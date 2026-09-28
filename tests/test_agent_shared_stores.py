"""Store selection and failure isolation for the shared (Postgres) A2A task and trace stores, without a database.

The two-replica behaviour itself is covered by ``test_agent_shared_stores_postgres.py``.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from query_intelligence.agent.pg import check_identifier, store_dsn
from query_intelligence.agent.telemetry import RecentTraceStore
from query_intelligence.agent.trace_store import PostgresTraceStore, build_trace_store

UNREACHABLE = "postgresql://nobody:nothing@127.0.0.1:1/none"


def test_store_dsn_follows_the_checkpointer_unless_overridden(monkeypatch):
    monkeypatch.delenv("QI_AGENT_TRACE_DB", raising=False)
    monkeypatch.delenv("QI_AGENT_CHECKPOINT_DB", raising=False)
    assert store_dsn("QI_AGENT_TRACE_DB") is None  # in memory by default

    monkeypatch.setenv("QI_AGENT_CHECKPOINT_DB", "/tmp/agent.sqlite")
    assert store_dsn("QI_AGENT_TRACE_DB") is None  # SQLite sessions are single-host: stores stay in memory

    monkeypatch.setenv("QI_AGENT_CHECKPOINT_DB", "postgresql://u:p@db:5432/finsight")
    assert store_dsn("QI_AGENT_TRACE_DB") == "postgresql://u:p@db:5432/finsight"

    monkeypatch.setenv("QI_AGENT_TRACE_DB", "memory")
    assert store_dsn("QI_AGENT_TRACE_DB") is None
    monkeypatch.setenv("QI_AGENT_TRACE_DB", "postgresql://other/db")
    assert store_dsn("QI_AGENT_TRACE_DB") == "postgresql://other/db"
    monkeypatch.setenv("QI_AGENT_TRACE_DB", "mysql://nope")
    assert store_dsn("QI_AGENT_TRACE_DB") is None


def test_table_names_are_validated():
    assert check_identifier("finsight_agent_traces") == "finsight_agent_traces"
    with pytest.raises(ValueError):
        check_identifier("traces; DROP TABLE x")


def test_unreachable_database_falls_back_to_in_memory_stores(monkeypatch):
    from a2a.server.tasks import InMemoryTaskStore

    from query_intelligence.agent.a2a_server import build_task_store

    monkeypatch.setenv("QI_AGENT_CHECKPOINT_DB", UNREACHABLE)
    monkeypatch.setenv("QI_STORE_CONNECT_TIMEOUT_S", "0.5")
    monkeypatch.delenv("QI_AGENT_TRACE_DB", raising=False)
    monkeypatch.delenv("QI_A2A_TASK_DB", raising=False)

    assert isinstance(build_trace_store(), RecentTraceStore)
    assert isinstance(build_task_store(), InMemoryTaskStore)


class _FlakyPool:
    """Accepts the schema setup, then fails every query (a database that went away)."""

    def __init__(self) -> None:
        self.calls = 0

    @contextmanager
    def connection(self):
        self.calls += 1
        if self.calls > 1:
            raise OSError("connection refused")
        yield self

    def execute(self, *_args, **_kwargs):
        return self

    def close(self) -> None:
        pass


def test_postgres_trace_store_keeps_serving_from_memory_when_the_database_fails():
    store = PostgresTraceStore(_FlakyPool(), max_rows=10, retention_days=1)

    store.emit({"trace_id": "t1", "owner": "key:a", "session_id": "s", "tools": [{"ok": True}]})

    assert store.write_errors == 1
    assert store.get("t1", owner="key:a")["trace_id"] == "t1"
    assert store.get("t1", owner="key:b") is None  # still owner-scoped
    assert [item["trace_id"] for item in store.recent(5, owner="key:a")] == ["t1"]
