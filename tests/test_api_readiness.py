"""Liveness (/health) vs deep readiness (/ready): checkpoint store, model config, retrieval index."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
from datetime import date
from types import SimpleNamespace

import pytest
from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.memory import make_checkpointer
from query_intelligence.agent.service import AgentService
from query_intelligence.api import readiness
from query_intelligence.api.app import create_app
from query_intelligence.api.readiness import ReadinessChecker
from query_intelligence.api.security import SecuritySettings


def _agent(checkpointer=None) -> AgentService:
    runtime = AgentRuntime(StubService(), build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    return AgentService(runtime, checkpointer=checkpointer, trace_sinks=[])


def _client(*, service=None, agent_service=None, app_config=None, security=None) -> TestClient:
    app = create_app(
        service=service or StubService(),
        app_config=app_config or {"deepseek": {"api_key": ""}},
        agent_service=agent_service,
        security=security or SecuritySettings(),
    )
    return TestClient(app)


class _Retriever:
    def __init__(self, documents: int, rows: int | None = None) -> None:
        self.documents = [{"doc_id": str(i)} for i in range(documents)]
        self.vectorizer = object()
        self.doc_matrix = SimpleNamespace(shape=(documents if rows is None else rows, 128))


def _service_with_index(documents: int, rows: int | None = None, tables: int = 2) -> StubService:
    service = StubService()
    service.retrieval_pipeline = SimpleNamespace(
        doc_retriever=_Retriever(documents, rows),
        sql_retriever=SimpleNamespace(structured_data={f"t{i}": {} for i in range(tables)}),
    )
    return service


def test_ready_when_all_checks_pass():
    client = _client(service=_service_with_index(10), agent_service=_agent())

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"checkpointer", "model_config", "retrieval_index"}
    assert body["checks"]["checkpointer"] == {"ok": True, "backend": "InMemorySaver", "persistent": False}
    assert body["checks"]["retrieval_index"]["documents"] == 10
    assert body["checks"]["model_config"]["llm"].startswith("not configured")


def test_ready_and_health_are_public_when_api_keys_are_required():
    client = _client(agent_service=_agent(), security=SecuritySettings(api_keys=("k1",), rate_limit_per_minute=1))

    assert [client.get("/ready").status_code for _ in range(3)] == [200, 200, 200]
    assert client.get("/health").status_code == 200
    assert client.get("/agent/traces").status_code == 401


def test_unopenable_checkpoint_db_is_unready_but_alive(monkeypatch, tmp_path):
    # B22: a root-owned or missing state volume. /health stays 200 (the process is alive), /ready is 503.
    monkeypatch.setenv("QI_AGENT_CHECKPOINT_DB", str(tmp_path / "missing-dir" / "agent.sqlite"))
    build = classmethod(lambda cls, runtime, **kwargs: _agent(checkpointer=make_checkpointer()))
    monkeypatch.setattr(AgentService, "from_service", build)
    client = _client()

    assert client.get("/health").status_code == 200
    response = client.get("/ready")

    assert response.status_code == 503
    check = response.json()["checks"]["checkpointer"]
    assert check["ok"] is False and check["error"].startswith("OperationalError")


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions and a non-root user")
def test_read_only_sqlite_checkpointer_is_unready(tmp_path):
    path = tmp_path / "agent.sqlite"
    SqliteSaver(sqlite3.connect(path, check_same_thread=False)).setup()
    path.chmod(stat.S_IRUSR)
    tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        client = _client(agent_service=_agent(checkpointer=saver))

        response = client.get("/ready")

        assert response.status_code == 503
        assert "readonly" in response.json()["checks"]["checkpointer"]["error"]
    finally:
        tmp_path.chmod(stat.S_IRWXU)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_writable_sqlite_checkpointer_is_ready(tmp_path):
    saver = SqliteSaver(sqlite3.connect(tmp_path / "agent.sqlite", check_same_thread=False))
    saver.setup()

    detail = readiness.probe_checkpointer(saver)

    assert detail == {"ok": True, "backend": "SqliteSaver", "persistent": True, "database": "sqlite"}


def test_unreachable_postgres_pool_fails_fast_and_redacts_the_password(monkeypatch):
    from psycopg_pool import ConnectionPool

    monkeypatch.setenv("QI_READY_DB_TIMEOUT_S", "0.5")
    pool = ConnectionPool("postgresql://app:s3cret-pw@127.0.0.1:1/finsight", min_size=1, open=True)
    try:
        checker = ReadinessChecker(
            {"checkpointer": lambda: readiness.probe_checkpointer(SimpleNamespace(conn=pool))}, timeout_s=3
        )

        ready, report = checker.run()
    finally:
        pool.close()

    assert ready is False
    assert report["checks"]["checkpointer"]["error"].startswith("PoolTimeout")
    assert "s3cret-pw" not in str(report)


@pytest.mark.skipif(not os.getenv("QI_TEST_POSTGRES_DSN"), reason="set QI_TEST_POSTGRES_DSN to test a live Postgres")
def test_live_postgres_checkpointer_is_ready():
    saver = make_checkpointer(os.environ["QI_TEST_POSTGRES_DSN"])
    try:
        detail = readiness.probe_checkpointer(saver)
    finally:
        saver.conn.close()

    assert detail["ok"] is True and detail["database"] == "postgres"


def test_error_messages_redact_dsn_credentials():
    error = RuntimeError("connection to postgresql://user:hunter2@db:5432/x failed; password=hunter2 host=db")

    text = readiness.describe_error(error)

    assert "hunter2" not in text and "postgresql://***@db:5432/x" in text


@pytest.mark.parametrize(
    ("section", "ok", "error"),
    [
        ({"api_key": ""}, True, None),
        ({"api_key": "sk-test", "base_url": "https://gateway.example.com/v1", "model": "m"}, True, None),
        ({"api_key": "sk-test", "base_url": "gateway.example.com", "model": "m"}, False, "base_url"),
        ({"api_key": "sk-test", "base_url": "https://g.example.com", "timeout_seconds": -1}, False, "timeout"),
        ({"api_key": "sk-test", "base_url": "https://g.example.com", "model": " "}, False, "model is empty"),
    ],
)
def test_model_config_sanity(section, ok, error):
    detail = readiness.check_model_config({"deepseek": section})

    assert detail["ok"] is ok
    if error:
        assert error in detail["error"]
    assert "sk-test" not in str(detail)


def test_configured_key_without_an_agent_llm_is_unready():
    config = {"deepseek": {"api_key": "sk-test", "base_url": "https://g.example.com", "model": "m"}}
    client = _client(agent_service=_agent(), app_config=config)

    response = client.get("/ready")  # the injected agent was built without an LLM client

    assert response.status_code == 503
    assert "no LLM client" in response.json()["checks"]["model_config"]["error"]


@pytest.mark.parametrize(
    ("documents", "rows", "tables", "error"),
    [(0, None, 2, "corpus is empty"), (10, 9, 2, "not fitted"), (10, None, 0, "structured data is empty")],
)
def test_retrieval_index_problems_make_the_replica_unready(documents, rows, tables, error):
    client = _client(service=_service_with_index(documents, rows, tables), agent_service=_agent())

    response = client.get("/ready")

    assert response.status_code == 503
    assert error in response.json()["checks"]["retrieval_index"]["error"]


def test_slow_check_times_out_and_is_not_started_twice():
    release = threading.Event()
    calls = []

    def slow():
        calls.append(1)
        release.wait(5)
        return {"ok": True}

    checker = ReadinessChecker({"slow": slow, "fast": lambda: {"ok": True}}, timeout_s=0.2)
    try:
        first_ready, first = checker.run()
        second_ready, second = checker.run()
    finally:
        release.set()

    assert not first_ready and first["checks"]["slow"]["error"] == "timed out after 0.2 s"
    assert not second_ready and second["checks"]["slow"]["error"] == "previous check is still running"
    assert first["checks"]["fast"] == {"ok": True}
    assert len(calls) == 1


def test_real_retrieval_pipeline_shape():
    # The check reads the attributes of the real DocumentRetriever.
    from query_intelligence.retrieval.doc_retriever import DocumentRetriever

    docs = [
        {"doc_id": "a", "title": "贵州茅台", "summary": "白酒", "body": "年报", "source_type": "news"},
        {"doc_id": "b", "title": "五粮液", "summary": "白酒", "body": "季报", "source_type": "news"},
    ]
    service = StubService()
    service.retrieval_pipeline = SimpleNamespace(
        doc_retriever=DocumentRetriever(docs), sql_retriever=SimpleNamespace(structured_data={"market_api": {}})
    )

    detail = readiness.check_retrieval_index(service)

    assert detail == {
        "ok": True,
        "backend": "DocumentRetriever",
        "documents": 2,
        "index_rows": 2,
        "structured_tables": 1,
    }


# ---- in-process doubles for the Postgres-only paths (round 10, F13: local coverage without a database) ----


class _FakePgPool:
    """Quacks like psycopg_pool.ConnectionPool: ``connection(timeout=…)`` and ``get_stats()``."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def connection(self, timeout: float | None = None):
        pool = self

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, *args):
                pool.queries.append(sql)
                return SimpleNamespace(fetchone=lambda: (1,))

        return _Conn()

    def get_stats(self) -> dict:
        return {"pool_size": 4, "pool_available": 3}


def test_a_postgres_pool_checkpointer_is_probed_with_select_1():
    pool = _FakePgPool()
    detail = readiness.probe_checkpointer(SimpleNamespace(conn=pool))
    assert detail == {
        "ok": True,
        "backend": "SimpleNamespace",
        "persistent": True,
        "database": "postgres",
        "pool": {"size": 4, "available": 3},
    }
    assert pool.queries == ["SELECT 1"]


def test_a_single_connection_checkpointer_is_probed_under_its_lock():
    executed: list[str] = []
    conn = SimpleNamespace(execute=lambda sql: executed.append(sql) or SimpleNamespace(fetchone=lambda: (1,)))
    detail = readiness.probe_checkpointer(SimpleNamespace(conn=conn, lock=threading.Lock()))
    assert detail == {"ok": True, "backend": "SimpleNamespace", "persistent": True} and executed == ["SELECT 1"]
    assert readiness.probe_checkpointer(None)["ok"] is False


def test_a_postgres_document_retriever_and_an_unknown_one():
    executed: list[str] = []
    connection = SimpleNamespace(execute=lambda sql: executed.append(sql) or SimpleNamespace(fetchone=lambda: (1,)))
    service = SimpleNamespace(
        retrieval_pipeline=SimpleNamespace(
            doc_retriever=SimpleNamespace(connection=connection),
            sql_retriever=SimpleNamespace(structured_data={"a": 1}),
        )
    )
    detail = readiness.check_retrieval_index(service)
    assert detail["ok"] and detail["database"] == "postgres" and detail["structured_tables"] == 1
    unknown = SimpleNamespace(retrieval_pipeline=SimpleNamespace(doc_retriever=object(), sql_retriever=None))
    assert readiness.check_retrieval_index(unknown) == {
        "ok": False,
        "backend": "object",
        "error": "unknown document retriever",
    }
