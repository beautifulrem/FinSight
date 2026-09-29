"""Rate limiter selection and failure policy (no database needed)."""

from __future__ import annotations

import logging

import pytest

from query_intelligence.api.rate_limit import PostgresTokenBucket, TokenBucket, build_rate_limiter
from query_intelligence.api.security import SecuritySettings


class _BrokenPool:
    """A pool whose connections always fail (database down after start-up)."""

    def __init__(self, fail_create: bool = False) -> None:
        self.fail_create = fail_create
        self.created = 0
        self.closed = False

    def connection(self):
        pool = self

        class _Conn:
            def __enter__(self):
                if pool.fail_create or pool.created:
                    raise ConnectionError("database unreachable")
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, *args, **kwargs):
                pool.created += 1

        return _Conn()

    def close(self) -> None:
        self.closed = True


def test_off_in_memory_or_postgres_by_configuration():
    assert build_rate_limiter(0) is None
    assert isinstance(build_rate_limiter(5, dsn=""), TokenBucket)


def test_unreachable_database_at_start_up_falls_back_to_the_in_process_bucket(caplog):
    with caplog.at_level(logging.WARNING):
        limiter = build_rate_limiter(5, dsn="postgresql://nobody:x@127.0.0.1:9/none?connect_timeout=1")

    assert isinstance(limiter, TokenBucket)
    assert "limiting per replica" in caplog.text


def test_database_errors_after_start_up_limit_per_replica_instead_of_failing(caplog):
    limiter = PostgresTokenBucket(_BrokenPool(), 2)  # table created, then every call fails

    with caplog.at_level(logging.WARNING):
        waits = [limiter.take("c") for _ in range(3)]

    assert waits[:2] == [0.0, 0.0] and waits[2] > 0  # the fallback bucket still limits
    assert caplog.text.count("shared rate limiter unavailable") == 1  # logged once, not per request
    limiter.close()
    assert limiter.pool.closed


def test_table_creation_is_retried_once_then_raises():
    with pytest.raises(ConnectionError):
        PostgresTokenBucket(_BrokenPool(fail_create=True), 2)


def test_settings_pick_up_the_shared_limiter_dsn(monkeypatch):
    monkeypatch.delenv("QI_AGENT_CHECKPOINT_DB", raising=False)
    monkeypatch.setenv("QI_RATE_LIMIT_DB", "postgresql://u:p@db:5432/x")
    assert SecuritySettings.from_env().rate_limit_db == "postgresql://u:p@db:5432/x"

    monkeypatch.setenv("QI_RATE_LIMIT_DB", "memory")
    monkeypatch.setenv("QI_AGENT_CHECKPOINT_DB", "postgresql://u:p@db:5432/x")
    assert SecuritySettings.from_env().rate_limit_db == ""

    monkeypatch.delenv("QI_RATE_LIMIT_DB")
    assert SecuritySettings.from_env().rate_limit_db == "postgresql://u:p@db:5432/x"  # follows the checkpointer
