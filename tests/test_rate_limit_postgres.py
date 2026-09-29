"""Two app instances (replicas) sharing one Postgres rate-limit bucket per client.

Opt-in integration test (skipped without a database), e.g.

    docker run -d --rm --name fs-rl-pg -e POSTGRES_PASSWORD=finsight -e POSTGRES_DB=finsight -p 55439:5432 \\
      postgres:16-alpine
    export QI_TEST_POSTGRES_DSN=postgresql://postgres:finsight@127.0.0.1:55439/finsight
    pytest tests/test_rate_limit_postgres.py -v
"""

from __future__ import annotations

import os
import threading
import uuid
from contextlib import ExitStack
from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.pg import open_pool
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app
from query_intelligence.api.rate_limit import PostgresTokenBucket
from query_intelligence.api.security import SecuritySettings

DSN = os.getenv("QI_TEST_POSTGRES_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="set QI_TEST_POSTGRES_DSN to run the shared rate-limit tests")
QUERY = {"query": "今天天气怎么样"}


def _replica(rate: int, keys: tuple[str, ...] = ()) -> TestClient:
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(
        service=stub,
        app_config={"deepseek": {"api_key": ""}},
        agent_service=AgentService(runtime, trace_sinks=[]),
        security=SecuritySettings(api_keys=keys, rate_limit_per_minute=rate, rate_limit_db=DSN),
    )
    return TestClient(app)


@pytest.fixture
def two_replicas(monkeypatch):
    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    key = f"k-{uuid.uuid4().hex[:8]}"  # a fresh principal, so earlier runs' rows do not interfere
    with ExitStack() as stack:
        replicas = [stack.enter_context(_replica(4, keys=(key,))) for _ in range(2)]
        yield replicas, {"X-API-Key": key}


def test_replicas_share_one_bucket_per_client(two_replicas):
    (a, b), headers = two_replicas
    assert type(a.app.state.rate_limiter).__name__ == "PostgresTokenBucket"

    statuses = [(a if i % 2 == 0 else b).post("/agent/chat", json=QUERY, headers=headers).status_code for i in range(6)]

    # 4 per minute in total, not 4 per replica (an in-process bucket would allow all 6)
    assert statuses == [200, 200, 200, 200, 429, 429]
    limited = b.post("/agent/chat", json=QUERY, headers=headers)
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1


def test_concurrent_takes_across_two_limiters_never_overspend():
    table = f"rl_test_{uuid.uuid4().hex[:8]}"
    now = {"t": 1_000_000.0}  # frozen clock: no refill during the test
    pools = [open_pool(DSN, name=f"rl-{i}", max_size=8) for i in range(2)]
    try:
        limiters = [PostgresTokenBucket(pool, 30, table=table, clock=lambda: now["t"]) for pool in pools]
        allowed: list[bool] = []
        lock = threading.Lock()

        def worker(limiter: PostgresTokenBucket) -> None:
            for _ in range(20):
                ok = limiter.take("shared-client") == 0
                with lock:
                    allowed.append(ok)

        threads = [threading.Thread(target=worker, args=(limiters[i % 2],)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(allowed) == 120 and sum(allowed) == 30  # exactly the capacity, under 6 concurrent callers
        now["t"] += 2.0  # 30/min refills one token per 2 s
        assert limiters[0].take("shared-client") == 0 and limiters[1].take("shared-client") > 0
    finally:
        with pools[0].connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        for pool in pools:
            pool.close()
