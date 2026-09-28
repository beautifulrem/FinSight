"""Two app instances (replicas) sharing one Postgres: A2A tasks and agent traces.

Opt-in integration test (skipped without a database), e.g.

    docker run -d --name fs-pg -e POSTGRES_PASSWORD=finsight -e POSTGRES_DB=finsight -p 55433:5432 postgres:16-alpine
    export QI_TEST_POSTGRES_DSN=postgresql://postgres:finsight@127.0.0.1:55433/finsight
    pytest tests/test_agent_shared_stores_postgres.py -v

Each "replica" is a separate ``create_app`` with its own agent service, trace store and A2A task store; they
only share the database, as two pods would.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import ExitStack
from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.memory import make_checkpointer
from query_intelligence.agent.pg import open_pool
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.trace_store import PostgresTraceStore
from query_intelligence.api.app import create_app
from query_intelligence.api.security import SecuritySettings

DSN = os.getenv("QI_TEST_POSTGRES_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="set QI_TEST_POSTGRES_DSN to run the shared-store Postgres tests")

KEY_A, KEY_B = {"X-API-Key": "key-a"}, {"X-API-Key": "key-b"}


@pytest.fixture
def replicas(monkeypatch):
    """Two started apps (lifespan entered, so their connection pools are closed at the end)."""
    with ExitStack() as stack:
        yield [stack.enter_context(_replica(monkeypatch)) for _ in range(2)]


def _replica(monkeypatch) -> TestClient:
    monkeypatch.setenv("QI_AGENT_CHECKPOINT_DB", DSN)  # traces and A2A tasks follow the checkpointer's DSN
    monkeypatch.delenv("QI_AGENT_TRACE_DB", raising=False)
    monkeypatch.delenv("QI_A2A_TASK_DB", raising=False)
    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    agent = AgentService(runtime, checkpointer=make_checkpointer(DSN), trace_sinks=[])
    app = create_app(
        service=stub,
        app_config={"deepseek": {"api_key": ""}},
        agent_service=agent,
        security=SecuritySettings(api_keys=("key-a", "key-b")),
    )
    return TestClient(app)


def _rpc(client: TestClient, method: str, params: dict, headers: dict) -> dict:
    response = client.post(
        "/a2a",
        headers={**headers, "A2A-Version": "1.0"},
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
    )
    assert response.status_code == 200
    return response.json()


def _send(client: TestClient, text: str, headers: dict, *, task_id=None, context_id=None) -> dict:
    message = {"messageId": uuid.uuid4().hex, "role": "ROLE_USER", "parts": [{"text": text}]}
    if task_id:
        message["taskId"] = task_id
    if context_id:
        message["contextId"] = context_id
    body = _rpc(client, "SendMessage", {"message": message}, headers)
    assert "error" not in body, body
    return body["result"]["task"]


def test_a2a_task_started_on_one_replica_is_served_and_resumed_by_the_other(replicas):
    replica_1, replica_2 = replicas

    pending = _send(replica_1, "它的市盈率呢", KEY_A)
    assert pending["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"

    # Replica 2 has never seen this task in memory: the lookup comes from Postgres.
    seen = _rpc(replica_2, "GetTask", {"id": pending["id"]}, KEY_A)["result"]
    assert seen["id"] == pending["id"] and seen["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    # Tasks are owner-scoped: another API key cannot read it.
    assert "error" in _rpc(replica_2, "GetTask", {"id": pending["id"]}, KEY_B)

    # The clarification reply lands on replica 2, which resumes the paused run (session state is in Postgres too).
    resumed = _send(replica_2, "贵州茅台", KEY_A, task_id=pending["id"], context_id=pending["contextId"])
    assert resumed["id"] == pending["id"] and resumed["status"]["state"] == "TASK_STATE_COMPLETED"
    answer = next(artifact for artifact in resumed["artifacts"] if artifact["name"] == "answer")
    assert "600519" in answer["parts"][0]["text"]

    # And replica 1 now sees the completed task, with its artifacts, and lists it by context.
    final = _rpc(replica_1, "GetTask", {"id": pending["id"]}, KEY_A)["result"]
    assert final["status"]["state"] == "TASK_STATE_COMPLETED"
    assert {artifact["name"] for artifact in final["artifacts"]} == {"answer", "evidence"}
    listed = _rpc(replica_1, "ListTasks", {"contextId": pending["contextId"]}, KEY_A)["result"]
    assert [task["id"] for task in listed["tasks"]] == [pending["id"]]
    assert _rpc(replica_1, "ListTasks", {"contextId": pending["contextId"]}, KEY_B)["result"].get("tasks", []) == []


def test_traces_written_by_one_replica_are_listed_and_served_by_the_other(replicas):
    replica_1, replica_2 = replicas
    session = f"pg-trace-{uuid.uuid4().hex[:8]}"

    answer = replica_1.post(
        "/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": session}, headers=KEY_A
    ).json()

    listing = replica_2.get("/agent/traces", params={"session_id": session}, headers=KEY_A).json()["traces"]
    assert [item["trace_id"] for item in listing] == [answer["trace_id"]]
    assert listing[0]["route"] == "workflow" and listing[0]["verification_passed"] is True
    trace = replica_2.get(f"/agent/traces/{answer['trace_id']}", headers=KEY_A).json()
    assert trace["query"] == "贵州茅台的市盈率是多少" and trace["tools"]

    # Owner scoping holds across replicas.
    assert replica_2.get(f"/agent/traces/{answer['trace_id']}", headers=KEY_B).status_code == 404
    assert replica_2.get("/agent/traces", params={"session_id": session}, headers=KEY_B).json()["traces"] == []

    # Feedback on replica 2 finds the trace written by replica 1.
    feedback = replica_2.post("/agent/feedback", json={"trace_id": answer["trace_id"], "rating": "up"}, headers=KEY_A)
    assert feedback.json() == {"ok": True}


def test_trace_retention_by_count_and_age():
    table = f"test_traces_{uuid.uuid4().hex[:8]}"
    pool = open_pool(DSN, name="test-traces")
    store = PostgresTraceStore(pool, table=table, max_rows=3, retention_days=1, prune_every=1)
    try:
        now = time.time()
        store.emit({"trace_id": "old", "owner": "local", "started_at": now - 3 * 86400, "tools": []})
        for index in range(5):
            store.emit({"trace_id": f"t{index}", "owner": "local", "started_at": now + index, "tools": []})

        assert [item["trace_id"] for item in store.recent(10)] == ["t4", "t3", "t2"]
        assert store.get("old") is None and store.get("t4")["trace_id"] == "t4"
        with pool.connection() as conn:
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 3
    finally:
        with pool.connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        store.close()
