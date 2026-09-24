from __future__ import annotations

import json
from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app


class LegacyStub(StubService):
    """Stub that also supports the legacy /chat pipeline call."""

    def __init__(self) -> None:
        super().__init__()
        self.pipeline_calls = 0

    def run_pipeline(self, query, user_profile=None, dialog_context=None, top_k=20, debug=False):
        self.pipeline_calls += 1
        nlu = self.analyze_query(query)
        return {
            "nlu_result": nlu,
            "retrieval_result": {"documents": [], "structured_data": [], "warnings": [], "coverage": {}},
        }


@pytest.fixture
def client_and_stub():
    stub = LegacyStub()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime))
    return TestClient(app), stub


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_agent_chat_endpoint(client_and_stub):
    client, _ = client_and_stub

    response = client.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": "api-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["session_id"] == "api-1"
    assert body["route"] == "workflow" and body["verification"]["passed"] is True
    assert body["risk_disclaimer"]


def test_agent_chat_generates_session_id(client_and_stub):
    client, _ = client_and_stub

    body = client.post("/agent/chat", json={"query": "今天天气怎么样"}).json()

    assert body["route"] == "refuse" and len(body["session_id"]) == 32


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x", "mode": "turbo"},
        {"query": "x", "session_id": "bad id with spaces"},
        {"query": "x" * 2001},
    ],
)
def test_agent_chat_validation(client_and_stub, payload):
    client, _ = client_and_stub

    assert client.post("/agent/chat", json=payload).status_code == 422


def test_clarification_resume_and_session_endpoints(client_and_stub):
    client, _ = client_and_stub

    pending = client.post("/agent/chat", json={"query": "这只股票能买吗", "session_id": "api-c"}).json()
    session = client.get("/agent/sessions/api-c").json()
    resumed = client.post("/agent/resume", json={"session_id": "api-c", "reply": "贵州茅台"})
    conflict = client.post("/agent/resume", json={"session_id": "api-c", "reply": "again"})

    assert pending["status"] == "needs_clarification"
    assert session["pending_clarification"]["type"] == "clarification" and session["turns"] == []
    assert resumed.status_code == 200 and resumed.json()["status"] == "ok"
    assert conflict.status_code == 409
    assert len(client.get("/agent/sessions/api-c").json()["turns"]) == 1
    assert client.get("/agent/sessions/bad%20id").status_code == 422


def test_agent_stream_endpoint_emits_sse(client_and_stub):
    client, _ = client_and_stub

    response = client.post("/agent/chat/stream", json={"query": "贵州茅台的市盈率是多少", "session_id": "api-s"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "session" and names[-1] == "done"
    assert "tool_call" in names and "answer" in names
    answer = next(data for name, data in events if name == "answer")
    assert answer["session_id"] == "api-s" and answer["verification"]["passed"] is True


def test_chat_mode_defaults_to_legacy_pipeline(client_and_stub, monkeypatch):
    client, stub = client_and_stub
    import query_intelligence.api.app as app_module

    monkeypatch.setattr(
        app_module,
        "build_chatbot_response",
        lambda **kwargs: {"answer": "legacy", "llm": {"status": "fallback"}},
    )

    legacy = client.post("/chat", json={"query": "贵州茅台的市盈率是多少"}).json()
    agent = client.post(
        "/chat", json={"query": "贵州茅台的市盈率是多少", "mode": "agent", "session_id": "api-m"}
    ).json()

    assert legacy == {"answer": "legacy", "llm": {"status": "fallback"}}
    assert stub.pipeline_calls == 1
    assert agent["session_id"] == "api-m" and agent["route"] == "workflow"  # no LLM configured -> downgraded
    assert "no_llm_configured:agent_route_downgraded_to_workflow" in agent["degraded"]
