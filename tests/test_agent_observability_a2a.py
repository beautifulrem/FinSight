"""Run inspector (/agent/traces), Prometheus metrics (/metrics) and the A2A endpoint."""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.a2a_server import answer_data, answer_text, session_for_context
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.telemetry import PrometheusTraceSink, RecentTraceStore, summarize_trace
from query_intelligence.api.app import create_app

A2A_HEADERS = {"A2A-Version": "1.0"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    monkeypatch.setenv("QI_FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime))
    return TestClient(app)


def _send(client: TestClient, text: str, *, task_id: str | None = None, context_id: str | None = None) -> dict:
    message = {"messageId": uuid.uuid4().hex, "role": "ROLE_USER", "parts": [{"text": text}]}
    if task_id:
        message["taskId"] = task_id
    if context_id:
        message["contextId"] = context_id
    response = client.post(
        "/a2a",
        headers=A2A_HEADERS,
        json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": message}},
    )
    assert response.status_code == 200
    body = response.json()
    assert "error" not in body, body
    return body["result"]["task"]


def test_traces_endpoint_lists_and_returns_runs(client):
    answer = client.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": "obs-1"}).json()

    listing = client.get("/agent/traces", params={"session_id": "obs-1"}).json()["traces"]
    assert [item["trace_id"] for item in listing] == [answer["trace_id"]]
    assert listing[0]["route"] == "workflow" and listing[0]["tool_calls"] >= 1
    assert listing[0]["verification_passed"] is True

    trace = client.get(f"/agent/traces/{answer['trace_id']}").json()
    assert trace["query"] == "贵州茅台的市盈率是多少"
    assert trace["tools"] and trace["nodes"]

    assert client.get("/agent/traces/doesnotexist").status_code == 404
    assert client.get("/agent/traces/bad id!").status_code in {404, 422}


def test_metrics_endpoint_counts_runs_and_tools(client):
    client.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": "obs-2"})

    text = client.get("/metrics").text

    assert 'finsight_agent_runs_total{answer_source="template",route="workflow"} 1.0' in text
    assert "finsight_tool_calls_total" in text and "finsight_agent_run_seconds_bucket" in text


def test_agent_card_is_discoverable(client):
    card = client.get("/.well-known/agent-card.json").json()

    assert card["name"] == "FinSight"
    assert {skill["id"] for skill in card["skills"]} == {"equity_research", "comparison", "macro_linkage"}
    assert card["capabilities"]["streaming"] is True
    assert card["supportedInterfaces"][0]["url"].endswith("/a2a")


def test_a2a_send_message_returns_cited_answer_artifacts(client):
    task = _send(client, "贵州茅台的市盈率是多少")

    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    artifacts = {artifact["name"]: artifact for artifact in task["artifacts"]}
    text = artifacts["answer"]["parts"][0]["text"]
    assert "[fundamental_600519.SH]" in text and "不构成投资建议" in text
    data = artifacts["evidence"]["parts"][0]["data"]
    assert data["verification"]["passed"] is True
    assert data["evidence_used"] == ["fundamental_600519.SH"]
    assert data["session_id"] == session_for_context(task["contextId"])


def test_a2a_clarification_is_input_required_then_resumes(client):
    task = _send(client, "它的市盈率呢")
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert "哪只" in task["status"]["message"]["parts"][0]["text"]

    resumed = _send(client, "贵州茅台", task_id=task["id"], context_id=task["contextId"])

    assert resumed["id"] == task["id"]
    assert resumed["status"]["state"] == "TASK_STATE_COMPLETED"
    answer = next(artifact for artifact in resumed["artifacts"] if artifact["name"] == "answer")
    assert "600519" in answer["parts"][0]["text"]


def test_a2a_follow_up_in_same_context_keeps_memory(client):
    first = _send(client, "贵州茅台的市盈率是多少")

    follow_up = _send(client, "它的市净率呢", context_id=first["contextId"])

    assert follow_up["status"]["state"] == "TASK_STATE_COMPLETED"
    data = next(artifact for artifact in follow_up["artifacts"] if artifact["name"] == "evidence")["parts"][0]["data"]
    assert data["session_id"] == session_for_context(first["contextId"])
    assert any("600519" in evidence for evidence in data["evidence_used"])


def test_a2a_can_be_disabled(monkeypatch):
    monkeypatch.setenv("QI_A2A_ENABLED", "0")
    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    stub = StubService()
    app = create_app(service=stub, app_config={"deepseek": {"api_key": ""}})

    assert TestClient(app).get("/.well-known/agent-card.json").status_code == 404


def test_answer_helpers_and_session_mapping():
    response = {
        "answer": "PE 24.6 [e1]",
        "key_points": ["估值适中 [e1]", " "],
        "risk_disclaimer": "不构成投资建议。",
        "trace_id": "t1",
        "evidence_used": ["e1"],
        "llm": {"model": "x"},
    }

    assert answer_text(response) == "PE 24.6 [e1]\n\n- 估值适中 [e1]\n\n不构成投资建议。"
    assert answer_data(response) == {"trace_id": "t1", "evidence_used": ["e1"]}
    assert session_for_context("7e266b09-d1b7-4bdf") == "a2a7e266b09d1b74bdf"
    assert session_for_context(None) is None


def test_recent_trace_store_is_bounded_and_filters_sessions():
    store = RecentTraceStore(capacity=2)
    for index in range(3):
        store.emit({"trace_id": f"t{index}", "session_id": "s" if index else "other", "tools": [{"ok": False}]})

    assert store.get("t0") is None
    assert [item["trace_id"] for item in store.recent(10)] == ["t2", "t1"]
    assert [item["trace_id"] for item in store.recent(10, session_id="s")] == ["t2", "t1"]
    assert summarize_trace({"tools": [{"ok": False}, {"ok": True}]})["tool_errors"] == 1


def test_prometheus_sink_records_llm_usage_cost_and_degradation():
    sink = PrometheusTraceSink()
    sink.emit(
        {
            "route": "agent",
            "answer_source": "llm_agent",
            "duration_ms": 1200,
            "tools": [{"tool": "get_price_history", "ok": True, "latency_ms": 30, "cached": True}],
            "llm_calls": [{}, {}],
            "model": "m",
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "cost": 0.002,
            "currency": "USD",
            "verification_passed": False,
            "degraded": ["llm_error: timeout"],
        }
    )

    text = sink.render()[0].decode()
    assert 'finsight_llm_calls_total{model="m"} 2.0' in text
    assert 'finsight_llm_tokens_total{kind="prompt",model="m"} 100.0' in text
    assert 'finsight_llm_cost_total{currency="USD",model="m"} 0.002' in text
    assert 'finsight_tool_calls_total{outcome="cached",tool="get_price_history"} 1.0' in text
    assert "finsight_verification_failures_total 1.0" in text
    assert 'finsight_degradations_total{flag="llm_error"} 1.0' in text


def test_prometheus_sink_labels_failover_calls_with_the_answering_model():
    sink = PrometheusTraceSink()
    sink.emit(
        {
            "route": "agent",
            "llm_calls": [
                {"model": "primary", "prompt_tokens": 300, "completion_tokens": 100},
                {"model": "fallback", "prompt_tokens": 100, "completion_tokens": 0},
            ],
            "model": "primary",
            "cost": 0.004,
            "currency": "USD",
        }
    )

    text = sink.render()[0].decode()
    assert 'finsight_llm_calls_total{model="fallback"} 1.0' in text
    assert 'finsight_llm_tokens_total{kind="prompt",model="fallback"} 100.0' in text
    assert 'finsight_llm_cost_total{currency="USD",model="primary"} 0.0032' in text
    assert 'finsight_llm_cost_total{currency="USD",model="fallback"} 0.0008' in text


def _keyed_client(monkeypatch):
    from query_intelligence.api.security import SecuritySettings

    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    monkeypatch.delenv("QI_API_KEYS", raising=False)
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(
        service=stub,
        app_config={"deepseek": {"api_key": ""}},
        agent_service=AgentService(runtime),
        security=SecuritySettings(api_keys=("key-a", "key-b")),
    )
    return TestClient(app)


def test_sessions_and_traces_are_scoped_to_the_api_key(monkeypatch):
    client = _keyed_client(monkeypatch)
    a, b = {"X-API-Key": "key-a"}, {"X-API-Key": "key-b"}

    answer = client.post(
        "/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": "own-1"}, headers=a
    ).json()

    # Another caller cannot read, continue or list that session and its traces.
    assert client.get("/agent/sessions/own-1", headers=b).status_code == 404
    assert (
        client.post("/agent/chat", json={"query": "它的市净率呢", "session_id": "own-1"}, headers=b).status_code == 404
    )
    assert client.get(f"/agent/traces/{answer['trace_id']}", headers=b).status_code == 404
    assert client.get("/agent/traces", headers=b).json()["traces"] == []
    # The owner can.
    assert client.get("/agent/sessions/own-1", headers=a).json()["turns"]
    assert [t["trace_id"] for t in client.get("/agent/traces", headers=a).json()["traces"]] == [answer["trace_id"]]


def test_feedback_is_recorded_for_known_traces(client, tmp_path):
    answer = client.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": "fb-1"}).json()

    ok = client.post("/agent/feedback", json={"trace_id": answer["trace_id"], "rating": "down", "comment": "PE 旧了"})
    missing = client.post("/agent/feedback", json={"trace_id": "nope", "rating": "up"})
    invalid = client.post("/agent/feedback", json={"trace_id": answer["trace_id"], "rating": "meh"})

    assert ok.json() == {"ok": True} and missing.status_code == 404 and invalid.status_code == 422
    import json

    record = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8"))
    assert record["trace_id"] == answer["trace_id"] and record["query"] == "贵州茅台的市盈率是多少"
    assert 'finsight_feedback_total{rating="down"} 1.0' in client.get("/metrics").text


def test_agent_card_url_follows_the_request():
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime))

    card = TestClient(app, base_url="http://agents.example:9000").get("/.well-known/agent-card.json").json()

    assert card["supportedInterfaces"][0]["url"] == "http://agents.example:9000/a2a"
