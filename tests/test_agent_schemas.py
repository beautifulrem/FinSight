from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import jsonschema
from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM, final_turn, tool_call_turn
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app
from scripts.export_agent_schemas import main as export_main

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def _schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _client(llm=None) -> TestClient:
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), llm, today=lambda: date(2026, 9, 24))
    app = create_app(service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime))
    return TestClient(app)


def test_committed_schemas_match_contracts():
    assert export_main(["--check"]) == 0


def test_requests_validate_against_schemas():
    jsonschema.validate({"query": "贵州茅台的市盈率是多少", "mode": "auto"}, _schema("agent_chat_request.schema.json"))
    jsonschema.validate({"session_id": "s-1", "reply": "贵州茅台"}, _schema("agent_resume_request.schema.json"))
    try:
        jsonschema.validate({"query": "x", "mode": "turbo"}, _schema("agent_chat_request.schema.json"))
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError("invalid mode accepted by schema")


def test_workflow_refuse_and_clarify_responses_match_schema():
    client = _client()
    schema = _schema("agent_chat_response.schema.json")

    bodies = [
        client.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": "sc-1"}).json(),
        client.post("/agent/chat", json={"query": "今天天气怎么样", "session_id": "sc-2"}).json(),
        client.post("/agent/chat", json={"query": "这只股票能买吗", "session_id": "sc-3"}).json(),
        client.post("/agent/resume", json={"session_id": "sc-3", "reply": "贵州茅台"}).json(),
    ]

    assert [body["status"] for body in bodies] == ["ok", "ok", "needs_clarification", "ok"]
    for body in bodies:
        jsonschema.validate(body, schema)


def test_llm_agent_response_matches_schema():
    llm = ScriptedLLM(
        [
            tool_call_turn(("get_fundamentals", {"target": "600519.SH"})),
            final_turn(
                {
                    "answer": "贵州茅台市盈率为 24.6 倍 [fundamental_600519.SH]。",
                    "key_points": ["市盈率 24.6 倍"],
                    "evidence_used": ["fundamental_600519.SH"],
                }
            ),
        ]
    )
    body = _client(llm).post("/agent/chat", json={"query": "贵州茅台的市盈率是多少", "mode": "agent"}).json()

    jsonschema.validate(body, _schema("agent_chat_response.schema.json"))
    assert body["llm"]["calls"] >= 1
