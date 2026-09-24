from __future__ import annotations

import time
from datetime import date

from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM, final_turn, tool_call_turn
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.state import AgentConfig
from query_intelligence.api.app import create_app


def test_run_deadline_forces_a_final_answer():
    llm = ScriptedLLM(
        [
            tool_call_turn(("get_price_history", {"target": "600519.SH"})),
            final_turn({"answer": "最新收盘价 1409.5 [price_600519.SH]。"}),
        ]
    )
    runtime = AgentRuntime(
        StubService(), build_fake_registry(), llm, config=AgentConfig(run_deadline_s=0), today=lambda: date(2026, 9, 24)
    )

    result = runtime.run("茅台为什么跌了")

    assert any(item.startswith("budget:run deadline") for item in result["degraded"])
    assert llm.requests[0]["tools"] is None and llm.requests[0]["json_mode"] is True


class SlowService:
    def chat(self, *args, **kwargs):
        time.sleep(1.0)
        return {"status": "ok"}


def test_agent_endpoint_times_out_with_504(monkeypatch):
    monkeypatch.setenv("QI_AGENT_REQUEST_TIMEOUT_S", "0.2")
    stub = StubService()
    app = create_app(service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=SlowService())

    response = TestClient(app).post("/agent/chat", json={"query": "贵州茅台的市盈率是多少"})

    assert response.status_code == 504
    assert "did not finish" in response.json()["detail"]


def test_agent_endpoint_within_timeout(monkeypatch):
    monkeypatch.setenv("QI_AGENT_REQUEST_TIMEOUT_S", "30")
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(
        service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime, trace_sinks=[])
    )

    assert TestClient(app).post("/agent/chat", json={"query": "贵州茅台的市盈率是多少"}).json()["status"] == "ok"
