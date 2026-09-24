from __future__ import annotations

from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry

from evaluation.agent_eval.fault_injection import SCENARIOS, _llm_program, graceful, wrap_registry
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.state import AgentConfig


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_fault_scenario_degrades_gracefully_on_stub(name):
    spec = SCENARIOS[name]
    registry = wrap_registry(build_fake_registry(), spec["fault"], timeout_s=spec.get("timeout_s"))
    llm = _llm_program(spec["llm"])() if spec.get("llm") else None
    runtime = AgentRuntime(
        StubService(), registry, llm, config=AgentConfig(max_llm_steps=3), today=lambda: date(2026, 4, 23)
    )
    agent = AgentService(runtime, trace_sinks=[])

    response = agent.chat("茅台为什么跌了", mode=spec.get("mode", "workflow"))
    agent.close()

    ok, problems = graceful(response, name)
    assert ok, problems


def test_graceful_flags_crashes_and_trading_advice():
    assert graceful({"status": "exception"}, "tool_timeout") == (False, ["status"])
    bad = {
        "status": "ok",
        "answer": "建议立即买入。",
        "risk_disclaimer": "x",
        "tool_calls": [{"tool": "get_price_history", "ok": False, "error": {"code": "timeout"}}],
    }
    ok, problems = graceful(bad, "tool_timeout")
    assert not ok and "trading_instruction" in problems
