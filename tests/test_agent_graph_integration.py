"""Agent graph over the real offline Query Intelligence service and tools (no LLM, no network)."""

from __future__ import annotations

from datetime import date

import pytest

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM, final_turn, tool_call_turn
from query_intelligence.agent.tools import build_registry_for_service


@pytest.fixture(scope="module")
def runtime(offline_service):
    registry = build_registry_for_service(offline_service)
    runtime = AgentRuntime(offline_service, registry, None, today=lambda: date(2026, 9, 24))
    yield runtime
    runtime.close()
    registry.shutdown()


def test_fact_question_workflow_is_verified(runtime):
    result = runtime.run("贵州茅台的市盈率是多少")

    assert result["route"] == "workflow"
    assert result["verification"]["passed"] is True
    assert {"price_600519.SH", "fundamental_600519.SH"} <= set(result["evidence_used"])
    assert "24.6" in result["answer"]


def test_why_question_without_llm_degrades_and_uses_news(runtime):
    result = runtime.run("茅台最近为什么跌了")

    assert result["route"] == "workflow"
    assert any(item.startswith("no_llm_configured") for item in result["degraded"])
    tools = [call["tool"] for call in result["tool_calls"]]
    assert "search_news" in tools and "get_price_history" in tools
    assert result["verification"]["passed"] is True
    assert result["answer"].startswith("现有证据不足以把结果归因于单一原因") or "可能" in result["answer"]


def test_macro_question_uses_macro_tool(runtime):
    result = runtime.run("CPI上升对白酒板块有什么影响？")

    assert "get_macro_indicators" in [call["tool"] for call in result["tool_calls"]]
    assert "macro_CPI_CN" in result["evidence_used"]


@pytest.mark.parametrize(("query", "route"), [("今天天气怎么样", "refuse"), ("这只股票能买吗", "clarify")])
def test_guard_routes(runtime, query, route):
    result = runtime.run(query)

    assert result["route"] == route
    assert result["tool_calls"] == []


def test_english_question_gets_english_answer(runtime):
    result = runtime.run("What is the PE ratio of Kweichow Moutai 600519.SH?")

    assert result["language"] == "en"
    assert result["answer"].startswith("Based on the evidence")
    assert result["risk_disclaimer"].startswith("This answer is based only on evidence")


def test_scripted_agent_over_real_tools(offline_service):
    registry = build_registry_for_service(offline_service)
    llm = ScriptedLLM(
        [
            tool_call_turn(
                ("get_fundamentals", {"target": "贵州茅台"}),
                ("get_fundamentals", {"target": "五粮液"}),
            ),
            final_turn(
                {
                    "answer": "贵州茅台 PE(TTM) 24.6 [fundamental_600519.SH]，行业为白酒 [industry_白酒]。",
                    "evidence_used": ["fundamental_600519.SH"],
                }
            ),
        ]
    )
    runtime = AgentRuntime(offline_service, registry, llm, today=lambda: date(2026, 9, 24))

    result = runtime.run("贵州茅台和五粮液的估值对比")

    assert result["route"] == "agent"
    assert result["verification"]["passed"] is True, result["verification"]
    assert {"fundamental_600519.SH", "industry_白酒"} <= set(result["evidence_used"])
    runtime.close()
    registry.shutdown()
