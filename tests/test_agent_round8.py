"""Round-8 rules, written from the round-4 review (D5-D8), with new wording (none repeats a reviewer probe).

* D5: fair-value / "what is it worth" questions are judgment: hedged, with a limitation, never one number as the value.
* D6: crypto funds are out of coverage (never fuzzy-matched to an A-share ETF); one policy for the short name 平安.
* D7: net margin is derived from the cited fundamentals; PEG and year-to-date returns are derived only when the
  evidence has what they need, and otherwise stated as unavailable.
* D8: the "no single cause" caveat only on causal (why) questions.

Offline snapshot values: 贵州茅台 revenue 1688.38 亿 / net profit 823.2 亿 (FY2025), 五粮液 1085 亿 / 378 亿,
中国平安 PE 8.7; 沪深300 has two closes (2026-04-21, 2026-04-22).
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.tools.defaults import build_registry_for_service


@pytest.fixture(scope="module")
def agent(offline_service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService

    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), ScriptedLLM([]))
    service = AgentService(runtime, trace_sinks=[])
    yield service
    runtime.close()


def _targets(result: dict) -> set[str]:
    return {call["arguments"].get("target") for call in result.get("tool_calls") or []}


def _entities(result: dict) -> set[str]:
    return {entity.get("symbol") for entity in (result.get("nlu_summary") or {}).get("entities") or []}


# --- D5: fair value / "what is it worth" is a judgment -------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "五粮液一股合理价位大概在哪",
        "按现在的业绩，平安估值应该是多少才合理",
        "你觉得茅台的内在价值有多少",
        "五粮液到底值多少钱",
        "How much is Wuliangye really worth?",
        "What's a fair value for Ping An shares?",
        "Estimate the intrinsic value of Moutai",
    ],
)
def test_fair_value_questions_are_judgments(query):
    from query_intelligence.agent.router import FAIR_VALUE_MARKERS, decide_route

    assert FAIR_VALUE_MARKERS.search(query)
    nlu = {"entities": [{"symbol": "600519.SH", "entity_type": "stock"}], "question_style": "fact"}
    assert "lexical:judgment_or_timing" in decide_route(nlu, query=query).reasons


@pytest.mark.parametrize(
    "query",
    [
        "五粮液现在多少钱一股",  # a price, not a value judgment
        "茅台总市值多少钱",
        "这只基金净值多少钱",
        "公允价值变动收益是多少",
        "Is Moutai worth buying?",  # already a judgment through "worth buying", not a fair-value request
        "茅台的估值多少倍",
    ],
)
def test_prices_and_reported_values_are_not_fair_value_requests(query):
    from query_intelligence.agent.router import FAIR_VALUE_MARKERS

    assert not FAIR_VALUE_MARKERS.search(query)


def test_a_fair_value_answer_is_hedged_with_a_limitation_and_no_value_of_its_own(agent):
    result = agent.chat("五粮液一股合理价位大概在哪", session_id="r8-fair-value")
    answer = str(result["answer"])
    assert {"conditional_prefix", "fair_value_hedge"} <= set(result["compliance_notes"])
    assert answer.startswith("基于当前证据只能做条件性判断")
    assert "不给出合理估值" in answer
    assert any("合理估值" in item for item in result["limitations"])
    assert "000858.SZ" in _targets(result)


def test_an_english_worth_question_is_hedged(agent):
    result = agent.chat("How much is Wuliangye really worth?", session_id="r8-worth-en")
    assert "fair_value_hedge" in result["compliance_notes"]
    assert "does not give a fair value" in str(result["answer"])


@pytest.mark.parametrize(
    "sentence",
    [
        "综合来看，茅台的合理估值约为1500元。",
        "我们测算每股内在价值在1320元左右。",
        "五粮液合理股价区间应在120元附近。",
        "Its fair value is about CNY 1,500 per share.",
        "We think the stock is worth about CNY 120.",
        "The intrinsic value of Moutai is around 1320.",
    ],
)
def test_a_single_number_presented_as_the_fair_value_is_removed(sentence):
    from query_intelligence.agent.compliance import apply_compliance, contains_trading_instruction

    assert contains_trading_instruction(sentence)
    guarded, notes = apply_compliance(
        {"answer": f"贵州茅台最新收盘价为 1409.5 元 [price_600519.SH]。{sentence}", "limitations": []},
        query="茅台合理估值多少",
        nlu_result={"question_style": "fact"},
    )
    assert "removed_trading_instruction" in notes
    assert "1500" not in guarded["answer"] and "1320" not in guarded["answer"]


@pytest.mark.parametrize(
    "sentence",
    [
        "贵州茅台 PE(TTM) 24.6 倍，行业 PE 27.3 倍 [fundamental_600519.SH]。",
        "公允价值变动收益为 3.2 亿元 [fundamental_601318.SH]。",
        "Net worth of the fund rose 2% [price_510300.SH].",
    ],
)
def test_valuation_facts_are_not_fair_value_claims(sentence):
    from query_intelligence.agent.compliance import contains_trading_instruction

    assert not contains_trading_instruction(sentence)
