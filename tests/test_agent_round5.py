"""Round-5 rules (reviewer round 3: C5, C7, C9, C10, C12, C20).

Every question here is new wording, not copied from the reviewer's battery, an independent set or a test set.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.tools import build_registry_for_service
from query_intelligence.chat.language import detect_query_language, requested_answer_language


@pytest.fixture
def agent(offline_service):
    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), ScriptedLLM([]))
    service = AgentService(runtime, trace_sinks=[])
    yield service
    runtime.close()


# --------------------------------------------------------------------------- C12: explicit answer language


@pytest.mark.parametrize(
    ("query", "language"),
    [
        ("麻烦用英语回复：招商银行的股息率", "en"),
        ("英文作答，沪深300ETF收在多少", "en"),
        ("Reply in Chinese please: how did Ping An close?", "zh"),
        ("Explain in Mandarin, please: what's the CSI 300 level?", "zh"),
        ("in English please, 中国平安市净率", "en"),
        # no instruction: the question's own language
        ("中国平安的市净率", "zh"),
        ("What's Ping An's P/B?", "en"),
    ],
)
def test_explicit_answer_language_overrides_detection(query, language):
    assert detect_query_language(query) == language


def test_the_last_language_instruction_wins():
    assert requested_answer_language("先用中文回答，算了还是用英文回答吧：茅台PE") == "en"
    assert requested_answer_language("茅台PE多少") is None


def test_agent_answers_in_the_requested_language(agent):
    english = agent.chat("麻烦用英语回复：五粮液的市净率")
    assert english["language"] == "en"
    assert "fundamentals" in str(english["answer"]) and "基本面" not in str(english["answer"])
    chinese = agent.chat("Reply in Chinese please: what is Kweichow Moutai's P/B?")
    assert chinese["language"] == "zh"
    assert "基本面" in str(chinese["answer"])


# --------------------------------------------------------------------------- C5: "it" inside a comparison


def _turn(name: str, symbol: str, query: str = "") -> dict:
    return {"query": query or name, "entities": [{"name": name, "symbol": symbol}], "named": True}


PING_AN = ("中国平安", "601318.SH")
WULIANGYE = ("五粮液", "000858.SZ")
MOUTAI = ("贵州茅台", "600519.SH")


@pytest.mark.parametrize(
    "query",
    ["Now compare that with Kweichow Moutai.", "Put it against Moutai.", "How does it stack up against Moutai?"],
)
def test_english_object_pronoun_in_a_comparison_keeps_the_earlier_target(query):
    from query_intelligence.agent.memory import resolve_comparison_anchor

    turns = [_turn(*PING_AN, query="What's Ping An's P/B?")]
    current = [{"canonical_name": "贵州茅台", "symbol": "600519.SH", "entity_type": "stock"}]
    rewritten, reason = resolve_comparison_anchor(query, turns, current)
    assert reason == "comparison_anchor:+中国平安"
    assert "中国平安" in rewritten and "Moutai" in rewritten


def test_compare_it_session_compares_both_and_the_next_which_keeps_both(agent):
    session = "r5-c5"
    agent.chat("How much did Ping An earn last year?", session_id=session)
    compared = agent.chat("Compare it against Kweichow Moutai", session_id=session)
    symbols = {call["arguments"].get("target") for call in compared["tool_calls"]}
    assert {"601318.SH", "600519.SH"} <= symbols, compared["route_reasons"]
    which = agent.chat("Which of the two looks cheaper?", session_id=session)
    assert {"601318.SH", "600519.SH"} <= {call["arguments"].get("target") for call in which["tool_calls"]}


# --------------------------------------------------------------------------- C7: "three of them" after two


def test_three_of_them_after_two_targets_uses_both_and_says_so(agent):
    from query_intelligence.agent.memory import resolve_group_reference

    rewritten, reason = resolve_group_reference("这三只谁的市净率最低", [_turn(*PING_AN), _turn(*WULIANGYE)])
    assert rewritten == "中国平安和五粮液谁的市净率最低"
    assert reason == "group_reference_count_mismatch:这三只->中国平安和五粮液"
    session = "r5-c7"
    agent.chat("中国平安和五粮液的市净率各是多少", session_id=session)
    result = agent.chat("这三只谁的市净率最低", session_id=session)
    assert "只讨论过中国平安和五粮液" in str(result["answer"])
    assert {"601318.SH", "000858.SZ"} <= {call["arguments"].get("target") for call in result["tool_calls"]}


@pytest.mark.parametrize(
    "query", ["三个月来的走势如何", "给我三只下周必涨的股票", "推荐三只白酒股", "Over the three months?"]
)
def test_counts_that_are_not_references(query):
    from query_intelligence.agent.memory import resolve_group_reference

    assert resolve_group_reference(query, [_turn(*PING_AN), _turn(*WULIANGYE), _turn(*MOUTAI)]) is None
