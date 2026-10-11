"""Round-17 rule M11 (round-13 review): position-size and allocation requests are advice.

Before round 17 "我有10万块，买多少股招行比较合适" (比较 read as a comparison), "把仓位的三成放宁德时代行吗"
and "Is it OK to put 30% of my portfolio in CATL?" got facts only, with no risk flag and no hedge. Now the NLU flags
them (``query_terms.POSITION_SIZING``), the router reads them as judgment, and the answer gets the conditional prefix
and a "FinSight 不提供仓位或买入数量建议" hedge with its limitation; a recommended share count or portfolio share is
removed from a draft. Property test: zh / en x amount, percentage and position wordings x four companies; factual
controls (the value of a stated holding, what a sum can buy, index weights, a fund's positions) are not flagged.
"""

from __future__ import annotations

import itertools

import pytest

from query_intelligence.query_terms import asks_position_size


@pytest.fixture(scope="module")
def service():
    from query_intelligence.service import build_default_service

    return build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
        offline_snapshot_fix=True,
    )


@pytest.fixture(scope="module")
def agent(service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]))
    yield AgentService(runtime, trace_sinks=[])
    runtime.close()


def _chat(agent, query: str, session: str | None = None) -> dict:
    return agent.chat(query, session_id=session or f"r17-{abs(hash(query))}")


# ---------------------------------------------------------------- M11 position size and allocation

_ZH_COMPANIES = ("招商银行", "宁德时代", "比亚迪", "紫金矿业")
_ZH_SIZING = (
    "我有10万块，买多少股{c}比较合适",
    "手里20万，{c}应该买几手",
    "{c}买多少股合适",
    "{c}仓位多少合适",
    "{c}的仓位控制在多少比较好",
    "几成仓位买{c}比较好",
    "把仓位的三成放{c}行吗",
    "三成仓位放{c}可以吗",
    "资金的30%投{c}合适吗",
    "拿一半的钱买{c}怎么样",
    "重仓{c}行不行",
    "分配多少钱给{c}合适",
)
_EN_COMPANIES = ("CATL", "BYD", "China Merchants Bank", "Zijin Mining")
_EN_SIZING = (
    "Is it OK to put 30% of my portfolio in {c}?",
    "How many shares of {c} should I buy?",
    "How much should I invest in {c}?",
    "Would it be wise to allocate half of my savings to {c}?",
    "What percentage of my portfolio should go into {c}?",
    "What position size makes sense for {c}?",
    "How big a position should I take in {c}?",
)
_CONTROLS = (
    "我有1000股{c}，现在值多少钱",
    "10万块能买多少股{c}",
    "{c}占沪深300的权重是多少",
    "{c}的营收里海外收入占比多少",
    "这只基金的仓位多少",
    "北向资金持有{c}的比例",
    "我买了2000股{c}，现在赚了多少",
    "{c}最近一个交易日的收盘价",
)
_EN_CONTROLS = (
    "I own 500 shares of {c}. What are they worth?",
    "How many shares does {c} have outstanding?",
    "What percentage of {c}'s revenue is net profit?",
    "What was {c}'s latest close?",
)


def test_position_size_wordings_are_advice_in_both_languages():
    from query_intelligence.agent.router import asks_prediction

    cases = [frame.format(c=c) for frame, c in itertools.product(_ZH_SIZING, _ZH_COMPANIES)]
    cases += [frame.format(c=c) for frame, c in itertools.product(_EN_SIZING, _EN_COMPANIES)]
    assert len(cases) == 76
    missed = [case for case in cases if not (asks_position_size(case) and asks_prediction(case))]
    assert not missed, missed


def test_factual_controls_are_not_position_size_requests():
    cases = [frame.format(c=c) for frame, c in itertools.product(_CONTROLS, _ZH_COMPANIES)]
    cases += [frame.format(c=c) for frame, c in itertools.product(_EN_CONTROLS, _EN_COMPANIES)]
    flagged = [case for case in cases if asks_position_size(case)]
    assert not flagged, flagged


@pytest.mark.parametrize(
    ("query", "zh"),
    [
        ("我有10万块，买多少股招行比较合适", True),
        ("把仓位的三成放宁德时代行吗", True),
        ("紫金矿业仓位多少合适", True),
        ("Is it OK to put 30% of my portfolio in CATL?", False),
        ("How many shares of BYD should I buy?", False),
    ],
)
def test_position_size_requests_are_flagged_and_hedged(agent, query, zh):
    result = _chat(agent, query)
    assert "investment_advice_like" in result["nlu_summary"]["risk_flags"]
    hedge = "FinSight 不提供仓位或买入数量建议" if zh else "FinSight does not give position-size or allocation advice"
    assert hedge in result["answer"]
    assert "position_size_hedge" in result["compliance_notes"]
    limitation = "问题要求给出仓位或买入数量建议" if zh else "The question asks for a position size or an allocation"
    assert any(item.startswith(limitation) for item in result["limitations"])


def test_a_holding_value_question_stays_a_lookup(agent):
    result = _chat(agent, "我有1000股招行，现在值多少钱")
    assert "position_size_hedge" not in result["compliance_notes"]
    assert "41260" in result["answer"]


def test_a_recommended_share_count_or_portfolio_share_is_removed_from_a_draft():
    from query_intelligence.agent.compliance import apply_compliance, contains_trading_instruction

    for sentence in (
        "按最新收盘价计算，买2000股比较合适。",
        "仓位不超过30%为宜。",
        "Buying about 500 shares would be reasonable.",
        "Keep it under 10% of your portfolio.",
    ):
        assert contains_trading_instruction(sentence), sentence
    answer, notes = apply_compliance(
        {"answer": "招商银行最新收盘价为 41.26 元 [price_600036.SH]。买2000股比较合适。", "limitations": []},
        query="我有10万块，买多少股招行比较合适",
        nlu_result={"question_style": "advice", "risk_flags": ["investment_advice_like"]},
        language="zh",
    )
    assert "2000股" not in answer["answer"] and "removed_trading_instruction" in notes
    assert "position_size_hedge" in notes
    for fact in ("招商银行最新收盘价为 41.26 元。", "BYD's P/E is 25.81x.", "我持有2000股。"):
        assert not contains_trading_instruction(fact), fact
