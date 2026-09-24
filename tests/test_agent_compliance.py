from __future__ import annotations

from datetime import date

import pytest

from query_intelligence.agent.compliance import apply_compliance, contains_trading_instruction
from query_intelligence.agent.evidence import AgentEvidence
from query_intelligence.chatbot import DEFAULT_RISK_DISCLAIMER_EN, DEFAULT_RISK_DISCLAIMER_ZH

TRADING_DAY = date(2026, 9, 24)  # Thursday
WEEKEND = date(2026, 9, 26)


def _nlu(style="fact", flags=()):
    return {"question_style": style, "risk_flags": list(flags), "product_type": {"label": "stock"}}


def _market(as_of="2026-04-22"):
    return AgentEvidence(evidence_id="price_600519.SH", kind="structured", source_type="market_api", as_of=as_of)


@pytest.mark.parametrize(
    "text",
    [
        "建议逢低买入贵州茅台。",
        "可以考虑加仓。",
        "目标价 2000 元。",
        "强烈推荐这只股票",
        "应该清仓离场",
        "You should buy this stock now.",
        "We recommend selling before earnings.",
        "Strong buy with a price target of 2000.",
    ],
)
def test_trading_instructions_are_detected(text):
    assert contains_trading_instruction(text)


@pytest.mark.parametrize(
    "text", ["收盘价为 1409.5 元。", "净利润同比增长 4.5%。", "The stock closed at 1409.5.", "不构成买卖建议。"]
)
def test_neutral_statements_are_not_flagged(text):
    assert not contains_trading_instruction(text)


def test_trading_sentences_removed_and_replaced_with_neutral_statement():
    answer = {
        "answer": "茅台收盘价 1409.5 元。建议逢低买入。",
        "key_points": ["收盘价 1409.5 元", "目标价 2000 元"],
        "evidence_used": ["price_600519.SH"],
    }

    guarded, notes = apply_compliance(answer, query="茅台怎么样", nlu_result=_nlu(), today=TRADING_DAY)

    assert "建议逢低买入" not in guarded["answer"]
    assert "不构成买卖建议" in guarded["answer"]
    assert guarded["key_points"] == ["收盘价 1409.5 元"]
    assert "removed_trading_instruction" in notes
    assert guarded["risk_disclaimer"] == DEFAULT_RISK_DISCLAIMER_ZH


def test_advice_questions_get_conditional_prefix_and_limitation():
    answer = {"answer": "茅台收盘价 1409.5 元，值得继续持有。", "key_points": [], "evidence_used": []}

    guarded, notes = apply_compliance(
        answer,
        query="茅台还值得持有吗",
        nlu_result=_nlu(style="advice", flags=["investment_advice_like"]),
        today=TRADING_DAY,
    )

    assert guarded["answer"].startswith("基于当前证据只能做条件性判断")
    assert "值得继续持有" not in guarded["answer"]
    assert "问题包含投资建议或预测属性" in guarded["limitations"]
    assert "conditional_prefix" in notes and "softened_judgment_or_causal_language" in notes


def test_why_questions_with_tool_failures_soften_causal_language():
    answer = {"answer": "主要原因是业绩下滑导致股价下跌。", "key_points": [], "evidence_used": []}

    guarded, _notes = apply_compliance(
        answer,
        query="茅台为什么跌",
        nlu_result=_nlu(style="why"),
        tool_failures=["get_price_history: timeout"],
        today=TRADING_DAY,
    )

    assert "可能相关的因素包括" in guarded["answer"]
    assert "导致" not in guarded["answer"]
    assert "get_price_history: timeout" in guarded["limitations"]


def test_freshness_note_for_stale_quote_on_trading_day():
    answer = {"answer": "收盘价 1409.5 元。", "key_points": ["收盘价 1409.5 元"], "evidence_used": []}

    guarded, notes = apply_compliance(
        answer, query="茅台今天涨了吗", nlu_result=_nlu(), market_evidence=[_market()], today=TRADING_DAY
    )

    assert guarded["key_points"][0] == "最新可用行情日期为 2026-04-22，不是今日（2026-09-24）实时行情。"
    assert guarded["answer"] == "收盘价 1409.5 元。"
    assert "market_freshness" in notes


def test_freshness_note_on_non_trading_day_and_missing_quote():
    weekend, _ = apply_compliance(
        {"answer": "x", "key_points": []},
        query="What is Moutai's price today?",
        nlu_result=_nlu(),
        market_evidence=[_market()],
        today=WEEKEND,
    )
    missing, _ = apply_compliance(
        {"answer": "x", "key_points": []}, query="茅台现在多少钱", nlu_result=_nlu(), today=TRADING_DAY
    )

    assert weekend["key_points"][0].startswith("Today (2026-09-26) is not a regular A-share trading day")
    assert weekend["risk_disclaimer"] == DEFAULT_RISK_DISCLAIMER_EN
    assert missing["key_points"][0].startswith("未获取到今日（2026-09-24）实时行情")


def test_non_current_questions_have_no_freshness_note():
    guarded, notes = apply_compliance(
        {"answer": "x", "key_points": []}, query="茅台的市盈率", nlu_result=_nlu(), market_evidence=[_market()]
    )

    assert guarded["key_points"] == [] and "market_freshness" not in notes


def test_disclaimer_with_trading_language_is_replaced():
    guarded, _ = apply_compliance(
        {"answer": "x", "risk_disclaimer": "Strong buy!"}, query="How is Moutai?", nlu_result=_nlu()
    )

    assert guarded["risk_disclaimer"] == DEFAULT_RISK_DISCLAIMER_EN
