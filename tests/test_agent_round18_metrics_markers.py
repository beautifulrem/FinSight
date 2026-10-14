"""Round-18 rule N16 (round-14 review, also asked in rounds 12 and 13): the hedge and missing-data checks of the
evaluation read FinSight's own hedge and gap statements, not any common word.

Before, ``score_turn`` counted a turn as hedged when its answer contained "可能", "无法", "not " or "cannot" anywhere,
and as stating missing data on "没有", "不足" or "not "; a plain factual sentence passed both. Now a hedge is a
compliance note that inserted or enforced one, or one of the product's hedge sentences; missing data is one of the
product's gap statements in the answer or its limitations. ``evaluation/human/score_head_to_head.py`` keeps the
generic words: it scores other assistants, which do not write FinSight's sentences.
"""

from __future__ import annotations

import pytest

from evaluation.agent_eval.metrics import score_turn


def _answer(text: str, *, notes: list[str] | None = None, limitations: list[str] | None = None) -> dict:
    return {
        "status": "ok",
        "route": "workflow",
        "answer": text,
        "compliance_notes": notes or [],
        "limitations": limitations or [],
        "risk_disclaimer": "以上内容仅基于系统检索到的证据生成，不构成投资建议。",
    }


@pytest.mark.parametrize(
    "text",
    [
        "茅台的市盈率可能是 21.4 倍。",
        "贵州茅台无法在一天内涨停两次。",
        "Moutai cannot be bought on margin by everyone.",
        "This is not a recommendation engine, the P/E is 21.4x.",
    ],
)
def test_a_common_word_is_not_a_hedge(text):
    assert score_turn(_answer(text), {"must_hedge": True})["checks"]["hedged"] is False


@pytest.mark.parametrize(
    ("text", "notes"),
    [
        ("贵州茅台最新收盘价 1409.5 元。", ["conditional_prefix"]),
        ("贵州茅台最新收盘价 1409.5 元。", ["position_size_hedge"]),
        ("以上证据只能提示可能的影响因素，不能据此确定单一原因。", []),
        ("This evidence only points to possible factors; it does not establish a single cause.", []),
        ("Based on the current evidence, this can only be a conditional assessment, not a definitive buy.", []),
    ],
)
def test_the_products_hedge_is_a_hedge(text, notes):
    assert score_turn(_answer(text, notes=notes), {"must_hedge": True})["checks"]["hedged"] is True


@pytest.mark.parametrize(
    "text",
    ["贵州茅台没有退市风险提示。", "五粮液净利润不足千亿。", "Moutai is not a bank.", "No, the close was 1409.5."],
)
def test_a_common_word_is_not_a_missing_data_statement(text):
    assert score_turn(_answer(text), {"must_state_missing": True})["checks"]["states_missing"] is False


@pytest.mark.parametrize(
    ("text", "limitations"),
    [
        ("本次没有检索到可用于回答该问题的证据。当前数据源中没有宁德时代（300750.SZ）的基本面数据。", []),
        ("贵州茅台最新收盘价 1409.5 元。", ["基本面数据未取到（当前数据源中没有相关记录）"]),
        ("无法计算五粮液的PEG：当前数据没有净利润增速；以下只列出可得的指标。", []),
        ("The current data sources do not include Wuliangye's revenue growth, so that part cannot be answered.", []),
        ("The data holds only the latest close for Wuliangye, so the maximum drawdown cannot be computed.", []),
    ],
)
def test_the_products_gap_statement_states_missing_data(text, limitations):
    result = score_turn(_answer(text, limitations=limitations), {"must_state_missing": True})
    assert result["checks"]["states_missing"] is True
