"""Round 15 (review-11 K17): a negative per-share gain is bracketed in the cost-basis product ("200 × (−90.5)", not
"200 × -90.5"), in both languages, and the sentence still verifies."""

from __future__ import annotations

import pytest

from query_intelligence.agent.composer import cost_basis_sentences
from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore
from query_intelligence.agent.verifier import verify_answer

_TOOL_LOG = [
    {
        "tool": "get_price_history",
        "ok": True,
        "data": {
            "symbol": "600519.SH",
            "name": "贵州茅台",
            "close": 1409.5,
            "as_of": "2026-04-22",
            "evidence_id": "price_600519.SH",
        },
    }
]


def _store() -> EvidenceStore:
    store = EvidenceStore()
    store.add(
        AgentEvidence(
            evidence_id="price_600519.SH",
            kind="structured",
            source_type="market",
            payload={"close": 1409.5, "symbol": "600519.SH"},
        )
    )
    return store


@pytest.mark.parametrize(
    ("question", "zh", "product"),
    [
        ("我200股茅台成本1500，现在盈亏多少", True, "200 × (−90.5) = -18100 元"),
        ("I hold 200 shares of Moutai at a cost of 1500; what's my P&L?", False, "200 × (−90.5) = CNY -18100"),
    ],
)
def test_a_negative_per_share_gap_is_bracketed_and_verifies(question, zh, product):
    sentence = cost_basis_sentences(_TOOL_LOG, question, zh)[0]
    assert product in sentence, sentence
    assert "× -" not in sentence
    report = verify_answer({"answer": sentence}, _store(), query=question, allow_derived=True)
    assert report.passed, report


def test_a_positive_per_share_gap_is_not_bracketed():
    sentence = cost_basis_sentences(_TOOL_LOG, "我200股茅台成本1400，现在盈亏多少", True)[0]
    assert "200 × 9.5 = 1900 元" in sentence, sentence
