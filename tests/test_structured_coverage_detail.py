from __future__ import annotations

from query_intelligence.contracts import RetrievalResult
from query_intelligence.retrieval.packager import RetrievalPackager


def test_retrieval_result_preserves_structured_coverage_detail() -> None:
    nlu_result = {
        "query_id": "q1",
        "product_type": {"label": "etf", "score": 0.99},
        "intent_labels": [{"label": "product_info", "score": 0.9}],
        "entities": [{"symbol": "510300.SH"}],
        "source_plan": ["market_api", "research_note"],
    }
    structured_items = [
        {"evidence_id": "price_510300.SH", "source_type": "market_api", "payload": {"symbol": "510300.SH", "close": 4.1}},
        {"evidence_id": "fund_nav_510300.SH", "source_type": "fund_nav", "payload": {"symbol": "510300.SH", "latest_nav": 4.08}},
        {"evidence_id": "fund_fee_510300.SH", "source_type": "fund_fee", "payload": {"symbol": "510300.SH", "management_fee": 0.5}},
        {"evidence_id": "fund_redemption_510300.SH", "source_type": "fund_redemption", "payload": {"symbol": "510300.SH", "redemption_status": "open"}},
        {"evidence_id": "index_daily_000300.SH", "source_type": "index_daily", "payload": {"symbol": "000300.SH", "close": 4000}},
        {"evidence_id": "index_valuation_000300.SH", "source_type": "index_valuation", "payload": {"symbol": "000300.SH", "pe": 12.5}},
        {"evidence_id": "macro_indicator_CPI_CN", "source_type": "macro_indicator", "payload": {"indicator_code": "CPI_CN", "metric_value": 0.8}},
    ]

    raw_result = RetrievalPackager().build(
        nlu_result=nlu_result,
        ranked_docs=[],
        structured_items=structured_items,
        groups=[],
        total_candidates=0,
        executed_sources=[item["source_type"] for item in structured_items],
    )
    result = RetrievalResult.model_validate(raw_result).model_dump(mode="json")

    assert result["coverage"]["price"] is True
    assert result["coverage"]["macro"] is True
    assert result["coverage_detail"] == {
        "price_history": True,
        "financials": False,
        "valuation": True,
        "industry_snapshot": False,
        "fund_nav": True,
        "fund_fee": True,
        "fund_redemption": True,
        "fund_profile": False,
        "index_daily": True,
        "index_valuation": True,
        "macro_indicator": True,
        "policy_event": False,
    }
