from __future__ import annotations

import pytest

from query_intelligence.service import build_demo_service


STOCK_SCOPE_SOURCES = {"market_api", "fundamental_sql", "news", "announcement", "research_note"}
FUND_MECHANISM_SOURCES = {"faq", "product_doc", "research_note"}


@pytest.fixture(scope="module")
def service():
    return build_demo_service()


def _source_types(items: list[dict]) -> set[str]:
    return {item["source_type"] for item in items}


@pytest.mark.parametrize(
    "query",
    [
        "你觉得中国平安怎么样？",
        "中国平安基本面怎么样？",
    ],
)
def test_stock_scope_queries_plan_full_runtime_sources(service, query: str) -> None:
    nlu_result = service.analyze_query(query, debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["product_type"]["label"] == "stock"
    assert STOCK_SCOPE_SOURCES <= set(nlu_result["source_plan"])
    assert retrieval_result["coverage"]["fundamentals"] is True
    assert "fundamental_sql" in _source_types(retrieval_result["structured_data"])
    assert "research_note" in _source_types(retrieval_result["documents"])


@pytest.mark.parametrize(
    "query",
    [
        "证券ETF费率包含哪些？",
        "ETF和LOF有什么区别？",
    ],
)
def test_etf_mechanism_queries_prefer_product_sources_without_fundamentals(service, query: str) -> None:
    nlu_result = service.analyze_query(query, debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["product_type"]["label"] == "etf"
    if query == "ETF和LOF有什么区别？":
        assert nlu_result["entities"] == []
    assert FUND_MECHANISM_SOURCES <= set(nlu_result["source_plan"])
    assert "fundamental_sql" not in nlu_result["source_plan"]
    assert "fundamental_sql" not in retrieval_result["executed_sources"]
    assert retrieval_result["coverage"]["product_mechanism"] is True


def test_index_why_query_plans_market_news_research_industry_and_macro(service) -> None:
    nlu_result = service.analyze_query("沪深300最近为什么跌？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["product_type"]["label"] == "index"
    assert set(nlu_result["source_plan"]) == {"market_api", "news", "research_note", "industry_sql", "macro_sql"}
    assert "fundamental_sql" not in nlu_result["source_plan"]
    assert "announcement" not in nlu_result["source_plan"]
    assert {"market_api", "industry_sql", "macro_sql"} <= set(retrieval_result["executed_sources"])
    assert retrieval_result["coverage"]["price"] is True
    assert retrieval_result["coverage"]["macro"] is True


def test_industry_hold_query_keeps_scope_sources_without_stock_entity(service) -> None:
    nlu_result = service.analyze_query("白酒板块还能拿吗？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["entities"] == []
    assert "missing_entity" not in nlu_result["missing_slots"]
    assert {"news", "research_note", "industry_sql", "macro_sql"} <= set(nlu_result["source_plan"])
    assert {"research_note", "industry_sql", "macro_sql"} <= set(retrieval_result["executed_sources"])
    assert retrieval_result["coverage"]["industry"] is True


def test_cpi_macro_query_filters_macro_sql_to_matching_indicator(service) -> None:
    nlu_result = service.analyze_query("CPI上行对A股有什么影响？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert set(nlu_result["source_plan"]) == {"macro_sql", "news", "research_note", "industry_sql"}
    assert "fundamental_sql" not in nlu_result["source_plan"]
    assert "announcement" not in nlu_result["source_plan"]
    assert {"macro_sql", "news", "research_note"} <= set(nlu_result["source_plan"])
    macro_ids = [item["evidence_id"] for item in retrieval_result["structured_data"] if item["source_type"] == "macro_sql"]
    assert macro_ids == ["macro_CPI_CN"]
    assert retrieval_result["coverage"]["macro"] is True


def test_policy_industry_query_retrieves_macro_and_industry_scope(service) -> None:
    nlu_result = service.analyze_query("降准利好哪些板块？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert {"macro_sql", "news", "research_note", "industry_sql"} <= set(nlu_result["source_plan"])
    assert {"macro_sql", "industry_sql"} <= set(retrieval_result["executed_sources"])
    macro_ids = {item["evidence_id"] for item in retrieval_result["structured_data"] if item["source_type"] == "macro_sql"}
    assert macro_ids == {"macro_M2_CN", "macro_CN10Y"}
    assert retrieval_result["coverage"]["industry"] is True
