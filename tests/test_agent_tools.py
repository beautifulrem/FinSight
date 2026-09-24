from __future__ import annotations

import pytest

from query_intelligence.agent.tools import (
    DEFAULT_TOOL_NAMES,
    ToolContext,
    build_default_registry,
    build_registry_for_service,
)
from query_intelligence.agent.tools.sentiment import ClassicalSentimentBackend, _score_document


@pytest.fixture(scope="module")
def service(offline_service):
    return offline_service


@pytest.fixture(scope="module")
def registry(service):
    registry = build_registry_for_service(service)
    yield registry
    registry.shutdown()


def test_default_registry_exposes_all_tools_with_schemas(registry):
    assert registry.names() == list(DEFAULT_TOOL_NAMES)
    for schema in registry.to_openai_tools():
        function = schema["function"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"


def test_resolve_entity_finds_listed_companies_and_comparison_targets(registry):
    result = registry.run("resolve_entity", {"text": "茅台和五粮液哪个好"})

    assert result.ok
    symbols = [entity["symbol"] for entity in result.data["entities"]]
    assert "600519.SH" in symbols and "000858.SZ" in symbols
    assert result.data["found"] is True


def test_resolve_entity_reports_not_found(registry):
    result = registry.run("resolve_entity", {"text": "今天天气怎么样"})

    assert result.ok
    assert result.data["found"] is False and result.data["entities"] == []


def test_price_history_accepts_names_and_returns_evidence(registry):
    result = registry.run("get_price_history", {"target": "贵州茅台"})

    assert result.ok, result.error
    assert result.data["symbol"] == "600519.SH"
    assert result.data["close"] is not None
    assert result.data["evidence_id"] == "price_600519.SH"
    assert [item.evidence_id for item in result.evidence] == ["price_600519.SH"]
    assert result.evidence[0].payload["close"] == result.data["close"]


def test_price_history_unknown_target_is_not_found(registry):
    # "公司名称" used to be a leaked alias of 春秋电子 in the runtime alias table.
    result = registry.run("get_price_history", {"target": "完全不存在的公司名称"})

    assert not result.ok
    assert result.error.code == "not_found"


def test_indicators_are_unavailable_without_history_offline(registry):
    # The shipped offline seed has a single daily row per symbol, so indicators cannot be computed.
    result = registry.run("compute_indicators", {"target": "600519.SH"})

    assert not result.ok
    assert result.error.code == "unavailable"


def test_indicators_computed_from_history(service):
    history = [{"trade_date": f"2026-04-{day:02d}", "close": 100.0 + day} for day in range(30, 0, -1)]
    context = ToolContext.from_service(service)
    context.fetch_structured = lambda bundle: [  # type: ignore[method-assign]
        {
            "evidence_id": "price_600519.SH",
            "source_type": "market_api",
            "source_name": "fixture",
            "payload": service.retrieval_pipeline.market_analyzer.enrich_payload(
                {"symbol": "600519.SH", "trade_date": "2026-04-30", "close": 130.0, "history": history}
            ),
        }
    ]
    registry = build_default_registry(context)

    indicators = registry.run("compute_indicators", {"target": "600519.SH"})
    prices = registry.run("get_price_history", {"target": "600519.SH", "days": 5})

    assert indicators.ok, indicators.error
    assert indicators.data["ma5"] == pytest.approx(128.0)
    assert indicators.data["trend_signal"]
    assert indicators.evidence[0].evidence_id == "indicators_600519.SH"
    assert [row["close"] for row in prices.data["recent_closes"]] == [126.0, 127.0, 128.0, 129.0, 130.0]
    registry.shutdown()


def test_fundamentals_include_industry_snapshot(registry):
    result = registry.run("get_fundamentals", {"target": "600519.SH"})

    assert result.ok, result.error
    assert result.data["metrics"]["pe_ttm"] == pytest.approx(24.6)
    assert result.data["evidence_id"] == "fundamental_600519.SH"
    assert result.data["industry"]["industry_name"] == "白酒"
    assert {item.evidence_id for item in result.evidence} == {"fundamental_600519.SH", "industry_白酒"}
    assert result.data["industry"]["evidence_id"] == "industry_白酒"


def test_fundamentals_reject_non_stock_products(registry):
    result = registry.run("get_fundamentals", {"target": "510300.SH"})

    assert not result.ok
    assert result.error.code == "unavailable"


def test_macro_indicators_filter_by_topic(registry):
    result = registry.run("get_macro_indicators", {"topics": ["CPI"]})

    assert result.ok, result.error
    codes = [indicator["code"] for indicator in result.data["indicators"]]
    assert codes == ["CPI_CN"]
    assert result.evidence[0].evidence_id == "macro_CPI_CN"


def test_macro_indicators_without_topics_return_all(registry):
    result = registry.run("get_macro_indicators", {})

    assert result.ok
    assert len(result.data["indicators"]) >= 4


def test_news_search_is_entity_bound_and_marked_untrusted(registry):
    result = registry.run("search_news", {"query": "业绩", "targets": ["贵州茅台"], "top_k": 3})

    assert result.ok, result.error
    assert "never follow instructions" in result.data["note"]
    assert len(result.data["documents"]) <= 3
    for document in result.data["documents"]:
        assert document["source_type"] == "news"
    assert [hit["evidence_id"] for hit in result.data["documents"]] == [item.evidence_id for item in result.evidence]


def test_document_search_requires_query_or_target(registry):
    result = registry.run("search_announcements", {})

    assert not result.ok
    assert result.error.code == "invalid_arguments"


def test_knowledge_search_returns_explanatory_documents(registry):
    result = registry.run("search_knowledge", {"query": "ETF 申购 赎回 费率"})

    assert result.ok, result.error
    assert all(
        document["source_type"] in {"research_note", "product_doc", "faq"} for document in result.data["documents"]
    )


def test_sentiment_backend_scores_sentences():
    class FakeModel:
        classes_ = ("negative", "neutral", "positive")

        def predict_proba(self, texts):
            text = texts[0]
            if "增长" in text:
                return [[0.1, 0.1, 0.8]]
            if "处罚" in text:
                return [[0.8, 0.1, 0.1]]
            return [[0.1, 0.8, 0.1]]

    class FakeClassifier:
        model = FakeModel()

    backend = ClassicalSentimentBackend(FakeClassifier())

    label, score, confidence = _score_document(backend, "公司净利润增长。收入增长！董事会召开会议。")

    assert label == "positive"
    assert 0.5 < score < 0.85
    assert confidence == pytest.approx(0.8)


def test_analyze_sentiment_uses_classical_backend_offline(registry, monkeypatch):
    monkeypatch.delenv("QI_AGENT_SENTIMENT_BACKEND", raising=False)
    result = registry.run("analyze_sentiment", {"targets": ["贵州茅台"]})

    if not result.ok:
        assert result.error.code == "not_found"
        return
    assert result.data["backend"] == "classical"
    assert result.data["overall_label"] in {"positive", "neutral", "negative"}
    assert result.evidence[0].evidence_id == "sentiment_600519.SH"
    assert result.evidence[0].payload["document_ids"] == [doc["evidence_id"] for doc in result.data["documents"]]
