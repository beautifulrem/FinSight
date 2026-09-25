"""Opt-in live smoke tests for the data source chains (real network, not run by default).

    QI_LIVE_TESTS=1 python -m pytest -q tests/test_data_sources_live.py

``QI_INTEGRATION_TESTS=1`` (the repository's existing live-test switch) also enables them. Assertions
check the contract (a live source served fresh data with provenance), not which upstream won, because
upstream availability changes (e.g. Eastmoney quote hosts throttle bursts of requests).
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pytest

from query_intelligence.integrations.sources import SourceRuntime

pytestmark = pytest.mark.skipif(
    os.getenv("QI_LIVE_TESTS") != "1" and os.getenv("QI_INTEGRATION_TESTS") != "1",
    reason="live data source tests disabled; set QI_LIVE_TESTS=1 to enable",
)


@pytest.fixture(scope="module")
def runtime() -> SourceRuntime:
    return SourceRuntime(call_timeout_s=20.0)


def _window() -> tuple[str, str]:
    today = date.today()
    return (today - timedelta(days=120)).strftime("%Y%m%d"), today.strftime("%Y%m%d")


@pytest.mark.parametrize(
    ("symbol", "name", "product_type"),
    [
        ("600519.SH", "贵州茅台", "stock"),
        ("300750.SZ", "宁德时代", "stock"),
        ("510300.SH", "沪深300ETF", "etf"),
        ("000300.SH", "沪深300", "index"),
    ],
)
def test_live_market_chain_serves_fresh_prices_with_provenance(runtime, symbol, name, product_type):
    from query_intelligence.integrations.akshare_market_provider import AKShareMarketProvider

    provider = AKShareMarketProvider.from_import(timeout=15, runtime=runtime)
    start, end = _window()

    payload = provider.fetch_bundle(symbol, name, product_type, start, end)["payload"]

    assert payload["close"] is not None and payload["close"] > 0
    assert len(payload["history"]) >= 20
    provenance = payload["provenance"]
    assert provenance["is_live"] is True
    assert provenance["freshness"] == "fresh", provenance
    assert provenance["source"] in {"eastmoney.quote", "sina.kline", "tencent.kline", "sina.quote", "efinance"}


def test_live_stock_fundamentals_and_valuation(runtime):
    from query_intelligence.integrations.akshare_market_provider import AKShareMarketProvider

    provider = AKShareMarketProvider.from_import(timeout=15, runtime=runtime)
    start, end = _window()

    fundamentals = provider.fetch_bundle("601318.SH", "中国平安", "stock", start, end)["fundamental_payload"]

    assert fundamentals["report_date"] and fundamentals["roe"] is not None
    assert fundamentals["provenance"]["freshness"] == "fresh"
    assert fundamentals["pe_ttm"] is not None and fundamentals["pb"] is not None


def test_live_macro_indicators_are_current(runtime):
    from query_intelligence.integrations.akshare_macro_provider import AKShareMacroProvider

    provider = AKShareMacroProvider.from_import(runtime=runtime)

    items, failures = provider.fetch_indicators_with_status({"normalized_query": "cpi pmi m2 国债 lpr"})
    codes = {item["payload"]["indicator_code"] for item in items}

    assert {"CPI_CN", "PMI_CN", "M2_CN", "CN10Y", "LPR1Y_CN"} <= codes, failures
    for item in items:
        assert item["payload"]["provenance"]["freshness"] == "fresh", item["payload"]


def test_live_news_and_announcements(runtime):
    from query_intelligence.integrations.akshare_provider import AKShareNewsProvider
    from query_intelligence.integrations.announcement_sources import FallbackAnnouncementProvider

    news = AKShareNewsProvider.from_import(runtime=runtime).fetch_news("600519.SH", "贵州茅台", limit=5)
    announcements = FallbackAnnouncementProvider.build_default(
        url="https://www.cninfo.com.cn/new/hisAnnouncement/query",
        static_base="https://static.cninfo.com.cn/",
        timeout=15,
        runtime=runtime,
    ).fetch_announcements("600519.SH", limit=5)

    assert news and all(doc["payload"]["provenance"]["source"] == "eastmoney.news" for doc in news)
    assert announcements
    assert all(doc["entity_symbols"] == ["600519.SH"] for doc in announcements)
    assert announcements[0]["payload"]["provenance"]["source"] in {"cninfo.announcement", "eastmoney.announcement"}
