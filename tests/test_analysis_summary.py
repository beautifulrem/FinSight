"""Pytest tests for the retrieval pipeline analysis_summary feature."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QI_USE_LIVE_MARKET", "1")
os.environ.setdefault("QI_USE_LIVE_MACRO", "1")
os.environ.setdefault("QI_USE_LIVE_NEWS", "1")

from query_intelligence.contracts import RetrievalResult
from query_intelligence.service import build_default_service, clear_service_caches


TEST_CASES = [
    {
        "query": "茅台今天股价多少",
        "desc": "个股行情查询",
        "expect": {"market_signal": True, "fundamental_signal": True, "macro_signal": False},
    },
    {
        "query": "最近CPI和PMI怎么样",
        "desc": "宏观指标查询",
        "expect": {"market_signal": False, "fundamental_signal": False, "macro_signal": True},
    },
    {
        "query": "沪深300ETF今天净值多少",
        "desc": "ETF净值查询",
        "expect": {"market_signal": True, "fundamental_signal": False, "macro_signal": False},
    },
    {
        "query": "比亚迪基本面怎么样",
        "desc": "基本面查询",
        "expect": {"market_signal": True, "fundamental_signal": True, "macro_signal": False},
    },
    {
        "query": "创业板指最近走势",
        "desc": "指数走势查询",
        "expect": {"market_signal": True, "fundamental_signal": False, "macro_signal": False},
    },
]


@pytest.fixture(scope="module")
def service():
    clear_service_caches()
    return build_default_service()


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_analysis_summary_exists(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    assert summary, "analysis_summary should not be empty"


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_expected_signals_present(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    for sig_name, expected in tc["expect"].items():
        actual = summary.get(sig_name) is not None
        assert actual == expected, f"{sig_name}: expected={'exists' if expected else 'absent'}, got={'exists' if actual else 'absent'}"


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_market_signal_fields(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    ms = summary.get("market_signal")
    if not ms:
        pytest.skip("no market_signal for this query")

    assert ms["trend_signal"] in ("bullish", "bearish", "neutral")
    rsi = ms.get("rsi_14")
    assert rsi is None or 0 <= rsi <= 100
    macd = ms.get("macd")
    assert macd is None or all(k in macd for k in ("macd_line", "signal_line", "histogram"))
    bb = ms.get("bollinger")
    assert bb is None or all(k in bb for k in ("upper", "middle", "lower"))
    pct = ms.get("pct_change_nd")
    assert pct is not None and any(k in pct for k in ("pct_3d", "pct_5d", "pct_10d", "pct_20d"))
    assert ms.get("volatility_20d") is not None


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_macro_signal_fields(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    mcs = summary.get("macro_signal")
    if not mcs:
        pytest.skip("no macro_signal for this query")

    assert len(mcs.get("indicators", [])) > 0
    assert mcs["overall"] in ("expansionary", "contractionary", "mixed")
    for ind in mcs["indicators"]:
        assert ind.get("direction") != "unknown" or ind.get("metric_value") is None


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_fundamental_signal_fields(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    fs = summary.get("fundamental_signal")
    if not fs:
        pytest.skip("no fundamental_signal for this query")

    assert fs["valuation_assessment"] in (
        "potentially_undervalued",
        "fair_range",
        "potentially_overvalued",
        "unknown",
    )


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_data_readiness(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    readiness = summary.get("data_readiness", {})
    assert readiness, "data_readiness should exist"
    assert isinstance(readiness.get("has_price_data"), bool)
    assert isinstance(readiness.get("has_news"), bool)


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_coverage_consistency(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    summary = rr.get("analysis_summary", {})
    coverage = rr.get("coverage", {})

    ms = summary.get("market_signal")
    if ms:
        assert coverage.get("price"), "market_signal exists but coverage.price is False"

    mcs = summary.get("macro_signal")
    if mcs:
        assert coverage.get("macro"), "macro_signal exists but coverage.macro is False"


@pytest.mark.parametrize(
    "tc",
    TEST_CASES,
    ids=[tc["desc"] for tc in TEST_CASES],
)
def test_retrieval_result_pydantic(service, tc):
    raw = service.run_pipeline(tc["query"])
    rr = raw.get("retrieval_result", {})
    RetrievalResult(**rr)  # should not raise
