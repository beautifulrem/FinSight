"""Offline tests for the source-layer reliability work.

* bounded source-call pool: abandoned-call accounting, saturation back-pressure, thread bound;
* opt-in, rate-limited active health probes (``/sources/health?probe=1``);
* cross-source validation of fundamentals (Sina vs THS: range, report period, cumulative vs quarter);
* stale snapshot industry records are refreshed live or labelled, never passed off as current;
* scrape-time Prometheus collector for source/LLM breakers and the pool.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from query_intelligence.data_loader import DATA_DIR, load_structured_data
from query_intelligence.integrations.akshare_market_provider import AKShareMarketProvider
from query_intelligence.integrations.ops_metrics import OpsMetricsCollector, llm_circuit_states
from query_intelligence.integrations.sources.cache import SourceCache
from query_intelligence.integrations.sources.crosscheck import (
    SINA,
    THS,
    is_cumulative,
    recomputed_yoy,
    reconcile_fundamentals,
)
from query_intelligence.integrations.sources.health import SourceHealthRegistry
from query_intelligence.integrations.sources.probe import ActiveProber, Probe
from query_intelligence.integrations.sources.provenance import build_provenance
from query_intelligence.integrations.sources.report import sources_health_report
from query_intelligence.integrations.sources.runtime import (
    SourceCallPool,
    SourcePoolSaturatedError,
    SourceRuntime,
    SourceTimeoutError,
)
from query_intelligence.retrieval.api_retriever import APIRetriever
from query_intelligence.retrieval.deduper import Deduper
from query_intelligence.retrieval.doc_retriever import DocumentRetriever
from query_intelligence.retrieval.feature_builder import FeatureBuilder
from query_intelligence.retrieval.packager import RetrievalPackager
from query_intelligence.retrieval.pipeline import RetrievalPipeline
from query_intelligence.retrieval.query_builder import QueryBuilder
from query_intelligence.retrieval.ranker import BaselineRanker
from query_intelligence.retrieval.selector import DocumentSelector
from query_intelligence.retrieval.sql_retriever import SQLRetriever

RECENT = (date.today() - timedelta(days=1)).isoformat()


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


# ---- bounded source-call pool ---------------------------------------------------------------


def test_timed_out_call_is_counted_as_abandoned_until_it_returns():
    pool = SourceCallPool(max_workers=2)
    release = threading.Event()

    with pytest.raises(SourceTimeoutError):
        pool.run(lambda: release.wait(5), timeout_s=0.05)

    stats = pool.stats()
    assert stats["timed_out_total"] == 1 and stats["abandoned_total"] == 1
    assert stats["abandoned_running"] == 1 and stats["busy"] == 1

    release.set()
    _wait_until(lambda: pool.stats()["busy"] == 0)
    stats = pool.stats()
    assert stats["abandoned_running"] == 0 and stats["abandoned_total"] == 1
    assert stats["completed_total"] == 1
    assert pool.run(lambda: 42, timeout_s=1) == 42


def test_saturated_pool_rejects_fast_and_does_not_trip_the_breaker():
    runtime = SourceRuntime(pool=SourceCallPool(max_workers=1), call_timeout_s=0.05)
    release = threading.Event()
    with pytest.raises(SourceTimeoutError):
        runtime.call("sina.kline", lambda: release.wait(5))

    started = time.perf_counter()
    with runtime.trace() as attempts, pytest.raises(SourcePoolSaturatedError):
        runtime.call("tencent.kline", lambda: "never runs")
    assert time.perf_counter() - started < 0.05  # rejected, not queued behind the hung call
    assert attempts == ["tencent.kline:saturated"]

    tencent = next(row for row in runtime.health.snapshot() if row["source"] == "tencent.kline")
    assert tencent["circuit"] == "closed" and tencent["calls"] == 0  # local back-pressure is not an upstream failure
    assert runtime.pool.stats()["rejected_total"] == 1

    release.set()
    _wait_until(lambda: runtime.pool.stats()["busy"] == 0)
    assert runtime.call("tencent.kline", lambda: "ok") == "ok"


def test_hung_upstream_cannot_grow_threads_beyond_the_pool():
    pool = SourceCallPool(max_workers=3)
    release = threading.Event()
    before = threading.active_count()
    outcomes = []
    for _ in range(12):
        try:
            pool.run(lambda: release.wait(5), timeout_s=0.02)
        except (SourceTimeoutError, SourcePoolSaturatedError) as exc:
            outcomes.append(type(exc).__name__)
    assert threading.active_count() - before <= 3
    assert outcomes.count("SourceTimeoutError") == 3
    assert outcomes.count("SourcePoolSaturatedError") == 9
    release.set()
    _wait_until(lambda: pool.stats()["abandoned_running"] == 0)
    pool.shutdown()


def test_nested_guarded_call_runs_inline_instead_of_deadlocking():
    runtime = SourceRuntime(pool=SourceCallPool(max_workers=1), call_timeout_s=2)

    def outer():
        return runtime.call("tencent.quote", lambda: "inner")

    with runtime.trace() as attempts:
        assert runtime.call("tencent.kline", outer) == "inner"
    assert attempts == ["tencent.quote:ok", "tencent.kline:ok"]


def test_health_report_exposes_pool_stats():
    runtime = SourceRuntime(pool=SourceCallPool(max_workers=4))

    class _Pipeline:
        source_runtime = runtime
        market_provider = None

    report = sources_health_report(_Pipeline())
    assert report["worker_pool"]["max_workers"] == 4
    assert {"abandoned_total", "abandoned_running", "rejected_total", "busy"} <= set(report["worker_pool"])
    assert "probe" not in report


# ---- active health probe --------------------------------------------------------------------


def _probes(calls: list[str]) -> list[Probe]:
    def ok():
        calls.append("sina")
        return [1]

    def down():
        calls.append("eastmoney")
        raise ConnectionError("Remote end closed connection")

    return [Probe("sina.kline", "akshare.stock_zh_a_daily", ok), Probe("eastmoney.quote", "akshare.x", down)]


def test_active_probe_records_health_and_is_rate_limited():
    clock = FakeClock()
    calls: list[str] = []
    prober = ActiveProber(min_interval_s=60, probes_factory=lambda: _probes(calls), clock=clock)
    runtime = SourceRuntime()

    class _Pipeline:
        source_runtime = runtime
        market_provider = object()

    first = sources_health_report(_Pipeline(), probe=True, prober=prober)
    assert first["probe"]["status"] == "completed"
    assert first["probe"]["ok"] == 1 and first["probe"]["total"] == 2
    by_source = {row["source"]: row for row in first["sources"]}
    assert by_source["sina.kline"]["status"] == "up"
    assert by_source["eastmoney.quote"]["status"] == "degraded"  # recorded like real traffic
    results = {row["source"]: row for row in first["probe"]["results"]}
    assert results["eastmoney.quote"]["outcome"] == "error(ConnectionError)"

    clock.advance(10)
    second = sources_health_report(_Pipeline(), probe=True, prober=prober)
    assert second["probe"]["status"] == "rate_limited"
    assert second["probe"]["retry_in_s"] == 50.0
    assert second["probe"]["last"]["ok"] == 1
    assert sorted(calls) == ["eastmoney", "sina"]  # no second round of upstream calls

    clock.advance(51)
    third = sources_health_report(_Pipeline(), probe=True, prober=prober)
    assert third["probe"]["status"] == "completed" and len(calls) == 4


def test_probe_is_skipped_when_live_data_is_off():
    class _Pipeline:
        source_runtime = SourceRuntime()
        market_provider = None

    prober = ActiveProber(probes_factory=lambda: pytest.fail("must not probe"))
    report = sources_health_report(_Pipeline(), probe=True, prober=prober)
    assert report["probe"]["status"] == "skipped"


def _app_is_wired() -> bool:
    """The API wiring lives in ``api/app.py`` (owned elsewhere); see ``deploy/patches/app-ops-wiring.patch``."""
    import inspect

    from query_intelligence.api import app as app_module

    return "OpsMetricsCollector" in inspect.getsource(app_module)


needs_app_wiring = pytest.mark.skipif(not _app_is_wired(), reason="api/app.py ops wiring not applied yet")


@needs_app_wiring
def test_sources_health_endpoint_accepts_probe_parameter():
    from fastapi.testclient import TestClient

    from query_intelligence.api.app import create_app

    class _Service:
        def __init__(self) -> None:
            self.retrieval_pipeline = _pipeline()

    client = TestClient(create_app(service=_Service(), app_config={"deepseek": {"api_key": ""}}))
    passive = client.get("/sources/health").json()
    assert "probe" not in passive and "worker_pool" in passive
    probed = client.get("/sources/health?probe=1").json()
    assert probed["probe"]["status"] == "skipped"  # offline pipeline: no upstream traffic


# ---- cross-source validation of fundamentals ------------------------------------------------

YI = 100_000_000


def _ths_wuliangye() -> dict[str, dict]:
    """THS levels and growth for 000858 as fetched on 2026-09-26 (cumulative, 亿元)."""
    rows = {
        "2025-03-31": (170.86, -50.95, 44.16, -68.56),
        "2025-06-30": (235.10, -53.58, 46.24, -75.74),
        "2025-09-30": (306.38, -54.89, 64.75, -74.03),
        "2025-12-31": (405.29, -54.55, 89.54, -71.89),
        "2026-03-31": (228.38, 33.67, 80.63, 82.57),
        "2026-06-30": (284.17, 20.87, 87.53, 89.30),
    }
    return {
        period: {
            "revenue": revenue * YI,
            "revenue_yoy": revenue_yoy,
            "net_profit": profit * YI,
            "netprofit_yoy": profit_yoy,
            "roe": 7.14,
            "grossprofit_margin": 80.29,
            "eps": None,
            "profit_dedt": None,
        }
        for period, (revenue, revenue_yoy, profit, profit_yoy) in rows.items()
    }


def _sina_wuliangye() -> dict[str, dict]:
    """Sina indicators for the same stock and day: growth disagrees with THS and the company's release."""
    return {
        "2026-03-31": {"revenue_yoy": -38.176, "netprofit_yoy": -45.8368, "roe": 6.30, "grossprofit_margin": None},
        "2026-06-30": {
            "revenue_yoy": -46.1509,
            "netprofit_yoy": -55.3213,
            "roe": 7.39,
            "grossprofit_margin": None,
            "eps": 2.25,
        },
    }


def _leaf_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [leaf for nested in value.values() for leaf in _leaf_values(nested)]
    if isinstance(value, list):
        return [leaf for nested in value for leaf in _leaf_values(nested)]
    return [value]


def test_b11_disagreement_is_resolved_against_reported_levels():
    report, check = reconcile_fundamentals(_sina_wuliangye(), _ths_wuliangye(), primary=SINA)

    assert check.status == "disagree_resolved"
    assert check.served_source == THS and check.compared_with == SINA
    assert check.disagreeing_fields == ["revenue_yoy", "netprofit_yoy"]
    assert check.resolution == {
        "revenue_yoy": "ths.finance:matches_reported_levels",
        "netprofit_yoy": "ths.finance:matches_reported_levels",
    }
    assert check.level_consistent["ths.finance:revenue_yoy"] is True
    assert check.level_consistent["sina.finance:revenue_yoy"] is False
    assert report["report_date"] == "2026-06-30"
    assert report["revenue_yoy"] == 20.87 and report["netprofit_yoy"] == 89.30
    assert report["eps"] == 2.25  # same-period gap filled from the other source
    assert "eps" in check.filled_from_other


def test_cross_check_metadata_carries_no_numbers():
    _report, check = reconcile_fundamentals(_sina_wuliangye(), _ths_wuliangye())
    metadata = check.as_metadata()
    for leaf in _leaf_values(metadata):
        assert leaf is None or isinstance(leaf, str | bool)
        if isinstance(leaf, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", leaf):
            assert not re.search(r"\d", leaf), leaf
    assert "不一致" in metadata["note"] and "同花顺" in metadata["note"]


def test_levels_are_cumulative_and_yoy_is_recomputed_both_ways():
    ths = _ths_wuliangye()
    assert is_cumulative(ths) is True
    recomputed = recomputed_yoy(ths, "2026-06-30", "revenue")
    assert recomputed["cumulative"] == pytest.approx(20.87, abs=0.01)
    # Q2 alone: (284.17-228.38) / (235.10-170.86) - 1
    assert recomputed["single_quarter"] == pytest.approx(-13.15, abs=0.01)

    quarterly = {period: dict(row) for period, row in ths.items()}
    quarterly["2026-06-30"]["revenue"] = 55.79 * YI  # a single-quarter level below Q1: not cumulative
    assert is_cumulative(quarterly) is False


def test_single_quarter_convention_is_recognised():
    ths = _ths_wuliangye()
    sina = {"2026-06-30": {"revenue_yoy": -13.15, "netprofit_yoy": None, "roe": 7.39}}
    _report, check = reconcile_fundamentals(sina, ths)
    assert check.conventions["sina.finance:revenue_yoy"] == "single_quarter"
    assert check.conventions["ths.finance:revenue_yoy"] == "cumulative"
    assert check.status == "disagree_resolved" and check.served_source == THS


def test_agreement_period_mismatch_single_source_and_range_checks():
    ths = _ths_wuliangye()
    agreeing = {"2026-06-30": {"revenue_yoy": 20.5, "netprofit_yoy": 89.0, "roe": 7.39, "grossprofit_margin": None}}
    report, check = reconcile_fundamentals(agreeing, ths)
    assert check.status == "agree" and check.served_source == SINA
    assert report["revenue_yoy"] == 20.5 and report["grossprofit_margin"] == 80.29

    older = {"2026-03-31": {"revenue_yoy": 33.0}}
    report, check = reconcile_fundamentals(older, ths)
    assert check.status == "period_mismatch" and check.served_source == THS
    assert check.report_period == "2026-06-30" and check.other_period == "2026-03-31"

    report, check = reconcile_fundamentals({}, ths)
    assert check.status == "single_source" and check.served_source == THS

    broken = {"2026-06-30": {"revenue_yoy": -180.0, "roe": 950.0, "netprofit_yoy": 88.9}}
    report, check = reconcile_fundamentals(broken, None)
    assert report["revenue_yoy"] is None and report["roe"] is None and report["netprofit_yoy"] == 88.9
    assert check.out_of_range == ["sina.finance:revenue_yoy", "sina.finance:roe"]

    assert reconcile_fundamentals(None, None) == (None, None)


class _DisagreeingModule:
    @staticmethod
    def stock_zh_a_hist(symbol, period, start_date, end_date, adjust, timeout=None):
        return [{"日期": RECENT, "开盘": 1, "最高": 1, "最低": 1, "收盘": 1, "涨跌幅": 0, "成交量": 1, "成交额": 1}]

    @staticmethod
    def stock_profile_cninfo(symbol):
        return [{"A股简称": "五粮液", "所属行业": "酒、饮料和精制茶制造业"}]

    @staticmethod
    def stock_financial_analysis_indicator(symbol, start_year):
        return [
            {
                "日期": date(2026, 6, 30),
                "净资产收益率(%)": 7.39,
                "主营业务收入增长率(%)": -46.1509,
                "净利润增长率(%)": -55.3213,
            }
        ]

    @staticmethod
    def stock_financial_abstract_ths(symbol, indicator):
        return [
            {
                "报告期": period,
                "营业总收入": f"{row['revenue'] / YI:.2f}亿",
                "营业总收入同比增长率": f"{row['revenue_yoy']}%",
                "净利润": f"{row['net_profit'] / YI:.2f}亿",
                "净利润同比增长率": f"{row['netprofit_yoy']}%",
                "净资产收益率": "7.14%",
                "销售毛利率": "80.29%",
            }
            for period, row in _ths_wuliangye().items()
        ]

    @staticmethod
    def stock_value_em(symbol):
        return [{"数据日期": date(2026, 9, 24), "PE(TTM)": 15.1, "市净率": 3.2}]


def test_provider_cross_checks_and_flags_disagreement_in_provenance_and_warnings():
    provider = AKShareMarketProvider(
        ak_module=_DisagreeingModule(), max_retries=0, retry_backoff_seconds=0, cross_check_fundamentals=True
    )
    fundamentals = provider.fetch_bundle("000858.SZ", "五粮液", "stock")["fundamental_payload"]

    assert fundamentals["revenue_yoy"] == pytest.approx(20.87)
    assert fundamentals["netprofit_yoy"] == pytest.approx(89.30)
    assert fundamentals["provenance"]["source"] == "ths.finance"
    assert fundamentals["provider_endpoint"] == "akshare.stock_financial_abstract_ths"
    cross = fundamentals["provenance"]["cross_check"]
    assert cross["status"] == "disagree_resolved" and cross["compared_with"] == "sina.finance"
    assert "新浪财经" in fundamentals["provenance"]["note"]
    assert fundamentals["provider_warnings"] == [
        "fundamentals_cross_source_disagree_resolved:000858:revenue_yoy,netprofit_yoy:served=ths.finance"
    ]
    attempts = fundamentals["provenance"]["attempts"]
    assert "sina.finance:ok" in attempts and "ths.finance:ok" in attempts  # both fetched
    json.dumps(fundamentals, allow_nan=False, default=str)


def test_provider_without_cross_check_keeps_sina_first_behaviour():
    provider = AKShareMarketProvider(ak_module=_DisagreeingModule(), max_retries=0, retry_backoff_seconds=0)
    fundamentals = provider.fetch_bundle("000858.SZ", "五粮液", "stock")["fundamental_payload"]
    assert fundamentals["provenance"]["source"] == "sina.finance"
    assert "cross_check" not in fundamentals["provenance"]
    assert all(not item.startswith("ths.finance") for item in fundamentals["provenance"]["attempts"])


# ---- stale snapshot industry records ----------------------------------------------------------


def _pipeline() -> RetrievalPipeline:
    structured = load_structured_data()
    return RetrievalPipeline(
        query_builder=QueryBuilder(),
        doc_retriever=DocumentRetriever(json.loads((Path(DATA_DIR) / "documents.json").read_text(encoding="utf-8"))),
        sql_retriever=SQLRetriever(structured),
        api_retriever=APIRetriever(structured),
        feature_builder=FeatureBuilder(),
        ranker=BaselineRanker(),
        deduper=Deduper(),
        selector=DocumentSelector(),
        packager=RetrievalPackager(),
    )


def _industry_bundle() -> dict:
    return {
        "query_id": "q",
        "normalized_query": "贵州茅台所在行业最近表现",
        "keywords": [],
        "entity_names": ["贵州茅台"],
        "symbols": ["600519.SH"],
        "industry_terms": [],
        "source_plan": ["industry_sql"],
        "product_type": "stock",
        "intent_labels": [],
        "topic_labels": [],
    }


class _IndustryProvider:
    def __init__(self, live: dict | None) -> None:
        self.live = live
        self.board_calls: list[str] = []

    def fetch_bundle(self, symbol, canonical_name, product_type, start_date, end_date):
        return {
            "source_type": "market_api",
            "source_name": "akshare_sina",
            "payload": {
                "symbol": symbol,
                "trade_date": RECENT,
                "close": 1.0,
                "history": [],
                "provenance": build_provenance(source="sina.kline", kind="market", as_of=RECENT),
            },
            "fundamental_payload": {},
        }

    def fetch_industry_board(self, name: str):
        self.board_calls.append(name)
        if self.live is None:
            raise ConnectionError("10jqka blocked")
        return {**self.live, "industry_name": name}


def _live_board() -> dict:
    return {
        "source_name": "akshare",
        "provider_endpoint": "akshare.stock_board_industry_index_ths",
        "trade_date": RECENT,
        "close": 1922.13,
        "pct_change": -2.69,
        "provenance": build_provenance(source="ths.industry", kind="industry", as_of=RECENT),
    }


def test_offline_mode_keeps_snapshot_industry_unchanged():
    items = _pipeline()._fetch_structured_items(_industry_bundle())
    industry = next(item for item in items if item["evidence_id"] == "industry_白酒")
    assert industry["payload"]["provenance"]["mode"] == "snapshot"
    assert "provider_warnings" not in industry["payload"]


def test_stale_snapshot_industry_is_replaced_by_live_board_and_cached():
    pipeline = _pipeline()
    provider = _IndustryProvider(_live_board())
    pipeline.market_provider = provider

    items = pipeline._fetch_structured_items(_industry_bundle())
    industry = next(item for item in items if item["evidence_id"] == "industry_白酒")
    assert industry["payload"]["trade_date"] == RECENT
    assert industry["payload"]["pct_change"] == -2.69
    assert "pe" not in industry["payload"]  # no stale snapshot fields mixed into the live record
    assert industry["payload"]["provenance"]["source"] == "ths.industry"
    assert industry["payload"]["provenance"]["is_live"] is True

    pipeline._fetch_structured_items(_industry_bundle())
    assert provider.board_calls == ["白酒"]  # second query served from the TTL cache


def test_stale_snapshot_industry_is_labelled_when_live_refresh_fails():
    pipeline = _pipeline()
    pipeline.market_provider = _IndustryProvider(None)

    items = pipeline._fetch_structured_items(_industry_bundle())
    industry = next(item for item in items if item["evidence_id"] == "industry_白酒")
    snapshot_date = load_structured_data()["industry_sql"]["白酒"]["trade_date"]
    assert industry["payload"]["provider_warnings"] == [f"industry_snapshot_stale:白酒:{snapshot_date}"]
    provenance = industry["payload"]["provenance"]
    assert provenance["mode"] == "snapshot" and provenance["freshness"] == "stale"
    assert provenance["fallback_reason"] == "live industry index unavailable"
    assert "勿当作今日行情" in provenance["note"]


class _IndustryIndexModule:
    @staticmethod
    def stock_board_industry_index_ths(symbol, start_date, end_date):
        return [
            {"日期": date(2026, 9, 23), "开盘价": 1974.0, "收盘价": 1975.245, "成交额": 9.95e9},
            {"日期": date(2026, 9, 24), "开盘价": 1966.468, "收盘价": 1922.131, "成交额": 8.15e9},
        ]


def test_provider_fetches_ths_industry_board_with_provenance():
    provider = AKShareMarketProvider(ak_module=_IndustryIndexModule(), max_retries=0, retry_backoff_seconds=0)
    board = provider.fetch_industry_board("白酒")
    assert board["trade_date"] == "2026-09-24"
    assert board["pct_change"] == pytest.approx(-2.69, abs=0.01)
    assert board["provenance"]["source"] == "ths.industry"
    assert board["provenance"]["endpoint"] == "akshare.stock_board_industry_index_ths"
    assert AKShareMarketProvider(ak_module=object()).fetch_industry_board("白酒") is None


# ---- scrape-time metrics ----------------------------------------------------------------------


class _FailingClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.fail = True

    def chat(self, messages, tools=None, **kwargs):
        from query_intelligence.agent.llm import AssistantTurn, LLMError

        if self.fail:
            raise LLMError("HTTP 400 invalid model", status_code=400)
        return AssistantTurn(content="ok", model=self.model)


def test_llm_breaker_state_goes_open_half_open_closed():
    from query_intelligence.agent.llm import FallbackLLM

    clock = FakeClock()
    primary, backup = _FailingClient("primary"), _FailingClient("backup")
    backup.fail = False
    llm = FallbackLLM([primary, backup], failure_threshold=3, cooldown_s=60, clock=clock)

    for _ in range(3):
        assert llm.chat([{"role": "user", "content": "x"}]).model == "backup"
    states = {row["model"]: row["state"] for row in llm_circuit_states(llm)}
    assert states == {"primary": "open", "backup": "closed"}

    clock.advance(61)
    assert {row["model"]: row["state"] for row in llm_circuit_states(llm)}["primary"] == "half_open"

    primary.fail = False
    assert llm.chat([{"role": "user", "content": "x"}]).model == "primary"  # the trial call succeeds
    assert {row["model"]: row["state"] for row in llm_circuit_states(llm)}["primary"] == "closed"


def test_ops_collector_renders_source_pool_and_llm_metrics():
    from prometheus_client import CollectorRegistry, generate_latest

    from query_intelligence.agent.llm import FallbackLLM

    clock = FakeClock()
    runtime = SourceRuntime(
        health=SourceHealthRegistry(failure_threshold=1, clock=clock),
        cache=SourceCache(clock=clock),
        pool=SourceCallPool(max_workers=2),
    )
    with pytest.raises(ConnectionError):
        runtime.call("eastmoney.quote", lambda: (_ for _ in ()).throw(ConnectionError("down")))
    runtime.call("sina.kline", lambda: 1)
    primary, backup = _FailingClient("primary"), _FailingClient("backup")
    backup.fail = False
    llm = FallbackLLM([primary, backup], failure_threshold=1, clock=clock)
    llm.chat([{"role": "user", "content": "x"}])

    registry = CollectorRegistry()
    registry.register(OpsMetricsCollector(lambda: runtime, lambda: llm))
    text = generate_latest(registry).decode()

    assert 'finsight_source_circuit_state{source="eastmoney.quote"} 2.0' in text
    assert 'finsight_source_circuit_state{source="sina.kline"} 0.0' in text
    assert 'finsight_source_calls_total{outcome="failure",source="eastmoney.quote"} 1.0' in text
    assert "finsight_source_pool_workers 2.0" in text
    assert "finsight_source_pool_abandoned_total 0.0" in text
    assert 'finsight_llm_circuit_state{model="primary"} 2.0' in text
    assert 'finsight_llm_client_calls_total{model="backup"} 1.0' in text


@needs_app_wiring
def test_metrics_endpoint_includes_ops_metrics():
    from fastapi.testclient import TestClient

    from query_intelligence.api.app import create_app

    class _Service:
        def __init__(self) -> None:
            self.retrieval_pipeline = _pipeline()

    client = TestClient(create_app(service=_Service(), app_config={"deepseek": {"api_key": ""}}))
    body = client.get("/metrics").text
    assert "finsight_source_circuit_state" in body and "finsight_source_pool_busy" in body
