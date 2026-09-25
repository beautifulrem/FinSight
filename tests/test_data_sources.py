"""Offline tests for robust live data acquisition: breaker, cache, fallback chains, and provenance.

No test here touches the network: providers get fake akshare modules / fake HTTP getters / fake
sessions, and pipelines are assembled from the small seed data in ``data/``.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest

from query_intelligence.data_loader import DATA_DIR, load_structured_data
from query_intelligence.integrations.akshare_macro_provider import AKShareMacroProvider
from query_intelligence.integrations.akshare_market_provider import AKShareMarketProvider
from query_intelligence.integrations.announcement_sources import (
    EastmoneyAnnouncementProvider,
    FallbackAnnouncementProvider,
)
from query_intelligence.integrations.cninfo_provider import CninfoAnnouncementProvider
from query_intelligence.integrations.sources import (
    AllSourcesFailedError,
    Candidate,
    CircuitOpenError,
    SourceCache,
    SourceHealthRegistry,
    SourceRuntime,
    SourceTimeoutError,
    build_provenance,
    freshness,
    snapshot_provenance,
)
from query_intelligence.integrations.sources.provenance import reason_zh
from query_intelligence.integrations.sources.values import to_iso_date, to_number
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

TODAY = date.today()
RECENT = (TODAY - timedelta(days=1)).isoformat()


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _runtime(clock: FakeClock | None = None, **kwargs: Any) -> SourceRuntime:
    clock = clock or FakeClock()
    health = SourceHealthRegistry(
        failure_threshold=kwargs.pop("failure_threshold", 3),
        cooldown_s=kwargs.pop("cooldown_s", 60.0),
        max_cooldown_s=kwargs.pop("max_cooldown_s", 600.0),
        clock=clock,
    )
    return SourceRuntime(health=health, cache=SourceCache(clock=clock), **kwargs)


def _fail(message: str = "boom"):
    def fetch():
        raise ConnectionError(message)

    return fetch


def _numbers_in(value: Any) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, int | float):
        found.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            found.extend(_numbers_in(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_numbers_in(nested))
    elif isinstance(value, str):
        try:
            float(value.strip().rstrip("%").replace(",", ""))
        except ValueError:
            pass
        else:
            found.append(value)
    return found


# ---- circuit breaker ---------------------------------------------------------------


def test_circuit_opens_after_consecutive_failures_and_short_circuits():
    clock = FakeClock()
    health = SourceHealthRegistry(failure_threshold=3, cooldown_s=60, clock=clock)
    for _ in range(3):
        health.acquire("eastmoney.quote")
        health.record_failure("eastmoney.quote", 12.0, "ConnectionError: reset")

    with pytest.raises(CircuitOpenError):
        health.acquire("eastmoney.quote")
    row = next(item for item in health.snapshot() if item["source"] == "eastmoney.quote")
    assert row["circuit"] == "open" and row["status"] == "down"
    assert row["failures"] == 3 and row["short_circuited"] == 1
    assert row["last_error"] == "ConnectionError: reset"
    assert 0 < row["retry_in_s"] <= 60


def test_half_open_trial_success_closes_and_failure_doubles_cooldown():
    clock = FakeClock()
    health = SourceHealthRegistry(failure_threshold=1, cooldown_s=10, max_cooldown_s=25, clock=clock)
    health.acquire("sina.kline")
    health.record_failure("sina.kline", 5.0, "timeout")

    clock.advance(10)
    health.acquire("sina.kline")  # half-open trial admitted
    with pytest.raises(CircuitOpenError):
        health.acquire("sina.kline")  # only one trial at a time
    health.record_failure("sina.kline", 5.0, "timeout")
    clock.advance(10)
    with pytest.raises(CircuitOpenError):
        health.acquire("sina.kline")  # cooldown doubled to 20s
    clock.advance(10)
    health.acquire("sina.kline")
    health.record_success("sina.kline", 3.0)

    row = next(item for item in health.snapshot() if item["source"] == "sina.kline")
    assert row["circuit"] == "closed" and row["status"] == "up"
    assert row["consecutive_failures"] == 0


def test_health_snapshot_lists_catalogued_sources_before_any_call():
    rows = SourceHealthRegistry().snapshot()
    by_source = {row["source"]: row for row in rows}
    assert {"eastmoney.quote", "sina.kline", "tencent.kline", "cninfo.announcement", "chinabond"} <= set(by_source)
    assert by_source["tencent.kline"]["status"] == "unknown"
    assert by_source["tencent.kline"]["label"] == "腾讯证券行情"


def test_runtime_call_enforces_hard_timeout_and_skips_open_circuit():
    runtime = _runtime(call_timeout_s=0.05, failure_threshold=1)
    calls = []

    def slow():
        calls.append(1)
        import time

        time.sleep(0.5)

    with runtime.trace() as attempts, pytest.raises(SourceTimeoutError):
        runtime.call("csindex", slow)
    with runtime.trace() as skipped, pytest.raises(CircuitOpenError):
        runtime.call("csindex", slow)

    assert attempts == ["csindex:timeout"]
    assert skipped == ["csindex:circuit_open"]
    assert len(calls) == 1  # the open circuit did not invoke the source


# ---- cache ---------------------------------------------------------------------------


def test_cache_ttl_stale_reads_and_copy_isolation():
    clock = FakeClock()
    cache = SourceCache(clock=clock)
    value = {"payload": {"close": 1.0}}
    cache.put("k", value, ttl_s=10)
    value["payload"]["close"] = 99.0  # caller mutation after put does not leak in

    hit = cache.get("k")
    assert hit == {"payload": {"close": 1.0}}
    hit["payload"]["close"] = 5.0  # mutation of a returned copy does not leak back
    assert cache.get("k")["payload"]["close"] == 1.0

    clock.advance(11)
    assert cache.get("k") is None
    assert cache.get_stale("k", max_stale_s=100)["payload"]["close"] == 1.0
    clock.advance(200)
    assert cache.get_stale("k", max_stale_s=100) is None


# ---- fallback chains -------------------------------------------------------------------


def test_chain_prefers_primary_then_falls_back_in_order():
    runtime = _runtime()
    order: list[str] = []

    def ok(name: str, value: Any):
        def fetch():
            order.append(name)
            return value

        return fetch

    primary = runtime.run_chain(
        "t", "a", [Candidate("sina.kline", "p", ok("p", 1)), Candidate("tencent.kline", "s", ok("s", 2))], ttl_s=0
    )
    assert (primary.value, primary.mode, primary.fallback_reason) == (1, "live", None)
    assert order == ["p"]

    def failing():
        order.append("f")
        raise ConnectionError("down")

    result = runtime.run_chain(
        "t",
        "b",
        [
            Candidate("eastmoney.quote", "akshare.stock_zh_a_hist", failing),
            Candidate("sina.kline", "akshare.stock_zh_a_daily", lambda: None),
            Candidate("tencent.kline", "tencent.fqkline", ok("t", 3)),
        ],
        ttl_s=0,
    )
    assert result.value == 3
    assert result.mode == "live_fallback"
    assert result.source_id == "tencent.kline"
    assert result.attempts == ["eastmoney.quote:error(ConnectionError)", "sina.kline:empty", "tencent.kline:ok"]
    assert result.fallback_reason == "eastmoney.quote:error(ConnectionError); sina.kline:empty"


def test_chain_serves_cache_then_last_known_good_then_raises():
    clock = FakeClock()
    runtime = _runtime(clock)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise TimeoutError("slow")
        return {"v": 1}

    first = runtime.run_chain("macro", "CPI", [Candidate("eastmoney.datacenter", "e", flaky)], ttl_s=60)
    cached = runtime.run_chain("macro", "CPI", [Candidate("eastmoney.datacenter", "e", flaky)], ttl_s=60)
    assert first.mode == "live" and not first.cache_hit
    assert cached.cache_hit and calls["n"] == 1

    clock.advance(61)
    stale = runtime.run_chain("macro", "CPI", [Candidate("eastmoney.datacenter", "e", flaky)], ttl_s=60)
    assert stale.mode == "last_known_good"
    assert stale.value == {"v": 1}
    assert stale.fallback_reason == "eastmoney.datacenter:timeout"

    with pytest.raises(AllSourcesFailedError) as excinfo:
        runtime.run_chain("macro", "PMI", [Candidate("eastmoney.datacenter", "e", _fail())], ttl_s=60)
    assert excinfo.value.attempts == ["eastmoney.datacenter:error(ConnectionError)"]


def test_chain_skips_source_with_open_circuit_without_calling_it():
    runtime = _runtime(failure_threshold=1)
    runtime.health.acquire("eastmoney.quote")
    runtime.health.record_failure("eastmoney.quote", 1.0, "blocked")
    called = []

    def primary():
        called.append(1)
        return 1

    result = runtime.run_chain(
        "market", "x", [Candidate("eastmoney.quote", "e", primary), Candidate("sina.kline", "s", lambda: 2)], ttl_s=0
    )
    assert result.value == 2 and not called
    assert result.fallback_reason == "eastmoney.quote:circuit_open"


# ---- provenance and value normalization ----------------------------------------------


def test_freshness_windows_per_kind():
    today = date(2026, 9, 25)
    assert freshness("market", "2026-09-24", today=today) == "fresh"
    assert freshness("market", "2026-04-22", today=today) == "stale"
    assert freshness("macro_monthly", "2026-08-01", today=today) == "fresh"
    assert freshness("fundamentals", "2026-06-30", today=today) == "fresh"
    assert freshness("market", None, today=today) == "unknown"
    assert freshness("unknown_kind", "2026-09-24", today=today) == "unknown"


def test_provenance_fields_and_note_carry_no_numbers():
    record = build_provenance(
        source="sina.kline",
        kind="market",
        as_of="20260924",
        mode="live_fallback",
        endpoint="akshare.stock_zh_a_daily",
        fallback_reason="eastmoney.quote:circuit_open",
        attempts=["eastmoney.quote:circuit_open", "sina.kline:ok"],
        today=date(2026, 9, 25),
    )
    assert record["as_of"] == "2026-09-24"
    assert record["is_live"] is True and record["freshness"] == "fresh"
    assert record["source_label"] == "新浪财经行情"
    assert record["note"] == "数据来自新浪财经行情，截至2026-09-24；因东方财富行情熔断中降级"
    assert _numbers_in(record) == []

    snapshot = snapshot_provenance(
        kind="market", as_of="2026-04-22", reason="live market data disabled", today=date(2026, 9, 25)
    )
    assert snapshot["is_live"] is False and snapshot["mode"] == "snapshot"
    assert snapshot["freshness"] == "stale"
    assert "离线快照" in snapshot["note"] and "未开启实时行情" in snapshot["note"]
    assert _numbers_in(snapshot) == []


def test_reason_zh_renders_attempt_labels():
    assert (
        reason_zh("eastmoney.quote:timeout; sina.kline:error(ConnectionError)")
        == "东方财富行情超时；新浪财经行情请求失败"
    )


def test_value_normalization_handles_placeholders_units_and_dates():
    assert to_number(float("nan")) is None
    assert to_number("445.17亿") == pytest.approx(44_517_000_000)
    assert to_number("1.47%") == pytest.approx(1.47)
    assert to_number("---（每年）") is None
    assert to_number("--") is None
    assert to_iso_date("2026年08月份") == "2026-08-01"
    assert to_iso_date("20260924") == "2026-09-24"


# ---- market provider -------------------------------------------------------------------


class _EastmoneyAndSinaDown:
    def __init__(self) -> None:
        self.hist_calls = 0

    def stock_zh_a_hist(self, symbol, period, start_date, end_date, adjust, timeout=None):
        self.hist_calls += 1
        raise ConnectionError("Remote end closed connection without response")

    @staticmethod
    def stock_zh_a_daily(symbol, start_date, end_date, adjust):
        raise ConnectionError("sina down")


class _FakeResponse:
    def __init__(self, *, payload: Any = None, text: str = "", content: bytes | None = None) -> None:
        self._payload = payload
        self.text = text
        if content is not None:
            self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def _tencent_get(url: str, headers: dict, timeout: float):
    rows = [
        [RECENT, "1250.010", "1237.000", "1256.130", "1231.050", "31239.000"],
        [(TODAY - timedelta(days=2)).isoformat(), "1240.000", "1251.240", "1255.000", "1238.000", "30000.000"],
    ]
    return _FakeResponse(payload={"code": 0, "data": {"sh600519": {"day": rows}}})


def test_market_provider_falls_back_to_tencent_with_provenance_and_volume_unit():
    provider = AKShareMarketProvider(
        ak_module=_EastmoneyAndSinaDown(), max_retries=0, retry_backoff_seconds=0, http_get=_tencent_get
    )

    bundle = provider.fetch_bundle("600519.SH", "贵州茅台", "stock")
    payload = bundle["payload"]

    assert bundle["source_name"] == "tencent"
    assert payload["close"] == 1237.0 and payload["trade_date"] == RECENT
    assert payload["pct_change_1d"] == pytest.approx(-1.1381, abs=1e-4)
    assert payload["volume_unit"] == "lot"
    assert payload["provider_endpoint"] == "tencent.fqkline"
    provenance = payload["provenance"]
    assert provenance["source"] == "tencent.kline"
    assert provenance["mode"] == "live_fallback" and provenance["is_live"] is True
    assert provenance["freshness"] == "fresh"
    assert provenance["attempts"] == [
        "eastmoney.quote:error(ConnectionError)",
        "sina.kline:error(ConnectionError)",
        "tencent.kline:ok",
    ]
    assert "东方财富行情请求失败" in provenance["note"]


def test_market_provider_circuit_breaker_stops_hammering_a_blocked_source():
    module = _EastmoneyAndSinaDown()
    provider = AKShareMarketProvider(
        ak_module=module,
        max_retries=0,
        retry_backoff_seconds=0,
        http_get=_tencent_get,
        runtime=_runtime(failure_threshold=2),
    )

    for _ in range(4):
        bundle = provider.fetch_bundle("600519.SH", "贵州茅台", "stock")

    assert module.hist_calls == 2  # the third and fourth fetches were short-circuited
    assert any("_skipped:circuit_open:eastmoney.quote" in warning for warning in bundle["provider_warnings"])
    assert bundle["payload"]["provenance"]["attempts"][0] == "eastmoney.quote:circuit_open"


class _FundamentalsModule:
    @staticmethod
    def stock_zh_a_hist(symbol, period, start_date, end_date, adjust, timeout=None):
        return [
            {
                "日期": RECENT,
                "开盘": 10.0,
                "最高": 11.0,
                "最低": 9.5,
                "收盘": 10.5,
                "涨跌幅": 1.0,
                "成交量": 100,
                "成交额": 1050,
            }
        ]

    @staticmethod
    def stock_individual_info_em(symbol, timeout=None):
        raise ConnectionError("push2 blocked")

    @staticmethod
    def stock_profile_cninfo(symbol):
        return [{"A股简称": "贵州茅台", "所属行业": "酒、饮料和精制茶制造业"}]

    @staticmethod
    def stock_financial_analysis_indicator(symbol, start_year):
        return [
            {"日期": date(2025, 12, 31), "净资产收益率(%)": 36.0, "销售毛利率(%)": 91.3, "加权每股收益(元)": 68.6},
            {
                "日期": date(2026, 6, 30),
                "净资产收益率(%)": 17.72,
                "销售毛利率(%)": math.nan,
                "加权每股收益(元)": 35.57,
                "净利润增长率(%)": -2.029,
                "主营业务收入增长率(%)": 1.4699,
                "扣除非经常性损益后的净利润(元)": 44464207646.01,
            },
        ]

    @staticmethod
    def stock_value_em(symbol):
        return [
            {"数据日期": date(2026, 9, 23), "PE(TTM)": 19.2, "市净率": 6.2},
            {"数据日期": date(2026, 9, 24), "PE(TTM)": 18.98901222, "市净率": 6.15454256},
        ]


def test_fundamentals_use_current_columns_valuation_source_and_cninfo_industry():
    provider = AKShareMarketProvider(ak_module=_FundamentalsModule(), max_retries=0, retry_backoff_seconds=0)

    bundle = provider.fetch_bundle("600519.SH", "贵州茅台", "stock")
    fundamentals = bundle["fundamental_payload"]

    assert fundamentals["report_date"] == "2026-06-30"
    assert fundamentals["roe"] == 17.72
    assert fundamentals["grossprofit_margin"] is None  # NaN in the source is reported as missing, not a number
    assert fundamentals["eps"] == 35.57
    assert fundamentals["netprofit_yoy"] == -2.029
    assert fundamentals["pe_ttm"] == pytest.approx(18.989, abs=1e-3)
    assert fundamentals["pb"] == pytest.approx(6.155, abs=1e-3)
    assert fundamentals["valuation_date"] == "2026-09-24"
    assert fundamentals["provenance"]["source"] == "sina.finance"
    assert fundamentals["valuation_provenance"]["source"] == "eastmoney.datacenter"
    json.dumps(bundle, allow_nan=False, default=str)  # no NaN leaks into JSON

    industry = bundle["payload"]["industry_snapshot"]
    assert industry["industry_name"] == "酒、饮料和精制茶制造业"
    assert industry["source_name"] == "cninfo_company_profile"
    assert industry["provider_endpoint"] == "akshare.stock_profile_cninfo"
    assert industry["coverage_level"] == "identity_only"


class _SinaFinanceDownThsUp(_FundamentalsModule):
    @staticmethod
    def stock_financial_analysis_indicator(symbol, start_year):
        raise ConnectionError("sina finance down")

    @staticmethod
    def stock_value_em(symbol):
        raise ConnectionError("datacenter down")

    @staticmethod
    def stock_financial_abstract_ths(symbol, indicator):
        return [
            {"报告期": "2026-03-31", "净利润": "272.43亿", "净资产收益率-摊薄": "10.06%"},
            {
                "报告期": "2026-06-30",
                "净利润": "445.17亿",
                "营业总收入": "922.78亿",
                "净利润同比增长率": "-1.95%",
                "销售毛利率": "89.56%",
                "净资产收益率-摊薄": "17.72%",
                "基本每股收益": "35.5700",
            },
        ]


def _tencent_quote_get(url: str, headers: dict, timeout: float):
    fields = ["1", "贵州茅台", "600519", *["0"] * 27, "20260924161444", *["0"] * 8, "18.99", *["0"] * 6, "6.15", "0"]
    return _FakeResponse(content=f'v_sh600519="{"~".join(fields)}";'.encode("gbk"))


def test_fundamentals_fall_back_to_ths_and_tencent_valuation():
    provider = AKShareMarketProvider(
        ak_module=_SinaFinanceDownThsUp(), max_retries=0, retry_backoff_seconds=0, http_get=_tencent_quote_get
    )

    fundamentals = provider.fetch_bundle("600519.SH", "贵州茅台", "stock")["fundamental_payload"]

    assert fundamentals["report_date"] == "2026-06-30"
    assert fundamentals["net_profit"] == pytest.approx(44_517_000_000)
    assert fundamentals["revenue"] == pytest.approx(92_278_000_000)
    assert fundamentals["grossprofit_margin"] == pytest.approx(89.56)
    assert fundamentals["roe"] == pytest.approx(17.72)
    assert fundamentals["provenance"]["source"] == "ths.finance"
    assert fundamentals["provenance"]["mode"] == "live_fallback"
    assert (fundamentals["pe_ttm"], fundamentals["pb"]) == (18.99, 6.15)
    assert fundamentals["valuation_provenance"]["source"] == "tencent.quote"
    assert fundamentals["valuation_date"] == "2026-09-24"


class _FundOverviewModule:
    def __init__(self) -> None:
        self.xq_calls = 0

    @staticmethod
    def fund_etf_hist_em(symbol, period, start_date, end_date, adjust):
        return [{"日期": RECENT, "收盘": 4.515, "涨跌幅": -1.63}]

    @staticmethod
    def fund_overview_em(symbol):
        return [
            {
                "基金代码": "510300（主代码）",
                "管理费率": "0.15%（每年）",
                "托管费率": "0.05%（每年）",
                "销售服务费率": "---（每年）",
                "最高赎回费率": "0.50%",
                "基金经理人": "柳军",
                "跟踪标的": "沪深300指数",
            }
        ]

    def fund_individual_detail_info_xq(self, symbol):
        self.xq_calls += 1
        raise KeyError("data")


def test_fund_details_come_from_fund_overview_and_skip_token_gated_xueqiu():
    module = _FundOverviewModule()
    provider = AKShareMarketProvider(ak_module=module, max_retries=0, retry_backoff_seconds=0)

    bundle = provider.fetch_bundle("510300.SH", "沪深300ETF", "etf")

    assert bundle["fund_fee_payload"]["management_fee"] == "0.15%（每年）"
    assert bundle["fund_fee_payload"]["sales_service_fee"] is None
    assert bundle["fund_fee_payload"]["redeem_fee"] == "0.50%"
    assert bundle["fund_profile_payload"]["fund_manager"] == "柳军"
    assert bundle["fund_profile_payload"]["tracking_index"] == "沪深300指数"
    assert bundle["fund_profile_payload"]["provenance"]["source"] == "eastmoney.fund"
    assert module.xq_calls == 0


class _IndexModule:
    @staticmethod
    def stock_zh_index_daily(symbol):
        return [
            {"date": TODAY - timedelta(days=2), "close": 1990.0, "volume": 100},
            {"date": TODAY - timedelta(days=1), "close": 2000.0, "volume": 120},
        ]

    @staticmethod
    def stock_zh_index_value_csindex(symbol):
        raise ConnectionError("not a CSI index")


def test_index_daily_shares_market_provenance_and_missing_valuation_is_explained():
    provider = AKShareMarketProvider(ak_module=_IndexModule(), max_retries=0, retry_backoff_seconds=0)

    bundle = provider.fetch_bundle("399006.SZ", "创业板指", "index")

    daily = bundle["index_daily_payload"]
    assert daily["provenance"]["source"] == "sina.kline"
    assert daily["provider_endpoint"] == "akshare.stock_zh_index_daily"
    assert bundle["payload"]["volume_unit"] == "share"
    valuation = bundle["index_valuation_payload"]["provenance"]
    assert valuation["source"] is None
    assert valuation["note"].startswith("未获取到实时数据")
    assert valuation["fallback_reason"] == "csindex:error(ConnectionError)"


# ---- macro provider ----------------------------------------------------------------------


def _month_label(offset: int) -> str:
    month = date(TODAY.year, TODAY.month, 1)
    for _ in range(offset):
        month = (month - timedelta(days=1)).replace(day=1)
    return f"{month.year}年{month.month:02d}月份"


class _MacroModule:
    def __init__(self) -> None:
        self.bond_calls = 0

    @staticmethod
    def macro_china_cpi():
        return [
            {"月份": _month_label(0), "全国-同比增长": math.nan},  # not yet released
            {"月份": _month_label(1), "全国-当月": 100.8, "全国-同比增长": 0.8},
            {"月份": _month_label(2), "全国-当月": 100.5, "全国-同比增长": 0.5},
        ]

    @staticmethod
    def macro_china_money_supply():
        return [{"月份": _month_label(1), "货币和准货币(M2)-数量(亿元)": 3568083.6, "货币和准货币(M2)-同比增长": 7.5}]

    @staticmethod
    def macro_china_pmi():
        return [{"月份": "2025年08月份", "制造业-指数": 49.4}]  # a dead feed: stale rows are rejected

    def bond_zh_us_rate(self, start_date):
        self.bond_calls += 1
        raise ConnectionError("datacenter down")

    @staticmethod
    def bond_china_yield(start_date, end_date):
        return [
            {"曲线名称": "中债商业银行普通债收益率曲线(AAA)", "日期": TODAY - timedelta(days=1), "10年": 1.85},
            {"曲线名称": "中债国债收益率曲线", "日期": TODAY - timedelta(days=1), "10年": 1.6738},
        ]

    @staticmethod
    def macro_china_lpr():
        return [
            {"TRADE_DATE": TODAY - timedelta(days=40), "LPR1Y": 3.0, "LPR5Y": 3.5},
            {"TRADE_DATE": TODAY - timedelta(days=5), "LPR1Y": 3.0, "LPR5Y": 3.5},
        ]


def test_macro_provider_reads_correct_columns_and_explains_fallbacks():
    provider = AKShareMacroProvider(ak_module=_MacroModule())

    items, failures = provider.fetch_indicators_with_status(
        {"normalized_query": "CPI M2 PMI 国债 LPR", "keywords": [], "entity_names": []}
    )
    by_code = {item["payload"]["indicator_code"]: item["payload"] for item in items}

    assert by_code["CPI_CN"]["metric_value"] == 0.8  # latest *released* month, not the NaN placeholder
    assert by_code["CPI_CN"]["metric_date"] == to_iso_date(_month_label(1))
    assert by_code["M2_CN"]["metric_value"] == 7.5  # YoY growth, never the 亿元 quantity
    assert by_code["M2_CN"]["unit"] == "%"
    assert by_code["CN10Y"]["metric_value"] == 1.6738
    assert by_code["CN10Y"]["provenance"]["source"] == "chinabond"
    assert by_code["CN10Y"]["provenance"]["mode"] == "live_fallback"
    assert by_code["CN10Y"]["provenance"]["fallback_reason"] == "eastmoney.datacenter:error(ConnectionError)"
    assert by_code["LPR1Y_CN"]["metric_value"] == 3.0 and by_code["LPR5Y_CN"]["metric_value"] == 3.5
    assert "PMI_CN" not in by_code
    assert failures["PMI_CN"] == "eastmoney.datacenter:empty"
    json.dumps(items, allow_nan=False)


def test_macro_provider_caches_results_between_calls():
    module = _MacroModule()
    provider = AKShareMacroProvider(ak_module=module)
    bundle = {"normalized_query": "国债收益率", "keywords": [], "entity_names": []}

    first = provider.fetch_indicators(bundle)
    second = provider.fetch_indicators(bundle)

    assert module.bond_calls == 1
    assert first[0]["payload"]["provenance"]["cache_hit"] is False
    assert second[0]["payload"]["provenance"]["cache_hit"] is True


# ---- announcements -------------------------------------------------------------------------


class _CninfoSession:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []

    def post(self, url, data, headers, timeout):
        self.posts.append((url, dict(data)))
        if "topSearch" in url:
            return _FakeResponse(payload=[{"code": "300750", "orgId": "GD165627", "zwjc": "宁德时代"}])
        return _FakeResponse(
            payload={
                "announcements": [
                    {
                        "secCode": "300750",
                        "announcementTitle": "关于绿色科技创新债券发行的公告",
                        "announcementTime": 1790160369000,
                        "adjunctUrl": "finalpage/2026-09-23/1.PDF",
                    }
                ]
            }
        )


def test_cninfo_resolves_org_id_once_and_filters_by_company():
    session = _CninfoSession()
    provider = CninfoAnnouncementProvider(session=session)

    first = provider.fetch_announcements("300750.SZ", limit=5)
    provider.fetch_announcements("300750.SZ", limit=5)

    queries = [data for url, data in session.posts if "hisAnnouncement" in url]
    assert queries[0]["stock"] == "300750,GD165627"
    assert sum("topSearch" in url for url, _ in session.posts) == 1  # orgId cached
    assert first[0]["payload"]["provenance"]["source"] == "cninfo.announcement"
    assert first[0]["entity_symbols"] == ["300750.SZ"]


class _EastmoneyNoticeSession:
    def get(self, url, params, headers, timeout):
        return _FakeResponse(
            payload={
                "data": {
                    "list": [
                        {
                            "art_code": "AN202609241829869787",
                            "codes": [{"stock_code": "300750"}],
                            "title_ch": "宁德时代:H股公告(翌日披露报表)",
                            "notice_date": "2026-09-24 00:00:00",
                        },
                        {"art_code": "X", "codes": [{"stock_code": "000001"}], "title": "unrelated"},
                    ]
                }
            }
        )


class _DownCninfo:
    timeout = 1

    def fetch_announcements(self, symbol, limit=10):
        raise ConnectionError("cninfo down")


def test_announcement_chain_falls_back_to_eastmoney_with_provenance():
    chain = FallbackAnnouncementProvider(
        providers=[
            ("cninfo.announcement", _DownCninfo()),
            ("eastmoney.announcement", EastmoneyAnnouncementProvider(session=_EastmoneyNoticeSession(), timeout=1)),
        ],
        runtime=_runtime(),
    )

    docs = chain.fetch_announcements("300750.SZ", limit=5)

    assert [doc["title"] for doc in docs] == ["宁德时代:H股公告(翌日披露报表)"]
    assert docs[0]["source_url"] == "https://data.eastmoney.com/notices/detail/300750/AN202609241829869787.html"
    assert docs[0]["publish_time"] == "2026-09-24T00:00:00"
    provenance = docs[0]["payload"]["provenance"]
    assert provenance["mode"] == "live_fallback"
    assert provenance["fallback_reason"] == "cninfo.announcement:error(ConnectionError)"
    assert chain.timeout == 2.0


# ---- retrieval pipeline --------------------------------------------------------------------


def _seed_documents() -> list[dict]:
    return json.loads((Path(DATA_DIR) / "documents.json").read_text(encoding="utf-8"))


def _pipeline(structured: dict | None = None) -> RetrievalPipeline:
    structured = structured or load_structured_data()
    return RetrievalPipeline(
        query_builder=QueryBuilder(),
        doc_retriever=DocumentRetriever(_seed_documents()),
        sql_retriever=SQLRetriever(structured),
        api_retriever=APIRetriever(structured),
        feature_builder=FeatureBuilder(),
        ranker=BaselineRanker(),
        deduper=Deduper(),
        selector=DocumentSelector(),
        packager=RetrievalPackager(),
    )


def _bundle(**overrides: Any) -> dict:
    bundle = {
        "query_id": "q",
        "normalized_query": "贵州茅台今天股价",
        "keywords": [],
        "entity_names": ["贵州茅台"],
        "symbols": ["600519.SH"],
        "industry_terms": [],
        "source_plan": ["market_api", "fundamental_sql"],
        "product_type": "stock",
        "intent_labels": [],
        "topic_labels": [],
    }
    bundle.update(overrides)
    return bundle


def test_offline_snapshot_items_are_labelled_as_snapshot():
    items = _pipeline()._fetch_structured_items(_bundle())

    market = next(item for item in items if item["source_type"] == "market_api")
    provenance = market["payload"]["provenance"]
    assert provenance["mode"] == "snapshot" and provenance["is_live"] is False
    assert provenance["source"] == "offline_snapshot"
    assert provenance["fallback_reason"] == "live market data disabled"
    assert provenance["as_of"] == market["payload"]["trade_date"]
    assert "离线快照" in provenance["note"]
    # The shipped seed payload itself is not mutated.
    assert "provenance" not in load_structured_data()["market_api"]["600519.SH"]


class _ScriptedMarketProvider:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def fetch_bundle(self, symbol, canonical_name, product_type, start_date, end_date):
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return json.loads(json.dumps(outcome))


def _live_bundle(close: float) -> dict:
    return {
        "source_type": "market_api",
        "source_name": "akshare_sina",
        "payload": {
            "symbol": "600519.SH",
            "trade_date": RECENT,
            "close": close,
            "history": [],
            "provenance": build_provenance(source="sina.kline", kind="market", as_of=RECENT),
        },
        "fundamental_payload": {},
    }


def test_pipeline_caches_market_bundles_and_serves_last_known_good():
    clock = FakeClock()
    pipeline = _pipeline()
    pipeline.source_runtime = _runtime(clock)
    provider = _ScriptedMarketProvider([_live_bundle(1237.0), ConnectionError("all sources down")])
    pipeline.market_provider = provider

    first = pipeline._fetch_structured_items(_bundle())
    cached = pipeline._fetch_structured_items(_bundle())
    assert provider.calls == 1
    assert next(i for i in cached if i["source_type"] == "market_api")["payload"]["provenance"]["cache_hit"] is True
    assert next(i for i in first if i["source_type"] == "market_api")["payload"]["close"] == 1237.0

    clock.advance(pipeline.market_bundle_ttl_s + 1)
    served = pipeline._fetch_structured_items(_bundle())

    market = next(item for item in served if item["source_type"] == "market_api")
    assert provider.calls == 2
    assert market["payload"]["close"] == 1237.0
    assert market["payload"]["provenance"]["mode"] == "last_known_good"
    assert market["payload"]["provenance"]["is_live"] is True
    assert "ConnectionError" in market["payload"]["provenance"]["fallback_reason"]
    assert any("market_served_last_known_good" in warning for warning in market["payload"]["provider_warnings"])
    assert not any(item["source_type"] == "provider_warning" for item in served)


def test_pipeline_uses_fresh_snapshot_but_never_a_stale_one_after_live_failure():
    structured = json.loads(json.dumps(load_structured_data()))
    structured["market_api"]["600519.SH"]["trade_date"] = RECENT
    fresh_pipeline = _pipeline(structured)
    fresh_pipeline.market_provider = _ScriptedMarketProvider([ConnectionError("down")])

    items = fresh_pipeline._fetch_structured_items(_bundle(source_plan=["market_api"]))

    market = next(item for item in items if item["source_type"] == "market_api")
    assert market["payload"]["provenance"]["mode"] == "snapshot"
    assert market["payload"]["provenance"]["fallback_reason"].startswith("live market fetch failed")
    assert any(item["source_type"] == "provider_warning" for item in items)

    stale_pipeline = _pipeline()
    stale_pipeline.market_provider = _ScriptedMarketProvider([ConnectionError("down")])
    stale_items = stale_pipeline._fetch_structured_items(_bundle(source_plan=["market_api"]))
    assert not any(item["source_type"] == "market_api" for item in stale_items)


class _FailingMacroProvider:
    def fetch_indicators_with_status(self, query_bundle):
        return [], {"CPI_CN": "eastmoney.datacenter:timeout"}


def test_snapshot_macro_items_explain_why_live_data_was_not_used():
    pipeline = _pipeline()
    pipeline.macro_provider = _FailingMacroProvider()

    items = pipeline._fetch_structured_items(
        {
            "normalized_query": "CPI 对市场有什么影响",
            "keywords": ["CPI"],
            "entity_names": [],
            "symbols": [],
            "source_plan": ["macro_sql"],
        }
    )

    cpi = next(item for item in items if item["payload"]["indicator_code"] == "CPI_CN")
    assert cpi["payload"]["provenance"]["mode"] == "snapshot"
    assert cpi["payload"]["provenance"]["fallback_reason"] == "live macro unavailable (eastmoney.datacenter:timeout)"


def test_packaged_retrieval_result_keeps_field_coverage_and_exposes_provenance():
    pipeline = _pipeline()
    query_bundle = _bundle(source_plan=["market_api", "news"])
    structured = pipeline.fetch_structured(query_bundle)
    documents, groups, total = pipeline.retrieve_documents(query_bundle, top_k=3)

    result = pipeline.packager.build(
        {
            "query_id": "q",
            "product_type": {"label": "stock"},
            "intent_labels": [],
            "topic_labels": [],
            "entities": [],
            "source_plan": ["market_api", "news"],
            "risk_flags": [],
        },
        documents,
        structured,
        groups,
        total,
        ["market_api", "news"],
    )

    market = next(item for item in result["structured_data"] if item["source_type"] == "market_api")
    assert "provenance" not in market["field_coverage"]["missing_fields"]
    assert market["payload"]["provenance"]["mode"] == "snapshot"
    assert result["documents"], "seed corpus should return at least one news document"
    assert all(doc["payload"]["provenance"]["source"] == "local_corpus" for doc in result["documents"])


# ---- agent tools ---------------------------------------------------------------------------


class _StubResolver:
    entities: ClassVar[list[dict[str, str]]] = [
        {"symbol": "600519.SH", "canonical_name": "贵州茅台", "entity_type": "stock"},
        {"symbol": "000300.SH", "canonical_name": "沪深300", "entity_type": "index"},
    ]


class _StubNLU:
    entity_resolver = _StubResolver()


@pytest.fixture(scope="module")
def tool_registry():
    from query_intelligence.agent.tools import ToolContext, build_default_registry

    registry = build_default_registry(ToolContext(nlu_pipeline=_StubNLU(), retrieval_pipeline=_pipeline()))
    yield registry
    registry.shutdown()


def test_price_tool_surfaces_provenance_in_data_and_evidence(tool_registry):
    result = tool_registry.run("get_price_history", {"target": "600519.SH"})

    assert result.ok, result.error
    provenance = result.data["provenance"]
    assert provenance["mode"] == "snapshot" and provenance["is_live"] is False
    assert result.evidence[0].payload["provenance"]["source"] == "offline_snapshot"
    assert _numbers_in(result.evidence[0].payload["provenance"]) == []


def test_fundamentals_macro_and_document_tools_surface_provenance(tool_registry):
    fundamentals = tool_registry.run("get_fundamentals", {"target": "600519.SH"})
    macro = tool_registry.run("get_macro_indicators", {"topics": ["CPI"]})
    news = tool_registry.run("search_news", {"query": "贵州茅台", "targets": ["600519.SH"], "top_k": 3})

    assert fundamentals.ok and fundamentals.data["provenance"]["mode"] == "snapshot"
    assert macro.ok and all(item["provenance"]["mode"] == "snapshot" for item in macro.data["indicators"])
    assert news.ok and news.data["documents"]
    assert all(hit["provenance"]["source"] == "local_corpus" for hit in news.data["documents"])
    assert all(item.payload["provenance"]["source"] == "local_corpus" for item in news.evidence)


# ---- API -------------------------------------------------------------------------------------


def test_sources_health_endpoint_reports_status_latency_and_last_error():
    from fastapi.testclient import TestClient

    from query_intelligence.api.app import create_app

    class _Service:
        def __init__(self) -> None:
            self.retrieval_pipeline = _pipeline()

    service = _Service()
    runtime = service.retrieval_pipeline.source_runtime
    runtime.health.acquire("sina.kline")
    runtime.health.record_success("sina.kline", 120.0)
    for _ in range(3):
        runtime.health.acquire("eastmoney.quote")
        runtime.health.record_failure("eastmoney.quote", 210.0, "ConnectionError: Remote end closed connection")

    client = TestClient(create_app(service=service, app_config={"deepseek": {"api_key": ""}}))
    response = client.get("/sources/health")

    assert response.status_code == 200
    body = response.json()
    by_source = {row["source"]: row for row in body["sources"]}
    assert by_source["sina.kline"]["status"] == "up" and by_source["sina.kline"]["last_latency_ms"] == 120.0
    assert by_source["eastmoney.quote"]["status"] == "down"
    assert by_source["eastmoney.quote"]["circuit"] == "open"
    assert by_source["eastmoney.quote"]["last_error"].startswith("ConnectionError")
    assert body["summary"]["down"] == 1
    assert body["live_providers"]["market"] is None
    assert body["circuit_breaker"]["failure_threshold"] == 3
