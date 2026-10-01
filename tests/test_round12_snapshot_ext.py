"""Round 12: the offline snapshot extension (data/snapshot/) and the derived price metrics it makes computable.

Written from the round-7 review's data-acquisition items with the author's own wording: widely asked names that the
v1 snapshot lacks (宁德时代, 招商银行, 黄金ETF, 上证指数, ...) get >= 250 daily closes, the FY2025 report and the
valuation on 2026-09-30, so EPS, the market cap, YoY growth, the year-to-date change, the 52-week high/low and the
maximum drawdown compute offline. The v1 snapshot is never changed, and the evaluation harness stays pinned to it.
"""

from __future__ import annotations

import json

import pytest

from query_intelligence.agent.tools.market import (
    JUMP_LIMIT,
    PriceHistoryOutput,
    jump_date,
    max_drawdown,
    max_drawdown_1y,
    max_drawdown_ytd,
    range_52w,
)
from query_intelligence.data_loader import DATA_DIR, SNAPSHOT_EXT_PATH, load_structured_data, merge_snapshot_extension

EXT_SYMBOLS = {"300750.SZ", "002594.SZ", "600036.SH", "000001.SZ", "518880.SH", "000001.SH"}


# --- the extension and the loader ------------------------------------------------------------------------------------
def test_the_extension_adds_names_and_never_changes_a_v1_record():
    base = json.loads((DATA_DIR / "structured_data.json").read_text(encoding="utf-8"))
    merged = load_structured_data(extended=True)
    for section in ("market_api", "fundamental_sql", "industry_sql", "macro_sql", "entity_to_industry"):
        for key, record in base[section].items():
            assert merged[section][key] == record, (section, key)
    assert set(merged["market_api"]) - set(base["market_api"]) >= EXT_SYMBOLS
    assert load_structured_data(extended=False) == base


def test_merge_keeps_the_base_record_when_both_have_a_key():
    base = {"market_api": {"A": {"close": 1}}, "fundamental_sql": {}}
    ext = {"market_api": {"A": {"close": 2}, "B": {"close": 3}}, "fundamental_sql": {"B": {"roe": 1}}}
    merged = merge_snapshot_extension(base, ext)
    assert merged["market_api"] == {"A": {"close": 1}, "B": {"close": 3}}
    assert merged["fundamental_sql"] == {"B": {"roe": 1}}
    assert base["market_api"] == {"A": {"close": 1}}  # the base dict is not mutated


def test_every_extension_instrument_has_a_year_of_closes_and_stocks_have_the_fields_for_derived_metrics():
    ext = json.loads(SNAPSHOT_EXT_PATH.read_text(encoding="utf-8"))
    for symbol, payload in ext["market_api"].items():
        assert len(payload["history"]) >= 250, symbol
        assert payload["trade_date"] == ext["as_of"] == "2026-09-30"
        assert payload["_snapshot"]["fetched_at"] and payload["_snapshot"]["endpoint"]
    for symbol, payload in ext["fundamental_sql"].items():
        for field in ("total_mv", "revenue_yoy", "netprofit_yoy", "eps", "pe_ttm", "revenue", "net_profit"):
            assert payload.get(field) is not None, (symbol, field)
        assert payload["report_date"] == "2025-12-31" and payload["valuation_date"] == "2026-09-30"


def test_the_build_is_reproducible_offline_from_the_committed_raw_records():
    from scripts.build_offline_snapshot import cmd_verify

    assert cmd_verify(None) == 0


def test_the_snapshot_extension_setting_follows_the_environment(monkeypatch):
    from query_intelligence.config import Settings
    from query_intelligence.data_loader import snapshot_ext_enabled

    monkeypatch.delenv("QI_OFFLINE_SNAPSHOT_EXT", raising=False)
    assert snapshot_ext_enabled() and Settings.from_env().offline_snapshot_ext
    monkeypatch.setenv("QI_OFFLINE_SNAPSHOT_EXT", "0")
    assert not snapshot_ext_enabled() and not Settings.from_env().offline_snapshot_ext


# --- services -------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ext_registry():
    from query_intelligence.agent.tools import build_registry_for_service
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
    )
    return build_registry_for_service(service)


def _data(result):
    return result.data.model_dump(mode="json") if hasattr(result.data, "model_dump") else result.data


def test_the_evaluation_harness_is_pinned_to_v1_unless_asked(monkeypatch):
    from evaluation.agent_eval.runner import build_offline_service, eval_snapshot_ext
    from query_intelligence.agent.tools import build_registry_for_service

    monkeypatch.delenv("QI_EVAL_SNAPSHOT_EXT", raising=False)
    assert not eval_snapshot_ext()
    pinned = build_registry_for_service(build_offline_service())
    assert not pinned.run("get_price_history", {"target": "300750.SZ"}).ok  # the v1 labels say: no offline price
    monkeypatch.setenv("QI_EVAL_SNAPSHOT_EXT", "1")
    assert eval_snapshot_ext()


def test_an_extension_quote_names_its_own_date_file_and_fetch_time(ext_registry):
    data = _data(ext_registry.run("get_price_history", {"target": "宁德时代", "days": 5}))
    assert data["as_of"] == "2026-09-30" and data["close"] == 291.11
    provenance = data["provenance"]
    assert provenance["endpoint"] == "data/snapshot/structured_data_ext.json"
    assert provenance["mode"] == "snapshot" and provenance["fetched_at"].startswith("2026-10-01")
    assert provenance["snapshot_version"] == "ext-2026-09-30" and "2026-09-30" in provenance["note"]
    # a v1 quote keeps its own date and the v1 file
    v1 = _data(ext_registry.run("get_price_history", {"target": "600519.SH", "days": 5}))
    assert v1["as_of"] == "2026-04-22" and v1["provenance"]["endpoint"] == "data/structured_data.json"
    assert "range_52w" not in v1 and "max_drawdown_1y" not in v1 and "year_start" not in v1


def test_extension_fundamentals_carry_market_cap_growth_eps_and_the_valuation_date(ext_registry):
    result = ext_registry.run("get_fundamentals", {"target": "600036.SH"})
    data = _data(result)
    metrics = data["metrics"]
    assert metrics["total_mv"] == 1040570829497.26 and metrics["eps"] == 5.7
    assert metrics["revenue_yoy"] == 0.01 and metrics["netprofit_yoy"] == 1.0477
    assert data["valuation_date"] == "2026-09-30" and data["metric_units"]["total_mv"] == "CNY"
    assert "_snapshot" not in result.evidence[0].payload


# --- derived metrics from the history -------------------------------------------------------------------------------
def _history(closes: list[tuple[str, float]]) -> list[dict]:
    return [{"trade_date": day, "close": close} for day, close in reversed(closes)]  # providers: latest first


def test_drawdown_is_the_largest_fall_from_a_running_peak():
    closes = [
        ("2026-01-02", 10.0),
        ("2026-01-05", 12.0),
        ("2026-01-06", 9.0),
        ("2026-01-07", 13.0),
        ("2026-01-08", 11.7),
    ]
    found = max_drawdown(closes, "x")
    assert (found.peak, found.trough, found.max_drawdown_pct) == (12.0, 9.0, -25.0)
    assert (found.peak_date, found.trough_date) == ("2026-01-05", "2026-01-06")
    rising = max_drawdown([("2026-01-02", 1.0), ("2026-01-05", 2.0)], "x")
    assert rising.max_drawdown_pct == 0.0


def test_52_week_metrics_need_a_history_covering_the_window():
    short = _history([("2026-04-21", 4.7), ("2026-04-22", 4.8)])
    assert range_52w(short) is None and max_drawdown_1y(short) is None
    year = [("2025-09-30", 10.0), ("2025-10-09", 11.0), ("2026-03-02", 9.0), ("2026-09-30", 9.5)]
    found = range_52w(_history(year))
    assert found.start_date == "2025-10-09" and found.end_date == "2026-09-30"  # 2025-09-30 is outside the 52 weeks
    assert (found.high, found.high_date, found.low, found.low_date, found.closes) == (
        11.0,
        "2025-10-09",
        9.0,
        "2026-03-02",
        3,
    )
    assert max_drawdown_1y(_history(year)).max_drawdown_pct == pytest.approx(-18.18)


def test_ytd_drawdown_starts_at_the_first_close_of_the_year():
    closes = [("2025-12-31", 10.5), ("2026-01-05", 10.0), ("2026-02-02", 12.0), ("2026-03-02", 10.2)]
    found = max_drawdown_ytd(_history(closes))
    assert found.start_date == "2026-01-05" and (found.peak, found.trough, found.max_drawdown_pct) == (
        12.0,
        10.2,
        -15.0,
    )
    assert max_drawdown_ytd(_history(closes[1:])) is None  # no 2025 close: the year's first close is not known


def test_an_ex_rights_jump_in_the_window_blocks_the_range_and_the_drawdown():
    closes = [("2025-09-30", 300.0), ("2025-10-09", 330.0), ("2026-01-05", 110.0), ("2026-09-30", 100.0)]
    assert jump_date(closes) == "2026-01-05" and JUMP_LIMIT == 0.21
    assert range_52w(_history(closes)) is None and max_drawdown_1y(_history(closes)) is None
    # a +20% day (the ChiNext/STAR limit) is a market move, not a jump
    assert jump_date([("2026-01-05", 100.0), ("2026-01-06", 120.0)]) is None


def test_a_short_history_serialises_without_the_new_fields():
    output = PriceHistoryOutput(
        symbol="600519.SH", name="贵州茅台", product_type="stock", as_of="2026-04-22", close=1409.5,
        recent_closes=[], source="tushare", evidence_id="price_600519.SH",
    )  # fmt: skip
    dumped = output.model_dump(mode="json")
    for key in ("year_start", "range_52w", "max_drawdown_1y", "max_drawdown_ytd", "price_jump_date"):
        assert key not in dumped


# --- question parsing -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "yearly", "day_high"),
    [
        ("中证500ETF过去52周最高收盘是多少", True, False),
        ("紫金矿业近一年的最高点和最低点", True, False),
        ("What's the 52-week low of 510500?", True, False),
        ("长江电力今天的最高价和52周最高价", True, True),
        ("长江电力今天最高价多少", False, True),
    ],
)
def test_a_52_week_question_is_not_the_days_high(query, yearly, day_high):
    from query_intelligence.agent.coverage import requested_price_fields

    request = requested_price_fields(query)
    assert request.range_52w is yearly and request.high is day_high


@pytest.mark.parametrize(
    ("query", "window"),
    [
        ("工商银行最近一年里最大回撤有多大", "1y"),
        ("科创50ETF的最大回撤", "1y"),
        ("长江电力年初以来的最大回撤", "ytd"),
        ("What is the YTD max drawdown of the gold ETF?", "ytd"),
        ("工商银行最新股价", ""),
    ],
)
def test_the_drawdown_window_comes_from_the_question(query, window):
    from query_intelligence.agent.coverage import drawdown_window

    assert drawdown_window(query) == window


# --- answers (deterministic path, extension on) ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def ext_agent():
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
    )
    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]))
    yield AgentService(runtime, trace_sinks=[])
    runtime.close()


def test_a_one_year_drawdown_is_stated_with_its_window_peak_and_trough(ext_agent):
    result = ext_agent.chat("工商银行过去一年里的最大回撤是多少", session_id="r12-dd-1y")
    answer = result["answer"]
    assert "近52周（2025-10-09 至 2026-09-30，按未复权收盘价）最大回撤 -16.73%" in answer
    assert "2025-11-25 收盘 8.31 元" in answer and "2026-02-27 收盘 6.92 元" in answer
    assert result["verification"]["passed"] and "price_601398.SH" in result["evidence_used"]


def test_a_ytd_drawdown_in_english(ext_agent):
    result = ext_agent.chat("What has China Merchants Bank's max drawdown been so far this year?", session_id="r12-dd")
    assert "year to date (2026-01-05 to 2026-09-30, unadjusted closes): maximum drawdown -16.92%" in result["answer"]
    assert result["verification"]["passed"]


def test_the_52_week_range_replaces_the_days_high_and_low(ext_agent):
    result = ext_agent.chat("上证指数过去52周收盘最高和最低各是多少", session_id="r12-52w")
    answer = result["answer"]
    assert "最高收盘 4242.572 点（2026-05-13），最低收盘 3764.155 点（2026-07-17）" in answer
    assert "最高价 3851.217" not in answer  # not the 2026-09-30 session's high
    assert result["verification"]["passed"]


def test_a_v1_target_still_states_the_52_week_range_as_unavailable(ext_agent):
    result = ext_agent.chat("五粮液52周最低价是多少", session_id="r12-52w-v1")
    assert any("收盘价不足一年，无法给出近一年的最高价和最低价" in item for item in result["limitations"])
    assert result["verification"]["passed"]  # the gap sentence carries no untraceable digits


def test_market_cap_is_in_yi_yuan_and_dated_by_the_valuation_day(ext_agent):
    result = ext_agent.chat("长江电力市值多大", session_id="r12-mv")
    answer = result["answer"]
    assert "长江电力（按 2026-09-30 收盘计算）：总市值 6983.23 亿元" in answer
    assert "流通市值" not in answer and "PE、PB 按 2026-09-30 收盘计算" in answer
    assert result["verification"]["passed"]


def test_reported_eps_has_its_unit_and_growth_fields_their_own_labels(ext_agent):
    eps = ext_agent.chat("紫金矿业的每股收益", session_id="r12-eps")["answer"]
    assert "每股收益 1.91 元" in eps and "隐含每股收益" not in eps  # reported, so no implied value
    growth = ext_agent.chat("中芯国际营收和净利润的同比增速", session_id="r12-yoy")["answer"]
    assert "营收增速 16.49%" in growth and "净利润增速 36.29%" in growth


def test_a_comparison_across_snapshot_dates_names_each_date(ext_agent):
    result = ext_agent.chat("五粮液和招商银行谁的当日涨幅更大", session_id="r12-dates")
    assert "所比较的行情日期不同（五粮液 2026-04-22，招商银行 2026-09-30）" in result["answer"]
    assert any("行情日期不同" in item for item in result["limitations"])


def test_holding_value_and_ytd_change_compute_for_an_extension_etf(ext_agent):
    held = ext_agent.chat("我手上有2000股黄金ETF，按收盘价算值多少", session_id="r12-hold")["answer"]
    assert "2000 × 8.633 = 17266 元" in held
    ytd = ext_agent.chat("黄金ETF年初至今表现如何", session_id="r12-ytd")["answer"]
    assert "今年首个交易日（2026-01-05）收盘 9.495 元" in ytd and "-9.08%" in ytd
