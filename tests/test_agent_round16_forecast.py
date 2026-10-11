"""Round-16 rules L8 and L20 (round-12 review), written with the engineer's own wording.

L8: a market value, a price, a dividend or earnings asked for a future year or horizon ("2031年", "五年后", "三年内",
"by 2030", "in five years"), this year's dividend, or a market value with a modal and a target and no time, is a
forecast: the answer states the dated history under a "FinSight does not forecast" hedge. Before round 16 a future
market value was read as a period lookup (today's market cap, no hedge), and in English the market cap was not even
fetched, so the answer said it was missing. Past periods stay lookups.

L20: English stop-loss / take-profit / entry and exit level requests get the advice flag and the conditional prefix,
like 止损 / 止盈 (a level "at what price" also gets the price-forecast hedge, like 止盈在多少).

The property test crosses future horizons with metrics and modal frames in both languages, with past controls.
"""

from __future__ import annotations

import itertools
from datetime import date

import pytest

from query_intelligence.agent.router import (
    asks_earnings_forecast,
    asks_market_value_forecast,
    asks_prediction,
    asks_price_forecast,
)

REFERENCE = date(2026, 10, 11)

_ZH_HORIZONS = ["2031年", "2029年底", "五年后", "三年内", "未来十年", "到2032年", "明年"]
_ZH_METRICS = {"market_value": "市值", "price": "股价", "dividend": "分红", "earnings": "净利润"}
_ZH_FRAMES = ["长江电力{h}的{m}会是多少", "{h}紫金矿业{m}能到多少", "中国神华{h}{m}大概有多少"]
_EN_HORIZONS = ["in 2031", "by 2029", "in five years", "within three years", "over the next decade", "next year"]
_EN_METRICS = {"market_value": "market cap", "price": "share price", "dividend": "dividend", "earnings": "net profit"}
_EN_FRAMES = [
    "What will Zijin's {m} be {h}?",
    "{h}, how high could CYPC's {m} go?",
    "Could Shenhua's {m} reach a record {h}?",
]

_ZH_PAST = ["长江电力2025年的{m}是多少", "紫金矿业去年的{m}", "中国神华五年前的{m}", "2024年末长江电力的{m}"]
_EN_PAST = ["What was Zijin's {m} in 2024?", "CYPC's {m} last year", "Shenhua's {m} at the end of 2025"]


def _specific(metric: str, query: str) -> bool:
    if metric == "market_value":
        return asks_market_value_forecast(query, as_of=REFERENCE)
    if metric == "price":
        return asks_price_forecast(query, as_of=REFERENCE)
    return asks_earnings_forecast(query, as_of=REFERENCE)


def _future_cases() -> list[tuple[str, str]]:
    cases = []
    for (metric, word), horizon, frame in itertools.product(_ZH_METRICS.items(), _ZH_HORIZONS, _ZH_FRAMES):
        cases.append((metric, frame.format(h=horizon, m=word)))
    for (metric, word), horizon, frame in itertools.product(_EN_METRICS.items(), _EN_HORIZONS, _EN_FRAMES):
        text = frame.format(h=horizon, m=word)
        cases.append((metric, text[0].upper() + text[1:]))
    return cases


def _past_cases() -> list[tuple[str, str]]:
    cases = [
        (metric, frame.format(m=word)) for (metric, word), frame in itertools.product(_ZH_METRICS.items(), _ZH_PAST)
    ]
    cases += [
        (metric, frame.format(m=word)) for (metric, word), frame in itertools.product(_EN_METRICS.items(), _EN_PAST)
    ]
    return cases


@pytest.mark.parametrize(("metric", "query"), _future_cases())
def test_a_metric_for_a_future_year_or_horizon_is_a_forecast(metric, query):
    assert _specific(metric, query), query
    assert asks_prediction(query, as_of=REFERENCE), query


@pytest.mark.parametrize(("metric", "query"), _past_cases())
def test_the_same_metric_for_a_past_period_is_a_lookup(metric, query):
    assert not _specific(metric, query), query
    assert not asks_prediction(query, as_of=REFERENCE), query


def test_the_property_set_covers_every_metric_and_language():
    future, past = _future_cases(), _past_cases()
    assert len(future) == 4 * (7 * 3 + 6 * 3) and len(past) == 4 * 7
    assert {metric for metric, _ in future} == set(_ZH_METRICS)


@pytest.mark.parametrize(
    "query",
    [
        "长江电力市值能突破1万亿吗",  # a target and a modal, no time
        "Will Zijin's market capitalisation top 1 trillion yuan?",
        "今年年底中国神华市值会有多少",  # a period still running, with a modal
        "紫金矿业今年分红多少",  # this year's dividend is not in the reported statements
        "What dividend will Shenhua pay this year?",
    ],
)
def test_targets_running_periods_and_this_years_dividend_are_forecasts(query):
    assert asks_prediction(query, as_of=REFERENCE)


@pytest.mark.parametrize(
    "query",
    [
        "长江电力市值多少",
        "紫金矿业市值排第几",
        "Can you tell me Zijin's market value?",
        "中国神华今年已经分红了吗",
        "紫金矿业近五年净利润",
        "中国神华股息率多少",
        "长江电力市值是否超过7000亿",  # a present-tense comparison, not a target
        "Is Zijin's market cap above 800 billion yuan?",
    ],
)
def test_plain_lookups_are_not_forecasts(query):
    assert not asks_prediction(query, as_of=REFERENCE)


# ---- L20: English trading levels ----


@pytest.mark.parametrize(
    "query",
    [
        "Set a stop-loss level for Zijin for me",
        "Where would a stop loss go on CYPC?",
        "Pick a take-profit target for Shenhua",
        "What's a good exit price for Zijin Mining?",
        "Suggest an entry level for China Shenhua",
        "trailing stop for CYPC?",
    ],
)
def test_english_trading_levels_are_advice_like_their_chinese_counterparts(query):
    assert asks_prediction(query, as_of=REFERENCE)


def test_a_take_profit_price_is_a_price_forecast_like_zhiying_zai_duoshao():
    assert asks_price_forecast("Take profit on Zijin at what price?", as_of=REFERENCE)
    assert asks_price_forecast("紫金矿业止盈在多少", as_of=REFERENCE)


# ---- end to end ----


@pytest.fixture(scope="module")
def agent():
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


def test_a_future_market_value_in_chinese_is_hedged(agent):
    result = agent.chat("长江电力2031年市值能突破1万亿吗", session_id="r16-mv-zh")
    assert "FinSight 不预测未来市值或股价" in result["answer"]
    assert "问题要求预测未来市值，FinSight 只提供历史数据" in result["limitations"]
    assert "总市值" in result["answer"]  # today's market value is stated, dated, as history


def test_a_future_market_value_in_english_states_the_market_cap_under_the_hedge(agent):
    result = agent.chat("Could Zijin Mining's market value reach 1 trillion yuan by 2030?", session_id="r16-mv-en")
    assert "FinSight does not forecast market values or prices" in result["answer"]
    assert "market cap CNY" in result["answer"]
    assert "does not include the requested market cap" not in result["answer"]


def test_this_years_dividend_is_hedged_and_named_as_missing(agent):
    result = agent.chat("长江电力今年能分红多少", session_id="r16-div")
    assert "FinSight 不预测未来业绩" in result["answer"]
    assert any("股息率" in item for item in result["limitations"])


@pytest.mark.parametrize(
    "query", ["Set a stop-loss level for Zijin for me", "Pick a take-profit target for Zijin Mining"]
)
def test_english_trading_levels_are_hedged_end_to_end(agent, query):
    result = agent.chat(query, session_id=f"r16-level-{abs(hash(query))}")
    assert "conditional assessment" in result["answer"]
    assert "The query has advice or forecast-like risk" in result["limitations"]
    assert "investment_advice_like" in result["nlu_summary"]["risk_flags"]
