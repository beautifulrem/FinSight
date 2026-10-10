"""Round-13 rule I2 (round-9 review), written with the engineer's own wording: a threshold forecast ("能不能站上105元",
"Could X fall below 80 yuan"), a modal close ("收盘会在多少"), a near-future time ("本周", "by Friday") and a "your
call" request for a price are price forecasts. On the template path the answer gets the "cannot predict" hedge and the
limitation; the same templates with a past time are plain lookups.

The property tests take the cross product of time expressions and price templates, so a new future-time word or a new
price template is checked against every counterpart, not just against the repro it was added for.
"""

from __future__ import annotations

import itertools

import pytest

# ---- property: any future time x any price template => hedge; past times => no hedge ----

_FUTURE_ZH = [
    "明天",
    "后天",
    "下周三",
    "下个交易日",
    "下个月",
    "月底",
    "年底前",
    "三个交易日后",
    "本周",
    "这周",
    "本月",
]
_PAST_ZH = ["昨天", "前天", "上周五", "上个月", "去年年底"]
_NAMES_ZH = ["招行", "比亚迪"]
# every template names a price level or a price number
_FUTURE_TEMPLATES_ZH = [
    "{n}{t}的收盘价会是多少",
    "{n}{t}股价大概会在什么价位",
    "{n}{t}能不能站上{p}元",
    "{n}{t}会不会跌破{p}元",
    "{t}{n}收盘会在多少",
    "{n}{t}能守住{p}块吗",
    "{n}{t}有没有可能涨到{p}",
    "{n}{t}有望重回{p}元吗",
]
_PAST_TEMPLATES_ZH = [
    "{n}{t}的收盘价是多少",
    "{n}{t}收在多少",
    "{n}{t}站上{p}元了吗",
    "{n}{t}有没有跌破{p}元",
    "{n}{t}的股价是多少",
]

_FUTURE_EN = [
    "tomorrow",
    "next Friday",
    "next week",
    "by Friday",
    "this week",
    "by the end of the month",
    "in two weeks",
]
_PAST_EN = ["yesterday", "last Friday", "last week", "last month"]
_NAMES_EN = ["CMB", "BYD"]
_FUTURE_TEMPLATES_EN = [
    "What will {n} close at {t}?",
    "Could {n} fall below {p} yuan {t}?",
    "Will {n} break above {p} {t}?",
    "Where will {n}'s share price be {t}?",
    "Can {n} hold {p} yuan {t}?",
    "What's your call for {n}'s close {t}?",
    "Is {n} likely to reach CNY {p} {t}?",
]
_PAST_TEMPLATES_EN = [
    "What did {n} close at {t}?",
    "Did {n} fall below {p} yuan {t}?",
    "Did {n} break above {p} {t}?",
    "What was {n}'s share price {t}?",
]


def _fill(templates, times, names):
    return [
        template.format(n=name, t=time, p=price)
        for template, time, name, price in itertools.product(templates, times, names, ["42", "88.5"])
    ]


def _hedge_notes(query: str, language: str) -> list[str]:
    from query_intelligence.agent.compliance import apply_compliance

    draft = {"answer": "最新可用收盘价为 40.1 元（2026-09-30）[price_600036.SH]。", "key_points": []}
    if language == "en":
        draft = {"answer": "CMB last available close was CNY 40.1 on 2026-09-30 [price_600036.SH].", "key_points": []}
    _, notes = apply_compliance(draft, query=query, nlu_result={}, language=language)
    return notes


def test_every_future_time_with_every_chinese_price_template_is_hedged():
    from query_intelligence.agent.router import asks_price_forecast

    queries = _fill(_FUTURE_TEMPLATES_ZH, _FUTURE_ZH, _NAMES_ZH)
    assert len(queries) == 8 * 11 * 2 * 2
    missed = [query for query in queries if not asks_price_forecast(query)]
    assert not missed, missed[:10]
    unhedged = [query for query in queries[::7] if "price_forecast_hedge" not in _hedge_notes(query, "zh")]
    assert not unhedged, unhedged[:10]


def test_every_future_time_with_every_english_price_template_is_hedged():
    from query_intelligence.agent.router import asks_price_forecast

    queries = _fill(_FUTURE_TEMPLATES_EN, _FUTURE_EN, _NAMES_EN)
    missed = [query for query in queries if not asks_price_forecast(query)]
    assert not missed, missed[:10]
    unhedged = [query for query in queries[::7] if "price_forecast_hedge" not in _hedge_notes(query, "en")]
    assert not unhedged, unhedged[:10]


def test_past_times_with_the_same_price_templates_are_lookups():
    from query_intelligence.agent.router import asks_price_forecast

    queries = _fill(_PAST_TEMPLATES_ZH, _PAST_ZH, _NAMES_ZH) + _fill(_PAST_TEMPLATES_EN, _PAST_EN, _NAMES_EN)
    flagged = [query for query in queries if asks_price_forecast(query)]
    assert not flagged, flagged[:10]
    hedged = [
        query
        for query in queries[::5]
        if "price_forecast_hedge" in _hedge_notes(query, "en" if query.isascii() else "zh")
    ]
    assert not hedged, hedged[:10]


@pytest.mark.parametrize(
    "query",
    [
        "比亚迪本周最高价是多少",  # a period in progress, but a lookup of what has happened
        "招行今天收盘价是多少",
        "What is today's close for BYD?",
        "茅台今年净利润能超过900亿吗",  # a threshold on an amount, not a price
        "招行ROE能达到20%吗",
        "Will Moutai's net profit reach 100 billion?",
        "Can you show me when BYD hit 80?",
        "茅台收盘后会公布多少分红",
        "茅台本周涨了多少",
        "茅台能不能站上MA20",  # a technical level, not a price
    ],
)
def test_lookalikes_without_a_future_price_are_not_price_forecasts(query):
    from query_intelligence.agent.router import asks_price_forecast

    assert not asks_price_forecast(query)


# ---- end to end on the template path (the review's repros reworded) ----


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
    agent = AgentService(runtime, trace_sinks=[])
    yield agent
    runtime.close()


@pytest.mark.parametrize(
    ("query", "close"),
    [
        ("比亚迪下周三收盘会在多少", "83.31"),
        ("招行月底能不能站上45元", "41.26"),
        ("宁德时代这周能守住280元吗", "291.11"),
    ],
)
def test_a_chinese_threshold_or_close_forecast_gets_the_hedge_and_the_dated_close(ext_agent, query, close):
    result = ext_agent.chat(query, session_id=f"r13-forecast-{abs(hash(query))}")
    assert "price_forecast_hedge" in result["compliance_notes"]
    assert "FinSight 不预测未来价格" in result["answer"]
    assert "最新可用收盘价" in result["answer"] and close in result["answer"]
    assert "问题要求预测未来价格，FinSight 只提供历史数据" in result["limitations"]
    assert "技术指标" not in result["answer"]  # "站上45元" is a price threshold, not "站上MA5"


def test_an_english_threshold_forecast_gets_the_hedge(ext_agent):
    result = ext_agent.chat("Might CATL slip under 250 yuan before next Friday?", session_id="r13-forecast-en")
    assert "price_forecast_hedge" in result["compliance_notes"]
    assert "FinSight does not predict future prices" in result["answer"]
    assert "291.11" in result["answer"]


def test_a_past_threshold_question_gets_no_hedge(ext_agent):
    result = ext_agent.chat("比亚迪上周五有没有跌破80元", session_id="r13-forecast-past")
    assert "price_forecast_hedge" not in result["compliance_notes"]
    assert "不预测未来价格" not in result["answer"]
