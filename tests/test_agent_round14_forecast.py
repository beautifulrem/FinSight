"""Round-14 rule J6 (round-10 review), written with the engineer's own wording: a price or an earnings figure asked for
a stated date or period after the reference date (the later of today and the data's latest date) is a forecast; the
same wording with a date on or before it is a lookup. Also opening gaps ("会高开吗"), the price to trade at ("the
exact price to sell at") and earnings asked with a modal for a period still being reported.

The property tests draw dates from a seeded generator around a fixed reference date and cross them with every
price and earnings template, zh and en, so a new date format or template is checked against every counterpart.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from query_intelligence.agent.router import asks_earnings_forecast, asks_price_forecast
from query_intelligence.agent.stated_dates import reference_date, stated_periods

REFERENCE = date(2026, 10, 9)  # a Friday; the data's latest close is earlier, so today is the reference
_SEED = 1414
_EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]  # fmt: skip
_CN_NUMERALS = "〇一二三四五六七八九"


def _cn(value: int) -> str:
    if value < 10:
        return _CN_NUMERALS[value]
    tens, units = divmod(value, 10)
    return ("" if tens == 1 else _CN_NUMERALS[tens]) + "十" + (_CN_NUMERALS[units] if units else "")


def _zh_day(day: date, rng: random.Random) -> str:
    forms = [
        f"{day.year}年{day.month}月{day.day}日",
        f"{day.year}-{day.month:02d}-{day.day:02d}",
        f"{day.year}/{day.month}/{day.day}",
        f"{day.year}年{day.month}月{day.day}号",
        f"{day.year}年{_cn(day.month)}月{_cn(day.day)}日",
    ]
    if day.year == REFERENCE.year:  # a year left out is the reference year
        forms += [f"{day.month}月{day.day}日", f"{_cn(day.month)}月{_cn(day.day)}号"]
    return rng.choice(forms)


def _en_day(day: date, rng: random.Random) -> str:
    month = _EN_MONTHS[day.month - 1]
    forms = [
        f"on {month} {day.day}, {day.year}",
        f"on {day.day} {month} {day.year}",
        f"on {day.year}-{day.month:02d}-{day.day:02d}",
        f"on {month[:3]} {day.day}, {day.year}",
    ]
    if day.year == REFERENCE.year:
        forms.append(f"on {month} {day.day}")
    return rng.choice(forms)


def _quarter_of(day: date) -> int:
    return (day.month - 1) // 3 + 1


def _zh_quarter(year: int, quarter: int, rng: random.Random) -> str:
    return rng.choice(
        [
            f"{year}年{'一二三四'[quarter - 1]}季度",
            f"{year}年第{quarter}季度",
            f"{year}Q{quarter}",
            f"{year}年Q{quarter}",
        ]
    )


def _en_quarter(year: int, quarter: int, rng: random.Random) -> str:
    ordinal = ["first", "second", "third", "fourth"][quarter - 1]
    return rng.choice([f"in Q{quarter} {year}", f"in the {ordinal} quarter of {year}", f"for {quarter}Q{year % 100}"])


def _days(rng: random.Random, *, future: bool, count: int) -> list[date]:
    out = []
    while len(out) < count:
        offset = rng.randint(1, 900)
        day = REFERENCE + timedelta(days=offset if future else -offset)
        out.append(day)
    return out


_NAMES_ZH = ["招行", "比亚迪", "宁德时代", "贵州茅台"]
_NAMES_EN = ["CMB", "BYD", "CATL", "Moutai"]
_ZH_PRICE_FUTURE = [
    "{n}{t}的收盘价是多少",
    "{n}{t}收盘价多少",
    "{t}{n}股价是多少",
    "{n}{t}的股价会在什么价位",
    "{n}{t}能不能站上{p}元",
]
_ZH_PRICE_PAST = ["{n}{t}的收盘价是多少", "{n}{t}收盘价多少", "{t}{n}股价是多少", "{n}{t}站上{p}元了吗"]
_EN_PRICE_FUTURE = [
    "What is {n}'s close {t}?",
    "What will {n} close at {t}?",
    "{n} share price {t}?",
    "Where will {n} trade {t}?",
    "Can {n} hold {p} yuan {t}?",
]
_EN_PRICE_PAST = ["What was {n}'s close {t}?", "What did {n} close at {t}?", "{n} closing price {t}?"]
_ZH_EARNINGS_FUTURE = [
    "{n}{t}的净利润是多少",
    "{n}{t}营收会是多少",
    "{n}{t}每股收益能超过{p}元吗",
    "{n}{t}归母净利润能有多少",
]
_ZH_EARNINGS_PAST = ["{n}{t}的净利润是多少", "{n}{t}营收是多少", "{n}{t}每股收益多少"]
_EN_EARNINGS_FUTURE = ["What is {n}'s net profit {t}?", "What will {n}'s revenue be {t}?", "{n} EPS {t}?"]
_EN_EARNINGS_PAST = ["What was {n}'s net profit {t}?", "What was {n}'s revenue {t}?"]


def _cases(templates: list[str], times: list[str], names: list[str], rng: random.Random) -> list[str]:
    return [
        rng.choice(templates).format(n=rng.choice(names), t=time, p=rng.choice(["42", "88.5", "1500"]))
        for time in times
    ]


def _generated(kind: str, *, future: bool) -> list[str]:
    rng = random.Random(f"{_SEED}-{kind}-{future}")
    days = _days(rng, future=future, count=60)
    if kind == "zh_price":
        return _cases(_ZH_PRICE_FUTURE if future else _ZH_PRICE_PAST, [_zh_day(d, rng) for d in days], _NAMES_ZH, rng)
    if kind == "en_price":
        return _cases(_EN_PRICE_FUTURE if future else _EN_PRICE_PAST, [_en_day(d, rng) for d in days], _NAMES_EN, rng)
    # earnings: whole quarters strictly after (or ending on or before) the reference
    quarters = []
    for day in days:
        year, quarter = day.year, _quarter_of(day)
        start = date(year, 3 * quarter - 2, 1)
        end = (date(year + (quarter == 4), (3 * quarter) % 12 + 1, 1)) - timedelta(days=1)
        if (future and start > REFERENCE) or (not future and end <= REFERENCE):
            quarters.append((year, quarter))
    if kind == "zh_earnings":
        times = [_zh_quarter(y, q, rng) for y, q in quarters] + [
            f"{y}年" for y, _ in quarters[:10] if (future and y > REFERENCE.year) or (not future and y < REFERENCE.year)
        ]
        return _cases(_ZH_EARNINGS_FUTURE if future else _ZH_EARNINGS_PAST, times, _NAMES_ZH, rng)
    times = [_en_quarter(y, q, rng) for y, q in quarters] + [
        f"in {y}" for y, _ in quarters[:10] if (future and y > REFERENCE.year) or (not future and y < REFERENCE.year)
    ]
    return _cases(_EN_EARNINGS_FUTURE if future else _EN_EARNINGS_PAST, times, _NAMES_EN, rng)


@pytest.mark.parametrize("kind", ["zh_price", "en_price"])
def test_a_price_on_any_generated_future_date_is_a_forecast(kind):
    queries = _generated(kind, future=True)
    assert len(queries) == 60
    missed = [query for query in queries if not asks_price_forecast(query, as_of=REFERENCE)]
    assert not missed, missed[:10]


@pytest.mark.parametrize("kind", ["zh_price", "en_price"])
def test_a_price_on_any_generated_past_date_is_a_lookup(kind):
    queries = _generated(kind, future=False)
    flagged = [query for query in queries if asks_price_forecast(query, as_of=REFERENCE)]
    assert not flagged, flagged[:10]


@pytest.mark.parametrize("kind", ["zh_earnings", "en_earnings"])
def test_earnings_for_any_generated_future_period_are_a_forecast(kind):
    queries = _generated(kind, future=True)
    assert len(queries) >= 40
    missed = [query for query in queries if not asks_earnings_forecast(query, as_of=REFERENCE)]
    assert not missed, missed[:10]


@pytest.mark.parametrize("kind", ["zh_earnings", "en_earnings"])
def test_earnings_for_any_generated_past_period_are_a_lookup(kind):
    queries = _generated(kind, future=False)
    assert len(queries) >= 40
    flagged = [query for query in queries if asks_earnings_forecast(query, as_of=REFERENCE)]
    assert not flagged, flagged[:10]
    assert not [query for query in queries if asks_price_forecast(query, as_of=REFERENCE)]


def test_the_reference_is_the_later_of_today_and_the_data_date():
    # the offline evaluation clock (2026-04-23) is earlier than the extended snapshot's closes (2026-09-30)
    reference = reference_date(date(2026, 4, 23), ["2026-09-30", "2026-04-22T15:00:00", None, "not a date"])
    assert reference == date(2026, 9, 30)
    assert asks_price_forecast("茅台2026年11月3日的收盘价是多少", as_of=reference)
    assert not asks_price_forecast("茅台2026年9月15日的收盘价是多少", as_of=reference)
    assert asks_price_forecast("茅台2026年9月15日的收盘价是多少", as_of=date(2026, 4, 23))


@pytest.mark.parametrize(
    "query",
    [
        "茅台2027年1月4日的收盘价是多少",
        "One number only: CMB's close on 2026-12-31?",
        "Where do you see CATL trading in Q1 2027?",
        "明年一季度招行股价大概在哪",
        "What's the Q4 2026 price for BYD?",
        "比亚迪十一月三号收盘价多少",
        "宁德时代下个月15号的收盘价",
        "宁德时代国庆后第一个交易日会高开吗",
        "招行明天高开还是低开",
        "Will BYD gap up on Monday?",
        "Should I dump my CATL shares next week? Tell me the exact price to sell at.",
        "茅台卖在多少合适",
        "At what price should I sell Moutai?",
    ],
)
def test_the_review_forms_reworded_are_price_forecasts(query):
    assert asks_price_forecast(query, as_of=REFERENCE)


@pytest.mark.parametrize(
    "query",
    [
        "比亚迪四季度净利润能超过100亿吗",
        "招行明年净利润会是多少",
        "What will CATL's EPS be next year?",
        "next year's EPS for BYD?",
        "宁德时代2027年营收",
        "Will Moutai's net profit top 100 billion this year?",
        "五粮液今年的分红能有多少",
    ],
)
def test_earnings_of_an_unreported_period_are_earnings_forecasts(query):
    assert asks_earnings_forecast(query, as_of=REFERENCE)


@pytest.mark.parametrize(
    "query",
    [
        "茅台2026年9月15日的收盘价是多少",
        "What was BYD's close on 2026-09-15?",
        "招行今天高开了多少",
        "BYD opened higher yesterday, why?",
        "茅台2026年以来涨了多少",
        "宁德时代2026年前三季度营收是多少",
        "比亚迪四季度净利润是多少",  # the last reported fourth quarter
        "茅台去年净利润是多少",
        "What was Moutai's revenue in 2025?",
        "招行ROE能达到20%吗",  # no period: a description as much as a forecast
        "茅台股价是多少",
        "Moutai's net profit for 2025?",
        "BYD may 3 times its peers' margin",
    ],
)
def test_lookups_with_dates_or_metrics_are_not_forecasts(query):
    assert not asks_price_forecast(query, as_of=REFERENCE)
    assert not asks_earnings_forecast(query, as_of=REFERENCE)


def test_year_less_and_cumulative_periods():
    (q4,) = stated_periods("四季度", REFERENCE)
    assert (q4.start, q4.end, q4.explicit_year) == (date(2026, 10, 1), date(2026, 12, 31), False)
    (nine,) = stated_periods("2026年前三季度", REFERENCE)
    assert (nine.start, nine.end) == (date(2026, 1, 1), date(2026, 9, 30))
    (relative,) = stated_periods("明年一季度", REFERENCE)
    assert relative.start == date(2027, 1, 1)
    assert stated_periods("能到2027吗", REFERENCE) == []  # a bare number is a price, not a year


# ---- end to end on the template path: the hedge, the limitation and the dated history ----


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
    runtime = AgentRuntime(
        service, build_registry_for_service(service), ScriptedLLM([]), today=lambda: date(2026, 10, 9)
    )
    agent = AgentService(runtime, trace_sinks=[])
    yield agent
    runtime.close()


def test_a_future_dated_close_gets_the_price_hedge(ext_agent):
    result = ext_agent.chat("招行2027年1月4日的收盘价是多少", session_id="r14-dated-close")
    assert "price_forecast_hedge" in result["compliance_notes"]
    assert "FinSight 不预测未来价格" in result["answer"] and "41.26" in result["answer"]
    assert "问题要求预测未来价格，FinSight 只提供历史数据" in result["limitations"]


def test_a_past_dated_close_is_a_lookup(ext_agent):
    result = ext_agent.chat("招行2026年9月15日的收盘价是多少", session_id="r14-dated-past")
    assert "price_forecast_hedge" not in result["compliance_notes"]
    assert "不预测" not in result["answer"]


def test_an_earnings_forecast_gets_the_earnings_hedge(ext_agent):
    result = ext_agent.chat("比亚迪明年净利润会是多少", session_id="r14-earnings")
    assert "earnings_forecast_hedge" in result["compliance_notes"]
    assert "FinSight 不预测未来业绩" in result["answer"]
    assert "问题要求预测未来业绩，FinSight 只提供已披露的历史财务数据" in result["limitations"]
    assert "326.19" in result["answer"]  # the reported FY2025 figure, with its period


def test_an_english_quarter_forecast_and_the_price_to_sell_at(ext_agent):
    result = ext_agent.chat("Where do you see BYD trading in Q2 2027?", session_id="r14-en-quarter")
    assert "FinSight does not predict future prices" in result["answer"]
    result = ext_agent.chat(
        "Should I sell all my CATL shares tomorrow? Tell me the exact price to sell at.", session_id="r14-en-sell"
    )
    assert "price_forecast_hedge" in result["compliance_notes"]
    assert "FinSight does not predict future prices" in result["answer"]


def test_a_gap_question_is_hedged(ext_agent):
    result = ext_agent.chat("比亚迪国庆后第一个交易日会高开吗", session_id="r14-gap")
    assert "price_forecast_hedge" in result["compliance_notes"]
    assert "FinSight 不预测未来价格" in result["answer"]
