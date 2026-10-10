"""Round-15 rule K11 (round-11 review), written with the engineer's own wording: earnings asked with a modal for a
periodic report named without a year ("三季报", "中报", "Q3 earnings", "interim results"), for the next report that has
not been released ("下一份年报", "upcoming interim results", "the next annual report"), or compared with a past
baseline ("…top last year's", "会比去年高吗") are earnings forecasts and get the hedge. The same report named without a
modal ("三季报净利润多少"), a written past year ("2025年报"), a past tense and a schedule question ("下一份年报什么时候
发布") stay lookups.

The property test crosses report-period expressions with earnings words and modal frames in both languages.
"""

from __future__ import annotations

import itertools
from datetime import date

import pytest

from query_intelligence.agent.router import asks_earnings_forecast

REFERENCE = date(2026, 10, 11)

_ZH_REPORTS = ["三季报", "中报", "半年报", "一季报", "年报", "季报", "下一份年报", "下一期财报", "即将发布的三季报"]
_ZH_WORDS = ["净利润", "营收", "归母净利润", "每股收益", "业绩"]
_ZH_FRAMES = [
    "宁德时代{r}{w}会同比增长吗",
    "招行{r}的{w}能超过去年吗",
    "美的{r}{w}会比去年高吗",
    "紫金{r}{w}有望增长吗",
    "长江电力{r}{w}会不会下滑",
]
_EN_REPORTS = ["Q3", "third-quarter", "interim", "quarterly", "upcoming annual", "next quarterly"]
_EN_WORDS = ["earnings", "results", "profit", "revenue"]
_EN_FRAMES = [
    "Will Midea's {r} {w} grow?",
    "Are CATL's {r} {w} going to beat last year's?",
    "Will Zijin's {r} {w} top the prior year's level?",
    "Is ICBC expected to report higher {r} {w}?",
]


def _forecasts() -> list[str]:
    zh = [
        frame.format(r=report, w=word) for frame, report, word in itertools.product(_ZH_FRAMES, _ZH_REPORTS, _ZH_WORDS)
    ]
    en = [
        frame.format(r=report, w=word) for frame, report, word in itertools.product(_EN_FRAMES, _EN_REPORTS, _EN_WORDS)
    ]
    return zh + en


@pytest.mark.parametrize("query", _forecasts())
def test_report_period_earnings_with_a_modal_are_forecasts(query):
    assert asks_earnings_forecast(query, as_of=REFERENCE)


def test_the_property_set_size():
    assert len(_forecasts()) == 5 * 9 * 5 + 4 * 6 * 4


_ZH_LOOKUPS = [
    "宁德时代{r}{w}是多少",
    "招行{r}{w}多少",
    "美的最新{r}的{w}",
    "宁德时代2025年{r}{w}增长了多少",
    "紫金去年{r}{w}同比增长多少",
]


@pytest.mark.parametrize(
    "query",
    [
        frame.format(r=report, w=word)
        for frame, report, word in itertools.product(_ZH_LOOKUPS, ["三季报", "中报", "年报", "季报"], _ZH_WORDS)
    ],
)
def test_a_report_without_a_modal_or_with_a_past_year_is_a_lookup(query):
    assert not asks_earnings_forecast(query, as_of=REFERENCE)


@pytest.mark.parametrize(
    "query",
    [
        "What were Midea's Q3 earnings?",
        "Midea Q3 results",
        "Did CATL's interim profit beat last year's?",
        "How much did ICBC earn in 2025?",
        "长江电力下一份年报什么时候发布",
        "When is the next annual report for Zijin due?",
        "宁德时代年报利润是不是增长了",
        "美的2025年报净利润比2024年增长了多少",
        "招行会比工行贵吗",  # a comparison of prices, not earnings
    ],
)
def test_past_reports_schedules_and_price_comparisons_are_not_earnings_forecasts(query):
    assert not asks_earnings_forecast(query, as_of=REFERENCE)


@pytest.mark.parametrize(
    "query",
    [
        "Will Midea's Q4 2026 net profit beat last year's?",
        "Can CATL's fourth-quarter profit top last quarter's?",
        "What will Zijin earn in Q4 compared with last year?",
        "How will ICBC's forthcoming annual results look?",
        "CMB's next earnings release: will revenue grow?",
        "宁德时代尚未发布的三季报净利润",
    ],
)
def test_a_past_baseline_or_an_unreleased_report_does_not_hide_the_forecast(query):
    assert asks_earnings_forecast(query, as_of=REFERENCE)


# ---- end to end on the template path: the hedge and the limitation ----


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
    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]), today=lambda: REFERENCE)
    agent = AgentService(runtime, trace_sinks=[])
    yield agent
    runtime.close()


def test_a_chinese_report_forecast_gets_the_earnings_hedge(ext_agent):
    result = ext_agent.chat("美的三季报营收会同比增长吗", session_id="r15-report-zh")
    assert "earnings_forecast_hedge" in result["compliance_notes"]
    assert "FinSight 不预测未来业绩" in result["answer"]
    assert "问题要求预测未来业绩，FinSight 只提供已披露的历史财务数据" in result["limitations"]


def test_an_english_quarter_forecast_gets_the_earnings_hedge(ext_agent):
    result = ext_agent.chat("Will Midea's Q3 profit beat last year's?", session_id="r15-report-en")
    assert "earnings_forecast_hedge" in result["compliance_notes"]
    assert "FinSight does not forecast future earnings" in result["answer"]


def test_the_latest_report_without_a_modal_is_not_hedged(ext_agent):
    result = ext_agent.chat("美的年报净利润是多少", session_id="r15-report-lookup")
    assert "earnings_forecast_hedge" not in result["compliance_notes"]
    assert "不预测未来业绩" not in result["answer"]
