"""Round-15 classes from the blind third real-claims set (evaluation/real_claims_web_v3/, after exposure): its README
lists six error classes; these tests use the engineer's own wordings, companies and days, not the set's sentences.
Values from the shipped default data (v1 + the 2026-09-30 extension + the corrections layer):

* 上证指数 2026-09-30: open 3839.253, high 3851.217, low 3833.086, close 3842.195 (2026-09-29 close 3830.451);
  2026-04-22 close 4106.258 (+0.5185%). 创业板指 2026-09-29: open 3136.527, high 3158.91, low 3128.479, close 3142.563;
  2026-04-22 close 3752.759 (+1.7301%, 2026-04-21 close 3688.937)
* 宁德时代 2026-09-30 close 291.11 (+1.5028%), turnover 86.14亿; 2026-04-22 turnover 201.74亿, 2026-04-21 254.12亿
* 创业板ETF (159915) latest bar 2026-04-22, turnover 69.24亿
* net profit (亿): 中国平安 1347.78 (/365 = 3.69), 贵州茅台 823.20 (/365 = 2.26), 宁德时代 722.01 (/365 = 1.98),
  比亚迪 326.19 (/365 = 0.894), 紫金矿业 517.77 (YoY 62.01%), 招商银行 1501.81, 工商银行 3685.62, 五粮液 89.54
  (YoY -71.89%), 中芯国际 50.41 (YoY 36.29%)
* revenue (亿): 比亚迪 8039.65, 紫金矿业 3490.79, 长江电力 862.42 (YoY 2.07%, prior 844.92), 宁德时代 4237.02
  (营业收入 = 营业总收入 by its report notes), 贵州茅台 1688.38 (营业总收入 1720.54 differs), 五粮液 YoY -54.55%,
  中芯国际 YoY 16.49%
* P/E TTM: 招商银行 6.86, 中信证券 10.19; ROE: 招商银行 11.8, 平安银行 7.73
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def checker():
    from evaluation.agent_eval.runner import build_offline_service
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    service = build_offline_service(snapshot_ext=True, snapshot_fix=True)
    registry = build_registry_for_service(service)

    def run(claim: str):
        return check_claim(claim, service=service, registry=registry, zh=not claim.isascii())

    return run


def _rows(report):
    return [
        (check.target, check.kind, check.metric, check.comparator, check.claimed, check.actual, check.status,
         check.reason, check.note[:80])
        for check in report.checks
    ]  # fmt: skip


def _verdict(checker, claim: str, verdict: str):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    return report


# ---- (1) metric wording ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("claim", ["中信证券目前的市盈TTM约为10.2倍", "招商银行市盈率(TTM)不到7倍"])
def test_pe_ttm_written_without_the_rate_character(checker, claim):
    report = _verdict(checker, claim, "supported")
    assert [check.metric for check in report.checks] == ["pe_ttm"], _rows(report)


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("中国平安2025年平均每天净赚约3.7亿元", "supported"),  # 1347.78 / 365 = 3.69
        ("宁德时代2025年日赚近2亿元", "supported"),  # 1.98
        ("贵州茅台2025年一天能赚5个亿以上", "contradicted"),  # 2.26
        ("BYD earned about RMB89 million a day in 2025", "supported"),  # 0.894亿
    ],
)
def test_a_daily_average_profit_is_the_years_net_profit_over_365_days(checker, claim, verdict):
    report = _verdict(checker, claim, verdict)
    (check,) = report.checks
    assert check.metric == "net_profit_daily" and "365" in check.note, _rows(report)


def test_a_days_earnings_without_an_average_word_are_not_a_daily_average(checker):
    report = checker("宁德时代今天赚了2亿元")
    assert all(check.metric != "net_profit_daily" for check in report.checks), _rows(report)


def test_bare_profit_is_net_profit_and_total_profit_is_not_served(checker):
    report = _verdict(checker, "工商银行2025年的利润约为3686亿元", "supported")
    assert report.checks[0].metric == "net_profit" and "利润" in report.checks[0].note, _rows(report)
    report = _verdict(checker, "宁德时代2025年利润总额超过800亿元", "unverifiable")
    assert all(check.metric != "net_profit" for check in report.checks), _rows(report)


def test_abbreviated_bar_words(checker):
    report = _verdict(checker, "创业板指9月29日开盘3136.53点，最高3158.91点，最低3128.48点，收报3142.56点", "supported")
    assert [check.metric for check in report.checks] == ["open", "high", "low", "close"], _rows(report)


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("创业板指4月22日涨了63.82点", "supported"),  # 3752.759 - 3688.937
        ("上证指数9月30日比前一日上涨11.74点", "supported"),  # 3842.195 - 3830.451
        ("创业板指4月22日涨了100点", "contradicted"),
    ],
)
def test_a_change_in_points_or_yuan_is_the_absolute_daily_change(checker, claim, verdict):
    report = _verdict(checker, claim, verdict)
    (check,) = report.checks
    assert check.metric == "change_1d", _rows(report)


@pytest.mark.parametrize(
    ("claim", "metrics"),
    [
        ("Ping An posted a 6.5% year-on-year increase in net profit for 2025", ["netprofit_yoy"]),
        ("China Merchants Bank booked RMB 150.2 billion in net profit for FY2025", ["net_profit"]),
        ("BYD reported RMB 804 billion in revenue and RMB 32.6 billion in net profit", ["revenue", "net_profit"]),
    ],
)
def test_an_english_number_before_its_metric(checker, claim, metrics):
    report = _verdict(checker, claim, "supported")
    assert [check.metric for check in report.checks] == metrics, _rows(report)


@pytest.mark.parametrize(
    "claim",
    [
        "长江电力2025年营收862.42亿元，上年为844.9亿元",
        "Midea's 2025 revenue of RMB458.5 billion compared with the previous year's RMB409 billion",
    ],
)
def test_a_prior_year_value_is_derived_from_the_level_and_its_yoy(checker, claim):
    report = _verdict(checker, claim, "supported")
    assert "prior year" in report.checks[-1].note, _rows(report)


# ---- (2) number-to-metric binding ------------------------------------------------------------------------------------


def test_an_amount_after_dao_belongs_to_the_metric_of_its_clause(checker):
    # 紫金: revenue 3490.79, net profit YoY 62.01, net profit 517.77
    report = _verdict(checker, "紫金矿业：营收3491亿，利润增62%到518亿", "supported")
    assert [check.metric for check in report.checks] == ["revenue", "netprofit_yoy", "net_profit"], _rows(report)


@pytest.mark.parametrize(
    ("claim", "verdict", "metric"),
    [
        ("宁德时代4月22日成交额201.7亿元，较前一交易日减少约52亿元", "supported", "amount_change_1d"),
        ("宁德时代4月22日成交额较前一交易日缩减约两成", "supported", "amount_change_pct_1d"),  # -20.61%
        ("宁德时代4月22日成交额较前一交易日增加52亿元", "contradicted", "amount_change_1d"),
    ],
)
def test_a_change_of_turnover_from_the_previous_session(checker, claim, verdict, metric):
    report = _verdict(checker, claim, verdict)
    assert report.checks[-1].metric == metric, _rows(report)


# ---- (3) dates -------------------------------------------------------------------------------------------------------


def test_a_month_and_day_without_a_year_use_that_days_bar(checker):
    report = _verdict(checker, "4月22日，创业板指收盘站上3750点", "supported")
    assert report.checks[0].as_of == "2026-04-22", _rows(report)


def test_a_daily_figure_with_no_date_that_does_not_fit_the_latest_session_is_unverifiable(checker):
    report = _verdict(checker, "截至收盘，宁德时代下跌2.51%", "unverifiable")  # the latest session is +1.50%
    assert report.checks[0].reason == "period_mismatch" and "no date" in report.checks[0].note, _rows(report)
    _verdict(checker, "截至收盘，宁德时代上涨1.50%", "supported")
    _verdict(checker, "宁德时代今天下跌2.51%", "contradicted")  # "今天" anchors the latest session


# ---- (4) entity resolution -------------------------------------------------------------------------------------------


def test_renminbi_is_not_a_company(checker):
    report = _verdict(checker, "比亚迪全年营业收入人民币8039.65亿元", "supported")
    assert {check.target for check in report.checks} == {"比亚迪"}, _rows(report)


def test_a_fund_code_in_brackets_names_the_fund_and_feeder_codes_are_asides(checker):
    report = _verdict(checker, "创业板ETF（159915，另有联接基金004744）成交额约69.2亿元", "supported")
    assert report.checks[0].target == "创业板ETF", _rows(report)


def test_a_respectively_list_binds_element_wise_with_a_company_outside_the_data(checker):
    report = checker("工商银行、招商银行、宁波银行、平安银行的净利润分别为3686亿、1502亿、271亿、426亿元")
    rows = [(check.target, check.status) for check in report.checks]
    assert rows[0] == ("工商银行", "supported") and rows[1] == ("招商银行", "supported"), _rows(report)
    assert rows[2][1] == "unverifiable" and rows[3] == ("平安银行", "supported"), _rows(report)


def test_a_vs_pair_binds_in_order_and_a_later_forecast_clause_does_not_make_it_a_forecast(checker):
    report = _verdict(checker, "平安银行与招行的ROE（7.7% vs 11.8%）预计2026年仍将承压", "supported")
    assert [check.target for check in report.checks] == ["平安银行", "招商银行"], _rows(report)


# ---- (5) a decline makes the figure negative -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("五粮液2025年净利润降幅为71.9%，营收降幅为54.6%", "supported"),
        ("五粮液2025年营收减少了约55%", "supported"),
        ("五粮液2025年营收降幅约为30%", "contradicted"),
    ],
)
def test_a_decline_word_makes_the_change_negative(checker, claim, verdict):
    report = _verdict(checker, claim, verdict)
    assert all(check.claimed is not None and check.claimed < 0 for check in report.checks), _rows(report)


def test_a_comparison_with_another_growth_rate_states_its_value(checker):
    # 中芯国际 net profit YoY 36.29 > revenue YoY 16.49 (stated)
    report = _verdict(checker, "中芯国际2025年净利润增幅36.3%，高于营收增幅的16.5%", "supported")
    metrics = [check.metric for check in report.checks]
    assert "netprofit_yoy" in metrics and "revenue_yoy" in metrics, _rows(report)


# ---- (6) 营业总收入 where the report states it equals 营业收入 -------------------------------------------------------


def test_total_operating_revenue_is_checkable_where_the_report_says_it_equals_operating_revenue(checker):
    report = _verdict(checker, "宁德时代2025年营业总收入4237亿元", "supported")
    assert "营业总收入" in report.checks[0].note, _rows(report)
    report = _verdict(checker, "贵州茅台2025年营业总收入1688.4亿元", "unverifiable")  # 1720.54 by its report
    assert report.checks[0].reason == "definition_mismatch", _rows(report)


def test_cheng_fen_is_a_constituent_not_tenths(checker):
    # "成份" (constituent) after a number is not 成 (tenths): "科创50成份" is no 500%
    report = _verdict(checker, "科创50成份ETF（588000）今天成交额约68.5亿元", "supported")
    assert [check.metric for check in report.checks] == ["amount"], _rows(report)


def test_an_english_period_end_close(checker):
    report = _verdict(checker, "The Shanghai Composite finished September at about 3,840", "supported")
    assert [check.metric for check in report.checks] == ["close"], _rows(report)


def test_a_published_price_with_the_current_price_word_is_anchored_to_the_latest_session(checker):
    # "股价" is the current price: a wrong figure is contradicted by the latest close, not left unverifiable
    report = _verdict(checker, "宁德时代股价低于250元", "contradicted")
    assert report.checks[0].reason is None, _rows(report)


def test_codes_inside_a_bracket_that_opens_with_words_are_asides(checker):
    # the bracket opens with the fund's own code (not a separate target) and then names feeder funds by code
    report = _verdict(checker, "创业板ETF（159915，旗下联接基金A类代码110026）当天成交额约69.2亿元", "supported")
    assert report.checks[0].target == "创业板ETF", _rows(report)
