"""Round-14 claim classes from the blind second real-claims set (evaluation/real_claims_web_v2/, after exposure): its
README lists the error classes; these tests use the engineer's own wordings, pairings and dates, not the set's
sentences. Values from the shipped default data (v1 + the 2026-09-30 extension + the corrections layer):

* definitions: ROE period-end for the extension companies, weighted for 贵州茅台; EPS diluted for 美的集团, basic for
  比亚迪; revenue 营业总收入 for 美的集团; net-profit YoY on total net profit for 比亚迪 / 工商银行 / 紫金矿业, 归母 for
  中芯国际; P/E is TTM for the extension, static (annual 归母 profit) for 贵州茅台
* daily bars from the extension's history: 中芯国际 2026-04-22 close 107.87, +0.3815%, 成交额 40.23亿, volume
  37,521,674 shares; 比亚迪 2026-04-22 close 101.62, +0.0886%; 宁德时代 2026-09-29 close 286.8, -1.7775%
* 平安银行 total market value 2245.26亿 at the 2026-09-30 close (valuation date 2026-09-30)
* net profit (亿): 工商银行 3685.62 (YoY 1.041%), 长江电力 345.03, 宁德时代 722.01 (YoY 42.18%); revenue 宁德时代
  4237.02 (YoY 17.04%), 比亚迪 8039.65 (YoY 3.46%); 扣非: 比亚迪 294.46, 美的集团 412.67
"""

from __future__ import annotations

import pytest


def _runner(service):
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    registry = build_registry_for_service(service)

    def run(claim: str):
        return check_claim(claim, service=service, registry=registry, zh=not claim.isascii())

    return run


@pytest.fixture(scope="module")
def checker():
    from evaluation.agent_eval.runner import build_offline_service

    return _runner(build_offline_service(snapshot_ext=True, snapshot_fix=True))


@pytest.fixture(scope="module")
def v1_checker():
    from evaluation.agent_eval.runner import build_offline_service

    return _runner(build_offline_service())


def _rows(report):
    return [
        (check.target, check.metric, check.comparator, check.claimed, check.actual, check.status, check.reason)
        for check in report.checks
    ]


# ---- (1) a named definition is checked against a field with that definition, or not at all ----


@pytest.mark.parametrize(
    "claim",
    [
        "宁德时代加权平均净资产收益率为21.4%",  # the served 21.42 is period-end
        "美的集团基本每股收益5.76元",  # the served 5.76 is diluted
        "比亚迪稀释每股收益3.58元",  # the served 3.58 is basic
        "宁德时代的动态市盈率约15.9倍",  # the served 15.85 is TTM
        "美的集团营业收入4585亿元",  # the served 4585.02 is 营业总收入
        "比亚迪归母净利润同比下降18.8%",  # the served -18.82 is on total net profit
    ],
)
def test_a_named_definition_other_than_the_served_one_is_unverifiable(checker, claim):
    report = checker(claim)
    (check,) = report.checks
    assert check.status == "unverifiable" and check.reason == "definition_mismatch", _rows(report)
    assert check.note, _rows(report)


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("贵州茅台加权平均ROE为32.53%", "supported"),  # weighted, served weighted
        ("美的集团稀释每股收益5.76元", "supported"),
        ("美的集团营业总收入超过4500亿元", "supported"),
        ("宁德时代的滚动市盈率约15.9倍", "supported"),
        ("中芯国际归母净利润同比增长36.3%", "supported"),  # 归母 YoY served
        ("宁德时代归母净利润722亿元", "supported"),
        ("比亚迪净利润同比下降18.8%", "supported"),  # no definition named: the served value
    ],
)
def test_a_named_definition_that_matches_the_served_one_is_checked(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)


def test_the_definition_carries_to_the_growth_and_the_parallel_clause(checker):
    report = checker("紫金矿业归母净利润约518亿元，同比增长62%")  # the YoY is on total net profit
    assert [check.metric for check in report.checks] == ["net_profit", "netprofit_yoy"], _rows(report)
    assert [check.status for check in report.checks] == ["supported", "unverifiable"], _rows(report)
    report = checker("宁德时代加权平均ROE约21%，连续三年保持在20%以上")
    assert {check.reason for check in report.checks} == {"definition_mismatch"}, _rows(report)


def test_a_named_definition_the_source_does_not_state_is_compared_with_a_note(v1_checker):
    report = v1_checker("贵州茅台加权平均ROE超过30%")  # the v1 statement states no definitions
    (check,) = report.checks
    assert check.status != "unverifiable" and "basis" in check.note, _rows(report)


# ---- (2) a price on a stated date uses that day's bar ----


@pytest.mark.parametrize(
    ("claim", "metrics"),
    [
        ("截至2026年4月22日收盘，比亚迪收于101.62元，微涨0.09%", ["close", "pct_change_1d"]),
        ("2026年9月29日宁德时代收报286.8元，下跌1.78%", ["close", "pct_change_1d"]),
        ("2026年4月22日中芯国际成交额约40.2亿元，成交量37.52万手", ["amount", "volume"]),
    ],
)
def test_a_price_on_a_stated_date_uses_that_days_bar(checker, claim, metrics):
    report = checker(claim)
    assert [check.metric for check in report.checks] == metrics, _rows(report)
    assert report.verdict == "supported", _rows(report)


def test_a_date_outside_the_history_is_a_period_mismatch(checker):
    report = checker("2025年5月6日中芯国际收于80元")  # the history starts on 2025-07-01
    (check,) = report.checks
    assert check.status == "unverifiable" and check.reason == "period_mismatch", _rows(report)


def test_a_turnover_rate_is_not_a_daily_change(checker):
    report = checker("2026年4月22日中芯国际换手率1.88%")
    (check,) = report.checks
    assert check.metric != "pct_change_1d" and check.status == "unverifiable", _rows(report)


# ---- (3) market value on the price date ----


def test_market_value_on_the_valuation_date_is_checked(checker):
    report = checker("截至2026年9月30日收盘，平安银行总市值约2245亿元")
    (check,) = report.checks
    assert check.metric == "market_cap" and check.status == "supported", _rows(report)
    report = checker("截至2026年4月22日收盘，平安银行总市值约2245亿元")  # no market value for that day
    (check,) = report.checks
    assert check.status == "unverifiable" and check.reason == "period_mismatch", _rows(report)


# ---- (4) English names, YoY wording and currencies ----


@pytest.mark.parametrize(
    ("claim", "target"),
    [
        ("Industrial & Commercial Bank of China earned net profit of about RMB368.6 billion", "工商银行"),
        ("Industrial and Commercial Bank of China's net profit was about RMB368.6 billion", "工商银行"),
        ("China Yangtze Power reported net profit of RMB34.5 billion", "长江电力"),
    ],
)
def test_english_company_names_resolve_to_the_right_company(checker, claim, target):
    report = checker(claim)
    (check,) = report.checks
    assert check.target == target and check.status == "supported", _rows(report)


def test_a_change_named_with_a_fundamental_is_its_growth_not_a_daily_move(checker):
    report = checker("Midea's diluted earnings per share rose 5% to RMB5.76")
    assert [check.metric for check in report.checks] == ["eps", "eps"], _rows(report)
    growth, level = report.checks
    assert growth.reason == "growth_unavailable" and level.status == "supported", _rows(report)


def test_net_profit_attributable_to_shareholders_is_a_net_profit_phrase(checker):
    report = checker("Net profit attributable to shareholders of CATL was about RMB72.2 billion")
    (check,) = report.checks
    assert check.metric == "net_profit" and check.status == "supported", _rows(report)


@pytest.mark.parametrize(
    "claim",
    ["BYD's revenue was about US$110 billion", "宁德时代营收约590亿美元", "CATL's net profit was about $10 billion"],
)
def test_a_foreign_currency_amount_is_not_compared_with_a_cny_figure(checker, claim):
    report = checker(claim)
    (check,) = report.checks
    assert check.status == "unverifiable" and check.reason == "currency", _rows(report)


# ---- (5) traditional characters ----


@pytest.mark.parametrize(
    ("claim", "metrics"),
    [
        ("比亞迪年度營收8,039.65億元，年增3.46%", ["revenue", "revenue_yoy"]),
        ("寧德時代淨利潤約722億元", ["net_profit"]),
        ("工商銀行每股盈餘1.0元", ["eps"]),
    ],
)
def test_traditional_characters_are_read(checker, claim, metrics):
    report = checker(claim)
    assert [check.metric for check in report.checks] == metrics, _rows(report)
    assert report.verdict == "supported", _rows(report)
    assert report.claim == claim  # the report keeps the claim as written


# ---- (6) 扣非 written out or in brackets ----


@pytest.mark.parametrize("claim", ["比亚迪扣除非经常性损益的净利润约294亿元", "美的集团归母净利润（扣非）约413亿元"])
def test_recurring_profit_written_out_or_in_brackets(checker, claim):
    report = checker(claim)
    (check,) = report.checks
    assert check.metric == "profit_dedt" and check.status == "supported", _rows(report)


# ---- (7) close words ----


@pytest.mark.parametrize("claim", ["比亚迪报83.31元，涨1.57%", "招商银行报收41.26元", "宁德时代收报291.11元/股"])
def test_bao_and_baoshou_are_the_close(checker, claim):
    report = checker(claim)
    assert report.checks[0].metric == "close", _rows(report)
    assert report.verdict == "supported", _rows(report)


# ---- (8) a base year is not the claim's period ----


@pytest.mark.parametrize(
    "claim",
    ["和2024年相比，比亚迪营收增长约3.5%", "比亚迪营收较 2024 年增长约3.5%", "与2024年对比，宁德时代营收增长了17%"],
)
def test_the_year_compared_with_is_not_the_period(checker, claim):
    report = checker(claim)
    (check,) = report.checks
    assert check.metric == "revenue_yoy" and check.status == "supported", _rows(report)


# ---- a year-on-year change in amount is derived from the YoY ----


@pytest.mark.parametrize(
    ("claim", "status"),
    [
        ("工商银行净利润比上年增加约38亿元", "supported"),  # 3685.62 - 3685.62 / 1.01041 = 37.97
        ("宁德时代营收同比增加约617亿元", "supported"),  # 4237.02 - 4237.02 / 1.170406 = 616.9
        ("宁德时代营收同比增加约500亿元", "contradicted"),
    ],
)
def test_a_year_on_year_change_in_amount_is_derived(checker, claim, status):
    report = checker(claim)
    (check,) = report.checks
    assert check.status == status and "derived" in check.note, _rows(report)


def test_a_derived_change_with_another_named_definition_is_not_checked(checker):
    report = checker("工商银行归母净利润比上年增加约38亿元")  # the YoY is on total net profit
    (check,) = report.checks
    assert check.reason == "definition_mismatch", _rows(report)


# ---- (9) companies outside the data ----


@pytest.mark.parametrize(
    "claim",
    ["宁德时代的净利润高于比亚迪、蔚来和小鹏三家的利润之和", "宁德时代净利润超过比亚迪、理想、零跑的净利润总和"],
)
def test_a_sum_with_companies_outside_the_data_is_unverifiable(checker, claim):
    report = checker(claim)
    assert report.checks and all(check.status == "unverifiable" for check in report.checks), _rows(report)
