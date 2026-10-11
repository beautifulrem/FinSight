"""Round 17 (review 13, M12b-e): weak direction words, same-company growth comparisons, 扣非 vs 归母 and growth
multiples in the claim checker (the definitions of M12a and the missing corrections layer of M13 are in
tests/test_agent_round17_claims.py, with the value table this file also uses).

* M12b: weak direction words without a number are bounded growth parts, with the bands written in docs/claim-check.md
  before the rules (flat |g| <= 2, slight 0 < |g| <= 5, modest 0 < |g| <= 10, sharp |g| >= 20).
* M12c: two growth rates of one company compared with 快于 / 慢于 / 明显 / 其 / its.
* M12d: an elided 扣非 vs 归母 comparison ("…归母净利润X亿元，扣非净利润更低").
* M12e: one growth rate as a multiple of the other ("净利润增速是营收增速的N倍").

The wordings, companies and pairings are the engineer's own (none is a review sentence or a slice text; the guard in
tests/test_agent_eval.py checks this).
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
        (check.target, check.kind, check.metric, check.comparator, check.claimed, check.claimed_high, check.actual,
         check.reference, check.reference_metric, check.status, check.reason, check.note[:70])
        for check in report.checks
    ] + [("unchecked", item.text) for item in report.unchecked]  # fmt: skip


# ---- M12b: weak direction words as bounded growth parts ------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "metric", "verdict"),
    [
        ("招商银行营收同比基本持平", "revenue_yoy", "supported"),  # 0.01, flat
        ("贵州茅台2025年营业收入大致持平", "revenue_yoy", "supported"),  # -1.21
        ("比亚迪营收基本持平", "revenue_yoy", "contradicted"),  # 3.46 is not flat
        ("比亚迪营收同比微增", "revenue_yoy", "supported"),  # 3.46
        ("贵州茅台净利润同比微降", "netprofit_yoy", "supported"),  # -4.53
        ("工商银行净利润略有下降", "netprofit_yoy", "contradicted"),  # +1.04: the wrong way
        ("长江电力净利润同比微增", "netprofit_yoy", "contradicted"),  # 6.13 is more than slight
        ("平安银行净利润同比略有下滑", "netprofit_yoy", "supported"),  # -4.21
        ("长江电力2025年净利润小幅增长", "netprofit_yoy", "supported"),  # 6.13
        ("平安银行营收小幅下降", "revenue_yoy", "contradicted"),  # -10.40 is more than modest
        ("紫金矿业净利润同比大增", "netprofit_yoy", "supported"),  # 62.01
        ("五粮液营收同比大幅下滑", "revenue_yoy", "supported"),  # -54.55
        ("比亚迪净利润大幅下降", "netprofit_yoy", "contradicted"),  # -18.82 is under the sharp band
        ("美的集团净利润同比大增", "netprofit_yoy", "contradicted"),  # 14.87
        ("Kweichow Moutai's revenue edged down year on year", "revenue_yoy", "supported"),  # -1.21
        ("CMB's net profit was broadly flat in 2025", "netprofit_yoy", "supported"),  # 1.05
        ("Wuliangye's net profit plunged in 2025", "netprofit_yoy", "supported"),  # -71.89
        ("Midea's revenue surged year on year", "revenue_yoy", "contradicted"),  # 12.11
        ("China Yangtze Power's net profit rose modestly in 2025", "netprofit_yoy", "supported"),  # 6.13
        ("ICBC's net profit edged lower", "netprofit_yoy", "contradicted"),  # +1.04
    ],
)
def test_a_weak_direction_word_is_a_bounded_growth_part(checker, claim, metric, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    (check,) = report.checks
    assert check.metric == metric and "band" in check.note, _rows(report)


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("长江电力净利润没有大幅增长", "supported"),  # 6.13 < 20
        ("紫金矿业净利润并非小幅增长", "supported"),  # 62.01 > 10
        ("招商银行净利润并没有基本持平", "contradicted"),  # 1.05 is flat
    ],
)
def test_a_negated_weak_direction_word_is_outside_its_band(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("招商银行营收3375亿元，净利润同比大增", "partially_supported"),  # NP +1.05
        ("比亚迪2025年净利润326亿元，同比小幅下降", "partially_supported"),  # -18.82
        ("中信证券营收748.5亿元，净利润同比大增", "supported"),  # +37.26
        ("工商银行营收同比基本持平，净利润微增", "supported"),  # 2.00 (on the band's edge), 1.04
    ],
)
def test_a_weak_direction_clause_next_to_a_number(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    assert len(report.checks) == 2, _rows(report)


@pytest.mark.parametrize(
    ("claim", "claimed", "verdict"),
    [
        ("贵州茅台净利润同比微降4.5%", -4.5, "supported"),  # -4.53
        ("比亚迪营收同比微增3.5%", 3.5, "supported"),  # 3.46
        ("工商银行营收同比微降2%", -2.0, "contradicted"),  # +2.00
    ],
)
def test_a_weak_direction_word_with_a_number_gives_its_sign(checker, claim, claimed, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    (check,) = report.checks
    assert check.claimed == claimed, _rows(report)


def test_a_weak_price_move_keeps_the_daily_convention(checker):
    report = checker("贵州茅台今天微涨")
    (check,) = report.checks
    assert check.metric == "pct_change_1d" and "band" not in check.note, _rows(report)


# ---- M12c: two growth rates of one company -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("中信证券营收增速快于其净利润增速", "contradicted"),  # 28.80 < 37.26
        ("宁德时代净利润增速明显快于营收增速", "supported"),  # 42.18 > 17.04
        ("紫金矿业营收增速明显慢于其净利润增速", "supported"),  # 14.96 < 62.01
        ("工商银行的净利润增速慢于营收增速", "supported"),  # 1.04 < 2.00
        ("CITIC Securities' revenue grew faster than its net profit", "contradicted"),
        ("China Yangtze Power's net profit growth outpaced its revenue growth", "supported"),  # 6.13 > 2.07
    ],
)
def test_two_growth_rates_of_one_company_with_qi_or_mingxian(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    (check,) = report.checks
    assert {check.metric, check.reference_metric} == {"revenue_yoy", "netprofit_yoy"}, _rows(report)


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("长江电力2025年营收862亿元，增速慢于其净利润增速", "supported"),  # 2.07 < 6.13
        ("美的集团营收同比增长12.1%，增速快于其净利润增速", "partially_supported"),  # 12.11 < 14.87
        ("比亚迪营收增长3.5%，增速明显快于其净利润", "supported"),  # 3.46 > -18.82
    ],
)
def test_a_growth_comparison_after_a_stated_figure(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    relation = report.checks[-1]
    assert (relation.metric, relation.reference_metric) == ("revenue_yoy", "netprofit_yoy"), _rows(report)


# ---- M12d: 扣非 vs 归母, elided --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("中信证券2025年归母净利润300.8亿元，扣非净利润更低", "partially_supported"),  # 扣非 303.03 > 300.76
        ("宁德时代归母净利润722亿元，扣非净利润更低", "supported"),  # 645.08
        ("美的集团净利润439亿元，扣非后净利润要低一些", "supported"),  # 412.67
        ("紫金矿业归母净利润518亿元，扣非净利润则更高", "partially_supported"),  # 507.24
        ("Ping An's attributable net profit was RMB134.8 billion, and its recurring net profit was even lower",
         "partially_supported"),  # 1437.73 > 1347.78
    ],
)  # fmt: skip
def test_an_elided_recurring_profit_comparison_is_checked(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    relation = report.checks[-1]
    assert (relation.metric, relation.reference_metric, relation.kind) == ("profit_dedt", "net_profit", "relation"), (
        _rows(report)
    )


def test_a_written_out_recurring_profit_comparison_still_works(checker):
    report = checker("招商银行扣非净利润高于归母净利润")  # 1500.07 < 1501.81
    assert report.verdict == "contradicted", _rows(report)


# ---- M12e: one growth rate as a multiple of the other --------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("宁德时代净利润增速约为营收增速的2.5倍", "supported"),  # 42.18 / 17.04 = 2.48
        ("中国平安净利润的增速大约是营收增速的3倍", "supported"),  # 6.50 / 2.10 = 3.10
        ("长江电力净利润增速接近营收增速的3倍", "supported"),  # 6.13 / 2.07 = 2.96
        ("中芯国际净利润增速是营收增速的3倍多", "contradicted"),  # 36.29 / 16.49 = 2.20
        ("美的集团净利润增速是营收增速的两倍", "contradicted"),  # 14.87 / 12.11 = 1.23
        ("CATL's net profit growth was about 2.5 times its revenue growth", "supported"),
    ],
)
def test_a_growth_rate_as_a_multiple_of_another(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    (check,) = report.checks
    assert (check.kind, check.metric, check.reference_metric) == ("ratio", "netprofit_yoy", "revenue_yoy"), _rows(
        report
    )


def test_a_growth_multiple_against_a_fall_is_contradicted(checker):
    report = checker("比亚迪净利润增速是营收增速的2倍")  # -18.82 against +3.46
    assert report.verdict == "contradicted", _rows(report)


def test_a_figure_as_a_multiple_of_another_figure_of_the_same_company(checker):
    report = checker("中国平安扣非净利润是归母净利润的1.07倍")  # 1437.73 / 1347.78 = 1.067
    assert report.verdict == "supported", _rows(report)
    (check,) = report.checks
    assert (check.kind, check.metric, check.reference_metric) == ("ratio", "profit_dedt", "net_profit"), _rows(report)
