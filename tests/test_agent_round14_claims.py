"""Round-14 claim forms (round-10 review J12), in the engineer's own wording and pairings. Values come from the offline
snapshot extension (data/snapshot/structured_data_ext.json; valuation at the 2026-09-30 close):

* 扣非净利润 / profit_dedt (亿): 宁德时代 645.08 (归母 722.01), 美的集团 412.67 (439.45), 比亚迪 294.46 (326.19)
* 流通市值 / circ_mv (亿): 中芯国际 2240.47 (total 9588.97), 比亚迪 2905.08 (total 7595.54); 贵州茅台 has none
* P/B: 宁德时代 3.55, 比亚迪 3.17, 美的集团 2.89, 招商银行 0.91, 平安银行 0.48; P/E: 比亚迪 25.81, 宁德时代 15.85,
  美的集团 13.77, 紫金矿业 11.71, 工商银行 7.89, 招商银行 6.86, 平安银行 5.17
* ROE (%): 紫金矿业 27.91, 宁德时代 21.42, 美的集团 19.69, 招商银行 11.8, 工商银行 8.68, 平安银行 7.73
* YoY (%): 宁德时代 net profit 42.18, 紫金矿业 net profit 62.01, 比亚迪 revenue 3.46
* net profit (亿): 工商银行 3685.62, 招商银行 1501.81, 平安银行 426.33; revenue: 比亚迪 8039.65, 宁德时代 4237.02

The v1 snapshot (``v1_checker``) has no profit_dedt for any company.
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
    from query_intelligence.service import build_default_service

    return _runner(
        build_default_service(
            use_live_market=False,
            use_live_macro=False,
            use_live_news=False,
            use_live_announcement=False,
            offline_snapshot_ext=True,
        )
    )


@pytest.fixture(scope="module")
def v1_checker():
    from evaluation.agent_eval.runner import build_offline_service

    return _runner(build_offline_service())


def _rows(report):
    return [
        (check.kind, check.target, check.metric, check.comparator, check.claimed, check.reference, check.status)
        for check in report.checks
    ]


# ---- 扣非净利润 and 流通市值 are their own metrics ----


@pytest.mark.parametrize(
    ("claim", "metrics", "verdict"),
    [
        ("比亚迪扣非净利润超过300亿元", ["profit_dedt"], "contradicted"),  # 294.46; 归母 326.19 would pass
        ("美的集团扣非归母净利润约413亿元", ["profit_dedt"], "supported"),  # 412.67
        ("宁德时代净利润超过700亿，扣除非经常性损益后的净利润不到650亿", ["net_profit", "profit_dedt"], "supported"),
        ("CATL's net profit excluding non-recurring items was above RMB65 billion", ["profit_dedt"], "contradicted"),
        ("中芯国际流通市值超过9000亿元", ["circ_mv"], "contradicted"),  # 2240.47; total 9588.97 would pass
        ("比亚迪的流通市值不到3000亿元", ["circ_mv"], "supported"),  # 2905.08; total 7595.54 would fail
        ("SMIC's free-float market cap is about RMB224 billion", ["circ_mv"], "supported"),
        ("中芯国际总市值超过9000亿元", ["market_cap"], "supported"),
    ],
)
def test_recurring_profit_and_free_float_value_are_distinct_metrics(checker, claim, metrics, verdict):
    report = checker(claim)
    assert [check.metric for check in report.checks] == metrics, _rows(report)
    assert report.verdict == verdict, _rows(report)


@pytest.mark.parametrize(
    ("fixture", "claim", "metric"),
    [
        ("v1_checker", "贵州茅台扣非净利润超过800亿元", "profit_dedt"),  # the v1 statement has no 扣非
        ("checker", "贵州茅台流通市值超过2万亿元", "circ_mv"),  # the corrected statement has no 流通市值
        ("checker", "比亚迪扣非净利润同比下降10%", "profit_dedt_yoy"),  # no 扣非 growth in any source
    ],
)
def test_a_missing_recurring_or_free_float_field_is_unverifiable_never_a_neighbour(request, fixture, claim, metric):
    report = request.getfixturevalue(fixture)(claim)
    (check,) = report.checks
    assert check.metric == metric and check.status == "unverifiable", _rows(report)
    assert check.reason in {"no_data", "growth_unavailable"} and check.note, _rows(report)


# ---- two bounds split by but / 但: both halves keep the subject ----


@pytest.mark.parametrize(
    ("claim", "subject", "verdict"),
    [
        ("BYD's P/B is below CATL's but above Midea's", "比亚迪", "supported"),  # 3.17 < 3.55, > 2.89
        ("Midea's P/E is above CATL's but below BYD's", "美的集团", "partially_supported"),  # 13.77 < 15.85
        ("比亚迪的市净率高于美的集团但低于宁德时代", "比亚迪", "supported"),
        ("宁德时代的市盈率高于美的集团但低于紫金矿业", "宁德时代", "partially_supported"),  # 15.85 > 11.71
        ("美的集团的市净率低于宁德时代，但高于比亚迪", "美的集团", "partially_supported"),  # 2.89 < 3.17
        ("美的集团市盈率低于20倍但高于15倍", "美的集团", "partially_supported"),  # 13.77
    ],
)
def test_two_bounds_split_by_but_check_both_halves_of_the_subject(checker, claim, subject, verdict):
    report = checker(claim)
    assert [check.target for check in report.checks] == [subject, subject], _rows(report)
    assert report.verdict == verdict, _rows(report)


# ---- a superlative with the subject first ----


@pytest.mark.parametrize(
    ("claim", "subject", "verdict"),
    [
        ("紫金矿业的ROE在紫金矿业、美的集团、宁德时代三家中最高", "紫金矿业", "supported"),
        ("工商银行的ROE在工行、招行和平安银行三家里最高", "工商银行", "contradicted"),
        ("平安银行是平安银行、招商银行、工商银行中市盈率最低的", "平安银行", "supported"),
        ("招行是招行、工行、平安银行三家里ROE最高的", "招商银行", "supported"),
        ("宁德时代的市盈率在宁德、比亚迪、美的集团三家中最低", "宁德时代", "contradicted"),  # 美的 13.77 is lower
    ],
)
def test_a_superlative_with_the_subject_named_first(checker, claim, subject, verdict):
    report = checker(claim)
    (check,) = report.checks
    assert check.kind == "relation" and check.target == subject, _rows(report)
    assert report.verdict == verdict, _rows(report)


# ---- a year is never a claimed value ----


@pytest.mark.parametrize(
    ("claim", "metric"),
    [
        ("CATL reported revenue of about RMB423.7 billion for 2025", "revenue"),
        ("In FY2025 BYD's net profit was RMB32.6 billion", "net_profit"),
        ("For fiscal 2025, Midea's revenue was about RMB458.5 billion", "revenue"),
        ("比亚迪2025 年营收约8040亿元", "revenue"),
    ],
)
def test_a_year_token_is_never_a_claimed_value(checker, claim, metric):
    report = checker(claim)
    (check,) = report.checks
    assert check.metric == metric and check.claimed != 2025, _rows(report)
    assert report.verdict == "supported", _rows(report)


# ---- "up N% year on year" / "同比增长N%" is the YoY of the metric it names ----


@pytest.mark.parametrize(
    ("claim", "metrics", "verdict"),
    [
        (
            "CATL's net profit came in around RMB72.2 billion in 2025, up 42% year on year",
            ["net_profit", "netprofit_yoy"],
            "supported",
        ),
        ("Zijin Mining's net profit rose about 62% year over year", ["netprofit_yoy"], "supported"),
        ("BYD's revenue grew 3.5% from a year earlier", ["revenue_yoy"], "supported"),
        ("美的集团2025年营收4585亿元，同比增长12%", ["revenue", "revenue_yoy"], "supported"),
    ],
)
def test_a_year_on_year_change_is_the_growth_of_the_named_metric(checker, claim, metrics, verdict):
    report = checker(claim)
    assert [check.metric for check in report.checks] == metrics, _rows(report)
    assert report.verdict == verdict, _rows(report)


def test_a_year_on_year_change_of_a_ratio_is_not_checked_against_its_level(checker):
    report = checker("招商银行ROE同比提升2个百分点")  # the sources have this year's ROE only
    (check,) = report.checks
    assert check.status == "unverifiable" and check.reason == "growth_unavailable", _rows(report)


# ---- "不到 / 超过 X的N倍" is a bound on the multiple ----


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("招商银行市净率还不到平安银行的2倍", "supported"),  # 1.89
        ("工商银行市盈率超过招商银行的1.2倍", "contradicted"),  # 1.15
    ],
)
def test_a_bound_before_x_times_another_value_bounds_the_multiple(checker, claim, verdict):
    report = checker(claim)
    (check,) = report.checks
    assert check.kind == "ratio" and check.comparator in {"lt", "gt"}, _rows(report)
    assert report.verdict == verdict, _rows(report)


def test_a_difference_without_bi_and_a_multiple_bound_in_one_claim(checker):
    report = checker("宁德时代ROE高出美的集团不到两个百分点，但市净率不足美的的1.3倍")  # 1.73; 1.23
    assert [check.kind for check in report.checks] == ["difference", "ratio"], _rows(report)
    assert {check.target for check in report.checks} == {"宁德时代"}, _rows(report)
    assert report.verdict == "supported", _rows(report)


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("招行ROE高出工行三个多百分点", "supported"),  # 3.12
        ("宁德时代的ROE高出美的集团约3个百分点", "contradicted"),  # 1.73
    ],
)
def test_a_difference_written_with_gaochu_and_no_bi(checker, claim, verdict):
    report = checker(claim)
    (check,) = report.checks
    assert check.kind == "difference", _rows(report)
    assert report.verdict == verdict, _rows(report)


# ---- "两个半X": two and a half times X ----


@pytest.mark.parametrize(
    ("claim", "claimed", "verdict"),
    [
        ("工商银行一年的净利润相当于两个半招商银行", 2.5, "supported"),  # 2.454
        ("招商银行净利润差不多相当于三个半平安银行", 3.5, "supported"),  # 3.523
        ("宁德时代的营收相当于两个比亚迪", 2.0, "contradicted"),  # 0.527
    ],
)
def test_n_and_a_half_of_another_company_is_a_multiple(checker, claim, claimed, verdict):
    report = checker(claim)
    (check,) = report.checks
    assert check.kind == "ratio" and check.claimed == pytest.approx(claimed), _rows(report)
    assert report.verdict == verdict, _rows(report)


def test_n_and_a_half_months_is_not_a_multiple():
    from query_intelligence.agent.claim_check import normalise

    assert "倍" not in normalise("比亚迪股价两个半月跌了两成")
    assert "倍" not in normalise("相当于两个半小时的成交")
