"""Round-15 claim classes (review 11, K6): published-style claims the checker got wrong, in the engineer's own wording
(the review's sentences are not copied). Values from the shipped default data (v1 + the 2026-09-30 extension + the
corrections layer):

* total market value (亿, 2026-09-30): 宁德时代 13470.39, 工商银行 29510.44, 招商银行 10405.71, 紫金矿业 7921.40,
  美的集团 6111.58, 平安银行 2245.26; 贵州茅台 17650.75 (2026-04-22)
* P/E: 招商银行 6.86, 工商银行 7.89, 平安银行 5.17 (mean 6.64); P/B: 紫金矿业 3.92, 中信证券 1.37, 比亚迪 3.17,
  工商银行 0.74; ROE: 宁德时代 21.42, 比亚迪 13.24, 美的集团 19.69 (mean 18.12)
* 归母 net profit (亿): 工商银行 3685.62, 招商银行 1501.81, 平安银行 426.33, 中信证券 300.76, 宁德时代 722.01, 比亚迪
  326.19, 美的集团 439.45, 紫金矿业 517.77, 中国平安 1347.78; 扣非: 宁德时代 645.08, 比亚迪 294.46, 美的集团 412.67,
  紫金矿业 507.24, 中国平安 1437.73
* revenue (亿): 工商银行 8382.70, 招商银行 3375.32, 平安银行 1314.42, 美的集团 4585.02 (YoY 12.11%), 宁德时代 4237.02
* net-profit YoY of 宁德时代 42.18% is on total net profit (not 归母)
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
        (check.target, check.kind, check.metric, check.comparator, check.claimed, check.actual, check.reference,
         check.reference_value, check.status, check.reason)
        for check in report.checks
    ]  # fmt: skip


def _verdict(checker, claim: str, verdict: str):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    return report


# ---- trillion units ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("宁德时代9月30日的总市值约为1.35万亿元", "supported"),
        ("CATL's market capitalisation was roughly RMB1.35 trillion on September 30", "supported"),
        ("ICBC's market cap is about 2.95 trillion yuan", "supported"),
        ("CMB market cap of about CNY 1.04tn", "supported"),
        ("Kweichow Moutai's market value is about 2.5 trillion yuan", "contradicted"),  # 1.77 trillion
    ],
)
def test_trillion_amounts_are_read_in_both_languages(checker, claim, verdict):
    report = _verdict(checker, claim, verdict)
    assert [check.metric for check in report.checks] == ["market_cap"], _rows(report)


# ---- means over several named companies --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim", "metric", "verdict", "mean"),
    [
        ("招商银行、工商银行和平安银行三家的市盈率平均下来约6.6倍", "pe_ttm", "supported", 6.64),
        ("宁德时代、比亚迪、美的集团的ROE均值约18%", "roe", "supported", 18.1167),
        ("紫金矿业和中信证券的平均市净率约为4倍", "pb", "contradicted", 2.645),
        ("The average P/E of CMB, ICBC and Ping An Bank is roughly 6.6x", "pe_ttm", "supported", 6.64),
    ],
)
def test_a_mean_of_several_companies_is_checked_against_their_mean(checker, claim, metric, verdict, mean):
    report = _verdict(checker, claim, verdict)
    (check,) = report.checks
    assert check.kind == "mean" and check.metric == metric, _rows(report)
    assert check.actual == pytest.approx(mean, abs=1e-3), _rows(report)
    assert len(check.operands) >= 2 and len(check.operand_evidence_ids) == len(check.operands)


def test_an_industry_average_is_not_a_mean_of_the_named_companies(checker):
    report = checker("贵州茅台市盈率21.4倍，低于白酒行业平均")
    assert all(check.kind != "mean" for check in report.checks), _rows(report)


def test_a_weighted_average_roe_is_a_definition_not_a_mean(checker):
    report = checker("贵州茅台加权平均ROE为32.53%")
    assert report.verdict == "supported" and report.checks[0].kind == "value", _rows(report)


# ---- a total of several companies as the compared side (never its first operand alone) ----------------------------


@pytest.mark.parametrize(
    ("claim", "verdict", "reference_total"),
    [
        # 7921.40 < 6111.58 + 2245.26 = 8356.84: false, although 紫金 is above 美的 alone
        ("紫金矿业的总市值比美的集团和平安银行加起来还高", "contradicted", 835684176537.86),
        ("Zijin Mining's market value exceeds that of Midea Group and Ping An Bank combined", "contradicted",
         835684176537.86),
        # 3685.62 > 1501.81 + 426.33 + 300.76 = 2228.90
        ("工商银行一家的净利润，超过了招行、平安银行和中信证券三家之和", "supported", 222890000000.0),
        ("ICBC's net profit is larger than CMB's, Ping An Bank's and CITIC Securities' combined", "supported",
         222890000000.0),
    ],
)  # fmt: skip
def test_a_total_of_several_companies_as_the_compared_side(checker, claim, verdict, reference_total):
    report = _verdict(checker, claim, verdict)
    (check,) = report.checks
    assert check.kind == "relation" and "+" in str(check.reference), _rows(report)
    assert check.reference_value == pytest.approx(reference_total, rel=1e-9), _rows(report)
    assert not report.unchecked, report.unchecked


def test_a_difference_from_a_total_of_several_companies(checker):
    # 3685.62 - 2228.90 = 1456.72亿: "1450多亿" is 1450-1460
    report = _verdict(checker, "工行净利润比招行、平安银行、中信证券三家加起来还多1450多亿元", "supported")
    (check,) = report.checks
    assert check.kind == "difference" and check.difference == pytest.approx(145672000000.0), _rows(report)


def test_a_total_of_several_companies_as_the_subject_of_a_relation(checker):
    # 6111.58 + 2245.26 = 8356.84 > 7921.40 (平安银行 alone is far below 紫金)
    report = _verdict(checker, "美的集团和平安银行的总市值合计超过紫金矿业", "supported")
    (check,) = report.checks
    assert check.target == "美的集团+平安银行" and check.actual == pytest.approx(835684176537.86), _rows(report)


def test_a_total_with_a_company_outside_the_data_stays_unverifiable(checker):
    report = _verdict(checker, "宁德时代的净利润超过了比亚迪、蔚来和小鹏三家的利润之和", "unverifiable")
    assert report.checks[0].reason == "no_data", _rows(report)


# ---- a share of an N-company total --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim", "verdict", "share"),
    [
        ("招商银行的净利润占招行、工行、平安银行三家合计的两成多", "supported", 0.2675),  # 1501.81 / 5613.76
        ("工商银行的营收占工行、招行、平安银行三家总和的六成以上", "supported", 0.6413),  # 8382.70 / 13072.44
        ("平安银行的净利润在招行、工行、平安银行三家合计中占约15%", "contradicted", 0.0759),  # 426.33 / 5613.76
        ("CMB accounts for about 27% of the combined net profit of CMB, ICBC and Ping An Bank", "supported", 0.2675),
    ],
)
def test_a_share_of_a_total_of_several_companies(checker, claim, verdict, share):
    report = _verdict(checker, claim, verdict)
    (check,) = report.checks
    assert check.kind == "ratio" and "+" in str(check.reference), _rows(report)
    assert check.ratio == pytest.approx(share, abs=1e-3), _rows(report)


# ---- an English year before the amount ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim", "metrics"),
    [
        ("Midea Group's 2025 revenue of RMB458.5 billion rose 12.1% from a year earlier", ["revenue", "revenue_yoy"]),
        ("CATL's 2025 net profit of RMB72.2 billion", ["net_profit"]),
    ],
)
def test_a_year_written_before_the_amount_is_not_a_value(checker, claim, metrics):
    report = _verdict(checker, claim, "supported")
    assert [check.metric for check in report.checks] == metrics, _rows(report)
    assert all(check.claimed != 2025 for check in report.checks)


# ---- a foreign line (H shares, Hong Kong, ADR, GDR) is never checked against A-share data -------------------------


@pytest.mark.parametrize(
    "claim",
    [
        "比亚迪H股的市净率约3.2倍",  # the A-share P/B 3.17 would "support" it
        "招商银行港股市盈率不到7倍",
        "ICBC's Hong Kong-listed shares have a P/B of about 0.74",
        "长江电力伦敦GDR的市盈率约19倍",
    ],
)
def test_a_claim_about_a_foreign_line_is_unverifiable(checker, claim):
    report = _verdict(checker, claim, "unverifiable")
    (check,) = report.checks
    assert check.reason == "no_data" and "A-share" in check.note, _rows(report)


def test_the_a_share_part_of_an_a_and_h_claim_is_still_checked(checker):
    report = checker("比亚迪A股市净率约3.2倍，H股约2.5倍")
    assert [(check.status, check.reason) for check in report.checks] == [
        ("supported", None),
        ("unverifiable", "no_data"),
    ], _rows(report)


# ---- 扣非 vs 归母 of the same company --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("中国平安的扣非净利润比归母净利润高", "supported"),  # 1437.73 > 1347.78
        ("比亚迪扣非净利润低于归母净利润", "supported"),  # 294.46 < 326.19
        ("宁德时代归母净利润比扣非净利润多出约77亿元", "supported"),  # 76.93
        ("美的集团扣非净利润比归母少了近27亿", "supported"),  # 26.78
        ("紫金矿业扣非净利润比归母净利润多10亿元", "contradicted"),  # 扣非 is 10.53亿 lower
        ("BYD's recurring net profit was about RMB3.2 billion below its net profit attributable to shareholders",
         "supported"),  # 31.73亿
    ],
)  # fmt: skip
def test_net_profit_after_non_recurring_items_against_attributable_net_profit(checker, claim, verdict):
    report = _verdict(checker, claim, verdict)
    (check,) = report.checks
    assert {check.metric, check.reference_metric} == {"profit_dedt", "net_profit"}, _rows(report)
    assert check.reference == check.target and check.reference_value is not None, _rows(report)


def test_two_figures_of_one_claim_bind_to_the_right_metric(checker):
    # "不足300亿" is 扣非 (294.46); "比归母少30多亿" is the gap 31.73
    report = _verdict(checker, "比亚迪扣非净利润不足300亿元，比归母净利润少30多亿", "supported")
    assert [check.kind for check in report.checks] == ["value", "difference"], _rows(report)


# ---- an implicit subject after 但 and a company's short name ----------------------------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        # 美的 439.45 / 宁德 722.01 = 0.61; "宁德" is 宁德时代's short name written after its full name
        "美的集团营收比宁德时代还多，但净利润只有宁德的六成左右",
        # the net profit is 比亚迪's (326.19), the subject of the first clause, not the compared 宁德时代 (722.01)
        "比亚迪的营收是宁德时代的1.9倍，但净利润只有约326亿元",
    ],
)
def test_the_subject_carries_over_a_contrast_clause(checker, claim):
    report = _verdict(checker, claim, "supported")
    assert all(check.target != "宁德时代" for check in report.checks), _rows(report)


# ---- a definition caution alone does not make a claim partially supported --------------------------------------------


def test_a_definition_caution_alone_leaves_the_verdict_supported(checker):
    report = checker("宁德时代2025年归母净利润722亿元，同比增长42%")
    assert report.verdict == "supported" and report.coverage == "partial", _rows(report)
    assert [check.reason for check in report.checks] == [None, "definition_mismatch"]


def test_another_unverifiable_part_still_makes_it_partial(checker):
    report = checker("宁德时代归母净利润722亿元，市销率约5倍")
    assert report.verdict == "partially_supported", _rows(report)
