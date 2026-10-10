"""Round-13 rules I8 and I19 (round-9 review), written with the engineer's own wording and other pairings than the
review's repros. Values come from the offline snapshot extension (data/snapshot/structured_data_ext.json, valuation
at the 2026-09-30 close) and the v1 snapshot:

* net profit (亿): 工商银行 3685.62, 招商银行 1501.81, 宁德时代 722.01, 平安银行 426.33, 比亚迪 326.19, 中芯国际 50.41
* revenue (亿): 比亚迪 8039.65, 宁德时代 4237.02
* P/E: 比亚迪 25.81, 宁德时代 15.85, 招商银行 6.86, 平安银行 5.17; P/B: 招商银行 0.91, 工商银行 0.74, 平安银行 0.48
* ROE (%): 美的集团 19.69, 比亚迪 13.24, 招商银行 11.8, 工商银行 8.68, 平安银行 7.73
* 宁德时代: close 291.11 (2026-09-30), 52-week closing high 460.0 (2026-05-06), low 286.8 (2026-09-29)
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def checker():
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools.defaults import build_registry_for_service
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
    )
    registry = build_registry_for_service(service)

    def run(claim: str):
        return check_claim(claim, service=service, registry=registry, zh=not claim.isascii())

    return run


def _rows(report):
    return [
        (check.kind, check.target, check.metric, check.comparator, check.claimed, check.status)
        for check in report.checks
    ]


# ---- X倍半 is X.5 times ----


@pytest.mark.parametrize(
    ("claim", "verdict", "claimed"),
    [
        ("工商银行的净利润大约是招商银行的两倍半", "supported", 2.5),  # 2.454
        ("宁德时代净利润差不多是平安银行的一倍半", "contradicted", 1.5),  # 1.694
        ("招商银行的净利润约为平安银行的三倍半", "supported", 3.5),  # 3.523
    ],
)
def test_x_and_a_half_times_is_read_as_x_point_five(checker, claim, verdict, claimed):
    report = checker(claim)
    (check,) = report.checks
    assert check.kind == "ratio" and check.claimed == pytest.approx(claimed), _rows(report)
    assert report.verdict == verdict, _rows(report)


# ---- 接近/将近一倍 is (80%, 100%] more; 一倍多 is [100%, 200%) more ----


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("比亚迪的营收比宁德时代高出接近一倍", "supported"),  # +89.75%
        ("招商银行市净率不足1倍，较平安银行高了将近一倍", "supported"),  # 0.91 < 1; +89.58%
        ("宁德时代的净利润比比亚迪多了将近一倍", "contradicted"),  # +121.35%: more than one fold
        ("宁德时代的净利润比比亚迪多了一倍多", "supported"),  # +121.35%
        ("工商银行的净利润比招商银行高出一倍多", "supported"),  # +145.41%
        ("招商银行的净利润比平安银行多出一倍多", "contradicted"),  # +252.26%: more than two folds
    ],
)
def test_near_one_fold_and_one_fold_more(checker, claim, verdict):
    report = checker(claim)
    relative = [check for check in report.checks if check.kind == "relative_difference"]
    assert len(relative) == 1, _rows(report)
    assert report.verdict == verdict, _rows(report)


# ---- 多一点 / 高一点 is not a number ----


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("平安银行的市盈率只有招商银行的七成多一点", "supported"),  # 0.754
        ("招商银行的ROE比工商银行高一点", "supported"),
        ("比亚迪的市净率比宁德时代低一点点", "supported"),  # 3.17 < 3.55
    ],
)
def test_a_little_more_is_not_a_stated_value(checker, claim, verdict):
    report = checker(claim)
    assert all(check.claimed != 1.0 or check.kind != "value" for check in report.checks), _rows(report)
    assert all(check.status != "unverifiable" for check in report.checks), _rows(report)
    assert report.verdict == verdict, _rows(report)


# ---- 52周 / 52-week is a window, never a value; the 52-week high is a reference ----


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("宁德时代的股价比52周最高收盘价低了三成多", "supported"),  # -36.71%
        ("宁德时代距52周高点已经跌去超过三成", "supported"),
        ("宁德时代股价较52周高点已经腰斩", "contradicted"),  # -36.71%, not about -50%
        ("CATL trades roughly 37% below its 52-week high", "supported"),
        ("宁德时代比52周低点高出不到2%", "supported"),  # +1.50%
    ],
)
def test_the_52_week_high_is_a_reference_not_the_number_52(checker, claim, verdict):
    report = checker(claim)
    assert all(check.claimed not in {52.0, -52.0} for check in report.checks), _rows(report)
    (check,) = report.checks
    assert check.kind == "relative_difference" and check.target == "宁德时代", _rows(report)
    assert check.reference_value in {460.0, 286.8}, _rows(report)
    assert report.verdict == verdict, _rows(report)


def test_a_52_week_claim_without_the_history_is_unverifiable_not_contradicted(checker):
    report = checker("贵州茅台比52周高点低了一成")  # the v1 history holds a few closes
    assert report.verdict == "unverifiable", _rows(report)
    assert all(check.claimed not in {52.0, -52.0} for check in report.checks)


# ---- English "N turns / points above" is a difference ----


@pytest.mark.parametrize(
    ("claim", "verdict", "difference"),
    [
        ("BYD is valued at about 26 times earnings, roughly 10 turns above CATL", "supported", 9.96),
        ("CATL's P/E sits about 10 turns lower than BYD's", "supported", -9.96),
        ("BYD's P/E is some 20 turns higher than CATL's", "contradicted", 9.96),
        ("Midea's ROE is about 6.5 points above BYD's", "supported", 6.45),
    ],
)
def test_turns_and_points_above_are_differences(checker, claim, verdict, difference):
    report = checker(claim)
    differences = [check for check in report.checks if check.kind == "difference"]
    assert len(differences) == 1, _rows(report)
    assert differences[0].difference == pytest.approx(difference, abs=0.01), _rows(report)
    assert not [check for check in report.checks if check.kind == "value" and check.claimed in {10.0, 20.0}]
    assert report.verdict == verdict, _rows(report)


# ---- the subject carries into a clause after 但 / 却 ----


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("茅台的市盈率低于白酒行业平均，但市净率高于行业平均", "supported"),  # 24.6 < 27.3; 8.1 > 6.2
        ("贵州茅台市盈率不如白酒行业平均高，市净率却比行业平均高", "supported"),
        ("茅台的市盈率低于白酒行业平均，但市净率也低于行业平均", "partially_supported"),
    ],
)
def test_the_subject_carries_into_a_contrast_clause(checker, claim, verdict):
    report = checker(claim)
    assert [check.target for check in report.checks] == ["贵州茅台", "贵州茅台"], _rows(report)
    assert report.verdict == verdict, _rows(report)


# ---- superlatives among the named targets ----


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("在招商银行、工商银行和平安银行三家中，平安银行的ROE最低", "supported"),
        ("招商银行、工商银行、平安银行这三家里，市净率最高的是招商银行", "supported"),
        ("招商银行、工商银行和平安银行当中工商银行的净资产收益率最高", "contradicted"),
        ("Of CMB, ICBC and Ping An Bank, ICBC has the highest ROE", "contradicted"),
        ("Among BYD, CATL and Midea, BYD has the highest P/E", "supported"),
    ],
)
def test_a_superlative_among_named_targets(checker, claim, verdict):
    report = checker(claim)
    (check,) = report.checks
    assert check.kind == "relation" and check.status != "unverifiable", _rows(report)
    assert report.verdict == verdict, _rows(report)


def test_a_superlative_with_one_named_target_is_not_checked(checker):
    report = checker("平安银行的ROE最低")
    assert report.verdict == "unverifiable"


# ---- a third / a quarter / half of ----


@pytest.mark.parametrize(
    ("claim", "verdict", "claimed"),
    [
        ("SMIC's net profit is less than a tenth of CATL's", "supported", 0.1),  # 0.0698
        ("BYD's net profit is less than half of CATL's", "supported", 0.5),  # 0.4518
        ("BYD's net profit is less than a third of CATL's", "contradicted", 0.3333),
        ("Ping An Bank's net profit is about a quarter of ICBC's", "contradicted", 0.25),  # 0.1157
        ("CMB's net profit is roughly two-fifths of ICBC's", "supported", 0.4),  # 0.4075
    ],
)
def test_english_fractions_of_another_value_are_multiples(checker, claim, verdict, claimed):
    report = checker(claim)
    (check,) = report.checks
    assert check.kind == "ratio" and check.claimed == pytest.approx(claimed, abs=1e-3), _rows(report)
    assert report.verdict == verdict, _rows(report)


# ---- I19: a contrastive conjunction is a clause break for the output-safety marker ----


@pytest.mark.parametrize(
    ("sentence", "clauses"),
    [
        ("茅台营收1688亿元但二季度直营占比升至45%。", 2),
        ("营收增长5%，但是净利润下降3%", 2),
        ("营收增长5%却净利润下降3%", 2),
        ("营收增长5%，不过毛利率下降2个百分点", 2),
        ("revenue rose 5% but net profit fell 3%", 2),
        ("revenue rose 5%; however, margins fell 2%", 2),
        ("不但营收增长5%而且利润增长3%", 2),  # 不但 is "not only": the split is at 而且
        ("五粮液吸收合并二季度提价7.5%", 1),
    ],
)
def test_contrastive_conjunctions_split_clauses(sentence, clauses):
    from query_intelligence.agent.output_safety import _clause_spans

    spans = _clause_spans(sentence)
    assert len(spans) == clauses, [sentence[a:b] for a, b in spans]
    assert all(end > start for start, end in spans)


# ---- numerals before 个百分点, a sector as a ratio reference, an English sector average ----


@pytest.mark.parametrize(
    ("claim", "verdict", "kind"),
    [
        ("招商银行与工商银行的ROE相差三个多百分点", "supported", "difference"),  # 3.12
        ("招商银行与工商银行的ROE相差两个百分点", "contradicted", "difference"),
        ("工商银行比平安银行的净资产收益率高约一个百分点", "supported", "difference"),  # 0.95
        ("保险行业平均市盈率11.8倍，不到白酒行业的一半", "supported", "ratio"),  # 11.8 / 27.3 = 0.43
        ("白酒板块平均PE是保险行业的三倍多", "contradicted", "ratio"),  # 2.31
        ("Wuliangye's P/E of 20.9x is about 23% below the baijiu average", "supported", "relative_difference"),
    ],
)
def test_point_numerals_sector_ratios_and_sector_averages(checker, claim, verdict, kind):
    report = checker(claim)
    assert any(check.kind == kind and check.status != "unverifiable" for check in report.checks), _rows(report)
    assert report.verdict == verdict, _rows(report)
