"""Round-12 rule H6 (round-8 review), written with the engineer's own wording: the claim checker reads a sum of
two companies (合计 / 加起来 / combined), 破 as a lower bound and 一半不到 / 七成都不到 as an upper bound, a stated
industry average introduced by 比起 / 对照 / 相较于 / 和…相比, a stated sector average compared with a company
value the claim states ("…, below Moutai's 24.6x"), English "beats / trails … by N", 高出一倍多 / 将近一倍 (one
fold is 100% of the compared value) and an anaphora to a stated value ("低于这一水平", "below that").
"""

from __future__ import annotations

import pytest

# ---- H6: claim checker ----


@pytest.fixture(scope="module")
def checker(offline_service):
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    registry = build_registry_for_service(offline_service)

    def run(claim: str, *, zh: bool | None = None):
        language = zh if zh is not None else not claim.isascii()
        return check_claim(claim, service=offline_service, registry=registry, zh=language)

    return run


@pytest.mark.parametrize(
    ("claim", "verdict", "total"),
    [
        # snapshot: turnover 茅台 37.94亿 + 五粮液 14.53亿 = 52.47亿; net profit 823.2亿 + 378亿 = 1201.2亿
        ("贵州茅台跟五粮液两家的成交额加在一起还不到50亿元", "contradicted", 52.47e8),
        ("五粮液和茅台的成交额合计超过50亿", "supported", 52.47e8),
        ("两家公司茅台与五粮液去年净利润总共约1200亿元", "supported", 1201.2e8),
        ("茅台、五粮液净利润之和不到1000亿", "contradicted", 1201.2e8),
        ("Moutai and Wuliangye together had turnover of more than 5 billion yuan", "supported", 52.47e8),
        ("The combined net profit of Moutai and Wuliangye is about 150 billion yuan", "contradicted", 1201.2e8),
    ],
)
def test_a_sum_of_two_companies_is_checked_against_the_sum(checker, claim, verdict, total):
    report = checker(claim)
    assert report.verdict == verdict
    (check,) = report.checks
    assert check.kind == "sum" and check.actual == pytest.approx(total, rel=1e-3)
    assert set(check.operands) == {"贵州茅台", "五粮液"}


@pytest.mark.parametrize(
    ("claim", "comparator", "verdict"),
    [
        ("五粮液全年营收已经破千亿", "ge", "supported"),
        ("中国平安股价破百元", "ge", "contradicted"),
        ("茅台去年营业收入站上1600亿", "gt", "supported"),
        ("茅台的PB只有五粮液的一半不到", "lt", "contradicted"),  # 8.1 / 5.4 = 1.5
        ("中国平安的市净率连五粮液的三成都不到", "lt", "supported"),  # 1.1 / 5.4 = 0.20
        ("五粮液营收只有茅台的七成都不到", "lt", "supported"),  # 1085 / 1688.38 = 0.64
    ],
)
def test_bounds_written_as_po_or_after_a_share(checker, claim, comparator, verdict):
    report = checker(claim)
    assert report.checks[0].comparator == comparator
    assert report.verdict == verdict


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        # a stated industry average introduced by a comparison frame, the company's own value, and the relation
        ("相较于保险行业约11.8倍的平均PE，中国平安8.7倍的市盈率偏低", "supported"),
        ("对照白酒业6.2倍的平均市净率，五粮液5.4倍的PB算是偏高", "partially_supported"),
        ("和保险业1.45倍的平均市净率相比，平安1.1倍的PB更低", "supported"),
        ("与保险业20倍的平均市盈率相比，平安8.7倍的市盈率偏低", "partially_supported"),
    ],
)
def test_a_stated_average_in_a_comparison_frame_is_the_industry_average(checker, claim, verdict):
    report = checker(claim)
    rows = [(check.kind, check.target, check.status) for check in report.checks]
    assert report.verdict == verdict, rows
    stated = [check for check in report.checks if check.kind in {"stated_reference", "value"} and check.actual]
    assert any("行业平均" in str(check.target) for check in stated), rows
    relations = [check for check in report.checks if check.kind == "relation"]
    assert len(relations) == 1 and relations[0].reference and "行业" in relations[0].reference, rows


def test_a_relation_to_a_stated_average_by_anaphora(checker):
    report = checker("保险业的平均市盈率在11.8倍左右，中国平安的PE低于该水平")
    assert report.verdict == "supported" and not report.unchecked
    relation = report.checks[-1]
    assert relation.kind == "relation" and relation.metric == "pe_ttm" and relation.comparator == "lt"
    assert relation.target == "中国平安" and relation.reference_value == pytest.approx(11.8)

    english = checker("The insurance sector's average P/B is about 1.45x, and Ping An trades below that.")
    assert english.verdict == "supported" and english.checks[-1].kind == "relation"

    higher = checker("白酒板块平均市净率6.2倍，茅台的PB比它高")
    assert higher.verdict == "supported" and higher.checks[-1].comparator == "gt"


@pytest.mark.parametrize(
    ("claim", "verdict", "difference"),
    [
        ("Moutai's return on equity beats Wuliangye's by roughly 3.6 points", "supported", 3.6),
        ("Wuliangye's ROE trails Moutai's by about 3.6 percentage points", "supported", -3.6),
        ("Ping An's P/E lags Wuliangye's by more than 20", "contradicted", -12.2),
        ("Moutai's revenue tops Wuliangye's by around 60 billion yuan", "supported", 603.38e8),
        ("Ping An's net profit is roughly 40 billion yuan above Wuliangye's", "contradicted", 832e8),
    ],
)
def test_english_differences_by_n(checker, claim, verdict, difference):
    report = checker(claim)
    (check,) = report.checks
    assert check.kind == "difference" and check.difference == pytest.approx(difference, rel=1e-3)
    assert report.verdict == verdict


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("茅台的净利润比五粮液高出一倍多", "supported"),  # 823.2 / 378 - 1 = 117.8%
        ("茅台营收比五粮液高出一倍多", "contradicted"),  # 1688.38 / 1085 - 1 = 55.6%
        ("茅台成交额比五粮液多了将近一倍", "contradicted"),  # 37.94 / 14.53 - 1 = 161%
        ("茅台的净利润比五粮液多了一倍左右", "supported"),
        ("五粮液PE比茅台低了将近一半", "contradicted"),  # -15.0%
    ],
)
def test_one_fold_differences_are_relative(checker, claim, verdict):
    report = checker(claim)
    (check,) = report.checks
    assert check.kind == "relative_difference" and check.status != "unverifiable"
    assert report.verdict == verdict


@pytest.mark.parametrize(
    ("claim", "verdict", "statuses"),
    [
        # the sector's stated average, the relation (sector vs company) and the company's stated value
        (
            "The baijiu industry's average P/E is roughly 20x, below Moutai's 24.6x",
            "partially_supported",
            {"value": "contradicted", "relation": "contradicted", "stated_reference": "supported"},
        ),
        (
            "白酒行业平均市盈率约30倍，高于茅台的24.6倍",
            "supported",
            {"value": "supported", "relation": "supported", "stated_reference": "supported"},
        ),
    ],
)
def test_a_sector_average_compared_with_a_company_value_it_states(checker, claim, verdict, statuses):
    report = checker(claim)
    assert report.verdict == verdict
    assert {check.kind: check.status for check in report.checks} == statuses
    relation = next(check for check in report.checks if check.kind == "relation")
    assert relation.reference == "贵州茅台" and relation.reference_value == pytest.approx(24.6)


def test_a_bracketed_stated_average_after_a_bound(checker):
    report = checker("中国平安的市净率低于保险行业均值（约1.45倍）")
    assert report.verdict == "supported"
    assert sorted(check.kind for check in report.checks) == ["relation", "stated_reference"]
