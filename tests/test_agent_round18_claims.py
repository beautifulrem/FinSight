"""Round 18 (review 14, N10 and N13): claim-check without the corrections layer, and four claim shapes.

* N10: with the corrections layer switched on but its file missing, a check decided on an uncorrected offline-snapshot
  (seed) value is marked ``degraded`` and says so in its note, and the report's coverage is never "full": before round
  18 the report stated the missing layer but kept a plain "supported" with full coverage for a seed value the healthy
  deployment contradicts.
* N13a: an exclusion ("除A外", "A除外", "不算A") leaves A out of the group the shared statement is about: the other
  companies are checked, A is not (before: only A was checked, a false contradiction).
* N13b: "A和B之和不足C" / "加起来还不到C": a sum compared with one company, like 不及 / 低于.
* N13c: a bare unit as a threshold ("突破万亿", "站上千亿", "超万亿"): 1万亿 / 1000亿.
* N13d: "降幅较营收收窄 / 扩大" compares the year-on-year rates of two figures of the company (before: read as a
  one-day price move against the previous session).

The wordings and company pairings are the engineer's own. Values: shipped default data (FY2025 report), see the table
in tests/test_agent_round17_claims.py; revenue 亿: 中国平安 10505.06, 比亚迪 8039.65, 贵州茅台 1688.38, 中芯国际
673.23; market cap 亿: 宁德时代 13470.39.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def checker(shipped_offline_service):
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    service = shipped_offline_service
    registry = build_registry_for_service(service)
    return lambda claim: check_claim(claim, service=service, registry=registry, zh=not claim.isascii())


def _checked(report) -> list[tuple[str | None, str | None, str]]:
    return [(check.target, check.metric, check.status) for check in report.checks]


# ---- N13a: exclusions ------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "verdict", "checked", "excluded"),
    [
        (
            "工行、招行、平安银行三家之中，除去平安银行之外，归母净利润都是正增长",
            "supported",
            {"工商银行", "招商银行"},
            ["平安银行"],
        ),
        (
            "宁德时代、比亚迪、美的集团里，除比亚迪以外净利润均同比上升",
            "supported",
            {"宁德时代", "美的集团"},
            ["比亚迪"],
        ),
        (
            "贵州茅台、五粮液、长江电力中，除长江电力外营收都在增长",
            "contradicted",
            {"贵州茅台", "五粮液"},
            ["长江电力"],
        ),
        ("比亚迪除外，宁德时代和美的集团的净利润同比均增长", "supported", {"宁德时代", "美的集团"}, ["比亚迪"]),
        (
            "中信证券、紫金矿业、五粮液三家，不算五粮液，净利润都同比增长了",
            "supported",
            {"中信证券", "紫金矿业"},
            ["五粮液"],
        ),
    ],
)
def test_an_excluded_company_is_left_out_of_the_group(checker, claim, verdict, checked, excluded):
    report = checker(claim)
    assert report.verdict == verdict, report.model_dump()
    assert {check.target for check in report.checks} == checked, _checked(report)
    assert report.excluded == excluded


def test_besides_is_not_an_exclusion(checker):
    # 除了…也: "besides", the named company is not left out of anything
    report = checker("除了贵州茅台外，五粮液的营收也下滑了")
    assert report.excluded == []
    assert ("五粮液", "revenue_yoy", "supported") in _checked(report), _checked(report)


def test_an_exclusion_of_something_that_is_not_a_company_is_left_alone(checker):
    report = checker("除房地产业务外，美的集团营收同比增长约12%")
    assert report.excluded == []
    assert _checked(report) == [("美的集团", "revenue_yoy", "supported")]


# ---- N13b: a sum against one company ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("美的集团与比亚迪的净利润合计还不到宁德时代一家", "contradicted"),  # 765.64 > 722.01
        ("比亚迪和中芯国际净利润加起来不足宁德时代", "supported"),  # 376.60 < 722.01
        ("招行与工行净利润之和不到中国平安一家", "contradicted"),  # 5187.43 > 1347.78
    ],
)
def test_a_sum_below_one_company(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, report.model_dump()
    assert [check.kind for check in report.checks] == ["relation"] and "+" in str(report.checks[0].target)


@pytest.mark.parametrize("claim", ["比亚迪市盈率不到30倍", "工商银行市净率不足1倍"])
def test_a_bound_with_a_number_is_still_a_bound(checker, claim):
    report = checker(claim)
    assert report.verdict == "supported" and report.checks[0].kind == "value", report.model_dump()


# ---- N13c: a bare unit as the threshold ------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "verdict", "metric"),
    [
        ("中国平安的营收已经跨过万亿门槛", "supported", "revenue"),
        ("比亚迪营业收入站上万亿", "contradicted", "revenue"),
        ("贵州茅台营收破千亿", "supported", "revenue"),
        ("中芯国际营收超过千亿", "contradicted", "revenue"),
        ("宁德时代总市值突破万亿", "supported", "market_cap"),
    ],
)
def test_a_bare_unit_threshold(checker, claim, verdict, metric):
    report = checker(claim)
    assert report.verdict == verdict and [check.metric for check in report.checks] == [metric], report.model_dump()
    assert report.checks[0].comparator in {"gt", "ge"}


# ---- N13d: a decline narrower or wider than another figure's -----------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("平安银行归母净利润降幅较营业收入明显收窄", "supported"),  # -4.21 vs -10.40
        ("五粮液净利润同比降幅较营收进一步扩大", "supported"),  # -71.89 vs -54.55
        ("贵州茅台净利润降幅较营收收窄", "contradicted"),  # -4.53 vs -1.21
        ("紫金矿业净利润增幅较营收明显扩大", "supported"),  # 62.01 vs 14.96
    ],
)
def test_a_decline_compared_with_revenue_is_a_growth_comparison(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, report.model_dump()
    assert all(check.metric != "pct_change_1d" for check in report.checks), _checked(report)
    assert any(check.reference_metric == "revenue_yoy" for check in report.checks), _checked(report)


def test_a_narrowing_decline_after_the_figures(checker):
    report = checker("平安银行净利润426.33亿元，同比下降4.21%，降幅较营收有所收窄")
    assert report.verdict == "supported" and report.coverage == "full", report.model_dump()


# ---- N10: the corrections layer switched on but missing --------------------------------------------------------
@pytest.fixture
def seed_checker(monkeypatch, tmp_path):
    from evaluation.agent_eval.runner import build_offline_service
    from query_intelligence import data_loader
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    monkeypatch.setattr(data_loader, "SNAPSHOT_FIX_PATH", tmp_path / "structured_data_fix.json")
    data_loader.clear_data_caches()
    service = build_offline_service(snapshot_ext=True, snapshot_fix=True)
    registry = build_registry_for_service(service)
    yield lambda claim: check_claim(claim, service=service, registry=registry, zh=not claim.isascii())
    monkeypatch.undo()
    data_loader.clear_data_caches()


@pytest.mark.parametrize(
    "claim",
    [
        "贵州茅台的PE在24.6倍上下",  # the v1 seed value; the corrected one is 21.44
        "Kweichow Moutai trades at roughly 24.6 times earnings",
    ],
)
def test_a_seed_value_without_the_corrections_layer_is_never_full_coverage(seed_checker, claim):
    report = seed_checker(claim)
    assert report.degraded == ["data_layer_missing:corrections"]
    assert report.coverage == "partial", report.model_dump()
    assert report.checks and all(check.degraded == ["data_layer_missing:corrections"] for check in report.checks)
    note = report.checks[0].note
    assert ("未经更正" in note) if not claim.isascii() else ("uncorrected" in note), note


def test_a_contradiction_on_a_seed_value_is_marked_too(seed_checker):
    report = seed_checker("贵州茅台市盈率不到22倍")  # true on corrected data, false on the seed
    assert report.coverage == "partial" and report.checks[0].degraded, report.model_dump()


def test_checks_with_the_layer_present_are_not_degraded(checker):
    report = checker("贵州茅台的PE在21.4倍上下")
    assert report.coverage == "full" and all(check.degraded == [] for check in report.checks), report.model_dump()


def test_a_definition_without_the_layer_is_not_degraded(seed_checker):
    report = seed_checker("市净率等于股价除以每股净资产")
    assert report.degraded == [] and all(check.degraded == [] for check in report.checks)
