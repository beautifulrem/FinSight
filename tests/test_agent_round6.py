"""Round-6 rules, written after the independent round-4 held-out slices were run once (evaluation/heldout_r4).

Each test uses new wording for a failure class of those slices; none repeats a held-out claim or question.
Claim values come from the offline snapshot: 贵州茅台 -0.1778% / 1409.5, 五粮液 -0.5337% / 100.64, 中国平安
+0.73% / 53.61 (2026-04-22); FY2025 P/B 8.1 / 5.4 / 1.1; industry 白酒 -1.05% (2026-04-21), 保险 +0.68%.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.claim_check import check_claim, normalise
from query_intelligence.agent.tools.defaults import build_registry_for_service


@pytest.fixture(scope="module")
def registry(offline_service):
    return build_registry_for_service(offline_service)


def _check(claim: str, offline_service, registry):
    return check_claim(claim, service=offline_service, registry=registry, zh=not claim.isascii())


def _summary(report) -> list[tuple]:
    return [(check.target, check.metric, check.comparator, check.status) for check in report.checks]


# --- class 1: bounded moves ("跌了不到X%", "fell less than half a percent") -------------------------------------
@pytest.mark.parametrize(
    ("claim", "status"),
    [
        ("五粮液上一个交易日跌了不到0.8%", "supported"),  # a fall of 0.53
        ("茅台上一个交易日跌幅不足0.1%", "contradicted"),  # a fall of 0.18
        ("中国平安上一个交易日跌了不到2%", "contradicted"),  # it rose: not a small fall
        ("Wuliangye lost less than half a percent in the latest session", "contradicted"),
        ("Moutai rose by less than two percent in the latest session", "contradicted"),  # it fell
        ("Ping An rose by less than two percent in the latest session", "supported"),
    ],
)
def test_bounded_moves_are_checked_not_unverifiable(claim, status, offline_service, registry):
    report = _check(claim, offline_service, registry)

    assert [check.status for check in report.checks] == [status]
    assert report.checks[0].metric == "pct_change_1d" and report.checks[0].comparator == "lt"


def test_english_fractions_and_number_words_become_numbers():
    assert "0.5 percent" in normalise("fell less than half a percent")
    assert "0.25 percent" in normalise("slipped by a quarter of a percent")
    assert "3 times" in normalise("three times Wuliangye's")
    assert "2 times" in normalise("twice Moutai's")
    assert "twice a year" in normalise("twice a year")


# --- class 2: qualitative move words (a documented convention) --------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "status", "comparator", "threshold"),
    [
        ("五粮液上一个交易日重挫", "contradicted", "ge", "3%"),
        ("中国平安上一个交易日小幅上涨", "supported", "lt", "1%"),
        ("茅台上一个交易日微跌", "supported", "lt", "1%"),
        ("Ping An plunged in the latest session", "contradicted", "ge", "3%"),
        ("五粮液上一个交易日没有暴跌", "supported", "lt", "3%"),  # negated: not a fall of 3% or more
    ],
)
def test_qualitative_moves_use_a_stated_threshold(claim, status, comparator, threshold, offline_service, registry):
    check = _check(claim, offline_service, registry).checks[0]

    assert (check.metric, check.comparator, check.status) == ("pct_change_1d", comparator, status)
    assert "convention" in check.note and threshold in check.note


# --- class 3: explicit dates -------------------------------------------------------------------------------------
def test_a_date_matching_the_trade_date_is_not_a_number_or_a_downgrade(offline_service, registry):
    report = _check("Moutai closed at 1409.5 yuan on Apr 22nd, 2026", offline_service, registry)

    assert _summary(report) == [("贵州茅台", "close", "eq", "supported")]
    assert report.checks[0].as_of == "2026-04-22"


@pytest.mark.parametrize(
    "claim", ["茅台4月20日收于1409.5元", "Moutai fell on 20 April", "2025年4月22日茅台收于1409.5元"]
)
def test_a_date_other_than_the_trade_date_is_a_period_mismatch(claim, offline_service, registry):
    check = _check(claim, offline_service, registry).checks[0]

    assert (check.status, check.reason) == ("unverifiable", "period_mismatch")


def test_latest_session_wording_is_not_a_multi_day_move(offline_service, registry):
    for claim in ("五粮液最近一个交易日收跌", "五粮液近1个交易日收跌"):
        assert _summary(_check(claim, offline_service, registry)) == [("五粮液", "pct_change_1d", "lt", "supported")]
    assert _check("五粮液近5个交易日下跌", offline_service, registry).checks[0].reason == "multi_day"


def test_may_as_a_verb_is_not_a_date(offline_service, registry):
    report = _check("Moutai may trade at 30 times earnings", offline_service, registry)

    assert report.checks[0].claimed == 30.0 and report.checks[0].reason == "forecast"


# --- class 4: multiples, relations, sectors, macro thresholds ------------------------------------------------------
def test_a_multiple_of_another_target_compares_the_ratio(offline_service, registry):
    check = _check("五粮液的市净率大概是中国平安的5倍", offline_service, registry).checks[0]

    assert (check.target, check.metric, check.reference, check.comparator) == ("五粮液", "pb", "中国平安", "approx")
    assert check.claimed == 5.0 and check.ratio == pytest.approx(5.4 / 1.1, rel=1e-3)
    assert check.status == "supported" and check.reference_value == 1.1
    english = _check("Wuliangye's P/B is roughly twice Moutai's", offline_service, registry).checks[0]
    assert (english.target, english.reference, english.status) == ("五粮液", "贵州茅台", "contradicted")


def test_a_multiple_of_moves_needs_both_moves_in_the_stated_direction(offline_service, registry):
    check = _check("中国平安的跌幅是茅台的4倍", offline_service, registry).checks[0]

    assert check.status == "contradicted" and "direction" in check.note  # Ping An rose


def test_outperformance_compares_daily_moves(offline_service, registry):
    report = _check("中国平安上一个交易日跑输沪深300", offline_service, registry)

    assert _summary(report) == [("中国平安", "pct_change_1d", "lt", "contradicted")]  # +0.73 vs +0.42


def test_relational_claims_on_amounts(offline_service, registry):
    report = _check("五粮液的净利润高于中国平安", offline_service, registry)

    assert _summary(report) == [("五粮液", "net_profit", "gt", "contradicted")]


def test_company_against_a_named_sector(offline_service, registry):
    report = _check("茅台的市净率高于白酒板块", offline_service, registry)

    assert _summary(report) == [("贵州茅台", "pb", "gt", "supported")]  # 8.1 vs 6.2
    assert report.checks[0].reference == "白酒行业" and report.checks[0].reference_value == 6.2


def test_sector_moves_use_the_industry_snapshot_and_its_date(offline_service, registry):
    insurance = _check("保险板块上一个交易日收涨", offline_service, registry).checks[0]
    assert (insurance.target, insurance.status, insurance.actual) == ("保险", "supported", 0.68)
    assert insurance.as_of_basis == "trade_date"
    baijiu = _check("白酒板块4月22日跌了", offline_service, registry).checks[0]
    assert (baijiu.status, baijiu.reason) == ("unverifiable", "period_mismatch")  # the snapshot is 2026-04-21


def test_a_sector_word_describing_a_company_is_not_a_target(offline_service, registry):
    report = _check("白酒龙头茅台市净率8.1倍", offline_service, registry)

    assert _summary(report) == [("贵州茅台", "pb", "eq", "supported")]


def test_the_pmi_line_is_fifty(offline_service, registry):
    check = _check("PMI已经回到荣枯线之上", offline_service, registry).checks[0]

    assert (check.metric, check.comparator, check.claimed, check.status) == ("pmi", "gt", 50.0, "supported")


def test_is_higher_than_is_not_read_as_a_move(offline_service, registry):
    report = _check("Moutai's P/B is higher than Ping An's", offline_service, registry)

    assert _summary(report) == [("贵州茅台", "pb", "gt", "supported")]


# --- class 5: several targets sharing one claim -----------------------------------------------------------------
def test_shared_claims_get_one_check_per_target(offline_service, registry):
    report = _check("五粮液和中国平安上一个交易日都涨了", offline_service, registry)

    assert _summary(report) == [
        ("五粮液", "pct_change_1d", "gt", "contradicted"),
        ("中国平安", "pct_change_1d", "gt", "supported"),
    ]
    assert report.verdict == "partially_supported"


def test_a_list_without_a_shared_word_keeps_the_nearest_target(offline_service, registry):
    report = _check("茅台和五粮液里，五粮液市净率5.4倍", offline_service, registry)

    assert _summary(report) == [("五粮液", "pb", "eq", "supported")]
