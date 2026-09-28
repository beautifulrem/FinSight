"""Claim check: numbers in a pasted claim against market and fundamental evidence."""

from __future__ import annotations

from datetime import date

import pytest
from agent_fakes import FUNDAMENTALS, StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.claim_check import check_claim, normalise
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app


@pytest.mark.parametrize(
    ("claim", "verdict", "statuses"),
    [
        ("贵州茅台市盈率只有15倍，已经严重低估", "contradicted", ["contradicted"]),
        ("贵州茅台市盈率约25倍", "supported", ["supported"]),
        ("贵州茅台收盘价1409.5元，当日下跌0.18%", "supported", ["supported", "supported"]),
        ("贵州茅台昨天大涨5%", "contradicted", ["contradicted"]),
        ("贵州茅台收盘价1409.5元，市盈率40倍", "partially_supported", ["supported", "contradicted"]),
        ("今天天气不错", "unverifiable", []),
    ],
)
def test_claims_are_checked_against_evidence(claim, verdict, statuses):
    report = check_claim(claim, service=StubService(), registry=build_fake_registry())

    assert report.verdict == verdict
    assert [check.status for check in report.checks] == statuses
    for check in report.checks:
        if check.status != "unverifiable":
            assert check.evidence_id and check.actual is not None


def test_contradiction_reports_the_actual_value_and_source():
    report = check_claim("贵州茅台市盈率只有15倍", service=StubService(), registry=build_fake_registry())

    check = report.checks[0]
    assert (check.metric, check.claimed, check.actual) == ("pe_ttm", 15.0, 24.6)
    assert check.evidence_id == "fundamental_600519.SH"
    assert "不构成投资建议" in report.disclaimer


def test_claim_check_endpoint(monkeypatch):
    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime))

    body = TestClient(app).post("/agent/claim-check", json={"claim": "贵州茅台市盈率只有15倍"}).json()

    assert body["verdict"] == "contradicted" and body["checks"][0]["actual"] == 24.6


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        # the metric word after a comma belongs to the next clause
        ("茅台市盈率只有15倍，股价昨天跌了5%", [("pe_ttm", "contradicted"), ("pct_change_1d", "contradicted")]),
        # English metric further than 12 characters before the number
        ("Moutai P/E ratio is currently about 25", [("pe_ttm", "supported")]),
        # "约" in the first clause does not widen the tolerance of the second
        ("茅台市盈率约24.6倍，营收1700亿", [("pe_ttm", "supported"), ("revenue", "contradicted")]),
        # a daily move is never compared x100 (17.78% is not -0.1778%)
        ("茅台昨天跌了17.78%", [("pct_change_1d", "contradicted")]),
    ],
)
def test_metric_binding_stays_within_the_numbers_clause(claim, expected):
    report = check_claim(claim, service=StubService(), registry=build_fake_registry())

    assert [(check.metric, check.status) for check in report.checks] == expected


def test_claim_reports_the_claimed_unit_and_source_provenance():
    report = check_claim("茅台营收1741亿", service=StubService(), registry=build_fake_registry())

    assert report.checks[0].claimed_unit == "亿"
    assert report.checks[0].status == "supported"
    assert all("provenance" in source for source in report.evidence_sources)


def test_claim_check_endpoint_validates_blank_claims_and_uses_the_requested_language(monkeypatch):
    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime))
    client = TestClient(app)

    assert client.post("/agent/claim-check", json={"claim": "     "}).status_code == 422
    body = client.post("/agent/claim-check", json={"claim": "贵州茅台市盈率只有15倍", "language": "en"}).json()
    assert "not investment advice" in body["disclaimer"]


def _check(claim: str):
    return check_claim(claim, service=StubService(), registry=build_fake_registry())


def test_acronyms_right_after_chinese_text_are_recognised():
    # B3: "\bPE\b" has no word boundary between 台 and P.
    report = _check("茅台PE为24.6倍，PB8.1倍")

    assert [(check.metric, check.status) for check in report.checks] == [("pe_ttm", "supported"), ("pb", "supported")]


@pytest.mark.parametrize(
    ("claim", "comparator", "status"),
    [
        ("贵州茅台ROE超过30%", "gt", "supported"),  # actual 33%: B4
        ("茅台ROE高于40%", "gt", "contradicted"),
        ("茅台ROE低于30%", "lt", "contradicted"),
        ("茅台市盈率不到30倍", "lt", "supported"),
        ("茅台市盈率不足20倍", "lt", "contradicted"),
        ("茅台PB 8倍以上", "ge", "supported"),
        ("茅台市盈率至少25倍", "ge", "contradicted"),  # bounds are literal
        ("茅台市盈率不超过25倍", "le", "supported"),
        ("茅台市盈率约25倍", "approx", "supported"),
        ("茅台市盈率25倍左右", "approx", "supported"),
        ("茅台市盈率30多倍", "gt", "contradicted"),
        ("Moutai's ROE is above 30%", "gt", "supported"),
        ("Moutai's P/E is less than 20", "lt", "contradicted"),
        ("Moutai's P/E is at least 20x", "ge", "supported"),
    ],
)
def test_comparators_are_evaluated_as_bounds(claim, comparator, status):
    (check,) = _check(claim).checks

    assert (check.comparator, check.status) == (comparator, status)


@pytest.mark.parametrize(
    ("claim", "low", "high", "status"),
    [
        ("茅台市盈率在20到30倍之间", 20.0, 30.0, "supported"),
        ("茅台市盈率20-30倍", 20.0, 30.0, "supported"),
        ("Moutai's P/E is between 25 and 30", 25.0, 30.0, "contradicted"),
    ],
)
def test_ranges_are_one_check(claim, low, high, status):
    (check,) = _check(claim).checks

    assert (check.comparator, check.claimed, check.claimed_high, check.status) == ("range", low, high, status)


@pytest.mark.parametrize(
    ("claim", "verdict", "checks"),
    [
        # B18: the negation applies to 15, not to 24.6
        ("茅台市盈率不是15倍而是24.6倍", "supported", [("ne", "supported"), ("eq", "supported")]),
        ("茅台市盈率不是24.6倍", "contradicted", [("ne", "contradicted")]),
        ("Moutai's P/E is not 15x", "supported", [("ne", "supported")]),
        ("茅台ROE没有超过30%", "contradicted", [("le", "contradicted")]),
        # number-less moves are checked against 0; the daily change was -0.18%
        ("茅台昨天并没有跌", "contradicted", [("ge", "contradicted")]),
        ("茅台昨天下跌了", "supported", [("lt", "supported")]),
        ("Moutai did not fall yesterday", "contradicted", [("ge", "contradicted")]),
    ],
)
def test_negation_flips_the_comparator(claim, verdict, checks):
    report = _check(claim)

    assert report.verdict == verdict
    assert [(check.comparator, check.status) for check in report.checks] == checks
    assert all(check.negated for check in report.checks if check.comparator == "ne")


def test_chinese_numerals_are_read():
    assert normalise("市盈率十五倍，市净率一点一倍，ROE超过三成，增速百分之二十四点六") == (
        "市盈率15倍，市净率1.1倍，ROE超过30%，增速24.6%"
    )
    assert [(check.claimed, check.status) for check in _check("茅台市盈率只有十五倍").checks] == [
        (15.0, "contradicted")
    ]
    assert _check("贵州茅台ROE超过三成").checks[0].status == "supported"


def test_growth_claims_use_yoy_fields_or_say_they_are_missing(monkeypatch):
    (check,) = _check("贵州茅台营收同比增长16%").checks
    assert (check.metric, check.status, check.reason) == ("revenue_yoy", "unverifiable", "growth_unavailable")

    fundamentals = {**FUNDAMENTALS["600519.SH"], "revenue_yoy": 16.2, "netprofit_yoy": -2.03}
    monkeypatch.setitem(FUNDAMENTALS, "600519.SH", fundamentals)
    report = _check("茅台营收同比增长16%，净利润同比下降2%")
    assert [(check.metric, check.claimed, check.status) for check in report.checks] == [
        ("revenue_yoy", 16.0, "supported"),
        ("netprofit_yoy", -2.0, "supported"),
    ]
    # "净利率" is a margin, not net-profit growth
    assert _check("茅台净利率48.8%").checks[0].metric == "net_margin"


def test_pe_as_of_is_the_valuation_date_when_the_source_gives_one(monkeypatch):
    # B26: P/E(TTM) is priced on a trade date, not on the report date.
    report = _check("茅台市盈率24.6倍")
    assert report.checks[0].as_of_basis == "report_date"

    fundamentals = {**FUNDAMENTALS["600519.SH"], "report_date": "2026-06-30", "valuation_date": "2026-09-24"}
    monkeypatch.setitem(FUNDAMENTALS, "600519.SH", fundamentals)
    (check,) = _check("茅台市盈率24.6倍").checks
    assert (check.as_of, check.as_of_basis) == ("2026-09-24", "valuation_date")


@pytest.mark.parametrize(
    ("claim", "reason"),
    [
        ("茅台市盈率24.6%", "unit_mismatch"),
        ("茅台ROE 33倍", "unit_mismatch"),
        ("茅台股价1409.5亿元", "unit_mismatch"),
        ("茅台营收1741", "no_unit"),
        ("茅台明年股价会涨到2000元", "forecast"),
        ("茅台今年以来涨了20%", "multi_day"),
        ("茅台股息率5%", "no_data"),
    ],
)
def test_checks_that_cannot_be_compared_say_why(claim, reason):
    (check,) = _check(claim).checks

    assert (check.status, check.reason) == ("unverifiable", reason)


def test_period_mismatch_and_interim_reports(monkeypatch):
    fundamentals = {**FUNDAMENTALS["600519.SH"], "report_date": "2025-12-31"}
    monkeypatch.setitem(FUNDAMENTALS, "600519.SH", fundamentals)
    assert _check("茅台2019年营收854亿").checks[0].reason == "period_mismatch"
    assert _check("茅台2025年营收1741亿").checks[0].status == "supported"

    monkeypatch.setitem(FUNDAMENTALS, "600519.SH", {**fundamentals, "report_date": "2026-06-30"})
    # an annual-looking amount is not compared with a half-year total
    assert _check("茅台营收1741亿").checks[0].reason == "period_mismatch"


def test_each_number_is_bound_to_the_target_named_before_it():
    report = _check("茅台PE 24.6倍，五粮液PE 15.2倍")
    assert [(check.target, check.status) for check in report.checks] == [
        ("贵州茅台", "supported"),
        ("五粮液", "supported"),
    ]
    # 24.6 is Moutai's P/E, not Wuliangye's
    assert _check("五粮液市盈率24.6倍，比茅台低").verdict == "contradicted"
    # parallel clause inherits the metric; "分别" assigns targets in order
    assert _check("茅台市盈率24.6倍，五粮液15.2倍").verdict == "supported"
    assert _check("茅台和五粮液的市盈率分别为24.6倍和15.2倍").verdict == "supported"


def test_index_names_and_tickers_are_not_claimed_numbers():
    report = _check("贵州茅台(600519.SH)收盘价1409.5元，600519市盈率24.6倍")

    assert [check.claimed for check in report.checks] == [1409.5, 24.6]


@pytest.mark.parametrize(
    ("claim", "claimed", "status"),
    [
        ("茅台昨日收跌0.18%", -0.18, "supported"),  # found by the held-out run (h011): 收跌 was read as up
        ("茅台昨天收涨0.18%", 0.18, "contradicted"),
        ("茅台昨天跌幅0.18%", -0.18, "supported"),
        ("Moutai fell 0.18% yesterday", -0.18, "supported"),
    ],
)
def test_the_move_word_before_a_number_sets_its_sign(claim, claimed, status):
    (check,) = _check(claim).checks

    assert (check.claimed, check.status) == (claimed, status)
