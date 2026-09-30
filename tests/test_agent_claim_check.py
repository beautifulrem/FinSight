"""Claim check: numbers in a pasted claim against market and fundamental evidence."""

from __future__ import annotations

from datetime import date

import pytest
from agent_fakes import FUNDAMENTALS, PRICES, StubService, build_fake_registry
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
    # 24.6 is Moutai's P/E, not Wuliangye's; the relation of the second clause is its own check (round 8, D2)
    report = _check("五粮液市盈率24.6倍，比茅台低")
    assert [(check.target, check.status) for check in report.checks] == [
        ("五粮液", "contradicted"),
        ("五粮液", "supported"),
    ]
    assert report.verdict == "partially_supported"
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


def test_roe_claims_use_the_declared_percent_unit():
    """ROE is normalised to percent (0.33 -> 33.0, metric_units roe=%); a percent ROE is never scaled x100."""
    from query_intelligence.agent.evidence import AgentEvidence
    from query_intelligence.agent.tools import ToolOutput, ToolRegistry, ToolSpec

    supported = check_claim("贵州茅台ROE约33%", service=StubService(), registry=build_fake_registry())
    assert supported.checks[0].status == "supported" and supported.checks[0].actual == 33.0

    registry = build_fake_registry()
    fundamentals = registry.get("get_fundamentals")
    low_roe = ToolRegistry()
    for spec in registry.specs():
        if spec.name != "get_fundamentals":
            low_roe.register(spec)

    def handler(args):
        payload = {"roe": 0.8, "pe_ttm": 24.6, "source_name": "tushare"}  # 0.8 percent, as Tushare serves it
        item = AgentEvidence(
            evidence_id="fundamental_600519.SH", kind="structured", source_type="fundamental_sql", payload=payload
        )
        return ToolOutput(data={"metrics": payload, "source": "tushare"}, evidence=[item])

    low_roe.register(
        ToolSpec(name="get_fundamentals", description="", input_model=fundamentals.input_model, handler=handler)
    )

    report = check_claim("贵州茅台ROE高达80%", service=StubService(), registry=low_roe)

    assert report.checks[0].status == "contradicted" and report.checks[0].actual == 0.8


@pytest.mark.parametrize(
    ("claim", "comparator", "direction", "status"),
    [
        # C2: a bound after a move word is about the size of the move in that direction (fake: Moutai -0.1778%).
        ("茅台昨天跌超0.1%", "gt", "down", "supported"),
        ("茅台昨天跌了超过1%", "gt", "down", "contradicted"),
        ("茅台昨日大跌超过3%", "gt", "down", "contradicted"),
        ("茅台昨天跌了不到1%", "lt", "down", "supported"),
        ("茅台昨天跌幅不超过0.1%", "le", "down", "contradicted"),
        ("茅台昨天没有跌超过1%", "le", "down", "supported"),
        ("Moutai fell more than 0.1% yesterday", "gt", "down", "supported"),
        ("Moutai dropped more than 1% yesterday", "gt", "down", "contradicted"),
        # a move the other way contradicts the bound, even a "smaller" one
        ("茅台昨天涨超0.1%", "gt", "up", "contradicted"),
        ("茅台昨天涨了不到1%", "lt", "up", "contradicted"),
        # a range after a move word is the size of the move
        ("茅台昨天跌了0.1%到0.3%", "range", "down", "supported"),
        ("茅台昨天跌了0.5%到1%", "range", "down", "contradicted"),
    ],
)
def test_bounds_on_a_move_compare_its_size_in_the_stated_direction(claim, comparator, direction, status):
    (check,) = _check(claim).checks
    _assert_move_check(check, comparator, direction, status)


@pytest.mark.parametrize(
    ("claim", "comparator", "direction", "status"),
    [
        ("茅台昨天跌了不到2%", "lt", "down", "contradicted"),  # it rose 1.25%: not a fall at all
        ("茅台昨天涨超1%", "gt", "up", "supported"),
        ("茅台昨天大涨超过3%", "gt", "up", "contradicted"),
        ("Moutai rose less than 2% yesterday", "lt", "up", "supported"),
        ("Moutai fell more than 1% yesterday", "gt", "down", "contradicted"),
    ],
)
def test_bounds_on_a_rise(monkeypatch, claim, comparator, direction, status):
    monkeypatch.setitem(PRICES["600519.SH"], "pct_change_1d", 1.25)
    (check,) = _check(claim).checks
    _assert_move_check(check, comparator, direction, status)


def _assert_move_check(check, comparator, direction, status):

    assert (check.metric, check.comparator, check.direction, check.status) == (
        "pct_change_1d",
        comparator,
        direction,
        status,
    )
    if direction == "down":
        assert check.claimed < 0  # "跌超1%" is shown as a move below -1%


def _macro_registry():
    """The fake registry plus a macro tool serving the seed readings (March 2026)."""
    from query_intelligence.agent.evidence import AgentEvidence
    from query_intelligence.agent.tools import ToolOutput, ToolSpec
    from query_intelligence.agent.tools.macro import MacroInput

    readings = {"CPI_CN": 0.8, "PMI_CN": 50.6, "M2_CN": 8.1, "CN10Y": 2.31}

    def handler(args):
        items = [
            AgentEvidence(
                evidence_id=f"macro_{code}",
                kind="structured",
                source_type="macro_sql",
                as_of="2026-03-31",
                payload={"indicator_code": code, "metric_date": "2026-03-31", "metric_value": value},
            )
            for code, value in readings.items()
        ]
        return ToolOutput(data={"indicators": [item.payload for item in items]}, evidence=items)

    registry = build_fake_registry()
    registry.register(ToolSpec("get_macro_indicators", "Macro.", MacroInput, handler, timeout_s=2))
    return registry


@pytest.mark.parametrize(
    ("claim", "metric", "comparator", "status", "reason"),
    [
        # C13: macro series are checked against get_macro_indicators
        ("CPI同比上涨0.8%", "cpi_yoy", "eq", "supported", None),
        ("CPI同比下降0.8%", "cpi_yoy", "eq", "contradicted", None),
        ("China's CPI rose 0.8% year on year", "cpi_yoy", "eq", "supported", None),
        ("PMI重回50以上", "pmi", "ge", "supported", None),
        ("PMI跌破50", "pmi", "lt", "contradicted", None),
        ("The PMI is above 50", "pmi", "gt", "supported", None),
        ("M2同比增长10%", "m2_yoy", "eq", "contradicted", None),
        ("10年期国债收益率低于2%", "cn10y", "lt", "contradicted", None),
        ("十年期国债收益率在2.3%左右", "cn10y", "approx", "supported", None),
        ("2月CPI同比上涨0.8%", "cpi_yoy", "eq", "unverifiable", "period_mismatch"),  # the reading is March's
        ("明年CPI会上涨3%", "cpi_yoy", "eq", "unverifiable", "forecast"),
        ("PMI回落了", "pmi", "lt", "unverifiable", "no_data"),  # a change: only the latest level is served
    ],
)
def test_macro_claims(claim, metric, comparator, status, reason):
    report = check_claim(claim, service=StubService(), registry=_macro_registry())

    (check,) = report.checks
    assert (check.metric, check.comparator, check.status, check.reason) == (metric, comparator, status, reason)
    if status != "unverifiable":
        assert check.evidence_id and check.evidence_id.startswith("macro_") and check.as_of == "2026-03-31"


def test_macro_claims_without_a_macro_source_say_why():
    (check,) = _check("CPI同比上涨0.8%").checks

    assert (check.status, check.reason) == ("unverifiable", "no_data")


@pytest.mark.parametrize(
    ("claim", "metric", "comparator", "status"),
    [
        # C13: relational claims between two named targets (fake P/E: Moutai 24.6, Wuliangye 15.2)
        ("茅台的市盈率比五粮液高", "pe_ttm", "gt", "supported"),
        ("五粮液的市盈率比茅台高", "pe_ttm", "gt", "contradicted"),
        ("茅台市盈率没有五粮液高", "pe_ttm", "le", "contradicted"),
        ("五粮液ROE低于茅台", "roe", "lt", "supported"),
        ("茅台市净率比五粮液低", "pb", "lt", "contradicted"),
        ("五粮液的市盈率比贵州茅台低", "pe_ttm", "lt", "supported"),
        ("茅台昨天跌得比五粮液多", "pct_change_1d", "lt", "supported"),  # -0.18% vs +1.25%
        # English relations: stub NLU is Chinese-only here; covered by the dev bench on the offline service
    ],
)
def test_relational_claims_between_two_targets(claim, metric, comparator, status):
    (check,) = _check(claim).checks

    assert (check.metric, check.comparator, check.status) == (metric, comparator, status)
    assert check.claimed is None and check.reference and check.reference_value is not None
    assert check.reference_evidence_id and check.reference_evidence_id != check.evidence_id


def test_relational_claims_against_the_market_are_unverifiable():
    (check,) = _check("茅台市盈率高于市场平均").checks

    assert (check.status, check.reason) == ("unverifiable", "no_reference")


@pytest.mark.parametrize(
    ("claim", "comparator", "status"),
    [
        ("Moutai trades at 24.6 times earnings", "eq", "supported"),
        ("Moutai trades below 10x earnings", "lt", "contradicted"),
        ("Moutai trades at more than 20 times trailing earnings", "gt", "supported"),
    ],
)
def test_times_earnings_is_a_pe_claim(claim, comparator, status):
    (check,) = _check(claim).checks

    assert (check.metric, check.comparator, check.status) == ("pe_ttm", comparator, status)


# ---------------------------------------------------------------------------------------------------------
# Round 8: every clause gets a check (D2), industry averages as subjects (D3), turnover and bounded ratios (D4),
# and clauses with nothing to check listed as "not checked".
# ---------------------------------------------------------------------------------------------------------
_AMOUNTS = {"600519.SH": 3793827534.0, "000858.SZ": 1452833705.0}


class _IndustryRegistry:
    """The fake tools plus what the offline service also returns: the 白酒 industry snapshot with the
    fundamentals, and the session's turnover (CNY) on the price rows."""

    def __init__(self) -> None:
        self.inner = build_fake_registry()

    def run(self, name, arguments=None):
        from query_intelligence.agent.evidence import AgentEvidence

        result = self.inner.run(name, arguments)
        if not result.ok:
            return result
        if name == "get_fundamentals":
            industry = AgentEvidence(
                evidence_id="industry_白酒",
                kind="structured",
                source_type="industry_sql",
                as_of="2026-04-22",
                payload={
                    "industry_name": "白酒",
                    "trade_date": "2026-04-22",
                    "pe": 27.3,
                    "pb": 6.2,
                    "pct_change": -1.05,
                },
            )
            return result.model_copy(update={"evidence": [*result.evidence, industry]})
        if name == "get_price_history":
            evidence = [
                item.model_copy(update={"payload": {**item.payload, "amount": _AMOUNTS[item.payload["symbol"]]}})
                for item in result.evidence
            ]
            return result.model_copy(update={"evidence": evidence})
        return result


def _check_all(claim: str, *, zh: bool = True):
    return check_claim(claim, service=StubService(), registry=_IndustryRegistry(), zh=zh)


def _rows(report):
    return [(check.metric, check.comparator, check.status) for check in report.checks]


def test_a_relation_is_checked_next_to_a_number_in_another_clause():
    # D2: the relation used to be dropped whenever any clause of the sentence stated a number.
    report = _check_all("茅台的市盈率比五粮液高，五粮液市盈率15.2倍")

    assert _rows(report) == [("pe_ttm", "gt", "supported"), ("pe_ttm", "eq", "supported")]
    assert report.checks[0].reference == "五粮液" and report.verdict == "supported"

    # the relation's subject may be named in the number's clause: both parts are checked
    report = _check_all("五粮液市盈率20倍，比茅台低")
    assert _rows(report) == [("pe_ttm", "eq", "contradicted"), ("pe_ttm", "lt", "supported")]
    assert report.verdict == "partially_supported"


def test_a_relation_in_the_clause_of_a_number_is_read_as_that_number():
    # "五粮液的15.2倍" is 五粮液's P/E (P/E is quoted in 倍), not 15.2 times it. Since round 10 (F1) the comparison
    # of the two is checked too, next to the value stated for 五粮液.
    report = _check_all("茅台市盈率24.6倍比五粮液的15.2倍高")

    assert [check.reference for check in report.checks] == [None, "五粮液", None]
    assert [check.target for check in report.checks] == ["贵州茅台", "贵州茅台", "五粮液"]
    assert [check.kind for check in report.checks] == ["value", "relation", "stated_reference"]


def test_an_industry_average_named_before_a_number_is_its_subject():
    # D3: "而行业平均27.3倍" is the industry's value, not the company's.
    report = _check_all("茅台市盈率24.6倍，而行业平均27.3倍")

    first, industry = report.checks
    assert (first.target, first.actual, first.status) == ("贵州茅台", 24.6, "supported")
    assert (industry.target, industry.actual, industry.status) == ("白酒行业平均", 27.3, "supported")
    assert industry.evidence_id == "industry_白酒" and industry.as_of_basis == "trade_date"
    assert report.verdict == "supported"

    english = _check_all("Moutai's P/E is 24.6x while the industry average is 27.3x", zh=False)
    assert english.checks[1].target == "baijiu (liquor) industry average"
    assert english.checks[1].status == "supported"


def test_an_industry_average_after_a_bound_is_the_other_side_of_the_comparison():
    # Round 9 (E1): "低于行业平均30倍" states the average (30) and a relation with it. Before, the 30 was a bound on
    # the company's own P/E, so a made-up average passed; now the relation and the stated average are both checked.
    report = _check_all("茅台市盈率低于行业平均30倍")
    assert _rows(report) == [("pe_ttm", "lt", "supported"), ("pe_ttm", "eq", "contradicted")]
    relation, average = report.checks
    assert (relation.target, relation.reference, relation.reference_value) == ("贵州茅台", "白酒行业平均", 27.3)
    assert (average.target, average.claimed, average.actual) == ("白酒行业平均", 30.0, 27.3)

    # "…，低于行业均值": a relation with the industry snapshot, checked next to the number
    report = _check_all("茅台市盈率24.6倍，低于行业均值")
    assert _rows(report) == [("pe_ttm", "eq", "supported"), ("pe_ttm", "lt", "supported")]
    assert report.checks[1].reference == "白酒行业平均" and report.checks[1].reference_value == 27.3


@pytest.mark.parametrize(
    ("claim", "comparator", "status", "reason"),
    [
        ("茅台昨天成交37.9亿元", "eq", "supported", None),
        ("茅台成交额约38亿", "approx", "supported", None),
        ("茅台成交额超过50亿", "gt", "contradicted", None),
        ("茅台本周累计成交200亿元", "eq", "unverifiable", "multi_day"),
        ("茅台成交额37.9%", "eq", "unverifiable", "unit_mismatch"),
    ],
)
def test_turnover_claims(claim, comparator, status, reason):
    # D4: 成交额 is the session's turnover from the price evidence (37.94 亿元 here)
    (check,) = _check_all(claim).checks

    assert (check.metric, check.comparator, check.status, check.reason) == ("amount", comparator, status, reason)


def test_volume_is_not_read_as_turnover():
    (check,) = _check_all("茅台成交量2.7万手").checks

    assert check.metric != "amount" and check.status == "unverifiable"


@pytest.mark.parametrize(
    ("claim", "comparator", "status"),
    [
        # D4: a bound where the verb stands ("不到五粮液的1.5倍"); fake revenue 1741.2 / 890 亿 = 1.96
        ("茅台营收不到五粮液的1.5倍", "lt", "contradicted"),
        ("茅台营收超过五粮液的1.5倍", "gt", "supported"),
        ("茅台营收至少是五粮液的两倍", "ge", "contradicted"),
        ("茅台营收没有五粮液的两倍", "lt", "supported"),
    ],
)
def test_bounded_ratio_claims(claim, comparator, status):
    (check,) = _check_all(claim).checks

    assert (check.metric, check.comparator, check.status) == ("revenue", comparator, status)
    assert check.reference == "五粮液" and check.ratio == pytest.approx(1.9564, abs=1e-3)


@pytest.mark.parametrize(
    ("claim", "unchecked", "checks"),
    [
        ("茅台市盈率24.6倍，ROE很高", ["ROE很高"], 1),
        ("茅台是好公司", ["茅台是好公司"], 0),
        ("茅台和五粮液的市盈率，分别是24.6倍和15.2倍", [], 2),  # the first clause names what the second checks
        ("茅台市盈率24.6倍，品牌力很强", [], 1),  # nothing factual left: no target, no metric
    ],
)
def test_clauses_with_nothing_to_check_are_listed(claim, unchecked, checks):
    report = _check_all(claim)

    assert [part.text for part in report.unchecked] == unchecked
    assert len(report.checks) == checks
    assert all(part.reason == "no_claim" for part in report.unchecked)


# ---------------------------------------------------------------------------------------------------------
# Round 9 (after the round-5 review and the exposure of the round-5 held-out slice): stated industry averages (E1),
# partial coverage (E2), fractions and ratio phrasings, binding to the clause's own company (E9), derived and
# computed values (E8, ETF daily change), one evidence entry per id (E14).
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "rows"),
    [
        # the average after the number ("10倍的行业平均水平") and in brackets after the phrase
        ("茅台PB低于10倍的行业平均水平", [("pb", "lt", "contradicted"), ("pb", "eq", "contradicted")]),
        ("茅台市净率高于行业平均水平（6.2倍）", [("pb", "gt", "supported"), ("pb", "eq", "supported")]),
        ("茅台市盈率不到行业均值27.3倍", [("pe_ttm", "lt", "supported"), ("pe_ttm", "eq", "supported")]),
        # Round 10 (F1): P/E is quoted in 倍, so without a ratio cue ("是…的", "只有…的", 还/更, a fraction)
        # "行业均值的2倍" is the average the claim states (2x, contradicted by 27.3), not twice the average
        ("茅台市盈率不到行业均值的2倍", [("pe_ttm", "lt", "supported"), ("pe_ttm", "eq", "contradicted")]),
        ("茅台市盈率只有行业均值的0.9倍", [("pe_ttm", "eq", "supported")]),  # "只有…的": a multiple (24.6 / 27.3)
        # a relation in the clause of the company's own number, and the stated average ("of 40x")
        (
            "Moutai's P/E of 24.6x is below the industry average of 40x",
            [("pe_ttm", "eq", "supported"), ("pe_ttm", "lt", "supported"), ("pe_ttm", "eq", "contradicted")],
        ),
    ],
)
def test_a_stated_industry_average_is_checked_against_the_snapshot(claim, rows):
    report = _check_all(claim, zh=not claim.isascii())

    assert _rows(report) == rows
    assert (
        report.checks[-1].evidence_id == "industry_白酒" or report.checks[-1].reference_evidence_id == "industry_白酒"
    )


def test_a_relation_after_the_number_of_its_clause_is_checked():
    # "Wuliangye's P/B of 3.9x is below Moutai's": before round 9 the relation was dropped because its clause states
    # a number; the number and the relation are now two checks.
    report = _check_all("五粮液市净率3.9倍低于茅台")

    assert _rows(report) == [("pb", "eq", "supported"), ("pb", "lt", "supported")]
    assert report.checks[1].reference == "贵州茅台"


def test_vocabulary_words_are_not_company_mentions():
    from query_intelligence.agent.claim_check import _vocabulary_mention

    # "均值" is an alias of 武汉天源 in the alias table: inside "行业均值" it is not a company (E9)
    assert _vocabulary_mention("茅台PB 8.1倍，高于行业均值4倍", "均值")
    assert _vocabulary_mention("平安市盈率不到保险业均值11.8倍", "均值")
    assert not _vocabulary_mention("均值科技市盈率30倍", "均值")
    assert not _vocabulary_mention("茅台PB 8.1倍", "")


def test_a_short_name_in_a_later_clause_binds_to_its_own_company():
    # E9: "贵州茅台" is written in full once and as "茅台" later; the later number is 茅台's, not 五粮液's
    report = _check_all("贵州茅台市盈率比五粮液高，茅台市净率8.1倍")

    assert [(check.target, check.metric, check.status) for check in report.checks] == [
        ("贵州茅台", "pe_ttm", "supported"),
        ("贵州茅台", "pb", "supported"),
    ]


@pytest.mark.parametrize(
    ("claim", "normalised"),
    [
        ("平安市盈率只有白酒行业平均的三分之一左右", "的0.3333倍左右"),
        ("五粮液营收不到茅台的六成", "的0.6倍"),
        ("五粮液营收是茅台的两成多", "的0.2倍多"),
        ("Moutai's revenue is more than double Wuliangye's", "more than 2 times"),
        ("Moutai grew at a double-digit pace", "double-digit"),  # not a multiple
        ("三分之一的营收来自海外", "三分之一"),  # a share of its own, not "of another value"
    ],
)
def test_fractions_and_multiples_are_normalised(claim, normalised):
    assert normalised in normalise(claim)


@pytest.mark.parametrize(
    ("claim", "comparator", "claimed", "status"),
    [
        # fake revenue: Moutai 1741.2 亿, Wuliangye 890 亿 (ratio 1.956; the inverse 0.511)
        ("茅台营收比五粮液的1.5倍还多", "gt", 1.5, "supported"),
        ("茅台营收比五粮液的两倍还多", "gt", 2.0, "contradicted"),
        ("茅台营收是五粮液的一倍有余", "gt", 1.0, "supported"),
        ("五粮液营收不到茅台的六成", "lt", 0.6, "supported"),
        ("五粮液营收是茅台的51%", "eq", 0.51, "supported"),
    ],
)
def test_ratio_phrasings(claim, comparator, claimed, status):
    (check,) = _check_all(claim, zh=not claim.isascii()).checks

    assert (check.metric, check.comparator, check.claimed, check.status) == ("revenue", comparator, claimed, status)
    assert check.ratio is not None and check.reference in {"五粮液", "贵州茅台"}


def test_a_multiple_of_the_industry_average():
    # "市盈率只有行业平均的一半": 24.6 / 27.3 = 0.90, not 0.5
    (check,) = _check_all("茅台市盈率只有行业平均的一半").checks

    assert (check.comparator, check.claimed, check.status) == ("eq", 0.5, "contradicted")
    assert check.reference == "白酒行业平均" and check.ratio == pytest.approx(0.9011, abs=1e-3)


@pytest.mark.parametrize(
    ("claim", "coverage", "verdict"),
    [
        ("茅台市盈率24.6倍", "full", "supported"),
        ("茅台市盈率24.6倍，ROE很高", "partial", "supported"),  # E2: the verdict is over checks, coverage says more
        ("茅台市盈率24.6倍，茅台股息率3%", "partial", "partially_supported"),  # an unverifiable part
        ("茅台是好公司", "none", "unverifiable"),
    ],
)
def test_coverage_says_whether_every_part_was_checked(claim, coverage, verdict):
    report = _check_all(claim)

    assert (report.verdict, report.coverage) == (verdict, coverage)


def test_evidence_sources_are_listed_once():
    # E14: a relation with the industry and the industry average share one snapshot
    report = _check_all("茅台市盈率24.6倍，低于行业均值")
    ids = [source["evidence_id"] for source in report.evidence_sources]

    assert len(ids) == len(set(ids)) and "industry_白酒" in ids


def test_derived_values_carry_their_arithmetic():
    from query_intelligence.agent.claim_check import _derived
    from query_intelligence.agent.evidence import AgentEvidence

    fundamentals = AgentEvidence(
        evidence_id="fundamental_600519.SH",
        kind="structured",
        source_type="fundamental_sql",
        payload={"revenue": 168838000000, "net_profit": 82320000000, "pe_ttm": 24.6},
    )
    (item, margin), note = _derived("net_margin", [fundamentals])
    assert item is fundamentals and margin == pytest.approx(48.7568, abs=1e-3)
    assert note.startswith("derived: net profit / revenue")
    assert _derived("peg", [fundamentals]) == (None, "")  # no growth rate: not derivable

    price = AgentEvidence(
        evidence_id="price_510300.SH",
        kind="structured",
        source_type="market_api",
        as_of="2026-04-22",
        payload={
            "close": 4.811,
            "pct_change_1d": None,
            "recent_closes": [{"date": "2026-04-21", "close": 4.776}, {"date": "2026-04-22", "close": 4.811}],
        },
    )
    (_item, change), note = _derived("pct_change_1d", [price])
    assert change == pytest.approx(0.7328, abs=1e-4)
    assert note.startswith("computed from the last two closes: 4.776 (2026-04-21) → 4.811 (2026-04-22)")
    stale = price.model_copy(update={"as_of": "2026-04-23"})
    assert _derived("pct_change_1d", [stale]) == (None, "")  # the closes must end at the quoted day


def test_metrics_the_sources_cannot_give_say_why():
    report = _check_all("茅台市销率10倍，茅台PEG为1.5")

    assert [(check.metric, check.reason) for check in report.checks] == [("ps", "no_data"), ("peg", "no_data")]
    assert "price-to-sales" in report.checks[0].note and "growth rate" in report.checks[1].note


def test_one_macro_series_against_another():
    report = check_claim("M2增速高于CPI，CPI低于10年期国债收益率", service=StubService(), registry=_macro_registry())

    assert [(c.metric, c.comparator, c.reference_value, c.status) for c in report.checks] == [
        ("m2_yoy", "gt", 0.8, "supported"),
        ("cpi_yoy", "lt", 2.31, "supported"),
    ]


# ---------------------------------------------------------------------------------------------------------
# Round 10 (after the round-6 review): a stated value for the compared side (F1), stated differences (F2), and
# numerals with 多 / 余 / 出头 / 左右 as bounded approximations (F7). Fake data: P/E 24.6 / 15.2, ROE 0.33 / 0.24,
# revenue 1741.2亿 / 890亿; 白酒 industry P/E 27.3.
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "rows", "kinds"),
    [
        # F1: P/E is quoted in 倍, so "平均的30倍" is the average the claim states, checked against the snapshot
        (
            "茅台市盈率24.6倍，比白酒行业平均的30倍低不少",
            [("pe_ttm", "eq", "supported"), ("pe_ttm", "lt", "supported"), ("pe_ttm", "eq", "contradicted")],
            ["value", "relation", "stated_reference"],
        ),
        (
            "茅台市盈率低于白酒行业平均的20倍",
            [("pe_ttm", "lt", "supported"), ("pe_ttm", "eq", "contradicted")],
            ["relation", "stated_reference"],
        ),
        # the same rule for a named target, and for a metric quoted in % ("五粮液的24%" is 五粮液's ROE)
        (
            "茅台市盈率低于五粮液的30倍",
            [("pe_ttm", "lt", "contradicted"), ("pe_ttm", "eq", "contradicted")],
            ["relation", "stated_reference"],
        ),
        (
            "茅台ROE高于五粮液的24%",
            [("roe", "gt", "supported"), ("roe", "eq", "supported")],
            ["relation", "stated_reference"],
        ),
        # explicit ratio cues keep the multiple: 是…的, 还/更 after 比, a fraction
        ("茅台市盈率是五粮液的1.6倍", [("pe_ttm", "eq", "supported")], ["ratio"]),
        ("茅台市盈率比五粮液的1.5倍还高", [("pe_ttm", "gt", "supported")], ["ratio"]),
        ("茅台市盈率不到白酒行业平均的三分之二", [("pe_ttm", "lt", "contradicted")], ["ratio"]),  # 0.90
    ],
)
def test_a_stated_value_of_the_compared_side_is_its_own_check(claim, rows, kinds):
    report = _check_all(claim)

    assert _rows(report) == rows
    assert [check.kind for check in report.checks] == kinds


def test_the_stated_industry_average_is_labelled_as_the_average():
    report = _check_all("茅台市盈率24.6倍，比白酒行业平均的30倍低不少")

    stated = report.checks[-1]
    assert stated.target == "白酒行业平均" and (stated.claimed, stated.actual) == (30.0, 27.3)
    assert report.verdict == "partially_supported"


@pytest.mark.parametrize(
    ("claim", "comparator", "claimed", "difference", "status"),
    [
        # F2: the number is the difference of the two, not the second company's own value
        ("茅台ROE比五粮液高出约9个百分点", "approx", 9.0, 9.0, "supported"),
        ("五粮液ROE比茅台高9个百分点", "eq", 9.0, -9.0, "contradicted"),  # the difference is the other way
        ("五粮液ROE比茅台低了9个百分点", "eq", -9.0, -9.0, "supported"),
        ("茅台ROE比五粮液高出不到5个百分点", "lt", 5.0, 9.0, "contradicted"),
        ("茅台的市盈率比五粮液高9.4倍", "eq", 9.4, 9.4, "supported"),  # P/E is quoted in 倍: 24.6 - 15.2
        ("茅台和五粮液的市盈率相差9.4倍左右", "approx", 9.4, 9.4, "supported"),  # no direction: the size
        ("茅台营收比五粮液多八百多亿", "gt", 800.0, 85120000000.0, "supported"),  # 800亿 < 851.2亿 < 900亿
    ],
)
def test_a_stated_difference_is_checked_against_both_values(claim, comparator, claimed, difference, status):
    (check,) = _check_all(claim).checks

    assert (check.comparator, check.claimed, check.status) == (comparator, claimed, status)
    assert check.kind == "difference" and check.reference in {"五粮液", "贵州茅台"}
    assert check.difference == pytest.approx(difference, abs=1e-6)


def test_a_percentage_difference_of_a_multiple_is_relative():
    # "比行业平均低了近10%": (24.6 - 27.3) / 27.3 = -9.9%
    (check,) = _check_all("茅台的PE比行业平均低了近10%").checks

    assert (check.kind, check.comparator, check.claimed, check.status) == (
        "relative_difference",
        "approx",
        -10.0,
        "supported",
    )
    assert check.difference == pytest.approx(-9.89, abs=0.01) and check.reference == "白酒行业平均"


def test_an_ambiguous_multiple_difference_is_unverifiable():
    # "高出一倍" of revenue: one or two times more? A multiple states it ("是…的两倍").
    (check,) = _check_all("茅台营收比五粮液高出一倍").checks

    assert (check.kind, check.status, check.reason) == ("difference", "unverifiable", "unit_mismatch")


@pytest.mark.parametrize(
    ("claim", "normalised", "comparator", "claimed", "high", "status"),
    [
        # F7: 多 / 余 is more than the number and less than its next step; 出头 is the lower half of that step
        ("茅台营收一千七百多亿", "1700多亿", "gt", 1700.0, 1800.0, "supported"),
        ("茅台营收一千六百余亿", "1600余亿", "gt", 1600.0, 1700.0, "contradicted"),  # 1741.2 is not below 1700
        ("茅台ROE三成出头", "30%出头", "gt", 30.0, 35.0, "supported"),
        ("茅台ROE四成出头", "40%出头", "gt", 40.0, 45.0, "contradicted"),  # 33%
        # 左右: 5%, or half the step of the last significant digit when wider (三成 = 30%: 25%-35%)
        ("茅台ROE三成左右", "30%左右", "approx", 30.0, None, "supported"),
        ("茅台ROE四成左右", "40%左右", "approx", 40.0, None, "contradicted"),  # 33% is not within 35%-45%
    ],
)
def test_numerals_with_more_or_about_are_bounded_approximations(claim, normalised, comparator, claimed, high, status):
    assert normalised in normalise(claim)
    (check,) = _check_all(claim).checks

    assert (check.comparator, check.claimed, check.claimed_high, check.status) == (comparator, claimed, high, status)
