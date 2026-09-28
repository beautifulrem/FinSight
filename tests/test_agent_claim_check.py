"""Claim check: numbers in a pasted claim against market and fundamental evidence."""

from __future__ import annotations

from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.claim_check import check_claim
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
