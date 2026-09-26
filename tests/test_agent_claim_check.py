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
