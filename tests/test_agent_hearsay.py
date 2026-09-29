"""Hearsay in the chat ("听说茅台市盈率只有15倍，是真的吗") gets an inline, deterministic fact check."""

from __future__ import annotations

from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.hearsay import claim_in_message, fact_check_for
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app


@pytest.mark.parametrize(
    ("message", "claim"),
    [
        ("听说茅台市盈率只有15倍，是真的吗", "茅台市盈率只有15倍"),
        ("据说五粮液ROE超过三成？", "五粮液ROE超过三成"),
        ("茅台昨天跌了5%，真的吗？", "茅台昨天跌了5%"),
        ("听说茅台的市盈率比五粮液高，对吗", "茅台的市盈率比五粮液高"),
        ("I heard that Moutai's P/E is only 15x, is that true?", "Moutai's P/E is only 15x"),
        ("Is it true that Moutai fell 5% yesterday?", "Moutai fell 5% yesterday"),
        ("贵州茅台最近走势怎么样？", None),
        ("茅台市盈率是多少", None),
        ("听说茅台管理层很好，是真的吗", None),
    ],
)
def test_claim_in_message(message, claim):
    assert claim_in_message(message) == claim


def test_fact_check_for_a_hearsay_question():
    report = fact_check_for("听说茅台市盈率只有15倍，是真的吗", service=StubService(), registry=build_fake_registry())

    assert report is not None and report["claim"] == "茅台市盈率只有15倍"
    assert report["verdict"] == "contradicted" and report["checks"][0]["actual"] == 24.6
    assert fact_check_for("贵州茅台的市盈率是多少", service=StubService(), registry=build_fake_registry()) is None


def test_fact_check_never_breaks_the_answer():
    class Broken(StubService):
        def analyze_query(self, *args, **kwargs):
            raise RuntimeError("nlu down")

    assert fact_check_for("听说茅台市盈率只有15倍，是真的吗", service=Broken(), registry=build_fake_registry()) is None


class _LegacyStub(StubService):
    def run_pipeline(self, query, user_profile=None, dialog_context=None, top_k=20, debug=False):
        return {
            "nlu_result": self.analyze_query(query),
            "retrieval_result": {"documents": [], "structured_data": [], "warnings": [], "coverage": {}},
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    stub = _LegacyStub()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime))
    return TestClient(app)


def test_agent_path_carries_the_fact_check(client):
    body = client.post("/agent/chat", json={"query": "听说茅台市盈率只有15倍，是真的吗"}).json()

    assert body["status"] == "ok"
    assert body["fact_check"]["verdict"] == "contradicted"
    assert body["fact_check"]["checks"][0]["metric"] == "pe_ttm"
    plain = client.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少"}).json()
    assert plain["fact_check"] is None


def test_agent_stream_answer_carries_the_fact_check(client):
    text = client.post("/agent/chat/stream", json={"query": "听说茅台昨天跌超0.1%，是真的吗"}).text

    answer = next(block for block in text.split("\n\n") if block.startswith("event: answer\n"))
    assert '"fact_check": {' in answer and '"direction": "down"' in answer and '"verdict": "supported"' in answer


def test_workflow_path_carries_the_fact_check(client, monkeypatch):
    import query_intelligence.api.app as app_module

    monkeypatch.setattr(app_module, "build_chatbot_response", lambda **kwargs: {"answer": "legacy"})

    body = client.post("/chat", json={"query": "听说茅台市盈率只有15倍，是真的吗"}).json()
    plain = client.post("/chat", json={"query": "贵州茅台的市盈率是多少"}).json()

    assert body["answer"] == "legacy" and body["fact_check"]["verdict"] == "contradicted"
    assert body["fact_check"]["checks"][0]["claimed"] == 15.0
    assert "fact_check" not in plain


def test_nlu_summary_names_carry_the_english_alias(client):
    body = client.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少"}).json()

    assert body["nlu_summary"]["entities"][0]["name_en"] == "Kweichow Moutai"
