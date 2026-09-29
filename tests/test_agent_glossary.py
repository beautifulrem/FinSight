"""Curated glossary of market concepts (round-5 C6): lookup rules, routing, plan and answer.

The questions here are new wording, not copied from any evaluation set.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.glossary import GLOSSARY, lookup_concept
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM
from query_intelligence.agent.planner import plan_from_nlu
from query_intelligence.agent.router import apply_finance_overrides, decide_route, glossary_concept
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.tools import build_registry_for_service


@pytest.mark.parametrize(
    ("query", "term"),
    [
        ("北向资金是啥", "北向资金"),
        ("北上资金最近净买入多少", "北向资金"),
        ("融资融券是什么意思", "融资融券"),
        ("两融余额高说明什么", "融资融券"),
        ("国家队是指什么", "国家队"),
        ("What are northbound funds?", "北向资金"),
        ("what does margin trading mean", "融资融券"),
        ("沪港通和深港通有什么区别", "沪深港通"),
        ("ST股能买吗", "ST股"),
        # an everyday meaning without a market cue is not a glossary question
        ("国家队队员名单什么时候公布", None),
        ("主力队员受伤了吗", None),
        # ordinary sentences must not match a Latin form inside a word
        ("the nationalteamsomething", None),
        ("茅台的市盈率是多少", None),
    ],
)
def test_lookup_concept(query, term):
    entry = lookup_concept(query)
    assert (entry.term if entry else None) == term
    assert glossary_concept(query) == term


def test_glossary_entries_state_no_figures():
    """Definitions are prose: a number in them could not be traced to a tool result by the verifier."""
    for entry in GLOSSARY:
        assert not any(char.isdigit() for char in entry.zh), entry.term
        assert not any(char.isdigit() for char in entry.en), entry.term
        assert entry.evidence_id == f"glossary_{entry.term}"


def _nlu(query: str) -> dict:
    return {
        "raw_query": query,
        "normalized_query": query,
        "question_style": "fact",
        "product_type": {"label": "out_of_scope"},
        "intent_labels": [],
        "entities": [],
        "risk_flags": ["out_of_scope_query"],
        "missing_slots": ["missing_entity"],
        "comparison_targets": [],
    }


def test_out_of_scope_concept_is_kept_in_scope_and_planned():
    nlu, reasons = apply_finance_overrides(_nlu("北向资金是啥"), "北向资金是啥")
    assert reasons == ["override:out_of_scope_glossary_concept:北向资金"]
    decision = decide_route(nlu, query="北向资金是啥")
    assert decision.route == "workflow" and decision.reasons == ["concept:glossary:北向资金"]
    plan = plan_from_nlu(nlu)
    assert [(call.tool, call.arguments["term"]) for call in plan.calls] == [("explain_concept", "北向资金")]


def test_a_named_security_is_not_answered_from_the_glossary():
    nlu = {
        **_nlu("ST股里的贵州茅台"),
        "product_type": {"label": "stock"},
        "risk_flags": [],
        "missing_slots": [],
        "entities": [
            {"canonical_name": "贵州茅台", "symbol": "600519.SH", "entity_type": "stock", "mention": "贵州茅台"}
        ],
    }
    plan = plan_from_nlu(nlu)
    assert "explain_concept" not in {call.tool for call in plan.calls}


@pytest.fixture
def agent(offline_service):
    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), ScriptedLLM([]))
    service = AgentService(runtime, trace_sinks=[])
    yield service
    runtime.close()


@pytest.mark.parametrize(
    ("query", "must_contain"),
    [
        ("融资融券是什么意思", ["融资融券", "glossary_融资融券", "数据序列"]),
        ("两融余额现在是多少", ["融资融券", "无法给出具体数值"]),
        ("What is margin trading?", ["margin", "glossary_融资融券"]),
    ],
)
def test_concept_questions_are_answered_not_refused(agent, query, must_contain):
    result = agent.chat(query)
    assert result["route"] == "workflow", result["route_reasons"]
    answer = str(result["answer"])
    for text in must_contain:
        assert text in answer, answer
    assert "glossary_融资融券" in (result.get("evidence_used") or [])
