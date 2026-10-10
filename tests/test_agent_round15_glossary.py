"""Round-15 (review 11, K12): definition questions about the labels the product shows (扣非 vs 归母 net profit, weighted
vs diluted / period-end ROE) are answered from the curated glossary with an evidence id, instead of a clarification
("which stock?") or "no usable evidence". A request for the figure without a company is still not a definition.
Wording is the engineer's own.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.glossary import GLOSSARY, lookup_concept
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.tools import build_registry_for_service


@pytest.mark.parametrize(
    ("query", "terms"),
    [
        ("扣非净利润和归母净利润的区别在哪", {"扣非净利润", "归母净利润"}),
        ("扣非和归母差在哪里，怎么理解", {"扣非净利润", "归母净利润"}),
        ("归母净利润是什么意思", {"归母净利润"}),
        ("What is recurring net profit?", {"扣非净利润"}),
        ("What does weighted average ROE mean?", {"加权平均ROE"}),
        ("加权ROE和摊薄ROE有什么不同", {"加权平均ROE"}),
        ("ROE（加权）指的是什么", {"加权平均ROE"}),
        # a figure without a company is not a definition question (it is still clarified)
        ("扣非净利润是多少", set()),
        ("weighted ROE above 20%", set()),
    ],
)
def test_label_definitions_are_glossary_terms(query, terms):
    entry = lookup_concept(query)
    assert ({entry.term} if entry else set()) <= terms and bool(entry) == bool(terms), entry


def test_label_entries_explain_both_sides():
    entries = {entry.term: entry for entry in GLOSSARY}
    assert "归母" in entries["扣非净利润"].zh and "attributable" in entries["扣非净利润"].en
    assert "扣非" in entries["归母净利润"].zh
    assert "摊薄" in entries["加权平均ROE"].zh and "period-end" in entries["加权平均ROE"].en


@pytest.fixture
def agent(offline_service):
    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), ScriptedLLM([]))
    service = AgentService(runtime, trace_sinks=[])
    yield service
    runtime.close()


@pytest.mark.parametrize(
    ("query", "evidence_id", "must_contain"),
    [
        ("扣非净利润和归母净利润有啥区别", "glossary_扣非净利润", ["非经常性损益", "归母"]),
        ("What does weighted average ROE mean?", "glossary_加权平均ROE", ["weighted", "period-end"]),
    ],
)
def test_label_definition_questions_are_answered_with_evidence(agent, query, evidence_id, must_contain):
    result = agent.chat(query)
    assert result["route"] != "clarify", result["route_reasons"]
    answer = str(result["answer"])
    assert evidence_id in (result.get("evidence_used") or []), (result["route_reasons"], answer)
    for text in must_contain:
        assert text in answer, answer
    assert "无法给出具体数值" not in answer and "definition only, with no figures" not in answer, answer
