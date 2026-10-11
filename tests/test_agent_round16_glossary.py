"""Round 16 (review 12, L11): every figure the KPI tiles show has a curated definition, and a definition question asked
inside a session is answered with the definition, not with the previous company's data. Wording is the engineer's own.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.glossary import GLOSSARY, lookup_concept
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.tools import build_registry_for_service

# The labels of the price, fundamentals and indicator tiles (frontend/src/lib/i18n.ts, kpi.*), zh and en, plus the
# figures the review asked for by name (毛利率, EPS 基本 / 稀释, 流通市值, 换手率)
TILE_LABELS = [
    ("收盘价", "收盘价"),
    ("日涨跌幅", "涨跌幅"),
    ("成交额", "成交额"),
    ("市盈率 TTM", "市盈率"),
    ("市净率", "市净率"),
    ("ROE", "ROE"),
    ("ROE（期末净资产）", "加权平均ROE"),
    ("营业收入", "营业总收入"),
    ("营业总收入", "营业总收入"),
    ("归母净利润", "归母净利润"),
    ("净利润（含少数股东损益）", "归母净利润"),
    ("行业市盈率", "市盈率"),
    ("行业市净率", "市净率"),
    ("净利率", "净利率"),
    ("估值日", "估值日"),
    ("MA20", "均线"),
    ("RSI(14)", "RSI"),
    ("20日波动率", "波动率"),
    ("毛利率", "毛利率"),
    ("基本每股收益", "每股收益"),
    ("稀释每股收益", "每股收益"),
    ("流通市值", "总市值"),
    ("总市值", "总市值"),
    ("换手率", "换手率"),
    ("TTM", "TTM"),
]


@pytest.mark.parametrize(("label", "term"), TILE_LABELS)
def test_every_tile_label_has_a_definition(label, term):
    entry = lookup_concept(f"{label}是什么意思")
    assert entry is not None and entry.term == term, (label, entry and entry.term)


@pytest.mark.parametrize(
    ("query", "term"),
    [
        ("市盈率指的是什么", "市盈率"),
        ("What is the P/B ratio?", "市净率"),
        ("What is a P/E ratio?", "市盈率"),
        ("ROE怎么理解", "ROE"),
        ("破净是什么意思", "市净率"),
        ("营业收入和营业总收入有什么区别", "营业总收入"),
        ("基本每股收益和稀释每股收益差在哪", "每股收益"),
        ("What does market cap mean?", "总市值"),
        ("How is net margin calculated?", "净利率"),
        ("What does gross margin mean?", "毛利率"),
    ],
)
def test_definition_questions_find_the_metric(query, term):
    entry = lookup_concept(query)
    assert entry is not None and entry.term == term, entry and entry.term


@pytest.mark.parametrize(
    "query",
    [
        "贵州茅台眼下的市盈率是几倍",  # a figure, not a definition
        "市值最大的是什么公司",  # a ranking: the cue is not about the word
        "今天涨幅最大的是什么",
        "PE above 20",
        "招商银行的市净率是多少",
    ],
)
def test_figure_questions_are_not_definitions(query):
    entry = lookup_concept(query)
    assert entry is None, entry.term


def test_the_new_entries_are_figure_labels_with_data():
    entries = {entry.term: entry for entry in GLOSSARY}
    for term in ("市盈率", "市净率", "ROE", "毛利率", "净利率", "营业总收入", "每股收益", "总市值", "TTM"):
        assert entries[term].definition_only and entries[term].has_data_series, term
    assert "破净" in entries["市净率"].zh and "TTM" in entries["市盈率"].zh
    assert "稀释" in entries["每股收益"].zh and "diluted" in entries["每股收益"].en
    assert "流通" in entries["总市值"].zh and "free-float" in entries["总市值"].en


@pytest.fixture
def agent(offline_service):
    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), ScriptedLLM([]))
    service = AgentService(runtime, trace_sinks=[])
    yield service
    runtime.close()


@pytest.mark.parametrize(
    ("first", "question", "evidence_id", "must_contain"),
    [
        ("平安现在市盈率多少", "归母净利润怎么理解", "glossary_归母净利润", "少数股东"),
        ("比亚迪的毛利率", "毛利率是什么意思", "glossary_毛利率", "营业成本"),
        ("招商银行市净率多少", "市净率应该怎么理解", "glossary_市净率", "每股净资产"),
        ("What is CATL's P/E?", "What does P/E mean?", "glossary_市盈率", "earnings per share"),
    ],
)
def test_a_definition_question_in_a_session_answers_the_definition(agent, first, question, evidence_id, must_contain):
    session = f"r16-{evidence_id}"
    agent.chat(first, session_id=session)
    result = agent.chat(question, session_id=session)
    used = result.get("evidence_used") or []
    assert evidence_id in used, (result["route_reasons"], used)
    assert not any(item.startswith(("fundamental_", "price_")) for item in used), (result["route_reasons"], used)
    assert must_contain in str(result["answer"]), result["answer"]
    assert not any(str(reason).startswith("ellipsis:") for reason in result["route_reasons"]), result["route_reasons"]


def test_a_metric_follow_up_still_carries_the_company(agent):
    agent.chat("平安现在市盈率多少", session_id="r16-carry")
    result = agent.chat("市净率呢", session_id="r16-carry")
    assert "fundamental_601318.SH" in (result.get("evidence_used") or []), result["route_reasons"]


def test_a_standalone_definition_is_answered_from_the_glossary(agent):
    result = agent.chat("市盈率指的是什么")
    assert "glossary_市盈率" in (result.get("evidence_used") or []), result["route_reasons"]
    assert "每股收益" in str(result["answer"]), result["answer"]


@pytest.mark.parametrize(
    ("query", "evidence_ids"),
    [
        ("市盈率和市净率有什么区别", {"glossary_市盈率", "glossary_市净率"}),
        ("What is the difference between P/E and P/B?", {"glossary_市盈率", "glossary_市净率"}),
        ("毛利率和净利率差在哪", {"glossary_毛利率", "glossary_净利率"}),
    ],
)
def test_a_question_about_two_labels_defines_both(agent, query, evidence_ids):
    result = agent.chat(query)
    used = set(result.get("evidence_used") or [])
    assert evidence_ids <= used, (result["route_reasons"], used)
    assert (result.get("verification") or {}).get("passed"), result.get("verification")
