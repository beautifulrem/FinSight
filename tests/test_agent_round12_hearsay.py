"""Round-12 rule H12 (round-8 review): hearsay cues are classes (a source that says, a report word, a request to
check, a confirmation question at the end), so "I read that … True?", "网上说…对吗", "有博主说…靠谱吗" start the inline
fact check; ordinary questions do not.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def agent(offline_service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), ScriptedLLM([]))
    service = AgentService(runtime, trace_sinks=[])
    yield service
    runtime.close()


# ---- H12: hearsay cues for the inline fact check ----


@pytest.mark.parametrize(
    ("message", "claim"),
    [
        (
            "I read that Ping An's P/E is 15x and Moutai's ROE is 33%. True?",
            "Ping An's P/E is 15x and Moutai's ROE is 33%",
        ),
        ("I saw a post saying Wuliangye's P/B is 3x. Is that correct?", "Wuliangye's P/B is 3x"),
        ("I've heard Moutai trades at 40x earnings. Correct?", "Moutai trades at 40x earnings"),
        ("A friend told me that Ping An's ROE is above 20%, right?", "Ping An's ROE is above 20%"),
        ("Apparently Moutai closed at 1500 yuan, can you verify?", "Moutai closed at 1500 yuan"),
        ("Fact-check: Wuliangye's revenue exceeds 100 billion yuan", "Wuliangye's revenue exceeds 100 billion yuan"),
        ("Is it true that Moutai's P/B is 8.1x?", "Moutai's P/B is 8.1x"),
        ("网上说五粮液ROE超过30%，对吗", "五粮液ROE超过30%"),
        ("有博主说中国平安市净率不到1倍，靠谱吗", "中国平安市净率不到1倍"),
        ("朋友告诉我茅台营收超过2000亿，是这样吗", "茅台营收超过2000亿"),
        ("看到新闻说五粮液营收破千亿，属实吗？", "五粮液营收破千亿"),
        ("群里都在传茅台跌了5%，真的假的", "茅台跌了5%"),
        ("茅台市盈率24.6倍，没错吧", "茅台市盈率24.6倍"),
        ("帮我核实一下：中国平安PE只有8.7倍", "中国平安PE只有8.7倍"),
    ],
)
def test_hearsay_cues_start_the_inline_fact_check(message, claim):
    from query_intelligence.agent.hearsay import claim_in_message

    assert claim_in_message(message) == claim


@pytest.mark.parametrize(
    "message",
    [
        "茅台市盈率是多少",
        "贵州茅台最近走势怎么样？",
        "听说茅台管理层很好，是真的吗",
        "What is Moutai's P/E?",
        "Compare Moutai and Wuliangye's ROE",
        "说说茅台2025年的营收",
        "I read the annual report, what was Moutai's revenue?",
    ],
)
def test_ordinary_questions_are_not_hearsay(message):
    from query_intelligence.agent.hearsay import claim_in_message

    assert claim_in_message(message) is None


def test_i_read_that_true_gets_the_inline_fact_check(agent):
    result = agent.chat("I read that Ping An's P/E is 15x and Moutai's ROE is 33%. True?", session_id="r12-hearsay-en")
    report = result["fact_check"]
    assert report["verdict"] == "partially_supported"
    assert [(check["metric"], check["status"]) for check in report["checks"]] == [
        ("pe_ttm", "contradicted"),
        ("roe", "supported"),
    ]
    assert result["answer"].startswith("**Fact check: the claim you heard partly matches the data.**")
