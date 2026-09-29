"""Round-5 rules (reviewer round 3: C5, C7, C9, C10, C12, C20).

Every question here is new wording, not copied from the reviewer's battery, an independent set or a test set.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.tools import build_registry_for_service
from query_intelligence.chat.language import detect_query_language, requested_answer_language


@pytest.fixture
def agent(offline_service):
    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), ScriptedLLM([]))
    service = AgentService(runtime, trace_sinks=[])
    yield service
    runtime.close()


# --------------------------------------------------------------------------- C12: explicit answer language


@pytest.mark.parametrize(
    ("query", "language"),
    [
        ("麻烦用英语回复：招商银行的股息率", "en"),
        ("英文作答，沪深300ETF收在多少", "en"),
        ("Reply in Chinese please: how did Ping An close?", "zh"),
        ("Explain in Mandarin, please: what's the CSI 300 level?", "zh"),
        ("in English please, 中国平安市净率", "en"),
        # no instruction: the question's own language
        ("中国平安的市净率", "zh"),
        ("What's Ping An's P/B?", "en"),
    ],
)
def test_explicit_answer_language_overrides_detection(query, language):
    assert detect_query_language(query) == language


def test_the_last_language_instruction_wins():
    assert requested_answer_language("先用中文回答，算了还是用英文回答吧：茅台PE") == "en"
    assert requested_answer_language("茅台PE多少") is None


def test_agent_answers_in_the_requested_language(agent):
    english = agent.chat("麻烦用英语回复：五粮液的市净率")
    assert english["language"] == "en"
    assert "fundamentals" in str(english["answer"]) and "基本面" not in str(english["answer"])
    chinese = agent.chat("Reply in Chinese please: what is Kweichow Moutai's P/B?")
    assert chinese["language"] == "zh"
    assert "基本面" in str(chinese["answer"])
