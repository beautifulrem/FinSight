"""Round-12 fixes from the round-8 review.

* H7: an English one-word follow-up on the LLM agent path ("Wuliangye's P/E ratio?" / "And Moutai?") was answered in
  Chinese. The agent prompt now shows the user's own words next to the resolved question and asks for English in so
  many words; a follow-up with no language signal ("PE?", "Moutai?") keeps the conversation's language; and
  ``language_violation`` catches a short Chinese answer to an English question (the old rule needed 40 Chinese
  characters).
"""

from __future__ import annotations

import pytest

# ---- H7: answer language ----


@pytest.mark.parametrize(
    ("text", "signal"),
    [
        ("PE?", False),
        ("P/E", False),
        ("600519.SH", False),
        ("Moutai?", False),
        ("why?", False),
        ("ROE？", False),
        ("And Moutai?", True),
        ("what about Ping An", True),
        ("茅台呢", True),
        ("ROE呢", True),
        ("请用英文回答", True),
    ],
)
def test_language_signal_of_short_follow_ups(text, signal):
    from query_intelligence.chat.language import has_language_signal

    assert has_language_signal(text) is signal


def test_a_follow_up_without_a_language_signal_keeps_the_session_language():
    from query_intelligence.chat.language import session_answer_language

    assert session_answer_language("PE?", "en") == "en"
    assert session_answer_language("PE?", "zh") == "zh"
    assert session_answer_language("Moutai?", "zh") == "zh"
    assert session_answer_language("And Moutai?", "zh") == "en"
    assert session_answer_language("茅台呢", "en") == "zh"
    assert session_answer_language("PE?", None) == "en"  # first turn: the message's own script


# the reviewer's answer to "And Moutai?" (round 8, L2b)
_CHINESE_ANSWER = "贵州茅台（600519.SH）的市盈率 PE(TTM) 为 24.6 倍，基于 FY2025 报告基本面 [fundamental_600519.SH]。"


def test_language_violation_catches_a_short_chinese_answer_to_an_english_question():
    from query_intelligence.agent.compliance import language_violation

    assert language_violation(_CHINESE_ANSWER, "And Moutai?", language="en")
    # English with Chinese names, tickers and citations is English
    english = "Kweichow Moutai (贵州茅台, 600519.SH) has a P/E (TTM) of 24.6x [fundamental_600519.SH]."
    assert not language_violation(english, "And Moutai?", language="en")
    names = "贵州茅台 and 五粮液 trade at 24.6x and 20.9x P/E [fundamental_600519.SH][fundamental_000858.SZ]."
    assert not language_violation(names, "Compare them", language="en")
    # the Chinese side is unchanged
    assert not language_violation(_CHINESE_ANSWER, "茅台呢", language="zh")


def _llm_service(offline_service, steps):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.state import AgentConfig
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    llm = ScriptedLLM(steps)
    # the shipped defaults (tests/conftest.py pins the older classic loop for older tests)
    config = AgentConfig(planner_prefetch=True, revise_policy="cite_repair", verify_derived=True)
    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), llm, config=config)
    return llm, runtime, AgentService(runtime, trace_sinks=[])


def _user_messages(request) -> str:
    return "\n".join(str(m.get("content")) for m in request["messages"] if m.get("role") == "user")


def test_an_english_one_word_follow_up_is_answered_in_english(offline_service):
    from query_intelligence.agent.llm import final_turn

    llm, runtime, service = _llm_service(
        offline_service,
        [
            final_turn({"answer": "Wuliangye's P/E (TTM) is 20.9x [fundamental_000858.SZ].", "evidence_used": []}),
            # the model answers the follow-up in Chinese, as DeepSeek did in the review
            final_turn({"answer": _CHINESE_ANSWER, "evidence_used": []}),
        ],
    )
    try:
        first = service.chat("Wuliangye's P/E ratio?", session_id="r12-h7", mode="agent")
        second = service.chat("And Moutai?", session_id="r12-h7", mode="agent")
    finally:
        runtime.close()
    assert first["answer"].startswith("Wuliangye's P/E")
    prompt = _user_messages(llm.requests[-1])
    assert "User's message: And Moutai?" in prompt
    assert "Answer language: English (write the answer" in prompt
    # the Chinese draft is caught and replaced by the English evidence summary
    assert "language_mismatch_fallback_to_template" in second["compliance_notes"]
    assert "Kweichow Moutai" in second["answer"] and "市盈率" not in second["answer"]
    assert "24.6" in second["answer"]


def test_an_acronym_only_follow_up_keeps_the_chinese_session_language(offline_service):
    from query_intelligence.agent.llm import final_turn

    llm, runtime, service = _llm_service(
        offline_service,
        [
            final_turn({"answer": "五粮液 ROE 为 29.4% [fundamental_000858.SZ]。", "evidence_used": []}),
            final_turn({"answer": "五粮液市盈率 PE(TTM) 为 20.9 倍 [fundamental_000858.SZ]。", "evidence_used": []}),
        ],
    )
    try:
        service.chat("五粮液的ROE是多少", session_id="r12-h7-zh", mode="agent")
        second = service.chat("PE?", session_id="r12-h7-zh", mode="agent")
    finally:
        runtime.close()
    assert "session_language:zh" in second["route_reasons"]
    assert "Answer language: Chinese" in _user_messages(llm.requests[-1])
    assert "language_mismatch_fallback_to_template" not in second["compliance_notes"]
    assert second["answer"].startswith("五粮液市盈率")


# ---- recorded environment switches (round-8 review, Context to-100 item 2) ----


def test_env_toggles_keep_switches_and_drop_secrets():
    from evaluation.agent_eval.runner import command_with_env, env_toggles

    env = env_toggles(
        {
            "QI_AGENT_PREFETCH": "1",
            "QI_PROMPT_VERSION": "v4",
            "DEEPSEEK_MODEL": "cline-pass/deepseek-v4.1-flash",
            "DEEPSEEK_API_KEY": "sk-secret",
            "QI_API_KEYS": "k1,k2",
            "QI_POSTGRES_URL": "postgresql://u:p@h/db",
            "QI_LLM_TOKEN_BUDGET": "x",
            "HOME": "/Users/x",
        }
    )
    assert env == {
        "DEEPSEEK_MODEL": "cline-pass/deepseek-v4.1-flash",
        "QI_AGENT_PREFETCH": "1",
        "QI_PROMPT_VERSION": "v4",
    }
    assert command_with_env("python -m m", {"QI_A": "1", "QI_B": "a b"}) == "QI_A=1 QI_B='a b' python -m m"


def test_an_ablation_records_its_env_switches(tmp_path, monkeypatch):
    import json

    from evaluation.agent_eval import ablation

    monkeypatch.setenv("QI_AGENT_FRAME_FALLBACK", "off")
    monkeypatch.setenv("QI_FAKE_SECRET_KEY", "never-recorded")
    out = tmp_path / "ablation.json"
    ablation.main(["--sets", "dev", "--limit", "1", "--out", str(out)])
    config = json.loads(out.read_text(encoding="utf-8"))["config"]
    assert config["env"]["QI_AGENT_FRAME_FALLBACK"] == "off"
    assert "QI_FAKE_SECRET_KEY" not in config["env"] and "never-recorded" not in json.dumps(config)
    assert config["command"].startswith("python -m evaluation.agent_eval.ablation --sets dev")
    assert "QI_AGENT_FRAME_FALLBACK=off" in config["command_with_env"]
    assert config["command_with_env"].endswith(config["command"])
