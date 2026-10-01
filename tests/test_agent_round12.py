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
    # restated after the (mostly Chinese) tool results, the last thing the model reads
    assert prompt.rstrip().endswith("Reminder: write the final answer, key points and limitations in English.")
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
    assert "Reminder: write the final answer" not in _user_messages(llm.requests[-1])
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


# ---- H8: model-derived numbers are re-derived from their operands, not deleted ----
#
# Drafts recorded online with DeepSeek before the fix (tests/fixtures/h8_recorded_drafts.json, written by
# ``python -m evaluation.agent_eval.session_llm_check --set h8``): the ratio 1.93 / 0.52 of two cited ROEs, a gap
# 14.2 whose second operand had no citation, and a relative 17.7% derived across a "；".

_DRAFTS_PATH = "tests/fixtures/h8_recorded_drafts.json"


def _recorded(session: str, turn: int, call: int = -1) -> dict:
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).resolve().parents[1] / _DRAFTS_PATH).read_text(encoding="utf-8"))
    turns = next(item for item in data["sessions"] if item["id"] == session)["turns"]
    return json.loads(turns[turn]["llm_calls"][call]["content"])


@pytest.fixture(scope="module")
def fundamentals_store(offline_service):
    from query_intelligence.agent.evidence import EvidenceStore
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    registry = build_registry_for_service(offline_service)
    store = EvidenceStore()
    for target in ("600519.SH", "000858.SZ", "601318.SH"):
        for item in registry.run("get_fundamentals", {"target": target}).evidence:
            store.add(item)
    return store


def test_a_recorded_ratio_of_two_cited_roes_passes_in_derived_mode(fundamentals_store):
    from query_intelligence.agent.verifier import verify_answer

    # "…二者之比约为 1.93（29.4 ÷ 15.2 ≈ 1.93），即五粮液 ROE 约为中国平安的 1.93 倍；…0.52 倍"
    draft = {"answer": _recorded("l3-ratio", 2, 0)["answer"]}
    assert "1.93" in draft["answer"] and "0.52" in draft["answer"]
    assert not verify_answer(draft, fundamentals_store, allow_derived=False).passed
    # before the fix the derived mode reported both as ROE figures contradicting the fundamentals
    # (document_market_numbers [1.93, 0.52]: "ROE 约为中国平安的 1.93 倍" read as an ROE of 1.93)
    after = verify_answer(draft, fundamentals_store, allow_derived=True)
    assert after.passed, after


def test_a_recorded_gap_with_an_uncited_operand_is_cite_repaired(fundamentals_store):
    from query_intelligence.agent.verifier import cite_repair, verify_answer

    # "…五粮液比中国平安高 14.2 个百分点（29.4 − 15.2）。" (29.4 uncited there)
    draft = {"answer": _recorded("l3-ratio", 1, 1)["answer"]}
    report = verify_answer(draft, fundamentals_store, allow_derived=True)
    assert not report.passed and 14.2 in report.unsupported_numbers
    assert cite_repair(draft, report, fundamentals_store, allow_derived=False) is None  # the old behaviour
    fixed = cite_repair(draft, report, fundamentals_store, allow_derived=True)
    assert fixed is not None and "高 14.2 个百分点（29.4 − 15.2）[fundamental_000858.SZ]" in fixed["answer"]
    assert verify_answer(fixed, fundamentals_store, allow_derived=True).passed


def test_a_recorded_relative_difference_across_a_semicolon_passes(fundamentals_store):
    from query_intelligence.agent.verifier import verify_answer

    draft = {"answer": _recorded("l2b-relative", 2, 0)["answer"]}  # "(24.6 − 20.9 = 3.7; 3.7 ÷ 20.9 ≈ 17.7%) [a][b]"
    assert "17.7%" in draft["answer"]
    report = verify_answer(draft, fundamentals_store, allow_derived=True)
    assert report.passed, report


def test_an_uncited_percent_formula_gets_its_operands_cited(fundamentals_store):
    """The round-8 reviewer's L2b shape: one uncited sentence with both operands, the gap, "× 100" and the result."""
    from query_intelligence.agent.verifier import cite_repair, verify_answer

    draft = {
        "answer": "Moutai's P/E (TTM) is 24.6x and Wuliangye's is 20.9x, so Moutai's is about 17.7% higher "
        "((24.6 − 20.9) / 20.9 × 100 = 17.7%)."
    }
    report = verify_answer(draft, fundamentals_store, allow_derived=True)
    assert not report.passed and 17.7 in report.unsupported_numbers and 100.0 not in report.unsupported_numbers
    fixed = cite_repair(draft, report, fundamentals_store, allow_derived=True)
    assert fixed is not None
    assert fixed["answer"].endswith("= 17.7%)[fundamental_600519.SH][fundamental_000858.SZ].")
    assert verify_answer(fixed, fundamentals_store, allow_derived=True).passed


@pytest.mark.parametrize(
    "answer",
    [
        # a gap that the operands do not reproduce (29.4 − 15.2 = 14.2)
        "五粮液 ROE 29.4%，中国平安 15.2%，五粮液高 12.5 个百分点。",
        # a ratio that the operands do not reproduce (29.4 ÷ 15.2 = 1.93)
        "五粮液 ROE 29.4%，中国平安 15.2%，前者约为后者的 2.10 倍。",
        # a relative difference off by more than rounding (17.7%)
        "Moutai's P/E is 24.6x and Wuliangye's 20.9x, so Moutai's is 19.4% higher.",
    ],
)
def test_a_derived_number_no_cited_operands_reproduce_is_still_rejected(fundamentals_store, answer):
    from query_intelligence.agent.verifier import cite_repair, repair_answer, verify_answer

    report = verify_answer({"answer": answer}, fundamentals_store, allow_derived=True)
    assert not report.passed
    assert cite_repair({"answer": answer}, report, fundamentals_store, allow_derived=True) is None
    repaired, _notes = repair_answer({"answer": answer}, report, fundamentals_store, zh="五" in answer)
    assert answer not in repaired["answer"]


def test_a_wrong_roe_next_to_a_comparison_word_is_still_a_metric_conflict(fundamentals_store):
    from query_intelligence.agent.verifier import verify_answer

    # 14.2 is a gap of the two ROEs, but "ROE 为 14.2%" states it as Ping An's ROE (15.2%) with no second operand
    answer = "中国平安 ROE 为 14.2%，低于五粮液 [fundamental_601318.SH][fundamental_000858.SZ]。"
    assert not verify_answer({"answer": answer}, fundamentals_store, allow_derived=True).passed


def test_recorded_drafts_keep_their_derived_numbers_end_to_end(offline_service):
    """The L3 session replayed with its recorded first drafts: the gap turn is cite-repaired (no revision call) and
    the ratio turn keeps 1.93 and 0.52."""
    from query_intelligence.agent.llm import final_turn, tool_call_turn

    steps = [
        final_turn(_recorded("l3-ratio", 0, 0)),
        tool_call_turn(("get_fundamentals", {"target": "601318.SH"})),
        final_turn(_recorded("l3-ratio", 1, 1)),
        final_turn(_recorded("l3-ratio", 2, 0)),
    ]
    llm, runtime, service = _llm_service(offline_service, steps)
    try:
        answers = [
            service.chat(query, session_id="r12-h8-l3", mode="agent")
            for query in ("中国平安ROE多少", "五粮液呢", "二者之比是多少")
        ]
    finally:
        runtime.close()
    gap, ratio = answers[1], answers[2]
    assert "14.2 个百分点" in gap["answer"] and gap["verification"]["passed"]
    assert any(item.startswith("verification_failed:citations_repaired") for item in gap["degraded"])
    assert "1.93" in ratio["answer"] and "0.52" in ratio["answer"]
    assert ratio["verification"]["passed"] and not ratio["degraded"]
    assert len(llm.requests) == 4  # no revision round trip


# ---- G4 measurement switch (QI_AGENT_FRAME_FALLBACK) ----


@pytest.mark.parametrize("fallback", [True, False])
def test_the_frame_fallback_switch(offline_service, fallback):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM, final_turn, tool_call_turn
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.state import AgentConfig
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    llm = ScriptedLLM(
        [
            tool_call_turn(("get_fundamentals", {"target": "000858.SZ"})),
            final_turn({"answer": "五粮液 ROE 为 29.4% [fundamental_000858.SZ]。", "evidence_used": []}),
            tool_call_turn(("get_fundamentals", {"target": "600519.SH"})),
            final_turn({"answer": "贵州茅台 ROE 为 33% [fundamental_600519.SH]。", "evidence_used": []}),
            final_turn({"answer": "本轮工具结果中没有五粮液的数据，无法核实两者的差值。", "evidence_used": []}),
        ]
    )
    config = AgentConfig(
        planner_prefetch=False, revise_policy="cite_repair", verify_derived=True, frame_fallback=fallback
    )
    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), llm, config=config)
    service = AgentService(runtime, trace_sinks=[])
    session = f"r12-g4-switch-{fallback}"
    try:
        for query in ("五粮液ROE是多少", "茅台的呢"):
            service.chat(query, session_id=session, mode="agent")
        gap = service.chat("两家差几个点", session_id=session, mode="agent")
    finally:
        runtime.close()
    if fallback:
        assert "frame_result_appended" in gap["degraded"] and "两者相差 3.6 个百分点" in gap["answer"]
    else:
        assert "frame_fallback_off:would_append" in gap["degraded"]
        assert "frame_result_appended" not in gap["degraded"] and "3.6" not in gap["answer"]
