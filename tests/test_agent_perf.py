"""Latency instrumentation and the performance switches of the agent (docs/performance.md)."""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest
from agent_fakes import StubService, build_fake_registry

from evaluation.agent_eval.profile import profile_records
from evaluation.agent_eval.runner import agent_config_from_overrides, turn_profile
from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import DeepSeekToolClient, FallbackLLM, ScriptedLLM, final_turn, tool_call_turn
from query_intelligence.agent.state import AgentConfig
from query_intelligence.agent.verifier import (
    VerificationReport,
    cite_repair,
    claim_numbers,
    failure_kinds,
    verify_answer,
)


def _completion() -> dict:
    return {
        "model": "deepseek-v4-flash",
        "choices": [{"message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


def test_http_stats_count_recovered_429s():
    replies = iter([httpx.Response(429, text="slow down"), httpx.Response(200, json=_completion())])
    client = DeepSeekToolClient(
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda _request: next(replies))),
        sleep=lambda _s: None,
    )
    client.chat([{"role": "user", "content": "hi"}])
    assert client.http_stats() == {"requests": 2, "http_429": 1, "retries": 1}
    assert FallbackLLM([client]).http_stats() == {"requests": 2, "http_429": 1, "retries": 1}


def test_failure_kinds_names_failed_checks():
    report = VerificationReport(passed=False, uncited_numbers=[1.5], invalid_citations=["x"])
    assert failure_kinds(report) == ["invalid_citations", "uncited_numbers"]
    assert failure_kinds(report.model_dump()) == ["invalid_citations", "uncited_numbers"]
    assert failure_kinds(VerificationReport(passed=True)) == []


def test_agent_config_overrides_parse_by_field_type():
    config = agent_config_from_overrides(["max_revisions=0", "agent_reasoning=low", "llm_compose=false"])
    assert config.max_revisions == 0 and config.agent_reasoning == "low" and config.llm_compose is False
    assert agent_config_from_overrides(["compose_reasoning=none"]).compose_reasoning is None
    with pytest.raises(SystemExit):
        agent_config_from_overrides(["no_such_field=1"])


def _response(calls: list[dict], tools: list[dict] | None = None) -> dict:
    return {
        "route": "agent",
        "answer_source": "llm_agent",
        "degraded": [],
        "llm": {"log": calls},
        "tool_calls": tools or [],
        "spans": [{"node": "agent_llm", "duration_ms": 1000.0}, {"node": "verify", "duration_ms": 2.0}],
    }


def test_profile_breaks_down_llm_time_by_node():
    first = {"node": "agent_llm", "step": 0, "latency_ms": 1000.0, "prompt_tokens": 100, "tool_calls": ["a", "b"]}
    final = {"node": "agent_llm", "step": 1, "latency_ms": 1500.0, "prompt_tokens": 300, "tool_calls": []}
    revise = {"node": "revise", "latency_ms": 2500.0, "trigger": ["uncited_numbers"]}
    tools = [{"tool": "a", "latency_ms": 5.0, "step": 1, "source": "llm"}]
    records = [
        {
            "turns": [
                {
                    "latency_ms": 5200.0,
                    "ttft_ms": 3000.0,
                    "profile": turn_profile(_response([first, final, revise], tools)),
                }
            ]
        },
        {"turns": [{"latency_ms": 2600.0, "profile": turn_profile(_response([first, final], tools))}]},
        {"turns": [{"latency_ms": 5.0, "profile": turn_profile({"route": "refuse"})}]},
    ]
    profile = profile_records(records)
    assert profile["llm_turns"] == 2
    assert profile["llm_calls_distribution"] == {2: 1, 3: 1}
    assert profile["revise"] == {
        "rate": 0.5,
        "trigger_counts": {"uncited_numbers": 1},
        "trigger_combinations": {"uncited_numbers": 1},
    }
    assert profile["llm_nodes"]["revise"]["share_of_llm_time"] == round(2500 / 7500, 3)
    assert profile["llm_nodes"]["agent_llm[1]"]["prompt_tokens_mean"] == 300
    assert profile["ttft_ms_p50"] == 3000.0
    json.dumps(profile)  # serialisable for the committed result


# --------------------------------------------------------------------------- graph switches


def _runtime(llm, **config) -> AgentRuntime:
    return AgentRuntime(
        StubService(), build_fake_registry(), llm, config=AgentConfig(**config), today=lambda: date(2026, 9, 24)
    )


_UNCITED = {"answer": "茅台最新收盘价为 1409.5 元。PE(TTM) 为 24.6 [fundamental_600519.SH]。", "evidence_used": []}


def test_cite_repair_skips_the_llm_revision_for_uncited_numbers():
    llm = ScriptedLLM(
        [
            tool_call_turn(
                ("get_price_history", {"target": "600519.SH"}), ("get_fundamentals", {"target": "600519.SH"})
            ),
            final_turn(_UNCITED),
        ]
    )
    result = _runtime(llm, revise_policy="cite_repair").run("茅台为什么跌了")

    assert result["llm"]["calls"] == 2  # no revise call
    assert result["verification"]["passed"] is True
    assert "1409.5 元[price_600519.SH]。" in result["answer"]
    assert "verification_failed:citations_repaired:uncited_numbers" in result["degraded"]
    assert {"price_600519.SH", "fundamental_600519.SH"} <= set(result["evidence_used"])


def test_default_policy_still_revises_with_the_llm():
    llm = ScriptedLLM(
        [
            tool_call_turn(
                ("get_price_history", {"target": "600519.SH"}), ("get_fundamentals", {"target": "600519.SH"})
            ),
            final_turn(_UNCITED),
            final_turn({"answer": "茅台最新收盘价为 1409.5 元 [price_600519.SH]。", "evidence_used": []}),
        ]
    )
    result = _runtime(llm).run("茅台为什么跌了")

    assert result["llm"]["calls"] == 3
    assert [entry["node"] for entry in result["llm"]["log"]][-1] == "revise"
    assert result["llm"]["log"][-1]["trigger"] == ["uncited_numbers"]


def test_cite_repair_leaves_unsupported_numbers_to_the_llm():
    invented = {"answer": "茅台最新收盘价为 1409.5 元，目标价 2600 元。", "evidence_used": []}
    llm = ScriptedLLM(
        [
            tool_call_turn(("get_price_history", {"target": "600519.SH"})),
            final_turn(invented),
            final_turn({"answer": "茅台最新收盘价为 1409.5 元 [price_600519.SH]。", "evidence_used": []}),
        ]
    )
    result = _runtime(llm, revise_policy="cite_repair").run("茅台为什么跌了")

    assert result["llm"]["log"][-1]["node"] == "revise"
    assert "2600" not in result["answer"]


def test_planner_prefetch_answers_in_one_llm_call():
    def answer(messages, _tools):
        prefetch = messages[-1]["content"]
        assert messages[-1]["role"] == "user" and '<tool_result name="get_fundamentals">' in prefetch
        assert "fundamental_600519.SH" in prefetch
        return final_turn({"answer": "茅台 PE(TTM) 为 24.6 [fundamental_600519.SH]。", "evidence_used": []})

    llm = ScriptedLLM([answer])
    result = _runtime(llm, planner_prefetch=True).run("贵州茅台的市盈率是多少", mode="agent")

    assert result["llm"]["calls"] == 1
    assert result["verification"]["passed"] is True
    assert [(call["tool"], call["source"]) for call in result["tool_calls"]] == [("get_fundamentals", "prefetch")]
    assert [span["node"] for span in result["spans"]][:3] == ["guard_in", "agent_prefetch", "agent_llm"]


def test_prefetched_calls_are_not_repeated():
    llm = ScriptedLLM(
        [
            tool_call_turn(("get_fundamentals", {"target": "600519.SH"})),
            final_turn({"answer": "茅台 PE(TTM) 为 24.6 [fundamental_600519.SH]。", "evidence_used": []}),
        ]
    )
    result = _runtime(llm, planner_prefetch=True).run("贵州茅台的市盈率是多少", mode="agent")

    assert "repeated_tool_calls:1" in result["degraded"]
    assert [call["source"] for call in result["tool_calls"]] == ["prefetch"]


# --------------------------------------------------------------------------- cite_repair / keep-alive


def _store() -> EvidenceStore:
    store = EvidenceStore()
    store.add(
        AgentEvidence(
            evidence_id="price_600519.SH", kind="structured", source_type="market_api", payload={"close": 1409.5}
        )
    )
    store.add(
        AgentEvidence(
            evidence_id="fundamental_600519.SH",
            kind="structured",
            source_type="fundamental_sql",
            payload={"pe_ttm": 24.6, "roe": 0.33},
        )
    )
    store.add(
        AgentEvidence(
            evidence_id="fundamental_000858.SZ",
            kind="structured",
            source_type="fundamental_sql",
            payload={"pe_ttm": 24.6, "roe": 0.24},
        )
    )
    return store


def _repair(answer: dict) -> dict | None:
    store = _store()
    report = verify_answer(answer, store)
    return cite_repair(answer, report, store)


def test_cite_repair_moves_a_misattributed_number_to_its_evidence():
    answer = {"answer": "茅台 ROE 为 33%，收盘价 1409.5 元 [price_600519.SH]。"}
    repaired = _repair(answer)
    assert repaired is not None
    assert repaired["answer"] == "茅台 ROE 为 33%，收盘价 1409.5 元 [price_600519.SH][fundamental_600519.SH]。"
    assert verify_answer(repaired, _store()).passed


def test_cite_repair_does_not_guess_between_items_with_the_same_value():
    # 24.6 is the PE of both companies: citing either would be a guess.
    assert _repair({"answer": "市盈率为 24.6 倍。"}) is None


def test_cite_repair_drops_invalid_ids_and_cites_key_points():
    answer = {"answer": "收盘价 1409.5 元 [price_x]。", "key_points": ["ROE 33%"], "evidence_used": ["price_x"]}
    repaired = _repair(answer)
    assert repaired is not None
    assert repaired["answer"] == "收盘价 1409.5 元[price_600519.SH]。"
    assert repaired["key_points"] == ["ROE 33%[fundamental_600519.SH]"]
    assert repaired["evidence_used"] == ["price_600519.SH", "fundamental_600519.SH"]
    assert verify_answer(repaired, _store()).passed


def test_cite_repair_refuses_unsupported_numbers():
    assert _repair({"answer": "收盘价 1409.5 元，目标价 2600 元。"}) is None


def test_llm_client_reuses_one_pooled_connection_client():
    pooled = DeepSeekToolClient(api_key="sk-test", keepalive=True)
    first, owned = pooled._client()
    assert owned is False and pooled._client()[0] is first
    pooled.close()
    one_off = DeepSeekToolClient(api_key="sk-test", keepalive=False)
    client, owned = one_off._client()
    assert owned is True
    client.close()


# --------------------------------------------------------------------------- verifier false positives (profile)


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("PE(TTM) 20.9x, PB 5.4x", [20.9, 5.4]),  # was [20, 5]: the token backtracked before the "x"
        ("revenue CNY 108.5bn, net profit CNY 37.8bn", [108.5, 37.8]),  # was [108, 37]
        ("4.746（04-16）、4.739（04-17）", [4.746, 4.739]),  # month-day dates are not claims
        ("中国10年期国债收益率 2.31%", [2.31]),  # bond tenor
        ("3) 板块情绪与资金面。", []),  # list marker
        ("增长 10-15%", [10.0, 15.0]),  # a range is still checked
        ("version v2x 3.5.2", []),
    ],
)
def test_claim_numbers_without_false_positives(text, numbers):
    assert claim_numbers(text) == numbers


def test_english_multiples_verify_against_fundamentals():
    answer = {"answer": "Moutai trades at 24.6x trailing earnings [fundamental_600519.SH]."}
    assert verify_answer(answer, _store()).passed


def test_derived_numbers_are_opt_in_and_need_their_operands():
    store = _store()
    derived = {
        "answer": "茅台 PE 24.6 倍，收盘价 1409.5 元，两者之比约 57.3 [fundamental_600519.SH][price_600519.SH]。"
    }
    assert not verify_answer(derived, store).passed
    assert verify_answer(derived, store, allow_derived=True).passed
    # without both operands in the sentence the result is not accepted
    alone = {"answer": "两者之比约 57.3 [fundamental_600519.SH][price_600519.SH]。"}
    assert not verify_answer(alone, store, allow_derived=True).passed


def test_stall_timeout_caps_streamed_requests_only():
    import time as _time

    from query_intelligence.agent.llm import call_timeout, llm_deadline

    with llm_deadline(_time.time() + 90, stall_s=20):
        assert call_timeout(120, streaming=True) == 20
        assert 85 < call_timeout(120) <= 90
    with llm_deadline(_time.time() + 90):
        assert 85 < call_timeout(120, streaming=True) <= 90


def test_stalled_stream_is_retried():
    calls = {"n": 0}
    body = 'data: {"choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("stalled", request=request)
        return httpx.Response(200, text=body)

    client = DeepSeekToolClient(
        api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _s: None
    )
    turn = client.chat([{"role": "user", "content": "hi"}], on_delta=lambda _text: None)
    assert turn.content == "{}"
    assert client.http_stats() == {"requests": 2, "timeout": 1, "retries": 1}


def test_defaults_are_the_measured_latency_configuration(monkeypatch):
    for name in (
        "QI_AGENT_PREFETCH",
        "QI_AGENT_REVISE_POLICY",
        "QI_AGENT_LLM_STALL_TIMEOUT_S",
        "QI_AGENT_VERIFY_DERIVED",
    ):
        monkeypatch.delenv(name, raising=False)
    config = AgentConfig()
    assert config.planner_prefetch is True
    assert config.revise_policy == "cite_repair"
    assert config.llm_stall_timeout_s == 20.0
    assert config.verify_derived is True
    monkeypatch.setenv("QI_AGENT_PREFETCH", "0")
    assert AgentConfig().planner_prefetch is False
