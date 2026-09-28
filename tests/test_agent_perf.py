"""Latency instrumentation and the performance switches of the agent (docs/performance.md)."""

from __future__ import annotations

import json

import httpx
import pytest

from evaluation.agent_eval.profile import profile_records
from evaluation.agent_eval.runner import agent_config_from_overrides, turn_profile
from query_intelligence.agent.llm import DeepSeekToolClient, FallbackLLM
from query_intelligence.agent.verifier import VerificationReport, failure_kinds


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
