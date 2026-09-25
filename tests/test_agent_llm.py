from __future__ import annotations

import json

import httpx
import pytest

from query_intelligence.agent.llm import (
    AssistantTurn,
    DeepSeekToolClient,
    LLMError,
    FallbackLLM,
    Pricing,
    ScriptedLLM,
    build_llm_from_config,
    Usage,
    final_turn,
    resolve_cost,
    tool_call_turn,
)

TOOLS = [{"type": "function", "function": {"name": "get_quote", "parameters": {"type": "object", "properties": {}}}}]


def _client(handler, **kwargs) -> DeepSeekToolClient:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return DeepSeekToolClient(api_key="sk-test", http_client=http_client, sleep=lambda _s: None, **kwargs)


def _completion(message: dict, usage: dict | None = None) -> dict:
    return {
        "model": "deepseek-v4-flash",
        "choices": [{"message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 60},
    }


def test_tool_call_request_and_response_parsing():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["authorization"]
        return httpx.Response(
            200,
            json=_completion(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "need a quote",
                    "tool_calls": [
                        {"id": "call_a", "type": "function", "function": {"name": "get_quote", "arguments": '{"t":1}'}}
                    ],
                },
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_cache_hit_tokens": 60,
                    "completion_tokens_details": {"reasoning_tokens": 12},
                },
            ),
        )

    client = _client(handler, thinking_type="enabled", reasoning_effort="high")
    turn = client.chat([{"role": "user", "content": "hi"}], TOOLS)

    body = captured["body"]
    assert captured["auth"] == "Bearer sk-test"
    assert body["tools"] == TOOLS and body["tool_choice"] == "auto"
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
    assert "temperature" not in body and "strict" not in json.dumps(body)
    assert turn.tool_calls[0].name == "get_quote" and turn.tool_calls[0].arguments == '{"t":1}'
    assert turn.usage.prompt_cache_hit_tokens == 60 and turn.usage.reasoning_tokens == 12
    # Thinking mode requires reasoning_content to be passed back on later requests.
    assert turn.as_message()["reasoning_content"] == "need a quote"
    assert turn.as_message()["tool_calls"][0]["function"]["name"] == "get_quote"


def test_json_mode_and_temperature_without_thinking():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion({"role": "assistant", "content": "{}"}))

    turn = _client(handler).chat([{"role": "user", "content": "hi"}], json_mode=True, max_tokens=50)

    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["temperature"] == 0.2 and captured["body"]["max_tokens"] == 50
    assert "tools" not in captured["body"]
    assert turn.content == "{}" and turn.tool_calls == []


def test_retries_on_rate_limit_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=_completion({"role": "assistant", "content": "ok"}))

    assert _client(handler).chat([{"role": "user", "content": "hi"}]).content == "ok"
    assert calls["count"] == 2


def test_client_errors_are_not_retried():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(LLMError) as info:
        _client(handler).chat([{"role": "user", "content": "hi"}])
    assert info.value.status_code == 400 and not info.value.retryable
    assert calls["count"] == 1


def test_timeouts_are_retryable_and_eventually_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError) as info:
        _client(handler, max_retries=1).chat([{"role": "user", "content": "hi"}])
    assert info.value.retryable


def test_missing_api_key_raises_before_any_request():
    client = DeepSeekToolClient(api_key="your_deepseek_api_key_here")
    assert not client.configured
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])


def test_from_chatbot_config_reads_deepseek_section():
    client = DeepSeekToolClient.from_chatbot_config(
        {
            "deepseek": {
                "api_key": "k",
                "model": "m",
                "base_url": "https://x/",
                "chat_path": "/v1/chat",
                "max_tokens": "",
            }
        }
    )
    assert client.model == "m" and client.url == "https://x/v1/chat" and client.max_tokens is None


def test_pricing_cost_and_env(monkeypatch):
    pricing = Pricing(input_cache_miss=2.0, input_cache_hit=0.5, output=8.0)
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000, prompt_cache_hit_tokens=400_000)

    assert pricing.cost(usage) == pytest.approx(0.6 * 2.0 + 0.4 * 0.5 + 0.5 * 8.0)

    monkeypatch.delenv("QI_LLM_PRICE_INPUT_MISS", raising=False)
    assert Pricing.from_env() is None
    monkeypatch.setenv("QI_LLM_PRICE_INPUT_MISS", "1")
    monkeypatch.setenv("QI_LLM_PRICE_INPUT_HIT", "0.1")
    monkeypatch.setenv("QI_LLM_PRICE_OUTPUT", "2")
    assert Pricing.from_env() == Pricing(input_cache_miss=1.0, input_cache_hit=0.1, output=2.0)


def test_gateway_envelope_and_reported_cost_are_parsed():
    body = _completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_quote", "arguments": "{}"}}],
        },
        usage={
            "prompt_tokens": 291,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 128},
            "completion_tokens_details": {"reasoning_tokens": 12},
            "cost": 0.0001965,
        },
    )
    client = _client(lambda request: httpx.Response(200, json={"success": True, "data": body}))

    turn = client.chat([{"role": "user", "content": "hi"}], TOOLS)

    assert [call.name for call in turn.tool_calls] == ["get_quote"]
    assert turn.usage.prompt_cache_hit_tokens == 128
    assert turn.usage.reasoning_tokens == 12
    assert turn.usage.reported_cost_usd == pytest.approx(0.0001965)


def test_resolve_cost_prefers_price_table_then_reported_cost(monkeypatch):
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=0, reported_cost_usd=0.5)
    monkeypatch.delenv("QI_LLM_USD_CNY", raising=False)

    assert resolve_cost(usage, Pricing(input_cache_miss=2.0, input_cache_hit=0.5, output=8.0)) == (
        2.0,
        "CNY",
        "price_table",
    )
    assert resolve_cost(usage, None) == (0.5, "USD", "provider_reported")
    monkeypatch.setenv("QI_LLM_USD_CNY", "7.1")
    assert resolve_cost(usage, None) == (pytest.approx(3.55), "CNY", "provider_reported")
    assert resolve_cost(Usage(), None) == (None, None, None)


def test_usage_addition():
    total = Usage(prompt_tokens=1, completion_tokens=2) + Usage(prompt_tokens=3, prompt_cache_hit_tokens=1)
    assert total.prompt_tokens == 4 and total.completion_tokens == 2 and total.total_tokens == 6


def test_scripted_llm_replays_steps_and_records_requests():
    script = ScriptedLLM(
        [
            tool_call_turn(("get_quote", {"target": "600519.SH"})),
            lambda messages, tools: final_turn({"answer": f"{len(messages)} messages"}),
            LLMError("boom"),
        ]
    )

    first = script.chat([{"role": "user", "content": "q"}], TOOLS)
    second = script.chat([{"role": "user", "content": "q"}, first.as_message()], TOOLS)

    assert first.tool_calls[0].name == "get_quote"
    assert json.loads(first.tool_calls[0].arguments) == {"target": "600519.SH"}
    assert json.loads(second.content) == {"answer": "2 messages"}
    assert script.requests[1]["messages"][1]["tool_calls"][0]["id"] == "call_0"
    with pytest.raises(LLMError):
        script.chat([], None)
    with pytest.raises(LLMError):
        script.chat([], None)


def test_assistant_turn_message_without_tools_is_plain():
    assert AssistantTurn(content="done").as_message() == {"role": "assistant", "content": "done"}


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_fallback_llm_fails_over_and_opens_circuit():
    clock = _Clock()
    broken = ScriptedLLM([LLMError("down", retryable=True)] * 5, model="primary")
    backup = ScriptedLLM([final_turn("ok")] * 5, model="backup")
    llm = FallbackLLM([broken, backup], failure_threshold=2, cooldown_s=30, clock=clock)

    assert llm.model == "primary"
    assert llm.chat([{"role": "user", "content": "q"}]).model == "backup"
    assert llm.chat([{"role": "user", "content": "q"}]).model == "backup"
    assert llm.stats()[0]["circuit_open"] is True
    calls_before = len(broken.requests)

    llm.chat([{"role": "user", "content": "q"}])
    assert len(broken.requests) == calls_before  # skipped while the circuit is open

    clock.now = 31  # half-open: primary gets one trial call
    llm.chat([{"role": "user", "content": "q"}])
    assert len(broken.requests) == calls_before + 1


def test_fallback_llm_raises_last_error_when_all_fail():
    llm = FallbackLLM(
        [ScriptedLLM([LLMError("a")], model="a"), ScriptedLLM([LLMError("b")], model="b")], failure_threshold=5
    )

    with pytest.raises(LLMError, match="b"):
        llm.chat([{"role": "user", "content": "q"}])


def test_fallback_llm_resets_failures_after_success():
    primary = ScriptedLLM([LLMError("blip"), final_turn("ok"), LLMError("blip")], model="p")
    backup = ScriptedLLM([final_turn("b1"), final_turn("b2")], model="b")
    llm = FallbackLLM([primary, backup], failure_threshold=2)

    assert llm.chat([]).model == "b"
    assert llm.chat([]).model == "p"
    assert llm.chat([]).model == "b"
    assert llm.stats()[0]["circuit_open"] is False


def test_build_llm_from_config_adds_fallback_models(monkeypatch):
    config = {"deepseek": {"api_key": "sk-x", "model": "m1", "base_url": "https://gw.example/api/v1"}}
    monkeypatch.delenv("QI_LLM_FALLBACK_MODELS", raising=False)
    assert isinstance(build_llm_from_config(config), DeepSeekToolClient)
    assert build_llm_from_config({"deepseek": {"api_key": ""}}) is None

    monkeypatch.setenv("QI_LLM_FALLBACK_MODELS", "m2, m1 ,m2")
    llm = build_llm_from_config(config)
    assert isinstance(llm, FallbackLLM)
    assert [client.model for client in llm.clients] == ["m1", "m2"]
    assert llm.clients[1].url == "https://gw.example/api/v1/chat/completions"
