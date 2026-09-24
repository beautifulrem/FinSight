"""LLM clients for the agent layer.

* ``DeepSeekToolClient`` talks to any OpenAI-compatible Chat Completions endpoint (DeepSeek by
  default) with tool calling. It does not use DeepSeek's Beta ``strict`` mode; tool arguments are
  validated by each tool's Pydantic model instead. In thinking mode every assistant
  ``reasoning_content`` must be sent back on later requests, so ``AssistantTurn.as_message``
  preserves it.
* ``ScriptedLLM`` replays scripted turns for tests and offline evaluation.

Token usage is always recorded. Cost is only computed when a ``Pricing`` is configured, because
provider prices change over time (and DeepSeek has peak/off-peak rates); nothing here hard-codes
a price.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field


class LLMError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str = Field(description="Raw JSON arguments string as produced by the model.")


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_cache_hit_tokens=self.prompt_cache_hit_tokens + other.prompt_cache_hit_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class AssistantTurn(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_content: str | None = None
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0

    def as_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in self.tool_calls
            ]
        if self.reasoning_content:
            message["reasoning_content"] = self.reasoning_content
        return message


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> AssistantTurn: ...


@dataclass(frozen=True)
class Pricing:
    """Price per one million tokens, in ``currency``."""

    input_cache_miss: float
    input_cache_hit: float
    output: float
    currency: str = "CNY"

    def cost(self, usage: Usage) -> float:
        hit = min(usage.prompt_cache_hit_tokens, usage.prompt_tokens)
        miss = usage.prompt_tokens - hit
        total = miss * self.input_cache_miss + hit * self.input_cache_hit + usage.completion_tokens * self.output
        return round(total / 1_000_000, 6)

    @classmethod
    def from_env(cls) -> Pricing | None:
        """Read ``QI_LLM_PRICE_INPUT_MISS``/``_INPUT_HIT``/``_OUTPUT`` (per 1M tokens) and ``QI_LLM_PRICE_CURRENCY``."""
        raw = {
            "input_cache_miss": os.getenv("QI_LLM_PRICE_INPUT_MISS"),
            "input_cache_hit": os.getenv("QI_LLM_PRICE_INPUT_HIT"),
            "output": os.getenv("QI_LLM_PRICE_OUTPUT"),
        }
        if not all(raw.values()):
            return None
        try:
            values = {key: float(value) for key, value in raw.items() if value is not None}
        except ValueError:
            return None
        return cls(currency=os.getenv("QI_LLM_PRICE_CURRENCY", "CNY"), **values)


class DeepSeekToolClient:
    """OpenAI-compatible chat client with tool calling."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        chat_path: str = "/chat/completions",
        timeout_s: float = 60.0,
        thinking_type: str = "",
        reasoning_effort: str = "",
        max_tokens: int | None = 4096,
        temperature: float = 0.2,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.url = f"{base_url.rstrip('/')}/{chat_path.lstrip('/')}"
        self.timeout_s = timeout_s
        self.thinking_type = thinking_type.strip()
        self.reasoning_effort = reasoning_effort.strip()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._http_client = http_client
        self._sleep = sleep

    @classmethod
    def from_chatbot_config(cls, config: dict[str, Any], **overrides: Any) -> DeepSeekToolClient:
        section = config.get("deepseek") or {}
        max_tokens = section.get("max_tokens")
        kwargs: dict[str, Any] = {
            "api_key": str(section.get("api_key") or ""),
            "model": str(section.get("model") or "deepseek-v4-flash"),
            "base_url": str(section.get("base_url") or "https://api.deepseek.com"),
            "chat_path": str(section.get("chat_path") or "/chat/completions"),
            "timeout_s": float(section.get("timeout_seconds") or 60),
            "thinking_type": str(section.get("thinking_type") or ""),
            "reasoning_effort": str(section.get("reasoning_effort") or ""),
            "max_tokens": int(max_tokens) if max_tokens not in {None, ""} else None,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def configured(self) -> bool:
        key = self.api_key.strip()
        return bool(key) and key.lower() not in {"your_deepseek_api_key_here", "changeme"}

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> AssistantTurn:
        if not self.configured:
            raise LLMError("LLM API key is not configured")
        body: dict[str, Any] = {"model": self.model, "messages": list(messages)}
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = tool_choice or "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.thinking_type:
            body["thinking"] = {"type": self.thinking_type}
        if self.reasoning_effort and self.thinking_type != "disabled":
            body["reasoning_effort"] = self.reasoning_effort
        limit = max_tokens if max_tokens is not None else self.max_tokens
        if limit is not None:
            body["max_tokens"] = limit
        if self.thinking_type != "enabled":
            body["temperature"] = self.temperature

        started = time.perf_counter()
        data = self._post_with_retries(body)
        return _parse_completion(data, model=self.model, latency_ms=(time.perf_counter() - started) * 1000)

    def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._post(body)
            except LLMError as exc:
                if not exc.retryable or attempt > self.max_retries:
                    raise
                self._sleep(self.retry_backoff_s * (2 ** (attempt - 1)))

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        client = self._http_client or httpx.Client(timeout=self.timeout_s)
        try:
            response = client.post(self.url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM request timed out: {exc}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"LLM transport error: {exc}", retryable=True) from exc
        finally:
            if self._http_client is None:
                client.close()
        if response.status_code >= 400:
            retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
            raise LLMError(
                f"LLM API returned HTTP {response.status_code}: {response.text[:300]}",
                retryable=retryable,
                status_code=response.status_code,
            )
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError("LLM API returned invalid JSON") from exc
        if not isinstance(data, dict) or not data.get("choices"):
            raise LLMError("LLM API response is missing choices")
        return data


def _parse_completion(data: dict[str, Any], *, model: str, latency_ms: float) -> AssistantTurn:
    choice = data["choices"][0]
    message = choice.get("message") or {}
    tool_calls = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        tool_calls.append(
            ToolCall(
                id=str(raw.get("id") or f"call_{index}"), name=str(function.get("name") or ""), arguments=arguments
            )
        )
    usage_raw = data.get("usage") or {}
    details = usage_raw.get("completion_tokens_details") or {}
    usage = Usage(
        prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
        completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        prompt_cache_hit_tokens=int(usage_raw.get("prompt_cache_hit_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
    )
    return AssistantTurn(
        content=message.get("content"),
        tool_calls=tool_calls,
        reasoning_content=message.get("reasoning_content"),
        finish_reason=choice.get("finish_reason"),
        usage=usage,
        model=str(data.get("model") or model),
        latency_ms=round(latency_ms, 2),
    )


ScriptStep = AssistantTurn | Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], AssistantTurn] | Exception


@dataclass
class ScriptedLLM:
    """Deterministic LLM for tests: returns scripted turns in order and records every request.

    A step may be an ``AssistantTurn``, a callable ``(messages, tools) -> AssistantTurn``, or an
    exception instance to raise.
    """

    steps: list[ScriptStep]
    model: str = "scripted"
    requests: list[dict[str, Any]] = field(default_factory=list)

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> AssistantTurn:
        request = {
            "messages": [dict(message) for message in messages],
            "tools": list(tools) if tools else None,
            "tool_choice": tool_choice,
            "json_mode": json_mode,
        }
        self.requests.append(request)
        if not self.steps:
            raise LLMError("ScriptedLLM has no more scripted turns")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step) and not isinstance(step, AssistantTurn):
            step = step(request["messages"], request["tools"])
        return step.model_copy(update={"model": step.model or self.model})


def tool_call_turn(*calls: tuple[str, dict[str, Any]], content: str = "", usage: Usage | None = None) -> AssistantTurn:
    """Helper for scripts: an assistant turn that requests the given tool calls."""
    return AssistantTurn(
        content=content,
        tool_calls=[
            ToolCall(id=f"call_{index}", name=name, arguments=json.dumps(arguments, ensure_ascii=False))
            for index, (name, arguments) in enumerate(calls)
        ],
        finish_reason="tool_calls",
        usage=usage or Usage(),
    )


def final_turn(content: str | dict[str, Any], usage: Usage | None = None) -> AssistantTurn:
    """Helper for scripts: a final assistant turn (dicts are JSON-encoded)."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return AssistantTurn(content=text, finish_reason="stop", usage=usage or Usage())
