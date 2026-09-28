"""LLM clients for the agent layer.

* ``DeepSeekToolClient`` talks to any OpenAI-compatible Chat Completions endpoint (DeepSeek by
  default) with tool calling. It does not use DeepSeek's Beta ``strict`` mode; tool arguments are
  validated by each tool's Pydantic model instead. In thinking mode every assistant
  ``reasoning_content`` must be sent back on later requests, so ``AssistantTurn.as_message``
  preserves it.
* ``ScriptedLLM`` replays scripted turns for tests and offline evaluation.

Token usage is always recorded. Cost comes from a configured ``Pricing`` or, when the endpoint is a
gateway that bills per request (OpenRouter-style ``usage.cost``, e.g. the Cline API), from the
provider-reported cost. Nothing here hard-codes a price, because provider prices change over time
(and DeepSeek has peak/off-peak rates).

Some gateways wrap the OpenAI-compatible body in an envelope (``{"success": true, "data": {...}}``);
``unwrap_completion`` accepts both shapes.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

_CALL_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar("llm_call_deadline", default=None)
_MIN_CALL_SECONDS = 2.0


class llm_deadline:
    """Bound every LLM request made inside the block by an absolute wall-clock deadline (``time.time()``).

    Each HTTP request (including retries and failover models) gets ``min(timeout_s, time left)``; with less
    than two seconds left the call fails fast with a non-retryable ``LLMError`` so the graph falls back to
    its deterministic path instead of waiting for a slow model past the run's deadline.
    """

    def __init__(self, deadline: float | None) -> None:
        self.deadline = deadline
        self._token: contextvars.Token[float | None] | None = None

    def __enter__(self) -> llm_deadline:
        self._token = _CALL_DEADLINE.set(self.deadline)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._token is not None:
            _CALL_DEADLINE.reset(self._token)


def call_timeout(default_s: float) -> float:
    """Timeout for the next LLM request: the client's timeout, capped by the active deadline."""
    deadline = _CALL_DEADLINE.get()
    if deadline is None:
        return default_s
    remaining = deadline - time.time()
    if remaining < _MIN_CALL_SECONDS:
        raise LLMError("run deadline reached; LLM call skipped", retryable=False)
    return min(default_s, remaining)


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
    reported_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_cache_hit_tokens=self.prompt_cache_hit_tokens + other.prompt_cache_hit_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            reported_cost_usd=round(self.reported_cost_usd + other.reported_cost_usd, 8),
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
        reasoning: str | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> AssistantTurn: ...


REASONING_LEVELS = ("off", "low", "medium", "high")


@dataclass(frozen=True)
class ModelCapabilities:
    """What a model accepts on an OpenAI-compatible endpoint (verified against the provider, see docs)."""

    tool_choice_none: bool = False
    reasoning_off: bool = False


# Matched by substring of the model id. Unknown models get the conservative defaults above.
_CAPABILITIES: tuple[tuple[str, ModelCapabilities], ...] = (
    ("deepseek", ModelCapabilities(tool_choice_none=True, reasoning_off=True)),
    ("glm", ModelCapabilities(tool_choice_none=False, reasoning_off=False)),
)


def model_capabilities(model: str) -> ModelCapabilities:
    lowered = model.lower()
    for needle, capabilities in _CAPABILITIES:
        if needle in lowered:
            return capabilities
    return ModelCapabilities()


def reasoning_style_for(base_url: str, configured: str = "auto") -> str:
    """``deepseek`` (``thinking`` + ``reasoning_effort``), ``openrouter`` (``reasoning`` object) or ``none``."""
    style = configured.strip().lower() or "auto"
    if style != "auto":
        return style
    url = base_url.lower()
    if "deepseek.com" in url:
        return "deepseek"
    if "openrouter.ai" in url or "cline.bot" in url:
        return "openrouter"
    return "none"


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


def resolve_cost(usage: Usage, pricing: Pricing | None) -> tuple[float | None, str | None, str | None]:
    """Return ``(cost, currency, source)`` for a run.

    A configured ``Pricing`` wins. Otherwise the gateway-reported USD cost is used, converted to CNY
    when ``QI_LLM_USD_CNY`` (exchange rate) is set. Returns ``(None, None, None)`` when neither is known.
    """
    if pricing is not None:
        return pricing.cost(usage), pricing.currency, "price_table"
    if usage.reported_cost_usd <= 0:
        return None, None, None
    rate = os.getenv("QI_LLM_USD_CNY", "").strip()
    try:
        fx = float(rate) if rate else 0.0
    except ValueError:
        fx = 0.0
    if fx > 0:
        return round(usage.reported_cost_usd * fx, 6), "CNY", "provider_reported"
    return round(usage.reported_cost_usd, 6), "USD", "provider_reported"


def unwrap_completion(data: Any) -> Any:
    """Return the OpenAI-compatible completion body, unwrapping a ``{"data": {...}}`` envelope."""
    if isinstance(data, dict) and not data.get("choices"):
        inner = data.get("data")
        if isinstance(inner, dict) and inner.get("choices"):
            return inner
    return data


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
        reasoning_style: str = "auto",
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
        self.reasoning_style = reasoning_style_for(base_url, reasoning_style)
        self.capabilities = model_capabilities(model)
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._http_client = http_client
        self._sleep = sleep
        self._stats: Counter[str] = Counter()
        self._stats_lock = threading.Lock()

    def http_stats(self) -> dict[str, int]:
        """Counters since start: ``requests`` (HTTP attempts), ``retries``, ``http_<status>`` for error
        responses (e.g. ``http_429``, including ones that a retry later recovered), ``timeout``, ``transport``."""
        with self._stats_lock:
            return dict(self._stats)

    def _count(self, key: str) -> None:
        with self._stats_lock:
            self._stats[key] += 1

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
            "reasoning_style": str(section.get("reasoning_style") or "auto"),
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
        reasoning: str | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """``reasoning`` overrides the configured thinking level for this call (``off``/``low``/...).

        With ``on_delta`` the request is streamed (server-sent events) and each content fragment is passed
        to it as it arrives; the returned turn is the same as without streaming."""
        if not self.configured:
            raise LLMError("LLM API key is not configured")
        body: dict[str, Any] = {"model": self.model, "messages": list(messages)}
        if tools:
            body["tools"] = list(tools)
            choice = tool_choice or "auto"
            if choice == "none" and not self.capabilities.tool_choice_none:
                # e.g. GLM returns an empty completion for tool_choice="none": drop the tools instead.
                body.pop("tools")
            else:
                body["tool_choice"] = choice
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        thinking_on = self._apply_reasoning(body, reasoning)
        limit = max_tokens if max_tokens is not None else self.max_tokens
        if limit is not None:
            body["max_tokens"] = limit
        if not thinking_on:
            body["temperature"] = self.temperature

        started = time.perf_counter()
        data = self._post_with_retries(body, on_delta)
        return _parse_completion(data, model=self.model, latency_ms=(time.perf_counter() - started) * 1000)

    def _apply_reasoning(self, body: dict[str, Any], reasoning: str | None) -> bool:
        """Write the provider's reasoning parameters; return whether DeepSeek thinking mode is on."""
        level = (reasoning or "").strip().lower() or None
        if level is not None and level not in REASONING_LEVELS:
            raise ValueError(f"reasoning must be one of {REASONING_LEVELS}, got {reasoning!r}")
        if level == "off" and not self.capabilities.reasoning_off:
            level = "low"  # e.g. GLM: reasoning is mandatory on this endpoint
        if self.reasoning_style == "deepseek":
            if level == "off":
                body["thinking"] = {"type": "disabled"}
                return False
            if level:
                body["thinking"] = {"type": "enabled"}
                body["reasoning_effort"] = level
                return True
            if self.thinking_type:
                body["thinking"] = {"type": self.thinking_type}
            if self.reasoning_effort and self.thinking_type != "disabled":
                body["reasoning_effort"] = self.reasoning_effort
            return self.thinking_type == "enabled"
        if self.reasoning_style == "openrouter":
            effort = level or self.reasoning_effort or None
            if effort == "off":
                body["reasoning"] = {"enabled": False}
            elif effort:
                body["reasoning"] = {"effort": effort}
        return False

    def _post_with_retries(self, body: dict[str, Any], on_delta: Callable[[str], None] | None = None) -> dict[str, Any]:
        attempt = 0
        emitted = [False]

        def forward(piece: str) -> None:
            emitted[0] = True
            on_delta(piece)  # type: ignore[misc]

        while True:
            attempt += 1
            self._count("requests")
            try:
                return self._post_stream(body, forward) if on_delta is not None else self._post(body)
            except LLMError as exc:
                if exc.status_code is not None:
                    self._count(f"http_{exc.status_code}")
                elif "timed out" in str(exc):
                    self._count("timeout")
                elif "transport error" in str(exc):
                    self._count("transport")
                # A stream that already produced text cannot be replayed without duplicating it.
                if not exc.retryable or attempt > self.max_retries or emitted[0]:
                    raise
                delay = self.retry_backoff_s * (2 ** (attempt - 1))
                deadline = _CALL_DEADLINE.get()
                if deadline is not None and time.time() + delay + _MIN_CALL_SECONDS > deadline:
                    raise
                self._count("retries")
                self._sleep(delay)

    def _post_stream(self, body: dict[str, Any], on_delta: Callable[[str], None]) -> dict[str, Any]:
        """Stream a completion and rebuild the non-streamed response shape (content, tool calls, usage)."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {**body, "stream": True, "stream_options": {"include_usage": True}}
        client = self._http_client or httpx.Client(timeout=self.timeout_s)
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        usage: dict[str, Any] = {}
        model = self.model
        try:
            with client.stream(
                "POST", self.url, headers=headers, json=payload, timeout=call_timeout(self.timeout_s)
            ) as response:
                if response.status_code >= 400:
                    text = response.read().decode("utf-8", "replace")
                    retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
                    raise LLMError(
                        f"LLM API returned HTTP {response.status_code}: {text[:300]}",
                        retryable=retryable,
                        status_code=response.status_code,
                    )
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = unwrap_completion(json.loads(data))
                    except json.JSONDecodeError:
                        continue
                    model = str(chunk.get("model") or model)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content.append(delta["content"])
                            on_delta(delta["content"])
                        piece = delta.get("reasoning_content") or delta.get("reasoning")
                        if isinstance(piece, str):
                            reasoning.append(piece)
                        for call in delta.get("tool_calls") or []:
                            slot = calls.setdefault(int(call.get("index", 0)), {"id": "", "name": "", "arguments": ""})
                            slot["id"] = call.get("id") or slot["id"]
                            function = call.get("function") or {}
                            slot["name"] += function.get("name") or ""
                            slot["arguments"] += function.get("arguments") or ""
                        finish_reason = choice.get("finish_reason") or finish_reason
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM request timed out: {exc}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"LLM transport error: {exc}", retryable=True) from exc
        finally:
            if self._http_client is None:
                client.close()
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if calls:
            message["tool_calls"] = [
                {
                    "id": slot["id"] or f"call_{index}",
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                }
                for index, slot in sorted(calls.items())
            ]
        return {"model": model, "choices": [{"message": message, "finish_reason": finish_reason}], "usage": usage}

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        client = self._http_client or httpx.Client(timeout=self.timeout_s)
        try:
            response = client.post(self.url, headers=headers, json=body, timeout=call_timeout(self.timeout_s))
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
            data = unwrap_completion(response.json())
        except json.JSONDecodeError as exc:
            raise LLMError("LLM API returned invalid JSON") from exc
        if not isinstance(data, dict) or not data.get("choices"):
            raise LLMError("LLM API response is missing choices")
        return data


class FallbackLLM:
    """Model routing with failover: try clients in order, skipping ones whose circuit is open.

    A client's circuit opens after ``failure_threshold`` consecutive ``LLMError`` failures and stays
    open for ``cooldown_s`` seconds, so a failing provider/model is not retried on every request.
    The returned ``AssistantTurn.model`` names the model that actually answered. When every client
    fails the last error is raised and the graph degrades to the deterministic planner.
    """

    def __init__(
        self,
        clients: Sequence[LLMClient],
        *,
        failure_threshold: int = 3,
        cooldown_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not clients:
            raise ValueError("FallbackLLM needs at least one client")
        self.clients = list(clients)
        self.model = self.clients[0].model
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failures = [0] * len(self.clients)
        self._opened_at: list[float | None] = [None] * len(self.clients)
        self._calls = [0] * len(self.clients)

    def _is_open(self, index: int) -> bool:
        opened = self._opened_at[index]
        if opened is None:
            return False
        if self._clock() - opened >= self.cooldown_s:
            self._opened_at[index] = None  # half-open: allow one trial call
            self._failures[index] = self.failure_threshold - 1
            return False
        return True

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        reasoning: str | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        last_error: LLMError | None = None
        for index, client in enumerate(self.clients):
            if self._is_open(index):
                continue
            self._calls[index] += 1
            try:
                turn = client.chat(
                    messages,
                    tools,
                    tool_choice=tool_choice,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    reasoning=reasoning,
                    on_delta=on_delta,
                )
            except LLMError as exc:
                last_error = exc
                self._failures[index] += 1
                if self._failures[index] >= self.failure_threshold:
                    self._opened_at[index] = self._clock()
                continue
            self._failures[index] = 0
            return turn if turn.model else turn.model_copy(update={"model": client.model})
        raise last_error or LLMError("all LLM clients are unavailable (circuits open)")

    def http_stats(self) -> dict[str, int]:
        """HTTP counters summed over every client (see ``DeepSeekToolClient.http_stats``)."""
        total: Counter[str] = Counter()
        for client in self.clients:
            stats = getattr(client, "http_stats", None)
            if callable(stats):
                total.update(stats())
        return dict(total)

    def stats(self) -> list[dict[str, Any]]:
        return [
            {
                "model": client.model,
                "calls": self._calls[index],
                "consecutive_failures": self._failures[index],
                "circuit_open": self._opened_at[index] is not None,
                "state": self._state(index),
            }
            for index, client in enumerate(self.clients)
        ]

    def _state(self, index: int) -> str:
        """``closed``, ``open``, or ``half_open`` once the cool-down has passed (next call is a trial)."""
        opened = self._opened_at[index]
        if opened is None:
            return "closed"
        return "half_open" if self._clock() - opened >= self.cooldown_s else "open"


def build_llm_from_config(config: dict[str, Any]) -> LLMClient | None:
    """Primary client from the ``deepseek`` config section plus optional failover models.

    ``QI_LLM_FALLBACK_MODELS`` (comma-separated) adds clients for the same endpoint and key with other
    model ids, e.g. ``cline-pass/glm-5.3-flash``. Returns ``None`` when no API key is configured.
    """
    primary = DeepSeekToolClient.from_chatbot_config(config)
    if not primary.configured:
        return None
    fallbacks = [model.strip() for model in os.getenv("QI_LLM_FALLBACK_MODELS", "").split(",") if model.strip()]
    fallbacks = [model for model in dict.fromkeys(fallbacks) if model != primary.model]
    if not fallbacks:
        return primary
    clients: list[LLMClient] = [primary]
    clients += [DeepSeekToolClient.from_chatbot_config(config, model=model) for model in fallbacks]
    return FallbackLLM(clients)


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
    prompt_details = usage_raw.get("prompt_tokens_details") or {}
    usage = Usage(
        prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
        completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        prompt_cache_hit_tokens=int(
            usage_raw.get("prompt_cache_hit_tokens") or prompt_details.get("cached_tokens") or 0
        ),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        reported_cost_usd=_float_or_zero(usage_raw.get("cost")),
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


def _float_or_zero(value: Any) -> float:
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return 0.0


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
        reasoning: str | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        request = {
            "messages": [dict(message) for message in messages],
            "tools": list(tools) if tools else None,
            "tool_choice": tool_choice,
            "json_mode": json_mode,
            "reasoning": reasoning,
        }
        self.requests.append(request)
        if not self.steps:
            raise LLMError("ScriptedLLM has no more scripted turns")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step) and not isinstance(step, AssistantTurn):
            step = step(request["messages"], request["tools"])
        if on_delta is not None and step.content:
            for index in range(0, len(step.content), 8):  # simulate streamed fragments
                on_delta(step.content[index : index + 8])
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
