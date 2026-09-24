"""Tool abstraction for the agent layer.

A tool wraps existing Query Intelligence code behind a typed contract:

* ``input_model`` validates arguments coming from an LLM (or the deterministic planner);
* the handler returns a ``ToolOutput`` with JSON-serializable data and evidence items;
* ``ToolRegistry.run`` adds timeouts, bounded retries with exponential backoff,
  TTL caching, and normalizes every failure into a ``ToolError`` so the agent loop
  never sees a raw exception.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from ..evidence import AgentEvidence

ToolErrorCode = Literal[
    "unknown_tool",
    "invalid_arguments",
    "timeout",
    "upstream_error",
    "not_found",
    "unavailable",
    "internal",
]


class ToolError(BaseModel):
    code: ToolErrorCode
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    tool: str
    ok: bool
    arguments: dict[str, Any]
    data: Any = None
    evidence: list[AgentEvidence] = []
    error: ToolError | None = None
    latency_ms: float = 0.0
    attempts: int = 0
    cached: bool = False

    def observation(self) -> dict[str, Any]:
        """Compact view returned to the LLM as the tool message content."""
        if not self.ok:
            assert self.error is not None
            return {"ok": False, "error": self.error.model_dump()}
        return {
            "ok": True,
            "data": self.data,
            "evidence": [item.prompt_view() for item in self.evidence],
        }


@dataclass
class ToolOutput:
    data: Any
    evidence: list[AgentEvidence] = field(default_factory=list)


class ToolFailure(Exception):
    """Raised by handlers for expected failures that should not be retried."""

    def __init__(self, code: ToolErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class TransientToolError(Exception):
    """Raised by handlers for upstream failures that are worth retrying."""


Handler = Callable[[BaseModel], ToolOutput]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Handler
    timeout_s: float = 15.0
    max_retries: int = 1
    retry_backoff_s: float = 0.2
    cache_ttl_s: float = 0.0
    retry_on: tuple[type[BaseException], ...] = (TransientToolError, TimeoutError, ConnectionError)

    def openai_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }


class _TTLCache:
    def __init__(self, max_entries: int = 256) -> None:
        self._data: dict[str, tuple[float, ToolOutput]] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def get(self, key: str) -> ToolOutput | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < time.monotonic():
                self._data.pop(key, None)
                return None
            return value

    def put(self, key: str, value: ToolOutput, ttl_s: float) -> None:
        with self._lock:
            if len(self._data) >= self._max_entries:
                oldest = min(self._data, key=lambda k: self._data[k][0])
                self._data.pop(oldest, None)
            self._data[key] = (time.monotonic() + ttl_s, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class ToolRegistry:
    def __init__(self, *, max_workers: int = 8, sleep: Callable[[float], None] = time.sleep) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent-tool")
        self._cache = _TTLCache()
        self._sleep = sleep

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def to_openai_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        selected = names or self.names()
        return [self._specs[name].openai_schema() for name in selected if name in self._specs]

    def clear_cache(self) -> None:
        self._cache.clear()

    def run(self, name: str, arguments: dict[str, Any] | str | None = None) -> ToolResult:
        started = time.perf_counter()
        spec = self._specs.get(name)
        raw_args = _coerce_arguments(arguments)
        if spec is None:
            return _failure(name, raw_args, "unknown_tool", f"unknown tool: {name}", started, attempts=0)
        if raw_args is None:
            return _failure(name, {}, "invalid_arguments", "arguments must be a JSON object", started, attempts=0)
        try:
            parsed = spec.input_model.model_validate(raw_args)
        except ValidationError as exc:
            return _failure(name, raw_args, "invalid_arguments", _validation_message(exc), started, attempts=0)

        normalized_args = parsed.model_dump(mode="json")
        cache_key = f"{name}:{json.dumps(normalized_args, sort_keys=True, ensure_ascii=False)}"
        if spec.cache_ttl_s > 0:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return _success(name, normalized_args, cached, started, attempts=0, cached=True)

        attempts = 0
        last_error: ToolError | None = None
        while attempts <= spec.max_retries:
            attempts += 1
            future = self._executor.submit(spec.handler, parsed)
            try:
                output = future.result(timeout=spec.timeout_s)
            except FutureTimeout:
                future.cancel()
                last_error = ToolError(code="timeout", message=f"{name} exceeded {spec.timeout_s:g}s", retryable=True)
            except ToolFailure as exc:
                return _failure(name, normalized_args, exc.code, str(exc), started, attempts=attempts)
            except spec.retry_on as exc:
                last_error = ToolError(code="upstream_error", message=_short(exc), retryable=True)
            except Exception as exc:  # normalized into a ToolError for the agent loop
                return _failure(name, normalized_args, "internal", _short(exc), started, attempts=attempts)
            else:
                if not isinstance(output, ToolOutput):
                    output = ToolOutput(data=output)
                if spec.cache_ttl_s > 0:
                    self._cache.put(cache_key, output, spec.cache_ttl_s)
                return _success(name, normalized_args, output, started, attempts=attempts)
            if attempts <= spec.max_retries:
                self._sleep(spec.retry_backoff_s * (2 ** (attempts - 1)))

        assert last_error is not None
        return ToolResult(
            tool=name,
            ok=False,
            arguments=normalized_args,
            error=last_error,
            latency_ms=_elapsed_ms(started),
            attempts=attempts,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _coerce_arguments(arguments: dict[str, Any] | str | None) -> dict[str, Any] | None:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    text = arguments.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
        except ImportError:  # pragma: no cover - json_repair is a declared dependency
            return None
        value = repair_json(text, return_objects=True)
    return value if isinstance(value, dict) else None


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:5]:
        location = ".".join(str(item) for item in error.get("loc", ())) or "arguments"
        parts.append(f"{location}: {error.get('msg')}")
    return "; ".join(parts)


def _success(
    name: str, arguments: dict[str, Any], output: ToolOutput, started: float, *, attempts: int, cached: bool = False
) -> ToolResult:
    data = output.data.model_dump(mode="json") if isinstance(output.data, BaseModel) else output.data
    return ToolResult(
        tool=name,
        ok=True,
        arguments=arguments,
        data=data,
        evidence=list(output.evidence),
        latency_ms=_elapsed_ms(started),
        attempts=attempts,
        cached=cached,
    )


def _failure(
    name: str, arguments: dict[str, Any], code: ToolErrorCode, message: str, started: float, *, attempts: int
) -> ToolResult:
    return ToolResult(
        tool=name,
        ok=False,
        arguments=arguments,
        error=ToolError(code=code, message=message, retryable=False),
        latency_ms=_elapsed_ms(started),
        attempts=attempts,
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _short(exc: BaseException, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else f"{text[: limit - 1]}…"
