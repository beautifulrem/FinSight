"""In-process observability for the agent API: recent traces and Prometheus metrics.

Both classes are ``TraceSink`` implementations (see ``tracing.py``), so they receive exactly the
trace that is written to disk / exported over OTLP.

* ``RecentTraceStore`` keeps the last N traces in memory for ``GET /agent/traces`` and
  ``GET /agent/traces/{trace_id}`` (the web UI's run inspector). It falls back to the JSON trace files
  for older traces.
* ``PrometheusTraceSink`` updates counters and histograms served at ``GET /metrics``: runs by route and
  answer source, run latency, tool calls/errors/latency per tool, LLM calls, tokens and cost,
  verification failures and degradations. ``prometheus-client`` is optional; without it the sink is a
  no-op and ``/metrics`` reports that metrics are unavailable.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

_LATENCY_BUCKETS_S = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)


class RecentTraceStore:
    def __init__(self, capacity: int = 200, *, trace_dir: str | Path | None = None) -> None:
        self.capacity = max(1, capacity)
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self._traces: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def emit(self, trace: dict[str, Any]) -> None:
        trace_id = trace.get("trace_id")
        if not trace_id:
            return
        with self._lock:
            self._traces[str(trace_id)] = trace
            self._traces.move_to_end(str(trace_id))
            while len(self._traces) > self.capacity:
                self._traces.popitem(last=False)

    def get(self, trace_id: str, *, owner: str | None = None) -> dict[str, Any] | None:
        """The trace, or ``None`` when it is unknown or belongs to another caller (``owner``)."""
        with self._lock:
            trace = self._traces.get(trace_id)
        if trace is not None:
            return trace if _visible(trace, owner) else None
        if self.trace_dir is None or not self.trace_dir.is_dir():
            return None
        for path in self.trace_dir.glob(f"*/{trace_id}.json"):
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            return trace if _visible(trace, owner) else None
        return None

    def recent(
        self, limit: int = 50, *, session_id: str | None = None, owner: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            traces = [trace for trace in reversed(self._traces.values()) if _visible(trace, owner)]
        if session_id:
            traces = [trace for trace in traces if trace.get("session_id") == session_id]
        return [summarize_trace(trace) for trace in traces[: max(0, limit)]]


def _visible(trace: dict[str, Any], owner: str | None) -> bool:
    # Traces written before ownership existed have no owner and are treated as local.
    return owner is None or str(trace.get("owner") or "local") == owner


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    tools = trace.get("tools") or []
    usage = trace.get("usage") or {}
    return {
        "trace_id": trace.get("trace_id"),
        "session_id": trace.get("session_id"),
        "query": trace.get("query"),
        "route": trace.get("route"),
        "answer_source": trace.get("answer_source"),
        "started_at": trace.get("started_at"),
        "duration_ms": trace.get("duration_ms"),
        "tool_calls": len(tools),
        "tool_errors": sum(1 for tool in tools if not tool.get("ok")),
        "llm_calls": len(trace.get("llm_calls") or []),
        "total_tokens": int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0),
        "cost": trace.get("cost"),
        "currency": trace.get("currency"),
        "verification_passed": trace.get("verification_passed"),
        "degraded": trace.get("degraded") or [],
    }


def prompt_version_of(trace: dict[str, Any]) -> str:
    """Prompt version (``v3``) of the run's first LLM call (``agent_system@v3#sha``); ``none`` without an LLM."""
    for call in trace.get("llm_calls") or []:
        ref = str(call.get("prompt") or "")
        if "@" in ref:
            return ref.split("@", 1)[1].split("#", 1)[0][:16] or "unknown"
    return "none"


def verification_outcome(trace: dict[str, Any]) -> str | None:
    """``passed`` / ``revised`` / ``repaired`` for answered runs; ``None`` for refusals and clarifications."""
    passed = trace.get("verification_passed")
    if passed is None:
        return None
    if passed is False:
        return "repaired"
    revised = any(call.get("node") == "revise" for call in trace.get("llm_calls") or [])
    return "revised" if revised else "passed"


class PrometheusTraceSink:
    """Aggregates traces into Prometheus metrics on a private registry."""

    def __init__(self) -> None:
        try:
            from prometheus_client import CollectorRegistry, Counter, Histogram
        except ImportError:
            self.registry = None
            return
        self.registry = CollectorRegistry()
        self.runs = Counter(
            "finsight_agent_runs_total",
            "Agent runs by route and answer source.",
            ["route", "answer_source"],
            registry=self.registry,
        )
        self.run_latency = Histogram(
            "finsight_agent_run_seconds",
            "End-to-end agent run latency.",
            ["route"],
            buckets=_LATENCY_BUCKETS_S,
            registry=self.registry,
        )
        self.tool_calls = Counter(
            "finsight_tool_calls_total", "Tool calls by tool and outcome.", ["tool", "outcome"], registry=self.registry
        )
        self.tool_latency = Histogram(
            "finsight_tool_seconds", "Tool call latency.", ["tool"], buckets=_LATENCY_BUCKETS_S, registry=self.registry
        )
        self.llm_calls = Counter("finsight_llm_calls_total", "LLM calls.", ["model"], registry=self.registry)
        self.llm_tokens = Counter(
            "finsight_llm_tokens_total", "LLM tokens by kind.", ["model", "kind"], registry=self.registry
        )
        self.llm_cost = Counter(
            "finsight_llm_cost_total", "LLM cost by currency.", ["model", "currency"], registry=self.registry
        )
        self.verification_failures = Counter(
            "finsight_verification_failures_total",
            "Runs whose draft answer failed citation/number verification.",
            registry=self.registry,
        )
        self.degradations = Counter(
            "finsight_degradations_total", "Degradation flags raised during runs.", ["flag"], registry=self.registry
        )
        self.feedback = Counter(
            "finsight_feedback_total",
            "User feedback on answers, by rating and the prompt version that produced the answer.",
            ["rating", "prompt_version"],
            registry=self.registry,
        )
        # Answer quality by prompt version. outcome: passed (first draft verified), revised (verified after an LLM
        # revision), repaired (still failing, deterministic repair applied). Labels stay low-cardinality: a
        # handful of prompt versions x 3 outcomes.
        self.verification_outcomes = Counter(
            "finsight_answer_verification_total",
            "Verified answers by prompt version and verification outcome.",
            ["prompt_version", "outcome"],
            registry=self.registry,
        )

        # Injection-filter redactions (C14): source = user_message | evidence | tool_output; outcome = answered |
        # refused. user_message/answered is the case the audit log previously missed.
        self.injection_redactions = Counter(
            "finsight_injection_redactions_total",
            "Runs in which the injection filter removed text, by source and whether the run was answered.",
            ["source", "outcome"],
            registry=self.registry,
        )

    @property
    def available(self) -> bool:
        return self.registry is not None

    def emit(self, trace: dict[str, Any]) -> None:
        if self.registry is None:
            return
        route = str(trace.get("route") or "unknown")
        self.runs.labels(route=route, answer_source=str(trace.get("answer_source") or "unknown")).inc()
        if trace.get("duration_ms") is not None:
            self.run_latency.labels(route=route).observe(float(trace["duration_ms"]) / 1000)
        for tool in trace.get("tools") or []:
            name = str(tool.get("tool") or "unknown")
            outcome = "cached" if tool.get("cached") else ("ok" if tool.get("ok") else "error")
            self.tool_calls.labels(tool=name, outcome=outcome).inc()
            if tool.get("latency_ms") is not None:
                self.tool_latency.labels(tool=name).observe(float(tool["latency_ms"]) / 1000)
        # Label by the model that answered each call, so failover traffic is attributed to the fallback model.
        default_model = str(trace.get("model") or "none")
        calls = trace.get("llm_calls") or []
        kinds = ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "reasoning_tokens")
        per_call = any(call.get(kind) for call in calls for kind in kinds)
        tokens_by_model: dict[str, dict[str, float]] = {}
        for call in calls:
            model = str(call.get("model") or default_model)
            self.llm_calls.labels(model=model).inc()
            if per_call:
                bucket = tokens_by_model.setdefault(model, {})
                for kind in kinds:
                    bucket[kind] = bucket.get(kind, 0.0) + float(call.get(kind) or 0)
        if not per_call:
            usage = trace.get("usage") or {}
            tokens_by_model = {default_model: {kind: float(usage.get(kind) or 0) for kind in kinds}}
        for model, bucket in tokens_by_model.items():
            for kind, value in bucket.items():
                if value:
                    self.llm_tokens.labels(model=model, kind=kind.removesuffix("_tokens")).inc(value)
        if trace.get("cost"):
            # The run's cost is split across models in proportion to their prompt + completion tokens.
            weights = {
                model: bucket.get("prompt_tokens", 0.0) + bucket.get("completion_tokens", 0.0)
                for model, bucket in tokens_by_model.items()
            }
            total = sum(weights.values())
            if not total:
                weights, total = {default_model: 1.0}, 1.0
            for model, weight in weights.items():
                if weight:
                    self.llm_cost.labels(model=model, currency=str(trace.get("currency") or "unknown")).inc(
                        float(trace["cost"]) * weight / total
                    )
        if trace.get("verification_passed") is False:
            self.verification_failures.inc()
        outcome = verification_outcome(trace)
        if outcome:
            self.verification_outcomes.labels(prompt_version=prompt_version_of(trace), outcome=outcome).inc()
        for flag in trace.get("degraded") or []:
            self.degradations.labels(flag=str(flag).split(":")[0]).inc()
        from .audit import redaction_sources

        outcome = "refused" if route == "refuse" else "answered"
        for source in redaction_sources(trace):
            self.injection_redactions.labels(source=source, outcome=outcome).inc()

    def record_feedback(self, rating: str, prompt_version: str = "none") -> None:
        if self.registry is not None:
            self.feedback.labels(rating=rating, prompt_version=prompt_version).inc()

    def render(self) -> tuple[bytes, str]:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST
