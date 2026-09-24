"""Run traces for the agent.

Every agent answer produces a trace with node spans, tool calls (latency, attempts, cache hits,
errors), LLM calls (latency, tokens, cache hits, requested tools), usage and cost, verification
status, and degradation flags.

Sinks:

* ``JsonFileTraceSink``: writes ``<dir>/<YYYY-MM-DD>/<trace_id>.json``. Enabled by default with
  ``outputs/traces`` (gitignored); set ``QI_AGENT_TRACE_DIR=off`` to disable or point it elsewhere.
* ``OTelTraceSink``: exports the trace as OpenTelemetry spans (explicit start/end times) through
  OTLP/HTTP. Enabled when ``QI_AGENT_OTEL=1`` or ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, and the
  optional ``opentelemetry-sdk`` / ``opentelemetry-exporter-otlp-proto-http`` packages are
  installed. Any OTLP backend works (Jaeger, Tempo, or Langfuse via its OTLP endpoint and
  ``OTEL_EXPORTER_OTLP_HEADERS``).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_TRACE_DIR = "outputs/traces"


class TraceSink(Protocol):
    def emit(self, trace: dict[str, Any]) -> None: ...


def build_trace(result: dict[str, Any], *, session_id: str | None = None) -> dict[str, Any]:
    spans = result.get("spans") or []
    starts = [span["started_at"] for span in spans if span.get("started_at") is not None]
    ends = [span["started_at"] + span.get("duration_ms", 0) / 1000 for span in spans if span.get("started_at")]
    started_at = min(starts) if starts else None
    duration_ms = round((max(ends) - started_at) * 1000, 2) if starts and ends else None
    llm = result.get("llm") or {}
    verification = result.get("verification") or {}
    return {
        "trace_id": result.get("run_id"),
        "session_id": session_id,
        "turn_index": result.get("turn_index"),
        "query": result.get("query"),
        "route": result.get("route"),
        "route_reasons": result.get("route_reasons") or [],
        "answer_source": result.get("answer_source"),
        "started_at": started_at,
        "duration_ms": duration_ms,
        "nodes": spans,
        "tools": [
            {
                key: call.get(key)
                for key in (
                    "tool",
                    "arguments",
                    "ok",
                    "error",
                    "latency_ms",
                    "started_at",
                    "attempts",
                    "cached",
                    "source",
                    "step",
                )
            }
            for call in result.get("tool_calls") or []
        ],
        "llm_calls": llm.get("log") or [],
        "model": llm.get("model"),
        "usage": llm.get("usage") or {},
        "cost": llm.get("cost"),
        "currency": llm.get("currency"),
        "verification_passed": verification.get("passed"),
        "unsupported_numbers": verification.get("unsupported_numbers") or [],
        "invalid_citations": verification.get("invalid_citations") or [],
        "compliance_notes": result.get("compliance_notes") or [],
        "degraded": result.get("degraded") or [],
        "evidence_count": len(result.get("evidence_sources") or []),
    }


class JsonFileTraceSink:
    def __init__(self, directory: str | Path = DEFAULT_TRACE_DIR) -> None:
        self.directory = Path(directory)

    def emit(self, trace: dict[str, Any]) -> None:
        day = datetime.fromtimestamp(trace.get("started_at") or datetime.now(UTC).timestamp(), UTC).strftime("%Y-%m-%d")
        folder = self.directory / day
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{trace.get('trace_id') or 'trace'}.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class OTelTraceSink:
    """Exports a finished trace as OpenTelemetry spans with the recorded timestamps."""

    def __init__(self, tracer_provider: Any = None) -> None:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        if tracer_provider is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            tracer_provider = TracerProvider(resource=Resource.create({"service.name": "finsight-agent"}))
            tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        self.provider = tracer_provider
        self.tracer = tracer_provider.get_tracer("finsight.agent")

    def emit(self, trace: dict[str, Any]) -> None:
        from opentelemetry import trace as otel_trace

        start = _ns(trace.get("started_at"))
        end = start + int((trace.get("duration_ms") or 0) * 1e6) if start else None
        root = self.tracer.start_span(
            "finsight.agent.run",
            start_time=start,
            attributes=_attributes(
                {
                    "finsight.trace_id": trace.get("trace_id"),
                    "finsight.session_id": trace.get("session_id"),
                    "finsight.route": trace.get("route"),
                    "finsight.answer_source": trace.get("answer_source"),
                    "finsight.verification_passed": trace.get("verification_passed"),
                    "finsight.degraded": ",".join(trace.get("degraded") or []),
                    "gen_ai.request.model": trace.get("model"),
                    "gen_ai.usage.input_tokens": (trace.get("usage") or {}).get("prompt_tokens"),
                    "gen_ai.usage.output_tokens": (trace.get("usage") or {}).get("completion_tokens"),
                }
            ),
        )
        context = otel_trace.set_span_in_context(root)
        for node in trace.get("nodes") or []:
            self._child(context, f"node.{node['node']}", node.get("started_at"), node.get("duration_ms"), {})
        for tool in trace.get("tools") or []:
            self._child(
                context,
                f"tool.{tool['tool']}",
                tool.get("started_at"),
                tool.get("latency_ms"),
                {
                    "finsight.tool.ok": tool.get("ok"),
                    "finsight.tool.attempts": tool.get("attempts"),
                    "finsight.tool.cached": tool.get("cached"),
                    "finsight.tool.source": tool.get("source"),
                    "finsight.tool.error": (tool.get("error") or {}).get("code"),
                },
            )
        for call in trace.get("llm_calls") or []:
            self._child(
                context,
                f"llm.{call.get('node')}",
                call.get("started_at"),
                call.get("latency_ms"),
                {
                    "gen_ai.request.model": call.get("model"),
                    "gen_ai.usage.input_tokens": call.get("prompt_tokens"),
                    "gen_ai.usage.output_tokens": call.get("completion_tokens"),
                    "finsight.llm.cache_hit_tokens": call.get("prompt_cache_hit_tokens"),
                    "finsight.llm.tool_calls": ",".join(call.get("tool_calls") or []),
                },
            )
        root.end(end_time=end)

    def _child(self, context: Any, name: str, started_at: float | None, duration_ms: float | None, attrs: dict) -> None:
        start = _ns(started_at)
        span = self.tracer.start_span(name, context=context, start_time=start, attributes=_attributes(attrs))
        span.end(end_time=start + int((duration_ms or 0) * 1e6) if start else None)

    def shutdown(self) -> None:
        self.provider.shutdown()


def sinks_from_env() -> list[TraceSink]:
    sinks: list[TraceSink] = []
    directory = os.getenv("QI_AGENT_TRACE_DIR", DEFAULT_TRACE_DIR).strip()
    if directory and directory.lower() not in {"off", "0", "false", "none"}:
        sinks.append(JsonFileTraceSink(directory))
    wants_otel = os.getenv("QI_AGENT_OTEL", "").lower() in {"1", "true", "yes"} or bool(
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    if wants_otel:
        try:
            sinks.append(OTelTraceSink())
        except ImportError:
            logger.warning("OpenTelemetry export requested but opentelemetry-sdk is not installed; skipping")
    return sinks


def emit(sinks: list[TraceSink], trace: dict[str, Any]) -> None:
    for sink in sinks:
        try:
            sink.emit(trace)
        except Exception:  # tracing must never break an answer
            logger.warning("trace sink %s failed", type(sink).__name__, exc_info=True)


def _ns(seconds: float | None) -> int | None:
    return int(seconds * 1e9) if seconds else None


def _attributes(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None and value != ""}
