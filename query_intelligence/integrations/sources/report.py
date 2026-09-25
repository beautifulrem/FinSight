"""Passive health report for ``GET /sources/health`` (never performs network I/O)."""

from __future__ import annotations

from typing import Any

from .provenance import utc_now_iso
from .runtime import SourceRuntime, get_default_runtime


def sources_health_report(retrieval_pipeline: Any = None) -> dict[str, Any]:
    runtime = getattr(retrieval_pipeline, "source_runtime", None)
    if not isinstance(runtime, SourceRuntime):
        runtime = get_default_runtime()
    sources = runtime.health.snapshot()
    counts: dict[str, int] = {}
    for row in sources:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "summary": counts,
        "circuit_breaker": {
            "failure_threshold": runtime.health.failure_threshold,
            "cooldown_s": runtime.health.cooldown_s,
            "max_cooldown_s": runtime.health.max_cooldown_s,
        },
        "call_timeout_s": runtime.call_timeout_s,
        "cache": {"enabled": runtime.cache_enabled, "entries": len(runtime.cache), "max_stale_s": runtime.max_stale_s},
        "sources": sources,
    }
    if retrieval_pipeline is not None:
        report["live_providers"] = {
            "market": type(retrieval_pipeline.market_provider).__name__
            if getattr(retrieval_pipeline, "market_provider", None)
            else None,
            "macro": type(retrieval_pipeline.macro_provider).__name__
            if getattr(retrieval_pipeline, "macro_provider", None)
            else None,
            "news": [type(provider).__name__ for provider in getattr(retrieval_pipeline, "news_providers", []) or []],
            "announcement": type(retrieval_pipeline.announcement_provider).__name__
            if getattr(retrieval_pipeline, "announcement_provider", None)
            else None,
        }
    return report
