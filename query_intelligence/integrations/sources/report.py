"""Health report for ``GET /sources/health``.

Passive by default (reports recorded outcomes, never calls an upstream). ``probe=True`` first runs a
rate-limited active probe round (``probe.py``) so the report reflects the sources' current state; every
source row then carries a ``probe`` field: the latest probe result, or ``{"probed": false, "reason": ...}``
for a source the round did not cover.
"""

from __future__ import annotations

from typing import Any

from .probe import ActiveProber, get_default_prober
from .provenance import utc_now_iso
from .runtime import SourceRuntime, get_default_runtime


def runtime_for(retrieval_pipeline: Any = None) -> SourceRuntime:
    runtime = getattr(retrieval_pipeline, "source_runtime", None)
    return runtime if isinstance(runtime, SourceRuntime) else get_default_runtime()


def sources_health_report(
    retrieval_pipeline: Any = None, *, probe: bool = False, prober: ActiveProber | None = None
) -> dict[str, Any]:
    runtime = runtime_for(retrieval_pipeline)
    probe_report: dict[str, Any] | None = None
    if probe:
        live = retrieval_pipeline is None or getattr(retrieval_pipeline, "market_provider", None) is not None
        if live:
            probe_report = (prober or get_default_prober()).probe(runtime)
        else:
            probe_report = {"status": "skipped", "reason": "live market data is disabled (QI_USE_LIVE_MARKET=0)"}
    sources = runtime.health.snapshot()
    if probe_report is not None:
        _annotate_probe_coverage(sources, probe_report)
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
        "worker_pool": runtime.pool.stats(),
        "sources": sources,
    }
    if probe_report is not None:
        report["probe"] = probe_report
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


def _annotate_probe_coverage(sources: list[dict[str, Any]], probe_report: dict[str, Any]) -> None:
    """Mark on every source row whether the latest probe round covered it (and how it went)."""
    if probe_report.get("status") == "skipped":
        for row in sources:
            row["probe"] = {"probed": False, "reason": probe_report.get("reason")}
        return
    round_ = probe_report if probe_report.get("status") == "completed" else probe_report.get("last")
    if not round_:
        for row in sources:
            row["probe"] = {"probed": False, "reason": f"no completed probe round yet ({probe_report.get('status')})"}
        return
    results = {result["source"]: result for result in round_.get("results") or []}
    reasons = {item["source"]: item["reason"] for item in round_.get("not_probed") or []}
    for row in sources:
        result = results.get(row["source"])
        if result is None:
            row["probe"] = {"probed": False, "reason": reasons.get(row["source"], "not in the probe round")}
        else:
            row["probe"] = {
                "probed": True,
                "ok": result["ok"],
                "outcome": result["outcome"],
                "latency_ms": result["latency_ms"],
                "probed_at": round_.get("probed_at"),
            }
