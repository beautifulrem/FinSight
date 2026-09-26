"""Prometheus collector for operational state that is not part of a run trace.

The trace-fed metrics in ``agent/telemetry.py`` only see finished runs. Breaker states and worker-pool
health are *current* state, so they are read at scrape time:

* ``finsight_source_circuit_state{source}`` – 0 closed, 1 half-open, 2 open (per live data source);
* ``finsight_source_calls_total{source,outcome}`` – success / failure / short_circuited;
* ``finsight_source_latency_ms{source}`` – smoothed latency of recent calls;
* ``finsight_source_pool_*`` – the bounded source-call pool: busy workers, abandoned (timed-out but
  still running) calls, rejections when saturated;
* ``finsight_llm_circuit_state{model}`` and ``finsight_llm_client_calls_total{model}`` – per-model
  failover state of ``FallbackLLM`` (primary and fallbacks), so a failover is visible even though the
  run-level ``model`` label names the configured primary.

Registered on the same private registry as the trace metrics by the API app.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

STATE_VALUES = {"closed": 0, "half_open": 1, "open": 2}


def llm_circuit_states(llm: Any) -> list[dict[str, Any]]:
    """Per-model breaker state of an LLM client (``FallbackLLM`` or a single client)."""
    if llm is None:
        return []
    stats = llm.stats() if callable(getattr(llm, "stats", None)) else None
    if not stats:
        return [{"model": str(getattr(llm, "model", "unknown")), "state": "closed", "calls": None, "failures": 0}]
    rows = []
    for row in stats:
        state = row.get("state") or ("open" if row.get("circuit_open") else "closed")
        rows.append(
            {
                "model": str(row.get("model")),
                "state": state,
                "calls": row.get("calls"),
                "failures": row.get("consecutive_failures", 0),
            }
        )
    return rows


class OpsMetricsCollector:
    def __init__(self, runtime_getter: Callable[[], Any], llm_getter: Callable[[], Any] | None = None) -> None:
        self._runtime_getter = runtime_getter
        self._llm_getter = llm_getter or (lambda: None)

    def describe(self) -> list:
        # Returning no descriptions keeps registration lazy (collect() may need the app to be built).
        return []

    def collect(self) -> Iterator[Any]:
        from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

        runtime = _safe(self._runtime_getter)
        if runtime is not None:
            state = GaugeMetricFamily(
                "finsight_source_circuit_state",
                "Live source breaker: 0 closed, 1 half-open, 2 open.",
                labels=["source"],
            )
            calls = CounterMetricFamily(
                "finsight_source_calls", "Live source calls by outcome.", labels=["source", "outcome"]
            )
            latency = GaugeMetricFamily(
                "finsight_source_latency_ms", "Smoothed latency of recent calls per source.", labels=["source"]
            )
            for row in runtime.health.snapshot():
                source = row["source"]
                state.add_metric([source], STATE_VALUES.get(row.get("circuit", "closed"), 0))
                if row.get("calls"):
                    calls.add_metric([source, "success"], row.get("successes", 0))
                    calls.add_metric([source, "failure"], row.get("failures", 0))
                if row.get("short_circuited"):
                    calls.add_metric([source, "short_circuited"], row["short_circuited"])
                if row.get("avg_latency_ms") is not None:
                    latency.add_metric([source], row["avg_latency_ms"])
            yield state
            yield calls
            yield latency
            pool = runtime.pool.stats()
            yield GaugeMetricFamily(
                "finsight_source_pool_workers", "Size of the source-call pool.", value=pool["max_workers"]
            )
            yield GaugeMetricFamily("finsight_source_pool_busy", "Busy source-call workers.", value=pool["busy"])
            yield GaugeMetricFamily(
                "finsight_source_pool_abandoned_running",
                "Timed-out upstream calls still occupying a worker.",
                value=pool["abandoned_running"],
            )
            yield CounterMetricFamily(
                "finsight_source_pool_abandoned",
                "Upstream calls abandoned after their timeout.",
                value=pool["abandoned_total"],
            )
            yield CounterMetricFamily(
                "finsight_source_pool_rejected",
                "Calls rejected because the pool was saturated.",
                value=pool["rejected_total"],
            )
        llm_rows = llm_circuit_states(_safe(self._llm_getter))
        if llm_rows:
            state = GaugeMetricFamily(
                "finsight_llm_circuit_state", "LLM model breaker: 0 closed, 1 half-open, 2 open.", labels=["model"]
            )
            calls = CounterMetricFamily(
                "finsight_llm_client_calls", "Calls attempted per model (including failed ones).", labels=["model"]
            )
            failures = GaugeMetricFamily(
                "finsight_llm_consecutive_failures", "Consecutive failures per model.", labels=["model"]
            )
            for row in llm_rows:
                state.add_metric([row["model"]], STATE_VALUES[row["state"]])
                if row["calls"] is not None:
                    calls.add_metric([row["model"]], row["calls"])
                failures.add_metric([row["model"]], row["failures"])
            yield state
            yield calls
            yield failures


def _safe(getter: Callable[[], Any]) -> Any:
    try:
        return getter()
    except Exception:
        return None
