"""Per-source health tracking with a circuit breaker.

States follow the classic breaker:

* ``closed`` – calls flow; ``failure_threshold`` consecutive failures open the circuit.
* ``open`` – calls are short-circuited (no network) until the cooldown elapses.
* ``half_open`` – exactly one trial call is let through; success closes the circuit, failure re-opens
  it with a doubled cooldown (capped at ``max_cooldown_s``).

The registry never performs network I/O itself, so ``snapshot()`` is safe to serve from a health
endpoint.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .catalog import CATALOG

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised instead of calling a source whose circuit is open."""

    def __init__(self, source_id: str, retry_in_s: float) -> None:
        super().__init__(f"circuit open for {source_id}; retry in {retry_in_s:.0f}s")
        self.source_id = source_id
        self.retry_in_s = retry_in_s


@dataclass
class _SourceState:
    state: str = CLOSED
    calls: int = 0
    successes: int = 0
    failures: int = 0
    short_circuited: int = 0
    consecutive_failures: int = 0
    last_latency_ms: float | None = None
    avg_latency_ms: float | None = None
    last_success_at: str | None = None
    last_failure_at: str | None = None
    last_error: str | None = None
    open_until: float = 0.0
    cooldown_s: float = 0.0
    trial_in_flight: bool = False


class SourceHealthRegistry:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_s: float = 60.0,
        max_cooldown_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_s = max(0.0, cooldown_s)
        self.max_cooldown_s = max(self.cooldown_s, max_cooldown_s)
        self._clock = clock
        self._wall_clock = wall_clock
        self._states: dict[str, _SourceState] = {}
        self._lock = threading.Lock()

    def _get(self, source_id: str) -> _SourceState:
        state = self._states.get(source_id)
        if state is None:
            state = _SourceState()
            self._states[source_id] = state
        return state

    def acquire(self, source_id: str) -> None:
        """Admit one call or raise ``CircuitOpenError``; every admitted call must be recorded."""
        with self._lock:
            state = self._get(source_id)
            now = self._clock()
            if state.state == OPEN:
                if now < state.open_until:
                    state.short_circuited += 1
                    raise CircuitOpenError(source_id, state.open_until - now)
                state.state = HALF_OPEN
                state.trial_in_flight = False
            if state.state == HALF_OPEN:
                if state.trial_in_flight:
                    state.short_circuited += 1
                    raise CircuitOpenError(source_id, 0.0)
                state.trial_in_flight = True

    def is_available(self, source_id: str) -> bool:
        with self._lock:
            state = self._states.get(source_id)
            if state is None or state.state == CLOSED:
                return True
            if state.state == OPEN:
                return self._clock() >= state.open_until
            return not state.trial_in_flight

    def record_success(self, source_id: str, latency_ms: float) -> None:
        with self._lock:
            state = self._get(source_id)
            state.calls += 1
            state.successes += 1
            state.consecutive_failures = 0
            self._record_latency(state, latency_ms)
            state.last_success_at = self._now_iso()
            state.state = CLOSED
            state.cooldown_s = 0.0
            state.trial_in_flight = False

    def record_failure(self, source_id: str, latency_ms: float, error: str) -> None:
        with self._lock:
            state = self._get(source_id)
            state.calls += 1
            state.failures += 1
            state.consecutive_failures += 1
            self._record_latency(state, latency_ms)
            state.last_failure_at = self._now_iso()
            state.last_error = error[:300]
            was_trial = state.state == HALF_OPEN
            state.trial_in_flight = False
            if was_trial or state.consecutive_failures >= self.failure_threshold:
                state.cooldown_s = (
                    min(self.max_cooldown_s, max(self.cooldown_s, state.cooldown_s * 2))
                    if was_trial
                    else self.cooldown_s
                )
                state.state = OPEN
                state.open_until = self._clock() + state.cooldown_s

    def snapshot(self) -> list[dict[str, Any]]:
        """Status of every catalogued source plus any other source that has been called."""
        with self._lock:
            now = self._clock()
            source_ids = list(CATALOG) + sorted(set(self._states) - set(CATALOG))
            rows = []
            for source_id in source_ids:
                info = CATALOG.get(source_id)
                state = self._states.get(source_id)
                row: dict[str, Any] = {
                    "source": source_id,
                    "label": info.label if info else source_id,
                    "upstream": info.upstream if info else None,
                    "kinds": list(info.kinds) if info else [],
                }
                if state is None:
                    row.update({"status": "unknown", "circuit": CLOSED, "calls": 0})
                    rows.append(row)
                    continue
                row.update(
                    {
                        "status": _status(state),
                        "circuit": state.state,
                        "calls": state.calls,
                        "successes": state.successes,
                        "failures": state.failures,
                        "short_circuited": state.short_circuited,
                        "consecutive_failures": state.consecutive_failures,
                        "last_latency_ms": state.last_latency_ms,
                        "avg_latency_ms": state.avg_latency_ms,
                        "last_success_at": state.last_success_at,
                        "last_failure_at": state.last_failure_at,
                        "last_error": state.last_error,
                        "retry_in_s": round(max(0.0, state.open_until - now), 1) if state.state == OPEN else None,
                    }
                )
                rows.append(row)
            return rows

    def reset(self) -> None:
        with self._lock:
            self._states.clear()

    def _record_latency(self, state: _SourceState, latency_ms: float) -> None:
        latency = round(latency_ms, 1)
        state.last_latency_ms = latency
        state.avg_latency_ms = (
            latency if state.avg_latency_ms is None else round(state.avg_latency_ms * 0.7 + latency * 0.3, 1)
        )

    def _now_iso(self) -> str:
        return self._wall_clock().isoformat(timespec="seconds")


def _status(state: _SourceState) -> str:
    if state.state != CLOSED:
        return "down"
    if state.consecutive_failures:
        return "degraded"
    return "up"
