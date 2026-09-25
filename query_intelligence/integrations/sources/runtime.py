"""Source runtime: guarded calls, attempt tracing, and ordered fallback chains.

``SourceRuntime.call`` is the single choke point for live I/O:

1. the per-source circuit breaker is consulted (an open circuit costs no network time);
2. the call runs under a hard wall-clock timeout (most akshare functions accept no timeout);
3. latency and outcome are recorded in the health registry;
4. the attempt is appended to the active trace so provenance can explain the fallback path.

``SourceRuntime.run_chain`` executes candidates in order (live primary -> live secondaries), serves
fresh cache hits, and falls back to the last known good value when every live candidate fails. The
shipped offline snapshot is the caller's final step because only the caller knows how to load it.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from .cache import SourceCache
from .health import CircuitOpenError, SourceHealthRegistry
from .provenance import LAST_KNOWN_GOOD, LIVE, LIVE_FALLBACK, utc_now_iso

if TYPE_CHECKING:
    from ...config import Settings

T = TypeVar("T")

_JS_ENGINE_LOCK = threading.Lock()
_js_engine_ready = False


def ensure_js_engine_ready() -> None:
    """Initialise the embedded V8 engine (``py_mini_racer``) once, before any concurrent use.

    Several akshare functions (Sina daily bars and indices, cninfo profiles, ...) decode responses with
    ``py_mini_racer``. V8's process-wide allocator is not safe to initialise from two threads at once:
    concurrent first use aborts the whole process (``Check failed: !pool->IsInitialized()``). Live calls
    run on worker threads and the agent runs tools in parallel, so the first call initialises the
    engine under a lock and every later call finds it ready. No-op when the package is missing.
    """
    global _js_engine_ready
    if _js_engine_ready:
        return
    with _JS_ENGINE_LOCK:
        if _js_engine_ready:
            return
        try:
            import py_mini_racer

            py_mini_racer.MiniRacer().eval("0")
        except Exception:  # missing package or unusable engine: the calls that need it fail on their own
            pass
        _js_engine_ready = True


_ACTIVE_TRACES: contextvars.ContextVar[tuple[list[str], ...]] = contextvars.ContextVar("source_traces", default=())


class SourceTimeoutError(TimeoutError):
    pass


class EmptyResultError(RuntimeError):
    """A source answered but returned nothing usable (empty or failed validation)."""


class AllSourcesFailedError(RuntimeError):
    def __init__(self, kind: str, attempts: list[str]) -> None:
        super().__init__(f"all live sources failed for {kind}: {'; '.join(attempts) or 'no candidates'}")
        self.kind = kind
        self.attempts = attempts


@dataclass
class Candidate:
    source_id: str
    endpoint: str
    fetch: Callable[[], Any]


@dataclass
class ChainResult:
    value: Any
    source_id: str
    endpoint: str
    mode: str
    fetched_at: str
    attempts: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    cache_hit: bool = False


def error_summary(exc: BaseException, limit: int = 160) -> str:
    text = " ".join(str(exc).split())
    summary = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return summary if len(summary) <= limit else summary[: limit - 3] + "..."


def attempt_label(source_id: str, exc: BaseException | None) -> str:
    if exc is None:
        return f"{source_id}:ok"
    if isinstance(exc, CircuitOpenError):
        return f"{source_id}:circuit_open"
    if isinstance(exc, SourceTimeoutError | TimeoutError):
        return f"{source_id}:timeout"
    if isinstance(exc, EmptyResultError):
        return f"{source_id}:empty"
    return f"{source_id}:error({type(exc).__name__})"


def fallback_reason_from(attempts: list[str]) -> str | None:
    """Human-readable reason built from the failed attempts that preceded the serving source."""
    failed = [item for item in attempts if not item.endswith(":ok")]
    return "; ".join(failed) or None


class SourceRuntime:
    def __init__(
        self,
        *,
        health: SourceHealthRegistry | None = None,
        cache: SourceCache | None = None,
        call_timeout_s: float = 10.0,
        cache_enabled: bool = True,
        max_stale_s: float = 24 * 3600.0,
    ) -> None:
        # ``is None`` checks: an empty SourceCache is falsy (it defines ``__len__``).
        self.health = health if health is not None else SourceHealthRegistry()
        self.cache = cache if cache is not None else SourceCache()
        self.call_timeout_s = call_timeout_s
        self.cache_enabled = cache_enabled
        self.max_stale_s = max_stale_s

    # ---- tracing --------------------------------------------------------

    @contextlib.contextmanager
    def trace(self) -> Iterator[list[str]]:
        """Collect attempt labels of every ``call`` made in this context (thread/context local).

        Traces nest: an attempt is recorded in every enclosing trace, so a caller can see the attempts
        a provider made internally.
        """
        attempts: list[str] = []
        token = _ACTIVE_TRACES.set((*_ACTIVE_TRACES.get(), attempts))
        try:
            yield attempts
        finally:
            _ACTIVE_TRACES.reset(token)

    def note(self, label: str) -> None:
        for attempts in _ACTIVE_TRACES.get():
            attempts.append(label)

    # ---- guarded calls --------------------------------------------------

    def call(self, source_id: str, fn: Callable[[], T], *, timeout_s: float | None = None) -> T:
        try:
            self.health.acquire(source_id)
        except CircuitOpenError as exc:
            self.note(attempt_label(source_id, exc))
            raise
        started = time.perf_counter()
        try:
            result = _run_with_timeout(fn, self.call_timeout_s if timeout_s is None else timeout_s)
        except BaseException as exc:
            self.health.record_failure(source_id, _elapsed_ms(started), error_summary(exc))
            self.note(attempt_label(source_id, exc))
            raise
        self.health.record_success(source_id, _elapsed_ms(started))
        self.note(attempt_label(source_id, None))
        return result

    # ---- fallback chains ------------------------------------------------

    def run_chain(
        self,
        kind: str,
        cache_key: str,
        candidates: list[Candidate],
        *,
        ttl_s: float,
        validate: Callable[[Any], bool] | None = None,
    ) -> ChainResult:
        """Try candidates in order; raise ``AllSourcesFailedError`` when neither live nor cache can serve."""
        key = f"{kind}:{cache_key}"
        if self.cache_enabled and ttl_s > 0:
            cached = self.cache.get(key)
            if cached is not None:
                cached.cache_hit = True
                return cached

        attempts: list[str] = []
        for index, candidate in enumerate(candidates):
            with self.trace() as trace:
                try:
                    value = self.call(candidate.source_id, candidate.fetch)
                    if value is None or (validate is not None and not validate(value)):
                        raise EmptyResultError(f"{candidate.endpoint} returned no usable data")
                except EmptyResultError as exc:
                    trace[-1:] = [attempt_label(candidate.source_id, exc)]
                    attempts.extend(trace)
                    continue
                except Exception:
                    attempts.extend(trace)
                    continue
            attempts.extend(trace)
            result = ChainResult(
                value=value,
                source_id=candidate.source_id,
                endpoint=candidate.endpoint,
                mode=LIVE if index == 0 else LIVE_FALLBACK,
                fetched_at=utc_now_iso(),
                attempts=attempts,
                fallback_reason=fallback_reason_from(attempts),
            )
            if self.cache_enabled and ttl_s > 0:
                self.cache.put(key, result, ttl_s)
            return result

        if self.cache_enabled:
            stale = self.cache.get_stale(key, self.max_stale_s)
            if stale is not None:
                stale.mode = LAST_KNOWN_GOOD
                stale.cache_hit = True
                stale.attempts = attempts
                stale.fallback_reason = fallback_reason_from(attempts)
                return stale
        raise AllSourcesFailedError(kind, attempts)


def _run_with_timeout(fn: Callable[[], T], timeout_s: float | None) -> T:
    ensure_js_engine_ready()
    if not timeout_s or timeout_s <= 0:
        return fn()
    outcome: dict[str, Any] = {}
    context = contextvars.copy_context()

    def target() -> None:
        try:
            outcome["value"] = context.run(fn)
        except BaseException as exc:  # re-raised in the caller thread
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True, name="source-call")
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        # The worker cannot be killed; it is a daemon and its late result is discarded.
        raise SourceTimeoutError(f"exceeded {timeout_s:g}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


_default_runtime: SourceRuntime | None = None
_default_lock = threading.Lock()


def runtime_from_settings(settings: Settings) -> SourceRuntime:
    ensure_js_engine_ready()
    health = SourceHealthRegistry(
        failure_threshold=settings.source_failure_threshold,
        cooldown_s=settings.source_cooldown_seconds,
        max_cooldown_s=settings.source_max_cooldown_seconds,
    )
    return SourceRuntime(
        health=health,
        call_timeout_s=settings.source_call_timeout_seconds,
        cache_enabled=settings.source_cache_enabled,
        max_stale_s=settings.source_max_stale_seconds,
    )


def get_default_runtime() -> SourceRuntime:
    """Process-wide runtime shared by the live providers and the ``/sources/health`` endpoint.

    Configured from the environment on first use (see ``Settings.source_*``).
    """
    global _default_runtime
    with _default_lock:
        if _default_runtime is None:
            from ...config import Settings

            _default_runtime = runtime_from_settings(Settings.from_env())
        return _default_runtime


def reset_default_runtime() -> None:
    global _default_runtime
    with _default_lock:
        _default_runtime = None
