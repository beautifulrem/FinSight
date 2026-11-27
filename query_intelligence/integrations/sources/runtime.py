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
import weakref
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from .cache import SourceCache
from .health import CircuitOpenError, SourceHealthRegistry
from .provenance import LAST_KNOWN_GOOD, LIVE, LIVE_FALLBACK, utc_now_iso

if TYPE_CHECKING:
    from ...config import Settings

T = TypeVar("T")

_JS_ENGINE_LOCK = threading.Lock()
_JS_CALL_RLOCK = threading.RLock()
_JS_DEFERRED_LOCK = threading.Lock()
_JS_DEFERRED_EXITS: list[Any] = []
_JS_CALL_DEPTH = 0
_JS_ACTIVE_REF: weakref.ReferenceType[Any] | None = None
_JS_MAX_HISTORY = 32
_js_engine_ready = False


def _drain_deferred_js_exits() -> None:
    global _JS_CALL_DEPTH
    while True:
        with _JS_DEFERRED_LOCK:
            if not _JS_DEFERRED_EXITS:
                return
            maker = _JS_DEFERRED_EXITS.pop()
        _JS_CALL_DEPTH += 1
        try:
            with contextlib.suppress(Exception):
                maker.__exit__(None, None, None)
        finally:
            _JS_CALL_DEPTH -= 1


@contextlib.contextmanager
def _js_gate() -> Iterator[None]:
    global _JS_CALL_DEPTH
    _JS_CALL_RLOCK.acquire()
    _JS_CALL_DEPTH += 1
    try:
        yield
    finally:
        _JS_CALL_DEPTH -= 1
        if _JS_CALL_DEPTH == 0:
            _drain_deferred_js_exits()
        _JS_CALL_RLOCK.release()


def _install_mini_racer_gate(cls: type[Any]) -> None:
    """Serialize ``MiniRacer`` against ``--single-threaded`` V8 and keep at most one live isolate.

    ``mini-racer >= 0.13`` initialises V8 14.4 with ``DEFAULT_V8_FLAGS = ("--single-threaded",)`` and
    spawns a C++ ``IsolateManager`` thread plus a Python ``asyncio`` loop thread per ``MiniRacer``
    instance. Concurrent context creation/evaluation/teardown across ``source-call`` worker threads can
    deadlock V8 on multi-core Linux runners, and ``akshare`` decoders rely on GC rather than ``close()``.
    Parking any previous live context before opening another keeps ``mr_context_count() <= 1`` while
    replaying top-level ``eval`` setup scripts transparently if a parked instance is used again.
    """
    if getattr(cls, "_finsight_serialized", False):
        return

    orig_init = cls.__init__
    orig_eval = cls.eval
    orig_execute = getattr(cls, "execute", None)
    orig_call = getattr(cls, "call", None)
    orig_close = getattr(cls, "close", None)

    if not callable(orig_close):
        # Legacy py-mini-racer (<0.10) without explicit close/_own_context_maker: serialize calls only.
        def _legacy_init(self: Any, *args: Any, **kwargs: Any) -> None:
            with _js_gate():
                orig_init(self, *args, **kwargs)

        def _legacy_eval(self: Any, *args: Any, **kwargs: Any) -> Any:
            with _js_gate():
                return orig_eval(self, *args, **kwargs)

        cls.__init__ = _legacy_init  # type: ignore[method-assign]
        cls.eval = _legacy_eval  # type: ignore[method-assign]
        if callable(orig_execute):

            def _legacy_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
                with _js_gate():
                    return orig_execute(self, *args, **kwargs)

            cls.execute = _legacy_execute  # type: ignore[method-assign]
        if callable(orig_call):

            def _legacy_call(self: Any, *args: Any, **kwargs: Any) -> Any:
                with _js_gate():
                    return orig_call(self, *args, **kwargs)

            cls.call = _legacy_call  # type: ignore[method-assign]
        cls._finsight_serialized = True
        return

    def _park_instance(inst: Any) -> None:
        maker = getattr(inst, "_own_context_maker", None)
        inst._own_context_maker = None
        inst._ctx = None
        if maker is not None:
            with contextlib.suppress(Exception):
                maker.__exit__(None, None, None)

    def _evict_other_active(target: Any) -> None:
        global _JS_ACTIVE_REF
        active = _JS_ACTIVE_REF() if _JS_ACTIVE_REF is not None else None
        if active is not None and active is not target:
            _park_instance(active)
            _JS_ACTIVE_REF = None
        elif active is None:
            _JS_ACTIVE_REF = None

    def _ensure_context(inst: Any) -> None:
        global _JS_ACTIVE_REF
        if getattr(inst, "_finsight_external_ctx", False) or getattr(inst, "_finsight_closed", False):
            return
        _evict_other_active(inst)
        if getattr(inst, "_ctx", None) is None:
            inst._finsight_internal_eval = True
            try:
                orig_init(inst)
                for snippet in getattr(inst, "_finsight_history", ()):
                    orig_eval(inst, snippet)
            finally:
                inst._finsight_internal_eval = False
            _JS_ACTIVE_REF = weakref.ref(inst)

    def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        global _JS_ACTIVE_REF
        has_external_ctx = bool(args and args[0] is not None) or (kwargs.get("context") is not None)
        with _js_gate():
            self._finsight_closed = False
            self._finsight_external_ctx = has_external_ctx
            self._finsight_history = []
            self._finsight_internal_eval = True
            try:
                if not has_external_ctx:
                    _evict_other_active(self)
                orig_init(self, *args, **kwargs)
            finally:
                self._finsight_internal_eval = False
            if not has_external_ctx:
                _JS_ACTIVE_REF = weakref.ref(self)

    def _wrapped_eval(self: Any, code: str, *args: Any, **kwargs: Any) -> Any:
        with _js_gate():
            _ensure_context(self)
            result = orig_eval(self, code, *args, **kwargs)
            if not getattr(self, "_finsight_internal_eval", False) and not getattr(
                self, "_finsight_external_ctx", False
            ):
                history = getattr(self, "_finsight_history", None)
                if isinstance(history, list) and len(history) < _JS_MAX_HISTORY:
                    history.append(code)
            return result

    def _wrap_internal_call(fn: Callable[..., Any]) -> Callable[..., Any]:
        def _method(self: Any, *args: Any, **kwargs: Any) -> Any:
            with _js_gate():
                _ensure_context(self)
                prev = getattr(self, "_finsight_internal_eval", False)
                self._finsight_internal_eval = True
                try:
                    return fn(self, *args, **kwargs)
                finally:
                    self._finsight_internal_eval = prev

        return _method

    def _wrapped_close(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _JS_ACTIVE_REF
        with _js_gate():
            self._finsight_closed = True
            history = getattr(self, "_finsight_history", None)
            if isinstance(history, list):
                history.clear()
            active = _JS_ACTIVE_REF() if _JS_ACTIVE_REF is not None else None
            if active is self:
                _JS_ACTIVE_REF = None
            return orig_close(self, *args, **kwargs)

    def _wrapped_del(self: Any) -> None:
        global _JS_ACTIVE_REF, _JS_CALL_DEPTH
        maker = getattr(self, "_own_context_maker", None)
        self._own_context_maker = None
        self._ctx = None
        if maker is None:
            return
        if _JS_CALL_RLOCK.acquire(blocking=False):
            _JS_CALL_DEPTH += 1
            try:
                active = _JS_ACTIVE_REF() if _JS_ACTIVE_REF is not None else None
                if active is None or active is self:
                    _JS_ACTIVE_REF = None
                with contextlib.suppress(Exception):
                    maker.__exit__(None, None, None)
            finally:
                _JS_CALL_DEPTH -= 1
                if _JS_CALL_DEPTH == 0:
                    _drain_deferred_js_exits()
                _JS_CALL_RLOCK.release()
        else:
            with _JS_DEFERRED_LOCK:
                _JS_DEFERRED_EXITS.append(maker)

    cls.__init__ = _wrapped_init  # type: ignore[method-assign]
    cls.eval = _wrapped_eval  # type: ignore[method-assign]
    if callable(orig_execute):
        cls.execute = _wrap_internal_call(orig_execute)  # type: ignore[method-assign]
    if callable(orig_call):
        cls.call = _wrap_internal_call(orig_call)  # type: ignore[method-assign]
    for helper_name in (
        "heap_stats",
        "heap_snapshot",
        "low_memory_notification",
        "set_hard_memory_limit",
        "set_soft_memory_limit",
        "was_hard_memory_limit_reached",
        "was_soft_memory_limit_reached",
    ):
        helper = getattr(cls, helper_name, None)
        if callable(helper):
            setattr(cls, helper_name, _wrap_internal_call(helper))
    cls.close = _wrapped_close  # type: ignore[method-assign]
    cls.__del__ = _wrapped_del  # type: ignore[method-assign]
    cls._finsight_serialized = True


def ensure_js_engine_ready() -> None:
    """Initialise the embedded V8 engine (``py_mini_racer``) once, before any concurrent use.

    Several akshare functions (Sina daily bars and indices, cninfo profiles, ...) decode responses with
    ``py_mini_racer``. V8's process-wide allocator is not safe to initialise from two threads at once:
    concurrent first use aborts the whole process (``Check failed: !pool->IsInitialized()``), and
    ``mini-racer >= 0.13`` runs V8 with ``--single-threaded`` where concurrent isolates can deadlock.
    Live calls run on worker threads and the agent runs tools in parallel, so the first call installs
    the serialization gate and warms the engine under a lock. No-op when the package is missing.
    """
    global _js_engine_ready
    if _js_engine_ready:
        return
    with _JS_ENGINE_LOCK:
        if _js_engine_ready:
            return
        try:
            import py_mini_racer

            _install_mini_racer_gate(py_mini_racer.MiniRacer)
            mr = py_mini_racer.MiniRacer()
            try:
                mr.eval("0")
            finally:
                close_mr = getattr(mr, "close", None)
                if callable(close_mr):
                    close_mr()
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
        pool: SourceCallPool | None = None,
    ) -> None:
        # ``is None`` checks: an empty SourceCache is falsy (it defines ``__len__``).
        self.health = health if health is not None else SourceHealthRegistry()
        self.cache = cache if cache is not None else SourceCache()
        self.call_timeout_s = call_timeout_s
        self.cache_enabled = cache_enabled
        self.max_stale_s = max_stale_s
        self.pool = pool if pool is not None else _default_pool

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
            result = _run_with_timeout(fn, self.call_timeout_s if timeout_s is None else timeout_s, self.pool)
        except SourcePoolSaturatedError as exc:
            # Local back-pressure: release the admission without blaming the upstream.
            self.health.release(source_id)
            self.note(f"{source_id}:saturated")
            raise exc
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


class SourcePoolSaturatedError(RuntimeError):
    """Every worker of the source-call pool is busy (usually with abandoned, hung upstream calls).

    This is local back-pressure, not an upstream failure, so it does not count against the source's
    circuit breaker.
    """


_IN_POOL_WORKER = threading.local()


class SourceCallPool:
    """Fixed-size worker pool for guarded upstream calls, with accounting for abandoned calls.

    Most akshare functions accept no timeout, so a hung upstream call cannot be interrupted. The caller
    stops waiting after the timeout and the call is *abandoned*: it keeps its worker busy until the
    socket gives up. With one new thread per call (the previous design) a hanging upstream under load
    grew the thread count without bound. Here at most ``max_workers`` calls run at once, and a call
    that finds every worker busy is rejected at once (``SourcePoolSaturatedError``) instead of queueing
    behind hung calls.

    ``stats()`` is served by ``/sources/health`` and exported on ``/metrics``.
    """

    def __init__(self, max_workers: int = 32) -> None:
        self.max_workers = max(1, int(max_workers))
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self.busy = 0
        self.max_busy = 0
        self.submitted_total = 0
        self.completed_total = 0
        self.timed_out_total = 0
        self.abandoned_total = 0
        self.abandoned_running = 0
        self.rejected_total = 0

    def run(self, fn: Callable[[], T], timeout_s: float) -> T:
        if getattr(_IN_POOL_WORKER, "active", False):
            # A guarded call made from inside another guarded call: the outer call's timeout already
            # bounds it, and dispatching it to the pool could deadlock when the pool is full.
            return fn()
        with self._lock:
            if self.busy >= self.max_workers:
                self.rejected_total += 1
                raise SourcePoolSaturatedError(
                    f"source-call pool saturated ({self.busy}/{self.max_workers} busy, "
                    f"{self.abandoned_running} abandoned calls still running)"
                )
            self.busy += 1
            self.max_busy = max(self.max_busy, self.busy)
            self.submitted_total += 1
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="source-call")
            executor = self._executor
        context = contextvars.copy_context()
        abandoned = threading.Event()

        def target() -> T:
            _IN_POOL_WORKER.active = True
            try:
                return context.run(fn)
            finally:
                _IN_POOL_WORKER.active = False

        def release(_future: Future) -> None:
            with self._lock:
                self.busy -= 1
                self.completed_total += 1
                if abandoned.is_set():
                    self.abandoned_running -= 1

        future = executor.submit(target)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError:
            with self._lock:
                self.timed_out_total += 1
                if not future.done():
                    # The worker cannot be interrupted; it finishes in the background, its late result
                    # is discarded, and ``release`` frees the slot when it eventually returns.
                    abandoned.set()
                    self.abandoned_total += 1
                    self.abandoned_running += 1
            raise SourceTimeoutError(f"exceeded {timeout_s:g}s") from None
        finally:
            # Registered after the wait so ``abandoned`` is decided first; runs at once if already done.
            future.add_done_callback(release)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "busy": self.busy,
                "max_busy": self.max_busy,
                "submitted_total": self.submitted_total,
                "completed_total": self.completed_total,
                "timed_out_total": self.timed_out_total,
                "abandoned_total": self.abandoned_total,
                "abandoned_running": self.abandoned_running,
                "rejected_total": self.rejected_total,
            }

    def shutdown(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


_default_pool = SourceCallPool()


def default_call_pool() -> SourceCallPool:
    return _default_pool


def _run_with_timeout[R](fn: Callable[[], R], timeout_s: float | None, pool: SourceCallPool | None = None) -> R:
    ensure_js_engine_ready()
    if not timeout_s or timeout_s <= 0:
        return fn()
    return (pool or _default_pool).run(fn, timeout_s)


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
        pool=SourceCallPool(settings.source_max_workers),
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
