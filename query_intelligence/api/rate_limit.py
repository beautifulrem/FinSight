"""Request rate limiting: one token bucket per client, in process or shared through Postgres.

* In memory (``TokenBucket``): per replica. With N replicas behind a load balancer a client gets up to N times
  the configured rate.
* Postgres (``PostgresTokenBucket``): one row per client in ``finsight_rate_buckets``, updated by a single
  atomic ``INSERT … ON CONFLICT DO UPDATE`` (the row lock serialises concurrent requests for the same client
  across replicas), with the database clock as the time source so replica clock skew does not matter.

``build_rate_limiter`` picks Postgres when a DSN is configured: ``QI_RATE_LIMIT_DB`` (a ``postgresql://`` DSN,
or ``memory`` to force the in-process bucket), else the checkpointer's ``QI_AGENT_CHECKPOINT_DB`` when that is
Postgres, the same switch as the shared trace and A2A task stores (``agent/pg.py``).

Failure policy: if the database cannot be reached at start-up, or a call fails later, the request is limited by
an in-process bucket instead (per replica, logged at most once a minute). Rate limiting degrades rather than
turning a database outage into a full API outage; authentication is unaffected.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MAX_RATE_CLIENTS = 10_000
DEFAULT_TABLE = "finsight_rate_buckets"
_IDLE_SECONDS = 60.0  # a bucket untouched for a minute has refilled to capacity: same as no row


class RateLimiter(Protocol):
    def take(self, client: str) -> float:
        """Consume one token; return 0 when allowed, else seconds until the next token."""
        ...


@dataclass
class TokenBucket:
    """Per-client token buckets in process, with bounded state.

    At most ``max_clients`` buckets are kept, least recently used first out. A bucket untouched for a full
    minute has refilled to capacity, which is the same as having no entry, so idle buckets are dropped
    first; evicting a still-draining bucket only happens under more than ``max_clients`` active clients.
    """

    rate_per_minute: int
    clock: Callable[[], float] = time.monotonic
    max_clients: int = DEFAULT_MAX_RATE_CLIENTS
    _state: OrderedDict[str, tuple[float, float]] = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, client: str) -> float:
        capacity = float(self.rate_per_minute)
        refill_per_second = capacity / 60.0
        now = self.clock()
        with self._lock:
            tokens, updated = self._state.get(client, (capacity, now))
            tokens = min(capacity, tokens + (now - updated) * refill_per_second)
            if tokens >= 1.0:
                self._state[client] = (tokens - 1.0, now)
                wait = 0.0
            else:
                self._state[client] = (tokens, now)
                wait = (1.0 - tokens) / refill_per_second
            self._state.move_to_end(client)
            self._evict(now)
            return wait

    def _evict(self, now: float) -> None:
        while self._state:
            oldest, (_tokens, updated) = next(iter(self._state.items()))
            if len(self._state) > self.max_clients or now - updated >= _IDLE_SECONDS:
                del self._state[oldest]
                continue
            break

    def close(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self._state)


_REFILLED = "LEAST(%(capacity)s, b.tokens + GREATEST(0, {now} - b.updated_at) * %(refill)s)"


class PostgresTokenBucket:
    """The same token bucket, stored in Postgres so every replica draws from one bucket per client."""

    def __init__(
        self,
        pool: Any,
        rate_per_minute: int,
        *,
        table: str = DEFAULT_TABLE,
        fallback: TokenBucket | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        from ..agent.pg import check_identifier

        self.pool = pool
        self.rate_per_minute = rate_per_minute
        self.table = check_identifier(table)
        self.fallback = fallback or TokenBucket(rate_per_minute)
        # tests may pin the clock; otherwise the database clock is shared by every replica
        self._clock = clock
        now = "%(now)s::double precision" if clock else "extract(epoch from clock_timestamp())"
        refilled = _REFILLED.format(now=now)
        self._take_sql = (
            f"INSERT INTO {self.table} AS b (client, tokens, updated_at, allowed) "
            f"VALUES (%(client)s, %(capacity)s - 1, {now}, true) "
            f"ON CONFLICT (client) DO UPDATE SET "
            f"tokens = CASE WHEN {refilled} >= 1 THEN {refilled} - 1 ELSE {refilled} END, "
            f"updated_at = {now}, allowed = {refilled} >= 1 "
            f"RETURNING allowed, tokens"
        )
        self._cleanup_sql = f"DELETE FROM {self.table} WHERE updated_at < {now} - %(idle)s"
        self._last_cleanup = 0.0
        self._last_warning = 0.0
        self._lock = threading.Lock()
        create = (
            f"CREATE TABLE IF NOT EXISTS {self.table} ("
            "client text PRIMARY KEY, tokens double precision NOT NULL, "
            "updated_at double precision NOT NULL, allowed boolean NOT NULL DEFAULT true)"
        )
        for attempt in range(2):
            try:
                with self.pool.connection() as conn:
                    conn.execute(create)
                break
            except Exception:
                # two replicas creating the table at once can collide on the catalog; the second try sees it
                if attempt:
                    raise

    def _params(self, **extra: Any) -> dict[str, Any]:
        capacity = float(self.rate_per_minute)
        params = {"capacity": capacity, "refill": capacity / 60.0, **extra}
        if self._clock is not None:
            params["now"] = self._clock()
        return params

    def take(self, client: str) -> float:
        try:
            with self.pool.connection() as conn:
                allowed, tokens = conn.execute(self._take_sql, self._params(client=client)).fetchone()
                self._maybe_cleanup(conn)
        except Exception as exc:  # database trouble: limit per replica rather than fail every request
            self._warn(exc)
            return self.fallback.take(client)
        if allowed:
            return 0.0
        return max(0.0, (1.0 - float(tokens)) / (self.rate_per_minute / 60.0))

    def _maybe_cleanup(self, conn: Any) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_cleanup < _IDLE_SECONDS:
                return
            self._last_cleanup = now
        conn.execute(self._cleanup_sql, self._params(idle=_IDLE_SECONDS))

    def _warn(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_warning >= 60.0:
            self._last_warning = now
            logger.warning("shared rate limiter unavailable, limiting per replica: %s", exc)

    def close(self) -> None:
        close = getattr(self.pool, "close", None)
        if callable(close):
            close()


def rate_limit_dsn() -> str | None:
    from ..agent.pg import store_dsn

    return store_dsn("QI_RATE_LIMIT_DB")


def build_rate_limiter(rate_per_minute: int, *, dsn: str | None = None) -> TokenBucket | PostgresTokenBucket | None:
    """``None`` when rate limiting is off; Postgres when a DSN is configured and reachable; else in process."""
    if not rate_per_minute:
        return None
    dsn = dsn if dsn is not None else rate_limit_dsn()
    if not dsn:
        return TokenBucket(rate_per_minute)
    from ..agent.pg import open_pool

    pool = None
    try:
        pool = open_pool(dsn, name="finsight-rate-limit", max_size=4)
        limiter = PostgresTokenBucket(pool, rate_per_minute)
    except Exception as exc:
        if pool is not None:
            pool.close()
        logger.warning("shared rate limiter unavailable at start-up, limiting per replica: %s", exc)
        return TokenBucket(rate_per_minute)
    logger.info("[startup] Rate limit shared through Postgres (%d/min per client).", rate_per_minute)
    return limiter
