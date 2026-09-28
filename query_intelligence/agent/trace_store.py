"""Trace store shared by all replicas, behind ``GET /agent/traces`` and ``GET /agent/traces/{trace_id}``.

``RecentTraceStore`` (``telemetry.py``) is a per-process ring, so with several replicas the run inspector
only finds a trace if the request lands on the replica that ran it. ``PostgresTraceStore`` keeps the same
interface (``emit`` / ``get`` / ``recent``, owner-scoped) on a Postgres table:

* one row per trace: ``trace_id``, ``owner`` (the API-key principal), ``session_id``, ``created_at`` (the run's
  start time), the full trace and its list summary (``summarize_trace``), both JSONB; ``recent`` reads only
  the summaries;
* bounded retention: rows older than ``QI_AGENT_TRACE_RETENTION_DAYS`` (default 14) and rows beyond the
  newest ``QI_AGENT_TRACE_MAX_ROWS`` (default 50000) are deleted every ``prune_every`` emits (default 50);
* failure isolation: every trace is also kept in a local ``RecentTraceStore``; if Postgres is unreachable,
  writes are logged and skipped and reads fall back to the local ring, so tracing never breaks an answer.

``build_trace_store`` chooses Postgres when ``QI_AGENT_TRACE_DB`` (or, by default, a Postgres
``QI_AGENT_CHECKPOINT_DB``) is set, else the in-memory ring. The HTTP API is unchanged.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from .pg import check_identifier, open_pool, store_dsn
from .telemetry import RecentTraceStore, summarize_trace

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "finsight_agent_traces"


class PostgresTraceStore:
    def __init__(
        self,
        pool: Any,
        *,
        table: str = DEFAULT_TABLE,
        max_rows: int | None = None,
        retention_days: float | None = None,
        prune_every: int = 50,
        fallback: RecentTraceStore | None = None,
    ) -> None:
        self.pool = pool
        self.table = check_identifier(table)
        self.max_rows = max_rows if max_rows is not None else int(os.getenv("QI_AGENT_TRACE_MAX_ROWS", "50000"))
        days = retention_days if retention_days is not None else float(os.getenv("QI_AGENT_TRACE_RETENTION_DAYS", "14"))
        self.retention_s = max(days, 0.0) * 86400
        self.prune_every = max(1, prune_every)
        self.fallback = fallback or RecentTraceStore()
        self.write_errors = 0
        self._emits = 0
        self._lock = threading.Lock()
        self.setup()

    def setup(self) -> None:
        t = self.table
        with self.pool.connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    trace_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    session_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    summary JSONB NOT NULL,
                    trace JSONB NOT NULL
                )"""
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_owner ON {t} (owner, created_at DESC)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_session ON {t} (owner, session_id, created_at DESC)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_created ON {t} (created_at)")

    # ------------------------------------------------------------------------------------------ sink

    def emit(self, trace: dict[str, Any]) -> None:
        trace_id = trace.get("trace_id")
        if not trace_id:
            return
        self.fallback.emit(trace)
        from psycopg.types.json import Jsonb

        try:
            with self.pool.connection() as conn:
                conn.execute(
                    f"""
                    INSERT INTO {self.table} (trace_id, owner, session_id, created_at, summary, trace)
                    VALUES (%s, %s, %s, COALESCE(to_timestamp(%s), now()), %s, %s)
                    ON CONFLICT (trace_id) DO UPDATE SET summary = EXCLUDED.summary, trace = EXCLUDED.trace
                    """,
                    (
                        str(trace_id),
                        str(trace.get("owner") or "local"),
                        trace.get("session_id"),
                        trace.get("started_at"),
                        Jsonb(summarize_trace(trace)),
                        Jsonb(trace, dumps=_dumps),
                    ),
                )
        except Exception:
            self.write_errors += 1
            logger.warning("trace %s not written to Postgres (kept in memory)", trace_id, exc_info=True)
            return
        with self._lock:
            self._emits += 1
            due = self._emits % self.prune_every == 0
        if due:
            self.prune()

    # ------------------------------------------------------------------------------------------ reads

    def get(self, trace_id: str, *, owner: str | None = None) -> dict[str, Any] | None:
        query = f"SELECT trace FROM {self.table} WHERE trace_id = %s"
        args: list[Any] = [trace_id]
        if owner is not None:
            query += " AND owner = %s"
            args.append(owner)
        try:
            with self.pool.connection() as conn:
                row = conn.execute(query, args).fetchone()
        except Exception:
            logger.warning("trace lookup in Postgres failed; using the local store", exc_info=True)
            return self.fallback.get(trace_id, owner=owner)
        if row:
            return row[0]
        # A trace whose write failed is only in the local ring; otherwise a missing row means unknown or pruned.
        return self.fallback.get(trace_id, owner=owner) if self.write_errors else None

    def recent(
        self, limit: int = 50, *, session_id: str | None = None, owner: str | None = None
    ) -> list[dict[str, Any]]:
        where, args = [], []
        if owner is not None:
            where.append("owner = %s")
            args.append(owner)
        if session_id:
            where.append("session_id = %s")
            args.append(session_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        try:
            with self.pool.connection() as conn:
                rows = conn.execute(
                    f"SELECT summary FROM {self.table} {clause} ORDER BY created_at DESC, trace_id DESC LIMIT %s",
                    [*args, max(0, limit)],
                ).fetchall()
        except Exception:
            logger.warning("trace listing from Postgres failed; using the local store", exc_info=True)
            return self.fallback.recent(limit, session_id=session_id, owner=owner)
        return [row[0] for row in rows]

    # ------------------------------------------------------------------------------------------ retention

    def prune(self) -> int:
        """Apply the age and row-count limits; returns the number of deleted traces."""
        deleted = 0
        try:
            with self.pool.connection() as conn:
                if self.retention_s:
                    deleted += conn.execute(
                        f"DELETE FROM {self.table} WHERE created_at < now() - make_interval(secs => %s)",
                        (self.retention_s,),
                    ).rowcount
                if self.max_rows:
                    deleted += conn.execute(
                        f"""
                        DELETE FROM {self.table} WHERE trace_id IN (
                            SELECT trace_id FROM {self.table} ORDER BY created_at DESC, trace_id DESC OFFSET %s
                        )""",
                        (self.max_rows,),
                    ).rowcount
        except Exception:
            logger.warning("trace pruning failed", exc_info=True)
        return deleted

    def close(self) -> None:
        self.pool.close()


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


def build_trace_store(trace_dir: str | Path | None = None) -> RecentTraceStore | PostgresTraceStore:
    """Postgres when configured and reachable, else the in-process ring (which also reads old JSON traces)."""
    local = RecentTraceStore(trace_dir=trace_dir)
    dsn = store_dsn("QI_AGENT_TRACE_DB")
    if not dsn:
        return local
    started = time.perf_counter()
    try:
        store = PostgresTraceStore(open_pool(dsn, name="agent-traces"), fallback=local)
    except Exception as exc:
        logger.warning("[startup] Postgres trace store unavailable (%s); traces stay in this process.", exc)
        return local
    logger.info("[startup] Agent traces are stored in Postgres (%.1fs).", time.perf_counter() - started)
    return store
