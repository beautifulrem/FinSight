"""Deep readiness for ``GET /ready`` (``GET /health`` stays a cheap liveness check).

A replica is ready to take traffic only when every check passes:

* ``checkpointer``: the agent service can be built (this opens the SQLite file or the Postgres pool)
  and its checkpoint store answers ``SELECT 1``; a SQLite store must also accept a write lock, so a
  read-only or root-owned state volume makes the pod unready instead of failing every chat request.
* ``model_config``: when an LLM key is configured, the endpoint URL, model id and timeout are sane and
  the agent actually holds an LLM client; without a key the agent runs in deterministic mode (ready).
* ``retrieval_index``: the document corpus is loaded and its TF-IDF matrix matches it (or, for the
  Postgres document repository, the connection answers ``SELECT 1``); structured data is loaded.

Checks run in worker threads with a deadline (``QI_READY_TIMEOUT_S``, default 5 s). A check that is still
running from an earlier probe (for example a Postgres pool waiting for a dead host) is reported as failed
without starting a second copy, so slow probes cannot pile up threads.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any
from urllib.parse import urlparse

CheckFn = Callable[[], dict[str, Any]]

DEFAULT_TIMEOUT_S = 5.0
_DSN_CREDENTIALS = re.compile(r"(\w+://)[^/@\s]+@")
_PASSWORD_KV = re.compile(r"(password\s*=\s*)\S+", re.IGNORECASE)


def _redact(text: str) -> str:
    return _PASSWORD_KV.sub(r"\1***", _DSN_CREDENTIALS.sub(r"\1***@", text))


def describe_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {_redact(message)[:200]}" if message else type(exc).__name__


class ReadinessChecker:
    def __init__(self, checks: dict[str, CheckFn], *, timeout_s: float | None = None) -> None:
        self.checks = checks
        if timeout_s is None:
            timeout_s = float(os.getenv("QI_READY_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)) or DEFAULT_TIMEOUT_S)
        self.timeout_s = max(0.1, timeout_s)
        self._executor = ThreadPoolExecutor(max_workers=max(1, len(checks)), thread_name_prefix="readiness")
        self._inflight: dict[str, Future] = {}
        self._lock = threading.Lock()

    def run(self) -> tuple[bool, dict[str, Any]]:
        started = time.perf_counter()
        futures: dict[str, Future | None] = {}
        with self._lock:
            for name, check in self.checks.items():
                previous = self._inflight.get(name)
                if previous is not None and not previous.done():
                    futures[name] = None
                    continue
                future = self._executor.submit(check)
                self._inflight[name] = future
                futures[name] = future
        deadline = started + self.timeout_s
        results: dict[str, dict[str, Any]] = {}
        for name, future in futures.items():
            if future is None:
                results[name] = {"ok": False, "error": "previous check is still running"}
                continue
            try:
                detail = future.result(timeout=max(0.0, deadline - time.perf_counter()))
                results[name] = {"ok": bool(detail.get("ok", True)), **{k: v for k, v in detail.items() if k != "ok"}}
            except FutureTimeout:
                results[name] = {"ok": False, "error": f"timed out after {self.timeout_s:g} s"}
            except Exception as exc:
                results[name] = {"ok": False, "error": describe_error(exc)}
        ready = all(row["ok"] for row in results.values())
        return ready, {
            "status": "ready" if ready else "not_ready",
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "checks": results,
        }


# ---- individual checks ----


def check_checkpointer(get_agent: Callable[[], Any]) -> dict[str, Any]:
    """Build the agent (opens the checkpoint store) and touch the store."""
    agent = get_agent()
    return probe_checkpointer(getattr(agent, "checkpointer", None))


def probe_checkpointer(saver: Any) -> dict[str, Any]:
    if saver is None:
        return {"ok": False, "error": "agent has no checkpointer"}
    kind = type(saver).__name__
    conn = getattr(saver, "conn", None)
    if conn is None:
        # InMemorySaver and other process-local stores: nothing external to reach.
        return {"ok": True, "backend": kind, "persistent": False}
    if isinstance(conn, sqlite3.Connection):
        lock = getattr(saver, "lock", None) or threading.Lock()
        with lock:
            # Rewrite the header's user_version inside a rolled-back transaction: a read-only file or mount
            # fails here ("attempt to write a readonly database") instead of on the first chat request.
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                conn.execute(f"PRAGMA user_version = {int(version)}")
            finally:
                conn.rollback()
        return {"ok": True, "backend": kind, "persistent": True, "database": "sqlite"}
    timeout = float(os.getenv("QI_READY_DB_TIMEOUT_S", "3") or 3)
    if hasattr(conn, "connection") and hasattr(conn, "get_stats"):
        # psycopg_pool.ConnectionPool
        with conn.connection(timeout=timeout) as connection:
            connection.execute("SELECT 1").fetchone()
        stats = conn.get_stats()
        return {
            "ok": True,
            "backend": kind,
            "persistent": True,
            "database": "postgres",
            "pool": {"size": stats.get("pool_size"), "available": stats.get("pool_available")},
        }
    lock = getattr(saver, "lock", None) or threading.Lock()
    with lock:
        conn.execute("SELECT 1").fetchone()
    return {"ok": True, "backend": kind, "persistent": True}


def check_model_config(chatbot_config: dict[str, Any], peek_agent: Callable[[], Any] | None = None) -> dict[str, Any]:
    """``peek_agent`` returns the agent if it is already built (it never builds one)."""
    section = chatbot_config.get("deepseek") or {}
    from ..agent.llm import DeepSeekToolClient

    client = DeepSeekToolClient.from_chatbot_config(chatbot_config)
    if not client.configured:
        return {"ok": True, "llm": "not configured (deterministic agent)"}
    problems: list[str] = []
    parsed = urlparse(str(section.get("base_url") or "https://api.deepseek.com"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        problems.append("base_url must be an absolute http(s) URL")
    if not client.model.strip():
        problems.append("model is empty")
    if client.timeout_s <= 0:
        problems.append("timeout_seconds must be positive")
    fallbacks = [model.strip() for model in os.getenv("QI_LLM_FALLBACK_MODELS", "").split(",") if model.strip()]
    agent = peek_agent() if peek_agent is not None else None
    if agent is not None and not problems and getattr(getattr(agent, "runtime", None), "llm", None) is None:
        problems.append("an API key is configured but the agent has no LLM client")
    detail: dict[str, Any] = {
        "ok": not problems,
        "llm": "configured",
        "model": client.model,
        "endpoint_host": parsed.netloc or None,
        "fallback_models": fallbacks,
        "reasoning_style": str(section.get("reasoning_style") or "auto"),
    }
    if problems:
        detail["error"] = "; ".join(problems)
    return detail


def check_retrieval_index(service: Any) -> dict[str, Any]:
    pipeline = getattr(service, "retrieval_pipeline", None)
    if pipeline is None:
        # Injected services (tests, embedded use) without the classic retrieval pipeline.
        return {"ok": True, "backend": "none", "note": "service has no retrieval pipeline"}
    retriever = getattr(pipeline, "doc_retriever", None)
    detail: dict[str, Any] = {"ok": True, "backend": type(retriever).__name__}
    problems: list[str] = []
    documents = getattr(retriever, "documents", None)
    if documents is not None:
        matrix = getattr(retriever, "doc_matrix", None)
        rows = getattr(matrix, "shape", (None,))[0]
        detail.update(documents=len(documents), index_rows=rows)
        if not documents:
            problems.append("document corpus is empty")
        if getattr(retriever, "vectorizer", None) is None or rows != len(documents):
            problems.append("TF-IDF index is not fitted to the corpus")
    elif hasattr(retriever, "connection"):
        retriever.connection.execute("SELECT 1").fetchone()
        detail["database"] = "postgres"
    else:
        problems.append("unknown document retriever")
    structured = getattr(getattr(pipeline, "sql_retriever", None), "structured_data", None)
    if isinstance(structured, dict):
        detail["structured_tables"] = len(structured)
        if not structured:
            problems.append("structured data is empty")
    if problems:
        detail.update(ok=False, error="; ".join(problems))
    return detail
