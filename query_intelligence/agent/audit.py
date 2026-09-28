"""Audit log of refusals and compliance edits.

Every agent run that the guard refused, and every compliance rule that changed an answer, produces one audit
event. The events are for security and compliance review: *what kind* of intervention happened, for *whom*,
and in *which run*, without storing what the user wrote.

An event is one JSON object::

    {"at": "2026-09-28T06:12:03Z", "event": "compliance_edit", "category": "removed_trading_instruction",
     "trace_id": "9f2c…", "principal": "key:3b7e0a51c2d4", "session_hash": "5d1f0c2a9e4b",
     "query_hash": "a41c7e02f9d3", "route": "agent", "answer_source": "llm_agent", "prompt_version": "v3"}

* ``event``: ``refusal`` (``category``: ``prompt_injection`` / ``out_of_scope``) or ``compliance_edit``
  (``category``: the compliance rule, e.g. ``removed_trading_instruction``, ``conditional_prefix``,
  ``causal_caveat``, ``softened_judgment_or_causal_language``, ``market_freshness``,
  ``language_mismatch_fallback_to_template``).
* ``principal`` is the caller id the API already uses for tenancy (``key:`` + a SHA-256 prefix of the API key,
  or ``local``). ``query_hash`` and ``session_hash`` are 12-hex-digit HMAC-SHA256 prefixes (key:
  ``QI_AUDIT_HASH_KEY``; plain SHA-256 when unset). They let a reviewer group repeated attempts without
  keeping the text. No query, answer or document text is written.

Sinks: the ``finsight.audit`` logger (one JSON line per event, to the service log), a daily-rotated JSONL file
``QI_AUDIT_LOG_PATH`` (default ``outputs/audit/audit.jsonl``; ``off`` disables it) that keeps
``QI_AUDIT_RETENTION_DAYS`` files (default 30), and the Prometheus counter
``finsight_audit_events_total{event, category}``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import logging.handlers
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_AUDIT_PATH = "outputs/audit/audit.jsonl"
_OFF = {"", "off", "0", "false", "none"}

audit_logger = logging.getLogger("finsight.audit")


def short_hash(value: str | None, key: bytes | None = None) -> str | None:
    if not value:
        return None
    data = value.encode("utf-8")
    digest = hmac.new(key, data, hashlib.sha256) if key else hashlib.sha256(data)
    return digest.hexdigest()[:12]


def audit_events(trace: dict[str, Any], *, hash_key: bytes | None = None) -> list[dict[str, Any]]:
    """The audit events for one trace (empty for runs without a refusal or compliance edit)."""
    base = {
        "trace_id": trace.get("trace_id"),
        "principal": str(trace.get("owner") or "local"),
        "session_hash": short_hash(trace.get("session_id"), hash_key),
        "query_hash": short_hash(trace.get("query"), hash_key),
        "route": trace.get("route"),
        "answer_source": trace.get("answer_source"),
        "prompt_version": trace.get("prompt_version") or "none",
    }
    at = datetime.fromtimestamp(trace.get("started_at") or datetime.now(UTC).timestamp(), UTC)
    stamp = at.isoformat(timespec="seconds").replace("+00:00", "Z")
    events = []
    if trace.get("route") == "refuse":
        events.append({"at": stamp, "event": "refusal", "category": trace.get("refusal_category") or "other", **base})
    for note in dict.fromkeys(trace.get("compliance_notes") or []):
        events.append({"at": stamp, "event": "compliance_edit", "category": str(note).split(":")[0], **base})
    return events


class AuditTraceSink:
    """A ``TraceSink``: turns each finished trace into audit events (log line, JSONL file, counter)."""

    def __init__(
        self,
        *,
        registry: Any = None,
        path: str | Path | None = None,
        retention_days: int | None = None,
        hash_key: str | None = None,
    ) -> None:
        key = hash_key if hash_key is not None else os.getenv("QI_AUDIT_HASH_KEY", "")
        self.hash_key = key.encode("utf-8") if key else None
        target = str(path if path is not None else os.getenv("QI_AUDIT_LOG_PATH", DEFAULT_AUDIT_PATH)).strip()
        days = retention_days if retention_days is not None else int(os.getenv("QI_AUDIT_RETENTION_DAYS", "30"))
        self.file_handler: logging.Handler | None = None
        self._file_logger: logging.Logger | None = None
        if target.lower() not in _OFF:
            try:
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                handler = logging.handlers.TimedRotatingFileHandler(
                    target, when="midnight", backupCount=max(days, 1), encoding="utf-8", utc=True
                )
            except OSError as exc:  # e.g. a read-only filesystem: keep the log line and the counter
                logging.getLogger(__name__).warning("audit log file %s unavailable: %s", target, exc)
                target = "off"
        if target.lower() not in _OFF:
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.file_handler = handler
            self._file_logger = logging.getLogger(f"finsight.audit.file.{id(self)}")
            self._file_logger.propagate = False
            self._file_logger.setLevel(logging.INFO)
            self._file_logger.addHandler(handler)
        self.counter = None
        if registry is not None:
            from prometheus_client import Counter

            self.counter = Counter(
                "finsight_audit_events_total",
                "Audit events: guard refusals and compliance edits, by category.",
                ["event", "category"],
                registry=registry,
            )

    def emit(self, trace: dict[str, Any]) -> None:
        for event in audit_events(trace, hash_key=self.hash_key):
            line = json.dumps(event, ensure_ascii=False, sort_keys=True)
            audit_logger.info(line)
            if self._file_logger is not None:
                self._file_logger.info(line)
            if self.counter is not None:
                self.counter.labels(event=event["event"], category=event["category"]).inc()

    def close(self) -> None:
        if self.file_handler is not None and self._file_logger is not None:
            self._file_logger.removeHandler(self.file_handler)
            self.file_handler.close()
