"""Freshness and provenance metadata attached to every returned record.

A provenance record answers "where did this number come from, how old is it, and why": source id and
label, the endpoint, when it was fetched, the as-of date of the data, whether it is live or a
shipped snapshot, and the fallback reason when a secondary source, the last-known-good cache, or the
snapshot was used.

Provenance deliberately contains no numeric values (booleans and strings only) so that it cannot make
an unsupported number look "traceable" to the agent's numeric-faithfulness check.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Any

from .catalog import source_label

LIVE = "live"
LIVE_FALLBACK = "live_fallback"
LAST_KNOWN_GOOD = "last_known_good"
SNAPSHOT = "snapshot"

# A record older than this (calendar days since its as-of date) is flagged ``stale``. Windows are
# generous enough to absorb weekends and the longest A-share holiday closures.
FRESHNESS_WINDOW_DAYS: dict[str, int] = {
    "market": 10,
    "index": 10,
    "fund_nav": 10,
    "valuation": 10,
    "index_valuation": 10,
    "industry": 10,
    "fundamentals": 200,
    "macro_daily": 10,
    "macro_monthly": 75,
    "news": 30,
    "announcement": 90,
}


def freshness(kind: str, as_of: Any, *, today: date | None = None) -> str:
    """``fresh`` / ``stale`` / ``unknown`` for a record of ``kind`` dated ``as_of``."""
    window = FRESHNESS_WINDOW_DAYS.get(kind)
    parsed = parse_date(as_of)
    if window is None or parsed is None:
        return "unknown"
    age = ((today or date.today()) - parsed).days
    return "fresh" if age <= window else "stale"


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if len(text) < 8:
        return None
    for candidate in (text[:10], text[:8]):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
    return None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_provenance(
    *,
    source: str | None,
    kind: str,
    as_of: Any,
    mode: str = LIVE,
    endpoint: str | None = None,
    fetched_at: str | None = None,
    fallback_reason: str | None = None,
    attempts: Iterable[str] = (),
    cache_hit: bool = False,
    today: date | None = None,
) -> dict[str, Any]:
    as_of_text = _as_of_text(as_of)
    record: dict[str, Any] = {
        "source": source,
        "source_label": source_label(source),
        "endpoint": endpoint,
        "is_live": mode in {LIVE, LIVE_FALLBACK, LAST_KNOWN_GOOD},
        "mode": mode,
        "fetched_at": fetched_at or (utc_now_iso() if mode != SNAPSHOT else None),
        "as_of": as_of_text,
        "freshness": freshness(kind, as_of_text, today=today),
        "fallback_reason": fallback_reason,
        "attempts": list(attempts),
        "cache_hit": cache_hit,
    }
    record["note"] = describe(record)
    return record


def snapshot_provenance(
    *,
    kind: str,
    as_of: Any,
    reason: str,
    source_name: str | None = None,
    today: date | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Provenance for a record served from the shipped offline snapshot.

    ``snapshot`` is the ``_snapshot`` metadata of a record from the snapshot extension
    (``data/snapshot/structured_data_ext.json``): its file, version, upstream endpoint and fetch time.
    """
    meta = snapshot or {}
    record = build_provenance(
        source="offline_snapshot",
        kind=kind,
        as_of=as_of,
        mode=SNAPSHOT,
        endpoint=str(meta.get("file") or "data/structured_data.json"),
        fallback_reason=reason,
        today=today,
    )
    if meta.get("fetched_at"):
        record["fetched_at"] = str(meta["fetched_at"])
    if meta.get("version"):
        record["snapshot_version"] = str(meta["version"])
    if meta.get("endpoint"):
        record["original_endpoint"] = str(meta["endpoint"])
    if source_name:
        record["original_source"] = str(source_name)
    record["source_label"] = "离线快照"
    record["note"] = describe(record)
    return record


def corpus_provenance(*, as_of: Any, source_name: str | None = None) -> dict[str, Any]:
    """Provenance for a document retrieved from the shipped local document corpus."""
    record = build_provenance(
        source="local_corpus", kind="document", as_of=as_of, mode=SNAPSHOT, endpoint="data/runtime/documents"
    )
    record["source_label"] = "本地文档库"
    if source_name:
        record["original_source"] = str(source_name)
    record["note"] = describe(record)
    return record


def with_mode(provenance: dict[str, Any], *, mode: str, fallback_reason: str | None) -> dict[str, Any]:
    updated = {**provenance, "mode": mode, "fallback_reason": fallback_reason}
    updated["is_live"] = mode in {LIVE, LIVE_FALLBACK, LAST_KNOWN_GOOD}
    if mode in {LAST_KNOWN_GOOD}:
        updated["cache_hit"] = True
    updated["note"] = describe(updated)
    return updated


def describe(record: dict[str, Any]) -> str:
    """One-line Chinese explanation: 数据来自X，截至Y；因Z降级."""
    label = record.get("source_label") or record.get("source")
    parts = [f"数据来自{label}" if label else "未获取到实时数据"]
    if record.get("as_of"):
        parts.append(f"截至{record['as_of']}")
    mode = record.get("mode")
    if mode == SNAPSHOT:
        parts.append("非实时（离线快照）")
    elif mode == LAST_KNOWN_GOOD:
        parts.append(f"沿用最近一次成功获取的实时数据（获取于{record.get('fetched_at')}）")
    if record.get("freshness") == "stale":
        parts.append("数据可能已过时")
    text = "，".join(parts)
    if record.get("fallback_reason"):
        text += f"；因{reason_zh(record['fallback_reason'])}降级"
    return text


_OUTCOME_ZH = {"circuit_open": "熔断中", "timeout": "超时", "empty": "无数据"}
_REASON_ZH = (
    ("live market data disabled", "未开启实时行情"),
    ("live macro data disabled", "未开启实时宏观数据"),
    ("live market fetch failed", "实时行情获取失败"),
    ("live market sources returned no rows", "实时行情源均无数据"),
    ("live macro unavailable", "实时宏观数据不可用"),
    ("live source returned no data", "实时源未返回该数据"),
    ("indicator not fetched from live sources", "该指标未从实时源获取"),
    ("industry board history unavailable", "行业指数行情不可用，仅有行业名称"),
    ("live industry index unavailable", "实时行业指数不可用，沿用离线快照（旧数据，勿当作今日行情）"),
    ("no live source returned data", "所有实时源均无数据"),
)


def reason_zh(reason: str) -> str:
    """Chinese rendering of a fallback reason (attempt labels such as ``sina.kline:timeout``)."""
    for prefix, text in _REASON_ZH:
        if reason.startswith(prefix):
            return text
    parts = []
    for item in reason.split("; "):
        source, _, outcome = item.rpartition(":")
        if not source:
            parts.append(item)
            continue
        label = source_label(source) or source
        outcome_text = _OUTCOME_ZH.get(outcome, "请求失败" if outcome.startswith("error") else outcome)
        parts.append(f"{label}{outcome_text}")
    return "；".join(dict.fromkeys(parts)) if parts else reason


def _as_of_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if text.isdigit():
        # Compact dates (20260924) would look like plain numbers to numeric-faithfulness checks.
        parsed = parse_date(text)
        return parsed.isoformat() if parsed else None
    return text or None
