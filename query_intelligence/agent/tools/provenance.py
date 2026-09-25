"""Typed view of the provenance metadata attached to live and snapshot records.

The retrieval layer stores provenance under ``payload["provenance"]`` (see
``query_intelligence/integrations/sources/provenance.py``). Tools surface it as an optional
``provenance`` field so the agent and the UI can say "数据来自X，截至Y，因Z降级".
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str | None = Field(default=None, description="Source id, e.g. 'sina.kline' or 'offline_snapshot'.")
    source_label: str | None = Field(default=None, description="Human-readable source name.")
    endpoint: str | None = None
    is_live: bool = False
    mode: str | None = Field(default=None, description="live | live_fallback | last_known_good | snapshot")
    fetched_at: str | None = Field(default=None, description="UTC time the live source was queried.")
    as_of: str | None = Field(default=None, description="Date the data refers to.")
    freshness: str | None = Field(default=None, description="fresh | stale | unknown")
    fallback_reason: str | None = None
    attempts: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    note: str | None = Field(default=None, description="One-line explanation for display.")


def provenance_from(payload: Any, key: str = "provenance") -> SourceProvenance | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, dict):
        return None
    try:
        return SourceProvenance.model_validate(value)
    except ValueError:
        return None
