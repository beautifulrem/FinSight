"""Evidence items produced by agent tools.

Tool outputs are normalized into ``AgentEvidence`` records so that answers can cite
stable ``evidence_id`` values and the verifier can check citations and numbers against
what tools actually returned. IDs follow the existing retrieval conventions
(``price_<symbol>``, ``fundamental_<symbol>``, ``macro_<code>``, ``<source_type>_<doc_id>``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

EvidenceKind = Literal["document", "structured"]

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.:-]+")
_MAX_EXCERPT_CHARS = 600


def safe_evidence_id(value: str) -> str:
    """Normalize an evidence id to the charset used by runtime document assets."""
    cleaned = _SAFE_ID.sub("_", value).strip("_")[:160]
    return cleaned or "evidence"


class AgentEvidence(BaseModel):
    evidence_id: str
    kind: EvidenceKind
    source_type: str
    title: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    provider: str | None = None
    as_of: str | None = None
    text_excerpt: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    produced_by: str | None = None

    @classmethod
    def from_document(cls, document: dict[str, Any], *, produced_by: str | None = None) -> AgentEvidence:
        text = document.get("text_excerpt") or document.get("summary") or document.get("body") or ""
        return cls(
            evidence_id=safe_evidence_id(str(document.get("evidence_id") or document.get("doc_id") or "doc")),
            kind="document",
            source_type=str(document.get("source_type") or "document"),
            title=document.get("title"),
            source_name=document.get("source_name"),
            source_url=document.get("source_url"),
            provider=document.get("provider"),
            as_of=document.get("publish_time") or document.get("retrieved_at"),
            text_excerpt=_truncate(str(text)) if text else None,
            payload={
                key: document[key]
                for key in ("entity_hits", "rank_score", "retrieval_score", "sentiment")
                if document.get(key) is not None
            },
            produced_by=produced_by,
        )

    @classmethod
    def from_structured(cls, item: dict[str, Any], *, produced_by: str | None = None) -> AgentEvidence:
        payload = item.get("payload") or {}
        return cls(
            evidence_id=safe_evidence_id(str(item.get("evidence_id") or "structured")),
            kind="structured",
            source_type=str(item.get("source_type") or "structured"),
            title=item.get("title") or payload.get("canonical_name") or payload.get("name"),
            source_name=item.get("source_name"),
            source_url=item.get("source_url"),
            provider=item.get("provider"),
            as_of=item.get("as_of") or item.get("retrieved_at"),
            payload=payload if isinstance(payload, dict) else {"value": payload},
            produced_by=produced_by,
        )

    def numbers(self) -> list[float]:
        """All numeric values carried by this evidence, used for numeric faithfulness checks."""
        values: list[float] = []
        _collect_numbers(self.payload, values)
        if self.text_excerpt:
            values.extend(extract_numbers(self.text_excerpt))
        return values

    def prompt_view(self) -> dict[str, Any]:
        """Compact representation handed to the LLM."""
        view: dict[str, Any] = {"evidence_id": self.evidence_id, "source_type": self.source_type}
        for key in ("title", "source_name", "as_of"):
            value = getattr(self, key)
            if value:
                view[key] = value
        if self.text_excerpt:
            view["text_excerpt"] = self.text_excerpt
        if self.payload:
            view["payload"] = _compact_payload(self.payload)
        return view


class EvidenceStore:
    """Per-run registry of evidence; keeps ids unique and stable within a run."""

    def __init__(self) -> None:
        self._items: dict[str, AgentEvidence] = {}

    def add(self, item: AgentEvidence) -> AgentEvidence:
        existing = self._items.get(item.evidence_id)
        if existing is None:
            self._items[item.evidence_id] = item
            return item
        if _fingerprint(existing) == _fingerprint(item):
            return existing
        suffix = 2
        while f"{item.evidence_id}_{suffix}" in self._items:
            suffix += 1
        renamed = item.model_copy(update={"evidence_id": f"{item.evidence_id}_{suffix}"})
        self._items[renamed.evidence_id] = renamed
        return renamed

    def extend(self, items: Iterable[AgentEvidence]) -> list[AgentEvidence]:
        return [self.add(item) for item in items]

    def get(self, evidence_id: str) -> AgentEvidence | None:
        return self._items.get(evidence_id)

    def ids(self) -> list[str]:
        return list(self._items)

    def items(self) -> list[AgentEvidence]:
        return list(self._items.values())

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._items

    def __len__(self) -> int:
        return len(self._items)


_NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z_\d]|\.[A-Za-z])")


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER.finditer(text):
        try:
            values.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return values


def _collect_numbers(value: Any, sink: list[float], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        sink.append(float(value))
    elif isinstance(value, str):
        stripped = value.strip().rstrip("%")
        try:
            sink.append(float(stripped.replace(",", "")))
        except ValueError:
            return
    elif isinstance(value, dict):
        for nested in value.values():
            _collect_numbers(nested, sink, depth + 1)
    elif isinstance(value, list | tuple):
        for nested in value[:500]:
            _collect_numbers(nested, sink, depth + 1)


def _compact_payload(payload: dict[str, Any], *, max_list: int = 5) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"history", "raw", "rows"} or str(key).startswith("_debug"):
            continue
        if isinstance(value, list) and len(value) > max_list:
            compact[key] = value[-max_list:]
            compact[f"{key}_count"] = len(value)
        else:
            compact[key] = value
    return compact


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_EXCERPT_CHARS else f"{text[: _MAX_EXCERPT_CHARS - 1]}…"


def _fingerprint(item: AgentEvidence) -> str:
    data = item.model_dump(exclude={"evidence_id", "produced_by"})
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
