"""Shared runtime context for agent tools.

Tools reuse the already-loaded NLU and retrieval components of a
``QueryIntelligenceService`` instead of rebuilding models or duplicating provider logic.
Data tools drive ``RetrievalPipeline.fetch_structured`` / ``retrieve_documents`` with a
narrowed query bundle (a single source type and explicit targets).
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .base import ToolFailure

if TYPE_CHECKING:
    from ...nlu.pipeline import NLUPipeline
    from ...retrieval.pipeline import RetrievalPipeline
    from ...service import QueryIntelligenceService

SYMBOL_PATTERN = re.compile(r"^\d{6}\.(SH|SZ|BJ)$", re.IGNORECASE)
MIN_TARGET_CONFIDENCE = 0.8
_PRODUCT_TYPES = {"stock", "etf", "fund", "index"}


@dataclass
class ResolvedTarget:
    symbol: str
    name: str
    entity_type: str

    @property
    def product_type(self) -> str:
        return self.entity_type if self.entity_type in _PRODUCT_TYPES else "stock"


@dataclass
class ToolContext:
    nlu_pipeline: NLUPipeline
    retrieval_pipeline: RetrievalPipeline
    structured_ttl_s: float = 60.0
    _structured_cache: dict[str, tuple[float, list[dict]]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _symbol_index: dict[str, dict[str, str]] | None = field(default=None, repr=False)

    @classmethod
    def from_service(cls, service: QueryIntelligenceService) -> ToolContext:
        return cls(nlu_pipeline=service.nlu_pipeline, retrieval_pipeline=service.retrieval_pipeline)

    # ---- entity helpers -------------------------------------------------

    def resolve_entities(self, text: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        normalized, _trace = self.nlu_pipeline.normalizer.normalize(text)
        resolver = self.nlu_pipeline.entity_resolver
        entities, comparison_targets, trace = resolver.resolve_exact(normalized)
        if not entities:
            entities, comparison_targets, trace = resolver.resolve(normalized)
        return entities, comparison_targets, trace

    def resolve_target(self, target: str) -> ResolvedTarget:
        """Accept a ticker (``600519.SH``) or a name/alias (``贵州茅台``) and return one listed target."""
        text = target.strip()
        if not text:
            raise ToolFailure("invalid_arguments", "target must not be empty")
        if SYMBOL_PATTERN.match(text):
            symbol = text.upper()
            row = self._symbols().get(symbol)
            if row is None:
                return ResolvedTarget(symbol=symbol, name=symbol, entity_type="stock")
            return ResolvedTarget(
                symbol=symbol, name=row["canonical_name"], entity_type=row.get("entity_type") or "stock"
            )
        entities, _targets, _trace = self.resolve_entities(text)
        listed = [entity for entity in entities if entity.get("symbol")]
        if not listed:
            raise ToolFailure("not_found", f"no listed security matches {text!r}")
        confident = [entity for entity in listed if float(entity.get("confidence") or 0.0) >= MIN_TARGET_CONFIDENCE]
        if not confident:
            candidate = listed[0]
            raise ToolFailure(
                "not_found",
                f"no confident match for {text!r}; closest candidate is {candidate['canonical_name']} "
                f"({candidate['symbol']}), confirm it with resolve_entity or pass the ticker",
            )
        listed = confident
        best = listed[0]
        return ResolvedTarget(
            symbol=str(best["symbol"]).upper(),
            name=str(best["canonical_name"]),
            entity_type=str(best.get("entity_type") or "stock"),
        )

    def _symbols(self) -> dict[str, dict[str, str]]:
        if self._symbol_index is None:
            index: dict[str, dict[str, str]] = {}
            for row in self.nlu_pipeline.entity_resolver.entities:
                symbol = (row.get("symbol") or "").upper()
                if symbol and symbol not in index:
                    index[symbol] = row
            self._symbol_index = index
        return self._symbol_index

    # ---- retrieval helpers ----------------------------------------------

    def bundle(
        self,
        *,
        source_plan: list[str],
        query: str = "",
        symbols: list[str] | None = None,
        names: list[str] | None = None,
        product_type: str = "stock",
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        from ...query_terms import INDUSTRY_TERMS

        normalized_query = query or " ".join(names or symbols or [])
        return {
            "query_id": "agent-tool",
            "normalized_query": normalized_query,
            "keywords": list(keywords or []),
            "entity_names": list(names or []),
            "symbols": list(symbols or []),
            "industry_terms": [term for term in INDUSTRY_TERMS if term in normalized_query],
            "source_plan": list(source_plan),
            "product_type": product_type,
            "intent_labels": [],
            "topic_labels": [],
            "time_scope": "unspecified",
        }

    def fetch_structured(self, bundle: dict[str, Any]) -> list[dict]:
        key = repr(
            (
                tuple(bundle["source_plan"]),
                tuple(bundle["symbols"]),
                tuple(bundle["entity_names"]),
                bundle["product_type"],
                bundle["normalized_query"],
            )
        )
        now = time.monotonic()
        with self._lock:
            cached = self._structured_cache.get(key)
            if cached and cached[0] > now:
                return cached[1]
        items = self.retrieval_pipeline.fetch_structured(bundle)
        if not provider_warnings(items):
            with self._lock:
                self._structured_cache[key] = (now + self.structured_ttl_s, items)
        return items

    def retrieve_documents(self, bundle: dict[str, Any], top_k: int) -> list[dict]:
        documents, _groups, _total = self.retrieval_pipeline.retrieve_documents(bundle, top_k)
        return documents[:top_k]


def provider_warnings(items: list[dict]) -> list[str]:
    warnings: list[str] = []
    for item in items:
        if item.get("source_type") == "provider_warning":
            warnings.extend(item.get("payload", {}).get("provider_warnings", []))
    return warnings
