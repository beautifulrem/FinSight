from __future__ import annotations

from ..query_terms import INDUSTRY_TERMS


class QueryBuilder:
    def build(self, nlu_result: dict) -> dict:
        entities = nlu_result.get("entities", [])
        entity_names = self._unique(
            [item.get("canonical_name") for item in entities if item.get("canonical_name")]
        )
        symbols = self._unique([item.get("symbol") for item in entities if item.get("symbol")])
        entity_targets = self._build_entity_targets(entities)
        normalized_query = nlu_result["normalized_query"]
        return {
            "query_id": nlu_result["query_id"],
            "normalized_query": normalized_query,
            "keywords": nlu_result.get("keywords", []),
            "entity_names": entity_names,
            "symbols": symbols,
            "entity_targets": entity_targets,
            "industry_terms": self._extract_industry_terms(normalized_query),
            "source_plan": nlu_result.get("source_plan", []),
            "product_type": nlu_result["product_type"]["label"],
            "intent_labels": [item["label"] for item in nlu_result.get("intent_labels", [])],
            "topic_labels": [item["label"] for item in nlu_result.get("topic_labels", [])],
            "time_scope": nlu_result.get("time_scope", "unspecified"),
        }

    def _build_entity_targets(self, entities: list[dict]) -> list[dict]:
        targets: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for item in entities:
            symbol = str(item.get("symbol") or "").strip()
            canonical_name = str(item.get("canonical_name") or "").strip()
            entity_type = str(item.get("entity_type") or "").strip()
            if not symbol and entity_type in {"sector", "financial_metric", "macro_indicator"}:
                continue
            if not (symbol or canonical_name):
                continue
            key = (symbol, "" if symbol else canonical_name)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                {
                    "symbol": symbol,
                    "canonical_name": canonical_name or symbol,
                    "entity_type": entity_type,
                }
            )
        return targets

    def _unique(self, values: list[str]) -> list[str]:
        unique_values: list[str] = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        return unique_values

    def _extract_industry_terms(self, normalized_query: str) -> list[str]:
        return [term for term in INDUSTRY_TERMS if term in normalized_query]
