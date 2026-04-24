from __future__ import annotations

from ..query_terms import INDUSTRY_TERMS


class QueryBuilder:
    def build(self, nlu_result: dict) -> dict:
        entity_names = self._unique(
            [item.get("canonical_name") for item in nlu_result.get("entities", []) if item.get("canonical_name")]
        )
        symbols = self._unique([item.get("symbol") for item in nlu_result.get("entities", []) if item.get("symbol")])
        normalized_query = nlu_result["normalized_query"]
        return {
            "query_id": nlu_result["query_id"],
            "normalized_query": normalized_query,
            "keywords": nlu_result.get("keywords", []),
            "entity_names": entity_names,
            "symbols": symbols,
            "industry_terms": self._extract_industry_terms(normalized_query),
            "source_plan": nlu_result.get("source_plan", []),
            "product_type": nlu_result["product_type"]["label"],
            "intent_labels": [item["label"] for item in nlu_result.get("intent_labels", [])],
            "topic_labels": [item["label"] for item in nlu_result.get("topic_labels", [])],
            "time_scope": nlu_result.get("time_scope", "unspecified"),
        }

    def _unique(self, values: list[str]) -> list[str]:
        unique_values: list[str] = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        return unique_values

    def _extract_industry_terms(self, normalized_query: str) -> list[str]:
        return [term for term in INDUSTRY_TERMS if term in normalized_query]
