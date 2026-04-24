from __future__ import annotations


DOCUMENT_SOURCE_TYPES = {"news", "announcement", "faq", "product_doc", "research_note"}


class DocumentSelector:
    def select(self, ranked_docs: list[dict], source_plan: list[str], top_k: int) -> list[dict]:
        preferred_sources = [source for source in source_plan if source in DOCUMENT_SOURCE_TYPES]
        selected: list[dict] = []
        seen_ids: set[str] = set()

        for source in preferred_sources:
            source_docs = [
                doc
                for doc in ranked_docs
                if doc["source_type"] == source and doc["evidence_id"] not in seen_ids and self._is_usable(doc)
            ]
            if source_docs:
                doc = max(source_docs, key=self._source_candidate_priority)
                selected.append(doc)
                seen_ids.add(doc["evidence_id"])
            if len(selected) >= top_k:
                return selected[:top_k]

        for doc in ranked_docs:
            if doc["evidence_id"] in seen_ids or not self._is_usable(doc):
                continue
            selected.append(doc)
            seen_ids.add(doc["evidence_id"])
            if len(selected) >= top_k:
                break
        return selected[:top_k]

    def _is_usable(self, doc: dict) -> bool:
        retrieval_score = doc.get("retrieval_score")
        if retrieval_score is None:
            return float(doc.get("rank_score", 0.0) or 0.0) >= 0.55
        return float(retrieval_score or 0.0) > 0.0

    def _source_candidate_priority(self, doc: dict) -> tuple:
        source_name = str(doc.get("source_name") or "").lower()
        has_url = bool(doc.get("source_url"))
        non_seed = "seed" not in source_name and not source_name.startswith("public_")
        return (
            1 if has_url else 0,
            1 if non_seed else 0,
            float(doc.get("rank_score", 0.0) or 0.0),
            float(doc.get("retrieval_score", 0.0) or 0.0),
        )
