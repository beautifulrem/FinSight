from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

MIN_RETRIEVAL_SCORE = {
    "news": 0.02,
    "announcement": 0.02,
    "research_note": 0.05,
    "faq": 0.05,
    "product_doc": 0.05,
}


@dataclass
class DocumentRetriever:
    documents: list[dict]

    def __post_init__(self) -> None:
        corpus = [f"{doc['title']} {doc['summary']} {doc['body']}" for doc in self.documents]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query_bundle: dict, top_k: int = 20) -> list[dict]:
        query_text = " ".join(
            [query_bundle["normalized_query"], *query_bundle["entity_names"], *query_bundle["keywords"], *query_bundle["symbols"]]
        ).strip()
        if not query_text:
            return []

        query_vector = self.vectorizer.transform([query_text])
        scores = linear_kernel(query_vector, self.doc_matrix).flatten()

        hits = []
        allowed_sources = self._allowed_sources(query_bundle)
        if not allowed_sources:
            return []
        for doc, score in zip(self.documents, scores, strict=False):
            if doc["source_type"] not in allowed_sources:
                continue
            requires_entity_hit = doc["source_type"] in {"news", "announcement", "research_note"}
            if requires_entity_hit and query_bundle["symbols"] and not any(symbol in doc.get("entity_symbols", []) for symbol in query_bundle["symbols"]):
                continue
            if self._should_skip_entity_bound_doc(query_bundle, doc):
                continue
            retrieval_score = float(round(max(score + self._term_hit_bonus(query_bundle, doc), 0.0), 4))
            if retrieval_score < MIN_RETRIEVAL_SCORE.get(doc["source_type"], 0.0):
                continue
            hits.append({**doc, "retrieval_score": retrieval_score})

        hits.sort(key=lambda item: item["retrieval_score"], reverse=True)
        return self._diversify_hits(hits, query_bundle.get("source_plan", []), top_k)

    def _allowed_sources(self, query_bundle: dict) -> set[str]:
        source_plan = query_bundle.get("source_plan")
        if source_plan is None:
            return {"news", "announcement", "faq", "product_doc", "research_note"}
        return set(source_plan).intersection({"news", "announcement", "faq", "product_doc", "research_note"})

    def _should_skip_entity_bound_doc(self, query_bundle: dict, doc: dict) -> bool:
        if query_bundle.get("symbols"):
            return False
        if not doc.get("entity_symbols"):
            return False
        intent_labels = set(query_bundle.get("intent_labels", []))
        topic_labels = set(query_bundle.get("topic_labels", []))
        product_type = query_bundle.get("product_type")
        industry_query = bool(query_bundle.get("industry_terms")) or "industry" in topic_labels
        if product_type in {"stock", "index", "generic_market"}:
            return not industry_query
        generic_product_query = bool(intent_labels.intersection({"product_info", "trading_rule_fee"})) or "product_mechanism" in topic_labels
        if product_type in {"fund", "etf"}:
            return generic_product_query
        return False

    def _term_hit_bonus(self, query_bundle: dict, doc: dict) -> float:
        terms = [
            *query_bundle.get("industry_terms", []),
            *query_bundle.get("keywords", []),
            *query_bundle.get("entity_names", []),
        ]
        terms = [term for term in terms if len(str(term).strip()) >= 2]
        if not terms:
            return 0.0
        text = " ".join(
            str(doc.get(field) or "")
            for field in ("title", "summary", "body")
        )
        matches = sum(1 for term in terms if str(term) in text)
        return min(0.06 * matches, 0.18)

    def _diversify_hits(self, hits: list[dict], source_plan: list[str], top_k: int) -> list[dict]:
        selected: list[dict] = []
        seen_ids: set[str] = set()
        preferred_sources = [source for source in source_plan if source in {"news", "announcement", "faq", "product_doc", "research_note"}]
        for source in preferred_sources:
            source_hit = next((hit for hit in hits if hit["source_type"] == source and hit["evidence_id"] not in seen_ids), None)
            if source_hit is None:
                continue
            selected.append(source_hit)
            seen_ids.add(source_hit["evidence_id"])
            if len(selected) >= top_k:
                return selected

        for hit in hits:
            if hit["evidence_id"] in seen_ids:
                continue
            selected.append(hit)
            seen_ids.add(hit["evidence_id"])
            if len(selected) >= top_k:
                break
        return selected
