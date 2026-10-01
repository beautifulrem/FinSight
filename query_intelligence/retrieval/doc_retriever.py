from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_RETRIEVAL_SCORE = {
    "news": 0.02,
    "announcement": 0.02,
    "research_note": 0.05,
    "faq": 0.05,
    "product_doc": 0.05,
}


logger = logging.getLogger(__name__)

_INDEX_VERSION = "char_wb-2-4-v1"
_INDEX_MEMO: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
_INDEX_MEMO_SIZE = 2
_INDEX_LOCK = threading.Lock()
# Transposed document matrices (CSR), one per fitted index, keyed by the matrix object's id. ``linear_kernel(q, M)``
# computes ``q @ M.T`` and scipy converts ``M.T`` (CSC) back to CSR on every call: on the offline corpus that
# conversion was ~87% of each search (~120 ms of ~270 ms, and seconds under a 4-user load). Doing it once per index
# gives the same products in the same order, so the scores are bit-identical (round 13, P95 driver of the LLM load).
_TRANSPOSED: dict[int, tuple[Any, Any]] = {}


def _fit_index(corpus: list[str]) -> tuple[Any, Any]:
    """TF-IDF vectorizer and document matrix for ``corpus``, reused when the corpus is unchanged.

    Fitting char n-grams over the full document set takes most of the service start-up time (~40 s), so
    the fitted index is memoised per process by corpus hash and, when ``QI_TFIDF_CACHE_DIR`` is set,
    persisted there (e.g. a volume shared by replicas) so a restart loads it instead of refitting.
    """
    digest = hashlib.sha256(_INDEX_VERSION.encode())
    for text in corpus:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    key = digest.hexdigest()
    with _INDEX_LOCK:
        if key in _INDEX_MEMO:
            _INDEX_MEMO.move_to_end(key)
            return _INDEX_MEMO[key]
        cache_dir = os.getenv("QI_TFIDF_CACHE_DIR", "").strip()
        path = Path(cache_dir) / f"tfidf-{key[:24]}.joblib" if cache_dir else None
        index = None
        if path is not None and path.is_file():
            try:
                index = joblib.load(path)
            except Exception as exc:  # a corrupt or incompatible cache file is rebuilt
                logger.warning("[retrieval] ignoring unreadable TF-IDF cache %s: %s", path, exc)
        if index is None:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            index = (vectorizer, vectorizer.fit_transform(corpus))
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(f".{os.getpid()}.tmp")
                    joblib.dump(index, tmp)
                    tmp.replace(path)
                except OSError as exc:
                    logger.warning("[retrieval] could not write TF-IDF cache %s: %s", path, exc)
        _INDEX_MEMO[key] = index
        while len(_INDEX_MEMO) > _INDEX_MEMO_SIZE:
            _INDEX_MEMO.popitem(last=False)
        return index


def _transposed(matrix: Any) -> Any:
    """``matrix.T`` as CSR, computed once per fitted matrix (kept alive with it, at most ``_INDEX_MEMO_SIZE + 2``)."""
    with _INDEX_LOCK:
        entry = _TRANSPOSED.get(id(matrix))
        if entry is not None and entry[0] is matrix:
            return entry[1]
        transposed = matrix.T.tocsr()
        _TRANSPOSED[id(matrix)] = (matrix, transposed)
        while len(_TRANSPOSED) > _INDEX_MEMO_SIZE + 2:
            _TRANSPOSED.pop(next(iter(_TRANSPOSED)))
        return transposed


@dataclass
class DocumentRetriever:
    documents: list[dict]

    def __post_init__(self) -> None:
        corpus = [f"{doc['title']} {doc['summary']} {doc['body']}" for doc in self.documents]
        # The fitted objects are shared (read-only) between retrievers built from the same corpus.
        self.vectorizer, self.doc_matrix = _fit_index(corpus)
        self._doc_matrix_t = _transposed(self.doc_matrix)

    def search(self, query_bundle: dict, top_k: int = 20) -> list[dict]:
        query_text = " ".join(
            [
                query_bundle["normalized_query"],
                *query_bundle["entity_names"],
                *query_bundle["keywords"],
                *query_bundle["symbols"],
            ]
        ).strip()
        if not query_text:
            return []

        scores = self._scores(self.vectorizer.transform([query_text]))

        hits = []
        allowed_sources = self._allowed_sources(query_bundle)
        if not allowed_sources:
            return []
        for doc, score in zip(self.documents, scores, strict=False):
            if doc["source_type"] not in allowed_sources:
                continue
            requires_entity_hit = doc["source_type"] in {"news", "announcement", "research_note"}
            if (
                requires_entity_hit
                and query_bundle["symbols"]
                and not any(symbol in doc.get("entity_symbols", []) for symbol in query_bundle["symbols"])
            ):
                continue
            if self._should_skip_entity_bound_doc(query_bundle, doc):
                continue
            retrieval_score = float(round(max(score + self._term_hit_bonus(query_bundle, doc), 0.0), 4))
            if retrieval_score < MIN_RETRIEVAL_SCORE.get(doc["source_type"], 0.0):
                continue
            hits.append({**doc, "retrieval_score": retrieval_score})

        hits.sort(key=lambda item: item["retrieval_score"], reverse=True)
        return self._diversify_hits(hits, query_bundle.get("source_plan", []), top_k)

    def _scores(self, query_vector: Any) -> Any:
        """Cosine scores of the query against every document: ``linear_kernel(query_vector, self.doc_matrix)``
        without re-transposing the corpus matrix on every call (bit-identical; see ``_TRANSPOSED``)."""
        return (query_vector @ self._doc_matrix_t).toarray().ravel()

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
        generic_product_query = (
            bool(intent_labels.intersection({"product_info", "trading_rule_fee"}))
            or "product_mechanism" in topic_labels
        )
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
        text = " ".join(str(doc.get(field) or "") for field in ("title", "summary", "body"))
        matches = sum(1 for term in terms if str(term) in text)
        return min(0.06 * matches, 0.18)

    def _diversify_hits(self, hits: list[dict], source_plan: list[str], top_k: int) -> list[dict]:
        selected: list[dict] = []
        seen_ids: set[str] = set()
        preferred_sources = [
            source
            for source in source_plan
            if source in {"news", "announcement", "faq", "product_doc", "research_note"}
        ]
        for source in preferred_sources:
            source_hit = next(
                (hit for hit in hits if hit["source_type"] == source and hit["evidence_id"] not in seen_ids), None
            )
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
