from __future__ import annotations

import json
import logging
import sys
import csv
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

from joblib import dump
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from training.progress import ProgressBar, StageProgress

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.config import Settings
from query_intelligence.nlu.pipeline import NLUPipeline
from query_intelligence.retrieval.doc_retriever import DocumentRetriever
from query_intelligence.retrieval.feature_builder import FeatureBuilder
from query_intelligence.retrieval.query_builder import QueryBuilder
from query_intelligence.retrieval.ranker import FEATURE_NAMES
from query_intelligence.training_data import load_training_rows
from query_intelligence.training_manifest import TrainingManifest, load_training_manifest
from query_intelligence.data_loader import load_documents
from query_intelligence.nlu.batch_linear import BATCH_LINEAR_BATCH_SIZE, BATCH_LINEAR_EPOCHS, estimate_batch_fit_steps


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    mapping: dict[str, dict[str, int]] = {}
    if not path.exists():
        return mapping
    if path.suffix.lower() == ".jsonl":
        for row in _read_jsonl(path):
            query = str(row.get("query") or "").strip()
            evidence_id = str(row.get("evidence_id") or row.get("doc_id") or "").strip()
            if not query or not evidence_id:
                continue
            mapping.setdefault(query, {})[evidence_id] = int(row.get("relevance") or 0)
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            query = str(row.get("query") or "").strip()
            evidence_id = str(row.get("evidence_id") or row.get("doc_id") or "").strip()
            if not query or not evidence_id:
                continue
            mapping.setdefault(query, {})[evidence_id] = int(row.get("relevance") or 0)
    return mapping


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _maybe_load_manifest(path: Path) -> TrainingManifest | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        return load_training_manifest(path)
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _normalize_document(row: dict, fallback_id: str) -> dict:
    document = dict(row)
    evidence_id = str(document.get("evidence_id") or document.get("doc_id") or fallback_id).strip()
    title = str(document.get("title") or "").strip()
    summary = str(document.get("summary") or document.get("abstract") or "").strip()
    body = str(document.get("body") or document.get("text") or summary or title).strip()
    source_type = str(document.get("source_type") or document.get("sample_family") or "research_note").strip() or "research_note"
    entity_symbols = document.get("entity_symbols") or []
    if isinstance(entity_symbols, str):
        entity_symbols = [item.strip() for item in entity_symbols.split("|") if item.strip()]
    document.update(
        {
            "evidence_id": evidence_id,
            "doc_id": str(document.get("doc_id") or evidence_id).strip(),
            "title": title,
            "summary": summary,
            "body": body,
            "source_type": source_type,
            "entity_symbols": list(entity_symbols),
        }
    )
    return document


def _load_documents_for_training(manifest: TrainingManifest | None) -> list[dict]:
    if manifest is None:
        return load_documents()
    documents = []
    for index, row in enumerate(_read_jsonl(manifest.retrieval_corpus_path), start=1):
        documents.append(_normalize_document(row, f"manifest-doc-{index}"))
    return documents


def _index_documents(documents: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for document in documents:
        evidence_id = str(document.get("evidence_id") or document.get("doc_id") or "").strip()
        if evidence_id:
            indexed[evidence_id] = document
    return indexed


def _cap_query_rows(rows: list[dict], limit: int = 10000) -> list[dict]:
    if len(rows) <= limit:
        return rows
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha1(str(row.get("query") or "").encode("utf-8")).hexdigest(),
    )
    return ordered[:limit]


def _build_ranker_train_rows(classification_rows: list[dict], qrels: dict[str, dict[str, int]]) -> list[dict]:
    if not qrels:
        return [row for row in classification_rows if row.get("split", "train") in {"train", "valid"}]

    rows_by_query = {str(row.get("query") or "").strip(): row for row in classification_rows if str(row.get("query") or "").strip()}
    qrel_rows: list[dict] = []
    for query in qrels:
        row = dict(rows_by_query.get(query, {}))
        row["query"] = query
        row.setdefault("product_type", "unknown")
        row.setdefault("intent_labels", [])
        row.setdefault("topic_labels", [])
        row.setdefault("expected_document_sources", [])
        row.setdefault("expected_structured_sources", [])
        row.setdefault("question_style", "")
        row.setdefault("sentiment_label", "")
        row["split"] = "train"
        row["_ranker_qrel_seed"] = True
        qrel_rows.append(row)
    return _cap_query_rows(qrel_rows)


def _build_lightweight_query_bundle(query: str) -> dict:
    return {
        "query_id": hashlib.sha1(query.encode("utf-8")).hexdigest(),
        "normalized_query": query,
        "keywords": [],
        "entity_names": [],
        "symbols": [],
        "industry_terms": [],
        "source_plan": [],
        "product_type": "unknown",
        "intent_labels": [],
        "topic_labels": [],
        "time_scope": "unspecified",
    }


def _fit_ranker_pipeline(X: list[list[float]], y: list[int]) -> Pipeline:
    if not X:
        raise RuntimeError("no ranker training rows generated")
    labels = np.array(y)
    classes = np.array(sorted(set(labels.tolist())))
    if len(classes) < 2:
        raise RuntimeError("ranker needs at least two classes")
    matrix = np.asarray(X, dtype=float)
    batch_size = BATCH_LINEAR_BATCH_SIZE
    scale_steps = estimate_batch_fit_steps(len(matrix), batch_size=batch_size, epochs=1)
    scale_progress = ProgressBar("ranker:scale", scale_steps, every=1)
    scale_progress.update(0, "starting", force=True)
    scaler = StandardScaler()
    completed = 0
    for start in range(0, len(matrix), batch_size):
        scaler.partial_fit(matrix[start : start + batch_size])
        completed += 1
        scale_progress.update(completed, "partial_fit scaler", force=completed == scale_steps)
    scale_progress.finish("scale complete")

    fit_steps = estimate_batch_fit_steps(len(matrix), batch_size=batch_size, epochs=BATCH_LINEAR_EPOCHS)
    fit_progress = ProgressBar("ranker:fit", fit_steps, every=1)
    fit_progress.update(0, "starting", force=True)
    classifier = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42)
    class_weights = compute_class_weight("balanced", classes=classes, y=labels)
    weight_map = dict(zip(classes, class_weights, strict=False))
    rng = np.random.default_rng(42)
    completed = 0
    first_batch = True
    for _ in range(BATCH_LINEAR_EPOCHS):
        indices = rng.permutation(len(matrix))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            x_batch = scaler.transform(matrix[batch_indices])
            y_batch = labels[batch_indices]
            sample_weight = np.array([weight_map[label] for label in y_batch], dtype=float)
            if first_batch:
                classifier.partial_fit(x_batch, y_batch, classes=classes, sample_weight=sample_weight)
                first_batch = False
            else:
                classifier.partial_fit(x_batch, y_batch, sample_weight=sample_weight)
            completed += 1
            fit_progress.update(completed, "partial_fit batches", force=completed == fit_steps)
    fit_progress.finish("fit complete")
    return Pipeline([("scaler", scaler), ("clf", classifier)])


def train_ranker_model(dataset_path: str | Path, output_path: str | Path, progress: StageProgress | None = None) -> dict:
    dataset_path = Path(dataset_path)
    manifest = _maybe_load_manifest(dataset_path)
    if progress:
        progress.advance("loading training rows")
    rows = load_training_rows(manifest.classification_path if manifest else dataset_path)
    qrels_path = manifest.qrels_path if manifest else dataset_path.with_name("retrieval_qrels.csv")
    qrels = _load_qrels(qrels_path) if qrels_path.exists() else {}
    train_rows = _build_ranker_train_rows(rows, qrels)
    seed_only = bool(qrels) and train_rows and all(row.get("_ranker_qrel_seed") for row in train_rows)
    logger.warning(
        "NLU predictions on training data may be overconfident — "
        "NLU pipeline was trained on the same split. Prefer qrels for ranker labels."
    )
    nlu = None if seed_only else NLUPipeline.build_default(Settings.from_env())
    training_documents = _load_documents_for_training(manifest)
    doc_retriever = None if seed_only else DocumentRetriever(training_documents)
    document_index = _index_documents(training_documents)
    indexed_doc_ids = sorted(document_index)
    query_builder = QueryBuilder()
    feature_builder = FeatureBuilder()
    if progress:
        progress.note(f"loaded {len(train_rows)} query rows and {len(training_documents)} documents")
        progress.advance("building features and fitting model")

    X: list[list[float]] = []
    y: list[int] = []
    label_source = "qrels" if qrels else "source_expectation"

    for row in train_rows:
        if row.get("_ranker_qrel_seed"):
            query_bundle = _build_lightweight_query_bundle(row["query"])
        else:
            nlu_result = nlu.run(query=row["query"], user_profile={}, dialog_context=[], debug=False)
            query_bundle = query_builder.build(nlu_result)
        expected_sources = set(row.get("expected_document_sources", []))
        qrels_for_query = qrels.get(row["query"], {})
        candidates = _build_qrel_candidates(qrels_for_query, document_index)
        if qrels_for_query:
            candidates.extend(
                _sample_negative_candidates(
                    row["query"],
                    indexed_doc_ids=indexed_doc_ids,
                    document_index=document_index,
                    excluded_ids=set(qrels_for_query),
                    limit=4,
                )
            )
        if not candidates and doc_retriever is not None:
            candidates = doc_retriever.search(query_bundle, top_k=20)
        if not candidates and doc_retriever is not None:
            candidates = _fallback_candidates(query_bundle, doc_retriever)
        query_keywords = query_bundle.get("keywords", []) + query_bundle.get("entity_names", [])
        for candidate in candidates:
            feature_json = feature_builder.build(query_bundle, candidate)
            X.append([float(feature_json.get(name, 0.0)) for name in FEATURE_NAMES])
            if qrels_for_query:
                relevance = int(qrels_for_query.get(candidate["evidence_id"], 0))
                y.append(1 if relevance >= 1 else 0)
            else:
                source_match = candidate["source_type"] in expected_sources
                content = f"{candidate.get('title', '')} {candidate.get('body', '')}"
                content_match = any(kw in content for kw in query_keywords) if query_keywords else False
                y.append(1 if source_match and content_match else 0)

    if not X:
        raise RuntimeError("no ranker training rows generated")
    if len(set(y)) < 2:
        X.append([0.0] * len(FEATURE_NAMES))
        y.append(0 if y[0] == 1 else 1)

    model = _fit_ranker_pipeline(X, y)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.advance("writing artifact")
    dump({"model": model, "feature_names": FEATURE_NAMES}, output_path)
    return {
        "training_rows": len(X),
        "positive_rows": int(sum(y)),
        "label_source": label_source,
        "feature_names": FEATURE_NAMES,
        "output": str(output_path),
    }


def _fallback_candidates(query_bundle: dict, doc_retriever: DocumentRetriever) -> list[dict]:
    allowed_sources = doc_retriever._allowed_sources(query_bundle)  # noqa: SLF001
    documents = doc_retriever.documents
    if allowed_sources:
        documents = [doc for doc in documents if doc.get("source_type") in allowed_sources]
    return [{**doc, "retrieval_score": 0.0} for doc in documents]


def _build_qrel_candidates(qrels_for_query: dict[str, int], document_index: dict[str, dict]) -> list[dict]:
    candidates: list[dict] = []
    for evidence_id in qrels_for_query:
        document = document_index.get(evidence_id)
        if document is None:
            continue
        candidates.append({**document, "retrieval_score": 1.0})
    return candidates


def _sample_negative_candidates(
    query: str,
    *,
    indexed_doc_ids: list[str],
    document_index: dict[str, dict],
    excluded_ids: set[str],
    limit: int,
) -> list[dict]:
    if not indexed_doc_ids or limit <= 0:
        return []
    start = int(hashlib.sha1(query.encode("utf-8")).hexdigest(), 16) % len(indexed_doc_ids)
    sampled: list[dict] = []
    cursor = 0
    while len(sampled) < limit and cursor < len(indexed_doc_ids):
        doc_id = indexed_doc_ids[(start + cursor) % len(indexed_doc_ids)]
        cursor += 1
        if doc_id in excluded_ids:
            continue
        document = document_index.get(doc_id)
        if document is None:
            continue
        sampled.append({**document, "retrieval_score": 0.0})
    return sampled


def main() -> None:
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/query_labels.csv")
    output_path = Path("models/ranker.joblib")
    progress = StageProgress("ranker", 3)
    metadata = train_ranker_model(dataset_path, output_path, progress=progress)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
