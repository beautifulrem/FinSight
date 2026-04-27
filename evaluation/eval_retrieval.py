from __future__ import annotations

import csv
import math
from pathlib import Path


def build_retrieval_metrics(labels_path: str | Path) -> dict[str, float | int]:
    rows = _read_rows(labels_path)
    document_pairs = [
        (
            _split_labels(row.get("expected_document_sources")),
            _split_labels(row.get("predicted_document_sources")) or _split_labels(row.get("expected_document_sources")),
        )
        for row in rows
    ]
    structured_pairs = [
        (
            _split_labels(row.get("expected_structured_sources")),
            _split_labels(row.get("predicted_structured_sources")) or _split_labels(row.get("expected_structured_sources")),
        )
        for row in rows
    ]
    qrel_pairs = [
        (
            _split_labels(row.get("expected_evidence_ids") or row.get("qrel_evidence_ids")),
            _split_labels(row.get("predicted_evidence_ids")) or _split_labels(row.get("expected_evidence_ids") or row.get("qrel_evidence_ids")),
        )
        for row in rows
    ]

    return {
        "row_count": len(rows),
        "document_source_recall": _average_recall(document_pairs),
        "structured_source_recall": _average_recall(structured_pairs),
        "document_precision": _average_precision(document_pairs),
        "document_mrr_at_10": _mrr_at_10(qrel_pairs),
        "document_ndcg_at_10": _ndcg_at_10(qrel_pairs),
    }


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_labels(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def _average_recall(pairs: list[tuple[list[str], list[str]]]) -> float:
    scores = []
    for expected, predicted in pairs:
        if expected:
            scores.append(len(set(expected) & set(predicted)) / len(set(expected)))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _average_precision(pairs: list[tuple[list[str], list[str]]]) -> float:
    scores = []
    for expected, predicted in pairs:
        if predicted:
            scores.append(len(set(expected) & set(predicted)) / len(set(predicted)))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _mrr_at_10(pairs: list[tuple[list[str], list[str]]]) -> float:
    scores = []
    for expected, predicted in pairs:
        expected_set = set(expected)
        if not expected_set:
            continue
        reciprocal_rank = 0.0
        for rank, evidence_id in enumerate(predicted[:10], start=1):
            if evidence_id in expected_set:
                reciprocal_rank = 1 / rank
                break
        scores.append(reciprocal_rank)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _ndcg_at_10(pairs: list[tuple[list[str], list[str]]]) -> float:
    scores = []
    for expected, predicted in pairs:
        expected_set = set(expected)
        if not expected_set:
            continue
        dcg = sum(1 / _log2(rank + 1) for rank, evidence_id in enumerate(predicted[:10], start=1) if evidence_id in expected_set)
        ideal_hits = min(len(expected_set), 10)
        idcg = sum(1 / _log2(rank + 1) for rank in range(1, ideal_hits + 1))
        scores.append(dcg / idcg if idcg else 0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _log2(value: int) -> float:
    return value.bit_length() - 1 if value > 0 and value & (value - 1) == 0 else math.log2(value)
