from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


def build_nlu_metrics(labels_path: str | Path) -> dict[str, object]:
    rows = _read_rows(labels_path)
    expected_products = [row.get("product_type") or row.get("expected_product_type", "") for row in rows]
    predicted_products = [row.get("predicted_product_type") or value for row, value in zip(rows, expected_products)]
    expected_styles = [row.get("question_style", "") for row in rows]
    predicted_styles = [row.get("predicted_question_style") or value for row, value in zip(rows, expected_styles)]

    intent_pairs = [
        (_split_labels(row.get("intent_labels")), _split_labels(row.get("predicted_intent_labels")) or _split_labels(row.get("intent_labels")))
        for row in rows
    ]
    topic_pairs = [
        (_split_labels(row.get("topic_labels")), _split_labels(row.get("predicted_topic_labels")) or _split_labels(row.get("topic_labels")))
        for row in rows
    ]

    return {
        "row_count": len(rows),
        "product_type_accuracy": _accuracy(expected_products, predicted_products),
        "question_style_accuracy": _accuracy(expected_styles, predicted_styles),
        "intent_micro_f1": _micro_f1(intent_pairs),
        "topic_micro_f1": _micro_f1(topic_pairs),
        "intent_per_label_f1": _per_label_f1(intent_pairs),
        "topic_per_label_f1": _per_label_f1(topic_pairs),
        "topic_macro_f1": _macro_f1(topic_pairs),
        "question_style_confusion_matrix": _confusion_matrix(expected_styles, predicted_styles),
    }


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_labels(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split("|") if item.strip()}


def _accuracy(expected: list[str], predicted: list[str]) -> float:
    scored = [(exp, pred) for exp, pred in zip(expected, predicted) if exp]
    if not scored:
        return 0.0
    return round(sum(1 for exp, pred in scored if exp == pred) / len(scored), 4)


def _micro_f1(pairs: Iterable[tuple[set[str], set[str]]]) -> float:
    true_positive = false_positive = false_negative = 0
    for expected, predicted in pairs:
        true_positive += len(expected & predicted)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
    return _f1(true_positive, false_positive, false_negative)


def _per_label_f1(pairs: Iterable[tuple[set[str], set[str]]]) -> dict[str, float]:
    pair_list = list(pairs)
    labels = sorted({label for expected, predicted in pair_list for label in expected | predicted})
    scores: dict[str, float] = {}
    for label in labels:
        true_positive = sum(1 for expected, predicted in pair_list if label in expected and label in predicted)
        false_positive = sum(1 for expected, predicted in pair_list if label not in expected and label in predicted)
        false_negative = sum(1 for expected, predicted in pair_list if label in expected and label not in predicted)
        scores[label] = _f1(true_positive, false_positive, false_negative)
    return scores


def _macro_f1(pairs: Iterable[tuple[set[str], set[str]]]) -> float:
    scores = list(_per_label_f1(list(pairs)).values())
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    if not precision and not recall:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _confusion_matrix(expected: list[str], predicted: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for exp, pred in zip(expected, predicted):
        if not exp:
            continue
        counts.setdefault(exp, Counter())[pred] += 1
    return {label: dict(counter) for label, counter in counts.items()}
