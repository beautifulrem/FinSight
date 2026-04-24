from __future__ import annotations

from typing import Any

from ..label_maps import autolabel_intents, autolabel_question_style, autolabel_topics


def infer_question_style(text: str) -> str:
    return autolabel_question_style(text)


def infer_classification_labels(text: str) -> dict[str, Any]:
    return {
        "intent_labels": autolabel_intents(text),
        "topic_labels": autolabel_topics(text),
        "question_style": infer_question_style(text),
    }


def build_autolabeled_classification_row(
    text: str,
    *,
    source_id: str,
    source_row_id: Any | None = None,
    product_type: str = "unknown",
    raw_label: Any | None = None,
    available_labels: list[str] | None = None,
) -> dict[str, Any]:
    labels = infer_classification_labels(text)
    return {
        "sample_family": "classification",
        "query": text,
        "text": text,
        "product_type": product_type,
        "intent_labels": labels["intent_labels"],
        "topic_labels": labels["topic_labels"],
        "question_style": labels["question_style"],
        "sentiment_label": None,
        "split_lock_key": f"{source_id}:{source_row_id or raw_label or text}",
        "source_id": source_id,
        "source_row_id": source_row_id,
        "raw_label": raw_label,
        "available_labels": list(available_labels or ["product_type", "intent_labels", "topic_labels", "question_style"]),
    }
