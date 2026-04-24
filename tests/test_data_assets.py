from __future__ import annotations

import csv
from pathlib import Path

from evaluation.eval_nlu import build_nlu_metrics
from evaluation.eval_retrieval import build_retrieval_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_query_labels_csv_exists_and_is_expanded() -> None:
    path = ROOT / "data" / "query_labels.csv"
    assert path.exists()
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert 200 <= len(rows) <= 300
    assert "split" in rows[0]
    assert "expected_document_sources" in rows[0]
    assert "expected_structured_sources" in rows[0]
    assert "question_style" in rows[0]
    assert "sentiment_label" in rows[0]


def test_retrieval_qrels_and_entity_annotations_exist() -> None:
    qrels_path = ROOT / "data" / "retrieval_qrels.csv"
    entity_ann_path = ROOT / "data" / "entity_bio_annotations.jsonl"
    assert qrels_path.exists()
    assert entity_ann_path.exists()

    with qrels_path.open("r", encoding="utf-8") as handle:
        qrels_rows = list(csv.DictReader(handle))
    assert len(qrels_rows) >= 1000

    ann_lines = entity_ann_path.read_text(encoding="utf-8").splitlines()
    assert 150 <= len(ann_lines) <= 300


def test_supervision_labels_do_not_cross_contaminate() -> None:
    intent_vocab = {
        "price_query", "market_explanation", "hold_judgment", "buy_sell_timing", "product_info",
        "risk_analysis", "peer_compare", "fundamental_analysis", "valuation_analysis",
        "macro_policy_impact", "event_news_query", "trading_rule_fee"
    }
    topic_vocab = {
        "price", "news", "industry", "macro", "policy", "fundamentals",
        "valuation", "risk", "comparison", "product_mechanism"
    }
    with (ROOT / "data" / "query_labels.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        intents = {label for label in row["intent_labels"].split("|") if label}
        topics = {label for label in row["topic_labels"].split("|") if label}
        assert intents <= intent_vocab
        assert topics <= topic_vocab


def test_eval_nlu_returns_metric_summary() -> None:
    result = build_nlu_metrics(ROOT / "data" / "query_labels.csv")

    assert "product_type_accuracy" in result
    assert "intent_micro_f1" in result
    assert "topic_macro_f1" in result
    assert "intent_per_label_f1" in result
    assert "topic_per_label_f1" in result
    assert "question_style_confusion_matrix" in result


def test_eval_retrieval_returns_metric_summary() -> None:
    result = build_retrieval_metrics(ROOT / "data" / "query_labels.csv")

    assert "document_source_recall" in result
    assert "structured_source_recall" in result
    assert "document_precision" in result
    assert "document_mrr_at_10" in result
    assert "document_ndcg_at_10" in result
