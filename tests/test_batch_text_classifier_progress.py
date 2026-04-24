from __future__ import annotations

from query_intelligence.nlu.classifiers import (
    MultiLabelClassifier,
    ProductTypeClassifier,
    SingleLabelTextClassifier,
    estimate_text_classifier_fit_steps,
)


def test_single_label_classifier_reports_partial_fit_batches() -> None:
    rows = [
        {"query": "茅台为什么跌", "question_style": "why"},
        {"query": "沪深300ETF是什么", "question_style": "fact"},
        {"query": "这个还能买吗", "question_style": "advice"},
        {"query": "CPI会影响市场吗", "question_style": "forecast"},
    ]
    events: list[tuple[int, int]] = []

    classifier = SingleLabelTextClassifier.build_from_records(
        rows,
        "question_style",
        fit_progress_callback=lambda done, total: events.append((done, total)),
        batch_size=2,
        epochs=2,
    )

    assert classifier.predict("茅台为什么跌")["label"] in {"why", "fact", "advice", "forecast"}
    assert events == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert estimate_text_classifier_fit_steps(len(rows), batch_size=2, epochs=2) == 4


def test_product_type_classifier_reports_partial_fit_batches() -> None:
    rows = [
        {"query": "茅台为什么跌", "product_type": "stock"},
        {"query": "沪深300ETF是什么", "product_type": "etf"},
        {"query": "这个基金费率高吗", "product_type": "fund"},
        {"query": "上证指数怎么看", "product_type": "index"},
    ]
    events: list[tuple[int, int]] = []

    classifier = ProductTypeClassifier.build_from_records(
        rows,
        fit_progress_callback=lambda done, total: events.append((done, total)),
        batch_size=2,
        epochs=2,
    )

    assert classifier.predict("沪深300ETF怎么样", [])["label"] in {"stock", "etf", "fund", "index"}
    assert events == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_multi_label_classifier_reports_partial_fit_batches() -> None:
    rows = [
        {"query": "茅台为什么跌", "intent_labels": ["market_explanation"]},
        {"query": "这只还能买吗", "intent_labels": ["buy_sell_timing"]},
        {"query": "这个基金费率高吗", "intent_labels": ["trading_rule_fee"]},
        {"query": "茅台和五粮液哪个好", "intent_labels": ["peer_compare"]},
    ]
    events: list[tuple[int, int]] = []

    classifier = MultiLabelClassifier.build_from_records(
        rows,
        "intent_labels",
        fit_progress_callback=lambda done, total: events.append((done, total)),
        batch_size=2,
        epochs=2,
    )

    assert classifier.predict("茅台为什么跌")
    assert events == [(1, 4), (2, 4), (3, 4), (4, 4)]
