from __future__ import annotations

from query_intelligence.nlu.clarification_gate import ClarificationGate
from query_intelligence.nlu.question_style_reranker import QuestionStyleReranker
from query_intelligence.nlu.source_plan_reranker import SourcePlanReranker
from query_intelligence.nlu.typo_linker import TypoLinker


def test_clarification_gate_reports_partial_fit_batches() -> None:
    rows = [
        {"query": "这个能买吗", "product_type": "stock", "intent_labels": ["buy_sell_timing"], "topic_labels": ["risk"], "time_scope": "unspecified", "entity_count": 0, "comparison_target_count": 0, "needs_clarification": True},
        {"query": "贵州茅台为什么跌", "product_type": "stock", "intent_labels": ["market_explanation"], "topic_labels": ["price"], "time_scope": "today", "entity_count": 1, "comparison_target_count": 0, "needs_clarification": False},
        {"query": "后面还值得拿吗", "product_type": "stock", "intent_labels": ["hold_judgment"], "topic_labels": ["risk"], "time_scope": "unspecified", "entity_count": 0, "comparison_target_count": 0, "needs_clarification": True},
        {"query": "ETF和LOF有什么区别", "product_type": "etf", "intent_labels": ["product_info"], "topic_labels": ["product_mechanism"], "time_scope": "unspecified", "entity_count": 0, "comparison_target_count": 2, "needs_clarification": False},
    ]
    events: list[tuple[int, int]] = []

    gate = ClarificationGate.build_from_rows(rows, fit_progress_callback=lambda done, total: events.append((done, total)), batch_size=2, epochs=2)

    assert 0.0 <= gate.predict_probability(**{**rows[0], "needs_clarification": None}) <= 1.0
    assert events == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_source_plan_reranker_reports_partial_fit_batches() -> None:
    rows = [
        {"query": "最近有哪些茅台新闻", "product_type": "stock", "intent_labels": ["event_news_query"], "topic_labels": ["news"], "time_scope": "recent_1w", "source": "news", "label": 1},
        {"query": "最近有哪些茅台新闻", "product_type": "stock", "intent_labels": ["event_news_query"], "topic_labels": ["news"], "time_scope": "recent_1w", "source": "market_api", "label": 0},
        {"query": "ETF和LOF有什么区别", "product_type": "etf", "intent_labels": ["product_info"], "topic_labels": ["product_mechanism"], "time_scope": "unspecified", "source": "faq", "label": 1},
        {"query": "ETF和LOF有什么区别", "product_type": "etf", "intent_labels": ["product_info"], "topic_labels": ["product_mechanism"], "time_scope": "unspecified", "source": "market_api", "label": 0},
    ]
    events: list[tuple[int, int]] = []

    reranker = SourcePlanReranker.build_from_rows(rows, fit_progress_callback=lambda done, total: events.append((done, total)), batch_size=2, epochs=2)

    assert reranker.score_candidates("最近有哪些茅台新闻", "stock", ["event_news_query"], ["news"], "recent_1w", ["news"], ["news"])
    assert events == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_question_style_reranker_reports_partial_fit_batches() -> None:
    rows = [
        {"query": "茅台为什么跌", "base_style": "why", "product_type": "stock", "intent_labels": ["market_explanation"], "topic_labels": ["price"], "entity_count": 1, "comparison_target_count": 0, "question_style": "why"},
        {"query": "茅台和五粮液哪个好", "base_style": "compare", "product_type": "stock", "intent_labels": ["peer_compare"], "topic_labels": ["comparison"], "entity_count": 2, "comparison_target_count": 2, "question_style": "compare"},
        {"query": "这只还能买吗", "base_style": "fact", "product_type": "stock", "intent_labels": ["buy_sell_timing"], "topic_labels": ["risk"], "entity_count": 0, "comparison_target_count": 0, "question_style": "advice"},
        {"query": "后面会怎么走", "base_style": "forecast", "product_type": "stock", "intent_labels": ["market_explanation"], "topic_labels": ["price"], "entity_count": 1, "comparison_target_count": 0, "question_style": "forecast"},
    ]
    events: list[tuple[int, int]] = []

    reranker = QuestionStyleReranker.build_from_rows(rows, fit_progress_callback=lambda done, total: events.append((done, total)), batch_size=2, epochs=2)

    assert reranker.predict(**{**rows[0], "question_style": None})["label"] in {"why", "compare", "advice", "forecast"}
    assert events == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_typo_linker_reports_partial_fit_batches() -> None:
    rows = [
        {"query": "茅苔", "mention": "茅苔", "alias": "茅台", "heuristic_score": 0.8, "label": 1},
        {"query": "茅苔", "mention": "茅苔", "alias": "五粮液", "heuristic_score": 0.0, "label": 0},
        {"query": "300ETF", "mention": "300ETF", "alias": "沪深300ETF", "heuristic_score": 0.8, "label": 1},
        {"query": "300ETF", "mention": "300ETF", "alias": "创业板ETF", "heuristic_score": 0.0, "label": 0},
    ]
    events: list[tuple[int, int]] = []

    linker = TypoLinker.build_from_rows(rows, fit_progress_callback=lambda done, total: events.append((done, total)), batch_size=2, epochs=2)

    assert 0.0 <= linker.predict_probability("茅苔", "茅苔", "茅台", 0.8) <= 1.0
    assert events == [(1, 4), (2, 4), (3, 4), (4, 4)]
