from __future__ import annotations

from scripts.evaluate_query_intelligence import EvalItem, build_summary
from query_intelligence.service import build_default_service, clear_service_caches


def test_eval_summary_checks_retrieval_metric_names_and_exports_failures() -> None:
    items = [
        EvalItem(
            query="贵州茅台最近为什么跌？",
            bucket="finance_generated_why",
            domain="finance",
            language="zh",
            expected_product_type="stock",
            expected_question_style="why",
            expected_intents=("market_explanation",),
            expected_topics=("price",),
            expected_sources=("news", "research_note"),
        ),
        EvalItem(
            query="这只适合继续定投吗？",
            bucket="finance_generated_clarification",
            domain="finance",
            language="zh",
            expected_product_type="fund",
            expected_question_style="advice",
            expected_intents=("hold_judgment",),
            expected_topics=("product_mechanism",),
            expected_sources=("faq", "product_doc"),
            expected_clarification=True,
        ),
        EvalItem(
            query="你好你是谁",
            bucket="ood_must_pass",
            domain="ood",
            language="zh",
            expected_out_of_scope=True,
        ),
    ]
    nlu_outputs = {
        "贵州茅台最近为什么跌？": {
            "product_type": {"label": "stock"},
            "question_style": "fact",
            "intent_labels": [],
            "topic_labels": [],
            "source_plan": [],
            "risk_flags": [],
            "missing_slots": [],
        },
        "这只适合继续定投吗？": {
            "product_type": {"label": "fund"},
            "question_style": "fact",
            "intent_labels": [{"label": "price_query"}],
            "topic_labels": [{"label": "price"}],
            "source_plan": [],
            "risk_flags": [],
            "missing_slots": [],
        },
        "你好你是谁": {
            "product_type": {"label": "out_of_scope"},
            "question_style": "fact",
            "intent_labels": [],
            "topic_labels": [],
            "source_plan": [],
            "risk_flags": ["out_of_scope_query"],
            "missing_slots": [],
        },
    }

    summary = build_summary(
        items,
        nlu_outputs,
        {
            "retrieval_eval_queries": 1,
            "ood_retrieval_queries": 1,
            "recall_at_10": 0.5,
            "mrr_at_10": 0.4,
            "ndcg_at_10": 0.4,
            "source_plan_support": 0.25,
            "ood_retrieval_abstention": 1.0,
        },
    )

    for metric in ["recall_at_10", "mrr_at_10", "ndcg_at_10", "source_plan_support"]:
        assert metric in summary["verdicts"]
    assert "source_plan_support" in summary["failed_metrics"]
    assert summary["failure_examples"]["question_style"]
    assert summary["failure_examples"]["intent"]
    assert summary["failure_examples"]["topic"]
    assert summary["failure_examples"]["source_plan"]
    assert summary["failure_examples"]["clarification"]


def test_personal_finance_english_query_is_not_rejected_as_ood_and_gets_research_source() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query(
        "What is the best way to invest in gold as a hedge against inflation without holding physical gold?",
        debug=True,
    )

    assert "out_of_scope_query" not in result["risk_flags"]
    assert "research_note" in result["source_plan"]


def test_generic_product_compare_and_dca_queries_keep_source_plan_without_missing_entity() -> None:
    clear_service_caches()
    service = build_default_service()

    compare_result = service.analyze_query("证券ETF和债券基金哪个更适合长期定投？", debug=True)
    referential_result = service.analyze_query("这只适合继续定投吗？", debug=True)

    assert compare_result["question_style"] == "advice"
    assert "clarification_required" not in compare_result["risk_flags"]
    assert set(compare_result["source_plan"]) == {"research_note", "faq", "product_doc"}
    assert referential_result["question_style"] == "advice"
    assert "clarification_required" in referential_result["risk_flags"]
    assert "missing_entity" in referential_result["missing_slots"]


def test_stock_why_and_macro_generated_patterns_survive_ood_and_keep_expected_labels() -> None:
    clear_service_caches()
    service = build_default_service()

    why_result = service.analyze_query("五粮液最近为什么跌？", debug=True)
    macro_result = service.analyze_query("美联储议息之后白酒板块会怎么走？", debug=True)
    fundamentals_result = service.analyze_query("平安银行在2024年第一季度的营收和净利润分别是多少？", debug=True)

    assert "out_of_scope_query" not in why_result["risk_flags"]
    assert why_result["question_style"] == "why"
    assert "market_explanation" in {item["label"] for item in why_result["intent_labels"]}
    assert "price" in {item["label"] for item in why_result["topic_labels"]}
    assert "news" in why_result["source_plan"]
    assert "macro_policy_impact" in {item["label"] for item in macro_result["intent_labels"]}
    assert {"macro", "policy"}.issubset({item["label"] for item in macro_result["topic_labels"]})
    assert "fundamental_analysis" in {item["label"] for item in fundamentals_result["intent_labels"]}
    assert "fundamentals" in {item["label"] for item in fundamentals_result["topic_labels"]}


def test_open_ended_known_stock_evaluation_uses_investment_analysis_labels() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("你觉得中国平安怎么样", debug=True)

    intent_labels = {item["label"] for item in result["intent_labels"]}
    topic_labels = {item["label"] for item in result["topic_labels"]}
    assert "out_of_scope_query" not in result["risk_flags"]
    assert result["product_type"]["label"] == "stock"
    assert result["question_style"] == "advice"
    assert {"hold_judgment", "fundamental_analysis", "market_explanation"}.issubset(intent_labels)
    assert {"fundamentals", "valuation", "risk", "price", "news"}.intersection(topic_labels)
    assert {"research_note", "fundamental_sql"}.issubset(set(result["source_plan"]))
    assert "macro_sql" not in result["source_plan"]


def test_unknown_company_open_evaluation_requires_clarification_not_ood() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("你觉得中华汽车怎么样", debug=True)

    assert "out_of_scope_query" not in result["risk_flags"]
    assert result["product_type"]["label"] == "stock"
    assert result["question_style"] == "advice"
    assert "clarification_required" in result["risk_flags"]
    assert "missing_entity" in result["missing_slots"]
    assert result["source_plan"] == []
