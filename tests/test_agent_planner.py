from __future__ import annotations

import pytest

from query_intelligence.agent.planner import MAX_CALLS, plan_from_nlu


def _nlu(**overrides):
    base = {
        "raw_query": "",
        "normalized_query": "",
        "question_style": "fact",
        "product_type": {"label": "stock", "score": 0.99},
        "intent_labels": [],
        "topic_labels": [],
        "entities": [],
        "missing_slots": [],
        "risk_flags": [],
        "source_plan": [],
        "keywords": [],
    }
    base.update(overrides)
    return base


MOUTAI = {"canonical_name": "贵州茅台", "symbol": "600519.SH", "entity_type": "stock"}
WULIANGYE = {"canonical_name": "五粮液", "symbol": "000858.SZ", "entity_type": "stock"}
HS300_ETF = {"canonical_name": "沪深300ETF", "symbol": "510300.SH", "entity_type": "etf"}


def _tools(plan):
    return [
        (call.tool, call.arguments.get("target") or tuple(call.arguments.get("targets", []))) for call in plan.calls
    ]


def test_out_of_scope_and_missing_entity_are_not_planned():
    assert plan_from_nlu(_nlu(risk_flags=["out_of_scope_query"])).skipped_reason == "out_of_scope"
    assert plan_from_nlu(_nlu(missing_slots=["missing_entity"])).skipped_reason == "missing_entity"


def test_single_stock_fact_question_uses_source_plan():
    plan = plan_from_nlu(
        _nlu(
            raw_query="贵州茅台的市盈率是多少",
            normalized_query="贵州茅台的市盈率是多少",
            intent_labels=[{"label": "price_query"}],
            entities=[MOUTAI, {"canonical_name": "市盈率", "symbol": None, "entity_type": "financial_metric"}],
            source_plan=["market_api", "fundamental_sql"],
        )
    )

    assert _tools(plan) == [("get_price_history", "600519.SH"), ("get_fundamentals", "600519.SH")]
    assert plan.targets == ["600519.SH"]
    assert all(call.reason for call in plan.calls)


def test_comparison_expands_every_target():
    plan = plan_from_nlu(
        _nlu(
            raw_query="贵州茅台和五粮液哪个更值得关注",
            question_style="advice",
            intent_labels=[{"label": "peer_compare"}],
            entities=[MOUTAI, WULIANGYE],
            source_plan=["market_api", "fundamental_sql", "news"],
        )
    )

    tools = _tools(plan)
    for symbol in ("600519.SH", "000858.SZ"):
        assert ("get_price_history", symbol) in tools
        assert ("get_fundamentals", symbol) in tools
        assert ("search_news", (symbol,)) in tools
    # Advice questions with news also look at document tone.
    assert ("analyze_sentiment", ("600519.SH",)) in tools


def test_why_question_pulls_news_and_sentiment_even_without_news_in_plan():
    plan = plan_from_nlu(
        _nlu(
            raw_query="茅台最近为什么跌了",
            normalized_query="贵州茅台最近为什么跌了",
            question_style="why",
            intent_labels=[{"label": "market_explanation"}],
            entities=[MOUTAI],
            source_plan=["market_api"],
            keywords=["下跌"],
        )
    )

    tools = _tools(plan)
    assert ("search_news", ("600519.SH",)) in tools
    assert ("analyze_sentiment", ("600519.SH",)) in tools
    news_call = next(call for call in plan.calls if call.tool == "search_news")
    assert news_call.arguments["query"] == "下跌"


def test_technical_terms_add_indicators():
    plan = plan_from_nlu(_nlu(raw_query="茅台现在的RSI和均线怎么样", entities=[MOUTAI], source_plan=["market_api"]))

    assert ("compute_indicators", "600519.SH") in _tools(plan)


def test_macro_question_uses_macro_tool_and_topic_news():
    plan = plan_from_nlu(
        _nlu(
            raw_query="CPI上升对白酒板块有什么影响",
            normalized_query="CPI上升对白酒板块有什么影响",
            product_type={"label": "macro"},
            intent_labels=[{"label": "macro_policy_impact"}],
            entities=[
                {"canonical_name": "CPI", "symbol": None, "entity_type": "macro_indicator"},
                {"canonical_name": "白酒", "symbol": None, "entity_type": "sector"},
            ],
            source_plan=["news", "industry_sql", "research_note", "macro_sql"],
        )
    )

    macro = next(call for call in plan.calls if call.tool == "get_macro_indicators")
    assert macro.arguments["topics"] == ["CPI"]
    assert ("search_news", ()) in _tools(plan)
    assert ("search_knowledge", ()) in _tools(plan)


def test_policy_entity_becomes_macro_topic():
    plan = plan_from_nlu(
        _nlu(
            raw_query="降息对银行股有什么影响",
            product_type={"label": "macro"},
            entities=[{"canonical_name": "降息", "symbol": None, "entity_type": "policy"}],
            source_plan=["macro_sql"],
        )
    )

    [macro] = [call for call in plan.calls if call.tool == "get_macro_indicators"]
    assert macro.arguments["topics"] == ["降息"]


def test_etf_fee_question_goes_to_knowledge_base():
    plan = plan_from_nlu(
        _nlu(
            raw_query="沪深300ETF的费率是多少",
            product_type={"label": "etf"},
            intent_labels=[{"label": "trading_rule_fee"}],
            entities=[HS300_ETF],
            source_plan=["research_note", "faq", "product_doc"],
        )
    )

    assert ("search_knowledge", ()) in _tools(plan)
    assert not any(call.tool == "get_fundamentals" for call in plan.calls)


def test_fundamentals_skipped_for_non_stocks():
    plan = plan_from_nlu(_nlu(entities=[HS300_ETF], source_plan=["market_api", "fundamental_sql"]))

    assert _tools(plan) == [("get_price_history", "510300.SH")]


def test_listed_target_without_plan_gets_default_snapshot():
    plan = plan_from_nlu(_nlu(entities=[MOUTAI]))

    assert _tools(plan) == [("get_price_history", "600519.SH")]


def test_no_entities_no_sources_falls_back_to_knowledge():
    plan = plan_from_nlu(_nlu(raw_query="什么是市盈率", normalized_query="什么是市盈率"))

    assert _tools(plan) == [("search_knowledge", ())]


def test_plan_is_capped_and_deduplicated():
    entities = [
        {"canonical_name": f"S{index}", "symbol": f"60000{index}.SH", "entity_type": "stock"} for index in range(5)
    ]
    plan = plan_from_nlu(
        _nlu(
            raw_query="这些股票的RSI、情绪和公告",
            question_style="why",
            entities=[*entities, entities[0]],
            source_plan=["market_api", "fundamental_sql", "news", "announcement"],
        )
    )

    assert len(plan.targets) == 3
    assert len(plan.calls) <= MAX_CALLS
    keys = [(call.tool, str(call.arguments)) for call in plan.calls]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    ("query", "expected_tool"),
    [
        ("宁德时代最近有什么公告", "search_announcements"),
        ("中国平安的市场情绪怎么样", "analyze_sentiment"),
    ],
)
def test_keyword_triggers(query, expected_tool):
    sources = ["announcement"] if "公告" in query else []
    plan = plan_from_nlu(
        _nlu(
            raw_query=query,
            entities=[{"canonical_name": "X", "symbol": "300750.SZ", "entity_type": "stock"}],
            source_plan=sources,
        )
    )

    assert expected_tool in [call.tool for call in plan.calls]
