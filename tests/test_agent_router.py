from __future__ import annotations

import pytest

from query_intelligence.agent.router import decide_route

MOUTAI = {"canonical_name": "贵州茅台", "symbol": "600519.SH", "entity_type": "stock"}
WULIANGYE = {"canonical_name": "五粮液", "symbol": "000858.SZ", "entity_type": "stock"}
PINGAN = {"canonical_name": "中国平安", "symbol": "601318.SH", "entity_type": "stock"}
ETF = {"canonical_name": "沪深300ETF", "symbol": "510300.SH", "entity_type": "etf"}
CPI = {"canonical_name": "CPI", "symbol": None, "entity_type": "macro_indicator"}
LIQUOR = {"canonical_name": "白酒", "symbol": None, "entity_type": "sector"}
RATE_CUT = {"canonical_name": "降息", "symbol": None, "entity_type": "policy"}
PE = {"canonical_name": "市盈率", "symbol": None, "entity_type": "financial_metric"}


def _nlu(query, *, entities=(), style="fact", intents=(), flags=(), missing=(), comparison=(), product="stock"):
    return {
        "raw_query": query,
        "normalized_query": query,
        "question_style": style,
        "product_type": {"label": product},
        "intent_labels": [{"label": label, "score": 0.8} for label in intents],
        "entities": list(entities),
        "risk_flags": list(flags),
        "missing_slots": list(missing),
        "comparison_targets": list(comparison),
    }


CASES = [
    # (query, nlu kwargs, expected route)
    ("今天天气怎么样", {"flags": ["out_of_scope_query"], "product": "out_of_scope"}, "refuse"),
    ("帮我写一首诗", {"product": "out_of_scope"}, "refuse"),
    ("这只股票能买吗", {"missing": ["missing_entity"], "flags": ["clarification_required"]}, "clarify"),
    ("能买吗", {"flags": ["clarification_required"]}, "clarify"),
    ("贵州茅台的市盈率是多少", {"entities": [MOUTAI, PE], "intents": ["price_query"]}, "workflow"),
    ("贵州茅台最新收盘价", {"entities": [MOUTAI], "intents": ["price_query"]}, "workflow"),
    ("中国平安的ROE是多少", {"entities": [PINGAN], "intents": ["fundamental_analysis"]}, "workflow"),
    ("沪深300ETF的费率是多少", {"entities": [ETF], "intents": ["trading_rule_fee"], "product": "etf"}, "workflow"),
    ("贵州茅台所属行业", {"entities": [MOUTAI]}, "workflow"),
    ("CPI最新数据", {"entities": [CPI], "product": "macro"}, "workflow"),
    ("什么是市盈率", {"entities": [PE]}, "workflow"),
    ("ETF怎么申购", {"intents": ["product_info"], "product": "etf"}, "workflow"),
    ("贵州茅台近期公告", {"entities": [MOUTAI]}, "workflow"),
    ("你觉得中国平安怎么样", {"entities": [PINGAN], "style": "advice"}, "workflow"),
    (
        "贵州茅台和五粮液哪个更值得关注",
        {"entities": [MOUTAI, WULIANGYE], "comparison": ["贵州茅台", "五粮液"]},
        "agent",
    ),
    ("茅台最近为什么跌了", {"entities": [MOUTAI], "style": "why", "intents": ["market_explanation"]}, "agent"),
    ("Why did Ping An fall recently?", {"entities": [PINGAN], "style": "why"}, "agent"),
    ("CPI上升对白酒板块有什么影响", {"entities": [CPI, LIQUOR], "intents": ["macro_policy_impact"]}, "agent"),
    ("降息对中国平安有什么影响", {"entities": [RATE_CUT, PINGAN]}, "agent"),
    ("对比茅台和五粮液的估值", {"entities": [MOUTAI, WULIANGYE]}, "agent"),
    ("茅台未来一个月会涨吗", {"entities": [MOUTAI], "style": "forecast"}, "agent"),
    ("茅台和沪深300ETF的走势相比如何", {"entities": [MOUTAI, ETF]}, "agent"),
    ("结合估值和新闻分析茅台", {"entities": [MOUTAI]}, "agent"),
    (
        "茅台的估值、公告和舆情",
        {"entities": [MOUTAI], "intents": ["valuation_analysis", "fundamental_analysis", "news_query"]},
        "agent",
    ),
    ("那它的市盈率呢", {"entities": [MOUTAI]}, "agent"),
    ("那五粮液呢", {"entities": [WULIANGYE]}, "agent"),
    ("Compare Moutai and Wuliangye", {"entities": [MOUTAI, WULIANGYE]}, "agent"),
    ("What is the impact of rate cuts on banks?", {"entities": [RATE_CUT], "product": "macro"}, "agent"),
    ("茅台为什么涨", {"entities": [MOUTAI], "style": "why"}, "agent"),
    ("茅台和五粮液分别的ROE", {"entities": [MOUTAI, WULIANGYE]}, "agent"),
    ("宁德时代和比亚迪还是茅台", {"entities": [MOUTAI, WULIANGYE, PINGAN]}, "agent"),
]


@pytest.mark.parametrize(("query", "kwargs", "expected"), CASES, ids=[case[0] for case in CASES])
def test_route_table(query, kwargs, expected):
    decision = decide_route(_nlu(query, **kwargs))

    assert decision.route == expected, decision.reasons
    assert decision.reasons


def test_route_table_has_enough_cases():
    assert len(CASES) >= 30


def test_mode_overrides_do_not_bypass_guards():
    out_of_scope = _nlu("今天天气怎么样", flags=["out_of_scope_query"])
    simple = _nlu("贵州茅台最新收盘价", entities=[MOUTAI])
    complex_query = _nlu("茅台最近为什么跌了", entities=[MOUTAI], style="why")

    assert decide_route(out_of_scope, mode="agent").route == "refuse"
    assert decide_route(simple, mode="agent").route == "agent"
    assert decide_route(complex_query, mode="workflow").route == "workflow"
    assert decide_route(complex_query, mode="workflow").reasons[0] == "mode:workflow"


def test_decision_exposes_features_and_score():
    decision = decide_route(_nlu("对比茅台和五粮液的估值", entities=[MOUTAI, WULIANGYE], style="compare"))

    assert decision.features["listed_entities"] == 2
    assert decision.complexity_score >= 2
    assert "multi_entity:2" in decision.reasons and "question_style:compare" in decision.reasons


@pytest.mark.parametrize(
    ("query", "reason", "route"),
    [
        ("最新一期PMI数据是多少？", "override:out_of_scope_with_macro_anchor", "workflow"),
        ("M2增速变化对股市流动性有什么影响", "override:out_of_scope_with_macro_anchor", "agent"),
        ("该股后面走势怎么样", "override:out_of_scope_with_finance_anchor", "clarify"),
        ("Is the stock overvalued?", "override:out_of_scope_with_finance_anchor", "clarify"),
    ],
)
def test_out_of_scope_false_positives_are_overridden(query, reason, route):
    from query_intelligence.agent.router import apply_finance_overrides

    nlu = _nlu(query, flags=["out_of_scope_query"], product="out_of_scope")
    patched, reasons = apply_finance_overrides(nlu, query)
    decision = decide_route(patched)

    assert reasons == [reason]
    assert decision.route == route
    assert "out_of_scope_query" not in patched["risk_flags"]


@pytest.mark.parametrize(
    "query", ["明天北京会下雨吗", "我跌倒了怎么办", "仓库管理怎么做", "其它问题", "Write a poem about the sea."]
)
def test_real_out_of_scope_queries_are_not_overridden(query):
    from query_intelligence.agent.router import apply_finance_overrides

    nlu = _nlu(query, flags=["out_of_scope_query"], product="out_of_scope")
    patched, reasons = apply_finance_overrides(nlu, query)

    assert reasons == [] and patched is nlu
    assert decide_route(patched).route == "refuse"


@pytest.mark.parametrize(
    "query", ["它的估值高吗", "这家公司的业绩怎么样", "How is that fund doing?", "Is it overvalued?"]
)
def test_dangling_references_ask_for_clarification(query):
    decision = decide_route(_nlu(query))

    assert decision.route == "clarify" and decision.reasons == ["dangling_reference"]


def test_references_with_a_resolved_entity_are_not_dangling():
    assert decide_route(_nlu("它的估值高吗", entities=[MOUTAI])).route != "clarify"
    assert decide_route(_nlu("其它行业的估值", entities=[LIQUOR])).route != "clarify"
