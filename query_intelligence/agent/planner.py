"""Deterministic tool planner.

Turns an NLU result into an ordered list of tool calls. It is used when no LLM is configured,
when the LLM fails, and as the "workflow" baseline in evaluation. Every planned call carries a
human-readable reason so the plan stays explainable.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

MAX_TARGETS = 3
MAX_CALLS = 14

_LISTED_TYPES = {"stock", "etf", "fund", "index"}
_TECHNICAL_TERMS = re.compile(
    r"rsi|macd|均线|ma5|ma20|布林|boll|技术面|技术指标|超买|超卖|金叉|死叉|趋势|走势|波动率"
    r"|volatility|moving average|technical",
    re.IGNORECASE,
)
_SENTIMENT_TERMS = re.compile(r"情绪|舆情|利好|利空|消息面|市场怎么看|sentiment|tone", re.IGNORECASE)
_PRICE_TERMS = re.compile(
    r"价格|股价|收盘|收在|点位|净值|涨跌|涨|跌|走势|行情|表现|多少钱|\bprice\b|\bclose\b|quote|moved|performance|"
    r"\brise\b|\bfall\b",
    re.IGNORECASE,
)
_VALUATION_TERMS = re.compile(
    r"估值|市盈率|市净率|\bPE\b|\bPB\b|ROE|净资产收益率|盈利|业绩|利润|营收|收入|基本面|财务|贵不贵|便宜|"
    r"valuation|valued|earnings|profit|revenue|fundamental|price-to-book|expensive|cheap",
    re.IGNORECASE,
)
_INDUSTRY_TERMS = re.compile(r"行业|板块|sector|industry", re.IGNORECASE)
_NEWS_TERMS = re.compile(r"新闻|消息|资讯|报道|\bnews\b", re.IGNORECASE)
_ANNOUNCEMENT_TERMS = re.compile(r"公告|披露|年报|季报|半年报|filing|announcement|disclosure", re.IGNORECASE)
_CAUSAL_TERMS = re.compile(r"为什么|原因|因素|影响|导致|驱动|\bwhy\b|\bimpact|\baffect|\bcause|\bdriver", re.IGNORECASE)
_JUDGMENT_STYLES = {"advice", "forecast", "compare"}
_MACRO_TERMS = (
    ("cpi", "CPI"),
    ("inflation", "CPI"),
    ("通胀", "通胀"),
    ("pmi", "PMI"),
    ("m2", "M2"),
    ("money supply", "M2"),
    ("社融", "社融"),
    ("国债", "国债"),
    ("bond yield", "国债"),
    ("利率", "利率"),
    ("interest rate", "利率"),
    ("降息", "降息"),
    ("rate cut", "降息"),
    ("降准", "降准"),
    ("lpr", "LPR"),
)
_PRICE_INTENTS = {"price_query", "market_explanation", "buy_sell_timing", "hold_judgment", "peer_compare"}
_FUNDAMENTAL_INTENTS = {"fundamental_analysis", "valuation_analysis", "peer_compare", "hold_judgment"}
_FUNDAMENTAL_TOPICS = {"fundamentals", "valuation", "earnings"}
_KNOWLEDGE_SOURCES = {"faq", "product_doc", "research_note"}
_MACRO_ENTITY_TYPES = {"macro_indicator", "policy"}


class PlannedCall(BaseModel):
    tool: str
    arguments: dict[str, Any]
    reason: str


class Plan(BaseModel):
    calls: list[PlannedCall] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    skipped_reason: str | None = None


def plan_from_nlu(nlu_result: dict[str, Any]) -> Plan:
    risk_flags = set(nlu_result.get("risk_flags") or [])
    if "out_of_scope_query" in risk_flags:
        return Plan(skipped_reason="out_of_scope")

    entities = nlu_result.get("entities") or []
    listed = _listed_targets(entities)
    if not listed and "missing_entity" in (nlu_result.get("missing_slots") or []):
        return Plan(skipped_reason="missing_entity")

    query = str(nlu_result.get("normalized_query") or nlu_result.get("raw_query") or "")
    raw_query = str(nlu_result.get("raw_query") or query)
    text = f"{raw_query} {query}"
    source_plan = list(nlu_result.get("source_plan") or [])
    sources = set(source_plan)
    intents = {item.get("label") for item in nlu_result.get("intent_labels") or []}
    topics = {item.get("label") for item in nlu_result.get("topic_labels") or []}
    style = str(nlu_result.get("question_style") or "")
    product = str((nlu_result.get("product_type") or {}).get("label") or "")

    calls: list[PlannedCall] = []

    def add(tool: str, arguments: dict[str, Any], reason: str) -> None:
        if len(calls) >= MAX_CALLS:
            return
        if any(call.tool == tool and call.arguments == arguments for call in calls):
            return
        calls.append(PlannedCall(tool=tool, arguments=arguments, reason=reason))

    # Candidate sources come from the NLU source_plan; NLU style/intents/topic scores and explicit
    # lexical cues prune the ones this question does not need, so each kept call has a reason.
    news_topic = any(
        item.get("label") == "news" and float(item.get("score", 0)) >= 0.7
        for item in nlu_result.get("topic_labels") or []
    )
    has_price_cue = bool(_PRICE_TERMS.search(text))
    has_valuation_cue = bool(_VALUATION_TERMS.search(text)) or bool(_INDUSTRY_TERMS.search(text))
    wants_technical = bool(_TECHNICAL_TERMS.search(text))
    causal = style == "why" or bool(_CAUSAL_TERMS.search(text))
    wants_news_docs = bool(_NEWS_TERMS.search(text)) or causal or news_topic
    wants_announcements = bool(_ANNOUNCEMENT_TERMS.search(text))
    wants_sentiment = bool(_SENTIMENT_TERMS.search(text)) or style == "advice"
    wants_price = (
        has_price_cue
        or style in {"why", *_JUDGMENT_STYLES}
        or bool(intents & {"market_explanation", "buy_sell_timing", "hold_judgment"})
    )
    wants_fundamentals = (
        has_valuation_cue
        or style in {"advice", "compare"}
        or bool(intents & {"fundamental_analysis", "valuation_analysis", "peer_compare"})
        or bool(topics & _FUNDAMENTAL_TOPICS)
    ) and ("fundamental_sql" in sources or "industry_sql" in sources or has_valuation_cue)
    wants_news = wants_news_docs and ("news" in sources or causal or bool(_NEWS_TERMS.search(text)))
    wants_announcements = wants_announcements and ("announcement" in sources or bool(_ANNOUNCEMENT_TERMS.search(text)))
    doc_query = " ".join(nlu_result.get("keywords") or [])

    for target in listed:
        symbol = target["symbol"]
        entity_type = target.get("entity_type") or "stock"
        name = target.get("canonical_name") or symbol
        if wants_price:
            add("get_price_history", {"target": symbol}, f"{name}: price cue / question style {style or 'n/a'}")
        if wants_technical:
            add("compute_indicators", {"target": symbol}, f"{name}: question mentions technical indicators")
        if wants_fundamentals and entity_type == "stock":
            add("get_fundamentals", {"target": symbol}, f"{name}: valuation/industry cue or style {style or 'n/a'}")
        if wants_news:
            add(
                "search_news",
                {"query": doc_query, "targets": [symbol], "top_k": 5},
                f"{name}: news evidence needed ({'causal question' if causal else 'news cue'})",
            )
        if wants_announcements:
            add("search_announcements", {"query": doc_query, "targets": [symbol], "top_k": 5}, f"{name}: filings")
        if wants_sentiment:
            add("analyze_sentiment", {"targets": [symbol], "top_k": 6}, f"{name}: tone of recent documents")

    macro_topics = _macro_topics(entities, text)
    if macro_topics or product == "macro" or (not listed and "macro_sql" in sources):
        add(
            "get_macro_indicators",
            {"topics": macro_topics, "query": query},
            "macro indicators requested by source_plan/product type",
        )
        if not listed and style == "why":
            add("search_news", {"query": query, "targets": [], "top_k": 5}, "macro/policy news context")

    if (sources & _KNOWLEDGE_SOURCES and not listed and not macro_topics) or (
        product in {"etf", "fund"} and _is_mechanism_question(intents, topics)
    ):
        add(
            "search_knowledge",
            {"query": query, "targets": [], "top_k": 5},
            "product rules / concepts from knowledge base",
        )

    if not calls and listed:
        for target in listed:
            add("get_price_history", {"target": target["symbol"]}, f"{target.get('canonical_name')}: default snapshot")
    if not calls and query:
        add(
            "search_knowledge",
            {"query": query, "targets": [], "top_k": 5},
            "no specific evidence plan; search knowledge",
        )

    return Plan(calls=calls, targets=[target["symbol"] for target in listed])


def _listed_targets(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    targets = []
    for entity in entities:
        symbol = entity.get("symbol")
        if not symbol or entity.get("entity_type") not in _LISTED_TYPES or symbol in seen:
            continue
        seen.add(symbol)
        targets.append(entity)
        if len(targets) >= MAX_TARGETS:
            break
    return targets


def _macro_topics(entities: list[dict[str, Any]], text: str) -> list[str]:
    topics = [
        str(entity.get("canonical_name"))
        for entity in entities
        if entity.get("entity_type") in _MACRO_ENTITY_TYPES and entity.get("canonical_name")
    ]
    lowered = text.lower()
    known = {topic.lower() for topic in topics}
    for term, topic in _MACRO_TERMS:
        if term in lowered and topic.lower() not in known:
            topics.append(topic)
            known.add(topic.lower())
    return topics[:8]


def _is_mechanism_question(intents: set[str | None], topics: set[str | None]) -> bool:
    return bool(intents & {"product_info", "trading_rule_fee"}) or "product_mechanism" in topics
