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

    wants_price = "market_api" in sources or bool(intents & _PRICE_INTENTS) or "price" in topics
    wants_technical = bool(_TECHNICAL_TERMS.search(text))
    wants_fundamentals = (
        "fundamental_sql" in sources or bool(intents & _FUNDAMENTAL_INTENTS) or bool(topics & _FUNDAMENTAL_TOPICS)
    )
    wants_news = "news" in sources or style == "why" or "news" in topics
    wants_announcements = "announcement" in sources
    wants_sentiment = bool(_SENTIMENT_TERMS.search(text)) or (style in {"why", "advice"} and wants_news)
    doc_query = " ".join(nlu_result.get("keywords") or [])

    for target in listed:
        symbol = target["symbol"]
        entity_type = target.get("entity_type") or "stock"
        name = target.get("canonical_name") or symbol
        if wants_price:
            add("get_price_history", {"target": symbol}, f"{name}: price-related question (source_plan/intent)")
        if wants_technical:
            add("compute_indicators", {"target": symbol}, f"{name}: question mentions technical indicators")
        if wants_fundamentals and entity_type == "stock":
            add("get_fundamentals", {"target": symbol}, f"{name}: fundamentals requested by source_plan/intent")
        if wants_news:
            add(
                "search_news",
                {"query": doc_query, "targets": [symbol], "top_k": 5},
                f"{name}: news evidence needed ({'why question' if style == 'why' else 'source_plan'})",
            )
        if wants_announcements:
            add("search_announcements", {"query": doc_query, "targets": [symbol], "top_k": 5}, f"{name}: filings")
        if wants_sentiment:
            add("analyze_sentiment", {"targets": [symbol], "top_k": 6}, f"{name}: tone of recent documents")

    macro_topics = _macro_topics(entities, text)
    if "macro_sql" in sources or product == "macro" or macro_topics:
        add(
            "get_macro_indicators",
            {"topics": macro_topics, "query": query},
            "macro indicators requested by source_plan/product type",
        )
        if wants_news and not listed:
            add("search_news", {"query": query, "targets": [], "top_k": 5}, "macro/policy news context")

    if (sources & _KNOWLEDGE_SOURCES and not listed) or (
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
    for term in ("cpi", "pmi", "m2", "国债", "利率", "降息", "降准", "lpr", "通胀", "社融"):
        if term in lowered and term not in [topic.lower() for topic in topics]:
            topics.append(term.upper() if term in {"cpi", "pmi", "m2", "lpr"} else term)
    return topics[:8]


def _is_mechanism_question(intents: set[str | None], topics: set[str | None]) -> bool:
    return bool(intents & {"product_info", "trading_rule_fee"}) or "product_mechanism" in topics
