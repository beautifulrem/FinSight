"""Deterministic tool planner.

Turns an NLU result into an ordered list of tool calls. It is used when no LLM is configured,
when the LLM fails, and as the "workflow" baseline in evaluation. Every planned call carries a
human-readable reason so the plan stays explainable.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .coverage import holding_value_request, requested_price_fields
from .glossary import lookup_concept

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
    r"价格|股价|收盘|收在|点位|净值|涨跌|涨|跌|走势|行情|表现|多少钱|回撤|\bprice\b|\bclose\b|quote|moved|performance|"
    r"drawdown|"
    r"\brise\b|\bfall\b|\brally\b|\brebound\b|\blos(?:e|t|ing)\b|\bgain(?:ed|s)?\b|percent(?:age)? change|"
    r"% change|daily change|\bchange\b",
    re.IGNORECASE,
)
_VALUATION_TERMS = re.compile(
    r"估值|市盈率|市净率|(?<![A-Za-z])(?:P/?E|P/?B)(?![A-Za-z])|ROE|净资产收益率|盈利|业绩|利润|营收|收入|基本面|"
    r"财务|贵不贵|便宜|毛利率|股息率|股息|市值|负债率|负债|杠杆|增速|现金流|净利率|赚钱|赚得|赚了|挣钱|盈利能力|"
    r"(?<![A-Za-z])PEG(?![A-Za-z])|市盈增长比|"
    # (round 9) a fair value or a valuation verdict is answered with the multiples and the industry, not the price
    # alone ("按DCF算…每股值多少", "是不是被低估了")
    r"低估|高估|值多少|值几|公道|公允|内在价值|(?<![A-Za-z])DCF(?![A-Za-z])|现金流折现|市销率|"
    r"(?<![A-Za-z])P/?S(?![A-Za-z])|price[- ]to[- ]sales|"
    r"undervalued|overvalued|\bworth\b(?! buying)|intrinsic value|fair value|"
    r"valuation|valued|earnings|profit|revenue|fundamental|price-to-(?:book|earnings)|expensive|cheap|margin|"
    r"dividend|market cap|debt|leverage|cash ?flow|growth rate|\bearns?\b|more profitable",
    re.IGNORECASE,
)
_INDUSTRY_TERMS = re.compile(r"行业|板块|sector|industry", re.IGNORECASE)
_EPS_TERMS = re.compile(r"每股收益|每股盈利|(?<![A-Za-z])EPS(?![A-Za-z])|earnings per share", re.IGNORECASE)
_NEWS_TERMS = re.compile(r"新闻|消息|资讯|报道|\bnews\b", re.IGNORECASE)
_ANNOUNCEMENT_TERMS = re.compile(r"公告|披露|年报|季报|半年报|filing|announcement|disclosure", re.IGNORECASE)
_CAUSAL_TERMS = re.compile(r"为什么|原因|因素|影响|导致|驱动|\bwhy\b|\bimpact|\baffect|\bcause|\bdriver", re.IGNORECASE)
_JUDGMENT_STYLES = {"advice", "forecast", "compare"}
_MACRO_TERMS = (
    (re.compile(r"cpi", re.I), "CPI"),
    (re.compile(r"inflation|deflation|通缩", re.I), "CPI"),
    (re.compile(r"通胀"), "通胀"),
    (re.compile(r"pmi", re.I), "PMI"),
    (re.compile(r"(?<![A-Za-z])m2(?![A-Za-z0-9])", re.I), "M2"),
    (re.compile(r"money supply", re.I), "M2"),
    (re.compile(r"社融"), "社融"),
    (re.compile(r"国债|\bcgb\b|government bond|treasury yield", re.I), "国债"),
    (re.compile(r"bond yield", re.I), "国债"),
    # "利率" but not the company margins 毛利率 / 净利率
    (re.compile(r"(?<![毛净])利率"), "利率"),
    (re.compile(r"interest rate", re.I), "利率"),
    (re.compile(r"降息"), "降息"),
    (re.compile(r"rate cut", re.I), "降息"),
    (re.compile(r"降准"), "降准"),
    (re.compile(r"lpr", re.I), "LPR"),
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
    query = str(nlu_result.get("normalized_query") or nlu_result.get("raw_query") or "")
    raw_query = str(nlu_result.get("raw_query") or query)
    # "北向资金是啥", "什么是两融": a market concept in the curated glossary is evidence of its own.
    concept = None if listed else lookup_concept(f"{raw_query} {query}")
    if concept is not None:
        return Plan(
            calls=[
                PlannedCall(
                    tool="explain_concept",
                    arguments={"term": concept.term, "query": raw_query[:200]},
                    reason=f"glossary concept: {concept.term}",
                )
            ]
        )
    if not listed and "missing_entity" in (nlu_result.get("missing_slots") or []):
        return Plan(skipped_reason="missing_entity")

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
    request = requested_price_fields(raw_query)
    sectors = [
        str(entity.get("canonical_name"))
        for entity in entities
        if entity.get("entity_type") == "sector" and entity.get("canonical_name")
    ]
    # A named sector next to a stock ("insurers" with Ping An in scope) asks for the industry snapshot, which
    # comes with the stock's fundamentals.
    sector_member = bool(sectors) or any(entity.get("match_type") == "session_sector_member" for entity in entities)
    # (round 11, G5) a holding's value needs the close; an EPS question gets the close for the implied EPS
    holding = holding_value_request(raw_query) is not None
    asks_eps = bool(_EPS_TERMS.search(text))
    has_price_cue = bool(_PRICE_TERMS.search(text)) or request.needs_quote or holding or asks_eps
    explicit_valuation_cue = bool(_VALUATION_TERMS.search(text)) or bool(_INDUSTRY_TERMS.search(text)) or asks_eps
    has_valuation_cue = explicit_valuation_cue or (sector_member and bool(listed))
    wants_technical = bool(_TECHNICAL_TERMS.search(text)) or request.needs_indicators
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
            arguments: dict[str, Any] = {"target": symbol}
            if request.closes > 10:
                arguments["days"] = min(request.closes, 30)
            add("get_price_history", arguments, f"{name}: price cue / question style {style or 'n/a'}")
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
    # "白酒板块整体跌了吗" with no member stock in scope: get_fundamentals returns the industry snapshot. A macro
    # question that mentions a sector ("CPI上升对白酒板块有什么影响") stays a macro question.
    sector_planned = bool(sectors) and not listed and not macro_topics and (explicit_valuation_cue or has_price_cue)
    if sector_planned:
        for sector in sectors[:MAX_TARGETS]:
            add("get_fundamentals", {"target": sector}, f"{sector}: industry snapshot for a sector question")

    # A question about a named security asks for macro data only when it names a macro topic: the product
    # classifier alone ("And the P/B?" read as macro) must not replace the carried target's metric with CPI/PMI.
    sector_only = bool(sectors) and not listed and not macro_topics
    if macro_topics or (
        not sector_only and ((product == "macro" and not listed) or (not listed and "macro_sql" in sources))
    ):
        add(
            "get_macro_indicators",
            {"topics": macro_topics, "query": query},
            "macro indicators requested by source_plan/product type",
        )
        if not listed and style == "why":
            add("search_news", {"query": query, "targets": [], "top_k": 5}, "macro/policy news context")

    if (sources & _KNOWLEDGE_SOURCES and not listed and not macro_topics and not sector_planned) or (
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
    known = {topic.lower() for topic in topics}
    for pattern, topic in _MACRO_TERMS:
        if pattern.search(text) and topic.lower() not in known:
            topics.append(topic)
            known.add(topic.lower())
    return topics[:8]


def _is_mechanism_question(intents: set[str | None], topics: set[str | None]) -> bool:
    return bool(intents & {"product_info", "trading_rule_fee"}) or "product_mechanism" in topics
