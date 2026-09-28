"""Route a query to refusal, clarification, the fixed workflow, or the agent loop.

The decision uses classical NLU output plus a few explicit lexical markers, and always returns
the reasons that fired, so routing stays explainable.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

Route = Literal["refuse", "clarify", "workflow", "agent"]
Mode = Literal["auto", "workflow", "agent"]

_LISTED_TYPES = {"stock", "etf", "fund", "index"}
_COMPLEX_STYLES = {"why", "compare", "forecast"}
_COMPLEX_INTENTS = {"market_explanation", "macro_policy_impact", "peer_compare"}
_MULTI_HOP_MARKERS = re.compile(
    r"结合|同时|并且|以及.*(影响|变化)|对比|相比|比较|分别|还是|哪个|影响|传导|联动|为什么|原因|归因|"
    r"意味着|关联|关系|是否匹配|综合来看|综合|"
    r"\bcompare|\bversus\b|\bvs\.?\b|\bimpact\b|\baffect|\beffect\b|\brelat(?:ion|ed|es)|\bcorrelat|"
    r"\blinked\b|\bwhy\b|\bcombined\b|\btogether with\b|\band then\b",
    re.IGNORECASE,
)
# Judgment and timing questions need valuation, fundamentals and news plus hedging: never a single lookup.
_JUDGMENT_MARKERS = re.compile(
    r"抄底|能不能买|能买吗|值得买|值不值得|要不要|该不该|会涨|会跌|能涨|还能涨|涨吗|跌吗|见底|高估|低估|买点|卖点|"
    r"止盈|止损|逃顶|上车|还能拿|拿得住|适合定投|适合买|值得持有|长期持有|"
    r"\bshould i\b|\bworth (?:buying|it)\b|\bwill\b.{0,40}\b(?:rise|fall|go up|go down|drop|rebound)\b|"
    r"\bgood time to\b|\bovervalued\b|\bundervalued\b|\bbottom(?:ed)?\b",
    re.IGNORECASE,
)
_WHY_MARKERS = re.compile(
    r"\bdr(?:ove|ives|iving)\b|\bdriver[s]?\b|\bbehind\b|\bwhat caused\b|\breasons? for\b|\bexplain\b",
    re.IGNORECASE,
)
_FOLLOW_UP_MARKERS = re.compile(r"^(那|那么|它|这只|这个|该股|那它|and |what about |how about )", re.IGNORECASE)
_MACRO_ANCHOR = re.compile(
    r"cpi|ppi|pmi|gdp|m2|lpr|社融|利率|国债|降息|降准|通胀|货币供应|宏观|bond yield|interest rate|inflation",
    re.IGNORECASE,
)
_FINANCE_ANCHOR = re.compile(
    r"股票|股价|个股|该股|这只票|那只票|基金|etf|lof|指数|估值|走势|行情|市盈率|市净率|收盘|涨跌|上涨|下跌|大涨|大跌|"
    r"涨停|跌停|业绩|财报|分红|满仓|加仓|减仓|清仓|仓位|会涨|会跌|能涨|涨吗|跌吗|抄底|买入|卖出|能买|值得买|"
    r"(?<![A-Za-z])(?:P/?E|P/?B|ROE)(?![A-Za-z])|股息|不良率|营收|净利|同行|涨了|跌了|费率|股价|"
    r"净资产收益率|毛利率|负债率|市值|公告|年报|季报|现金流|"
    r"\bstocks?\b|\bshares?\b|\bfunds?\b|valuation|overvalued|undervalued|earnings|share price|dividend|"
    r"\bbuy\b|\bsell\b|\binvest(?:ing|ment)?\b|\bdrop\b|\bgo up\b|\brise\b|\bfall\b|"
    r"\brevenue\b|\bnet (?:profit|income)\b|\bmargins?\b|market cap|\bprice\b|\bannouncements?\b",
    re.IGNORECASE,
)
_DANGLING_REFERENCE = re.compile(
    r"(?<!其)它|这只|这支|这个基金|这个标的|这个指数|这个股票|这家|那家|该股|该公司|该基金|那只|那支|"
    r"\bit\b|\bits\b|\b(?:this|that) (?:stock|fund|company|one|etf|index|bank)\b|"
    # plural references ("这两家哪个更值得关注") and dangling why follow-ups ("为什么会这样") in a session
    # without earlier targets: a clarification, never an off-topic refusal
    r"这两家|这两只|这两个|两家公司|两者|二者|它们|\bboth (?:of them|companies|stocks)\b|\bthese two\b|\bthe two\b",
    re.IGNORECASE,
)
# Dangling "why" follow-ups that name neither a target nor an aspect: "为什么会这样", "怎么回事", "why did that
# happen?". Only whole short questions qualify ("大盘今天怎么回事" names the market and is not dangling).
_DANGLING_WHY_ZH = re.compile(
    r"^(?:那|那么|所以|但|可)?(?:这|那)?(?:是)?(?:为什么|为何|怎么|咋)(?:会|能|就)?"
    r"(?:这样|如此|这么\S{0,3}|那样|回事|了)?(?:呢|啊|呀)?[？?。!！]*$|"
    r"^(?:那|那么)?(?:这|那)?(?:是)?(?:什么原因|啥原因)(?:呢|啊|导致的)?[？?。!！]*$|"
    r"^(?:那|那么)?(?:背后的)?原因(?:是什么|是啥|呢|何在)[？?。!！]*$"
)
_DANGLING_WHY_EN = re.compile(
    r"^(?:and |so |but |ok,? )?(?:why(?: is| was| did| does| do| has| would)?(?: that| this)?"
    r"(?: happen(?:ing|ed)?| so| the case)?|how come|what caused (?:that|this)|"
    r"what(?:'s| is| was) behind (?:that|this)|what drove (?:that|this)|"
    r"what(?:'s| is) the reason(?: for (?:that|this))?)\s*[?.!]*$",
    re.IGNORECASE,
)


def is_dangling_why(query: str) -> bool:
    text = (query or "").strip()
    return bool(_DANGLING_WHY_ZH.match(text) or _DANGLING_WHY_EN.match(text))


def has_macro_content(query: str) -> bool:
    return bool(_MACRO_ANCHOR.search(query))


def has_finance_content(query: str) -> bool:
    return bool(_FINANCE_ANCHOR.search(query) or _MACRO_ANCHOR.search(query))


def apply_finance_overrides(nlu_result: dict[str, Any], query: str) -> tuple[dict[str, Any], list[str]]:
    """Correct classical-NLU out-of-scope false positives with explicit finance anchors.

    Returns the (possibly patched) NLU result and the reasons for any change.
    """
    flags = set(nlu_result.get("risk_flags") or [])
    product = (nlu_result.get("product_type") or {}).get("label")
    if "out_of_scope_query" not in flags and product != "out_of_scope":
        return nlu_result, []
    patched = dict(nlu_result)
    patched["risk_flags"] = [flag for flag in nlu_result.get("risk_flags") or [] if flag != "out_of_scope_query"]
    if _MACRO_ANCHOR.search(query):
        patched["product_type"] = {"label": "macro", "score": 0.5}
        patched["source_plan"] = ["macro_sql", "news"]
        return patched, ["override:out_of_scope_with_macro_anchor"]
    if _FINANCE_ANCHOR.search(query):
        patched["product_type"] = {"label": "unknown", "score": 0.5}
        patched["missing_slots"] = sorted({*(nlu_result.get("missing_slots") or []), "missing_entity"})
        return patched, ["override:out_of_scope_with_finance_anchor"]
    if (_DANGLING_REFERENCE.search(query) or is_dangling_why(query)) and len(query) <= 40:
        # A short follow-up about "it" is a conversation turn without context, not an off-topic request.
        patched["product_type"] = {"label": "unknown", "score": 0.5}
        patched["missing_slots"] = sorted({*(nlu_result.get("missing_slots") or []), "missing_entity"})
        return patched, ["override:out_of_scope_dangling_reference"]
    return nlu_result, []


_DEFINITION = re.compile(
    r"什么是|是什么|什么意思|含义|定义|概念|怎么算|如何计算|计算公式|区别|"
    r"\bwhat (?:is|are) (?:a|an|the)?\s*(?:p/?e|p/?b|roe|price|dividend)|\bdefin|\bmeaning\b|\bexplain\b",
    re.IGNORECASE,
)
_CONCEPT_TYPES = {"sector", "financial_metric", "macro_indicator", "policy"}


def drop_fuzzy_concepts(nlu_result: dict[str, Any], query: str) -> tuple[dict[str, Any], list[str]]:
    """Drop fuzzy-matched concept entities whose name is not in the question.

    Fuzzy alias matching is useful for company names with typos, but for short concept names it produces
    false hits ("那家公司最近有公告吗" -> sector 有色金属), which would ground a question that names no target.
    """
    kept, dropped = [], []
    for entity in nlu_result.get("entities") or []:
        name = str(entity.get("canonical_name") or "")
        fuzzy = "fuzzy" in str(entity.get("match_type") or "")
        if fuzzy and entity.get("entity_type") in _CONCEPT_TYPES and name and name.lower() not in query.lower():
            dropped.append(name)
        else:
            kept.append(entity)
    if not dropped:
        return nlu_result, []
    return {**nlu_result, "entities": kept}, [f"dropped_fuzzy_concept:{name}" for name in dropped]


class RouteDecision(BaseModel):
    route: Route
    reasons: list[str] = Field(default_factory=list)
    complexity_score: int = 0
    features: dict[str, Any] = Field(default_factory=dict)


def entity_types_of(entities: list[dict[str, Any]]) -> set[str]:
    return {str(entity.get("entity_type")) for entity in entities if entity.get("entity_type")}


def decide_route(nlu_result: dict[str, Any], *, mode: Mode = "auto", query: str | None = None) -> RouteDecision:
    risk_flags = set(nlu_result.get("risk_flags") or [])
    entities = nlu_result.get("entities") or []
    listed = {
        entity.get("symbol")
        for entity in entities
        if entity.get("symbol") and entity.get("entity_type") in _LISTED_TYPES
    }
    text = query or str(nlu_result.get("raw_query") or nlu_result.get("normalized_query") or "")

    if "out_of_scope_query" in risk_flags or (nlu_result.get("product_type") or {}).get("label") == "out_of_scope":
        return RouteDecision(route="refuse", reasons=["nlu:out_of_scope_query"])
    missing = set(nlu_result.get("missing_slots") or [])
    # A metric alone ("市净率是多少") names no target: only listed, macro, policy or sector entities count.
    targeted = listed or entity_types_of(entities) & {"macro_indicator", "policy", "sector"}
    if "missing_entity" in missing and not targeted:
        return RouteDecision(route="clarify", reasons=["nlu:missing_entity"])
    if "clarification_required" in risk_flags and not listed and not entities:
        return RouteDecision(route="clarify", reasons=["nlu:clarification_required"])

    if not targeted and (_DANGLING_REFERENCE.search(text) or is_dangling_why(text)):
        return RouteDecision(route="clarify", reasons=["dangling_reference"])
    if not targeted and "financial_metric" in entity_types_of(entities) and not _DEFINITION.search(text):
        # "市净率是多少": a company metric with no company ("什么是市净率" is a concept question).
        return RouteDecision(route="clarify", reasons=["metric_without_target"])

    reasons: list[str] = []
    style = str(nlu_result.get("question_style") or "")
    intents = {
        item.get("label") for item in nlu_result.get("intent_labels") or [] if float(item.get("score", 1)) >= 0.5
    }
    entity_types = {entity.get("entity_type") for entity in entities}
    comparison_targets = nlu_result.get("comparison_targets") or []

    if len(listed) >= 2:
        reasons.append(f"multi_entity:{len(listed)}")
    if len(comparison_targets) >= 2:
        reasons.append("comparison_targets")
    multi_hop = bool(_MULTI_HOP_MARKERS.search(text))
    anchored_to_market = bool(listed) or "sector" in entity_types
    if style in _COMPLEX_STYLES:
        # The style classifier alone is noisy (比亚迪 reads as a comparison): with a single target it needs
        # lexical support before a question counts as complex.
        supported = len(listed) >= 2 or multi_hop or bool(_WHY_MARKERS.search(text)) or style == "forecast"
        if supported:
            reasons.append(f"question_style:{style}")
    for intent in sorted(intents & _COMPLEX_INTENTS):
        # A macro question counts as macro-to-market only when it names a market target.
        if intent == "macro_policy_impact" and not (anchored_to_market or multi_hop):
            continue
        reasons.append(f"intent:{intent}")
    if len(intents) >= 3:
        reasons.append(f"multi_intent:{len(intents)}")
    if entity_types & {"macro_indicator", "policy"} and (listed or "sector" in entity_types):
        reasons.append("cross_domain:macro_to_market")
    if multi_hop:
        reasons.append("lexical:multi_hop_marker")
    if _JUDGMENT_MARKERS.search(text):
        reasons.append("lexical:judgment_or_timing")
    if _WHY_MARKERS.search(text):
        reasons.append("lexical:why")
    if _FOLLOW_UP_MARKERS.search(text.strip()):
        reasons.append("lexical:follow_up")

    score = len(reasons)
    features = {
        "listed_entities": len(listed),
        "question_style": style,
        "intents": sorted(label for label in intents if label),
        "entity_types": sorted(label for label in entity_types if label),
    }
    if mode == "workflow":
        return RouteDecision(
            route="workflow", reasons=["mode:workflow", *reasons], complexity_score=score, features=features
        )
    if mode == "agent":
        return RouteDecision(route="agent", reasons=["mode:agent", *reasons], complexity_score=score, features=features)
    route: Route = "agent" if score >= 1 else "workflow"
    if not reasons:
        reasons.append("simple:single_lookup")
    return RouteDecision(route=route, reasons=reasons, complexity_score=score, features=features)
