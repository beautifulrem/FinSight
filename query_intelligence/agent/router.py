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
    r"结合|同时|并且|以及.*(影响|变化)|对比|相比|比较|分别|还是|哪个|影响|传导|联动|为什么|原因|归因"
    r"|\bcompare|\bversus\b|\bvs\.?\b|\bimpact\b|\bwhy\b|\bcombined\b|\btogether with\b|\band then\b",
    re.IGNORECASE,
)
_FOLLOW_UP_MARKERS = re.compile(r"^(那|那么|它|这只|这个|该股|那它|and |what about |how about )", re.IGNORECASE)
_MACRO_ANCHOR = re.compile(
    r"cpi|ppi|pmi|gdp|m2|lpr|社融|利率|国债|降息|降准|通胀|货币供应|宏观|bond yield|interest rate|inflation",
    re.IGNORECASE,
)
_FINANCE_ANCHOR = re.compile(
    r"股票|股价|个股|该股|这只票|那只票|基金|etf|lof|指数|估值|走势|行情|市盈率|市净率|收盘|涨跌|上涨|下跌|大涨|大跌|"
    r"涨停|跌停|业绩|财报|分红|满仓|加仓|减仓|清仓|仓位|"
    r"\bstocks?\b|\bshares?\b|\bfunds?\b|valuation|overvalued|undervalued|earnings|share price|dividend",
    re.IGNORECASE,
)
_DANGLING_REFERENCE = re.compile(
    r"(?<!其)它|这只|这支|这个基金|这个标的|这家|该股|该公司|该基金|那只|那支|"
    r"\bit\b|\bthis (?:stock|fund|company|one)\b|\bthat (?:stock|fund|company|one)\b",
    re.IGNORECASE,
)


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
    return nlu_result, []


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
    if "missing_entity" in missing and not entities:
        return RouteDecision(route="clarify", reasons=["nlu:missing_entity"])
    if "clarification_required" in risk_flags and not listed and not entities:
        return RouteDecision(route="clarify", reasons=["nlu:clarification_required"])

    grounded = listed or entity_types_of(entities) & {"macro_indicator", "policy", "sector"}
    if not grounded and _DANGLING_REFERENCE.search(text):
        return RouteDecision(route="clarify", reasons=["dangling_reference"])

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
    if style in _COMPLEX_STYLES:
        reasons.append(f"question_style:{style}")
    for intent in sorted(intents & _COMPLEX_INTENTS):
        reasons.append(f"intent:{intent}")
    if len(intents) >= 3:
        reasons.append(f"multi_intent:{len(intents)}")
    if entity_types & {"macro_indicator", "policy"} and (listed or "sector" in entity_types):
        reasons.append("cross_domain:macro_to_market")
    if _MULTI_HOP_MARKERS.search(text):
        reasons.append("lexical:multi_hop_marker")
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
