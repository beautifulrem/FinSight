"""Shared fakes for agent tests: a stub NLU service and deterministic tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from query_intelligence.agent.evidence import AgentEvidence
from query_intelligence.agent.tools import ToolFailure, ToolOutput, ToolRegistry, ToolSpec

MOUTAI = {"canonical_name": "贵州茅台", "symbol": "600519.SH", "entity_type": "stock"}
WULIANGYE = {"canonical_name": "五粮液", "symbol": "000858.SZ", "entity_type": "stock"}

PRICES = {
    "600519.SH": {"name": "贵州茅台", "close": 1409.5, "pct_change_1d": -0.1778, "as_of": "2026-04-22"},
    "000858.SZ": {"name": "五粮液", "close": 101.2, "pct_change_1d": 1.25, "as_of": "2026-04-22"},
}
FUNDAMENTALS = {
    "600519.SH": {"pe_ttm": 24.6, "pb": 8.1, "roe": 0.33, "revenue": 174120000000},
    "000858.SZ": {"pe_ttm": 15.2, "pb": 3.9, "roe": 0.24, "revenue": 89000000000},
}


def nlu_for(query: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "query_id": "q",
        "raw_query": query,
        "normalized_query": query,
        "question_style": "fact",
        "product_type": {"label": "stock", "score": 0.99},
        "intent_labels": [],
        "topic_labels": [],
        "entities": [],
        "comparison_targets": [],
        "missing_slots": [],
        "risk_flags": [],
        "source_plan": [],
        "keywords": [],
    }
    if "天气" in query or "weather" in query.lower():
        base.update(product_type={"label": "out_of_scope"}, risk_flags=["out_of_scope_query"])
    elif "这只股票" in query:
        base.update(missing_slots=["missing_entity"], risk_flags=["clarification_required"])
    elif "五粮液" in query and "茅台" in query:
        base.update(
            question_style="compare",
            entities=[MOUTAI, WULIANGYE],
            comparison_targets=["贵州茅台", "五粮液"],
            source_plan=["market_api", "fundamental_sql"],
        )
    elif "为什么" in query or "why" in query.lower():
        base.update(
            question_style="why",
            entities=[MOUTAI],
            intent_labels=[{"label": "market_explanation", "score": 0.9}],
            source_plan=["market_api", "news"],
        )
    elif "茅台" in query or "moutai" in query.lower():
        base.update(entities=[MOUTAI], source_plan=["market_api", "fundamental_sql"])
    return base


class StubService:
    """Implements the subset of QueryIntelligenceService used by the agent."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def analyze_query(self, query, user_profile=None, dialog_context=None, debug=False):
        self.calls.append({"query": query, "dialog_context": dialog_context or []})
        return nlu_for(query)


class TargetInput(BaseModel):
    target: str = Field(min_length=1)


class DocsInput(BaseModel):
    query: str = ""
    targets: list[str] = Field(default_factory=list)
    top_k: int = 5


class SentimentInput(BaseModel):
    targets: list[str]
    top_k: int = 6
    query: str = ""


def build_fake_registry(*, fail: set[str] | None = None, news_text: str | None = None) -> ToolRegistry:
    fail = fail or set()

    def price(args: TargetInput) -> ToolOutput:
        if "get_price_history" in fail:
            raise TimeoutError("upstream timeout")
        row = PRICES.get(args.target)
        if row is None:
            raise ToolFailure("not_found", f"no data for {args.target}")
        evidence_id = f"price_{args.target}"
        data = {"symbol": args.target, **row, "evidence_id": evidence_id}
        evidence = AgentEvidence(
            evidence_id=evidence_id, kind="structured", source_type="market_api", as_of=row["as_of"], payload=data
        )
        return ToolOutput(data=data, evidence=[evidence])

    def fundamentals(args: TargetInput) -> ToolOutput:
        if "get_fundamentals" in fail:
            raise ToolFailure("unavailable", "fundamentals source down")
        metrics = FUNDAMENTALS[args.target]
        evidence_id = f"fundamental_{args.target}"
        data = {
            "symbol": args.target,
            "name": PRICES[args.target]["name"],
            "report_date": "2025-12-31",
            "metrics": metrics,
            "evidence_id": evidence_id,
            "industry": None,
        }
        evidence = AgentEvidence(
            evidence_id=evidence_id, kind="structured", source_type="fundamental_sql", payload=metrics
        )
        return ToolOutput(data=data, evidence=[evidence])

    def news(args: DocsInput) -> ToolOutput:
        text = news_text or "贵州茅台2025年度净利润823.20亿元，同比增长4.5%"
        evidence = AgentEvidence(
            evidence_id="news_1",
            kind="document",
            source_type="news",
            title="贵州茅台发布年报",
            source_name="财经日报",
            as_of="2026-04-16",
            text_excerpt=text,
        )
        document = {
            "evidence_id": "news_1",
            "source_type": "news",
            "title": evidence.title,
            "source_name": "财经日报",
            "publish_time": "2026-04-16",
            "excerpt": text,
        }
        return ToolOutput(data={"documents": [document], "targets": args.targets}, evidence=[evidence])

    def sentiment(args: SentimentInput) -> ToolOutput:
        evidence = AgentEvidence(
            evidence_id="sentiment_600519.SH",
            kind="structured",
            source_type="sentiment_summary",
            payload={"mean_score": 0.62, "label_counts": {"positive": 2, "neutral": 1}, "neutral_score": 0.5},
        )
        data = {
            "targets": ["贵州茅台"],
            "label_counts": {"positive": 2, "neutral": 1},
            "mean_score": 0.62,
            "evidence_id": "sentiment_600519.SH",
        }
        return ToolOutput(data=data, evidence=[evidence])

    registry = ToolRegistry(sleep=lambda _s: None)
    registry.register(ToolSpec("get_price_history", "Price.", TargetInput, price, timeout_s=2, max_retries=1))
    registry.register(ToolSpec("get_fundamentals", "Fundamentals.", TargetInput, fundamentals, timeout_s=2))
    registry.register(ToolSpec("search_news", "News.", DocsInput, news, timeout_s=2))
    registry.register(ToolSpec("analyze_sentiment", "Sentiment.", SentimentInput, sentiment, timeout_s=2))
    return registry
