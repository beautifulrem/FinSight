from __future__ import annotations

from datetime import date

from agent_fakes import StubService, build_fake_registry

from query_intelligence.agent.followups import next_questions, sentiment_summary
from query_intelligence.agent.graph import AgentRuntime

MOUTAI = {"canonical_name": "贵州茅台", "symbol": "600519.SH", "entity_type": "stock"}
WULIANGYE = {"canonical_name": "五粮液", "symbol": "000858.SZ", "entity_type": "stock"}


def _nlu(entities, style="fact", flags=()):
    return {
        "entities": entities,
        "question_style": style,
        "risk_flags": list(flags),
        "product_type": {"label": "stock"},
    }


def _log(*tools):
    return [{"tool": tool, "ok": True, "data": {}} for tool in tools]


def test_suggestions_point_at_uncovered_evidence():
    questions = next_questions(
        query="贵州茅台最新收盘价",
        route="workflow",
        nlu_result=_nlu([MOUTAI]),
        tool_log=_log("get_price_history"),
        zh=True,
    )

    reasons = [item["reason"] for item in questions]
    assert reasons == ["missing_fundamentals", "missing_technicals", "missing_news"]
    assert questions[0]["question"] == "贵州茅台的估值和盈利指标（PE、ROE）如何？"


def test_comparison_and_english_suggestions():
    questions = next_questions(
        query="Compare Moutai and Wuliangye",
        route="agent",
        nlu_result=_nlu([MOUTAI, WULIANGYE]),
        tool_log=_log(
            "get_price_history", "get_fundamentals", "compute_indicators", "search_news", "analyze_sentiment"
        ),
        zh=False,
    )

    assert questions[0]["reason"] == "compare_targets"
    assert questions[0]["question"].startswith("How do 贵州茅台 and 五粮液 differ")


def test_refusal_and_clarification_suggestions():
    refused = next_questions(query="天气", route="refuse", nlu_result={}, tool_log=[], zh=True)

    assert refused and all(item["reason"] == "finance_reentry_followup" for item in refused)
    assert next_questions(query="x", route="clarify", nlu_result={}, tool_log=[], zh=True) == []


def test_advice_context_suggestions_never_propose_trades():
    questions = next_questions(
        query="茅台还值得持有吗",
        route="workflow",
        nlu_result=_nlu([MOUTAI], style="advice", flags=["investment_advice_like"]),
        tool_log=_log("get_price_history"),
        zh=True,
    )

    assert len(questions) == 3
    assert not any(marker in item["question"] for item in questions for marker in ("买入", "卖出", "持有", "止损"))


def test_sentiment_summary_uses_latest_successful_call():
    log = [
        {"tool": "analyze_sentiment", "ok": True, "data": {"overall_label": "negative", "mean_score": 0.3}},
        {"tool": "analyze_sentiment", "ok": False, "data": None},
        {
            "tool": "analyze_sentiment",
            "ok": True,
            "data": {"overall_label": "positive", "mean_score": 0.7, "targets": ["A"]},
        },
    ]

    assert sentiment_summary(log)["overall_label"] == "positive"
    assert sentiment_summary(_log("get_price_history")) is None


def test_agent_result_contains_sentiment_and_next_questions():
    runtime = AgentRuntime(StubService(), build_fake_registry(), None, today=lambda: date(2026, 9, 24))

    result = runtime.run("茅台最近的市场情绪怎么样")

    assert "label_counts" in result["sentiment"]
    assert result["sentiment"]["evidence_id"] == "sentiment_600519.SH"
    assert 1 <= len(result["next_questions"]) <= 3
