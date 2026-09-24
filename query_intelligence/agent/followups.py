"""Online next-question suggestions and sentiment summary for agent answers.

Suggestions are deterministic: they point at evidence the current turn did not cover yet (valuation,
technicals, news, peers, macro links), so they are cheap, explainable, and always answerable by the
agent's tools. They are then passed through the existing judgment/causal sanitizer from
``scripts/llm_response.py`` so they never suggest trading actions.
"""

from __future__ import annotations

from typing import Any

from .compliance import _guards


def sentiment_summary(tool_log: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in reversed(tool_log):
        if entry.get("tool") == "analyze_sentiment" and entry.get("ok"):
            data = entry.get("data") or {}
            return {
                "targets": data.get("targets") or [],
                "overall_label": data.get("overall_label"),
                "mean_score": data.get("mean_score"),
                "label_counts": data.get("label_counts") or {},
                "backend": data.get("backend"),
                "evidence_id": data.get("evidence_id"),
            }
    return None


def next_questions(
    *,
    query: str,
    route: str,
    nlu_result: dict[str, Any],
    tool_log: list[dict[str, Any]],
    zh: bool,
    limit: int = 3,
) -> list[dict[str, Any]]:
    guards = _guards()
    if route == "refuse":
        return guards._out_of_scope_next_question_response(zh=zh)["predictions"][:limit]
    if route == "clarify":
        return []

    used = {entry.get("tool") for entry in tool_log if entry.get("ok")}
    names = [
        str(entity.get("canonical_name"))
        for entity in nlu_result.get("entities") or []
        if entity.get("symbol") and entity.get("entity_type") in {"stock", "etf", "fund", "index"}
    ][:2]
    is_stock = any(entity.get("entity_type") == "stock" for entity in nlu_result.get("entities") or [])
    candidates: list[tuple[str, str]] = []

    if len(names) >= 2:
        candidates.append(
            (
                f"{names[0]}和{names[1]}在估值与盈利质量上有什么差异？"
                if zh
                else f"How do {names[0]} and {names[1]} differ in valuation and earnings quality?",
                "compare_targets",
            )
        )
    for name in names[:1]:
        if is_stock and "get_fundamentals" not in used:
            candidates.append(
                (
                    f"{name}的估值和盈利指标（PE、ROE）如何？" if zh else f"What are {name}'s PE and ROE?",
                    "missing_fundamentals",
                )
            )
        if "compute_indicators" not in used:
            candidates.append(
                (
                    f"{name}近期的均线和 RSI 指标怎么样？" if zh else f"What do {name}'s moving averages and RSI show?",
                    "missing_technicals",
                )
            )
        if not used & {"search_news", "search_announcements"}:
            candidates.append(
                (
                    f"{name}最近有哪些重要新闻或公告？" if zh else f"What recent news or announcements affect {name}?",
                    "missing_news",
                )
            )
        if "analyze_sentiment" not in used:
            candidates.append(
                (
                    f"{name}近期新闻的整体语气如何？" if zh else f"What is the tone of recent news about {name}?",
                    "missing_sentiment",
                )
            )
        if len(names) == 1:
            candidates.append(
                (
                    f"{name}与同行业公司相比表现如何？" if zh else f"How does {name} compare with its industry peers?",
                    "peer_context",
                )
            )
    if "get_macro_indicators" in used:
        candidates.append(
            (
                "这些宏观指标变化对哪些行业影响更大？"
                if zh
                else "Which sectors are most sensitive to these macro indicators?",
                "macro_transmission",
            )
        )
    if not candidates:
        candidates.append(
            (
                "还需要哪些证据才能更完整地回答这个问题？"
                if zh
                else "What further evidence would make this answer more complete?",
                "evidence_gap",
            )
        )

    predictions = [
        {"question": question, "score": round(0.9 - index * 0.05, 4), "reason": reason}
        for index, (question, reason) in enumerate(dict.fromkeys(candidates))
    ]
    sanitized = guards._sanitize_next_questions_for_context(predictions, query=query, nlu_result=nlu_result)
    return sanitized[:limit]
