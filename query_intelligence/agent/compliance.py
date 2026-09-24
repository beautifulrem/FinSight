"""Compliance guard for agent answers.

Reuses the judgment/causal softening rules from ``scripts/llm_response.py`` and the market-freshness
and disclaimer conventions from ``query_intelligence/chatbot.py``, and additionally removes direct
trading instructions (buy/sell/position calls, price targets) that a model might still produce.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

from ..chatbot import (
    DEFAULT_RISK_DISCLAIMER_EN,
    DEFAULT_RISK_DISCLAIMER_ZH,
    _asks_for_current_market_data,
    _is_known_non_trading_day,
    detect_query_language,
)
from .evidence import AgentEvidence

_ROOT = Path(__file__).resolve().parents[2]

_TRADING_ZH = re.compile(
    r"(?:建议|推荐|可以|应该|应当|不妨|宜|可考虑|考虑|赶紧|立即|果断)\s*(?:逢低|逢高|适当|分批|立即|果断|继续)?\s*"
    r"(?:买入|卖出|加仓|减仓|清仓|满仓|全仓|抄底|建仓|止损|止盈|持有|增持|减持|入场|离场|上车)"
    r"|强烈推荐|强烈建议|目标价|梭哈|满仓|全仓买入"
)
_TRADING_EN = re.compile(
    r"\b(?:you should|we recommend|i recommend|i suggest|consider|it is a good time to|now is the time to)\s+"
    r"(?:buy|sell|short|add|accumulate|trim|dump|hold|exit|enter)\w*\b"
    r"|\b(?:strong buy|strong sell|price target|target price|go all[- ]in)\b",
    re.IGNORECASE,
)
_NEUTRAL_ZH = "是否交易取决于个人风险承受能力、投资期限和持仓情况，以上仅为证据梳理，不构成买卖建议。"
_NEUTRAL_EN = (
    "Whether to trade depends on your risk tolerance, horizon, and existing positions; the above is an evidence "
    "summary, not a trading recommendation."
)


def _guards() -> ModuleType:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from scripts import llm_response

    return llm_response


def contains_trading_instruction(text: str) -> bool:
    return bool(_TRADING_ZH.search(text) or _TRADING_EN.search(text))


def apply_compliance(
    answer: dict[str, Any],
    *,
    query: str,
    nlu_result: dict[str, Any],
    tool_failures: list[str] | None = None,
    market_evidence: list[AgentEvidence] | None = None,
    today: date | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(guarded_answer, notes)``; notes name the rules that changed the answer."""
    guards = _guards()
    zh = detect_query_language(query) == "zh"
    pseudo_retrieval = {"warnings": list(tool_failures or []), "coverage": {}}
    guarded = dict(answer)
    notes: list[str] = []

    text = str(guarded.get("answer") or "")
    softened = guards._soften_answer_text(text, nlu_result, pseudo_retrieval, None, query=query, zh=zh)
    points = [str(point) for point in guarded.get("key_points") or [] if str(point).strip()]
    softened_points = guards._soften_key_points(points, nlu_result, query=query, zh=zh)
    if softened != text or softened_points != points:
        notes.append("softened_judgment_or_causal_language")

    softened, removed_answer = _strip_trading_sentences(softened, zh=zh)
    cleaned_points: list[str] = []
    removed_points = 0
    for point in softened_points:
        if contains_trading_instruction(point):
            removed_points += 1
            continue
        cleaned_points.append(point)
    if removed_answer or removed_points:
        notes.append("removed_trading_instruction")

    if guards._needs_conditional_answer_guard(nlu_result, pseudo_retrieval, None, query):
        prefix = guards._conditional_answer_prefix(nlu_result, zh=zh, query=query).strip()
        if prefix and prefix not in softened:
            softened = f"{prefix}{'' if zh else ' '}{softened}".strip()
            notes.append("conditional_prefix")

    limitations = [str(item) for item in guarded.get("limitations") or [] if str(item).strip()]
    limitations = guards._append_unique(
        limitations, guards._guardrail_limitations(pseudo_retrieval, nlu_result, zh=zh, query=query)
    )

    freshness_point = _freshness_note(query, market_evidence or [], zh=zh, today=today or date.today())
    if freshness_point:
        cleaned_points = [freshness_point, *[point for point in cleaned_points if point != freshness_point]]
        limitations = guards._append_unique(limitations, [freshness_point])
        notes.append("market_freshness")

    guarded["answer"] = softened
    guarded["key_points"] = cleaned_points
    guarded["limitations"] = limitations
    disclaimer = str(guarded.get("risk_disclaimer") or "").strip()
    default_disclaimer = DEFAULT_RISK_DISCLAIMER_ZH if zh else DEFAULT_RISK_DISCLAIMER_EN
    if not disclaimer or contains_trading_instruction(disclaimer):
        guarded["risk_disclaimer"] = default_disclaimer
    return guarded, notes


def _strip_trading_sentences(text: str, *, zh: bool) -> tuple[str, int]:
    sentences = [part for part in re.split(r"(?<=[。！？!?；;])|(?<=\.)\s+", text) if part]
    kept: list[str] = []
    removed = 0
    for sentence in sentences:
        if contains_trading_instruction(sentence):
            removed += 1
            continue
        kept.append(sentence)
    if removed:
        neutral = _NEUTRAL_ZH if zh else _NEUTRAL_EN
        kept.append(neutral if zh else f" {neutral}")
    return "".join(kept).strip(), removed


def _freshness_note(query: str, market_evidence: list[AgentEvidence], *, zh: bool, today: date) -> str | None:
    if not _asks_for_current_market_data(query):
        return None
    today_text = today.isoformat()
    dates = sorted({str(item.as_of)[:10] for item in market_evidence if item.as_of}, reverse=True)
    if _is_known_non_trading_day(today):
        latest = f"，最新可用交易日行情日期为 {dates[0]}" if dates and zh else ""
        latest_en = f"; the latest available trading-day quote is from {dates[0]}" if dates and not zh else ""
        return (
            f"今天（{today_text}）不是 A 股常规交易日，没有当日涨跌行情{latest}。"
            if zh
            else f"Today ({today_text}) is not a regular A-share trading day{latest_en}."
        )
    if not dates:
        return (
            f"未获取到今日（{today_text}）实时行情，不能判断今天的涨跌。"
            if zh
            else f"Today's ({today_text}) real-time quote was not retrieved, so today's move cannot be determined."
        )
    if dates[0] != today_text:
        return (
            f"最新可用行情日期为 {dates[0]}，不是今日（{today_text}）实时行情。"
            if zh
            else f"The latest available market date is {dates[0]}, not today's ({today_text}) real-time quote."
        )
    return None
