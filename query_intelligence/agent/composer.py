"""Deterministic answer composition from tool results, and parsing of LLM answer drafts.

The template composer is used when no LLM is configured or the LLM fails. It only restates values
that tools returned (so the verifier can trace every number) and cites the evidence ids.
"""

from __future__ import annotations

import json
from typing import Any

_MAX_DOCS_PER_TOOL = 3


def parse_answer(content: str | None) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    parsed: Any = None
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json

                parsed = repair_json(text, return_objects=True)
            except Exception:  # json_repair may raise on pathological input
                parsed = None
    if not isinstance(parsed, dict) or not str(parsed.get("answer") or "").strip():
        return {"answer": (content or "").strip(), "key_points": [], "evidence_used": [], "limitations": []}
    return {
        "answer": str(parsed.get("answer") or "").strip(),
        "key_points": _string_list(parsed.get("key_points")),
        "evidence_used": _string_list(parsed.get("evidence_used")),
        "limitations": _string_list(parsed.get("limitations")),
    }


def compose_template(tool_log: list[dict[str, Any]], *, zh: bool, question_style: str = "") -> dict[str, Any]:
    facts: list[str] = []
    evidence_used: list[str] = []
    limitations: list[str] = []
    for entry in tool_log:
        if not entry.get("ok"):
            error = entry.get("error") or {}
            limitations.append(_failure_text(entry.get("tool", ""), error, zh=zh))
            continue
        renderer = _RENDERERS.get(str(entry.get("tool")))
        if renderer is None:
            continue
        for sentence in renderer(entry.get("data") or {}, zh):
            if sentence and sentence not in facts:
                facts.append(sentence)
        for evidence_id in entry.get("evidence_ids") or []:
            if evidence_id not in evidence_used:
                evidence_used.append(evidence_id)

    if facts:
        lead = "根据本次检索到的证据：" if zh else "Based on the evidence retrieved for this question: "
        answer = lead + ("" if zh else " ").join(facts)
        if question_style == "why":
            answer += (
                "以上证据只能提示可能的影响因素，不能据此确定单一原因。"
                if zh
                else " This evidence only points to possible factors; it does not establish a single cause."
            )
    else:
        answer = (
            "本次没有检索到可用于回答该问题的证据。" if zh else "No usable evidence was retrieved for this question."
        )
    return {
        "answer": answer,
        "key_points": facts[:8],
        "evidence_used": evidence_used,
        "limitations": list(dict.fromkeys(limitations)),
    }


def _price(data: dict[str, Any], zh: bool) -> list[str]:
    name, symbol, eid = data.get("name"), data.get("symbol"), data.get("evidence_id")
    close, pct, as_of = data.get("close"), data.get("pct_change_1d"), data.get("as_of")
    if close is None:
        return []
    if zh:
        change = f"，当日涨跌幅 {_num(pct)}%" if pct is not None else ""
        return [f"{name}（{symbol}）最新可用收盘价为 {_num(close)}（{as_of}）{change} [{eid}]。"]
    change = f", daily change {_num(pct)}%" if pct is not None else ""
    return [f"{name} ({symbol}) last available close was {_num(close)} on {as_of}{change} [{eid}]."]


def _indicators(data: dict[str, Any], zh: bool) -> list[str]:
    name, eid = data.get("name"), data.get("evidence_id")
    parts = []
    for key, label in (("ma5", "MA5"), ("ma20", "MA20"), ("rsi_14", "RSI(14)"), ("volatility_20d", "20D vol")):
        if data.get(key) is not None:
            parts.append(f"{label} {_num(data[key])}")
    if not parts:
        return []
    missing = [name.upper().replace("_14", "(14)") for name in data.get("unavailable") or []]
    trend = data.get("trend_signal")
    if zh:
        trend_text = f"，趋势信号为 {trend}" if trend else ""
        gap = f"（历史数据不足，无法计算 {'、'.join(missing)}）" if missing else ""
        return [f"{name} 技术指标：{'，'.join(parts)}{trend_text}{gap} [{eid}]。"]
    trend_text = f", trend signal {trend}" if trend else ""
    gap = f" (not enough history to compute {', '.join(missing)})" if missing else ""
    return [f"{name} technical indicators: {', '.join(parts)}{trend_text}{gap} [{eid}]."]


def _fundamentals(data: dict[str, Any], zh: bool) -> list[str]:
    sentences = []
    metrics = data.get("metrics") or {}
    name, eid, period = data.get("name"), data.get("evidence_id"), data.get("report_date")
    parts = []
    if metrics.get("pe_ttm") is not None:
        parts.append(f"PE(TTM) {_num(metrics['pe_ttm'])}")
    if metrics.get("pb") is not None:
        parts.append(f"PB {_num(metrics['pb'])}")
    roe = metrics.get("roe")
    if roe is not None:
        roe_pct = roe * 100 if abs(float(roe)) <= 1 else roe
        parts.append(f"ROE {_num(roe_pct)}%")
    for key, label_zh, label_en in (("revenue", "营业收入", "revenue"), ("net_profit", "净利润", "net profit")):
        value = metrics.get(key)
        if value is not None and abs(float(value)) >= 1e6:
            parts.append(
                f"{label_zh} {_num(value / 1e8)} 亿元" if zh else f"{label_en} {_num(value / 1e8)} hundred million CNY"
            )
    if parts and eid:
        if zh:
            sentences.append(f"{name} 基本面（报告期 {period}）：{'，'.join(parts)} [{eid}]。")
        else:
            sentences.append(f"{name} fundamentals (period {period}): {', '.join(parts)} [{eid}].")
    industry = data.get("industry") or {}
    industry_metrics = industry.get("metrics") or {}
    industry_parts = [
        f"{label} {_num(industry_metrics[key])}"
        for key, label in (("pe", "PE"), ("pb", "PB"), ("pct_change", "涨跌幅%" if zh else "change %"))
        if industry_metrics.get(key) is not None
    ]
    if industry_parts and industry.get("evidence_id"):
        if zh:
            sentences.append(
                f"所属行业 {industry.get('industry_name')}：{'，'.join(industry_parts)} [{industry['evidence_id']}]。"
            )
        else:
            sentences.append(
                f"Industry {industry.get('industry_name')}: {', '.join(industry_parts)} [{industry['evidence_id']}]."
            )
    return sentences


def _macro(data: dict[str, Any], zh: bool) -> list[str]:
    sentences = []
    for indicator in data.get("indicators") or []:
        if indicator.get("value") is None:
            continue
        unit = indicator.get("unit") or ""
        if zh:
            sentences.append(
                f"{indicator.get('code')} 最新值 {_num(indicator['value'])}{unit}（{indicator.get('date')}）"
                f" [{indicator.get('evidence_id')}]。"
            )
        else:
            sentences.append(
                f"{indicator.get('code')} latest value {_num(indicator['value'])}{unit} ({indicator.get('date')})"
                f" [{indicator.get('evidence_id')}]."
            )
    return sentences


def _documents(data: dict[str, Any], zh: bool) -> list[str]:
    sentences = []
    for document in (data.get("documents") or [])[:_MAX_DOCS_PER_TOOL]:
        title = str(document.get("title") or "").strip()
        if not title:
            continue
        source = document.get("source_name") or document.get("source_type")
        when = str(document.get("publish_time") or "")[:10]
        if zh:
            sentences.append(f"相关资料：《{title}》（{source}，{when}） [{document.get('evidence_id')}]。")
        else:
            sentences.append(f'Related document: "{title}" ({source}, {when}) [{document.get("evidence_id")}].')
    return sentences


def _sentiment(data: dict[str, Any], zh: bool) -> list[str]:
    counts = data.get("label_counts") or {}
    total = sum(int(value) for value in counts.values())
    if not total:
        return []
    targets = "、".join(data.get("targets") or []) if zh else ", ".join(data.get("targets") or [])
    positive, neutral, negative = (int(counts.get(key, 0)) for key in ("positive", "neutral", "negative"))
    eid = data.get("evidence_id")
    if zh:
        return [
            f"{targets} 近期 {total} 篇新闻/公告的语气分布：正面 {positive} 篇、中性 {neutral} 篇、负面 {negative} 篇，"
            f"模型均值 {_num(data.get('mean_score'))}（0.5 为中性） [{eid}]。"
        ]
    return [
        f"Tone of {total} recent documents about {targets}: {positive} positive, {neutral} neutral, "
        f"{negative} negative; mean model score {_num(data.get('mean_score'))} (0.5 is neutral) [{eid}]."
    ]


def _entities(data: dict[str, Any], zh: bool) -> list[str]:
    return []


_RENDERERS = {
    "get_price_history": _price,
    "compute_indicators": _indicators,
    "get_fundamentals": _fundamentals,
    "get_macro_indicators": _macro,
    "search_news": _documents,
    "search_announcements": _documents,
    "search_knowledge": _documents,
    "analyze_sentiment": _sentiment,
    "resolve_entity": _entities,
}


def _failure_text(tool: str, error: dict[str, Any], *, zh: bool) -> str:
    code = error.get("code") or "error"
    message = str(error.get("message") or "")[:160]
    return (
        f"{tool} 未返回可用数据（{code}：{message}）" if zh else f"{tool} returned no usable data ({code}: {message})"
    )


def _num(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
