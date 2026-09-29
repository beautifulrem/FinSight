"""Deterministic answer composition from tool results, and parsing of LLM answer drafts.

The template composer is used when no LLM is configured or the LLM fails. It only restates values
that tools returned (so the verifier can trace every number) and cites the evidence ids.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .coverage import (
    EXTRA_METRIC_FIELDS,
    PriceRequest,
    asks_about_industry,
    coverage_gaps,
    failed_target_statements,
    indicator_gaps,
    industry_gaps,
    macro_gaps,
    non_stock_fundamental_gaps,
    requested_metrics,
    requested_price_fields,
)

_MAX_DOCS_PER_TOOL = 3


def answer_json_status(content: str | None) -> str:
    """``ok`` (valid JSON with an answer), ``repaired`` (needed json_repair) or ``failed`` (used as plain text)."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    try:
        parsed = json.loads(text)
        return "ok" if isinstance(parsed, dict) and str(parsed.get("answer") or "").strip() else "failed"
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json

        parsed = repair_json(text, return_objects=True)
    except Exception:
        return "failed"
    return "repaired" if isinstance(parsed, dict) and str(parsed.get("answer") or "").strip() else "failed"


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


def compose_template(
    tool_log: list[dict[str, Any]],
    *,
    zh: bool,
    question_style: str = "",
    query: str = "",
    names: dict[str, str] | None = None,
    types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Restate tool results with citations.

    With ``query``, metrics and periods the question asks for but the evidence lacks are stated first
    (``coverage_gaps``), and targets whose data could not be retrieved are named, so a question about
    "2019年营收" or a dividend yield is never answered silently with other numbers. Requested price details
    (recent closes, high/low, volume, N-day return, price vs. MA) are stated when the evidence has them and
    named as unavailable when it does not; a sector question leads with the industry snapshot.
    """
    facts: list[str] = []
    evidence_used: list[str] = []
    limitations: list[str] = []
    extra_keys = {field for metric in requested_metrics(query) for field in metric.fields}
    request = requested_price_fields(query)
    industry_first = asks_about_industry(query)
    for entry in tool_log:
        if not entry.get("ok"):
            error = entry.get("error") or {}
            limitations.append(_failure_text(entry.get("tool", ""), error, zh=zh))
            continue
        tool = str(entry.get("tool"))
        renderer = _RENDERERS.get(tool)
        if renderer is None:
            continue
        data = entry.get("data") or {}
        if tool == "get_price_history":
            sentences = _price(data, zh, request)
        elif tool == "compute_indicators":
            sentences = _indicators(data, zh, request)
        elif tool == "get_fundamentals":
            sentences = _fundamentals(data, zh, industry_first=industry_first)
            if extra_keys:
                sentences = [*sentences, *_extra_metrics(data, extra_keys, zh)]
        else:
            sentences = renderer(data, zh)
        for sentence in sentences:
            if sentence and sentence not in facts:
                facts.append(sentence)
        if sentences and renderer is _documents:
            limitations.append(DOCUMENT_TEXT_LIMITATION_ZH if zh else DOCUMENT_TEXT_LIMITATION_EN)
        for evidence_id in entry.get("evidence_ids") or []:
            if evidence_id not in evidence_used:
                evidence_used.append(evidence_id)

    gaps: list[str] = []
    if query:
        gaps = [
            *coverage_gaps(query, tool_log, zh=zh, names=names),
            *industry_gaps(query, tool_log, zh=zh),
            *non_stock_fundamental_gaps(query, tool_log, zh=zh, names=names, types=types),
            *macro_gaps(query, tool_log, zh=zh),
            # a failed indicator tool is already named by failed_target_statements when nothing else was found
            *(indicator_gaps(query, tool_log, zh=zh, names=names) if facts else []),
        ]
        gaps = list(dict.fromkeys(gaps))
    separator = "" if zh else " "
    if facts:
        lead = "根据本次检索到的证据：" if zh else "Based on the evidence retrieved for this question: "
        answer = lead + separator.join(facts)
        if gaps:
            answer = separator.join(gaps) + separator + answer
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
        missing = failed_target_statements(query, tool_log, zh=zh, names=names) if query else []
        if missing or gaps:
            answer += separator + separator.join([*gaps, *missing])
    return {
        "answer": answer,
        "key_points": facts[:8],
        "evidence_used": evidence_used,
        "limitations": list(dict.fromkeys([*gaps, *limitations])),
    }


def _extra_metrics(data: dict[str, Any], keys: set[str], zh: bool) -> list[str]:
    """Requested metrics beyond the standard snapshot (e.g. a live source's dividend yield), when present."""
    metrics = data.get("metrics") or {}
    eid, name, period = data.get("evidence_id"), data.get("name"), data.get("report_date")
    parts = []
    for key in sorted(keys):
        value = metrics.get(key)
        if value is None or not eid:
            continue
        metric = EXTRA_METRIC_FIELDS[key]
        parts.append(f"{metric.zh if zh else metric.en} {_num(value)}")
    if not parts:
        return []
    if zh:
        return [f"{name}（报告期 {period}）：{'，'.join(parts)} [{eid}]。"]
    return [f"{name} (period {period}): {', '.join(parts)} [{eid}]."]


def _price(data: dict[str, Any], zh: bool, request: PriceRequest | None = None) -> list[str]:
    name, symbol, eid = data.get("name"), data.get("symbol"), data.get("evidence_id")
    close, pct, as_of = data.get("close"), data.get("pct_change_1d"), data.get("as_of")
    if close is None:
        return []
    if zh:
        change = f"，当日涨跌幅 {_num(pct)}%" if pct is not None else ""
        sentences = [f"{name}（{symbol}）最新可用收盘价为 {_num(close)}（{as_of}）{change} [{eid}]。"]
    else:
        change = f", daily change {_num(pct)}%" if pct is not None else ""
        sentences = [f"{name} ({symbol}) last available close was {_num(close)} on {as_of}{change} [{eid}]."]
    if request is not None and request.needs_quote:
        sentences.extend(_price_details(data, zh, request))
    return sentences


_QUOTE_FIELDS = (
    ("open", "开盘价", "open"),
    ("high", "最高价", "high"),
    ("low", "最低价", "low"),
    ("volume", "成交量", "volume"),
    ("amount", "成交额", "turnover"),
)


def _price_details(data: dict[str, Any], zh: bool, request: PriceRequest) -> list[str]:
    """Recent closes, previous close, open/high/low, volume and turnover when asked for; missing ones are named."""
    name, eid, as_of = data.get("name"), data.get("evidence_id"), data.get("as_of")
    closes = [row for row in data.get("recent_closes") or [] if row.get("close") is not None]
    stated: list[str] = []
    missing: list[str] = []
    sentences: list[str] = []
    if request.closes:
        shown = closes[-request.closes :]
        listing = ("、" if zh else ", ").join(
            f"{row.get('date')} {_num(row['close'])}" if zh else f"{row.get('date')}: {_num(row['close'])}"
            for row in shown
        )
        if zh:
            head = (
                f"最近{len(shown)}个交易日收盘价"
                if len(shown) >= request.closes
                else f"数据中只有最近{len(shown)}个交易日的收盘价（所问为{request.closes}个）"
            )
            sentences.append(f"{name}{head}：{listing} [{eid}]。")
        else:
            head = "most recent closes" if len(shown) >= request.closes else "only available recent closes"
            sentences.append(f"{name}'s {head}: {listing} [{eid}].")
    if request.previous_close:
        if len(closes) >= 2:
            previous = closes[-2]
            stated.append(
                f"前一交易日（{previous.get('date')}）收盘价 {_num(previous['close'])}"
                if zh
                else f"previous close {_num(previous['close'])} on {previous.get('date')}"
            )
        else:
            missing.append("前一交易日收盘价" if zh else "the previous close")
    for key, label_zh, label_en in _QUOTE_FIELDS:
        if not getattr(request, key):
            continue
        value = data.get(key)
        if value is None or (key in {"volume", "amount"} and not float(value)):
            missing.append(label_zh if zh else f"the {label_en}")
            continue
        unit = ""
        if key == "volume":
            unit_name = data.get("volume_unit")
            unit = (
                {"lot": "手", "share": "股"}.get(str(unit_name), "（单位以数据源为准）")
                if zh
                else {"lot": " lots", "share": " shares"}.get(str(unit_name), " (units as reported by the source)")
            )
        stated.append(f"{label_zh} {_num(value)}{unit}" if zh else f"{label_en} {_num(value)}{unit}")
    if stated:
        sentences.append(
            f"{name}（{as_of}）：{'，'.join(stated)} [{eid}]。"
            if zh
            else f"{name} ({as_of}): {', '.join(stated)} [{eid}]."
        )
    if missing:
        sentences.append(
            f"当前数据中没有{name}的{'、'.join(missing)}。"
            if zh
            else f"The current data does not include {', '.join(missing)} for {name}."
        )
    return sentences


def _indicators(data: dict[str, Any], zh: bool, request: PriceRequest | None = None) -> list[str]:
    name, eid = data.get("name"), data.get("evidence_id")
    parts = []
    for key, label in (("ma5", "MA5"), ("ma20", "MA20"), ("rsi_14", "RSI(14)"), ("volatility_20d", "20D vol")):
        if data.get(key) is not None:
            parts.append(f"{label} {_num(data[key])}")
    returns = data.get("pct_change_nd") or {}
    for days in request.return_days if request else ():
        value = returns.get(f"pct_{days}d")
        if value is not None:
            parts.append(f"近{days}日涨跌幅 {_num(value)}%" if zh else f"{days}-day return {_num(value)}%")
    if not parts:
        return []
    missing = [name.upper().replace("_14", "(14)") for name in data.get("unavailable") or []]
    trend = data.get("trend_signal")
    sentences = []
    if zh:
        trend_text = f"，趋势信号为 {trend}" if trend else ""
        gap = f"（历史数据不足，无法计算 {'、'.join(missing)}）" if missing else ""
        sentences.append(f"{name} 技术指标：{'，'.join(parts)}{trend_text}{gap} [{eid}]。")
    else:
        trend_text = f", trend signal {trend}" if trend else ""
        gap = f" (not enough history to compute {', '.join(missing)})" if missing else ""
        sentences.append(f"{name} technical indicators: {', '.join(parts)}{trend_text}{gap} [{eid}].")
    if request is not None and request.above_ma:
        sentences.extend(_price_vs_ma(data, zh, request))
    return sentences


def _price_vs_ma(data: dict[str, Any], zh: bool, request: PriceRequest) -> list[str]:
    """ "站上MA5了吗": compare the latest close with each requested moving average that could be computed."""
    name, eid, close = data.get("name"), data.get("evidence_id"), data.get("latest_close")
    sentences = []
    for days in request.moving_averages or (5,):
        average = data.get(f"ma{days}")
        if close is None or average is None:
            continue
        above = float(close) >= float(average)
        if zh:
            relation = "高于" if above else "低于"
            state = "站上" if above else "位于其下方"
            sentences.append(
                f"{name} 最新收盘价 {_num(close)} {relation} MA{days} {_num(average)}，即{state} MA{days} [{eid}]。"
            )
        else:
            relation = "above" if above else "below"
            sentences.append(
                f"{name}'s latest close {_num(close)} is {relation} its MA{days} of {_num(average)} [{eid}]."
            )
    return sentences


def _fundamentals(data: dict[str, Any], zh: bool, *, industry_first: bool = False) -> list[str]:
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
        # normalised payloads state ROE in percent (metric_units); older payloads may hold a fraction
        in_percent = (data.get("metric_units") or {}).get("roe") == "%"
        roe_pct = roe if in_percent or abs(float(roe)) > 1 else roe * 100
        parts.append(f"ROE {_num(roe_pct)}%")
    for key, label_zh, label_en in (("revenue", "营业收入", "revenue"), ("net_profit", "净利润", "net profit")):
        value = metrics.get(key)
        if value is not None and abs(float(value)) >= 1e6:
            parts.append(
                f"{label_zh} {_num(value / 1e8)} 亿元" if zh else f"{label_en} {_num(value / 1e8)} hundred million CNY"
            )
    if parts and eid:
        label = data.get("period")
        if zh:
            report = f"{label[2:]}年年报" if label and label.startswith("FY") else label
            suffix = f"，{report}" if report else ""
            sentences.append(f"{name} 基本面（报告期 {period}{suffix}）：{'，'.join(parts)} [{eid}]。")
        else:
            suffix = f", {label}" if label else ""
            sentences.append(f"{name} fundamentals (period {period}{suffix}): {', '.join(parts)} [{eid}].")
    industry = data.get("industry") or {}
    industry_metrics = industry.get("metrics") or {}
    industry_parts = [
        f"{label} {_num(industry_metrics[key])}"
        for key, label in (("pe", "PE"), ("pb", "PB"), ("pct_change", "涨跌幅%" if zh else "change %"))
        if industry_metrics.get(key) is not None
    ]
    if industry_parts and industry.get("evidence_id"):
        if zh:
            industry_sentence = (
                f"所属行业 {industry.get('industry_name')}：{'，'.join(industry_parts)} [{industry['evidence_id']}]。"
                if eid
                else f"{industry.get('industry_name')}行业：{'，'.join(industry_parts)} [{industry['evidence_id']}]。"
            )
        else:
            industry_sentence = (
                f"Industry {industry.get('industry_name')}: {', '.join(industry_parts)} [{industry['evidence_id']}]."
            )
        sentences = [industry_sentence, *sentences] if industry_first else [*sentences, industry_sentence]
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


# Neutral document categories by source type: the template names what kind of document it cites, never its title.
_DOCUMENT_CATEGORY = {
    "news": ("新闻", "news article"),
    "announcement": ("公告", "company announcement"),
    "research_note": ("研究报告", "research note"),
    "product_doc": ("产品资料", "product document"),
    "faq": ("常见问题解答", "FAQ entry"),
}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DOCUMENT_TEXT_LIMITATION_ZH = "资料标题和原文属于第三方内容，未经核实，回答中不引用；可在证据列表中查看。"
DOCUMENT_TEXT_LIMITATION_EN = (
    "Document titles and text are unverified third-party content and are not quoted in the answer; "
    "see the evidence list."
)


def _documents(data: dict[str, Any], zh: bool) -> list[str]:
    """Cite retrieved documents by category, source and date; never by title.

    Titles and excerpts are third-party text. A blocklist over titles (links, directives, advice words) was
    bypassed by homoglyphs, slang and plausible fake headlines ("证监会：…立案调查", "每10股派现1000元"), which no
    shape check can tell from real ones. So the deterministic answer states only facts FinSight controls: the
    document category (from the source type), the publisher when it is a plain name, and the date. The title
    stays visible in the evidence list, labelled as a source.
    """
    from ..text_safety import safe_headline

    sentences = []
    for document in (data.get("documents") or [])[:_MAX_DOCS_PER_TOOL]:
        eid = document.get("evidence_id")
        if not eid:
            continue
        category_zh, category_en = _DOCUMENT_CATEGORY.get(str(document.get("source_type") or ""), ("资料", "document"))
        # the publisher name comes from the data provider, but is still shown only when it is inert text
        source = safe_headline(str(document.get("source_name") or "")[:40]) if document.get("source_name") else None
        when = str(document.get("publish_time") or "")[:10]
        when = when if _DATE.match(when) else ""
        if zh:
            origin = f"{source}{'于' + when if when else ''}发布的" if source else (f"{when}的" if when else "")
            sentences.append(f"相关资料：{origin}一篇{category_zh} [{eid}]。")
        else:
            origin = f" from {source}" if source else ""
            dated = f" ({when})" if when else ""
            sentences.append(f"Related document: a {category_en}{origin}{dated} [{eid}].")
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
