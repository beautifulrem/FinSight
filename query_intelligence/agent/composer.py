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
    PROFIT_GROWTH_FIELDS,
    PriceRequest,
    asks_about_industry,
    coverage_gaps,
    drawdown_gaps,
    failed_target_statements,
    flow_gaps,
    foreign_macro_gaps,
    indicator_gaps,
    industry_gaps,
    macro_gaps,
    non_stock_fundamental_gaps,
    requested_metrics,
    requested_price_fields,
    year_to_date_gaps,
)
from .names import english_display

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
    wanted = requested_metrics(query)
    extra_keys = {field for metric in wanted for field in metric.fields}
    derive = {metric.key for metric in wanted if metric.derivable_from}
    margins: list[tuple[str, float]] = []
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
        if not data.get("product_type") and (types or {}).get(str(data.get("symbol"))):
            # indicator outputs carry no product type: an index level is in points, not CNY
            data = {**data, "product_type": (types or {})[str(data.get("symbol"))]}
        if tool == "get_price_history":
            sentences = _price(data, zh, request)
        elif tool == "compute_indicators":
            sentences = _indicators(data, zh, request)
        elif tool == "get_fundamentals":
            sentences = _fundamentals(data, zh, industry_first=industry_first)
            if extra_keys:
                sentences = [*sentences, *_extra_metrics(data, extra_keys, zh)]
            if derive:
                derived, margin = _derived_metrics(data, derive, zh)
                sentences = [*sentences, *derived]
                if margin is not None:
                    margins.append((str(data.get("name") or data.get("symbol") or ""), margin))
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

    if len(margins) >= 2:
        facts.append(_margin_ranking(margins, zh))
    gaps: list[str] = []
    if query:
        gaps = [
            *coverage_gaps(query, tool_log, zh=zh, names=names),
            *industry_gaps(query, tool_log, zh=zh),
            *non_stock_fundamental_gaps(query, tool_log, zh=zh, names=names, types=types),
            *macro_gaps(query, tool_log, zh=zh),
            *foreign_macro_gaps(query, zh=zh),
            *flow_gaps(query, tool_log, zh=zh),
            *year_to_date_gaps(query, tool_log, zh=zh),
            *drawdown_gaps(query, tool_log, zh=zh),
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
    limitations = list(dict.fromkeys([*gaps, *limitations]))
    key_points = facts[:8]
    if not zh:
        # English answers name targets and industries in English ("Kweichow Moutai", "baijiu"), not 贵州茅台/白酒.
        answer = english_display(answer)
        key_points = [english_display(point) for point in key_points]
        limitations = [english_display(item) for item in limitations]
    return {
        "answer": answer,
        "key_points": key_points,
        "evidence_used": evidence_used,
        "limitations": limitations,
    }


def _derived_metrics(data: dict[str, Any], keys: set[str], zh: bool) -> tuple[list[str], float | None]:
    """Net margin (net profit / revenue) and PEG (P/E / profit growth) from the cited fundamentals, with the operands
    in the sentence so the verifier and the reader can check the arithmetic. Returns ``(sentences, net margin)``."""
    metrics = data.get("metrics") or {}
    name, eid, period = data.get("name"), data.get("evidence_id"), data.get("report_date")
    sentences: list[str] = []
    margin = None
    if not eid:
        return sentences, margin
    revenue, profit = metrics.get("revenue"), metrics.get("net_profit")
    reported_margin = any(metrics.get(field) is not None for field in ("net_margin", "netprofit_margin"))
    if "net_margin" in keys and not reported_margin and revenue and profit is not None and float(revenue) > 0:
        margin = round(float(profit) / float(revenue) * 100, 2)
        sentences.append(
            f"{name} 净利率（净利润 ÷ 营业收入，报告期 {period}）：{_money(profit, zh)} ÷ {_money(revenue, zh)} ≈ "
            f"{_num(margin)}% [{eid}]。"
            if zh
            else f"{name} net margin (net profit / revenue, period {period}): {_money(profit, zh)} / "
            f"{_money(revenue, zh)} ≈ {_num(margin)}% [{eid}]."
        )
    pe = metrics.get("pe_ttm") if metrics.get("pe_ttm") is not None else metrics.get("pe")
    growth = next((metrics[field] for field in PROFIT_GROWTH_FIELDS if metrics.get(field) is not None), None)
    if "peg" in keys and metrics.get("peg") is None and pe is not None and growth is not None:
        if float(growth) > 0:
            peg = round(float(pe) / float(growth), 2)
            sentences.append(
                f"{name} PEG（市盈率 ÷ 净利润增速）：{_num(pe)} ÷ {_num(growth)} ≈ {_num(peg)} [{eid}]。"
                if zh
                else f"{name} PEG (P/E / net profit growth): {_num(pe)} / {_num(growth)} ≈ {_num(peg)} [{eid}]."
            )
        else:
            sentences.append(
                f"{name}的净利润增速不为正，PEG 不适用 [{eid}]。"
                if zh
                else f"{name}'s net profit growth is not positive, so PEG does not apply [{eid}]."
            )
    return sentences, margin


def _margin_ranking(margins: list[tuple[str, float]], zh: bool) -> str:
    """Which target has the higher net margin, in words (the figures are in the sentences above)."""
    ordered = sorted(margins, key=lambda item: item[1], reverse=True)
    names = [name for name, _margin in ordered]
    if ordered[0][1] == ordered[-1][1]:
        return "按上述口径，各标的净利率相同。" if zh else "On this basis the net margins are equal."
    if zh:
        return f"按上述口径，净利率由高到低为：{'、'.join(names)}。"
    return f"On this basis, net margin from highest to lowest: {', '.join(names)}."


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
        parts.append(f"{metric.zh if zh else metric.en} {_metric_value(key, value, data, zh)}")
    if not parts:
        return []
    if zh:
        return [f"{name}（报告期 {period}）：{'，'.join(parts)} [{eid}]。"]
    return [f"{name} (period {period}): {', '.join(parts)} [{eid}]."]


def _price(data: dict[str, Any], zh: bool, request: PriceRequest | None = None) -> list[str]:
    name, symbol, eid = data.get("name"), data.get("symbol"), data.get("evidence_id")
    close, pct, as_of = data.get("close"), data.get("pct_change_1d"), data.get("as_of")
    quote = data.get("intraday") if data.get("price_basis") == "intraday" else None
    if quote and quote.get("price") is not None:
        return _intraday_price(name, symbol, eid, quote, zh)
    if close is None:
        return []
    if zh:
        change = f"，当日涨跌幅 {_num(pct)}%" if pct is not None else ""
        sentences = [f"{name}（{symbol}）最新可用收盘价为 {_px(close, data, zh)}（{as_of}）{change} [{eid}]。"]
    else:
        change = f", daily change {_num(pct)}%" if pct is not None else ""
        sentences = [f"{name} ({symbol}) last available close was {_px(close, data, zh)} on {as_of}{change} [{eid}]."]
    if request is not None and request.needs_quote:
        sentences.extend(_price_details(data, zh, request))
    if request is not None and request.year_to_date:
        sentences.extend(_year_to_date(data, zh))
    return sentences


def _year_to_date(data: dict[str, Any], zh: bool) -> list[str]:
    """The change from the first close of the year to the latest close, with both closes in the sentence (the
    verifier checks the percent change against them); without ``year_start`` the gap is stated by coverage."""
    start, close = data.get("year_start") or {}, data.get("close")
    if not start or start.get("close") in (None, 0) or close is None:
        return []
    change = (float(close) - float(start["close"])) / abs(float(start["close"])) * 100
    name, eid, as_of = data.get("name"), data.get("evidence_id"), data.get("as_of")
    if zh:
        return [
            f"{name}今年以来：今年首个交易日（{start.get('date')}）收盘 {_px(start['close'], data, zh)}，"
            f"最新（{as_of}）收盘 {_px(close, data, zh)}，涨跌幅 {_num(round(change, 2))}% [{eid}]。"
        ]
    return [
        f"{name} year to date: first close of the year ({start.get('date')}) {_px(start['close'], data, zh)}, latest "
        f"close ({as_of}) {_px(close, data, zh)}, a change of {_num(round(change, 2))}% [{eid}]."
    ]


def _intraday_price(name: Any, symbol: Any, eid: Any, quote: dict[str, Any], zh: bool) -> list[str]:
    """A 今天 question during trading hours: the real-time quote, labelled as intraday, not a close.

    Date and time are written together so the verifier reads them as a timestamp, not as numbers.
    """
    stamp = str(quote.get("quote_time") or "")[:19].replace("T", " ")
    price, previous, change = quote.get("price"), quote.get("prev_close"), quote.get("pct_change")
    has_change = previous is not None and change is not None
    if zh:
        versus = f"，较前收 {_num(previous)} 涨跌 {_num(change)}%" if has_change else ""
        label = f"{stamp} 北京时间，盘中价格，非收盘价"
        return [f"{name}（{symbol}）盘中实时价为 {_num(price)}（{label}）{versus} [{eid}]。"]
    versus = f", {_num(change)}% against the previous close of {_num(previous)}" if has_change else ""
    return [f"{name} ({symbol}) intraday price is {_num(price)} ({stamp} Beijing time; not a close){versus} [{eid}]."]


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
            f"{row.get('date')} {_px(row['close'], data, zh)}"
            if zh
            else f"{row.get('date')}: {_px(row['close'], data, zh)}"
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
                f"前一交易日（{previous.get('date')}）收盘价 {_px(previous['close'], data, zh)}"
                if zh
                else f"previous close {_px(previous['close'], data, zh)} on {previous.get('date')}"
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
        if key == "amount":
            stated.append(f"{label_zh} {_money(value, zh)}" if zh else f"{label_en} {_money(value, zh)}")
            continue
        if key in {"open", "high", "low"}:
            stated.append(f"{label_zh} {_px(value, data, zh)}" if zh else f"{label_en} {_px(value, data, zh)}")
            continue
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
            value = _px(data[key], data, zh) if key.startswith("ma") else _num(data[key])
            parts.append(f"{label} {value}")
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
                f"{name} 最新收盘价 {_px(close, data, zh)} {relation} MA{days} {_px(average, data, zh)}，"
                f"即{state} MA{days} [{eid}]。"
            )
        else:
            relation = "above" if above else "below"
            sentences.append(
                f"{name}'s latest close {_px(close, data, zh)} is {relation} its MA{days} of "
                f"{_px(average, data, zh)} [{eid}]."
            )
    return sentences


def _fundamentals(data: dict[str, Any], zh: bool, *, industry_first: bool = False) -> list[str]:
    sentences = []
    metrics = data.get("metrics") or {}
    name, eid, period = data.get("name"), data.get("evidence_id"), data.get("report_date")
    parts = []
    if metrics.get("pe_ttm") is not None:
        parts.append(f"PE(TTM) {_times(metrics['pe_ttm'], zh)}")
    if metrics.get("pb") is not None:
        parts.append(f"PB {_times(metrics['pb'], zh)}")
    roe = metrics.get("roe")
    if roe is not None:
        # normalised payloads state ROE in percent (metric_units); older payloads may hold a fraction
        in_percent = (data.get("metric_units") or {}).get("roe") == "%"
        roe_pct = roe if in_percent or abs(float(roe)) > 1 else roe * 100
        parts.append(f"ROE {_num(roe_pct)}%")
    for key, label_zh, label_en in (("revenue", "营业收入", "revenue"), ("net_profit", "净利润", "net profit")):
        value = metrics.get(key)
        if value is not None and abs(float(value)) >= 1e6:
            parts.append(f"{label_zh if zh else label_en} {_money(value, zh)}")
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
        f"{label} {_times(industry_metrics[key], zh) if key in {'pe', 'pb'} else _num(industry_metrics[key]) + '%'}"
        for key, label in (("pe", "PE"), ("pb", "PB"), ("pct_change", "涨跌幅" if zh else "change"))
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
        f"Tone of {total} documents published recently about {targets}: {positive} positive, {neutral} neutral, "
        f"{negative} negative; mean model score {_num(data.get('mean_score'))} (0.5 is neutral) [{eid}]."
    ]


def _entities(data: dict[str, Any], zh: bool) -> list[str]:
    return []


def _concept(data: dict[str, Any], zh: bool) -> list[str]:
    """State a glossary definition, and say that FinSight tracks no data series for the concept."""
    term, eid = data.get("term"), data.get("evidence_id")
    text = str((data.get("definition_zh") if zh else data.get("definition_en")) or "")
    if not term or not text or not eid:
        return []
    sentences = [f"{term}：{text} [{eid}]。" if zh else f"{term}: {text} [{eid}]."]
    if not data.get("has_data_series"):
        sentences.append(
            f"当前数据源不包含{term}的数据序列，因此只能给出概念说明，无法给出具体数值。"
            if zh
            else f"The configured sources carry no data series for {term}, so this is a definition only, with no "
            "figures."
        )
    return sentences


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
    "explain_concept": _concept,
}


def _failure_text(tool: str, error: dict[str, Any], *, zh: bool) -> str:
    code = error.get("code") or "error"
    message = str(error.get("message") or "")[:160]
    return (
        f"{tool} 未返回可用数据（{code}：{message}）" if zh else f"{tool} returned no usable data ({code}: {message})"
    )


# Extra metrics that are ratios (stated in percent, like ROE) and amounts (stated in CNY).
_PERCENT_FIELDS = {
    "dividend_yield",
    "dv_ratio",
    "dv_ttm",
    "debt_to_assets",
    "debt_ratio",
    "liability_ratio",
    "gross_margin",
    "grossprofit_margin",
    "net_margin",
    "netprofit_margin",
}
_MONEY_FIELDS = {"operating_cash_flow", "n_cashflow_act", "free_cash_flow"}


def _metric_value(key: str, value: Any, data: dict[str, Any], zh: bool) -> str:
    unit = (data.get("metric_units") or {}).get(key)
    if key in _PERCENT_FIELDS or unit == "%":
        number = float(value)
        # normalised payloads state ratios in percent; older payloads may hold a fraction
        return f"{_num(number if unit == '%' or abs(number) > 1 else number * 100)}%"
    if key in _MONEY_FIELDS:
        return _money(value, zh)
    if key == "debt_to_equity":
        return _times(value, zh)
    return _num(value)


def _money(value: Any, zh: bool) -> str:
    """An amount in CNY with its unit: "37.94 亿元" / "CNY 3.79 bn" (two decimals at the chosen scale)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    size = abs(number)
    if zh:
        if size >= 1e8:
            return f"{_num(round(number / 1e8, 2))} 亿元"
        if size >= 1e4:
            return f"{_num(round(number / 1e4, 2))} 万元"
        return f"{_num(number)} 元"
    if size >= 1e9:
        return f"CNY {_num(round(number / 1e9, 2))} bn"
    if size >= 1e6:
        return f"CNY {_num(round(number / 1e6, 2))} mn"
    return f"CNY {_num(number)}"


def _px(value: Any, data: dict[str, Any], zh: bool) -> str:
    """A price with its unit: an index level in points, anything else in CNY ("1409.5 元" / "CNY 1409.5")."""
    kind = str(data.get("product_type") or "")
    if kind == "index":
        return f"{_num(value)} 点" if zh else f"{_num(value)} points"
    return f"{_num(value)} 元" if zh else f"CNY {_num(value)}"


def _times(value: Any, zh: bool) -> str:
    """A valuation multiple: "24.6 倍" / "24.6x"."""
    return f"{_num(value)} 倍" if zh else f"{_num(value)}x"


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
