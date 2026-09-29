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
from .verifier import _MARKET_METRIC

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
        else:
            sentences = renderer(data, zh)
        for sentence in sentences:
            if sentence and sentence not in facts:
                facts.append(sentence)
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
        parts.append(f"{metric.zh if zh else metric.en} {_metric_value(key, value, data, zh)}")
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
        sentences = [f"{name}（{symbol}）最新可用收盘价为 {_px(close, data, zh)}（{as_of}）{change} [{eid}]。"]
    else:
        change = f", daily change {_num(pct)}%" if pct is not None else ""
        sentences = [f"{name} ({symbol}) last available close was {_px(close, data, zh)} on {as_of}{change} [{eid}]."]
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


_MAX_TITLE_CHARS = 60
# A quoted headline must be inert text: no links, markup or code, and nothing addressed to the reader or to
# an AI ("请…", "AI 助手…", "readers should…"), and no advice or ratings (the compliance guard would strip
# them from the model's own words, so a quoted title must not smuggle them in either).
_ACTIVE_TITLE = re.compile(
    r"[a-z][a-z0-9+.-]*:(?://|[^\s]*\()|javascript:|data:|www\.|\]\(|[<>{}`|]|&#|\\u[0-9a-f]{4}",
    re.IGNORECASE,
)
_DIRECTIVE_TITLE = re.compile(
    r"请|务必|必须|不得|助手|机器人|读者|建议|推荐|评级|建仓|加仓|减仓|买入|卖出|"
    r"\b(?:assistant|chatbot|readers?|you|your|must|should|recommend\w*|rated|must-buy|"
    r"(?:include|mention|report|summari[sz]e|repeat|output) (?:this|that|the following)|ignore)\b",
    re.IGNORECASE,
)


def quotable_title(title: str) -> str | None:
    """The title as inert text for the template answer, or ``None`` when it must not be echoed.

    Document titles are third-party text, like excerpts: the untrusted-data envelope protects the model, but
    the template path copies titles into the answer verbatim, so a title is only quoted when it is plainly a
    headline. Invisible characters are removed and matching runs on the NFKC form (so full-width and
    zero-width obfuscation does not hide a payload); the title is withheld when the injection filter flags
    it or it carries links/markup, directives, advice or ratings, and it is cut to ``_MAX_TITLE_CHARS``.
    """
    from .compliance import contains_trading_instruction
    from .injection import _INVISIBLE, _normalise, sanitize_untrusted_text

    text = re.sub(r"\s+", " ", _INVISIBLE.sub("", title)).strip()
    plain = _normalise(text)  # full-width letters folded for matching only; the quote keeps CJK punctuation
    if not text:
        return None
    _cleaned, flagged = sanitize_untrusted_text(plain)
    if flagged or _ACTIVE_TITLE.search(plain) or _DIRECTIVE_TITLE.search(plain) or contains_trading_instruction(plain):
        return None
    if len(text) > _MAX_TITLE_CHARS:
        text = text[: _MAX_TITLE_CHARS - 1].rstrip("，,、；;：: ") + "…"
    return text.replace("《", "〈").replace("》", "〉").replace('"', "'")


def _documents(data: dict[str, Any], zh: bool) -> list[str]:
    """Cite retrieved documents by source and date, quoting the title only when it is inert text.

    Titles that state market metrics with numbers ("shares closed up 12.34%", "市盈率55倍") are not
    repeated: prices, multiples and daily moves come only from market data (the same source-precedence
    rule the verifier applies to LLM drafts). Titles that fail ``quotable_title`` are withheld and only the
    source and date are given. The documents stay in the evidence list either way.
    """
    sentences = []
    for document in (data.get("documents") or [])[:_MAX_DOCS_PER_TOOL]:
        raw_title = str(document.get("title") or "").strip()
        if not raw_title or (_MARKET_METRIC.search(raw_title) and re.search(r"\d", raw_title)):
            continue
        title = quotable_title(raw_title)
        source = document.get("source_name") or document.get("source_type")
        when = str(document.get("publish_time") or "")[:10]
        eid = document.get("evidence_id")
        if zh:
            quoted = f"《{title}》" if title else "一篇资料（标题未引用）"
            sentences.append(f"相关资料：{quoted}（{source}，{when}） [{eid}]。")
        else:
            quoted = f'"{title}"' if title else "a document (title not quoted)"
            sentences.append(f"Related document: {quoted} ({source}, {when}) [{eid}].")
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
