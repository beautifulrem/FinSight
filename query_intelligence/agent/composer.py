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
    holding_value_request,
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
    frame_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Restate tool results with citations.

    With ``query``, metrics and periods the question asks for but the evidence lacks are stated first
    (``coverage_gaps``), and targets whose data could not be retrieved are named, so a question about
    "2019年营收" or a dividend yield is never answered silently with other numbers. Requested price details
    (recent closes, high/low, volume, N-day return, price vs. MA) are stated when the evidence has them and
    named as unavailable when it does not; a sector question leads with the industry snapshot.

    (round 11) With a ``frame_request`` (a gap, ratio or which-is-higher question read against the session's
    comparison frame, ``frame.py``) the computed comparison leads the answer, and an operand without data is named
    instead of computing anything from other figures.
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
        elif renderer is _documents:
            # the question's targets decide which knowledge documents are about it (round 10, F11)
            targets = [*(data.get("targets") or []), *(names or {}).values(), *(names or {}).keys()]
            sentences = _documents({**data, "targets": targets}, zh)
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
    if "eps" in {metric.key for metric in wanted}:
        facts.extend(fact for fact in _implied_eps(tool_log, zh) if fact not in facts)
    holding = holding_value_request(query) if query else None
    if holding is not None:
        facts = [*_holding_value(holding[0], tool_log, zh), *facts]
    frame_gaps: list[str] = []
    if frame_request:
        framed, frame_gaps = frame_sentences(frame_request, tool_log, zh)
        facts = [*framed, *(fact for fact in facts if fact not in framed)]
    else:
        for sentence in _arithmetic(query, tool_log, zh):
            if sentence not in facts:
                facts.append(sentence)
    gaps: list[str] = list(frame_gaps)
    if query:
        gaps += [
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


def _holding_value(shares: int, tool_log: list[dict[str, Any]], zh: bool) -> list[str]:
    """(round 11, G5) "我有1000股五粮液，值多少钱": shares × the latest close, with the date, both operands in the
    sentence, and a note that this is a market value at the close, not a tradable price, a valuation or advice."""
    for entry in tool_log:
        data = entry.get("data") or {}
        close, eid = data.get("close"), data.get("evidence_id")
        if entry.get("tool") != "get_price_history" or not entry.get("ok") or close is None or not eid:
            continue
        value = round(shares * float(close), 2)
        name, as_of = data.get("name") or data.get("symbol"), data.get("as_of")
        if zh:
            return [
                f"按 {as_of} 的收盘价 {_px(close, data, zh)} 计算，{shares} 股{name}的市值约为 {shares} × "
                f"{_num(close)} = {_num(value)} 元 [{eid}]。",
                "这是按最近收盘价计算的持仓市值，不是可成交价格，也不是估值判断或投资建议。",
            ]
        return [
            f"At the {as_of} close of {_px(close, data, zh)}, {shares} shares of {name} are worth {shares} × "
            f"{_num(close)} = CNY {_num(value)} [{eid}].",
            "This is the market value at the last close, not a tradable price, a valuation or investment advice.",
        ]
    return []


def _implied_eps(tool_log: list[dict[str, Any]], zh: bool) -> list[str]:
    """(round 11, G5) EPS when no source reports it: the value implied by the latest close and the P/E (TTM), with
    both operands and a label saying it is derived, not the company's reported figure."""
    closes = {
        str((entry.get("data") or {}).get("symbol")): entry.get("data") or {}
        for entry in tool_log
        if entry.get("tool") == "get_price_history" and entry.get("ok")
    }
    sentences = []
    for entry in tool_log:
        data = entry.get("data") or {}
        metrics = data.get("metrics") or {}
        price = closes.get(str(data.get("symbol")))
        if entry.get("tool") != "get_fundamentals" or not entry.get("ok") or not price or not data.get("evidence_id"):
            continue
        if any(metrics.get(field) is not None for field in ("eps", "basic_eps", "diluted_eps", "eps_ttm")):
            continue
        pe, close = metrics.get("pe_ttm"), price.get("close")
        if not pe or float(pe) <= 0 or close is None or not price.get("evidence_id"):
            continue
        eps = _num(round(float(close) / float(pe), 2))
        cites = f"[{price['evidence_id']}][{data['evidence_id']}]"
        if zh:
            sentences.append(
                f"{data.get('name')}按最新收盘价与市盈率(TTM)反推的隐含每股收益约为 {_num(close)} 元 ÷ "
                f"{_num(pe)} ≈ {eps} 元（推算值，不是公司披露的每股收益；收盘价日期 {price.get('as_of')}） {cites}。"
            )
        else:
            sentences.append(
                f"{data.get('name')}'s EPS implied by the latest close and the P/E (TTM) is about CNY {_num(close)} / "
                f"{_num(pe)} ≈ CNY {eps} (a derived figure, not the company's reported EPS; close of "
                f"{price.get('as_of')}) {cites}."
            )
    return sentences


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


# (round 9, E5/E8) A question that asks for the difference ("差了多少个百分点", "营收差多少亿", "How big is the gap?")
# or the ratio ("PE是行业的几倍") of one metric for two targets, or for a target and its industry: the template derives
# it from the two cited values and writes both operands next to the result, so the verifier (``allow_derived``) and the
# reader can check the arithmetic.
_ASKS_DIFFERENCE = re.compile(
    r"差了?(?:有|是|大概)?(?:多少|几|多大)|相差|差距|差额|差值|(?:高|低|多|少)(?:了|出)?(?:多少|几)|"
    r"\bdifference\b|\bgap\b|\bspread\b|\bhow much (?:higher|lower|more|less|bigger|smaller)\b",
    re.IGNORECASE,
)
_ASKS_RATIO = re.compile(r"(?:是|为|相当于)[^，。？?,]{0,12}?的?(?:几|多少)倍|几倍于|\bhow many times\b", re.IGNORECASE)
_ASKS_INDUSTRY = re.compile(r"行业|板块|\bsector\b|\bindustry\b", re.IGNORECASE)
# (key, zh label, en label, pattern): the metric the question names first is the one compared
_ARITHMETIC_METRICS: tuple[tuple[str, str, str, re.Pattern[str]], ...] = (
    # (round 10, F8) net margin, derived from the cited revenue and net profit: named before "百分点" so a margin gap
    # "差几个百分点" is not read as a gap in daily change
    (
        "net_margin",
        "净利率",
        "net margin",
        re.compile(r"净利率|净利润率|销售净利率|\bnet (?:profit )?margins?\b", re.IGNORECASE),
    ),
    (
        "pct_change",
        "当日涨跌幅",
        "daily change",
        re.compile(
            r"涨跌幅|涨幅|跌幅|百分点|涨|跌|\bdaily change\b|\bpercent(?:age)? change\b|"
            r"\b(?:rose|fell|gained|dropped|moved)\b",
            re.IGNORECASE,
        ),
    ),
    ("close", "收盘价", "close", re.compile(r"收盘价?|股价|\bclos(?:e|ing price)\b|\bshare price\b", re.IGNORECASE)),
    # (round 10, F11) turnover: "哪个成交更活跃", "成交额谁大", "which traded more"
    (
        "amount",
        "成交额",
        "turnover",
        re.compile(
            r"成交额|成交金额|成交(?:更|最|比较)?(?:活跃|大|多|少|旺)|\bturnover\b|\btrading value\b|\btraded more\b",
            re.IGNORECASE,
        ),
    ),
    ("pe", "市盈率", "P/E", re.compile(r"市盈率|(?<![A-Za-z])P/?E(?![A-Za-z])|price[- ]to[- ]earnings", re.IGNORECASE)),
    ("pb", "市净率", "P/B", re.compile(r"市净率|(?<![A-Za-z])P/?B(?![A-Za-z])|price[- ]to[- ]book", re.IGNORECASE)),
    ("roe", "ROE", "ROE", re.compile(r"净资产收益率|(?<![A-Za-z])ROE(?![A-Za-z])|return on equity", re.IGNORECASE)),
    ("revenue", "营业收入", "revenue", re.compile(r"营收|营业收入|收入|\brevenues?\b|\bsales\b", re.IGNORECASE)),
    (
        "net_profit",
        "净利润",
        "net profit",
        re.compile(r"净利润|净利(?!率)|(?<!毛)利润(?!率)|净赚|\bnet (?:profit|income)\b|\bearnings\b", re.IGNORECASE),
    ),
)


def _arithmetic_operands(tool_log: list[dict[str, Any]], key: str, zh: bool) -> tuple[list[tuple], tuple | None]:
    """``(companies, industry)``: ``(name, value, evidence id, data)`` per target for ``key`` in tool order, and the
    first target's industry value when the industry snapshot has it."""
    companies: list[tuple] = []
    industry = None
    seen: set[str] = set()
    for entry in tool_log:
        if not entry.get("ok"):
            continue
        data = entry.get("data") or {}
        eid, symbol = data.get("evidence_id"), str(data.get("symbol") or "")
        value = None
        found = metric_value(entry, key)
        if found is not None:
            value, data = found
        if key in {"pe", "pb", "pct_change"} and industry is None and entry.get("tool") == "get_fundamentals":
            snapshot = data.get("industry") or {}
            industry_value = (snapshot.get("metrics") or {}).get(key)
            if industry_value is not None and snapshot.get("evidence_id") and key != "pct_change":
                sector = snapshot.get("industry_name")
                label = f"{sector}行业" if zh else f"the {sector} industry"
                industry = (label, industry_value, snapshot["evidence_id"], {})
        if value is None or not eid or not symbol or symbol in seen:
            continue
        seen.add(symbol)
        companies.append((str(data.get("name") or symbol), float(value), eid, data))
    return companies, industry


_PRICE_KEYS = {"pct_change", "close", "amount", "volume"}
_FUNDAMENTAL_FIELDS = {
    "pe": ("pe_ttm", "pe"),
    "pb": ("pb",),
    "roe": ("roe",),
    "revenue": ("revenue",),
    "net_profit": ("net_profit",),
    "gross_margin": ("gross_margin", "grossprofit_margin"),
    "dividend_yield": ("dividend_yield", "dv_ratio", "dv_ttm"),
    "eps": ("eps", "basic_eps", "diluted_eps", "eps_ttm"),
    "market_cap": ("total_mv", "market_cap", "total_market_cap"),
}
_RATIO_FIELDS = {"roe", "gross_margin", "dividend_yield"}


def metric_value(entry: dict[str, Any], key: str) -> tuple[float, dict[str, Any]] | None:
    """``(value, data)`` of one metric in one tool result, in the unit the template states it (ratios in percent,
    amounts in CNY), or ``None``. A daily change computed from the last two closes is marked ``_computed_change``."""
    data = entry.get("data") or {}
    tool = entry.get("tool")
    if tool == "get_price_history" and key in _PRICE_KEYS:
        value = data.get({"pct_change": "pct_change_1d"}.get(key, key))
        if key == "pct_change" and value is None and (computed := _computed_change(data)):
            # the change the price sentence states as computed from the last two closes: it is compared, but
            # never restated without its closes (the verifier checks it against them in that sentence)
            return computed[1], {**data, "_computed_change": True}
        if key == "volume" and value is not None and not float(value):
            value = None
        return (float(value), data) if value is not None else None
    if tool != "get_fundamentals" or key in _PRICE_KEYS:
        return None
    metrics = data.get("metrics") or {}
    if key == "net_margin":
        revenue, profit = metrics.get("revenue"), metrics.get("net_profit")
        if revenue and profit is not None and float(revenue) >= 1e6:
            return round(float(profit) / float(revenue) * 100, 2), data
        return None
    fields = _FUNDAMENTAL_FIELDS.get(key, (key,))
    field = next((name for name in fields if metrics.get(name) is not None), None)
    if field is None:
        return None
    value = float(metrics[field])
    if key in _RATIO_FIELDS:
        in_percent = (data.get("metric_units") or {}).get(field) == "%"
        value = value if in_percent or abs(value) > 1 else value * 100
    if key in {"revenue", "net_profit"} and abs(value) < 1e6:
        return None  # an amount the template does not state (see _fundamentals)
    return value, data


def _arithmetic(query: str, tool_log: list[dict[str, Any]], zh: bool) -> list[str]:
    """The difference or ratio a question asks for, derived from two cited values (see ``_ASKS_DIFFERENCE``); for a
    comparison that asks for neither (round 10, F10: "哪个更低", "谁跌得多", "比较…的市盈率"), which value is higher."""
    difference, ratio = bool(_ASKS_DIFFERENCE.search(query or "")), bool(_ASKS_RATIO.search(query or ""))
    if not (difference or ratio):
        return _comparison_verdict(query, tool_log, zh) if _ASKS_COMPARISON.search(query or "") else []
    named = [
        (match.start(), key, label_zh, label_en)
        for key, label_zh, label_en, pattern in _ARITHMETIC_METRICS
        if (match := pattern.search(query))
    ]
    if not named:
        return []
    _position, key, label_zh, label_en = min(named)
    companies, industry = _arithmetic_operands(tool_log, key, zh)
    if len(companies) >= 2:
        first, second = companies[0], companies[1]
    elif len(companies) == 1 and industry is not None and _ASKS_INDUSTRY.search(query):
        first, second = companies[0], industry
    else:
        return []

    def shown(value: float, data: dict[str, Any]) -> str:
        if key in {"pct_change", "roe"}:
            return f"{_num(value)}%"
        if key == "close":
            return _px(value, data, zh)
        if key in {"pe", "pb"}:
            return _times(value, zh)
        return _money(value, zh)

    if first[3].get("_computed_change") or second[3].get("_computed_change"):
        # a change computed from closes has no stored value to derive a gap from: say which is higher, and why the
        # gap is not stated
        verdict = _comparison_verdict(query, tool_log, zh)
        computed = next(name for name, _value, _eid, data in (first, second) if data.get("_computed_change"))
        note = (
            f"{computed}的涨跌幅由最近两个收盘价推算（数据源未提供），因此不另行计算两者差值。"
            if zh
            else f"{computed}'s change is computed from its last two closes (the source has none), so no gap is stated."
        )
        return [*verdict, note]
    (name_a, a, eid_a, data_a), (name_b, b, eid_b, data_b) = first, second
    cites = f"[{eid_a}]" + (f"[{eid_b}]" if eid_b != eid_a else "")
    if key == "net_margin":
        return [] if ratio else [_margin_gap(first, second, cites, zh)]
    label = label_zh if zh else label_en
    operands = (
        f"{name_a}{label} {shown(a, data_a)}，{name_b} {shown(b, data_b)}"
        if zh
        else f"{name_a} {label} {shown(a, data_a)}, {name_b} {shown(b, data_b)}"
    )
    if ratio and b:
        times = _num(round(a / b, 2))
        return [
            f"{operands}，前者约为后者的 {times} 倍 {cites}。"
            if zh
            else f"{operands}: the former is about {times} times the latter {cites}."
        ]
    gap = abs(a - b)
    if key in {"pct_change", "roe"}:
        gap_text = f"{_num(round(gap, 2))} 个百分点" if zh else f"{_num(round(gap, 2))} percentage points"
    elif key == "close":
        gap_text = _px(round(gap, 3), data_a, zh)
    elif key in {"pe", "pb"}:
        gap_text = _num(round(gap, 2))
    else:
        gap_text = _money(gap, zh)
    if a == b:
        return [f"{operands}，两者相同 {cites}。" if zh else f"{operands}: they are equal {cites}."]
    higher = name_a if a > b else name_b
    return [
        f"{operands}，两者相差 {gap_text}（{higher}更高） {cites}。"
        if zh
        else f"{operands}: a difference of {gap_text} ({higher} is higher) {cites}."
    ]


# (round 10, F10) A comparison: "谁/哪个…更高/低/多/少", "比较/对比/相比", "compare", "which … higher".
_ASKS_COMPARISON = re.compile(
    r"(?:谁|哪个|哪一个|哪只|哪家|哪边)[^，。？?,;；]{0,10}?(?:高|低|大|小|多|少|贵|便宜|强|弱|活跃)|比较|对比|相比|"
    r"\bcompar(?:e|ed|ing|ison)\b|\bversus\b|\bvs\.?(?=\s)|"
    r"\bwhich\b[^.?!]{0,40}\b(?:higher|lower|bigger|smaller|more|less|cheaper|larger)\b",
    re.IGNORECASE,
)


def _margin_gap(first: tuple, second: tuple, cites: str, zh: bool) -> str:
    """Two net margins and their gap in percentage points, each margin with its net profit and revenue in the same
    sentence (the verifier derives the margins and the gap from the four cited amounts)."""
    parts = []
    for name, margin, _eid, data in (first, second):
        metrics = data.get("metrics") or {}
        profit, revenue = _money(metrics.get("net_profit"), zh), _money(metrics.get("revenue"), zh)
        parts.append(f"{name} {profit} {'÷' if zh else '/'} {revenue} ≈ {_num(margin)}%")
    (name_a, a, _ea, _da), (name_b, b, _eb, _db) = first, second
    gap = _num(round(abs(a - b), 2))
    higher = name_a if a > b else name_b
    if zh:
        return f"净利率：{'，'.join(parts)}，两者相差 {gap} 个百分点（{higher}更高） {cites}。"
    return f"Net margin: {', '.join(parts)}, a gap of {gap} percentage points ({higher} is higher) {cites}."


_OPERATION_WORDS = {
    "difference": ("差值", "the difference"),
    "ratio": ("倍数", "the ratio"),
    "relative": ("相对差异", "the relative difference"),
    "which": ("高低", "which is higher"),
}


def frame_sentences(request: dict[str, Any], tool_log: list[dict[str, Any]], zh: bool) -> tuple[list[str], list[str]]:
    """(round 11) The comparison a frame request asks for, from this run's tool results: ``(sentences, gaps)``.

    Each sentence writes both operands with their evidence ids next to the result (difference in the metric's unit or
    in percentage points, ratio to two decimals, relative difference in percent of the second operand, or which is
    higher), so the verifier re-derives it. An operand without data is named in ``gaps`` and nothing is computed from
    other figures."""
    from .frame import METRIC_BY_KEY, metric_label, operand_name, operand_value

    key = str(request.get("metric") or "")
    operation = str(request.get("operation") or "difference")
    metric = METRIC_BY_KEY.get(key)
    label = metric_label(key, zh)
    found, missing = [], []
    for operand in request.get("operands") or []:
        hit = operand_value(tool_log, operand, key)
        name = operand_name(operand, zh)
        if hit is None:
            missing.append(name)
        else:
            found.append((name, hit[0], hit[1], hit[2]))
    word_zh, word_en = _OPERATION_WORDS.get(operation, _OPERATION_WORDS["difference"])
    if missing or len(found) < 2 or metric is None:
        names = missing or [operand_name(item, zh) for item in request.get("operands") or []]
        gap = (
            f"当前数据中没有{'、'.join(names)}的{label}，因此无法计算两者的{word_zh}。"
            if zh
            else f"The data has no {label} for {' or '.join(names)}, so {word_en} cannot be computed."
        )
        return [], [gap]
    kind = metric.kind

    def shown(value: float, data: dict[str, Any]) -> str:
        if kind == "points":
            return f"{_num(value)}%"
        if kind == "multiple":
            return _times(value, zh)
        if kind == "money":
            return _money(value, zh)
        if kind == "price":
            return _px(value, data, zh)
        return _num(value)

    cites = "".join(dict.fromkeys(f"[{eid}]" for _name, _value, eid, _data in found))
    if len(found) > 2:
        ordered = sorted(found, key=lambda item: item[1], reverse=True)
        parts = [f"{name} {shown(value, data)}" for name, value, _eid, data in ordered]
        text = f"{label}由高到低：{'、'.join(parts)}" if zh else f"{label} from highest to lowest: {', '.join(parts)}"
        return [f"{text} {cites}。" if zh else f"{text} {cites}."], []
    (name_a, a, _ea, data_a), (name_b, b, _eb, data_b) = found[0], found[1]
    if data_a.get("_computed_change") or data_b.get("_computed_change"):
        # a change computed from two closes is compared (which side is higher) but not restated without its closes,
        # and no gap or ratio is derived from it
        computed = name_a if data_a.get("_computed_change") else name_b
        note = (
            f"{computed}的涨跌幅由最近两个收盘价推算（数据源未提供），因此不另行计算两者的{word_zh}。"
            if zh
            else f"{computed}'s change is computed from its last two closes (the source has none), so {word_en} is "
            "not stated."
        )

        def stated(value: float, data: dict[str, Any]) -> str:
            if data.get("_computed_change"):
                return "（按收盘价推算，见上文）" if zh else "(computed from closes, see above)"
            return f" {shown(value, data)}" if zh else f"({shown(value, data)})"

        if a == b:
            relation = "持平" if zh else "is level with"
        else:
            relation = ("高于" if a > b else "低于") if zh else ("is higher than" if a > b else "is lower than")
        verdict = (
            f"{label}：{name_a}{stated(a, data_a)} {relation} {name_b}{stated(b, data_b)} {cites}。"
            if zh
            else f"{label}: {name_a} {stated(a, data_a)} {relation} {name_b} {stated(b, data_b)} {cites}."
        )
        return [verdict], [] if operation == "which" else [note]
    if operation == "which":
        if a == b:
            relation = "持平" if zh else "is level with"
        else:
            relation = ("高于" if a > b else "低于") if zh else ("is higher than" if a > b else "is lower than")
        if zh:
            return [f"{label}：{name_a} {shown(a, data_a)} {relation} {name_b} {shown(b, data_b)} {cites}。"], []
        return [f"{label}: {name_a} ({shown(a, data_a)}) {relation} {name_b} ({shown(b, data_b)}) {cites}."], []
    if key == "net_margin":
        sentence = _margin_gap(found[0], found[1], cites, zh)
        if operation == "difference":
            return [sentence], []
        note = (
            "净利率本身是推算值，这里给出两者相差的百分点，不再计算其比值。"
            if zh
            else "Net margins are derived figures, so the gap in percentage points is given instead of a ratio."
        )
        return [sentence], [note]
    operands = (
        f"{name_a}{label} {shown(a, data_a)}，{name_b} {shown(b, data_b)}"
        if zh
        else f"{name_a} {label} {shown(a, data_a)}, {name_b} {shown(b, data_b)}"
    )
    if operation == "ratio":
        if not b:
            return [], ["后者为零，无法计算倍数。" if zh else "The second value is zero, so no ratio can be computed."]
        times = _num(round(a / b, 2))
        return [
            f"{operands}，前者约为后者的 {times} 倍 {cites}。"
            if zh
            else f"{operands}: the former is about {times} times the latter {cites}."
        ], []
    if operation == "relative":
        if not b:
            note = "基数为零，无法计算相对差异。" if zh else "The base value is zero, so no percentage can be computed."
            return [], [note]
        pct = _num(round(abs(a - b) / abs(b) * 100, 2))
        if a == b:
            return [f"{operands}，两者相同 {cites}。" if zh else f"{operands}: they are equal {cites}."], []
        lower = a < b
        industry_base = any(item.get("kind") == "industry" for item in (request.get("operands") or [])[1:2])
        if zh:
            if kind == "multiple" and industry_base:
                relation = f"{name_a}相对{name_b}{'折价' if lower else '溢价'}约 {pct}%"
            else:
                relation = f"{name_a}比{name_b}{'低' if lower else '高'}约 {pct}%（以{name_b}为基数）"
            return [f"{operands}，{relation} {cites}。"], []
        side = "below" if lower else "above"
        extra = f" (a {'discount' if lower else 'premium'})" if kind == "multiple" and industry_base else ""
        return [f"{operands}: {name_a} is about {pct}% {side} {name_b}{extra} {cites}."], []
    gap = abs(a - b)
    if kind == "points":
        gap_text = f"{_num(round(gap, 2))} 个百分点" if zh else f"{_num(round(gap, 2))} percentage points"
    elif kind == "money":
        gap_text = _money(gap, zh)
    elif kind == "price":
        gap_text = _px(round(gap, 3), data_a, zh)
    else:
        gap_text = _num(round(gap, 2))
    if a == b:
        return [f"{operands}，两者相同 {cites}。" if zh else f"{operands}: they are equal {cites}."], []
    higher = name_a if a > b else name_b
    return [
        f"{operands}，两者相差 {gap_text}（{higher}更高） {cites}。"
        if zh
        else f"{operands}: a difference of {gap_text} ({higher} is higher) {cites}."
    ], []


def frame_result(request: dict[str, Any], tool_log: list[dict[str, Any]]) -> float | None:
    """The number a frame request computes (gap, ratio or percent), for checking whether an LLM draft states it."""
    from .frame import operand_value

    key, operation = str(request.get("metric") or ""), str(request.get("operation") or "difference")
    values = []
    for operand in (request.get("operands") or [])[:2]:
        hit = operand_value(tool_log, operand, key)
        if hit is None or hit[2].get("_computed_change"):
            return None
        values.append(hit[0])
    if len(values) < 2 or operation == "which":
        return None
    a, b = values
    if operation == "ratio" and key != "net_margin":
        return round(a / b, 2) if b else None
    if operation == "relative" and key != "net_margin":
        return round(abs(a - b) / abs(b) * 100, 2) if b else None
    return round(abs(a - b), 2)


def _comparison_verdict(query: str, tool_log: list[dict[str, Any]], zh: bool) -> list[str]:
    """One sentence saying which cited value is higher, for the metric the comparison names first: "市净率：中国平安
    1.1 倍 低于 五粮液 5.4 倍"; three or more targets are ordered from highest to lowest. A comparison that names no
    metric ("谁更好") is a judgment and gets no verdict."""
    named = [
        (match.start(), key, label_zh, label_en)
        for key, label_zh, label_en, pattern in _ARITHMETIC_METRICS
        if (match := pattern.search(query))
    ]
    if not named:
        return []
    _position, key, label_zh, label_en = min(named)
    if key == "net_margin":
        return []  # the margins are ranked in words next to their derivation (``_margin_ranking``)
    companies, industry = _arithmetic_operands(tool_log, key, zh)
    operands = list(companies)
    if len(operands) == 1 and industry is not None and _ASKS_INDUSTRY.search(query):
        operands.append(industry)
    if len(operands) < 2:
        return []

    def shown(value: float, data: dict[str, Any]) -> str:
        if data.get("_computed_change"):
            return "（按收盘价推算，见上文）" if zh else "(computed from closes, see above)"
        if key in {"pct_change", "roe"}:
            return f"{_num(value)}%"
        if key == "close":
            return _px(value, data, zh)
        if key in {"pe", "pb"}:
            return _times(value, zh)
        return _money(value, zh)

    label = label_zh if zh else label_en
    cites = "".join(dict.fromkeys(f"[{eid}]" for _name, _value, eid, _data in operands))
    if len(operands) == 2:
        (name_a, a, _eid_a, data_a), (name_b, b, _eid_b, data_b) = operands
        if a == b:
            relation = "持平" if zh else "is level with"
        else:
            relation = ("高于" if a > b else "低于") if zh else ("is higher than" if a > b else "is lower than")
        text = (
            f"{label}：{name_a}{shown(a, data_a) if data_a.get('_computed_change') else ' ' + shown(a, data_a)} "
            f"{relation} {name_b}{shown(b, data_b) if data_b.get('_computed_change') else ' ' + shown(b, data_b)}"
            if zh
            else f"{label}: {name_a} ({shown(a, data_a)}) {relation} {name_b} ({shown(b, data_b)})"
        )
    else:
        ordered = sorted(operands, key=lambda item: item[1], reverse=True)
        parts = [f"{name} {shown(value, data)}" for name, value, _eid, data in ordered]
        text = f"{label}由高到低：{'、'.join(parts)}" if zh else f"{label} from highest to lowest: {', '.join(parts)}"
    falling = key == "pct_change" and re.search(r"跌|\b(?:fell|dropped|lost)\b", query, re.IGNORECASE)
    if falling and all(value > 0 for _name, value, _eid, _data in operands):
        text += "（两者当日均为上涨）" if zh else " (both rose on the day)"
    return [f"{text} {cites}。" if zh else f"{text} {cites}."]


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
    computed = _computed_change(data) if pct is None else None
    if zh:
        change = f"，当日涨跌幅 {_num(pct)}%" if pct is not None else ""
        if computed:
            previous, change_pct = computed
            change = (
                f"，按前一交易日收盘 {_px(previous, data, zh)} 计算的当日涨跌幅约 {_num(change_pct)}%"
                "（数据源未提供涨跌幅）"
            )
        sentences = [f"{name}（{symbol}）最新可用收盘价为 {_px(close, data, zh)}（{as_of}）{change} [{eid}]。"]
    else:
        change = f", daily change {_num(pct)}%" if pct is not None else ""
        if computed:
            previous, change_pct = computed
            change = (
                f", a daily change of about {_num(change_pct)}% computed from the previous close of "
                f"{_px(previous, data, zh)} (the source reports no daily change)"
            )
        sentences = [f"{name} ({symbol}) last available close was {_px(close, data, zh)} on {as_of}{change} [{eid}]."]
    if request is not None and request.needs_quote:
        sentences.extend(_price_details(data, zh, request))
    if request is not None and request.year_to_date:
        sentences.extend(_year_to_date(data, zh))
    return sentences


def _computed_change(data: dict[str, Any]) -> tuple[float, float] | None:
    """``(previous close, change in %)`` from the last two closes, for a fund or index whose source leaves the daily
    change empty (510300 offline). Only when the latest close is the quoted one; labelled "computed" by the caller."""
    closes = [row for row in data.get("recent_closes") or [] if isinstance(row, dict) and row.get("close")]
    if len(closes) < 2 or str(closes[-1].get("date") or "")[:10] != str(data.get("as_of") or "")[:10]:
        return None
    previous, latest = float(closes[-2]["close"]), float(closes[-1]["close"])
    if not previous or latest != float(data.get("close") or 0):
        return None
    return previous, round((latest / previous - 1) * 100, 2)


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
# (round 10, F11) A corpus label ("fincprg", "fiqa", "fir_bench_reports"): the public dataset a document came from,
# not a publisher, so it is never written as "X发布的"; and a knowledge document (research note, product document,
# FAQ) that does not mention any requested target is background from the corpus, not about the question's target.
_CORPUS_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")
_KNOWLEDGE_TYPES = {"research_note", "product_doc", "faq"}


def _mentions_target(document: dict[str, Any], targets: list[str]) -> bool:
    text = f"{document.get('title') or ''} {document.get('excerpt') or ''}"
    for name in targets:
        name = str(name or "").strip()
        code = re.search(r"\d{6}", name)
        tail = name[-2:] if len(name) >= 4 and re.fullmatch(r"[\u4e00-\u9fff]{2}", name[-2:]) else ""
        if (name and name in text) or (code and code.group(0) in text) or (tail and tail in text):
            return True
    return False


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
    targets = [str(name) for name in data.get("targets") or [] if name]
    documents = [
        document
        for document in data.get("documents") or []
        if not (targets and document.get("source_type") in _KNOWLEDGE_TYPES and not _mentions_target(document, targets))
    ]
    for document in documents[:_MAX_DOCS_PER_TOOL]:
        eid = document.get("evidence_id")
        if not eid:
            continue
        category_zh, category_en = _DOCUMENT_CATEGORY.get(str(document.get("source_type") or ""), ("资料", "document"))
        # the publisher name comes from the data provider, but is still shown only when it is inert text and not a
        # corpus label
        publisher = str(document.get("source_name") or "")[:40]
        source = safe_headline(publisher) if publisher and not _CORPUS_LABEL.match(publisher) else None
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


# (round 10, F11) A failed tool is named by the data it would have given and a plain reason, never by its internal
# name and error code ("get_price_history: not_found"); the graph gives the compliance guard the same note, so the
# limitation appears once.
_TOOL_DATA = {
    "get_price_history": ("行情数据", "market data"),
    "compute_indicators": ("技术指标", "technical indicators"),
    "get_fundamentals": ("基本面数据", "fundamentals"),
    "get_macro_indicators": ("宏观数据", "macro data"),
    "search_news": ("新闻", "news"),
    "search_announcements": ("公告", "announcements"),
    "search_knowledge": ("研究资料", "research documents"),
    "analyze_sentiment": ("舆情分析", "sentiment analysis"),
    "resolve_entity": ("标的识别", "the security lookup"),
    "explain_concept": ("概念解释", "the concept lookup"),
}
_FAILURE_REASON = {
    "not_found": ("当前数据源中没有相关记录", "the configured sources have no record"),
    "timeout": ("数据源响应超时", "the source timed out"),
    "unavailable": ("可用数据不足", "there is not enough data"),
    "upstream_error": ("数据源暂时不可用", "the source is temporarily unavailable"),
    "invalid_arguments": ("请求参数无效", "the request was invalid"),
}


def failure_note(tool: str, code: str | None, *, zh: bool) -> str:
    """ "行情数据未取到（当前数据源中没有相关记录）" / "No market data: the configured sources have no record"."""
    label_zh, label_en = _TOOL_DATA.get(tool, ("数据", "data"))
    reason_zh, reason_en = _FAILURE_REASON.get(str(code or ""), ("数据源返回错误", "the source returned an error"))
    return f"{label_zh}未取到（{reason_zh}）" if zh else f"No {label_en}: {reason_en}"


def _failure_text(tool: str, error: dict[str, Any], *, zh: bool) -> str:
    return failure_note(tool, error.get("code"), zh=zh)


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
