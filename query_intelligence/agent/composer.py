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
    METRICS,
    PROFIT_GROWTH_FIELDS,
    PriceRequest,
    asks_about_industry,
    bind_positions,
    coverage_gaps,
    drawdown_gaps,
    failed_target_statements,
    flow_gaps,
    foreign_macro_gaps,
    holding_lots,
    holding_positions,
    holding_purchase,
    holding_value_request,
    in_fund_units,
    indicator_gaps,
    industry_gaps,
    macro_gaps,
    name_forms,
    non_stock_fundamental_gaps,
    range_52w_gaps,
    requested_metrics,
    requested_price_fields,
    without_holding_value,
    year_to_date_gaps,
)
from .frame import TURNOVER
from .names import english_display

_MAX_DOCS_PER_TOOL = 3

# (round 13) The definition a figure uses, when the data states it (``definitions``, from the corrections layer
# data/snapshot/structured_data_fix.json): the answer names it next to the figure ("ROE（加权）", "营业总收入",
# "归母净利润", "basic EPS"), since the same word covers different figures (weighted vs period-end ROE, ...).
_BASIS_FIELD = {
    "roe": "roe_basis",
    "revenue": "revenue_basis",
    "net_profit": "net_profit_basis",
    "eps": "eps_basis",
    "netprofit_yoy": "netprofit_yoy_basis",
    "revenue_yoy": "revenue_yoy_basis",
}
# basis -> (zh label, en label, zh short, en short): the label stands for the metric's name, the short form names
# the definition in a comparison note
_BASIS_LABELS: dict[str, dict[str, tuple[str, str, str, str]]] = {
    "roe_basis": {
        "weighted": ("ROE（加权）", "ROE (weighted average)", "加权平均", "weighted average"),
        "period_end": ("ROE（期末净资产）", "ROE (on period-end equity)", "按期末净资产", "on period-end equity"),
    },
    "revenue_basis": {
        # (round 14, review-10 J13) English labels carry no Chinese
        "营业收入": ("营业收入", "operating revenue", "营业收入", "operating revenue"),
        "营业总收入": ("营业总收入", "total operating revenue", "营业总收入", "total operating revenue"),
    },
    "net_profit_basis": {
        "归母": ("归母净利润", "net profit attributable to shareholders", "归母", "attributable to shareholders"),
        "total": (
            "净利润（含少数股东损益）",
            "net profit incl. minority interests",
            "含少数股东损益",
            "incl. minority interests",
        ),
    },
    "eps_basis": {
        "basic": ("基本每股收益", "basic EPS", "基本", "basic"),
        "diluted": ("稀释每股收益", "diluted EPS", "稀释", "diluted"),
    },
    "netprofit_yoy_basis": {
        "归母": ("归母净利润增速", "growth of net profit attributable to shareholders", "归母", "attributable"),
        "total": (
            "净利润（含少数股东损益）增速",
            "growth of net profit incl. minority interests",
            "含少数股东损益",
            "incl. minority interests",
        ),
    },
    "revenue_yoy_basis": {
        "营业收入": ("营业收入增速", "operating revenue growth", "营业收入", "operating revenue"),
        "营业总收入": ("营业总收入增速", "total operating revenue growth", "营业总收入", "total operating revenue"),
        "营业收入(2024年调整后)": (
            "营业收入增速（对比重述后的上年数据）",
            "operating revenue growth (vs the restated prior year)",
            "对比重述后的上年数据",
            "vs the restated prior year",
        ),
    },
}


def _basis(data: dict[str, Any], field: str) -> tuple[str, str, str, str] | None:
    key = _BASIS_FIELD.get(field)
    definitions = data.get("definitions") if isinstance(data.get("definitions"), dict) else {}
    return _BASIS_LABELS.get(key or "", {}).get(str(definitions.get(key) or "")) if key else None


def basis_label(data: dict[str, Any], field: str, zh: bool, default: str) -> str:
    """The metric's name with its definition when the data states one ("ROE（加权）"), else ``default``."""
    found = _basis(data, field)
    return (found[0] if zh else found[1]) if found else default


def basis_note(field: str, operands: list[tuple[str, dict[str, Any]]], zh: bool) -> list[str]:
    """(round 13) One sentence naming the definition each compared figure uses, when the data states any: "口径：
    五粮液的ROE为加权平均，宁德时代的ROE为按期末净资产；口径不同，比较仅供参考。" (no number of its own)."""
    if field not in _BASIS_FIELD or len(operands) < 2:
        return []
    found = [(name, _basis(data, field)) for name, data in operands]
    if all(basis is None for _name, basis in found):
        return []
    metric = {"roe": "ROE", "revenue": "收入", "net_profit": "净利润", "eps": "每股收益"}.get(field, field)
    metric_en = {"revenue": "revenue", "net_profit": "net profit", "eps": "EPS"}.get(field, field.upper())
    shorts = {basis[2] if zh else basis[3] for _name, basis in found if basis is not None}
    if all(basis is not None for _name, basis in found) and len(shorts) == 1:
        word = shorts.pop()
        return [f"口径：以上{metric}均为{word}口径。" if zh else f"Definition: each {metric_en} above is {word}."]
    if zh:
        parts = [f"{name}的{metric}为{basis[2] if basis else '数据未标明口径'}" for name, basis in found]
        return [f"口径：{'，'.join(parts)}；口径不同，比较仅供参考。"]
    parts = [f"{name}'s {metric_en} is {basis[3] if basis else 'of unstated definition'}" for name, basis in found]
    return [f"Definitions: {'; '.join(parts)}. They differ, so the comparison is indicative only."]


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
    wanted = requested_metrics(without_holding_value(query) if query else query)
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
            sentences = _fundamentals(data, zh, industry_first=industry_first, turnover_rate=request.turnover_rate)
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
            quoted = any(marker in item for item in sentences for marker in _HEADLINE_MARKERS)
            if quoted:
                limitations.append(DOCUMENT_HEADLINE_LIMITATION_ZH if zh else DOCUMENT_HEADLINE_LIMITATION_EN)
            else:
                limitations.append(DOCUMENT_TEXT_LIMITATION_ZH if zh else DOCUMENT_TEXT_LIMITATION_EN)
        for evidence_id in entry.get("evidence_ids") or []:
            if evidence_id not in evidence_used:
                evidence_used.append(evidence_id)

    if len(margins) >= 2:
        facts.append(_margin_ranking(margins, zh))
    if "eps" in {metric.key for metric in wanted}:
        facts.extend(fact for fact in _implied_eps(tool_log, zh) if fact not in facts)
    holding = holding_value_request(query) if query else None
    positions = holding_positions(query) if query else []
    several = _several_holdings(positions, tool_log, query, zh, purchase=holding_purchase(query)) if positions else []
    if several:
        facts = [*several, *facts]
    elif holding is not None:
        holding_facts = _holding_value(
            holding[0],
            tool_log,
            zh,
            units=in_fund_units(query),
            lots=holding_lots(query),
            purchase=holding_purchase(query),
        )
        facts = [*holding_facts, *facts]
    costed = cost_basis_sentences(tool_log, query, zh) if query else []
    if costed:
        facts = [*costed, *facts]
    dated, dated_gaps = dated_close_sentences(query, tool_log, zh) if query else ([], [])
    if dated:
        facts = [*dated, *(fact for fact in facts if fact not in dated)]
    frame_gaps: list[str] = []
    if frame_request:
        framed, frame_gaps = frame_sentences(frame_request, tool_log, zh)
        facts = [*framed, *(fact for fact in facts if fact not in framed)]
    else:
        for sentence in _arithmetic(query, tool_log, zh):
            if sentence not in facts:
                facts.append(sentence)
    gaps: list[str] = [*frame_gaps, *dated_gaps]
    if query:
        gaps += _price_date_note(query, tool_log, zh, frame_request)
        gaps += [
            *news_gaps(query, tool_log, zh=zh),
            # (round 15, K12) a glossary definition is not the data a metric question asked for
            *coverage_gaps(query, _data_tools(tool_log), zh=zh, names=names),
            *industry_gaps(query, tool_log, zh=zh),
            *_missing_industry_snapshot(query, tool_log, zh),
            *non_stock_fundamental_gaps(query, tool_log, zh=zh, names=names, types=types),
            *macro_gaps(query, tool_log, zh=zh),
            *foreign_macro_gaps(query, zh=zh),
            *flow_gaps(query, tool_log, zh=zh),
            *year_to_date_gaps(query, tool_log, zh=zh),
            *drawdown_gaps(query, tool_log, zh=zh),
            *range_52w_gaps(query, tool_log, zh=zh),
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


def _holding_value(
    shares: int,
    tool_log: list[dict[str, Any]],
    zh: bool,
    *,
    units: bool = False,
    lots: int | None = None,
    purchase: bool = False,
) -> list[str]:
    """(round 11, G5) "我有1000股五粮液，值多少钱": shares × the latest close, with the date, both operands in the
    sentence, and a note that this is a market value at the close, not a tradable price, a valuation or advice.
    (round 12) A holding stated in fund units ("两万份沪深300ETF", "4000 units") is written in units, one stated in
    lots ("3手", 100 shares each) as lots and shares."""
    for entry in tool_log:
        data = entry.get("data") or {}
        close, eid = data.get("close"), data.get("evidence_id")
        if entry.get("tool") != "get_price_history" or not entry.get("ok") or close is None or not eid:
            continue
        value = round(shares * float(close), 2)
        name, as_of = data.get("name") or data.get("symbol"), data.get("as_of")
        unit = "份" if units else "股"
        held = f"{lots} 手（{shares} 股）" if lots else f"{shares} {unit}"
        held_en = f"{lots} lots ({shares} shares)" if lots else f"{shares} {'units' if units else 'shares'}"
        if purchase:
            # (round 13) "买三手要多少钱": the same count × close, as the amount needed before trading costs
            if zh:
                return [
                    f"按 {as_of} 的收盘价 {_px(close, data, zh)} 计算，买入 {held}{name}约需 {shares} × "
                    f"{_num(close)} = {_num(value)} 元 [{eid}]。",
                    "这是按最近收盘价估算的买入金额，未含佣金、印花税等交易费用；实际成交价会不同，这也不是投资建议。",
                ]
            return [
                f"At the {as_of} close of {_px(close, data, zh)}, buying {held_en} of {name} takes about {shares} × "
                f"{_num(close)} = CNY {_num(value)} [{eid}].",
                "This is an estimate at the last close before commissions and stamp duty; the actual fill price will "
                "differ, and this is not investment advice.",
            ]
        if zh:
            return [
                f"按 {as_of} 的收盘价 {_px(close, data, zh)} 计算，{held}{name}的市值约为 {shares} × "
                f"{_num(close)} = {_num(value)} 元 [{eid}]。",
                "这是按最近收盘价计算的持仓市值，不是可成交价格，也不是估值判断或投资建议。",
            ]
        return [
            f"At the {as_of} close of {_px(close, data, zh)}, {held_en} of {name} are worth {shares} × "
            f"{_num(close)} = CNY {_num(value)} [{eid}].",
            "This is the market value at the last close, not a tradable price, a valuation or investment advice.",
        ]
    return []


def _several_holdings(
    positions: list[dict[str, Any]], tool_log: list[dict[str, Any]], query: str, zh: bool, *, purchase: bool = False
) -> list[str]:
    """(round 13, review-9 I5) Several positions in one question ("我有200股茅台和3手五粮液，一共值多少"): each
    count × its own target's close and the total, in one sentence with every operand. ``[]`` when a position's
    target cannot be told or has no close (the single-holding answer then states what it can)."""
    closes: dict[str, dict[str, Any]] = {}
    for entry in tool_log:
        data = entry.get("data") or {}
        if entry.get("tool") == "get_price_history" and entry.get("ok") and data.get("close") is not None:
            closes.setdefault(str(data.get("symbol")), data)
    candidates = {symbol: name_forms(str(data.get("name") or symbol), symbol) for symbol, data in closes.items()}
    bound = bind_positions(query, positions, candidates)
    if not closes or any(key is None for key in bound):
        return []
    parts, values, cites = [], [], []
    for position, symbol in zip(positions, bound, strict=True):
        data = closes[str(symbol)]
        close, eid, name = data["close"], data.get("evidence_id"), data.get("name") or symbol
        shares = position["shares"]
        value = round(shares * float(close), 2)
        values.append(value)
        cites.append(f"[{eid}]")
        unit = "份" if position["units"] else "股"
        held = f"{position['lots']} 手（{shares} 股）" if position["lots"] else f"{shares} {unit}"
        held_en = (
            f"{position['lots']} lots ({shares} shares)"
            if position["lots"]
            else f"{shares} {'units' if position['units'] else 'shares'}"
        )
        parts.append(
            f"{held}{name} {shares} × {_num(close)} = {_num(value)} 元"
            if zh
            else f"{held_en} of {name}: {shares} × {_num(close)} = CNY {_num(value)}"
        )
    total = _num(round(sum(values), 2))
    joined_cites = "".join(dict.fromkeys(cites))
    dates = sorted({str(closes[str(symbol)].get("as_of") or "") for symbol in bound} - {""})
    when = "、".join(dates) if zh else ", ".join(dates)
    if zh:
        verb = "买入约需" if purchase else "市值约为"
        sentence = f"按最新收盘价（{when}）计算：{'，'.join(parts)}，合计{verb} {total} 元 {joined_cites}。"
        note = (
            "这是按最近收盘价估算的金额，未含佣金、印花税等交易费用，不是可成交价格，也不是估值判断或投资建议。"
            if purchase
            else "这是按最近收盘价计算的持仓市值，不是可成交价格，也不是估值判断或投资建议。"
        )
        return [sentence, note]
    # not "worth CNY …": the output layer reads that as a fair-value statement
    verb = "buying them all takes about" if purchase else "market value"
    sentence = f"At the latest closes ({when}): {', '.join(parts)}, {verb} CNY {total} in total {joined_cites}."
    note = (
        "This is an estimate at the last closes before commissions and stamp duty, not investment advice."
        if purchase
        else "This is the market value at the last closes, not a tradable price, a valuation or investment advice."
    )
    return [sentence, note]


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
        profit_word = basis_label(data, "net_profit", zh, "净利润" if zh else "net profit")
        revenue_word = basis_label(data, "revenue", zh, "营业收入" if zh else "revenue")
        sentences.append(
            f"{name} 净利率（{profit_word} ÷ {revenue_word}，报告期 {period}）：{_money(profit, zh)} ÷ "
            f"{_money(revenue, zh)} ≈ {_num(margin)}% [{eid}]。"
            if zh
            else f"{name} net margin ({profit_word} / {revenue_word}, period {period}): {_money(profit, zh)} / "
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
        re.compile(r"净利率|净利润率|销售净利率|\bnet[- ](?:profit[- ])?margins?\b", re.IGNORECASE),
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
        # (round 12) the comparison frame's turnover vocabulary ("成交了多少钱", "交易更活跃", "value traded")
        re.compile(rf"{TURNOVER.pattern}|成交(?:更|最|比较)?(?:大|多|少)", re.IGNORECASE),
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
    # (round 14, review-10 J8) each from its own field, never from a neighbour's
    "float_market_cap": ("circ_mv",),
    "profit_dedt": ("profit_dedt",),
    "operating_profit": ("operate_profit", "operating_profit"),
    "gross_profit": ("gross_profit",),
    "total_profit": ("total_profit",),
}
_RATIO_FIELDS = {"roe", "gross_margin", "dividend_yield"}
_STATEMENT_AMOUNTS = {"revenue", "net_profit", "profit_dedt", "operating_profit", "gross_profit", "total_profit"}


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
    if key in _STATEMENT_AMOUNTS and abs(value) < 1e6:
        return None  # an amount the template does not state (see _fundamentals)
    return value, data


def _arithmetic(query: str, tool_log: list[dict[str, Any]], zh: bool) -> list[str]:
    """The difference or ratio a question asks for, derived from two cited values (see ``_ASKS_DIFFERENCE``); for a
    comparison that asks for neither (round 10, F10: "哪个更低", "谁跌得多", "比较…的市盈率"), which value is higher.
    (round 13) When the compared figures carry definitions, a note names them (``basis_note``)."""
    sentences = _arithmetic_sentences(query, tool_log, zh)
    if not sentences:
        return sentences
    named = [
        (match.start(), key)
        for key, _label_zh, _label_en, pattern in _ARITHMETIC_METRICS
        if (match := pattern.search(query or ""))
    ]
    if not named:
        return sentences
    key = min(named)[1]
    companies, _industry = _arithmetic_operands(tool_log, key, zh)
    note = basis_note(key, [(name, data) for name, _value, _eid, data in companies], zh)
    return [*sentences, *(line for line in note if line not in sentences)]


def _arithmetic_sentences(query: str, tool_log: list[dict[str, Any]], zh: bool) -> list[str]:
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
    if len(companies) > 2:
        # (round 13, review-9 I4) a gap or ratio of three or more targets the frame did not read: never compute it from
        # the first two silently; order them instead (the frame computes every pair when it reads the question)
        return _comparison_verdict(query, tool_log, zh)
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


def _price_date_note(
    query: str, tool_log: list[dict[str, Any]], zh: bool, frame_request: dict[str, Any] | None
) -> list[str]:
    """(round 12) A comparison of daily price data (change, close, turnover) whose targets are dated on different
    trading days (the v1 snapshot is as of 2026-04-22, the extension as of 2026-09-30, a live quote today): each
    target's date is named, so a reader does not take two different sessions for the same one."""
    if frame_request:
        key = str(frame_request.get("metric") or "")
    else:
        asked = _ASKS_COMPARISON.search(query) or _ASKS_DIFFERENCE.search(query) or _ASKS_RATIO.search(query)
        named = [
            (match.start(), metric_key)
            for metric_key, _zh, _en, pattern in _ARITHMETIC_METRICS
            if (match := pattern.search(query))
        ]
        if not asked or not named:
            return []
        key = min(named)[1]
    if key not in _PRICE_KEYS:
        return []
    companies, _industry = _arithmetic_operands(tool_log, key, zh)
    dated = [(name, str(data.get("as_of") or "")[:10]) for name, _value, _eid, data in companies]
    if len({day for _name, day in dated if day}) < 2:
        return []
    if zh:
        listing = "，".join(f"{name} {day}" for name, day in dated if day)
        return [f"注意：所比较的行情日期不同（{listing}），各为该标的最新可用交易日，不是同一交易日的数据。"]
    listing = ", ".join(f"{name} {day}" for name, day in dated if day)
    return [
        f"Note: the compared market data are from different trading days ({listing}); each is that target's latest "
        "available session, not the same day."
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
    "share": ("占比", "the share"),
    "share_total": ("占合计的比例", "the share of the combined total"),
    "sum": ("合计", "the total"),
    "spread": ("极差", "the spread"),
    "mean": ("平均值", "the average"),
    "total_vs_one": ("合计与另一方的差", "the total against the other side"),
}
_GROUP_ZH = {2: "两", 3: "三", 4: "四", 5: "五", 6: "六"}
_GROUP_EN = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def frame_sentences(request: dict[str, Any], tool_log: list[dict[str, Any]], zh: bool) -> tuple[list[str], list[str]]:
    """(round 11) The comparison a frame request asks for, from this run's tool results: ``(sentences, gaps)``.

    Each sentence writes both operands with their evidence ids next to the result (difference in the metric's unit or
    in percentage points, ratio to two decimals, relative difference in percent of the second operand, or which is
    higher), so the verifier re-derives it. An operand without data is named in ``gaps`` and nothing is computed from
    other figures. (round 13) When the compared figures carry definitions, a note names them after the result.

    (round 15, review-11 K4/K10) A message with two operations ("两家加一起是多少，宁德占其中几成") has the further
    ones in ``request["also"]``: each is answered after the first, with its own operands, values and evidence ids."""
    sentences, gaps = _one_frame_sentences(request, tool_log, zh)
    for extra in request.get("also") or []:
        more, more_gaps = _one_frame_sentences(extra, tool_log, zh)
        sentences += [line for line in more if line not in sentences]
        gaps += [line for line in more_gaps if line not in gaps]
    return sentences, gaps


def _one_frame_sentences(
    request: dict[str, Any], tool_log: list[dict[str, Any]], zh: bool
) -> tuple[list[str], list[str]]:
    from .frame import operand_name, operand_value

    sentences, gaps = _frame_sentences(request, tool_log, zh)
    if not sentences:
        return sentences, gaps
    key = str(request.get("metric") or "")
    operands = []
    for operand in request.get("operands") or []:
        hit = operand_value(tool_log, operand, key)
        if hit is not None and operand.get("kind") != "industry":
            operands.append((operand_name(operand, zh), hit[2]))
    note = basis_note(key, operands, zh)
    return [*sentences, *(line for line in note if line not in sentences)], gaps


def _frame_sentences(request: dict[str, Any], tool_log: list[dict[str, Any]], zh: bool) -> tuple[list[str], list[str]]:
    from .frame import METRIC_BY_KEY, metric_label, operand_name, operand_value

    key = str(request.get("metric") or "")
    operation = str(request.get("operation") or "difference")
    metric = METRIC_BY_KEY.get(key)
    label = metric_label(key, zh)
    if any(operand.get("metric") for operand in request.get("operands") or []):
        return _metric_pair_sentences(request, tool_log, zh)
    found, missing = [], []
    word_zh, word_en = _OPERATION_WORDS.get(operation, _OPERATION_WORDS["difference"])
    unresolved = [operand_name(item, zh) for item in request.get("operands") or [] if item.get("kind") == "unresolved"]
    if unresolved:
        # (round 14, review-10 J4) a comparison whose second operand could not be identified is never answered with
        # the other operand alone: say which one is missing
        names = "、".join(f"「{name}」" for name in unresolved) if zh else ", ".join(f'"{name}"' for name in unresolved)
        count = len(request.get("operands") or [])
        group = "两者" if count <= 2 else f"{_GROUP_ZH.get(count, '各')}者"
        return [], [
            f"未能识别{names}对应的证券，因此无法计算{group}{label}的{word_zh}；以下只列出能识别的部分。"
            if zh
            else f"Could not tell which security {names} refers to, so {word_en} of the {label} cannot be computed; "
            "only the identified part is listed below."
        ]
    for operand in request.get("operands") or []:
        hit = operand_value(tool_log, operand, key)
        if hit is not None and operand.get("kind") == "industry" and not operand.get("industry"):
            # "比行业低百分之多少" in one message: the industry is named from its snapshot ("保险行业平均")
            operand = {**operand, "industry": hit[2].get("industry_name")}
        name = operand_name(operand, zh)
        if hit is None:
            missing.append(name)
        else:
            found.append((name, hit[0], hit[1], hit[2]))
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
    computed = any(data.get("_computed_change") for _name, _value, _eid, data in found)
    many = operation in {"sum", "spread", "share_total", "mean", "total_vs_one"} or (
        request.get("pairwise") and len(found) > 2
    )
    totals = {"sum", "share_total", "total_vs_one"}
    if many and not computed and (operation not in totals or kind in {"money", "count"}):
        return _many_operand_sentences(operation, found, label, kind, shown, cites, zh, request), []
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
    if operation == "which" and key == "net_margin":
        # derived margins are stated with their amounts (the verifier re-derives them), the higher side named
        return [_margin_gap(found[0], found[1], cites, zh)], []
    if operation == "which":
        if a == b:
            relation = "持平" if zh else "is level with"
        else:
            relation = ("高于" if a > b else "低于") if zh else ("is higher than" if a > b else "is lower than")
        industry_base = any(item.get("kind") == "industry" for item in (request.get("operands") or [])[1:2])
        if industry_base and a != b and b and kind in {"multiple", "points"}:
            # (round 14, review-10 J14) a target against its industry average: higher or lower, by how much and by
            # what percentage of the average
            gap = _num(round(abs(a - b), 2))
            gap_text = (
                f"{gap} 个百分点"
                if kind == "points" and zh
                else f"{gap} percentage points"
                if kind == "points"
                else gap
            )
            pct = _num(round(abs(a - b) / abs(b) * 100, 2))
            if zh:
                side = "高" if a > b else "低"
                return [
                    f"{label}：{name_a} {shown(a, data_a)} {relation} {name_b} {shown(b, data_b)}，{side} {gap_text}"
                    f"（约 {pct}%，以{name_b}为基数） {cites}。"
                ], []
            side = "above" if a > b else "below"
            return [
                f"{label}: {name_a} ({shown(a, data_a)}) {relation} {name_b} ({shown(b, data_b)}), {gap_text} {side} "
                f"it (about {pct}% of {name_b}) {cites}."
            ], []
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
    if operation == "share":
        # (round 13, review-9 I1) "A相当于B的百分之多少": A ÷ B × 100, with the division written out
        if not b:
            return [], ["后者为零，无法计算占比。" if zh else "The base value is zero, so no share can be computed."]
        sentences = [_share_sentence(found[0], found[1], label, shown, cites, zh, request)]
        claimed = request.get("claimed") or {}
        if claimed.get("multiple"):
            sentences.append(_claimed_multiple_verdict(a / b, claimed, zh))
        return sentences, []
    if operation == "ratio":
        if not b:
            return [], ["后者为零，无法计算倍数。" if zh else "The second value is zero, so no ratio can be computed."]
        times = _num(round(a / b, 2))
        sentences = [
            f"{operands}，前者约为后者的 {times} 倍 {cites}。"
            if zh
            else f"{operands}: the former is about {times} times the latter {cites}."
        ]
        claimed = request.get("claimed") or {}
        if claimed.get("multiple"):
            sentences.append(_claimed_multiple_verdict(a / b, claimed, zh))
        return sentences, []
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
    side = f"{higher}更高" if zh else f"{higher} is higher"
    direction = request.get("direction")
    if direction == "fall" and a < 0 and b < 0:
        # (round 12) "多跌了多少" after two falls: the side that fell more
        lower = name_b if a > b else name_a
        side = f"{lower}跌得更多" if zh else f"{lower} fell more"
    elif direction == "rise" and a > 0 and b > 0:
        side = f"{higher}涨得更多" if zh else f"{higher} rose more"
    return [
        f"{operands}，两者相差 {gap_text}（{side}） {cites}。"
        if zh
        else f"{operands}: a difference of {gap_text} ({side}) {cites}."
    ], []


def _share_sentence(first: tuple, second: tuple, label: str, shown: Any, cites: str, zh: bool, request: dict) -> str:
    """(round 13, review-9 I1) "A相当于B的百分之多少": both values, the division and the share in percent (and in 成
    when the question counts in 成)."""
    (name_a, a, _ea, data_a), (name_b, b, _eb, data_b) = first, second
    pct = _num(round(a / b * 100, 2))
    division = "÷"  # written out in both languages: the verifier derives a share only next to a written division
    if zh:
        return (
            f"{name_a}{label} {shown(a, data_a)}，{name_b} {shown(b, data_b)}，{shown(a, data_a)} {division} "
            f"{shown(b, data_b)} ≈ {pct}%，即{name_a}约为{name_b}的 {pct}% {cites}。"
        )
    return (
        f"{name_a} {label} {shown(a, data_a)}, {name_b} {shown(b, data_b)}: {shown(a, data_a)} {division} "
        f"{shown(b, data_b)} ≈ {pct}%, so {name_a} is about {pct}% of {name_b} {cites}."
    )


def _many_operand_sentences(
    operation: str, found: list[tuple], label: str, kind: str, shown: Any, cites: str, zh: bool, request: dict
) -> list[str]:
    """(round 13, review-9 I3/I4) A total of two or more operands, the spread between the highest and the lowest,
    or a two-operand comparison asked of three or more targets (every pair, said as such). All values and the
    results are in one sentence, so the verifier re-derives each result from the cited values."""
    from itertools import combinations

    def gap_text(value: float) -> str:
        if kind == "points":
            return f"{_num(round(value, 2))} 个百分点" if zh else f"{_num(round(value, 2))} percentage points"
        if kind == "money":
            return _money(value, zh)
        if kind == "price":
            return _px(round(value, 3), found[0][3], zh)
        return _num(round(value, 2))

    values = [f"{name} {shown(value, data)}" for name, value, _eid, data in found]
    listing = ("，" if zh else ", ").join(values)
    if operation == "mean":
        # (round 15, review-11 K4) the mean of every operand, with every value next to it
        mean = sum(value for _name, value, _eid, _data in found) / len(found)
        if kind == "money":
            mean_text = _money(mean, zh)
        elif kind == "points":
            mean_text = f"{_num(round(mean, 2))}%"
        elif kind == "multiple":
            mean_text = _times(round(mean, 2), zh)
        elif kind == "price":
            mean_text = _px(round(mean, 3), found[0][3], zh)
        else:
            mean_text = _num(round(mean, 2))
        count = len(found)
        if zh:
            return [f"{label}：{listing}，{_GROUP_ZH.get(count, str(count))}者平均约 {mean_text} {cites}。"]
        group = _GROUP_EN.get(count, "operands")
        return [f"{label}: {listing}; the average of the {group} is about {mean_text} {cites}."]
    if operation == "total_vs_one":
        return [_total_vs_one_sentence(found, label, kind, shown, cites, zh, request)]
    if operation in {"sum", "share_total"}:
        total = sum(value for _name, value, _eid, _data in found)
        total_text = _money(total, zh) if kind == "money" else _num(round(total, 2))
        # (round 14, review-10 J13) the total is the sum of the exact values; when the rounded figures shown add up to
        # a different last digit, the answer says so instead of showing an arithmetic that does not add up
        note = ""
        if kind == "money" and _shown_sum_differs([value for _name, value, _eid, _data in found], total, zh):
            note = "（按未四舍五入的原值相加）" if zh else " (summed from the unrounded figures)"
        if operation == "sum":
            return [
                f"{label}：{listing}，合计 {total_text}{note} {cites}。"
                if zh
                else f"{label}: {listing}; combined {total_text}{note} {cites}."
            ]
        # (round 14, review-10 J3) a share of the combined total: A ÷ (A + B), with the total and the division written
        name_a, a, _ea, data_a = found[0]
        if not total:
            return []
        pct = _num(round(a / total * 100, 2))
        group = f"{_GROUP_ZH.get(len(found), str(len(found)))}者"
        if zh:
            return [
                f"{label}：{listing}，合计 {total_text}{note}，{shown(a, data_a)} ÷ {total_text} ≈ {pct}%，"
                f"即{name_a}约占{group}合计的 {pct}% {cites}。"
            ]
        return [
            f"{label}: {listing}; combined {total_text}{note}, and {shown(a, data_a)} ÷ {total_text} ≈ {pct}%, so "
            f"{name_a} is about {pct}% of the combined total {cites}."
        ]
    if operation == "spread":
        ordered = sorted(found, key=lambda item: item[1], reverse=True)
        (high, top, _e1, _d1), (low, bottom, _e2, _d2) = ordered[0], ordered[-1]
        spread = gap_text(top - bottom)
        if zh:
            return [f"{label}：{listing}，最高的{high}比最低的{low}高 {spread} {cites}。"]
        return [f"{label}: {listing}; the highest ({high}) is {spread} above the lowest ({low}) {cites}."]
    pairs = []
    # every pair from the higher value to the lower (ratios ≥ 1, relative differences "above"); a share the other way
    ranked = sorted(found, key=lambda item: item[1], reverse=True)
    ordered_pairs = [(low, high) if operation == "share" else (high, low) for high, low in combinations(ranked, 2)]
    for (name_a, a, _ea, data_a), (name_b, b, _eb, data_b) in ordered_pairs:
        if operation == "difference":
            pairs.append(
                f"{name_a}与{name_b}相差 {gap_text(abs(a - b))}"
                if zh
                else f"{name_a} vs {name_b}: {gap_text(abs(a - b))}"
            )
        elif operation == "ratio" and b:
            times = _num(round(a / b, 2))
            pairs.append(f"{name_a}约为{name_b}的 {times} 倍" if zh else f"{name_a} is {times} times {name_b}")
        elif operation == "relative" and b:
            pct = _num(round(abs(a - b) / abs(b) * 100, 2))
            side = ("高" if a > b else "低") if zh else ("above" if a > b else "below")
            pairs.append(f"{name_a}比{name_b}{side}约 {pct}%" if zh else f"{name_a} is about {pct}% {side} {name_b}")
        elif operation == "share" and b:
            pct = _num(round(a / b * 100, 2))
            pairs.append(
                f"{shown(a, data_a)} ÷ {shown(b, data_b)} ≈ {pct}%（{name_a}约为{name_b}的 {pct}%）"
                if zh
                else f"{shown(a, data_a)} ÷ {shown(b, data_b)} ≈ {pct}% ({name_a} is about {pct}% of {name_b})"
            )
    count = len(found)
    if zh:
        return [f"{label}：{listing}；问题涉及 {count} 个标的，逐对计算：{'，'.join(pairs)} {cites}。"]
    return [
        f"{label}: {listing}; the question names {count} targets, so each pair is computed: {'; '.join(pairs)} {cites}."
    ]


def _total_vs_one_sentence(
    found: list[tuple], label: str, kind: str, shown: Any, cites: str, zh: bool, request: dict
) -> str:
    """(round 15, review-11 K1) A total compared with another value: every operand, the total of the side with two or
    more operands (never including the other side), and the gap; a yes / no question gets the answer first."""
    split = int(request.get("split") or 1)
    left, right = found[:split], found[split:]

    def amount(value: float) -> str:
        return _money(value, zh) if kind == "money" else _num(round(value, 2))

    def side(items: list[tuple]) -> tuple[str, float]:
        total = sum(value for _name, value, _eid, _data in items)
        if len(items) == 1:
            name, value, _eid, data = items[0]
            return f"{name}（{shown(value, data)}）" if zh else f"{name} ({shown(value, data)})", total
        names = "、".join(item[0] for item in items) if zh else " and ".join(item[0] for item in items)
        return (f"{names}合计 {amount(total)}" if zh else f"{names} combined ({amount(total)})"), total

    listing = ("，" if zh else ", ").join(f"{name} {shown(value, data)}" for name, value, _eid, data in found)
    left_text, left_total = side(left)
    right_text, right_total = side(right)
    gap = amount(abs(left_total - right_total))
    more = left_total > right_total
    asked = request.get("asked")
    verdict = ""
    if asked in {"more", "less"} and left_total != right_total:
        yes = more if asked == "more" else not more
        verdict = ("是的，" if yes else "不是，") if zh else ("Yes: " if yes else "No: ")
    if zh:
        if left_total == right_total:
            return f"{label}：{listing}；{left_text}与{right_text}相同 {cites}。"
        return f"{verdict}{label}：{listing}；{left_text}，比{right_text}{'多' if more else '少'} {gap} {cites}。"
    if left_total == right_total:
        return f"{verdict}{label}: {listing}; {left_text} equals {right_text} {cites}."
    return f"{verdict}{label}: {listing}; {left_text} is {gap} {'more' if more else 'less'} than {right_text} {cites}."


def _display_scale(value: float, zh: bool) -> float:
    size = abs(value)
    if zh:
        return 1e8 if size >= 1e8 else 1e4 if size >= 1e4 else 1.0
    return 1e9 if size >= 1e9 else 1e6 if size >= 1e6 else 1.0


def _shown_sum_differs(values: list[float], total: float, zh: bool) -> bool:
    """Whether the operands as shown (two decimals at their scale) add up to another figure than the total as shown;
    only when every figure is shown at the same scale (otherwise the shown figures are not directly added)."""
    scale = _display_scale(total, zh)
    if any(_display_scale(value, zh) != scale for value in values):
        return False
    shown = round(sum(round(value / scale, 2) for value in values), 2)
    return abs(shown - round(total / scale, 2)) > 1e-9


def _claimed_multiple_verdict(ratio: float, claimed: dict[str, Any], zh: bool) -> str:
    """(round 13) Whether the multiple a question states holds ("大约是平安的两倍吗", "more than double …?"): yes or no
    for the bound asked (above, below, or about: within 5% for 大约 / about, 2% otherwise), then where the ratio lies.
    The multiple is written as the user wrote it."""
    multiple, written = float(claimed["multiple"]), str(claimed.get("written") or "")
    tolerance = 0.05 if claimed.get("approx") else 0.02
    close = abs(ratio - multiple) <= multiple * tolerance
    bound = claimed.get("bound") or "eq"
    yes = ratio > multiple if bound == "gt" else ratio < multiple if bound == "lt" else close
    if zh:
        times = written if written.endswith(("倍", "半", "成")) else f"{written}倍"
        relation = "大约是" if close else ("超过" if ratio > multiple else "不到")
        return f"{'是的' if yes else '不是'}，{relation}{times}。"
    times = written if re.search(r"[A-Za-z]$", written) and written.lower() != "x" else f"{written} times"
    if re.fullmatch(r"\d+(?:\.\d+)?\s?x", written, re.IGNORECASE):
        times = written
    relation = "about" if close else ("more than" if ratio > multiple else "less than")
    return f"So {'yes' if yes else 'no'}: it is {relation} {times}."


def _metric_pair_sentences(
    request: dict[str, Any], tool_log: list[dict[str, Any]], zh: bool
) -> tuple[list[str], list[str]]:
    """(round 13) Two metrics of one target ("两个率相差多少个点" after its gross and net margins): both values in one
    sentence (a derived net margin with its net profit and revenue) and their gap, ratio or relative difference."""
    from .frame import METRIC_BY_KEY, metric_label, operand_name, operand_value

    operands = list(request.get("operands") or [])[:2]
    operation = str(request.get("operation") or "difference")
    name = operand_name(operands[0], zh) if operands else ""
    found, missing = [], []
    for operand in operands:
        key = str(operand.get("metric") or request.get("metric") or "")
        hit = operand_value(tool_log, operand, key)
        if hit is None:
            missing.append(metric_label(key, zh))
        else:
            found.append((key, hit[0], hit[1], hit[2]))
    if missing or len(found) < 2:
        return [], [
            f"当前数据中没有{name}的{'、'.join(missing)}，因此无法计算两者的差值。"
            if zh
            else f"The data has no {' or '.join(missing)} for {name}, so the two cannot be compared."
        ]
    kind = METRIC_BY_KEY[found[0][0]].kind

    def shown(key: str, value: float, data: dict[str, Any]) -> str:
        if key == "net_margin":
            metrics = data.get("metrics") or {}
            division = "÷" if zh else "/"
            amounts = f"{_money(metrics.get('net_profit'), zh)} {division} {_money(metrics.get('revenue'), zh)}"
            return f"{amounts} ≈ {_num(value)}%"
        if kind == "points":
            return f"{_num(value)}%"
        if kind == "multiple":
            return _times(value, zh)
        if kind == "money":
            return _money(value, zh)
        return _px(value, data, zh) if kind == "price" else _num(value)

    labels = [metric_label(key, zh) for key, *_rest in found]
    (key_a, a, eid_a, data_a), (key_b, b, eid_b, data_b) = found
    cites = "".join(dict.fromkeys(f"[{eid}]" for eid in (eid_a, eid_b)))
    parts = (
        f"{name}{labels[0]} {shown(key_a, a, data_a)}，{labels[1]} {shown(key_b, b, data_b)}"
        if zh
        else f"{name} {labels[0]} {shown(key_a, a, data_a)}, {labels[1]} {shown(key_b, b, data_b)}"
    )
    derived = "net_margin" in {key_a, key_b}
    if operation in {"ratio", "relative"} and not derived and b:
        if operation == "ratio":
            times = _num(round(a / b, 2))
            return [
                f"{parts}，前者约为后者的 {times} 倍 {cites}。"
                if zh
                else f"{parts}: the former is about {times} times the latter {cites}."
            ], []
        pct = _num(round(abs(a - b) / abs(b) * 100, 2))
        side = ("低" if a < b else "高") if zh else ("below" if a < b else "above")
        return [
            f"{parts}，前者比后者{side}约 {pct}% {cites}。"
            if zh
            else f"{parts}: the former is about {pct}% {side} the latter {cites}."
        ], []
    gap = round(abs(a - b), 2)
    if kind == "points":
        gap_text = f"{_num(gap)} 个百分点" if zh else f"{_num(gap)} percentage points"
    elif kind == "money":
        gap_text = _money(abs(a - b), zh)
    else:
        gap_text = _num(gap)
    higher = labels[0] if a > b else labels[1]
    if a == b:
        return [f"{parts}，两者相同 {cites}。" if zh else f"{parts}: they are equal {cites}."], []
    sentence = (
        f"{parts}，两者相差 {gap_text}（{higher}更高） {cites}。"
        if zh
        else f"{parts}: a difference of {gap_text} ({higher} is higher) {cites}."
    )
    notes = []
    if operation in {"ratio", "relative", "share"}:
        notes.append(
            "净利率本身是推算值，这里给出两者相差的百分点，不再计算其比值。"
            if zh
            else "The net margin is a derived figure, so the gap in percentage points is given instead of a ratio."
        )
    return [sentence], notes


def frame_values(request: dict[str, Any], tool_log: list[dict[str, Any]]) -> list[float]:
    """(round 15, review-11 K3) Every number the comparison frame computes for a request (and for the further
    operations of the same message): the result, and the totals it states on the way (the combined total of a share
    of the total or of a mean, each side of a total compared with another value). The verifier accepts these in a cited
    sentence, so a correct computation is never deleted."""
    from .frame import operand_value

    values: list[float] = []
    for item in [request, *(request.get("also") or [])] if request else []:
        result = frame_result(item, tool_log)
        if result is not None:
            values.append(result)
        operation = str(item.get("operation") or "")
        if operation not in {"sum", "share_total", "mean", "total_vs_one"}:
            continue
        found = []
        for operand in item.get("operands") or []:
            hit = operand_value(tool_log, operand, str(operand.get("metric") or item.get("metric") or ""))
            if hit is None:
                break
            found.append(float(hit[0]))
        else:
            if len(found) >= 2:
                values.append(sum(found))
                if operation == "total_vs_one":
                    split = int(item.get("split") or 1)
                    values += [sum(found[:split]), sum(found[split:])]
    return [value for value in dict.fromkeys(values) if value]


def frame_result(request: dict[str, Any], tool_log: list[dict[str, Any]]) -> float | None:
    """The number a frame request computes (gap, ratio or percent), for checking whether an LLM draft states it."""
    from .frame import operand_value

    key, operation = str(request.get("metric") or ""), str(request.get("operation") or "difference")
    values = []
    many = operation in {"sum", "spread", "share_total", "mean", "total_vs_one"}
    for operand in (request.get("operands") or [])[: None if many else 2]:
        hit = operand_value(tool_log, operand, str(operand.get("metric") or key))
        if hit is None or hit[2].get("_computed_change"):
            return None
        values.append(hit[0])
    if len(values) < 2 or operation == "which" or (request.get("pairwise") and not many):
        return None
    if operation == "sum":
        return round(sum(values), 2)
    if operation == "mean":
        return round(sum(values) / len(values), 2)
    if operation == "total_vs_one":
        split = int(request.get("split") or 1)
        return round(abs(sum(values[:split]) - sum(values[split:])), 2)
    if operation == "spread":
        return round(max(values) - min(values), 2)
    if operation == "share_total":
        return round(values[0] / sum(values) * 100, 2) if sum(values) else None
    a, b = values[:2]
    if operation == "share":
        return round(a / b * 100, 2) if b else None
    if any(operand.get("metric") for operand in request.get("operands") or []):
        key = "net_margin" if "net_margin" in {o.get("metric") for o in request["operands"]} else key
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


# Metrics priced on a trading day (market cap, P/S, dividend yield): dated by the valuation date, not the report period.
_VALUATION_DATED = {"total_mv", "market_cap", "circ_mv", "total_market_cap", "ps_ttm", "ps"}
# Fields whose label differs from their metric's ("流通市值" is not the total market cap).
_FIELD_LABELS = {"circ_mv": ("流通市值", "free-float market cap")}


def _extra_metrics(data: dict[str, Any], keys: set[str], zh: bool) -> list[str]:
    """Requested metrics beyond the standard snapshot (e.g. a live source's dividend yield), when present.

    One value per metric (its first reported field: ``total_mv`` before ``circ_mv``), labelled by the most specific
    metric (``revenue_yoy`` is the revenue growth, not a bare "growth"); market cap and P/S are dated by the
    valuation date when the data has one, statement metrics by the report period."""
    metrics = data.get("metrics") or {}
    eid, name, period = data.get("evidence_id"), data.get("name"), data.get("report_date")
    if not eid:
        return []
    chosen: dict[str, str] = {}
    for key in sorted(keys):
        metric = EXTRA_METRIC_FIELDS[key]
        present = [field for field in metric.fields if field in keys and metrics.get(field) is not None]
        if present and metric.key not in chosen:
            chosen[metric.key] = present[0]
    valuation_date = data.get("valuation_date")
    dated: list[str] = []
    periodic: list[str] = []
    for metric in (item for item in METRICS if item.key in chosen):
        key = chosen[metric.key]
        label_zh, label_en = _FIELD_LABELS.get(key, (metric.zh, metric.en))
        label = basis_label(data, key, zh, label_zh if zh else label_en)
        part = f"{label} {_metric_value(key, metrics[key], data, zh)}"
        (dated if valuation_date and key in _VALUATION_DATED else periodic).append(part)
    sentences = []
    if periodic:
        sentences.append(
            f"{name}（报告期 {period}）：{'，'.join(periodic)} [{eid}]。"
            if zh
            else f"{name} (period {period}): {', '.join(periodic)} [{eid}]."
        )
    if dated:
        sentences.append(
            f"{name}（按 {valuation_date} 收盘计算）：{'，'.join(dated)} [{eid}]。"
            if zh
            else f"{name} (at the {valuation_date} close): {', '.join(dated)} [{eid}]."
        )
    return sentences


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
        # (round 14, review-10 J13) the daily change to two decimals, as the KPI tile shows it
        change = f"，当日涨跌幅 {_pct2(pct)}%" if pct is not None else ""
        if computed:
            previous, change_pct = computed
            change = (
                f"，按前一交易日收盘 {_px(previous, data, zh)} 计算的当日涨跌幅约 {_num(change_pct)}%"
                "（数据源未提供涨跌幅）"
            )
        sentences = [f"{name}（{symbol}）最新可用收盘价为 {_px(close, data, zh)}（{as_of}）{change} [{eid}]。"]
    else:
        change = f", daily change {_pct2(pct)}%" if pct is not None else ""
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
        sentences.extend(_year_to_date(data, zh, first_close_base=request.ytd_first_close))
    if request is not None and request.range_52w:
        sentences.extend(_range_52w(data, zh))
        sentences.extend(_from_52w_extremes(data, zh, request))
    if request is not None and request.drawdown:
        sentences.extend(_drawdown(data, request.drawdown, zh))
    return sentences


def _range_52w(data: dict[str, Any], zh: bool) -> list[str]:
    """(round 12) The 52-week high and low close with their dates and the window; without ``range_52w`` (history
    shorter than 52 weeks) the gap is stated by coverage."""
    found = data.get("range_52w") or {}
    if found.get("high") is None or found.get("low") is None:
        return []
    name, eid = data.get("name"), data.get("evidence_id")
    start, end, count = found.get("start_date"), found.get("end_date"), found.get("closes")
    if zh:
        return [
            f"{name}近52周（{start} 至 {end}，{count} 个交易日，按未复权收盘价）：最高收盘 "
            f"{_px(found['high'], data, zh)}（{found.get('high_date')}），最低收盘 {_px(found['low'], data, zh)}"
            f"（{found.get('low_date')}） [{eid}]。"
        ]
    return [
        f"{name} over the 52 weeks from {start} to {end} ({count} sessions, unadjusted closes): highest close "
        f"{_px(found['high'], data, zh)} on {found.get('high_date')}, lowest close {_px(found['low'], data, zh)} on "
        f"{found.get('low_date')} [{eid}]."
    ]


def _from_52w_extremes(data: dict[str, Any], zh: bool, request: PriceRequest) -> list[str]:
    """(round 14, review-10 J4) The latest close against the 52-week low ("反弹了多少") or high ("较高点跌了多少"),
    with both closes and the division written out so the verifier and the reader can recompute it."""
    found = data.get("range_52w") or {}
    close, name, eid = data.get("close"), data.get("name"), data.get("evidence_id")
    if close is None or not eid:
        return []
    sentences = []
    low, high = found.get("low"), found.get("high")
    if request.rebound_52w and low:
        pct = _num(round((float(close) - float(low)) / float(low) * 100, 2))
        # (round 16) the distance in yuan as well as in percent ("比一年来的最低收盘价高出多少元")
        gap = _num(round(float(close) - float(low), 3))
        if zh:
            sentences.append(
                f"{name}最新收盘 {_px(close, data, zh)}较近52周最低收盘 {_px(low, data, zh)}（{found.get('low_date')}）"
                f"{'反弹' if float(close) >= float(low) else '下跌'} {_num(close)} − {_num(low)} = {gap} 元，即 "
                f"({_num(close)} − {_num(low)}) ÷ {_num(low)} ≈ {pct}% [{eid}]。"
            )
        else:
            sentences.append(
                f"{name}'s latest close of {_px(close, data, zh)} is {_num(close)} − {_num(low)} = CNY {gap}, or "
                f"({_num(close)} − {_num(low)}) ÷ {_num(low)} ≈ {pct}%, above its 52-week low close of "
                f"{_px(low, data, zh)} on {found.get('low_date')} [{eid}]."
            )
    if request.back_to_high_52w and high and float(close):
        # (round 15, review-11 K4) the rise from the latest close back to the 52-week high, in percent of the close
        pct = _num(round((float(high) - float(close)) / float(close) * 100, 2))
        if zh:
            sentences.append(
                f"{name}最新收盘 {_px(close, data, zh)}，要回到近52周最高收盘 {_px(high, data, zh)}"
                f"（{found.get('high_date')}）需上涨 {_num(high)} − {_num(close)} = "
                f"{_num(round(float(high) - float(close), 3))} 元，即 ({_num(high)} − {_num(close)}) ÷ {_num(close)} ≈ "
                f"{pct}% [{eid}]。"
            )
        else:
            sentences.append(
                f"From its latest close of {_px(close, data, zh)}, {name} would need to rise {_num(high)} − "
                f"{_num(close)} = CNY {_num(round(float(high) - float(close), 3))}, or ({_num(high)} − "
                f"{_num(close)}) ÷ {_num(close)} ≈ {pct}%, to get back to its 52-week high close of "
                f"{_px(high, data, zh)} on {found.get('high_date')} [{eid}]."
            )
    if request.off_high_52w and high:
        pct = _num(round((float(high) - float(close)) / float(high) * 100, 2))
        # (round 16, review-12 L13) the gap in yuan as well as in percent ("还差多少元")
        gap = _num(round(float(high) - float(close), 3))
        if zh:
            sentences.append(
                f"{name}最新收盘 {_px(close, data, zh)}较近52周最高收盘 {_px(high, data, zh)}"
                f"（{found.get('high_date')}）低 {_num(high)} − {_num(close)} = {gap} 元，即 ({_num(high)} − "
                f"{_num(close)}) ÷ {_num(high)} ≈ {pct}% [{eid}]。"
            )
        else:
            sentences.append(
                f"{name}'s latest close of {_px(close, data, zh)} is {_num(high)} − {_num(close)} = CNY {gap}, or "
                f"({_num(high)} − {_num(close)}) ÷ {_num(high)} ≈ {pct}%, below its 52-week high close of "
                f"{_px(high, data, zh)} on {found.get('high_date')} [{eid}]."
            )
    return sentences


# (round 16) the last session of a stated year
_YEAR_LAST_SESSION = re.compile(
    r"((?:19|20)\d{2})\s*年(?:的)?(?:最后一个交易日|最后一天|最后一个工作日|收官日|年末|年底)|"
    r"\blast (?:trading day|session) of ((?:19|20)\d{2})\b|"
    r"\b((?:19|20)\d{2})['’]?s? (?:last|final) (?:trading day|session)\b",
    re.IGNORECASE,
)
_CLOSE_ASKED = re.compile(r"收盘|收于|报收|股价|价格|\bclos(?:e|ed|ing)\b|\bprice\b|\btrad(?:ed|ing) at\b", re.I)


def dated_closes(data: dict[str, Any]) -> dict[str, float]:
    """(round 15, review-11 K8) Every dated close the price payload carries: the recent closes, the year's first
    close, the previous year's last close, the 52-week high and low, and the drawdown peaks and troughs."""
    found: dict[str, float] = {}

    def put(day: Any, close: Any) -> None:
        if day and isinstance(close, int | float):
            found.setdefault(str(day)[:10], float(close))

    for row in data.get("recent_closes") or []:
        if isinstance(row, dict):
            put(row.get("date"), row.get("close"))
    for key in ("year_start", "prior_year_end"):
        row = data.get(key) or {}
        put(row.get("date"), row.get("close"))
    extremes = data.get("range_52w") or {}
    put(extremes.get("high_date"), extremes.get("high"))
    put(extremes.get("low_date"), extremes.get("low"))
    for key in ("max_drawdown_1y", "max_drawdown_ytd"):
        row = data.get(key) or {}
        put(row.get("peak_date"), row.get("peak"))
        put(row.get("trough_date"), row.get("trough"))
    if data.get("as_of") and data.get("close") is not None:
        put(data.get("as_of"), data.get("close"))
    return found


def dated_close_sentences(query: str, tool_log: list[dict[str, Any]], zh: bool) -> tuple[list[str], list[str]]:
    """(round 15, review-11 K8) A close asked for a stated day ("招商银行2025年12月31日收盘价"): the close of that day
    when the payload carries it (``dated_closes``), with its date; a day after the data's last bar is said to be
    after the data (a future day is also hedged as a forecast by compliance); a day the payload does not carry is
    named missing. ``(facts, gaps)``."""
    from datetime import date

    from .stated_dates import stated_periods

    if not _CLOSE_ASKED.search(query or ""):
        return [], []
    facts: list[str] = []
    gaps: list[str] = []
    for entry in tool_log:
        data = entry.get("data") or {}
        if entry.get("tool") != "get_price_history" or not entry.get("ok") or not data.get("as_of"):
            continue
        try:
            as_of = date.fromisoformat(str(data["as_of"])[:10])
        except ValueError:
            continue
        days = [period for period in stated_periods(query, as_of) if period.kind == "day"]
        last = _YEAR_LAST_SESSION.search(query or "")
        prior = data.get("prior_year_end") or {}
        name, eid = data.get("name") or data.get("symbol"), data.get("evidence_id")
        if last is not None and not days:
            # (round 16) "2025年最后一个交易日收盘多少": the previous year's last close the payload carries
            year = next(group for group in last.groups() if group)
            if str(prior.get("date") or "").startswith(year) and prior.get("close") is not None and eid:
                value, day = _px(prior["close"], data, zh), str(prior["date"])[:10]
                facts.append(
                    f"{name} {year}年最后一个交易日（{day}）的收盘价为 {value} [{eid}]。"
                    if zh
                    else f"{name} closed at {value} on {day}, the last trading day of {year} [{eid}]."
                )
            break
        if not days:
            continue
        closes = dated_closes(data)
        for period in days:
            day = period.start.isoformat()
            if day in closes and eid and day != str(data["as_of"])[:10]:
                value = _px(closes[day], data, zh)
                sentence = f"{name} {day} 的收盘价为 {value} [{eid}]。"
                facts.append(sentence if zh else f"{name} closed at {value} on {day} [{eid}].")
            elif period.start > as_of:
                gaps.append(
                    f"当前数据截至 {as_of.isoformat()}，没有{name} {day} 的收盘价；以下是最新可用收盘价。"
                    if zh
                    else f"The data ends on {as_of.isoformat()}, so there is no close for {name} on {day}; the latest "
                    "available close is given below."
                )
            elif day not in closes:
                gaps.append(
                    f"当前数据中没有{name} {day} 的收盘价（可能不是交易日，或不在数据提供的日期中）；"
                    "以下是最新可用收盘价。"
                    if zh
                    else f"The data has no close for {name} on {day} (not a trading day, or not among the dates the "
                    "data carries); the latest available close is given below."
                )
        break
    return facts, gaps


def cost_basis_sentences(tool_log: list[dict[str, Any]], query: str, zh: bool) -> list[str]:
    """(round 14, review-10 J4) Gain or loss against the cost price the question states ("我300股成本90元，浮亏多
    少"): (close − cost) per share and in percent of the cost, times the shares when a count is stated, at the latest
    close of the one target asked about; the cost is attributed to the user. Not advice, before fees and dividends."""
    from .coverage import cost_basis_request

    asked = cost_basis_request(query)
    if asked is None:
        return []
    prices = [
        entry.get("data") or {}
        for entry in tool_log
        if entry.get("tool") == "get_price_history" and entry.get("ok") and (entry.get("data") or {}).get("close")
    ]
    if len({str(data.get("symbol")) for data in prices}) != 1:
        return []  # no close, or several targets: which position the cost belongs to is not known
    data = prices[0]
    close, eid, name, as_of = float(data["close"]), data.get("evidence_id"), data.get("name"), data.get("as_of")
    if not eid:
        return []
    if asked.get("lots_detail"):
        return _lots_sentences(asked["lots_detail"], close, data, zh)
    cost = float(asked["cost"])
    diff = round(close - cost, 4)
    pct = _num(round(diff / cost * 100, 2))
    per_share = _num(round(diff, 2))
    # (round 15, review-11 K17) a negative factor is bracketed in the product: "200 × (−90.5)", not "200 × -90.5"
    factor = f"(−{per_share.lstrip('-')})" if per_share.startswith("-") else per_share
    gain = diff >= 0
    shares, lots = asked.get("shares"), asked.get("lots")
    if zh:
        word = "浮盈" if gain else "浮亏"
        sentence = (
            f"按 {as_of} 收盘价 {_px(close, data, zh)}和你给出的成本价 {_num(cost)} 元计算，{name}每股{word} "
            f"{_num(close)} − {_num(cost)} = {per_share} 元（{pct}%）"
        )
        if shares:
            held = f"{lots} 手（{shares} 股）" if lots else f"{shares} 股"
            total = _num(round(shares * round(diff, 2), 2))
            sentence += f"，{held}合计 {shares} × {factor} = {total} 元"
        return [
            f"{sentence} [{eid}]。",
            "这是按最近收盘价计算的账面盈亏，未含佣金、印花税等交易费用和分红，不是卖出建议或投资建议。",
        ]
    word = "gain" if gain else "loss"
    sentence = (
        f"At the {as_of} close of {_px(close, data, zh)} and your stated cost of CNY {_num(cost)}, {name} shows a "
        f"paper {word} of {_num(close)} − {_num(cost)} = CNY {per_share} per share ({pct}%)"
    )
    if shares:
        held = f"{lots} lots ({shares} shares)" if lots else f"{shares} shares"
        total = _num(round(shares * round(diff, 2), 2))
        sentence += f", or {shares} × {factor} = CNY {total} on {held}"
    return [
        f"{sentence} [{eid}].",
        "This is a paper gain or loss at the last close before fees, taxes and dividends, not a recommendation to sell "
        "or investment advice.",
    ]


def _lots_sentences(lots: list[dict[str, Any]], close: float, data: dict[str, Any], zh: bool) -> list[str]:
    """(round 15, review-11 K2) Several purchases at different costs: each lot's shares × (close − cost), their sum,
    and the average cost; the costs are the user's, the close is the latest one."""
    eid, name, as_of = data.get("evidence_id"), data.get("name"), data.get("as_of")
    shares = sum(int(lot["shares"]) for lot in lots)
    paid = sum(int(lot["shares"]) * float(lot["cost"]) for lot in lots)
    gains = [round(int(lot["shares"]) * (close - float(lot["cost"])), 2) for lot in lots]
    total = round(sum(gains), 2)
    average = round(paid / shares, 4)
    pct = _num(round(total / paid * 100, 2)) if paid else "0"
    if zh:
        parts = [
            f"{lot['shares']} 股 × ({_num(close)} − {_num(lot['cost'])}) = {_num(gain)} 元"
            for lot, gain in zip(lots, gains, strict=True)
        ]
        word = "浮盈" if total >= 0 else "浮亏"
        return [
            f"按 {as_of} 收盘价 {_px(close, data, zh)}和你给出的{_GROUP_ZH.get(len(lots), '各')}笔买入成本"
            f"分别计算，{name}："
            f"{'；'.join(parts)}；合计{word} {_num(total)} 元（{shares} 股，你给出的平均成本 "
            f"{_num(round(average, 3))} 元，约 {pct}%） [{eid}]。",
            "这是按最近收盘价计算的账面盈亏，未含佣金、印花税等交易费用和分红，不是卖出建议或投资建议。",
        ]
    parts = [
        f"{lot['shares']} × ({_num(close)} − {_num(lot['cost'])}) = CNY {_num(gain)}"
        for lot, gain in zip(lots, gains, strict=True)
    ]
    word = "gain" if total >= 0 else "loss"
    return [
        f"At the {as_of} close of {_px(close, data, zh)} and your {_GROUP_EN.get(len(lots), 'several')} stated "
        f"purchase costs, {name}: "
        f"{'; '.join(parts)}; a combined paper {word} of CNY {_num(total)} on {shares} shares (your stated average "
        f"cost CNY {_num(round(average, 3))}, about {pct}%) [{eid}].",
        "This is a paper gain or loss at the last close before fees, taxes and dividends, not a recommendation to sell "
        "or investment advice.",
    ]


def _drawdown(data: dict[str, Any], window: str, zh: bool) -> list[str]:
    """(round 12) The maximum drawdown of the close over the asked window, with the peak and the trough (date and
    close) in the sentence so the reader and the verifier can recompute it; a missing window is stated by coverage."""
    found = data.get(f"max_drawdown_{window}") or {}
    if found.get("max_drawdown_pct") is None:
        return []
    name, eid = data.get("name"), data.get("evidence_id")
    start, end, pct = found.get("start_date"), found.get("end_date"), _num(found["max_drawdown_pct"])
    if zh:
        span = "今年以来" if window == "ytd" else "近52周"
        if not found["max_drawdown_pct"]:
            return [f"{name}{span}（{start} 至 {end}，按未复权收盘价）收盘价没有低于此前高点，最大回撤为 0% [{eid}]。"]
        return [
            f"{name}{span}（{start} 至 {end}，按未复权收盘价）最大回撤 {pct}%：从 {found.get('peak_date')} 收盘 "
            f"{_px(found['peak'], data, zh)} 回落至 {found.get('trough_date')} 收盘 {_px(found['trough'], data, zh)} "
            f"[{eid}]。"
        ]
    span = "year to date" if window == "ytd" else "over the past 52 weeks"
    if not found["max_drawdown_pct"]:
        return [f"{name} {span} ({start} to {end}, unadjusted closes) never closed below an earlier high: a maximum "
                f"drawdown of 0% [{eid}]."]  # fmt: skip
    return [
        f"{name} {span} ({start} to {end}, unadjusted closes): maximum drawdown {pct}%, from a close of "
        f"{_px(found['peak'], data, zh)} on {found.get('peak_date')} to {_px(found['trough'], data, zh)} on "
        f"{found.get('trough_date')} [{eid}]."
    ]


def _implied_change(data: dict[str, Any]) -> tuple[float, float, float] | None:
    """``(implied previous close, change in yuan, reported change in %)`` from the latest close and the reported daily
    change (round 13), rounded to the price's precision; ``None`` without both."""
    close, pct = data.get("close"), data.get("pct_change_1d")
    if close is None or pct is None or float(pct) <= -100 or not data.get("evidence_id"):
        return None
    previous = float(close) / (1 + float(pct) / 100)
    digits = 3 if abs(float(close)) < 10 else 2
    return round(previous, digits), round(float(close) - previous, digits), float(pct)


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


def _year_to_date(data: dict[str, Any], zh: bool, *, first_close_base: bool = False) -> list[str]:
    """The year-to-date change, with both closes in the sentence (the verifier checks the percent change against
    them); without ``year_start`` the gap is stated by coverage.

    (round 13, I11) The base is the previous year's last close (``prior_year_end``, the 年初至今 basis of quote
    vendors); a payload recorded before round 13 has only the year's first close, and the sentence then says that
    this is its base. (round 14) When the user names the year's first close as the base, that base is used."""
    start, close = data.get("year_start") or {}, data.get("close")
    base = {} if first_close_base else data.get("prior_year_end") or {}
    if not start or start.get("close") in (None, 0) or close is None:
        return []
    name, eid, as_of = data.get("name"), data.get("evidence_id"), data.get("as_of")
    if base and base.get("close") not in (None, 0):
        change = (float(close) - float(base["close"])) / abs(float(base["close"])) * 100
        if zh:
            return [
                f"{name}今年以来（以上年最后一个交易日收盘为基准）：{base.get('date')} 收盘 "
                f"{_px(base['close'], data, zh)}，最新（{as_of}）收盘 {_px(close, data, zh)}，"
                f"涨跌幅 {_num(round(change, 2))}% [{eid}]。"
            ]
        return [
            f"{name} year to date (from the previous year's last close): {base.get('date')} close "
            f"{_px(base['close'], data, zh)}, latest close ({as_of}) {_px(close, data, zh)}, a change of "
            f"{_num(round(change, 2))}% [{eid}]."
        ]
    change = (float(close) - float(start["close"])) / abs(float(start["close"])) * 100
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
    if request.turnover_rate:
        rate = data.get("turnover_rate")
        if rate is None:
            missing.append("换手率（需要流通股本，数据源未提供）" if zh else "the turnover rate (needs the free float)")
        else:
            stated.append(f"换手率 {_num(rate)}%" if zh else f"turnover rate {_num(rate)}%")
    if request.change_amount:
        if len(closes) >= 2 and closes[-1].get("close") is not None:
            previous, latest = closes[-2], closes[-1]
            delta = round(float(latest["close"]) - float(previous["close"]), 3)
            if zh:
                move = f"{'上涨' if delta > 0 else '下跌' if delta < 0 else '持平'} {_px(abs(delta), data, zh)}"
                sentences.append(
                    f"{name}最新收盘 {_px(latest['close'], data, zh)}（{latest.get('date')}），前一交易日"
                    f"（{previous.get('date')}）收盘 {_px(previous['close'], data, zh)}，{move} [{eid}]。"
                )
            else:
                move = f"{'up' if delta > 0 else 'down' if delta < 0 else 'unchanged'} {_px(abs(delta), data, zh)}"
                sentences.append(
                    f"{name} closed at {_px(latest['close'], data, zh)} on {latest.get('date')} against "
                    f"{_px(previous['close'], data, zh)} on {previous.get('date')}, {move} [{eid}]."
                )
        elif (implied := _implied_change(data)) is not None:
            # (round 13) "茅台今天每股跌了多少元" with the close and the reported % change but no previous close: the
            # previous close implied by them (close ÷ (1 + change)) and the change in yuan, with both operands and the
            # formula in the sentence and labelled as derived
            previous_close, delta, pct = implied
            sign = "−" if pct < 0 else "+"
            close = data.get("close")
            if zh:
                move = f"{'上涨' if delta > 0 else '下跌' if delta < 0 else '持平'}约 {_px(abs(delta), data, zh)}"
                sentences.append(
                    f"{name}最新收盘 {_px(close, data, zh)}（{as_of}），当日涨跌幅 {_num(pct)}%，"
                    f"据此推算前一交易日收盘约 {_num(close)} ÷ (1 {sign} {_num(abs(pct))}%) ≈ "
                    f"{_px(previous_close, data, zh)}，每股{move}"
                    f"（推算值，数据源未直接提供涨跌金额） [{eid}]。"
                )
            else:
                move = (
                    f"{'up' if delta > 0 else 'down' if delta < 0 else 'unchanged'} about {_px(abs(delta), data, zh)}"
                )
                sentences.append(
                    f"{name} closed at {_px(close, data, zh)} on {as_of} with a daily change of {_num(pct)}%, which "
                    f"implies a previous close of about {_num(close)} / (1 {sign} {_num(abs(pct))}%) ≈ "
                    f"{_px(previous_close, data, zh)}, so it was {move} per share (derived; the source states no "
                    f"change in yuan) [{eid}]."
                )
        else:
            # no previous close and no % change in the data: the change in yuan is named missing; the reported
            # daily change in percent stays in the price sentence
            missing.append(
                "前一交易日收盘价，因此无法给出涨跌金额（涨跌幅见上）"
                if zh
                else "the previous close, so the change in yuan cannot be given (the daily change in percent is above)"
            )
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


_INDUSTRY_METRIC_WORDS = (
    ("市盈率", "P/E", re.compile(r"市盈率|(?<![A-Za-z])P/?E(?![A-Za-z])|price[- ]to[- ]earnings", re.I)),
    ("市净率", "P/B", re.compile(r"市净率|(?<![A-Za-z])P/?B(?![A-Za-z])|price[- ]to[- ]book", re.I)),
    ("ROE", "ROE", re.compile(r"净资产收益率|(?<![A-Za-z])ROE(?![A-Za-z])|return on equity", re.I)),
)


def _missing_industry_snapshot(query: str, tool_log: list[dict[str, Any]], zh: bool) -> list[str]:
    """(round 13, I10) A question about a company's industry average when the data has no industry snapshot for
    the company (the 2026-09-30 snapshot extension carries none): say so, instead of answering with the company's
    own figures only. ``industry_gaps`` covers an industry snapshot that lacks the asked metric."""
    if not asks_about_industry(query):
        return []
    asked = [(zh_label, en_label) for zh_label, en_label, pattern in _INDUSTRY_METRIC_WORDS if pattern.search(query)]
    sentences: list[str] = []
    for entry in tool_log:
        if entry.get("tool") != "get_fundamentals" or not entry.get("ok"):
            continue
        data = entry.get("data") or {}
        if data.get("industry") or not data.get("symbol") or data.get("product_type") not in {None, "stock"}:
            continue
        name = str(data.get("name") or data.get("symbol"))
        if zh:
            what = "、".join(label for label, _en in asked) or "估值指标"
            sentences.append(f"当前数据没有{name}所属行业的平均{what}，无法与行业平均比较。")
        else:
            what = ", ".join(en for _zh, en in asked) or "valuation multiples"
            sentences.append(
                f"The industry average {what} of {name}'s industry is not available in the current data, so no "
                "comparison with the industry average can be made."
            )
    return list(dict.fromkeys(sentences))


def _fundamentals(
    data: dict[str, Any], zh: bool, *, industry_first: bool = False, turnover_rate: bool = False
) -> list[str]:
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
        parts.append(f"{basis_label(data, 'roe', zh, 'ROE')} {_num(roe_pct)}%")
    for key, label_zh, label_en in (("revenue", "营业收入", "revenue"), ("net_profit", "净利润", "net profit")):
        value = metrics.get(key)
        if value is not None and abs(float(value)) >= 1e6:
            parts.append(f"{basis_label(data, key, zh, label_zh if zh else label_en)} {_money(value, zh)}")
    if parts and eid:
        label = data.get("period")
        # (round 12) P/E and P/B are priced on the valuation date, not at the report period: say which day
        valued = data.get("valuation_date")
        priced = valued and valued != period and (metrics.get("pe_ttm") is not None or metrics.get("pb") is not None)
        if zh:
            report = f"{label[2:]}年年报" if label and label.startswith("FY") else label
            suffix = f"，{report}" if report else ""
            suffix += f"；PE、PB 按 {valued} 收盘计算" if priced else ""
            sentences.append(f"{name} 基本面（报告期 {period}{suffix}）：{'，'.join(parts)} [{eid}]。")
        else:
            suffix = f", {label}" if label else ""
            suffix += f"; P/E and P/B at the {valued} close" if priced else ""
            sentences.append(f"{name} fundamentals (period {period}{suffix}): {', '.join(parts)} [{eid}].")
    industry = data.get("industry") or {}
    industry_metrics = industry.get("metrics") or {}
    industry_fields = [("pe", "PE"), ("pb", "PB"), ("pct_change", "涨跌幅" if zh else "change")]
    if turnover_rate and industry_metrics.get("turnover"):
        # (round 13) "白酒行业今天换手率多少": the industry snapshot's turnover is a rate in percent (tools/units.py
        # declares it "%"); 0 is a row whose source reports none
        industry_fields.append(("turnover", "换手率" if zh else "turnover rate"))
    industry_parts = [
        f"{label} {_times(industry_metrics[key], zh) if key in {'pe', 'pb'} else _num(industry_metrics[key]) + '%'}"
        for key, label in industry_fields
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


# (round 13, I16) What a document is about, in FinSight's own words: a fixed set of neutral topics, matched on a title
# that passes the headline shape check (``text_safety.safe_headline``, the same check the evidence list applies). The
# answer names at most two topics and never the title or any of its words, so a planted headline can change at most
# which of these labels is shown; a title that fails the check gives no topic at all. The labels avoid event words
# ("回购", "立案", "退市"): a label must not restate a claim a planted document could make.
_DOCUMENT_TOPICS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "公司治理",
        "corporate governance",
        re.compile(
            r"股东大会|股东会|董事会|监事会|董事|监事|高管|董事长|总经理|总裁|独立董事|任命|聘任|辞职|离任|换届|章程|"
            r"\b(?:board|chairman|chairwoman|ceo|cfo|director|appoint\w*|resign\w*|governance)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "财务业绩",
        "financial results",
        re.compile(
            r"业绩|财报|年报|半年报|季报|年度报告|季度报告|中期报告|营收|营业收入|收入|净利|净赚|利润|盈利|扭亏|亏损|毛利|"
            r"\b(?:earnings|results|revenue|profit|annual\s+report|interim\s+report|quarterly)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "股东回报",
        "shareholder payouts",
        re.compile(r"分红|派息|派现|每股派|红利|股息|回购|\b(?:dividends?|buybacks?|repurchases?)\b", re.IGNORECASE),
    ),
    (
        "股权变动",
        "share ownership",
        re.compile(
            r"增持|减持|持股|股权|质押|解禁|定增|配股|股本|\b(?:stake|shareholding|placement|share\s+issue)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "业务经营",
        "business operations",
        re.compile(
            r"产品|业务|销量|销售|渠道|批价|产能|产量|项目|合作|签约|中标|订单|并购|收购|扩产|新品|门店|出口|出海|海外|研发|"
            r"\b(?:products?|sales|orders?|contracts?|acquisitions?|mergers?|partnerships?|launch\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "监管合规",
        "regulatory matters",
        re.compile(
            r"证监会|监管|问询|处罚|罚款|警示|合规|诉讼|仲裁|\b(?:regulat\w*|csrc|inquiry|penalt\w*|lawsuit)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "宏观政策",
        "macro policy",
        re.compile(
            r"政策|央行|利率|降准|降息|经济|GDP|CPI|PMI|\b(?:policy|economy|economic|inflation)\b", re.IGNORECASE
        ),
    ),
    (
        "市场行情",
        "market moves",
        re.compile(
            r"行业|板块|市场|指数|大盘|行情|主力资金|资金[^，。,]{0,4}(?:流入|流出)|北向|市值|股价|估值|ETF|"
            r"\b(?:sector|industry|market|index|valuation|shares?\s+(?:rose|fell))\b",
            re.IGNORECASE,
        ),
    ),
)
_MAX_TOPICS = 2


def document_topics(title: str | None, *, zh: bool) -> list[str]:
    """(round 13, I16) Up to two neutral topic labels for a document, from a title that passes ``safe_headline``;
    ``[]`` when the title fails the check or matches no topic. The labels are fixed strings, never title text."""
    from ..text_safety import safe_headline

    headline = safe_headline(title)
    if headline is None:
        return []
    topics = [
        zh_label if zh else en_label for zh_label, en_label, pattern in _DOCUMENT_TOPICS if pattern.search(headline)
    ]
    return topics[:_MAX_TOPICS]


# (round 15, review-11 K13) A news or announcement question whose document search returned nothing: before round 15
# the answer restated whatever else the tools gave (the close) and said nothing about the news, so "What's the latest
# news on BYD?" read as if the close were the news. The search found no document for the target, and the answer says
# so first, by kind (news / announcements) and target; a failed search is named by ``failure_note`` instead.
_NEWS_ASKED = re.compile(
    r"新闻|公告|消息|资讯|动态|报道|快讯|舆情|\bnews\b|\bheadlines?\b|\bannouncements?\b|\bfilings?\b|"
    r"\bpress\s+releases?\b|\bwhat(?:'s|\s+is)\s+happening\b",
    re.IGNORECASE,
)
_DOCUMENT_KIND = {"search_news": ("新闻", "news"), "search_announcements": ("公告", "announcements")}


def news_gaps(query: str, tool_log: list[dict[str, Any]], *, zh: bool) -> list[str]:
    """ "当前数据中没有找到比亚迪的相关新闻。" for each news / announcement search that ran, succeeded and returned no
    document, when the question asks for news or announcements."""
    if not _NEWS_ASKED.search(query or ""):
        return []
    gaps = []
    for entry in tool_log:
        kind = _DOCUMENT_KIND.get(str(entry.get("tool")))
        data = entry.get("data") or {}
        if not kind or not entry.get("ok") or data.get("documents"):
            continue
        targets = [str(name) for name in data.get("targets") or [] if name]
        if zh:
            gaps.append(f"当前数据中没有找到{'、'.join(targets) + '的' if targets else ''}相关{kind[0]}。")
        else:
            about = f" about {', '.join(targets)}" if targets else ""
            gaps.append(f"No {kind[1]}{about} was found in the data.")
    return list(dict.fromkeys(gaps))


DOCUMENT_TEXT_LIMITATION_ZH = "资料标题和原文属于第三方内容，未经核实，回答中不引用；可在证据列表中查看。"
DOCUMENT_TEXT_LIMITATION_EN = (
    "Document titles and text are unverified third-party content and are not quoted in the answer; "
    "see the evidence list."
)
# (round 15, K13) when a headline is quoted (``quotable_headline``), the limitation says so instead
DOCUMENT_HEADLINE_LIMITATION_ZH = (
    "资料原文属于第三方内容，未经核实，回答中不引用；所引标题同样未经核实，可在证据列表中查看。"
)
_HEADLINE_MARKERS = ("（第三方标题，未经核实）", "(a third-party headline, not verified)")
DOCUMENT_HEADLINE_LIMITATION_EN = (
    "Document text is unverified third-party content and is not quoted in the answer; the headlines quoted are "
    "unverified too; see the evidence list."
)


def _documents(data: dict[str, Any], zh: bool) -> list[str]:
    """Cite retrieved documents by category, topic, source and date; never by title.

    Titles and excerpts are third-party text. A blocklist over titles (links, directives, advice words) was
    bypassed by homoglyphs, slang and plausible fake headlines ("证监会：…立案调查", "每10股派现1000元"), which no
    shape check can tell from real ones. So the deterministic answer states only facts FinSight controls: the
    document category (from the source type), the publisher when it is a plain name, the date, and (round 13, I16)
    what the document is about as one or two fixed topic labels (``document_topics``; only from a title that passes
    the headline shape check). The title stays visible in the evidence list, labelled as a source.
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
        topics = document_topics(str(document.get("title") or ""), zh=zh)
        headline = quotable_headline(document, targets, zh=zh)
        if zh:
            origin = f"{source}{'于' + when if when else ''}发布的" if source else (f"{when}的" if when else "")
            about = f"关于{'、'.join(topics)}的" if topics else ""
            quoted = f"，标题为「{headline}」（第三方标题，未经核实）" if headline else ""
            sentences.append(f"相关资料：{origin}一篇{about}{category_zh}{quoted} [{eid}]。")
        else:
            origin = f" from {source}" if source else ""
            dated = f" ({when})" if when else ""
            about = f" about {' and '.join(topics)}" if topics else ""
            quoted = f', headlined "{headline}" (a third-party headline, not verified)' if headline else ""
            sentences.append(f"Related document: a {category_en}{about}{origin}{dated}{quoted} [{eid}].")
    return sentences


# (round 15, review-11 K13) What happened, where a headline can be shown safely: a news or announcement title is quoted
# in the answer, attributed as an unverified third-party headline, only when it passes every check the evidence list
# applies (``safe_headline``: the shape rules; ``regulatory_title_withheld``: regulatory action from a non-official
# source), states no figure at all (no digit, no Chinese-numeral figure: figures come from the tools, and a headline is
# where a planted one would be shown), names one of the question's targets and is in the answer's language. Stricter
# than the evidence list, because the answer is read as FinSight's own text: no colon or question mark (a "结论：…",
# "用户：…" transcript or an "X宣布：…" lead-in), no judgment or recommendation words (建议, 观点, 目标, 看好, 低估…),
# no ending on a dangling word or a Chinese numeral (a title cut before its claim: "…一致给出茅台目标", "…同比下滑百"),
# and a whole headline, not a short label ("茅台快讯"). Anything else keeps the topic label only, as before.
_QUOTABLE_TYPES = {"news", "announcement"}
_CJK = re.compile(r"[一-鿿]")
_NOT_QUOTABLE = re.compile(
    r"[:：?？]|建议|观点|结论|研判|点评|目标|能买|能不能|值得|机会|黄金坑|低估|高估|看好|看空|看多|利好|利空|必然|"
    r"稳了|抢筹|布局|上涨空间|下跌空间|预计|预期|有望|或将|将会|"
    r"\b(?:should|could|would|may|might|expects?|expected|likely|upside|downside|target|bullish|bearish|"
    r"recommend\w*|opinion|view)\b",
    re.IGNORECASE,
)
_DANGLING_END = re.compile(
    r"(?:[零〇一二两三四五六七八九十百千万亿的了在将为至于与和及把被对向从起是给让比同达超破至起称]|宣布|表示|透露|指出)\s*$|"
    r"\b(?:the|a|an|to|at|of|by|for|from|with|and|or|as|on|in|than|its|their|has|have|is|are|was|were)\s*$",
    re.IGNORECASE,
)


def quotable_headline(document: dict[str, Any], targets: list[str], *, zh: bool) -> str | None:
    from ..text_safety import fold, regulatory_title_withheld, safe_headline
    from .output_safety import unit_figures

    if str(document.get("source_type") or "") not in _QUOTABLE_TYPES:
        return None
    title = str(document.get("title") or "")
    headline = safe_headline(title)
    if headline is None or bool(_CJK.search(headline)) != zh:
        return None
    if re.search(r"\d", fold(headline)) or unit_figures(headline, spelled=True):
        return None
    if _NOT_QUOTABLE.search(headline) or _DANGLING_END.search(headline):
        return None
    if len(_CJK.findall(headline)) < 8 if zh else len(headline.split()) < 4:
        return None
    if regulatory_title_withheld(
        title,
        source_type=document.get("source_type"),
        source_name=document.get("source_name"),
        source_url=document.get("url") or document.get("source_url"),
    ):
        return None
    if not targets or not _mentions_target({"title": headline}, targets):
        return None
    return headline


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


def _data_tools(tool_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tool log without glossary lookups: "扣非和归母有什么区别" answered from the glossary has no missing data."""
    return [entry for entry in tool_log if entry.get("tool") != "explain_concept"]


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
_MONEY_FIELDS = {
    "operating_cash_flow",
    "n_cashflow_act",
    "free_cash_flow",
    "total_mv",
    "circ_mv",
    "market_cap",
    "total_market_cap",
    "profit_dedt",
    "operate_profit",
    "operating_profit",
    "gross_profit",
    "total_profit",
}
_PER_SHARE_FIELDS = {"eps", "basic_eps", "diluted_eps", "eps_ttm", "bps"}


def _metric_value(key: str, value: Any, data: dict[str, Any], zh: bool) -> str:
    unit = (data.get("metric_units") or {}).get(key)
    if key in _PERCENT_FIELDS or unit == "%":
        number = float(value)
        # normalised payloads state ratios in percent; older payloads may hold a fraction
        return f"{_num(number if unit == '%' or abs(number) > 1 else number * 100)}%"
    if key in _MONEY_FIELDS:
        return _money(value, zh)
    if key in _PER_SHARE_FIELDS or unit == "CNY/share":
        return f"{_num(value)} 元" if zh else f"CNY {_num(value)}"
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


def _pct2(value: Any) -> str:
    """A percentage to two decimals ("1.8514" → "1.85"), as the KPI tiles show it."""
    try:
        return _num(round(float(value), 2))
    except (TypeError, ValueError):
        return _num(value)


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
