"""Hearsay in the chat: "听说茅台市盈率只有15倍，是真的吗" is a claim to check, not only a question.

``claim_in_message`` finds the claim inside such a message (the same rule as the web UI's ``claimInMessage``);
``fact_check_for`` runs the deterministic claim check on it (no LLM) and returns the report as a plain dict,
which ``/chat`` (workflow) and the agent's result carry as ``fact_check`` for the UI to render inline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .claim_check import check_claim

logger = logging.getLogger(__name__)

_CUE = re.compile(
    r"听说|据说|传言|传闻|有人说|网上说|听人说|据传|号称|是真的吗|真的吗|是真的么|对吗|对不对|是不是真的|属实|靠谱吗|"
    r"\bis it true\b|\bi heard\b|\bsomeone said\b|\brumou?r\b|\bis (?:that|this) (?:true|right)\b",
    re.I,
)
_CONTENT = re.compile(
    r"\d|[一二两三四五六七八九十]+(?:点[〇零一二三四五六七八九]+)?(?:倍|成|%|元|亿)|涨了|跌了|大涨|大跌|涨停|跌停|"
    r"比.{1,12}(?:高|低|贵|便宜|多|少)|高于|低于|\b(?:rose|fell|higher than|lower than)\b",
    re.I,
)
_LEADING = re.compile(
    r"^\s*(?:我)?(?:听说|据说|传言|传闻|有人说|网上说|听人说|据传|I heard(?: that)?|someone said(?: that)?|"
    r"is it true(?: that)?)[，,：:\s]*",
    re.I,
)
_TRAILING = re.compile(
    r"[，,。\s]*(?:这|这个|这话|这是)?(?:是真的吗|真的吗|是真的么|对吗|对不对|是不是真的|属实吗?|靠谱吗|"
    r",?\s*is (?:that|this|it) (?:true|right))?\s*[？?！!。.]*\s*$",
    re.I,
)


def claim_in_message(message: str) -> str | None:
    """The claim inside a hearsay chat message, or None. "听说茅台市盈率只有15倍，是真的吗？" → "茅台市盈率只有15倍"."""
    text = (message or "").strip()
    if len(text) < 4 or not _CUE.search(text) or not _CONTENT.search(text):
        return None
    claim = _TRAILING.sub("", _LEADING.sub("", text)).strip()
    return claim if len(claim) >= 2 else None


def fact_check_for(message: str, *, service: Any, registry: Any, zh: bool = True) -> dict[str, Any] | None:
    """The claim-check report for a hearsay message, or None when the message is not hearsay or the check found
    no number, move or relation to compare. Best effort: a failure never breaks the chat answer."""
    claim = claim_in_message(message)
    if claim is None:
        return None
    try:
        report = check_claim(claim, service=service, registry=registry, zh=zh)
    except Exception:  # the answer does not depend on it
        logger.exception("inline fact check failed")
        return None
    if not report.checks:
        return None
    return report.model_dump()


# ---------------------------------------------------------------------------------------------------------
# The answer text: the claimed number next to the actual one ("你说的市盈率15倍与数据不符：数据为24.6倍").
# ---------------------------------------------------------------------------------------------------------
_VERDICT = {
    "supported": ("与数据相符", "matches the data"),
    "contradicted": ("与数据不符", "does not match the data"),
    "partially_supported": ("部分与数据相符", "partly matches the data"),
    "unverifiable": ("无法用现有数据核实", "cannot be verified with the available data"),
}
_STATUS = {
    "supported": ("相符", "matches"),
    "contradicted": ("不符", "does not match"),
    "unverifiable": ("无法核实", "cannot be verified"),
}
_METRIC_LABEL = {
    "pe_ttm": ("市盈率", "P/E"),
    "pb": ("市净率", "P/B"),
    "roe": ("ROE", "ROE"),
    "close": ("收盘价", "close"),
    "pct_change_1d": ("涨跌幅", "daily change"),
    "amount": ("成交额", "turnover"),
    "revenue": ("营收", "revenue"),
    "net_profit": ("净利润", "net profit"),
    "gross_margin": ("毛利率", "gross margin"),
    "net_margin": ("净利率", "net margin"),
    "dividend_yield": ("股息率", "dividend yield"),
    "market_cap": ("市值", "market cap"),
    "eps": ("每股收益", "EPS"),
    "debt_ratio": ("资产负债率", "debt ratio"),
    "revenue_yoy": ("营收同比", "revenue growth"),
    "netprofit_yoy": ("净利润同比", "net profit growth"),
}
_CMP = {
    "eq": ("", ""),
    "ne": ("不是", "not "),
    "gt": ("超过", "above "),
    "ge": ("至少", "at least "),
    "lt": ("不到", "below "),
    "le": ("不超过", "at most "),
    "approx": ("约", "about "),
}
_REL = {"gt": ("高于", "higher than"), "ge": ("不低于", "not lower than"), "lt": ("低于", "lower than")}
_REL |= {"le": ("不高于", "not higher than")}
_MOVE = {"lt": ("下跌", "fell"), "gt": ("上涨", "rose"), "ge": ("没有下跌", "did not fall")}
_MOVE |= {"le": ("没有上涨", "did not rise")}
_REASON = {
    "no_target": ("没有识别到具体的公司、基金或指数", "no listed company, fund or index was recognised"),
    "no_metric": ("没有识别到指标", "the metric was not recognised"),
    "no_data": ("数据源没有这项数据", "the sources have no value for it"),
    "growth_unavailable": ("数据源没有同比增速", "the sources have no year-on-year growth"),
    "unit_mismatch": ("单位与指标不符", "the unit does not fit the metric"),
    "no_unit": ("金额缺少单位", "the amount has no unit"),
    "forecast": ("这是预测或假设，不是已披露的事实", "it is a forecast or hypothetical"),
    "period_mismatch": ("说法所指的时期与数据不同", "it is about another period than the data"),
    "multi_day": ("说的是多日变化，数据只有最近一个交易日", "it is about several days; only the latest session exists"),
    "no_reference": ("数据源没有市场平均值", "the sources have no market average"),
}
_PERCENT_LEVELS = {"roe", "gross_margin", "net_margin", "dividend_yield", "debt_ratio", "cn10y", "lpr_1y", "lpr_5y"}
_SIGNED = {"pct_change_1d", "revenue_yoy", "netprofit_yoy", "cpi_yoy", "ppi_yoy", "m2_yoy", "gdp_yoy"}
_AMOUNTS = {"revenue", "net_profit", "amount", "market_cap"}
_MAX_SENTENCES = 4


def _pick(pair: tuple[str, str], zh: bool) -> str:
    return pair[0] if zh else pair[1]


def _value(metric: str | None, value: float, zh: bool, *, signed: bool = True) -> str:
    """An actual value in the metric's unit: "24.6倍", "-0.53%", "33%", "1409.5元", "14.53 亿元"."""
    from .composer import _money, _num

    if metric in {"pe_ttm", "pb"}:
        return f"{_num(round(value, 2))}倍" if zh else f"{_num(round(value, 2))}x"
    if metric in _SIGNED:
        return f"{value:+.2f}%" if signed else f"{_num(round(abs(value), 2))}%"
    if metric in _PERCENT_LEVELS:
        return f"{_num(round(value, 2))}%"
    if metric in {"close", "eps"}:
        return f"{_num(round(value, 2))}元" if zh else f"CNY {_num(round(value, 2))}"
    if metric in _AMOUNTS:
        return _money(value, zh).replace(" 亿元", "亿元").replace(" 万元", "万元")
    return _num(round(value, 2))


def _claimed(check: dict[str, Any], zh: bool) -> str:
    """The claimed number as written, with its comparator: "15倍", "超过30%", "跌超过1%", "1688亿"."""
    from .composer import _num

    metric, value = check.get("metric"), float(check.get("claimed") or 0.0)
    comparator = str(check.get("comparator") or "eq")
    unit = str(check.get("claimed_unit") or "")
    if metric in _AMOUNTS and unit:
        text = f"{_num(abs(value))}{unit}" if zh else f"{_num(abs(value))} {unit}"
    elif metric in _SIGNED and check.get("direction"):
        text = _value(metric, value, zh, signed=False)
    else:
        text = _value(metric, value, zh, signed=metric in _SIGNED)
    if comparator == "range" and check.get("claimed_high") is not None:
        high = float(check["claimed_high"])
        high_text = _value(metric, high, zh, signed=not check.get("direction"))
        text = f"{text}到{high_text}" if zh else f"{text} to {high_text}"
    else:
        text = f"{_pick(_CMP.get(comparator, ('', '')), zh)}{text}"
    if check.get("direction"):
        down = check["direction"] == "down"
        move = ("跌" if down else "涨") if zh else ("fell " if down else "rose ")
        return f"{move}{text}"
    return text


def _name(value: Any, zh: bool) -> str:
    from .names import english_display, english_name

    text = str(value or "")
    return text if zh else (english_name(text) or english_display(text))


def _check_sentence(check: dict[str, Any], zh: bool) -> str:
    metric = check.get("metric")
    target = _name(check.get("target"), zh) or ("该标的" if zh else "the target")
    label = _pick(_METRIC_LABEL.get(str(metric), ("", "")), zh)
    if metric == "pct_change_1d" and check.get("direction"):
        label = ""  # "贵州茅台跌超过1%": the move word says it
    status = str(check.get("status") or "unverifiable")
    comparator = str(check.get("comparator") or "eq")
    reference = _name(check.get("reference"), zh) if check.get("reference") else None
    space = "" if zh else " "
    if check.get("ratio") is not None or (reference and check.get("claimed") is not None):
        multiple = f"{float(check.get('claimed') or 0):g}"
        bound = _pick(_CMP.get(comparator, ("", "")), zh)
        said = (
            f"{target}{label}{bound or '是'}{reference}的{multiple}倍"
            if zh
            else f"{target} {label} {bound or ''}{multiple} times {reference}'s"
        )
    elif reference:
        word = _pick(_REL.get(comparator, ("接近", "close to")), zh)
        said = f"{target}{label}{word}{reference}" if zh else f"{target} {label} {word} {reference}"
    elif metric == "pct_change_1d" and not check.get("claimed") and comparator in _MOVE:
        said = f"{target}{_pick(_MOVE[comparator], zh)}" if zh else f"{target} {_pick(_MOVE[comparator], zh)}"
    else:
        said = (
            space.join(part for part in (target, label, _claimed(check, zh)) if part)
            if not zh
            else (f"{target}{label}{_claimed(check, zh)}")
        )
    said = " ".join(said.split()) if not zh else said
    verdict = _pick(_STATUS.get(status, _STATUS["unverifiable"]), zh)
    if status == "unverifiable" or check.get("actual") is None:
        why = _pick(_REASON.get(str(check.get("reason") or "no_data"), _REASON["no_data"]), zh)
        return f"“{said}”{verdict}：{why}。" if zh else f'"{said}" {verdict}: {why}.'
    actual = _value(metric, float(check["actual"]), zh)
    as_of = f"，截至 {check['as_of']}" if zh and check.get("as_of") else ""
    as_of_en = f", as of {check['as_of']}" if not zh and check.get("as_of") else ""
    if check.get("ratio") is not None:
        facts = (
            f"实际约为{float(check['ratio']):.2f}倍" if zh else f"the actual multiple is {float(check['ratio']):.2f}"
        )
    elif reference and check.get("reference_value") is not None:
        other = _value(metric, float(check["reference_value"]), zh)
        facts = (
            f"数据为{target} {actual}、{reference} {other}"
            if zh
            else f"the data: {target} {actual}, {reference} {other}"
        )
    else:
        facts = f"数据为 {actual}" if zh else f"the data shows {actual}"
    return f"“{said}”{verdict}，{facts}{as_of}。" if zh else f'"{said}" {verdict}; {facts}{as_of_en}.'


def fact_check_prose(report: dict[str, Any] | None, *, zh: bool = True) -> str:
    """The answer's opening for a hearsay question: the verdict, then each claimed number next to the actual
    one ("核查结论：与数据不符。“贵州茅台市盈率15倍”不符，数据为 24.6倍，截至 2025-12-31。"). Deterministic,
    from the claim-check report the card shows; empty when there is no report."""
    if not report or not report.get("checks"):
        return ""
    verdict = _pick(_VERDICT.get(str(report.get("verdict")), _VERDICT["unverifiable"]), zh)
    checks = list(report["checks"])
    sentences = [_check_sentence(check, zh) for check in checks[:_MAX_SENTENCES]]
    if len(checks) > _MAX_SENTENCES:
        rest = len(checks) - _MAX_SENTENCES
        sentences.append(f"另有 {rest} 项见下方核查卡片。" if zh else f"{rest} more in the fact-check card below.")
    unchecked = len(report.get("unchecked") or [])
    if unchecked:
        sentences.append(
            f"另有 {unchecked} 处说法没有可核查的数字，未核查。"
            if zh
            else f"{unchecked} part(s) of the claim had nothing to check and were not checked."
        )
    head = f"**核查结论：你听到的说法{verdict}。**" if zh else f"**Fact check: the claim you heard {verdict}.**"
    return head + ("" if zh else " ") + ("" if zh else " ").join(sentences)
