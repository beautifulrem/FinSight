"""Evidence verifier: citations must exist and numbers must be traceable to tool outputs.

Checks are claim-level. The answer and each key point are split into sentences; a number in a
sentence that cites evidence ids must be found in *those* evidence items, not merely somewhere in the
run. A number found only in other evidence is reported as ``misattributed`` (for example a PE ratio of
one company cited with another company's evidence id); a number found nowhere is ``unsupported``.
Numbers in sentences without a citation are rejected (``uncited``); with ``require_citations=False`` they
fall back to the whole evidence store. Matching allows common unit
conversions (percent <-> ratio, 万/亿, thousand/million/billion) and rounding. Dates, evidence ids,
ticker codes, and window parameters such as ``RSI(14)`` or ``近5日`` are not treated as factual claims.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .coverage import holding_value_request
from .evidence import _NUMBER as _NUMBER_TOKEN
from .evidence import EvidenceStore, _collect_numbers, extract_numbers

_SCALES = (1.0, 100.0, 0.01, 1e-4, 1e-8, 1e4, 1e8, 1e-3, 1e-6, 1e-9, 1e3, 1e6, 1e9)
# Scale factors (evidence value -> stated value) allowed for the unit written after a number. Evidence
# amounts may be stored in 元, 千元 (Tushare), 万元 or 亿元, so each unit lists the conversions that can
# legitimately produce it; a bare number must match as is. Restricting scales by unit keeps a wrong
# number from matching an unrelated value by an arbitrary power of ten.
_UNIT_SCALES: tuple[tuple[re.Pattern[str], tuple[float, ...]], ...] = (
    (re.compile(r"^\s*(?:%|％|个百分点|百分点|pct|percentage points?|bp)", re.IGNORECASE), (1.0, 100.0, 0.01)),
    (re.compile(r"^\s*万亿"), (1e-12, 1e-9, 1e-8, 1e-4, 1.0)),
    (re.compile(r"^\s*(?:亿|hundred million)", re.IGNORECASE), (1e-8, 1e-5, 1e-4, 1.0)),
    (re.compile(r"^\s*千万"), (1e-7, 1e-4, 1e-3, 1.0)),
    (re.compile(r"^\s*百万"), (1e-6, 1e-3, 1e-2, 1.0)),
    (re.compile(r"^\s*万"), (1e-4, 0.1, 1.0)),
    (re.compile(r"^\s*千(?!元)"), (1e-3, 1.0)),
    (re.compile(r"^\s*(?:billion|bn)\b", re.IGNORECASE), (1e-9, 1e-6, 1e-5, 0.1, 1.0)),
    (re.compile(r"^\s*(?:million|mn|m)\b", re.IGNORECASE), (1e-6, 1e-3, 1e-2, 100.0, 1.0)),
    (re.compile(r"^\s*(?:thousand|k)\b", re.IGNORECASE), (1e-3, 1.0)),
)
_BARE_SCALES = (1.0,)
# Market metrics must come from market/fundamental data, not from third-party text: a closing price or a
# valuation multiple that is only backed by a news excerpt is exactly what a poisoned document would plant.
_MARKET_METRIC = re.compile(
    r"收盘|收于|股价|现价|最新价|市盈率|市净率|涨跌幅|当日(?:上涨|下跌|涨|跌)|"
    r"\bclos(?:e|ed|ing)\b|share price|last price|\bP/?E\b|\bP/?B\b|price[- ]to[- ](?:earnings|book)",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[([^\[\]\s]{2,160})\]")
# Reported fundamentals that structured evidence can carry (``get_fundamentals`` payload keys). When the run has
# structured evidence for one of them, a figure for it that only document text supports is treated like a
# document-only market metric: a poisoned "修订说明：ROE已修订为47.7%" must not override the fundamentals data.
_FUNDAMENTAL_METRICS: dict[str, re.Pattern[str]] = {
    "roe": re.compile(r"(?<![A-Za-z])ROE(?![A-Za-z])|净资产收益率|return on (?:average )?equity", re.IGNORECASE),
    "eps": re.compile(r"(?<![A-Za-z])EPS(?![A-Za-z])|每股收益|每股盈利|earnings per share", re.IGNORECASE),
    "dps": re.compile(
        r"每\s*(?:10|十)\s*股\s*派|每股(?:派发|派|分配)?(?:现金)?(?:红利|分红|股利|股息|派息|派现)|每股派|"
        r"(?<![A-Za-z])DPS(?![A-Za-z])|dividends? per share|per[- ]share (?:cash )?dividend|"
        r"(?:cash )?dividend of(?= .{0,30}?per share)|"
        r"(?:for )?(?:every|per) 10 shares",
        re.IGNORECASE,
    ),
    "bps": re.compile(r"(?<![A-Za-z])BPS(?![A-Za-z])|每股净资产|book value per share", re.IGNORECASE),
}
_PER_TEN_SHARES = re.compile(r"(?:10|十)\s*股|10 shares", re.IGNORECASE)
_METRIC_KEYS: dict[str, re.Pattern[str]] = {
    "roe": re.compile(r"^roe(?:_\w+)?$", re.IGNORECASE),
    "eps": re.compile(r"^(?:\w+_)?eps(?:_\w+)?$", re.IGNORECASE),
    "dps": re.compile(r"^(?:cash_)?(?:dividend_per_share|dps)(?:_\w+)?$", re.IGNORECASE),
    "bps": re.compile(r"^(?:bps|book_value_per_share)(?:_\w+)?$", re.IGNORECASE),
}
# English month names ("May" only capitalised: "may 5%" is the verb).
EN_MONTH = (
    r"(?:(?i:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)|May)"
)
_DATE_PATTERNS = (
    re.compile(r"\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?"),
    re.compile(r"\d{4}/\d{1,2}/\d{1,2}"),
    re.compile(r"\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?"),
    re.compile(r"\d{1,2}月\d{1,2}日"),
    re.compile(r"(?:\bin|\bsince|\bby|\bFY|财年)\s*(?:19|20)\d{2}\b", re.IGNORECASE),
    re.compile(r"(?:19|20)\d{2}\s*(?:年报|年度|annual|fiscal|full[- ]year)", re.IGNORECASE),
    re.compile(r"\bQ[1-4]\b", re.IGNORECASE),
    # "04-16": month-day without a year, as models write daily series ("4.746（04-16）"); not "10-15%"
    re.compile(r"(?<![\d.\-/])(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?![\d.%％])"),
    # English dates: "April 22", "Apr. 22nd, 2026", "22 April"
    re.compile(rf"\b{EN_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+(?:19|20)\d{{2}}\b)?"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{EN_MONTH}\b(?:,?\s+(?:19|20)\d{{2}}\b)?"),
)
_PARAMETER_PATTERNS = (
    re.compile(r"(?:RSI|MA|EMA|SMA|MACD|BOLL)\s*[\(（]?\s*\d+(?:\s*[,，]\s*\d+)*\s*[\)）]?", re.IGNORECASE),
    re.compile(r"(?:近|过去|最近|前|后|未来)\s*\d+\s*(?:个)?(?:交易日|日|天|周|个月|月|年|季度)"),
    # a count ("5 个交易日", "3 篇"), not a decimal's tail nor a percentage-point figure ("1.26 个百分点", round 9)
    re.compile(r"(?<![\d.])\d+\s*(?:个)?(?:交易日|日均线|日线|篇|条|家|只|个(?!\s*百分点)|项|名|位)"),
    re.compile(
        r"\b\d+[- ]?(?:day|days|week|weeks|month|months|year|years|articles?|items?|documents?)\b", re.IGNORECASE
    ),
    re.compile(r"\d{6}\.(?:SH|SZ|BJ)", re.IGNORECASE),
    # bond tenors ("10年期国债") and list markers ("3) …", "（2）…", "1. …", "2、…")
    re.compile(r"\d+\s*年期"),
    re.compile(r"(?:^|(?<=[\s。；;：:，,]))[（(]?\d{1,2}[)）](?=\s|[一-鿿A-Za-z])"),
    re.compile(r"(?:^|(?<=\n))\s*\d{1,2}[.、](?=\s|[一-鿿])"),
)


class VerificationReport(BaseModel):
    passed: bool
    cited_ids: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    unsupported_numbers: list[float] = Field(default_factory=list)
    misattributed_numbers: list[float] = Field(
        default_factory=list,
        description="Numbers present in the run's evidence but not in the evidence cited next to them",
    )
    document_market_numbers: list[float] = Field(
        default_factory=list,
        description=(
            "Prices, valuation multiples or daily moves supported only by document text, not market data; and "
            "ROE / EPS / dividend or book value per share figures that differ from the run's structured fundamentals"
        ),
    )
    uncited_numbers: list[float] = Field(
        default_factory=list, description="Numbers in sentences that cite no evidence (LLM drafts only)"
    )
    checked_numbers: int = 0
    missing_citations: bool = False

    def feedback(self) -> str:
        problems = []
        if self.invalid_citations:
            problems.append(f"These evidence ids do not exist in this run: {', '.join(self.invalid_citations)}.")
        if self.unsupported_numbers:
            values = ", ".join(_format_number(value) for value in self.unsupported_numbers)
            problems.append(f"These numbers are not found in any tool output: {values}.")
        if self.misattributed_numbers:
            values = ", ".join(_format_number(value) for value in self.misattributed_numbers)
            problems.append(
                f"These numbers do not appear in the evidence cited in the same sentence: {values}. "
                "Cite the evidence id that actually contains each number."
            )
        if self.document_market_numbers:
            values = ", ".join(_format_number(value) for value in self.document_market_numbers)
            problems.append(
                f"These prices, valuation figures or reported fundamentals (ROE, EPS, dividend per share) are only "
                f"backed by news or document text, or contradict the fundamentals data: {values}. State market "
                "metrics and fundamentals only from market or fundamental data evidence, or drop them."
            )
        if self.uncited_numbers:
            values = ", ".join(_format_number(value) for value in self.uncited_numbers)
            problems.append(
                f"These numbers appear in sentences without a citation: {values}. Put the evidence id right after "
                "each sentence that states a number."
            )
        if self.missing_citations:
            problems.append("The answer cites no evidence ids although evidence is available.")
        return " ".join(problems)


_FAILURE_KINDS = (
    "invalid_citations",
    "unsupported_numbers",
    "misattributed_numbers",
    "document_market_numbers",
    "uncited_numbers",
    "missing_citations",
)


def failure_kinds(report: VerificationReport | dict[str, Any]) -> list[str]:
    """Names of the checks a verification report failed (e.g. ``["uncited_numbers"]``)."""
    data = report.model_dump() if isinstance(report, VerificationReport) else report
    return [kind for kind in _FAILURE_KINDS if data.get(kind)]


def answer_texts(answer: dict[str, Any]) -> list[str]:
    texts = [str(answer.get("answer") or "")]
    texts.extend(str(point) for point in answer.get("key_points") or [])
    return [text for text in texts if text.strip()]


def cited_ids(answer: dict[str, Any]) -> list[str]:
    ids: list[str] = [str(item) for item in answer.get("evidence_used") or [] if str(item).strip()]
    for text in answer_texts(answer):
        ids.extend(match.group(1) for match in _CITATION.finditer(text))
    return list(dict.fromkeys(ids))


def _cleaned(text: str) -> str:
    cleaned = _CITATION.sub(" ", text)
    for pattern in (*_DATE_PATTERNS, *_PARAMETER_PATTERNS):
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


def claim_numbers(text: str) -> list[float]:
    """Claimed values in ``text``: Arabic numbers and Chinese numerals with a quantity unit ("三十倍")."""
    return [value for value, *_ in claim_values(text)]


_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_NUMERAL = "零〇一二两三四五六七八九十百千"
# A Chinese numeral is a claim only with a quantity unit right after it (or after "百分之"), so words such as
# 一些, 统一, 十分, 千万 or 一季度 are not numbers. "成" is a tenth ("三成" = 30%), except in compounds (成长, 成本).
_CN_QUANTITY = re.compile(
    rf"(?<![几数多余第{_CN_NUMERAL}])(?P<percent>百分之)?(?P<num>[{_CN_NUMERAL}]+(?:点[{_CN_NUMERAL[:10]}]+)?)"
    r"(?:(?P<big>万亿|亿|万)(?=[元股手个]|美元|港元|$|[，。；、,.;\s）)])|"
    r"(?P<unit>倍|个百分点|成(?=[，。；、,.;\s）)]|左右|以上|以下|多|的|仓|$)|元|美元|港元)"
    r"|(?(percent)|(?!)))"
)


def _parse_cn_integer(text: str) -> int | None:
    total, current = 0, None
    for char in text:
        if char in _CN_DIGITS:
            if current is not None:  # "三五" is a range, "一二三" a list: not a single number
                return None
            current = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            total += (1 if current is None else current) * _CN_UNITS[char]
            current = None
        else:
            return None
    return total + (current or 0)


def _parse_cn_number(text: str) -> float | None:
    whole, _, fraction = text.partition("点")
    integer = _parse_cn_integer(whole) if whole else 0
    if integer is None or (fraction and any(char not in _CN_DIGITS for char in fraction)):
        return None
    return float(f"{integer}.{''.join(str(_CN_DIGITS[char]) for char in fraction)}") if fraction else float(integer)


def _chinese_values(cleaned: str) -> list[tuple[int, float, tuple[float, ...], float]]:
    """``(position, value, scales, rounding)`` for Chinese numerals with a unit ("约为三十倍", "三成", "百分之十五")."""
    values = []
    for match in _CN_QUANTITY.finditer(cleaned):
        value = _parse_cn_number(match.group("num"))
        if value is None or value == 0:
            continue
        unit, big = match.group("unit"), match.group("big")
        numeral = match.group("num")
        decimals = len(numeral.partition("点")[2])
        # a numeral ending in 十/百/千 is a round figure ("约三十倍" ~ 25-35); otherwise it is exact to its last digit
        rounding = 0.5 * _CN_UNITS[numeral[-1]] if numeral[-1] in _CN_UNITS else 0.5 * 10**-decimals
        if match.group("percent") or unit == "个百分点":
            scales = _UNIT_SCALES[0][1]
        elif unit == "成":  # tenths: "三成" is 30% give or take half a tenth
            value, rounding, scales = value * 10, 5.0, _UNIT_SCALES[0][1]
        elif big:
            scales = next(scales for pattern, scales in _UNIT_SCALES if pattern.search(big))
        else:
            scales = _BARE_SCALES
        values.append((match.start(), value, scales, rounding))
    return values


_UP = re.compile(
    r"(?:上涨|涨幅|涨了|上升|增长|增加|提高|走高|反弹|"
    r"\brose\b|\bup\b|\bgain(?:ed|s)?\b|\bincrease[ds]?\b|\bhigher\b)"
)
_DOWN = re.compile(
    r"(?:下跌|跌幅|跌了|下降|减少|回落|走低|下滑|"
    r"\bfell\b|\bdown\b|\bdecline[ds]?\b|\bdecrease[ds]?\b|\blower\b)"
)


_HYPOTHETICAL = re.compile(
    r"若|如果|假如|假设|假定|倘若|是否|能否|能不能|会不会|\bif\b|\bwhether\b|\bassum(?:e|ing)\b|\bsuppose\b",
    re.IGNORECASE,
)


def _stated_sign(token: str, before: str) -> int | None:
    """-1/+1 when the text states a direction ("-2.35", "下跌 2.35%", "rose 2%"), else ``None``."""
    if token.startswith("-"):
        return -1
    window = before[-8:].replace("涨跌幅", "").replace("涨跌", "")
    ups, downs = list(_UP.finditer(window)), list(_DOWN.finditer(window))
    if not ups and not downs:
        return None
    last_up = ups[-1].end() if ups else -1
    last_down = downs[-1].end() if downs else -1
    return 1 if last_up > last_down else -1


def claim_values(text: str) -> list[tuple[float, tuple[float, ...], float, int | None]]:
    """Claimed numbers with the scale factors their unit allows, the rounding tolerance of their precision
    and the direction the text states (``-1``/``+1``/``None``), in order of appearance.

    A number written with ``d`` decimals can differ from the evidence by at most half a unit in its
    last place (``0.5 * 10**-d``) plus 0.05% for binary rounding; "24.6" matches 24.63 but not 24.8.
    Chinese numerals count when a quantity unit follows them ("三十倍" = 30, "三成" = 30% ± 5,
    "百分之十五" = 15%); a numeral ending in 十/百/千 is read as a round figure (三十 = 30 ± 5). Vague amounts
    ("几十倍", "数成") are not claims and are not checked.
    """
    cleaned = _cleaned(text)
    values = []
    for match in _NUMBER_TOKEN.finditer(cleaned):
        token = match.group(0).replace(",", "")
        try:
            value = float(token)
        except ValueError:
            continue
        tail = cleaned[match.end() : match.end() + 24]
        scales = next((scales for pattern, scales in _UNIT_SCALES if pattern.search(tail)), _BARE_SCALES)
        decimals = len(token.split(".")[1]) if "." in token else 0
        sign = _stated_sign(token, cleaned[: match.start()])
        values.append((match.start(), (value, scales, 0.5 * 10**-decimals, sign)))
    for position, value, scales, rounding in _chinese_values(cleaned):
        values.append((position, (value, scales, rounding, _stated_sign("", cleaned[:position]))))
    return [claim for _position, claim in sorted(values, key=lambda item: item[0])]


class MetricClaim(BaseModel):
    """A reported fundamental stated in text: "ROE 为 47.7%", "每10股派现1000元" (``value`` is per share: 100)."""

    metric: str
    value: float
    stated: float
    scales: tuple[float, ...]
    rounding: float
    start: int
    end: int


def metric_claims(text: str) -> list[MetricClaim]:
    """ROE / EPS / dividend-per-share / book-value-per-share figures in ``text``: the first number within 24
    characters after the metric name (for dividends also the number just before an English "per 10 shares")."""
    cleaned = _cleaned(text)
    found: list[MetricClaim] = []
    for metric, pattern in _FUNDAMENTAL_METRICS.items():
        for match in pattern.finditer(cleaned):
            window = cleaned[match.end() : match.end() + 24]
            values = claim_values(window)
            offset = match.end()
            if not values and metric == "dps":
                window = cleaned[max(0, match.start() - 24) : match.start()]
                values = claim_values(window)[-1:]
                offset = max(0, match.start() - 24)
            if not values:
                continue
            value, scales, rounding, _sign = values[0]
            factor = 0.1 if metric == "dps" and _PER_TEN_SHARES.search(match.group(0)) else 1.0
            found.append(
                MetricClaim(
                    metric=metric,
                    value=value * factor,
                    stated=value,
                    scales=scales,
                    rounding=rounding * factor,
                    start=match.start(),
                    end=offset + len(window),
                )
            )
    return found


def structured_metric_values(store: EvidenceStore, metric: str) -> list[tuple[float, bool]]:
    """Values of ``metric`` in the structured evidence of the run (empty when no tool returned it)."""
    key_pattern = _METRIC_KEYS[metric]
    values: list[tuple[float, bool]] = []
    for item in store.items():
        if item.kind != "structured":
            continue
        for key, raw in (item.payload or {}).items():
            if not key_pattern.match(str(key)) or isinstance(raw, bool):
                continue
            try:
                values.append((float(str(raw).rstrip("%").replace(",", "")), True))
            except ValueError:
                continue
    return values


def claim_units(answer: dict[str, Any]) -> list[str]:
    """Sentences of the answer and of each key point: the unit a citation applies to."""
    units: list[str] = []
    for text in answer_texts(answer):
        units.extend(sentence for sentence in _split_sentences(text) if sentence.strip())
    return units


def _bound_units(answer: dict[str, Any]) -> list[tuple[str, str]]:
    """``(claim unit, whole sentence containing it)`` for the answer and each key point."""
    pairs: list[tuple[str, str]] = []
    for text in answer_texts(answer):
        for sentence in whole_sentences(text):
            pairs.extend((unit, sentence) for unit in _split_sentences(sentence) if unit.strip())
    return pairs


def verify_answer(
    answer: dict[str, Any],
    store: EvidenceStore,
    *,
    query: str = "",
    binding: str = "claim",
    market_precedence: bool = True,
    require_citations: bool = True,
    allow_derived: bool = False,
) -> VerificationReport:
    """``binding="claim"`` (default) checks each number against the evidence cited in its sentence with a
    unit- and precision-aware tolerance. ``"run"`` checks against all evidence of the run, and ``"legacy"``
    also uses the original loose matching (any of 13 scales, ±max(0.011, 0.5%)); both are kept only so
    ``evaluation/agent_eval/verifier_stress.py`` can measure the improvement.

    ``market_precedence`` (on for LLM drafts) rejects prices, valuation multiples and daily moves backed
    only by document text, and ROE / EPS / dividend or book value per share figures that differ from the value
    of the same metric in the run's structured evidence (only when a tool returned that metric). Template
    answers quote documents with explicit attribution ("相关资料：《…》")
    and are deterministic, so the graph turns it off for them. ``require_citations`` (on by default, for
    every draft) rejects numbers in sentences that cite no evidence instead of accepting any number of the
    run, so an uncited "预计明年涨幅21.4%" cannot borrow the 21.4 of an unrelated PE; template sentences
    always cite, so the switch only matters for model-written text.

    Known limits: numbers are bound to the *evidence items* cited in their sentence, not to fields, so a value
    reused for another metric of the same cited item (a PE of 21.4 restated as "涨幅21.4%" with the same
    citation) still passes; the claim checker (``claim_check.py``) binds metrics, the verifier does not.
    Stated directions are checked against signed structured values: "上涨2.35%" does not match -2.35.

    ``allow_derived`` (off by default; ``AgentConfig.verify_derived``) also accepts, in a sentence that cites
    evidence, a number equal to the difference, sum, ratio or percent change of two other numbers stated in
    the same sentence that the cited evidence supports ("茅台 ROE 33%，平安 15.2%，高 17.8 个百分点"). The
    operands must be written next to the result, so the arithmetic can be checked by the reader too."""
    ids = cited_ids(answer)
    invalid = [evidence_id for evidence_id in ids if evidence_id not in store]
    known = _evidence_numbers(store)
    query_numbers = claim_numbers(query)
    holding = holding_value_request(query)
    if holding is not None and float(holding[0]) not in query_numbers:
        query_numbers.append(float(holding[0]))  # "两千股": a count the claim parser reads only with some units
    unsupported: list[float] = []
    misattributed: list[float] = []
    document_market: list[float] = []
    uncited: list[float] = []
    checked = 0
    for unit, sentence in _bound_units(answer):
        unit_ids = []
        if binding == "claim":
            # a clause cut off by "；" is bound by the citation that closes its sentence
            unit_ids = [match.group(1) for match in _CITATION.finditer(unit) if match.group(1) in store] or [
                match.group(1) for match in _CITATION.finditer(sentence) if match.group(1) in store
            ]
        scope = _evidence_numbers(store, unit_ids) if unit_ids else known
        market_scope = (
            _evidence_numbers(store, [i for i in (unit_ids or store.ids()) if _is_structured(store, i)])
            if binding == "claim" and market_precedence and _MARKET_METRIC.search(unit)
            else None
        )
        claims = claim_values(unit)
        supported_claims = (
            [(v, sc) for v, sc, r, sg in claims if v and _is_supported(v, scope, sc, r, sg)]
            if allow_derived and unit_ids and binding == "claim"
            else []
        )
        operands = [v for v, _sc in supported_claims]
        # amounts written with a magnitude unit (亿元, CNY bn): their share in percent is a derived figure too
        amounts = [v for v, sc in supported_claims if _is_amount(sc)]
        for value, scales, rounding, sign in claims:
            if binding == "legacy":
                scales, rounding, sign = _SCALES, None, None
            if value == 0 and binding != "legacy":
                continue  # zero counts ("0 negative") carry no checkable magnitude
            checked += 1
            # Numbers from the question may be echoed or used hypothetically ("若收益率达到 10%"), but not
            # asserted as facts next to a citation ("市盈率为 99 倍 [price_…]").
            echo_allowed = not unit_ids or bool(_HYPOTHETICAL.search(unit))
            if echo_allowed and _is_supported(value, query_numbers, _BARE_SCALES):
                continue
            # (round 11, G5) a product with a number the user stated ("1000 × 100.64 = 100640 元" for "我有1000股"): the
            # user's count is an operand, and its product with a supported operand of the sentence is derived
            if (
                allow_derived
                and unit_ids
                and "×" in unit
                and query_numbers
                and (
                    _is_supported(value, query_numbers, _BARE_SCALES)
                    or _is_product(value, rounding, query_numbers, operands)
                )
            ):
                continue
            if _is_supported(value, scope, scales, rounding, sign):
                document_only = market_scope is not None and not _is_supported(
                    value, market_scope, scales, rounding, sign
                )
                if document_only and value not in document_market:
                    document_market.append(value)
                if require_citations and not unit_ids and binding == "claim" and value not in uncited:
                    uncited.append(value)
                continue
            shares = amounts if _is_percent(scales) else []
            if operands and _is_derived(value, rounding, [v for v in operands if v != value], shares=shares):
                continue
            if unit_ids and _is_supported(value, known, scales, rounding, sign):
                if value not in misattributed:
                    misattributed.append(value)
            elif value not in unsupported:
                unsupported.append(value)
        if binding == "claim" and market_precedence:
            # Reported fundamentals (ROE, EPS, dividend / book value per share) take the structured value when the
            # run has one: a different figure for the same metric from a document is a conflict, not a fact.
            for claim in metric_claims(unit):
                reference = structured_metric_values(store, claim.metric)
                if not reference or claim.stated in unsupported or claim.stated in document_market:
                    continue
                if not _is_supported(claim.value, reference, claim.scales, claim.rounding):
                    document_market.append(claim.stated)
    valid_cited = [evidence_id for evidence_id in ids if evidence_id in store]
    missing = len(store) > 0 and not valid_cited
    return VerificationReport(
        passed=not invalid
        and not unsupported
        and not misattributed
        and not document_market
        and not uncited
        and not missing,
        cited_ids=valid_cited,
        invalid_citations=invalid,
        unsupported_numbers=unsupported,
        misattributed_numbers=misattributed,
        document_market_numbers=document_market,
        uncited_numbers=uncited,
        checked_numbers=checked,
        missing_citations=missing,
    )


def repair_answer(
    answer: dict[str, Any],
    report: VerificationReport,
    store: EvidenceStore,
    *,
    zh: bool,
    fallback: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Delete whole sentences that state an unverified number, and invalid citations. Returns ``(answer, notes)``.

    Sentences are never cut into clauses: a sentence is kept verbatim or dropped (see ``whole_sentences``).
    When a dropped sentence is followed by one that only makes sense after it ("因此…", "This means…"),
    that sentence goes too; a plain connector ("此外，", "However, ") in front of a kept sentence is removed
    instead. If no cited statement survives, ``fallback`` (the deterministic template answer composed from
    the same tool results) replaces the draft; without one a short "evidence is not enough" answer is used.
    """
    repaired = dict(answer)
    notes: list[str] = []
    invalid = set(report.invalid_citations)
    # The report lists values, not positions: a value is held against a sentence only in the role it was
    # reported for, so an uncited restatement of a figure does not take down the cited sentence stating it.
    anywhere = [*report.unsupported_numbers, *report.document_market_numbers]
    cited_only = [*anywhere, *report.misattributed_numbers]
    uncited_only = [*anywhere, *report.uncited_numbers]

    def has_unsupported(text: str) -> bool:
        cites = any(match.group(1) in store for match in _CITATION.finditer(text))
        rejected = cited_only if cites else uncited_only
        return bool(rejected) and any(_is_supported(value, rejected, _BARE_SCALES) for value in claim_numbers(text))

    def drop_invalid(text: str) -> str:
        cleaned = _CITATION.sub(lambda match: "" if match.group(1) in invalid else match.group(0), text)
        return re.sub(r"[ \t]+([。．.，,；;！？!?])", r"\1", cleaned) if cleaned != text else text

    def prune(text: str) -> tuple[str, int]:
        kept: list[str] = []
        removed = 0
        previous_dropped = False
        for raw in whole_sentences(drop_invalid(text)):
            if not raw.strip():
                continue
            if has_unsupported(raw) or (previous_dropped and _DEPENDENT_OPENING.search(raw)):
                removed += 1
                previous_dropped = True
                continue
            kept.append(_strip_connector(raw) if previous_dropped or not kept else raw)
            previous_dropped = False
        return "".join(kept).strip(), removed

    text, removed = prune(str(answer.get("answer") or ""))
    points: list[str] = []
    for point in answer.get("key_points") or []:
        cleaned, dropped = prune(str(point))
        if cleaned:
            points.append(cleaned)
        removed += int(bool(dropped))

    if not _cites_store(text, store) and fallback and str(fallback.get("answer") or "").strip():
        limitations = [*(answer.get("limitations") or []), *(fallback.get("limitations") or [])]
        repaired.update(
            answer=str(fallback["answer"]),
            key_points=list(fallback.get("key_points") or []),
            evidence_used=[item for item in fallback.get("evidence_used") or [] if item in store],
            limitations=list(dict.fromkeys(limitations)),
        )
        notes.append(
            "模型回答中的数字无法由证据核实，已改用基于工具结果的模板回答。"
            if zh
            else "The model's answer stated figures the evidence does not support; replaced with the evidence summary."
        )
        return repaired, notes

    repaired["answer"] = text
    repaired["key_points"] = points
    valid = [evidence_id for evidence_id in cited_ids(repaired) if evidence_id in store]
    if not valid and len(store):
        valid = store.ids()[:5]
    repaired["evidence_used"] = valid

    if removed:
        notes.append(
            f"已删除 {removed} 处无法由证据核实的表述。"
            if zh
            else f"Removed {removed} statement(s) not supported by evidence."
        )
    if invalid:
        notes.append("已移除不存在的证据引用。" if zh else "Removed citations to non-existent evidence.")
    if not repaired["answer"]:
        repaired["answer"] = (
            "现有证据不足以支持完整回答，以下仅列出可核实的要点。"
            if zh
            else "The available evidence is not enough for a complete answer; only verifiable points are listed."
        )
    return repaired, notes


CITATION_REPAIRABLE = frozenset({"uncited_numbers", "misattributed_numbers", "invalid_citations", "missing_citations"})


def cite_repair(
    answer: dict[str, Any],
    report: VerificationReport,
    store: EvidenceStore,
    *,
    query: str = "",
    market_precedence: bool = True,
) -> dict[str, Any] | None:
    """Fix a draft whose only problems are citations, without an LLM call; ``None`` when that is not possible.

    Applies when every failed check is in ``CITATION_REPAIRABLE``: numbers stated without a citation, numbers
    cited with the wrong evidence id, ids that do not exist, or no valid citation at all. Invalid ids are
    removed; a number that its sentence does not support gets the id of the **one** evidence item of the run
    that contains it (structured evidence preferred over document text) appended to its sentence. A number
    found in two or more items is ambiguous and is not guessed. The text is otherwise unchanged; the caller
    re-verifies the result and only uses it when it passes.
    """
    if report.passed or not set(failure_kinds(report)) <= CITATION_REPAIRABLE:
        return None
    invalid = set(report.invalid_citations)
    query_numbers = claim_numbers(query)
    structured = [evidence_id for evidence_id in store.ids() if _is_structured(store, evidence_id)]
    documents = [evidence_id for evidence_id in store.ids() if evidence_id not in structured]
    numbers = {evidence_id: _evidence_numbers(store, [evidence_id]) for evidence_id in store.ids()}

    def owner(value: float, scales: tuple[float, ...], rounding: float, sign: int | None, market: bool) -> str | None:
        for group in (structured,) if market else (structured, documents):
            found = [i for i in group if _is_supported(value, numbers[i], scales, rounding, sign)]
            if len(found) == 1:
                return found[0]
            if found:
                return None  # ambiguous: several items state this value
        return None

    def fix(text: str) -> str | None:
        if invalid:
            text = _CITATION.sub(lambda match: "" if match.group(1) in invalid else match.group(0), text)
            text = re.sub(r"[ \t]+([。．.，,；;！？!?])", r"\1", text)
        out = []
        for sentence in whole_sentences(text):
            sentence_ids = [m.group(1) for m in _CITATION.finditer(sentence) if m.group(1) in store]
            for unit in _split_sentences(sentence):
                if not unit.strip():
                    continue
                unit_ids = [m.group(1) for m in _CITATION.finditer(unit) if m.group(1) in store] or sentence_ids
                scope = _evidence_numbers(store, unit_ids) if unit_ids else []
                market = market_precedence and bool(_MARKET_METRIC.search(unit))
                echo_allowed = not unit_ids or bool(_HYPOTHETICAL.search(unit))
                added: list[str] = []
                for value, scales, rounding, sign in claim_values(unit):
                    if value == 0 or (echo_allowed and _is_supported(value, query_numbers, _BARE_SCALES)):
                        continue
                    if unit_ids and _is_supported(value, scope, scales, rounding, sign):
                        continue
                    evidence_id = owner(value, scales, rounding, sign, market)
                    if evidence_id is None:
                        return None
                    if evidence_id not in added and evidence_id not in unit_ids:
                        added.append(evidence_id)
                if added:
                    citation = "".join(f"[{evidence_id}]" for evidence_id in added)
                    body = unit.rstrip()
                    terminal = re.search(r"[。！？!?；;.]+$", body)
                    cut = terminal.start() if terminal else len(body)
                    sentence = sentence.replace(unit, body[:cut] + citation + body[cut:] + unit[len(body) :], 1)
            out.append(sentence)
        return "".join(out)

    repaired = dict(answer)
    text = fix(str(answer.get("answer") or ""))
    if text is None:
        return None
    points = []
    for point in answer.get("key_points") or []:
        fixed = fix(str(point))
        if fixed is None:
            return None
        points.append(fixed)
    repaired["answer"] = text
    repaired["key_points"] = points
    repaired["evidence_used"] = [evidence_id for evidence_id in cited_ids(repaired) if evidence_id in store]
    return repaired


def _cites_store(text: str, store: EvidenceStore) -> bool:
    """True when ``text`` keeps at least one citation of this run (or the run has no evidence and text remains)."""
    if not len(store):
        return bool(text.strip())
    return any(match.group(1) in store for match in _CITATION.finditer(text))


def _strip_connector(sentence: str) -> str:
    """Drop a leading "此外，" / "However, " whose antecedent sentence was removed; re-capitalise English."""
    match = _CONNECTOR_OPENING.match(sentence)
    if not match or match.end() >= len(sentence.rstrip()):
        return sentence
    rest = sentence[match.end() :]
    return match.group(1) + (rest[:1].upper() + rest[1:] if rest[:1].isascii() else rest)


def _is_structured(store: EvidenceStore, evidence_id: str) -> bool:
    item = store.get(evidence_id)
    return item is not None and item.kind == "structured"


def _evidence_numbers(store: EvidenceStore, evidence_ids: list[str] | None = None) -> list[tuple[float, bool]]:
    """``(value, signed)`` pairs. Structured payload values keep their sign; numbers read from text do not
    ("同比下降1.21%" yields 1.21), so they only match by magnitude."""
    values: list[tuple[float, bool]] = []
    items = store.items() if evidence_ids is None else [store.get(evidence_id) for evidence_id in evidence_ids]
    for item in items:
        if item is None:
            continue
        signed: list[float] = []
        _collect_numbers(item.payload, signed)
        values.extend((value, True) for value in signed)
        for text in (item.title, item.text_excerpt):
            if text:
                values.extend((value, False) for value in extract_numbers(text))
    return values


def _is_supported(
    value: float,
    known: list[float] | list[tuple[float, bool]],
    scales: tuple[float, ...] = _SCALES,
    rounding: float | None = None,
    sign: int | None = None,
) -> bool:
    """``rounding`` is the stated number's precision tolerance; ``None`` keeps the legacy loose tolerance.

    With a precision tolerance, x100 (percent) conversion only applies to fractional evidence (|v| <= 1.5,
    e.g. ROE 0.33 -> 33%), and a stated direction must agree with the sign of signed evidence values.
    """
    for entry in known:
        base, signed = (entry, False) if isinstance(entry, int | float) else entry
        for scale in scales:
            if rounding is not None and scale == 100.0 and abs(base) > 1.5:
                continue
            target = base * scale
            loose = max(0.011, abs(target) * 0.005)
            tolerance = loose if rounding is None else rounding + abs(target) * 0.0005 + 1e-9
            if sign is not None and signed and target != 0 and (target > 0) != (sign > 0):
                continue
            # Signs are otherwise compared loosely: "下跌 1.2%" legitimately restates a change of -1.2.
            if abs(abs(value) - abs(target)) <= tolerance:
                return True
    return False


def _is_amount(scales: tuple[float, ...]) -> bool:
    """A number written with a magnitude unit (亿, 万, bn, mn), not a percent or a plain multiple."""
    return bool(scales) and min(scales) < 0.001 and 0.01 not in scales


def _is_percent(scales: tuple[float, ...]) -> bool:
    return 0.01 in scales and 100.0 in scales


def _is_derived(value: float, rounding: float | None, operands: list[float], *, shares: list[float] = ()) -> bool:
    """``value`` is a - b, a + b, a / b or the percent change (a - b) / b of two stated operands, or, for a value
    written in percent, the share a / b of two stated amounts (a net margin: net profit / revenue, a <= b) or the
    difference of two such shares of four stated amounts (a net-margin gap, round 10). The share
    is limited to amounts: allowing a / b * 100 for any pair (multiples, ratios) raised the stress test's derived
    false-accept rate from 0.020 to 0.028."""
    tolerance = 0.5 if rounding is None else rounding
    pairs = [(a, b, False) for i, a in enumerate(operands) for j, b in enumerate(operands) if i != j]
    share_pairs = [(i, j) for i, a in enumerate(shares) for j, b in enumerate(shares) if i != j and 0 < a <= b]
    pairs += [(shares[i], shares[j], True) for i, j in share_pairs]
    for a, b, share in pairs:
        if share:
            candidates = [a / b * 100]
        else:
            candidates = [a - b, a + b]
            if b:
                candidates += [a / b, (a - b) / abs(b) * 100]
        for candidate in candidates:
            if candidate and abs(abs(value) - abs(candidate)) <= tolerance + abs(candidate) * 0.0005 + 1e-9:
                return True
    # (round 10, F8) the gap in percentage points between two such shares of four stated amounts (two net margins:
    # "823.2 亿 ÷ 1688.38 亿 ≈ 48.76%，378 亿 ÷ 1085 亿 ≈ 34.84%，相差 13.92 个百分点"), the amounts all disjoint
    for i, j in share_pairs:
        for k, m in share_pairs:
            if len({i, j, k, m}) < 4:
                continue
            candidate = (shares[i] / shares[j] - shares[k] / shares[m]) * 100
            if candidate and abs(abs(value) - abs(candidate)) <= tolerance + abs(candidate) * 0.0005 + 1e-9:
                return True
    return False


def _is_product(value: float, rounding: float | None, factors: list[float], operands: list[float]) -> bool:
    """``value`` is a stated number from the question times a supported operand of the same sentence."""
    tolerance = 0.5 if rounding is None else rounding
    for factor in factors:
        for operand in operands:
            candidate = factor * operand
            if candidate and abs(abs(value) - abs(candidate)) <= tolerance + abs(candidate) * 0.0005 + 1e-9:
                return True
    return False


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])|(?<=\.)\s+", text)
    return [part for part in parts if part]


_OPENERS = "(（《“「【"
_CLOSERS = ")）》”」】"
_CITATION_RUN = re.compile(r"(?:[ \t]*\[[^\[\]\s]{2,160}\])+")
_TRAILING_SPACE = re.compile(r"\s*")


def whole_sentences(text: str) -> list[str]:
    """Split ``text`` into whole sentences, keeping every character (``"".join(result) == text``).

    Unlike the claim units used for verification, a sentence only ends at 。！？!? or at a full stop
    followed by whitespace, outside brackets and quotes, and not at list numbering ("1. "); ``；`` and
    ``;`` stay inside the sentence. Citations written right after the terminator ("growth. [id]") and the
    whitespace that follows belong to the sentence they close, so dropping a sentence never leaves an
    orphan citation or glues its neighbours together.
    """
    sentences: list[str] = []
    start = depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS:
            depth = max(0, depth - 1)
        boundary = char == "\n"
        if not boundary and depth == 0:
            if char in "。！？!?":
                boundary = True
            elif char == "." and (index + 1 == len(text) or text[index + 1].isspace()):
                boundary = not re.fullmatch(r"\s*\d{1,2}\.", text[start : index + 1])
        if boundary:
            end = index + 1
            citations = _CITATION_RUN.match(text, end)
            if citations:
                end = citations.end()
            end = _TRAILING_SPACE.match(text, end).end()
            sentences.append(text[start:end])
            start = index = end
            continue
        index += 1
    if start < len(text):
        sentences.append(text[start:])
    return sentences


# A sentence that opens with one of these only makes sense after the sentence before it.
_DEPENDENT_OPENING = re.compile(
    r"^\s*(?:因此|所以|因而|从而|由此|故而|这意味着|这表明|这说明|这显示|这一|其中|对此|"
    r"therefore\b|thus\b|hence\b|as a result\b|consequently\b|so\b|which\b|"
    r"this (?:means|suggests|shows|indicates|implies)\b|that (?:means|suggests|shows)\b)",
    re.IGNORECASE,
)
# Connectors that can simply be dropped when the sentence before them is gone.
_CONNECTOR_OPENING = re.compile(
    r"^(\s*)(?:此外|另外|与此同时|同时|而且|并且|但是|然而|不过|相比之下|相较之下|对比之下|另一方面|但|"
    r"also\b|in addition\b|additionally\b|moreover\b|furthermore\b|meanwhile\b|however\b|but\b|"
    r"by contrast\b|in contrast\b|on the other hand\b|besides\b)\s*[，,、:：]?\s*",
    re.IGNORECASE,
)
_CONTINUATION_OPENING = re.compile(r"^\s*(?:[，,；;、：:)）\]]|以及|(?:and|or|while|whereas)\b)", re.IGNORECASE)
_TERMINAL = re.compile(r"[。！？.!?…][”」’\"')）]*$")
_BRACKET_PAIRS = (("(", ")"), ("（", "）"), ("[", "]"), ("《", "》"), ("“", "”"), ("【", "】"))


def readability_issues(text: str) -> list[str]:
    """Surface defects of an edited answer: empty or dangling sentences, broken brackets, orphan citations.

    Used by tests and by ``evaluation/agent_eval/verifier_stress.py`` to measure repaired answers. Each issue
    is ``"<kind>: <sentence excerpt>"``; an empty list means the text reads as whole sentences.
    """
    issues: list[str] = []
    stripped = text.strip()
    if not stripped:
        return ["empty"]
    for opener, closer in _BRACKET_PAIRS:
        if stripped.count(opener) != stripped.count(closer):
            issues.append(f"unbalanced_brackets: {opener}{closer}")
    if re.search(r"\.\s*(?:\[[^\]]*\]\s*)*。|。\s*(?:\[[^\]]*\]\s*)*\.(?!\d)", stripped):
        issues.append("mixed_terminators")
    for sentence in whole_sentences(stripped):
        body = sentence.strip()
        if not body:
            continue
        content = _CITATION.sub("", body).strip()
        excerpt = body[:40]
        if not re.search(r"[\w一-鿿]", content):
            issues.append(f"orphan_citation: {excerpt}" if _CITATION.search(body) else f"stray_punctuation: {excerpt}")
            continue
        if body.startswith("["):
            issues.append(f"starts_with_citation: {excerpt}")
        if _CONTINUATION_OPENING.search(content):
            issues.append(f"starts_mid_clause: {excerpt}")
        elif _DEPENDENT_OPENING.search(content) or _CONNECTOR_OPENING.match(content):
            issues.append(f"starts_with_connector: {excerpt}")
        elif re.match(r"[a-z]", content) and not re.match(r"[a-z]+[A-Z0-9(]", content):  # "down 1.2%", not "iPhone"
            issues.append(f"starts_lowercase: {excerpt}")
    body = _CITATION.sub("", stripped).rstrip()
    if body and not _TERMINAL.search(body):
        issues.append(f"unterminated: {body[-40:]}")
    return issues


def _format_number(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:g}"
