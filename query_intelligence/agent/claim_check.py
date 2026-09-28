"""Check a pasted market claim against live (or snapshot) data: "茅台市盈率只有15倍，股价跌了5%".

The claim is parsed with the classical NLU (targets) and a rule-based reader for the numbers; the price
and fundamentals tools fetch evidence for every target, and each claimed number is compared with the
evidence value of *its* target and metric. The rules are documented in ``docs/claim-check.md``; in short:

* **Numbers.** Arabic and simple Chinese numerals ("十五倍", "一点一倍", "三成", "百分之三十"); dates,
  durations, tickers and index names ("沪深300") are not claims. "20到30倍" / "between 20 and 30" is one
  range. A move without a number ("昨天下跌了", "并没有跌") is a check of the daily change against 0.
* **Metric.** The nearest metric word in the number's clause whose unit fits the number ("24.6倍" →
  P/E, "33%" → ROE). A clause with no metric word inherits the previous number's metric when the unit is
  the same ("茅台市盈率24.6倍，五粮液20.9倍"). A percentage next to revenue / net profit is YoY growth.
* **Target.** The nearest company, fund or index named before the number.
* **Comparator** (``eq ne gt ge lt le approx range``), read between the previous number and this one plus
  the words right after it (以上/以下/左右/多). Negation ("不是", "没有", "not") flips it: eq → ne,
  gt → le, lt → ge, ...
* **Status.** ``eq`` matches within half a unit of the last written digit or 2% (5% for ``approx``);
  ``ne`` is the opposite; bounds and ranges are literal. ``unverifiable`` when there is no target, no
  metric, no data, a unit that does not fit the metric, an amount with no unit, a forecast, a period
  other than the data's, or a multi-day move (only the daily change is available).

The verdict is deterministic (no LLM) and every check cites the evidence id, source and as-of date (for
P/E and P/B the valuation date when the source gives one, see ``as_of_basis``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from .evidence import _NUMBER as _NUMBER_TOKEN
from .evidence import AgentEvidence
from .tools import ToolRegistry
from .verifier import _BARE_SCALES, _DATE_PATTERNS, _HYPOTHETICAL, _PARAMETER_PATTERNS, _UNIT_SCALES, _stated_sign

Status = Literal["supported", "contradicted", "unverifiable"]
Comparator = Literal["eq", "ne", "gt", "ge", "lt", "le", "approx", "range"]
Reason = Literal[
    "no_target",
    "no_metric",
    "no_data",
    "growth_unavailable",
    "unit_mismatch",
    "no_unit",
    "forecast",
    "period_mismatch",
    "multi_day",
]

# ---------------------------------------------------------------------------------------------------------
# Metrics: payload keys, the words that name them, and the unit classes a number for them may carry.
# ---------------------------------------------------------------------------------------------------------
_PERCENT, _MULTIPLE, _PRICE, _AMOUNT, _POINTS = "percent", "multiple", "price", "amount", "points"


@dataclass(frozen=True)
class _Metric:
    keys: tuple[str, ...]
    words: re.Pattern[str] | None  # None: never named directly (growth, see _metric_for)
    units: frozenset[str | None]
    fraction: bool = False  # the payload may hold 0.33 for a claimed 33%
    fundamental: bool = True


def _words(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_RATIO = frozenset({_PERCENT, None})
_METRICS: dict[str, _Metric] = {
    "close": _Metric(
        ("close",),
        _words(
            r"收盘价?|收于|收报|股价|现价|价格|报价|最新价|点位|\bclos(?:e|ed|ing)\b|"
            r"\bshare price\b|\bprice\b(?![- ]to[- ])"
        ),
        frozenset({_PRICE, _POINTS, None}),
        fundamental=False,
    ),
    "pct_change_1d": _Metric(
        ("pct_change_1d",),
        _words(
            r"涨跌幅|涨幅|跌幅|大涨|大跌|收涨|收跌|上涨|下跌|涨了|跌了|(?<!涨)跌(?![幅破到至])|涨(?![跌幅到至])|"
            r"\bup\b|\bdown\b|\brose\b|\bfell\b|\bgained\b|\blost\b|\bdropped\b|\bjumped\b|\bslid\b|\bclimbed\b"
        ),
        _RATIO,
        fundamental=False,
    ),
    "pe_ttm": _Metric(
        ("pe_ttm", "pe"),
        _words(r"市盈率|(?<![A-Za-z])P/?E(?![A-Za-z])|price[- ]to[- ]earnings"),
        frozenset({_MULTIPLE, None}),
    ),
    "pb": _Metric(
        ("pb",), _words(r"市净率|(?<![A-Za-z])P/?B(?![A-Za-z])|price[- ]to[- ]book"), frozenset({_MULTIPLE, None})
    ),
    "roe": _Metric(
        ("roe",), _words(r"(?<![A-Za-z])ROE(?![A-Za-z])|净资产收益率|return on equity"), _RATIO, fraction=True
    ),
    "revenue": _Metric(
        ("revenue", "total_revenue"),
        _words(r"营收|营业总?收入|收入|\brevenues?\b|\bsales\b"),
        frozenset({_AMOUNT, _PRICE, None}),
    ),
    "net_profit": _Metric(
        ("net_profit", "n_income"),
        _words(r"归母净利润|净利润|净利|\bnet (?:profit|income|earnings)\b"),
        frozenset({_AMOUNT, _PRICE, None}),
    ),
    "gross_margin": _Metric(
        ("gross_margin", "grossprofit_margin"), _words(r"毛利率|gross (?:profit )?margin"), _RATIO, fraction=True
    ),
    "net_margin": _Metric(
        ("net_margin", "netprofit_margin"), _words(r"净利率|净利润率|net (?:profit )?margin"), _RATIO, fraction=True
    ),
    "dividend_yield": _Metric(("dividend_yield",), _words(r"股息率|dividend yield"), _RATIO),
    "market_cap": _Metric(
        ("total_mv", "market_cap"), _words(r"总市值|市值|market (?:cap|capitali[sz]ation|value)"), frozenset({_AMOUNT})
    ),
    "eps": _Metric(("eps",), _words(r"每股收益|(?<![A-Za-z])EPS(?![A-Za-z])"), frozenset({_PRICE, None})),
    "debt_ratio": _Metric(
        ("debt_to_assets", "debt_ratio"), _words(r"资产负债率|负债率|debt[- ]to[- ]assets?|debt ratio"), _RATIO
    ),
    # Growth is not named by its own word: a percentage next to revenue / net profit (see _metric_for).
    "revenue_yoy": _Metric(("revenue_yoy", "or_yoy", "tr_yoy"), None, _RATIO),
    "netprofit_yoy": _Metric(("netprofit_yoy",), None, _RATIO),
}
_GROWTH_OF = {"revenue": "revenue_yoy", "net_profit": "netprofit_yoy"}
_LEVELS = {"revenue", "net_profit", "eps"}  # flows: an interim report holds a year-to-date total
_GROWTH_WORDS = re.compile(
    r"同比|环比|增速|增幅|增长|下降|下滑|减少|\bYoY\b|year[- ]on[- ]year|\bgrowth\b|\bgrew\b", re.I
)

_UNIT = re.compile(
    r"\s*(万亿|亿元|亿|千万|百万|万元|万|元|块|倍|%|个百分点|百分点|点|x(?![A-Za-z])|times\b|"
    r"trillion|billion|million|bn\b|mn\b|tn\b|yuan\b|RMB\b|CNY\b|percentage points?\b|percent\b|pct\b|points?\b|pts\b)",
    re.I,
)
_UNIT_CLASS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:%|个百分点|百分点|percentage points?|percent|pct)$", re.I), _PERCENT),
    (re.compile(r"^(?:倍|x|times)$", re.I), _MULTIPLE),
    (re.compile(r"^(?:元|块|yuan|RMB|CNY)$", re.I), _PRICE),
    (re.compile(r"^(?:点|points?|pts)$", re.I), _POINTS),
    (re.compile(r"^(?:万亿|亿元|亿|千万|百万|万元|万|trillion|billion|million|bn|mn|tn)$", re.I), _AMOUNT),
)

# ---------------------------------------------------------------------------------------------------------
# Comparators, negation and context markers.
# ---------------------------------------------------------------------------------------------------------
_COMPARATOR_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("le", _words(r"至多|最多|以内|\bat most\b|\bup to\b")),
    ("ge", _words(r"至少|起码|最少|\bat least\b")),
    (
        "lt",
        _words(r"低于|小于|少于|不到|不足|不及|跌破|\bbelow\b|\bunder\b|\bless than\b|\blower than\b|\bfewer than\b"),
    ),
    (
        "gt",
        _words(
            r"超过|超出|高于|大于|多于|逾|突破|站上|超(?![跌买卖额级大])|\babove\b|\bover\b|\bmore than\b|"
            r"\bgreater than\b|\bhigher than\b|\bexceed(?:s|ed|ing)?\b|\bin excess of\b|\bupwards of\b"
        ),
    ),
    (
        "approx",
        _words(
            r"约|大约|大概|将近|接近|差不多|近(?=\s*\d)|\babout\b|\baround\b|\broughly\b|\bapproximately\b|"
            r"\bnearly\b|\balmost\b|\bsome\b|~(?=\s*\d)"
        ),
    ),
)
_POST_COMPARATOR: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ge", _words(r"^\s*(?:或?以上|\bor (?:more|above|higher)\b|\+)")),
    ("le", _words(r"^\s*(?:或?以下|以内|\bor (?:less|below|lower)\b)")),
    ("approx", _words(r"^\s*(?:左右|上下|出头)")),
)
_NEGATION = _words(
    r"不是|并非|并不是|绝非|并没有|没有|没(?!有)|未(?!来)|不(?=超|高于|低于|大于|小于|少于|多于|在)|"
    r"\bnot\b|n't\b|\bnever\b|\bno\b(?=\s+(?:more|less|higher|lower|greater|fewer)\b)"
)
_FLIP: dict[str, Comparator] = {"eq": "ne", "approx": "ne", "ne": "eq", "gt": "le", "ge": "lt", "lt": "ge", "le": "gt"}
_RANGE_GAP = re.compile(r"\s*(?:到|至|~|～|-|－|—|–)\s*", re.I)
_RANGE_GAP_EN = re.compile(r"\s*(?:and|to)\s*", re.I)
_RESPECTIVELY = re.compile(r"分别|\brespectively\b", re.I)

_FORECAST = re.compile(
    r"预计|预期|预测|有望|目标价|明年|后年|明天|将(?!近)|会(?!计|议)|\bwill\b|\bforecasts?\b|\bexpect(?:s|ed)?\b|"
    r"\btarget price\b|\bnext (?:year|quarter|month|week)\b|\bcould\b|\bmight\b|\bwould\b",
    re.I,
)
_MULTI_DAY = re.compile(
    r"今年|年初|年内|本周|本月|本季|上周|上月|去年|近\s*[\d一二三四五六七八九十两几]+\s*个?\s*(?:交易日|日|天|周|月|年|季度)|"
    r"过去|累计|区间|以来|\bthis (?:year|week|month|quarter)\b|\byear[- ]to[- ]date\b|\bYTD\b|"
    r"\b(?:past|last)\s+(?:\d+\s+)?(?:days?|weeks?|months?|years?)\b|\bsince\b",
    re.I,
)
_YEAR = re.compile(
    r"((?:19|20)\d{2})\s*(?:年|财年|年度)|\bFY\s*((?:19|20)\d{2})\b|"
    r"\b(?:in|for|during)\s+((?:19|20)\d{2})\b|((?:19|20)\d{2})\s*(?:annual|fiscal|full[- ]year)",
    re.I,
)
_PERIOD_MONTH: tuple[tuple[re.Pattern[str], str], ...] = (
    (_words(r"一季度|第一季度|一季报|\bQ1\b|first quarter"), "03"),
    (_words(r"上半年|中报|半年报|半年度|\bH1\b|first half|interim"), "06"),
    (_words(r"三季度|前三季度|三季报|\bQ3\b|third quarter|nine months"), "09"),
    (_words(r"全年|年报|年度|\bannual\b|full[- ]year|\bFY\b"), "12"),
)
_MOVE = re.compile(
    r"大涨|大跌|收涨|收跌|上涨|下跌|走高|走低|下挫|上扬|涨了|跌了|(?<!涨)跌(?![幅破到至])|涨(?![跌幅到至])|"
    r"\b(?:rose|rise[sn]?|fell|fall(?:s|en)?|gained|gain|dropped|drop|declined|decline|climbed|climb|"
    r"slipped|rallied|slumped|went (?:up|down)|go (?:up|down)|(?:was|is|closed) (?:up|down|higher|lower))\b",
    re.I,
)
_DOWN_MOVE = re.compile(
    r"跌|下挫|走低|\b(?:fell|fall(?:s|en)?|dropped|drop|declined|decline|slipped|slumped|went down|go down)\b|"
    r"\b(?:was|is|closed) (?:down|lower)\b",
    re.I,
)
_CLAUSE_BREAK = re.compile(r"[，,。；;！!？?\n]|\bbut\b|而且|并且|但是|同时|\band\b(?!\s*[-+]?\d)")
_SENTENCE_BREAK = re.compile(r"[。；;！!？?\n]")
_INDEX_NAMES = re.compile(
    r"(?:沪深|中证|上证|深证|创业板|科创|北证|国证)\s*\d{2,4}|\bCSI\s*\d{3,4}|\b(?:SSE|STAR)\s*50\b", re.I
)
_TICKER = re.compile(r"(?<![\d.])\d{6}(?:\.(?:SH|SZ|BJ))?(?![\d.]|\s*(?:元|亿|万|倍|%|点))", re.I)
_METRIC_DIGITS = re.compile(r"(?<![A-Za-z])(P/?E|P/?B|ROE|EPS)(?=[-+]?\d)", re.I)
_LETTER_UNITS = re.compile(r"(\d)(?=(?:x|X|bn|mn|tn|pct|k)(?![A-Za-z]))")
_FULLWIDTH = str.maketrans("０１２３４５６７８９．％＋－", "0123456789.%+-")

_CN_DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9), strict=True))
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_NUMERAL = r"[零〇一二两三四五六七八九十百千]+(?:点[零〇一二三四五六七八九]+)?"
_CN_BEFORE_UNIT = re.compile(rf"({_CN_NUMERAL})(?=倍|%|元|块|亿|万|个百分点|成|点(?![零〇一二三四五六七八九]))")
_CN_PERCENT = re.compile(rf"百分之\s*({_CN_NUMERAL}|\d+(?:\.\d+)?)")
_TENTHS = re.compile(r"(\d+(?:\.\d+)?)成(?![交本功为])")

_LOOK_BEHIND = 40
_LOOK_AHEAD = 8
_REL_TOLERANCE = 0.02
_APPROX_TOLERANCE = 0.05


class ClaimCheck(BaseModel):
    target: str | None = None
    metric: str | None = None
    claimed: float
    claimed_high: float | None = Field(default=None, description="Upper bound of a range claim ('20到30倍').")
    claimed_unit: str | None = None
    comparator: Comparator = Field(
        default="eq",
        description="How the claim relates to the value: eq, ne (negated), gt, ge, lt, le, approx, range.",
    )
    negated: bool = Field(default=False, description="The claim was negated ('不是15倍', 'did not fall').")
    actual: float | None = None
    status: Status
    reason: Reason | None = Field(default=None, description="Why a check is unverifiable (machine-readable).")
    evidence_id: str | None = None
    source: str | None = None
    as_of: str | None = None
    as_of_basis: str | None = Field(
        default=None,
        description="What as_of is: trade_date (price), valuation_date (P/E, P/B), or report_date.",
    )
    note: str = ""


class ClaimReport(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "partially_supported", "unverifiable"]
    checks: list[ClaimCheck] = Field(default_factory=list)
    targets: list[dict[str, Any]] = Field(default_factory=list)
    evidence_sources: list[dict[str, Any]] = Field(default_factory=list)
    disclaimer: str


@dataclass
class _Number:
    """One claimed number (or number-less move) as read from the claim."""

    start: int
    end: int
    value: float
    rounding: float
    scales: tuple[float, ...]
    unit: str | None
    unit_class: str | None
    comparator: Comparator = "eq"
    negated: bool = False
    high: float | None = None
    metric: str | None = None
    unit_mismatch: bool = False
    target: dict[str, Any] | None = None
    reasons: list[tuple[Reason, str]] = field(default_factory=list)
    years: list[str] = field(default_factory=list)
    period_month: str | None = None


def check_claim(claim: str, *, service: Any, registry: ToolRegistry, zh: bool = True) -> ClaimReport:
    nlu = service.analyze_query(claim)
    targets: list[dict[str, Any]] = []
    for entity in nlu.get("entities") or []:
        listed = entity.get("symbol") and entity.get("entity_type") in {"stock", "etf", "fund", "index"}
        if listed and all(target["symbol"] != entity["symbol"] for target in targets):
            targets.append(
                {
                    "name": entity.get("canonical_name"),
                    "symbol": entity.get("symbol"),
                    "_mentions": [entity.get("mention"), entity.get("canonical_name")],
                }
            )
    numbers = read_numbers(claim, targets)
    wanted = {number.metric for number in numbers if not number.reasons}
    evidence = _fetch([number.target for number in numbers if number.target and not number.reasons], wanted, registry)
    checks = [_check(number, evidence) for number in numbers]
    statuses = {check.status for check in checks}
    if not checks or statuses == {"unverifiable"}:
        verdict = "unverifiable"
    elif statuses == {"supported"}:
        verdict = "supported"
    elif "supported" not in statuses:
        verdict = "contradicted"
    else:
        verdict = "partially_supported"
    disclaimer = (
        "核查只比对声明中的数字与所列数据源，不评价观点本身，也不构成投资建议。"
        if zh
        else "This check compares the claim's numbers with the listed data sources; it is not investment advice."
    )
    public_targets = [{"name": target["name"], "symbol": target["symbol"]} for target in targets]
    return ClaimReport(
        claim=claim,
        verdict=verdict,
        checks=checks,
        targets=public_targets,
        evidence_sources=[_source(item) for items in evidence.values() for item in items],
        disclaimer=disclaimer,
    )


# ---------------------------------------------------------------------------------------------------------
# Reading the claim
# ---------------------------------------------------------------------------------------------------------
def normalise(claim: str) -> str:
    """Full-width digits, Chinese numerals ("十五倍" → "15倍", "三成" → "30%"), "15x" → "15 x"."""
    text = claim.translate(_FULLWIDTH).replace("个百分点", "百分点")  # "42个" would read as a count
    text = _CN_PERCENT.sub(lambda m: f"{_cn_number(m.group(1))}%", text)
    text = _CN_BEFORE_UNIT.sub(lambda m: _cn_number(m.group(1)), text)
    text = _TENTHS.sub(lambda m: f"{_trim(float(m.group(1)) * 10)}%", text)
    text = _METRIC_DIGITS.sub(r"\1 ", text)
    return _LETTER_UNITS.sub(r"\1 ", text)


def _trim(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _cn_number(text: str) -> str:
    """ "二十四点六" → "24.6"; Arabic input is returned as is."""
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return text
    integer, _, decimals = text.partition("点")
    total, digit = 0, 0
    for char in integer:
        if char in _CN_DIGITS:
            digit = _CN_DIGITS[char]
        else:
            total += (digit or 1) * _CN_UNITS[char]
            digit = 0
    total += digit
    tail = "".join(str(_CN_DIGITS[char]) for char in decimals)
    return f"{total}.{tail}" if tail else str(total)


def _masked(text: str, targets: list[dict[str, Any]]) -> tuple[str, list[tuple[int, dict[str, Any]]]]:
    """Blank out target names, index names and tickers (their digits are not claims); target positions."""
    positions: list[tuple[int, dict[str, Any]]] = []
    lowered = text.lower()
    spans: list[tuple[int, int]] = []
    for target in targets:
        for mention in _surface_forms(target, lowered):
            start = lowered.find(mention.lower())
            while start >= 0:
                if not any(a <= start < b for a, b in spans):
                    spans.append((start, start + len(mention)))
                    positions.append((start, target))
                start = lowered.find(mention.lower(), start + len(mention))
    for pattern in (_INDEX_NAMES, _TICKER):
        spans.extend(match.span() for match in pattern.finditer(text))
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            chars[index] = "＠"
    return "".join(chars), sorted(positions, key=lambda item: item[0])


def _surface_forms(target: dict[str, Any], lowered: str) -> list[str]:
    """How the target may be written: the NLU mention, the canonical name, or the longest part of a Chinese
    name that occurs in the claim ("茅台" for 贵州茅台; the NLU reports the canonical name as the mention)."""
    names = sorted({str(item) for item in target.get("_mentions") or [] if item}, key=len, reverse=True)
    found = [name for name in names if name.lower() in lowered]
    if found:
        return found
    for name in names:
        if not re.fullmatch(r"[\u4e00-\u9fff]{3,}", name):
            continue
        for size in range(len(name) - 1, 1, -1):
            parts = [name[i : i + size] for i in range(len(name) - size, -1, -1)]  # suffixes first
            hit = next((part for part in parts if part in lowered), None)
            if hit:
                return [hit]
    return []


def _blank(text: str) -> str:
    """Remove dates, durations and indicator parameters, keeping every character position."""
    for pattern in (*_DATE_PATTERNS, *_PARAMETER_PATTERNS):
        text = pattern.sub(lambda match: " " * len(match.group(0)), text)
    return text


def _clause_bounds(text: str, position: int, pattern: re.Pattern[str] = _CLAUSE_BREAK) -> tuple[int, int]:
    start = 0
    for match in pattern.finditer(text, 0, position):
        start = match.end()
    after = pattern.search(text, position)
    return start, after.start() if after else len(text)


def _unit_at(text: str, end: int) -> tuple[str | None, str | None, int, str | None]:
    """Unit after a number, its class, the index after it, and a 多/余 ("30多倍") comparator."""
    rest = text[end : end + 24]
    post: str | None = None
    skip = 0
    if rest[:1] in {"多", "余"}:
        post, skip = "gt", 1
    match = _UNIT.match(rest[skip:])
    if not match:
        return None, None, end + skip, post
    unit = match.group(1)
    unit_class = next((name for pattern, name in _UNIT_CLASS if pattern.match(unit)), None)
    return unit, unit_class, end + skip + match.end(), post


def _scales(unit: str | None, unit_class: str | None) -> tuple[float, ...]:
    if unit_class != _AMOUNT or unit is None:
        return _BARE_SCALES
    return next((scales for pattern, scales in _UNIT_SCALES if pattern.search(unit)), _BARE_SCALES)


def read_numbers(claim: str, targets: list[dict[str, Any]]) -> list[_Number]:
    """Every claimed number (and number-less move) with metric, target, comparator and context flags."""
    norm, positions = _masked(normalise(claim), targets)
    text = _blank(norm)
    raw = []
    for match in _NUMBER_TOKEN.finditer(text):
        token = match.group(0).replace(",", "")
        try:
            value = float(token)
        except ValueError:
            continue
        unit, unit_class, unit_end, post = _unit_at(text, match.end())
        decimals = len(token.split(".")[1]) if "." in token else 0
        raw.append(
            _Number(
                start=match.start(),
                end=unit_end,
                value=value,
                rounding=0.5 * 10**-decimals,
                scales=_scales(unit, unit_class),
                unit=unit,
                unit_class=unit_class,
                comparator=post or "eq",
            )
        )
    numbers = _merge_ranges(text, raw)
    previous_end = 0
    previous: _Number | None = None
    for number in numbers:
        clause_start, clause_end = _clause_bounds(text, number.start)
        stretch = text[max(clause_start, previous_end) : number.start]
        before = text[max(clause_start, number.start - _LOOK_BEHIND) : number.start]
        after = text[number.end : min(clause_end, number.end + _LOOK_AHEAD)]
        _read_comparator(number, stretch, after)
        # "下跌0.18%" / "fell 0.18%" is -0.18 (a written "-0.18" is already negative).
        if number.value > 0 and number.comparator != "range" and _direction(before) == -1:
            number.value = -number.value
        _metric_for(number, before, after, norm[clause_start:clause_end], previous)
        _context(number, norm, clause_start, clause_end)
        previous_end = number.end
        previous = number if number.metric else previous
    numbers.extend(_moves(text, norm, [(number.start, number.end) for number in numbers]))
    numbers.sort(key=lambda item: item.start)
    _bind_targets(numbers, positions, targets, text)
    for number in numbers:
        if number.target is None and not number.reasons:
            number.reasons.append(("no_target", "no listed company, fund or index"))
    return numbers


def _direction(before: str) -> int | None:
    """-1/+1 for the move word just before a number ("收跌0.53%", "fell 0.18%"), else the verifier's rule."""
    window = before[-8:].replace("涨跌幅", "").replace("涨跌", "")
    moves = list(_MOVE.finditer(window))
    if moves:
        return -1 if _DOWN_MOVE.search(moves[-1].group(0)) else 1
    return _stated_sign("", before)


def _merge_ranges(text: str, numbers: list[_Number]) -> list[_Number]:
    merged: list[_Number] = []
    index = 0
    while index < len(numbers):
        number = numbers[index]
        following = numbers[index + 1] if index + 1 < len(numbers) else None
        if following is not None:
            gap = text[number.end : following.start]
            head = text[max(0, number.start - 12) : number.start]
            is_range = bool(_RANGE_GAP.fullmatch(gap)) or (
                bool(_RANGE_GAP_EN.fullmatch(gap)) and re.search(r"\bbetween\b\s*$", head, re.I) is not None
            )
            compatible = number.unit_class in {None, following.unit_class}
            if is_range and compatible and following.value >= number.value >= 0:
                merged.append(
                    _Number(
                        start=number.start,
                        end=following.end,
                        value=number.value,
                        high=following.value,
                        rounding=min(number.rounding, following.rounding),
                        scales=following.scales,
                        unit=following.unit,
                        unit_class=following.unit_class,
                        comparator="range",
                    )
                )
                index += 2
                continue
        merged.append(number)
        index += 1
    return merged


def _read_comparator(number: _Number, stretch: str, after: str) -> None:
    if number.comparator == "range":
        comparator: str = "range"
    elif number.comparator != "eq":  # "30多倍"
        comparator = number.comparator
    else:
        comparator = next((name for name, pattern in _POST_COMPARATOR if pattern.search(after)), "eq")
    if comparator == "eq":
        # The comparator word closest to the number ("不是超过" is still about "超过").
        best: tuple[int, str] | None = None
        for name, pattern in _COMPARATOR_WORDS:
            for match in pattern.finditer(stretch):
                if best is None or match.end() > best[0]:
                    best = (match.end(), name)
        if best is not None:
            comparator = best[1]
    if _NEGATION.search(stretch):
        number.negated = True
        if comparator != "range":
            comparator = _FLIP[comparator]
    number.comparator = comparator  # type: ignore[assignment]


def _nearest_metrics(before: str, after: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for name, metric in _METRICS.items():
        if metric.words is None:
            continue
        for match in metric.words.finditer(before):
            found.append((len(before) - match.end(), name))
        match = metric.words.search(after)
        # Ties go to the word before the number ("市盈率15倍"), which is how claims are usually written.
        if match:
            found.append((match.start() + 1, name))
    return sorted(found)


def _metric_for(
    number: _Number,
    before: str,
    after: str,
    clause: str,
    previous: _Number | None,
) -> None:
    """Metric of a number: the nearest metric word in its clause whose unit fits; growth for a percentage next
    to revenue / net profit; otherwise the previous number's metric for a parallel clause."""
    found = _nearest_metrics(before, after)  # both are cut at the clause boundaries
    compatible = [name for _distance, name in found if number.unit_class in _METRICS[name].units]
    metric: str | None = None
    if number.unit_class == _PERCENT or number.unit_class is None:
        # A percentage next to revenue / net profit is growth ("营收同比增长16%", "net profit up 15%").
        # The nearest metric word other than a move word must be the line item ("净利率48.8%" is a margin).
        named = [name for _distance, name in found if name != "pct_change_1d"]
        if named and named[0] in _GROWTH_OF and number.unit_class == _PERCENT:
            metric = _GROWTH_OF[named[0]]
    if metric is None and compatible:
        metric = compatible[0]
    if metric is None and found:
        metric = found[0][1]
        number.unit_mismatch = True
    if metric is None and number.unit_class == _POINTS:
        metric = "close"
    if metric is None and previous is not None and previous.metric:
        prior = previous.metric
        base = next((base for base, growth in _GROWTH_OF.items() if prior in {base, growth}), None)
        if number.unit_class == _PERCENT and base and _GROWTH_WORDS.search(clause):
            metric = _GROWTH_OF[base]  # "净利润850亿元，同比增长15%"
        elif number.unit_class == previous.unit_class and number.unit_class in _METRICS[prior].units:
            metric = prior  # "茅台市盈率24.6倍，五粮液20.9倍"
    number.metric = metric


def _context(number: _Number, norm: str, clause_start: int, clause_end: int) -> None:
    clause = norm[clause_start:clause_end]
    if number.metric is None:
        number.reasons.append(("no_metric", "metric not recognised"))
        return
    if _FORECAST.search(clause) or _HYPOTHETICAL.search(clause):
        number.reasons.append(("forecast", "a forecast or hypothetical, not a reported fact"))
        return
    if number.metric == "pct_change_1d" and _MULTI_DAY.search(clause):
        number.reasons.append(("multi_day", "a multi-day move; only the latest daily change is available"))
        return
    if number.unit_mismatch:
        unit = number.unit or "?"
        number.reasons.append(("unit_mismatch", f"the unit '{unit}' does not fit {number.metric}"))
        return
    if number.metric in {"revenue", "net_profit", "market_cap"} and number.unit_class is None:
        number.reasons.append(("no_unit", "an amount without a unit (亿/万/billion)"))
        return
    # The year named in the number's clause, else earlier in its sentence ("2025年营收1741亿，净利润850亿").
    sentence_start, _sentence_end = _clause_bounds(norm, number.start, _SENTENCE_BREAK)
    for scope in (clause, norm[sentence_start : number.start]):
        number.years = [next(group for group in match.groups() if group) for match in _YEAR.finditer(scope)]
        if number.years:
            break
    number.period_month = next((month for pattern, month in _PERIOD_MONTH if pattern.search(clause)), None)


def _moves(text: str, norm: str, taken: list[tuple[int, int]]) -> list[_Number]:
    """Number-less moves: "昨天下跌了" (< 0), "并没有跌" (≥ 0), "did not fall"."""
    moves: list[_Number] = []
    for match in _MOVE.finditer(text):
        clause_start, clause_end = _clause_bounds(text, match.start())
        if any(clause_start <= start < clause_end for start, _end in taken):
            continue  # the clause states a number: the move is its direction
        if any(clause_start <= move.start < clause_end for move in moves):
            continue
        clause = norm[clause_start:clause_end]
        comparator: Comparator = "lt" if _DOWN_MOVE.search(match.group(0)) else "gt"
        negated = bool(_NEGATION.search(text[clause_start : match.start()]))
        number = _Number(
            start=match.start(),
            end=match.end(),
            value=0.0,
            rounding=0.0,
            scales=_BARE_SCALES,
            unit="%",
            unit_class=_PERCENT,
            comparator=_FLIP[comparator] if negated else comparator,
            negated=negated,
            metric="pct_change_1d",
        )
        if _FORECAST.search(clause) or _HYPOTHETICAL.search(clause):
            number.reasons.append(("forecast", "a forecast or hypothetical, not a reported fact"))
        elif _MULTI_DAY.search(clause):
            number.reasons.append(("multi_day", "a multi-day move; only the latest daily change is available"))
        moves.append(number)
    return moves


def _bind_targets(
    numbers: list[_Number], positions: list[tuple[int, dict[str, Any]]], targets: list[dict[str, Any]], text: str
) -> None:
    """Each number belongs to the nearest target named before it ("分别" assigns targets in order)."""
    if not targets:
        return
    placed = {id(target) for _position, target in positions}
    unplaced = [target for target in targets if id(target) not in placed]
    for number in numbers:
        before = [target for position, target in positions if position < number.start]
        # No target named before the number: the first one the NLU found but we could not place (NLU order
        # is the order of appearance), else the first one named after it.
        number.target = before[-1] if before else (unplaced or [target for _p, target in positions] or targets)[0]
    # "茅台和五粮液PE分别为24.6倍和20.9倍": the k-th number goes to the k-th target named.
    for match in _RESPECTIVELY.finditer(text):
        start, end = _clause_bounds(text, match.start(), _SENTENCE_BREAK)
        in_clause = [number for number in numbers if start <= number.start < end and number.start > match.start()]
        named = list(dict.fromkeys(id(target) for position, target in positions if start <= position < match.start()))
        ordered = [next(target for _p, target in positions if id(target) == key) for key in named]
        if len(ordered) > 1 and len(ordered) == len(in_clause):
            for number, target in zip(in_clause, ordered, strict=True):
                number.target = target


# ---------------------------------------------------------------------------------------------------------
# Fetching and comparing
# ---------------------------------------------------------------------------------------------------------
def _fetch(
    targets: list[dict[str, Any]], metrics: set[str | None], registry: ToolRegistry
) -> dict[str, list[AgentEvidence]]:
    evidence: dict[str, list[AgentEvidence]] = {}
    wanted_market = any(metric and not _METRICS[metric].fundamental for metric in metrics)
    wanted_fundamental = any(metric and _METRICS[metric].fundamental for metric in metrics)
    for target in list({target["symbol"]: target for target in targets}.values())[:3]:
        items: list[AgentEvidence] = []
        if wanted_market:
            result = registry.run("get_price_history", {"target": target["symbol"]})
            items.extend(result.evidence if result.ok else [])
        if wanted_fundamental:
            result = registry.run("get_fundamentals", {"target": target["symbol"]})
            items.extend(
                item for item in (result.evidence if result.ok else []) if item.evidence_id.startswith("fundamental_")
            )
        evidence[target["symbol"]] = items
    return evidence


def _check(number: _Number, evidence: dict[str, list[AgentEvidence]]) -> ClaimCheck:
    base = ClaimCheck(
        target=number.target["name"] if number.target else None,
        metric=number.metric,
        claimed=number.value,
        claimed_high=number.high,
        claimed_unit=number.unit,
        comparator=number.comparator,
        negated=number.negated,
        status="unverifiable",
    )
    if number.reasons:
        reason, note = number.reasons[0]
        return base.model_copy(update={"reason": reason, "note": note})
    assert number.metric is not None and number.target is not None
    metric = _METRICS[number.metric]
    found = None
    for item in evidence.get(number.target["symbol"]) or []:
        for key in metric.keys:
            actual = item.payload.get(key)
            if isinstance(actual, int | float) and not isinstance(actual, bool) and math.isfinite(actual):
                found = (item, float(actual))
                break
        if found:
            break
    if found is None:
        if number.metric in {"revenue_yoy", "netprofit_yoy"}:
            note = "the fundamentals source has no year-on-year growth for this item"
            return base.model_copy(update={"reason": "growth_unavailable", "note": note})
        return base.model_copy(update={"reason": "no_data", "note": "no data for this metric"})
    item, actual = found
    as_of, basis = _as_of(item, number.metric)
    base = base.model_copy(
        update={
            "actual": actual,
            "evidence_id": item.evidence_id,
            "source": _source(item)["source_name"],
            "as_of": as_of,
            "as_of_basis": basis,
        }
    )
    mismatch = _period_mismatch(number, item) if metric.fundamental else None
    if mismatch:
        return base.model_copy(update={"reason": "period_mismatch", "note": mismatch})
    return base.model_copy(update={"status": _compare(number, actual), "note": _interim_note(number, item)})


def _as_of(item: AgentEvidence, metric: str) -> tuple[str | None, str | None]:
    if item.source_type == "market_api" or item.evidence_id.startswith("price_"):
        return item.as_of, "trade_date"
    payload = item.payload
    if metric in {"pe_ttm", "pb", "dividend_yield", "market_cap"}:
        provenance = payload.get("valuation_provenance")
        valuation_date = payload.get("valuation_date") or (
            provenance.get("as_of") if isinstance(provenance, dict) else None
        )
        if valuation_date:
            return str(valuation_date), "valuation_date"
    report_date = payload.get("report_date") or item.as_of
    return (str(report_date) if report_date else None), "report_date"


def _period_mismatch(number: _Number, item: AgentEvidence) -> str | None:
    report_date = str(item.payload.get("report_date") or item.as_of or "")
    if not re.match(r"\d{4}-\d{2}", report_date):
        return None
    if number.years and report_date[:4] not in number.years:
        return f"the claim is about {', '.join(number.years)}; the data is for {report_date}"
    if number.period_month and report_date[5:7] != number.period_month:
        return f"the claim is about another reporting period; the data is for {report_date}"
    interim = report_date[5:7] in {"03", "06", "09"}
    if interim and number.metric in _LEVELS and not number.years and not number.period_month:
        # "茅台营收1741亿" is usually the annual figure; comparing it with a half-year total would be wrong.
        return f"the claim names no period and the latest report is year to date ({report_date})"
    return None


def _interim_note(number: _Number, item: AgentEvidence) -> str:
    """Growth from an interim report is year to date: say so when the claim names no period."""
    report_date = str(item.payload.get("report_date") or "")
    interim = re.match(r"\d{4}-(?:03|06|09)-", report_date) is not None
    if number.metric in _GROWTH_OF.values() and interim and not number.period_month:
        return f"compared with the report for the period ending {report_date} (year to date)"
    return ""


def _compare(number: _Number, actual: float) -> Status:
    metric = _METRICS[number.metric or ""]
    scales = number.scales
    if number.unit_class in {_PERCENT, None} and metric.fraction and abs(actual) <= 1.5:
        scales = (1.0, 100.0)
    reference = number.high if number.comparator == "range" and number.high is not None else number.value
    expected = actual * _nearest_scale(actual, reference, scales)
    tolerance = max(number.rounding, abs(expected) * _REL_TOLERANCE) + 1e-9
    claimed = number.value
    comparator = number.comparator
    if comparator in {"eq", "ne", "approx"}:
        if comparator == "approx":
            tolerance = max(number.rounding, abs(expected) * _APPROX_TOLERANCE) + 1e-9
        same_direction = number.metric != "pct_change_1d" or (claimed >= 0) == (expected >= 0) or expected == 0
        close = same_direction and abs(claimed - expected) <= tolerance
        holds = not close if comparator == "ne" else close
    elif comparator == "gt":
        holds = expected > claimed
    elif comparator == "ge":
        holds = expected >= claimed
    elif comparator == "lt":
        holds = expected < claimed
    elif comparator == "le":
        holds = expected <= claimed
    else:  # range
        inside = claimed <= expected <= (number.high if number.high is not None else claimed)
        holds = not inside if number.negated else inside
    return "supported" if holds else "contradicted"


def _nearest_scale(actual: float, claimed: float, scales: tuple[float, ...]) -> float:
    """The unit conversion that puts the evidence value closest to the claim (1741亿 vs 174120000000)."""
    if not actual or not claimed:
        return scales[0] if 1.0 not in scales else 1.0
    return min(scales, key=lambda scale: abs(math.log(abs(actual * scale)) - math.log(abs(claimed))))


def _source(item: AgentEvidence) -> dict[str, Any]:
    provenance = item.payload.get("provenance") if isinstance(item.payload.get("provenance"), dict) else {}
    return {
        "evidence_id": item.evidence_id,
        "source_name": item.source_name or provenance.get("source_label") or item.provider or item.produced_by,
        "as_of": item.as_of,
        "title": item.title,
        "provenance": provenance
        or ({"mode": "snapshot", "is_live": False} if (item.source_name or "") in {"seed", "snapshot"} else None),
    }
