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
* **Bounded approximations** (round 10). "八百多亿" / "一千六百余亿" / "三倍多" is more than the number and less than
  the next step of its last significant digit (800-900亿, 3-4倍); "三成出头" is the lower half of that step (30%-35%).
* **Stated values and differences** (round 10). "比白酒行业平均的30倍低": P/E is quoted in 倍, so the 30 is the
  average the claim states (its own check) next to the comparison; a multiple needs a ratio cue ("是/只有…的N倍",
  "比…的N倍还高", a fraction). "茅台ROE比五粮液高出3.6个百分点" / "相差…" / "多赚…亿": the difference of the two.
* **Status.** ``eq`` matches within half a unit of the last written digit or 2% (for ``approx``: 5%, or half the
  step of the last significant digit when wider); ``ne`` is the opposite; bounds and ranges are literal.
  ``unverifiable`` when there is no target, no
  metric, no data, a unit that does not fit the metric, an amount with no unit, a forecast, a period
  other than the data's, or a multi-day move (only the daily change is available).

The verdict is deterministic (no LLM) and every check cites the evidence id, source and as-of date (for
P/E and P/B the valuation date when the source gives one, see ``as_of_basis``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field

from .coverage import METRICS as COVERAGE_METRICS
from .coverage import PROFIT_GROWTH_FIELDS
from .evidence import _NUMBER as _NUMBER_TOKEN
from .evidence import AgentEvidence
from .names import INDUSTRY_EN, english_aliases, english_name
from .tools import ToolRegistry
from .verifier import _BARE_SCALES, _DATE_PATTERNS, _HYPOTHETICAL, _PARAMETER_PATTERNS, _UNIT_SCALES, EN_MONTH

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
    "no_reference",
]

# ---------------------------------------------------------------------------------------------------------
# Metrics: payload keys, the words that name them, and the unit classes a number for them may carry.
# ---------------------------------------------------------------------------------------------------------
_PERCENT, _MULTIPLE, _PRICE, _AMOUNT, _POINTS, _BASIS = "percent", "multiple", "price", "amount", "points", "bp"


@dataclass(frozen=True)
class _Metric:
    keys: tuple[str, ...]
    words: re.Pattern[str] | None  # None: never named directly (growth, see _metric_for)
    units: frozenset[str | None]
    fraction: bool = False  # the payload may hold 0.33 for a claimed 33%
    fundamental: bool = True
    macro: str | None = None  # indicator family served by get_macro_indicators (CPI, PMI, M2, CN10Y, ...)
    directional: bool = False  # a change: "跌超1%" is about the size of the fall (see _compare)


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
            r"\bup\b|\bdown\b|\brose\b|\bfell\b|\bgained\b|\blost\b|\bdropped\b|\bjumped\b|\bslid\b|\bclimbed\b|"
            r"\bslipped\b|\bdipped\b|\bdeclined\b|\badvanced\b|\brallied\b|\bedged\b|\btumbled\b|\bplunged\b|"
            r"\bsoared\b|\bsurged\b"
        ),
        _RATIO,
        fundamental=False,
        directional=True,
    ),
    # Turnover of the latest session ("五粮液昨天成交14.5亿元", "turnover of CNY 1.45 bn"): an amount, never a rate.
    "amount": _Metric(
        ("amount",),
        _words(r"成交额|成交金额|成交(?![量价均])|交易额|\bturnover\b|\btrad(?:ed|ing) value\b"),
        frozenset({_AMOUNT}),
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
        # "净赚800多亿", "一年赚的钱" (round 9, E8): colloquial words for the year's net profit.
        _words(
            r"归母净利润|净利润|净利(?!率)|净赚|赚的钱|(?:一年|全年|年度?)能?赚了?|\bnet (?:profit|income|earnings)\b"
        ),
        frozenset({_AMOUNT, _PRICE, None}),
    ),
    "gross_margin": _Metric(
        ("gross_margin", "grossprofit_margin"), _words(r"毛利率|gross (?:profit )?margin"), _RATIO, fraction=True
    ),
    # The chat's vocabulary (coverage.METRICS): net margin is derived from net profit / revenue when not reported.
    "net_margin": _Metric(
        ("net_margin", "netprofit_margin"),
        _words(r"销售净利率|净利润率|净利率|net (?:profit )?margin"),
        _RATIO,
        fraction=True,
    ),
    "peg": _Metric(
        ("peg", "peg_ratio"), _words(r"(?<![A-Za-z])PEG(?![A-Za-z])|市盈增长比"), frozenset({_MULTIPLE, None})
    ),
    # Named so that a claim about them says why it cannot be checked instead of "metric not recognised".
    "ps": _Metric(
        ("ps_ttm", "ps"),
        _words(r"市销率|(?<![A-Za-z])P/?S(?![A-Za-z])|price[- ]to[- ]sales"),
        frozenset({_MULTIPLE, None}),
    ),
    "max_drawdown": _Metric(
        ("max_drawdown",), _words(r"最大回撤|\bmax(?:imum)? drawdown\b"), _RATIO, fundamental=False
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
    "revenue_yoy": _Metric(("revenue_yoy", "or_yoy", "tr_yoy"), None, _RATIO, directional=True),
    "netprofit_yoy": _Metric(("netprofit_yoy",), None, _RATIO, directional=True),
    # Macro series (get_macro_indicators): the words are matched on the unmasked text ("M2", "10年期").
    "cpi_yoy": _Metric(
        ("metric_value",),
        _words(r"(?<![A-Za-z])CPI(?![A-Za-z])|居民消费价格(?:指数)?|消费者物价(?:指数)?|consumer price(?:s| index)?"),
        _RATIO,
        fundamental=False,
        macro="CPI",
        directional=True,
    ),
    "ppi_yoy": _Metric(
        ("metric_value",),
        _words(r"(?<![A-Za-z])PPI(?![A-Za-z])|工业生产者出厂价格(?:指数)?|producer price(?:s| index)?"),
        _RATIO,
        fundamental=False,
        macro="PPI",
        directional=True,
    ),
    "pmi": _Metric(
        ("metric_value",),
        _words(r"(?<![A-Za-z])PMI(?![A-Za-z])|采购经理人?指数|purchasing managers'? index"),
        frozenset({_POINTS, None}),
        fundamental=False,
        macro="PMI",
    ),
    "m2_yoy": _Metric(
        ("metric_value",),
        _words(r"(?<![A-Za-z])M2(?![A-Za-z\d])|广义货币(?:供应量?)?|broad money|money supply"),
        _RATIO,
        fundamental=False,
        macro="M2",
        directional=True,
    ),
    "cn10y": _Metric(
        ("metric_value",),
        _words(
            r"(?:10|十)\s*年期?\s*国债(?:收益率|利率)?|国债收益率|(?<![A-Za-z])(?:CN)?10Y(?![A-Za-z])|"
            r"CGB yield|10[- ]year (?:(?:china |chinese )?(?:government |treasury )?(?:bond )?)?yield"
        ),
        _RATIO,
        fundamental=False,
        macro="CN10Y",
    ),
    "lpr_5y": _Metric(
        ("metric_value",),
        _words(r"5\s*年期?(?:以上)?\s*(?:的)?\s*(?:LPR|贷款市场报价利率)|5[- ]year (?:LPR|loan prime rate)"),
        _RATIO,
        fundamental=False,
        macro="LPR5Y",
    ),
    "lpr_1y": _Metric(
        ("metric_value",),
        _words(r"(?<![A-Za-z])LPR(?![A-Za-z])|贷款市场报价利率|loan prime rate"),
        _RATIO,
        fundamental=False,
        macro="LPR1Y",
    ),
    "gdp_yoy": _Metric(
        ("metric_value",),
        _words(r"(?<![A-Za-z])GDP(?![A-Za-z])|国内生产总值|经济增速"),
        _RATIO,
        fundamental=False,
        macro="GDP",
        directional=True,
    ),
}
_MACRO_NAMES = {
    "CPI": ("CPI 同比", "CPI YoY"),
    "PPI": ("PPI 同比", "PPI YoY"),
    "PMI": ("制造业 PMI", "Manufacturing PMI"),
    "M2": ("M2 同比", "M2 YoY"),
    "CN10Y": ("10 年期国债收益率", "China 10Y government bond yield"),
    "LPR1Y": ("1 年期 LPR", "1-year LPR"),
    "LPR5Y": ("5 年期以上 LPR", "5-year LPR"),
    "GDP": ("GDP 同比", "GDP YoY"),
}
# Industry snapshots (get_fundamentals' industry_sql item) name some metrics differently.
_INDUSTRY_KEYS = {"pe_ttm": ("pe_ttm", "pe"), "pb": ("pb",), "pct_change_1d": ("pct_change", "pct_change_1d")}
_GROWTH_OF = {"revenue": "revenue_yoy", "net_profit": "netprofit_yoy"}
_LEVELS = {"revenue", "net_profit", "eps"}  # flows: an interim report holds a year-to-date total
_DAILY = {"pct_change_1d", "amount"}  # one session's value: "本周" / "近5日" is a multi-day claim
_GROWTH_WORDS = re.compile(
    r"同比|环比|增速|增幅|增长|下降|下滑|减少|\bYoY\b|year[- ]on[- ]year|\bgrowth\b|\bgrew\b", re.I
)

_UNIT = re.compile(
    r"\s*(万亿|亿元|亿|千万|百万|万元|万|元|块|倍|%|个百分点|百分点|个?基点|点|x(?![A-Za-z])|times\b|"
    r"trillion|billion|million|bn\b|mn\b|tn\b|yuan\b|RMB\b|CNY\b|percentage points?\b|per ?cent\b|pct\b|"
    r"basis points?\b|bps?\b|points?\b|pts\b)",
    re.I,
)
_UNIT_CLASS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:个?基点|basis points?|bps?)$", re.I), _BASIS),  # a change in a rate: never a level
    (re.compile(r"^(?:%|个百分点|百分点|percentage points?|per ?cent|pct)$", re.I), _PERCENT),
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
            # "近10%": the stretch read for a number ends right before it, so 近 / ~ may end the text (round 10)
            r"约|大约|大概|将近|接近|差不多|近(?=\s*(?:\d|$))|\babout\b|\baround\b|\broughly\b|\bapproximately\b|"
            r"\bnearly\b|\balmost\b|\bsome\b|~(?=\s*(?:\d|$))"
        ),
    ),
)
_POST_COMPARATOR: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ge", _words(r"^\s*(?:或?以上|\bor (?:more|above|higher)\b|\+)")),
    ("le", _words(r"^\s*(?:或?以下|以内|\bor (?:less|below|lower)\b)")),
    ("approx", _words(r"^\s*(?:左右|上下)")),
    ("gt", _words(r"^\s*(?:多(?!少)|有余|出头|之?上方|之上)")),  # "三倍多", "两倍有余", "三成出头", "PMI在50上方"
    ("lt", _words(r"^\s*(?:之?下方|之下)")),
)
# "八百多亿", "一千六百余亿", "三倍多", "七倍有余": more than the number but less than the next step of its last
# significant digit (800多 is 800-900, 三倍多 is 3-4). "三成出头", "八百亿出头": in the lower half of that step
# (三成出头 is 30%-35%). Round 10 (F7): read as bounded approximations, not as an open "more than".
_OVER_WORD = re.compile(r"^\s*(?:(多(?!少)|余|有余)|(出头))")
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
    r"\btarget price\b|\bnext (?:year|quarter|month|week)\b|\bcould\b|\bmight\b|\bwould\b|(?-i:\bmay\b)",
    re.I,
)
_MULTI_DAY = re.compile(
    # "近5个交易日" is a multi-day move; "最近一个交易日" / "近1日" is the latest daily change.
    r"今年|年初|年内|本周|本月|本季|上周|上月|去年|"
    r"近\s*(?![1一]\s*个?\s*(?:交易日|日|天)(?![\d一二三四五六七八九十]))[\d一二三四五六七八九十两几]+\s*个?\s*"
    r"(?:交易日|日|天|周|月|年|季度)|"
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
    r"slipped|rallied|slumped|went (?:up|down)|go (?:up|down)|(?:was|is|closed) (?:up|down|higher|lower))\b"
    r"(?!\s+than\b)",  # "is higher than Wuliangye's" is a comparison, not a move
    re.I,
)
_DOWN_MOVE = re.compile(
    r"跌|下挫|走低|\b(?:fell|fall(?:s|en)?|dropped|drop|declined|decline|slipped|slumped|went down|go down)\b|"
    r"\b(?:was|is|closed) (?:down|lower)\b",
    re.I,
)
# Change words used of macro series only ("PMI回落了", "CPI回升"): for a share, 回落 is not a daily move.
_MACRO_MOVE = re.compile(r"回落|回升|上升|下降|下滑|下行|反弹|攀升|\b(?:increased|decreased|eased|picked up)\b", re.I)
# Of those, the words for a change from the previous reading ("CPI同比回升": the YoY rate rose, not "YoY > 0").
_MACRO_CHANGE = re.compile(r"回落|回升|反弹|攀升|下行|\b(?:eased|picked up)\b", re.I)
# Qualitative move words without a number (a documented convention, stated in the check's note):
# a big move is at least 3% in its direction, a small one less than 1%.
BIG_MOVE_PCT, SMALL_MOVE_PCT = 3.0, 1.0
_BIG_MOVE = re.compile(
    r"大跌|暴跌|重挫|大幅(?:下跌|下挫|收跌|走低|上涨|收涨|走高|拉升)|跳水|大涨|暴涨|飙升|"
    r"\b(?:plung(?:e|es|ed|ing)|tumbl(?:e|es|ed|ing)|plummet(?:s|ed|ing)?|crash(?:es|ed|ing)?|"
    r"soar(?:s|ed|ing)?|surg(?:e|es|ed|ing)|skyrocket(?:s|ed|ing)?|(?:fell|dropped|rose|jumped) sharply)\b",
    re.I,
)
_SMALL_MOVE = re.compile(
    r"小幅(?:收)?(?:下跌|下挫|跌|走低|上涨|涨|走高)|微跌|微涨|小跌|小涨|"
    r"\b(?:edged|inched|ticked) (?:up|down|higher|lower)\b|\bdipped\b|"
    r"\b(?:fell|rose|dropped|gained|slipped) slightly\b|"
    r"\bslightly (?:lower|higher)\b",
    re.I,
)
_QUALITATIVE_DOWN = re.compile(
    r"跌|下挫|走低|跳水|\b(?:plung|tumbl|plummet|crash|fell|dropped|slipped|dipped)|\b(?:down|lower)\b", re.I
)
# "跌幅较前一交易日扩大": a comparison with the previous session's move, which the sources do not carry.
_WIDENING = re.compile(
    r"(?:跌幅|涨幅|降幅|增幅|跌势|涨势)[^，,。；;]{0,12}?(?:扩大|收窄|缩小|加大|加深)|"
    r"\b(?:losses|gains|decline|fall|rise)\s+(?:widened|narrowed|deepened)\b",
    re.I,
)
# Explicit dates ("4月22日", "2026-04-22", "April 22", "22 April"): a daily value is checked against its date.
_EN_MONTHS = {
    name: index
    for index, names in enumerate(
        (("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june")),
        start=1,
    )
    for name in names
} | {
    name: index
    for index, names in enumerate(
        (("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
         ("dec", "december")),
        start=7,
    )
    for name in names
}  # fmt: skip
_CLAIM_DATES = (
    re.compile(r"(?P<y>(?:19|20)\d{2})-(?P<m>\d{1,2})-(?P<d>\d{1,2})"),
    re.compile(r"(?:(?P<y>(?:19|20)\d{2})\s*年\s*)?(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*[日号]"),
    re.compile(rf"\b(?P<mn>{EN_MONTH})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y>(?:19|20)\d{{2}})\b)?"),
    re.compile(
        rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mn>{EN_MONTH})\b(?:,?\s+(?P<y>(?:19|20)\d{{2}})\b)?"
    ),
)
# "茅台的市净率是五粮液的1.5倍" / "Moutai's P/B is 1.5 times Wuliangye's": a multiple of another target's value.
_APPROX_WORDS = r"(?:约|大约|大概|接近|将近|差不多|近|约为|\babout\b|\baround\b|\broughly\b|\bnearly\b|\balmost\b)?"
# A bound may stand where the verb does: "营收不到五粮液的1.5倍", "超过五粮液的两倍", "至少是五粮液的1.2倍".
_RATIO_VERB = r"(?:是|为|相当于|达到?|等于|有|不到|不足|不及|低于|小于|少于|超过|超出|高于|大于|多于|至少|至多|最多|比)"
_RATIO_BEFORE = re.compile(
    rf"(?P<verb>{_RATIO_VERB})\s*{_APPROX_WORDS}\s*(?P<ref>＠+)\s*(?P<of>的|'s|’s)?\s*{_APPROX_WORDS}\s*$", re.I
)
# A multiple of an industry average (round 9): "市盈率只有白酒行业平均的三分之一", "不到行业均值的一半".
_RATIO_BEFORE_AVERAGE = re.compile(
    rf"(?P<verb>{_RATIO_VERB})\s*{_APPROX_WORDS}\s*(?P<ref>＠+)?\s*(?:的)?\s*(?:[\u4e00-\u9fff]{{1,4}}(?=行业|板块))?"
    rf"(?:行业|板块)?\s*(?:的)?"
    rf"(?:平均|均值)(?:水平|值)?\s*的\s*{_APPROX_WORDS}\s*$"
)
# "比五粮液的1.5倍还多": the comparison word after a multiple introduced by 比.
_RATIO_THAN = re.compile(r"\s*(?:还|更)?\s*要?\s*(多|高|大|贵|少|低|小|便宜)")
_RATIO_AFTER = re.compile(  # used with .match(text, pos): anchored at the end of the number
    r"\s*(?:(?:that|those) of\s+|as (?:high|large|big|much) as\s+)?(?:the\s+)?(?P<ref>＠+)", re.I
)
# Several targets sharing one claim: "茅台和五粮液都跌超0.5%", "Moutai and Wuliangye both fell ..."
_SHARED = re.compile(r"都|(?<![平人])均(?![值线价])|皆|全都|\bboth\b|\ball\b(?![\s-]+(?:of|time)\b)", re.I)
# Sector targets: a sector entity counts as a claim target only when written as a sector ("白酒板块").
_SECTOR_WORD = re.compile(
    r"\s*(?:板块|行业|指数|概念)|[\s-]*(?:sector|industry|stocks)\b", re.I
)  # used with .match(pos)
# The direction a word gives a change ("跌超1%", "fell more than 1%", "营收同比增长", "CPI同比下降").
_UP_WORDS = re.compile(
    r"大涨|收涨|上涨|涨幅|涨了|走高|上扬|上升|增长|增加|提高|反弹|攀升|回升|涨(?![跌停破到至])|"
    r"\b(?:rose|rise[sn]?|risen|gained|gains?|jumped|climbed|climbs?|rallied|increased?|grew|went up|go up|"
    r"(?:was|is|closed|ended) (?:up|higher))\b|\bup\b(?!\s+to\b)",
    re.I,
)
_DOWN_WORDS = re.compile(
    r"大跌|收跌|下跌|跌幅|跌了|下挫|走低|下降|下滑|减少|回落|下行|跌(?![停破到至])|"
    r"\b(?:fell|fall(?:s|en)?|dropped|drops?|declined|declines?|decreased?|slipped|slumped|lost|shed|went down|"
    r"go down|(?:was|is|closed|ended) (?:down|lower))\b|\bdown\b",
    re.I,
)
_DIRECTION_WINDOW = 20
_CLAUSE_BREAK = re.compile(r"[，,。；;！!？?\n]|\bbut\b|而且|并且|但是|同时|\band\b(?!\s*[-+]?\d)")
_SENTENCE_BREAK = re.compile(r"[。；;！!？?\n]")
_INDEX_NAMES = re.compile(
    r"(?:沪深|中证|上证|深证|创业板|科创|北证|国证)\s*\d{2,4}|\bCSI\s*\d{3,4}|\b(?:SSE|STAR)\s*50\b", re.I
)
_TICKER = re.compile(r"(?<![\d.])\d{6}(?:\.(?:SH|SZ|BJ))?(?![\d.]|\s*(?:元|亿|万|倍|%|点))", re.I)
# Digits inside names and dates that are not claims: "M2", "10Y", "3月CPI".
_NAME_DIGITS = re.compile(
    r"(?<![A-Za-z])M[012](?![A-Za-z\d])|(?<![A-Za-z])(?:CN)?10Y(?![A-Za-z])|(?<![\d.])\d{1,2}\s*月份?(?![\d日])", re.I
)
_MONTH = re.compile(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月份?(?![\d日])")
_TIMES_EARNINGS = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:x|times)\s+(?:(?:its|trailing|TTM|last year's)\s+)?(?:earnings|profits?)\b", re.I
)
_METRIC_DIGITS = re.compile(r"(?<![A-Za-z])(P/?E|P/?B|ROE|EPS)(?=[-+]?\d)", re.I)
_LETTER_UNITS = re.compile(r"(\d)(?=(?:x|X|bn|mn|tn|pct|k)(?![A-Za-z]))")
_FULLWIDTH = str.maketrans("０１２３４５６７８９．％＋－", "0123456789.%+-")

_CN_DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9), strict=True))
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_NUMERAL = r"[零〇一二两三四五六七八九十百千]+(?:点[零〇一二三四五六七八九]+)?"
# "一千六百多亿", "八百余亿", "三十多倍": 多 / 余 may stand between the numeral and its unit (round 10, F7).
_CN_BEFORE_UNIT = re.compile(
    rf"({_CN_NUMERAL})(?=(?:多|余)?(?:倍|%|元|块|亿|万|个百分点|成|点(?![零〇一二三四五六七八九])))"
)
_CN_PERCENT = re.compile(rf"百分之\s*({_CN_NUMERAL}|\d+(?:\.\d+)?)")
_TENTHS = re.compile(r"(\d+(?:\.\d+)?)成(?![交本功为])")

_LOOK_BEHIND = 40
_LOOK_AHEAD = 8
_REL_TOLERANCE = 0.02
_APPROX_TOLERANCE = 0.05


class ClaimCheck(BaseModel):
    target: str | None = None
    metric: str | None = None
    claimed: float | None = Field(
        default=None,
        description="The claimed number, with the sign its move word gives ('跌超1%' is -1); None for a relation.",
    )
    claimed_high: float | None = Field(default=None, description="Upper bound of a range claim ('20到30倍').")
    claimed_unit: str | None = None
    comparator: Comparator = Field(
        default="eq",
        description="How the claim relates to the value: eq, ne (negated), gt, ge, lt, le, approx, range.",
    )
    negated: bool = Field(default=False, description="The claim was negated ('不是15倍', 'did not fall').")
    direction: Literal["up", "down"] | None = Field(
        default=None,
        description=(
            "The direction a move word states for a change ('跌超1%' is down). A bound or range then applies "
            "to the size of the move in that direction, and a move the other way contradicts it."
        ),
    )
    reference: str | None = Field(
        default=None,
        description="What the target is compared with in a relational claim ('茅台PE比五粮液高': 五粮液).",
    )
    reference_value: float | None = None
    reference_evidence_id: str | None = None
    ratio: float | None = Field(
        default=None,
        description=(
            "For a multiple claim ('市净率是五粮液的1.5倍'): the target's value divided by the reference's. "
            "`claimed` is then the claimed multiple."
        ),
    )
    difference: float | None = Field(
        default=None,
        description=(
            "For a difference claim ('茅台ROE比五粮液高出3.6个百分点', round 10): the target's value minus the "
            "reference's (in percent of the reference's value when `kind` is 'relative_difference'). `claimed` is "
            "then the stated difference, signed by the stated direction (高出 +, 低 -)."
        ),
    )
    kind: Literal["value", "stated_reference", "relation", "ratio", "difference", "relative_difference"] = Field(
        default="value",
        description=(
            "What the check compares: a target's own value; 'stated_reference', the value the claim states for the "
            "compared side ('比白酒行业平均的30倍低': the average is 30x, round 10); a relation of two values; a "
            "multiple of the other side; or a stated (absolute or relative) difference of the two."
        ),
    )
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


class UncheckedClause(BaseModel):
    """A part of the claim that names a target or a metric but has no number, move or comparison to check."""

    text: str = Field(description="The clause as normalised for reading ('十五倍' is written '15倍').")
    reason: Literal["no_claim"] = "no_claim"
    note: str = "no number, move or comparison to check in this part of the claim"


class ClaimReport(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "partially_supported", "unverifiable"]
    checks: list[ClaimCheck] = Field(default_factory=list)
    targets: list[dict[str, Any]] = Field(default_factory=list)
    evidence_sources: list[dict[str, Any]] = Field(default_factory=list)
    unchecked: list[UncheckedClause] = Field(
        default_factory=list,
        description=(
            "Clauses that name a target or a metric but were not checked ('ROE很高'); the verdict is over "
            "`checks` only, and the UI lists these as 'not checked' rows."
        ),
    )
    coverage: Literal["full", "partial", "none"] = Field(
        default="full",
        description=(
            "How much of the claim the verdict covers (round 9, E2): 'full' when every part was checked and decided, "
            "'partial' when some part is unchecked or unverifiable, 'none' when nothing was decided. A 'supported' "
            "verdict with partial coverage means only that the checked numbers agree (the UI says 'partly checked')."
        ),
    )
    disclaimer: str
    labels_en: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "(round 11, G11) English for every Chinese target or reference label in the checks ('白酒行业平均' -> "
            "'baijiu (liquor) industry average'), from the server's name tables, so the English UI never shows them "
            "in Chinese"
        ),
    )


_INDUSTRY_LABEL = re.compile(r"^(?P<industry>[一-鿿]+?)(?:行业|板块)(?P<average>平均|均值|平均水平)?$")


def english_label(label: str) -> str | None:
    """The English for a check's target or reference label: a company ("贵州茅台" -> "Kweichow Moutai"), an industry
    or its average ("白酒行业平均" -> "baijiu (liquor) industry average"); ``None`` when there is none."""
    match = _INDUSTRY_LABEL.match(label or "")
    if match:
        industry = match.group("industry")
        english = INDUSTRY_EN.get(industry)
        if english:
            return f"{english} industry{' average' if match.group('average') else ''}"
    if label in INDUSTRY_EN:
        return INDUSTRY_EN[label]
    return english_name(label)


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
    month: tuple[str | None, str] | None = None  # (year, month) named for a macro value ("3月CPI")
    direction: int | None = None  # -1/+1 from a move word, for directional metrics
    relation: bool = False  # "茅台PE比五粮液高": compared with ``reference``, not with a number
    reference: dict[str, Any] | None = None
    reference_kind: Literal["target", "industry", "market", "macro"] | None = None
    ratio: bool = False  # "是五粮液的1.5倍": ``value`` is a multiple of the reference's value
    date: tuple[str | None, int, int] | None = None  # (year, month, day) named for a daily value
    convention: str | None = None  # the threshold behind a qualitative move word ("大跌": at least 3%)
    industry: bool = False  # "而行业平均11.8倍": the number is the target's industry average, not the target's
    # The unit of the last significant written digit (800 → 100, 24.6 → 0.1, 3 → 1): how precise a round number is.
    step: float = 1.0
    over: Literal["more", "just_over"] | None = None  # "八百多亿" / "三成出头" (see _OVER_WORD)
    # "比白酒行业平均的30倍低", "低于五粮液的20倍": the number is the value the claim states for the compared side
    stated_reference: bool = False
    # "茅台ROE比五粮液高出3.6个百分点": the number is the difference target - reference (round 10, F2);
    # "relative" when it is a percentage of the reference's value ("比行业平均低了近10%").
    difference: Literal["absolute", "relative"] | None = None
    difference_unsigned: bool = False  # "两者相差3.6个百分点": no direction stated


def check_claim(claim: str, *, service: Any, registry: ToolRegistry, zh: bool = True) -> ClaimReport:
    nlu = service.analyze_query(claim)
    targets: list[dict[str, Any]] = []
    for entity in nlu.get("entities") or []:
        listed = entity.get("symbol") and entity.get("entity_type") in {"stock", "etf", "fund", "index"}
        if listed and _vocabulary_mention(claim, str(entity.get("mention") or "")):
            continue  # "行业均值": "均值" is also an alias of an unrelated listed company (round 9, E9)
        if listed and all(target["key"] != entity["symbol"] for target in targets):
            targets.append(
                {
                    "name": entity.get("canonical_name"),
                    "symbol": entity.get("symbol"),
                    "key": entity.get("symbol"),
                    "_mentions": [entity.get("mention"), entity.get("canonical_name")],
                }
            )
        elif entity.get("entity_type") == "sector" and _written_as_sector(claim, entity):
            # "白酒板块跌超1%", "the baijiu industry average": checked against the industry snapshot.
            name = str(entity.get("canonical_name") or entity.get("mention"))
            if all(target["key"] != f"sector:{name}" for target in targets):
                targets.append(
                    {
                        "name": name,
                        "symbol": None,
                        "key": f"sector:{name}",
                        "sector": name,
                        "_mentions": [entity.get("mention"), name],
                    }
                )
    for name, phrase in _SECTOR_EN_PHRASE.items():
        # "the insurance industry", "brokerage stocks": English sector names the NLU does not always tag
        match = phrase.search(claim)
        if match and all(target.get("sector") != name for target in targets):
            targets.append({"name": name, "symbol": None, "key": f"sector:{name}", "sector": name, "_mentions": []})
    reading = _read(claim, targets, zh=zh)
    numbers = reading.numbers
    evidence = _fetch([number for number in numbers if not number.reasons], registry)
    checks = [_check(number, evidence, zh=zh) for number in numbers]
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
    unchecked = _unchecked(reading)
    decided = [check for check in checks if check.status != "unverifiable"]
    if not decided:
        coverage = "none"
    elif len(decided) < len(checks) or unchecked:
        coverage = "partial"
    else:
        coverage = "full"
    public_targets = [
        {"name": target["name"], "symbol": target["symbol"], "name_en": english_name(target["name"], target["symbol"])}
        for target in targets
        if target["symbol"]
    ]
    labels = {str(value) for check in checks for value in (check.target, check.reference) if value}
    labels_en = {
        label: english for label in sorted(labels) if re.search(r"[一-鿿]", label) and (english := english_label(label))
    }
    return ClaimReport(
        claim=claim,
        verdict=verdict,
        checks=checks,
        targets=public_targets,
        labels_en=labels_en,
        # One entry per evidence id: a sector target and a company's industry share one snapshot (round 9, E14).
        evidence_sources=list(
            {item.evidence_id: _source(item) for items in evidence.values() for item in items}.values()
        ),
        unchecked=unchecked,
        coverage=coverage,
        disclaimer=disclaimer,
    )


# ---------------------------------------------------------------------------------------------------------
# Reading the claim
# ---------------------------------------------------------------------------------------------------------
def _vocabulary_mention(claim: str, mention: str) -> bool:
    """The NLU matched a listed company on a word of the claim's own vocabulary: every occurrence of the mention
    lies inside an industry or market reference ("行业均值", "所属行业的平均水平") or a metric word. The alias table
    lists some such words as short names ("均值" for 武汉天源), and a number next to them would bind to that company."""
    if not mention:
        return False
    patterns = [_INDUSTRY_SUBJECT, _INDUSTRY_REFERENCE, _MARKET_REFERENCE, _STATED_AVERAGE_WORDS, _AVERAGE_ALONE]
    patterns += [metric.words for metric in _METRICS.values() if metric.words is not None]
    spans = [match.span() for pattern in patterns for match in pattern.finditer(claim) if match.end() > match.start()]
    starts = [index for index in range(len(claim)) if claim.startswith(mention, index)]
    return bool(starts) and all(
        any(low <= start and start + len(mention) <= high for low, high in spans) for start in starts
    )


def _written_as_sector(claim: str, entity: dict[str, Any]) -> bool:
    """A sector entity is a claim target when written as a sector ("白酒板块", "保险行业", "baijiu industry"),
    not when it only describes a company ("白酒龙头茅台")."""
    lowered = claim.lower()
    canonical = str(entity.get("canonical_name") or "")
    for name in (entity.get("mention"), canonical, *_sector_english(canonical)):
        if not name:
            continue
        if re.search(r"(?:板块|行业)$", str(name)):
            return True
        start = lowered.find(str(name).lower())
        while start >= 0:
            if _SECTOR_WORD.match(lowered, start + len(str(name))):
                return True
            start = lowered.find(str(name).lower(), start + 1)
    return False


# English names of the sectors with an industry snapshot ("the baijiu industry", "insurance stocks"), plus any
# English alias in the alias table.
_SECTOR_EN = {
    "白酒": ("baijiu", "liquor"),
    "保险": ("insurance", "insurers"),
    "证券": ("brokerage", "securities", "brokers"),
    "银行": ("banking", "banks"),
}
_SECTOR_SAME = {"券商": "证券"}  # the NLU's canonical sector name for another alias
_SECTOR_EN_PHRASE = {
    name: re.compile(rf"\b(?:{'|'.join(words)})[\s-]+(?:sector|industry|stocks)\b", re.I)
    for name, words in _SECTOR_EN.items()
}


@lru_cache(maxsize=64)
def _sector_english(name: str) -> tuple[str, ...]:
    from ..data_loader import load_aliases

    rows = load_aliases()
    ids = {row.get("entity_id") for row in rows if row.get("alias_text") == name}
    found = {str(row["alias_text"]) for row in rows if row.get("entity_id") in ids and str(row["alias_text"]).isascii()}
    found.update(_SECTOR_EN.get(_SECTOR_SAME.get(name, name), ()))
    return tuple(sorted(found, key=len, reverse=True))


_EN_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}  # fmt: skip
_EN_FRACTION = re.compile(
    r"\b(?:(half)|(a\s+quarter|one\s+quarter))\s+(?:of\s+)?(?:a|one)\s+(percent(?:age point)?|per cent)\b", re.I
)
_EN_WORD_NUMBER = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten)(?:\s+and\s+a\s+(half))?"
    r"(?=\s+(?:percent|per cent|percentage points?|times)\b)",
    re.I,
)
# "跌破荣枯线": the PMI's expansion/contraction line is 50 by definition.
_PMI_LINE = re.compile(
    r"荣枯线|荣枯分界线|枯荣线|\bthe\s+(?:50[- ]point\s+)?(?:boom[- ]bust|expansion[- ]contraction)\s+line\b", re.I
)


_EN_MULTIPLES = {"double": 2, "triple": 3, "quadruple": 4}
_EN_MULTIPLE = re.compile(r"(?<![A-Za-z-])(double|triple|quadruple)\b(?![- ]+digits?\b)", re.I)
_CN_FRACTION = re.compile(
    r"的\s*([一二两三四五六七八九十]+|[1-9]\d?)\s*分之\s*([一二两三四五六七八九十]+|[1-9]\d?)(?!\s*[倍%])"
)
_OF_TENTHS = re.compile(r"的\s*([一二两三四五六七八九]|[1-9](?:\.\d)?)\s*成(?![交本功为])")


def normalise(claim: str) -> str:
    """Full-width digits, Chinese numerals ("十五倍" → "15倍", "三成" → "30%"), English number words ("half a
    percent" → "0.5 percent", "three times" → "3 times", "double" → "2 times"), shares of another value ("的三分之一"
    → "的0.3333倍", "的六成" → "的0.6倍"), "荣枯线" → "50", "15x" → "15 x"."""
    text = claim.translate(_FULLWIDTH).replace("个百分点", "百分点")  # "42个" would read as a count
    text = _EN_FRACTION.sub(lambda m: f"{0.5 if m.group(1) else 0.25} {m.group(3)}", text)
    text = _EN_WORD_NUMBER.sub(
        lambda m: _trim(_EN_NUMBER_WORDS[m.group(1).lower()] + (0.5 if m.group(2) else 0.0)), text
    )
    text = re.sub(r"(?<![A-Za-z])twice\b(?!\s+(?:a|per)\b)", "2 times", text, flags=re.I)
    text = _EN_MULTIPLE.sub(lambda m: f"{_EN_MULTIPLES[m.group(1).lower()]} times", text)  # "more than double"
    text = _PMI_LINE.sub("50", text)
    text = re.sub(r"的一半", "的0.5倍", text)
    # A share of another value is a multiple (round 9): "的三分之一" → "的0.3333倍", "的六成" → "的0.6倍".
    text = _CN_FRACTION.sub(
        lambda m: f"的{_trim(round(float(_cn_number(m.group(2))) / float(_cn_number(m.group(1))), 4))}倍", text
    )
    text = _OF_TENTHS.sub(lambda m: f"的{_trim(float(_cn_number(m.group(1))) / 10)}倍", text)
    text = _CN_PERCENT.sub(lambda m: f"{_cn_number(m.group(1))}%", text)
    text = _CN_BEFORE_UNIT.sub(lambda m: _cn_number(m.group(1)), text)
    text = _TENTHS.sub(lambda m: f"{_trim(float(m.group(1)) * 10)}%", text)
    text = _TIMES_EARNINGS.sub(r"\1x P/E", text)  # "trades at 8.7 times earnings" is a P/E
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
    # Longest names first, over all targets: "平安银行" is placed before 中国平安's short form "平安" could take it.
    forms = sorted(
        ((mention, target) for target in targets for mention in _surface_forms(target, lowered)),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for mention, target in forms:
        start = lowered.find(mention.lower())
        while start >= 0:
            end = start + len(mention)
            # English names are whole words ("ping an" is not inside "ping an bank" twice).
            whole = not mention.isascii() or not (
                lowered[start - 1 : start].isalpha() or lowered[end : end + 1].isalpha()
            )
            if whole and not any(a <= start < b or a < end <= b for a, b in spans):
                spans.append((start, end))
                positions.append((start, target))
            start = lowered.find(mention.lower(), end)
    for pattern in (_INDEX_NAMES, _TICKER, _NAME_DIGITS):
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
        # The short form may be written too, in another clause ("五粮液净利润比中国平安多，…，平安ROE 15.2%"): a
        # number there belongs to this target, not to the previous one (round 9, E9).
        rest = lowered
        for name in found:
            rest = rest.replace(name.lower(), " " * len(name))
        return found + ([] if target.get("sector") else _short_forms(names, rest, suffix_only=True))
    aliases = _sector_english(str(target["name"])) if target.get("sector") else english_aliases(target.get("name"))
    english = [alias for alias in aliases if re.search(rf"\b{re.escape(alias.lower())}\b", lowered)]
    if english:
        return english  # "Moutai's P/E is higher than Wuliangye's"
    return _short_forms(names, lowered)


def _short_forms(names: list[str], lowered: str, *, suffix_only: bool = False) -> list[str]:
    """The longest part of a Chinese name of three or more characters that occurs in the text (suffixes first:
    "茅台" for 贵州茅台, "平安" for 中国平安). ``suffix_only`` next to the full name: "中国" is not 中国平安."""
    for name in names:
        if not re.fullmatch(r"[\u4e00-\u9fff]{3,}", name):
            continue
        for size in range(len(name) - 1, 1, -1):
            parts = [name[i : i + size] for i in range(len(name) - size, -1, -1)]  # suffixes first
            hit = next((part for part in parts[: 1 if suffix_only else None] if part in lowered), None)
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


def read_numbers(claim: str, targets: list[dict[str, Any]], *, zh: bool = True) -> list[_Number]:
    """Every claimed number, number-less move and relation ("茅台PE比五粮液高") with metric, target,
    comparator and context flags."""
    return _read(claim, targets, zh=zh).numbers


@dataclass
class _Reading:
    numbers: list[_Number]
    text: str  # normalised, names masked, dates blanked (same length as ``plain``)
    plain: str  # normalised claim
    positions: list[tuple[int, dict[str, Any]]]


def _read(claim: str, targets: list[dict[str, Any]], *, zh: bool = True) -> _Reading:
    plain = normalise(claim)
    norm, positions = _masked(plain, targets)
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
                step=_step(token),
                over="more" if post else None,  # "800多亿": 多 / 余 between the number and its unit
            )
        )
    numbers = _merge_ranges(text, raw)
    previous_end = 0
    previous: _Number | None = None
    averages: list[_Number] = []  # "低于3倍的行业平均水平": the relation with the industry (round 9, E1)
    shares = _share_values(claim)
    for number in numbers:
        clause_start, clause_end = _clause_bounds(text, number.start)
        stretch = text[max(clause_start, previous_end) : number.start]
        before = text[max(clause_start, number.start - _LOOK_BEHIND) : number.start]
        after = text[number.end : min(clause_end, number.end + _LOOK_AHEAD)]
        _read_comparator(number, stretch, after)
        if _difference(number, text, positions):
            pass  # "茅台ROE比五粮液高出约3.6个百分点": the difference of the two, checked in ``_check_difference``
        elif _ratio_reference(number, text, positions, shares):
            # "茅台的市净率大约是五粮液的1.5倍": the metric is the one compared, the 倍 is the multiple.
            sentence_start, sentence_end = _clause_bounds(text, number.start, _SENTENCE_BREAK)
            number.metric = _relation_metric(
                text, number.start, number.end, clause_start, clause_end
            ) or _relation_metric(text, number.start, number.end, sentence_start, sentence_end)
            if number.metric == "pct_change_1d":
                number.direction = _direction(text[clause_start : number.start])
        else:
            _metric_for(
                number,
                before,
                after,
                norm[clause_start:clause_end],
                previous,
                plain_before=plain[max(clause_start, number.start - _LOOK_BEHIND) : number.start],
                plain_after=plain[number.end : min(clause_end, number.end + _LOOK_AHEAD)],
            )
            # "跌超1%" / "fell more than 0.1%": the sign of the move word (a written "-0.18" is already negative).
            _apply_direction(number, stretch[-_DIRECTION_WINDOW:])
            number.industry = _industry_subject(text, number.start, positions, max(clause_start, previous_end))
            if number.metric is None:
                # "茅台和五粮液的市盈率，分别是24.6倍和20.9倍": the metric named earlier in the sentence (after the
                # previous number), when its clause names none.
                sentence_start, _sentence_end = _clause_bounds(text, number.start, _SENTENCE_BREAK)
                # (round 9: up to the number, so a metric beyond the look-behind window of its own clause counts:
                # "trades at a P/E under the baijiu industry average of 30x")
                earlier = text[max(sentence_start, previous_end) : number.start]
                fitting = [
                    name for _d, name in _nearest_metrics(earlier, "") if number.unit_class in _METRICS[name].units
                ]
                number.metric = fitting[0] if fitting else None
        _context(number, norm, clause_start, clause_end, plain)
        if not number.ratio:
            averaged = _stated_reference(number, text, max(clause_start, previous_end), positions, shares)
            if averaged is not None:
                averaged.reasons = list(number.reasons)
                averages.append(averaged)
        previous_end = number.end
        previous = number if number.metric and not (number.ratio or number.difference) else previous
    relations = _relations(text, plain, positions, [number.start for number in numbers]) + averages
    relations += _macro_relations(text, plain, [number.start for number in numbers], zh=zh)
    taken = [(number.start, number.end) for number in [*numbers, *relations]]
    numbers.extend(_moves(text, norm, taken, plain))
    numbers[:] = _bind_targets(numbers, positions, targets, text)
    _assign_dates([*numbers, *relations], plain)
    for number in numbers:
        metric = _METRICS[number.metric] if number.metric else None
        if metric and metric.macro:
            number.target = _macro_target(metric.macro, zh=zh)
    numbers.extend(relations)
    numbers.sort(key=lambda item: item.start)
    for number in numbers:
        if number.target is None and not number.reasons:
            number.reasons.append(("no_target", "no listed company, fund or index"))
    return _Reading(numbers, text, plain, positions)


def _unchecked(reading: _Reading) -> list[UncheckedClause]:
    """Clauses that name a target or a metric but produced no check ("茅台PE 24.6倍，ROE很高": "ROE很高"), so a
    report never drops part of a claim silently. A clause that only names what the rest of its sentence checks
    ("茅台和五粮液的市盈率，分别是24.6倍和20.9倍") is covered by those checks."""
    text, plain, numbers = reading.text, reading.plain, reading.numbers
    spans, start = [], 0
    for match in _CLAUSE_BREAK.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    found: list[UncheckedClause] = []
    for clause_start, clause_end in spans:
        words = plain[clause_start:clause_end].strip()
        if not words or any(clause_start <= number.start < clause_end for number in numbers):
            continue
        named = {target["key"] for position, target in reading.positions if clause_start <= position < clause_end}
        metrics = {name for _distance, name in _nearest_metrics(text[clause_start:clause_end], "")}
        metrics |= {name for _distance, name in _nearest_metrics(plain[clause_start:clause_end], "", macro=True)}
        if not named and not metrics:
            continue  # "是真的吗", "据说": nothing factual
        sentence_start, sentence_end = _clause_bounds(text, clause_start, _SENTENCE_BREAK)
        in_sentence = [number for number in numbers if sentence_start <= number.start < sentence_end]
        keys = {
            str(subject.get("key") or subject.get("macro"))
            for number in in_sentence
            for subject in (number.target, number.reference)
            if subject
        }
        checked = {number.metric for number in in_sentence}
        if named <= keys and metrics <= checked:
            continue
        found.append(UncheckedClause(text=words))
    return found


def _direction(window: str) -> int | None:
    """-1/+1 for the last move word before a number ("收跌0.53%", "跌超1%", "fell by more than 0.1%")."""
    window = window.replace("涨跌幅", "").replace("涨跌", "")
    ups = [match.end() for match in _UP_WORDS.finditer(window)]
    downs = [match.end() for match in _DOWN_WORDS.finditer(window)]
    if not ups and not downs:
        return None
    return 1 if max(ups, default=-1) > max(downs, default=-1) else -1


def _apply_direction(number: _Number, window: str) -> None:
    """A change with a move word: remember the direction and give the claimed number its sign."""
    if number.metric is None or not _METRICS[number.metric].directional:
        return
    direction = _direction(window)
    if direction is None:
        return
    number.direction = direction
    if direction == -1:
        number.value = -abs(number.value)
        if number.high is not None:
            number.high = -abs(number.high)


# "而行业平均11.8倍", "the industry average is 11.8x": the industry average as the subject of a number.
_INDUSTRY_SUBJECT = re.compile(
    r"(?:其?所在|所属|同)?(?:行业|板块|同行|同业)(?:的)?(?:平均|均值|中位数|整体)(?:水平)?|"
    r"\b(?:the\s+)?(?:industry|sector|peer)(?:'s)?\s+(?:average|median|mean)\b",
    re.I,
)
_BOUND_WORDS = tuple(pattern for name, pattern in _COMPARATOR_WORDS if name in {"gt", "ge", "lt", "le"})


def _industry_subject(text: str, number_start: int, positions: list[tuple[int, dict[str, Any]]], start: int) -> bool:
    """ "中国平安市盈率8.7倍，而行业平均11.8倍": an industry average named after the last target (and after the
    previous number) and before this number is what the number is about. Not when it is the other side of a
    comparison: "低于行业平均11.8倍" is a bound on the target's own value."""
    named = [position for position, _target in positions if start <= position < number_start]
    stretch = text[named[-1] if named else start : number_start]
    match = _INDUSTRY_SUBJECT.search(stretch)
    if match is None:
        return False
    lead = stretch[: match.start()]
    compared = _REL_BI.search(lead) or _REL_WORD.search(lead) or re.search(r"\bthan\b", lead, re.I)
    return not (compared or any(pattern.search(lead) for pattern in _BOUND_WORDS))


# An industry average stated with its number on the other side of a bound (round 9, E1): "低于3倍的行业平均水平",
# "低于白酒行业35倍的平均估值", "不到保险业均值11.8倍", "低于行业平均水平（3倍）", "below the sector average of 3x".
_AVERAGE_WORD = r"(?:平均|均值|中位数)(?:水平|估值|值|数)?"
_STATED_AVERAGE_WORDS = re.compile(
    rf"(?:其?所在|所属|同)?(?:行业|板块|同行|同业|(?<=[^\x00-\x7f])业)(?:的)?{_AVERAGE_WORD}|"
    r"\b(?:the\s+)?(?:(?:[a-z]+|＠+)[- ])?(?:industry|sector|peers?)(?:'s)?\s+(?:average|median|mean)\b",
    re.I,
)
# "…6.2倍的平均值": the average word on its own ("均值科技" as a name is not an average)
_AVERAGE_ALONE = re.compile(r"平均(?:值|水平|估值|数)?|(?<=的)均值|中位数")
# The number after the phrase: "行业平均11.8倍", "行业均值为1.45倍", "行业平均水平（3倍）", "sector average of 3x".
_AVERAGE_BEFORE_NUMBER = re.compile(r"\s*(?:的|为|是|约为?|在|of|at|:|：|\(|（)?\s*", re.I)
# The phrase after the number: "3倍的行业平均水平", "35倍的平均估值" (the sector named before the number).
_AVERAGE_AFTER_NUMBER = re.compile(
    rf"\s*(?:的)?\s*(?:＠+\s*)?(?:(?:行业|板块)(?:的)?)?{_AVERAGE_WORD}|\s*(?:(?:industry|sector)\s+)?average\b", re.I
)
_STATED_AVERAGE_METRICS = {"pe_ttm", "pb"}  # the metrics an industry snapshot carries as a level


# "低于五粮液的20倍", "比五粮液的15.2倍高", "高于白酒行业的30倍": a named target (or sector) and 的 between the
# comparison cue and a number in the metric's unit (round 10, F1): the value the claim states for that target.
_STATED_TARGET = re.compile(
    rf"\s*{_APPROX_WORDS}\s*(?P<ref>＠+)\s*(?:行业|板块)?\s*(?:的)?\s*(?P<average>{_AVERAGE_WORD})?\s*(?:的|'s|’s)\s*"
    rf"{_APPROX_WORDS}\s*",
    re.I,
)


def _stated_reference(
    number: _Number,
    text: str,
    stretch_start: int,
    positions: list[tuple[int, dict[str, Any]]],
    shares: set[float] | None = None,
) -> _Number | None:
    """ "中国平安PB低于3倍的行业平均水平": the stated average (3) is a fact about the industry, and the bound is a
    relation of the company with that industry. Before round 9 the 3 was read as a bound on the company's own value,
    so a made-up average passed. The number becomes the industry's (checked against the snapshot, like "而行业平均
    11.8倍"); the returned relation compares the company with the industry snapshot (or the named sector).

    Round 10 (F1): the cue may also be 比 with the comparison word after the number ("比白酒行业平均的30倍低不少"),
    and the compared side may be a named target ("茅台市盈率低于五粮液的20倍", "ROE比五粮液的29.4%高"): the number is
    that target's stated value, checked against its own data, next to the relation of the two. ``_ratio_reference``
    leaves these forms to this function when the number is in the metric's own unit and no ratio cue is written."""
    if number.metric is None or number.unit_mismatch or number.ratio or number.difference:
        return None
    if number.unit_class == _MULTIPLE and round(number.value, 4) in (shares or set()):
        return None  # "的三分之二" is a share of the other side, never a value stated for it
    stretch = text[stretch_start : number.start]
    comparator: str = number.comparator
    cue = None
    if comparator in {"gt", "ge", "lt", "le"}:
        for pattern in _BOUND_WORDS:
            for match in pattern.finditer(stretch):
                if cue is None or match.start() > cue.start():
                    cue = match
    elif comparator in {"eq", "approx"}:
        # "比白酒行业平均的30倍低不少": 比 before the compared side, the comparison word right after the number
        than = _RATIO_THAN.match(text, number.end)
        cue = None if than is None else next(reversed(list(_REL_BI.finditer(stretch))), None)
        if cue is not None and than is not None:
            comparator = "gt" if than.group(1) in {"多", "高", "大", "贵"} else "lt"
    if cue is None:
        return None
    between = stretch[cue.end() :]
    cue_at = stretch_start + cue.start()
    at = dict(positions)
    named = _STATED_TARGET.fullmatch(between)
    reference: dict[str, Any] | None = None
    if named is not None and stretch_start + cue.end() + named.start("ref") in at:
        reference = at[stretch_start + cue.end() + named.start("ref")]
        if named.group("average") and not reference.get("sector"):
            return None  # "五粮液平均的": not a stated value
        if reference.get("sector") and number.metric not in _INDUSTRY_KEYS:
            return None  # a sector's snapshot carries P/E, P/B and the daily change only
    else:
        if number.metric not in _STATED_AVERAGE_METRICS:
            return None
        phrase = _STATED_AVERAGE_WORDS.search(between)
        before = phrase is not None and _AVERAGE_BEFORE_NUMBER.fullmatch(between, phrase.end()) is not None
        after = _AVERAGE_AFTER_NUMBER.match(text, number.end) if not before else None
        if not before and after is None:
            return None
        # A sector named between the bound word and the average ("低于白酒行业35倍的平均估值") is the reference.
        reference = next(
            (target for position, target in positions if cue_at <= position < number.start and target.get("sector")),
            None,
        )
    ref_key = (reference or {}).get("key")
    companies = [
        target
        for position, target in positions
        if position < cue_at and not target.get("sector") and target.get("key") != ref_key
    ]
    if not companies:
        return None
    relation = _Number(
        start=cue_at,
        end=stretch_start + cue.end(),
        value=0.0,
        rounding=0.0,
        scales=_BARE_SCALES,
        unit=None,
        unit_class=None,
        comparator=comparator,  # type: ignore[arg-type]
        negated=number.negated,
        metric=number.metric,
        target=companies[-1],
        relation=True,
        reference=reference,
        reference_kind="target" if reference else "industry",
    )
    number.comparator = "approx" if number.comparator == "approx" else "eq"
    number.negated = False
    number.industry = reference is None
    number.stated_reference = True
    if reference is not None:
        number.target = reference
    return relation


# Round 10 (F2): a stated difference of two values. "茅台ROE比五粮液高出约3.6个百分点", "茅台比五粮液多赚四百多亿",
# "五粮液的PE比茅台低3.7倍左右" (P/E is quoted in 倍), "茅台的PE比行业平均低了近10%" (a percentage of a metric not
# quoted in percent is relative), and without a direction "茅台和五粮液的ROE差了3.6个百分点", "…相差约3.6个百分点".
_DIFF_LEAD = r"(?:(?:约|大约|大概|将近|接近|近|不到|不足|超过|超|逾|至少|起码|最多|至多|足足|整整|有|达到?)\s*)*"
_DIFF_AFTER_REFERENCE = re.compile(
    r"(?P<mid>[^，,。；;！!？?\d＠]{0,12}?)(?:还|更|要)*\s*"
    r"(?P<adj>高出|多出|多赚|少赚|高|多|大|贵|低|少|小|便宜)(?:了)?\s*" + _DIFF_LEAD + r"$"
)
_DIFF_SPREAD = re.compile(r"(?:相差|差距(?:为|是|有|达到?)?|差了|差)(?!不多)\s*" + _DIFF_LEAD + r"$")
_DIFF_UP = {"高出", "多出", "多赚", "高", "多", "大", "贵"}


def _difference(number: _Number, text: str, positions: list[tuple[int, dict[str, Any]]]) -> bool:
    """ "茅台ROE比五粮液高出约3.6个百分点": the number is the difference of the target's value and the reference's, not
    either value (before round 10 the 3.6 was read as 五粮液's own ROE and the true claim was contradicted). Sets
    ``difference``, ``reference``, the metric and the direction (高出 +1, 低 -1; none for "相差"); the subject is bound
    in ``_bind_targets``. The number keeps its comparator ("约", "不到", "四百多亿")."""
    if number.comparator == "range":
        return False
    clause_start, clause_end = _clause_bounds(text, number.start)
    sentence_start, sentence_end = _clause_bounds(text, number.start, _SENTENCE_BREAK)
    head = text[clause_start : number.start]
    direction: int | None = None
    reference: tuple[Literal["target", "industry", "market"], dict[str, Any] | None, int] | None = None
    cue_start = cue_end = -1
    adjective = ""
    for bi in reversed(list(_REL_BI.finditer(head))):
        found = _reference_at(text, clause_start + bi.end(), positions)
        if found is None:
            continue
        gap = _DIFF_AFTER_REFERENCE.fullmatch(text, found[2], number.start)
        if gap is None or "＠" in gap.group("mid"):
            continue
        reference, adjective = found, gap.group("adj")
        direction = 1 if adjective in _DIFF_UP else -1
        cue_start, cue_end = clause_start + bi.start(), clause_start + bi.end()
        break
    if reference is None:
        spread = _DIFF_SPREAD.search(head)
        if spread is None:
            return False
        cue_start, cue_end = clause_start + spread.start(), clause_start + spread.end()
        named: list[tuple[int, dict[str, Any]]] = []
        for position, target in positions:
            if sentence_start <= position < cue_start and all(target["key"] != other["key"] for _p, other in named):
                named.append((position, target))
        industry = None
        if named:
            industry = _INDUSTRY_REFERENCE.search(text, named[-1][0], cue_start)
            industry = industry if industry is not None and industry.end() > industry.start() else None
        if industry is not None:
            reference = ("industry", None, industry.end())
        elif len(named) >= 2:
            reference = ("target", named[-1][1], cue_start)
        else:
            return False
    kind, ref_target, _ref_end = reference
    subjects = [
        target
        for position, target in positions
        if sentence_start <= position < cue_start and target["key"] != (ref_target or {}).get("key")
    ]
    if not subjects:
        return False
    metric = _relation_metric(text, cue_start, cue_end, clause_start, clause_end) or _relation_metric(
        text, cue_start, cue_end, sentence_start, sentence_end
    )
    if metric is None and adjective in {"多赚", "少赚"}:
        metric = "net_profit"  # "多赚四百多亿": the year's net profit
    if metric is None or _METRICS[metric].macro:
        return False
    number.metric, number.relation, number.reference, number.reference_kind = metric, True, ref_target, kind
    if number.unit_class in _METRICS[metric].units:
        number.difference = "absolute"  # "高出3.6个百分点" of ROE, "低3.7倍" of P/E, "多赚400亿"
    elif number.unit_class == _PERCENT and number.unit == "%":
        number.difference = "relative"  # "PE比行业平均低了近10%": a percentage of the reference's value
    else:
        number.difference = "absolute"
        number.unit_mismatch = True  # "营收比五粮液高出两倍": 2 or 3 times? stated as a multiple, it is checkable
    if direction is not None and metric == "pct_change_1d" and re.search(r"跌", text[clause_start:clause_end]):
        direction = -direction  # "跌幅比五粮液大0.3个百分点": it fell more, a lower change
    number.difference_unsigned = direction is None
    number.direction = direction
    if direction is not None:
        number.value = direction * abs(number.value)
        number.high = None if number.high is None else direction * abs(number.high)
    return True


_RATIO_CUE_VERBS = {"是", "为", "相当于", "等于", "达", "达到", "有"}  # "是/为/相当于/只有…的N倍": a multiple


def _share_values(claim: str) -> set[float]:
    """The multiples that ``normalise`` writes for fraction and share words ("的一半" → 0.5, "的三分之一" → 0.3333,
    "的六成" → 0.6): a fraction of another value is always a multiple, never a stated value."""
    text = claim.translate(_FULLWIDTH)
    values = {0.5} if "的一半" in text else set()
    for match in _CN_FRACTION.finditer(text):
        values.add(round(float(_cn_number(match.group(2))) / float(_cn_number(match.group(1))), 4))
    for match in _OF_TENTHS.finditer(text):
        values.add(round(float(_cn_number(match.group(1))) / 10, 4))
    return values


def _in_metric_unit(number: _Number, text: str) -> bool:
    """The number is written in the unit of the metric compared (the metric named nearest in its clause, else in its
    sentence): "市盈率…的30倍" (P/E is quoted in 倍), "ROE…的29.4%"."""
    clause_start, clause_end = _clause_bounds(text, number.start)
    sentence_start, sentence_end = _clause_bounds(text, number.start, _SENTENCE_BREAK)
    metric = _relation_metric(text, number.start, number.end, clause_start, clause_end) or _relation_metric(
        text, number.start, number.end, sentence_start, sentence_end
    )
    return metric is not None and number.unit_class in _METRICS[metric].units


def _ratio_reference(
    number: _Number, text: str, positions: list[tuple[int, dict[str, Any]]], shares: set[float] | None = None
) -> bool:
    """ "是五粮液的1.5倍" / "1.5 times Wuliangye's": a multiple of a named target's value. Sets ``ratio`` and
    ``reference`` (the subject is bound in ``_bind_targets``). Since round 9 also "比五粮液的1.5倍还多", "是茅台的64%"
    (a share is a multiple), and a multiple of the industry average ("只有白酒行业平均的三分之一")."""
    share = number.unit_class == _PERCENT and number.comparator != "range"
    if not (number.unit_class == _MULTIPLE or share) or number.comparator == "range":
        return False
    if _AVERAGE_AFTER_NUMBER.match(text, number.end):
        return False  # "低于白酒行业35倍的平均估值": the industry average itself, not a multiple of it
    at = dict(positions)
    before = _RATIO_BEFORE.search(text[: number.start])
    after = None if share else _RATIO_AFTER.match(text, number.end)
    average = _RATIO_BEFORE_AVERAGE.search(text[: number.start])
    for match in (before, after, average):
        if match is None:
            continue
        ref_at = match.start("ref") if match.group("ref") else None
        if match is not average and ref_at not in at:
            continue
        if share and (match is not before or match.group("of") != "的"):
            continue  # "是茅台的64%" is a share; "比茅台高5%" is not a multiple
        # Round 10 (F1): when the number's unit is the metric's own unit (P/E and P/B are quoted in 倍, ROE in %),
        # "X的N倍" is ambiguous: X's value (N倍) or N times X's value. It is a multiple only with an explicit ratio cue:
        # a ratio verb ("是/为/相当于/只有…的N倍"), 还/更 after a 比 comparison ("比…的1.5倍还高"), a fraction or share
        # word ("一半", "三分之一", "六成"), or English "N times X's". Otherwise it is the value the claim states for X
        # ("比白酒行业平均的30倍低": the average is 30x), checked as such by ``_stated_reference``.
        explicit = match is after or round(number.value, 4) in (shares or set())
        explicit = explicit or (match is not after and match.group("verb") in _RATIO_CUE_VERBS)
        than = None
        if match is not after and match.group("verb") == "比":
            than = _RATIO_THAN.match(text, number.end)
            if than is None:
                continue  # "比五粮液的1.5倍" needs 多/高/少/低 after it
            explicit = explicit or bool(re.match(r"\s*(?:还|更)", than.group(0)))
        if not explicit and _in_metric_unit(number, text):
            continue  # "市盈率24.6倍比五粮液的15.2倍高": 五粮液's P/E, not 15.2 times it
        if than is not None:
            number.comparator = "gt" if than.group(1) in {"多", "高", "大", "贵"} else "lt"
        number.ratio = number.relation = True
        if match is average and ref_at not in at:
            number.reference, number.reference_kind = None, "industry"
        else:
            number.reference, number.reference_kind = at[ref_at], "target"
        if share:
            number.value, number.rounding = number.value / 100, number.rounding / 100
            number.step /= 100
            number.high = None if number.high is None else number.high / 100
            number.unit, number.unit_class = "倍", _MULTIPLE
        if match is not after and match.group("verb") == "有" and re.search(r"没有?\s*$", text[: match.start()]):
            # "营收没有五粮液的两倍" asserts less than twice, not "any multiple but two"
            number.comparator, number.negated = "lt", True
        return True
    return False


def _claim_dates(plain: str) -> list[tuple[int, tuple[str | None, int, int]]]:
    """Explicit dates in the claim: (position, (year or None, month, day))."""
    found: list[tuple[int, tuple[str | None, int, int]]] = []
    for pattern in _CLAIM_DATES:
        for match in pattern.finditer(plain):
            if any(start <= match.start() < start + 1 for start, _date in found):
                continue
            groups = match.groupdict()
            month = int(groups["m"]) if groups.get("m") else _EN_MONTHS.get(str(groups.get("mn")).lower().rstrip("."))
            day = int(groups["d"])
            if month and 1 <= month <= 12 and 1 <= day <= 31:
                found.append((match.start(), (groups.get("y"), month, day)))
    return sorted(found)


def _assign_dates(numbers: list[_Number], plain: str) -> None:
    """The date named in a number's sentence (the nearest before it, else the first after it)."""
    dates = _claim_dates(plain)
    if not dates:
        return
    for number in numbers:
        start, end = _clause_bounds(plain, number.start, _SENTENCE_BREAK)
        inside = [(position, date) for position, date in dates if start <= position < end]
        earlier = [date for position, date in inside if position < number.start]
        number.date = earlier[-1] if earlier else (inside[0][1] if inside else None)


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


def _step(token: str) -> float:
    """The unit of the last significant digit of a written number: "800" → 100, "1600" → 100, "24.6" → 0.1, "3" → 1."""
    if "." in token:
        return 10.0 ** -len(token.split(".")[1])
    digits = token.lstrip("+-").rstrip("0")
    return 10.0 ** (len(token.lstrip("+-")) - len(digits)) if digits else 1.0


def _read_comparator(number: _Number, stretch: str, after: str) -> None:
    if number.comparator == "range":
        comparator: str = "range"
    elif number.comparator != "eq":  # "30多倍"
        comparator = number.comparator
    else:
        comparator = next((name for name, pattern in _POST_COMPARATOR if pattern.search(after)), "eq")
        over = _OVER_WORD.match(after)
        if over and comparator == "gt":
            number.over = "more" if over.group(1) else "just_over"
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
    if comparator == "gt" and number.over and number.high is None and number.value >= 0:
        # "八百多亿" is 800 < x < 900; "三成出头" is 30% < x <= 35% (the upper bound is kept in ``high``).
        number.high = number.value + (number.step if number.over == "more" else number.step / 2)
    else:
        number.over = None


def _nearest_metrics(before: str, after: str, *, macro: bool = False) -> list[tuple[int, str]]:
    """Metric words around a number, nearest first. ``macro`` selects the macro series (matched on the
    unmasked text, where "M2" and "10年期" are still written out) instead of the company metrics."""
    found: list[tuple[int, int, str]] = []
    for name, metric in _METRICS.items():
        if metric.words is None or bool(metric.macro) != macro:
            continue
        for match in metric.words.finditer(before):
            found.append((len(before) - match.end(), -len(match.group(0)), name))
        match = metric.words.search(after)
        # Ties go to the word before the number ("市盈率15倍"), which is how claims are usually written, then
        # to the longer word ("5年期LPR" over "LPR").
        if match:
            found.append((match.start() + 1, -len(match.group(0)), name))
    return [(distance, name) for distance, _length, name in sorted(found)]


def _metric_for(
    number: _Number,
    before: str,
    after: str,
    clause: str,
    previous: _Number | None,
    *,
    plain_before: str = "",
    plain_after: str = "",
) -> None:
    """Metric of a number: a macro series named in its clause ("CPI同比上涨0.8%"); else the nearest metric
    word in its clause whose unit fits; growth for a percentage next to revenue / net profit; otherwise the
    previous number's metric for a parallel clause."""
    macro = _nearest_metrics(plain_before, plain_after, macro=True)
    if macro:
        fitting = [name for _distance, name in macro if number.unit_class in _METRICS[name].units]
        number.metric = fitting[0] if fitting else macro[0][1]
        number.unit_mismatch = not fitting
        return
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


def _context(number: _Number, norm: str, clause_start: int, clause_end: int, plain: str = "") -> None:
    clause = norm[clause_start:clause_end]
    if number.metric and _METRICS[number.metric].macro:
        # "3月CPI同比上涨0.8%": the month the value is for (checked against the indicator's date).
        months = list(_MONTH.finditer((plain or norm)[clause_start:clause_end]))
        if months:
            number.month = (months[-1].group(1), months[-1].group(2))
    if number.metric is None:
        number.reasons.append(("no_metric", "metric not recognised"))
        return
    if _FORECAST.search(clause) or _HYPOTHETICAL.search(clause):
        number.reasons.append(("forecast", "a forecast or hypothetical, not a reported fact"))
        return
    if number.metric in _DAILY and _MULTI_DAY.search(clause):
        if number.metric == "pct_change_1d":
            number.reasons.append(("multi_day", "a multi-day move; only the latest daily change is available"))
        else:
            number.reasons.append(("multi_day", "a multi-day turnover; only the latest session's is available"))
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


def _moves(text: str, norm: str, taken: list[tuple[int, int]], plain: str = "") -> list[_Number]:
    """Number-less moves: "昨天下跌了" (< 0), "并没有跌" (≥ 0), "did not fall", "CPI同比上涨" (> 0); qualitative
    moves by convention ("大跌": a fall of at least 3%, "小幅下跌": less than 1%); "跌幅扩大" (unverifiable)."""
    moves: list[_Number] = []
    found = [(match, False) for match in _MOVE.finditer(text)]
    found += [(match, False) for pattern in (_BIG_MOVE, _SMALL_MOVE, _WIDENING) for match in pattern.finditer(text)]
    found += [(match, True) for match in _MACRO_MOVE.finditer(text)]  # "PMI回落了": macro series only
    for match, macro_only in sorted(found, key=lambda pair: pair[0].start()):
        clause_start, clause_end = _clause_bounds(text, match.start())
        if any(clause_start <= start < clause_end for start, _end in taken):
            continue  # the clause states a number (the move is its direction) or a relation
        if any(clause_start <= move.start < clause_end for move in moves):
            continue
        clause = norm[clause_start:clause_end]
        down = _DOWN_WORDS if macro_only else _DOWN_MOVE
        comparator: Comparator = "lt" if down.search(match.group(0)) else "gt"
        negated = bool(_NEGATION.search(text[clause_start : match.start()]))
        macro = _nearest_metrics((plain or norm)[clause_start : match.start()], "", macro=True) or _nearest_metrics(
            "", (plain or norm)[match.end() : clause_end], macro=True
        )
        if macro_only and not macro:
            continue
        metric = macro[0][1] if macro else "pct_change_1d"
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
            metric=metric,
        )
        segment = text[clause_start:clause_end]
        widening = None if macro else _WIDENING.search(segment)
        qualitative = None if macro else (_BIG_MOVE.search(segment) or _SMALL_MOVE.search(segment))
        if widening:
            number.comparator, number.direction = "gt", (-1 if _DOWN_WORDS.search(widening.group(0)) else 1)
        elif qualitative:
            _qualitative(number, qualitative.group(0), big=bool(_BIG_MOVE.fullmatch(qualitative.group(0))))
        if _FORECAST.search(clause) or _HYPOTHETICAL.search(clause):
            number.reasons.append(("forecast", "a forecast or hypothetical, not a reported fact"))
        elif widening:
            note = "compares with the previous session's move; only the latest daily change is available"
            number.reasons.append(("multi_day", note))
        elif macro and (not _METRICS[metric].directional or _MACRO_CHANGE.fullmatch(match.group(0))):
            # "PMI回落了": a change from the previous reading; only the latest level is available.
            number.reasons.append(("no_data", "only the latest reading is available, not its change"))
        elif not macro and _MULTI_DAY.search(clause):
            number.reasons.append(("multi_day", "a multi-day move; only the latest daily change is available"))
        moves.append(number)
    return moves


def _qualitative(number: _Number, word: str, *, big: bool) -> None:
    """ "大跌" / "plunged": a move of at least 3% in its direction; "小幅下跌" / "edged down": less than 1%. The
    thresholds are a convention and the check's note says so."""
    direction = -1 if _QUALITATIVE_DOWN.search(word) else 1
    threshold = BIG_MOVE_PCT if big else SMALL_MOVE_PCT
    comparator: Comparator = "ge" if big else "lt"
    number.value = direction * threshold
    number.direction = direction
    number.comparator = _FLIP[comparator] if number.negated else comparator
    size = f"at least {_trim(threshold)}%" if big else f"less than {_trim(threshold)}%"
    number.convention = f"convention: '{word}' means a move of {size} in that direction"


# ---------------------------------------------------------------------------------------------------------
# Relations: "茅台的市盈率比五粮液高", "五粮液ROE低于茅台", "Moutai's P/E is higher than Wuliangye's"
# ---------------------------------------------------------------------------------------------------------
_REL_BI = re.compile(r"比")
_REL_WORD = re.compile(
    r"高于|大于|超过|多于|强于|好于|优于|跑赢|高过|大过|多过|强过|胜过|赢过|低于|小于|少于|不及|不如|逊于|弱于|差于|跑输|低过"
)
_REL_NOT_AS = re.compile(r"没有|没")
_REL_EN = re.compile(
    r"\b(higher|greater|bigger|larger|more|lower|smaller|less|cheaper|pricier)\b[^,.;!?]{0,40}?\bthan\b|"
    r"\b(above|below|exceeds?|exceeded|trails?|trailed|beats?|outperform(?:s|ed)?|underperform(?:s|ed)?|lag(?:s|ged)?)\b",
    re.I,
)
_REL_ADJ = re.compile(r"(?<!大)(活跃|高|大(?![涨跌])|多|贵|低|小|少|便宜)")
_REL_GT_WORDS = {"高", "大", "多", "贵", "活跃", "高于", "大于", "超过", "多于", "强于", "好于", "优于", "跑赢"}
_REL_GT_WORDS |= {"高过", "大过", "多过", "强过", "胜过", "赢过"}
_REL_GT_EN = {"higher", "greater", "bigger", "larger", "more", "pricier", "above", "exceed", "exceeds", "exceeded"}
_REL_GT_EN |= {"beat", "beats", "outperform", "outperforms", "outperformed"}
# "跑赢沪深300" / "outperformed the CSI 300": with no metric named, a comparison of the daily moves.
_PERFORMANCE_WORDS = {
    "跑赢",
    "跑输",
    "强于",
    "弱于",
    "强过",
    "胜过",
    "赢过",
    "beat",
    "beats",
    "outperform",
    "outperforms",
}
_PERFORMANCE_WORDS |= {"outperformed"}
_PERFORMANCE_WORDS |= {"underperform", "underperforms", "underperformed", "lag", "lags", "lagged"}
_REFERENCE_FILLER = re.compile(r"\s*(?:了|过)?\s*(?:that of|those of|the|its|其|它的|它)?\s*", re.I)
_INDUSTRY_REFERENCE = re.compile(
    r"(?:其?所在|所属|同)?(?:行业|板块|同行|同业)(?:的)?(?:平均|均值|整体|中位数)?(?:水平)?|"
    r"(?:its\s+)?(?:industry|sector|peers?)(?:\s+(?:average|median|mean|peers))?",
    re.I,
)
_MARKET_REFERENCE = re.compile(
    r"(?:大盘|全市场|市场|A股|沪深两市)(?:的)?(?:平均|均值|整体)?(?:水平)?|(?:the\s+)?(?:broader\s+)?market(?:\s+average)?",
    re.I,
)


def _relations(
    text: str, plain: str, positions: list[tuple[int, dict[str, Any]]], number_starts: list[int]
) -> list[_Number]:
    """Relational claims between two named targets, or a target and its industry: one check each. A sentence
    that states a number is checked on its numbers ("五粮液市盈率24.6倍，比茅台低" checks the 24.6)."""
    cues: list[tuple[int, int, str | None]] = []  # (start, end, comparator word or None: read the adjective)
    cues.extend((match.start(), match.end(), None) for match in _REL_BI.finditer(text))
    cues.extend((match.start(), match.end(), match.group(0)) for match in _REL_WORD.finditer(text))
    cues.extend((match.start(), match.end(), "not_as") for match in _REL_NOT_AS.finditer(text))
    relations: list[_Number] = []
    for start, end, word in sorted(cues):
        if any(relation.start <= start < relation.end for relation in relations):
            continue
        relation = _relation_at(text, plain, positions, number_starts, start, end, word)
        if relation is not None:
            relations.append(relation)
    for match in _REL_EN.finditer(text):
        if any(relation.start <= match.start() < relation.end for relation in relations):
            continue
        word = (match.group(1) or match.group(2)).lower()
        relation = _relation_at(text, plain, positions, number_starts, match.start(), match.end(), f"en:{word}")
        if relation is not None:
            relations.append(relation)
    return relations


def _relation_at(
    text: str,
    plain: str,
    positions: list[tuple[int, dict[str, Any]]],
    number_starts: list[int],
    start: int,
    end: int,
    word: str | None,
) -> _Number | None:
    sentence_start, sentence_end = _clause_bounds(text, start, _SENTENCE_BREAK)
    clause_start, clause_end = _clause_bounds(text, start)
    # A clause that states a number is checked on it ("茅台PE 24.6倍比五粮液的20.9倍高"); a relation in a clause
    # of its own is checked too, even when another clause of the sentence states a number ("茅台的市盈率比五粮液
    # 高，中国平安市盈率8.7倍" is two checks; "五粮液市盈率24.6倍，比茅台低" checks the 24.6 and the relation).
    # A number before the cue with nothing stated on the compared side is a relation too ("Ping An's P/B of 1.1x is
    # below the sector average", round 9): the number is checked on its own and the comparison as a relation.
    if any(start <= position < clause_end for position in number_starts):
        return None
    reference = _reference_at(text, end, positions)
    if reference is None:
        return None
    kind, ref_target, ref_end = reference
    ref_key = ref_target["key"] if ref_target else None
    subjects = [
        target for position, target in positions if sentence_start <= position < start and target["key"] != ref_key
    ]
    if not subjects:
        return None
    # The comparator: the cue word ("高于"), else the adjective after the reference ("比五粮液高").
    negated = False
    adjective = None
    if word is None or word == "not_as":
        adjective = _REL_ADJ.search(text, ref_end, clause_end)
        if adjective is None:
            return None
        comparator: Comparator = "gt" if adjective.group(1) in _REL_GT_WORDS else "lt"
        relation_end = adjective.end()
        negated = word == "not_as" or bool(re.search(r"(?:不|没有?|并不|并非)\s*$", text[clause_start:start]))
    elif word.startswith("en:"):
        comparator = "gt" if word[3:] in _REL_GT_EN else "lt"
        relation_end = ref_end
        negated = bool(re.search(r"\bnot\b|n't\b|\bnever\b", text[clause_start:start], re.I))
    else:
        comparator = "gt" if word in _REL_GT_WORDS else "lt"
        relation_end = ref_end
        negated = bool(re.search(r"(?:不|没有?|并不|并非)\s*$", text[clause_start:start]))
    # The metric: the nearest metric word in the clause, else in the sentence.
    metric = _relation_metric(text, start, end, clause_start, clause_end) or _relation_metric(
        text, start, end, sentence_start, sentence_end
    )
    if metric is None and (word or "").removeprefix("en:") in _PERFORMANCE_WORDS:
        metric = "pct_change_1d"
    if word in {None, "not_as"} and adjective is not None and adjective.group(1) == "活跃":
        metric = "amount"  # "比创业板ETF更活跃": more actively traded, i.e. a higher turnover (round 9)
    fell = re.search(r"跌|\b(?:fell|fall|dropped|declined|lost)\b", text[clause_start:clause_end], re.I)
    if metric == "pct_change_1d" and fell:
        comparator = "lt" if comparator == "gt" else "gt"  # "跌幅比五粮液大": it fell more, a lower change
    number = _Number(
        start=start,
        end=relation_end,
        value=0.0,
        rounding=0.0,
        scales=_BARE_SCALES,
        unit=None,
        unit_class=None,
        comparator=_FLIP[comparator] if negated else comparator,
        negated=negated,
        metric=metric,
        target=subjects[-1],
        relation=True,
        reference=ref_target,
        reference_kind=kind,
    )
    clause = plain[clause_start:clause_end]
    if metric is None:
        number.reasons.append(("no_metric", "the compared metric is not named"))
    elif _FORECAST.search(clause) or _HYPOTHETICAL.search(clause):
        number.reasons.append(("forecast", "a forecast or hypothetical, not a reported fact"))
    elif kind == "market":
        number.reasons.append(("no_reference", "compared with the market average, which the sources do not provide"))
    elif metric == "pct_change_1d" and _MULTI_DAY.search(clause):
        number.reasons.append(("multi_day", "a multi-day move; only the latest daily change is available"))
    return number


def _macro_relations(text: str, plain: str, number_starts: list[int], *, zh: bool) -> list[_Number]:
    """ "M2增速高于CPI": one macro series against another in a clause with no number (round 9). Both values come from
    ``get_macro_indicators``; the comparison is of their latest readings."""
    relations: list[_Number] = []
    for match in _REL_WORD.finditer(text):
        clause_start, clause_end = _clause_bounds(text, match.start())
        if any(clause_start <= position < clause_end for position in number_starts):
            continue
        # The subject: the series named before the cue in its clause, else earlier in the sentence
        # ("M2增速8.1%，高于CPI").
        sentence_start, _sentence_end = _clause_bounds(text, match.start(), _SENTENCE_BREAK)
        left = _nearest_metrics(plain[clause_start : match.start()], "", macro=True) or _nearest_metrics(
            plain[sentence_start : match.start()], "", macro=True
        )
        right = _nearest_metrics("", plain[match.end() : clause_end], macro=True)
        if not left or not right or left[0][1] == right[0][1]:
            continue
        subject, other = _METRICS[left[0][1]], _METRICS[right[0][1]]
        comparator: Comparator = "gt" if match.group(0) in _REL_GT_WORDS else "lt"
        negated = bool(re.search(r"(?:不|没有?|并不|并非)\s*$", text[clause_start : match.start()]))
        relations.append(
            _Number(
                start=match.start(),
                end=match.end(),
                value=0.0,
                rounding=0.0,
                scales=_BARE_SCALES,
                unit=None,
                unit_class=None,
                comparator=_FLIP[comparator] if negated else comparator,
                negated=negated,
                metric=left[0][1],
                target=_macro_target(str(subject.macro), zh=zh),
                relation=True,
                reference={**_macro_target(str(other.macro), zh=zh), "metric": right[0][1]},
                reference_kind="macro",
            )
        )
    return relations


def _macro_target(family: str, *, zh: bool) -> dict[str, Any]:
    label = _MACRO_NAMES.get(family, (family, family))[0 if zh else 1]
    return {"name": label, "symbol": None, "macro": family}


def _reference_at(
    text: str, index: int, positions: list[tuple[int, dict[str, Any]]]
) -> tuple[Literal["target", "industry", "market"], dict[str, Any] | None, int] | None:
    """What a comparison cue points at: a named target, the industry, or the market."""
    start = _REFERENCE_FILLER.match(text, index).end()  # type: ignore[union-attr]
    for position, target in positions:
        if position == start:
            end = start
            while end < len(text) and text[end] == "＠":
                end += 1
            if text[end : end + 2] in {"'s", "’s"}:
                end += 2
            return "target", target, end
    for kind, pattern in (("industry", _INDUSTRY_REFERENCE), ("market", _MARKET_REFERENCE)):
        match = pattern.match(text, start)
        if match and match.end() > start:
            return kind, None, match.end()  # type: ignore[return-value]
    return None


def _relation_metric(text: str, cue_start: int, cue_end: int, start: int, end: int) -> str | None:
    """The metric word nearest to a relation's cue within ``text[start:end]``; a move word only when no other
    metric is named ("涨得比五粮液多"). The cue itself is searched too ("a higher P/E than")."""
    found = _nearest_metrics(text[start:cue_start], text[cue_start:end])
    found = [(distance, name) for distance, name in found if not _METRICS[name].macro]
    named = [name for _distance, name in found if name != "pct_change_1d"]
    if named:
        return named[0]
    return found[0][1] if found else None


_COMPARED_SIDE = re.compile(r"(?:比|高于|低于|大于|小于|超过|不及|不如|跑赢|跑输|than)\s*(?:了|过|the|its)?\s*$", re.I)


def _bind_targets(
    numbers: list[_Number], positions: list[tuple[int, dict[str, Any]]], targets: list[dict[str, Any]], text: str
) -> list[_Number]:
    """Each number belongs to the nearest target named before it ("分别" assigns targets in order; "都" gives
    each target named before it its own check). Returns the numbers, with shared claims expanded."""
    if not targets:
        return numbers
    placed = {id(target) for _position, target in positions}
    unplaced = [target for target in targets if id(target) not in placed]
    # Targets named only as the compared side ("比五粮液", "高于茅台"): not the subject of a later clause (round 10).
    compared = {
        position for position, _target in positions if _COMPARED_SIDE.search(text, max(0, position - 8), position)
    }
    for number in numbers:
        before = [target for position, target in positions if position < number.start]
        if (number.ratio or number.difference) and number.reference is not None:
            # "五粮液的跌幅大约是茅台的三倍": the subject is named before the reference
            before = [target for target in before if target["key"] != number.reference["key"]]
        clause_start, _clause_end = _clause_bounds(text, number.start)
        subjects = [
            target
            for position, target in positions
            if position < number.start and not (position in compared and position < clause_start)
        ]
        if subjects and before and before[-1] is not subjects[-1] and not number.relation:
            # "茅台ROE比五粮液高出3.6个百分点，一年营收一千六百多亿": the revenue is 茅台's, not the compared side's
            before = subjects
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
    expanded: list[_Number] = []
    for number in numbers:
        shared = [] if number.relation else _shared_targets(number, numbers, positions, text)
        if len(shared) < 2:
            expanded.append(number)
            continue
        for target in shared:
            expanded.append(replace(number, target=target, reasons=list(number.reasons), years=list(number.years)))
    return expanded


def _shared_targets(
    number: _Number, numbers: list[_Number], positions: list[tuple[int, dict[str, Any]]], text: str
) -> list[dict[str, Any]]:
    """ "茅台和五粮液4月22日都跌超0.5%": the targets named since the previous number of the sentence, when a
    "都 / 均 / both / all" stands between the last of them and the number."""
    start, _end = _clause_bounds(text, number.start, _SENTENCE_BREAK)
    start = max([start, *(other.end for other in numbers if other is not number and other.end <= number.start)])
    named = [(position, target) for position, target in positions if start <= position < number.start]
    if len(named) < 2 or not _SHARED.search(text, named[-1][0], number.start):
        return []
    ordered: list[dict[str, Any]] = []
    for _position, target in named:
        if all(target["key"] != other["key"] for other in ordered):
            ordered.append(target)
    return ordered


# ---------------------------------------------------------------------------------------------------------
# Fetching and comparing
# ---------------------------------------------------------------------------------------------------------
def _fetch(numbers: list[_Number], registry: ToolRegistry) -> dict[str, list[AgentEvidence]]:
    """Evidence per target key (a symbol: price and/or fundamentals; ``sector:<name>``: the industry
    snapshot), per industry (``industry:<symbol>``, the target's industry snapshot) and per macro family
    (``macro:CPI``)."""
    evidence: dict[str, list[AgentEvidence]] = {}
    subjects: dict[str, dict[str, Any]] = {}
    wanted_market = wanted_fundamental = False
    families: set[str] = set()
    for number in numbers:
        if number.metric is None:
            continue
        metric = _METRICS[number.metric]
        if metric.macro:
            families.add(metric.macro)
            if number.reference_kind == "macro" and number.reference:
                families.add(str(number.reference["macro"]))
            continue
        wanted_market = wanted_market or not metric.fundamental
        # The industry snapshot comes with the fundamentals.
        wanted_fundamental = (
            wanted_fundamental or metric.fundamental or number.reference_kind == "industry" or number.industry
        )
        for subject in (number.target, number.reference):
            if subject and subject.get("key"):
                subjects.setdefault(subject["key"], subject)
    for key, subject in list(subjects.items())[:4]:
        if subject.get("sector"):
            result = registry.run("get_fundamentals", {"target": subject["sector"]})
            evidence[key] = list(result.evidence) if result.ok else []
            continue
        items: list[AgentEvidence] = []
        if wanted_market:
            result = registry.run("get_price_history", {"target": key})
            items.extend(result.evidence if result.ok else [])
        if wanted_fundamental:
            result = registry.run("get_fundamentals", {"target": key})
            for item in result.evidence if result.ok else []:
                if item.evidence_id.startswith("fundamental_"):
                    items.append(item)
                elif _is_industry(item):
                    evidence.setdefault(f"industry:{key}", []).append(item)
        evidence[key] = items
    if families:
        result = registry.run("get_macro_indicators", {"topics": sorted(families)})
        for item in result.evidence if result.ok else []:
            code = item.payload.get("indicator_code") if isinstance(item.payload, dict) else None
            if code:
                evidence.setdefault(f"macro:{_macro_family(str(code))}", []).append(item)
    return evidence


def _is_industry(item: AgentEvidence) -> bool:
    return item.source_type == "industry_sql" or item.evidence_id.startswith("industry_")


def _keys(subject: dict[str, Any] | None, metric: str) -> tuple[str, ...]:
    """Payload keys of a metric for a target: a sector's industry snapshot names some metrics differently."""
    if subject and subject.get("sector"):
        return _INDUSTRY_KEYS.get(metric, ())
    return _METRICS[metric].keys


def _macro_family(code: str) -> str:
    """ "CPI_CN" → "CPI", "LPR1Y_CN" → "LPR1Y", "UST10Y_CN_PROXY" / "CN10Y" → "CN10Y"."""
    code = code.upper()
    if "10Y" in code:
        return "CN10Y"
    return code.removesuffix("_CN")


def _value(items: list[AgentEvidence], keys: tuple[str, ...]) -> tuple[AgentEvidence, float] | None:
    for item in items:
        for key in keys:
            actual = item.payload.get(key) if isinstance(item.payload, dict) else None
            if isinstance(actual, int | float) and not isinstance(actual, bool) and math.isfinite(actual):
                return item, float(actual)
    return None


def _industry_average(target: dict[str, Any] | None) -> bool:
    """The number is about the target's industry average (a sector target is its industry already)."""
    return bool(target) and not (target or {}).get("sector")


def _industry_label(key: str | None, evidence: dict[str, list[AgentEvidence]], *, zh: bool) -> str:
    """ "保险行业平均" / "insurance industry average", from the target's industry snapshot."""
    industry = next((item.payload.get("industry_name") for item in evidence.get(f"industry:{key}") or []), None)
    if zh:
        return f"{industry}行业平均" if industry else "行业平均"
    english = INDUSTRY_EN.get(str(industry), str(industry)) if industry else None
    return f"{english} industry average" if english else "industry average"


def _check(number: _Number, evidence: dict[str, list[AgentEvidence]], *, zh: bool = True) -> ClaimCheck:
    industry = number.industry and _industry_average(number.target)
    key = (number.target or {}).get("key")
    if industry:
        target_label = _industry_label(key, evidence, zh=zh)
    elif number.stated_reference and (number.target or {}).get("sector"):
        target_label = _sector_average_label(number.target, zh=zh)  # "白酒行业平均": the average the claim states
    else:
        target_label = _target_name(number.target, zh=zh)
    base = ClaimCheck(
        target=target_label,
        kind=_kind(number),
        metric=number.metric,
        claimed=None if number.relation and not (number.ratio or number.difference) else number.value,
        claimed_high=number.high,
        claimed_unit=number.unit,
        comparator=number.comparator,
        negated=number.negated,
        direction=None if number.direction is None else ("up" if number.direction > 0 else "down"),
        reference=_reference_name(number, evidence, zh=zh),
        status="unverifiable",
        note=number.convention or "",
    )
    if number.reasons:
        reason, note = number.reasons[0]
        return base.model_copy(update={"reason": reason, "note": note})
    assert number.metric is not None and number.target is not None
    metric = _METRICS[number.metric]
    if metric.macro:
        found = _value(evidence.get(f"macro:{metric.macro}") or [], metric.keys)
        if found is None:
            note = f"the macro sources do not provide {metric.macro}"
            return base.model_copy(update={"reason": "no_data", "note": note})
    elif industry:
        items = evidence.get(f"industry:{key}") or []
        keys = _INDUSTRY_KEYS.get(number.metric)
        found = _value(items, keys) if keys else None
        if found is None:
            note = "the industry snapshot has no value for this metric" if items else "no industry snapshot"
            return base.model_copy(update={"reason": "no_data", "note": note})
    else:
        found = _value(evidence.get(number.target.get("key") or "") or [], _keys(number.target, number.metric))
    derived_note = ""
    if found is None and not industry and not number.target.get("sector"):
        found, derived_note = _derived(number.metric, evidence.get(number.target.get("key") or "") or [])
    if found is None:
        if number.metric in _UNAVAILABLE:
            return base.model_copy(update={"reason": "no_data", "note": _UNAVAILABLE[number.metric]})
        if number.metric in {"revenue_yoy", "netprofit_yoy"}:
            note = "the fundamentals source has no year-on-year growth for this item"
            return base.model_copy(update={"reason": "growth_unavailable", "note": note})
        return base.model_copy(update={"reason": "no_data", "note": "no data for this metric"})
    item, actual = found
    if number.metric == "amount" and actual <= 0:
        # Index rows carry turnover 0 when the source does not report it: no data, not a value to compare.
        return base.model_copy(update={"reason": "no_data", "note": "the source reports no turnover for this target"})
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
    if metric.macro:
        mismatch = _macro_period_mismatch(number, as_of)
    elif basis == "trade_date":
        mismatch = _date_mismatch(number, as_of)
    else:
        mismatch = _period_mismatch(number, item) if metric.fundamental else None
    if mismatch:
        return base.model_copy(update={"reason": "period_mismatch", "note": mismatch})
    in_percent = any(
        (item.payload.get("metric_units") or {}).get(key) == "%" for key in _METRICS[number.metric or ""].keys
    )
    if number.relation:
        return _check_relation(number, base, evidence, declared_percent=in_percent)
    status = _compare(number, actual, declared_percent=in_percent or bool(derived_note))
    note = "; ".join(part for part in (derived_note, number.convention, _interim_note(number, item)) if part)
    return base.model_copy(update={"status": status, "note": note})


# Metrics a claim may name that the sources neither report nor determine (round 9, E8): say why.
_COVERAGE_METRICS = {metric.key: metric for metric in COVERAGE_METRICS}
_UNAVAILABLE = {
    "ps": "the sources carry no price-to-sales ratio, and it cannot be computed without the market cap",
    "max_drawdown": "a maximum drawdown needs the full price history of the period; only the latest closes are served",
    "market_cap": "the sources carry no market cap for this target",
    "peg": _COVERAGE_METRICS["peg"].needs_en,
}


def _payload_number(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def _derived(metric: str | None, items: list[AgentEvidence]) -> tuple[tuple[AgentEvidence, float] | None, str]:
    """A value the source does not report but its evidence determines, with the arithmetic in the note (round 9):

    * net margin = net profit / revenue, when both are reported (the chat derives it the same way,
      ``coverage.METRICS["net_margin"].derivable_from``);
    * PEG = P/E / net-profit growth, when the growth is reported and positive;
    * the daily change of a fund or index whose source leaves ``pct_change_1d`` empty, from its last two closes
      (labelled "computed")."""
    for item in items:
        payload = item.payload if isinstance(item.payload, dict) else {}
        if metric in {"net_margin", "peg"} and not _COVERAGE_METRICS[metric].derivable(payload):
            continue
        if metric == "net_margin":
            revenue = _payload_number(payload, _METRICS["revenue"].keys)
            profit = _payload_number(payload, _METRICS["net_profit"].keys)
            if revenue and revenue > 0 and profit is not None:
                value = profit / revenue * 100
                return (item, round(value, 4)), (
                    f"derived: net profit / revenue = {_trim(profit)} / {_trim(revenue)} = {value:.2f}%"
                )
        elif metric == "peg":
            pe = _payload_number(payload, _METRICS["pe_ttm"].keys)
            growth = _payload_number(payload, PROFIT_GROWTH_FIELDS)
            if pe is not None and growth is not None and growth > 0:
                return (
                    item,
                    round(pe / growth, 4),
                ), f"derived: P/E / net profit growth = {_trim(pe)} / {_trim(growth)}"
        elif metric == "pct_change_1d":
            closes = [
                row
                for row in payload.get("recent_closes") or []
                if isinstance(row, dict) and isinstance(row.get("close"), int | float) and row.get("close")
            ]
            latest = str(item.as_of or payload.get("as_of") or "")[:10]
            if len(closes) >= 2 and str(closes[-1].get("date") or "")[:10] == latest:
                previous, last = closes[-2], closes[-1]
                value = (float(last["close"]) / float(previous["close"]) - 1) * 100
                return (item, round(value, 4)), (
                    f"computed from the last two closes: {_trim(float(previous['close']))} ({previous.get('date')}) → "
                    f"{_trim(float(last['close']))} ({last.get('date')}); the source reports no daily change"
                )
    return None, ""


def _kind(number: _Number) -> str:
    if number.difference:
        return "difference" if number.difference == "absolute" else "relative_difference"
    if number.ratio:
        return "ratio"
    if number.relation:
        return "relation"
    return "stated_reference" if number.stated_reference else "value"


def _sector_average_label(target: dict[str, Any] | None, *, zh: bool) -> str:
    name = str((target or {}).get("name") or "")
    if zh:
        return f"{name}行业平均"
    return f"{_SECTOR_EN.get(_SECTOR_SAME.get(name, name), (name,))[0]} industry average"


def _target_name(target: dict[str, Any] | None, *, zh: bool) -> str | None:
    """The target as written in a check; a sector in an English report reads "baijiu industry", not 白酒."""
    if not target:
        return None
    name = str(target.get("name") or target.get("symbol"))
    if target.get("sector") and not zh:
        return f"{_SECTOR_EN.get(_SECTOR_SAME.get(name, name), (name,))[0]} industry"
    return name


def _reference_name(number: _Number, evidence: dict[str, list[AgentEvidence]], *, zh: bool) -> str | None:
    if not number.relation:
        return None
    if number.reference_kind == "target" and number.reference:
        name = str(number.reference.get("name") or number.reference.get("symbol"))
        if number.reference.get("sector"):
            english = _SECTOR_EN.get(_SECTOR_SAME.get(name, name), (name,))[0]
            return f"{name}行业" if zh else f"{english} industry"
        return name
    if number.reference_kind == "market":
        return "市场平均" if zh else "market average"
    if number.reference_kind == "macro":
        return str((number.reference or {}).get("name"))
    return _industry_label((number.target or {}).get("key"), evidence, zh=zh)


def _check_relation(
    number: _Number, base: ClaimCheck, evidence: dict[str, list[AgentEvidence]], *, declared_percent: bool = False
) -> ClaimCheck:
    """ "茅台PE比五粮液高": the target's value against the reference's (a named target, a sector or the
    target's industry); "是五粮液的1.5倍": their ratio against the claimed multiple."""
    metric = number.metric or ""
    if number.reference_kind == "macro":
        other = str((number.reference or {}).get("metric") or "")
        items = evidence.get(f"macro:{(number.reference or {}).get('macro')}") or []
        reference = _value(items, _METRICS[other].keys) if other in _METRICS else None
        if reference is None:
            return base.model_copy(update={"reason": "no_data", "note": "no reading for the compared series"})
        item, value = reference
        base = base.model_copy(update={"reference_value": value, "reference_evidence_id": item.evidence_id})
        actual = float(base.actual or 0.0)
        holds = {"gt": actual > value, "ge": actual >= value, "lt": actual < value, "le": actual <= value}.get(
            number.comparator, math.isclose(actual, value, rel_tol=_REL_TOLERANCE)
        )
        return base.model_copy(update={"status": "supported" if holds else "contradicted"})
    if number.reference_kind == "industry":
        keys = _INDUSTRY_KEYS.get(metric)
        items = evidence.get(f"industry:{(number.target or {}).get('key')}") or []
        reference = _value(items, keys) if keys else None
        if reference is None:
            note = "the industry snapshot has no value for this metric" if items else "no industry snapshot"
            return base.model_copy(update={"reason": "no_data", "note": note})
    else:
        subject = number.reference or {}
        reference = _value(evidence.get(subject.get("key") or "") or [], _keys(subject, metric))
        if reference is None:
            return base.model_copy(update={"reason": "no_data", "note": "no data for the compared target"})
    item, value = reference
    as_of, basis = _as_of(item, metric)
    mismatch = (
        _date_mismatch(number, as_of) if basis == "trade_date" else _period_mismatch(number, item)
        if _METRICS[metric].fundamental else None
    )  # fmt: skip
    if mismatch:
        return base.model_copy(update={"reason": "period_mismatch", "note": f"{mismatch} (compared side)"})
    base = base.model_copy(update={"reference_value": value, "reference_evidence_id": item.evidence_id})
    actual = float(base.actual or 0.0)
    if number.ratio:
        return _check_ratio(number, base, actual, value)
    if number.difference:
        if _METRICS[metric].fraction and not declared_percent and max(abs(actual), abs(value)) <= 1.5:
            actual, value = actual * 100, value * 100  # an undeclared fraction (0.33) against percentage points
        return _check_difference(number, base, actual, value)
    holds = {
        "gt": actual > value,
        "ge": actual >= value,
        "lt": actual < value,
        "le": actual <= value,
    }.get(number.comparator, math.isclose(actual, value, rel_tol=_REL_TOLERANCE))
    return base.model_copy(update={"status": "supported" if holds else "contradicted"})


def _check_ratio(number: _Number, base: ClaimCheck, actual: float, value: float) -> ClaimCheck:
    """ "市净率大约是五粮液的1.5倍", "跌幅大约是茅台的三倍": actual / reference against the claimed multiple. For
    moves, both must go the stated way (a fall is not three times a rise)."""
    if value == 0:
        return base.model_copy(update={"reason": "no_data", "note": "the compared value is 0; no ratio"})
    if number.metric == "pct_change_1d":
        direction = number.direction or (1 if value > 0 else -1)
        if actual * direction <= 0 or value * direction <= 0:
            note = "the two moves are not both in the stated direction"
            return base.model_copy(update={"status": "contradicted", "note": note})
    elif actual <= 0 or value <= 0:
        return base.model_copy(update={"reason": "no_data", "note": "a multiple of a negative value is not defined"})
    ratio = actual / value
    claimed, high = number.value, number.high
    comparator = number.comparator
    if comparator == "approx":
        tolerance = _approx_tolerance(number, claimed)
    else:
        tolerance = max(number.rounding, abs(claimed) * _REL_TOLERANCE) + 1e-9
    close = abs(ratio - claimed) <= tolerance
    bounded = number.over is not None and high is not None  # "三倍多": 3 < ratio < 4
    holds = {
        "eq": close,
        "approx": close,
        "ne": not close,
        "gt": ratio > claimed and (not bounded or ratio < (high or 0.0) + (1e-9 if number.over == "just_over" else 0)),
        "ge": ratio >= claimed,
        "lt": ratio < claimed,
        "le": ratio <= claimed,
        "range": claimed <= ratio <= (high if high is not None else claimed),
    }[comparator]
    note = f"ratio {ratio:.2f} = {_trim(actual)} / {_trim(value)}"
    return base.model_copy(
        update={"ratio": round(ratio, 4), "status": "supported" if holds else "contradicted", "note": note}
    )


def _check_difference(number: _Number, base: ClaimCheck, actual: float, value: float) -> ClaimCheck:
    """ "茅台ROE比五粮液高出约3.6个百分点": target - reference against the stated difference, in the stated direction
    (a difference the other way contradicts it); "相差3.6个百分点" is the size of the difference either way. A relative
    difference ("比行业平均低了近10%") is a percentage of the reference's value. The comparator applies to the size of
    the difference: "高出不到4个百分点" is 0 < difference < 4, "多赚四百多亿" 400亿 < difference < 500亿."""
    relative = number.difference == "relative"
    if relative:
        if value <= 0:
            return base.model_copy(update={"reason": "no_data", "note": "a percentage of a non-positive value"})
        difference = (actual - value) / value * 100
        note = f"relative difference {difference:+.2f}% = ({_trim(actual)} - {_trim(value)}) / {_trim(value)}"
    else:
        difference = actual - value
        note = f"difference {_trim(round(difference, 4))} = {_trim(actual)} - {_trim(value)}"
    base = base.model_copy(update={"difference": round(difference, 4), "note": note})
    if number.unit_mismatch:
        note = (
            f"the unit '{number.unit}' does not fit a difference of {number.metric} ('高出两倍' could mean 2 or 3 "
            "times; a multiple states it: '是…的3倍')"
        )
        return base.model_copy(update={"reason": "unit_mismatch", "note": note})
    size = abs(difference) if number.difference_unsigned else difference * (number.direction or 1)
    scaled = size * _nearest_scale(size, number.value, number.scales)
    bound = abs(number.value)
    comparator = number.comparator
    if comparator == "approx":
        tolerance = _approx_tolerance(number, bound)
    else:
        tolerance = max(number.rounding, bound * _REL_TOLERANCE) + 1e-9
    close = scaled > 0 and abs(scaled - bound) <= tolerance if bound else abs(scaled) <= tolerance
    if comparator == "gt" and number.over and number.high is not None:
        high = abs(number.high)
        holds = bound < scaled <= high + 1e-9 if number.over == "just_over" else bound < scaled < high
    else:
        holds = {
            "eq": close,
            "approx": close,
            "ne": not close,
            "gt": scaled > bound,
            "ge": scaled >= bound,
            # "高出不到4个百分点" still says it is higher: a difference the other way contradicts it
            "lt": 0 < scaled < bound if not number.negated else scaled < bound,
            "le": 0 < scaled <= bound if not number.negated else scaled <= bound,
            "range": min(bound, abs(number.high or bound)) <= scaled <= max(bound, abs(number.high or bound)),
        }[comparator]
    return base.model_copy(update={"status": "supported" if holds else "contradicted"})


def _date_mismatch(number: _Number, as_of: str | None) -> str | None:
    """ "4月21日跌超1%" against the 2026-04-22 close: another trading day (a matching date is fine)."""
    if number.date is None or not as_of or not re.match(r"\d{4}-\d{2}-\d{2}", as_of):
        return None
    year, month, day = number.date
    if (month, day) == (int(as_of[5:7]), int(as_of[8:10])) and (not year or year == as_of[:4]):
        return None
    named = f"{year}-{month:02d}-{day:02d}" if year else f"{month:02d}-{day:02d}"
    return f"the claim is about {named}; the data is for the trading day {as_of[:10]}"


def _macro_period_mismatch(number: _Number, as_of: str | None) -> str | None:
    """ "2月CPI同比上涨0.7%" against the March reading: another period."""
    if not as_of or not re.match(r"\d{4}-\d{2}", as_of):
        return None
    if number.month:
        year, month = number.month
        if int(month) != int(as_of[5:7]) or (year and year != as_of[:4]):
            named = f"{year}-{int(month):02d}" if year else f"month {int(month)}"
            return f"the claim is about {named}; the latest reading is for {as_of[:7]}"
    elif number.years and as_of[:4] not in number.years:
        return f"the claim is about {', '.join(number.years)}; the latest reading is for {as_of[:7]}"
    return None


def _as_of(item: AgentEvidence, metric: str) -> tuple[str | None, str | None]:
    if item.source_type == "market_api" or item.evidence_id.startswith("price_"):
        return item.as_of, "trade_date"
    if _is_industry(item):
        trade_date = item.payload.get("trade_date") or item.as_of
        return (str(trade_date) if trade_date else None), "trade_date"
    payload = item.payload
    if _METRICS[metric].macro:
        indicator_date = payload.get("metric_date") or item.as_of
        return (str(indicator_date) if indicator_date else None), "indicator_date"
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


def _approx_tolerance(number: _Number, value: float) -> float:
    """ "约 / 左右 / 接近" (round 10, F7): within 5% of the value, or half the step of the number's last significant
    digit when that is wider ("八百亿左右" is 750-850亿, "三成左右" is 25%-35%, "市盈率25倍左右" is 24.5-25.5 or 5%)."""
    return max(number.rounding, number.step / 2, abs(value) * _APPROX_TOLERANCE) + 1e-9


def _compare(number: _Number, actual: float, *, declared_percent: bool = False) -> Status:
    metric = _METRICS[number.metric or ""]
    scales = number.scales
    # Normalised payloads declare percent units (tools/units.py); only an undeclared fraction may be x100.
    if number.unit_class in {_PERCENT, None} and metric.fraction and abs(actual) <= 1.5 and not declared_percent:
        scales = (1.0, 100.0)
    reference = number.high if number.comparator == "range" and number.high is not None else number.value
    expected = actual * _nearest_scale(actual, reference, scales)
    tolerance = max(number.rounding, abs(expected) * _REL_TOLERANCE) + 1e-9
    claimed = number.value
    comparator = number.comparator
    if comparator in {"eq", "ne", "approx"}:
        if comparator == "approx":
            tolerance = _approx_tolerance(number, expected)
        same_direction = number.metric != "pct_change_1d" or (claimed >= 0) == (expected >= 0) or expected == 0
        close = same_direction and abs(claimed - expected) <= tolerance
        holds = not close if comparator == "ne" else close
    elif comparator == "gt" and number.over and number.high is not None:
        # "八百多亿": above the number and below its next step; for a move, on the size of the move.
        size = expected * number.direction if number.direction is not None else expected
        low, high = abs(claimed) if number.direction is not None else claimed, abs(number.high)
        holds = low < size <= high + 1e-9 if number.over == "just_over" else low < size < high
    elif number.direction is not None:
        # "跌超1%" is about the size of the fall: change <= -1. "跌不到1%": -1 < change <= 0 (a rise is not a
        # smaller fall). Negated bounds ("没有跌超过1%") hold for a move the other way.
        size = expected * number.direction
        bound = abs(claimed)
        if comparator == "range":
            low, high = sorted((abs(claimed), abs(number.high if number.high is not None else claimed)))
            inside = low <= size <= high
            holds = not inside if number.negated else inside
        else:
            holds = {"gt": size > bound, "ge": size >= bound, "lt": size < bound, "le": size <= bound}[comparator]
            if comparator in {"lt", "le"} and not number.negated:
                holds = holds and size > 0  # "跌了不到1%" asserts a fall: a flat day or a rise contradicts it
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
