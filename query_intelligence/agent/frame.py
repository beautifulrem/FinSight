"""The session comparison frame (round 11, G1-G4).

A conversation that compares targets usually spreads the comparison over several turns: "五粮液的ROE多少" → "茅台呢"
→ "两者差几个点", or "中国平安市盈率多少" → "保险行业平均呢" → "折价了百分之多少". Earlier rounds joined such a gap
question to the turn before it with phrase rules, which failed whenever the chain was one turn longer or worded
differently. The frame is a small piece of state instead:

* After every answered turn, ``next_frame`` keeps ``{metric, operands}`` for the last metric discussed: the operands
  are the targets (and an industry average) the metric was asked for, in the order the user brought them in, with the
  value and evidence id each one had in that turn. A turn that asks the same metric for another target ("X呢", "and
  Moutai's?") adds an operand; a turn about another metric starts a new frame.
* ``resolve_frame_question`` reads a gap, ratio, relative-difference or which-is-higher question ("差几个点",
  "前者是后者的几倍", "折价了百分之多少", "哪个更大", "what's the ratio between them?") against the frame: the metric
  and the two operands come from the frame (or from the question where it names them), "前者/后者" and "the
  former/the latter" follow the frame's order. The question is rewritten to name both operands and the metric, and a
  ``request`` tells the planner which data to fetch and the composer what to compute (``composer.frame_sentences``),
  so the answer never falls back to whatever data a bare question happens to fetch (prices).
* Metric names are matched longest first: 净利率 is not 净利润 or 净利, 市净率 is not 市盈率, 毛利率 is a metric.

The frame lives in the turn records (``turns[-1]["frame"]``), so it survives checkpoints like the rest of the session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MAX_OPERANDS = 4
# (round 14) the second half of a long Chinese name that is a generic word, not a short name (招商银行 → 银行 would
# match every bank)
GENERIC_NAME_TAILS = frozenset({"银行", "证券", "保险", "集团", "股份", "控股", "电力", "时代", "科技", "医药", "电器"})


@dataclass(frozen=True)
class FrameMetric:
    key: str
    zh: str
    en: str
    # "points": a percentage (gap in percentage points); "multiple": P/E, P/B; "money": an amount in CNY;
    # "price": a per-share price; "count": a plain count (volume)
    kind: str
    tool: str  # the tool whose output carries it: get_fundamentals or get_price_history
    terms: tuple[str, ...]  # literal Chinese words and English regex fragments


# (round 12) Turnover however it is asked: "成交额", "成交了多少钱", "交易额", "哪个交易更活跃", "trading value", "value
# traded", "which traded more". One vocabulary for the frame, the price answer's details (``coverage._AMOUNT``), the
# template's comparisons (``composer._ARITHMETIC_METRICS``) and the ellipsis aspect, so a turnover question is answered
# with the turnover and a later gap or ratio has a metric. Regex terms start with "(" or "\\" (see ``_alternative``).
TURNOVER_TERMS: tuple[str, ...] = (
    "成交金额",
    "成交额",
    "交易金额",
    "交易额",
    r"(?:成交了?(?:多少钱|多少金额|多少亿|几亿|多少万元?))",
    r"(?:(?:成交|交易)得?(?:更|最|比较|很|十分|非常|不)?(?:活跃|旺盛?|火爆|清淡|冷清))",
    r"\bturnover\b(?! rate| ratio)",
    r"\btrading value\b",
    r"\bvalue traded\b",
    r"\btraded value\b",
    r"\b(?:more|most|less|least) (?:actively )?traded\b",
    r"\btraded (?:more|less)\b",
)
TURNOVER = re.compile(
    "|".join(term if term.startswith(("(", "\\")) else re.escape(term) for term in TURNOVER_TERMS), re.IGNORECASE
)
_DAY_MOVE = (
    r"(?:\b(?:move[sd]?|r[io]se|fell|fall|gain(?:ed)?|drop(?:ped)?|climb(?:ed)?|slip(?:ped)?|lost)\b"
    r"(?=[^.?!]{0,20}\b(?:today|yesterday|on the day|in the last session|on the last session)\b|\s*[?,]))"
)

FRAME_METRICS: tuple[FrameMetric, ...] = (
    FrameMetric(
        "net_margin",
        "净利率",
        "net margin",
        "points",
        "get_fundamentals",
        (
            "净利润率",
            "销售净利率",
            "净利率",
            "利润率",
            r"\bnet[- ](?:profit[- ])?margins?\b",
            r"\bprofit[- ]margins?\b",
        ),
    ),
    FrameMetric(
        "gross_margin",
        "毛利率",
        "gross margin",
        "points",
        "get_fundamentals",
        ("毛利润率", "销售毛利率", "毛利率", r"\bgross (?:profit )?margins?\b"),
    ),
    FrameMetric(
        "roe",
        "ROE",
        "ROE",
        "points",
        "get_fundamentals",
        ("净资产收益率", r"(?<![A-Za-z])ROE(?![A-Za-z])", r"\breturns? on equity\b"),
    ),
    FrameMetric(
        "pe",
        "市盈率",
        "P/E",
        "multiple",
        "get_fundamentals",
        (
            "市盈率",
            r"(?<![A-Za-z])P/?E(?![A-Za-z])",
            r"\bprice[- ]to[- ]earnings(?: ratio)?\b",
            r"\bearnings multiples?\b",
        ),
    ),
    FrameMetric(
        "pb",
        "市净率",
        "P/B",
        "multiple",
        "get_fundamentals",
        ("市净率", r"(?<![A-Za-z])P/?B(?![A-Za-z])", r"\bprice[- ]to[- ]book(?: ratio)?\b", r"\bbook multiples?\b"),
    ),
    FrameMetric(
        "eps",
        "每股收益",
        "EPS",
        "price",
        "get_fundamentals",
        (
            "每股收益",
            "每股盈利",
            "每股净利润",
            r"(?:(?:一|每)股(?:能|可以|大概|大约)?赚(?:了)?(?:多少|几))",
            r"(?<![A-Za-z])EPS(?![A-Za-z])",
            r"\bearnings per share\b",
        ),
    ),
    FrameMetric(
        "dividend_yield",
        "股息率",
        "dividend yield",
        "points",
        "get_fundamentals",
        ("股息率", r"\bdividend yields?\b"),
    ),
    # (round 14, review-10 J8) the free-float market cap is its own metric (``circ_mv``), never the total
    FrameMetric(
        "float_market_cap",
        "流通市值",
        "free-float market cap",
        "money",
        "get_fundamentals",
        (
            "流通市值",
            "流通盘市值",
            r"\bfree[- ]?float(?:ing)? market (?:cap|value|capitali[sz]ation)\b",
            r"\bfloat(?:ing)? market (?:cap|value)\b",
            r"\bcirculating market (?:cap|value)\b",
            r"\bfree[- ]float\b",
        ),
    ),
    FrameMetric(
        "market_cap",
        "总市值",
        "market cap",
        "money",
        "get_fundamentals",
        ("总市值", "市值", r"\bmarket (?:cap|capitali[sz]ation|value)\b"),
    ),
    # (round 14, review-10 J8) net profit after non-recurring items (``profit_dedt``) is its own metric, never 归母
    FrameMetric(
        "profit_dedt",
        "扣非净利润",
        "net profit excluding non-recurring items",
        "money",
        "get_fundamentals",
        (
            "扣除非经常性损益后的净利润",
            "扣除非经常性损益后净利润",
            "扣非归母净利润",
            "扣非后净利润",
            "扣非净利润",
            "扣非净利",
            "扣非利润",
            "扣非",
            r"\b(?:net )?(?:profit|income|earnings) (?:excluding|ex|before|after) non-?recurring"
            r"(?: items| gains and losses)?\b",
            r"\b(?:excluding|ex-?) non-?recurring(?: items| gains and losses)?\b",
            r"\b(?:recurring|core|underlying) (?:net )?(?:profit|earnings|income)\b",
            r"\bdeducted net profit\b",
        ),
    ),
    # (round 14, review-10 J8) metrics the data does not have: named, so a comparison or a question about them says
    # they are missing instead of answering with a neighbour (营业利润 is not 净利润, 毛利润 is not 毛利率)
    FrameMetric(
        "operating_profit",
        "营业利润",
        "operating profit",
        "money",
        "get_fundamentals",
        ("营业利润", "经营利润", r"\boperating (?:profit|income)\b", r"\bEBIT\b"),
    ),
    FrameMetric(
        "gross_profit",
        "毛利润",
        "gross profit",
        "money",
        "get_fundamentals",
        ("毛利润", "毛利额", r"\bgross profits?\b"),
    ),
    FrameMetric(
        "total_profit",
        "利润总额",
        "total profit",
        "money",
        "get_fundamentals",
        ("利润总额", "税前利润", r"\bpre-?tax (?:profit|income)\b", r"\bprofit before tax\b"),
    ),
    FrameMetric(
        "revenue",
        "营业收入",
        "revenue",
        "money",
        "get_fundamentals",
        (
            "营业总收入",
            "营业收入",
            "营收",
            "收入",
            r"\brevenues?\b",
            r"\bsales\b",
            r"\btop line\b",
            r"\bout-?sell(?:s|ing)?\b",
        ),
    ),
    FrameMetric(
        "net_profit",
        "净利润",
        "net profit",
        "money",
        "get_fundamentals",
        (
            "归母净利润",
            "净利润",
            "净利",
            "利润",
            r"\bnet (?:profit|income)\b",
            r"\bprofits?\b",
            r"\bearnings\b",
            # (round 13) "Ping An out-earns Wuliangye", "earns less than": what a company earns is its net profit
            r"\b(?:out-?)?earn(?:s|ed)?\b",
            # (round 14) "两家去年一共赚了多少": what companies earned (not "一股赚多少", EPS, which is longer)
            r"(?:赚(?:了|的钱|到的钱|得)?(?=[^，。？?；;]{0,4}(?:多少|几)))",
        ),
    ),
    FrameMetric(
        "pct_change",
        "涨跌幅",
        "daily change",
        "points",
        "get_price_history",
        (
            "涨跌幅",
            "涨幅",
            "跌幅",
            r"(?:涨|跌)(?:得|了)",
            r"\b(?:daily|percent(?:age)?|price) change\b",
            # (round 12) "How much did X move today?", "X fell yesterday, by how much?": a move on the day
            _DAY_MOVE,
        ),
    ),
    FrameMetric(
        "amount",
        "成交额",
        "turnover",
        "money",
        "get_price_history",
        TURNOVER_TERMS,
    ),
    FrameMetric(
        "volume",
        "成交量",
        "volume",
        "count",
        "get_price_history",
        ("成交量", r"(?:成交了?(?:多少|几[十百千万亿]*)(?:手|股))", r"\btrading volume\b", r"\bvolume\b"),
    ),
    FrameMetric(
        "close",
        "收盘价",
        "close",
        "price",
        "get_price_history",
        (
            "最新收盘价",
            "收盘价",
            "收盘",
            "股价",
            "价格",
            "净值",
            # (round 13) "Which of A and B closed higher?"
            r"\bclos(?:e|es|ed|ing price)\b",
            r"\bshare price\b",
            r"\bprice\b",
        ),
    ),
)
METRIC_BY_KEY = {metric.key: metric for metric in FRAME_METRICS}


def _literal_length(term: str) -> int:
    """The length of the text a term matches, for longest-first ordering ("净利率" before "净利")."""
    return len(re.sub(r"\(\?<?[!=][^)]*\)|\\b|[\\?()|:]", "", term))


_TERMS = sorted(
    ((term, metric.key) for metric in FRAME_METRICS for term in metric.terms),
    key=lambda item: _literal_length(item[0]),
    reverse=True,
)


def _alternative(index: int, term: str) -> str:
    body = term if term.startswith(("(", "\\")) else re.escape(term)
    return f"(?P<m{index}>{body})"


_METRIC_PATTERN = re.compile("|".join(_alternative(i, term) for i, (term, _key) in enumerate(_TERMS)), re.IGNORECASE)
_KEY_BY_GROUP = {f"m{index}": key for index, (_term, key) in enumerate(_TERMS)}
# "净利润是营收的百分之几", "净利润占营收多少", "营收里有多少变成净利润", "net profit as a percentage of revenue",
# (round 12) "What share of that revenue is left as net profit?": the net margin, not two metrics. Shared with
# ``coverage.METRICS`` (the net-margin metric), so the frame and the template read the same phrasings.
NET_MARGIN_SHARE_SOURCE = (
    # (round 12, H9) "一年赚的钱占收入多大比例": colloquial profit words
    r"(?:净利润|净利|净赚|利润|赚的钱|赚到的钱|赚的)[^，。？?,.!！]{0,4}?(?:是|为|占|相当于|等于|在)[^，。？?,.!！]{0,4}?"
    r"(?:营收|营业收入|收入|销售额)[^，。？?,.!！]{0,6}?(?:百分之|比例|比重|百分比|占比|几成|多少|多大|几)|"
    r"(?:净利润|净利|利润)(?:率)?(?:与|和|跟)(?:营收|营业收入|收入)(?:之)?比|"
    r"(?:营收|营业收入|收入|销售额)[^，。？?,.!！]{0,6}?(?:中|里)[^，。？?,.!！]{0,10}?(?:净利润|净利|净赚|利润)|"
    r"\b(?:net )?(?:profit|income|earnings) (?:is |as )?(?:a |what )?(?:share|percent(?:age)?|proportion|fraction) "
    r"of (?:the )?(?:revenue|sales)\b|"
    r"\b(?:percent(?:age)?|share|fraction|proportion|portion|part|how much) of (?:its |the |that |this |their |each |"
    r"every )?(?:revenue|sales|top line)\b[^.?!]{0,30}?\b(?:net )?(?:profit|income|earnings)\b|"
    r"\bwhat (?:percent(?:age)?|share|fraction) of (?:its |the |that |this |their )?(?:revenue|sales)\b"
)
_NET_MARGIN_SHARE = re.compile(NET_MARGIN_SHARE_SOURCE, re.IGNORECASE)


def metric_mentions(text: str) -> list[str]:
    """Frame metrics the text names, in order, each once (longest name first at each position)."""
    if _NET_MARGIN_SHARE.search(text or ""):
        return ["net_margin"]
    keys: list[str] = []
    for match in _METRIC_PATTERN.finditer(text or ""):
        key = _KEY_BY_GROUP[str(match.lastgroup)]
        if key not in keys:
            keys.append(key)
    return keys


def metric_of(text: str) -> str | None:
    """The first frame metric the text names, or ``None``."""
    found = metric_mentions(text)
    return found[0] if found else None


def metric_label(key: str, zh: bool) -> str:
    metric = METRIC_BY_KEY.get(key)
    if metric is None:
        return key
    return metric.zh if zh else metric.en


# --------------------------------------------------------------------------- questions that read the frame

# (round 12, H1) The operation of a comparison question is parsed from three kinds of words instead
# of a list of whole phrases:
# * a unit word saying what is asked: 倍 / 之比 / ratio / "1.2x" → a ratio; 百分之 / % / 几成 /
#   percent → a relative difference; 多少 / 几 / points / how much → a difference;
# * a comparative: 高 / 低 / 贵 / 多跌 / 少涨 / 多成交, higher / above / more;
# * an anchor relating two operands: A是B的, 比, 差, 相对, 前者 / 后者, 二者 / 两者 / 它们,
#   the former / the latter, than, them.
# "差了多少倍" is a ratio; "市盈率是多少倍" (one operand, no anchor) is not a comparison;
# "ROE是百分之多少" is not a relative difference.
#
# (round 13) Comparative verbs as a class: 领先 / 落后 / 跑赢 / 跑输 / 超出 / 胜过 / 输给, "outperform", "trail",
# "lead", "out-earn", "out-sell" relate two operands the way 高 / 低 / 比 do. With a unit word ("几个百分点", "多少
# 元") or a question quantity they ask a difference; with "by what multiple" / "几倍" a ratio. A colloquial count of one
# operand in the other ("相当于几个茅台", "抵得上几个", "顶得上几个") is a ratio, and so are imperatives ("divide the
# second by the first", "后者除以前者"); "subtract" / "减去" ask a difference.
# (round 14, review-10 J4) 高过 / 低过 / 强过 / 弱过 / 超过 / 多赚 / 少赚, "outstrip", "surpass", "edge out" join the
# class.
_VERB_COMPARATIVE_ZH = (
    r"领先|落后|跑赢|跑输|超出|胜过|胜出|输给|赢了|高出|低出|多出|少出|高过|低过|强过|弱过|超过|多赚|少赚|甩开"
)
_VERB_COMPARATIVE_EN = (
    r"\b(?:out-?(?:earn|sell|perform|pace|grow|gain|trade|rank)|underperform|lead|trail|lag|beat|exceed|top)"
    r"(?:s|ed|ing|ged)?\b|"
    r"\b(?:outstrip(?:s|ped|ping)?|surpass(?:es|ed|ing)?|outdo(?:es|ne|ing)?|outshine[sd]?|edge[sd]? (?:out|past))\b"
)
# "是平安的两倍吗", "是不是三倍左右", "Is it about twice Ping An's?": a ratio checked against a stated multiple
_CLAIMED_MULTIPLE_ZH = re.compile(
    r"(?:是|为|有|相当于|等于|是不是|有没有|是否|超过|不到|达到)[^，。？?,;；]{1,14}?的\s*"
    r"(?P<approx>大约|大概|约|将近|接近|差不多)?\s*(?P<k>\d+(?:\.\d+)?|[一两二三四五六七八九十]|半)\s*倍"
    # (round 14, review-10 J4) "三倍半": three and a half times
    r"(?P<half>半)?(?P<post>还要多|还多|以上|有余|多|左右|上下|不到)?\s*(?:吗|么|嘛|吧|没有|不|呢)?\s*[？?]?\s*$"
)
# "是不是平安的三成以上", "有没有茅台的六成": a share stated in tenths
_CLAIMED_TENTHS_ZH = re.compile(
    r"(?:是|为|有|相当于|等于|是不是|有没有|是否|超过|不到|达到)[^，。？?,;；]{1,14}?的\s*(?:大约|大概|约|将近|接近)?\s*"
    r"(?P<k>[一二两三四五六七八九]|[1-9])\s*成(?:以上|还多|多|左右|不到)?\s*(?:吗|么|嘛|吧|没有|不|呢)?\s*[？?]?\s*$"
)
_CLAIMED_HALF_ZH = re.compile(
    r"(?:是|为|有|相当于|等于|是不是|有没有|是否|超过|不到|达到)[^，。？?,;；]{1,14}?的\s*(?:大约|大概|约|将近|接近)?\s*一半"
    r"(?:还要多|还多|以上|多|左右|不到)?\s*(?:吗|么|嘛|吧|没有|不|呢)?\s*[？?]?\s*$"
)
# (round 13, review-9 I7) the bound a yes/no multiple question asks about: "两倍还多 / 以上 / more than double" (above),
# "不到两倍 / less than twice" (below), else "about"
_ABOVE_WORDS = re.compile(
    r"超过|多于|高于|至少|还要多|还多|以上|有余|有没有|达到|够不够|"
    r"\bexceeds?\b|\bmore than\b|\bover\b|\bat least\b|\babove\b",
    re.I,
)
_BELOW_WORDS = re.compile(r"不到|不足|少于|低于|\bless than\b|\bunder\b|\bbelow\b|\bnot even\b", re.I)
# the multiple stands before the other operand: "twice Ping An's", "three times as high as the latter", "double that"
_CLAIMED_MULTIPLE_EN = re.compile(
    r"^(?:is|are|was|were|does|do|did|has)\b[^.?!]{0,60}?\b"
    r"(?P<approx>about |roughly |around |nearly |approximately |close to )?"
    r"(?P<k>twice|double|triple|half|"
    # (round 14, review-10 J4) "over eight times", "three and a half times"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)(?: and a half)? times|"
    r"\d+(?:\.\d+)?\s?(?:x|times))\s+"
    r"(?:(?:as \w+ as|that of|those of|of|the size of|what)\s+)?(?:the\s+)?"
    r"(?:latter|former|first|second|that|its|their|his|her|(?-i:[A-Z])[\w-]*)",
    re.IGNORECASE,
)
_RATIO_EXPLICIT = re.compile(
    r"倍数关系|几倍关系|比值|之比|几倍于|\bratios?\b|\bhow many times\b|"
    r"\btimes (?:as (?:high|large|big|much)|bigger|larger|higher|more|greater|the)\b|\bmultiple of\b|"
    r"(?<![\w.])\d+(?:\.\d+)?\s?[x×](?![A-Za-z])|"
    r"(?:相当于|抵得上|抵得过|顶得上|顶得过|能顶|能抵|顶|抵)(?:了)?(?:几|多少)个|除以|"
    r"\bby what (?:multiple|factor)\b|\bwhat multiple\b|\bhow many [\w' -]{1,30}?(?:'s|’s) worth\b|"
    r"\bdivide\b[^.?!]{0,40}?\bby\b|\bdivided by\b",
    re.IGNORECASE,
)
_RATIO_UNIT = re.compile(r"(?:几|多少)倍")
_ANCHOR = re.compile(
    r"(?:是|为|相当于|等于)[^，。？?,;；]{1,14}?的|比(?!例|较|重)|差|相对|较之|前者|后者|前一|后一|第一个|第二个|二者|两者|"
    r"两个|两家|两只|俩|它们|(?<=[一-鿿A-Za-z0-9])、(?=[一-鿿A-Za-z0-9])|"
    r"\bthe former\b|\bthe latter\b|\bthan\b|\bthem\b|\bthe two\b|\bboth\b|\bversus\b|\bvs\.?|"
    + _VERB_COMPARATIVE_ZH
    + "|"
    + _VERB_COMPARATIVE_EN,
    re.IGNORECASE,
)
_RELATIVE_EXPLICIT = re.compile(
    r"折价|溢价|\b(?:premium|discount)\b|"
    r"\bby what (?:percent(?:age)?|share)\b|"
    r"\bpercent(?:age)? (?:higher|lower|more|less|above|below|premium|discount|bigger|smaller|cheaper)\b",
    re.IGNORECASE,
)
_RELATIVE_UNIT = re.compile(
    r"百分之(?:多少|几)|(?:多少|几)(?:个)?(?:百分比|%)|几成|多少成|\s%|\bhow many percent\b|"
    r"\bin percent(?:age)?(?! points?)(?: terms)?\b|\bpercentage[- ]wise\b|\bas a percent(?:age)?\b",
    re.IGNORECASE,
)
# a comparative: 高 / 低 / 贵 / 便宜 / 大 / 小, "多" / "少" only before a verb or 了 / 出 (not the 多少 of a question)
_COMPARATIVE = re.compile(
    r"高|低|贵|便宜|(?<!多)大|(?<!多|大)小|超出|超过|领先|落后|多(?=[了出跌涨赚亏卖成交])|(?<!多)少(?=[了出跌涨赚亏卖成交])|"
    r"\b(?:higher|lower|more|less|above|below|under|bigger|smaller|larger|cheaper|greater)\b",
    re.IGNORECASE,
)
# "差几个点", "相差多少", "高了多少", "大多少", "difference", "gap", "how much higher", "by how much".
_DIFFERENCE = re.compile(
    r"差了?(?:有|是|大概|大约)?(?:多少|几|多大)|相差|差距|差额|差值|差多少|"
    # (round 14, review-10 J4) "强多少", "弱了几个点"
    r"(?:高|低|多|少|大|小|贵|便宜|强|弱)(?:了|出)?(?:有|是|大概|大约)?(?:多少|几)|"
    # (round 12) "多跌了多少", "少涨了几个点", "多成交了多少钱": more / less of a verb, by how much
    r"(?:多|少)(?:跌|涨|赚|亏|卖|成交|交易)了?(?:有|是|大概|大约)?(?:多少|几)|"
    # (round 13) a comparative verb, its object, then the quantity asked: "领先前者几个百分点", "落后茅台多少",
    # "跑赢五粮液几个点", "超出行业多少"; "subtract", "减去", "相减"
    rf"(?:{_VERB_COMPARATIVE_ZH})了?[^，。？?,;；]{{0,12}}?(?:有|是|大概|大约)?(?:多少|几)|减去|相减|"
    r"\bsubtract\b|\bminus\b|"
    r"\bdifferences?\b|\bgap\b|\bspread\b|\bdiffer\b|"
    r"\bhow much (?:higher|lower|more|less|bigger|smaller|larger|cheaper)\b|"
    r"\bby how (?:much|many)\b|\bhow far (?:apart|above|below|under|ahead|behind)\b|"
    r"\bhow many (?:more |fewer )?(?:percentage |basis )?points\b|\b(?:points?|percent) apart\b",
    re.IGNORECASE,
)
# "谁更高", "哪个更大", "which one is lower": which side is higher (the answer states both values).
_WHICH = re.compile(
    r"(?:谁|哪个|哪一个|哪只|哪家|哪边|哪一家|哪一只)[^，。？?,;；]{0,6}?(?:更|比较|相对)?"
    r"(?:高|低|大|小|多|少|贵|便宜|强|弱|活跃)(?!档|端|效|级|点位)|"
    r"\bwhich\b[^.?!]{0,30}\b(?:higher|lower|bigger|smaller|larger|more|less|cheaper|greater)\b|"
    r"\b(?:higher|lower|bigger|larger) of the two\b|"
    # (round 14, review-10 J14) "比行业平均高还是低", "is it above or below the sector average?"
    r"(?:高|低|大|小|多|少)还是(?:高|低|大|小|多|少)|"
    r"\b(?:higher|lower|above|below|bigger|smaller|more|less) or "
    r"(?:higher|lower|above|below|bigger|smaller|more|less)\b",
    re.IGNORECASE,
)
# (round 13) a question about two of several operands ("两者", "the two", "前者/后者")
_TWO = re.compile(r"两者|二者|两个|两家|两只|俩|\bthe two\b|\bboth\b|\bthe pair\b", re.IGNORECASE)
# (round 13) "Divide the second by the first": a bare "the first / the second" before by / from / and / an end
_BARE_ORDINAL_END = r"(?=\s+(?:by|from|and|to|over|minus|plus|divided|times|one)\b|\s*['’]s\b|\s*[.?!,;]|\s*$)"
_ORDINAL = re.compile(
    r"(?P<first>前者|前一个|前一家|前一只|第一个|第一家|\bthe former\b|\bthe first (?:one|company|stock|fund)\b|"
    rf"\bthe first{_BARE_ORDINAL_END})|"
    r"(?P<last>后者|后一个|后一家|后一只|第二个|第二家|\bthe latter\b|\bthe second (?:one|company|stock|fund)\b|"
    rf"\bthe second{_BARE_ORDINAL_END})",
    re.IGNORECASE,
)
# Words that make a gap question about something else: advice, forecasts, news.
_NOT_A_GAP = re.compile(
    r"值得|该不该|要不要|能不能买|买哪|选哪|推荐|会涨|会跌|明天|下周|未来|预测|新闻|公告|为什么|原因|"
    r"\bshould i\b|\bbuy\b|\bsell\b|\bforecast\b|\bwhy\b|\bnews\b",
    re.IGNORECASE,
)
# (round 12, H11) a question with more than one clause ("how big is the discount? Someone told me …"): the clause
# that asks the comparison is parsed; the cap applies to that clause, not to the whole message
_CLAUSE = re.compile(r"[^。！？!?；;]+[。！？!?；;]?")
MAX_CLAUSE_CHARS = 100
MAX_CLAUSE_WORDS = 25


def _bound(text: str) -> str:
    return "gt" if _ABOVE_WORDS.search(text) else "lt" if _BELOW_WORDS.search(text) else "eq"


def claimed_check(text: str) -> dict[str, Any] | None:
    """The stated multiple with the bound the question asks about: ``{multiple, written, approx, bound}``."""
    found = claimed_multiple(text)
    if found is None:
        return None
    return {"multiple": found[0], "written": found[1], "approx": found[2], "bound": _bound(text)}


def claimed_multiple(text: str) -> tuple[float, str, bool] | None:
    """(round 13) ``(multiple, as written, approximate)`` when a ratio question states the multiple it checks ("五粮液
    的股价大约是平安的两倍吗" → (2.0, "两", True); "Is Moutai's P/B roughly twice Ping An's?" → (2.0, "twice", True)),
    else ``None``. The answer computes the ratio and says whether the stated multiple holds."""
    text = (text or "").strip()
    tenths = _CLAIMED_TENTHS_ZH.search(text)
    if tenths is not None:
        raw = tenths.group("k")
        value = (float(raw) if raw.isdigit() else float(_CN_SMALL.get(raw, 0))) / 10
        return (value, f"{raw}成", bool(re.search(r"大约|大概|约|将近|接近|左右", text))) if value else None
    half = _CLAIMED_HALF_ZH.search(text)
    if half is not None:
        return 0.5, "一半", bool(re.search(r"大约|大概|约|将近|接近|差不多|左右", text))
    found = _CLAIMED_MULTIPLE_ZH.search(text)
    if found is not None:
        raw = found.group("k")
        value = 0.5 if raw == "半" else float(raw) if raw[0].isdigit() else float(_CN_SMALL.get(raw, 0))
        approx = bool(found.group("approx") or re.search(r"大约|大概|约|将近|接近|差不多|左右|上下", text))
        if found.group("half") and raw != "半":
            return (value + 0.5, f"{raw}倍半", approx) if value > 0 else None
        return (value, raw, approx) if value > 0 else None
    found = _CLAIMED_MULTIPLE_EN.search(text)
    if found is not None:
        raw = found.group("k").strip()
        words = {"twice": 2.0, "double": 2.0, "triple": 3.0, "half": 0.5}
        number = re.match(r"\d+(?:\.\d+)?", raw)
        spelled = re.match(r"(\w+)( and a half)? times", raw.lower())
        if number:
            value = float(number.group(0))
        elif spelled and spelled.group(1) in _EN_SMALL:
            value = _EN_SMALL[spelled.group(1)] + (0.5 if spelled.group(2) else 0.0)
        else:
            value = words.get(raw.lower(), 0.0)
        approx = bool(found.group("approx")) or bool(
            re.search(r"\b(?:about|roughly|around|nearly|approximately|close to)\b", text, re.IGNORECASE)
        )
        return (value, raw, approx) if value > 0 else None
    return None


_CN_SMALL = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_EN_SMALL = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "fifteen": 15.0,
    "twenty": 20.0,
}
# (round 13) a total of two operands: "加起来", "合计", "一共", "combined", "in total", "the sum of"
# (round 14, review-10 J4) 总计 / 共计 / 合在一块 / 加总 / "altogether" / "all told" as well
_SUM = re.compile(
    r"加起来|加(?:在|到)?一(?:块|起)|合(?:在)?一(?:起|块)|合起来|合计|一共|总共|总和|之和|相加|再加上|总计|共计|加总|"
    r"\bcombined\b|\bin total\b|\btogether\b|\bsum of\b|\badd(?:ed|s)? up\b|\bput together\b|"
    r"\btotal (?:of|for)\b|\bin aggregate\b|\baltogether\b|\ball told\b",
    re.IGNORECASE,
)
# (round 14, review-10 J4) weaker total words, read only after the gap, ratio, share and which productions: 加 between
# two names ("工行加招行的净利润"; not 加权 / 加息 / 加仓 / 再加800股), "plus", "between them" ("how much do they earn
# between them"; "the gap between them" is a gap), "add X to it", "what's the total"
_SUM_WEAK = re.compile(
    r"(?<=[一-鿿A-Za-z）)])加(?![权息速大入仓码剧强快紧倍盟油班薪工减乘除上起一在到])(?=[一-鿿A-Za-z])|"
    r"\bplus\b|\bbetween (?:them|the two(?: of them)?|both of them)\b|"
    r"\b(?:what(?:'s| is| are)|how (?:much|big) is) the (?:total|sum)\b|\btotal\s*[?？]\s*$|\bsum them\b",
    re.IGNORECASE,
)
# a bare total asked of the operands under discussion: "Total?", "And combined?", "合计呢", "一共多少"
_SUM_BARE = re.compile(
    r"(?:^|[.。,，]\s*)(?:and\s+|so\s+)?(?:the\s+|in\s+)?(?:total|combined|altogether|sum)\s*[?？]\s*$|"
    r"(?:^|[，,。]\s*)(?:那)?(?:合计|总共|一共|加起来|总计|共计)(?:呢|多少|是多少|有多少)\s*[？?]?\s*$",
    re.IGNORECASE,
)
# "Add CMB's to ICBC's", "add the second to it": a total of the two named or referred operands
# (not "add more to …", "add to my position": buying)
_SUM_ADD_TO = re.compile(
    r"\badd(?:ing)?\s+(?!(?:more|some|a bit|a little|to)\b)[^.?!]{1,60}?\bto\s+(?!(?:my|your|our|a|an)\b)",
    re.IGNORECASE,
)
_JOINED = re.compile(
    r"和|与|跟|及|、|(?<!再)加(?![起一在到权息仓])|\band\b|\bof\b|\bboth\b|\btheir\b|\bthese\b|\bthose\b|\bplus\b|\bit\b|"
    r"\bthat\b|\bthis\b",
    re.IGNORECASE,
)
_QUANTITY = re.compile(r"多少|几|\bhow (?:much|many|big)\b|\bwhat\b|\btotal\b|[?？]", re.IGNORECASE)
# (round 14, review-10 J3) a share of a combined total: "A占两家合计的多少", "两者合计中A占几成", "A's revenue as a
# percentage of A and B combined", "what share of their combined profit": A ÷ (A + B), never A ÷ B
_TOTAL_WORDS_ZH = r"合计|总和|之和|加起来|加在一起|合起来|合在一起|总共|一共|共计|总计|总量|两(?:家|者|只|个)(?:之)?和"
_SHARE_TOTAL = re.compile(
    rf"(?:占|是|为|相当于)[^，。？?；;]{{0,24}}?(?:{_TOTAL_WORDS_ZH})(?:的?(?:数|值|数字|金额))?[^，。？?；;]{{0,4}}?的?\s*"
    r"(?:大约|大概|约)?\s*(?:百分之(?:多少|几)|(?:几|多少)成|(?:多少|几)(?:个)?(?:百分比|%)|多大比例|多少比例|多大比重|"
    r"几分之几|比重|比例|占比|份额|多少|几)|"
    rf"(?:{_TOTAL_WORDS_ZH})(?:里|中|之中|当中|里面)[^。？?；;]{{0,16}}?(?:占|是)(?:比)?[^，。？?；;]{{0,4}}?"
    r"(?:多少|几成|百分之|多大|比例|比重|几)|"
    r"\b(?:share|percent(?:age)?|fraction|proportion|portion|part|slice)\b[^.?!]{0,60}?"
    r"\b(?:combined|together|put together|joint|between them|in total|their total|the total|the sum|"
    r"(?:both|the two)(?: companies| firms| banks| stocks)?['’]? total)\b|"
    # "Of A and B's combined net profit, what percentage is A's?"
    r"\b(?:combined|joint)\b[^.?!]{0,60}?\b(?:what|how (?:big|large|much)(?: of)?)\s+(?:a\s+)?"
    r"(?:share|percent(?:age)?|fraction|proportion|portion|part)\b",
    re.IGNORECASE,
)
# a metric whose name starts with "total" is not a combined total ("total market cap", "CATL's total revenue")
_TOTAL_METRIC_NAME = re.compile(
    r"\btotal (?=market (?:cap|value|capitali[sz]ation)\b)|(?<=['’]s )total (?=(?:operating )?(?:revenue|sales)\b)",
    re.IGNORECASE,
)
# (round 13, review-9 I1) a share of another value: "A相当于B的百分之多少", "A是B的几成", "A占B的百分之几",
# "what percent of B is A", "A is what percentage of B", "as a percentage of B": A ÷ B × 100, not the relative
# difference (A − B) ÷ B
_SHARE = re.compile(
    r"(?:相当于|是|占|为|等于|只有|才)[^，。？?,;；]{1,14}?的\s*(?:大约|大概|约)?\s*"
    r"(?:百分之(?:多少|几)|(?:几|多少)成|(?:多少|几)(?:个)?(?:百分比|%)|多大比例|多少比例|几分之几|比重|比例|占比)|"
    r"\bwhat (?:percent(?:age)?|share|fraction|proportion|portion) of\b|"
    r"\b(?:is|are|was|were)\s+what\s+(?:percent(?:age)?|share|fraction|proportion)\s+of\b|"
    r"\bas an? (?:percent(?:age)?|share|fraction|proportion) of\b|\bhow many percent of\b",
    re.IGNORECASE,
)
# (round 13, review-9 I4) the spread of several operands: "最高的比最低的高多少", "highest minus lowest", "the spread
# between the highest and the lowest", "最大和最小差多少"; (round 14, review-10 J4) "the most and the least expensive",
# "the cheapest and the priciest" (a valuation spread, P/E unless another metric is named)
_SPREAD = re.compile(
    r"最(?:高|大|多|贵)[^，。？?,;；]{0,10}?(?:比|和|与|跟|减|减去|-|−)[^，。？?,;；]{0,6}?最(?:低|小|少|便宜)|"
    r"最(?:低|小|少|便宜)[^，。？?,;；]{0,10}?(?:比|和|与|跟)[^，。？?,;；]{0,6}?最(?:高|大|多|贵)|极差|"
    r"\b(?:highest|largest|biggest|top)\b[^.?!]{0,30}?\b(?:minus|and|less|over|vs\.?|versus|to)\b[^.?!]{0,12}?"
    r"\b(?:lowest|smallest|bottom)\b|\b(?:lowest|smallest)\b[^.?!]{0,30}?\b(?:and|to|vs\.?)\b[^.?!]{0,12}?"
    r"\b(?:highest|largest|biggest)\b|\brange (?:between|from)\b|"
    r"\b(?:most|least) expensive\b[^.?!]{0,40}?\b(?:and|vs\.?|versus|to|minus|from)\b[^.?!]{0,20}?"
    r"\b(?:least|most) expensive\b|\b(?:most|least) and (?:the )?(?:least|most) expensive\b|"
    r"\bcheapest\b[^.?!]{0,40}?\b(?:and|vs\.?|versus|to|minus|from)\b[^.?!]{0,20}?"
    r"\b(?:most expensive|priciest|dearest)\b|"
    r"\b(?:most expensive|priciest|dearest)\b[^.?!]{0,40}?\b(?:and|vs\.?|versus|to|minus|from)\b[^.?!]{0,20}?"
    r"\bcheapest\b",
    re.IGNORECASE,
)
# a spread of how "expensive" the operands are, with no metric named: P/E
_EXPENSIVE = re.compile(r"贵|便宜|\bexpensive\b|\bcheap(?:est|er)?\b|\bpric(?:ey|iest)\b|\bdearest\b", re.IGNORECASE)
# A single target's own extremes ("最高价和最低价", "52周最高和最低收盘", "highest and lowest close in the last
# 52 weeks")
# are price details of one target, not the spread of several operands
_OWN_EXTREMES = re.compile(
    r"最(?:高|低)(?:价|点|位|收盘|成交)|(?:52|五十二)\s*周|近一年|一年来|今年以来|今天|当日|盘中|日内|"
    r"\b(?:highest|lowest)\s+(?:close|closing|price|level|point|print)s?\b|\b52[- ]week\b|\bintraday\b|"
    r"\b(?:today|this year|last year|past year|in the last)\b",
    re.IGNORECASE,
)
# (round 13, review-9 I6) how many shares of B one share of A buys: "一股茅台能换几股五粮液" (a ratio of the closes)
_SWAP_SHARES = re.compile(
    r"(?:能|可以)?换(?:成)?(?:几|多少)(?:股|份|手)|\bhow many (?:shares|units) of\b[^.?!]{0,40}\bone\b", re.I
)
SUM_KINDS = frozenset({"money", "count"})  # a total of P/Es, prices or rates is not a value


# (round 13) "the P/E ratios of A and B": a metric's name, not a question for their ratio
_METRIC_RATIO_NAME = re.compile(
    r"(?:(?<![A-Za-z])P/?[EB]|\bprice[- ]to[- ](?:earnings|book)|\bpayout|\bdebt(?:[- ]to[- ]equity)?|\bsharpe)"
    r"\s+ratios?\b",
    re.IGNORECASE,
)
# (round 14) how a comparison refers to its operands besides naming them
_ANAPHORA = re.compile(
    r"两者|二者|两个|两家|两只|三家|三只|三者|俩|它们|他们|它|其|\bthe two\b|\bboth\b|\bthem\b|\bthey\b|\btheir\b|"
    r"\bthe pair\b|\bit\b|\bthat\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- the comparison grammar (round 14)

# (round 14, review-10 J4) A comparison question is read as OPERATION × OPERANDS × METRIC instead of a list of
# phrase classes checked in turn:
#
#   OPERATION  ← the first production of ``GRAMMAR`` whose word classes (``_LEXICON``) all occur in the clause
#                and none of whose excluded classes does;
#   OPERANDS   ← named (two or more targets the question names) | named + carried (one named, the frame's other) |
#                ordinal (前者 / 后者 / the former) | anaphoric (两者 / 它们 / both / them / between them) |
#                carried (none written: the frame's);
#   METRIC     ← the longest metric name in the clause (``metric_mentions``), else the frame's; a spread of how
#                "expensive" the operands are with no metric named is a spread of P/E.
#
# The word classes are still regular expressions (a lexicon), but which operation a question asks is decided by one
# ordered table, so a new wording is a new word in a class, and ``tests/test_agent_round14_grammar.py`` generates
# wordings from templates (operation phrases × entity pairs × metric names, zh and en, seeded) and checks the
# operation, the operands and the computed value. Rule: a two-operand question is never answered with one operand —
# it is computed, or the answer says which operand could not be identified (``unresolved`` operands).


@dataclass(frozen=True)
class Production:
    operation: str
    all_of: tuple[str, ...]
    none_of: tuple[str, ...] = ()


_LEXICON: dict[str, Any] = {
    "EXTREME_PAIR": lambda text: _SPREAD.search(text),
    "OWN_EXTREMES": lambda text: _OWN_EXTREMES.search(text),
    "SHARE_OF_TOTAL": lambda text: _SHARE_TOTAL.search(text),
    "SHARE": lambda text: _SHARE.search(text),
    "CLAIMED_TENTHS": lambda text: _CLAIMED_TENTHS_ZH.search(text),
    "RATIO_WORD": lambda text: _RATIO_EXPLICIT.search(text),
    "SWAP": lambda text: _SWAP_SHARES.search(text),
    "RATIO_UNIT": lambda text: _RATIO_UNIT.search(text),
    "ANCHOR": lambda text: _ANCHOR.search(text),
    "CLAIMED_MULTIPLE": lambda text: claimed_multiple(text) is not None,
    "RELATIVE_WORD": lambda text: _RELATIVE_EXPLICIT.search(text),
    "RELATIVE_UNIT": lambda text: _RELATIVE_UNIT.search(text),
    "COMPARATIVE": lambda text: _COMPARATIVE.search(text),
    "SUM_WORD": lambda text: _SUM.search(text),
    "SUM_WEAK": lambda text: _SUM_WEAK.search(text),
    "SUM_BARE": lambda text: _SUM_BARE.search(text),
    "SUM_ADD_TO": lambda text: _SUM_ADD_TO.search(text),
    "JOINED": lambda text: _ANCHOR.search(text) or _JOINED.search(text),
    "QUANTITY": lambda text: _QUANTITY.search(text),
    "DIFFERENCE": lambda text: _DIFFERENCE.search(text),
    "WHICH": lambda text: _WHICH.search(text),
}
GRAMMAR: tuple[Production, ...] = (
    Production("spread", ("EXTREME_PAIR",), ("OWN_EXTREMES",)),
    Production("share_total", ("SHARE_OF_TOTAL",)),
    Production("share", ("SHARE",)),
    Production("share", ("CLAIMED_TENTHS",)),  # "是不是平安的三成以上": a share in 成, checked against the tenths
    Production("ratio", ("RATIO_WORD",)),
    Production("ratio", ("SWAP",)),
    Production("ratio", ("RATIO_UNIT", "ANCHOR")),
    Production("ratio", ("CLAIMED_MULTIPLE",)),  # "有没有…的三倍半", "over eight times …?": yes / no and the ratio
    Production("relative", ("RELATIVE_WORD",)),
    Production("relative", ("RELATIVE_UNIT", "COMPARATIVE")),
    Production("relative", ("RELATIVE_UNIT", "ANCHOR")),
    Production("sum", ("SUM_WORD", "JOINED")),
    Production("difference", ("DIFFERENCE",)),
    Production("which", ("WHICH",)),
    Production("sum", ("SUM_WEAK", "JOINED", "QUANTITY")),
    Production("sum", ("SUM_ADD_TO", "QUANTITY")),
    Production("sum", ("SUM_BARE",)),
)
COMPARISON_OPERATIONS = frozenset(production.operation for production in GRAMMAR)


@dataclass(frozen=True)
class FrameParse:
    """One comparison clause read by the grammar: the operation, how it refers to its operands besides naming them
    (``ordinal``, ``anaphoric`` or ``none``), the metric it names and the word classes that decided it."""

    operation: str | None
    reference: str
    metric: str | None
    classes: tuple[str, ...]


def _normalised(text: str) -> str:
    return _TOTAL_METRIC_NAME.sub("", _METRIC_RATIO_NAME.sub("metric", text))


def parse_clause(text: str) -> FrameParse:
    """The grammar's reading of one clause (see ``GRAMMAR``)."""
    clause = _normalised(text or "")
    seen: dict[str, bool] = {}

    def has(name: str) -> bool:
        if name not in seen:
            seen[name] = bool(_LEXICON[name](clause))
        return seen[name]

    operation = next(
        (
            production.operation
            for production in GRAMMAR
            if all(has(name) for name in production.all_of) and not any(has(name) for name in production.none_of)
        ),
        None,
    )
    reference = "ordinal" if _ORDINAL.search(clause) else "anaphoric" if _ANAPHORA.search(clause) else "none"
    classes = tuple(sorted(name for name, present in seen.items() if present))
    return FrameParse(operation, reference, metric_of(text or ""), classes)


def _operation_of(text: str) -> str | None:
    return parse_clause(text).operation


def comparison_clause(text: str) -> str | None:
    """The clause of the question that asks a comparison (the whole question when it has one clause)."""
    text = (text or "").strip()
    for clause in (match.group(0).strip() for match in _CLAUSE.finditer(text)):
        if clause and len(clause) <= _clause_cap(clause) and _operation_of(clause):
            return clause
    return None


def _clause_cap(clause: str) -> int:
    """(round 14) An English clause spells its names out ("China Yangtze Power and China Merchants Bank combined"),
    so its cap is in words (25), not characters; a Chinese clause keeps ``MAX_CLAUSE_CHARS``."""
    if re.search(r"[一-鿿]", clause):
        return MAX_CLAUSE_CHARS
    return MAX_CLAUSE_CHARS if len(clause.split()) > MAX_CLAUSE_WORDS else len(clause)


def frame_operation(text: str) -> str | None:
    """The comparison a question asks (``difference``, ``ratio``, ``relative``, ``share``, ``share_total``, ``sum``,
    ``spread`` or ``which``), else ``None``.

    A net-margin question written as a share ("净利润是营收的百分之几") is one metric, not a comparison; advice,
    forecast and news questions are not comparisons either."""
    from .coverage import holding_value_request

    text = (text or "").strip()
    if not text or _NOT_A_GAP.search(text) or _NET_MARGIN_SHARE.search(text):
        return None
    if holding_value_request(text) is not None:
        return None  # "我有2000股，合计值多少钱": a holding's value, not a total of two operands
    clause = comparison_clause(text)
    return _operation_of(clause) if clause else None


def without_valuation_extremes(text: str) -> str:
    """(round 14, review-10 J4) "最贵的和最便宜的差多少", "the most and the least expensive on P/E": a spread of a
    valuation metric is arithmetic, not a valuation judgment, so the extremes words are blanked for the judgment check
    (every other judgment word stays)."""
    if not text or not _SPREAD.search(_normalised(text)) or _NOT_A_GAP.search(text):
        return text
    return re.sub(
        r"最(?:贵|便宜)|\b(?:most|least) (?:and (?:the )?(?:least|most) )?expensive\b|\bcheapest\b|\bpriciest\b|"
        r"\bdearest\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )


def is_frame_question(text: str) -> bool:
    return frame_operation(text) is not None


def opening_comparison(text: str) -> bool:
    """(round 13) A gap, ratio, relative or total question that refers to its operands by order ("the second", "前者")
    or names the metric compared: a finance comparison even when nothing in the session supplies the operands."""
    operation = frame_operation(text)
    return operation in {"difference", "ratio", "relative", "sum", "share", "share_total", "spread"} and bool(
        _ORDINAL.search(text or "") or metric_of(text or "") or _TWO.search(text or "")
    )


# --------------------------------------------------------------------------- the frame


def operand_key(operand: dict[str, Any]) -> str:
    return str(operand.get("symbol") or f"industry:{operand.get('industry')}")


def operand_name(operand: dict[str, Any], zh: bool) -> str:
    if operand.get("kind") == "industry":
        from .names import INDUSTRY_EN

        industry = str(operand.get("industry") or "")
        if not industry:
            return "所属行业平均" if zh else "its industry average"
        return f"{industry}行业平均" if zh else f"the {INDUSTRY_EN.get(industry, industry)} industry average"
    return str(operand.get("name") or operand.get("symbol") or "")


_INDUSTRY_WORDS = re.compile(
    r"行业|板块|同行|同业|[一-鿿]{1,4}业(?:的)?(?:平均|整体|均值)|平均水平|\bsector\b|\bindustry\b|\bpeers?\b",
    re.IGNORECASE,
)
# Aspects that are not frame metrics: a turn about them ends the frame's metric ("茅台最近走势怎么样").
_OTHER_ASPECT = re.compile(
    r"走势|行情|新闻|公告|舆情|情绪|均线|波动率|技术|分红|业绩|财报|估值|基本面|"
    r"\btrend\b|\bnews\b|\bannouncements?\b|\bsentiment\b|\bvolatility\b|\bmoving average\b|\bvaluation\b",
    re.IGNORECASE,
)


def _short(text: str) -> bool:
    return len(text) <= 24 if not re.search(r"[A-Za-z]{3,}", text) else len(text.split()) <= 10


def _industry_operands(query: str, targets: list[dict[str, Any]], tool_log: list[dict[str, Any]]) -> list[dict]:
    """The industry average a turn asked about ("保险行业平均是多少", "and the sector average?"), from the industry
    snapshot that came with a target's fundamentals."""
    if not _INDUSTRY_WORDS.search(query or ""):
        return []
    symbols = {str(target.get("symbol")) for target in targets}
    for entry in tool_log:
        data = entry.get("data") or {}
        industry = (data.get("industry") or {}).get("industry_name")
        if entry.get("tool") != "get_fundamentals" or not entry.get("ok") or not industry:
            continue
        if symbols and str(data.get("symbol") or "") not in symbols:
            continue
        return [{"kind": "industry", "industry": str(industry), "member": data.get("symbol") or industry}]
    return []


def operand_value(tool_log: list[dict[str, Any]], operand: dict[str, Any], key: str) -> tuple[Any, Any, dict] | None:
    """``(value, evidence_id, data)`` of ``key`` for one operand in this run's tool results, or ``None``."""
    from .composer import metric_value

    if operand.get("kind") == "unresolved":
        return None
    for entry in tool_log:
        if not entry.get("ok"):
            continue
        data = entry.get("data") or {}
        if operand.get("kind") == "industry":
            snapshot = data.get("industry") or {}
            wanted = operand.get("industry")
            same = (
                snapshot.get("industry_name") == wanted if wanted else str(data.get("symbol")) == operand.get("member")
            )
            if entry.get("tool") != "get_fundamentals" or not same:
                continue
            value = (snapshot.get("metrics") or {}).get(key)
            if value is not None and snapshot.get("evidence_id") and key in {"pe", "pb", "pct_change"}:
                return float(value), snapshot["evidence_id"], {"industry_name": snapshot.get("industry_name")}
            continue
        if str(data.get("symbol") or "") != str(operand.get("symbol") or ""):
            continue
        found = metric_value(entry, key)
        if found is not None and data.get("evidence_id"):
            value, data = found
            return value, data.get("evidence_id"), data
    return None


def next_frame(
    previous: dict[str, Any] | None,
    *,
    query: str,
    route: str,
    targets: list[dict[str, Any]],
    tool_log: list[dict[str, Any]],
    request: dict[str, Any] | None = None,
    named: bool = True,
) -> dict[str, Any] | None:
    """The frame after a finished turn (see the module docstring).

    (round 12, H11) ``named`` is False when the turn's targets were carried by a rewrite ("那PB呢"): a metric switch
    on a pair under comparison keeps the pair ("茅台PE → 五粮液呢 → 那PB呢 → 差多少" compares both PBs).

    ``targets`` are the turn's listed targets in order of mention (``{"name", "symbol"}``). Refusals and
    clarifications keep the previous frame; a turn about something that is not a frame metric (a trend, news) with
    targets of its own starts a frame without a metric, so a later gap question asks which metric is meant."""
    previous = previous or None
    if route in {"refuse", "clarify"}:
        return previous
    if request and any(item.get("metric") for item in request.get("operands") or []):
        return previous  # (round 13) two metrics of one target: the frame stays as it was
    if request and request.get("metric") and request.get("operands"):
        metric = str(request["metric"])
        operands = [dict(item) for item in request["operands"] if item.get("kind") != "unresolved"]
    else:
        metric = metric_of(query)
        operands = [
            {"kind": "target", "name": str(t.get("name") or t.get("symbol")), "symbol": str(t["symbol"])}
            for t in targets
            if t.get("symbol")
        ]
        operands += _industry_operands(query, targets, tool_log)
        if not operands:
            return previous
        if metric is None and previous and previous.get("metric") and _short(query) and not _OTHER_ASPECT.search(query):
            # "保险行业平均是多少", "五粮液呢", "and Moutai's?": the same metric for another operand
            metric = str(previous["metric"])
        earlier = [dict(item) for item in (previous or {}).get("operands") or []]
        if (
            metric
            and previous
            and previous.get("metric")
            and previous.get("metric") != metric
            and not named
            and len(earlier) >= 2
            and {operand_key(item) for item in operands} <= {operand_key(item) for item in earlier}
        ):
            operands = [
                {key: item[key] for key in ("kind", "name", "symbol", "industry", "member") if key in item}
                for item in earlier
            ]
        elif metric and previous and previous.get("metric") == metric:
            merged = {operand_key(item): item for item in previous.get("operands") or []}
            for item in operands:
                merged.pop(operand_key(item), None)
                merged[operand_key(item)] = item
            operands = list(merged.values())
    operands = operands[-MAX_OPERANDS:]
    for operand in operands:
        found = operand_value(tool_log, operand, metric) if metric else None
        if found is not None:
            operand["value"], operand["evidence_id"] = found[0], found[1]
        elif previous and previous.get("metric") == metric:
            # keep the value an earlier turn found (this turn did not fetch it again)
            earlier = next(
                (item for item in previous.get("operands") or [] if operand_key(item) == operand_key(operand)), {}
            )
            if earlier.get("value") is not None:
                operand["value"], operand["evidence_id"] = earlier["value"], earlier.get("evidence_id")
    return {"metric": metric, "operands": operands}


def latest_frame(turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    for turn in reversed(turns):
        if "frame" in turn:
            return turn.get("frame") or None
    return None


def _mention_position(text: str, entity: dict[str, Any], fallback: int) -> int:
    """Where the question names an entity (its mention, canonical name or English name), for operand order."""
    from .names import english_aliases, english_name, load_synonyms

    lowered = text.lower()
    canonical = str(entity.get("canonical_name") or entity.get("name") or "")
    names = [entity.get("mention"), canonical, english_name(canonical), *english_aliases(canonical)]
    names += [alias for alias, name in (load_synonyms().get("alias") or {}).items() if name == canonical]
    if len(canonical) >= 4 and re.fullmatch(r"[一-鿿]+", canonical) and canonical[2:] not in GENERIC_NAME_TAILS:
        names.append(canonical[2:])  # 贵州茅台 → 茅台, 中国平安 → 平安
    found = [lowered.find(str(name).lower()) for name in names if name]
    found = [position for position in found if position >= 0]
    return min(found) if found else 10_000 + fallback


def _mention_names(entity: dict[str, Any]) -> list[str]:
    from .names import english_aliases, english_name, load_synonyms

    canonical = str(entity.get("canonical_name") or entity.get("name") or "")
    names = [entity.get("mention"), canonical, english_name(canonical), *english_aliases(canonical)]
    names += [alias for alias, name in (load_synonyms().get("alias") or {}).items() if name == canonical]
    if len(canonical) >= 4 and re.fullmatch(r"[一-鿿]+", canonical) and canonical[2:] not in GENERIC_NAME_TAILS:
        names.append(canonical[2:])  # 贵州茅台 → 茅台, 中国平安 → 平安 (not 招商银行 → 银行)
    return sorted({str(name) for name in names if name}, key=len, reverse=True)


def _mention_count(text: str, entity: dict[str, Any]) -> int:
    """How often the question names an entity (any of its forms, longest first, without double counting)."""
    lowered = text.lower()
    spans: list[tuple[int, int]] = []
    for name in _mention_names(entity):
        for match in re.finditer(re.escape(name.lower()), lowered):
            if not any(start < match.end() and match.start() < end for start, end in spans):
                spans.append(match.span())
    return len(spans)


def _operand_form(named: int, reference: str, known: int) -> str:
    """(round 14) How the question supplied its operands: ``named``, ``named+carried``, ``ordinal``, ``anaphoric``
    or ``carried`` (recorded in the frame request for the trace)."""
    if named >= 2:
        return "named"
    if reference in {"ordinal", "anaphoric"} and named < 2:
        return reference
    return "named+carried" if named == 1 and known else "carried"


# (round 14, review-10 J4) a second operand the question names but the entity resolver did not: a capitalised name in
# English ("Zijin's", "than Zijin"), a name around 比 / 和 in Chinese
_EN_NAME = re.compile(r"(?<![\w/])(?-i:[A-Z])[A-Za-z&-]{2,}(?:\s+(?-i:[A-Z])[A-Za-z&-]{2,}){0,2}")
_EN_NOT_NAME = re.compile(
    r"^(?:by|how|what|which|whose|who|is|are|was|were|does|do|did|has|have|the|and|of|in|on|for|than|its|their|"
    r"compared|versus|between|among|i|my|so|tell|give|show|please|cny|rmb|roe|eps|p/?e|p/?b|net|gross|operating|"
    r"revenue|profit|margin|market|cap|close|price|dividend|yield|turnover|volume|sector|industry|average|both|it|"
    r"they|them|over|about|more|less|higher|lower)$",
    re.IGNORECASE,
)
_ZH_NAME = re.compile(
    r"比(?P<b>[一-鿿]{2,6}?)(?:的|高|低|多|少|贵|便宜|强|弱|大|小|还)|"
    r"(?P<a>[一-鿿]{2,6}?)(?:和|与|跟)(?P<c>[一-鿿]{2,6}?)(?:的|两|二|谁|哪|相比|比)"
)
_ZH_NOT_NAME = re.compile(
    r"前者|后者|行业|板块|同行|同业|平均|大盘|指数|去年|今年|上月|上周|昨天|之前|以前|它|其|两者|二者|两家|三家|"
    r"市场|整体|水平|当前|现在|最新|多少|几|哪个|谁"
)


def unresolved_operand(clause: str, entities: list[dict[str, Any]]) -> tuple[str, int] | None:
    """``(name as written, position)`` of an operand the comparison names that matches no resolved entity, or
    ``None``. Only one resolved entity is expected; metric and question words are not names."""
    forms = [name.lower() for entity in entities for name in _mention_names(entity)]

    def known(word: str) -> bool:
        lowered = word.lower()
        return any(form in lowered or lowered in form for form in forms if len(form) >= 2)

    if re.search(r"[一-鿿]", clause):
        for match in _ZH_NAME.finditer(clause):
            for group in ("b", "a", "c"):
                word = match.group(group)
                if word and not known(word) and not _ZH_NOT_NAME.search(word) and metric_of(word) is None:
                    return word, match.start(group)
        return None
    for match in _EN_NAME.finditer(clause):
        words = [word for word in match.group(0).split() if not _EN_NOT_NAME.match(word)]
        word = " ".join(words)
        if word and len(word) >= 3 and not known(word) and metric_of(word) is None:
            return word, match.start()
    return None


# "how many times Ping An's is Moutai's?": the second operand is the numerator
_TIMES_INVERTED = re.compile(r"\bhow many times\b(?P<rest>.*)", re.IGNORECASE)


def resolve_frame_question(
    query: str,
    turns: list[dict[str, Any]],
    listed: list[dict[str, Any]],
    sectors: list[str] | None = None,
) -> tuple[str, str, dict[str, Any]] | None:
    """A gap / ratio / relative / which question read against the session frame: ``(rewritten, reason, request)``.

    The metric is the one the question names, else the frame's; the operands are the targets the question names
    (two or more, in the order the question names them), else the frame's (one named target joins the frame's other
    operand). "前者/后者" follow the frame's order. (round 12, H2) A question naming two targets (or a target and its
    industry, or two industries) and a metric is computed the same way without any frame ("五粮液的PE比茅台低
    百分之多少", "Which of Moutai and Wuliangye has the higher P/B, and by how much?", "白酒板块的平均市盈率比保险
    板块高多少").
    ``None`` when there is no such question or nothing supplies a metric and two operands."""
    from .memory import _TRIPLE, strip_filler

    text = strip_filler(query).strip()
    operation = frame_operation(text)
    if operation is None:
        return None
    clause = comparison_clause(text) or text
    frame = latest_frame(turns) or {}
    known = [dict(item) for item in frame.get("operands") or []]
    if _TRIPLE.search(clause) and len(known) < 3 and len([e for e in listed if e.get("symbol")]) < 3:
        # "这三家谁最高" after two targets: the group reference and its count check ask which third one is meant
        return None
    named_metric = metric_of(clause)
    entities = [entity for entity in listed if entity.get("symbol")]
    ordered = sorted(enumerate(entities), key=lambda item: _mention_position(text, item[1], item[0]))
    named = [
        {"kind": "target", "name": str(e.get("canonical_name") or e.get("symbol")), "symbol": str(e["symbol"])}
        for _index, e in ordered
    ]
    industries = list(dict.fromkeys(sector for sector in sectors or [] if sector))
    if len(industries) >= 2 and not named:
        # (round 12, H9) two industry averages: each from its own industry snapshot
        operands = [{"kind": "industry", "industry": name, "member": name} for name in industries]
    elif len(named) >= 2:
        operands = named
    elif len(named) == 1:
        others = [item for item in known if operand_key(item) != operand_key(named[0])]
        if others:
            in_frame = named[0]["symbol"] in {i.get("symbol") for i in known}
            operands = [named[0], others[-1]] if in_frame else [others[-1], named[0]]
        else:
            operands = named  # a target against its industry (below), or nothing to compare
    else:
        operands = known
    if operation == "share_total" and len(named) == 1:
        # (round 14, review-10 J3) "比亚迪占两家合计的多少": the named target is the numerator
        operands = [named[0], *[item for item in operands if operand_key(item) != operand_key(named[0])]]
    if (
        _INDUSTRY_WORDS.search(clause)
        and len(named) < 2
        and len(industries) < 2
        and not any(i.get("kind") == "industry" for i in operands)
    ):
        # "比行业便宜百分之几": the latest target against its own industry average
        targets = [item for item in operands if item.get("kind") != "industry"]
        if len(targets) != 1:
            # "Which of the two is cheaper relative to its own industry?": each target against its own industry is
            # not one comparison of two operands; the comparison path answers it
            return None
        member = targets[0]
        operands = [member, {"kind": "industry", "member": member.get("symbol")}]
    if (
        len(named) == 1
        and len(operands) == 1
        and operands[0].get("kind") == "target"
        and (stray := unresolved_operand(clause, entities))
    ):
        # (round 14, review-10 J4) "By how many points does <unknown>'s ROE outstrip Midea's?": never answered with
        # the one operand that resolved; the answer says which operand could not be identified
        unknown = {"kind": "unresolved", "name": stray[0]}
        before = stray[1] < _mention_position(clause, entities[0], 0)
        operands = [unknown, operands[0]] if before else [operands[0], unknown]
    zh = bool(re.search(r"[一-鿿]", text))
    if not named_metric and len(named) < 2 and len(industries) < 2 and len(operands) < 2 and operation != "sum":
        # (round 13) "两个率相差多少个点" after "X的毛利率" → "那净利率呢": two metrics of one target
        pair = _metric_pair(turns)
        if pair is not None:
            return _metric_pair_question(text, operation, pair, zh)
    swap = operation == "ratio" and _SWAP_SHARES.search(clause) is not None
    metric = named_metric or ("close" if swap else frame.get("metric"))  # "一股茅台能换几股五粮液": closes
    if not named_metric and operation == "spread" and _EXPENSIVE.search(clause):
        metric = "pe"  # (round 14, review-10 J4) "the most and the least expensive": a spread of P/E
    if not metric or len(operands) < 2:
        return None
    if operation in {"sum", "share_total"} and METRIC_BY_KEY[metric].kind not in SUM_KINDS:
        return None  # "两只的市盈率加起来": a total of multiples is not a value
    if len({operand_key(item) for item in operands}) < 2:
        return None
    ordinals = [match for match in _ORDINAL.finditer(clause)]
    first_name, last_name = operand_name(operands[0], zh), operand_name(operands[-1], zh)
    if ordinals and len(named) < 2:
        first, last = operands[0], operands[-1]
        picked = [first if match.group("first") else last for match in ordinals]
        operands = picked + [item for item in (first, last) if item not in picked]
    elif _TRIPLE.search(clause):
        operands = operands[-3:]
    elif operation in {"which", "spread", "sum", "share_total"} or len(named) >= 3:
        pass  # (round 13, review-9 I3/I4) every operand: a total, a spread or a comparison of all the named targets
    elif len(operands) > 2 and _TWO.search(clause):
        operands = operands[-2:]  # "两者差多少" after three targets: the last two discussed
    if operation == "share_total" and len(named) >= 2 and not (ordinals and len(named) < 2):
        # (round 14, review-10 J3) the numerator of a share of a combined total is the target the question is about:
        # the one named again outside the total ("比亚迪的营收占比亚迪和宁德时代两家合计的多少", "BYD's revenue as a
        # percentage of BYD and CATL combined"), else the first one named
        counts = {str(entity.get("symbol")): _mention_count(text, entity) for entity in entities}
        top = max(operands, key=lambda item: (counts.get(str(item.get("symbol")), 0), -operands.index(item)))
        operands = [top, *[item for item in operands if item is not top]]
    if operation == "share" and len(named) >= 2 and (base := _SHARE.search(clause)) is not None:
        # (round 13, review-9 I1) the base of a share is the operand written after 相当于 / "of": "五粮液相当于茅台的
        # 百分之多少" is 五粮液 ÷ 茅台, "What fraction of A's does B earn?" is B ÷ A
        after = [
            item for item in ordered_entities(entities, operands) if _mention_position(text, item, 0) >= base.start()
        ]
        if after and str(after[0].get("symbol")) == str(operands[0].get("symbol")) and len(operands) == 2:
            operands = [operands[1], operands[0]]
    if operation == "ratio" and (inverted := _TIMES_INVERTED.search(clause)):
        # "how many times A is B" asks B / A; "how many times bigger is A than B" asks A / B (A and B named, or
        # the former / the latter)
        rest = inverted.group("rest")
        verb = re.search(r"\b(?:is|are|was)\b", rest)
        if len(named) >= 2:
            a, b = (_mention_position(rest, item, 0) for item in ordered_entities(entities, operands))
        else:
            spots = [match.start() for match in _ORDINAL.finditer(rest)]
            a, b = (spots[0], spots[1]) if len(spots) >= 2 else (10_000, 10_000)
        if verb and a < verb.start() < b < 10_000:
            operands = [operands[1], operands[0], *operands[2:]]
    names = [operand_name(item, zh) for item in operands]
    label = metric_label(metric, zh)
    if zh:
        joined = "和".join(names)
        if ordinals and len(named) < 2:
            # keep the user's words but say who 前者/后者 are
            named_text = _ORDINAL.sub(lambda m: first_name if m.group("first") else last_name, text)
            rewritten = f"{joined}的{label}，{named_text}"
        else:
            rewritten = f"{joined}的{label}，{text}"
    else:
        joined = " and ".join(names)
        rewritten = f"{label} of {joined}: {text}"
    clean = [
        {key: item[key] for key in ("kind", "name", "symbol", "industry", "member") if item.get(key) is not None}
        for item in operands
    ]
    request: dict[str, Any] = {"operation": operation, "metric": metric, "operands": clean}
    request["operand_form"] = _operand_form(len(named), parse_clause(clause).reference, len(known))
    if metric == "pct_change" and (direction := _direction(clause)):
        # (round 12) "多跌了多少" after two falls: the gap names the side that fell more, not the "higher" change
        request["direction"] = direction
    if operation in {"difference", "ratio", "relative", "share"} and len(operands) > 2:
        # (round 13, review-9 I4) a two-operand comparison of three or more targets: every pair, said as such
        request["pairwise"] = True
    if operation in {"ratio", "share"} and (claimed := claimed_check(clause)) is not None:
        # (round 13) "大约是平安的两倍吗", "more than double …?": the ratio, and whether the stated multiple holds
        request["claimed"] = claimed
    reason = f"frame:{operation}:{metric}:{'|'.join(operand_name(item, True) for item in operands)}"
    return rewritten, reason, request


def _metric_pair(turns: list[dict[str, Any]]) -> tuple[dict[str, Any], str, str] | None:
    """``(operand, earlier metric, latest metric)`` when the last two metric frames are about one and the same target
    with two different metrics of one kind ("五粮液的毛利率" → "那净利率呢": two rates), else ``None``."""
    frames = [turn.get("frame") or {} for turn in turns if "frame" in turn]
    if not frames:
        return None
    latest = frames[-1]
    operands = latest.get("operands") or []
    metric = latest.get("metric")
    if not metric or len(operands) != 1 or operands[0].get("kind") == "industry":
        return None
    key = operand_key(operands[0])
    for earlier in reversed(frames[:-1]):
        other = earlier.get("metric")
        if not other or other == metric:
            continue
        mine = [item for item in earlier.get("operands") or [] if operand_key(item) == key]
        if len(earlier.get("operands") or []) != 1 or not mine:
            return None
        if METRIC_BY_KEY[other].kind != METRIC_BY_KEY[metric].kind:
            return None  # "毛利率" and "市盈率": a percentage and a multiple have no gap
        return dict(operands[0]), str(other), str(metric)
    return None


def _metric_pair_question(
    text: str, operation: str, pair: tuple[dict[str, Any], str, str], zh: bool
) -> tuple[str, str, dict[str, Any]] | None:
    operand, first, second = pair
    if operation in {"sum", "which"} and METRIC_BY_KEY[first].kind == "points":
        return None
    operation = (
        "difference" if operation == "spread" else operation
    )  # a total of two rates means nothing; "which is higher" of two rates is stated by the gap
    name = operand_name(operand, zh)
    labels = [metric_label(first, zh), metric_label(second, zh)]
    rewritten = f"{name}的{labels[0]}和{labels[1]}，{text}" if zh else f"{name}'s {labels[0]} and {labels[1]}: {text}"
    base = {key: operand[key] for key in ("kind", "name", "symbol") if operand.get(key) is not None}
    request = {
        "operation": operation,
        "metric": first,
        "operands": [{**base, "metric": first}, {**base, "metric": second}],
    }
    reason = f"frame:{operation}:{first}-{second}:{operand_name(operand, True)}"
    return rewritten, reason, request


def ordered_entities(entities: list[dict[str, Any]], operands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The NLU entities of the first two operands, in operand order (for locating them in the question)."""
    by_symbol = {str(entity.get("symbol")): entity for entity in entities}
    return [by_symbol.get(str(item.get("symbol")), item) for item in operands[:2]]


_FALL_WORDS = re.compile(r"跌|下挫|回落|\b(?:fell|fall|falls|dropped|drop|lost|declined?)\b", re.IGNORECASE)
_RISE_WORDS = re.compile(r"涨|上扬|\b(?:rose|rise|rises|gained|gain|climbed?)\b", re.IGNORECASE)


def _direction(text: str) -> str | None:
    fall, rise = bool(_FALL_WORDS.search(text)), bool(_RISE_WORDS.search(text))
    return "fall" if fall and not rise else "rise" if rise and not fall else None


def frame_calls(request: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The tool calls that return every operand of a frame request (target or industry member)."""
    metric = METRIC_BY_KEY.get(str(request.get("metric") or ""))
    if metric is None:
        return []
    calls: list[tuple[str, dict[str, Any]]] = []
    for operand in request.get("operands") or []:
        target = operand.get("symbol") or operand.get("member")
        own = METRIC_BY_KEY.get(str(operand.get("metric") or "")) or metric  # (round 13) a metric per operand
        tool = "get_fundamentals" if operand.get("kind") == "industry" else own.tool
        if target and (tool, {"target": target}) not in calls:
            calls.append((tool, {"target": str(target)}))
    return calls


def memory_view(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    """The frame for the LLM's session memory: metric, the earlier values and where they came from."""
    if not frame or not frame.get("metric") or not frame.get("operands"):
        return None
    values = []
    for operand in frame["operands"]:
        row = {"target": operand_name(operand, True)}
        if operand.get("symbol"):
            row["symbol"] = operand["symbol"]
        if operand.get("value") is not None:
            row["value_in_earlier_turn"] = operand["value"]
            row["earlier_evidence_id"] = operand.get("evidence_id")
        values.append(row)
    return {
        "metric": metric_label(str(frame["metric"]), zh=False),
        "operands_in_order": values,
        "note": "Values from earlier turns. Their evidence ids cannot be cited in this turn: call the tool again for "
        "any operand a gap, ratio or which-is-higher question needs, then compute it.",
    }
