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
    FrameMetric(
        "market_cap",
        "总市值",
        "market cap",
        "money",
        "get_fundamentals",
        ("总市值", "流通市值", "市值", r"\bmarket (?:cap|capitali[sz]ation)\b"),
    ),
    FrameMetric(
        "revenue",
        "营业收入",
        "revenue",
        "money",
        "get_fundamentals",
        ("营业总收入", "营业收入", "营收", "收入", r"\brevenues?\b", r"\bsales\b", r"\btop line\b"),
    ),
    FrameMetric(
        "net_profit",
        "净利润",
        "net profit",
        "money",
        "get_fundamentals",
        ("归母净利润", "净利润", "净利", "利润", r"\bnet (?:profit|income)\b", r"\bprofits?\b", r"\bearnings\b"),
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
            r"\bclos(?:e|es|ing price)\b",
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
_RATIO_EXPLICIT = re.compile(
    r"倍数关系|比值|之比|几倍于|\bratios?\b|\bhow many times\b|"
    r"\btimes (?:as (?:high|large|big|much)|bigger|larger|higher|more|greater|the)\b|\bmultiple of\b|"
    r"(?<![\w.])\d+(?:\.\d+)?\s?[x×](?![A-Za-z])",
    re.IGNORECASE,
)
_RATIO_UNIT = re.compile(r"(?:几|多少)倍")
_ANCHOR = re.compile(
    r"(?:是|为|相当于|等于)[^，。？?,;；]{1,14}?的|比(?!例|较|重)|差|相对|较之|前者|后者|前一|后一|第一个|第二个|二者|两者|"
    r"两个|两家|两只|俩|它们|\bthe former\b|\bthe latter\b|\bthan\b|\bthem\b|\bthe two\b|\bboth\b|\bversus\b|\bvs\.?",
    re.IGNORECASE,
)
_RELATIVE_EXPLICIT = re.compile(
    r"折价|溢价|\b(?:premium|discount)\b|\bhow many percent\b|\bin percent(?:age)?(?: terms)?\b|"
    r"\bby what (?:percent(?:age)?|share)\b|"
    r"\bpercent(?:age)? (?:higher|lower|more|less|above|below|premium|discount|bigger|smaller|cheaper)\b",
    re.IGNORECASE,
)
_RELATIVE_UNIT = re.compile(r"百分之(?:多少|几)|(?:多少|几)(?:个)?(?:百分比|%)|几成|多少成|\s%", re.IGNORECASE)
# a comparative: 高 / 低 / 贵 / 便宜 / 大 / 小, "多" / "少" only before a verb or 了 / 出 (not the 多少 of a question)
_COMPARATIVE = re.compile(
    r"高|低|贵|便宜|(?<!多)大|(?<!多|大)小|超出|超过|领先|落后|多(?=[了出跌涨赚亏卖成交])|(?<!多)少(?=[了出跌涨赚亏卖成交])|"
    r"\b(?:higher|lower|more|less|above|below|bigger|smaller|larger|cheaper|greater)\b",
    re.IGNORECASE,
)
# "差几个点", "相差多少", "高了多少", "大多少", "difference", "gap", "how much higher", "by how much".
_DIFFERENCE = re.compile(
    r"差了?(?:有|是|大概|大约)?(?:多少|几|多大)|相差|差距|差额|差值|差多少|"
    r"(?:高|低|多|少|大|小|贵|便宜)(?:了|出)?(?:有|是|大概|大约)?(?:多少|几)|"
    # (round 12) "多跌了多少", "少涨了几个点", "多成交了多少钱": more / less of a verb, by how much
    r"(?:多|少)(?:跌|涨|赚|亏|卖|成交|交易)了?(?:有|是|大概|大约)?(?:多少|几)|"
    r"\bdifferences?\b|\bgap\b|\bspread\b|\bdiffer\b|"
    r"\bhow much (?:higher|lower|more|less|bigger|smaller|larger|cheaper)\b|"
    r"\bby how (?:much|many)\b|\bhow far (?:apart|above|below)\b|"
    r"\bhow many (?:more |fewer )?(?:percentage |basis )?points\b|\b(?:points?|percent) apart\b",
    re.IGNORECASE,
)
# "谁更高", "哪个更大", "which one is lower": which side is higher (the answer states both values).
_WHICH = re.compile(
    r"(?:谁|哪个|哪一个|哪只|哪家|哪边|哪一家|哪一只)[^，。？?,;；]{0,6}?(?:更|比较|相对)?"
    r"(?:高|低|大|小|多|少|贵|便宜|强|弱|活跃)(?!档|端|效|级|点位)|"
    r"\bwhich\b[^.?!]{0,30}\b(?:higher|lower|bigger|smaller|larger|more|less|cheaper|greater)\b|"
    r"\b(?:higher|lower|bigger|larger) of the two\b",
    re.IGNORECASE,
)
_ORDINAL = re.compile(
    r"(?P<first>前者|前一个|前一家|前一只|第一个|第一家|\bthe former\b|\bthe first (?:one|company|stock|fund)\b)|"
    r"(?P<last>后者|后一个|后一家|后一只|第二个|第二家|\bthe latter\b|\bthe second (?:one|company|stock|fund)\b)",
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


def _operation_of(text: str) -> str | None:
    if _RATIO_EXPLICIT.search(text) or (_RATIO_UNIT.search(text) and _ANCHOR.search(text)):
        return "ratio"
    if _RELATIVE_EXPLICIT.search(text) or (
        _RELATIVE_UNIT.search(text) and (_COMPARATIVE.search(text) or _ANCHOR.search(text))
    ):
        return "relative"
    if _DIFFERENCE.search(text):
        return "difference"
    if _WHICH.search(text):
        return "which"
    return None


def comparison_clause(text: str) -> str | None:
    """The clause of the question that asks a comparison (the whole question when it has one clause)."""
    text = (text or "").strip()
    for clause in (match.group(0).strip() for match in _CLAUSE.finditer(text)):
        if clause and len(clause) <= MAX_CLAUSE_CHARS and _operation_of(clause):
            return clause
    return None


def frame_operation(text: str) -> str | None:
    """``ratio``, ``relative``, ``difference`` or ``which`` for a question that compares two values, else ``None``.

    A net-margin question written as a share ("净利润是营收的百分之几") is one metric, not a comparison; advice,
    forecast and news questions are not comparisons either."""
    text = (text or "").strip()
    if not text or _NOT_A_GAP.search(text) or _NET_MARGIN_SHARE.search(text):
        return None
    clause = comparison_clause(text)
    return _operation_of(clause) if clause else None


def is_frame_question(text: str) -> bool:
    return frame_operation(text) is not None


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
    if request and request.get("metric") and request.get("operands"):
        metric, operands = str(request["metric"]), [dict(item) for item in request["operands"]]
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
    if len(canonical) >= 4 and re.fullmatch(r"[一-鿿]+", canonical):
        names.append(canonical[2:])  # 贵州茅台 → 茅台, 中国平安 → 平安
    found = [lowered.find(str(name).lower()) for name in names if name]
    found = [position for position in found if position >= 0]
    return min(found) if found else 10_000 + fallback


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
    if _TRIPLE.search(clause) and len(known) < 3:
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
    metric = named_metric or frame.get("metric")
    if not metric or len(operands) < 2:
        return None
    ordinals = [match for match in _ORDINAL.finditer(clause)]
    zh = bool(re.search(r"[一-鿿]", text))
    first_name, last_name = operand_name(operands[0], zh), operand_name(operands[-1], zh)
    if ordinals and not named:
        first, last = operands[0], operands[-1]
        picked = [first if match.group("first") else last for match in ordinals]
        operands = picked + [item for item in (first, last) if item not in picked]
    elif _TRIPLE.search(clause):
        operands = operands[-3:]
    elif operation != "which" and len(operands) > 2:
        operands = operands[-2:]
    if operation == "ratio" and len(named) >= 2 and (inverted := _TIMES_INVERTED.search(clause)):
        # "how many times A is B" asks B / A; "how many times bigger is A than B" asks A / B
        rest = inverted.group("rest")
        a, b = (_mention_position(rest, item, 0) for item in ordered_entities(entities, operands))
        verb = re.search(r"\b(?:is|are|was)\b", rest)
        if verb and a < verb.start() < b < 10_000:
            operands = [operands[1], operands[0], *operands[2:]]
    names = [operand_name(item, zh) for item in operands]
    label = metric_label(metric, zh)
    if zh:
        joined = "和".join(names)
        if ordinals and not named:
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
    request = {"operation": operation, "metric": metric, "operands": clean}
    if metric == "pct_change" and (direction := _direction(clause)):
        # (round 12) "多跌了多少" after two falls: the gap names the side that fell more, not the "higher" change
        request["direction"] = direction
    reason = f"frame:{operation}:{metric}:{'|'.join(operand_name(item, True) for item in operands)}"
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
        tool = "get_fundamentals" if operand.get("kind") == "industry" else metric.tool
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
