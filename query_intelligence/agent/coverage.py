"""What FinSight covers, and what a question asks for that the evidence does not have.

* ``out_of_coverage``: crypto assets and foreign (US / Hong Kong) equities are finance questions, but
  FinSight's data covers China A-shares, funds/ETFs, indices and China macro only. Such questions get a
  clear "not covered" answer instead of a "which stock?" clarification (which could never succeed).
* ``coverage_gaps``: the deterministic composer restates whatever the tools returned. When the question
  asks for a metric the tools did not return (dividend yield, debt ratio, growth rate) or for a period
  the data does not cover ("2019年营收" with 2025 statements), the gap is stated explicitly instead of
  silently answering with other numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- out of coverage

_CRYPTO = re.compile(
    r"比特币|以太坊|狗狗币|莱特币|瑞波币|泰达币|加密货币|加密资产|数字货币|虚拟货币|币圈|山寨币|稳定币|"
    r"(?<![A-Za-z])(?:BTC|ETH|USDT|DOGE|XRP|SOL)(?![A-Za-z])|\bbitcoin\b|\bethereum\b|\bcrypto(?:currenc(?:y|ies))?\b|"
    r"\bdogecoin\b|\bstablecoins?\b",
    re.IGNORECASE,
)
# US / Hong Kong listed names and markets. Concept-sector phrasing ("苹果概念股", "特斯拉产业链") is an
# A-share theme and stays in scope.
_FOREIGN_EQUITY = re.compile(
    r"(?:苹果公司|苹果股票|苹果股价|苹果的股[价票]|苹果(?=的?(?:市盈率|市净率|市值|财报|营收|净利润|股价|股票|能买|值得买|会涨|会跌))|"
    r"特斯拉|英伟达|微软|谷歌|亚马逊|脸书|奈飞|伯克希尔|台积电|"
    r"腾讯控股|阿里巴巴|美团|小米集团|港股|美股|中概股|纳斯达克|道琼斯|标普500|恒生指数|恒指)"
    r"(?!概念|产业链|供应链|链)|"
    r"\b(?:apple|tesla|nvidia|microsoft|google|alphabet|amazon|netflix|berkshire|tencent|alibaba|meituan|tsmc)\b|"
    r"(?<![A-Za-z])(?:AAPL|TSLA|NVDA|MSFT|GOOGL?|AMZN|NFLX|META|BABA)(?![A-Za-z])|"
    r"\b(?:us|u\.s\.|american|hong kong) (?:stocks?|shares|equities|market)\b|\bnasdaq\b|\bs&p 500\b|\bdow jones\b|"
    r"\bhang seng\b|\bnyse\b",
    re.IGNORECASE,
)


# A question about how a foreign market affects A-shares ("美股大跌对A股有什么影响") is in scope.
_A_SHARE_ANCHOR = re.compile(
    r"A股|沪深|上证|深证|创业板|科创板|北交所|\bA-?shares?\b|\bChin(?:a|ese) (?:stocks?|equities|market)\b",
    re.IGNORECASE,
)


def out_of_coverage(query: str) -> str | None:
    """``crypto`` or ``foreign_equity`` when the question is about an asset FinSight has no data for."""
    text = query or ""
    if _A_SHARE_ANCHOR.search(text):
        return None
    if _CRYPTO.search(text):
        return "crypto"
    if _FOREIGN_EQUITY.search(text):
        return "foreign_equity"
    return None


def out_of_coverage_text(category: str, *, zh: bool) -> str:
    if category == "crypto":
        return (
            "FinSight 只覆盖 A 股、公募基金/ETF、指数和中国宏观数据，没有比特币等加密资产的行情或数据，"
            "因此无法回答这个问题。可以换成 A 股或基金相关的问题，例如「贵州茅台最新收盘价是多少？」。"
            if zh
            else "FinSight only covers China A-shares, funds/ETFs, indices and China macro data. It has no data on "
            "crypto assets such as Bitcoin, so it cannot answer this. Try an A-share or fund question instead, "
            'for example "What was Kweichow Moutai\'s latest close?"'
        )
    return (
        "FinSight 只覆盖 A 股、公募基金/ETF、指数和中国宏观数据，没有美股、港股个股或海外市场的数据，"
        "因此无法回答这个问题。可以换成 A 股或基金相关的问题，例如「贵州茅台最新收盘价是多少？」。"
        if zh
        else "FinSight only covers China A-shares, funds/ETFs, indices and China macro data. It has no data on US "
        "or Hong Kong listed stocks or overseas markets, so it cannot answer this. Try an A-share or fund question "
        'instead, for example "What was Kweichow Moutai\'s latest close?"'
    )


# --------------------------------------------------------------------------- requested metrics


@dataclass(frozen=True)
class Metric:
    key: str
    zh: str
    en: str
    pattern: re.Pattern[str]
    fields: tuple[str, ...]  # keys in get_fundamentals ``metrics`` that answer it
    derivable_from: tuple[str, ...] = ()  # present together, the metric can be derived (net margin)


def _metric(key: str, zh: str, en: str, pattern: str, fields: tuple[str, ...], derivable: tuple[str, ...] = ()):
    return Metric(key, zh, en, re.compile(pattern, re.IGNORECASE), fields, derivable)


# Order matters: growth rates are matched before the level they grow ("营收增速" is not "营收").
METRICS: tuple[Metric, ...] = (
    _metric(
        "revenue_growth",
        "营收增速",
        "revenue growth",
        r"(?:营收|营业收入|收入)(?:的)?(?:增速|增长率|同比增长|同比|增幅|增长)|revenue growth|sales growth",
        ("revenue_yoy", "revenue_growth", "or_yoy", "tr_yoy"),
    ),
    _metric(
        "profit_growth",
        "净利润增速",
        "net profit growth",
        r"(?:净利润|净利|利润)(?:的)?(?:增速|增长率|同比增长|同比|增幅|增长)|(?:profit|earnings|income) growth",
        ("netprofit_yoy", "net_profit_yoy", "profit_growth", "dt_netprofit_yoy"),
    ),
    _metric(
        "growth",
        "增速",
        "growth rate",
        r"增速|增长率|同比增长|同比增幅|\bgrowth(?: rate)?\b|\byoy\b|year[- ]over[- ]year",
        (
            "revenue_yoy",
            "revenue_growth",
            "or_yoy",
            "tr_yoy",
            "netprofit_yoy",
            "net_profit_yoy",
            "profit_growth",
            "dt_netprofit_yoy",
        ),
    ),
    _metric(
        "market_cap",
        "总市值",
        "market cap",
        r"总市值|流通市值|市值|market cap(?:itali[sz]ation)?",
        ("total_mv", "market_cap", "circ_mv", "total_market_cap"),
    ),
    _metric(
        "dividend_yield",
        "股息率",
        "dividend yield",
        r"股息率|股息|dividend yield|\bdividends?\b",
        ("dividend_yield", "dv_ratio", "dv_ttm"),
    ),
    _metric(
        "debt_ratio",
        "资产负债率",
        "debt ratio",
        r"资产负债率|负债率|负债水平|杠杆率|debt[- ]to[- ](?:asset|equity)|debt ratio|leverage|gearing",
        ("debt_to_assets", "debt_ratio", "liability_ratio", "debt_to_equity"),
    ),
    _metric(
        "gross_margin",
        "毛利率",
        "gross margin",
        r"毛利率|gross margin",
        ("gross_margin", "grossprofit_margin"),
    ),
    _metric(
        "net_margin",
        "净利率",
        "net margin",
        r"净利率|净利润率|net (?:profit )?margin",
        ("net_margin", "netprofit_margin"),
        ("revenue", "net_profit"),
    ),
    _metric(
        "cash_flow",
        "现金流",
        "cash flow",
        r"现金流|cash ?flow",
        ("operating_cash_flow", "n_cashflow_act", "free_cash_flow"),
    ),
)
_METRIC_BY_KEY = {metric.key: metric for metric in METRICS}
# Labels of extra metrics rendered when present (the template renders PE/PB/ROE/revenue/net profit itself).
EXTRA_METRIC_FIELDS = {field: metric for metric in METRICS for field in metric.fields}


def requested_metrics(query: str) -> list[Metric]:
    """Metrics beyond the standard snapshot (PE/PB/ROE/revenue/net profit) that the question asks for."""
    text = query or ""
    found: list[Metric] = []
    consumed: list[tuple[int, int]] = []
    for metric in METRICS:
        for match in metric.pattern.finditer(text):
            if any(start <= match.start() < end for start, end in consumed):
                continue
            consumed.append(match.span())
            if metric not in found:
                found.append(metric)
    return found


_YEAR = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*(?:年|财年|年度)|"
    r"\b(?:in|for|during|of|fy|fiscal|financial year)\s*((?:19|20)\d{2})\b|"
    r"\b((?:19|20)\d{2})\s+(?:revenue|sales|net|profit|earnings|results|annual|full[- ]year|report|financials?|"
    r"dividends?|roe|eps|p/?e|p/?b)",
    re.IGNORECASE,
)
_DATE_LIKE = re.compile(r"(?:19|20)\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?")
_FORWARD = re.compile(r"预计|预测|将会|会不会|目标|展望|明年|forecast|expect|will\b|outlook|target", re.IGNORECASE)


def requested_years(query: str) -> list[int]:
    """Calendar years a question explicitly asks about ("2019年营收", "2023 revenue")."""
    text = _DATE_LIKE.sub(" ", query or "")
    years = []
    for match in _YEAR.finditer(text):
        year = int(next(group for group in match.groups() if group))
        if 1990 <= year <= 2100 and year not in years:
            years.append(year)
    return years


def _year_of(value: Any) -> int | None:
    match = re.match(r"\s*((?:19|20)\d{2})", str(value or ""))
    return int(match.group(1)) if match else None


def coverage_gaps(
    query: str, tool_log: list[dict[str, Any]], *, zh: bool, names: dict[str, str] | None = None
) -> list[str]:
    """Sentences stating which requested metrics or periods the retrieved data does not contain.

    Only checked against tools that returned data: a failed tool is already reported as missing.
    """
    names = names or {}
    sentences: list[str] = []
    wanted = requested_metrics(query)
    if _MACRO_GROWTH.search(query or ""):
        # "M2增速" / "M2 growth": the macro indicator is itself the growth rate, not a company metric.
        wanted = [metric for metric in wanted if metric.key != "growth"]
    years = requested_years(query)
    forward = bool(_FORWARD.search(query or ""))
    fundamentals = [entry for entry in tool_log if entry.get("tool") == "get_fundamentals" and entry.get("ok")]
    asked_fundamentals = any(entry.get("tool") == "get_fundamentals" for entry in tool_log)
    for entry in fundamentals:
        data = entry.get("data") or {}
        metrics = data.get("metrics") or {}
        name = str(data.get("name") or names.get(str(data.get("symbol"))) or data.get("symbol") or "")
        missing = [
            metric
            for metric in wanted
            if not any(metrics.get(field) is not None for field in metric.fields)
            and not (metric.derivable_from and all(metrics.get(field) is not None for field in metric.derivable_from))
        ]
        if missing:
            labels = ("、".join(metric.zh for metric in missing)) if zh else ", ".join(metric.en for metric in missing)
            sentences.append(
                f"当前数据源没有{name}的{labels}数据，无法回答这一项；以下只列出可得的指标。"
                if zh
                else f"The current data sources do not include {name}'s {labels}, so that part cannot be answered; "
                "only the available metrics are listed below."
            )
        period = str(data.get("report_date") or "")
        period_year = _year_of(period)
        quarter = requested_quarter(query)
        if quarter and _month_of(period) and _month_of(period) != quarter[1] and not forward:
            # "今年一季度的净利润" with annual statements only: the quarter is not in the data.
            sentences.append(
                f"当前数据中没有所问的{quarter[0]}数据：{name}可得的财务数据报告期为 {period}，"
                "以下数字均属于该报告期，不是所问季度的。"
                if zh
                else f"Data for the requested period ({quarter[0]}) is not available: the latest financial statements "
                f"for {name} are for the period {period}, and the figures below are for that period."
            )
        if years and period_year and period_year not in years and not forward:
            asked = _years_text(years, zh=zh)
            sentences.append(
                f"当前数据中没有所问的{asked}数据：{name}可得的财务数据报告期为 {period}，"
                f"以下数字均属于该报告期，不是{asked}的。"
                if zh
                else f"Data for {asked} is not available: the latest financial statements for {name} are for the "
                f"period {period}, and the figures below are for that period, not {asked}."
            )
    if wanted and not asked_fundamentals:
        labels = ("、".join(metric.zh for metric in wanted)) if zh else ", ".join(metric.en for metric in wanted)
        if any(entry.get("ok") for entry in tool_log):
            sentences.append(
                f"本次检索的证据中没有所问的{labels}数据。"
                if zh
                else f"The retrieved evidence does not include the requested {labels}."
            )
    if years and not fundamentals and not forward:
        for entry in tool_log:
            if entry.get("tool") != "get_price_history" or not entry.get("ok"):
                continue
            data = entry.get("data") or {}
            as_of_year = _year_of(data.get("as_of"))
            if as_of_year and as_of_year not in years and min(years) < as_of_year:
                name = str(data.get("name") or data.get("symbol") or "")
                asked = _years_text(years, zh=zh)
                sentences.append(
                    f"当前数据中没有{name}{asked}的行情，只有截至 {data.get('as_of')} 的最新数据。"
                    if zh
                    else f"There is no {asked} market data for {name}; only the latest data (as of "
                    f"{data.get('as_of')}) is available."
                )
    return list(dict.fromkeys(sentences))


_MACRO_GROWTH = re.compile(r"(?<![A-Za-z])m[12](?![A-Za-z0-9])|gdp|cpi|ppi|社融|货币供应|money supply", re.I)
# Macro indicators a question can name, and the indicator codes that answer them.
_MACRO_REQUESTS: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    ("CPI", re.compile(r"cpi|居民消费价格", re.I), ("CPI",)),
    ("PPI", re.compile(r"ppi|工业生产者出厂价格", re.I), ("PPI",)),
    ("PMI", re.compile(r"pmi|采购经理", re.I), ("PMI",)),
    ("M2", re.compile(r"(?<![A-Za-z])m2(?![A-Za-z0-9])|money supply|货币供应", re.I), ("M2",)),
    ("LPR", re.compile(r"lpr|贷款市场报价利率|loan prime rate", re.I), ("LPR",)),
    ("GDP", re.compile(r"gdp|国内生产总值", re.I), ("GDP",)),
    ("社融", re.compile(r"社融|社会融资|total social financing", re.I), ("TSF", "SOCIAL_FINANCING")),
    ("10Y", re.compile(r"国债|\bcgb\b|government bond|treasury", re.I), ("10Y",)),
)


def macro_gaps(query: str, tool_log: list[dict[str, Any]], *, zh: bool) -> list[str]:
    """Macro indicators named in the question that the macro tool did not return ("1-year LPR" with no LPR data)."""
    entries = [entry for entry in tool_log if entry.get("tool") == "get_macro_indicators" and entry.get("ok")]
    if not entries:
        return []
    codes = [
        str(indicator.get("code") or "").upper()
        for entry in entries
        for indicator in (entry.get("data") or {}).get("indicators") or []
        if indicator.get("value") is not None
    ]
    missing = [
        label
        for label, pattern, keys in _MACRO_REQUESTS
        if pattern.search(query or "") and not any(key in code for key in keys for code in codes)
    ]
    if not missing:
        return []
    labels = "、".join(missing) if zh else ", ".join(missing)
    return [
        f"当前数据源没有{labels}的数据，无法回答这一项；以下只列出可得的宏观指标。"
        if zh
        else f"The current data sources do not include {labels}, so that part cannot be answered; only the "
        "available macro indicators are listed below."
    ]


# Foreign central banks and US macro data ("美联储加息对A股有什么影响", "Will a Fed hike hurt A-shares?"): the
# question is in scope (its target is the A-share market), but no source carries the foreign series.
_FOREIGN_MACRO = re.compile(
    r"美联储|联储|FOMC|美国(?:的)?(?:加息|降息|利率|通胀|CPI|国债|经济|就业|非农)|美债|欧洲央行|欧央行|日本央行|日央行|"
    r"\bfed\b|federal reserve|\bu\.?s\.? (?:rates?|inflation|cpi|treasur(?:y|ies)|economy|jobs)|\becb\b|\bboj\b|"
    r"bank of japan|european central bank",
    re.IGNORECASE,
)


def foreign_macro_gaps(query: str, *, zh: bool) -> list[str]:
    """State that foreign central-bank and US macro data are not covered, when the question is about them."""
    if not _FOREIGN_MACRO.search(query or ""):
        return []
    return [
        "当前数据源只有中国的宏观指标（如CPI、PMI、货币供应量和国债收益率），没有美联储等境外央行的政策利率或美国经济数据，"
        "因此无法用数据说明其对A股的具体影响；以下只列出可得的国内证据。"
        if zh
        else "The configured sources carry China macro indicators only (CPI, PMI, money supply, government bond "
        "yields); they have no Federal Reserve or other foreign central-bank rates and no US economic data, so the "
        "effect on A-shares cannot be shown with data. Only the available domestic evidence is listed."
    ]


# Positions and fund flows of an investor group ("国家队最近是不是在加仓中国平安", "Is northbound money flowing into
# Moutai?"): no configured source carries holdings, flows or margin balances. A company's own shareholders are not
# included: their increases and reductions are disclosed in announcements.
_FLOW_SUBJECT = re.compile(
    r"国家队|中央汇金|汇金|证金|社保基金|社保|养老金|险资|公募基金|私募基金|机构投资者|主力资金|主力|游资|北向资金|南向资金|"
    r"北上资金|外资|陆股通|沪股通|深股通|聪明钱|\bnational team\b|\bstate(?:-backed)? funds?\b|"
    r"\b(?:northbound|southbound) (?:money|funds?|capital|investors|flows?)\b|"
    r"\bforeign (?:investors|funds|money|capital)\b|\binstitutional investors?\b|\bsmart money\b",
    re.IGNORECASE,
)
_FLOW_ACTION = re.compile(
    r"加仓|减仓|增持|减持|建仓|清仓|持仓|持股|抄底|出货|买入|卖出|买进|抛售|扫货|买了|卖了|在买|在卖|流入|流出|净买|净卖|"
    r"进场|离场|"
    r"\b(?:buy(?:ing)?|sell(?:ing)?|bought|sold|adding|trimming|accumulating|dumping|inflows?|outflows?|flow(?:ing|ed)?|"
    r"holdings?|positions?|stakes?|net (?:buying|selling|purchases?))\b",
    re.IGNORECASE,
)
_FLOW_DIRECT = re.compile(
    r"资金流向|资金流入|资金流出|两融余额|融资余额|融券余额|持仓数据|\b(?:fund|money|capital) flows?\b|"
    r"\bmargin (?:balance|debt|financing balance)\b|\binstitutional holdings\b",
    re.IGNORECASE,
)


def flow_gaps(query: str, tool_log: list[dict[str, Any]] | None = None, *, zh: bool) -> list[str]:
    """State that holdings and fund-flow data are not covered, when the question asks about them."""
    subject = _FLOW_SUBJECT.search(query or "")
    direct = _FLOW_DIRECT.search(query or "")
    if not direct and not (subject and _FLOW_ACTION.search(query or "")):
        return []
    concept = any(entry.get("tool") == "explain_concept" and entry.get("ok") for entry in tool_log or [])
    if concept and not (subject and _FLOW_ACTION.search(query or "")):
        return []  # a glossary answer already says the concept has no data series
    if subject and _FLOW_ACTION.search(query or ""):
        who = subject.group(0)
        if zh:
            return [f"当前数据源没有{who}的持仓或资金流向数据，无法判断其是否在买入或卖出；以下只列出可得的证据。"]
        return [
            f"The configured sources have no holdings or fund-flow data for {who}, so whether they are buying or "
            "selling cannot be shown; only the available evidence is listed."
        ]
    what = direct.group(0)  # type: ignore[union-attr]
    if zh:
        return [f"当前数据源没有{what}数据，无法回答这一项；以下只列出可得的证据。"]
    return [
        f"The configured sources do not include {what} data, so that part cannot be answered; only the available "
        "evidence is listed."
    ]


def _years_text(years: list[int], *, zh: bool) -> str:
    # "2019年" / "FY2019": forms the verifier reads as dates, not as claimed values.
    return "、".join(f"{year}年" for year in years) if zh else ", ".join(f"FY{year}" for year in years)


# Prefix of the get_fundamentals error for a sector without an industry snapshot (tools/fundamentals.py).
INDUSTRY_NOT_FOUND = "no industry snapshot for"


def failed_target_statements(
    query: str, tool_log: list[dict[str, Any]], *, zh: bool, names: dict[str, str] | None = None
) -> list[str]:
    """Name the targets whose data could not be retrieved (instead of a generic "no evidence" line)."""
    names = names or {}
    per_target: dict[str, list[str]] = {}
    sectors: list[str] = []
    for entry in tool_log:
        if entry.get("ok"):
            continue
        arguments = entry.get("arguments") or {}
        target = arguments.get("target") or ((arguments.get("targets") or [None])[0])
        if not target:
            continue
        if str((entry.get("error") or {}).get("message") or "").startswith(INDUSTRY_NOT_FOUND):
            # "半导体板块估值高吗": the sector is known, the sources have no snapshot for it
            if str(target) not in sectors:
                sectors.append(str(target))
            continue
        kind = {
            "get_fundamentals": ("基本面", "fundamentals"),
            "get_price_history": ("行情", "market data"),
            "compute_indicators": ("技术指标", "technical indicators"),
        }.get(str(entry.get("tool")))
        if kind is None:
            continue
        per_target.setdefault(str(target), [])
        label = kind[0] if zh else kind[1]
        if label not in per_target[str(target)]:
            per_target[str(target)].append(label)
    statements = [
        f"当前数据源没有{sector}行业的估值和行情快照，因此无法判断该行业的估值高低。"
        if zh
        else f"The configured sources have no valuation or market snapshot for the {sector} sector, so its "
        "valuation cannot be assessed."
        for sector in sectors
    ]
    for target, kinds in per_target.items():
        name = names.get(target)
        who = f"{name}（{target}）" if zh and name else (f"{name} ({target})" if name else target)
        statements.append(
            f"当前数据源中没有{who}的{'、'.join(kinds)}数据。"
            if zh
            else f"The current data sources have no {', '.join(kinds)} for {who}."
        )
    return statements


def metric_label(key: str, *, zh: bool) -> str:
    metric = _METRIC_BY_KEY.get(key)
    if metric is None:
        return key
    return metric.zh if zh else metric.en


# --------------------------------------------------------------------------- requested price details

_COUNT_WORDS = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20,
}  # fmt: skip
_COUNT = r"(\d{1,2}|[一两二三四五六七八九十]|one|two|three|four|five|six|seven|eight|nine|ten|twenty)"
_RECENT_CLOSES = re.compile(
    rf"(?:最近|近|过去|前)\s*{_COUNT}\s*(?:个)?(?:交易日|天|日)[^？?。]{{0,8}}?收盘|"
    rf"收盘价?[^？?。]{{0,6}}?(?:最近|近|过去)\s*{_COUNT}\s*(?:个)?(?:交易日|天|日)|"
    rf"(?:last|past|previous|recent)\s+{_COUNT}\s+(?:closes|closing prices|sessions|trading days|days)|"
    rf"{_COUNT}\s+(?:most recent|latest|recent)\s+closes",
    re.IGNORECASE,
)
_CLOSES_WORD = re.compile(r"收盘|\bclos(?:es|ing)\b", re.IGNORECASE)
_PREVIOUS_CLOSE = re.compile(
    r"前一(?:天|日|个交易日)(?:的)?收盘|前收|昨收|上一(?:个)?交易日(?:的)?收盘|\b(?:previous|prior|prev\.?) close\b|"
    r"\bclose (?:before|the day before)\b",
    re.IGNORECASE,
)
# The day's high/low, not "谁的ROE最高" or "is the P/E high or low".
_HIGH = re.compile(
    r"最高价|最高点|日内高点|最高(?=(?:和|与|、|及)最低)|(?:当天|当日|今天|日内)(?:的)?最高|"
    r"\b(?:daily|day'?s|intraday|session|today'?s)\s+highs?\b|\bhighs? and (?:the )?lows?\b|\bhigh/low\b|"
    r"\bhigh price\b",
    re.IGNORECASE,
)
_LOW = re.compile(
    r"最低价|最低点|日内低点|(?<=最高和)最低|(?<=最高与)最低|(?<=最高、)最低|(?:当天|当日|今天|日内)(?:的)?最低|"
    r"\b(?:daily|day'?s|intraday|session|today'?s)\s+lows?\b|\bhighs? and (?:the )?lows?\b|\bhigh/low\b|\blow price\b",
    re.IGNORECASE,
)
_OPEN = re.compile(r"开盘价?|\bopen(?:ing)?(?: price)?\b(?! interest)", re.IGNORECASE)
_VOLUME = re.compile(r"成交量|量能|\b(?:trading )?volume\b", re.IGNORECASE)
_AMOUNT = re.compile(r"成交额|成交金额|\bturnover\b|\bvalue traded\b", re.IGNORECASE)
_RETURN_DAYS = re.compile(
    rf"(?:近|过去|最近)?\s*{_COUNT}\s*(?:个)?(?:交易日|日|天)(?:的)?(?:收益率?|回报|涨幅|跌幅|涨跌幅?|表现)|"
    rf"\b{_COUNT}[- ](?:day|session)s?\s+(?:return|change|performance|gain|move)\b",
    re.IGNORECASE,
)
_MOVING_AVERAGE = re.compile(
    r"(?<![A-Za-z])MA\s*(\d{1,3})(?!\d)|(\d{1,3})\s*(?:日|天)均线|\b(\d{1,3})[- ]day moving average\b", re.IGNORECASE
)
_ABOVE_MA = re.compile(
    r"站上|站稳|跌破|均线(?:上方|之上|下方|之下)|"
    r"\b(?:above|below) (?:the |its )?(?:\d+[- ]day )?(?:ma\d*|moving average)",
    re.I,
)


def _count(token: str) -> int:
    token = token.lower()
    return int(token) if token.isdigit() else _COUNT_WORDS.get(token, 0)


@dataclass(frozen=True)
class PriceRequest:
    """Price details a question asks for beyond the latest close and daily change."""

    closes: int = 0  # number of recent closes to list
    previous_close: bool = False
    high: bool = False
    low: bool = False
    open: bool = False
    volume: bool = False
    amount: bool = False
    return_days: tuple[int, ...] = ()
    moving_averages: tuple[int, ...] = ()
    above_ma: bool = False

    @property
    def needs_quote(self) -> bool:
        return bool(
            self.closes or self.previous_close or self.high or self.low or self.open or self.volume or self.amount
        )

    @property
    def needs_indicators(self) -> bool:
        return bool(self.return_days or self.moving_averages or self.above_ma)


def requested_price_fields(query: str) -> PriceRequest:
    """Parse the price details asked for.

    Examples: "最近五个交易日的收盘价", "daily high and low", "3-day return", "MA5站上了吗".
    """
    text = query or ""
    closes = 0
    match = _RECENT_CLOSES.search(text)
    if match:
        closes = _count(next(group for group in match.groups() if group))
    elif re.search(r"\b(?:last|latest|recent) (?:two|2) closes\b|最近两(?:个|次)收盘", text, re.IGNORECASE):
        closes = 2
    returns = sorted({_count(next(g for g in m.groups() if g)) for m in _RETURN_DAYS.finditer(text)} - {0})
    averages = sorted({int(next(g for g in m.groups() if g)) for m in _MOVING_AVERAGE.finditer(text)})
    above = bool(_ABOVE_MA.search(text))
    if above and not averages:
        averages = [5]
    return PriceRequest(
        closes=closes if _CLOSES_WORD.search(text) or match else 0,
        previous_close=bool(_PREVIOUS_CLOSE.search(text)),
        high=bool(_HIGH.search(text)),
        low=bool(_LOW.search(text)),
        open=bool(_OPEN.search(text)),
        volume=bool(_VOLUME.search(text)),
        amount=bool(_AMOUNT.search(text)),
        return_days=tuple(returns),
        moving_averages=tuple(averages),
        above_ma=above,
    )


# --------------------------------------------------------------------------- reporting periods

_QUARTER = re.compile(
    r"(?P<q>[一二三四1-4])季度|第(?P<q2>[一二三四1-4])季度|\bQ(?P<q3>[1-4])\b|"
    r"\b(?P<q4>first|second|third|fourth) quarter\b|(?P<h>上半年|半年报|中报|\bH1\b|\bfirst half\b|\binterim\b)",
    re.IGNORECASE,
)
_QUARTER_MONTH = {"1": 3, "一": 3, "first": 3, "2": 6, "二": 6, "second": 6, "3": 9, "三": 9, "third": 9}
_QUARTER_MONTH.update({"4": 12, "四": 12, "fourth": 12})


def requested_quarter(query: str) -> tuple[str, int] | None:
    """``(label, period-end month)`` for a quarter or half-year request ("今年一季度", "Q3", "上半年")."""
    match = _QUARTER.search(query or "")
    if not match:
        return None
    if match.group("h"):
        return match.group("h"), 6
    token = next(
        group for group in (match.group("q"), match.group("q2"), match.group("q3"), match.group("q4")) if group
    )
    return match.group(0), _QUARTER_MONTH[token.lower()]


def _month_of(value: Any) -> int | None:
    found = re.match(r"\s*(?:19|20)\d{2}-(\d{1,2})", str(value or ""))
    return int(found.group(1)) if found else None


# --------------------------------------------------------------------------- industry and product scope

_INDUSTRY_SCOPE = re.compile(r"行业|板块|同行|同业|\bsector\b|\bindustry\b|\bpeers?\b", re.IGNORECASE)
# Standard fundamentals a question can ask of a company or of its industry snapshot (industry keys in brackets).
_STANDARD: tuple[tuple[str, str, str, re.Pattern[str], tuple[str, ...]], ...] = (
    ("pe", "市盈率", "P/E", re.compile(r"市盈率|(?<![A-Za-z])P/?E(?![A-Za-z])|price[- ]to[- ]earnings", re.I), ("pe",)),
    ("pb", "市净率", "P/B", re.compile(r"市净率|(?<![A-Za-z])P/?B(?![A-Za-z])|price[- ]to[- ]book", re.I), ("pb",)),
    ("roe", "ROE", "ROE", re.compile(r"净资产收益率|(?<![A-Za-z])ROE(?![A-Za-z])|return on equity", re.I), ("roe",)),
    ("revenue", "营业收入", "revenue", re.compile(r"营收|营业收入|\brevenue\b", re.I), ("revenue",)),
    (
        "net_profit",
        "净利润",
        "net profit",
        re.compile(r"净利润|净利(?!率)|\bnet (?:profit|income)\b", re.I),
        ("net_profit",),
    ),
    ("gross_margin", "毛利率", "gross margin", re.compile(r"毛利率|gross margin", re.I), ("gross_margin",)),
)
_FUNDAMENTAL_ASK = re.compile(
    r"市盈率|市净率|净资产收益率|ROE|营收|营业收入|净利润|毛利率|股息率|基本面|财报|(?<![A-Za-z])P/?[EB](?![A-Za-z])|"
    r"\brevenue\b|\bnet (?:profit|income)\b|\bgross margin\b|\bearnings\b|\bfundamentals?\b",
    re.IGNORECASE,
)


def asks_about_industry(query: str) -> bool:
    return bool(_INDUSTRY_SCOPE.search(query or ""))


def industry_gaps(query: str, tool_log: list[dict[str, Any]], *, zh: bool) -> list[str]:
    """Industry metrics the question asks for that the industry snapshot does not contain.

    "ROE跟保险行业平均比呢？": the 保险 snapshot has PE/PB/change but no ROE, so the industry side is stated
    as missing instead of being silently skipped.
    """
    if not asks_about_industry(query):
        return []
    wanted = [item for item in _STANDARD if item[3].search(query or "")]
    sentences: list[str] = []
    seen: set[str] = set()
    for entry in tool_log:
        if entry.get("tool") != "get_fundamentals" or not entry.get("ok"):
            continue
        industry = (entry.get("data") or {}).get("industry") or {}
        name = str(industry.get("industry_name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        metrics = industry.get("metrics") or {}
        missing = [item for item in wanted if not any(metrics.get(key) is not None for key in item[4])]
        if missing:
            labels = "、".join(item[1] for item in missing) if zh else ", ".join(item[2] for item in missing)
            sentences.append(
                f"当前数据源的{name}行业快照没有{labels}，无法给出行业层面的这一项。"
                if zh
                else f"The {name} industry snapshot in the current data has no {labels}, so the industry side of "
                "that cannot be given."
            )
    return sentences


def non_stock_fundamental_gaps(
    query: str,
    tool_log: list[dict[str, Any]],
    *,
    zh: bool,
    names: dict[str, str] | None = None,
    types: dict[str, str] | None = None,
) -> list[str]:
    """P/E, ROE or revenue asked of an ETF, fund or index: company fundamentals do not exist for it."""
    if not types or not _FUNDAMENTAL_ASK.search(query or ""):
        return []
    fetched = {
        str((entry.get("arguments") or {}).get("target"))
        for entry in tool_log
        if entry.get("tool") == "get_fundamentals"
    }
    sentences = []
    for symbol, kind in types.items():
        if kind in {"etf", "fund", "index"} and symbol not in fetched:
            name = (names or {}).get(symbol) or symbol
            label = {"etf": "ETF", "fund": "基金" if zh else "fund", "index": "指数" if zh else "index"}[kind]
            sentences.append(
                f"市盈率、ROE、营收等基本面指标当前只覆盖个股；{name}（{symbol}）是{label}，当前数据中没有它的这类数据。"
                if zh
                else f"P/E, ROE, revenue and similar fundamentals are only covered for individual stocks; {name} "
                f"({symbol}) is an {label}, so the current data has none for it."
            )
    return sentences


_NAMED_INDICATORS = (
    ("RSI(14)", "rsi_14", re.compile(r"(?<![A-Za-z])RSI", re.I)),
    ("MACD", "macd", re.compile(r"MACD", re.I)),
    ("20D vol", "volatility_20d", re.compile(r"波动率|volatility", re.I)),
    ("Bollinger", "bollinger", re.compile(r"布林|bollinger", re.I)),
)


def indicator_gaps(
    query: str, tool_log: list[dict[str, Any]], *, zh: bool, names: dict[str, str] | None = None
) -> list[str]:
    """Requested technical indicators or N-day returns that could not be computed or are null."""
    request = requested_price_fields(query)
    if not request.needs_indicators and not re.search(
        r"RSI|MACD|均线|moving average|volatility|波动率", query or "", re.I
    ):
        return []
    sentences = []
    for entry in tool_log:
        if entry.get("tool") != "compute_indicators":
            continue
        target = str((entry.get("arguments") or {}).get("target") or "")
        name = (names or {}).get(target) or target
        if not entry.get("ok"):
            sentences.append(
                f"当前数据不足以计算{name}的所问技术指标（历史收盘价不够）。"
                if zh
                else f"The current data is not enough to compute the requested indicators for {name} "
                "(too little price history)."
            )
            continue
        data = entry.get("data") or {}
        returns = data.get("pct_change_nd") or {}
        missing = [f"{days}日涨跌幅" if zh else f"{days}-day return" for days in request.return_days
                   if returns.get(f"pct_{days}d") is None]  # fmt: skip
        missing += [f"MA{days}" for days in request.moving_averages if data.get(f"ma{days}") is None]
        for label, key, pattern in _NAMED_INDICATORS:
            if pattern.search(query or "") and data.get(key) is None and label not in missing:
                missing.append(label)
        if missing:
            joined = "、".join(missing) if zh else ", ".join(missing)
            sentences.append(
                f"当前数据中没有{name}的{joined}（历史数据不足）。"
                if zh
                else f"The current data has no {joined} for {name} (not enough history)."
            )
    return sentences
