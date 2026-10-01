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

from .frame import NET_MARGIN_SHARE_SOURCE, TURNOVER

# --------------------------------------------------------------------------- out of coverage

# Crypto assets by name, ticker or shape. (round 9, E7) Tokens are also named by their ticker next to a fund word
# ("BTC ETF", "ETH现货ETF"), by their project name ("Solana ETF", "币安币"), or as "<X>币" + a fund word ("某某币ETF");
# 人民币 / 港币 / 美元 and money-market funds (货币ETF, 货币基金) are not crypto.
_CRYPTO_TICKERS = r"BTC|ETH|USDT|USDC|DOGE|XRP|SOL|BNB|ADA|DOT|TRX|LTC|SHIB|AVAX|TON|LINK"
_NOT_CRYPTO_BI = "".join(
    f"(?<!{prefix})" for prefix in ("人民", "港", "美", "日", "外", "货", "硬", "纸", "钱", "欧", "英")
)
_CRYPTO = re.compile(
    r"比特币|以太坊|以太币|以太(?=\s*(?:ETF|ETP|现货|期货|基金))|狗狗币|莱特币|瑞波币?|泰达币|币安币?|柴犬币|波场币?|"
    r"艾达币|波卡币|索拉纳|加密货币|加密资产|数字货币|虚拟货币|币圈|山寨币|稳定币|"
    rf"(?<![A-Za-z])(?:{_CRYPTO_TICKERS})(?=\s*(?:ETF|ETP|现货|期货|基金|\bfunds?\b|\btrusts?\b))|"
    r"(?<![A-Za-z])(?:BTC|ETH|USDT|USDC|DOGE|XRP|SOL|BNB|SHIB)(?![A-Za-z])|"
    rf"{_NOT_CRYPTO_BI}币\s*(?=ETF|ETP|现货|期货|基金)|"
    r"\bbitcoin\b|\bethereum\b|\bether\b|\bsolana\b|\bcardano\b|\bpolkadot\b|\bbinance\b|\bripple\b|\btether\b|"
    r"\bcrypto(?:currenc(?:y|ies))?\b|\b(?:doge|lite|stable)coins?\b|\bstablecoins?\b|\btokens?\s+(?:etf|fund)s?\b",
    re.IGNORECASE,
)
# (round 10, F6) Chinese companies listed only in Hong Kong or the US, by the name users write. Several contain or
# resemble an A-share name (平安好医生 / 平安健康 ~ 中国平安, 药明生物 ~ 药明康德, 京东健康 ~ 京东方), so the whole name
# is matched first and the A-share lookalike inside it is not a target (``agent/graph.py`` ``_named_targets``).
# Dual A+H listings (比亚迪, 药明康德, 中芯国际) are A-shares and are not listed here; media and product words
# (网易财经, 百度一下, 腾讯新闻) are not companies asked about.
_HK_US_LISTED = (
    r"平安好医生|平安健康(?:医疗)?|腾讯音乐|腾讯(?!新闻|财经|网|视频|会议|文档|云)|阿里健康|阿里影业|京东健康|京东物流|"
    r"京东集团|京东(?!方)|小米(?:集团|公司)|网易(?!财经|新闻|号|云)|百度(?!一下|搜索|指数|百科|地图|贴吧)|拼多多|快手|"
    r"哔哩哔哩|蔚来(?:汽车)?|理想汽车|小鹏汽车|零跑汽车|携程|贝壳找房|农夫山泉|海底捞|泡泡玛特|蒙牛(?:乳业)?|华润啤酒|"
    r"药明生物|安踏(?:体育)?|李宁公司|中国飞鹤|百胜中国|名创优品|知乎|微博|爱奇艺|金山软件|联想集团|"
    # (round 11, G6) Hong Kong listed subsidiaries and lookalikes of A-share names (比亚迪电子 ~ 比亚迪, 华润置地 ~
    # 华润微, 中国海外发展 ~ 中国海油) and Hong Kong-only blue chips
    r"比亚迪电子|吉利汽车|舜宇光学(?:科技)?|华润置地|华润电力|华润啤酒|华润万象生活|中国海外发展|中信股份|"
    r"石药集团|中国生物制药|碧桂园(?:服务)?|融创中国|长江和记|汇丰控股|友邦保险|香港交易所|港交所|银河娱乐|"
    r"中国旺旺|康师傅|周大福|申洲国际|恒基地产|新鸿基地产|港铁公司"
)
# (round 11, G6) An H share or a Hong Kong ticker: "中国平安H股", "Ping An H shares", "2318.HK", "HK0285". The
# company may also be listed in Shanghai or Shenzhen, but the question asks for the Hong Kong line, which FinSight
# has no data for: the A-share target inside the span is a lookalike, never answered with A-share data.
_H_SHARE = (
    r"(?:(?![和与跟及或比对同的、，,])[一-鿿A-Za-z]){2,8}?\s*(?:的\s*)?H\s*股|H\s*股(?!东)|港股通|"
    r"(?<![\w.])\d{4,5}\s*\.\s*HK(?![A-Za-z])|(?<![A-Za-z])HK\s?\d{4,5}(?!\d)|(?<![\w.])\d{4,5}\s+HK(?![A-Za-z])|"
    r"\b(?:[a-z][\w&.'-]*\s+){1,3}H[- ]?shares?\b|\bH[- ]shares?\b|\bhong kong[- ]listed\b|\bHKEX\b|"
    # (round 12) the Hong Kong line described in words: "它在港交所挂牌的那部分股票", "平安在香港上市的股份",
    # "its shares listed in Hong Kong", "Ping An's Hong Kong listing". As with "…H股", the name before it is part of
    # the span, so the A-share target it names is a lookalike, not answered with A-share data.
    r"(?:(?![和与跟及或比对同的、，,])[一-鿿A-Za-z]){0,8}?(?:在|于)?(?:香港|港交所|联交所|香港交易所)"
    r"(?:上市|挂牌|发行|交易)|"
    r"\b(?:[a-z][\w&.'-]*\s+){0,3}?(?:hong kong (?:listing|line|counter)|"
    r"shares? (?:listed|traded|quoted) (?:in|on(?: the)?) hong kong)\b|"
    r"\b(?:listed|traded|quoted) (?:in|on(?: the)?) hong kong\b"
)
# US / Hong Kong listed names and markets. Concept-sector phrasing ("苹果概念股", "特斯拉产业链") is an
# A-share theme and stays in scope.
_FOREIGN_EQUITY = re.compile(
    r"(?:苹果公司|苹果股票|苹果股价|苹果的股[价票]|苹果(?=的?(?:市盈率|市净率|市值|财报|营收|净利润|股价|股票|能买|值得买|会涨|会跌))|"
    r"特斯拉|英伟达|微软|谷歌|亚马逊|脸书|奈飞|伯克希尔|台积电|"
    r"腾讯控股|阿里巴巴|美团|小米集团|港股|美股|中概股|纳斯达克|道琼斯|标普500|恒生指数|恒指|" + _HK_US_LISTED + r")"
    r"(?!概念|产业链|供应链|链)|"
    r"\b(?:apple|tesla|nvidia|microsoft|google|alphabet|amazon|netflix|berkshire|tencent|alibaba|meituan|tsmc)\b|"
    r"\b(?:ping an (?:good doctor|healthcare)|tencent music|jd (?:health|logistics)|jd\.com|wuxi biologics|"
    r"alibaba health|xiaomi|baidu|netease|pinduoduo|pdd holdings|kuaishou|bilibili|nio inc|li auto|xpeng)\b|"
    r"(?<![A-Za-z])(?:AAPL|TSLA|NVDA|MSFT|GOOGL?|AMZN|NFLX|META|BABA)(?![A-Za-z])|"
    r"\b(?:us|u\.s\.|american|hong kong) (?:stocks?|shares|equities|market)\b|\bnasdaq\b|\bs&p 500\b|\bdow jones\b|"
    r"\bhang seng\b|\bnyse\b|"
    r"\b(?:byd electronic|geely(?: auto(?:mobile)?)?|sunny optical|china resources land|china overseas land|"
    r"citic limited|cspc|sino biopharm|country garden|hsbc|aia group|ck hutchison)\b|" + _H_SHARE,
    re.IGNORECASE,
)


# A question about how a foreign market affects A-shares ("美股大跌对A股有什么影响") is in scope.
_A_SHARE_ANCHOR = re.compile(
    r"A股|沪深|上证|深证|创业板|科创板|北交所|\bA-?shares?\b|\bChin(?:a|ese) (?:stocks?|equities|market)\b",
    re.IGNORECASE,
)


def foreign_equity_spans(query: str) -> list[tuple[int, int]]:
    """Where the question names a US / Hong Kong listed company or market: an A-share name inside such a span
    ("平安" in 平安好医生) is part of the foreign name, not a target of its own."""
    return [match.span() for match in _FOREIGN_EQUITY.finditer(query or "")]


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


def asks_h_share(query: str) -> bool:
    """Whether the question asks for a Hong Kong (H-share) line or ticker ("中国平安H股", "2318.HK")."""
    return bool(re.search(_H_SHARE, query or "", re.IGNORECASE))


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
    # One field of every group present, the metric can be derived (net margin: revenue and net profit).
    derivable_from: tuple[tuple[str, ...], ...] = ()
    # Why it cannot be given when the inputs are missing ("PEG needs the profit growth rate").
    needs_zh: str = ""
    needs_en: str = ""

    def derivable(self, metrics: dict[str, Any]) -> bool:
        return bool(self.derivable_from) and all(
            any(metrics.get(field) is not None for field in group) for group in self.derivable_from
        )


def _metric(
    key: str,
    zh: str,
    en: str,
    pattern: str,
    fields: tuple[str, ...],
    derivable: tuple[tuple[str, ...], ...] = (),
    needs: tuple[str, str] = ("", ""),
) -> Metric:
    return Metric(key, zh, en, re.compile(pattern, re.IGNORECASE), fields, derivable, *needs)


# Profit growth rates a source may report (percent), in order of preference.
PROFIT_GROWTH_FIELDS = ("netprofit_yoy", "net_profit_yoy", "profit_growth", "dt_netprofit_yoy")


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
        r"毛利率|毛利润率|gross margin",
        ("gross_margin", "grossprofit_margin"),
    ),
    # (round 9, E8) net profit as a share of revenue, however it is phrased: "净利润占营收的比重", "(销售)利润率",
    # "每卖100元能落下几块净利润", "profit as a percentage of sales"; not the gross margin ("毛利润率").
    _metric(
        "net_margin",
        "净利率",
        "net margin",
        r"净利率|净利润率|销售净利率|(?<![毛])利润率|"
        r"每(?:赚|卖|收|收入|实现)?[^，。？?,.!！]{0,3}?\d+\s*(?:块|元)(?:钱)?(?:的)?(?:营收|收入|销售额)?"
        r"[^，。？?,.!！]{0,10}?(?:净利润|净利|净赚|利润|落袋)|"
        r"net[- ](?:profit[- ])?margin|profit[- ]margin|(?:net )?(?:profit|income|earnings) as a (?:share|percentage|"
        r"proportion|percent) of (?:revenue|sales)|"
        # (round 11, G5; round 12) "净利润是营收的百分之几", "营收里有多少变成净利润", "What share of that revenue is
        # left as net profit?": one vocabulary with the comparison frame
        f"{NET_MARGIN_SHARE_SOURCE}",
        ("net_margin", "netprofit_margin"),
        (("revenue",), ("net_profit",)),
    ),
    # (round 11, G5) EPS: stated when a source reports it; otherwise named as missing (no share count to derive it),
    # and the template adds the value implied by the latest close and the P/E (TTM), labelled as such.
    _metric(
        "eps",
        "每股收益",
        "EPS",
        r"每股收益|每股盈利|每股净利润?|(?<![A-Za-z])EPS(?![A-Za-z])|earnings per share",
        ("eps", "basic_eps", "diluted_eps", "eps_ttm"),
    ),
    # P/S needs the market cap, which no configured source has: stated as not computable, never replaced by P/E.
    _metric(
        "ps",
        "市销率",
        "P/S",
        r"市销率|(?<![A-Za-z])P/?S(?![A-Za-z])|price[- ]to[- ]sales",
        ("ps_ttm", "ps"),
        (),
        (
            "市销率等于总市值除以营业收入，当前数据没有总市值或市销率",
            "P/S is the market cap divided by revenue, and the data has neither the market cap nor a P/S figure",
        ),
    ),
    # PEG = P/E ÷ profit growth (percent): derivable only when a source reports the growth rate.
    _metric(
        "peg",
        "PEG",
        "PEG",
        r"(?<![A-Za-z])PEG(?![A-Za-z])|市盈增长比|市盈率相对盈利增长比率",
        ("peg", "peg_ratio"),
        (("pe_ttm", "pe"), PROFIT_GROWTH_FIELDS),
        (
            "PEG 等于市盈率除以净利润增速，当前数据没有净利润增速",
            "PEG is the P/E divided by the net profit growth rate, and the data has no growth rate",
        ),
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


# (round 11, G5) "我有1000股五粮液，按最新收盘价值多少钱", "500 shares of Moutai, what are they worth?": the value of a
# stated holding at the last close (shares × close), not a fair-value question. "一股/每股值多少" is a fair value, and
# "10股派…" a dividend per share, so neither is a holding.
_HOLDING_VALUE_ZH = re.compile(
    r"(?<![每\d])(?P<n>\d[\d,，]*|[两二三四五六七八九十百千万][零〇一二两三四五六七八九十百千万]*|一[十百千万][零〇一二两三四五六七八九十百千万]*)"
    # (round 12) fund units count like shares: "两万份沪深300ETF"
    r"\s*(?:股(?![价票东份息市权本派送转配])|份)[^。？?！!；;]{0,24}?"
    r"(?:值|市值|价值|总值|总额|合计|一共|总共|算下来|折合)[^。？?！!；;]{0,4}?(?:多少|几)"
)
_HOLDING_VALUE_EN = re.compile(
    r"\b(?P<n>\d[\d,]*)\s+(?:shares?|units?)\b[^.?!]{0,60}?\b(?:worth|value|how much)\b|"
    # (round 12) "I own 500 shares of X. What are they worth at the close?": the question in the next sentence
    r"\b(?P<n3>\d[\d,]*)\s+(?:shares?|units?)\b[^.?!]{0,60}\.\s+(?:so\s+|and\s+)?(?:what|how much)\b[^.?!]{0,40}?"
    r"\b(?:worth|value)\b|"
    r"\bhow much (?:are|is|would) (?:my )?(?P<n2>\d[\d,]*)\s+(?:shares?|units?)\b",
    re.IGNORECASE,
)
# A fund holding is counted in units (份), a stock holding in shares (股).
_FUND_UNITS = re.compile(r"\d\s*份|[两二三四五六七八九十百千万]\s*份|\bunits?\b", re.IGNORECASE)
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _cn_integer(text: str) -> int:
    total, section, digit = 0, 0, 0
    for char in text:
        if char in _CN_DIGIT:
            digit = _CN_DIGIT[char]
        elif char == "万":
            total += (section + digit) * 10000
            section, digit = 0, 0
        else:
            section += (digit or 1) * _CN_UNIT[char]
            digit = 0
    return total + section + digit


def holding_value_request(query: str) -> tuple[int, tuple[int, int]] | None:
    """``(shares, span)`` when the question asks what a stated number of shares is worth, else ``None``."""
    text = query or ""
    match = _HOLDING_VALUE_ZH.search(text) or _HOLDING_VALUE_EN.search(text)
    if match is None:
        return None
    groups = match.groupdict()
    raw = (groups.get("n") or groups.get("n2") or groups.get("n3") or "").replace(",", "").replace("，", "")
    shares = int(raw) if raw.isdigit() else _cn_integer(raw)
    return (shares, match.span()) if shares > 1 else None


def in_fund_units(query: str) -> bool:
    """Whether a holding is stated in fund units ("两万份", "4000 units"), not shares."""
    return bool(_FUND_UNITS.search(query or ""))


_HOLDING_COUNT = re.compile(
    r"(?<![每\d])(?P<n>\d[\d,，]*|[两二三四五六七八九十百千万][零〇一二两三四五六七八九十百千万]*)\s*(?:股(?![价票东份息市权本派送转配])|份)|"
    r"\b(?P<n2>\d[\d,]*)\s+(?:shares?|units?)\b",
    re.IGNORECASE,
)


def stated_holding_count(text: str) -> int | None:
    """A number of shares or units stated without a value question ("同样300股", "如果是500股", "800 shares")."""
    match = _HOLDING_COUNT.search(text or "")
    if match is None:
        return None
    raw = (match.group("n") or match.group("n2") or "").replace(",", "").replace("，", "")
    count = int(raw) if raw.isdigit() else _cn_integer(raw)
    return count if count > 1 else None


def without_holding_value(query: str) -> str:
    """The question with a holding-value request blanked out, for the fair-value and judgment checks."""
    found = holding_value_request(query)
    if found is None:
        return query
    start, end = found[1]
    return f"{query[:start]} {query[end:]}"


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
            if not any(metrics.get(field) is not None for field in metric.fields) and not metric.derivable(metrics)
        ]
        for metric in [metric for metric in missing if metric.needs_zh]:
            # "茅台的PEG": say what the metric needs and that it is missing, not just "no PEG data"
            missing.remove(metric)
            sentences.append(
                f"无法计算{name}的{metric.zh}：{metric.needs_zh}；以下只列出可得的指标。"
                if zh
                else f"{name}'s {metric.en} cannot be computed: {metric.needs_en}; only the available metrics are "
                "listed below."
            )
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
    if any(entry.get("tool") == "explain_concept" and entry.get("ok") for entry in tool_log or []):
        return []  # the glossary answer already says the concept has no data series
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


# A year-to-date move ("今年涨了多少", "年初至今收益", "YTD return", "how has it done this year"), not this year's
# statements ("今年营收").
_YEAR_TO_DATE = re.compile(
    r"今年(?:以来|到现在|至今|迄今)?(?:的|一共|总共|累计|整体|共){0,3}(?:涨|跌|收益|回报|表现|涨幅|跌幅|走势)|"
    r"年初(?:至今|到现在|以来)|年内(?:的|累计|共){0,2}(?:涨|跌|收益|回报|涨幅|跌幅|表现)|"
    r"\bYTD\b|\byear[- ]to[- ]date\b|\bso far this year\b|\bsince the (?:start|beginning) of (?:the|this) year\b|"
    r"\bthis year\b.{0,20}\b(?:return|gain|perform|rise|rose|risen|fall|fell|fallen|up|down|change|move)|"
    r"\b(?:return|gain|performance|change|move|up|down|rise|rose|fall|fell)\b.{0,20}\bthis year\b",
    re.IGNORECASE,
)


def asks_year_to_date(query: str) -> bool:
    return bool(_YEAR_TO_DATE.search(query or ""))


def year_to_date_gaps(query: str, tool_log: list[dict[str, Any]], *, zh: bool) -> list[str]:
    """A year-to-date question whose price data does not reach back to the first trading day of the year.

    ``get_price_history`` reports ``year_start`` (the first close of the latest close's year) only when its history
    also has a close from the year before, so that close is known to be the year's first. Without it the change
    since the start of the year is stated as unavailable, instead of answering with the latest daily move.
    """
    if not asks_year_to_date(query):
        return []
    sentences = []
    for entry in tool_log:
        if entry.get("tool") != "get_price_history" or not entry.get("ok"):
            continue
        data = entry.get("data") or {}
        if data.get("year_start") or data.get("close") is None:
            continue
        name = str(data.get("name") or data.get("symbol") or "")
        sentences.append(
            f"当前数据中没有{name}今年首个交易日的收盘价，无法计算今年以来的涨跌幅；以下只列出最新一日的行情。"
            if zh
            else f"The data has no close for {name} on the first trading day of this year, so the year-to-date "
            "change cannot be computed; only the latest session is listed below."
        )
    return list(dict.fromkeys(sentences))


# (round 9, E8) A maximum drawdown over a period ("近一年最大回撤", "max drawdown this year") needs the whole period's
# closes; the price tool returns the latest few. Stated as not computable instead of answering with the daily move.
_DRAWDOWN = re.compile(r"最大回撤|回撤幅度|最大跌幅|\bmax(?:imum)?\.? drawdown\b|\bdrawdown\b", re.IGNORECASE)


def asks_drawdown(query: str) -> bool:
    return bool(_DRAWDOWN.search(query or ""))


def drawdown_gaps(query: str, tool_log: list[dict[str, Any]], *, zh: bool) -> list[str]:
    """A drawdown question: the price data holds only the latest closes, so the drawdown is stated as unavailable."""
    if not asks_drawdown(query):
        return []
    sentences = []
    for entry in tool_log:
        if entry.get("tool") != "get_price_history" or not entry.get("ok"):
            continue
        data = entry.get("data") or {}
        if data.get("close") is None:
            continue
        name = str(data.get("name") or data.get("symbol") or "")
        closes = len(data.get("recent_closes") or [])
        held_zh = f"最近 {closes} 个交易日的收盘价" if closes >= 2 else "最新一个交易日的收盘价"
        held_en = f"the latest {closes} closes" if closes >= 2 else "the latest close"
        sentences.append(
            f"当前数据只有{name}{held_zh}，无法计算所问期间的最大回撤；以下只列出最新行情。"
            if zh
            else f"The data holds only {held_en} for {name}, so the maximum drawdown over the requested period "
            "cannot be computed; only the latest session is listed below."
        )
    return list(dict.fromkeys(sentences))


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
_AMOUNT = TURNOVER  # (round 12) the comparison frame's turnover vocabulary ("成交了多少钱", "trading value")
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
    year_to_date: bool = False

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
        year_to_date=asks_year_to_date(text),
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
