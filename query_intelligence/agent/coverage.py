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


def _years_text(years: list[int], *, zh: bool) -> str:
    # "2019年" / "FY2019": forms the verifier reads as dates, not as claimed values.
    return "、".join(f"{year}年" for year in years) if zh else ", ".join(f"FY{year}" for year in years)


def failed_target_statements(
    query: str, tool_log: list[dict[str, Any]], *, zh: bool, names: dict[str, str] | None = None
) -> list[str]:
    """Name the targets whose data could not be retrieved (instead of a generic "no evidence" line)."""
    names = names or {}
    per_target: dict[str, list[str]] = {}
    for entry in tool_log:
        if entry.get("ok"):
            continue
        arguments = entry.get("arguments") or {}
        target = arguments.get("target") or ((arguments.get("targets") or [None])[0])
        if not target:
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
    statements = []
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
