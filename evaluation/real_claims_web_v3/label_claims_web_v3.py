"""Label the third blind set from the offline data files only (deterministic)::

    python -m evaluation.real_claims_web_v3.label_claims_web_v3   # collected_v3.jsonl -> claims_web_v3.jsonl

A person wrote each claim's *parts* (below, keyed by candidate id): the entity, the metric, the number as written, a
qualifier, and the basis the sentence names. Every value a part is compared with is computed here from
``data/structured_data.json``, ``data/snapshot/structured_data_ext.json``, ``data/snapshot/structured_data_fix.json``
and ``data/snapshot/corrections/sources.json``, with the corrections layer over the extension over the v1 seed (the
shipped default). The rules are in ``README.md`` ("Labelling rules"), written before this script.
"""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COLLECTED = HERE / "collected_v3.jsonl"
OUT = HERE / "claims_web_v3.jsonl"
DATA = ROOT / "data"

SYMBOLS = {
    "宁德时代": "300750.SZ",
    "比亚迪": "002594.SZ",
    "美的集团": "000333.SZ",
    "紫金矿业": "601899.SH",
    "中芯国际": "688981.SH",
    "招商银行": "600036.SH",
    "工商银行": "601398.SH",
    "平安银行": "000001.SZ",
    "中信证券": "600030.SH",
    "长江电力": "600900.SH",
    "贵州茅台": "600519.SH",
    "五粮液": "000858.SZ",
    "中国平安": "601318.SH",
    "上证指数": "000001.SH",
    "创业板指": "399006.SZ",
    "沪深300指数": "000300.SH",
    "科创50ETF": "588000.SH",
    "创业板ETF": "159915.SZ",
    "中证500ETF": "510500.SH",
    "黄金ETF": "518880.SH",
    "沪深300ETF": "510300.SH",
    "证券ETF": "512880.SH",
}
UNITS = {
    "": 1.0,
    "元": 1.0,
    "%": 1.0,
    "点": 1.0,
    "倍": 1.0,
    "亿": 1e8,
    "万亿": 1e12,
    "billion": 1e9,
    "万手": 1e6,  # 1 手 = 100 shares
}
UNIT_SCALE = {"元": 1.0, "千元": 1e3, "百万元": 1e6, "%": 1.0, "元/股": 1.0, "股": 1.0}
LOOSE = {"approx", "ge", "gt", "near", "lt", "more"}
ANNUAL = {
    "revenue",
    "net_profit",
    "profit_dedt",
    "revenue_yoy",
    "netprofit_yoy",
    "roe",
    "gross_margin",
    "net_margin",
    "eps",
    "daily_net_profit",
    "prior_revenue",
}
RATIOS = {"revenue_yoy", "netprofit_yoy", "roe", "gross_margin", "net_margin"}
VALUATION = {"total_mv", "pe_ttm", "pb"}
BAR = {"close", "open", "high", "low", "pct", "change", "amount", "volume", "amount_prev_diff", "amount_prev_pct"}


# ---------------------------------------------------------------- data


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


SEED = _load(DATA / "structured_data.json")
EXT = _load(DATA / "snapshot" / "structured_data_ext.json")
FIX = _load(DATA / "snapshot" / "structured_data_fix.json")
SOURCES = _load(DATA / "snapshot" / "corrections" / "sources.json")


def fundamentals(symbol: str) -> dict[str, Any]:
    if symbol in FIX["fundamental_sql"]:
        row = dict(FIX["fundamental_sql"][symbol])
    else:
        row = dict(EXT["fundamental_sql"][symbol])
        row["definitions"] = FIX["definitions"][symbol]
    if "gross_margin" not in row:
        row["gross_margin"] = row.get("grossprofit_margin")
    return row


def bars(symbol: str) -> dict[str, dict[str, Any]]:
    """Daily bars by date: corrections layer over extension over v1 seed."""
    out: dict[str, dict[str, Any]] = {}
    seed = SEED["market_api"].get(symbol)
    if seed:
        history = seed.get("history") or [{k: seed[k] for k in seed if k != "history"}]
        for bar in history:
            bar = dict(bar)
            if seed.get("source_name") == "tushare":  # tushare units: amount in 千元, volume in 手
                bar["amount"] = bar["amount"] * 1000
                bar["volume"] = bar["volume"] * 100
            out[bar["trade_date"]] = bar
    for layer in (EXT, FIX):
        rec = layer["market_api"].get(symbol)
        if rec:
            for bar in rec.get("history", []):
                out[bar["trade_date"]] = dict(bar)
    return out


def bar_on(symbol: str, date: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    history = bars(symbol)
    if date not in history:
        return None, None
    earlier = sorted(d for d in history if d < date)
    return history[date], (history[earlier[-1]] if earlier else None)


def _num(text: str) -> float:
    return float(text.replace(",", "").replace("%", "").strip())


def report(symbol: str) -> dict[str, Any]:
    for company in SOURCES["companies"]:
        if company["symbol"] == symbol:
            return company
    return SOURCES["extension_definitions"][symbol]


def report_figures(symbol: str) -> dict[str, tuple[float, float | None]]:
    """Report figure id -> (value in base units, prior-year value or None)."""
    out: dict[str, tuple[float, float | None]] = {}
    for fig in report(symbol)["figures"]:
        scale = UNIT_SCALE[fig["unit"]]
        prior = _num(fig["prior"]) * scale if fig.get("prior") else None
        out[fig["id"]] = (_num(fig["value"]) * scale, prior)
    return out


def primary(symbol: str, key: str) -> float | None:
    """The annual report's figure on a named basis (None when the data files do not carry it)."""
    figs = report_figures(symbol)

    def val(fid: str) -> float | None:
        return figs[fid][0] if fid in figs else None

    def yoy(fid: str) -> float | None:
        if fid not in figs or figs[fid][1] is None:
            return None
        value, prior = figs[fid]
        return (value / prior - 1) * 100

    if key in ("revenue_yoy", "net_profit_parent_yoy") and key in figs:
        return figs[key][0]
    if key == "revenue_yoy":
        return yoy("revenue")
    if key == "net_profit_parent_yoy":
        return yoy("net_profit_parent")
    if key == "net_profit_total_yoy":
        return yoy("net_profit_total")
    if key == "prior_revenue":
        return figs["revenue"][1] if "revenue" in figs else None
    if key == "roe_period_end":
        np_, eq = val("net_profit_parent"), val("equity_parent")
        return np_ / eq * 100 if np_ and eq else None
    if key == "gross_margin":
        rev, cost = val("revenue"), val("cost_of_sales")
        return (rev - cost) / rev * 100 if rev and cost else None
    if key.startswith("net_margin:"):
        _, num, den = key.split(":")
        n, d = (
            val(num),
            val(den) or (val("revenue") if den == "revenue_total" and equal_bases(symbol, "revenue") else None),
        )
        return n / d * 100 if n and d else None
    if key == "revenue_total" and key not in figs and equal_bases(symbol, "revenue"):
        return val("revenue")
    if key.startswith("daily:"):
        v = val(key.split(":", 1)[1])
        return v / 365 if v else None
    return val(key)


def equal_bases(symbol: str, what: str) -> bool:
    """What the annual-report notes in sources.json state as equal on both bases."""
    notes = " ".join(report(symbol).get("notes", []))
    if what == "revenue":
        return "营业收入 = 营业总收入" in notes
    if what == "eps":
        return "basic EPS = diluted EPS" in notes or "基本/稀释" in json.dumps(report(symbol), ensure_ascii=False)
    if what == "net_profit":
        return "归母净利润 = 净利润" in notes
    return False


def served_key(symbol: str, metric: str) -> str:
    """The report figure id that the served field is on (from the stated definitions)."""
    d = fundamentals(symbol)["definitions"]
    rev = "revenue_total" if d["revenue_basis"] == "营业总收入" else "revenue"
    return {
        "revenue": rev,
        "net_profit": "net_profit_parent",
        "profit_dedt": "profit_dedt",
        "revenue_yoy": "revenue_yoy",
        "netprofit_yoy": "net_profit_parent_yoy" if d["netprofit_yoy_basis"] == "归母" else "net_profit_total_yoy",
        "roe": "roe_weighted" if d["roe_basis"] == "weighted" else "roe_period_end",
        "gross_margin": "gross_margin",
        "eps": "eps_diluted" if d["eps_basis"] == "diluted" else "eps_basic",
        "net_margin": f"net_margin:net_profit_parent:{rev}",
        "daily_net_profit": "daily:net_profit_parent",
        "prior_revenue": "prior_revenue",
    }[metric]


def fact(entity: str, metric: str, date: str | None = None) -> tuple[float | None, str]:
    """The offline value of one metric (base units: 元, %, 点, shares) and how it was obtained."""
    symbol = SYMBOLS[entity]
    if metric in BAR:
        bar, prev = bar_on(symbol, date or "")
        if bar is None:
            return None, f"no {symbol} bar on {date}"
        if metric == "pct":
            if bar.get("pct_change_1d") is not None:
                return float(bar["pct_change_1d"]), f"{symbol} {date} pct_change_1d"
            if prev:
                return (bar["close"] / prev["close"] - 1) * 100, f"{symbol} {date} close / previous close"
            return None, "no % change"
        if metric == "change":
            if prev:
                return bar["close"] - prev["close"], f"{symbol} {date} close − previous close"
            pct = bar.get("pct_change_1d")
            return bar["close"] - bar["close"] / (1 + pct / 100), f"{symbol} {date} close − close/(1+pct)"
        if metric in ("amount_prev_diff", "amount_prev_pct"):
            if not prev or not prev.get("amount") or not bar.get("amount"):
                return None, "no previous amount"
            if metric == "amount_prev_diff":
                return bar["amount"] - prev["amount"], f"{symbol} amount {date} − {prev['trade_date']}"
            return (bar["amount"] / prev["amount"] - 1) * 100, f"{symbol} amount {date} / {prev['trade_date']} − 1"
        value = bar.get(metric)
        return (float(value) if value is not None else None), f"{symbol} {date} {metric}"
    row = fundamentals(symbol)
    if metric in VALUATION:
        if date and date != row["valuation_date"]:
            return None, f"valuation is on {row['valuation_date']}, not {date}"
        return float(row[metric]), f"{symbol} {metric} on {row['valuation_date']}"
    if metric == "net_margin":
        return row["net_profit"] / row["revenue"] * 100, f"{symbol} net_profit / revenue"
    if metric == "daily_net_profit":
        return row["net_profit"] / 365, f"{symbol} net_profit / 365"
    if metric == "prior_revenue":
        return row["revenue"] / (1 + row["revenue_yoy"] / 100), f"{symbol} revenue / (1 + revenue_yoy)"
    value = row.get(metric)
    return (float(value) if value is not None else None), f"{symbol} {metric} ({row.get('source_name')})"


# ---------------------------------------------------------------- rules


def half_unit(written: str) -> float:
    digits = written.lstrip("+-").replace(",", "")
    if "." in digits:
        return 0.5 * 10 ** -len(digits.split(".")[1])
    return 0.5


def holds(written: str, unit: str, q: str, actual: float, signed: bool) -> tuple[bool, str]:
    claimed = float(written.replace(",", "")) * UNITS[unit]
    half = half_unit(written) * UNITS[unit]
    if signed and claimed != 0 and actual != 0 and math.copysign(1, claimed) != math.copysign(1, actual):
        return False, "sign differs"
    c, a = (abs(claimed), abs(actual)) if signed else (claimed, actual)
    if q == "exact":
        tol = max(half, 0.01 * abs(a))
        return abs(c - a) <= tol + 1e-12 * max(1.0, abs(a)), f"|claimed − actual| ≤ max(half unit, 1%) = {tol:.6g}"
    if q == "approx":
        return abs(a - c) <= 0.05 * abs(c), "within 5% of the claimed number"
    if q == "ge":
        return a >= c, "actual ≥ N"
    if q == "gt":
        return a > c, "actual > N"
    if q == "near":
        return 0.9 * c <= a <= c, "0.9 N ≤ actual ≤ N"
    if q == "lt":
        return a < c, "actual < N"
    if q == "more":
        return c < a < c + 2 * half, "N < actual < N + one unit of the last digit"
    raise ValueError(q)


def basis_null(part: dict[str, Any]) -> str | None:
    """Reason a named basis makes the part undecidable from the offline data, or None."""
    named = part.get("named")
    if not named:
        return None
    symbol = SYMBOLS[part["e"]]
    served = served_key(symbol, part["m"])
    if named == served:
        return None
    a, b = primary(symbol, named), primary(symbol, served)
    if named.startswith("revenue") and served.startswith("revenue") and equal_bases(symbol, "revenue"):
        return None
    if a is not None and b is not None and abs(a - b) <= max(5e-4 * abs(b), 1e-9):
        return None
    return f"names {named}; the offline field is on {served}"


def agree(o: float, p: float) -> bool:
    return abs(o - p) <= max(5e-4 * abs(p), 0.005)


def evaluate_part(part: dict[str, Any], *, definition_aware: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"reading": part["text"]}
    if "null" in part:
        return out | {"holds": None, "reason": part["null"], "metric": part.get("m")}
    if "rel" in part:
        a, ra = fact(part["a"]["e"], part["a"]["m"], part["a"].get("date"))
        b, rb = fact(part["b"]["e"], part["b"]["m"], part["b"].get("date"))
        ok = {"gt": a > b, "lt": a < b, "abs_gt": abs(a) > abs(b)}[part["rel"]]
        return out | {
            "holds": ok,
            "actual": [round(a, 6), round(b, 6)],
            "rule": f"relation {part['rel']}: {ra} vs {rb}",
        }
    if definition_aware:
        reason = basis_null(part)
        if reason:
            value, how = fact(part["e"], part["m"], part.get("date"))
            return out | {"holds": None, "reason": reason, "actual": value, "source": how, "metric": part["m"]}
    value, how = fact(part["e"], part["m"], part.get("date"))
    if value is None:
        return out | {"holds": None, "reason": how, "metric": part["m"]}
    ok, rule = holds(part["v"], part.get("u", ""), part.get("q", "exact"), value, part.get("signed", False))
    return out | {"holds": ok, "actual": value, "rule": f"{part.get('q', 'exact')}: {rule}", "source": how}


def verdict(values: list[bool | None]) -> str:
    if values and all(v is True for v in values):
        return "supported"
    if any(v is True for v in values):
        return "partially_supported"
    if any(v is False for v in values):
        return "contradicted"
    return "unverifiable"


def primary_row(index: int, part: dict[str, Any]) -> dict[str, Any] | None:
    if "null" in part or "rel" in part or part["m"] not in ANNUAL:
        return None
    symbol = SYMBOLS[part["e"]]
    served = served_key(symbol, part["m"])
    pbasis = part.get("pbasis") or part.get("named") or served
    offline, _ = fact(part["e"], part["m"])
    rep = primary(symbol, pbasis)
    rep_served = primary(symbol, served)
    row: dict[str, Any] = {"part": index, "basis": pbasis, "served_basis": served, "offline": offline}
    if rep is None:
        return row | {"report": None, "offline_vs_primary": "no_primary", "claim_vs_primary": None}
    if pbasis == served or agree(offline, rep):
        ovp = "agree" if agree(offline, rep) else "value_differs"
    else:
        ovp = "value_differs" if rep_served is not None and not agree(offline, rep_served) else "definition_differs"
    unit = part.get("u", "")
    claim_ok, _ = holds(part["v"], unit, part.get("q", "exact"), rep, part.get("signed", False))
    return row | {"report": rep, "offline_vs_primary": ovp, "claim_vs_primary": claim_ok}


# ---------------------------------------------------------------- parts (written by a person, per candidate id)


def P(text: str, e: str, m: str, v: str, u: str = "", q: str = "exact", **kw: Any) -> dict[str, Any]:
    return {"text": text, "e": e, "m": m, "v": v, "u": u, "q": q, **kw}


def N(text: str, reason: str, m: str | None = None) -> dict[str, Any]:
    return {"text": text, "null": reason, "m": m}


def R(text: str, rel: str, a: tuple[str, str], b: tuple[str, str], date: str | None = None) -> dict[str, Any]:
    return {
        "text": text,
        "rel": rel,
        "a": {"e": a[0], "m": a[1], "date": date},
        "b": {"e": b[0], "m": b[1], "date": date},
    }


D0930, D0422 = "2026-09-30", "2026-04-22"
TURNOVER = "换手率 is not in the offline data"
PP = "a change in percentage points is not in the offline data"
DEDT_YOY = "扣非 YoY is not in the offline data"
PRIOR_MARGIN = "prior-year margins are not in the offline data"

PARTS: dict[str, dict[str, Any]] = {
    "cC16": {
        "parts": [P("全天成交额近70亿元", "创业板ETF", "amount", "70", "亿", "near", date=D0422)],
        "context": ["较上日大幅放量"],
    },
    "cC22": {
        "parts": [
            P("今日成交额68.50亿元", "科创50ETF", "amount", "68.50", "亿", date=D0930),
            P("较上一交易日增加8.28亿元", "科创50ETF", "amount_prev_diff", "8.28", "亿", date=D0930, signed=True),
            P("环比增幅为13.75%", "科创50ETF", "amount_prev_pct", "13.75", "%", date=D0930, signed=True),
        ]
    },
    "cD06": {
        "parts": [
            P("今日下跌2.47%", "科创50ETF", "pct", "-2.47", "%", date=D0930, signed=True),
            P("成交额68.50亿元", "科创50ETF", "amount", "68.50", "亿", date=D0930),
            N("换手率7.43%", TURNOVER, "turnover_rate"),
            N("规模937.13亿元", "fund size is not in the offline data", "fund_size"),
        ]
    },
    "cC15": {
        "parts": [
            P("沪指涨0.52%", "上证指数", "pct", "0.52", "%", date=D0422, signed=True),
            N("深成指涨1.3%", "深证成指 is not in the offline data", "pct"),
            P("创业板指涨1.73%", "创业板指", "pct", "1.73", "%", date=D0422, signed=True),
        ]
    },
    "cC19": {
        "parts": [P("上证指数收盘指数站上4100点", "上证指数", "close", "4100", "点", "ge", date=D0422)],
        "context": ["A股大盘指数延续上涨势头", "创业板指收盘指数创下近10年新高"],
    },
    "cD00": {
        "parts": [
            P("上证指数上涨0.52%", "上证指数", "pct", "0.52", "%", date=D0422, signed=True),
            P("报收4106.26点", "上证指数", "close", "4106.26", "点", date=D0422),
            P("成功重返4100点整数关口", "上证指数", "close", "4100", "点", "ge", date=D0422),
        ]
    },
    "cD10": {"parts": [P("where it ended September, at 3,842", "上证指数", "close", "3842", "点", date=D0930)]},
    "cD16": {
        "parts": [
            P("开3839.25", "上证指数", "open", "3839.25", "点", date=D0930),
            P("最高3851.22", "上证指数", "high", "3851.22", "点", date=D0930),
            P("最低3833.09", "上证指数", "low", "3833.09", "点", date=D0930),
            P("收3842.19", "上证指数", "close", "3842.19", "点", date=D0930),
            P("涨11.74点", "上证指数", "change", "11.74", "点", date=D0930, signed=True),
            P("（+0.31%）", "上证指数", "pct", "0.31", "%", date=D0930, signed=True),
        ]
    },
    "cD17": {
        "parts": [
            P("沪指报3842.19点", "上证指数", "close", "3842.19", "点", date=D0930),
            P("涨0.31%", "上证指数", "pct", "0.31", "%", date=D0930, signed=True),
            N("深成指报12887.62点", "深证成指 is not in the offline data", "close"),
            N("跌0.11%", "深证成指 is not in the offline data", "pct"),
            P("创指报3135.28点", "创业板指", "close", "3135.28", "点", date=D0930),
            P("跌0.23%", "创业板指", "pct", "-0.23", "%", date=D0930, signed=True),
        ]
    },
    "cB01": {
        "parts": [
            P("当前市盈TTM为10.19", "中信证券", "pe_ttm", "10.19", date=D0930),
            N("在过去5年中处于个股0.08%的位置", "a 5-year valuation percentile is not in the offline data", "pe_pct"),
        ],
        "context": ["低于行业中位数"],
    },
    "cB22": {
        "parts": [
            P("2025年加权平均净资产收益率10.59%", "中信证券", "roe", "10.59", "%", named="roe_weighted"),
            N("同比增加2.5个百分点", PP, "roe"),
        ]
    },
    "cE12": {
        "parts": [
            P("营业收入748.54亿元", "中信证券", "revenue", "748.54", "亿", named="revenue"),
            P("归母净利润300.76亿元", "中信证券", "net_profit", "300.76", "亿", named="net_profit_parent"),
            N("在全行业150家券商中占比…13.8%", "industry totals are not in the offline data", "share"),
            N("…和13.7%", "industry totals are not in the offline data", "share"),
        ]
    },
    "cB09": {
        "parts": [
            P("a 6.45% year-on-year rise in net profit in 2025", "中国平安", "netprofit_yoy", "6.45", "%", signed=True)
        ]
    },
    "cB17": {
        "parts": [
            N("国寿…1541亿元", "中国人寿 is not in the offline data", "net_profit"),
            P("平安…1348亿元", "中国平安", "net_profit", "1348", "亿"),
            N("太保…535亿元", "中国太保 is not in the offline data", "net_profit"),
            N("人保…466亿元", "中国人保 is not in the offline data", "net_profit"),
            N("新华…363亿元", "新华保险 is not in the offline data", "net_profit"),
        ]
    },
    "cB25": {
        "parts": [
            P("收入1.05万亿", "中国平安", "revenue", "1.05", "万亿"),
            P("利润增7%", "中国平安", "netprofit_yoy", "7", "%", signed=True),
            P("到1348亿", "中国平安", "net_profit", "1348", "亿"),
        ]
    },
    "cE01": {
        "parts": [P("中国平安日赚3.68亿", "中国平安", "daily_net_profit", "3.68", "亿")],
        "context": ["堪称“超级印钞机”"],
        "note": (
            "The article's preceding sentence derives 3.68亿 from 1344亿 of 营运利润; the sentence itself says 日赚."
        ),
    },
    "cA01": {
        "parts": [
            P("报收于111.99元", "中芯国际", "close", "111.99", "元", date=D0930),
            P("下跌4.7%", "中芯国际", "pct", "-4.7", "%", date=D0930, signed=True),
            N("换手率2.14%", TURNOVER, "turnover_rate"),
            P("成交量42.87万手", "中芯国际", "volume", "42.87", "万手", date=D0930),
            P("成交额48.56亿元", "中芯国际", "amount", "48.56", "亿", date=D0930),
        ]
    },
    "cA20": {
        "parts": [
            P("全年实现销售收入人民币673.23亿元", "中芯国际", "revenue", "673.23", "亿"),
            P("同比增长16.5%", "中芯国际", "revenue_yoy", "16.5", "%", signed=True),
        ],
        "context": ["继续巩固全球纯晶圆代工企业第二位置"],
    },
    "cA21": {
        "parts": [
            P(
                "去年的净利率为10.71%",
                "中芯国际",
                "net_margin",
                "10.71",
                "%",
                pbasis="net_margin:net_profit_parent:revenue",
            )
        ],
        "context": ["算上全部成本后，公司产品或服务的附加值一般"],
        "note": (
            "10.71% equals total net profit (incl. minority interests) / revenue in the report; "
            "归母 / revenue is 7.49%."
        ),
    },
    "cC27": {
        "parts": [P("中芯国际(688981)下跌4.70%", "中芯国际", "pct", "-4.70", "%", date=D0930, signed=True)],
        "context": ["前五大权重股全部下跌"],
    },
    "cC07": {
        "parts": [
            P("2025年总营收405.29亿元", "五粮液", "revenue", "405.29", "亿", named="revenue_total"),
            P("（同比-55%）", "五粮液", "revenue_yoy", "-55", "%", signed=True),
            P("归母净利润89.54亿元", "五粮液", "net_profit", "89.54", "亿", named="net_profit_parent"),
            P("（同比-72%）", "五粮液", "netprofit_yoy", "-72", "%", signed=True, named="net_profit_parent_yoy"),
            P("扣非净利润88.16亿元", "五粮液", "profit_dedt", "88.16", "亿"),
            N("（同比-72%）", DEDT_YOY, "dedt_yoy"),
        ]
    },
    "cC08": {
        "parts": [
            P("扣非归母净利润88.16亿元", "五粮液", "profit_dedt", "88.16", "亿"),
            N("同比下降72.23%", DEDT_YOY, "dedt_yoy"),
        ]
    },
    "cC09": {
        "parts": [
            P(
                "归属于母公司股东的净利润为8,954,257,202.51元",
                "五粮液",
                "net_profit",
                "8954257202.51",
                "元",
                named="net_profit_parent",
            )
        ]
    },
    "cC12": {
        "parts": [
            P("净利润降幅为71.9%", "五粮液", "netprofit_yoy", "-71.9", "%", signed=True),
            P("营收降幅的54.6%", "五粮液", "revenue_yoy", "-54.6", "%", signed=True),
            R("净利润降幅…显著大于营收降幅", "abs_gt", ("五粮液", "netprofit_yoy"), ("五粮液", "revenue_yoy")),
        ]
    },
    "cA00": {"parts": [P("宁德时代滚动市盈率为15.85倍", "宁德时代", "pe_ttm", "15.85", "倍", date=D0930)]},
    "cA08": {
        "parts": [
            P("non-recurring net profit reached 64.51 billion yuan", "宁德时代", "profit_dedt", "64.51", "billion"),
            N("(9.34 billion USD)", "USD amounts are not in the offline data", "profit_dedt"),
            N("up 43.37% year-on-year", DEDT_YOY, "dedt_yoy"),
        ]
    },
    "cA26": {
        "parts": [
            N("从2024年的约14.9%", PRIOR_MARGIN, "net_margin"),
            P(
                "提升至2025年的约18.1%",
                "宁德时代",
                "net_margin",
                "18.1",
                "%",
                "approx",
                pbasis="net_margin:net_profit_parent:revenue",
            ),
        ],
        "note": "约18.1% equals total net profit / revenue in the report (18.12%); 归母 / revenue is 17.04%.",
    },
    "cB03": {
        "parts": [
            P("报收于8.28元", "工商银行", "close", "8.28", "元", date=D0930),
            P("上涨1.47%", "工商银行", "pct", "1.47", "%", date=D0930, signed=True),
            N("换手率0.11%", TURNOVER, "turnover_rate"),
            P("成交量293.17万手", "工商银行", "volume", "293.17", "万手", date=D0930),
            P("成交额24.18亿元", "工商银行", "amount", "24.18", "亿", date=D0930),
        ]
    },
    "cB18": {
        "parts": [
            P("净利润3,707.66亿元", "工商银行", "net_profit", "3707.66", "亿", pbasis="net_profit_total"),
            P("营业收入8,382.70亿元", "工商银行", "revenue", "8382.70", "亿", named="revenue"),
            N("手续费及佣金净收入1,111.71亿元", "fee income is not in the offline data", "fee_income"),
        ],
        "context": ["三项核心盈利指标同步实现正增长"],
    },
    "cB23": {
        "parts": [
            N("2003年的223.7亿", "2003 profit is not in the offline data", "net_profit"),
            P("增加到2025年的3686亿", "工商银行", "net_profit", "3686", "亿"),
            N("翻了16.5倍", "needs the 2003 profit, which is not in the offline data", "ratio"),
        ]
    },
    "cB04": {
        "parts": [
            P("报收于11.57元", "平安银行", "close", "11.57", "元", date=D0930),
            P("上涨1.94%", "平安银行", "pct", "1.94", "%", date=D0930, signed=True),
            N("换手率0.54%", TURNOVER, "turnover_rate"),
            P("成交量104.54万手", "平安银行", "volume", "104.54", "万手", date=D0930),
            P("成交额12.06亿元", "平安银行", "amount", "12.06", "亿", date=D0930),
        ]
    },
    "cB19": {
        "parts": [
            P("实现净利润426.33亿元", "平安银行", "net_profit", "426.33", "亿", pbasis="net_profit_total"),
            P("同比下降4.2%", "平安银行", "netprofit_yoy", "-4.2", "%", signed=True),
        ]
    },
    "cD20": {
        "parts": [
            P("[平安银行]…ROE…~9%", "平安银行", "roe", "9", "%", "approx", pbasis="roe_weighted"),
            P("招行的ROE…13.44%", "招商银行", "roe", "13.44", "%", pbasis="roe_weighted"),
        ],
        "context": ["差距…预计2026年仍难以明显收敛"],
    },
    "cB00": {
        "parts": [
            P("当前市盈TTM为6.86", "招商银行", "pe_ttm", "6.86", date=D0930),
            N("在过去5年中处于个股59.01%的位置", "a 5-year valuation percentile is not in the offline data", "pe_pct"),
        ],
        "context": ["高于行业中位数"],
    },
    "cB13": {
        "parts": [
            P("RMB 150.2 billion in net profit for FY2025", "招商银行", "net_profit", "150.2", "billion"),
            P(
                "growing 1.21% year-over-year",
                "招商银行",
                "netprofit_yoy",
                "1.21",
                "%",
                signed=True,
                pbasis="net_profit_parent_yoy",
            ),
        ]
    },
    "cB24": {
        "parts": [
            N("中国石油…净利润均超过千亿元", "中国石油 is not in the offline data", "net_profit"),
            N("中国人寿…", "中国人寿 is not in the offline data", "net_profit"),
            P("招商银行…净利润…超过千亿元", "招商银行", "net_profit", "1000", "亿", "gt"),
            N("中国移动…", "中国移动 is not in the offline data", "net_profit"),
            N("中国海油…", "中国海油 is not in the offline data", "net_profit"),
            P("中国平安净利润…超过千亿元", "中国平安", "net_profit", "1000", "亿", "gt"),
        ]
    },
    "cA04": {
        "parts": [
            P("比亚迪涨1.57%", "比亚迪", "pct", "1.57", "%", date=D0930, signed=True),
            P("成交额17.96亿元", "比亚迪", "amount", "17.96", "亿", date=D0930),
        ],
        "context": ["人气排名38位"],
    },
    "cA11": {
        "parts": [
            P("RMB 803.97 billion in revenue", "比亚迪", "revenue", "803.97", "billion"),
            P("RMB 32.62 billion in net profit", "比亚迪", "net_profit", "32.62", "billion"),
        ]
    },
    "cE05": {
        "parts": [
            P("A net margin of 4.1%", "比亚迪", "net_margin", "4.1", "%"),
            N("was down from 5.2% a year ago", PRIOR_MARGIN, "net_margin"),
        ]
    },
    "cA22": {
        "parts": [
            P("扣非归母净利润507.24亿元", "紫金矿业", "profit_dedt", "507.24", "亿"),
            N("同比增长60.05%", DEDT_YOY, "dedt_yoy"),
        ],
        "context": ["业绩再创历史新高"],
    },
    "cE09": {
        "parts": [
            P("紫金矿业517.77亿元的净利润", "紫金矿业", "net_profit", "517.77", "亿"),
            P("宁德时代（722.01亿元）", "宁德时代", "net_profit", "722.01", "亿"),
            R("仅次于…宁德时代", "lt", ("紫金矿业", "net_profit"), ("宁德时代", "net_profit")),
        ],
        "context": ["暂列第二位"],
    },
    "cE10": {
        "parts": [P("“日赚近1.4亿元”", "紫金矿业", "daily_net_profit", "1.4", "亿", "near")],
        "context": ["不仅刷新了公司自身的历史纪录"],
    },
    "cA02": {
        "parts": [
            P("报收于80.1元", "美的集团", "close", "80.1", "元", date=D0930),
            P("下跌1.91%", "美的集团", "pct", "-1.91", "%", date=D0930, signed=True),
            N("换手率0.58%", TURNOVER, "turnover_rate"),
            P("成交量40.17万手", "美的集团", "volume", "40.17", "万手", date=D0930),
            P("成交额32.32亿元", "美的集团", "amount", "32.32", "亿", date=D0930),
        ]
    },
    "cA16": {
        "parts": [P("2025年毛利率26.4%", "美的集团", "gross_margin", "26.4", "%")],
        "context": ["同比基本持平"],
    },
    "cE02": {
        "parts": [
            P("归母净利润439.45亿元", "美的集团", "net_profit", "439.45", "亿", named="net_profit_parent"),
            P("同比增长14.03%", "美的集团", "netprofit_yoy", "14.03", "%", signed=True, named="net_profit_parent_yoy"),
            P("相当于每天净赚1.2亿元", "美的集团", "daily_net_profit", "1.2", "亿"),
            P("净利率达到了9.7%", "美的集团", "net_margin", "9.7", "%", pbasis="net_margin:net_profit_parent:revenue"),
        ],
        "note": (
            "The dropped leading clause gives revenue as 营业收入 4564.52亿; "
            "the served revenue is 营业总收入 4585.02亿."
        ),
    },
    "cC02": {
        "parts": [
            P(
                "the average ROE of Kweichow Moutai reached 34.36%",
                "贵州茅台",
                "roe",
                "34.36",
                "%",
                named="roe_weighted",
            )
        ]
    },
    "cC05": {"parts": [P("市值约1.7万亿元", "贵州茅台", "total_mv", "1.7", "万亿", "approx", date=D0422)]},
    "cE08": {
        "parts": [P("茅台一天赚4.7个亿", "贵州茅台", "daily_net_profit", "4.7", "亿")],
        "note": "The article's context ranks companies by 2025 revenue; 4.7亿 a day matches revenue / 365, not profit.",
    },
    "cB12": {
        "parts": [
            P("revenue was 86.24 billion", "长江电力", "revenue", "86.24", "billion"),
            P("an increase of 2.07%", "长江电力", "revenue_yoy", "2.07", "%", signed=True),
            P("the previous year's 84.49 billion", "长江电力", "prior_revenue", "84.49", "billion"),
        ]
    },
    "cB20": {"parts": [P("加权平均净资产收益率15.90%", "长江电力", "roe", "15.90", "%", named="roe_weighted")]},
    "cB26": {
        "parts": [
            P("营业总收入862.42亿元", "长江电力", "revenue", "862.42", "亿", named="revenue_total"),
            P("同比增长2.07%", "长江电力", "revenue_yoy", "2.07", "%", signed=True),
        ]
    },
}


# ---------------------------------------------------------------- output


def category(row: dict[str, Any], parts: list[dict[str, Any]], rel: set[str]) -> str:
    if row["lang"] != "zh":
        return "english_or_traditional"
    if row["candidate_id"] in rel or any(p.get("q") in LOOSE for p in parts):
        return "loose_or_relational"
    metrics = {p.get("m") for p in parts if "rel" not in p}
    if metrics & VALUATION:
        return "market_value"
    if metrics & (BAR | {"turnover_rate", "fund_size"}):
        return "daily_price"
    if metrics <= RATIOS | {"dedt_yoy", None}:
        return "annual_ratios"
    return "annual_figures"


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return float(Decimal(repr(value)).quantize(Decimal("0.000001"))) if abs(value) < 1e6 else round(value, 2)
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def label(row: dict[str, Any], rel: set[str]) -> dict[str, Any]:
    spec = PARTS[row["candidate_id"]]
    parts = spec["parts"]
    aware = [evaluate_part(p, definition_aware=True) for p in parts]
    blind = [evaluate_part(p, definition_aware=False) for p in parts]
    prim = [r for i, p in enumerate(parts) if (r := primary_row(i, p))]
    summary = sorted({r["offline_vs_primary"] for r in prim}) or ["no_annual_part"]
    out = dict(row)
    out["category"] = category(row, parts, rel)
    out["expected_verdict"] = verdict([p["holds"] for p in aware])
    out["expected_verdict_definition_blind"] = verdict([p["holds"] for p in blind])
    out["parts"] = [{k: _round(v) for k, v in p.items()} for p in aware]
    changed = [i for i, (a, b) in enumerate(zip(aware, blind, strict=True)) if a["holds"] != b["holds"]]
    if changed:
        out["definition_blind_parts"] = {str(i): blind[i]["holds"] for i in changed}
    out["context_not_checked"] = spec.get("context", [])
    out["primary"] = [{k: _round(v) for k, v in r.items()} for r in prim]
    out["offline_vs_primary"] = "/".join(summary)
    if spec.get("note"):
        out["label_note"] = spec["note"]
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from evaluation.real_claims_web_v3.select_claims_web_v3 import REL

    rows = [json.loads(line) for line in COLLECTED.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [r["candidate_id"] for r in rows if r["candidate_id"] not in PARTS]
    if missing:
        raise SystemExit(f"no parts for {missing}")
    labelled = [label(r, REL) for r in rows]
    with OUT.open("w", encoding="utf-8") as fh:
        for r in labelled:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for r in labelled:
        counts[r["expected_verdict"]] = counts.get(r["expected_verdict"], 0) + 1
    print(f"{len(labelled)} claims -> {OUT.name}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
