"""Label the blind second real-claims set (``collected_v2.jsonl``) from the data files only.

Every expected verdict is computed here from the shipped data files:

- ``data/structured_data.json`` (v1 snapshot),
- ``data/snapshot/structured_data_ext.json`` (2026-09-30 extension),
- ``data/snapshot/structured_data_fix.json`` (corrections layer: replacement values and field definitions),
- ``data/snapshot/corrections/sources.json`` (primary-source figures from the annual reports, used only for the
  ``offline_vs_primary`` / ``primary_vs_claim`` columns and to see where two bases coincide).

No checker code is imported or read. A person reads each claim once and writes its *parts* below (company, metric,
stated value as written, qualifier, named basis); the numbers the parts are compared with all come from the files.
The tolerance rules are the ones written in ``README.md`` before labelling.

    python -m evaluation.real_claims_web_v2.label_claims_web_v2          # writes claims_web_v2.jsonl
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
COLLECTED = HERE / "collected_v2.jsonl"
OUTPUT = HERE / "claims_web_v2.jsonl"

UNITS = {
    "元": 1.0,
    "万": 1e4,
    "亿": 1e8,
    "万亿": 1e12,
    "bn": 1e9,
    "%": 1.0,
    "pt": 1.0,
    "倍": 1.0,
    "x": 1.0,
    "股": 1.0,
    "手": 100.0,
    "万手": 1e6,
}

# metric -> the definitions key that states its basis
BASIS_FIELD = {
    "roe": "roe_basis",
    "eps": "eps_basis",
    "revenue": "revenue_basis",
    "netprofit_yoy": "netprofit_yoy_basis",
    "net_profit": "net_profit_basis",
    "prior_net_profit": "netprofit_yoy_basis",
    "net_profit_change": "netprofit_yoy_basis",
}
# fields whose basis is fixed by the field name rather than stated per company
FIXED_BASIS = {"pe_ttm": "ttm"}
# the v1 seed bars from tushare carry tushare's units: amount in 千元, volume in 手
SOURCE_SCALE = {"tushare": {"amount": 1e3, "volume": 100.0}}

# (metric, basis) -> the sources.json figure id that holds the primary value of that basis
PRIMARY_FIGURE = {
    ("revenue", "营业收入"): "revenue",
    ("revenue", "营业总收入"): "revenue_total",
    ("net_profit", "归母"): "net_profit_parent",
    ("net_profit", "total"): "net_profit_total",
    ("eps", "basic"): "eps_basic",
    ("eps", "diluted"): "eps_diluted",
    ("roe", "weighted"): "roe_weighted",
    ("profit_dedt", None): "profit_dedt",
}


# --------------------------------------------------------------------------------------------- data


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class OfflineData:
    """The values the shipped default serves: v1 base, the extension on top, the corrections layer on top."""

    def __init__(self) -> None:
        base = _load("data/structured_data.json")
        ext = _load("data/snapshot/structured_data_ext.json")
        fix = _load("data/snapshot/structured_data_fix.json")
        self.sources = _load("data/snapshot/corrections/sources.json")
        self.fund: dict[str, dict[str, Any]] = {}
        for layer in (base, ext, fix):
            for sym, row in layer.get("fundamental_sql", {}).items():
                self.fund[sym] = row  # each layer replaces the whole statement
        self.defs: dict[str, dict[str, str]] = dict(fix.get("definitions", {}))
        for sym, row in fix.get("fundamental_sql", {}).items():
            if row.get("definitions"):
                self.defs[sym] = row["definitions"]
        self.bars: dict[str, dict[str, dict[str, Any]]] = {}
        self.bar_source: dict[str, str] = {}
        for layer in (base, ext, fix):
            for sym, row in layer.get("market_api", {}).items():
                self.bar_source[sym] = row.get("source_name", "")
                hist = row.get("history") or [
                    {k: row.get(k) for k in ("trade_date", "open", "high", "low", "close", "pct_change_1d",
                                             "volume", "amount")}
                ]
                self.bars[sym] = {b["trade_date"]: dict(b) for b in hist}
        self.primary: dict[str, dict[str, dict[str, Any]]] = {}
        self.notes: dict[str, list[str]] = {}
        for comp in self.sources.get("companies", []):
            self.primary[comp["symbol"]] = {f["id"]: f for f in comp["figures"]}
            self.notes[comp["symbol"]] = comp.get("notes", [])
        for sym, comp in self.sources.get("extension_definitions", {}).items():
            self.primary[sym] = {f["id"]: f for f in comp["figures"]}
            self.notes[sym] = comp.get("notes", [])

    # ---- primary figures (annual reports)
    def primary_value(self, sym: str, fig_id: str, prior: bool = False) -> float | None:
        fig = self.primary.get(sym, {}).get(fig_id)
        if not fig:
            return None
        raw = fig.get("prior") if prior else fig.get("value")
        if raw in (None, ""):
            return None
        val = float(str(raw).replace(",", "").rstrip("%"))
        scale = {"元": 1.0, "千元": 1e3, "百万元": 1e6}.get(fig.get("unit", ""), 1.0)
        return val * scale

    def bases_coincide(self, sym: str, metric: str, a: str, b: str) -> bool:
        """True when the annual report shows the two bases give the same figure for this company."""
        if a == b:
            return True
        if metric == "revenue" and any("营业收入 = 营业总收入" in n for n in self.notes.get(sym, [])):
            return True
        if metric in ("netprofit_yoy", "prior_net_profit", "net_profit_change"):
            pairs = [("net_profit_parent", False), ("net_profit_parent", True)]
            other = [("net_profit_total", False), ("net_profit_total", True)]
            vals = [self.primary_value(sym, f, p) for f, p in pairs + other]
            return None not in vals and vals[0] == vals[2] and vals[1] == vals[3]
        fa = PRIMARY_FIGURE.get((metric, a))
        fb = PRIMARY_FIGURE.get((metric, b))
        if not fa or not fb:
            return False
        va, vb = self.primary_value(sym, fa), self.primary_value(sym, fb)
        return va is not None and va == vb

    # ---- served values
    def fundamental(self, sym: str, metric: str) -> float | None:
        row = self.fund.get(sym)
        if not row:
            return None
        if metric == "net_margin":
            if row.get("net_profit") and row.get("revenue"):
                return row["net_profit"] / row["revenue"] * 100
            return None
        if metric == "gross_margin":
            val = row.get("gross_margin", row.get("grossprofit_margin"))
            return None if val is None else float(val)
        if metric == "prior_net_profit":
            if row.get("net_profit") is None or row.get("netprofit_yoy") is None:
                return None
            return row["net_profit"] / (1 + row["netprofit_yoy"] / 100)
        if metric == "net_profit_change":
            prior = self.fundamental(sym, "prior_net_profit")
            return None if prior is None else row["net_profit"] - prior
        if metric == "prior_revenue":
            if row.get("revenue") is None or row.get("revenue_yoy") is None:
                return None
            return row["revenue"] / (1 + row["revenue_yoy"] / 100)
        val = row.get(metric)
        return None if val is None else float(val)

    def valuation_date(self, sym: str) -> str | None:
        row = self.fund.get(sym) or {}
        return row.get("valuation_date")

    def bar(self, sym: str, date: str) -> dict[str, Any] | None:
        return self.bars.get(sym, {}).get(date)

    def price(self, sym: str, date: str, metric: str) -> float | None:
        bars = self.bars.get(sym, {})
        bar = bars.get(date)
        if not bar:
            return None
        dates = sorted(bars)
        idx = dates.index(date)
        prev = bars[dates[idx - 1]]["close"] if idx > 0 else None
        if metric == "pct":
            if bar.get("pct_change_1d") is not None:
                return float(bar["pct_change_1d"])
            return None if prev is None else (bar["close"] / prev - 1) * 100
        if metric == "change":
            if prev is not None:
                return bar["close"] - prev
            if bar.get("pct_change_1d") is not None:
                return bar["close"] - bar["close"] / (1 + bar["pct_change_1d"] / 100)
            return None
        val = bar.get(metric)
        if val is None:
            return None
        return float(val) * SOURCE_SCALE.get(self.bar_source.get(sym, ""), {}).get(metric, 1.0)


# --------------------------------------------------------------------------------------------- rules


def _half_step(stated: str, unit: str) -> float:
    digits = stated.lstrip("+-")
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    return 0.5 * float(Decimal(1).scaleb(-decimals)) * UNITS[unit]


def judge(actual: float, stated: str, unit: str, qual: str, signed: bool) -> tuple[bool, str]:
    """Apply the README's tolerance rule. Returns (holds, rule text)."""
    claimed = float(stated) * UNITS[unit]
    if signed and claimed != 0 and actual != 0 and math.copysign(1, claimed) != math.copysign(1, actual):
        return False, "sign differs"
    a, c = (abs(actual), abs(claimed)) if signed else (actual, claimed)
    if qual == "exact":
        tol = max(_half_step(stated, unit), 0.01 * abs(a))
        return abs(a - c) <= tol + 1e-9, f"|{a:.6g} - {c:.6g}| <= {tol:.6g}"
    if qual == "about":
        tol = 0.05 * abs(c)
        return abs(a - c) <= tol + 1e-9, f"约: |{a:.6g} - {c:.6g}| <= 5% of claim"
    if qual == "near":
        return 0.9 * c - 1e-9 <= a <= c + 1e-9, f"近: 0.9*{c:.6g} <= {a:.6g} <= {c:.6g}"
    if qual == "ge":
        return a >= c - 1e-9, f"突破/站上: {a:.6g} >= {c:.6g}"
    if qual == "gt":
        return a > c, f"超/逾/多: {a:.6g} > {c:.6g}"
    if qual == "lt":
        return a < c, f"不到/低于: {a:.6g} < {c:.6g}"
    if qual == "gt_next":  # "N倍多", "N千多亿": N < a < N + one step of the stated digit
        step = 2 * _half_step(stated, unit)
        return c < a < c + step, f"N多: {c:.6g} < {a:.6g} < {c + step:.6g}"
    raise ValueError(qual)


# --------------------------------------------------------------------------------------------- parts


def eval_part(data: OfflineData, part: dict[str, Any]) -> dict[str, Any]:
    out = dict(part)
    if "null" in part:
        out.update(holds=None, reason=part["null"])
        return out
    kind = part.get("kind", "value")
    if kind == "cmp":  # relation between two served values: "A 高于 B"
        a = _value(data, part["a"])
        b = _value(data, part["b"])
        if a is None or b is None:
            out.update(holds=None, reason="a compared value is not in the offline data")
            return out
        holds = {">": a > b, "<": a < b}[part["op"]]
        out.update(holds=holds, actual=[round(a, 6), round(b, 6)], rule=f"{a:.6g} {part['op']} {b:.6g}")
        return out
    if kind in ("ratio", "diff"):
        a = _value(data, part["a"])
        b = _value(data, part["b"])
        if a is None or b is None:
            out.update(holds=None, reason="a compared value is not in the offline data")
            return out
        actual = a / b if kind == "ratio" else a - b
        holds, rule = judge(actual, part["v"], part["u"], part.get("q", "exact"), part.get("signed", False))
        out.update(holds=holds, actual=round(actual, 6), rule=rule)
        return out
    ref = {k: part[k] for k in ("s", "m", "date", "basis") if k in part}
    blocked = _definition_block(data, ref)
    if blocked:
        out.update(holds=None, reason=blocked)
        return out
    actual = _value(data, ref)
    if actual is None:
        where = part.get("date", "FY2025")
        out.update(holds=None, reason=f"{part['m']} for {part['s']} on {where} is not in the offline data")
        return out
    holds, rule = judge(actual, part["v"], part["u"], part.get("q", "exact"), part.get("signed", False))
    out.update(holds=holds, actual=round(actual, 6), rule=rule)
    return out


def _definition_block(data: OfflineData, ref: dict[str, Any]) -> str | None:
    basis = ref.get("basis")
    if not basis:
        return None
    if ref["m"] in FIXED_BASIS:
        served = FIXED_BASIS[ref["m"]]
        return None if basis == served else f"definition: claim names {basis}, the offline {ref['m']} is {served}"
    field = BASIS_FIELD.get(ref["m"])
    if not field:
        return None
    served = data.defs.get(ref["s"], {}).get(field)
    if served is None or data.bases_coincide(ref["s"], ref["m"], basis, served):
        return None
    return f"definition: claim names {basis}, the offline {ref['m']} is {served}"


def _value(data: OfflineData, ref: dict[str, Any]) -> float | None:
    date = ref.get("date", "FY2025")
    metric = ref["m"]
    if metric in ("close", "pct", "change", "amount", "open", "high", "low", "volume"):
        return data.price(ref["s"], date, metric)
    if metric in ("pe_ttm", "pb", "total_mv", "circ_mv"):
        if data.valuation_date(ref["s"]) != date:
            return None
        return data.fundamental(ref["s"], metric)
    return data.fundamental(ref["s"], metric)


def primary_check(data: OfflineData, part: dict[str, Any]) -> dict[str, Any] | None:
    """Compare the offline value and the claim with the annual report's figure on the claim's basis."""
    if part.get("kind", "value") != "value" or ("null" in part and "m" not in part):
        return None
    sym, metric = part.get("s"), part.get("m")
    if metric not in ("revenue", "net_profit", "eps", "roe", "profit_dedt", "netprofit_yoy", "revenue_yoy",
                      "prior_net_profit", "net_profit_change"):
        return None
    served_basis = data.defs.get(sym, {}).get(BASIS_FIELD.get(metric, ""), None)
    basis = part.get("pbasis") or part.get("basis") or served_basis
    if metric in ("prior_net_profit", "net_profit_change"):
        fid = {"total": "net_profit_total"}.get(basis or "", "net_profit_parent")
        cur, prior = data.primary_value(sym, fid), data.primary_value(sym, fid, prior=True)
        primary = None if cur is None or prior is None else (prior if metric == "prior_net_profit" else cur - prior)
    elif metric in ("netprofit_yoy", "revenue_yoy"):
        cur_id = {"netprofit_yoy": {"归母": "net_profit_parent", "total": "net_profit_total"},
                  "revenue_yoy": {None: "revenue"}}[metric]
        fid = cur_id.get(basis if metric == "netprofit_yoy" else None) or cur_id.get("归母")
        cur, prior = data.primary_value(sym, fid), data.primary_value(sym, fid, prior=True)
        primary = None if not cur or not prior else (cur / prior - 1) * 100
    else:
        fid = PRIMARY_FIGURE.get((metric, basis if metric != "profit_dedt" else None))
        primary = data.primary_value(sym, fid) if fid else None
    if primary is None:
        return {"basis": basis, "primary": None}
    served = data.fundamental(sym, metric)
    res: dict[str, Any] = {"basis": basis, "primary": round(primary, 6)}
    if served is not None:
        same_basis = basis == served_basis or data.bases_coincide(sym, metric, basis or "", served_basis or "")
        close = abs(served - primary) <= 0.001 * abs(primary) + 0.005
        if not same_basis:
            res["offline_vs_primary"] = "definition_differs" + (" (values within 0.1%)" if close else "")
        else:
            res["offline_vs_primary"] = "agree" if close else "value_differs"
        res["offline"] = round(served, 6)
    if "v" in part and "u" in part:
        res["claim_vs_primary"] = judge(primary, part["v"], part["u"], part.get("q", "exact"),
                                        part.get("signed", False))[0]
    return res


def verdict(parts: list[dict[str, Any]]) -> str:
    vals = [p["holds"] for p in parts]
    if vals and all(v is True for v in vals):
        return "supported"
    if any(v is True for v in vals):
        return "partially_supported"
    if any(v is False for v in vals):
        return "contradicted"
    return "unverifiable"


# --------------------------------------------------------------------------------------------- claims

CATL, BYD, MIDEA, ZIJIN, SMIC = "300750.SZ", "002594.SZ", "000333.SZ", "601899.SH", "688981.SH"
CMB, ICBC, PAB, CITIC, CYPC = "600036.SH", "601398.SH", "000001.SZ", "600030.SH", "600900.SH"
PINGAN, WLY, MOUTAI = "601318.SH", "000858.SZ", "600519.SH"
SSE, CHINEXT, GOLD_ETF, CSI500_ETF = "000001.SH", "399006.SZ", "518880.SH", "510500.SH"
D0930, D0422 = "2026-09-30", "2026-04-22"


def V(s: str, m: str, v: str, u: str, q: str = "exact", **kw: Any) -> dict[str, Any]:
    """One stated number: company, metric, the number as written, its unit, its qualifier."""
    part: dict[str, Any] = {"s": s, "m": m, "v": v, "u": u, "q": q}
    part.update(kw)
    return part


def G(s: str, m: str, v: str, **kw: Any) -> dict[str, Any]:
    """A signed growth rate or change in % (the sign is in ``v``)."""
    return V(s, m, v, "%", signed=True, **kw)


def N(text: str, reason: str) -> dict[str, Any]:
    """A stated number the offline data cannot decide."""
    return {"text": text, "null": reason}


PARTS: dict[str, dict[str, Any]] = {
    "v01": {"category": "loose_or_relational", "parts": [
        V(CATL, "net_profit", "722.01", "亿", basis="归母"),
        N("超过比亚迪、上汽…零跑的利润总和", "ten of the eleven carmakers are not in the offline data"),
    ]},
    "v02": {"category": "daily_price", "parts": [
        G(CATL, "pct", "1.5", date=D0930), V(CATL, "close", "291.11", "元", date=D0930),
    ]},
    "v03": {"category": "market_value", "parts": [
        V(CATL, "pe_ttm", "15.56", "倍", date=D0930, basis="动态"),
        V(CATL, "pb", "3.55", "倍", date=D0930),
    ]},
    "v04": {"category": "annual_figures", "parts": [
        V(CATL, "profit_dedt", "645.08", "亿"),
        N("扣非净利润同比增长43.37%", "扣非 YoY is not in the offline data"),
    ]},
    "v05": {"category": "loose_or_relational", "parts": [
        {"kind": "ratio", "a": {"s": BYD, "m": "revenue"}, "b": {"s": CATL, "m": "revenue"},
         "v": "1.9", "u": "倍", "q": "about"},
    ]},
    "v06": {"category": "annual_figures", "parts": [
        V(BYD, "net_profit", "326.19", "亿", basis="归母"),
        V(BYD, "prior_net_profit", "402.54", "亿", basis="归母"),
        G(BYD, "netprofit_yoy", "-19", basis="归母"),
    ]},
    "v07": {"category": "annual_ratios", "parts": [
        N("毛利率从19%", "the 2024 gross margin is not in the offline data"),
        V(BYD, "gross_margin", "18", "%"),
    ], "context": ["近五年来毛利率的最低点"]},
    "v08": {"category": "market_value", "parts": [
        V(BYD, "total_mv", "7595.37", "亿", date=D0930),
    ], "context": ["稳居榜首"]},
    "v09": {"category": "annual_figures", "parts": [
        V(MIDEA, "revenue", "4547", "亿"),
        G(MIDEA, "revenue_yoy", "11.6"),
        V(MIDEA, "net_profit", "442.9", "亿", basis="归母"),
        G(MIDEA, "netprofit_yoy", "14.9", basis="归母"),
    ], "context": ["增速连续四年跑赢营收增速 (a four-year relation)"]},
    "v10": {"category": "annual_ratios", "parts": [
        V(MIDEA, "roe", "17.3", "%", "about", basis="weighted"),
        V(MIDEA, "roe", "17", "%", "gt", basis="weighted"),
    ], "context": ["连续多年 (earlier years)"]},
    "v11": {"category": "english_or_traditional", "parts": [
        G(MIDEA, "netprofit_yoy", "14.0", basis="归母"),
        V(MIDEA, "net_profit", "43.9", "bn", basis="归母"),
        N("basic EPS advanced 6.6%", "EPS growth is not in the offline data"),
        V(MIDEA, "eps", "5.80", "元", basis="basic"),
    ]},
    "v12": {"category": "annual_figures", "parts": [
        V(ZIJIN, "revenue", "3491", "亿"),
        G(ZIJIN, "revenue_yoy", "15"),
        V(ZIJIN, "net_profit", "518", "亿", basis="归母"),
        G(ZIJIN, "netprofit_yoy", "62", basis="归母"),
    ]},
    "v13": {"category": "daily_price", "parts": [
        V(ZIJIN, "close", "35.53", "元", date=D0422), G(ZIJIN, "pct", "-0.36", date=D0422),
    ]},
    "v14": {"category": "english_or_traditional", "parts": [
        N("revenue of US$49.7 billion", "USD amounts are not in the offline data"),
        G(ZIJIN, "revenue_yoy", "15"),
        N("net profit … of US$7.4 billion", "USD amounts are not in the offline data"),
        G(ZIJIN, "netprofit_yoy", "62", basis="归母"),
    ]},
    "v15": {"category": "annual_ratios", "parts": [
        V(SMIC, "eps", "0.63", "元", basis="basic"),
        V(SMIC, "gross_margin", "21.0", "%"),
        N("同比上升3.0个百分点", "the 2024 gross margin is not in the offline data"),
    ]},
    "v16": {"category": "annual_ratios", "parts": [
        V(SMIC, "gross_margin", "21.62", "%"),
        N("上年同期18.59%", "the 2024 gross margin is not in the offline data"),
        N("提高3.03个百分点", "the 2024 gross margin is not in the offline data"),
    ]},
    "v17": {"category": "daily_price", "parts": [
        V(SMIC, "close", "107.87", "元", date=D0422), G(SMIC, "pct", "0.38", date=D0422),
        N("换手率1.88%", "换手率 is not in the offline data"),
        V(SMIC, "volume", "37.52", "万手", date=D0422), V(SMIC, "amount", "40.23", "亿", date=D0422),
    ]},
    "v18": {"category": "annual_figures", "parts": [
        V(CMB, "revenue", "3375", "亿"), V(CMB, "net_profit", "1502", "亿", basis="归母"),
    ]},
    "v19": {"category": "annual_ratios", "parts": [
        V(CMB, "roe", "13.44", "%", basis="weighted"),
        N("同比下降1.05个百分点", "the 2024 ROE is not in the offline data"),
    ]},
    "v20": {"category": "market_value", "parts": [
        V(CMB, "total_mv", "1.04", "万亿", date=D0930),
    ], "context": ["居第五名", "股份行里唯一一家站稳万亿市值 (other banks)"]},
    "v21": {"category": "annual_figures", "parts": [
        V(ICBC, "net_profit", "3685.62", "亿", basis="归母"),
        G(ICBC, "netprofit_yoy", "0.7", basis="归母"),
        N("日均盈利超10亿元", "a per-day figure is not one of the derivations the rules list"),
    ]},
    "v22": {"category": "english_or_traditional", "parts": [
        V(ICBC, "net_profit", "3707.66", "亿", pbasis="total"),
        V(ICBC, "net_profit_change", "38.20", "亿", signed=True, pbasis="total"),
        G(ICBC, "netprofit_yoy", "1.0", pbasis="total"),
    ], "primary_note": "agree on the claim's basis (IFRS total 淨利潤 370,766百万, +1.04%); the offline net_profit is "
       "归母 (3685.62亿), 0.6% from the claim, so the 1% rule passes it"},
    "v23": {"category": "english_or_traditional", "parts": [
        G(ICBC, "netprofit_yoy", "0.7", pbasis="归母"),
        V(ICBC, "net_profit", "368.6", "bn"),
    ], "context": ["the world's largest lender by assets"]},
    "v24": {"category": "market_value", "parts": [
        V(ICBC, "total_mv", "2.95", "万亿", date=D0930),
    ], "context": ["稳居榜首"]},
    "v25": {"category": "annual_ratios", "parts": [
        V(PAB, "net_margin", "32.43", "%"),
        N("同比增6.9%", "the 2024 net margin is not in the offline data"),
    ]},
    "v26": {"category": "annual_figures", "parts": [
        V(PAB, "net_profit", "426.33", "亿"),
        G(PAB, "netprofit_yoy", "-4.2"),
        N("环比下降达68.12%", "a sequential (环比) change is not in the offline data"),
    ]},
    "v27": {"category": "daily_price", "parts": [
        V(PAB, "close", "11.57", "元", date=D0930), G(PAB, "pct", "1.94", date=D0930),
        V(PAB, "total_mv", "2245.26", "亿", date=D0930),
    ]},
    "v28": {"category": "loose_or_relational", "parts": [
        V(CITIC, "net_profit", "300.76", "亿", basis="归母"),
        N("国泰海通（278.09亿元）", "国泰海通 is not in the offline data"),
        N("高出22.67亿元", "国泰海通 is not in the offline data"),
    ]},
    "v29": {"category": "annual_ratios", "parts": [
        V(CITIC, "roe", "9.4", "%", "about"),
        N("归母权益3199.30亿元", "equity is not in the offline data"),
    ]},
    "v30": {"category": "loose_or_relational", "parts": [
        N("摩根大通全年营收1856亿美元", "摩根大通 is not in the offline data"),
        N("净利润570亿美元", "摩根大通 is not in the offline data"),
        N("一家的利润=13.6家中信证券", "a ratio to 摩根大通's USD profit is not in the offline data"),
    ]},
    "v31": {"category": "daily_price", "parts": [
        V(CITIC, "close", "25.86", "元", date=D0930), G(CITIC, "pct", "0.15", date=D0930),
        V(CITIC, "total_mv", "4040.44", "亿", date=D0930),
    ]},
    "v32": {"category": "annual_figures", "parts": [
        V(CYPC, "net_profit", "335.03", "亿", basis="归母"),
        V(CYPC, "net_profit_change", "20.07", "亿", signed=True, basis="归母"),
    ]},
    "v33": {"category": "english_or_traditional", "parts": [
        V(CYPC, "net_profit", "34.503", "bn", basis="归母"),
        V(CYPC, "net_profit_change", "2.007", "bn", signed=True, basis="归母"),
        G(CYPC, "netprofit_yoy", "6.17", basis="归母"),
    ]},
    "v34": {"category": "annual_figures", "parts": [
        V(CYPC, "revenue", "858.82", "亿", basis="营业总收入", pbasis="营业收入"),
        G(CYPC, "revenue_yoy", "1.65"),
    ], "primary_note": "the claim quotes the preliminary 业绩快报 (858.82亿, +1.65%); the annual report and "
       "the offline data agree on 862.42亿, +2.07% (营业收入 = 营业总收入 for 长江电力)"},
    "v35": {"category": "annual_figures", "parts": [
        V(PINGAN, "profit_dedt", "1437.73", "亿"),
        N("同比增长22.5%", "扣非 YoY is not in the offline data"),
    ]},
    "v36": {"category": "annual_figures", "parts": [
        V(PINGAN, "net_profit", "1347", "亿"),
        G(PINGAN, "netprofit_yoy", "6.45", basis="归母"),
    ]},
    "v37": {"category": "annual_figures", "parts": [
        V(PINGAN, "net_profit", "1348", "亿", basis="归母"),
        V(PINGAN, "profit_dedt", "1438", "亿"),
        N("同比增长22.5%", "扣非 YoY is not in the offline data"),
    ]},
    "v38": {"category": "daily_price", "parts": [
        V(PINGAN, "close", "58.03", "元", date=D0422), G(PINGAN, "pct", "0", date=D0422),
        V(PINGAN, "high", "58.60", "元", date=D0422), V(PINGAN, "low", "52.23", "元", date=D0422),
    ], "primary_note": "price: the offline bar is the corrections layer's (Sina kline, cross-checked with Tencent), "
       "low 57.65; the published low 52.23 differs (unresolved against an exchange record)"},
    "v39": {"category": "annual_figures", "parts": [
        V(WLY, "revenue", "405.29", "亿"), G(WLY, "revenue_yoy", "-54.55"),
        V(WLY, "net_profit", "89.54", "亿", basis="归母"), G(WLY, "netprofit_yoy", "-71.89", basis="归母"),
    ]},
    "v40": {"category": "annual_ratios", "parts": [
        V(WLY, "eps", "2.3068", "元", basis="basic"),
        N("同比下降71.89%", "EPS growth is not in the offline data"),
    ]},
    "v41": {"category": "annual_ratios", "parts": [
        V(WLY, "roe", "7.47", "%"),
        N("减少16.43个百分点", "the 2024 ROE is not in the offline data"),
    ], "context": ["在已披露的同业公司中排名第11"]},
    "v42": {"category": "daily_price", "parts": [
        V(MOUTAI, "close", "1409.50", "元", date=D0422), G(MOUTAI, "pct", "-0.18", date=D0422),
    ]},
    "v43": {"category": "daily_price", "parts": [
        V(MOUTAI, "amount", "37.94", "亿", date=D0422),
    ]},
    "v44": {"category": "english_or_traditional", "parts": [
        V(MOUTAI, "net_profit", "823.2", "亿", basis="归母"), G(MOUTAI, "netprofit_yoy", "-4.53", basis="归母"),
    ]},
    "v45": {"category": "annual_ratios", "parts": [
        V(MOUTAI, "net_margin", "48.8", "%"),
        N("同比降1.7pct", "the 2024 net margin is not in the offline data"),
    ]},
    "v46": {"category": "daily_price", "parts": [
        G(CHINEXT, "pct", "1.7", q="gt", date=D0422),
        V(CHINEXT, "close", "3700", "pt", "ge", date=D0422),
    ], "context": ["再创逾十年新高"]},
    "v47": {"category": "daily_price", "parts": [
        V(SSE, "close", "3842.19", "pt", date=D0930), G(SSE, "pct", "0.31", date=D0930),
    ]},
    "v48": {"category": "daily_price", "parts": [
        V(GOLD_ETF, "close", "8.339", "元", date=D0930), G(GOLD_ETF, "pct", "0.97", date=D0930),
    ], "primary_note": "price: the published 8.339 / +0.97% differs from the offline bar 8.633 / +1.34%; the first "
       "set's w58 (财中社) states 8.633 / +1.34% (unresolved against an exchange record)"},
    "v49": {"category": "daily_price", "parts": [
        V(CSI500_ETF, "close", "7.472", "元", date=D0930), G(CSI500_ETF, "pct", "-0.05", date=D0930),
        V(CSI500_ETF, "amount", "16.7", "亿", date=D0930),
    ]},
    "v50": {"category": "daily_price", "parts": [
        V(ZIJIN, "close", "29.79", "元", date=D0930), G(ZIJIN, "pct", "1.12", date=D0930),
    ]},
}


def label_all() -> list[dict[str, Any]]:
    data = OfflineData()
    rows = [json.loads(line) for line in COLLECTED.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for row in rows:
        spec = PARTS[row["id"]]
        parts = [eval_part(data, p) for p in spec["parts"]]
        blind_parts = [eval_part(data, {k: v for k, v in p.items() if k != "basis"}) for p in spec["parts"]]
        primaries = [primary_check(data, p) for p in spec["parts"]]
        rec = dict(row)
        rec.update(
            category=spec["category"],
            expected_verdict=verdict(parts),
            expected_verdict_definition_blind=verdict(blind_parts),
            parts=parts,
            context_not_checked=spec.get("context", []),
            primary=[p for p in primaries if p],
            offline_vs_primary=_offline_vs_primary(primaries, spec),
        )
        out.append(rec)
    return out


def _offline_vs_primary(primaries: list[dict[str, Any] | None], spec: dict[str, Any]) -> str:
    if spec.get("primary_note"):
        return spec["primary_note"]
    flags = [p.get("offline_vs_primary") for p in primaries if p and p.get("offline_vs_primary")]
    if not flags:
        if spec["category"] in ("daily_price", "market_value"):
            return "price / valuation: no independent record in the data files"
        return "no annual-report figure for these parts in sources.json"
    if all(f == "agree" for f in flags):
        return "agree"
    return "; ".join(sorted(set(flags)))


def main() -> None:
    rows = label_all()
    OUTPUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["expected_verdict"]] = counts.get(r["expected_verdict"], 0) + 1
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(rows)} claims, {counts}")


if __name__ == "__main__":
    main()
