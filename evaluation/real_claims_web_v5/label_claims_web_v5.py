"""Label the blind fifth real-claims set (``collected_v5.jsonl``) from the data files only.

Copied from ``evaluation/real_claims_web_v4/label_claims_web_v4.py`` (its data reader, tolerance rules, definition
rule, primary-source comparison and verdict aggregation are unchanged), with this set's additions written in
``README.md`` before labelling:

- "双位数增长 / double-digit growth": 10 <= growth < 100 (%), qualifier ``double_digit``;
- intraday highs and lows ("盘中触及") are outside the bar scope and null;
- the unit "million" (1e6); "万手" volume compared in shares.

Every value a part is compared with comes from:

- ``data/structured_data.json`` (v1 snapshot),
- ``data/snapshot/structured_data_ext.json`` (2026-09-30 extension),
- ``data/snapshot/structured_data_fix.json`` (corrections layer: replacement values and field definitions),
- ``data/snapshot/corrections/sources.json`` (annual-report figures, used for the ``primary`` columns and to see where
  two bases coincide).

No checker code is imported or read. A person read each claim once and wrote its *parts* below.

    python -m evaluation.real_claims_web_v5.label_claims_web_v5          # writes claims_web_v5.jsonl
"""

from __future__ import annotations

import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
COLLECTED = HERE / "collected_v5.jsonl"
OUTPUT = HERE / "claims_web_v5.jsonl"

UNITS = {
    "元": 1.0,
    "万": 1e4,
    "亿": 1e8,
    "万亿": 1e12,
    "million": 1e6,  # new in this set (a unit, not a rule)
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


# --------------------------------------------------------------------------------------------- data (as v4)


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
                    {
                        k: row.get(k)
                        for k in ("trade_date", "open", "high", "low", "close", "pct_change_1d", "volume", "amount")
                    }
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


# --------------------------------------------------------------------------------------------- rules (as v4)


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
    if qual == "double_digit":  # new in this set: "双位数增长" = 10 <= growth < 100
        return 10 - 1e-9 <= a < 100, f"双位数: 10 <= {a:.6g} < 100"
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
    if kind in ("ratio", "diff", "multiple_gap"):
        a = _value(data, part["a"])
        b = _value(data, part["b"])
        if a is None or b is None:
            out.update(holds=None, reason="a compared value is not in the offline data")
            return out
        if kind == "ratio":
            actual = a / b
        elif kind == "diff":
            actual = a - b
        else:  # new in this set: "差 / 相差 N 倍" = larger / smaller - 1
            actual = max(a, b) / min(a, b) - 1
        holds, rule = judge(actual, part["v"], part["u"], part.get("q", "exact"), part.get("signed", False))
        out.update(holds=holds, actual=round(actual, 6), rule=rule)
        return out
    if kind in ("sum", "mean", "share", "sum_cmp"):  # new in this set
        items = [_value(data, ref) for ref in part["items"]]
        if any(v is None for v in items):
            out.update(holds=None, reason="a named company is not in the offline data")
            return out
        total = sum(items)
        if kind == "sum_cmp":
            a = _value(data, part["a"])
            if a is None:
                out.update(holds=None, reason="a compared value is not in the offline data")
                return out
            holds = {">": a > total, "<": a < total}[part["op"]]
            out.update(holds=holds, actual=[round(a, 6), round(total, 6)], rule=f"{a:.6g} {part['op']} {total:.6g}")
            return out
        if kind == "sum":
            actual = total
        elif kind == "mean":
            actual = total / len(items)
        else:
            a = _value(data, part["a"])
            if a is None:
                out.update(holds=None, reason="a compared value is not in the offline data")
                return out
            actual = a / total * 100
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
    """Compare the offline value and the claim with the annual report's figure on the claim's basis (as v2)."""
    if part.get("kind", "value") != "value" or ("null" in part and "m" not in part):
        return None
    sym, metric = part.get("s"), part.get("m")
    if metric not in (
        "revenue",
        "net_profit",
        "eps",
        "roe",
        "profit_dedt",
        "netprofit_yoy",
        "revenue_yoy",
        "prior_net_profit",
        "net_profit_change",
    ):
        return None
    served_basis = data.defs.get(sym, {}).get(BASIS_FIELD.get(metric, ""), None)
    basis = part.get("pbasis") or part.get("basis") or served_basis
    if metric in ("prior_net_profit", "net_profit_change"):
        fid = {"total": "net_profit_total"}.get(basis or "", "net_profit_parent")
        cur, prior = data.primary_value(sym, fid), data.primary_value(sym, fid, prior=True)
        primary = None if cur is None or prior is None else (prior if metric == "prior_net_profit" else cur - prior)
    elif metric in ("netprofit_yoy", "revenue_yoy"):
        cur_id = {
            "netprofit_yoy": {"归母": "net_profit_parent", "total": "net_profit_total"},
            "revenue_yoy": {None: "revenue"},
        }[metric]
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
        res["claim_vs_primary"] = judge(
            primary, part["v"], part["u"], part.get("q", "exact"), part.get("signed", False)
        )[0]
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
SSE, CHINEXT, HS300 = "000001.SH", "399006.SZ", "000300.SH"
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


PP = "a change in percentage points needs the prior-year value, which is not in the offline data"
DEDT_YOY = "扣非 YoY is not in the offline data"
PER_DAY = "a per-day figure is not one of the derivations the rules list"
USD = "USD amounts are not in the offline data"
SZCI = "深证成指 is not in the offline data"
BSE50 = "北证50 is not in the offline data"
INTRADAY = "an intraday high is outside the set's bar scope (new in this set)"


def _not_in_data(name: str) -> str:
    return f"{name} is not in the offline data"


# id -> (candidate_id, spec); the candidate id guards against a changed selection
PARTS: dict[str, tuple[str, dict[str, Any]]] = {
    "w5_01": (
        "c091",
        {
            "category": "english_or_traditional",
            "parts": [
                V(SSE, "close", "3842.19", "pt", date=D0930),
                G(SSE, "pct", "0.31", date=D0930),
            ],
        },
    ),
    "w5_02": (
        "c036",
        {
            "category": "annual_figures",
            "parts": [
                G(CMB, "revenue_yoy", "0.01"),
                G(CMB, "netprofit_yoy", "1.2", basis="归母"),
            ],
        },
    ),
    "w5_03": (
        "c014",
        {
            "category": "daily_price",
            "parts": [V(CATL, "close", "291.11", "元", date=D0930)],
        },
    ),
    "w5_04": (
        "c078",
        {
            "category": "english_or_traditional",
            "parts": [
                G(MOUTAI, "netprofit_yoy", "-4.5"),
                V(MOUTAI, "net_profit", "82.32", "bn"),
                N("(US$12.07 billion)", USD),
                G(MOUTAI, "revenue_yoy", "-1.2"),
                V(MOUTAI, "revenue", "168.84", "bn"),
            ],
            "context": ["both well below market expectations and short of the company's own guidance"],
        },
    ),
    "w5_05": (
        "c017",
        {
            "category": "english_or_traditional",
            "parts": [
                V(CATL, "eps", "16.14", "元"),
                N(
                    "compared with 11.58 yuan in the previous year",
                    "the 2024 EPS is not in the offline data and not one of the listed derivations",
                ),
            ],
        },
    ),
    "w5_06": (
        "c031",
        {
            "category": "english_or_traditional",
            "parts": [
                V(ZIJIN, "net_profit", "51.8", "bn", basis="归母"),
                N("($7.52 billion)", USD),
                G(ZIJIN, "netprofit_yoy", "62", basis="归母"),
            ],
            "context": ["setting a new historical high since the company's founding"],
        },
    ),
    "w5_07": (
        "c008",
        {
            "category": "annual_ratios",
            "parts": [
                V(CATL, "net_profit", "722", "亿"),
                V(CATL, "net_margin", "18", "%", "gt"),
            ],
            "primary_note": "the report's 销售净利率 18.12% is total net profit (incl. minority interests) "
            "/ 营业收入; the rule's net margin uses the served 归母净利润 (17.04%)",
        },
    ),
    "w5_08": (
        "c074",
        {
            "category": "annual_ratios",
            "parts": [
                V(MOUTAI, "roe", "30", "%", "about"),
                N("较上年下降约4个百分点", PP),
            ],
            "context": ["依然处于优秀水平"],
        },
    ),
    "w5_09": (
        "c040",
        {
            "category": "annual_figures",
            "parts": [
                V(CMB, "revenue", "3375.32", "亿"),
                V(CMB, "net_profit", "1501.81", "亿", basis="归母"),
            ],
            "context": ["稳坐“股份行一哥”宝座 (a ranking)"],
        },
    ),
    "w5_10": (
        "c068",
        {
            "category": "annual_ratios",
            "parts": [
                N(
                    "2021 年 14.22%、2022 年 14.86%、2023 年 14.12%、2024 年 15.45%",
                    "ROE for 2021-2024 is not in the offline data",
                ),
                V(CYPC, "roe", "15.59", "%", basis="weighted"),
            ],
            "primary_note": "15.59% is the period-end ROE the data serves (归母净利润 / 期末归母净资产); "
            "the report's weighted ROE is 15.90%",
        },
    ),
    "w5_11": (
        "c006",
        {
            "category": "annual_ratios",
            "parts": [
                V(CATL, "gross_margin", "26.27", "%"),
                N("较上年增加1.83个百分点", PP),
                V(CATL, "net_margin", "18.12", "%"),
                N("较上年增加3.2个百分点", PP),
            ],
            "primary_note": "18.12% is total net profit (incl. minority interests) / 营业收入 in the report; "
            "the rule's net margin uses the served 归母净利润 (17.04%)",
        },
    ),
    "w5_12": (
        "c028",
        {
            "category": "annual_figures",
            "parts": [
                V(MIDEA, "revenue", "4585", "亿", basis="营业总收入"),
                V(MIDEA, "net_profit", "439.5", "亿", basis="归母"),
                G(MIDEA, "revenue_yoy", "10", q="double_digit", text="收入……实现了双位数增长"),
                G(MIDEA, "netprofit_yoy", "10", q="double_digit", text="利润……实现了双位数增长"),
            ],
            "context": ["业绩历史新高"],
        },
    ),
    "w5_13": (
        "c021",
        {
            "category": "annual_figures",
            "parts": [
                V(BYD, "revenue", "8040", "亿"),
                N("毛利润1426.60亿元", "gross profit is not in the offline data"),
                V(BYD, "gross_margin", "17.74", "%"),
                N("汽车相关业务毛利率20.49%", "a segment margin is not in the offline data"),
                N("电子相关业务毛利率6.29%", "a segment margin is not in the offline data"),
            ],
        },
    ),
    "w5_14": (
        "c065",
        {
            "category": "annual_figures",
            "parts": [
                V(CYPC, "revenue", "862.42", "亿"),
                G(CYPC, "revenue_yoy", "2.07"),
                V(CYPC, "net_profit", "345.03", "亿", basis="归母"),
                G(CYPC, "netprofit_yoy", "6.18", basis="归母"),
                V(CYPC, "profit_dedt", "334.46", "亿"),
                N("扣非同比增长2.89%", DEDT_YOY),
            ],
        },
    ),
    "w5_15": (
        "c063",
        {
            "category": "market_value",
            "parts": [
                V(CITIC, "total_mv", "4213.61", "亿", date=D0930),
                N("超越第二名1074亿元", _not_in_data("国泰海通")),
            ],
            "context": ["稳居榜首 (a ranking)"],
        },
    ),
    "w5_16": (
        "c046",
        {
            "category": "market_value",
            "parts": [
                V(CMB, "total_mv", "1.04", "万亿", date=D0930),
                V(CMB, "total_mv", "1", "万亿", "ge", date=D0930, text="站稳万亿市值"),
            ],
            "context": ["居第五名 (a ranking)", "股份行里唯一一家 (a comparison with banks not in the data)"],
        },
    ),
    "w5_17": (
        "c077",
        {
            "category": "english_or_traditional",
            "parts": [
                V(MOUTAI, "revenue", "168.838", "bn", "about"),
                G(MOUTAI, "revenue_yoy", "-1.21", q="about"),
            ],
        },
    ),
    "w5_18": (
        "c033",
        {
            "category": "annual_figures",
            "parts": [
                V(SMIC, "revenue", "673.23", "亿", "about"),
                G(SMIC, "revenue_yoy", "16.5"),
                V(SMIC, "net_profit", "50.41", "亿", "about", basis="归母"),
                G(SMIC, "netprofit_yoy", "36.3", basis="归母"),
                V(SMIC, "gross_margin", "21.6", "%"),
                V(SMIC, "net_margin", "10.7", "%"),
            ],
            "primary_note": "the report's 净利率 10.7% is total net profit (72.09亿) / 营业收入; the rule's net "
            "margin uses the served 归母净利润 (7.49%); revenue, 归母净利润 and their growth agree with the report",
        },
    ),
    "w5_19": (
        "c100",
        {
            "category": "annual_figures",
            "parts": [
                V(MOUTAI, "revenue", "1720.54", "亿", basis="营业总收入"),
                G(
                    MOUTAI,
                    "revenue_yoy",
                    "-1.20",
                    note="the growth of 营业总收入; revenue_yoy has no definition field in the rules (v4), "
                    "so it is compared with the served growth of 营业收入",
                ),
                V(MOUTAI, "revenue", "1688.38", "亿", basis="营业收入"),
                G(MOUTAI, "revenue_yoy", "-1.21"),
                V(MOUTAI, "net_profit", "823.20", "亿", basis="归母"),
                G(MOUTAI, "netprofit_yoy", "-4.53", basis="归母"),
                V(MOUTAI, "eps", "65.66", "元", basis="basic"),
            ],
        },
    ),
    "w5_20": (
        "c067",
        {
            "category": "annual_figures",
            "parts": [
                V(CYPC, "revenue", "862.42", "亿"),
                G(CYPC, "revenue_yoy", "2.07"),
                V(CYPC, "net_profit", "345.03", "亿", basis="归母"),
                G(CYPC, "netprofit_yoy", "6.17", basis="归母"),
            ],
        },
    ),
    "w5_21": (
        "c084",
        {
            "category": "daily_price",
            "parts": [
                V(SSE, "close", "3842.19", "pt", date=D0930),
                G(SSE, "pct", "0.31", date=D0930),
                N("深证成指（399001）收于12887.62点", SZCI),
                N("深证成指跌0.11%", SZCI),
                V(CHINEXT, "close", "3135.28", "pt", date=D0930),
                G(CHINEXT, "pct", "-0.23", date=D0930),
                N("北证50收于1039.61点", BSE50),
                N("北证50涨0.70%", BSE50),
            ],
        },
    ),
    "w5_22": (
        "c032",
        {
            "category": "daily_price",
            "parts": [
                V(ZIJIN, "close", "29.79", "元", date=D0930),
                G(ZIJIN, "pct", "1.12", date=D0930),
                N("换手率0.48%", "a turnover rate is not in the offline data"),
                V(ZIJIN, "volume", "98.22", "万手", date=D0930),
                V(ZIJIN, "amount", "29.22", "亿", date=D0930),
            ],
        },
    ),
    "w5_23": (
        "c101",
        {
            "category": "loose_or_relational",
            "parts": [
                {
                    "kind": "sum",
                    "text": "比亚迪、吉利、上汽、长城、长安五家头部车企，2025年净利润加在一起是735亿元",
                    "items": [
                        {"s": BYD, "m": "net_profit"},
                        {"s": "吉利汽车", "m": "net_profit"},
                        {"s": "上汽集团", "m": "net_profit"},
                        {"s": "长城汽车", "m": "net_profit"},
                        {"s": "长安汽车", "m": "net_profit"},
                    ],
                    "v": "735",
                    "u": "亿",
                },
                N(
                    "勉强和宁德时代打个平手",
                    "a relation to the five-company sum, which includes companies not in the offline data",
                ),
            ],
        },
    ),
    "w5_24": (
        "c047",
        {
            "category": "daily_price",
            "parts": [
                N("盘中触及8.30元/股", INTRADAY),
                V(ICBC, "close", "8.28", "元", date=D0930),
                G(ICBC, "pct", "1.47", date=D0930),
            ],
        },
    ),
    "w5_25": (
        "c061",
        {
            "category": "annual_figures",
            "parts": [
                V(CITIC, "revenue", "748.54", "亿"),
                G(CITIC, "revenue_yoy", "28.79"),
                V(CITIC, "net_profit", "300.76", "亿", basis="归母"),
                G(CITIC, "netprofit_yoy", "38.58", basis="归母"),
                V(CITIC, "net_profit", "300", "亿", "ge", basis="归母", text="首次突破300亿元大关"),
            ],
            "context": ["首次 (突破300亿元 for the first time)"],
        },
    ),
    "w5_26": (
        "c088",
        {
            "category": "daily_price",
            "parts": [
                G(SSE, "pct", "0.31", date=D0930),
                N("深证成指跌0.11%", SZCI),
                G(CHINEXT, "pct", "-0.23", date=D0930),
            ],
        },
    ),
    "w5_27": (
        "c034",
        {
            "category": "annual_figures",
            "parts": [
                N("年营收93.3亿美元", USD),
                G(SMIC, "revenue_yoy", "16.2"),
                N("归母净利润6.85亿美元", USD),
                V(SMIC, "net_profit", "50.4", "亿", basis="归母"),
                G(SMIC, "netprofit_yoy", "24.9", basis="归母"),
            ],
            "primary_note": "the sentence's growth rates go with the US-dollar figures; the A-share report "
            "gives 营业收入 +16.49% and 归母净利润 +36.29%",
        },
    ),
    "w5_28": (
        "c058",
        {
            "category": "annual_figures",
            "parts": [
                V(CITIC, "revenue", "748.54", "亿"),
                G(CITIC, "revenue_yoy", "28.79"),
                V(CITIC, "net_profit", "300.76", "亿", basis="归母"),
                G(CITIC, "netprofit_yoy", "38.58", basis="归母"),
            ],
        },
    ),
    "w5_29": (
        "c025",
        {
            "category": "english_or_traditional",
            "parts": [G(BYD, "netprofit_yoy", "-19")],
        },
    ),
    "w5_30": (
        "c027",
        {
            "category": "loose_or_relational",
            "parts": [
                N("2025全年分红324亿", "dividends are not in the offline data"),
                N("回购116亿", "buybacks are not in the offline data"),
                V(MIDEA, "net_profit", "439", "亿"),
                N(
                    "（分红加回购）约当全年净利",
                    "a relation to dividends and buybacks, which are not in the offline data",
                ),
                N("分红率近100%", "the payout ratio is not in the offline data"),
            ],
            "context": ["A股罕见"],
        },
    ),
    "w5_31": (
        "c055",
        {
            "category": "annual_figures",
            "parts": [
                V(PAB, "net_profit", "426.33", "亿"),
                G(PAB, "netprofit_yoy", "-4.2"),
            ],
        },
    ),
    "w5_32": (
        "c064",
        {
            "category": "annual_figures",
            "parts": [
                V(CITIC, "revenue", "748.54", "亿"),
                G(CITIC, "revenue_yoy", "28.79"),
                V(CITIC, "net_profit", "300.76", "亿", basis="归母"),
                G(CITIC, "netprofit_yoy", "38.58", basis="归母"),
            ],
            "context": ["主要财务指标创历史新高"],
        },
    ),
    "w5_33": (
        "c089",
        {
            "category": "daily_price",
            "parts": [
                V(SSE, "close", "3842", "pt", date=D0930),
                N("7月低点3721点", "a close on another date is outside the scope"),
                N("仅有121点空间", "a gap to a close outside the scope"),
            ],
            "context": ["3750点属于强支撑区间 (an opinion)"],
        },
    ),
    "w5_34": (
        "c043",
        {
            "category": "annual_ratios",
            "parts": [
                V(CMB, "roe", "13", "%", "gt", basis="weighted"),
                N("江苏银行的ROE也在13%以上", _not_in_data("江苏银行")),
                N("宁波银行的ROE也在13%以上", _not_in_data("宁波银行")),
            ],
        },
    ),
    "w5_35": (
        "c054",
        {
            "category": "annual_figures",
            "parts": [
                V(PAB, "revenue", "1314.42", "亿"),
                G(PAB, "revenue_yoy", "-10.4"),
            ],
        },
    ),
    "w5_36": (
        "c052",
        {
            "category": "english_or_traditional",
            "parts": [
                V(ICBC, "revenue", "801.4", "bn"),
                N("($116 billion)", USD),
                G(ICBC, "revenue_yoy", "1.9"),
                G(ICBC, "netprofit_yoy", "1"),
                V(ICBC, "net_profit", "370.77", "bn"),
            ],
            "primary_note": "801.4bn / +1.9% is the IFRS operating income of the HKEX results; the A-share "
            "report's 营业收入 is 8382.70亿 (+2.0%), which the offline data serves; 370.77bn is total net profit "
            "(净利润), the served 归母净利润 is 368.56bn",
        },
    ),
    "w5_37": (
        "c083",
        {
            "category": "english_or_traditional",
            "parts": [
                V(PINGAN, "profit_dedt", "143773", "million"),
                N("up 22.5% year on year", DEDT_YOY),
            ],
        },
    ),
    "w5_38": (
        "c051",
        {
            "category": "annual_figures",
            "parts": [
                G(ICBC, "revenue_yoy", "2", q="ge", text="营业收入增幅均超2%（含2%）"),
                G(ICBC, "revenue_yoy", "2.00"),
                V(ICBC, "revenue", "8382.70", "亿"),
                N("农业银行：增幅超2%、同比增长2.10%、营业收入7253.06亿元", _not_in_data("农业银行")),
                N("交通银行：增幅超2%、同比增长2.02%、营业收入2650.71亿元", _not_in_data("交通银行")),
            ],
        },
    ),
    "w5_39": (
        "c044",
        {
            "category": "annual_figures",
            "parts": [
                V(ICBC, "revenue", "8382.70", "亿"),
                G(ICBC, "revenue_yoy", "2"),
            ],
            "context": ["蝉联榜首 (a ranking)"],
        },
    ),
    "w5_40": (
        "c024",
        {
            "category": "english_or_traditional",
            "parts": [
                G(BYD, "revenue_yoy", "3.5"),
                V(BYD, "revenue", "804", "bn"),
                N("($116 billion)", USD),
                N("eclipsing rival Tesla's full year revenue of $94.8 billion", _not_in_data("Tesla") + "; USD amount"),
            ],
            "context": ["another record"],
        },
    ),
    "w5_41": (
        "c066",
        {
            "category": "annual_figures",
            "parts": [
                V(CYPC, "revenue", "862.42", "亿", basis="营业总收入"),
                G(CYPC, "revenue_yoy", "2.07"),
                V(CYPC, "net_profit", "345.03", "亿", basis="归母"),
                G(CYPC, "netprofit_yoy", "6.17", basis="归母"),
            ],
        },
    ),
    "w5_42": (
        "c082",
        {
            "category": "annual_figures",
            "parts": [
                N("集团总资产达13.898万亿元", "total assets are not in the offline data"),
                V(PINGAN, "net_profit", "1347.78", "亿", basis="归母"),
                N("日均盈利超3.69亿元", PER_DAY),
            ],
            "context": ["盈利规模稳居全球保险机构前列 (a ranking)"],
        },
    ),
    "w5_43": (
        "c081",
        {
            "category": "annual_figures",
            "parts": [
                N("归母营运利润达到1344.15亿元", "营运利润 is not in the offline data"),
                N("营运利润同比增长10.3%", "营运利润 is not in the offline data"),
                V(PINGAN, "net_profit", "1347.78", "亿", basis="归母"),
                G(PINGAN, "netprofit_yoy", "6.5", basis="归母"),
                V(PINGAN, "revenue", "10505.06", "亿"),
                G(PINGAN, "revenue_yoy", "2.1"),
            ],
        },
    ),
    "w5_44": (
        "c030",
        {
            "category": "annual_figures",
            "parts": [
                V(ZIJIN, "revenue", "3490.79", "亿"),
                V(ZIJIN, "net_profit", "517.77", "亿", basis="归母"),
                G(ZIJIN, "netprofit_yoy", "61.55", basis="归母"),
            ],
        },
    ),
    "w5_45": (
        "c056",
        {
            "category": "annual_figures",
            "parts": [
                V(PAB, "revenue", "1314.42", "亿"),
                G(PAB, "revenue_yoy", "-10.4"),
                V(PAB, "net_profit", "426.33", "亿", basis="归母"),
                G(PAB, "netprofit_yoy", "-4.2", basis="归母"),
            ],
            "context": ["经营压力仍未缓解"],
        },
    ),
    "w5_46": (
        "c086",
        {
            "category": "daily_price",
            "parts": [
                G(HS300, "pct", "0.29", date=D0930),
                G(CHINEXT, "pct", "-0.23", date=D0930),
                {
                    "kind": "cmp",
                    "text": "沪深300（+0.29%）明显强于创业板指（-0.23%）",
                    "a": {"s": HS300, "m": "pct", "date": D0930},
                    "b": {"s": CHINEXT, "m": "pct", "date": D0930},
                    "op": ">",
                },
            ],
            "context": ["价值风格占优；蓝筹权重股主导市场"],
        },
    ),
    "w5_47": (
        "c097",
        {
            "category": "daily_price",
            "parts": [
                N("上证50指数下跌0.05%", _not_in_data("上证50")),
                G(HS300, "pct", "0.66", date=D0422),
            ],
        },
    ),
    "w5_48": (
        "c029",
        {
            "category": "annual_figures",
            "parts": [
                V(ZIJIN, "revenue", "3490.8", "亿"),
                G(ZIJIN, "revenue_yoy", "14.96"),
                V(ZIJIN, "net_profit", "517.77", "亿", basis="归母"),
                G(ZIJIN, "netprofit_yoy", "61.55", basis="归母"),
                V(ZIJIN, "profit_dedt", "507.24", "亿"),
                N("扣非同比增长60.05%", DEDT_YOY),
            ],
        },
    ),
    "w5_49": (
        "c041",
        {
            "category": "annual_figures",
            "parts": [
                V(PAB, "revenue", "1314.42", "亿"),
                V(PAB, "net_profit", "426.33", "亿"),
                G(PAB, "netprofit_yoy", "-4.21"),
            ],
        },
    ),
    "w5_50": (
        "c026",
        {
            "category": "annual_figures",
            "parts": [
                V(MIDEA, "net_profit", "439.5", "亿", basis="归母"),
                G(MIDEA, "netprofit_yoy", "14.0", basis="归母"),
            ],
        },
    ),
}


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


def label_all() -> list[dict[str, Any]]:
    data = OfflineData()
    rows = [json.loads(line) for line in COLLECTED.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sorted(r["id"] for r in rows) != sorted(PARTS):
        raise SystemExit("collected_v5.jsonl ids differ from the labelled ids")
    out = []
    for row in rows:
        cid, spec = PARTS[row["id"]]
        if row["candidate_id"] != cid:
            raise SystemExit(f"{row['id']}: candidate {row['candidate_id']} differs from the labelled {cid}")
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


def main() -> None:
    rows = label_all()
    OUTPUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(rows)} claims")
    print("expected:", dict(Counter(r["expected_verdict"] for r in rows)))
    print("definition-blind:", dict(Counter(r["expected_verdict_definition_blind"] for r in rows)))
    print("categories:", dict(Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()
