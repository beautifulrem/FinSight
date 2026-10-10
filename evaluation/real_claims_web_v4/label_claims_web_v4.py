"""Label the blind fourth real-claims set (``collected_v4.jsonl``) from the data files only.

Modelled on ``evaluation/real_claims_web_v2/label_claims_web_v2.py`` (its data reader, tolerance rules, definition
rule, primary-source comparison and verdict aggregation are copied here unchanged), with this set's additions written in
``README.md`` before labelling:

- "差 / 相差 N 倍" as a multiple of the difference (larger / smaller - 1), kind ``multiple_gap``;
- a sum of several companies as the compared side (kind ``sum_cmp`` / ``sum``), an average across named companies
  (kind ``mean``), a share of a total (kind ``share``); each null if a named company is not in the data;
- 扣非 vs 归母 relations and gaps from the two served fields (``profit_dedt``, ``net_profit``) with the v2 kinds;
- H-share / ADR / GDR / HK-dollar figures null, USD and EUR amounts null.

Every value a part is compared with comes from:

- ``data/structured_data.json`` (v1 snapshot),
- ``data/snapshot/structured_data_ext.json`` (2026-09-30 extension),
- ``data/snapshot/structured_data_fix.json`` (corrections layer: replacement values and field definitions),
- ``data/snapshot/corrections/sources.json`` (annual-report figures, used for the ``primary`` columns and to see where
  two bases coincide).

No checker code is imported or read. A person read each claim once and wrote its *parts* below.

    python -m evaluation.real_claims_web_v4.label_claims_web_v4          # writes claims_web_v4.jsonl
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
COLLECTED = HERE / "collected_v4.jsonl"
OUTPUT = HERE / "claims_web_v4.jsonl"

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


# --------------------------------------------------------------------------------------------- data (as v2)


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


# --------------------------------------------------------------------------------------------- rules (as v2)


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
SSE, CHINEXT = "000001.SH", "399006.SZ"
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

# id -> (candidate_id, spec); the candidate id guards against a changed selection
PARTS: dict[str, tuple[str, dict[str, Any]]] = {
    "w3_01": ("c091", {
        "category": "annual_figures",
        "parts": [
            V(WLY, "revenue", "405.29", "亿"),
            G(WLY, "revenue_yoy", "-54.55"),
            V(WLY, "net_profit", "89.54", "亿", basis="归母"),
            G(WLY, "netprofit_yoy", "-71.89", basis="归母"),
        ],
    }),
    "w3_02": ("c036", {
        "category": "annual_figures",
        "parts": [
            V(SMIC, "revenue", "673.23", "亿"),
            G(SMIC, "revenue_yoy", "16.5"),
            V(SMIC, "net_profit", "50.41", "亿", basis="归母"),
            G(SMIC, "netprofit_yoy", "36.3", basis="归母"),
        ],
    }),
    "w3_03": ("c014", {
        "category": "annual_figures",
        "parts": [V(CATL, "net_profit", "722.01", "亿")],
        "context": ["位居创业板2025年度“盈利王” (a ranking)"],
    }),
    "w3_04": ("c078", {
        "category": "annual_ratios",
        "parts": [
            V(CYPC, "gross_margin", "61.67", "%"),
            N("同比提升2.54pct", PP),
            V(CYPC, "net_margin", "39.99", "%"),
            N("同比提升1.53pct", PP),
        ],
    }),
    "w3_05": ("c017", {
        "category": "loose_or_relational",
        "parts": [
            V(CATL, "revenue", "4000", "亿"),
            V(CATL, "net_profit", "722", "亿"),
        ],
    }),
    "w3_06": ("c031", {
        "category": "annual_figures",
        "parts": [V(BYD, "revenue", "8040", "亿")],
        "context": ["唯一进入全球前十的中国车企 (a global ranking)"],
    }),
    "w3_07": ("c008", {
        "category": "annual_ratios",
        "parts": [
            N("动力电池系统毛利率23.8%", "a segment margin is not in the offline data"),
            V(CATL, "net_margin", "17", "%", "about"),
        ],
    }),
    "w3_08": ("c074", {
        "category": "annual_figures",
        "parts": [
            V(CITIC, "revenue", "748.30", "亿"),
            G(CITIC, "revenue_yoy", "28.75"),
            V(CITIC, "net_profit", "300.51", "亿", basis="归母"),
            G(CITIC, "netprofit_yoy", "38.46", basis="归母"),
        ],
        "primary_note": "the sentence quotes the preliminary 业绩快报 (748.30亿, 300.51亿, +38.46%); the annual report "
        "gives 748.54亿 营业收入 and 300.76亿 归母净利润 (+38.58%)",
    }),
    "w3_09": ("c040", {
        "category": "english_or_traditional",
        "parts": [
            V(SMIC, "net_profit", "5.041", "bn", basis="归母"),
            G(SMIC, "netprofit_yoy", "36.3", basis="归母"),
        ],
    }),
    "w3_10": ("c068", {
        "category": "annual_figures",
        "parts": [
            V(PAB, "revenue", "1314.42", "亿"),
            G(PAB, "revenue_yoy", "-10.4"),
            V(PAB, "net_profit", "426.33", "亿", basis="归母"),
            G(PAB, "netprofit_yoy", "-4.2", basis="归母"),
        ],
    }),
    "w3_11": ("c010", {
        "category": "loose_or_relational",
        "parts": [
            {
                "kind": "multiple_gap",
                "text": "营收规模上相差近一倍",
                "a": {"s": BYD, "m": "revenue"},
                "b": {"s": CATL, "m": "revenue"},
                "v": "1",
                "u": "倍",
                "q": "near",
            },
            V(BYD, "revenue", "8039.65", "亿"),
            V(CATL, "revenue", "4237.02", "亿"),
        ],
    }),
    "w3_12": ("c104", {
        "category": "daily_price",
        "parts": [
            G(SSE, "pct", "0.27", date=D0930),
            N("深成指跌0.06%", "深证成指 is not in the offline data"),
            G(CHINEXT, "pct", "-0.17", date=D0930),
        ],
        "primary_note": "price: the sentence sits in the page's 11:31 午评 block although it says 截至收盘; no "
        "independent record in the data files",
    }),
    "w3_13": ("c028", {
        "category": "annual_figures",
        "parts": [G(BYD, "netprofit_yoy", "-19")],
    }),
    "w3_14": ("c021", {
        "category": "annual_figures",
        "parts": [
            V(BYD, "revenue", "8039.65", "亿", basis="营业总收入"),
            G(BYD, "revenue_yoy", "3.46"),
            V(BYD, "net_profit", "326.19", "亿", basis="归母"),
            G(BYD, "netprofit_yoy", "-18.97", basis="归母"),
            V(BYD, "profit_dedt", "294.46", "亿"),
            N("扣非同比下降20.38%", DEDT_YOY),
        ],
    }),
    "w3_15": ("c065", {
        "category": "annual_figures",
        "parts": [
            V(CMB, "revenue", "3375.32", "亿"),
            G(CMB, "revenue_yoy", "0.01"),
            V(CMB, "net_profit", "1500", "亿", "ge", basis="归母"),
            V(CMB, "net_profit", "1501.81", "亿", basis="归母"),
            G(CMB, "netprofit_yoy", "1.21", basis="归母"),
            N("相当于日赚4.11亿元", PER_DAY),
        ],
        "context": ["首次 (突破1500亿元 for the first time)"],
    }),
    "w3_16": ("c063", {
        "category": "annual_figures",
        "parts": [
            V(CMB, "revenue", "3375.32", "亿"),
            G(CMB, "revenue_yoy", "0.01"),
            V(CMB, "net_profit", "1501.81", "亿", basis="归母"),
            G(CMB, "netprofit_yoy", "1.21", basis="归母"),
        ],
    }),
    "w3_17": ("c046", {
        "category": "annual_figures",
        "parts": [
            V(ZIJIN, "revenue", "3490.79", "亿"),
            G(ZIJIN, "revenue_yoy", "14.96"),
        ],
    }),
    "w3_18": ("c077", {
        "category": "annual_figures",
        "parts": [
            V(CYPC, "revenue", "862.42", "亿"),
            G(CYPC, "revenue_yoy", "2.07"),
            V(CYPC, "net_profit", "345.03", "亿", basis="归母"),
            G(CYPC, "netprofit_yoy", "6.17", basis="归母"),
        ],
    }),
    "w3_19": ("c033", {
        "category": "english_or_traditional",
        "parts": [
            G(BYD, "netprofit_yoy", "-19", basis="归母"),
            V(BYD, "net_profit", "32.6", "bn", basis="归母"),
            N("(€4.4bn)", "EUR amounts are not in the offline data"),
            V(BYD, "prior_net_profit", "40.3", "bn", basis="归母"),
        ],
        "context": ["a filing to the Hong Kong Stock Exchange (the figures are the company's, not the H-share line's)"],
    }),
    "w3_20": ("c100", {
        "category": "annual_figures",
        "parts": [
            V(MIDEA, "revenue", "4585.02", "亿", "about"),
            V(MIDEA, "prior_revenue", "4090.84", "亿"),
            G(MIDEA, "revenue_yoy", "12.1"),
        ],
    }),
    "w3_21": ("c067", {
        "category": "annual_figures",
        "parts": [
            V(PAB, "net_profit", "426.33", "亿"),
            G(PAB, "netprofit_yoy", "-4.2"),
        ],
    }),
    "w3_22": ("c084", {
        "category": "annual_figures",
        "parts": [V(MOUTAI, "revenue", "1688.38", "亿")],
        "context": ["继续稳居榜首 (a provincial ranking)"],
    }),
    "w3_23": ("c101", {
        "category": "annual_figures",
        "parts": [
            V(MIDEA, "net_profit", "439.45", "亿", "about", basis="归母"),
            G(MIDEA, "netprofit_yoy", "14.0", basis="归母"),
        ],
    }),
    "w3_24": ("c047", {
        "category": "annual_ratios",
        "parts": [V(ZIJIN, "eps", "1.95", "元", basis="basic")],
    }),
    "w3_25": ("c061", {
        "category": "english_or_traditional",
        "parts": [
            V(CMB, "revenue", "337.5", "bn"),
            G(CMB, "revenue_yoy", "0.01"),
        ],
    }),
    "w3_26": ("c088", {
        "category": "english_or_traditional",
        "parts": [
            V(MOUTAI, "revenue", "168.8", "bn"),
            N("(USD $23.3 billion)", USD),
            G(MOUTAI, "revenue_yoy", "-1.2"),
            G(MOUTAI, "netprofit_yoy", "-4.5"),
            V(MOUTAI, "net_profit", "82.3", "bn"),
            N("($12.07 billion)", USD),
        ],
    }),
    "w3_27": ("c058", {
        "category": "annual_figures",
        "parts": [
            V(ICBC, "revenue", "8013.95", "亿"),
            G(ICBC, "revenue_yoy", "1.9"),
            V(ICBC, "net_profit", "3685.62", "亿", basis="归母"),
            G(ICBC, "netprofit_yoy", "0.7", basis="归母"),
        ],
        "primary_note": "8013.95亿 / +1.9% is the IFRS operating income of the HKEX results; the A-share report's "
        "营业收入 is 8382.70亿 (+2.0%), which the offline data serves",
    }),
    "w3_28": ("c055", {
        "category": "annual_figures",
        "parts": [
            V(ICBC, "revenue", "8013.95", "亿"),
            G(ICBC, "revenue_yoy", "1.9"),
            V(ICBC, "net_profit", "3707.66", "亿", basis="归母", pbasis="total"),
            G(ICBC, "netprofit_yoy", "1.0", basis="归母", pbasis="total"),
        ],
        "context": ["cites the H-share code 01398.HK (company figures, compared as such)"],
        "primary_note": "the sentence's figures are the IFRS results announced in Hong Kong: operating income 8013.95亿 "
        "(+1.9%) and net profit 3707.66亿 (+1.0%), which is total net profit although the sentence says 归属于母公司股东; "
        "the A-share report's 营业收入 is 8382.70亿 and 归母净利润 3685.62亿",
    }),
    "w3_29": ("c064", {
        "category": "annual_ratios",
        "parts": [
            N("平均总资产收益率（ROAA）1.19%", "ROA is not in the offline data"),
            V(CMB, "roe", "13.44", "%", basis="weighted"),
            N("同比下降0.09个百分点", PP),
            N("同比下降1.05个百分点", PP),
        ],
    }),
    "w3_30": ("c089", {
        "category": "english_or_traditional",
        "parts": [
            V(MOUTAI, "revenue", "168.838", "bn"),
            G(MOUTAI, "revenue_yoy", "-1.21"),
            V(MOUTAI, "net_profit", "82.32", "bn", basis="归母"),
            G(MOUTAI, "netprofit_yoy", "-4.53", basis="归母"),
        ],
        "context": ["first-ever decline in both annual revenue and net profit since its listing"],
    }),
    "w3_31": ("c085", {
        "category": "annual_figures",
        "parts": [
            V(MOUTAI, "net_profit", "823.2", "亿"),
            N("万亿市值对应PE仅12倍", "a valuation with no stated date"),
        ],
        "context": ["万亿市值 (no stated date)"],
    }),
    "w3_32": ("c094", {
        "category": "english_or_traditional",
        "parts": [
            V(WLY, "revenue", "40.53", "bn"),
            G(WLY, "revenue_yoy", "-55", q="near"),
            G(WLY, "netprofit_yoy", "-72"),
            V(WLY, "net_profit", "8.95", "bn"),
        ],
    }),
    "w3_33": ("c043", {
        "category": "annual_ratios",
        "parts": [
            V(ZIJIN, "roe", "33.04", "%", basis="weighted"),
            N("同比提升7.15个百分点", PP),
        ],
        "context": ["稳居全球矿业行业的顶级水平"],
    }),
    "w3_34": ("c039", {
        "category": "english_or_traditional",
        "parts": [
            N("sales revenue totaled $9.327 billion", USD),
            G(SMIC, "revenue_yoy", "16.2"),
        ],
    }),
    "w3_35": ("c098", {
        "category": "loose_or_relational",
        "parts": [
            V(MIDEA, "net_profit", "439.45", "亿"),
            N("每天净赚超过1.2亿元", PER_DAY),
        ],
    }),
    "w3_36": ("c054", {
        "category": "loose_or_relational",
        "parts": [
            N("净利润超过 187 亿美元", USD),
            V(PINGAN, "net_profit", "1267.5", "亿", "about"),
        ],
        "context": ["位列利润榜第 9 位 (Fortune China 500 ranking)"],
    }),
    "w3_37": ("c099", {
        "category": "annual_ratios",
        "parts": [V(MIDEA, "net_margin", "9.63", "%")],
        "context": ["创历史新高"],
    }),
    "w3_38": ("c052", {
        "category": "annual_figures",
        "parts": [V(PINGAN, "revenue", "10505.06", "亿")],
    }),
    "w3_39": ("c049", {
        "category": "annual_figures",
        "parts": [
            V(PINGAN, "revenue", "11403.24", "亿"),
            N("营运利润1344.15亿元", "营运利润 is not in the offline data"),
            N("营运利润同比增长10.30%", "营运利润 is not in the offline data"),
            V(PINGAN, "net_profit", "1347.78", "亿", basis="归母"),
            G(PINGAN, "netprofit_yoy", "6.45", basis="归母"),
        ],
        "context": ["cites the H-share code 02318 (company figures, compared as such)"],
        "primary_note": "11403.24亿 is the HKEX (IFRS) total income; the A-share report's 营业收入 is 10505.06亿, "
        "which the offline data serves",
    }),
    "w3_40": ("c051", {
        "category": "annual_figures",
        "parts": [
            V(PINGAN, "net_profit", "1348", "亿", basis="归母"),
            G(PINGAN, "netprofit_yoy", "6.45", basis="归母"),
            V(PINGAN, "profit_dedt", "1438", "亿"),
            N("扣非上升22.5%", DEDT_YOY),
        ],
    }),
    "w3_41": ("c044", {
        "category": "annual_figures",
        "parts": [
            V(ZIJIN, "net_profit", "518", "亿", basis="归母"),
            G(ZIJIN, "netprofit_yoy", "62", basis="归母"),
            V(ZIJIN, "profit_dedt", "507", "亿"),
            N("扣非同比增长60%", DEDT_YOY),
        ],
    }),
    "w3_42": ("c037", {
        "category": "annual_ratios",
        "parts": [
            N("产能利用率上升至93.5%", "capacity utilisation is not in the offline data"),
            V(SMIC, "gross_margin", "22", "%"),
        ],
    }),
    "w3_43": ("c073", {
        "category": "annual_ratios",
        "parts": [
            V(CITIC, "roe", "10.58", "%", basis="weighted"),
            N("较上年的8.09%", "the 2024 ROE is not in the offline data"),
            N("提升2.49个百分点", PP),
        ],
        "primary_note": "the sentence quotes the preliminary 业绩快报 weighted ROE 10.58%; the annual report gives "
        "10.59%",
    }),
    "w3_44": ("c066", {
        "category": "annual_figures",
        "parts": [
            V(PAB, "revenue", "1314.42", "亿"),
            G(PAB, "revenue_yoy", "-10.4"),
        ],
    }),
    "w3_45": ("c056", {
        "category": "annual_figures",
        "parts": [
            V(ICBC, "revenue", "8013.95", "亿"),
            G(ICBC, "revenue_yoy", "1.9"),
        ],
        "primary_note": "8013.95亿 / +1.9% is the IFRS operating income; the A-share report's 营业收入 is 8382.70亿 "
        "(+2.0%), which the offline data serves",
    }),
    "w3_46": ("c092", {
        "category": "annual_figures",
        "parts": [
            G(WLY, "revenue_yoy", "-54.55"),
            G(WLY, "netprofit_yoy", "-71.89", basis="归母"),
        ],
    }),
    "w3_47": ("c102", {
        "category": "loose_or_relational",
        "parts": [
            N("中国人寿业绩增长超40%", "中国人寿 is not in the offline data"),
            G(CATL, "netprofit_yoy", "40", q="gt"),
            G(ZIJIN, "netprofit_yoy", "40", q="gt"),
        ],
        "context": ["TOP20 (a ranking)"],
    }),
    "w3_48": ("c090", {
        "category": "annual_figures",
        "parts": [V(WLY, "net_profit", "89.54", "亿", basis="归母")],
    }),
    "w3_49": ("c103", {
        "category": "daily_price",
        "parts": [
            G(SSE, "pct", "0.31", date=D0930),
            V(SSE, "close", "3842.19", "pt", date=D0930),
            N("深证成指跌0.11%", "深证成指 is not in the offline data"),
            G(CHINEXT, "pct", "-0.23", date=D0930),
        ],
    }),
    "w3_50": ("c069", {
        "category": "annual_figures",
        "parts": [
            V(PAB, "revenue", "1314.42", "亿"),
            G(PAB, "revenue_yoy", "-10.4"),
            V(PAB, "net_profit", "426.33", "亿", basis="归母"),
            G(PAB, "netprofit_yoy", "-4.2", basis="归母"),
        ],
    }),
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
        raise SystemExit("collected_v4.jsonl ids differ from the labelled ids")
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
