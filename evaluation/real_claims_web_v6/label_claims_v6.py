"""Label the sixth blind claim set: collected_v6.jsonl -> claims_v6.jsonl.

Every value a part is compared with is computed here from the FinSight-r2 data files (read-only):
data/structured_data.json, data/snapshot/structured_data_fix.json, data/snapshot/structured_data_ext.json and
data/snapshot/corrections/sources.json (primary-source comparison and the definition rule's same-figure exception only).
Standard library only; nothing from query_intelligence is imported; the claim checker is never run.

The per-claim parts (what each sentence asserts) are written by hand in claim_parts_v6.py (PARTS).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/Volumes/Remi/FinSight-r2")
BASE = REPO / "data/structured_data.json"
FIX = REPO / "data/snapshot/structured_data_fix.json"
EXT = REPO / "data/snapshot/structured_data_ext.json"
SOURCES = REPO / "data/snapshot/corrections/sources.json"

SYM = {
    "宁德时代": "300750.SZ", "比亚迪": "002594.SZ", "美的集团": "000333.SZ", "紫金矿业": "601899.SH",
    "中芯国际": "688981.SH", "招商银行": "600036.SH", "工商银行": "601398.SH", "平安银行": "000001.SZ",
    "中信证券": "600030.SH", "长江电力": "600900.SH", "贵州茅台": "600519.SH", "五粮液": "000858.SZ",
    "中国平安": "601318.SH", "上证指数": "000001.SH", "创业板指": "399006.SZ", "沪深300": "000300.SH",
    "中证500ETF": "510500.SH", "黄金ETF": "518880.SH", "科创50ETF": "588000.SH", "沪深300ETF": "510300.SH",
    "创业板ETF": "159915.SZ", "证券ETF": "512880.SH",
}
INDICES = {"000001.SH", "399006.SZ", "000300.SH"}
UNITS = {"元": 1.0, "亿": 1e8, "万亿": 1e12, "千亿": 1e11, "百亿": 1e10, "万": 1e4, "billion": 1e9, "trillion": 1e12,
         "million": 1e6, "%": 1.0, "倍": 1.0, "元/股": 1.0, "点": 1.0, "股": 1.0, "手": 100.0, "万手": 1e6,
         "万股": 1e4, "亿股": 1e8, "x": 1.0, "": 1.0}
SIGNED = {"revenue_yoy", "netprofit_yoy", "pct", "change", "prior_growth"}

# ----------------------------------------------------------------------------------------------- data layer


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


BASE_D, FIX_D, EXT_D, SRC_D = _load(BASE), _load(FIX), _load(EXT), _load(SOURCES)


def fundamentals(sym: str) -> dict | None:
    row: dict = {}
    for layer in (BASE_D.get("fundamental_sql", {}), EXT_D.get("fundamental_sql", {}), FIX_D.get("fundamental_sql", {})):
        if sym in layer:
            row = {**row, **{k: v for k, v in layer[sym].items() if k != "_snapshot"}}
            row["_layer"] = "fix" if layer is FIX_D.get("fundamental_sql") else (
                "ext" if layer is EXT_D.get("fundamental_sql") else "base")
    if not row:
        return None
    defs = row.get("definitions") or FIX_D.get("definitions", {}).get(sym) or {}
    row["definitions"] = defs
    return row


def bars(sym: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for layer in (BASE_D.get("market_api", {}), EXT_D.get("market_api", {}), FIX_D.get("market_api", {})):
        rec = layer.get(sym)
        if not rec:
            continue
        hist = rec.get("history") or [rec]
        for b in hist:
            out[b["trade_date"]] = {k: b.get(k) for k in ("open", "high", "low", "close", "pct_change_1d", "volume", "amount")}
        if rec.get("trade_date") and rec["trade_date"] not in out:
            out[rec["trade_date"]] = {k: rec.get(k) for k in ("open", "high", "low", "close", "pct_change_1d", "volume", "amount")}
    return out


BASIS_KEY = {"revenue": "revenue_basis", "revenue_yoy": "revenue_yoy_basis", "net_profit": "net_profit_basis",
             "netprofit_yoy": "netprofit_yoy_basis", "roe": "roe_basis", "eps": "eps_basis",
             "gross_margin": "gross_margin_basis"}
BASIS_ALIASES = {"营业收入": "营业收入", "营业总收入": "营业总收入", "归母": "归母", "total": "total", "weighted": "weighted",
                 "period_end": "period_end", "basic": "basic", "diluted": "diluted"}


def served_basis(sym: str, metric: str) -> str | None:
    f = fundamentals(sym)
    if not f:
        return None
    if metric == "prior_net_profit":  # derived as net_profit / (1 + netprofit_yoy): a 归母 prior only if the YoY is 归母
        return f["definitions"].get("netprofit_yoy_basis")
    b = f["definitions"].get(BASIS_KEY.get(metric, ""), None)
    if b and b.startswith("营业收入("):
        return "营业收入"
    return b


def fact(sym: str, metric: str, date: str | None = None) -> tuple[float | None, str]:
    """Served value (base units: 元, %, 元/股, 倍, 股) and a short provenance note."""
    if metric in ("close", "pct", "change", "amount", "volume", "prev_close"):
        bs = bars(sym)
        if date not in bs:
            return None, f"no served bar for {sym} on {date}"
        b = bs[date]
        prev_dates = sorted(d for d in bs if d < date)
        prev = bs[prev_dates[-1]]["close"] if prev_dates else None
        prev_note = f"previous served close {prev_dates[-1]}" if prev_dates else ""
        if prev is None and b.get("pct_change_1d") is not None:
            prev = b["close"] / (1 + b["pct_change_1d"] / 100)
            prev_note = "previous close derived as close/(1+pct)"
        if metric == "close":
            return b["close"], f"bar {date}"
        if metric == "prev_close":
            return prev, prev_note
        if metric == "pct":
            if b.get("pct_change_1d") is not None:
                return b["pct_change_1d"], f"bar {date} pct_change_1d"
            if prev:
                return (b["close"] / prev - 1) * 100, f"pct derived from {prev_note}"
            return None, "no pct"
        if metric == "change":
            return (b["close"] - prev, f"close − {prev_note}") if prev is not None else (None, "no previous close")
        if metric == "amount":
            return b.get("amount"), f"bar {date} amount"
        if metric == "volume":
            return b.get("volume"), f"bar {date} volume (shares)"
    f = fundamentals(sym)
    if not f:
        return None, f"{sym} not in fundamentals"
    if metric in ("total_mv", "pe_ttm", "pb"):
        if f.get("valuation_date") != date:
            return None, f"valuation served for {f.get('valuation_date')}, claim date {date}"
        return f.get(metric), f"{metric} on {date}"
    if metric == "gross_margin":
        v = f.get("gross_margin", f.get("grossprofit_margin"))
        return v, "gross margin"
    if metric == "net_margin":
        if f.get("net_profit") is None or not f.get("revenue"):
            return None, "no net margin"
        return f["net_profit"] / f["revenue"] * 100, "net_profit/revenue (derived)"
    if metric == "prior_revenue":
        return f["revenue"] / (1 + f["revenue_yoy"] / 100), "revenue/(1+revenue_yoy) (derived)"
    if metric == "prior_net_profit":
        return f["net_profit"] / (1 + f["netprofit_yoy"] / 100), "net_profit/(1+netprofit_yoy) (derived)"
    if metric == "nonrecurring":
        return f["net_profit"] - f["profit_dedt"], "net_profit − profit_dedt (derived)"
    if metric in f:
        return f[metric], f"{metric} ({f['_layer']})"
    return None, f"{metric} not served"


# ------------------------------------------------------------------------------------------- primary sources

def _num(s: str) -> float:
    s = s.strip().replace(",", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()%")
    v = float(s)
    return -v if neg else v


PU = {"元": 1.0, "千元": 1e3, "百万元": 1e6, "%": 1.0, "元/股": 1.0, "股": 1.0}


def primary_figures(sym: str) -> dict[str, dict]:
    figs = None
    for c in SRC_D.get("companies", []):
        if c.get("document") == sym or c.get("symbol") == sym:
            figs = c["figures"]
    if figs is None and sym in SRC_D.get("extension_definitions", {}):
        figs = SRC_D["extension_definitions"][sym]["figures"]
    out: dict[str, dict] = {}
    for f in figs or []:
        m = PU.get(f["unit"], 1.0)
        rec = {"value": _num(f["value"]) * m, "quote": f["quote"]}
        if f.get("prior"):
            rec["prior"] = _num(f["prior"]) * m
        out[f["id"]] = rec
    return out


def primary(sym: str, metric: str, basis: str | None) -> tuple[float | None, str]:
    """The annual report's figure for a metric on a basis (None basis = the served basis)."""
    pf = primary_figures(sym)
    if not pf:
        return None, "no transcription"
    b = basis or served_basis(sym, metric)

    def g(k):
        return pf[k]["value"] if k in pf else None

    def yoy(k):
        if k in pf and pf[k].get("prior"):
            return (pf[k]["value"] / pf[k]["prior"] - 1) * 100
        return None

    if metric == "revenue":
        k = "revenue_total" if b == "营业总收入" and "revenue_total" in pf else "revenue"
        return g(k), k
    if metric == "revenue_yoy":
        if "revenue_yoy" in pf:
            return g("revenue_yoy"), "revenue_yoy"
        return yoy("revenue"), "revenue vs prior (computed)"
    if metric == "net_profit":
        k = "net_profit_total" if b == "total" else "net_profit_parent"
        return g(k), k
    if metric == "netprofit_yoy":
        if b == "total":
            return yoy("net_profit_total"), "net_profit_total vs prior (computed)"
        if "net_profit_parent_yoy" in pf:
            return g("net_profit_parent_yoy"), "net_profit_parent_yoy"
        return yoy("net_profit_parent"), "net_profit_parent vs prior (computed)"
    if metric == "profit_dedt":
        return g("profit_dedt"), "profit_dedt"
    if metric == "roe":
        if b == "weighted":
            return g("roe_weighted"), "roe_weighted"
        if g("net_profit_parent") and g("equity_parent"):
            return g("net_profit_parent") / g("equity_parent") * 100, "net_profit_parent/equity_parent (computed)"
        return None, "no period-end ROE inputs"
    if metric == "eps":
        k = "eps_diluted" if b == "diluted" else "eps_basic"
        return g(k), k
    if metric == "gross_margin":
        if g("cost_of_sales") and g("revenue"):
            return (1 - g("cost_of_sales") / g("revenue")) * 100, "(revenue−cost_of_sales)/revenue (computed)"
        return None, "no cost of sales transcribed"
    if metric == "net_margin":
        if b == "total" and g("net_profit_total") and g("revenue"):
            return g("net_profit_total") / g("revenue") * 100, "net_profit_total/revenue (computed)"
        if g("net_profit_parent") and g("revenue"):
            return g("net_profit_parent") / g("revenue") * 100, "net_profit_parent/revenue (computed)"
        return None, "no inputs"
    if metric == "prior_revenue":
        return (pf["revenue"].get("prior") if "revenue" in pf else None), "revenue prior"
    if metric == "prior_net_profit":
        k = "net_profit_total" if b == "total" else "net_profit_parent"
        return (pf[k].get("prior") if k in pf else None), f"{k} prior"
    return None, f"{metric} not transcribed"


def same_figure(sym: str, metric: str, named: str) -> bool:
    """Does the report give the same figure on the named basis and on the served basis?"""
    served = served_basis(sym, metric)
    a, _ = primary(sym, metric, named)
    b, _ = primary(sym, metric, served)
    if a is None or b is None:
        return False
    return abs(a - b) <= 1e-9 * max(1.0, abs(b)) or (metric.endswith("yoy") and abs(a - b) < 0.005)


# --------------------------------------------------------------------------------------------- evaluators

def decimals(s: str) -> int:
    s = s.lstrip("+-")
    return len(s.split(".")[1]) if "." in s else 0


def holds_value(claimed_s: str, unit: str, qual: str, actual: float | None, signed: bool) -> bool | None:
    if actual is None:
        return None
    m = UNITS[unit]
    c = float(claimed_s) * m
    step = 10 ** (-decimals(claimed_s)) * m
    if signed and qual != "exact":
        if (c > 0) != (actual > 0) and c != 0:
            return False
        c, actual = abs(c), abs(actual)
    if qual == "exact":
        if signed and c != 0 and (c > 0) != (actual > 0):
            return False
        return abs(c - actual) <= max(step / 2, 0.01 * abs(actual)) + 1e-12
    if qual == "approx":
        return abs(c - actual) <= 0.05 * abs(c) + 1e-12
    if qual == "gte":
        return actual >= c
    if qual == "gt":
        return actual > c
    if qual == "near":
        return 0.9 * c <= actual <= c
    if qual == "lt":
        return actual < c
    if qual == "n_more":
        return c < actual < c + step
    if qual == "double_digit":
        return 10 <= actual < 100
    raise ValueError(qual)


ANNUAL_BANDS = {
    "up": lambda g: g > 0, "down": lambda g: g < 0,
    "slight_up": lambda g: 0 < g <= 5, "slight_down": lambda g: -5 <= g < 0,
    "flat": lambda g: abs(g) <= 1, "strong_up": lambda g: g >= 20, "strong_down": lambda g: g <= -20,
    "halved": lambda g: -60 <= g <= -40, "doubled": lambda g: (1 + g / 100) >= 1.9,
}


def daily_band(band: str, c: float, is_index: bool, sym: str) -> bool:
    strong = 1.5 if is_index else 3.0
    limit = 19.9 if sym.startswith(("300", "688")) else 9.9
    return {
        "up": c > 0, "down": c < 0, "slight_up": 0 < c <= 1, "slight_down": -1 <= c < 0, "flat": abs(c) <= 0.2,
        "strong_up": c >= strong, "strong_down": c <= -strong, "limit_up": c >= limit, "limit_down": c <= -limit,
    }[band]


def ref_value(ref: dict) -> tuple[float | None, str]:
    sym = SYM[ref["entity"]]
    return fact(sym, ref["metric"], ref.get("date"))


def eval_part(p: dict) -> dict:
    """Return the part with actual, served basis, holds (expected), holds_blind and primary comparison."""
    out = dict(p)
    kind = p["kind"]
    if kind == "null":
        out.update(holds=None, holds_blind=None)
        return out
    if kind == "definition":
        sym = SYM[p["entity"]]
        sb = served_basis(sym, p["metric"])
        out["served_basis"] = sb
        if not p["standard"]:
            out.update(holds=False, holds_blind=False)
        elif p.get("basis_named") and sb and p["basis_named"] != sb and not same_figure(sym, p["metric"], p["basis_named"]):
            out.update(holds=None, holds_blind=True)
        else:
            out.update(holds=True, holds_blind=True)
        return out
    if kind == "value":
        sym = SYM[p["entity"]]
        actual, note = fact(sym, p["metric"], p.get("date"))
        signed = p["metric"] in SIGNED
        h = holds_value(p["claimed"], p.get("unit", ""), p.get("qual", "exact"), actual, signed)
        out.update(actual=actual, actual_note=note)
        sb = served_basis(sym, p["metric"])
        out["served_basis"] = sb
        named = p.get("basis_named")
        mismatch = bool(named and sb and named != sb and not same_figure(sym, p["metric"], named))
        out["basis_mismatch"] = mismatch
        out.update(holds=None if mismatch else h, holds_blind=h)
        if p.get("date") is None and p["metric"] in ("revenue", "revenue_yoy", "net_profit", "netprofit_yoy",
                                                      "profit_dedt", "roe", "eps", "gross_margin", "net_margin",
                                                      "prior_revenue", "prior_net_profit"):
            pv, pnote = primary(sym, p["metric"], named)
            if pv is not None:
                # the derived prior (net_profit / (1 + yoy)) has a 归母 numerator: compare it with the report's 归母 prior
                psv, _ = primary(sym, p["metric"], "归母" if p["metric"] == "prior_net_profit" else sb)
                # agreement: within 1% of the report figure; a percentage also agrees within 0.01 (printed rounding)
                pct_metric = p["metric"] in ("revenue_yoy", "netprofit_yoy", "roe", "gross_margin", "net_margin")
                tol = max(0.01 * abs(psv), 0.01 if pct_metric else 0.0) + 1e-9 if psv is not None else None
                out["primary"] = {
                    "report_value": pv, "report_field": pnote,
                    "primary_holds": holds_value(p["claimed"], p.get("unit", ""), p.get("qual", "exact"), pv, signed),
                    "served_vs_report": None if (psv is None or actual is None) else
                    ("agree" if abs(actual - psv) <= tol else f"differ (served {actual:.6g}, report {psv:.6g})"),
                }
                if p.get("alt_primary_basis"):
                    av, anote = primary(sym, p["metric"], p["alt_primary_basis"])
                    out["primary"]["alt_basis"] = {
                        "basis": p["alt_primary_basis"], "report_value": av, "report_field": anote,
                        "primary_holds": None if av is None else
                        holds_value(p["claimed"], p.get("unit", ""), p.get("qual", "exact"), av, signed)}
        return out
    if kind == "direction":
        sym = SYM[p["entity"]]
        actual, note = fact(sym, p["metric"], p.get("date"))
        out.update(actual=actual, actual_note=note)
        if actual is None:
            out.update(holds=None, holds_blind=None)
            return out
        if p["metric"] == "pct":
            h = daily_band(p["band"], actual, sym in INDICES, sym)
        else:
            h = ANNUAL_BANDS[p["band"]](actual)
        named = p.get("basis_named")
        sb = served_basis(sym, p["metric"])
        mismatch = bool(named and sb and named != sb and not same_figure(sym, p["metric"], named))
        out.update(served_basis=sb, basis_mismatch=mismatch, holds=None if mismatch else h, holds_blind=h)
        if p["metric"] in ("revenue_yoy", "netprofit_yoy"):
            pv, pnote = primary(sym, p["metric"], named)
            if pv is not None:
                out["primary"] = {"report_value": pv, "report_field": pnote, "primary_holds": ANNUAL_BANDS[p["band"]](pv)}
        return out
    if kind == "relation":
        a, an = ref_value(p["a"])
        b, bn = ref_value(p["b"])
        out.update(a_actual=a, b_actual=b, a_note=an, b_note=bn)
        if a is None or b is None:
            out.update(holds=None, holds_blind=None)
            return out
        op = p["op"]
        if op == "gt":
            h = a > b
        elif op == "lt":
            h = a < b
        elif op == "gt_magnitude_both_negative":
            h = a < 0 and b < 0 and abs(a) > abs(b)
        elif op == "lt_magnitude_both_negative":
            h = a < 0 and b < 0 and abs(a) < abs(b)
        elif op == "opposite_sign":
            h = a * b < 0
        elif op == "ratio":
            out["computed"] = a / b
            h = holds_value(p["claimed"], "", p.get("qual", "exact"), a / b, False)
        elif op == "diff":
            out["computed"] = a - b
            h = holds_value(p["claimed"], p.get("unit", ""), p.get("qual", "exact"), a - b, False)
        elif op == "ratio_minus_one":
            out["computed"] = a / b - 1
            h = holds_value(p["claimed"], "", p.get("qual", "exact"), a / b - 1, False)
        else:
            raise ValueError(op)
        mismatch = False
        for side in ("a", "b"):
            r = p[side]
            if r.get("basis_named"):
                sym = SYM[r["entity"]]
                sb = served_basis(sym, r["metric"])
                if sb and r["basis_named"] != sb and not same_figure(sym, r["metric"], r["basis_named"]):
                    mismatch = True
        out.update(basis_mismatch=mismatch, holds=None if mismatch else h, holds_blind=h)
        return out
    if kind == "sum":
        vals = [ref_value(r)[0] for r in p["refs"]]
        if any(v is None for v in vals):
            out.update(holds=None, holds_blind=None)
            return out
        total = sum(vals)
        if p.get("op") == "average":
            total /= len(vals)
        out["computed"] = total
        h = holds_value(p["claimed"], p.get("unit", ""), p.get("qual", "exact"), total, False)
        out.update(holds=h, holds_blind=h)
        return out
    raise ValueError(kind)


def category(period: str, parts: list[dict]) -> str:
    """One category per claim, from its parts (first match in this order)."""
    kinds = {p["kind"] for p in parts}
    if period != "FY2025":
        return "valuation" if any(p.get("metric") in ("pe_ttm", "pb", "total_mv") for p in parts) else "daily bar / index"
    if "definition" in kinds:
        return "definition"
    rel = [p for p in parts if p["kind"] == "relation"]
    if rel:
        ents = {x["entity"] for p in rel for x in (p["a"], p["b"])}
        return "cross-company comparison" if len(ents) > 1 else "same-company comparison"
    if "direction" in kinds:
        return "weak direction word"
    if any(p.get("metric") == "profit_dedt" or "扣非" in p.get("what", "") for p in parts):
        return "扣非 figures"
    return "annual figures"


def verdict(values: list[bool | None]) -> str:
    if all(v is None for v in values):
        return "unverifiable"
    if all(v is True for v in values):
        return "supported"
    if any(v is True for v in values):
        return "partially_supported"
    return "contradicted"


def _clean(o):
    if isinstance(o, float):
        return None if math.isnan(o) else round(o, 6)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def main() -> None:
    from claim_parts_v6 import CONTEXT, PARTS  # hand-written parts

    rows = [json.loads(line) for line in open(HERE / "collected_v6.jsonl", encoding="utf-8") if line.strip()]
    out = []
    for r in rows:
        spec = PARTS[r["id"]]
        parts = [eval_part(p) for p in spec]
        for i, p in enumerate(parts, 1):
            p["part_id"] = f"{r['id']}.{i}"
        exp = verdict([p["holds"] for p in parts])
        blind = verdict([p["holds_blind"] for p in parts])
        prim = [p["primary"] for p in parts if "primary" in p]
        row = {**r, "category": category(r["period"], parts), "parts": parts, "expected_verdict": exp, "expected_verdict_definition_blind": blind,
               "context_not_checked": CONTEXT.get(r["id"], []),
               "primary_source_agreement": (None if not prim else
                                            ("agree" if all(x.get("served_vs_report") in (None, "agree") for x in prim)
                                             else "differ")),
               "definition_difference": any(p.get("basis_mismatch") for p in parts)}
        out.append(_clean(row))
    with open(HERE / "claims_v6.jsonl", "w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r["expected_verdict"] for r in out))
    print("definition-blind", Counter(r["expected_verdict_definition_blind"] for r in out))


if __name__ == "__main__":
    main()
