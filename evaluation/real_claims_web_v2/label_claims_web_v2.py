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
}

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
        for layer in (base, ext, fix):
            for sym, row in layer.get("market_api", {}).items():
                hist = row.get("history") or [
                    {k: row.get(k) for k in ("trade_date", "open", "high", "low", "close", "pct_change_1d",
                                             "volume", "amount")}
                ]
                self.bars[sym] = {b["trade_date"]: dict(b) for b in hist}
        self.primary: dict[str, dict[str, dict[str, Any]]] = {}
        for comp in self.sources.get("companies", []):
            self.primary[comp["symbol"]] = {f["id"]: f for f in comp["figures"]}
        for sym, comp in self.sources.get("extension_definitions", {}).items():
            self.primary[sym] = {f["id"]: f for f in comp["figures"]}

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
        if metric == "netprofit_yoy":
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
        return None if val is None else float(val)


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
    field = BASIS_FIELD.get(ref["m"])
    if not basis or not field:
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
    if metric not in ("revenue", "net_profit", "eps", "roe", "profit_dedt", "netprofit_yoy", "revenue_yoy"):
        return None
    served_basis = data.defs.get(sym, {}).get(BASIS_FIELD.get(metric, ""), None)
    basis = part.get("basis") or served_basis
    if metric in ("netprofit_yoy", "revenue_yoy"):
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

PARTS: dict[str, dict[str, Any]] = {}


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
