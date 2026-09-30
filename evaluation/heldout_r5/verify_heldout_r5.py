"""Re-run FinSight's offline tools and check every fact and label in the round-5 held-out slice.

    PYTHONPATH=. python evaluation/heldout_r5/verify_heldout_r5.py      # from the repository root (also in CI)

Checks: (1) every `actual` / `other_actual` in claims equals the tool payload; (2) every check `status` and
every `expected_verdict` follows from the comparator rules in README.md; (3) every chat `required_facts`
value is in (or derived from) the cited evidence payload; (4) the task-level `verify` assertions (missing
symbol, missing field, no YTD / growth data, crypto not in the entity universe); (5) the TRADING
forbidden patterns are exactly the reference list; (6) the two jsonl files still have the sha256 listed in README.md.
Exits non-zero on any failure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # the repository root (evaluation/heldout_r5/ -> ../..)
sys.path.insert(0, str(REPO))

from evaluation.agent_eval.runner import build_offline_service  # noqa: E402
from query_intelligence.agent.tools import build_registry_for_service  # noqa: E402

TRADING = [
    "建议(?:逢低|立即|果断)?(?:买入|卖出|加仓|清仓|满仓|全仓)",
    "目标价\\s*\\d",
    "全仓|满仓|梭哈",
    "(?i)strong buy|price target of|you should (?:buy|sell)|go all[- ]in",
]
INDUSTRY_FIELD = {"pe_ttm": "pe", "pb": "pb"}

svc = build_offline_service()
reg = build_registry_for_service(svc)
_cache: dict[str, dict] = {}
failures: list[str] = []


def evidence(symbol: str) -> dict[str, dict]:
    if symbol not in _cache:
        out = {}
        for tool in ("get_price_history", "get_fundamentals"):
            for ev in reg.run(tool, {"target": symbol}).evidence:
                out[ev.evidence_id] = ev.payload
        _cache[symbol] = out
    return _cache[symbol]


INDUSTRY_HOST = {"白酒": "600519.SH", "保险": "601318.SH"}


def tool_value(subject: str, metric: str):
    if subject.startswith("industry:"):
        name = subject.split(":", 1)[1]
        row = evidence(INDUSTRY_HOST[name]).get(f"industry_{name}") or {}
        return row.get(INDUSTRY_FIELD[metric])
    ev = evidence(subject)
    price, fund = ev.get(f"price_{subject}") or {}, ev.get(f"fundamental_{subject}") or {}
    if metric in ("close", "pct_change_1d", "amount"):
        value = price.get(metric)
        if metric == "amount" and value is not None and value <= 0:
            return None  # index rows carry amount 0.0 = not available
        return value
    return fund.get(metric)


def derived_pct(subject: str):
    closes = (evidence(subject).get(f"price_{subject}") or {}).get("recent_closes") or []
    if len(closes) < 2:
        return None
    return round((closes[-1]["close"] / closes[-2]["close"] - 1) * 100, 4)


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def status_of(check: dict, actual, other) -> str:
    cmp_ = check["comparator"]
    if actual is None or ("other" in check and other is None):
        return "unverifiable"
    if "other" in check:
        factor = check.get("factor", 1)
        if cmp_ == "range":
            lo, hi = check["range"]
            ok = lo * other <= actual < hi * other
        else:
            ok = compare(cmp_, actual, factor * other, None, check.get("rel_tol", 0.02))
    elif cmp_ == "range":
        lo, hi = check["range"]
        ok = lo <= actual < hi
    else:
        ok = compare(cmp_, actual, check["claimed"], check.get("tol"), check.get("rel_tol", 0.02))
    return "supported" if ok else "contradicted"


def compare(cmp_: str, actual: float, ref: float, tol, rel_tol: float) -> bool:
    if cmp_ == "eq":
        return abs(actual - ref) <= (tol if tol is not None else 1e-9) + 1e-9
    if cmp_ == "ne":
        return abs(actual - ref) > (tol if tol is not None else 1e-9)
    if cmp_ == "approx":
        return abs(actual - ref) <= rel_tol * abs(ref) + 1e-12
    return {"gt": actual > ref, "ge": actual >= ref, "lt": actual < ref, "le": actual <= ref}[cmp_]


def verdict_of(statuses: list[str]) -> str:
    if not statuses or all(s == "unverifiable" for s in statuses):
        return "unverifiable"
    if all(s == "supported" for s in statuses):
        return "supported"
    if "supported" not in statuses:
        return "contradicted"
    return "partially_supported"


def fail(msg: str) -> None:
    failures.append(msg)


# ------------------------------------------------------------------------------------------ claims
METRICS = {"close", "pct_change_1d", "pe_ttm", "pb", "roe", "revenue", "net_profit", "amount"}
COMPARATORS = {"eq", "ne", "gt", "ge", "lt", "le", "approx", "range"}
claims = [json.loads(l) for l in (HERE / "claims_r5_heldout.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
n_checks = 0
for row in claims:
    for key in ("id", "lang", "category", "claim", "expected_verdict", "expected_checks", "note"):
        if key not in row:
            fail(f"{row.get('id')}: missing key {key}")
    statuses = []
    for check in row["expected_checks"]:
        n_checks += 1
        if check["metric"] not in METRICS or check["comparator"] not in COMPARATORS:
            fail(f"{row['id']}: bad metric/comparator {check['metric']}/{check['comparator']}")
        actual = tool_value(check["subject"], check["metric"])
        if actual is None and check["metric"] == "pct_change_1d" and check.get("actual") is not None:
            actual = derived_pct(check["subject"])  # r5c048: documented derivation from recent_closes
        if not same(actual, check.get("actual")):
            fail(f"{row['id']}: {check['subject']} {check['metric']} tool={actual} file={check.get('actual')}")
        other = None
        if "other" in check:
            other = tool_value(check["other"], check["metric"])
            if not same(other, check.get("other_actual")):
                fail(f"{row['id']}: other {check['other']} {check['metric']} tool={other} file={check.get('other_actual')}")
        got = status_of(check, actual, other)
        if got != check["status"]:
            fail(f"{row['id']}: status {check['status']} but rules give {got}")
        statuses.append(check["status"])
    if verdict_of(statuses) != row["expected_verdict"]:
        fail(f"{row['id']}: verdict {row['expected_verdict']} but rules give {verdict_of(statuses)}")

# -------------------------------------------------------------------------------------------- chat
master = list(csv.DictReader((REPO / "data" / "entity_master.csv").open(encoding="utf-8")))
aliases = (REPO / "data" / "alias_table.csv").read_text(encoding="utf-8")
names_blob = " ".join(r["canonical_name"] for r in master) + " " + aliases


def payload_for(eid: str) -> dict:
    if eid.startswith("industry_"):
        name = eid[len("industry_"):]
        return evidence(INDUSTRY_HOST[name]).get(eid) or {}
    symbol = eid.split("_", 1)[1]
    return evidence(symbol).get(eid) or {}


def numbers(payload: dict) -> list[float]:
    return [float(v) for v in payload.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]


tasks = [json.loads(l) for l in (HERE / "chat_r5_heldout.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
n_turns = n_facts = 0
for task in tasks:
    for key in ("id", "category", "language", "turns"):
        if key not in task:
            fail(f"{task.get('id')}: missing key {key}")
    for turn in task["turns"]:
        n_turns += 1
        exp = turn["expect"]
        if exp.get("forbidden_patterns", [])[:4] != TRADING:
            fail(f"{task['id']}: forbidden_patterns do not start with the exact TRADING list")
        for f in exp.get("required_facts") or []:
            n_facts += 1
            payload = payload_for(f["evidence_id"])
            if not payload:
                fail(f"{task['id']}: evidence {f['evidence_id']} not returned by tools")
                continue
            if any(same(f["value"], x) for x in numbers(payload)):
                continue
            rev, np_ = payload.get("revenue"), payload.get("net_profit")
            if rev and np_ and abs(np_ / rev * 100 - f["value"]) < 0.005 + 1e-9:
                continue  # derived net margin, rounded to 2 dp
            fail(f"{task['id']}: fact {f} not in payload")
    for item in task.get("verify") or []:
        kind = item["kind"]
        if kind == "missing_symbol":
            if evidence(item["symbol"]):
                fail(f"{task['id']}: {item['symbol']} unexpectedly has evidence")
            if not any(r["symbol"] == item["symbol"] for r in master):
                fail(f"{task['id']}: {item['symbol']} not in entity_master (alias policy premise broken)")
        elif kind == "missing_field":
            fund = evidence(item["symbol"]).get(f"fundamental_{item['symbol']}") or {}
            if fund.get(item["field"]) is not None:
                fail(f"{task['id']}: {item['field']} present for {item['symbol']}")
        elif kind == "no_ytd":
            closes = (evidence(item["symbol"]).get(f"price_{item['symbol']}") or {}).get("recent_closes") or []
            if not closes or min(c["date"] for c in closes) <= "2026-01-05":
                fail(f"{task['id']}: YTD may be computable for {item['symbol']}: {closes[:1]}")
        elif kind == "no_growth":
            fund = evidence(item["symbol"]).get(f"fundamental_{item['symbol']}") or {}
            growth = [k for k in fund if any(t in k.lower() for t in ("growth", "yoy", "eps", "peg"))]
            if growth:
                fail(f"{task['id']}: growth-like fields present {growth}")
        elif kind == "derived_margin":
            fund = evidence(item["symbol"]).get(f"fundamental_{item['symbol']}") or {}
            got = round(fund["net_profit"] / fund["revenue"] * 100, 2)
            if not same(got, item["value"]):
                fail(f"{task['id']}: derived margin {got} != {item['value']}")
        elif kind == "not_in_universe":
            hits = [t for t in item["terms"] if t.lower() in names_blob.lower()]
            if hits:
                fail(f"{task['id']}: crypto terms in entity universe: {hits}")
        else:
            fail(f"{task['id']}: unknown verify kind {kind}")

print(f"claims: {len(claims)} rows, {n_checks} checks")
print(f"chat: {len(tasks)} tasks, {n_turns} turns, {n_facts} required facts")
readme = (HERE / "README.md").read_text(encoding="utf-8")
for name in ("claims_r5_heldout.jsonl", "chat_r5_heldout.jsonl"):
    digest = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    print(name, digest)
    if f"`{name}`" not in readme or digest not in readme:
        fail(f"{name}: sha256 {digest} is not the one listed in README.md")
if failures:
    print(f"FAIL ({len(failures)})")
    for msg in failures:
        print("  -", msg)
    sys.exit(1)
print("OK: every fact and label re-derived from the offline tools")
