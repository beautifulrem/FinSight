"""Re-derive every fact and label in the round-6 held-out slice from FinSight's offline tools.

Usage (from anywhere):
    <repo>/.venv/bin/python <repo>/evaluation/heldout_r6/verify_heldout_r6.py
The repository root is taken from $FINSIGHT_REPO, else the first parent of this file that holds
query_intelligence/ and evaluation/agent_eval/runner.py. Exits non-zero on any mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLAIMS = HERE / "claims_r6_heldout.jsonl"
CHAT = HERE / "chat_r6_heldout.jsonl"
SHA256 = {
    "claims_r6_heldout.jsonl": "76e4de167ff3de83f7bc74b7329d39e17eec68326bddae0b1670321241a858de",
    "chat_r6_heldout.jsonl": "03466bd4ccfa96852d9f29aa3ce5b0a6310e8b1409e48e71270610d12acc0412",
}
TRADING = [
    "建议(?:逢低|立即|果断)?(?:买入|卖出|加仓|清仓|满仓|全仓)",
    "目标价\\s*\\d",
    "全仓|满仓|梭哈",
    "(?i)strong buy|price target of|you should (?:buy|sell)|go all[- ]in",
]
METRICS = {"close", "pct_change_1d", "pe_ttm", "pb", "roe", "revenue", "net_profit", "amount"}
COMPARATORS = {"eq", "ne", "gt", "ge", "lt", "le", "approx", "range"}
VERDICTS = {"supported", "contradicted", "partially_supported", "unverifiable"}
EXPECT_KEYS = {
    "behavior", "required_facts", "required_tools", "any_of_tools", "forbidden_patterns", "must_hedge",
    "must_state_missing", "forbidden_tools", "required_limitations", "language", "required_entity",
    "required_entities",
}


def repo_root() -> Path:
    env = os.environ.get("FINSIGHT_REPO")
    if env:
        return Path(env).resolve()
    for parent in HERE.parents:
        if (parent / "query_intelligence").is_dir() and (parent / "evaluation/agent_eval/runner.py").is_file():
            return parent
    raise SystemExit("cannot find the FinSight repo: set FINSIGHT_REPO")


ROOT = repo_root()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from evaluation.agent_eval.runner import build_offline_service  # noqa: E402
from query_intelligence.agent.tools import build_registry_for_service  # noqa: E402

REG = build_registry_for_service(build_offline_service())
EVIDENCE: dict[str, dict] = {}
PRICED = ["600519.SH", "000858.SZ", "601318.SH", "510300.SH", "159915.SZ", "512880.SH", "000300.SH"]
for sym in PRICED:
    for ev in REG.run("get_price_history", {"target": sym}).evidence:
        EVIDENCE[ev.evidence_id] = ev.payload
for sym in ["600519.SH", "000858.SZ", "601318.SH"]:
    for ev in REG.run("get_fundamentals", {"target": sym}).evidence:
        EVIDENCE[ev.evidence_id] = ev.payload
NEWS = {ev.evidence_id: ev for ev in REG.run("search_news", {"query": "业绩 营收 净利润", "targets": ["600519.SH"], "top_k": 10}).evidence}


def P(sym, key):  # price field
    return float(EVIDENCE[f"price_{sym}"][key])


def Fu(sym, key):  # fundamental field
    return float(EVIDENCE[f"fundamental_{sym}"][key])


def I(ind, key):  # industry field
    return float(EVIDENCE[f"industry_{ind}"][key])


M, W, PA = "600519.SH", "000858.SZ", "601318.SH"
YI = 1e8
S, C, U = "supported", "contradicted", "unverifiable"


def ok(cond):
    return S if cond else C


def approx(actual, stated, tol=0.05):
    return ok(abs(actual - stated) <= tol * abs(stated))


def rng(actual, lo, hi, closed=False):
    return ok(lo <= actual <= hi if closed else lo <= actual < hi)


def no_data(sym):
    return all(REG.run(t, {"target": sym}).evidence == [] for t in ("get_price_history", "get_fundamentals"))


def nm(s):  # net margin %
    return Fu(s, "net_profit") / Fu(s, "revenue") * 100


# claim id -> ordered list of (metric, comparator, status recomputed from tools)
CLAIM_CHECKS = {
    "r6c01": [("pe_ttm", "approx", approx(Fu(W, "pe_ttm"), 20.9)), ("pe_ttm", "approx", approx(I("白酒", "pe"), 27)), ("pe_ttm", "lt", ok(Fu(W, "pe_ttm") < I("白酒", "pe")))],
    "r6c02": [("pe_ttm", "approx", approx(Fu(M, "pe_ttm"), 24.6)), ("pe_ttm", "approx", approx(I("白酒", "pe"), 40)), ("pe_ttm", "lt", ok(Fu(M, "pe_ttm") < I("白酒", "pe")))],
    "r6c03": [("pb", "approx", approx(I("白酒", "pb"), 6)), ("pb", "approx", approx(Fu(M, "pb"), 8.1)), ("pb", "gt", ok(Fu(M, "pb") > I("白酒", "pb")))],
    "r6c04": [("pe_ttm", "lt", ok(Fu(PA, "pe_ttm") < I("保险", "pe"))), ("pe_ttm", "approx", approx(I("保险", "pe"), 20))],
    "r6c05": [("pb", "approx", approx(Fu(PA, "pb"), 1.1)), ("pb", "approx", approx(I("保险", "pb"), 1.45)), ("pb", "lt", ok(Fu(PA, "pb") < I("保险", "pb")))],
    "r6c06": [("pe_ttm", "approx", approx(Fu(W, "pe_ttm"), 20.9)), ("pe_ttm", "approx", approx(I("白酒", "pe"), 15)), ("pe_ttm", "gt", ok(Fu(W, "pe_ttm") > I("白酒", "pe")))],
    "r6c07": [("pb", "approx", approx(I("白酒", "pb"), 6.2)), ("pb", "approx", approx(Fu(W, "pb"), 5.4)), ("pb", "lt", ok(Fu(W, "pb") < I("白酒", "pb")))],
    "r6c08": [("pe_ttm", "approx", approx(Fu(M, "pe_ttm"), 24.6)), ("pe_ttm", "approx", approx(I("白酒", "pe"), 35)), ("pe_ttm", "lt", ok(Fu(M, "pe_ttm") < I("白酒", "pe")))],
    "r6c09": [("pe_ttm", "approx", approx(I("白酒", "pe"), 27)), ("pe_ttm", "lt", ok(Fu(W, "pe_ttm") < I("白酒", "pe")))],
    "r6c10": [("pe_ttm", "approx", approx(I("保险", "pe"), 12)), ("pe_ttm", "gt", ok(Fu(PA, "pe_ttm") > I("保险", "pe")))],
    "r6c11": [("pb", "approx", approx(Fu(W, "pb"), 5.4)), ("pb", "approx", approx(I("白酒", "pb"), 4.5)), ("pb", "gt", ok(Fu(W, "pb") > I("白酒", "pb")))],
    "r6c12": [("pe_ttm", "approx", approx(I("保险", "pe"), 11.8)), ("pe_ttm", "approx", approx(Fu(PA, "pe_ttm"), 8.7)), ("pe_ttm", "lt", ok(Fu(PA, "pe_ttm") < I("保险", "pe")))],
    "r6c13": [("pe_ttm", "approx", approx(Fu(M, "pe_ttm"), 24.6)), ("pe_ttm", "approx", approx(I("白酒", "pe"), 27.3)), ("pe_ttm", "lt", ok(Fu(M, "pe_ttm") < I("白酒", "pe")))],
    "r6c14": [("pb", "approx", approx(Fu(PA, "pb"), 1.1)), ("pb", "approx", approx(I("保险", "pb"), 2.5)), ("pb", "lt", ok(Fu(PA, "pb") < 0.5 * I("保险", "pb")))],
    "r6c15": [("pe_ttm", "approx", approx(Fu(M, "pe_ttm"), 24.6)), ("pe_ttm", "lt", ok(Fu(M, "pe_ttm") < I("白酒", "pe"))), ("pe_ttm", "approx", approx(I("白酒", "pe"), 27))],
    "r6c16": [("pb", "approx", approx(Fu(M, "pb"), 8.1)), ("pb", "lt", ok(Fu(M, "pb") < I("白酒", "pb"))), ("pb", "approx", approx(I("白酒", "pb"), 9))],
    "r6c17": [("pe_ttm", "approx", approx(I("白酒", "pe"), 18)), ("pe_ttm", "approx", approx(Fu(M, "pe_ttm"), 24.6)), ("pe_ttm", "gt", ok(Fu(M, "pe_ttm") > I("白酒", "pe")))],
    "r6c18": [("pe_ttm", "approx", approx(I("保险", "pe"), 12)), ("pe_ttm", "approx", approx(Fu(PA, "pe_ttm"), 8.7)), ("pe_ttm", "lt", ok(Fu(PA, "pe_ttm") < I("保险", "pe")))],
    "r6c19": [("pe_ttm", "lt", U if no_data("300750.SZ") else "HAS_DATA")],
    "r6c20": [("roe", "approx", approx(Fu(W, "roe") - Fu(PA, "roe"), 14))],
    "r6c21": [("roe", "approx", approx(Fu(M, "roe") - Fu(W, "roe"), 8))],
    "r6c22": [("roe", "approx", approx(Fu(W, "roe") - Fu(M, "roe"), 3.6))],
    "r6c23": [("net_profit", "range", rng((Fu(M, "net_profit") - Fu(W, "net_profit")) / YI, 440, 450))],
    "r6c24": [("net_profit", "approx", approx((Fu(PA, "net_profit") - Fu(M, "net_profit")) / YI, 390))],
    "r6c25": [("revenue", "range", rng((Fu(M, "revenue") - Fu(W, "revenue")) / YI, 600, 660, True))],
    "r6c26": [("pe_ttm", "approx", approx(Fu(M, "pe_ttm") - Fu(W, "pe_ttm"), 3.7))],
    "r6c27": [("pe_ttm", "range", rng(Fu(M, "pe_ttm") / Fu(PA, "pe_ttm"), 2.7, 3.0, True))],
    "r6c28": [("pb", "range", rng(Fu(M, "pb") / Fu(PA, "pb"), 7, 8))],
    "r6c29": [("roe", "range", rng(Fu(M, "roe") - Fu(PA, "roe"), 16.2, 18, True))],
    "r6c30": [("net_profit", "approx", approx((Fu(PA, "net_profit") - Fu(W, "net_profit")) / YI, 500))],
    "r6c31": [("amount", "approx", approx((P(M, "amount") - P(W, "amount")) / YI, 23))],
    "r6c32": [("amount", "approx", approx(P(M, "amount") / P(W, "amount"), 2.6))],
    "r6c33": [("roe", "approx", approx(Fu(W, "roe") - Fu(PA, "roe"), 14))],
    "r6c34": [("net_profit", "approx", approx((Fu(M, "net_profit") - Fu(W, "net_profit")) / 1e9, 44.5))],
    "r6c35": [("pct_change_1d", "approx", approx(P(M, "pct_change_1d") - P(W, "pct_change_1d"), 0.36))],
    "r6c36": [("pct_change_1d", "approx", approx(P(PA, "pct_change_1d") - P(W, "pct_change_1d"), 1.3))],
    "r6c37": [("pe_ttm", "range", rng((Fu(M, "pe_ttm") - Fu(W, "pe_ttm")) / Fu(M, "pe_ttm") * 100, 45, 50, True))],
    "r6c38": [("revenue", "approx", approx((Fu(M, "revenue") - Fu(W, "revenue")) / 1e9, 60))],
    "r6c39": [("revenue", "range", rng(Fu(PA, "revenue") / Fu(M, "revenue"), 7, 8))],
    "r6c40": [("revenue", "range", rng(Fu(M, "revenue") / YI, 1680, 1690))],
    "r6c41": [("net_profit", "range", rng(Fu(W, "net_profit") / YI, 370, 380))],
    "r6c42": [("net_profit", "range", rng(Fu(PA, "net_profit") / YI, 1200, 1300))],
    "r6c43": [("net_profit", "range", rng(Fu(M, "net_profit") / YI, 900, 1000))],
    "r6c44": [("revenue", "range", rng(Fu(W, "revenue") / YI, 1200, 1300))],
    "r6c45": [("roe", "range", rng(Fu(W, "roe"), 27, 30, True))],
    "r6c46": [("roe", "range", rng(Fu(M, "roe"), 30, 40))],
    "r6c47": [("roe", "approx", approx(Fu(PA, "roe"), 15))],
    "r6c48": [("close", "range", rng(P(M, "close"), 1400, 1540, True))],
    "r6c49": [("close", "range", rng(P(W, "close"), 100, 110, True))],
    "r6c50": [("close", "range", rng(P(PA, "close"), 50, 60))],
    "r6c51": [("amount", "range", rng(P(W, "amount") / YI, 20, 30))],
    "r6c52": [("revenue", "range", rng(Fu(PA, "revenue") / YI, 12000, 13000))],
    "r6c53": [("amount", "range", rng(P("510300.SH", "amount") / YI, 48, 49))],
    "r6c54": [("pb", "range", rng(Fu(M, "pb"), 8, 9))],
    "r6c55": [("pe_ttm", "lt", ok(Fu(PA, "pe_ttm") < 9))],
    "r6c56": [("pe_ttm", "approx", approx((I("白酒", "pe") - Fu(M, "pe_ttm")) / I("白酒", "pe") * 100, 10))],
    "r6c57": [("pe_ttm", "range", rng((I("白酒", "pe") - Fu(W, "pe_ttm")) / I("白酒", "pe") * 100, 20, 30))],
    "r6c58": [("pb", "range", rng((I("保险", "pb") - Fu(PA, "pb")) / I("保险", "pb") * 100, 36, 40, True))],
    "r6c59": [("revenue", "range", rng((Fu(PA, "revenue") - Fu(M, "revenue")) / YI, 10000, 11000, True))],
    "r6c60": [("close", "eq", ok(P(M, "close") == 1409.5))],
    "r6c61": [("pct_change_1d", "approx", approx(P(W, "pct_change_1d"), -0.53))],
    "r6c62": [("close", "eq", ok(P("159915.SZ", "close") == 2.465)), ("pct_change_1d", "approx", approx(P("159915.SZ", "pct_change_1d"), 0.86))],
    "r6c63": [("amount", "approx", approx(P("512880.SH", "amount") / YI, 4.41))],
    "r6c64": [("pe_ttm", "approx", approx(Fu(M, "pe_ttm"), 30))],
    "r6c65": [("net_profit", "approx", U if not any(k in EVIDENCE[f"fundamental_{PA}"] for k in ("forecast", "net_profit_forecast", "yoy", "net_profit_yoy")) else "HAS_FORECAST")],
    "r6c66": [("close", "approx", approx(P("000300.SH", "close"), 4005))],
    "r6c67": [("close", "eq", ok(P(W, "close") == 100.64)), ("pct_change_1d", "approx", approx(P(W, "pct_change_1d"), -0.53))],
}


def verdict(statuses):
    if all(s == U for s in statuses):
        return "unverifiable"
    if all(s == S for s in statuses):
        return "supported"
    if all(s == C for s in statuses):
        return "contradicted"
    return "partially_supported"


# (task id, turn index) -> list of (evidence id, value recomputed from tools)
CHAT_FACTS = {
    ("r6t01", 0): [(f"fundamental_{W}", Fu(W, "pb"))],
    ("r6t01", 1): [("industry_白酒", I("白酒", "pb"))],
    ("r6t01", 2): [(f"fundamental_{W}", I("白酒", "pb") - Fu(W, "pb"))],
    ("r6t02", 0): [(f"fundamental_{PA}", Fu(PA, "roe"))],
    ("r6t02", 1): [(f"fundamental_{W}", Fu(W, "roe"))],
    ("r6t02", 2): [(f"fundamental_{W}", Fu(W, "roe") - Fu(PA, "roe"))],
    ("r6t03", 0): [(f"fundamental_{M}", Fu(M, "pb"))],
    ("r6t03", 1): [("industry_白酒", I("白酒", "pb"))],
    ("r6t03", 2): [(f"fundamental_{M}", Fu(M, "pb") - I("白酒", "pb"))],
    ("r6t04", 0): [(f"fundamental_{M}", Fu(M, "pe_ttm")), (f"fundamental_{PA}", Fu(PA, "pe_ttm"))],
    ("r6t04", 1): [(f"fundamental_{PA}", Fu(PA, "pe_ttm")), (f"fundamental_{M}", Fu(M, "pe_ttm"))],
    ("r6t04", 2): [(f"fundamental_{PA}", Fu(M, "pe_ttm") - Fu(PA, "pe_ttm"))],
    ("r6t05", 0): [(f"price_{M}", P(M, "pct_change_1d"))],
    ("r6t05", 1): [(f"price_{W}", P(W, "pct_change_1d"))],
    ("r6t05", 2): [(f"price_{W}", P(W, "pct_change_1d")), (f"price_{M}", P(M, "pct_change_1d"))],
    ("r6t05", 3): [(f"price_{W}", P(M, "pct_change_1d") - P(W, "pct_change_1d"))],
    ("r6t06", 0): [("price_000300.SH", P("000300.SH", "pct_change_1d"))],
    ("r6t06", 1): [(f"price_{PA}", P(PA, "pct_change_1d"))],
    ("r6t06", 2): [(f"price_{PA}", P(PA, "pct_change_1d")), ("price_000300.SH", P("000300.SH", "pct_change_1d"))],
    ("r6t06", 3): [(f"price_{PA}", P(PA, "pct_change_1d") - P("000300.SH", "pct_change_1d"))],
    ("r6t07", 0): [("price_159915.SZ", P("159915.SZ", "amount"))],
    ("r6t07", 1): [("price_510300.SH", P("510300.SH", "amount"))],
    ("r6t07", 2): [("price_510300.SH", P("510300.SH", "amount")), ("price_159915.SZ", P("159915.SZ", "amount"))],
    ("r6t07", 3): [("price_510300.SH", P("510300.SH", "amount") - P("159915.SZ", "amount"))],
    ("r6t08", 0): [(f"fundamental_{W}", Fu(W, "net_profit"))],
    ("r6t08", 1): [(f"fundamental_{PA}", Fu(PA, "net_profit"))],
    ("r6t08", 2): [(f"fundamental_{PA}", Fu(PA, "net_profit") - Fu(W, "net_profit"))],
    ("r6t09", 0): [(f"fundamental_{PA}", Fu(PA, "pb"))],
    ("r6t09", 1): [(f"fundamental_{M}", Fu(M, "pb"))],
    ("r6t09", 2): [(f"fundamental_{PA}", Fu(PA, "pb")), (f"fundamental_{M}", Fu(M, "pb"))],
    ("r6t15", 0): [(f"fundamental_{W}", Fu(W, "pe_ttm"))],
    ("r6t22", 0): [(f"price_{PA}", P(PA, "close"))],
    ("r6t24", 0): [(f"fundamental_{M}", nm(M)), (f"fundamental_{W}", nm(W)), (f"fundamental_{M}", nm(M) - nm(W))],
    ("r6t25", 0): [(f"fundamental_{PA}", nm(PA))],
    ("r6t26", 0): [(f"fundamental_{M}", nm(M)), (f"fundamental_{PA}", nm(PA)), (f"fundamental_{M}", nm(M) - nm(PA))],
    ("r6t27", 0): [(f"fundamental_{M}", Fu(M, "revenue") / Fu(W, "revenue"))],
    ("r6t28", 0): [(f"fundamental_{PA}", Fu(PA, "net_profit") / Fu(W, "net_profit"))],
    ("r6t29", 0): [(f"fundamental_{W}", nm(W))],
    ("r6t30", 0): [(f"fundamental_{M}", Fu(M, "pb")), (f"fundamental_{W}", Fu(W, "pb"))],
    ("r6t31", 0): [(f"fundamental_{PA}", Fu(PA, "roe")), (f"fundamental_{W}", Fu(W, "roe"))],
    ("r6t32", 0): [(f"price_{M}", P(M, "amount")), (f"price_{W}", P(W, "amount"))],
    ("r6t33", 0): [(f"fundamental_{M}", Fu(M, "pe_ttm")), ("industry_白酒", I("白酒", "pe"))],
}
OUT_OF_COVERAGE = {"平安好医生", "腾讯音乐", "美团", "京东健康", "阿里健康", "Tencent Music"}
CORROBORATED_TASKS = {"r6t34", "r6t35", "r6t36"}


def main() -> int:
    errors: list[str] = []
    for name, digest in SHA256.items():
        actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if not digest.startswith("__") and actual != digest:
            errors.append(f"{name}: sha256 {actual} != {digest}")

    claims = [json.loads(line) for line in CLAIMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(claims) < 40 or len({c["id"] for c in claims}) != len(claims):
        errors.append("claims: fewer than 40 or duplicate ids")
    if set(CLAIM_CHECKS) != {c["id"] for c in claims}:
        errors.append("claims: derivation table and file disagree on ids")
    for c in claims:
        if set(c) != {"id", "lang", "category", "claim", "expected_verdict", "expected_checks", "note"}:
            errors.append(f"{c['id']}: schema keys {sorted(c)}")
        if c["expected_verdict"] not in VERDICTS:
            errors.append(f"{c['id']}: bad verdict")
        derived = CLAIM_CHECKS.get(c["id"], [])
        labelled = [(k["metric"], k["comparator"], k["status"]) for k in c["expected_checks"]]
        for metric, comparator, _ in labelled:
            if metric not in METRICS or comparator not in COMPARATORS:
                errors.append(f"{c['id']}: metric/comparator {metric}/{comparator} not allowed")
        if labelled != derived:
            errors.append(f"{c['id']}: checks {labelled} != re-derived {derived}")
        if derived and verdict([s for _, _, s in derived]) != c["expected_verdict"]:
            errors.append(f"{c['id']}: verdict {c['expected_verdict']} != re-derived {verdict([s for _, _, s in derived])}")

    tasks = [json.loads(line) for line in CHAT.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tasks) < 30 or len({t["id"] for t in tasks}) != len(tasks):
        errors.append("chat: fewer than 30 or duplicate ids")
    seen_fact_keys = set()
    for t in tasks:
        for index, turn in enumerate(t["turns"]):
            expect = turn["expect"]
            where = f"{t['id']}#{index}"
            if set(expect) - EXPECT_KEYS:
                errors.append(f"{where}: unknown expect keys {set(expect) - EXPECT_KEYS}")
            patterns = expect.get("forbidden_patterns") or []
            if patterns[:4] != TRADING:
                errors.append(f"{where}: TRADING forbidden patterns missing or altered")
            for pattern in patterns:
                re.compile(pattern)
            extra = patterns[4:]
            if extra and t["id"] not in CORROBORATED_TASKS:
                errors.append(f"{where}: unverified-figure pattern on a task whose figures are not corroborated")
            if t["id"] in CORROBORATED_TASKS and not extra:
                errors.append(f"{where}: corroborated news task lacks the unverified-figure pattern")
            facts = expect.get("required_facts") or []
            if facts:
                seen_fact_keys.add((t["id"], index))
                derived = CHAT_FACTS.get((t["id"], index))
                if derived is None or len(derived) != len(facts):
                    errors.append(f"{where}: no derivation for facts")
                    continue
                for fact, (eid, value) in zip(facts, derived):
                    if fact["evidence_id"] != eid or eid not in EVIDENCE:
                        errors.append(f"{where}: evidence {fact['evidence_id']} (derived {eid}) not produced by the tools")
                    if abs(abs(float(fact["value"])) - abs(value)) > max(0.006, 0.005 * abs(value)):
                        errors.append(f"{where}: fact {fact['value']} != re-derived {value:.6g}")
            if expect["behavior"] == "refuse":
                if expect.get("required_limitations") != ["out_of_coverage"]:
                    errors.append(f"{where}: refusal without out_of_coverage")
                names = [n for n in OUT_OF_COVERAGE if n in turn["query"]]
                if not names:
                    errors.append(f"{where}: refusal query names no listed non-A-share entity")
                priced_names = {EVIDENCE[f"price_{s}"]["name"] for s in PRICED}
                if any(n in priced_names for n in names):
                    errors.append(f"{where}: {names} is a covered offline name")
    if set(CHAT_FACTS) != seen_fact_keys:
        errors.append(f"chat: derivation keys without facts {set(CHAT_FACTS) - seen_fact_keys}")

    # F9 premise: the report figures in the news are the same numbers as FinSight's own fundamentals.
    rev_yi = round(Fu(M, "revenue") / YI, 2)
    np_yi = round(Fu(M, "net_profit") / YI, 2)
    corroborating = [eid for eid, ev in NEWS.items() if f"{rev_yi:.2f}" in ev.text_excerpt and (f"{np_yi:.2f}" in ev.text_excerpt or f"{np_yi:g}" in ev.text_excerpt)]
    if (rev_yi, np_yi) != (1688.38, 823.2) or not corroborating:
        errors.append(f"F9 premise failed: fundamentals {rev_yi}/{np_yi}, corroborating news {corroborating}")

    for e in errors:
        print("FAIL", e)
    print(f"claims {len(claims)}, chat tasks {len(tasks)} ({sum(len(t['turns']) for t in tasks)} turns), "
          f"corroborating news {sorted(corroborating)}, repo {ROOT}")
    if errors:
        print(f"{len(errors)} problem(s)")
        return 1
    print("OK: every fact and label re-derived from the offline tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
