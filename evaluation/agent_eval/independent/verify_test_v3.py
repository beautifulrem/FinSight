"""Verify test_v3.jsonl and router_labels_independent_v1.jsonl against fresh offline tool output.

Checks
  * every required fact: the tool that produces its evidence id is re-run now, and the value is found in that
    evidence's payload (structured evidence: numeric match anywhere in the payload; document evidence: the
    number appears in the document's title / excerpt, which is the document content the tool returns);
  * every must_state_missing turn has a probe showing the data really is absent (tool fails / field null /
    period absent);
  * out-of-coverage turns do not resolve to an A-share / A-share ETF;
  * ids unique, schema keys valid, behaviours / routes valid, forbidden_patterns == TRADING_PATTERNS,
    tool names exist, size / balance thresholds.

Run: python verify_test_v3.py   (exit code 0 = all passed)
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from evaluation.agent_eval.build_tasks import TRADING_PATTERNS  # noqa: E402
from evaluation.agent_eval.runner import build_offline_service  # noqa: E402
from query_intelligence.agent.tools import build_registry_for_service  # noqa: E402

TASK_KEYS = {"id", "category", "language", "turns", "note"}
TURN_KEYS = {"query", "expect", "note"}
EXPECT_KEYS = {
    "behavior",
    "required_tools",
    "any_of_tools",
    "required_facts",
    "required_entity",
    "required_entities",
    "must_hedge",
    "must_state_missing",
    "forbidden_patterns",
    "forbidden_tools",
    "language",
    "required_limitations",
}
BEHAVIORS = {"answer", "clarify", "refuse"}
ROUTES = {"refuse", "clarify", "workflow", "agent"}
SYMBOL = re.compile(r"\d{6}\.(?:SH|SZ)")

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


svc = build_offline_service()
reg = build_registry_for_service(svc)
TOOLS = set(reg.names())
_cache: dict[str, object] = {}


def run(tool: str, args: dict):
    key = tool + json.dumps(args, sort_keys=True, ensure_ascii=False)
    if key not in _cache:
        _cache[key] = reg.run(tool, args)
    return _cache[key]


def evidence_for(evidence_id: str):
    """Re-run the tool that produces ``evidence_id`` and return the matching evidence record (or None)."""
    calls: list[tuple[str, dict]] = []
    if evidence_id.startswith("price_"):
        calls = [("get_price_history", {"target": evidence_id[6:], "days": 10})]
    elif evidence_id.startswith("fundamental_"):
        calls = [("get_fundamentals", {"target": evidence_id[12:]})]
    elif evidence_id.startswith("industry_"):
        calls = [("get_fundamentals", {"target": s}) for s in ("600519.SH", "000858.SZ", "601318.SH")]
    elif evidence_id.startswith("macro_"):
        calls = [("get_macro_indicators", {"topics": []})]
    elif evidence_id.startswith("indicators_"):
        calls = [("compute_indicators", {"target": evidence_id[11:]})]
    elif evidence_id.startswith("sentiment_"):
        calls = [("analyze_sentiment", {"targets": evidence_id[10:].split("_")})]
    else:
        symbols = SYMBOL.findall(evidence_id)
        calls = [(t, {"query": "", "targets": symbols}) for t in ("search_news", "search_announcements")]
    for tool, args in calls:
        result = run(tool, args)
        for ev in getattr(result, "evidence", None) or []:
            if ev.evidence_id == evidence_id:
                return ev
    return None


def numbers_in(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from numbers_in(v)


def fact_ok(fact) -> tuple[bool, str]:
    ev = evidence_for(fact["evidence_id"])
    if ev is None:
        return False, "evidence id not produced by fresh tool run"
    value = float(fact["value"])
    if ev.kind == "document":
        text = f"{ev.title or ''} {ev.text_excerpt or ''}"
        found = any(abs(float(m) - value) < 1e-9 for m in re.findall(r"-?\d+(?:\.\d+)?", text))
        return found, "value not in document title/excerpt"
    found = any(abs(n - value) < 1e-9 for n in numbers_in(ev.payload))
    return found, "value not in evidence payload"


# ------------------------------------------------------------------ missing-data probes
def fails(tool, args):
    return lambda: not run(tool, args).ok


def field_null(tool, args, field):
    return lambda: run(tool, args).ok and run(tool, args).evidence[0].payload.get(field) is None


def fund_period_is(sym, date):
    return lambda: run("get_fundamentals", {"target": sym}).evidence[0].payload.get("report_date") == date


def macro_dates_not(prefix):
    def probe():
        r = run("get_macro_indicators", {"topics": []})
        dates = [str(ev.payload.get("metric_date")) for ev in r.evidence]
        return bool(dates) and not any(d.startswith(prefix) for d in dates)

    return probe


def no_macro_code(fragment):
    def probe():
        r = run("get_macro_indicators", {"topics": [fragment]})
        codes = [str(ev.payload.get("indicator_code", "")) for ev in (r.evidence if r.ok else [])]
        return not any(fragment.lower() in c.lower() for c in codes)

    return probe


def closes(sym):
    return run("get_price_history", {"target": sym, "days": 30}).evidence[0].payload["recent_closes"]


PROBES = {
    ("v3_miss_zh_01", 0): fails("get_price_history", {"target": "300750.SZ"}),
    ("v3_miss_zh_02", 0): fails("get_fundamentals", {"target": "600036.SH"}),
    ("v3_miss_zh_03", 0): fund_period_is("600519.SH", "2025-12-31"),
    ("v3_miss_zh_04", 0): macro_dates_not("2023-12"),
    ("v3_miss_zh_05", 0): fails("compute_indicators", {"target": "600519.SH"}),
    ("v3_miss_zh_06", 0): fails("compute_indicators", {"target": "601318.SH"}),
    ("v3_miss_zh_07", 0): fails("get_fundamentals", {"target": "510300.SH"}),
    ("v3_miss_en_01", 0): fails("get_fundamentals", {"target": "300750.SZ"}),
    ("v3_miss_en_02", 0): fails("get_price_history", {"target": "002594.SZ"}),
    ("v3_miss_en_03", 0): lambda: all(c["date"] != "2025-06-30" for c in closes("600519.SH")),
    ("v3_miss_en_04", 0): no_macro_code("LPR"),
    ("v3_miss_en_05", 0): fails("analyze_sentiment", {"targets": ["600030.SH"]}),
    ("v3_miss_en_06", 0): lambda: len(closes("600519.SH")) < 20,
    ("v3_tech_zh_04", 0): field_null("compute_indicators", {"target": "510300.SH"}, "ma20"),
    ("v3_tech_en_03", 0): field_null("compute_indicators", {"target": "510300.SH"}, "rsi_14"),
    ("v3_multi_zh_05", 2): field_null("compute_indicators", {"target": "510300.SH"}, "rsi_14"),
    ("v3_multi_zh_07", 0): fails("get_price_history", {"target": "300750.SZ"}),
    ("v3_multi_zh_07", 1): fails("get_price_history", {"target": "002594.SZ"}),
}


def main() -> int:
    tasks = [json.loads(line) for line in open(REPO / "evaluation/agent_eval/tasks/agent_eval_test_v3.jsonl", encoding="utf-8")]
    router = [json.loads(line) for line in open(REPO / "evaluation/agent_eval/tasks/router_labels_independent_v1.jsonl", encoding="utf-8")]

    # ids
    ids = [t["id"] for t in tasks]
    check(len(ids) == len(set(ids)), f"duplicate task ids: {[i for i, c in Counter(ids).items() if c > 1]}")
    rids = [r["id"] for r in router]
    check(len(rids) == len(set(rids)), "duplicate router ids")
    rq = [r["query"] for r in router]
    check(len(rq) == len(set(rq)), f"duplicate router queries: {[q for q, c in Counter(rq).items() if c > 1]}")
    single_q = [t["turns"][0]["query"] for t in tasks if len(t["turns"]) == 1]
    check(len(single_q) == len(set(single_q)), "duplicate single-turn queries")

    n_facts = n_missing = 0
    for t in tasks:
        tid = t["id"]
        check(set(t) <= TASK_KEYS and {"id", "category", "language", "turns"} <= set(t), f"{tid}: task keys {set(t)}")
        check(t["language"] in {"zh", "en", "mixed"}, f"{tid}: language {t['language']}")
        check(1 <= len(t["turns"]) <= 4, f"{tid}: {len(t['turns'])} turns")
        for i, turn in enumerate(t["turns"]):
            where = f"{tid}[{i}]"
            check(set(turn) == TURN_KEYS, f"{where}: turn keys {set(turn)}")
            check(bool(str(turn.get("note", "")).strip()), f"{where}: empty note")
            check(bool(str(turn.get("query", "")).strip()), f"{where}: empty query")
            exp = turn["expect"]
            check(set(exp) <= EXPECT_KEYS, f"{where}: unknown expect keys {set(exp) - EXPECT_KEYS}")
            check(exp.get("behavior") in BEHAVIORS, f"{where}: behavior {exp.get('behavior')}")
            check(exp.get("forbidden_patterns") == TRADING_PATTERNS, f"{where}: forbidden_patterns != TRADING_PATTERNS")
            for p in exp.get("forbidden_patterns") or []:
                re.compile(p)
            if "language" in exp:
                check(exp["language"] in {"zh", "en"}, f"{where}: language {exp['language']}")
            for key in ("required_tools", "any_of_tools", "forbidden_tools"):
                unknown = set(exp.get(key) or []) - TOOLS
                check(not unknown, f"{where}: unknown tools in {key}: {unknown}")
            if exp.get("required_entities"):
                check(len(exp["required_entities"]) >= 2, f"{where}: required_entities needs 2+")
            if exp["behavior"] != "answer":
                for key in ("required_facts", "required_tools", "any_of_tools", "must_hedge", "must_state_missing"):
                    check(not exp.get(key), f"{where}: {key} set on a {exp['behavior']} turn")
            for fact in exp.get("required_facts") or []:
                n_facts += 1
                check(set(fact) == {"evidence_id", "value"}, f"{where}: fact keys {set(fact)}")
                ok, why = fact_ok(fact)
                check(ok, f"{where}: fact {fact} -> {why}")
            if exp.get("must_state_missing"):
                n_missing += 1
                probe = PROBES.get((tid, i))
                check(probe is not None, f"{where}: must_state_missing without a data probe")
                if probe is not None:
                    try:
                        check(bool(probe()), f"{where}: probe says data is NOT missing")
                    except Exception as exc:  # noqa: BLE001
                        check(False, f"{where}: probe error {exc!r}")
            if "out_of_coverage" in (exp.get("required_limitations") or []):
                r = run("resolve_entity", {"text": turn["query"]})
                ents = r.data["entities"] if r.ok else []
                a_share = [e for e in ents if SYMBOL.fullmatch(str(e.get("symbol") or ""))]
                check(not a_share, f"{where}: out-of-coverage query resolves to A-share {a_share}")
    unused = set(PROBES) - {
        (t["id"], i) for t in tasks for i, turn in enumerate(t["turns"]) if turn["expect"].get("must_state_missing")
    }
    check(not unused, f"probes for turns that are not must_state_missing: {unused}")

    # router schema + balance
    for r in router:
        check(set(r) == {"id", "query", "expected_route", "note"}, f"{r.get('id')}: router keys {set(r)}")
        check(r["expected_route"] in ROUTES, f"{r['id']}: route {r['expected_route']}")
        check(bool(r["note"].strip()), f"{r['id']}: empty note")
    routes = Counter(r["expected_route"] for r in router)
    for route in ROUTES:
        check(routes[route] >= 30, f"router route {route} has only {routes[route]}")

    # size thresholds
    turns = sum(len(t["turns"]) for t in tasks)
    multi = [t for t in tasks if len(t["turns"]) >= 2]
    check(len(tasks) >= 110, f"only {len(tasks)} tasks")
    check(turns >= 150, f"only {turns} turns")
    check(len(multi) >= 15, f"only {len(multi)} multi-turn tasks")
    check(len(router) >= 150, f"only {len(router)} router labels")

    langs = Counter(t["language"] for t in tasks)
    print(f"tasks={len(tasks)} turns={turns} multi_turn_tasks={len(multi)} languages={dict(langs)}")
    print(f"categories={dict(Counter(t['category'] for t in tasks))}")
    print(f"facts_checked={n_facts} missing_probes={n_missing} router={len(router)} routes={dict(routes)}")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
