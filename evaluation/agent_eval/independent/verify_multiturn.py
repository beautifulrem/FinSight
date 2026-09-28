"""Validate multiturn_v1.jsonl against the real offline FinSight tools.

Usage (from the FinSight-r2 repo root):
    PYTHONPATH=. .venv/bin/python \
        evaluation/agent_eval/independent/verify_multiturn.py

Checks
1. every line parses as JSON; ids unique; 3-6 turns per task; expect keys are ones score_turn reads;
2. forbidden_patterns on every turn == evaluation.agent_eval.build_tasks.TRADING_PATTERNS;
3. every required_fact: the tool that produces its evidence_id is re-run now, the evidence id is present,
   and the value equals a numeric leaf of that evidence payload (|diff| <= 1e-9);
4. every required_entity resolves to itself via resolve_entity; entity mentions used in queries resolve
   to the expected symbol;
5. required_facts / required_entity only on answer turns; clarify/refuse turns carry no facts;
6. must_state_missing turns: the underlying tool really fails / the field really is null or absent
   (spot-checked per case below).
Exit code 1 on any failure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from evaluation.agent_eval.build_tasks import TRADING_PATTERNS
from evaluation.agent_eval.runner import build_offline_service
from query_intelligence.agent.tools import build_registry_for_service

HERE = Path(__file__).resolve().parent
TASKS = HERE.parents[0] / "tasks" / "agent_eval_multiturn_v1.jsonl"
EXPECT_KEYS = {
    "behavior", "required_tools", "any_of_tools", "required_facts", "required_entity",
    "must_hedge", "must_state_missing", "forbidden_patterns",
}
TOOLS = {
    "resolve_entity", "get_price_history", "compute_indicators", "get_fundamentals", "get_macro_indicators",
    "search_news", "search_announcements", "search_knowledge", "analyze_sentiment",
}
MENTIONS = {
    "贵州茅台": "600519.SH", "茅台": "600519.SH", "五粮液": "000858.SZ", "中国平安": "601318.SH",
    "沪深300ETF": "510300.SH", "创业板ETF": "159915.SZ", "证券ETF": "512880.SH", "沪深300": "000300.SH",
    "宁德时代": "300750.SZ", "上证指数": "000001.SH", "Kweichow Moutai": "600519.SH", "Moutai": "600519.SH",
    "Wuliangye": "000858.SZ", "Ping An Insurance": "601318.SH", "Ping An": "601318.SH",
    "CSI 300 ETF": "510300.SH", "ChiNext ETF": "159915.SZ", "securities ETF 512880": "512880.SH",
    "CSI 300 index": "000300.SH", "China Merchants Bank": "600036.SH",
}
# Symbols whose price-history tool must fail (used for must_state_missing turns).
NO_MARKET = ["300750.SZ", "600036.SH", "000001.SH"]


def numeric_leaves(value, sink):
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        sink.append(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            numeric_leaves(v, sink)
    elif isinstance(value, list):
        for v in value:
            numeric_leaves(v, sink)


def producer(evidence_id: str):
    if evidence_id.startswith("price_"):
        return "get_price_history", {"target": evidence_id[len("price_"):]}
    if evidence_id.startswith("fundamental_"):
        return "get_fundamentals", {"target": evidence_id[len("fundamental_"):]}
    if evidence_id.startswith("indicators_"):
        return "compute_indicators", {"target": evidence_id[len("indicators_"):]}
    if evidence_id == "industry_白酒":
        return "get_fundamentals", {"target": "600519.SH"}
    if evidence_id == "industry_保险":
        return "get_fundamentals", {"target": "601318.SH"}
    if evidence_id.startswith("macro_"):
        return "get_macro_indicators", {"topics": []}
    raise ValueError(f"no producer for {evidence_id}")


def main() -> int:
    errors: list[str] = []
    raw = TASKS.read_bytes()
    tasks = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"line {n}: {exc}")
    ids = [t["id"] for t in tasks]
    for tid, count in Counter(ids).items():
        if count > 1:
            errors.append(f"duplicate id {tid}")

    svc = build_offline_service()
    reg = build_registry_for_service(svc)
    cache: dict[str, dict] = {}

    def evidence_payloads(evidence_id: str):
        tool, args = producer(evidence_id)
        key = json.dumps([tool, args], ensure_ascii=False, sort_keys=True)
        if key not in cache:
            result = reg.run(tool, args)
            cache[key] = {e.evidence_id: e.payload for e in result.evidence}
        return cache[key].get(evidence_id)

    fact_checks = 0
    entities = set()
    for t in tasks:
        if set(t) - {"id", "category", "language", "turns"}:
            errors.append(f"{t['id']}: unexpected task keys {set(t)}")
        if not 3 <= len(t["turns"]) <= 6:
            errors.append(f"{t['id']}: {len(t['turns'])} turns")
        for i, turn in enumerate(t["turns"], 1):
            where = f"{t['id']}#{i}"
            exp = turn["expect"]
            if not turn.get("query") or not turn.get("note"):
                errors.append(f"{where}: missing query/note")
            if set(exp) - EXPECT_KEYS:
                errors.append(f"{where}: unknown expect keys {set(exp) - EXPECT_KEYS}")
            if exp.get("behavior") not in {"answer", "clarify", "refuse"}:
                errors.append(f"{where}: bad behavior")
            if exp.get("forbidden_patterns") != TRADING_PATTERNS:
                errors.append(f"{where}: forbidden_patterns differ from TRADING_PATTERNS")
            for tool in (exp.get("required_tools") or []) + (exp.get("any_of_tools") or []):
                if tool not in TOOLS:
                    errors.append(f"{where}: unknown tool {tool}")
            if exp["behavior"] != "answer" and (exp.get("required_facts") or exp.get("required_entity")):
                errors.append(f"{where}: facts/entity on a non-answer turn")
            if exp.get("required_entity"):
                entities.add(exp["required_entity"])
            for fact in exp.get("required_facts") or []:
                fact_checks += 1
                payload = evidence_payloads(fact["evidence_id"])
                if payload is None:
                    errors.append(f"{where}: evidence {fact['evidence_id']} not produced by the tool")
                    continue
                leaves: list[float] = []
                numeric_leaves(payload, leaves)
                if not any(abs(leaf - float(fact["value"])) <= 1e-9 for leaf in leaves):
                    errors.append(f"{where}: value {fact['value']} not in {fact['evidence_id']}")

    for symbol in sorted(entities):
        data = reg.run("resolve_entity", {"text": symbol}).data
        data = data.model_dump() if hasattr(data, "model_dump") else data
        got = [e["symbol"] for e in (data or {}).get("entities", [])]
        if symbol not in got:
            errors.append(f"required_entity {symbol} does not resolve to itself (got {got})")
    for mention, symbol in MENTIONS.items():
        data = reg.run("resolve_entity", {"text": mention}).data
        data = data.model_dump() if hasattr(data, "model_dump") else data
        got = [e["symbol"] for e in (data or {}).get("entities", [])]
        if not got or got[0] != symbol:
            errors.append(f"mention {mention!r} resolves to {got}, expected {symbol}")
    for mention in ("Apple", "苹果公司", "Bitcoin", "比特币", "英伟达", "狗狗币", "Amazon", "S&P 500"):
        data = reg.run("resolve_entity", {"text": mention}).data
        data = data.model_dump() if hasattr(data, "model_dump") else data
        got = [e["symbol"] for e in (data or {}).get("entities", [])]
        print(f"out-of-scope mention {mention!r} -> {got or 'unresolved'}")

    # must_state_missing spot checks (the data really is absent).
    for sym in NO_MARKET:
        if reg.run("get_price_history", {"target": sym}).ok:
            errors.append(f"{sym} unexpectedly has market data")
    for sym in ("300750.SZ", "600036.SH", "159915.SZ"):
        if reg.run("get_fundamentals", {"target": sym}).ok:
            errors.append(f"{sym} unexpectedly has fundamentals")
    for sym in ("600519.SH", "159915.SZ", "512880.SH"):
        if reg.run("compute_indicators", {"target": sym}).ok:
            errors.append(f"{sym} unexpectedly has indicators")
    ind = {e.evidence_id: e.payload for e in reg.run("compute_indicators", {"target": "510300.SH"}).evidence}
    p = ind["indicators_510300.SH"]
    if p.get("rsi_14") is not None or p.get("volatility_20d") is not None:
        errors.append("510300 RSI/volatility unexpectedly present")
    funds = {s: {e.evidence_id: e.payload for e in reg.run("get_fundamentals", {"target": s}).evidence} for s in ("600519.SH", "601318.SH")}
    if funds["600519.SH"]["fundamental_600519.SH"].get("gross_margin") is not None:
        errors.append("Moutai gross_margin unexpectedly present")
    if funds["601318.SH"]["fundamental_601318.SH"].get("gross_margin") is not None:
        errors.append("Ping An gross_margin unexpectedly present")
    for s, key in (("600519.SH", "fundamental_600519.SH"), ("601318.SH", "fundamental_601318.SH")):
        pl = funds[s][key]
        for absent in ("dividend_yield", "total_mv", "market_cap", "netprofit_yoy", "revenue_yoy"):
            if pl.get(absent) is not None:
                errors.append(f"{s} unexpectedly has {absent}")
        if pl.get("report_date") != "2025-12-31":
            errors.append(f"{s} report_date is {pl.get('report_date')}")
    if "roe" in funds["601318.SH"]["industry_保险"]:
        errors.append("insurance industry snapshot unexpectedly has ROE")
    macro_codes = {e.payload.get("indicator_code") for e in reg.run("get_macro_indicators", {"topics": []}).evidence}
    if any("LPR" in str(code) for code in macro_codes):
        errors.append("LPR unexpectedly present in macro data")

    turns = [turn for t in tasks for turn in t["turns"]]
    print(json.dumps({
        "tasks": len(tasks),
        "turns": len(turns),
        "by_category": dict(sorted(Counter(t["category"] for t in tasks).items())),
        "by_language": dict(sorted(Counter(t["language"] for t in tasks).items())),
        "turns_by_behavior": dict(Counter(turn["expect"]["behavior"] for turn in turns)),
        "must_hedge_turns": sum(1 for turn in turns if turn["expect"].get("must_hedge")),
        "must_state_missing_turns": sum(1 for turn in turns if turn["expect"].get("must_state_missing")),
        "fact_checks": fact_checks,
        "macro_codes": sorted(macro_codes),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }, ensure_ascii=False, indent=1))
    if errors:
        print("ERRORS:\n" + "\n".join(errors))
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
