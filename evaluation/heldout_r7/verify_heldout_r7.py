"""Verify chat_r7_heldout.jsonl against the real offline FinSight tools.

Checks, for every conversation and turn:

1. schema: ``id``/``category``/``language``/``turns``; each turn has ``query``, ``expect`` and ``note``; ids are
   unique; ``expect.language`` equals the conversation language; ``forbidden_patterns`` is exactly the
   four TRADING patterns;
2. every required fact: its ``derive`` spec is recomputed from fresh tool output; the stored value equals
   the recomputation at the stated precision; the fact's ``evidence_id`` is really returned by the tool;
   the scorer's own matcher (``verifier._is_supported``) accepts the stored value for the exact result;
3. discrimination: for a derived fact (gap, ratio, margin, holding value), an answer that only lists the
   operands must NOT satisfy it (the G1 failure mode "lists two prices" cannot pass by accident);
4. ``verify_absent`` fields are really absent/null in tool output (labels behind ``must_state_missing``);
5. refusals carry ``required_limitations: ["out_of_coverage"]`` and the snapshot holds no ``.HK`` security
   for the HK names used;
6. a synthetic gold response per turn passes ``metrics.score_turn`` (the expect dict is scorable as written).

Repo discovery: ``FINSIGHT_REPO`` if set, else walk up from this file, else from the working directory, to
the directory containing ``query_intelligence/``. Run with the repo's interpreter, e.g.::

    FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python verify_heldout_r7.py [tasks.jsonl]

Exit status 0 when every check passes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS_FILE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "chat_r7_heldout.jsonl"


def find_repo() -> Path:
    env = os.environ.get("FINSIGHT_REPO")
    if env and (Path(env) / "query_intelligence").is_dir():
        return Path(env).resolve()
    for start in (HERE, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "query_intelligence").is_dir():
                return candidate
    raise SystemExit("cannot find the FinSight repo: set FINSIGHT_REPO or run from inside it")


REPO = find_repo()
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from evaluation.agent_eval.metrics import score_turn  # noqa: E402
from evaluation.agent_eval.runner import build_offline_service  # noqa: E402
from query_intelligence.agent.tools import build_registry_for_service  # noqa: E402
from query_intelligence.agent.verifier import _is_supported, claim_numbers  # noqa: E402

TRADING = [
    "建议(?:逢低|立即|果断)?(?:买入|卖出|加仓|清仓|满仓|全仓)",
    "目标价\\s*\\d",
    "全仓|满仓|梭哈",
    "(?i)strong buy|price target of|you should (?:buy|sell)|go all[- ]in",
]
CATEGORIES = {"gap_followup", "derived_metric", "fair_value_implied", "hk_out_of_coverage", "control"}
HK_NAMES = ["中国平安H股", "02318.HK", "腾讯控股", "Tencent Holdings", "比亚迪电子", "BYD Electronic", "00700.HK"]

registry = build_registry_for_service(build_offline_service())
_cache: dict[str, dict] = {}
errors: list[str] = []
warnings: list[str] = []


def fail(message: str) -> None:
    errors.append(message)


def evidence(evidence_id: str) -> dict | None:
    if evidence_id not in _cache:
        kind, _, target = evidence_id.partition("_")
        tool = "get_price_history" if kind == "price" else "get_fundamentals"
        for item in registry.run(tool, {"target": target}).evidence:
            _cache[item.evidence_id] = item.payload
    return _cache.get(evidence_id)


def field(ref: str) -> float:
    evidence_id, _, name = ref.partition(":")
    payload = evidence(evidence_id)
    if payload is None:
        raise LookupError(f"tool did not return {evidence_id}")
    if payload.get(name) is None:
        raise LookupError(f"{ref} missing in tool output")
    return float(payload[name])


def operands(spec: dict) -> list[str]:
    refs = [spec["a"]] + ([spec["b"]] if "b" in spec else [])
    if spec["op"] == "margin_gap":
        refs += [ref.split(":")[0] + ":revenue" for ref in refs]
    return refs


def exact(spec: dict) -> float:
    a = field(spec["a"])
    op = spec["op"]
    if op == "value":
        return a
    if op == "mul":
        return a * spec["k"]
    b = field(spec["b"])
    if op == "sub":
        return abs(a - b)
    if op == "div":
        return a / b
    if op == "pct":
        return a / b * 100
    if op == "margin_gap":
        ra = field(spec["a"].split(":")[0] + ":revenue")
        rb = field(spec["b"].split(":")[0] + ":revenue")
        return abs(a / ra - b / rb) * 100
    raise ValueError(f"unknown op {op}")


def render(value: float, language: str) -> str:
    """How an answer would naturally state the number (亿 / billion for large amounts)."""
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f}亿元" if language == "zh" else f"RMB {value / 1e9:.2f} billion"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text


def gold_response(task: dict, turn: dict) -> dict:
    expect = turn["expect"]
    language = task["language"]
    facts = expect.get("required_facts") or []
    behavior = expect.get("behavior", "answer")
    parts = [render(float(item["value"]), language) for item in facts]
    if expect.get("must_hedge"):
        parts.append("仅为条件性测算，不构成投资建议" if language == "zh" else "a conditional estimate, not investment advice")
    if expect.get("must_state_missing"):
        parts.append("工具未返回该数据" if language == "zh" else "this figure is not available in the data")
    tools = sorted(set(expect.get("required_tools") or []) | set((expect.get("any_of_tools") or [])[:1]))
    entities = set(expect.get("required_entities") or []) | (
        {expect["required_entity"]} if expect.get("required_entity") else set()
    )
    response = {
        "route": "refuse" if behavior == "refuse" else ("clarify" if behavior == "clarify" else "workflow"),
        "answer": "；".join(parts) or "ok",
        "key_points": [],
        "evidence_used": sorted({item["evidence_id"] for item in facts}),
        "tool_calls": [{"tool": tool, "ok": True} for tool in tools],
        "risk_disclaimer": "仅供参考",
        "language": language,
        "limitations": list(expect.get("required_limitations") or []),
        "nlu_summary": {"entities": [{"symbol": symbol} for symbol in sorted(entities)]},
    }
    if behavior == "clarify":
        response["status"] = "needs_clarification"
    return response


def main() -> int:
    raw = TASKS_FILE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    tasks = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = Counter(task.get("id") for task in tasks)
    for tid, count in ids.items():
        if count > 1:
            fail(f"duplicate id {tid}")

    facts_checked = 0
    for task in tasks:
        tid = task.get("id")
        if set(task) != {"id", "category", "language", "turns"}:
            fail(f"{tid}: top-level keys {sorted(task)}")
        if task.get("category") not in CATEGORIES:
            fail(f"{tid}: unknown category {task.get('category')}")
        if task.get("language") not in {"zh", "en"}:
            fail(f"{tid}: language {task.get('language')}")
        for index, turn in enumerate(task.get("turns") or [], start=1):
            where = f"{tid} t{index} {turn.get('query')!r}"
            if not {"query", "expect", "note"} <= set(turn) or not str(turn.get("note") or "").strip():
                fail(f"{where}: needs query, expect, note")
                continue
            expect = turn["expect"]
            if expect.get("forbidden_patterns") != TRADING:
                fail(f"{where}: forbidden_patterns differ from TRADING")
            if expect.get("language") != task["language"]:
                fail(f"{where}: expect.language {expect.get('language')} != {task['language']}")
            if expect.get("behavior", "answer") == "refuse" and "out_of_coverage" not in (
                expect.get("required_limitations") or []
            ):
                fail(f"{where}: refusal without out_of_coverage")

            for ref in turn.get("verify_absent") or []:
                eid, _, name = ref.partition(":")
                payload = evidence(eid)
                if payload is None:
                    fail(f"{where}: {eid} not returned (cannot confirm {name} absent)")
                elif payload.get(name) is not None:
                    fail(f"{where}: {ref} is present ({payload[name]}) but labelled missing")

            for item in expect.get("required_facts") or []:
                facts_checked += 1
                spec = item.get("derive")
                if not spec:
                    fail(f"{where}: fact without derive spec")
                    continue
                try:
                    value_exact = exact(spec)
                    operand_values = [field(ref) for ref in operands(spec)]
                except LookupError as exc:
                    fail(f"{where}: {exc}")
                    continue
                stored = float(item["value"])
                half_unit = 0.5 * 10 ** (-spec["dp"]) + 1e-9
                if abs(stored - value_exact) > half_unit:
                    fail(f"{where}: stored {stored} vs recomputed {value_exact} (dp {spec['dp']})")
                if item["evidence_id"] != spec["a"].partition(":")[0]:
                    fail(f"{where}: evidence_id {item['evidence_id']} is not the first operand's id")
                if evidence(item["evidence_id"]) is None:
                    fail(f"{where}: tool did not return {item['evidence_id']}")
                if not _is_supported(stored, [value_exact]):
                    fail(f"{where}: scorer would not match stored {stored} to exact {value_exact}")
                if not _is_supported(stored, claim_numbers(render(stored, task["language"]))):
                    fail(f"{where}: scorer would not match {stored} to its natural rendering")
                if spec["op"] != "value" and _is_supported(stored, operand_values):
                    fail(f"{where}: derived {stored} is matched by an operand {operand_values}")
                # wrong-metric collisions (e.g. PB gap vs a PE figure of the same company): warn only
                others = []
                for ref in [r for r in operands(spec) if not r.startswith("price_")]:  # metric-name collisions
                    payload = evidence(ref.partition(":")[0]) or {}
                    used = {r.partition(":")[2] for r in operands(spec)}
                    others += [
                        float(v)
                        for k, v in payload.items()
                        if k not in used and isinstance(v, (int, float)) and not isinstance(v, bool)
                    ]
                if others and _is_supported(stored, others):
                    warnings.append(f"{where}: {stored} also matches another field of the same evidence")

            score = score_turn(gold_response(task, turn), expect)
            if not score["success"]:
                failed = [name for name, ok in score["checks"].items() if not ok]
                fail(f"{where}: synthetic gold response fails score_turn {failed}")

    for name in HK_NAMES:
        result = registry.run("get_price_history", {"target": name})
        hk = [item.evidence_id for item in result.evidence if str(item.payload.get("symbol", "")).endswith(".HK")]
        if hk:
            fail(f"snapshot unexpectedly covers HK security for {name}: {hk}")
        mapped = [item.evidence_id for item in result.evidence]
        if mapped:
            warnings.append(f"info: get_price_history({name!r}) returns A-share/fund evidence {mapped}")

    by_category = Counter(task["category"] for task in tasks)
    by_language = Counter(task["language"] for task in tasks)
    by_turns = Counter(len(task["turns"]) for task in tasks)
    print(f"file: {TASKS_FILE}")
    print(f"sha256: {digest}")
    print(f"repo: {REPO}")
    print(f"conversations: {len(tasks)}  turns: {sum(by_turns[n] * n for n in by_turns)}  facts checked: {facts_checked}")
    print(f"by category: {dict(sorted(by_category.items()))}")
    print(f"by language: {dict(sorted(by_language.items()))}")
    print(f"by turn count: {dict(sorted(by_turns.items()))}")
    for message in warnings:
        print(f"WARN {message}")
    for message in errors:
        print(f"FAIL {message}")
    print("OK" if not errors else f"{len(errors)} failure(s)")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
