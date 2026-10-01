"""Verify chat_r8_heldout.jsonl and claims_r8_heldout.jsonl against the real offline FinSight tools.

Chat, for every conversation and turn:

1. schema: ``id``/``category``/``language``/``turns``; each turn has ``query``, ``expect`` and ``note``; ids unique;
   ``expect.language`` equals the conversation language; ``forbidden_patterns`` is exactly the four TRADING patterns;
2. every required fact: its ``derive`` spec is recomputed here (own implementation, not imported from the build
   script) from fresh tool output; the stored value equals the recomputation at the stated precision; the fact's
   ``evidence_id`` is returned by the tool and is the first operand's id; the scorer's matcher
   (``verifier._is_supported``) accepts the stored value for the exact result and for its natural rendering;
3. discrimination: a derived fact (gap, ratio, relative %, margin, holding value, sum, yuan change) is NOT matched by
   any of its operands, neither as a raw number nor as the scorer would read the operand from an answer ("1409.5 元":
   the matcher scales by powers of ten, so a 100-share lot would be matched by the price itself);
4. ``verify_absent`` fields are absent/null in tool output (labels behind ``must_state_missing``);
5. refusals carry ``required_limitations: ["out_of_coverage"]``; no HK name returns a ``.HK`` security;
6. a synthetic gold response per turn passes ``metrics.score_turn``.

Claims: every row is re-derived by re-running the build script's predicates against fresh tool output (the
predicates are code, shown in ``build_heldout_r8.py``); the stored verdict and per-part truth values must match.

Repo discovery: ``FINSIGHT_REPO`` if set, else walk up from this file, else from the working directory.

    FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python verify_heldout_r8.py

Exit status 0 when every check passes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAT_FILE = HERE / "chat_r8_heldout.jsonl"
CLAIMS_FILE = HERE / "claims_r8_heldout.jsonl"


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
CATEGORIES = {
    "gap_lexicon",
    "single_turn_compare",
    "holding_value",
    "metric_aspect",
    "hk_lookalike",
    "injection_prediction",
    "control",
}
HK_NAMES = ["比亚迪港股", "中芯国际港股", "招商银行H股", "中国平安H股", "01211.HK", "00981.HK", "03968.HK", "02318.HK"]

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
    if spec["op"] == "gm_minus_nm":
        refs += [spec["b"].split(":")[0] + ":revenue"]
    return refs


def exact(spec: dict) -> float:
    a = field(spec["a"])
    op = spec["op"]
    if op == "value":
        return a
    if op == "mul":
        return a * spec["k"]
    b = field(spec["b"])
    if op == "chg_yuan":
        return abs(a - a / (1 + b / 100))
    if op == "sub":
        return abs(a - b)
    if op == "sum":
        return a + b
    if op == "div":
        return a / b
    if op == "pct":
        return a / b * 100
    if op == "rel":
        return abs(a - b) / abs(b) * 100
    if op == "margin_gap":
        ra = field(spec["a"].split(":")[0] + ":revenue")
        rb = field(spec["b"].split(":")[0] + ":revenue")
        return abs(a / ra - b / rb) * 100
    if op == "gm_minus_nm":
        rb = field(spec["b"].split(":")[0] + ":revenue")
        return abs(a - b / rb * 100)
    raise ValueError(f"unknown op {op}")


def render(value: float, language: str) -> str:
    """How an answer would naturally state the number (亿 / billion for large amounts)."""
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f}亿元" if language == "zh" else f"RMB {value / 1e9:.2f} billion"
    return f"{value:.4f}".rstrip("0").rstrip(".")


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


def verify_chat() -> tuple[str, int, list[dict]]:
    raw = CHAT_FILE.read_bytes()
    tasks = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    for tid, count in Counter(task.get("id") for task in tasks).items():
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
                if abs(stored - value_exact) > 0.5 * 10 ** (-spec["dp"]) + 1e-9:
                    fail(f"{where}: stored {stored} vs recomputed {value_exact} (dp {spec['dp']})")
                if item["evidence_id"] != spec["a"].partition(":")[0]:
                    fail(f"{where}: evidence_id {item['evidence_id']} is not the first operand's id")
                if evidence(item["evidence_id"]) is None:
                    fail(f"{where}: tool did not return {item['evidence_id']}")
                if not _is_supported(stored, [value_exact]):
                    fail(f"{where}: scorer would not match stored {stored} to exact {value_exact}")
                if not _is_supported(stored, claim_numbers(render(stored, task["language"]))):
                    fail(f"{where}: scorer would not match {stored} to its natural rendering")
                if spec["op"] != "value":
                    if _is_supported(stored, operand_values):
                        fail(f"{where}: derived {stored} is matched by an operand {operand_values}")
                    for operand in operand_values:
                        for text in (render(operand, task["language"]), f"{operand:g}元", f"{operand:g}%"):
                            if _is_supported(stored, claim_numbers(text)):
                                fail(f"{where}: derived {stored} is matched by operand text {text!r}")
                others = []
                for ref in [r for r in operands(spec) if not r.startswith("price_")]:
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
            warnings.append(f"info: get_price_history({name!r}) returns non-HK evidence {mapped}")
    return hashlib.sha256(raw).hexdigest(), facts_checked, tasks


def verify_claims() -> tuple[str, list[dict]]:
    raw = CLAIMS_FILE.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    spec = importlib.util.spec_from_file_location("build_heldout_r8", HERE / "build_heldout_r8.py")
    build = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(build)  # re-runs every predicate against fresh tool output; writes nothing
    rebuilt = {row["id"]: row for row in build.CLAIMS}
    for row in rows:
        again = rebuilt.get(row["id"])
        if again is None:
            fail(f"claim {row['id']}: not produced by the build script")
            continue
        for key in ("claim", "expected_verdict", "parts", "lang", "category", "also_acceptable"):
            if row.get(key) != again.get(key):
                fail(f"claim {row['id']}: {key} differs from a fresh derivation ({row.get(key)!r} vs {again.get(key)!r})")
        if row["expected_verdict"] not in {"supported", "contradicted", "partially_supported"}:
            fail(f"claim {row['id']}: unexpected verdict {row['expected_verdict']}")
    if set(rebuilt) != {row["id"] for row in rows}:
        fail("claims file and build script disagree on the claim ids")
    return hashlib.sha256(raw).hexdigest(), rows


def main() -> int:
    chat_digest, facts_checked, tasks = verify_chat()
    claims_digest, claims = verify_claims()
    by_category = Counter(task["category"] for task in tasks)
    by_language = Counter(task["language"] for task in tasks)
    by_turns = Counter(len(task["turns"]) for task in tasks)
    print(f"chat file: {CHAT_FILE.name} sha256 {chat_digest}")
    print(f"claims file: {CLAIMS_FILE.name} sha256 {claims_digest}")
    print(f"repo: {REPO}")
    print(f"conversations: {len(tasks)}  turns: {sum(n * c for n, c in by_turns.items())}  facts checked: {facts_checked}")
    print(f"by category: {dict(sorted(by_category.items()))}")
    print(f"by language: {dict(sorted(by_language.items()))}")
    print(f"by turn count: {dict(sorted(by_turns.items()))}")
    print(f"claims: {len(claims)}  verdicts: {dict(Counter(c['expected_verdict'] for c in claims))}")
    for message in warnings:
        print(f"WARN {message}")
    for message in errors:
        print(f"FAIL {message}")
    print("OK" if not errors else f"{len(errors)} failure(s)")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
