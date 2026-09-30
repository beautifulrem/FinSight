"""Generate the 100 FinSight answers the owner labels (``labels/answers_to_label.csv``).

Questions are single-turn tasks sampled with a fixed seed from test v3 and the held-out set. About half are
answered on the deterministic path (``mode=auto``, no LLM) and half on the LLM agent path (``mode=agent``).
Every tool call runs against the offline tools directly (the same offline data the product serves when live
sources are off, as ``runner --no-replay`` does), not against the recorded evaluation snapshots: a call
missing from a snapshot would come back as an evaluation artifact ("not recorded in the evaluation snapshot")
and the labellers would be judging that instead of the product. FinSight's automatic score (task success,
verification) of every answer is kept in ``labels/answers_meta.jsonl``; the CSV is shown to the owner in a
shuffled order without the path or the automatic score, so the labels are blind to both.

    source /path/to/llmenv.sh && python -m evaluation.human.generate_answers --llm deepseek --force

Generation fails, and writes nothing, when an answer or its sources mention evaluation internals
(``EVAL_LEAK_PATTERNS``: evaluation snapshot, replay, fixtures, task sets …) or when a tool call hit a
replay gap. LLM calls are sequential and capped (``--max-llm-requests``, HTTP attempts including retries). On
HTTP 429, or when the cap is reached, generation stops and writes nothing; ``--allow-fallback`` instead
answers the remaining LLM items on the deterministic path and records the reason per row and in
``labels/generation.json`` (``--llm none --allow-fallback`` answers all 100 without a key, for testing).
"""

from __future__ import annotations

import argparse
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..agent_eval.metrics import llm_failure_flags, score_turn
from ..agent_eval.runner import (
    EVAL_TODAY,
    SNAPSHOT_NAME,
    TASK_SETS,
    _make_llm,
    build_offline_service,
    load_tasks,
)
from .common import (
    HUMAN_DIR,
    answer_text,
    describe_evidence,
    display,
    read_csv,
    run_config,
    sha256_file,
    write_csv,
    write_jsonl,
    write_result,
)

SEED = 20260930
N_TOTAL = 100
N_LLM = 50
SETS = ("test_v3", "holdout")
LABEL_COLUMNS = ("correct", "supported_by_sources", "compliant", "overall_good", "comment")
CSV_COLUMNS = ("id", "question", "answer", "sources", *LABEL_COLUMNS)
LABELS_DIR = HUMAN_DIR / "labels"
CSV_PATH = LABELS_DIR / "answers_to_label.csv"
META_PATH = LABELS_DIR / "answers_meta.jsonl"
GENERATION_PATH = LABELS_DIR / "generation.json"
REJECTED_PATH = HUMAN_DIR.parents[1] / "outputs" / "human_labels_rejected.jsonl"  # gitignored
ALLOWED_MODEL_PREFIX = "cline-pass/"
# Worst case of HTTP attempts one agent turn can still make (tool steps + compose + revise, with retries).
REQUESTS_HEADROOM = 8
# Wording that belongs to the evaluation harness, never to a product answer. A match in an answer or its
# sources fails generation. The offline data's own provenance label ("离线快照" / "offline snapshot", "industry
# snapshot" titles) is product behaviour (the offline deployment shows it too) and is not matched.
EVAL_LEAK_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"评估快照|评测快照|评估集|评测集|测试集|回放",
        r"未(?:被)?(?:记录|录制|收录)(?:在|于)?[^。；;\n]{0,8}快照",
        r"evaluation snapshot|eval(?:uation)?[ _-](?:set|task|fixture|harness|run)s?\b",
        r"not (?:recorded|captured) in (?:the |this )?(?:\w+ )?snapshot",
        r"\breplay(?:ed|ing)?\b|\bfixtures?\b|\bagent_eval\b|\beval_today\b",
        r"\btest_v\d\b|\bholdout(?:_v\d)?\b|\bheld-out\b|snapshot_\w+\.json",
    )
)
REPLAY_MISS = "not recorded in the evaluation snapshot"


def sample_items(seed: int = SEED, n: int = N_TOTAL, n_llm: int = N_LLM) -> list[dict[str, Any]]:
    """The labelled questions, in presentation order, each with its planned answer path and id ``L001``…

    The pool is every single-turn task of test v3 then the held-out set, in file order. ``n`` are drawn
    without replacement; the last ``n_llm`` drawn go to the LLM agent path. The drawn items are then
    shuffled again (same generator) so ids and row order do not reveal the path.
    """
    pool = [(name, task) for name in SETS for task in load_tasks(TASK_SETS[name][0]) if len(task["turns"]) == 1]
    rng = random.Random(seed)
    chosen = rng.sample(pool, n)
    items = [
        {"set": name, "task": task, "planned_path": "llm_agent" if index >= n - n_llm else "deterministic"}
        for index, (name, task) in enumerate(chosen)
    ]
    rng.shuffle(items)
    for index, item in enumerate(items, start=1):
        item["id"] = f"L{index:03d}"
    return items


def _slim_evidence(item: dict[str, Any]) -> dict[str, Any]:
    keep = ("evidence_id", "kind", "title", "source_name", "source_type", "source_url", "as_of", "produced_by")
    return {**{key: item.get(key) for key in keep}, "summary": describe_evidence(item)}


def _http_requests(llm: Any) -> int:
    stats = llm.http_stats() if llm is not None and hasattr(llm, "http_stats") else {}
    return int(stats.get("requests", 0))


def _http_429(llm: Any) -> int:
    stats = llm.http_stats() if llm is not None and hasattr(llm, "http_stats") else {}
    return int(stats.get("http_429", 0))


def fallback_reason(row: dict[str, Any], stop_reason: str | None) -> str | None:
    """Why a row planned for the LLM agent path was not answered by the LLM, else ``None``.

    Either the LLM was not used at all (no key, request cap, HTTP 429: ``stop_reason``), or an LLM call failed
    inside the turn and the graph fell back (``degraded`` LLM-failure flags). Refusals and clarifications
    answered by the guard before any LLM call are the agent path working as designed, not fallbacks.
    """
    if row["planned_path"] == "llm_agent" and row["path"] != "llm_agent":
        return stop_reason
    flags = llm_failure_flags(row.get("degraded"))
    if row["path"] == "llm_agent" and flags:
        return "LLM call failed in this turn; the graph fell back: " + ", ".join(flag[:80] for flag in flags)
    return None


def meta_row(
    item: dict[str, Any],
    response: dict[str, Any],
    *,
    path: str,
    mode: str,
    model: str | None,
    fallback_reason: str | None,
    latency_ms: float,
) -> dict[str, Any]:
    turn = item["task"]["turns"][0]
    score = score_turn(response, turn["expect"])
    verification = response.get("verification") or {}
    return {
        "id": item["id"],
        "set": item["set"],
        "task_id": item["task"]["id"],
        "category": item["task"].get("category"),
        "language": item["task"].get("language"),
        "question": turn["query"],
        "planned_path": item["planned_path"],
        "path": path,
        "mode": mode,
        "model": model,
        "fallback_reason": fallback_reason,
        "route": response.get("route"),
        "answer_source": response.get("answer_source"),
        "degraded": response.get("degraded") or [],
        "expected_behavior": score["expected_behavior"],
        "auto": {
            "task_success": score["success"],
            "checks": score["checks"],
            "uncited_success": score["uncited_success"],
            "verification_passed": verification.get("passed") if verification else None,
            "no_forbidden_content": score["checks"]["no_forbidden_content"],
        },
        "evidence_used": response.get("evidence_used") or [],
        "evidence": [_slim_evidence(entry) for entry in response.get("evidence_sources") or []],
        "llm_calls": (response.get("llm") or {}).get("calls", 0),
        "latency_ms": latency_ms,
        "answer": answer_text(response),
    }


class GenerationStopped(RuntimeError):
    """The LLM half could not be completed (HTTP 429 or the request cap) and fallback was not allowed."""


def eval_leaks(text: str) -> list[str]:
    """Evaluation-internal phrases in ``text`` (empty when the answer reads as a product answer)."""
    return [match.group(0) for pattern in EVAL_LEAK_PATTERNS for match in pattern.finditer(text or "")]


def tool_call_summary(response: dict[str, Any]) -> dict[str, Any]:
    calls = response.get("tool_calls") or []
    errors = [call for call in calls if not call.get("ok")]
    return {
        "calls": len(calls),
        "errors": [
            {
                "tool": call.get("tool"),
                "code": (call.get("error") or {}).get("code"),
                "message": (call.get("error") or {}).get("message"),
            }
            for call in errors
        ],
    }


def generate(
    items: list[dict[str, Any]], *, llm: Any, max_requests: int, allow_fallback: bool = False, progress=print
) -> dict[str, Any]:
    """Answer every item with the offline tools called directly (no replay snapshot, so no replay gaps).

    Without ``allow_fallback``, HTTP 429 or the request cap raises ``GenerationStopped`` at that item.
    """
    model = getattr(llm, "model", None) if llm is not None else None
    stop_reason: str | None = None if llm is not None else "no LLM configured (--llm none)"
    if stop_reason and not allow_fallback and any(item["planned_path"] == "llm_agent" for item in items):
        raise GenerationStopped(f"{stop_reason}: the LLM half needs --llm deepseek (or --allow-fallback)")

    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.state import AgentConfig
    from query_intelligence.agent.tools import build_registry_for_service

    service = build_offline_service()
    registry = build_registry_for_service(service)
    agents: dict[bool, Any] = {}

    def agent_for(with_llm: bool) -> Any:
        if with_llm not in agents:
            runtime = AgentRuntime(
                service, registry, llm if with_llm else None, config=AgentConfig(), today=lambda: EVAL_TODAY
            )
            agents[with_llm] = AgentService(runtime, trace_sinks=[])
        return agents[with_llm]

    rows = []
    # Deterministic items first (fast), then LLM items one at a time.
    ordered = sorted(items, key=lambda item: item["planned_path"] == "llm_agent")
    try:
        for count, item in enumerate(ordered, start=1):
            query = item["task"]["turns"][0]["query"]
            use_llm = item["planned_path"] == "llm_agent" and stop_reason is None
            if use_llm and _http_requests(llm) + REQUESTS_HEADROOM > max_requests:
                stop_reason = f"LLM request cap reached ({_http_requests(llm)} of {max_requests})"
                if not allow_fallback:
                    raise GenerationStopped(f"{stop_reason} at {item['id']} ({count - 1}/{len(ordered)} answered)")
                use_llm = False
            mode = "agent" if use_llm else "auto"
            before_429 = _http_429(llm)
            started = time.perf_counter()
            response = agent_for(use_llm).chat(query, session_id=f"label-{item['id']}", mode=mode)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            degraded = response.get("degraded") or []
            hit_429 = _http_429(llm) > before_429 or any("HTTP 429" in str(flag) for flag in degraded)
            if use_llm and hit_429:
                stop_reason = "HTTP 429 from the gateway"
                if not allow_fallback:
                    raise GenerationStopped(f"{stop_reason} at {item['id']} ({count - 1}/{len(ordered)} answered)")
            row = meta_row(
                item,
                response,
                path="llm_agent" if use_llm else "deterministic",
                mode=mode,
                model=model if use_llm else None,
                fallback_reason=None,
                latency_ms=latency_ms,
            )
            row["fallback_reason"] = fallback_reason(row, stop_reason)
            row["tools"] = tool_call_summary(response)
            rows.append(row)
            if progress and (count % 10 == 0 or use_llm):
                progress(f"{count}/{len(ordered)} {item['id']} {mode} requests={_http_requests(llm)} {query[:30]}")
    finally:
        for agent in agents.values():
            agent.close()
    by_id = {row["id"]: row for row in rows}
    return {
        "rows": [by_id[item["id"]] for item in items],
        "stop_reason": stop_reason,
        "model": model,
        "http": llm.http_stats() if llm is not None and hasattr(llm, "http_stats") else {},
    }


def check_rows(meta: list[dict[str, Any]], csv: list[dict[str, Any]]) -> list[str]:
    """Problems that make the label set unusable: evaluation wording in a shown cell, or a replay gap."""
    problems = []
    for row in csv:
        for column in ("answer", "sources"):
            leaks = eval_leaks(row[column])
            if leaks:
                problems.append(f"{row['id']} {column} mentions evaluation internals: {sorted(set(leaks))}")
    for row in meta:
        misses = [error for error in (row.get("tools") or {}).get("errors", []) if REPLAY_MISS in str(error)]
        if misses:
            problems.append(f"{row['id']} has {len(misses)} tool call(s) that hit a replay gap")
    return problems


NO_SOURCES = "（本回答没有引用证据）"


def sources_text(row: dict[str, Any]) -> str:
    """The cited evidence of one answer, one line each, in citation order."""
    summaries = {entry["evidence_id"]: entry["summary"] for entry in row["evidence"]}
    lines = [summaries.get(eid, f"[{eid}]（该证据的详情未随回答返回）") for eid in row["evidence_used"]]
    return "\n".join(lines) or NO_SOURCES


def csv_rows(meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": row["id"], "question": row["question"], "answer": row["answer"], "sources": sources_text(row)}
        for row in meta
    ]


def _has_labels(path: Path) -> bool:
    if not path.exists():
        return False
    return any(any(row.get(column) for column in LABEL_COLUMNS) for row in read_csv(path))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Generate FinSight answers for human labelling.")
    parser.add_argument("--llm", choices=["none", "deepseek"], default="none")
    parser.add_argument("--model", default="", help="Default: DEEPSEEK_MODEL. Must be a cline-pass/* model.")
    parser.add_argument("--max-llm-requests", type=int, default=120, help="HTTP attempts incl. retries (<=200).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite a CSV that already has labels.")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="On HTTP 429 / the request cap / --llm none, answer LLM items deterministically instead of stopping.",
    )
    args = parser.parse_args(argv)
    if _has_labels(CSV_PATH) and not args.force:
        raise SystemExit(f"{display(CSV_PATH)} already has labels; refusing to overwrite (use --force).")
    llm = _make_llm(args.llm, args.model)
    if llm is not None and not str(getattr(llm, "model", "")).startswith(ALLOWED_MODEL_PREFIX):
        raise SystemExit(f"only {ALLOWED_MODEL_PREFIX}* models may be used, got {getattr(llm, 'model', None)!r}")
    if args.max_llm_requests > 200:
        raise SystemExit("--max-llm-requests must be at most 200")
    config = run_config("evaluation.human.generate_answers", argv)
    items = sample_items(args.seed)
    started = time.perf_counter()
    try:
        result = generate(items, llm=llm, max_requests=args.max_llm_requests, allow_fallback=args.allow_fallback)
    except GenerationStopped as exc:
        raise SystemExit(f"generation stopped, nothing written: {exc}") from exc
    meta = result["rows"]
    shown = csv_rows(meta)
    problems = check_rows(meta, shown)
    if problems:
        write_jsonl(REJECTED_PATH, meta)
        raise SystemExit(
            f"generation rejected, label files not written ({len(problems)} problems; rows kept in "
            f"{display(REJECTED_PATH)}):\n  " + "\n  ".join(problems)
        )
    write_jsonl(META_PATH, meta)
    write_csv(CSV_PATH, CSV_COLUMNS, shown)
    paths = [row["path"] for row in meta]
    tool_errors = [error for row in meta for error in row["tools"]["errors"]]
    summary = {
        "config": {
            **config,
            "seed": args.seed,
            "pool": {name: display(TASK_SETS[name][0]) for name in SETS},
            "tools": "offline tools called directly (no replay snapshot, as runner --no-replay)",
            "data": SNAPSHOT_NAME,
            "eval_today": EVAL_TODAY.isoformat(),
            "model": result["model"],
            "max_llm_requests": args.max_llm_requests,
            "wall_seconds": round(time.perf_counter() - started, 1),
        },
        "answers": len(meta),
        "planned": {"deterministic": N_TOTAL - N_LLM, "llm_agent": N_LLM},
        "actual": {"deterministic": paths.count("deterministic"), "llm_agent": paths.count("llm_agent")},
        "llm_fallbacks": [
            {"id": row["id"], "reason": row["fallback_reason"]} for row in meta if row["fallback_reason"]
        ],
        "llm_stop_reason": result["stop_reason"],
        "llm_http": result["http"],
        "llm_calls_recorded": sum(row["llm_calls"] or 0 for row in meta),
        "tool_calls": {
            "total": sum(row["tools"]["calls"] for row in meta),
            "errors": len(tool_errors),
            "errors_by_code": dict(Counter(str(error["code"]) for error in tool_errors).most_common()),
            "replay_gaps": 0,  # check_rows refuses any; every call ran against the offline tools
        },
        "eval_wording_check": "passed: no answer or sources cell matches EVAL_LEAK_PATTERNS",
        "auto_task_success": {
            path: round(
                sum(row["auto"]["task_success"] for row in meta if row["path"] == path) / max(1, paths.count(path)),
                4,
            )
            for path in ("deterministic", "llm_agent")
        },
        "files": {
            "csv": display(CSV_PATH),
            "csv_sha256": sha256_file(CSV_PATH),
            "meta": display(META_PATH),
            "meta_sha256": sha256_file(META_PATH),
        },
    }
    write_result(GENERATION_PATH, summary)
    print({key: summary[key] for key in ("answers", "actual", "llm_stop_reason", "llm_calls_recorded")})
    return summary


if __name__ == "__main__":
    main()
