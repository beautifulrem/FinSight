"""Generate the 100 FinSight answers the owner labels (``labels/answers_to_label.csv``).

Questions are single-turn tasks sampled with a fixed seed from test v3 and the held-out set. About half are
answered on the deterministic path (``mode=auto``, no LLM) and half on the LLM agent path (``mode=agent``),
both over the committed tool snapshots, so FinSight's automatic score (task success, verification) of every
answer is reproducible. The CSV is shown to the owner in a shuffled order without the path or the automatic
score (kept in ``labels/answers_meta.jsonl``) so the labels are blind to both.

    python -m evaluation.human.generate_answers                    # deterministic half only (no key needed)
    source /path/to/llmenv.sh && python -m evaluation.human.generate_answers --llm deepseek

LLM calls are sequential and capped (``--max-llm-requests``, HTTP attempts including retries). On HTTP 429,
or when the cap is reached, the remaining LLM items are answered on the deterministic path and the reason is
recorded per row and in ``labels/generation.json``.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

from ..agent_eval.metrics import llm_failure_flags, score_turn
from ..agent_eval.runner import (
    EVAL_TODAY,
    SNAPSHOT_NAME,
    TASK_SETS,
    _make_llm,
    build_offline_service,
    build_registry,
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
ALLOWED_MODEL_PREFIX = "cline-pass/"
# Worst case of HTTP attempts one agent turn can still make (tool steps + compose + revise, with retries).
REQUESTS_HEADROOM = 8


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


def generate(items: list[dict[str, Any]], *, llm: Any, max_requests: int, progress=print) -> dict[str, Any]:
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.state import AgentConfig

    service = build_offline_service()
    agents: dict[tuple[str, bool], Any] = {}
    misses: dict[str, Any] = {}

    def agent_for(set_name: str, with_llm: bool) -> Any:
        key = (set_name, with_llm)
        if key not in agents:
            registry, holder = build_registry(service, snapshot=TASK_SETS[set_name][1], record=False)
            misses[f"{set_name}{'-llm' if with_llm else ''}"] = holder
            runtime = AgentRuntime(
                service, registry, llm if with_llm else None, config=AgentConfig(), today=lambda: EVAL_TODAY
            )
            agents[key] = AgentService(runtime, trace_sinks=[])
        return agents[key]

    model = getattr(llm, "model", None) if llm is not None else None
    stop_reason: str | None = None if llm is not None else "no LLM configured (--llm none)"
    rows = []
    # Deterministic items first (fast), then LLM items one at a time.
    ordered = sorted(items, key=lambda item: item["planned_path"] == "llm_agent")
    for count, item in enumerate(ordered, start=1):
        query = item["task"]["turns"][0]["query"]
        use_llm = item["planned_path"] == "llm_agent" and stop_reason is None
        if use_llm and _http_requests(llm) + REQUESTS_HEADROOM > max_requests:
            stop_reason = f"LLM request cap reached ({_http_requests(llm)} of {max_requests})"
            use_llm = False
        mode = "agent" if use_llm else "auto"
        before_429 = _http_429(llm)
        started = time.perf_counter()
        response = agent_for(item["set"], use_llm).chat(query, session_id=f"label-{item['id']}", mode=mode)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        hit_429 = _http_429(llm) > before_429 or any("HTTP 429" in str(flag) for flag in response.get("degraded") or [])
        if use_llm and hit_429:
            stop_reason = "HTTP 429 from the gateway"
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
        rows.append(row)
        if progress and (count % 10 == 0 or use_llm):
            progress(f"{count}/{len(ordered)} {item['id']} {mode} requests={_http_requests(llm)} {query[:30]}")
    for agent in agents.values():
        agent.close()
    by_id = {row["id"]: row for row in rows}
    return {
        "rows": [by_id[item["id"]] for item in items],
        "stop_reason": stop_reason,
        "model": model,
        "http": llm.http_stats() if llm is not None and hasattr(llm, "http_stats") else {},
        "snapshot_misses": {name: len(getattr(holder, "misses", []) or []) for name, holder in misses.items()},
    }


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
    parser.add_argument("--max-llm-requests", type=int, default=190, help="HTTP attempts incl. retries (<=200).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite a CSV that already has labels.")
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
    result = generate(items, llm=llm, max_requests=args.max_llm_requests)
    meta = result["rows"]
    write_jsonl(META_PATH, meta)
    write_csv(CSV_PATH, CSV_COLUMNS, csv_rows(meta))
    paths = [row["path"] for row in meta]
    summary = {
        "config": {
            **config,
            "seed": args.seed,
            "pool": {name: display(TASK_SETS[name][0]) for name in SETS},
            "snapshots": {name: display(TASK_SETS[name][1]) for name in SETS},
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
        "snapshot_misses": result["snapshot_misses"],
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
