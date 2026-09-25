"""Ablation: compare answer paths on the same tasks.

Offline (no LLM key needed):

* ``legacy``   – the original ``/chat`` path (Query Intelligence pipeline + template answer, as served
  when no LLM key is configured). It has no refusal/clarification behaviour of its own, so its
  behaviour is inferred from the NLU flags (out-of-scope -> refuse, missing entity -> clarify); this
  is generous to the legacy path. Tools are inferred from ``executed_sources``.
* ``workflow`` – the agent's deterministic planner + template composer with verification and
  compliance.

Online (``--llm deepseek``, needs ``DEEPSEEK_API_KEY``): ``pure_llm`` (no tools), ``workflow_llm``
(planner + LLM composition) and ``agent`` (LLM tool loop), with ``--repeats`` for pass^k.
``legacy_llm`` (the original ``/chat`` client) always runs once per task, so it reports pass^1.

    python -m evaluation.agent_eval.ablation                      # offline: legacy vs workflow
    python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3
    python -m evaluation.agent_eval.ablation --sets dev,holdout,test_v2

Every mode stores per-task outcomes (``task_outcomes``: success of each repeat), and each set gets
paired comparisons between paths (paired bootstrap over tasks + exact McNemar on pass^k).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.prompts import prompt_refs
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.verifier import verify_answer
from query_intelligence.chatbot import DeepSeekClient, build_chatbot_response

from .metrics import paired_comparison
from .runner import (
    DEFAULT_OUTPUT_DIR,
    EVAL_TODAY,
    TASK_SETS,
    _display_path,
    _git_commit,
    _make_llm,
    _task_meta,
    _turn_record,
    build_offline_service,
    build_registry,
    load_tasks,
    map_tasks,
    run_agent_tasks,
    run_pure_llm_tasks,
    summarize,
)

HOLDOUT_TASKS, HOLDOUT_SNAPSHOT = TASK_SETS["holdout"]
# Paths compared on every set (a minus b), when both ran.
COMPARISONS = (("agent", "workflow_llm"), ("agent", "workflow"), ("workflow_llm", "workflow"))
_SOURCE_TO_TOOL = {
    "market_api": "get_price_history",
    "fundamental_sql": "get_fundamentals",
    "industry_sql": "get_fundamentals",
    "macro_sql": "get_macro_indicators",
    "macro_indicator": "get_macro_indicators",
    "news": "search_news",
    "announcement": "search_announcements",
    "research_note": "search_knowledge",
    "faq": "search_knowledge",
    "product_doc": "search_knowledge",
}


def legacy_response(
    service: Any, query: str, dialog_context: list[dict[str, Any]], client: DeepSeekClient | None = None
) -> dict[str, Any]:
    pipeline = service.run_pipeline(query, dialog_context=dialog_context, top_k=10)
    client = client or DeepSeekClient({"deepseek": {"api_key": ""}})
    # The freshness guard reads "today"; pin it to the evaluation date like the agent paths (runs on a
    # weekend used to take the non-trading-day branch and drop the hedging sentence).
    answer = build_chatbot_response(
        query=query, pipeline_result=pipeline, deepseek_client=client, as_of_date=EVAL_TODAY
    )
    nlu, retrieval = pipeline["nlu_result"], pipeline["retrieval_result"]
    flags = set(nlu.get("risk_flags") or [])
    route = "workflow"
    if "out_of_scope_query" in flags:
        route = "refuse"
    elif "missing_entity" in (nlu.get("missing_slots") or []) and not nlu.get("entities"):
        route = "clarify"
    store = EvidenceStore()
    for document in retrieval.get("documents") or []:
        store.add(AgentEvidence.from_document(document))
    structured = retrieval.get("structured_data") or []
    for item in structured:
        store.add(AgentEvidence.from_structured(item))
    tools = [
        {"tool": tool, "ok": True}
        for tool in dict.fromkeys(_SOURCE_TO_TOOL.get(source) for source in retrieval.get("executed_sources") or [])
        if tool
    ]
    if any((item.get("payload") or {}).get("_market_analysis") for item in structured):
        tools.append({"tool": "compute_indicators", "ok": True})
    return {
        **answer,
        "route": route,
        "tool_calls": tools,
        "verification": verify_answer(answer, store, query=query).model_dump(),
        "nlu_summary": {
            "entities": [
                {"name": entity.get("canonical_name"), "symbol": entity.get("symbol")}
                for entity in nlu.get("entities") or []
            ]
        },
        "llm": {"calls": 0, "usage": {}},
    }


def run_legacy_tasks(
    tasks: list[dict[str, Any]], service: Any, client: DeepSeekClient | None = None, *, workers: int = 1
) -> list[dict[str, Any]]:
    def run_task(task: dict[str, Any]) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        turns = []
        for turn in task["turns"]:
            started = time.perf_counter()
            response = legacy_response(service, turn["query"], context, client)
            turns.append(_turn_record(turn, response, round((time.perf_counter() - started) * 1000, 2)))
            context.append({"role": "user", "content": turn["query"]})
        return [{"task": _task_meta(task), "repeat": 0, "turns": turns}]

    return map_tasks(run_task, tasks, workers=workers)


def _agent_records(
    service, tasks, snapshot: Path, *, mode: str, llm=None, repeats: int = 1, workers: int = 1
) -> list[dict[str, Any]]:
    registry, _holder = build_registry(service, snapshot=snapshot, record=False, live_fallback=llm is not None)
    runtime = AgentRuntime(service, registry, llm, today=lambda: EVAL_TODAY)
    agent = AgentService(runtime, trace_sinks=[])
    try:
        return run_agent_tasks(tasks, agent, mode=mode, repeats=repeats, workers=workers)
    finally:
        agent.close()


ONLINE_MODES = ("legacy_llm", "pure_llm", "workflow_llm", "agent")
# Deterministic paths run once; ``legacy_llm`` also runs once (its client predates repeats).
SINGLE_RUN_MODES = frozenset({"legacy", "workflow", "legacy_llm"})


def comparisons(modes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        f"{a}_vs_{b}": paired_comparison(modes[a]["task_outcomes"], modes[b]["task_outcomes"])
        for a, b in COMPARISONS
        if a in modes and b in modes
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    _git_commit()  # record the commit at start, not when the run finishes
    parser = argparse.ArgumentParser(description="Ablation over answer paths.")
    parser.add_argument("--llm", choices=["none", "deepseek"], default="none")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Run tasks concurrently in LLM modes.")
    parser.add_argument(
        "--sets", default="dev,holdout", help="Comma-separated task sets: dev, holdout, test_v2 (untouched test set)."
    )
    parser.add_argument(
        "--modes", default=",".join(ONLINE_MODES), help="Online modes to run with --llm (comma-separated)."
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR / "ablation.json"))
    args = parser.parse_args(argv)
    online_modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = sorted(set(online_modes) - set(ONLINE_MODES))
    if unknown:
        raise SystemExit(f"unknown --modes: {unknown}")

    llm = _make_llm(args.llm)
    service = build_offline_service()
    requested = [item.strip() for item in args.sets.split(",") if item.strip()]
    unknown_sets = sorted(set(requested) - set(TASK_SETS))
    if unknown_sets:
        raise SystemExit(f"unknown --sets: {unknown_sets} (known: {sorted(TASK_SETS)})")
    sets = {name: (load_tasks(TASK_SETS[name][0]), TASK_SETS[name][1]) for name in requested}
    results: dict[str, dict[str, Any]] = {}
    for set_name, (tasks, snapshot) in sets.items():
        runs: dict[str, list[dict[str, Any]]] = {
            "legacy": run_legacy_tasks(tasks, service),
            "workflow": _agent_records(service, tasks, snapshot, mode="workflow"),
        }
        if llm is not None:
            from query_intelligence.chatbot import load_chatbot_config

            workers = args.workers
            if "legacy_llm" in online_modes:
                client = DeepSeekClient(load_chatbot_config())
                runs["legacy_llm"] = run_legacy_tasks(tasks, service, client, workers=workers)
            if "pure_llm" in online_modes:
                runs["pure_llm"] = run_pure_llm_tasks(tasks, llm, repeats=args.repeats, workers=workers)
            if "workflow_llm" in online_modes:
                runs["workflow_llm"] = _agent_records(
                    service, tasks, snapshot, mode="workflow", llm=llm, repeats=args.repeats, workers=workers
                )
            if "agent" in online_modes:
                runs["agent"] = _agent_records(
                    service, tasks, snapshot, mode="agent", llm=llm, repeats=args.repeats, workers=workers
                )
        results[set_name] = {
            name: summarize(records, config={"mode": name}, repeats=1 if name in SINGLE_RUN_MODES else args.repeats)
            for name, records in runs.items()
        }

    report = {
        "config": {
            "llm": getattr(llm, "model", None),
            "repeats": args.repeats,
            "prompts": prompt_refs(),
            "commit": _git_commit(),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "eval_today": EVAL_TODAY.isoformat(),
            "task_files": {name: _display_path(TASK_SETS[name][0]) for name in sets},
            "snapshots": {name: _display_path(TASK_SETS[name][1]) for name in sets},
            "sets": list(sets),
            "online_modes": online_modes if llm is not None else [],
            "workers": args.workers,
            "command": "python -m evaluation.agent_eval.ablation "
            + " ".join(argv if argv is not None else sys.argv[1:]),
        },
        "results": {
            set_name: {
                mode: {
                    "summary": data["summary"],
                    "by_category": data["by_category"],
                    "failed_checks": data["failed_checks"],
                    "failures": data["failures"],
                    "task_outcomes": data["task_outcomes"],
                }
                for mode, data in modes.items()
            }
            for set_name, modes in results.items()
        },
        "comparisons": {set_name: comparisons(modes) for set_name, modes in results.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for set_name, modes in report["results"].items():
        for mode, data in modes.items():
            summary = data["summary"]
            print(
                f"{set_name:8s} {mode:12s} success={summary['task_success']} facts={summary['fact_recall']} "
                f"tool_p={summary['tool_precision']} hedged={summary['hedged_when_required']} "
                f"behavior={summary['behavior_accuracy']} p95={summary['latency_ms_p95']}"
            )
    return report


if __name__ == "__main__":
    main()
