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

    python -m evaluation.agent_eval.ablation                      # offline: legacy vs workflow
    python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3
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
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.verifier import verify_answer
from query_intelligence.chatbot import DeepSeekClient, build_chatbot_response

from .runner import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SNAPSHOT,
    DEFAULT_TASKS,
    EVAL_DIR,
    EVAL_TODAY,
    _display_path,
    _git_commit,
    _make_llm,
    _task_meta,
    _turn_record,
    build_offline_service,
    build_registry,
    load_tasks,
    run_agent_tasks,
    run_pure_llm_tasks,
    summarize,
)

HOLDOUT_TASKS = EVAL_DIR / "tasks" / "agent_eval_holdout_v1.jsonl"
HOLDOUT_SNAPSHOT = EVAL_DIR / "fixtures" / "snapshot_holdout_v1.json"
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
    answer = build_chatbot_response(query=query, pipeline_result=pipeline, deepseek_client=client)
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
    tasks: list[dict[str, Any]], service: Any, client: DeepSeekClient | None = None
) -> list[dict[str, Any]]:
    records = []
    for task in tasks:
        context: list[dict[str, Any]] = []
        turns = []
        for turn in task["turns"]:
            started = time.perf_counter()
            response = legacy_response(service, turn["query"], context, client)
            turns.append(_turn_record(turn, response, round((time.perf_counter() - started) * 1000, 2)))
            context.append({"role": "user", "content": turn["query"]})
        records.append({"task": _task_meta(task), "repeat": 0, "turns": turns})
    return records


def _agent_records(service, tasks, snapshot: Path, *, mode: str, llm=None, repeats: int = 1) -> list[dict[str, Any]]:
    registry, _holder = build_registry(service, snapshot=snapshot, record=False, live_fallback=llm is not None)
    runtime = AgentRuntime(service, registry, llm, today=lambda: EVAL_TODAY)
    agent = AgentService(runtime, trace_sinks=[])
    try:
        return run_agent_tasks(tasks, agent, mode=mode, repeats=repeats)
    finally:
        agent.close()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Ablation over answer paths.")
    parser.add_argument("--llm", choices=["none", "deepseek"], default="none")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR / "ablation.json"))
    args = parser.parse_args(argv)

    llm = _make_llm(args.llm)
    service = build_offline_service()
    sets = {
        "dev": (load_tasks(DEFAULT_TASKS), DEFAULT_SNAPSHOT),
        "holdout": (load_tasks(HOLDOUT_TASKS), HOLDOUT_SNAPSHOT),
    }
    results: dict[str, dict[str, Any]] = {}
    for set_name, (tasks, snapshot) in sets.items():
        runs: dict[str, list[dict[str, Any]]] = {
            "legacy": run_legacy_tasks(tasks, service),
            "workflow": _agent_records(service, tasks, snapshot, mode="workflow"),
        }
        if llm is not None:
            from query_intelligence.chatbot import load_chatbot_config

            runs["legacy_llm"] = run_legacy_tasks(tasks, service, DeepSeekClient(load_chatbot_config()))
            runs["pure_llm"] = run_pure_llm_tasks(tasks, llm, repeats=args.repeats)
            runs["workflow_llm"] = _agent_records(
                service, tasks, snapshot, mode="workflow", llm=llm, repeats=args.repeats
            )
            runs["agent"] = _agent_records(service, tasks, snapshot, mode="agent", llm=llm, repeats=args.repeats)
        results[set_name] = {
            name: summarize(
                records, config={"mode": name}, repeats=args.repeats if name not in {"legacy", "workflow"} else 1
            )
            for name, records in runs.items()
        }

    report = {
        "config": {
            "llm": getattr(llm, "model", None),
            "repeats": args.repeats,
            "commit": _git_commit(),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "eval_today": EVAL_TODAY.isoformat(),
            "dev_tasks": _display_path(DEFAULT_TASKS),
            "holdout_tasks": _display_path(HOLDOUT_TASKS),
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
                }
                for mode, data in modes.items()
            }
            for set_name, modes in results.items()
        },
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
