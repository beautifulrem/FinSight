"""Run the agent evaluation task set.

Examples::

    # offline, deterministic (no LLM): planner + template composer over the replay snapshot
    python -m evaluation.agent_eval.runner --mode workflow

    # online, with the configured OpenAI-compatible LLM (DEEPSEEK_API_KEY), three repeats for pass^3
    python -m evaluation.agent_eval.runner --mode agent --llm deepseek --repeats 3

    # re-record the tool snapshot from the offline runtime assets
    python -m evaluation.agent_eval.runner --mode workflow --record

Modes: ``workflow`` (deterministic plan; LLM only composes when one is configured), ``agent`` (LLM tool
loop), ``auto`` (router decides), ``pure_llm`` (LLM answers without any tools; requires an LLM).
Results are written to ``outputs/agent_eval/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from query_intelligence.agent.composer import parse_answer
from query_intelligence.agent.evidence import EvidenceStore
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import LLMClient, LLMError
from query_intelligence.agent.prompts import ANSWER_CONTRACT
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.state import AgentConfig
from query_intelligence.agent.tools import ToolRegistry, build_registry_for_service
from query_intelligence.agent.verifier import verify_answer

from .metrics import aggregate, breakdown, failed_checks, score_turn
from .replay import RecordingRegistry, ReplayRegistry

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS = EVAL_DIR / "tasks" / "agent_eval_v1.jsonl"
DEFAULT_SNAPSHOT = EVAL_DIR / "fixtures" / "snapshot_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "agent_eval"
SNAPSHOT_NAME = "offline-runtime-assets (market/fundamental/macro seed as of 2026-04-22)"
EVAL_TODAY = date(2026, 4, 23)  # fixed "today" so freshness notes are reproducible

PURE_LLM_SYSTEM = (
    "You are a financial assistant for China-market questions. Answer from your own knowledge; no tools are "
    f"available. Do not give buy/sell instructions. Answer in the user's language. {ANSWER_CONTRACT}"
)


def load_tasks(path: str | Path = DEFAULT_TASKS) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_offline_service():
    from query_intelligence.service import build_default_service

    return build_default_service(
        use_live_market=False, use_live_macro=False, use_live_news=False, use_live_announcement=False
    )


def build_registry(
    service: Any, *, snapshot: Path | None, record: bool, live_fallback: bool = False
) -> tuple[ToolRegistry, RecordingRegistry | ReplayRegistry | None]:
    live = build_registry_for_service(service)
    if record:
        recorder = RecordingRegistry(live)
        return recorder, recorder
    if snapshot is not None and snapshot.exists():
        replay = ReplayRegistry.from_file(snapshot, live.specs(), fallback=live if live_fallback else None)
        return replay, replay
    return live, None


def run_agent_tasks(
    tasks: list[dict[str, Any]],
    agent: AgentService,
    *,
    mode: str,
    repeats: int = 1,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for index, task in enumerate(tasks, start=1):
        for repeat in range(repeats):
            session = f"eval-{task['id']}-{mode}-{repeat}"
            turns = []
            for turn in task["turns"]:
                started = time.perf_counter()
                response = agent.chat(turn["query"], session_id=session, mode=mode)
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                turns.append(_turn_record(turn, response, latency_ms))
            records.append({"task": _task_meta(task), "repeat": repeat, "turns": turns})
        if progress and index % 25 == 0:
            progress(f"{index}/{len(tasks)} tasks")
    return records


def run_pure_llm_tasks(tasks: list[dict[str, Any]], llm: LLMClient, *, repeats: int = 1) -> list[dict[str, Any]]:
    """Baseline without tools: the model answers from parametric knowledge only."""
    records = []
    for task in tasks:
        for repeat in range(repeats):
            history: list[dict[str, Any]] = []
            turns = []
            for turn in task["turns"]:
                messages = [
                    {"role": "system", "content": PURE_LLM_SYSTEM},
                    *history,
                    {"role": "user", "content": turn["query"]},
                ]
                started = time.perf_counter()
                try:
                    reply = llm.chat(messages, json_mode=True)
                    draft = parse_answer(reply.content)
                    usage = reply.usage.model_dump()
                except LLMError as exc:
                    draft = {"answer": "", "key_points": [], "evidence_used": [], "limitations": [str(exc)]}
                    usage = {}
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                report = verify_answer(draft, EvidenceStore(), query=turn["query"])
                response = {
                    **draft,
                    "route": "pure_llm",
                    "tool_calls": [],
                    "verification": report.model_dump(),
                    "risk_disclaimer": "",
                    "llm": {"calls": 1, "usage": usage, "model": getattr(llm, "model", None)},
                }
                turns.append(_turn_record(turn, response, latency_ms))
                history += [
                    {"role": "user", "content": turn["query"]},
                    {"role": "assistant", "content": draft["answer"]},
                ]
            records.append({"task": _task_meta(task), "repeat": repeat, "turns": turns})
    return records


def summarize(records: list[dict[str, Any]], *, config: dict[str, Any], repeats: int) -> dict[str, Any]:
    failures = [
        {
            "task": record["task"]["id"],
            "category": record["task"]["category"],
            "query": turn["query"],
            "failed_checks": [name for name, passed in turn["score"]["checks"].items() if not passed],
            "route": turn["score"]["route"],
            "tools_used": turn["score"]["tools_used"],
            "answer_excerpt": turn["answer_excerpt"],
        }
        for record in records
        for turn in record["turns"]
        if not turn["score"]["success"]
    ]
    return {
        "config": config,
        "summary": aggregate(records, repeats=repeats),
        "by_category": breakdown(records, "category"),
        "by_language": breakdown(records, "language"),
        "failed_checks": failed_checks(records),
        "failures": failures,
        "records": records,
    }


def _turn_record(turn: dict[str, Any], response: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    return {
        "query": turn["query"],
        "latency_ms": latency_ms,
        "score": score_turn(response, turn["expect"]),
        "answer_excerpt": str(response.get("answer") or (response.get("clarification") or {}).get("question") or "")[
            :300
        ],
    }


def _task_meta(task: dict[str, Any]) -> dict[str, Any]:
    return {key: task.get(key) for key in ("id", "category", "language")}


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _make_llm(kind: str) -> LLMClient | None:
    if kind == "none":
        return None
    from query_intelligence.agent.llm import DeepSeekToolClient
    from query_intelligence.chatbot import load_chatbot_config

    client = DeepSeekToolClient.from_chatbot_config(load_chatbot_config())
    if not client.configured:
        raise SystemExit("--llm deepseek requires DEEPSEEK_API_KEY (or deepseek.api_key in config/app_config.json)")
    return client


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run the FinSight agent evaluation.")
    parser.add_argument("--mode", choices=["workflow", "agent", "auto", "pure_llm"], default="workflow")
    parser.add_argument("--llm", choices=["none", "deepseek"], default="none")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--no-replay", action="store_true", help="Use the offline tools directly.")
    parser.add_argument("--record", action="store_true", help="Record a new tool snapshot to --snapshot.")
    parser.add_argument("--live-fallback", action="store_true", help="Run and record calls missing from the snapshot.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    if args.category:
        tasks = [task for task in tasks if task["category"] in set(args.category)]
    if args.limit:
        tasks = tasks[: args.limit]
    llm = _make_llm(args.llm)
    started = time.perf_counter()

    if args.mode == "pure_llm":
        if llm is None:
            raise SystemExit("pure_llm mode needs --llm deepseek")
        records = run_pure_llm_tasks(tasks, llm, repeats=args.repeats)
        snapshot_info = None
    else:
        service = build_offline_service()
        snapshot = None if args.no_replay else Path(args.snapshot)
        registry, holder = build_registry(
            service, snapshot=snapshot, record=args.record, live_fallback=args.live_fallback
        )
        runtime = AgentRuntime(service, registry, llm, config=AgentConfig(), today=lambda: EVAL_TODAY)
        agent = AgentService(runtime, trace_sinks=[])
        records = run_agent_tasks(tasks, agent, mode=args.mode, repeats=args.repeats, progress=print)
        agent.close()
        if args.record and isinstance(holder, RecordingRegistry):
            holder.save(args.snapshot, snapshot=SNAPSHOT_NAME)
        snapshot_info = {
            "path": str(Path(args.snapshot).relative_to(ROOT)) if not args.no_replay else None,
            "recorded": args.record,
            "misses": len(holder.misses) if isinstance(holder, ReplayRegistry) else None,
        }

    config = {
        "mode": args.mode,
        "llm": getattr(llm, "model", None) if llm else None,
        "tasks_file": str(Path(args.tasks).relative_to(ROOT)) if Path(args.tasks).is_relative_to(ROOT) else args.tasks,
        "snapshot": snapshot_info,
        "repeats": args.repeats,
        "commit": _git_commit(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "eval_today": EVAL_TODAY.isoformat(),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "command": "python -m evaluation.agent_eval.runner " + " ".join(argv if argv is not None else []),
    }
    report = summarize(records, config=config, repeats=args.repeats)
    out = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / f"{args.mode}{'-' + args.llm if llm else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"out": str(out), **report["summary"]}, ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main()
