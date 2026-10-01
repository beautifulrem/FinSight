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
import functools
import json
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from query_intelligence.agent.composer import parse_answer
from query_intelligence.agent.evidence import EvidenceStore
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import LLMClient, LLMError, Pricing, Usage, resolve_cost
from query_intelligence.agent.prompts import ANSWER_CONTRACT, prompt_refs
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.state import AgentConfig
from query_intelligence.agent.tools import ToolRegistry, build_registry_for_service
from query_intelligence.agent.verifier import verify_answer
from query_intelligence.chat.language import detect_user_language

from .metrics import aggregate, breakdown, failed_checks, score_turn, task_outcomes
from .replay import RecordingRegistry, ReplayRegistry

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS = EVAL_DIR / "tasks" / "agent_eval_v1.jsonl"
DEFAULT_SNAPSHOT = EVAL_DIR / "fixtures" / "snapshot_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "agent_eval"
# name -> (task file, tool snapshot). ``test_v2`` is the untouched test set: never used for tuning.
TASK_SETS: dict[str, tuple[Path, Path]] = {
    "dev": (DEFAULT_TASKS, DEFAULT_SNAPSHOT),
    "holdout": (EVAL_DIR / "tasks" / "agent_eval_holdout_v1.jsonl", EVAL_DIR / "fixtures" / "snapshot_holdout_v1.json"),
    "test_v2": (EVAL_DIR / "tasks" / "agent_eval_test_v2.jsonl", EVAL_DIR / "fixtures" / "snapshot_test_v2.json"),
    # Written independently (see evaluation/agent_eval/tasks/README_multiturn_v1.md); first run at cf01eef.
    "multiturn_v1": (
        EVAL_DIR / "tasks" / "agent_eval_multiturn_v1.jsonl",
        EVAL_DIR / "fixtures" / "snapshot_multiturn_v1.json",
    ),
    # Written independently (see evaluation/agent_eval/tasks/README_test_v3.md); run once at the final commit.
    "test_v3": (EVAL_DIR / "tasks" / "agent_eval_test_v3.jsonl", EVAL_DIR / "fixtures" / "snapshot_test_v3.json"),
}
SNAPSHOT_NAME = "offline-runtime-assets (market/fundamental/macro seed as of 2026-04-22)"
EVAL_TODAY = date(2026, 4, 23)  # fixed "today" so freshness notes are reproducible

PURE_LLM_SYSTEM = (
    "You are a financial assistant for China-market questions. Answer from your own knowledge; no tools are "
    f"available. Do not give buy/sell instructions. Answer in the user's language. {ANSWER_CONTRACT}"
)


def load_tasks(path: str | Path = DEFAULT_TASKS) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_snapshot_ext() -> bool:
    """Whether the evaluation's offline tools see the snapshot extension (``QI_EVAL_SNAPSHOT_EXT``, default off).

    Every task set and held-out slice was labelled against the v1 snapshot (``data/structured_data.json``),
    including absence labels such as "宁德时代 has no offline price"; the extension (data/snapshot/) adds those
    names, so evaluation stays pinned to v1 unless this is set. Tasks that need the extension are recorded into
    their replay fixture with it on (docs/data-sources.md, "Offline snapshot").
    """
    import os

    return os.getenv("QI_EVAL_SNAPSHOT_EXT", "").strip().lower() in {"1", "true", "yes"}


def build_offline_service(*, snapshot_ext: bool | None = None):
    from query_intelligence.service import build_default_service

    return build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=eval_snapshot_ext() if snapshot_ext is None else snapshot_ext,
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


def map_tasks(
    fn: Callable[[dict[str, Any]], list[dict[str, Any]]],
    tasks: list[dict[str, Any]],
    *,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Apply ``fn`` to every task (optionally on a thread pool) and flatten the records in task order.

    Tasks are independent (each uses its own session id), so they can run concurrently; results are
    always returned in input order so reports do not depend on scheduling.
    """
    results: list[list[dict[str, Any]]] = []
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            results.append(fn(task))
            if progress and index % 25 == 0:
                progress(f"{index}/{len(tasks)} tasks")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, records in enumerate(pool.map(fn, tasks), start=1):
                results.append(records)
                if progress and index % 25 == 0:
                    progress(f"{index}/{len(tasks)} tasks")
    return [record for records in results for record in records]


def streamed_chat(
    agent: AgentService, query: str, *, session_id: str, mode: str
) -> tuple[dict[str, Any], float | None]:
    """Run one turn through ``AgentService.stream`` (the SSE path) and return ``(response, ttft_ms)``.

    ``ttft_ms`` is the time from the call to the first ``answer_delta`` event, i.e. the first answer text a
    streaming client can show; ``None`` when the turn streamed no answer text (template answers, refusals,
    clarifications). The response has the same shape as ``AgentService.chat``.
    """
    started = time.perf_counter()
    ttft_ms: float | None = None
    response: dict[str, Any] | None = None
    for event in agent.stream(query, session_id=session_id, mode=mode):
        kind, data = event.get("event"), event.get("data") or {}
        if kind == "answer_delta" and ttft_ms is None and data.get("text"):
            ttft_ms = round((time.perf_counter() - started) * 1000, 2)
        elif kind == "answer":
            response = dict(data)
        elif kind == "clarification":
            payload = {key: value for key, value in data.items() if key != "session_id"}
            response = {"status": "needs_clarification", "session_id": session_id, "clarification": payload}
        elif kind == "error":
            raise RuntimeError(f"agent stream failed: {data.get('message')}")
    if response is None:
        raise RuntimeError("agent stream ended without an answer or clarification event")
    return response, ttft_ms


def run_agent_tasks(
    tasks: list[dict[str, Any]],
    agent: AgentService,
    *,
    mode: str,
    repeats: int = 1,
    progress: Callable[[str], None] | None = None,
    workers: int = 1,
    stream: bool = False,
) -> list[dict[str, Any]]:
    """``stream=True`` runs every turn through the streaming path and records time to first answer token."""

    def run_task(task: dict[str, Any]) -> list[dict[str, Any]]:
        records = []
        for repeat in range(repeats):
            session = f"eval-{task['id']}-{mode}-{repeat}"
            turns = []
            for turn in task["turns"]:
                started = time.perf_counter()
                ttft_ms = None
                if stream:
                    response, ttft_ms = streamed_chat(agent, turn["query"], session_id=session, mode=mode)
                else:
                    response = agent.chat(turn["query"], session_id=session, mode=mode)
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                turns.append(_turn_record(turn, response, latency_ms, ttft_ms=ttft_ms))
            records.append({"task": _task_meta(task), "repeat": repeat, "turns": turns})
        return records

    return map_tasks(run_task, tasks, workers=workers, progress=progress)


def run_pure_llm_tasks(
    tasks: list[dict[str, Any]], llm: LLMClient, *, repeats: int = 1, workers: int = 1
) -> list[dict[str, Any]]:
    """Baseline without tools: the model answers from parametric knowledge only."""

    def run_task(task: dict[str, Any]) -> list[dict[str, Any]]:
        records = []
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
                degraded: list[str] = []
                try:
                    reply = llm.chat(messages, json_mode=True)
                    draft = parse_answer(reply.content)
                    usage = reply.usage.model_dump()
                except LLMError as exc:
                    draft = {"answer": "", "key_points": [], "evidence_used": [], "limitations": [str(exc)]}
                    usage = {}
                    degraded = [f"llm_error:{exc}"]
                cost, currency, _source = resolve_cost(Usage(**usage), Pricing.from_env())
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                report = verify_answer(draft, EvidenceStore(), query=turn["query"])
                answer = str(draft.get("answer") or "")
                response = {
                    **draft,
                    "route": "pure_llm",
                    # The answer's own language, so the language check scores the model and not a missing field.
                    "language": detect_user_language(answer) if answer.strip() else None,
                    "degraded": degraded,
                    "tool_calls": [],
                    "verification": report.model_dump(),
                    "risk_disclaimer": "",
                    "llm": {
                        "calls": 1,
                        "usage": usage,
                        "model": getattr(llm, "model", None),
                        "cost": cost,
                        "currency": currency,
                    },
                }
                turns.append(_turn_record(turn, response, latency_ms))
                history += [
                    {"role": "user", "content": turn["query"]},
                    {"role": "assistant", "content": draft["answer"]},
                ]
            records.append({"task": _task_meta(task), "repeat": repeat, "turns": turns})
        return records

    return map_tasks(run_task, tasks, workers=workers)


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
        "task_outcomes": task_outcomes(records),
        "records": records,
    }


def _turn_record(
    turn: dict[str, Any], response: dict[str, Any], latency_ms: float, *, ttft_ms: float | None = None
) -> dict[str, Any]:
    record = {
        "query": turn["query"],
        "latency_ms": latency_ms,
        "score": score_turn(response, turn["expect"]),
        "answer_excerpt": str(response.get("answer") or (response.get("clarification") or {}).get("question") or "")[
            :300
        ],
        "profile": turn_profile(response),
    }
    if ttft_ms is not None:
        record["ttft_ms"] = ttft_ms
    return record


_LLM_PROFILE_KEYS = (
    "node",
    "step",
    "model",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "prompt_cache_hit_tokens",
    "reasoning_tokens",
    "context_chars",
    "tool_calls",
    "finish_reason",
    "trigger",
)


def turn_profile(response: dict[str, Any]) -> dict[str, Any]:
    """Where the time of one turn went: every LLM call (node, latency, tokens, context composition), every
    tool call (latency, step, cache) and every graph node (duration). Read by ``evaluation.agent_eval.profile``."""
    llm = response.get("llm") or {}
    return {
        "route": response.get("route"),
        "answer_source": response.get("answer_source"),
        "degraded": response.get("degraded") or [],
        "llm_calls": [{key: entry.get(key) for key in _LLM_PROFILE_KEYS} for entry in llm.get("log") or []],
        "tools": [
            {key: call.get(key) for key in ("tool", "ok", "latency_ms", "step", "source", "cached")}
            for call in response.get("tool_calls") or []
        ],
        "spans": [
            {"node": span.get("node"), "duration_ms": span.get("duration_ms")} for span in response.get("spans") or []
        ],
    }


def _task_meta(task: dict[str, Any]) -> dict[str, Any]:
    return {key: task.get(key) for key in ("id", "category", "language")}


def _display_path(value: str | Path) -> str:
    path = Path(value).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _command(module: str, argv: list[str] | None) -> str:
    """The command line as run, with paths inside the repository made relative so results are portable."""
    args = argv if argv is not None else sys.argv[1:]
    shown = [_display_path(arg) if arg.startswith(str(ROOT)) else arg for arg in args]
    return f"python -m {module} " + " ".join(shown)


# (round 12) Environment variables that change what a run measures: every QI_* switch (QI_AGENT_PREFETCH,
# QI_PROMPT_VERSION, QI_AGENT_FRAME_FALLBACK, …) and the model / failover ids. Names that can hold a credential or a
# connection string are never recorded.
_ENV_PREFIXES = ("QI_",)
_ENV_NAMES = ("DEEPSEEK_MODEL",)
_ENV_SECRET = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "DSN", "URL", "URI", "AUTH", "COOKIE")


def env_toggles(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The run's environment switches (``QI_*``, ``DEEPSEEK_MODEL``), sorted, without anything secret-shaped."""
    import os

    source = os.environ if environ is None else environ
    return {
        name: value
        for name, value in sorted(source.items())
        if (name.startswith(_ENV_PREFIXES) or name in _ENV_NAMES)
        and not any(marker in name.upper() for marker in _ENV_SECRET)
    }


def command_with_env(command: str, env: dict[str, str]) -> str:
    """``QI_X=1 QI_Y=off python -m …``: the command as it has to be typed to repeat the run."""
    import shlex

    prefix = " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())
    return f"{prefix} {command}".strip()


def command_fields(module: str, argv: list[str] | None) -> dict[str, Any]:
    """``command`` (as before), ``env`` (the switches in effect) and ``command_with_env`` for a result's config."""
    command = _command(module, argv)
    env = env_toggles()
    return {"command": command, "env": env, "command_with_env": command_with_env(command, env)}


@functools.cache
def _git_commit() -> str | None:
    """Commit the evaluation code ran from, suffixed ``-dirty`` when the working tree had changes.

    Cached: every ``main`` calls it before running, so a commit made while a long online run is in
    progress is not attributed to that run.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if status else commit


def agent_config_from_overrides(pairs: list[str]) -> AgentConfig:
    """``AgentConfig`` with ``FIELD=VALUE`` overrides (values parsed by the field's default type; ``none`` → None)."""
    import dataclasses

    base = AgentConfig()
    fields = {item.name: getattr(base, item.name) for item in dataclasses.fields(AgentConfig)}
    updates: dict[str, Any] = {}
    for pair in pairs:
        name, sep, raw = pair.partition("=")
        name = name.strip()
        if not sep or name not in fields:
            raise SystemExit(f"--agent-config expects FIELD=VALUE with a known field, got {pair!r}")
        current = fields[name]
        value: Any = raw.strip()
        if value.lower() in {"none", "null"}:
            value = None
        elif isinstance(current, bool):
            value = value.lower() in {"1", "true", "on", "yes"}
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        updates[name] = value
    return dataclasses.replace(base, **updates)


LLM_HELP = (
    "LLM client. 'deepseek' is the OpenAI-compatible client configured by DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL; "
    "it names the client, not the model."
)
MODEL_HELP = (
    "Model id sent to the client (e.g. cline-pass/glm-5.3-flash). Defaults to DEEPSEEK_MODEL, else "
    "deepseek.model in the config. Passing it here puts the model into the recorded command."
)


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm", choices=["none", "deepseek"], default="none", help=LLM_HELP)
    parser.add_argument("--model", default="", help=MODEL_HELP)


def _make_llm(kind: str, model: str = "") -> LLMClient | None:
    if kind == "none":
        return None
    from query_intelligence.agent.llm import build_llm_from_config
    from query_intelligence.chatbot import load_chatbot_config

    config = load_chatbot_config()
    if model:
        config = {**config, "deepseek": {**(config.get("deepseek") or {}), "model": model}}
    client = build_llm_from_config(config)
    if client is None:
        raise SystemExit("--llm deepseek requires DEEPSEEK_API_KEY (or deepseek.api_key in config/app_config.json)")
    return client


def llm_config(kind: str, model_arg: str, llm: LLMClient | None) -> dict[str, Any]:
    """Which client and model a run used. ``llm`` is kept for older readers; ``model`` is explicit."""
    model = getattr(llm, "model", None) if llm is not None else None
    return {
        "llm": model,
        "model": model,
        "llm_client": kind,
        "model_source": None if llm is None else ("--model" if model_arg else "DEEPSEEK_MODEL / deepseek.model"),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    _git_commit()  # record the commit at start, not when the run finishes
    parser = argparse.ArgumentParser(description="Run the FinSight agent evaluation.")
    parser.add_argument("--mode", choices=["workflow", "agent", "auto", "pure_llm"], default="workflow")
    add_llm_arguments(parser)
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--no-replay", action="store_true", help="Use the offline tools directly.")
    parser.add_argument("--record", action="store_true", help="Record a new tool snapshot to --snapshot.")
    parser.add_argument("--live-fallback", action="store_true", help="Run calls missing from the snapshot.")
    parser.add_argument(
        "--record-missing",
        action="store_true",
        help="Like --live-fallback, and add the missing calls to --snapshot without changing recorded ones.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1, help="Run tasks concurrently (LLM modes).")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    if args.category:
        tasks = [task for task in tasks if task["category"] in set(args.category)]
    if args.limit:
        tasks = tasks[: args.limit]
    llm = _make_llm(args.llm, args.model)
    started = time.perf_counter()

    if args.mode == "pure_llm":
        if llm is None:
            raise SystemExit("pure_llm mode needs --llm deepseek")
        records = run_pure_llm_tasks(tasks, llm, repeats=args.repeats, workers=args.workers)
        snapshot_info = None
    else:
        service = build_offline_service()
        snapshot = None if args.no_replay else Path(args.snapshot)
        registry, holder = build_registry(
            service, snapshot=snapshot, record=args.record, live_fallback=args.live_fallback or args.record_missing
        )
        runtime = AgentRuntime(service, registry, llm, config=AgentConfig(), today=lambda: EVAL_TODAY)
        agent = AgentService(runtime, trace_sinks=[])
        records = run_agent_tasks(
            tasks, agent, mode=args.mode, repeats=args.repeats, progress=print, workers=args.workers
        )
        agent.close()
        if args.record and isinstance(holder, RecordingRegistry):
            holder.save(args.snapshot, snapshot=SNAPSHOT_NAME)
        if args.record_missing and isinstance(holder, ReplayRegistry):
            print(f"added {holder.save_missing(args.snapshot)} calls to {_display_path(args.snapshot)}")
        snapshot_info = {
            "path": _display_path(args.snapshot) if not args.no_replay else None,
            "recorded": args.record,
            "misses": len(holder.misses) if isinstance(holder, ReplayRegistry) else None,
        }

    config = {
        "mode": args.mode,
        **llm_config(args.llm, args.model, llm),
        "tasks_file": _display_path(args.tasks),
        "snapshot": snapshot_info,
        "repeats": args.repeats,
        "prompts": prompt_refs(),
        "commit": _git_commit(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "eval_today": EVAL_TODAY.isoformat(),
        "wall_seconds": round(time.perf_counter() - started, 1),
        **command_fields("evaluation.agent_eval.runner", argv),
    }
    report = summarize(records, config=config, repeats=args.repeats)
    out = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / f"{args.mode}{'-' + args.llm if llm else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"out": str(out), **report["summary"]}, ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main()
