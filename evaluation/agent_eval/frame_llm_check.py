"""Online check of the comparison frame on the LLM agent path (round 11, G4).

Ten multi-turn sessions written by the author (zh/en): a metric for one target, the same metric for another ("X呢"),
then a gap, ratio or relative difference. Every turn runs in ``mode=agent`` with the configured LLM over the offline
tools, sequentially. For the gap turn the run records whether the answer states the expected value, whether the
model stated it itself or the deterministic fallback appended it (``degraded: frame_result_appended``), the route,
verification, LLM calls and latency. It stops at the call budget or on the first HTTP 429.

    python -m evaluation.agent_eval.frame_llm_check --llm deepseek --max-calls 60 \
        --out outputs/agent_eval/frame-llm-check.json

The LLM is configured like the runner (``DEEPSEEK_API_KEY`` / ``DEEPSEEK_BASE_URL`` / ``DEEPSEEK_MODEL``).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.prompts import prompt_refs
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.state import AgentConfig
from query_intelligence.agent.tools import build_registry_for_service

from .runner import (
    EVAL_TODAY,
    _git_commit,
    _make_llm,
    add_llm_arguments,
    build_offline_service,
    command_fields,
)

# (turns, expected value of the last turn's comparison)
SESSIONS: list[tuple[list[str], float]] = [
    (["中国平安的净资产收益率", "五粮液那边呢", "两家差了几个百分点"], 14.2),
    (["茅台的营业收入多少", "再看五粮液的", "前一个是后一个的几倍"], 1.56),
    (["五粮液的市盈率", "白酒行业平均市盈率呢", "折价百分之多少"], 23.44),
    (["What is Wuliangye's P/B?", "and Ping An's?", "how many times bigger is the first?"], 4.91),
    (["证券ETF今天成交额", "那沪深300ETF呢", "哪个大，大多少"], 44.11),
    (["贵州茅台净利率多少", "中国平安呢", "差多少个点"], 38.83),
    (["Moutai ROE?", "and Ping An's, and what's the gap?"], 17.8),
    (["中国平安市净率多少", "茅台呢？两者相差多少"], 7.0),
    (["五粮液的净利润", "茅台呢", "后者比前者多多少亿"], 445.2),
    (["What's Ping An's P/E?", "And the insurance industry average?", "What's the discount in percent?"], 26.27),
]


def _states(text: str, value: float) -> bool:
    """Whether the answer states the expected value, as written or in 亿/万/bn/mn units (445.2 亿 = 44520000000 元)."""
    from query_intelligence.agent.graph import _states_number

    return _states_number(text, value) or _states_number(text, value * 1e8)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    commit = _git_commit()
    parser = argparse.ArgumentParser(description=__doc__)
    add_llm_arguments(parser)
    parser.add_argument("--max-calls", type=int, default=60)
    parser.add_argument("--only", default="", help="comma-separated session indices to run (default: all)")
    parser.add_argument("--out", default="outputs/agent_eval/frame-llm-check.json")
    args = parser.parse_args(argv)
    llm = _make_llm(args.llm, args.model)
    if llm is None:
        raise SystemExit("an LLM is required (--llm deepseek)")
    service = build_offline_service()
    runtime = AgentRuntime(
        service, build_registry_for_service(service), llm, config=AgentConfig(), today=lambda: EVAL_TODAY
    )
    agent = AgentService(runtime, trace_sinks=[])
    calls, sessions, stopped = 0, [], None
    try:
        only = {int(item) for item in args.only.split(",") if item.strip()}
        for index, (turns, expected) in enumerate(SESSIONS):
            if only and index not in only:
                continue
            if calls >= args.max_calls - 4:
                stopped = f"call budget ({calls} of {args.max_calls})"
                break
            record: dict[str, Any] = {"session": index, "turns": [], "expected": expected}
            for query in turns:
                started = time.perf_counter()
                result = agent.chat(query, session_id=f"frame-llm-{index}", mode="agent")
                used = int((result.get("llm") or {}).get("calls") or 0)
                calls += used
                errors = [item for item in result.get("degraded") or [] if "429" in str(item)]
                record["turns"].append(
                    {
                        "query": query,
                        "route": result.get("route"),
                        "answer_source": result.get("answer_source"),
                        "llm_calls": used,
                        "latency_s": round(time.perf_counter() - started, 2),
                        "verification_passed": (result.get("verification") or {}).get("passed"),
                        "degraded": result.get("degraded") or [],
                        "frame_reason": next(
                            (r for r in result.get("route_reasons") or [] if r.startswith("frame:")), None
                        ),
                        "answer": result.get("answer"),
                    }
                )
                if errors:
                    stopped = "HTTP 429"
                    break
            last = record["turns"][-1]
            appended = "frame_result_appended" in last["degraded"]
            record["states_expected"] = _states(str(last["answer"] or ""), expected)
            record["model_stated_it"] = record["states_expected"] and not appended
            record["fallback_appended"] = appended
            record["refused"] = last["route"] == "refuse"
            sessions.append(record)
            if stopped:
                break
    finally:
        runtime.close()
    done = len(sessions)
    summary = {
        "sessions": done,
        "gap_turn_states_expected": sum(s["states_expected"] for s in sessions),
        "model_stated_it": sum(s["model_stated_it"] for s in sessions),
        "fallback_appended": sum(s["fallback_appended"] for s in sessions),
        "refused": sum(s["refused"] for s in sessions),
        "gap_turn_verified": sum(bool(s["turns"][-1]["verification_passed"]) for s in sessions),
        "llm_calls": calls,
        "stopped": stopped,
    }
    report = {
        "config": {
            "commit": commit,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **command_fields("evaluation.agent_eval.frame_llm_check", argv),
            "model": getattr(llm, "model", None),
            "prompts": prompt_refs(),
            "tools": "offline runtime assets (no replay snapshot)",
        },
        "summary": summary,
        "sessions": sessions,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main()
