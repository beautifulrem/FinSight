"""Where does agent time go? Latency breakdown of an online ablation run.

Reads the per-turn ``profile`` that ``runner._turn_record`` stores (every LLM call with its node, step,
latency, tokens and context composition; every tool call; every graph node's duration) from a raw
ablation output in ``outputs/agent_eval/`` and reports, for the turns that called the LLM:

* LLM calls per turn and their distribution, and the time per call by node (``agent_llm`` step 0, 1, …,
  ``revise``, ``compose``) with each node's share of all LLM time;
* prompt tokens and context composition (characters of system / user / assistant / tool results /
  tool schemas) per tool-loop step, completion and reasoning tokens per node;
* graph-node time (LLM nodes vs tools vs verification and the rest);
* revise frequency and which verification checks triggered it;
* what the slowest 10% of turns have in common.

    python -m evaluation.agent_eval.profile outputs/agent_eval/perf-baseline.json            # markdown tables
    python -m evaluation.agent_eval.profile outputs/agent_eval/perf-baseline.json --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .metrics import percentile

_CONTEXT_PARTS = ("system", "user", "assistant", "tool", "tool_schemas")


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None]
    return round(statistics.fmean(clean), 1) if clean else None


def _call_label(call: dict[str, Any]) -> str:
    if call.get("node") == "agent_llm":
        step = call.get("step") or 0
        return f"agent_llm[{step}]" if step < 3 else "agent_llm[3+]"
    return str(call.get("node"))


def llm_turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turns that made at least one LLM call and carry a profile."""
    return [turn for record in records for turn in record["turns"] if (turn.get("profile") or {}).get("llm_calls")]


def profile_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    turns = llm_turns(records)
    if not turns:
        return {"llm_turns": 0}
    latencies = [turn["latency_ms"] for turn in turns]
    calls_per_turn = [len(turn["profile"]["llm_calls"]) for turn in turns]

    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        for call in turn["profile"]["llm_calls"]:
            by_node[_call_label(call)].append(call)
    total_llm_ms = sum(call.get("latency_ms") or 0 for calls in by_node.values() for call in calls)
    nodes = {}
    for label in sorted(by_node):
        calls = by_node[label]
        ms = [call.get("latency_ms") or 0 for call in calls]
        chars = {
            part: _mean([(call.get("context_chars") or {}).get(part, 0) for call in calls]) for part in _CONTEXT_PARTS
        }
        nodes[label] = {
            "calls": len(calls),
            "calls_per_llm_turn": round(len(calls) / len(turns), 3),
            "latency_ms_p50": percentile(ms, 0.5),
            "latency_ms_p95": percentile(ms, 0.95),
            "latency_ms_mean": _mean(ms),
            "share_of_llm_time": round(sum(ms) / total_llm_ms, 3) if total_llm_ms else None,
            "prompt_tokens_mean": _mean([call.get("prompt_tokens") or 0 for call in calls]),
            "cache_hit_tokens_mean": _mean([call.get("prompt_cache_hit_tokens") or 0 for call in calls]),
            "completion_tokens_mean": _mean([call.get("completion_tokens") or 0 for call in calls]),
            "reasoning_tokens_mean": _mean([call.get("reasoning_tokens") or 0 for call in calls]),
            "tool_calls_requested_mean": _mean([len(call.get("tool_calls") or []) for call in calls]),
            "context_chars_mean": chars,
        }

    # Graph-node time per turn (sum over repeated nodes), averaged over LLM turns.
    span_ms: dict[str, list[float]] = defaultdict(list)
    for turn in turns:
        per_turn: Counter[str] = Counter()
        for span in turn["profile"].get("spans") or []:
            per_turn[str(span.get("node"))] += span.get("duration_ms") or 0
        for node, value in per_turn.items():
            span_ms[node].append(value)
    turn_total = sum(latencies)
    graph_nodes = {
        node: {
            "turns": len(values),
            "ms_mean_when_run": _mean(values),
            "share_of_turn_time": round(sum(values) / turn_total, 3) if turn_total else None,
        }
        for node, values in sorted(span_ms.items(), key=lambda item: -sum(item[1]))
    }

    tools = [tool for turn in turns for tool in turn["profile"].get("tools") or []]
    tool_ms = [tool.get("latency_ms") or 0 for tool in tools]
    tool_rounds = [
        len({tool.get("step") for tool in turn["profile"].get("tools") or [] if tool.get("source") == "llm"})
        for turn in turns
    ]

    revised = [turn for turn in turns if any(call.get("node") == "revise" for call in turn["profile"]["llm_calls"])]
    triggers = Counter(
        kind
        for turn in revised
        for call in turn["profile"]["llm_calls"]
        if call.get("node") == "revise"
        for kind in call.get("trigger") or ["unknown"]
    )
    trigger_sets = Counter(
        "+".join(call.get("trigger") or ["unknown"])
        for turn in revised
        for call in turn["profile"]["llm_calls"]
        if call.get("node") == "revise"
    )

    threshold = percentile(latencies, 0.9) or 0
    slow = [turn for turn in turns if turn["latency_ms"] >= threshold]

    def describe(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "turns": len(group),
            "latency_ms_mean": _mean([turn["latency_ms"] for turn in group]),
            "llm_calls_mean": _mean([len(turn["profile"]["llm_calls"]) for turn in group]),
            "revised_share": round(
                sum(1 for turn in group if any(c.get("node") == "revise" for c in turn["profile"]["llm_calls"]))
                / len(group),
                3,
            )
            if group
            else None,
            "tool_rounds_mean": _mean(
                [
                    len({t.get("step") for t in turn["profile"].get("tools") or [] if t.get("source") == "llm"})
                    for turn in group
                ]
            ),
            "max_single_call_ms_mean": _mean(
                [max(call.get("latency_ms") or 0 for call in turn["profile"]["llm_calls"]) for turn in group]
            ),
            "degraded": dict(
                Counter(
                    str(flag).split(":")[0] for turn in group for flag in turn["profile"].get("degraded") or []
                ).most_common(6)
            ),
        }

    return {
        "llm_turns": len(turns),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_p99": percentile(latencies, 0.99),
        "ttft_ms_p50": percentile([t["ttft_ms"] for t in turns if t.get("ttft_ms") is not None], 0.5),
        "ttft_ms_p95": percentile([t["ttft_ms"] for t in turns if t.get("ttft_ms") is not None], 0.95),
        "llm_calls_per_turn": _mean(calls_per_turn),
        "llm_calls_distribution": dict(sorted(Counter(calls_per_turn).items())),
        "llm_time_share_of_turn": round(total_llm_ms / turn_total, 3) if turn_total else None,
        "llm_nodes": nodes,
        "graph_nodes": graph_nodes,
        "tools": {
            "calls_per_turn": _mean([len(turn["profile"].get("tools") or []) for turn in turns]),
            "latency_ms_p50": percentile(tool_ms, 0.5),
            "latency_ms_p95": percentile(tool_ms, 0.95),
            "llm_tool_rounds_distribution": dict(sorted(Counter(tool_rounds).items())),
        },
        "revise": {
            "rate": round(len(revised) / len(turns), 3),
            "trigger_counts": dict(triggers.most_common()),
            "trigger_combinations": dict(trigger_sets.most_common(8)),
        },
        "slowest_10pct": describe(slow) | {"threshold_ms": threshold},
        "all_llm_turns": describe(turns),
    }


def profile_report(report: dict[str, Any], modes: tuple[str, ...] = ("agent", "workflow_llm")) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for set_name, set_modes in report["results"].items():
        for mode in modes:
            records = (set_modes.get(mode) or {}).get("records")
            if records:
                out[f"{set_name}/{mode}"] = profile_records(records)
    return out


def markdown(profiles: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, data in profiles.items():
        if not data.get("llm_turns"):
            continue
        lines.append(f"### {key}: {data['llm_turns']} LLM turns")
        lines.append("")
        lines.append(
            f"Turn latency P50/P95/P99 {data['latency_ms_p50']}/{data['latency_ms_p95']}/{data['latency_ms_p99']} ms; "
            f"TTFT P50/P95 {data['ttft_ms_p50']}/{data['ttft_ms_p95']} ms; LLM calls/turn {data['llm_calls_per_turn']} "
            f"{data['llm_calls_distribution']}; LLM share of turn time {data['llm_time_share_of_turn']}; "
            f"revise rate {data['revise']['rate']} (triggers {data['revise']['trigger_counts']})."
        )
        lines.append("")
        lines.append(
            "| LLM node | calls/turn | P50 ms | P95 ms | share of LLM time | prompt tok | cached | completion | "
            "reasoning | chars sys/user/asst/tool/schemas |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for label, node in data["llm_nodes"].items():
            chars = node["context_chars_mean"]
            lines.append(
                f"| {label} | {node['calls_per_llm_turn']} | {node['latency_ms_p50']} | {node['latency_ms_p95']} | "
                f"{node['share_of_llm_time']} | {node['prompt_tokens_mean']} | {node['cache_hit_tokens_mean']} | "
                f"{node['completion_tokens_mean']} | {node['reasoning_tokens_mean']} | "
                + "/".join(str(round(chars[part] or 0)) for part in _CONTEXT_PARTS)
                + " |"
            )
        lines.append("")
        lines.append("| Graph node | turns | mean ms when run | share of turn time |")
        lines.append("|---|---:|---:|---:|")
        for node, item in data["graph_nodes"].items():
            lines.append(f"| {node} | {item['turns']} | {item['ms_mean_when_run']} | {item['share_of_turn_time']} |")
        lines.append("")
        tools = data["tools"]
        lines.append(
            f"Tools: {tools['calls_per_turn']} calls/turn, latency P50/P95 {tools['latency_ms_p50']}/"
            f"{tools['latency_ms_p95']} ms, LLM tool rounds per turn {tools['llm_tool_rounds_distribution']}."
        )
        slow, every = data["slowest_10pct"], data["all_llm_turns"]
        lines.append(
            f"Slowest 10% (>= {slow['threshold_ms']} ms): {slow['llm_calls_mean']} LLM calls, revised "
            f"{slow['revised_share']}, {slow['tool_rounds_mean']} tool rounds, longest call "
            f"{slow['max_single_call_ms_mean']} ms (all LLM turns: {every['llm_calls_mean']} calls, "
            f"revised {every['revised_share']}, "
            f"{every['tool_rounds_mean']} rounds, longest call {every['max_single_call_ms_mean']} ms); "
            f"degraded {slow['degraded']}."
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Latency breakdown of an online ablation run.")
    parser.add_argument("ablation", help="Raw ablation output (outputs/agent_eval/*.json) with per-turn records.")
    parser.add_argument("--json", default="", help="Also write the breakdown as JSON here.")
    args = parser.parse_args(argv)
    report = json.loads(Path(args.ablation).read_text(encoding="utf-8"))
    profiles = profile_report(report)
    print(markdown(profiles))
    if args.json:
        Path(args.json).write_text(json.dumps(profiles, ensure_ascii=False, indent=1), encoding="utf-8")
    return profiles


if __name__ == "__main__":
    main()
