"""Where does a slow request spend its time? Stage breakdown of agent traces (round 13, Cost item 1).

Reads the per-run trace files a server writes to ``QI_AGENT_TRACE_DIR`` (``scripts/chaos_drill`` puts them next to its
``--out``) and splits every run's wall time into: ``guard_in`` (NLU + routing), LLM wait (sum of the LLM calls),
tool stage (``agent_prefetch`` + ``agent_tools`` + ``execute_plan`` node time), verify + revise outside the LLM calls,
and the rest. Tool latencies are listed per tool, so a cheap lookup that took seconds (queueing behind CPU-bound
retrieval in the shared tool pool) is visible.

    python -m scripts.trace_stages outputs/chaos/r13-llm-load/traces --out docs/results/perf/agent/x-stages.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

TOOL_NODES = frozenset({"agent_prefetch", "agent_tools", "execute_plan"})


def stages(trace: dict[str, Any]) -> dict[str, Any]:
    nodes: dict[str, float] = defaultdict(float)
    for node in trace.get("nodes") or []:
        nodes[node["node"]] += float(node.get("duration_ms") or 0.0)
    llm = sum(float(call.get("latency_ms") or 0.0) for call in trace.get("llm_calls") or [])
    llm_in_nodes = sum(
        float(call.get("latency_ms") or 0.0)
        for call in trace.get("llm_calls") or []
        if call.get("node") in {"agent_llm", "compose", "revise"}
    )
    tool_stage = sum(value for name, value in nodes.items() if name in TOOL_NODES)
    total = float(trace.get("duration_ms") or sum(nodes.values()))
    tools: dict[str, list[float]] = defaultdict(list)
    for tool in trace.get("tools") or []:
        tools[tool["tool"]].append(round(float(tool.get("latency_ms") or 0.0), 1))
    other_llm_nodes = sum(nodes.get(name, 0.0) for name in ("agent_llm", "compose", "revise")) - llm_in_nodes
    return {
        "trace_id": trace.get("trace_id"),
        "query": trace.get("query"),
        "route": trace.get("route"),
        "total_ms": round(total, 1),
        "guard_in_ms": round(nodes.get("guard_in", 0.0), 1),
        "llm_ms": round(llm, 1),
        "llm_calls": len(trace.get("llm_calls") or []),
        "tool_stage_ms": round(tool_stage, 1),
        "verify_ms": round(nodes.get("verify", 0.0), 1),
        "llm_node_overhead_ms": round(max(other_llm_nodes, 0.0), 1),
        "tools_ms": dict(tools),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    ordered = sorted(rows, key=lambda row: row["total_ms"])
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    totals = {key: sum(row[key] for row in rows) for key in ("total_ms", "guard_in_ms", "llm_ms", "tool_stage_ms")}

    def share(row: dict[str, Any]) -> dict[str, float]:
        return {
            key.replace("_ms", ""): round(row[key] / row["total_ms"], 3) if row["total_ms"] else 0.0
            for key in ("guard_in_ms", "llm_ms", "tool_stage_ms")
        }

    tool_latency: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for tool, values in row["tools_ms"].items():
            tool_latency[tool].extend(values)
    return {
        "runs": len(rows),
        "p95_run": {**{k: v for k, v in p95.items() if k != "tools_ms"}, "share": share(p95), "tools_ms": p95["tools_ms"]},
        "all_runs_share": {
            key.replace("_ms", ""): round(totals[key] / totals["total_ms"], 3)
            for key in ("guard_in_ms", "llm_ms", "tool_stage_ms")
        },
        "median_ms": {
            key: round(statistics.median(row[key] for row in rows), 1)
            for key in ("total_ms", "guard_in_ms", "llm_ms", "tool_stage_ms")
        },
        "tool_latency_ms": {
            tool: {"n": len(values), "median": round(statistics.median(values), 1), "max": round(max(values), 1)}
            for tool, values in sorted(tool_latency.items())
        },
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Stage breakdown of agent traces.")
    parser.add_argument("trace_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    rows = [
        stages(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(Path(args.trace_dir).rglob("*.json"))
    ]
    report = {"trace_dir": args.trace_dir, **summarise(rows), "per_run": sorted(rows, key=lambda r: -r["total_ms"])}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_run"}, ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main()
