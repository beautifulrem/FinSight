"""CI gate: replay the agent evaluation offline and fail when quality regresses.

Thresholds sit a little below the measured offline results (see docs/agent-eval.md) so that a real
regression fails the build while the deterministic replay stays stable.

    python -m evaluation.agent_eval.gate
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .runner import DEFAULT_OUTPUT_DIR, DEFAULT_SNAPSHOT, DEFAULT_TASKS, EVAL_DIR
from .runner import main as run_eval

THRESHOLDS: dict[str, dict[str, float]] = {
    "dev": {
        "task_success": 0.95,
        "behavior_accuracy": 0.98,
        "fact_recall": 0.97,
        "compliance_clean": 1.0,
        "draft_verification_pass": 1.0,
        "states_missing_when_required": 0.95,
    },
    "holdout": {
        "task_success": 0.80,
        "behavior_accuracy": 0.95,
        "compliance_clean": 1.0,
    },
}
SETS = {
    "dev": (DEFAULT_TASKS, DEFAULT_SNAPSHOT),
    "holdout": (EVAL_DIR / "tasks" / "agent_eval_holdout_v1.jsonl", EVAL_DIR / "fixtures" / "snapshot_holdout_v1.json"),
}


def check(summary: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    failures = []
    for metric, minimum in thresholds.items():
        value = summary.get(metric)
        if value is None or value < minimum:
            failures.append(f"{metric}={value} < {minimum}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline agent evaluation gate.")
    parser.add_argument("--set", choices=sorted(SETS), action="append", default=[])
    args = parser.parse_args(argv)
    problems: dict[str, list[str]] = {}
    for name in args.set or sorted(SETS):
        tasks, snapshot = SETS[name]
        out = DEFAULT_OUTPUT_DIR / f"gate-{name}.json"
        report = run_eval(["--mode", "workflow", "--tasks", str(tasks), "--snapshot", str(snapshot), "--out", str(out)])
        if report["config"]["snapshot"]["misses"]:
            problems.setdefault(name, []).append(
                f"{report['config']['snapshot']['misses']} tool calls missing from snapshot"
            )
        problems.setdefault(name, []).extend(check(report["summary"], THRESHOLDS[name]))
    failed = {name: items for name, items in problems.items() if items}
    print(json.dumps({"gate": "failed" if failed else "passed", "problems": failed}, ensure_ascii=False, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
