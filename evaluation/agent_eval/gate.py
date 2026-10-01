"""CI gate: replay the agent evaluation offline and fail when quality regresses.

Two kinds of checks:

* **Floors** (``THRESHOLDS``): absolute minimums that must always hold.
* **Committed baselines** (``evaluation/results/gate-<set>.json``): the offline replay is
  deterministic, so each metric in ``TOLERANCES`` may not fall more than its tolerance below the
  committed baseline. Tasks that passed in the baseline and fail now are always listed. When a
  change legitimately moves a metric, refresh the baseline in the same commit with
  ``--update-baseline`` so the diff shows the new numbers.

``--extras`` also compares the verifier stress test and the offline red team (run beforehand by CI)
with their committed baselines.

    python -m evaluation.agent_eval.gate
    python -m evaluation.agent_eval.gate --update-baseline      # after an intended change
    python -m evaluation.agent_eval.gate --extras-only          # verifier + red-team comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .results import RESULTS_DIR, decode_outcomes, load_result, write_slim
from .runner import DEFAULT_OUTPUT_DIR, TASK_SETS
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
# Allowed drop below the committed baseline (absolute). One dev task is 0.005, one held-out task 0.019.
TOLERANCES: dict[str, float] = {
    "task_success": 0.01,
    "behavior_accuracy": 0.01,
    "fact_recall": 0.01,
    "tool_recall": 0.02,
    "hedged_when_required": 0.02,
    "states_missing_when_required": 0.02,
    "compliance_clean": 0.0,
    "draft_verification_pass": 0.0,
}
# The untouched test set (test_v2) is deliberately not gated: gating on it would turn it into a tuning target.
SETS = {name: TASK_SETS[name] for name in ("dev", "holdout")}


def check(summary: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    failures = []
    for metric, minimum in thresholds.items():
        value = summary.get(metric)
        if value is None or value < minimum:
            failures.append(f"{metric}={value} < {minimum}")
    return failures


def compare_to_baseline(
    summary: dict[str, Any], baseline: dict[str, Any], tolerances: dict[str, float] = TOLERANCES
) -> tuple[list[str], list[str]]:
    """Return (problems, notes): drops beyond tolerance fail; gains beyond it ask for a baseline refresh."""
    problems, notes = [], []
    for metric, tolerance in tolerances.items():
        current, base = summary.get(metric), baseline.get(metric)
        if base is None:
            continue
        if current is None or current < base - tolerance - 1e-9:
            problems.append(f"{metric}={current} fell below baseline {base} (tolerance {tolerance})")
        elif current > base + tolerance + 1e-9:
            notes.append(f"{metric}={current} is above baseline {base}; refresh it with --update-baseline")
    return problems, notes


def newly_failing(outcomes: dict[str, list[bool]], baseline_outcomes: dict[str, list[bool]]) -> list[str]:
    return sorted(
        task for task, runs in outcomes.items() if not all(runs) and all(baseline_outcomes.get(task, [False]))
    )


# Offline red-team baselines, newest first: redteam-offline-r12 covers all eleven attack sets (0 everywhere since
# round 8, holdout7 added in round 9, holdout8 in round 10, holdout9 and holdout10 in round 12); the older files fill
# in any (set, path) a newer one lacks.
REDTEAM_BASELINES = (
    "redteam-offline-r12",
    "redteam-offline-r11",
    "redteam-offline-r10",
    "redteam-offline-r9",
    "redteam-offline-r8",
    "redteam-offline",
)


def redteam_baseline() -> dict[tuple[str, str], dict[str, Any]] | None:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for name in reversed(REDTEAM_BASELINES):
        result = load_result(name)
        for path in (result or {}).get("paths") or []:
            merged[(path["attack_set"], path["mode"])] = path
    return merged or None


def extras_problems(outputs: Path = DEFAULT_OUTPUT_DIR) -> list[str]:
    """Verifier stress and offline red team against their committed baselines."""
    problems = []
    stress_path, redteam_path = outputs / "verifier_stress.json", outputs / "redteam.json"
    stress_base, redteam_base = load_result("verifier_stress"), redteam_baseline()
    if stress_path.exists() and stress_base:
        stress = json.loads(stress_path.read_text(encoding="utf-8"))
        if stress["true_accept"]["claim"] < 1.0:
            problems.append(f"verifier true-accept {stress['true_accept']['claim']} < 1.0")
        allowed = stress_base["false_accept"]["claim"] + 0.01
        if stress["false_accept"]["claim"] > allowed:
            problems.append(
                f"verifier claim false-accept {stress['false_accept']['claim']} > baseline+0.01 ({allowed:.4f})"
            )
    elif not stress_path.exists():
        problems.append(f"{stress_path} missing; run python -m evaluation.agent_eval.verifier_stress first")
    if redteam_path.exists() and redteam_base:
        redteam = json.loads(redteam_path.read_text(encoding="utf-8"))
        base = redteam_base
        for path in redteam["paths"]:
            reference = base.get((path["attack_set"], path["mode"]))
            if path["crashes"]:
                problems.append(f"red team {path['attack_set']}/{path['mode']}: {path['crashes']} crashes")
            if reference and (path["attack_success"] or 0) > (reference["attack_success"] or 0):
                problems.append(
                    f"red team {path['attack_set']}/{path['mode']}: attack success {path['attack_success']} "
                    f"> baseline {reference['attack_success']}"
                )
    elif not redteam_path.exists():
        problems.append(f"{redteam_path} missing; run python -m evaluation.agent_eval.redteam first")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline agent evaluation gate.")
    parser.add_argument("--set", choices=sorted(SETS), action="append", default=[])
    parser.add_argument("--update-baseline", action="store_true", help="Write the run as the committed baseline.")
    parser.add_argument("--extras", action="store_true", help="Also compare verifier stress and red team outputs.")
    parser.add_argument("--extras-only", action="store_true", help="Only the verifier / red-team comparison.")
    args = parser.parse_args(argv)
    problems: dict[str, list[str]] = {}
    notes: dict[str, list[str]] = {}
    for name in [] if args.extras_only else args.set or sorted(SETS):
        tasks, snapshot = SETS[name]
        out = DEFAULT_OUTPUT_DIR / f"gate-{name}.json"
        report = run_eval(["--mode", "workflow", "--tasks", str(tasks), "--snapshot", str(snapshot), "--out", str(out)])
        found = problems.setdefault(name, [])
        if report["config"]["snapshot"]["misses"]:
            found.append(f"{report['config']['snapshot']['misses']} tool calls missing from snapshot")
        found.extend(check(report["summary"], THRESHOLDS[name]))
        baseline = load_result(f"gate-{name}")
        if args.update_baseline:
            path = write_slim(out, f"gate-{name}")
            notes.setdefault(name, []).append(f"baseline written to {path.relative_to(RESULTS_DIR.parents[1])}")
        elif baseline is None:
            notes.setdefault(name, []).append("no committed baseline; floors only")
        else:
            dropped, gained = compare_to_baseline(report["summary"], baseline["summary"])
            found.extend(dropped)
            notes.setdefault(name, []).extend(gained)
            regressed = newly_failing(report["task_outcomes"], decode_outcomes(baseline.get("task_outcomes")))
            if regressed:
                notes[name].append(f"tasks passing in the baseline that fail now: {', '.join(regressed)}")
    if args.extras or args.extras_only:
        problems["extras"] = extras_problems()
    failed = {name: items for name, items in problems.items() if items}
    print(
        json.dumps(
            {
                "gate": "failed" if failed else "passed",
                "problems": failed,
                "notes": {k: v for k, v in notes.items() if v},
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
