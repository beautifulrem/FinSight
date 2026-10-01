"""Slim evaluation outputs into committed evidence under ``evaluation/results/``.

``outputs/`` is gitignored, so the raw JSON behind the documented numbers is not on GitHub. This
module keeps what a reader needs to trace every headline number (config with commit, prompts and
command; summaries; per-category tables; failed checks; deduplicated failures; per-task outcomes)
and drops the bulky per-turn records.

Older ablation files did not store per-task outcomes. They are reconstructed from the failure list
(each failing turn of each repeat is listed): exact for single-turn tasks and for pass^k; for a
multi-turn task where two different turns failed, the number of failed repeats is taken as the
largest per-turn count, and the file records whether the reconstruction reproduced the stored task
success exactly.

    python -m evaluation.agent_eval.results outputs/agent_eval/ablation-final.json   # -> evaluation/results/
    python -m evaluation.agent_eval.results outputs/agent_eval/gate-dev.json outputs/agent_eval/redteam.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .metrics import CI_METHOD, COMPARISONS, outcome_cis, paired_comparison, task_success_value
from .profile import profile_report
from .runner import ROOT, TASK_SETS, load_tasks

RESULTS_DIR = ROOT / "evaluation" / "results"
EXCERPT_CHARS = 160
# Paths that fail (almost) every task by construction: keep their failure rows but not the answer text.
NO_EXCERPT_MODES = frozenset({"pure_llm", "legacy", "legacy_llm"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def dedupe_failures(failures: list[dict[str, Any]], *, excerpts: bool = True) -> list[dict[str, Any]]:
    """One entry per (task, query, failed checks) with a count across repeats; excerpts shortened."""
    merged: dict[tuple, dict[str, Any]] = {}
    for failure in failures:
        key = (failure["task"], failure["query"], tuple(failure.get("failed_checks") or []))
        if key in merged:
            merged[key]["count"] += 1
            continue
        merged[key] = {
            "task": failure["task"],
            "category": failure.get("category"),
            "query": failure["query"],
            "failed_checks": list(key[2]),
            "count": 1,
            "route": failure.get("route"),
            "tools_used": failure.get("tools_used"),
        }
        if excerpts:
            merged[key]["answer_excerpt"] = str(failure.get("answer_excerpt") or "")[:EXCERPT_CHARS]
    return list(merged.values())


def reconstruct_outcomes(
    tasks: list[dict[str, Any]], failures: list[dict[str, Any]], runs: int
) -> tuple[dict[str, list[bool]], list[str]]:
    """Per-task outcomes from a failure list (one entry per failing turn per repeat).

    Returns the outcomes and the ids of multi-turn tasks whose failed-repeat count is ambiguous.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for failure in failures:
        counts[failure["task"]][failure["query"]] += 1
    outcomes, ambiguous = {}, []
    for task in tasks:
        per_turn = counts.get(task["id"], Counter())
        failed = min(runs, max(per_turn.values(), default=0))
        if sum(1 for value in per_turn.values() if value) > 1 and sum(per_turn.values()) > failed:
            ambiguous.append(task["id"])
        outcomes[task["id"]] = [False] * failed + [True] * (runs - failed)
    return outcomes, ambiguous


def encode_outcomes(outcomes: dict[str, list[bool]] | None) -> dict[str, str] | None:
    """Compact form for committed files: ``{"task": "110"}`` = repeats 1 and 2 succeeded, 3 failed."""
    if outcomes is None:
        return None
    return {task: "".join("1" if ok else "0" for ok in runs) for task, runs in outcomes.items()}


def decode_outcomes(encoded: dict[str, Any] | None) -> dict[str, list[bool]]:
    """Inverse of ``encode_outcomes``; also accepts the list-of-bools form written by the ablation."""
    return {
        task: [char == "1" for char in runs] if isinstance(runs, str) else [bool(item) for item in runs]
        for task, runs in (encoded or {}).items()
    }


def _pass_key(summary: dict[str, Any]) -> str | None:
    return next((key for key in summary if key.startswith("pass^")), None)


def slim_mode(
    data: dict[str, Any], tasks: list[dict[str, Any]] | None, *, mode: str
) -> tuple[dict[str, Any], list[str]]:
    """Slim one mode of an ablation set; returns the entry and notes about repairs made."""
    notes: list[str] = []
    summary = dict(data["summary"])
    outcomes = decode_outcomes(data["task_outcomes"]) if data.get("task_outcomes") is not None else None
    reconstruction = None
    if outcomes is None and tasks is not None:
        task_turns = sum(len(task["turns"]) for task in tasks)
        runs = max(1, round(summary["turns"] / task_turns))
        outcomes, ambiguous = reconstruct_outcomes(tasks, data.get("failures") or [], runs)
        success = round(sum(task_success_value(item) for item in outcomes.values()) / len(outcomes), 4)
        reconstruction = {
            "method": "from failure list",
            "runs_per_task": runs,
            "ambiguous_multi_turn_tasks": ambiguous,
            "reproduces_stored_task_success": abs(success - (summary.get("task_success") or 0)) < 1e-4,
        }
        if runs == 1 and summary.get("repeats", 1) != 1:
            old_key = _pass_key(summary)
            value = summary.pop(old_key) if old_key else None
            summary["pass^1"] = value
            notes.append(
                f"{mode}: the stored summary said repeats={summary.get('repeats')} and labelled {old_key}, but "
                f"the file has {summary['turns']} turns for {task_turns} task turns, i.e. one run per task; "
                "relabelled pass^1."
            )
            summary["repeats"] = 1
    if outcomes:
        # Recompute the outcome CIs; keep CIs the run computed for other metrics (task_success_uncited).
        kept = {key: ci for key, ci in (summary.get("ci") or {}).items() if key.endswith("_uncited")}
        summary["ci"] = {**outcome_cis(outcomes), **kept}
        summary["ci_method"] = CI_METHOD
    entry = {
        "summary": summary,
        "by_category": data.get("by_category"),
        "failed_checks": data.get("failed_checks"),
        "failures": dedupe_failures(data.get("failures") or [], excerpts=mode.split("/")[-1] not in NO_EXCERPT_MODES),
        "task_outcomes": encode_outcomes(outcomes),
    }
    if reconstruction:
        entry["task_outcomes_reconstruction"] = reconstruction
    return entry, notes


def slim_ablation(report: dict[str, Any]) -> dict[str, Any]:
    config = dict(report["config"])
    notes: list[str] = []
    results: dict[str, dict[str, Any]] = {}
    for set_name, modes in report["results"].items():
        task_file = (config.get("task_files") or {}).get(set_name) or config.get(f"{set_name}_tasks")
        tasks = load_tasks(ROOT / task_file) if task_file else load_tasks(TASK_SETS[set_name][0])
        results[set_name] = {}
        for mode, data in modes.items():
            results[set_name][mode], mode_notes = slim_mode(data, tasks, mode=f"{set_name}/{mode}")
            notes += mode_notes
    comparisons = {
        set_name: {
            f"{a}_vs_{b}": paired_comparison(
                decode_outcomes(modes[a]["task_outcomes"]), decode_outcomes(modes[b]["task_outcomes"])
            )
            for a, b in COMPARISONS
            if a in modes and b in modes and modes[a]["task_outcomes"] and modes[b]["task_outcomes"]
        }
        for set_name, modes in results.items()
    }
    slim = {"kind": "ablation", "config": config, "notes": notes, "results": results, "comparisons": comparisons}
    if report.get("llm_http"):
        slim["llm_http"] = report["llm_http"]
    profiles = profile_report(report)
    if profiles:
        # Latency breakdown from the per-turn records, which are not committed (evaluation.agent_eval.profile).
        slim["profile"] = profiles
    return slim


def slim_run(report: dict[str, Any]) -> dict[str, Any]:
    """A single ``runner`` output (e.g. the offline gate runs)."""
    outcomes = decode_outcomes(report["task_outcomes"]) if report.get("task_outcomes") is not None else None
    if outcomes is None and report.get("records"):
        outcomes = defaultdict(list)
        for record in sorted(report["records"], key=lambda item: item.get("repeat", 0)):
            outcomes[record["task"]["id"]].append(all(turn["score"]["success"] for turn in record["turns"]))
        outcomes = dict(outcomes)
    summary = dict(report["summary"])
    if outcomes:
        # Recompute the outcome CIs; keep CIs the run computed for other metrics (task_success_uncited).
        kept = {key: ci for key, ci in (summary.get("ci") or {}).items() if key.endswith("_uncited")}
        summary["ci"] = {**outcome_cis(outcomes), **kept}
        summary["ci_method"] = CI_METHOD
    return {
        "kind": "run",
        "config": report["config"],
        "summary": summary,
        "by_category": report.get("by_category"),
        "by_language": report.get("by_language"),
        "failed_checks": report.get("failed_checks"),
        "failures": dedupe_failures(report.get("failures") or []),
        "task_outcomes": encode_outcomes(outcomes),
    }


def slim_redteam(report: dict[str, Any]) -> dict[str, Any]:
    paths = [{key: value for key, value in path.items() if key != "results"} for path in report["paths"]]
    return {"kind": "redteam", "config": report["config"], "paths": paths}


def slim_faults(report: dict[str, Any]) -> dict[str, Any]:
    scenarios = []
    for scenario in report["scenarios"]:
        results = scenario.get("results") or []
        scenarios.append(
            {
                **{key: value for key, value in scenario.items() if key != "results"},
                "tool_errors": sorted({code for result in results for code in result.get("tool_errors") or [] if code}),
            }
        )
    return {
        "kind": "fault_injection",
        "config": report["config"],
        "overall_graceful_rate": report["overall_graceful_rate"],
        "scenarios": scenarios,
    }


def slim(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("kind") == "output_safety_audit":
        # the counts, per-set tables and samples; the per-edit list stays in outputs/
        return {key: value for key, value in report.items() if key != "units"}
    if "results" in report and "config" in report:
        return slim_ablation(report)
    if "paths" in report:
        return slim_redteam(report)
    if "scenarios" in report:
        return slim_faults(report)
    if "false_accept" in report:
        return {"kind": "verifier_stress", **report}
    if "summary" in report:
        return slim_run(report)
    raise ValueError("unrecognised evaluation output")


def write_slim(source: Path, name: str, *, extra_notes: list[str] | None = None) -> Path:
    report = json.loads(source.read_text(encoding="utf-8"))
    slimmed = slim(report)
    config = slimmed.get("config")
    if isinstance(config, dict) and "model" not in config:
        # Every committed result names its model (None = no LLM); older runs only had ``llm``.
        slimmed["config"] = {**config, "model": config.get("llm")}
    slimmed["source"] = {
        "file": str(source.resolve().relative_to(ROOT)) if source.resolve().is_relative_to(ROOT) else source.name,
        "sha256_16": _sha256(source),
        "bytes": source.stat().st_size,
        "slimmed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if extra_notes:
        slimmed["notes"] = [*(slimmed.get("notes") or []), *extra_notes]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(slimmed, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    return out


# A red-team path where most runs hit HTTP 429 measured the template fallback, not the LLM.
MAX_REDTEAM_429_RATE = 0.5


def invalid_runs(report: dict[str, Any]) -> list[str]:
    """Runs that must not be committed without ``--allow-invalid``: those an ablation marked invalid, and red-team
    LLM paths where more than half of the runs got HTTP 429 (their answers came from the template fallback)."""
    invalid = list(report.get("invalid_runs") or [])
    for path in report.get("paths") or []:
        if (path.get("llm_429_rate") or 0) > MAX_REDTEAM_429_RATE:
            invalid.append(f"{path.get('attack_set')}/{path.get('mode')} (429 rate {path['llm_429_rate']})")
    return invalid


def load_result(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write slim, committable evaluation summaries.")
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--name", default="", help="Output name (single source only); defaults to the file stem.")
    parser.add_argument("--note", action="append", default=[], help="Provenance note to attach.")
    parser.add_argument(
        "--allow-invalid", action="store_true", help="Commit a run that ablation marked invalid (e.g. mostly 429s)."
    )
    args = parser.parse_args(argv)
    if args.name and len(args.sources) > 1:
        raise SystemExit("--name needs exactly one source")
    for source in args.sources:
        path = Path(source)
        invalid = invalid_runs(json.loads(path.read_text(encoding="utf-8")))
        if invalid and not args.allow_invalid:
            raise SystemExit(f"{source} has invalid runs {invalid}; rerun, or pass --allow-invalid with a --note")
        out = write_slim(path, args.name or path.stem, extra_notes=args.note)
        print(f"{source} -> {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
