"""Render ``docs/agent-eval.md`` from the committed evaluation evidence in ``evaluation/results/``.

Only the block between the GENERATED markers is rewritten; hand-written interpretation outside the
markers is preserved. Every number comes from a file in ``evaluation/results/`` (slimmed from a run
with ``python -m evaluation.agent_eval.results``); each file records the commit, prompts, date and
command of its run, and the provenance table at the end of the block lists them all.

    python -m evaluation.agent_eval.report                    # render from evaluation/results/
    python -m evaluation.agent_eval.report --check            # exit 1 if the doc is out of date

Task success and pass^k carry percentile-bootstrap 95% CIs over tasks (2000 resamples, fixed
seed). Paired comparisons use a paired bootstrap over tasks and an exact McNemar test on pass^k.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .results import RESULTS_DIR, decode_outcomes, load_result

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "agent-eval.md"
BEGIN = "<!-- BEGIN GENERATED: python -m evaluation.agent_eval.report -->"
END = "<!-- END GENERATED -->"

# Which committed result plays which role in the page.
PRIMARY = "ablation-final"  # DeepSeek, dev + held-out, all paths
TEST_V2 = "ablation-test_v2-deepseek"  # DeepSeek, untouched test set
# GLM. The first run of all three sets overlapped with the DeepSeek test-set run on the same gateway and
# its dev rows had LLM errors on 23-28% of turns, so dev comes from a sequential rerun; for each set the
# first file that has it wins.
SECOND_MODEL = ("ablation-glm-dev", "ablation-glm")
# DeepSeek agent on the test set rerun at lower concurrency: the main run hit gateway HTTP 429s.
RATE_LIMIT_CHECK = "ablation-test_v2-deepseek-agent-w3"
OFFLINE = "ablation-offline"  # deterministic paths on all sets at the current evaluation commit
VARIANCE_RUNS = ("ablation-online-deepseek-v4.1-flash", "ablation-online-v1", "ablation-final")
PROMPT_AB = ("ablation-online-v1", "ablation-online-v2")
GATE_RUNS = ("gate-dev", "gate-holdout")
SET_TITLES = {"dev": "Development set", "holdout": "Held-out set", "test_v2": "Untouched test set v2"}

SUMMARY_ROWS = [
    ("task_success", "Task success (dealbreaker-gated)"),
    ("pass^k", "pass^k (all k repeats succeed)"),
    ("behavior_accuracy", "Behaviour accuracy (answer / clarify / refuse)"),
    ("fact_recall", "Required facts stated and cited"),
    ("tool_recall", "Tool recall (required tools used)"),
    ("tool_precision", "Tool precision (calls that were relevant)"),
    ("hedged_when_required", "Hedged when required (why / judgment / advice)"),
    ("states_missing_when_required", "States missing data when required"),
    ("compliance_clean", "No trading instructions"),
    ("draft_verification_pass", "Draft passed evidence verification"),
    ("latency_ms_p50", "Latency P50 (ms)"),
    ("latency_ms_p95", "Latency P95 (ms)"),
    ("llm_calls_per_turn", "LLM calls per turn"),
    ("tokens_per_turn", "LLM tokens per turn"),
    ("reasoning_tokens_per_turn", "Reasoning tokens per turn"),
    ("cache_hit_ratio", "Prompt cache-hit ratio"),
    ("first_pass_verification", "LLM draft verified on first pass"),
    ("revise_rate", "LLM drafts sent back for revision"),
    ("llm_error_rate", "Turns with an LLM error (fallback used)"),
    ("json_repair_rate", "Answer drafts that needed JSON repair"),
    ("json_failure_rate", "Answer drafts that were not JSON (used as text)"),
    ("cost_per_task", "Cost per task"),
]
COMPACT_ROWS = [
    "task_success",
    "pass^k",
    "fact_recall",
    "hedged_when_required",
    "tool_precision",
    "latency_ms_p95",
    "llm_error_rate",
    "cost_per_task",
]


def _fmt(value: Any, key: str = "") -> str:
    if value is None:
        return "–"
    if key.startswith("cost") and isinstance(value, float):
        return f"{value:.5f}"
    if isinstance(value, float):
        return f"{value:.1f}" if value > 1.5 else f"{value:.3f}"
    return str(value)


def pass_key(summary: dict[str, Any]) -> str | None:
    return next((key for key in summary if key.startswith("pass^")), None)


def fmt_ci(value: float | None, ci: list[float] | None) -> str:
    """``0.956 [0.90, 0.99]``."""
    if value is None:
        return "–"
    if not ci:
        return f"{value:.3f}"
    return f"{value:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def cell(summary: dict[str, Any], key: str) -> str:
    ci = summary.get("ci") or {}
    if key == "task_success":
        return fmt_ci(summary.get("task_success"), ci.get("task_success"))
    if key == "pass^k":
        name = pass_key(summary)
        if name is None:
            return "–"
        return f"{fmt_ci(summary[name], ci.get(name))} (k={name.split('^')[1]})"
    return _fmt(summary.get(key), key)


def _label(key: str, label: str, modes: dict[str, Any]) -> str:
    if key != "cost_per_task":
        return label
    currency = next(
        (d["summary"].get("cost_currency") for d in modes.values() if d["summary"].get("cost_currency")), ""
    )
    return f"{label} ({currency})" if currency else label


def set_table(modes: dict[str, Any], rows: list[tuple[str, str]] | list[str]) -> list[str]:
    labels = dict(SUMMARY_ROWS)
    lines = ["| Metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for row in rows:
        key, label = (row, labels[row]) if isinstance(row, str) else row
        lines.append(
            f"| {_label(key, label, modes)} | "
            + " | ".join(cell(data["summary"], key) for data in modes.values())
            + " |"
        )
    return lines


def category_table(modes: dict[str, Any]) -> list[str]:
    categories = sorted({name for data in modes.values() for name in data.get("by_category") or {}})
    lines = ["| Category | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for category in categories:
        cells = []
        for data in modes.values():
            entry = (data.get("by_category") or {}).get(category)
            cells.append(f"{entry['task_success']:.2f} ({entry['tasks']})" if entry else "–")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    return lines


def failure_table(data: dict[str, Any], *, limit: int = 40) -> list[str]:
    seen: dict[str, dict[str, Any]] = {}
    for failure in data.get("failures") or []:
        entry = seen.setdefault(failure["task"], {"query": failure["query"], "checks": [], "count": 0})
        entry["checks"] += [check for check in failure["failed_checks"] if check not in entry["checks"]]
        entry["count"] += failure.get("count", 1)
    lines = ["| Task | Query | Failed checks |", "|---|---|---|"]
    for task, entry in list(seen.items())[:limit]:
        query = entry["query"].replace("|", "\\|")
        lines.append(f"| {task} | {query} | {', '.join(entry['checks'])} |")
    if len(seen) > limit:
        lines.append(f"| … | {len(seen) - limit} more in the result file | |")
    return lines


def verdict(comparison: dict[str, Any], a: str, b: str) -> str:
    """Primary test: the paired-bootstrap 95% CI of the task-success difference excludes 0."""
    ts = comparison["task_success"]
    if not ts["significant"]:
        return "no significant difference"
    return f"{a if ts['diff'] > 0 else b} better"


def diff_ci(entry: dict[str, Any]) -> str:
    return f"{entry['diff']:+.3f} [{entry['ci'][0]:+.3f}, {entry['ci'][1]:+.3f}]"


def comparison_table(comparisons: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Set | a − b | Tasks | Δ task success [95% CI] | Δ pass^k [95% CI] "
        "| McNemar on pass^k (a-only / b-only, p) | Verdict (Δ task success CI) |",
        "|---|---|---|---|---|---|---|",
    ]
    for set_name, items in comparisons.items():
        for name, result in items.items():
            if not result.get("tasks"):
                continue
            a, b = name.split("_vs_")
            mc = result["mcnemar"]
            lines.append(
                f"| {set_name} | {a} − {b} | {result['tasks']} | {diff_ci(result['task_success'])} | "
                f"{diff_ci(result['pass^k'])} | {mc['a_only_pass']} / {mc['b_only_pass']}, p={mc['p_value']:.3f} | "
                f"{verdict(result, a, b)} |"
            )
    return lines


def _header(result: dict[str, Any], name: str) -> list[str]:
    config = result["config"]
    prompts = ", ".join(f"`{ref}`" for ref in (config.get("prompts") or {}).values())
    return [
        f"Source `evaluation/results/{name}.json`: commit `{config.get('commit')}`, run {config.get('run_at')}, "
        f"LLM {config.get('llm') or 'none (offline)'}, prompts {prompts or '–'}.",
        "",
        "```bash",
        config.get("command", ""),
        "```",
        "",
    ]


def ablation_section(
    result: dict[str, Any],
    name: str,
    *,
    full: bool = True,
    failures_for=("workflow", "agent"),
    sets: set[str] | None = None,
) -> list[str]:
    lines = _header(result, name)
    for set_name, modes in result["results"].items():
        if sets is not None and set_name not in sets:
            continue
        first = next(iter(modes.values()))["summary"]
        title = SET_TITLES.get(set_name, set_name)
        lines += [
            f"#### {title} ({first['tasks']} tasks, {first['turns'] // max(1, first.get('repeats') or 1)} turns)",
            "",
        ]
        lines += set_table(modes, SUMMARY_ROWS if full else COMPACT_ROWS)
        lines.append("")
        if full:
            lines += [f"Task success by category ({title.lower()}):", "", *category_table(modes), ""]
            for mode in failures_for:
                data = modes.get(mode)
                if data and data.get("failures"):
                    lines += [f"Remaining {mode} failures ({title.lower()}; each task once across repeats):", ""]
                    lines += [*failure_table(data), ""]
    comparisons = {
        name: items
        for name, items in (result.get("comparisons") or {}).items()
        if items and (sets is None or name in sets)
    }
    if comparisons:
        lines += ["Paired comparisons (same tasks, a − b):", "", *comparison_table(comparisons), ""]
    for note in result.get("notes") or []:
        lines.append(f"* Note: {note}")
    if result.get("notes"):
        lines.append("")
    return lines


def rate_limit_section(main: dict[str, Any], rerun: dict[str, Any], name: str) -> list[str]:
    """The same path rerun with fewer concurrent requests, and a paired comparison across the two files."""
    from .metrics import paired_comparison

    lines = [
        "### Rate-limit check: DeepSeek agent on the test set at lower concurrency",
        "",
        *_header(rerun, name),
    ]
    set_name = "test_v2"
    main_modes, rerun_modes = main["results"].get(set_name) or {}, rerun["results"].get(set_name) or {}
    rows = {"agent (main run)": main_modes.get("agent"), "agent (rerun)": rerun_modes.get("agent")}
    rows["workflow_llm (main run)"] = main_modes.get("workflow_llm")
    rows = {label: data for label, data in rows.items() if data}
    lines += set_table(rows, ["task_success", "pass^k", "llm_error_rate", "latency_ms_p95", "cost_per_task"])
    lines.append("")
    if rerun_modes.get("agent") and main_modes.get("workflow_llm"):
        comparison = paired_comparison(
            decode_outcomes(rerun_modes["agent"]["task_outcomes"]),
            decode_outcomes(main_modes["workflow_llm"]["task_outcomes"]),
        )
        lines += [
            "Paired comparison, agent (rerun) − workflow_llm (main run), same commit and tasks:",
            "",
            *comparison_table({set_name: {"agent_vs_workflow_llm": comparison}}),
            "",
        ]
    return lines


def variance_section(runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "### Run-to-run spread (three DeepSeek runs of the same task sets)",
        "",
        "The three online runs differ in code and prompts as well as in sampling, so the spread is an upper "
        "bound on pure sampling noise; it is the right yardstick for claims that one run beat another.",
        "",
        "| Set | Path | " + " | ".join(f"`{r['config'].get('commit')}` ({name})" for name, r in runs) + " | Spread |",
        "|---|---|" + "---|" * (len(runs) + 1),
    ]
    for set_name in ("dev", "holdout"):
        for mode in ("legacy", "legacy_llm", "workflow_llm", "agent"):
            for metric in ("task_success", "pass^k"):
                if mode in {"legacy", "legacy_llm"} and metric == "pass^k":
                    continue
                values, cells = [], []
                for _name, result in runs:
                    summary = ((result["results"].get(set_name) or {}).get(mode) or {}).get("summary")
                    if not summary:
                        cells.append("–")
                        continue
                    key = "task_success" if metric == "task_success" else pass_key(summary)
                    values.append(summary[key])
                    cells.append(fmt_ci(summary[key], (summary.get("ci") or {}).get(key)))
                spread = f"{max(values) - min(values):.3f}" if len(values) > 1 else "–"
                label = f"{mode} {metric if metric == 'task_success' else 'pass^3'}"
                lines.append(f"| {set_name} | {label} | " + " | ".join(cells) + f" | {spread} |")
    lines.append("")
    return lines


def cross_model_section(first: dict[str, Any], second: dict[str, Any], extra: dict[str, Any] | None) -> list[str]:
    """Same path, two model families: which conclusions transfer."""
    first_model, second_model = first["config"].get("llm"), second["config"].get("llm")
    sources = {"dev": first, "holdout": first}
    if extra:
        sources["test_v2"] = extra
    short = [str(model).split("/")[-1] for model in (first_model, second_model)]
    lines = [
        f"### Second model family: {short[1]} vs {short[0]}",
        "",
        "Commits per row are listed; rows on different commits compare models *and* code. "
        "Δ is agent − workflow_llm task success with its paired-bootstrap 95% CI (* = CI excludes 0).",
        "",
        f"| Set | Path | {short[0]} task success | {short[0]} pass^3 | {short[1]} task success | {short[1]} pass^3 "
        f"| Δ agent − workflow_llm, {short[0]} | Δ agent − workflow_llm, {short[1]} "
        f"| Commits ({short[0]} / {short[1]}) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for set_name in ("dev", "holdout", "test_v2"):
        a_result = sources.get(set_name) or {}
        a_modes = (a_result.get("results") or {}).get(set_name) or {}
        b_modes = second["results"].get(set_name) or {}
        if not b_modes:
            continue
        commits = f"`{(a_result.get('config') or {}).get('commit')}` / `{b_modes.get('_commit', '')}`"
        for mode in ("workflow_llm", "agent"):
            cells = []
            for modes in (a_modes, b_modes):
                summary = (modes.get(mode) or {}).get("summary")
                if not summary:
                    cells += ["–", "–"]
                    continue
                key = pass_key(summary)
                cells += [cell(summary, "task_success"), fmt_ci(summary[key], (summary.get("ci") or {}).get(key))]
            deltas = []
            for result in (a_result, second):
                comparison = ((result or {}).get("comparisons") or {}).get(set_name, {}).get("agent_vs_workflow_llm")
                if mode == "agent" and comparison and comparison.get("tasks"):
                    ts = comparison["task_success"]
                    deltas.append(diff_ci(ts) + (" *" if ts["significant"] else ""))
                else:
                    deltas.append("")
            lines.append(f"| {set_name} | {mode} | " + " | ".join(cells + deltas) + f" | {commits} |")
    lines.append("")
    return lines


_AB_KEYS = [
    ("task_success", "Task success"),
    ("pass^k", "pass^3"),
    ("tool_precision", "Tool precision"),
    ("first_pass_verification", "Draft verified on first pass"),
    ("llm_calls_per_turn", "LLM calls per turn"),
    ("tokens_per_turn", "Tokens per turn"),
    ("latency_ms_p95", "Latency P95 (ms)"),
    ("cost_per_task", "Cost per task"),
]


def prompt_ab_section(baseline: dict[str, Any], variant: dict[str, Any]) -> list[str]:
    base_refs = baseline["config"].get("prompts") or {}
    variant_refs = variant["config"].get("prompts") or {}
    lines = [
        "### Prompt A/B (v1 vs v2, same commit)",
        "",
        f"Baseline `{base_refs.get('agent_system')}` vs variant `{variant_refs.get('agent_system')}` "
        f"(both started at `{baseline['config'].get('commit')}`; "
        f"variant command `{variant['config'].get('command')}`). "
        "Paired comparison: bootstrap over tasks and McNemar on pass^3.",
        "",
    ]
    from .metrics import paired_comparison

    for set_name in ("dev", "holdout"):
        for mode in ("workflow_llm", "agent"):
            base = (baseline["results"].get(set_name) or {}).get(mode)
            other = (variant["results"].get(set_name) or {}).get(mode)
            if not base or not other:
                continue
            lines += [f"{set_name} · {mode}:", "", "| Metric | v1 (baseline) | v2 (variant) |", "|---|---|---|"]
            for key, label in _AB_KEYS:
                lines.append(f"| {label} | {cell(base['summary'], key)} | {cell(other['summary'], key)} |")
            comparison = paired_comparison(
                decode_outcomes(other["task_outcomes"]), decode_outcomes(base["task_outcomes"])
            )
            ts, mc = comparison["task_success"], comparison["mcnemar"]
            lines += [
                "",
                f"v2 − v1: task success {diff_ci(ts)}, pass^3 {diff_ci(comparison['pass^k'])}, "
                f"McNemar p={mc['p_value']:.3f} → {verdict(comparison, 'v2', 'v1')}.",
                "",
            ]
    return lines


def faults_section(faults: dict[str, Any]) -> list[str]:
    lines = [
        f"### Fault injection (overall graceful rate {faults['overall_graceful_rate']:.2f})",
        "",
        f"Command `{faults['config'].get('command')}` at commit `{faults['config'].get('commit')}`. Faults are "
        "simulated with stub tools and a scripted LLM, not injected into real providers.",
        "",
        "| Scenario | Expectation | Runs | Graceful | Tool errors seen |",
        "|---|---|---|---|---|",
    ]
    for scenario in faults["scenarios"]:
        lines.append(
            f"| {scenario['scenario']} | {scenario['expectation']} | {scenario['runs']} | "
            f"{scenario['graceful_rate']:.2f} | {', '.join(scenario.get('tool_errors') or []) or '–'} |"
        )
    lines.append("")
    return lines


def stress_section(stress: dict[str, Any]) -> list[str]:
    modes = list(stress["false_accept"])
    lines = [
        f"### Verifier stress test ({stress['gold_answers']} gold answers, {stress['variants']} corrupted variants)",
        "",
        f"Command: `{stress['config']['command']}` at commit `{stress['config']['commit']}`. Lower is better; "
        "true-accept must stay 1.0.",
        "",
        "| | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes),
        "| True-accept (gold answers passing) | " + " | ".join(_fmt(stress["true_accept"][m]) for m in modes) + " |",
        "| False-accept, all corruptions | " + " | ".join(_fmt(stress["false_accept"][m]) for m in modes) + " |",
    ]
    for kind, values in stress["by_kind"].items():
        cells = " | ".join(_fmt(values.get(f"false_accept_{m}")) for m in modes)
        lines.append(f"| False-accept, {kind} ({values['variants']}) | {cells} |")
    lines.append("")
    repair = stress.get("repair")
    if repair:
        lines += [
            f"Repair of the {repair['repaired_answers']} rejected variants (whole-sentence deletion, template "
            "fallback when nothing cited survives):",
            "",
            "| Readable (no dangling clause, stray punctuation, orphan citation) | Containing a fragment "
            "| Repaired answer verifies | Untouched sentences kept | Template fallback |",
            "|---|---|---|---|---|",
            f"| {_fmt(repair['readable'])} | {_fmt(repair.get('with_fragment'))} | "
            f"{_fmt(repair['passes_verification'])} | {_fmt(repair['retained_sentences'])} | "
            f"{_fmt(repair['template_fallback'])} |",
            "",
        ]
    return lines


def _attack_counts(attacks: Any) -> str:
    if not isinstance(attacks, dict):
        return str(attacks)
    return ", ".join(f"{name} {count}" for name, count in attacks.items())


def redteam_section(redteam: dict[str, Any], title: str) -> list[str]:
    config = redteam["config"]
    lines = [
        f"### {title}",
        "",
        f"Command: `{config['command']}` at commit `{config['commit']}` (LLM: {config.get('llm') or 'none'}). "
        f"Attacks: {_attack_counts(config['attacks'])}; variants: {', '.join(config['variants'])}. "
        "Only runs in which a document tool returned the poisoned text are counted.",
        "",
        "| Attack set | Path | Runs | Attack success | Redaction by lexical filter | Crashes |",
        "|---|---|---|---|---|---|",
    ]
    for path in redteam["paths"]:
        lines.append(
            f"| {path['attack_set']} | {path['mode']} | {path['exposed_runs']} | {_fmt(path['attack_success'])} | "
            f"{_fmt(path['redaction'])} | {path['crashes']} |"
        )
    lines.append("")
    successes = [(path, item) for path in redteam["paths"] for item in path.get("successes") or []]
    if successes:
        lines += [
            "Successful attacks:",
            "",
            "| Set | Path | Attack | Variant | Answer excerpt |",
            "|---|---|---|---|---|",
        ]
        for path, item in successes[:15]:
            excerpt = item["answer_excerpt"][:120].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {path['attack_set']} | {path['mode']} | {item['attack']} | {item['variant']} | {excerpt} |"
            )
        lines.append("")
    return lines


def gate_section(runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "### Offline gate baselines (CI compares against these)",
        "",
        "| Run | Commit | Tasks | Task success [95% CI] | Behaviour | Facts | Snapshot misses |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in runs:
        summary = run["summary"]
        lines.append(
            f"| {name} | `{run['config'].get('commit')}` | {summary['tasks']} | {cell(summary, 'task_success')} | "
            f"{_fmt(summary.get('behavior_accuracy'))} | {_fmt(summary.get('fact_recall'))} | "
            f"{(run['config'].get('snapshot') or {}).get('misses')} |"
        )
    lines.append("")
    return lines


def provenance_section(names: list[str]) -> list[str]:
    lines = [
        "### Provenance of every number above",
        "",
        "| File in `evaluation/results/` | Kind | Commit | Run at (UTC) | Source file (sha256/16) |",
        "|---|---|---|---|---|",
    ]
    for name in names:
        result = load_result(name)
        if not result:
            continue
        config, source = result.get("config") or {}, result.get("source") or {}
        lines.append(
            f"| `{name}.json` | {result.get('kind')} | `{config.get('commit')}` | {config.get('run_at')} | "
            f"`{source.get('file')}` ({source.get('sha256_16')}) |"
        )
    lines.append("")
    return lines


def render() -> str:
    lines = [
        BEGIN,
        "",
        "Rendered by `python -m evaluation.agent_eval.report` from the committed files in `evaluation/results/`. "
        "Task success and pass^k show the value and a 95% bootstrap CI over tasks "
        "(2000 resamples, seed 20260926). With 53 held-out tasks one task is 1.9 points and the CI half-width "
        "is about 6–9 points; with 121 test tasks one task is 0.8 points.",
        "",
    ]
    used: list[str] = []

    def take(name: str) -> dict[str, Any] | None:
        result = load_result(name)
        if result is not None and name not in used:
            used.append(name)
        return result

    primary = take(PRIMARY)
    if primary:
        lines += ["### Development and held-out sets, all paths (DeepSeek V4.1 Flash)", ""]
        lines += ablation_section(primary, PRIMARY)
    test_v2 = take(TEST_V2)
    if test_v2:
        lines += ["### Untouched test set v2 (DeepSeek V4.1 Flash)", ""]
        lines += ablation_section(test_v2, TEST_V2)
    rate_check = take(RATE_LIMIT_CHECK)
    if test_v2 and rate_check:
        lines += rate_limit_section(test_v2, rate_check, RATE_LIMIT_CHECK)
    second_files = [(name, take(name)) for name in SECOND_MODEL]
    second_files = [(name, result) for name, result in second_files if result]
    if second_files:
        merged: dict[str, Any] = {"config": second_files[0][1]["config"], "results": {}, "comparisons": {}}
        lines += [f"### All sets with a second model family ({merged['config'].get('llm')})", ""]
        for name, result in second_files:
            fresh = [set_name for set_name in result["results"] if set_name not in merged["results"]]
            for set_name in fresh:
                merged["results"][set_name] = {**result["results"][set_name], "_commit": result["config"].get("commit")}
                merged["comparisons"][set_name] = (result.get("comparisons") or {}).get(set_name, {})
            if fresh:
                lines += ablation_section(result, name, full=False, sets=set(fresh))
        if primary:
            lines += cross_model_section(primary, merged, test_v2)
    offline = take(OFFLINE)
    if offline:
        lines += ["### Deterministic paths on all three sets at the evaluation commit", ""]
        lines += ablation_section(offline, OFFLINE, full=False)
    variance = [(name, take(name)) for name in VARIANCE_RUNS]
    if all(result for _name, result in variance):
        lines += variance_section(variance)  # type: ignore[arg-type]
    baseline, variant = take(PROMPT_AB[0]), take(PROMPT_AB[1])
    if baseline and variant:
        lines += prompt_ab_section(baseline, variant)
    gates = [(name, take(name)) for name in GATE_RUNS]
    if all(result for _name, result in gates):
        lines += gate_section(gates)  # type: ignore[arg-type]
    faults = take("fault_injection")
    if faults:
        lines += faults_section(faults)
    stress = take("verifier_stress")
    if stress:
        lines += stress_section(stress)
    online_redteam = take("redteam-online")
    if online_redteam:
        lines += redteam_section(online_redteam, "Prompt-injection red team (online)")
    offline_redteam = take("redteam-offline")
    if offline_redteam:
        lines += redteam_section(offline_redteam, "Prompt-injection red team (offline workflow path, CI baseline)")
    lines += provenance_section(used)
    lines.append(END)
    return "\n".join(lines)


def splice(text: str, block: str) -> str:
    if BEGIN in text and END in text:
        return text[: text.index(BEGIN)] + block + text[text.index(END) + len(END) :]
    return text.rstrip() + "\n\n" + block + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render docs/agent-eval.md from evaluation/results/.")
    parser.add_argument("--check", action="store_true", help="Fail if the generated block is out of date.")
    args = parser.parse_args(argv)
    if not RESULTS_DIR.exists():
        raise SystemExit(f"{RESULTS_DIR} does not exist; slim runs with python -m evaluation.agent_eval.results")
    block = render()
    text = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else "# FinSight Agent Evaluation\n"
    updated = splice(text, block)
    if args.check:
        if updated != text:
            print(f"{DOC_PATH.relative_to(ROOT)} is out of date; run python -m evaluation.agent_eval.report")
            return 1
        print(f"{DOC_PATH.relative_to(ROOT)} is up to date")
        return 0
    DOC_PATH.write_text(updated, encoding="utf-8")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
