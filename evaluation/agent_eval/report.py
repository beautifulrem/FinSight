"""Render ``docs/agent-eval.md`` from the committed evaluation evidence in ``evaluation/results/``.

Only the block between the GENERATED markers is rewritten; hand-written interpretation outside the
markers is preserved. Every number comes from a file in ``evaluation/results/`` (slimmed from a run
with ``python -m evaluation.agent_eval.results``); each file records the commit, prompts, date and
command of its run, and the provenance table at the end of the block lists them all.

    python -m evaluation.agent_eval.report                    # render from evaluation/results/
    python -m evaluation.agent_eval.report --check            # exit 1 if the doc is out of date, or if
                                                              # README.md / README_CN.md cite a result
                                                              # file that is missing or not rendered

Task success and pass^k carry percentile-bootstrap 95% CIs over tasks (2000 resamples, fixed
seed). Paired comparisons use a paired bootstrap over tasks and an exact McNemar test on pass^k.
Every online table shows the share of turns where an LLM call failed and the run fell back
(``llm_error_rate``), and how many of those failures were HTTP 429 (gateway rate limit / quota).
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from .metrics import pass_all_value, task_success_value, uncited_success_bounds
from .results import RESULTS_DIR, decode_outcomes, load_result

ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = ROOT / "docs" / "agent-eval.md"
README_PATHS = (ROOT / "README.md", ROOT / "README_CN.md")
BEGIN = "<!-- BEGIN GENERATED: python -m evaluation.agent_eval.report -->"
END = "<!-- END GENERATED -->"

# Which committed result plays which role in the page.
# 9536abf: test v3 (first LLM run), held-out, test v2 and multi-turn v1 (both after exposure); GLM on test v3.
FINAL4_ONLINE = (
    "ablation-final4-deepseek-testv3-holdout",
    "ablation-final4-deepseek-testv2-multiturn",
    "ablation-final4-glm-testv3",
)
FINAL_ONLINE = ("ablation-final2-deepseek", "ablation-final2-glm")  # d1c007c: held-out + test v2, both models
PRIMARY = "ablation-final"  # DeepSeek, dev + held-out, all paths
TEST_V2 = "ablation-test_v2-deepseek"  # DeepSeek, test set v2, first runs (before exposure)
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
TEST_V3 = ("test_v3-auto-nollm-first-run", "ablation-test_v3-purellm-deepseek")
MULTITURN = (
    "multiturn_v1-auto-nollm-first-run",
    "ablation-multiturn_v1-deepseek-first-run",
    "multiturn_v1-auto-nollm-after-fixes",
)
PERF_RUNS = (
    "perf-baseline-deepseek",
    "perf-verifierfix-deepseek",
    "perf-merged-defaults-deepseek",
    "perf-merged-citerepair-stall-deepseek",
    "perf-merged-prefetch-deepseek",
    "perf-glm-holdout-off",
    "perf-glm-holdout-off2",
    "perf-glm-holdout-on",
    "perf-glm-holdout-on2",
)
# Paired perf comparisons (a − b): each change against the run it was measured against.
PERF_PAIRS = (
    ("perf-verifierfix-deepseek", "perf-baseline-deepseek"),
    ("perf-merged-citerepair-stall-deepseek", "perf-merged-defaults-deepseek"),
    ("perf-merged-prefetch-deepseek", "perf-merged-defaults-deepseek"),
    ("perf-merged-prefetch-deepseek", "perf-merged-citerepair-stall-deepseek"),
)
STRESS_RUNS = ("verifier_stress", "verifier_stress-round9", "verifier_stress-perf-8a85ae5")
REDTEAM_RUNS = (
    ("redteam-r9-holdout7-llm", "Prompt-injection red team, LLM paths on holdout7 after round 9 (3d7afd5)"),
    (
        "redteam-offline-r9",
        "Prompt-injection red team, offline template path after round 9 (all eight sets, CI baseline)",
    ),
    (
        "redteam-holdout7-prefix",
        "Prompt-injection red team, round-5 reviewer's attacks (holdout7), template path, before the round-9 fix",
    ),
    ("redteam-r8-llm", "Prompt-injection red team, LLM paths after round 8 (0473968)"),
    (
        "redteam-offline-r8",
        "Prompt-injection red team, offline template path after round 8 (all seven sets)",
    ),
    (
        "redteam-holdout6-prefix",
        "Prompt-injection red team, round-4 reviewer's attacks (holdout6), template path, before the round-8 fix",
    ),
    ("redteam-final4-llm", "Prompt-injection red team, LLM paths at the final online commit (9536abf)"),
    ("redteam-offline-r6", "Prompt-injection red team, offline template path after round 4 (all six sets)"),
    ("redteam-holdout5-first-run", "Prompt-injection red team, independent round-4 attacks (holdout5), first run"),
    ("redteam-final2", "Prompt-injection red team, LLM paths at the round-2 online commit (d1c007c)"),
    ("redteam-online", "Prompt-injection red team, LLM paths (earlier online run)"),
    ("redteam-offline", "Prompt-injection red team (offline workflow path, earlier CI baseline)"),
)
SUPERSEDED = ("ablation-test_v2-deepseek-concurrent",)
ROUTER_RUNS = (
    "router_eval-d78b313",
    "router_eval-round2",
    "router_eval-round3",
    "router_eval-round3b",
    "router_eval-round4-own",
    "router_eval-independent_v1-first-run",
    "router_eval-round4-independent-after-exposure",
    "router_eval-independent_v2-first-run",
    "router_eval-round9-own",
    "router_eval-independent_v2-round9",
)
CLAIM_BENCH_RUNS = (
    "claim_bench-dev-baseline",
    "claim_bench-dev",
    "claim_bench-holdout",
    "claim_bench-holdout-after-round8",
    "claim_bench-heldout_r4-first-run",
    "claim_bench-heldout_r4-after-exposure",
    "claim_bench-holdout-after-round9",
    "claim_bench-heldout_r4-after-round9",
    "claim_bench-heldout_r5-first-run",
    "claim_bench-heldout_r5-after-exposure",
)
# Round-4 held-out slices (evaluation/heldout_r4/, independent author): first run, then after exposure.
HELDOUT_R4_RUNS = ("multiturn_r4_heldout-auto-nollm-first-run", "multiturn_r4_heldout-after-exposure")
# Round-5 held-out chat slice (evaluation/heldout_r5/, independent author): first run, then after exposure (round 9).
HELDOUT_R5_RUNS = ("chat_heldout_r5-auto-nollm-first-run", "chat_heldout_r5-auto-nollm-after-exposure")
# Other committed evidence the READMEs cite, summarised as one row each.
EXTRA_EVIDENCE = (
    "redteam-r7-targeted",
    "redteam-r8-d1-targeted",
    "injection_classifier-r4",
    "verifier_stress-clause-salvage",
)

SET_TITLES = {
    "dev": "Development set",
    "holdout": "Held-out set",
    "test_v2": "Test set v2",
    "test_v3": "Test set v3 (independent author)",
    "multiturn_v1": "Multi-turn set v1 (independent author)",
}

# First run vs after exposure. Test v2 was written blind and first run at f7bf624 / 38a3069; the failure
# classes those runs exposed (elliptical follow-ups) were read and fixed at da3ec8b, so every later run on
# it is after exposure.
TEST_V2_EXPOSED_AT = "da3ec8b"
TEST_V2_FIRST_RUN_COMMITS = frozenset({"f7bf624", "38a3069"})
FILE_STATUS = {
    "multiturn_v1-auto-nollm-first-run": "**first run** (before any fix for its failure classes)",
    "ablation-multiturn_v1-deepseek-first-run": "**first run** on the LLM paths "
    "(before any fix for its failure classes)",
    "multiturn_v1-auto-nollm-after-fixes": "**after exposure**: rules were written for the failure classes the "
    "first run showed, so this is not an unseen measurement",
    "router_eval-d78b313": "author's own labels (tuned against)",
    "router_eval-round2": "author's own labels (tuned against)",
    "router_eval-round3": "author's own labels (tuned against)",
    "router_eval-round3b": "author's own labels (tuned against)",
    "router_eval-round4-own": "author's own labels (tuned against)",
    "router_eval-independent_v1-first-run": "**first run** of an independent set",
    "router_eval-round4-independent-after-exposure": "**after exposure** (fixes were generalised from its misses)",
    "router_eval-independent_v2-first-run": "**first run** of a fresh independent set",
    "router_eval-round9-own": "author's own labels (tuned against)",
    "router_eval-independent_v2-round9": "**after exposure** (HEAD after round 9; first run 0.8008)",
    "chat_heldout_r5-auto-nollm-first-run": "**first run** of the independent round-5 held-out chat slice",
    "chat_heldout_r5-auto-nollm-after-exposure": "**after exposure** (round 9 fixed the classes its first run showed)",
    "claim_bench-dev-baseline": "development claims, before tuning",
    "claim_bench-dev": "development claims, after tuning (tuned on)",
    "claim_bench-holdout": "**held-out claims, run once** (hashed file; later fixes are not re-scored here)",
    "claim_bench-holdout-after-round8": "held-out claims **after exposure** (round 8: the review's h038 class, "
    "industry averages, was fixed; not a fresh estimate)",
    "claim_bench-holdout-after-round9": "held-out claims **after exposure**, re-run at the round-9 commit",
    "claim_bench-heldout_r4-after-round9": "independent round-4 claim slice **after exposure**, re-run at the round-9 "
    "commit",
    "claim_bench-heldout_r5-first-run": "**first and only pre-fix run** of the independent round-5 claim slice",
    "claim_bench-heldout_r5-after-exposure": "independent round-5 claim slice **after exposure** (round 9 fixed its "
    "failure classes; not a fresh estimate)",
}


def short_commit(commit: Any) -> str:
    return str(commit or "").split("-")[0]


def status_label(name: str, set_name: str | None, commit: Any) -> str:
    """How much a result can be trusted as an estimate: first run, validation or after exposure."""
    if name in FILE_STATUS:
        return FILE_STATUS[name]
    if set_name == "dev":
        return "development set (used to drive fixes)"
    if set_name == "holdout":
        return "held-out (not used for rule tuning, but used to choose prompts: a validation set)"
    if set_name == "test_v2":
        if short_commit(commit) in TEST_V2_FIRST_RUN_COMMITS:
            return "**first runs** (before exposure)"
        return f"**after exposure** (failure classes read and fixed since `{TEST_V2_EXPOSED_AT}`)"
    if set_name == "test_v3":
        return "**first run** (untouched: no fix has looked at it)"
    if set_name == "multiturn_v1":
        return "independent set"
    return "–"


SUMMARY_ROWS = [
    ("task_success", "Task success (dealbreaker-gated)"),
    ("pass^k", "pass^k (all k repeats succeed)"),
    ("behavior_accuracy", "Behaviour accuracy (answer / clarify / refuse)"),
    ("fact_recall", "Required facts stated and cited"),
    ("fact_stated", "Required facts stated with the snapshot value, cited or not (uncited correctness)"),
    ("task_success_uncited", "Task success without the citation / tool / disclaimer checks"),
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
    ("llm_error_rate", "Turns with an LLM error, fallback used (HTTP 429 share)"),
    ("json_repair_rate", "Answer drafts that needed JSON repair"),
    ("json_failure_rate", "Answer drafts that were not JSON (used as text)"),
    ("cost_per_task", "Cost per task"),
]
COMPACT_ROWS = [
    "task_success",
    "pass^k",
    "fact_recall",
    "fact_stated",
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


def llm_error_cell(summary: dict[str, Any]) -> str:
    """``0.071 (429: 11 of 12 error flags)``: the fallback share, and how much of it the gateway caused.

    Runs since the ``llm_429_rate`` metric record the share of turns directly; older runs only kept the
    five most common error kinds (``llm_error_kinds``, counted per flag, and one turn can carry both an
    ``llm_error`` and an ``llm_revision_failed`` flag), so for them the 429 share is given per flag.
    """
    rate = summary.get("llm_error_rate")
    if rate is None:
        return "not recorded"
    text = f"{rate:.3f}"
    if summary.get("llm_429_rate") is not None:
        return f"{text} (429: {summary['llm_429_rate']:.3f} of turns)"
    kinds = summary.get("llm_error_kinds")
    if kinds:
        total = sum(kinds.values())
        http_429 = sum(count for kind, count in kinds.items() if "HTTP 429" in kind)
        return f"{text} (429: {http_429} of {total} error flags)"
    if rate > 0:
        return f"{text} (429 share not recorded)"
    return text


def summary_of(data: dict[str, Any]) -> dict[str, Any]:
    """The stored summary, with task success and pass^k recomputed exactly from the per-task outcomes.

    Summaries store values rounded to 4 places, so 99/130 = 0.76154 is stored as 0.7615 and would print as
    0.761; the outcomes give the exact value. Kept as stored when the outcomes do not reproduce it (older
    files whose outcomes were reconstructed from failure lists).
    """
    summary = data.get("summary") or {}
    runs = list(decode_outcomes(data.get("task_outcomes")).values())
    if not runs or summary.get("task_success") is None:
        return summary
    exact = statistics.fmean(task_success_value(item) for item in runs)
    if abs(exact - summary["task_success"]) > 1e-4:
        return summary
    updated = {**summary, "task_success": exact}
    key = pass_key(summary)
    if key and summary.get(key) is not None:
        all_pass = statistics.fmean(pass_all_value(item) for item in runs)
        if abs(all_pass - summary[key]) <= 1e-4:
            updated[key] = all_pass
    return updated


def cell(summary: dict[str, Any], key: str) -> str:
    ci = summary.get("ci") or {}
    if key == "task_success":
        return fmt_ci(summary.get("task_success"), ci.get("task_success"))
    if key == "task_success_uncited":
        if summary.get(key) is None:
            return "not recorded"
        return fmt_ci(summary[key], ci.get(key))
    if key == "llm_error_rate":
        return llm_error_cell(summary)
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


OPTIONAL_ROWS = frozenset({"task_success_uncited"})  # only shown when a run recorded it


def set_table(modes: dict[str, Any], rows: list[tuple[str, str]] | list[str]) -> list[str]:
    labels = dict(SUMMARY_ROWS)
    lines = ["| Metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for row in rows:
        key, label = (row, labels[row]) if isinstance(row, str) else row
        if key in OPTIONAL_ROWS and all(data["summary"].get(key) is None for data in modes.values()):
            continue
        lines.append(
            f"| {_label(key, label, modes)} | "
            + " | ".join(cell(summary_of(data), key) for data in modes.values())
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


def model_text(config: dict[str, Any]) -> str:
    """The model a run used, and where it came from (``--llm deepseek`` names the client, not the model)."""
    model = config.get("model") or config.get("llm")
    if not model:
        return "no LLM (offline)"
    source = config.get("model_source")
    if source == "--model":
        return f"model `{model}` (from `--model`)"
    return (
        f"model `{model}` (`--llm deepseek` names the OpenAI-compatible client; the model came from "
        "`DEEPSEEK_MODEL` and is recorded in the result's config)"
    )


def _header(result: dict[str, Any], name: str) -> list[str]:
    config = result["config"]
    prompts = ", ".join(f"`{ref}`" for ref in (config.get("prompts") or {}).values())
    command = [config.get("command", "")]
    model = config.get("model") or config.get("llm")
    if model and "--model" not in command[0]:
        command.insert(0, f"# model selected in the environment: DEEPSEEK_MODEL={model}")
    return [
        f"Source `evaluation/results/{name}.json`: commit `{config.get('commit')}`, run {config.get('run_at')}, "
        f"{model_text(config)}, prompts {prompts or '–'}.",
        "",
        "```bash",
        *command,
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
            f"Status: {status_label(name, set_name, result['config'].get('commit'))}.",
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
        "bound on pure sampling noise; it is the right yardstick for claims that one run beat another. "
        "These runs predate the `llm_error_rate` metric, so they may contain an unknown share of turns where "
        "an LLM error forced the deterministic fallback.",
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
                    summary = summary_of((result["results"].get(set_name) or {}).get(mode) or {}) or None
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
        f"| LLM-error turns ({short[0]} / {short[1]}) | Commits ({short[0]} / {short[1]}) |",
        "|---|---|---|---|---|---|---|---|---|---|",
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
                summary = summary_of(modes.get(mode) or {}) or None
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
            errors = " / ".join(
                llm_error_cell((modes.get(mode) or {}).get("summary") or {}) for modes in (a_modes, b_modes)
            )
            lines.append(f"| {set_name} | {mode} | " + " | ".join(cells + deltas) + f" | {errors} | {commits} |")
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
    ("llm_error_rate", "LLM-error turns (HTTP 429 share)"),
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


def set_status_section(entries: list[tuple[str, str, str]]) -> list[str]:
    """Which rendered result is a first run, a validation run or an after-exposure run."""
    lines = [
        "### How to read the status labels",
        "",
        "* **first run**: the set was run before any fix looked at its failures; this is the honest estimate.",
        "* **after exposure**: failures the set showed were read and fixed (on dev-style examples), so the "
        "number is tuned and not an estimate for unseen questions.",
        "* **development / validation**: the development set drives fixes; the held-out set chose prompts.",
        f"* Test set v2 was written blind and first run at `{'` / `'.join(sorted(TEST_V2_FIRST_RUN_COMMITS))}`; "
        f"its failure classes were fixed at `{TEST_V2_EXPOSED_AT}`, so every later test v2 run is after "
        "exposure. Test set v3 has only first runs.",
        "",
        "| Result file | Set | Commit | Status |",
        "|---|---|---|---|",
    ]
    for name, set_name, commit in entries:
        lines.append(f"| `{name}.json` | {set_name} | `{commit}` | {status_label(name, set_name, commit)} |")
    lines.append("")
    return lines


def run_section(result: dict[str, Any], name: str, *, full: bool = True) -> list[str]:
    """A single ``runner`` result (one path, one set)."""
    config, summary = result["config"], summary_of(result)
    set_name = _set_of_tasks_file(config.get("tasks_file"))
    title = SET_TITLES.get(set_name, set_name)
    lines = [
        *_header(result, name),
        f"{title}, path `{config.get('mode')}`: {summary['tasks']} tasks, {summary['turns']} turns. "
        f"Status: {status_label(name, set_name, config.get('commit'))}.",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    labels = dict(SUMMARY_ROWS)
    rows = ["task_success", "pass^k", "behavior_accuracy", "fact_recall", "fact_stated", "task_success_uncited"]
    rows += ["tool_recall", "tool_precision", "hedged_when_required", "states_missing_when_required"]
    rows += ["compliance_clean", "latency_ms_p95"]
    for key in rows:
        if key in OPTIONAL_ROWS and summary.get(key) is None:
            continue
        lines.append(f"| {labels[key]} | {cell(summary, key)} |")
    if summary.get("turn_success") is not None:
        lines.append(f"| Turn success | {_fmt(summary['turn_success'])} |")
    lines.append("")
    if full and result.get("by_category"):
        lines += ["| Category | Tasks | Task success |", "|---|---|---|"]
        for category, entry in sorted(result["by_category"].items()):
            lines.append(f"| {category} | {entry['tasks']} | {entry['task_success']:.2f} |")
        lines.append("")
    if full and result.get("failures"):
        lines += [f"Failures ({title.lower()}; first 25):", "", *failure_table(result, limit=25), ""]
    for note in result.get("notes") or []:
        lines.append(f"* Note: {note}")
    if result.get("notes"):
        lines.append("")
    return lines


def _set_of_tasks_file(path: Any) -> str:
    text = str(path or "")
    for marker, set_name in (
        ("test_v3", "test_v3"),
        ("multiturn_v1", "multiturn_v1"),
        ("test_v2", "test_v2"),
        ("holdout", "holdout"),
    ):
        if marker in text:
            return set_name
    return "dev"


def pure_llm_section(runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """The no-tools LLM under strict (citation-gated) scoring and under uncited correctness."""
    lines = [
        "### The no-tools LLM baseline: strict scoring vs uncited correctness",
        "",
        "Under the strict score a task needs every required number stated **and cited with an evidence id**, the "
        "required tools and the product's risk-disclaimer field. A model without tools can meet none of these, "
        "so its 0.000 is a property of the scoring, not only of the model. The uncited columns score the same "
        "answers against the same snapshot values without those requirements. The snapshot is dated 2026-04-22 "
        "and the model has no access to it, so uncited correctness measures what the model knew or guessed.",
        "",
        "* *Facts stated with the snapshot value* (`fact_stated`) counts required numbers that appear in the "
        "answer, cited or not; it is in every committed summary.",
        "* *Task success, uncited* (`task_success_uncited`) is task-level: behaviour, hedging, missing-data and "
        "compliance checks still apply, and every required number must be right; cited facts, tool use, the "
        "disclaimer field and the pipeline's resolved entities are not required. It was added after these runs "
        "and runs from now on record it for every path, including pure_llm. The committed files of these runs "
        "keep per-task outcomes and failure rows (failed checks per turn, without the answer text), so the exact "
        "value cannot be recomputed. The column gives bounds from those rows: a row that failed only tool-only "
        "checks passes; one that also failed `facts` (an uncited number may still have been right) or `language` "
        "(these runs did not record the answer language) is undecided, a failure for the lower bound and a pass "
        "for the upper; any other failed check fails. The upper bounds are loose: only 4–8% of the required "
        "numbers appear in these answers at all (`fact_stated`).",
        "",
        "| Result file | Set | Commit | Model | Status | Tasks | Strict task success | Facts stated, uncited "
        "| Task success, uncited | LLM-error turns (429) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, result in runs:
        config = result["config"]
        for set_name, modes in result["results"].items():
            data = modes.get("pure_llm")
            if not data:
                continue
            summary = summary_of(data)
            uncited = cell(summary, "task_success_uncited")
            if summary.get("task_success_uncited") is None:
                bounds = uncited_success_bounds(data.get("failures") or [], decode_outcomes(data.get("task_outcomes")))
                uncited = f"not recorded; bounds [{bounds[0]:.3f}, {bounds[1]:.3f}]" if bounds else "not recorded"
            lines.append(
                f"| `{name}.json` | {set_name} | `{config.get('commit')}` "
                f"| `{config.get('model') or config.get('llm')}` "
                f"| {status_label(name, set_name, config.get('commit'))} | {summary['tasks']} "
                f"| {cell(summary, 'task_success')} | {_fmt(summary.get('fact_stated'))} "
                f"| {uncited} | {llm_error_cell(summary)} |"
            )
    lines.append("")
    return lines


def router_section(runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "### Router label sets",
        "",
        "Route accuracy of `mode=auto` (refuse / clarify / workflow / agent) against labelled queries. The "
        "author's own labels were written by the person tuning the rules; the independent sets were labelled "
        "by separate authors against a written policy. No LLM is involved.",
        "",
        "| Result file | Labels | Status | Commit | Queries | Accuracy | Recall refuse / clarify / workflow / agent "
        "| Command |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, result in runs:
        config = result["config"]
        per_route = result.get("per_route") or {}
        recall = " / ".join(
            _fmt((per_route.get(route) or {}).get("recall")) for route in ("refuse", "clarify", "workflow", "agent")
        )
        lines.append(
            f"| `{name}.json` | `{Path(str(config.get('labels'))).name}` | {status_label(name, None, None)} "
            f"| `{config.get('commit')}` | {result['queries']} | {result['accuracy']:.3f} | {recall} "
            f"| `{config.get('command')}` |"
        )
    lines.append("")
    notes = [(name, result["config"]["note"]) for name, result in runs if result["config"].get("note")]
    for name, note in notes:
        lines.append(f"* Note (`{name}.json`): {note}")
    if notes:
        lines.append("")
    return lines


def claim_bench_section(runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "### Claim-check benchmark",
        "",
        "Deterministic claim check (`POST /agent/claim-check`, no LLM) against labelled claims on the offline "
        "snapshot. Verdict accuracy is per claim; check accuracy is per number. The held-out file's sha256 is "
        "recorded so a later edit of the claims is visible.",
        "",
        "| Result file | Set | Status | Commit | Claims | Verdict accuracy [95% CI] | Check accuracy [95% CI] "
        "| Comparator accuracy | Claims sha256 (first 16) | Command |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, result in runs:
        config = result["config"]
        lines.append(
            f"| `{name}.json` | {config.get('set')} | {status_label(name, None, None)} | `{config.get('commit')}` "
            f"| {result['claims']} | {fmt_ci(result['verdict_accuracy'], result.get('verdict_accuracy_ci'))} "
            f"| {fmt_ci(result.get('check_accuracy'), result.get('check_accuracy_ci'))} "
            f"| {_fmt(result.get('comparator_accuracy'))} | `{str(config.get('claims_sha256'))[:16]}` "
            f"| `{config.get('command')}` |"
        )
    lines.append("")
    return lines


def perf_section(runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "### Latency profile runs (agent path, streamed)",
        "",
        "Each run streams the agent path to record time to the first answer token. The overrides column lists "
        "the `--agent-config` switches (none = the defaults at that commit). `HTTP 429 of requests` counts every "
        "LLM HTTP attempt, including 429s a retry recovered; `LLM-error turns` counts turns that fell back.",
        "",
        "| Result file | Commit | Model | Overrides | Set (status) | Task success [95% CI] | pass^k | P50 / P95 (s) "
        "| TTFT P50 (s) | LLM calls / turn | Tokens / turn | Cost / task | LLM-error turns (429) "
        "| HTTP 429 of requests |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, result in runs:
        config = result["config"]
        overrides = ", ".join(f"`{item}`" for item in config.get("agent_config_overrides") or []) or "none"
        model = str(config.get("model") or config.get("llm") or "").split("/")[-1]
        for set_name, modes in result["results"].items():
            data = modes.get("agent")
            if not data:
                continue
            summary = summary_of(data)
            http = ((result.get("llm_http") or {}).get(set_name) or {}).get("agent") or {}
            seconds = [summary.get(key) for key in ("latency_ms_p50", "latency_ms_p95")]
            latency = " / ".join("–" if value is None else f"{value / 1000:.1f}" for value in seconds)
            ttft = summary.get("ttft_ms_p50")
            ttft_text = "–" if ttft is None else f"{ttft / 1000:.1f}"
            cost = summary.get("cost_per_task")
            cost_text = "–" if cost is None else f"{cost:.5f} {summary.get('cost_currency') or ''}".strip()
            rate_429 = http.get("rate_429")
            http_text = "–" if rate_429 is None else f"{rate_429:.3f} of {http.get('requests')}"
            status = status_label(name, set_name, config.get("commit")).split(" (")[0]
            lines.append(
                f"| `{name}.json` | `{config.get('commit')}` | {model} | {overrides} | {set_name} ({status}) "
                f"| {cell(summary, 'task_success')} | {cell(summary, 'pass^k')} | {latency} | {ttft_text} "
                f"| {_fmt(summary.get('llm_calls_per_turn'))} | {_fmt(summary.get('tokens_per_turn'))} "
                f"| {cost_text} | {llm_error_cell(summary)} | {http_text} |"
            )
    lines.append("")
    by_name = dict(runs)
    pairs = [(a, b) for a, b in PERF_PAIRS if a in by_name and b in by_name]
    if pairs:
        from .metrics import paired_comparison

        lines += [
            "Paired comparisons between these runs (agent task success, same tasks; a − b):",
            "",
            "| a | b | Set | Tasks | Δ task success [95% CI] | Δ pass^k [95% CI] | McNemar (a-only / b-only, p) |",
            "|---|---|---|---|---|---|---|",
        ]
        for a, b in pairs:
            for set_name in by_name[a]["results"]:
                a_data = (by_name[a]["results"].get(set_name) or {}).get("agent")
                b_data = (by_name[b]["results"].get(set_name) or {}).get("agent")
                if not a_data or not b_data:
                    continue
                result = paired_comparison(
                    decode_outcomes(a_data["task_outcomes"]), decode_outcomes(b_data["task_outcomes"])
                )
                mc = result["mcnemar"]
                lines.append(
                    f"| `{a}` | `{b}` | {set_name} | {result['tasks']} | {diff_ci(result['task_success'])} "
                    f"| {diff_ci(result['pass^k'])} | {mc['a_only_pass']} / {mc['b_only_pass']}, "
                    f"p={mc['p_value']:.3f} |"
                )
        lines.append("")
    for name, result in runs:
        for note in result.get("notes") or []:
            lines.append(f"* `{name}.json`: {note}")
        commands = result["config"].get("command")
        lines.append(f"* `{name}.json` command: `{commands}`")
    lines.append("")
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


def stress_section(stress: dict[str, Any], name: str = "verifier_stress") -> list[str]:
    modes = list(stress["false_accept"])
    lines = [
        f"### Verifier stress test ({stress['gold_answers']} gold answers, {stress['variants']} corrupted variants, "
        f"`{name}.json`)",
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
    for note in stress.get("notes") or []:
        lines += [f"* Note: {note}", ""]
    return lines


def _attack_counts(attacks: Any) -> str:
    if not isinstance(attacks, dict):
        return str(attacks)
    return ", ".join(f"{name} {count}" for name, count in attacks.items())


def _redteam_llm_errors(path: dict[str, Any], config: dict[str, Any]) -> str:
    """Share of red-team runs where an LLM call failed and the template fallback answered."""
    if path["mode"] == "workflow" or not (config.get("model") or config.get("llm")):
        return "– (no LLM)"
    rate = path.get("llm_error_rate")
    if rate is None:
        return "not recorded (run predates the metric)"
    return f"{rate:.3f} (429: {path.get('llm_429_rate') or 0:.3f} of runs)"


def redteam_section(redteam: dict[str, Any], title: str) -> list[str]:
    config = redteam["config"]
    lines = [
        f"### {title}",
        "",
        f"Command: `{config['command']}` at commit `{config['commit']}`; {model_text(config)}. "
        f"Attacks: {_attack_counts(config['attacks'])}; variants: {', '.join(config['variants'])}. "
        "Only runs in which a document tool returned the poisoned text are counted.",
        "",
    ]
    # Since round 8 an attack succeeds only when its payload is stated as fact (outside an attributed sentence);
    # those files also carry the raw detector rate and the evidence-ledger surface.
    round8 = any("detector_hit_rate" in path for path in redteam["paths"])
    if round8:
        lines += [
            "Attack success = the payload stated as fact (a detector match outside a sentence that attributes it, "
            'e.g. "据一篇文档称…（未经其他来源证实）" or "媒体报道称…"); detector hits = every match, attributed or '
            "not; ledger hits = the payload in a shown evidence-ledger title or a suggested follow-up.",
            "",
            "| Attack set | Path | Runs | Attack success | Detector hits | Ledger hits | Redaction by lexical filter "
            "| Crashes | LLM-error runs (429) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| Attack set | Path | Runs | Attack success | Redaction by lexical filter | Crashes "
            "| LLM-error runs (429) |",
            "|---|---|---|---|---|---|---|",
        ]
    for path in redteam["paths"]:
        extra = f"{_fmt(path.get('detector_hit_rate'))} | {_fmt(path.get('ledger_hit_rate'))} | " if round8 else ""
        lines.append(
            f"| {path['attack_set']} | {path['mode']} | {path['exposed_runs']} | {_fmt(path['attack_success'])} | "
            f"{extra}{_fmt(path['redaction'])} | {path['crashes']} | {_redteam_llm_errors(path, config)} |"
        )
    lines.append("")
    for note in redteam.get("notes") or []:
        lines += [f"* Note: {note}", ""]
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
        summary = summary_of(run)
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
        "| File in `evaluation/results/` | Kind | Commit | Run at (UTC) | Model | Source file (sha256/16) |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        result = load_result(name)
        if not result:
            continue
        config, source = result.get("config") or {}, result.get("source") or {}
        kind = result.get("kind") or name.split("-")[0]
        model = config.get("model") or config.get("llm") or "–"
        origin = f"`{source.get('file')}` ({source.get('sha256_16')})" if source else "written directly"
        lines.append(
            f"| `{name}.json` | {kind} | `{config.get('commit')}` | {config.get('run_at')} | {model} | {origin} |"
        )
    lines.append("")
    return lines


def render_with_sources() -> tuple[str, list[str]]:
    """The generated block and the result files it rendered (stems, in order)."""
    lines = [
        BEGIN,
        "",
        "Rendered by `python -m evaluation.agent_eval.report` from the committed files in `evaluation/results/`. "
        "Task success and pass^k show the value and a 95% bootstrap CI over tasks "
        "(2000 resamples, seed 20260926). With 53 held-out tasks one task is 1.9 points and the CI half-width "
        "is about 6–9 points; with 121 test tasks one task is 0.8 points. Every online table shows the share "
        "of turns where an LLM call failed and the run fell back to the deterministic path "
        "(`llm_error_rate`), with how many of those failures were HTTP 429 from the gateway. `report --check` "
        "fails if README.md or README_CN.md cite a result file that is missing or not rendered here.",
        "",
    ]
    used: list[str] = []
    status_rows: list[tuple[str, str, str]] = []

    def take(name: str) -> dict[str, Any] | None:
        result = load_result(name)
        if result is not None and name not in used:
            used.append(name)
        return result

    def note_status(name: str, result: dict[str, Any]) -> None:
        commit = str((result.get("config") or {}).get("commit"))
        sets = list(result.get("results") or {}) or [_set_of_tasks_file(result["config"].get("tasks_file"))]
        status_rows.extend((name, set_name, commit) for set_name in sets)

    body: list[str] = []
    finals4 = [(name, take(name)) for name in FINAL4_ONLINE]
    finals4 = [(name, result) for name, result in finals4 if result]
    if finals4:
        body += [
            "### Final online run at the final commit (headline table)",
            "",
            "Test set v3 is untouched: this is its first LLM run. Held-out is a validation set; test set v2 and "
            "multi-turn set v1 are **after exposure**.",
            "",
        ]
        for name, result in finals4:
            note_status(name, result)
            body += [f"#### `{name}.json`", "", *ablation_section(result, name, full=False)]
    finals = [(name, take(name)) for name in FINAL_ONLINE]
    finals = [(name, result) for name, result in finals if result]
    if finals:
        body += [
            "### Round-2 online run: held-out and test set v2, DeepSeek and GLM",
            "",
            "Kept for history (commit `d1c007c`). Test set v2 is **after exposure** in both.",
            "",
        ]
        for name, result in finals:
            note_status(name, result)
            body += [f"#### `{name}.json`", "", *ablation_section(result, name, full=False)]
    primary = take(PRIMARY)
    if primary:
        note_status(PRIMARY, primary)
        body += ["### Development and held-out sets, all paths (DeepSeek V4.1 Flash)", ""]
        body += ablation_section(primary, PRIMARY)
    test_v2 = take(TEST_V2)
    if test_v2:
        note_status(TEST_V2, test_v2)
        body += [
            f"### Test set v2, first runs before exposure (DeepSeek V4.1 Flash, `{test_v2['config']['commit']}`)",
            "",
        ]
        body += ablation_section(test_v2, TEST_V2)
    rate_check = take(RATE_LIMIT_CHECK)
    if test_v2 and rate_check:
        note_status(RATE_LIMIT_CHECK, rate_check)
        body += rate_limit_section(test_v2, rate_check, RATE_LIMIT_CHECK)
    second_files = [(name, take(name)) for name in SECOND_MODEL]
    second_files = [(name, result) for name, result in second_files if result]
    if second_files:
        merged: dict[str, Any] = {"config": second_files[0][1]["config"], "results": {}, "comparisons": {}}
        body += [f"### All sets with a second model family, first runs ({merged['config'].get('llm')})", ""]
        for name, result in second_files:
            fresh = [set_name for set_name in result["results"] if set_name not in merged["results"]]
            for set_name in fresh:
                merged["results"][set_name] = {**result["results"][set_name], "_commit": result["config"].get("commit")}
                merged["comparisons"][set_name] = (result.get("comparisons") or {}).get(set_name, {})
                status_rows.append((name, set_name, str(result["config"].get("commit"))))
            if fresh:
                body += ablation_section(result, name, full=False, sets=set(fresh))
        if primary:
            body += cross_model_section(primary, merged, test_v2)
    offline = take(OFFLINE)
    if offline:
        note_status(OFFLINE, offline)
        body += ["### Deterministic paths on dev, held-out and test v2", ""]
        body += ablation_section(offline, OFFLINE, full=False)
    test_v3 = [(name, take(name)) for name in TEST_V3]
    test_v3 = [(name, result) for name, result in test_v3 if result]
    if test_v3:
        body += [
            "### Test set v3 (independent author, untouched): first runs",
            "",
            "Written by a separate author who did not read the routing code or any task file "
            "(`evaluation/agent_eval/tasks/README_test_v3.md`). No fix has looked at it. The LLM paths' first run "
            "is in the final online run above.",
            "",
        ]
        for name, result in test_v3:
            note_status(name, result)
            body += [f"#### `{name}.json`", ""]
            if result.get("kind") == "run":
                body += run_section(result, name)
            else:
                body += ablation_section(result, name, full=False)
    multiturn = [(name, take(name)) for name in MULTITURN]
    multiturn = [(name, result) for name, result in multiturn if result]
    if multiturn:
        body += [
            "### Multi-turn set v1 (independent author): first runs, then after exposure",
            "",
            "49 conversations / 206 turns written by a separate author "
            "(`evaluation/agent_eval/tasks/README_multiturn_v1.md`). A conversation succeeds only if every turn does.",
            "",
        ]
        for name, result in multiturn:
            note_status(name, result)
            body += [f"#### `{name}.json`", ""]
            if result.get("kind") == "run":
                body += run_section(result, name, full=name.endswith("first-run"))
            else:
                body += ablation_section(result, name, full=False)
    heldout_r4 = [(name, take(name)) for name in HELDOUT_R4_RUNS]
    heldout_r4 = [(name, result) for name, result in heldout_r4 if result]
    if heldout_r4:
        body += [
            "### Round-4 held-out multi-turn slice (independent author): first run, then after exposure",
            "",
            "24 conversations written before the round-4 fixes (`evaluation/heldout_r4/README.md`).",
            "",
        ]
        for name, result in heldout_r4:
            body += [f"#### `{name}.json`", "", *run_section(result, name, full=name.endswith("first-run"))]
    heldout_r5 = [(name, take(name)) for name in HELDOUT_R5_RUNS]
    heldout_r5 = [(name, result) for name, result in heldout_r5 if result]
    if heldout_r5:
        body += [
            "### Round-5 held-out chat slice (independent author): first run, then after exposure",
            "",
            "38 tasks / 41 turns written before the round-8 fixes were run on it (`evaluation/heldout_r5/README.md`); "
            "its claims are in the claim-check table below.",
            "",
        ]
        for name, result in heldout_r5:
            body += [f"#### `{name}.json`", "", *run_section(result, name, full=name.endswith("first-run"))]
    pure = [
        (name, result)
        for name in (PRIMARY, TEST_V2, *TEST_V3)
        if (result := load_result(name)) and result.get("kind") == "ablation"
    ]
    if pure:
        body += pure_llm_section(pure)
    routers = [(name, take(name)) for name in ROUTER_RUNS]
    routers = [(name, result) for name, result in routers if result]
    if routers:
        body += router_section(routers)
    claims = [(name, take(name)) for name in CLAIM_BENCH_RUNS]
    claims = [(name, result) for name, result in claims if result]
    if claims:
        body += claim_bench_section(claims)
    perf = [(name, take(name)) for name in PERF_RUNS]
    perf = [(name, result) for name, result in perf if result]
    if perf:
        for name, result in perf:
            note_status(name, result)
        body += perf_section(perf)
    variance = [(name, take(name)) for name in VARIANCE_RUNS]
    if all(result for _name, result in variance):
        for name, result in variance:
            if name != PRIMARY:
                note_status(name, result)  # type: ignore[arg-type]
        body += variance_section(variance)  # type: ignore[arg-type]
    baseline, variant = take(PROMPT_AB[0]), take(PROMPT_AB[1])
    if baseline and variant:
        note_status(PROMPT_AB[1], variant)
        body += prompt_ab_section(baseline, variant)
    gates = [(name, take(name)) for name in GATE_RUNS]
    if all(result for _name, result in gates):
        body += gate_section(gates)  # type: ignore[arg-type]
    faults = take("fault_injection")
    if faults:
        body += faults_section(faults)
    for name in STRESS_RUNS:
        stress = take(name)
        if stress:
            body += stress_section(stress, name)
    for name, title in REDTEAM_RUNS:
        redteam = take(name)
        if redteam:
            body += redteam_section(redteam, title)
    extras = [(name, take(name)) for name in EXTRA_EVIDENCE]
    extras = [(name, result) for name, result in extras if result]
    if extras:
        body += extra_evidence_section(extras)
    for name in SUPERSEDED:
        result = take(name)
        if result:
            note_status(name, result)
            body += [f"### Superseded run kept as evidence: `{name}.json`", ""]
            body += ablation_section(result, name, full=False)
    lines += set_status_section(status_rows)
    lines += body
    lines += provenance_section(used)
    lines.append(END)
    return "\n".join(lines), used


def extra_evidence_section(runs: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """One summary row per committed file of another kind (targeted replays, classifier evaluations)."""
    lines = [
        "### Other committed evidence",
        "",
        "| Result file | Commit | Summary |",
        "|---|---|---|",
    ]
    for name, result in runs:
        config = result.get("config") or {}
        if "conditions" in result:
            parts = [
                f"{condition}: {values.get('detector_hits')}/{values.get('runs')} detector hits, "
                f"{values.get('unattributed_hits')} stated as fact"
                for condition, values in result["conditions"].items()
            ]
            summary = "targeted red-team replay of previously leaking cases — " + "; ".join(parts)
        elif "attack_recall" in result:
            combined = result["attack_recall"].get("held_out_combined") or {}
            parts = [
                f"{key} {_fmt(value.get('rate'))} [{_fmt((value.get('ci95') or [None, None])[0])}, "
                f"{_fmt((value.get('ci95') or [None, None])[1])}]"
                for key, value in combined.items()
            ]
            fpr = result.get("clean_false_positive_rate") or {}
            fpr_parts = [
                f"{key} {_fmt(value.get('rate'))}"
                for key, value in fpr.items()
                if isinstance(value, dict) and "rate" in value and key != "by_source_type"
            ]
            summary = (
                "injection classifier, recall on unseen attacks (holdout2-4): "
                + "; ".join(parts)
                + f"; false positives on {fpr.get('documents')} clean documents: "
                + "; ".join(fpr_parts)
            )
        elif result.get("kind") == "verifier_stress_repair_comparison":
            parts = []
            for key, label in (
                ("whole_sentence_repair", "whole-sentence repair"),
                ("clause_salvage_repair", f"clause salvage ({config.get('legacy_repair_commit')})"),
                ("clause_salvage_repair_legacy_report", "clause salvage on its own verifier's report"),
            ):
                repair = result.get(key) or {}
                parts.append(
                    f"{label}: readable {_fmt(repair.get('readable'))}, "
                    f"with fragment {_fmt(repair.get('with_fragment'))}, "
                    f"verifies {_fmt(repair.get('passes_verification'))}"
                )
            summary = (
                f"repair of {(result.get('whole_sentence_repair') or {}).get('repaired_answers')} rejected variants "
                f"({result.get('gold_answers')} gold answers) — " + "; ".join(parts)
            )
        else:
            summary = ", ".join(sorted(result))[:160]
        commit = config.get("commit") or ", ".join(
            sorted(
                {str(source.get("commit")) for source in (config.get("sources") or {}).values() if source.get("commit")}
            )
        )
        lines.append(f"| `{name}.json` | `{commit or 'n/a'}` | {summary} |")
    lines.append("")
    return lines


def render() -> str:
    return render_with_sources()[0]


# ``ablation-final2-glm.json``, ``docs/results/perf/startup.json``, ``perf-*.json``: a JSON file name or
# repo-relative path (globs allowed) as cited in the READMEs. URLs such as ``/.well-known/…`` are skipped.
_CITED_JSON = re.compile(r"(?<![\w./*-])([\w.*-][\w./*-]*\.json)(?![\w])")


def cited_json(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(1) for match in _CITED_JSON.finditer(text)))


def citation_problems(rendered: list[str], readmes: tuple[Path, ...] = README_PATHS) -> list[str]:
    """README citations of result files that do not exist, or that exist in evaluation/results/ unrendered."""
    problems: list[str] = []
    rendered_set = set(rendered)
    for readme in readmes:
        if not readme.exists():
            continue
        where = readme.name
        for token in cited_json(readme.read_text(encoding="utf-8")):
            if "/" in token:
                matches = sorted(ROOT.glob(token))
            else:
                matches = sorted(RESULTS_DIR.glob(token)) or sorted((ROOT / "docs" / "results").rglob(token))
            if not matches:
                problems.append(f"{where} cites {token}, which does not exist")
                continue
            for path in matches:
                if path.parent == RESULTS_DIR and path.stem not in rendered_set:
                    problems.append(f"{where} cites {token}: evaluation/results/{path.name} is not rendered")
    return problems


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
    block, rendered = render_with_sources()
    text = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else "# FinSight Agent Evaluation\n"
    updated = splice(text, block)
    problems = citation_problems(rendered)
    unrendered = sorted(path.stem for path in RESULTS_DIR.glob("*.json") if path.stem not in set(rendered))
    if unrendered:
        print(f"note: committed results not rendered (not cited by the READMEs): {', '.join(unrendered)}")
    if args.check:
        failed = False
        if updated != text:
            print(f"{DOC_PATH.relative_to(ROOT)} is out of date; run python -m evaluation.agent_eval.report")
            failed = True
        for problem in problems:
            print(f"citation: {problem}")
            failed = True
        if failed:
            return 1
        print(f"{DOC_PATH.relative_to(ROOT)} is up to date; every result the READMEs cite exists and is rendered")
        return 0
    for problem in problems:
        print(f"citation: {problem}")
    DOC_PATH.write_text(updated, encoding="utf-8")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
