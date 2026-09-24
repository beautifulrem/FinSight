"""Render ``docs/agent-eval.md`` from evaluation outputs.

Only the block between the GENERATED markers is rewritten; hand-written interpretation outside the
markers is preserved. Every number comes from ``outputs/agent_eval/*.json`` produced by the commands
listed in the report.

    python -m evaluation.agent_eval.report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "agent_eval"
DOC_PATH = ROOT / "docs" / "agent-eval.md"
BEGIN = "<!-- BEGIN GENERATED: python -m evaluation.agent_eval.report -->"
END = "<!-- END GENERATED -->"

SUMMARY_ROWS = [
    ("task_success", "Task success (dealbreaker-gated)"),
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
    ("tokens_per_turn", "LLM tokens per turn"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.1f}" if value > 1.5 else f"{value:.3f}"
    return str(value)


def render(ablation: dict[str, Any], faults: dict[str, Any] | None) -> str:
    config = ablation["config"]
    lines = [
        BEGIN,
        "",
        f"Generated from commit `{config.get('commit')}` at {config.get('run_at')} "
        f"(LLM: {config.get('llm') or 'none — offline'}; evaluation date fixed to {config.get('eval_today')}).",
        "",
        "Commands:",
        "",
        "```bash",
        config.get("command", "python -m evaluation.agent_eval.ablation"),
        (faults or {}).get("config", {}).get("command", "python -m evaluation.agent_eval.fault_injection"),
        "```",
        "",
    ]
    for set_name, title in (("dev", "Development set"), ("holdout", "Held-out set")):
        modes = ablation["results"].get(set_name) or {}
        if not modes:
            continue
        first = next(iter(modes.values()))["summary"]
        lines += [
            f"### {title} ({first['tasks']} tasks, {first['turns']} turns)",
            "",
            "| Metric | " + " | ".join(modes) + " |",
            "|---|" + "---|" * len(modes),
        ]
        for key, label in SUMMARY_ROWS:
            lines.append(f"| {label} | " + " | ".join(_fmt(data["summary"].get(key)) for data in modes.values()) + " |")
        pass_keys = sorted({key for data in modes.values() for key in data["summary"] if key.startswith("pass^")})
        for key in pass_keys:
            lines.append(f"| {key} | " + " | ".join(_fmt(data["summary"].get(key)) for data in modes.values()) + " |")
        lines.append("")
        categories = sorted({name for data in modes.values() for name in data["by_category"]})
        lines += [
            f"Task success by category ({title.lower()}):",
            "",
            "| Category | " + " | ".join(modes) + " |",
            "|---|" + "---|" * len(modes),
        ]
        for category in categories:
            cells = []
            for data in modes.values():
                entry = data["by_category"].get(category)
                cells.append(f"{entry['task_success']:.2f} ({entry['tasks']})" if entry else "–")
            lines.append(f"| {category} | " + " | ".join(cells) + " |")
        lines.append("")
        workflow = modes.get("workflow")
        if workflow and workflow["failures"]:
            lines += [
                f"Remaining workflow failures ({title.lower()}):",
                "",
                "| Task | Query | Failed checks |",
                "|---|---|---|",
            ]
            for failure in workflow["failures"]:
                query = failure["query"].replace("|", "\\|")
                lines.append(f"| {failure['task']} | {query} | {', '.join(failure['failed_checks'])} |")
            lines.append("")

    if faults:
        lines += [
            f"### Fault injection (overall graceful rate {faults['overall_graceful_rate']:.2f})",
            "",
            "| Scenario | Expectation | Runs | Graceful | Tool errors seen |",
            "|---|---|---|---|---|",
        ]
        for scenario in faults["scenarios"]:
            errors = sorted({code for result in scenario["results"] for code in result["tool_errors"] if code})
            lines.append(
                f"| {scenario['scenario']} | {scenario['expectation']} | {scenario['runs']} | "
                f"{scenario['graceful_rate']:.2f} | {', '.join(errors) or '–'} |"
            )
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def main() -> None:
    ablation = json.loads((OUTPUT_DIR / "ablation.json").read_text(encoding="utf-8"))
    faults_path = OUTPUT_DIR / "fault_injection.json"
    faults = json.loads(faults_path.read_text(encoding="utf-8")) if faults_path.exists() else None
    block = render(ablation, faults)
    if DOC_PATH.exists():
        text = DOC_PATH.read_text(encoding="utf-8")
        if BEGIN in text and END in text:
            text = text[: text.index(BEGIN)] + block + text[text.index(END) + len(END) :]
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
    else:
        text = "# FinSight Agent Evaluation\n\n" + block + "\n"
    DOC_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
