"""Render ``docs/agent-eval.md`` from evaluation outputs.

Only the block between the GENERATED markers is rewritten; hand-written interpretation outside the
markers is preserved. Every number comes from ``outputs/agent_eval/*.json`` produced by the commands
listed in the report.

    python -m evaluation.agent_eval.report --ablation outputs/agent_eval/ablation-online-v1.json \
        --prompt-ab outputs/agent_eval/ablation-online-v2.json

Optional inputs that are rendered when present: ``fault_injection.json``, ``verifier_stress.json`` and
``redteam.json`` in ``outputs/agent_eval/``.
"""

from __future__ import annotations

import argparse
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
    ("llm_calls_per_turn", "LLM calls per turn"),
    ("tokens_per_turn", "LLM tokens per turn"),
    ("reasoning_tokens_per_turn", "Reasoning tokens per turn"),
    ("cache_hit_ratio", "Prompt cache-hit ratio"),
    ("first_pass_verification", "LLM draft verified on first pass"),
    ("revise_rate", "LLM drafts sent back for revision"),
    ("cost_per_task", "Cost per task"),
]


def _fmt(value: Any, key: str = "") -> str:
    if value is None:
        return "–"
    if key.startswith("cost"):
        return f"{value:.5f}"
    if isinstance(value, float):
        return f"{value:.1f}" if value > 1.5 else f"{value:.3f}"
    return str(value)


def render(
    ablation: dict[str, Any],
    faults: dict[str, Any] | None,
    *,
    prompt_ab: dict[str, Any] | None = None,
    stress: dict[str, Any] | None = None,
    redteam: dict[str, Any] | None = None,
) -> str:
    config = ablation["config"]
    lines = [
        BEGIN,
        "",
        f"Generated from commit `{config.get('commit')}` at {config.get('run_at')} "
        f"(LLM: {config.get('llm') or 'none — offline'}, {config.get('repeats', 1)} repeat(s) for LLM modes; "
        f"evaluation date fixed to {config.get('eval_today')}). Prompts: "
        + ", ".join(f"`{ref}`" for ref in (config.get("prompts") or {}).values())
        + ". Costs are the gateway-reported per-request charges.",
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
            if key == "cost_per_task":
                currency = next(
                    (d["summary"].get("cost_currency") for d in modes.values() if d["summary"].get("cost_currency")), ""
                )
                label = f"{label} ({currency})" if currency else label
            lines.append(
                f"| {label} | " + " | ".join(_fmt(data["summary"].get(key), key) for data in modes.values()) + " |"
            )
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
        for mode in ("workflow", "agent"):
            data = modes.get(mode)
            if not data or not data["failures"]:
                continue
            seen: set[str] = set()
            lines += [
                f"Remaining {mode} failures ({title.lower()}; each task listed once across repeats):",
                "",
                "| Task | Query | Failed checks |",
                "|---|---|---|",
            ]
            for failure in data["failures"]:
                if failure["task"] in seen:
                    continue
                seen.add(failure["task"])
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
    if prompt_ab:
        lines += _prompt_ab_section(ablation, prompt_ab)
    if stress:
        lines += _stress_section(stress)
    if redteam:
        lines += _redteam_section(redteam)
    lines.append(END)
    return "\n".join(lines)


_AB_KEYS = [
    ("task_success", "Task success"),
    ("pass^3", "pass^3"),
    ("tool_precision", "Tool precision"),
    ("first_pass_verification", "Draft verified on first pass"),
    ("llm_calls_per_turn", "LLM calls per turn"),
    ("tokens_per_turn", "Tokens per turn"),
    ("latency_ms_p95", "Latency P95 (ms)"),
    ("cost_per_task", "Cost per task"),
]


def _prompt_ab_section(baseline: dict[str, Any], variant: dict[str, Any]) -> list[str]:
    base_refs = baseline["config"].get("prompts") or {}
    variant_refs = variant["config"].get("prompts") or {}
    lines = [
        "### Prompt A/B",
        "",
        f"Baseline `{base_refs.get('agent_system')}` vs variant `{variant_refs.get('agent_system')}` "
        f"(commit `{variant['config'].get('commit')}`, command `{variant['config'].get('command')}`).",
        "",
    ]
    for set_name in ("dev", "holdout"):
        for mode in ("workflow_llm", "agent"):
            base = ((baseline["results"].get(set_name) or {}).get(mode) or {}).get("summary")
            other = ((variant["results"].get(set_name) or {}).get(mode) or {}).get("summary")
            if not base or not other:
                continue
            lines += [f"{set_name} · {mode}:", "", "| Metric | baseline | variant |", "|---|---|---|"]
            for key, label in _AB_KEYS:
                lines.append(f"| {label} | {_fmt(base.get(key), key)} | {_fmt(other.get(key), key)} |")
            lines.append("")
    return lines


def _stress_section(stress: dict[str, Any]) -> list[str]:
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
    return lines


def _redteam_section(redteam: dict[str, Any]) -> list[str]:
    config = redteam["config"]
    lines = [
        "### Prompt-injection red team",
        "",
        f"Command: `{config['command']}` at commit `{config['commit']}` (LLM: {config.get('llm') or 'none'}). "
        f"Attacks: {config['attacks']}; variants: {', '.join(config['variants'])}. Only runs in which a "
        "document tool returned the poisoned text are counted.",
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


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render docs/agent-eval.md from evaluation outputs.")
    parser.add_argument("--ablation", default=str(OUTPUT_DIR / "ablation.json"))
    parser.add_argument("--prompt-ab", default="", help="Ablation output of a prompt variant to compare.")
    args = parser.parse_args(argv)
    ablation = json.loads(Path(args.ablation).read_text(encoding="utf-8"))
    block = render(
        ablation,
        _load(OUTPUT_DIR / "fault_injection.json"),
        prompt_ab=_load(Path(args.prompt_ab)) if args.prompt_ab else None,
        stress=_load(OUTPUT_DIR / "verifier_stress.json"),
        redteam=_load(OUTPUT_DIR / "redteam.json"),
    )
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
