"""Verifier stress test: how often does the evidence verifier accept a wrong number?

Gold answers come from the offline workflow over the development tasks (replayed tools); every gold
answer passes verification. Each number in a gold answer is then corrupted in two ways:

* ``perturb_<pct>``: the number is scaled by (1 ± pct) and written with the same decimals, as a model
  that misreads or miscalculates would.
* ``swap``: the number is replaced by a number that exists in *another* evidence item of the same
  run but not in the evidence cited by that sentence (the wrong company, period or metric). This is
  the failure a run-level check cannot see.

Every corrupted answer is checked by three verifier modes:

* ``legacy``: the original verifier. A number passes if it matches any number anywhere in the run's
  evidence under any of 13 unit scales with a tolerance of max(0.011, 0.5%).
* ``run``: unit- and precision-aware matching (a number written with d decimals may be off by half a
  unit in the last place) but still against all evidence of the run.
* ``claim``: the current verifier. Like ``run``, but each number must be found in the evidence cited
  in its own sentence.
* ``claim_derived``: ``claim`` plus the opt-in derived-number rule (a number equal to the difference, sum,
  ratio or percent change of two supported numbers stated in the same cited sentence passes).

The false-accept rate is the share of corrupted answers that still pass. The true-accept rate is the
share of gold answers that pass; it must stay at 1.0 for every mode.

Every corrupted answer the claim verifier rejects is then repaired as the graph repairs an LLM draft
(``repair_answer`` with the gold template answer as fallback), and the ``repair`` section reports how
readable the result is (``verifier.readability_issues``), whether it contains a fragment (a sentence that
is not a whole sentence of the draft or the fallback), whether it verifies, and how much of the untouched
content survives.

    python -m evaluation.agent_eval.verifier_stress
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.verifier import (
    _CITATION,
    claim_numbers,
    readability_issues,
    repair_answer,
    verify_answer,
    whole_sentences,
)
from query_intelligence.chatbot import detect_query_language

from .runner import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SNAPSHOT,
    DEFAULT_TASKS,
    EVAL_TODAY,
    _command,
    _display_path,
    _git_commit,
    build_offline_service,
    build_registry,
    load_tasks,
)

PERTURBATIONS = (0.01, 0.05, 0.2)
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w])")
_MODES = ("legacy", "run", "claim", "claim_derived")


def gold_answers(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    service = build_offline_service()
    registry, _ = build_registry(service, snapshot=DEFAULT_SNAPSHOT, record=False)
    runtime = AgentRuntime(service, registry, None, today=lambda: EVAL_TODAY)
    graph = runtime.build_graph()
    golds = []
    try:
        for task in tasks:
            turn = task["turns"][0]
            if turn.get("expect", {}).get("behavior", "answer") != "answer":
                continue
            state = graph.invoke(runtime.initial_state(turn["query"], mode="workflow"))
            draft = state.get("draft") or {}
            store = EvidenceStore()
            for item in (state.get("evidence") or {}).values():
                store.add(AgentEvidence.model_validate(item))
            if not draft.get("answer") or not len(store):
                continue
            if not verify_answer(draft, store, query=turn["query"], market_precedence=False).passed:
                continue
            golds.append({"task": task["id"], "query": turn["query"], "draft": draft, "store": store})
    finally:
        runtime.close()
    return golds


def _claim_tokens(sentence: str) -> list[re.Match[str]]:
    """Number tokens in a sentence that the verifier treats as claims (not dates, ids or windows)."""
    claims = claim_numbers(sentence)
    citation_spans = [match.span() for match in _CITATION.finditer(sentence)]
    tokens = []
    for match in _NUMBER.finditer(sentence):
        if any(start <= match.start() < end for start, end in citation_spans):
            continue
        value = float((match.group(1) + (match.group(2) or "")).replace(",", ""))
        if any(abs(value - claim) < 1e-9 for claim in claims):
            tokens.append(match)
    return tokens


def _format_like(original: str, value: float) -> str:
    decimals = len(original.split(".")[1]) if "." in original else 0
    text = f"{value:.{decimals}f}"
    return text if decimals or "." not in text else text.split(".")[0]


def corruptions(gold: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    answer = str(gold["draft"]["answer"])
    store: EvidenceStore = gold["store"]
    sentences = re.split(r"(?<=[。！？!?；;])|(?<=\.)(?=\s)", answer)  # keeps every character: offsets stay exact
    variants = []
    offset = 0
    for sentence in sentences:
        cited = [match.group(1) for match in _CITATION.finditer(sentence) if match.group(1) in store]
        cited_numbers = [value for evidence_id in cited for value in store.get(evidence_id).numbers()]
        other_numbers = sorted(
            {
                value
                for item in store.items()
                if item.evidence_id not in cited
                for value in item.numbers()
                if abs(value) >= 1 and all(abs(value - own) > max(0.011, abs(own) * 0.005) for own in cited_numbers)
            }
        )
        for token in _claim_tokens(sentence):
            original = token.group(0)
            value = float(original.replace(",", ""))
            start, end = offset + token.start(), offset + token.end()
            for pct in PERTURBATIONS:
                sign = rng.choice((-1, 1))
                replacement = _format_like(original.replace(",", ""), value * (1 + sign * pct))
                if replacement != original.replace(",", ""):
                    variants.append(_variant(gold, answer, start, end, replacement, f"perturb_{int(pct * 100)}pct"))
            if cited and other_numbers:
                swapped = rng.choice(other_numbers)
                replacement = _format_like("0.00" if abs(swapped) < 100 else "0", swapped)
                variants.append(_variant(gold, answer, start, end, replacement, "swap"))
        offset += len(sentence)
    return variants


def _variant(gold: dict[str, Any], answer: str, start: int, end: int, replacement: str, kind: str) -> dict[str, Any]:
    draft = dict(gold["draft"])
    draft["answer"] = answer[:start] + replacement + answer[end:]
    draft["key_points"] = []  # key points repeat the gold numbers; test the corrupted sentence alone
    return {"kind": kind, "draft": draft, "original": answer[start:end], "replacement": replacement}


def _check(draft: dict[str, Any], gold: dict[str, Any], mode: str) -> bool:
    # Gold answers are template answers, for which the graph turns market precedence off.
    # ``claim_derived``: claim binding plus the opt-in derived-number rule (AgentConfig.verify_derived).
    binding = "claim" if mode == "claim_derived" else mode
    return verify_answer(
        draft,
        gold["store"],
        query=gold["query"],
        binding=binding,
        market_precedence=False,
        allow_derived=mode == "claim_derived",
    ).passed


def _repair(variant: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Repair a rejected variant the way the graph does and describe the result.

    ``readable``: no readability issue (``verifier.readability_issues``: dangling connectors or clauses,
    broken brackets, orphan citations, mixed terminators). ``passes``: the repaired answer verifies.
    ``retained``: share of the gold answer's untouched sentences that survive verbatim (content kept).
    ``fragment``: the repaired answer contains a sentence that is not a whole sentence of the draft or of the
    template fallback (a clause cut out of a sentence, e.g. a figure without its subject).
    """
    store = gold["store"]
    report = verify_answer(variant["draft"], store, query=gold["query"], market_precedence=False)
    zh = detect_query_language(gold["query"]) == "zh"
    # as in the graph for LLM drafts, the template answer of the same run is the fallback
    repaired, _notes = repair_answer(variant["draft"], report, store, zh=zh, fallback=gold["draft"])
    text = str(repaired.get("answer") or "")
    issues = readability_issues(text)
    corrupted = variant["draft"]["answer"]
    untouched = [s.strip() for s in whole_sentences(gold["draft"]["answer"]) if s.strip() and s in corrupted]
    whole = {s.strip() for source in (corrupted, gold["draft"]["answer"]) for s in whole_sentences(source)}
    return {
        "readable": not issues,
        "issues": [issue.split(":")[0] for issue in issues],
        "passes": verify_answer(repaired, store, query=gold["query"], market_precedence=False).passed,
        "retained": (sum(1 for s in untouched if s in text) / len(untouched)) if untouched else 1.0,
        "fallback": text.strip() == str(gold["draft"]["answer"]).strip(),
        "fragment": any(s.strip() and s.strip() not in whole for s in whole_sentences(text)),
        "answer": text,
    }


def run(tasks: list[dict[str, Any]], *, seed: int = 7) -> dict[str, Any]:
    rng = random.Random(seed)
    golds = gold_answers(tasks)
    true_accept = {mode: 0 for mode in _MODES}
    accepted: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for gold in golds:
        gold_answer = dict(gold["draft"])
        for mode in _MODES:
            true_accept[mode] += int(_check(gold_answer, gold, mode))
        for variant in corruptions(gold, rng):
            kind = variant["kind"]
            totals[kind] = totals.get(kind, 0) + 1
            outcome = {}
            for mode in _MODES:
                passed = _check(variant["draft"], gold, mode)
                accepted.setdefault(kind, {m: 0 for m in _MODES})[mode] += int(passed)
                outcome[mode] = passed
            if not outcome["claim"]:
                repairs.append({"kind": kind, "task": gold["task"], **_repair(variant, gold)})
            if outcome["legacy"] and not outcome["claim"] and len(examples) < 8:
                examples.append(
                    {
                        "task": gold["task"],
                        "kind": kind,
                        "original": variant["original"],
                        "replacement": variant["replacement"],
                        "answer": variant["draft"]["answer"][:240],
                    }
                )
    by_kind = {
        kind: {
            "variants": totals[kind],
            **{f"false_accept_{mode}": round(accepted[kind][mode] / totals[kind], 4) for mode in _MODES},
        }
        for kind in sorted(totals)
    }
    all_variants = sum(totals.values())
    issue_counts: dict[str, int] = {}
    for item in repairs:
        for issue in set(item["issues"]):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    def share(key: str) -> float | None:
        return round(sum(1 for item in repairs if item[key]) / len(repairs), 4) if repairs else None

    repair_summary = {
        "repaired_answers": len(repairs),
        "readable": share("readable"),
        "passes_verification": share("passes"),
        "template_fallback": share("fallback"),
        "with_fragment": share("fragment"),
        "retained_sentences": round(sum(item["retained"] for item in repairs) / len(repairs), 4) if repairs else None,
        "answers_with_issue": dict(sorted(issue_counts.items())),
        "unreadable_examples": [
            {"task": item["task"], "kind": item["kind"], "issues": item["issues"], "answer": item["answer"][:240]}
            for item in repairs
            if not item["readable"]
        ][:6],
    }
    return {
        "gold_answers": len(golds),
        "true_accept": {mode: round(true_accept[mode] / len(golds), 4) if golds else None for mode in _MODES},
        "variants": all_variants,
        "false_accept": {
            mode: round(sum(accepted[kind][mode] for kind in accepted) / all_variants, 4) if all_variants else None
            for mode in _MODES
        },
        "by_kind": by_kind,
        "examples_caught_only_by_claim_binding": examples,
        "repair": repair_summary,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    _git_commit()  # record the commit at start, not when the run finishes
    parser = argparse.ArgumentParser(description="Verifier false-accept stress test.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR / "verifier_stress.json"))
    args = parser.parse_args(argv)
    results = run(load_tasks(DEFAULT_TASKS), seed=args.seed)
    report = {
        "config": {
            "tasks": _display_path(DEFAULT_TASKS),
            "seed": args.seed,
            "perturbations": list(PERTURBATIONS),
            "model": None,  # deterministic verifier, no LLM
            "commit": _git_commit(),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": _command("evaluation.agent_eval.verifier_stress", argv),
        },
        **results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("gold_answers", "true_accept", "variants", "false_accept", "by_kind", "repair")
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return report


if __name__ == "__main__":
    main()
