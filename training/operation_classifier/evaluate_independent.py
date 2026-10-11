"""(round 16, review-12 L17) The comparison operation on wordings its template author did not write.

    python -m training.operation_classifier.evaluate_independent --label first-run
    python -m training.operation_classifier.evaluate_independent --label after-exposure-round16

``evaluation/operation_wordings_independent_v1.jsonl`` was written by a separate agent that saw only the list of
operations (one line each), the company names and the metric names, never ``frame.py``, the classifier's templates,
the tests or the task sets (``evaluation/operation_wordings_independent_v1.README.md``). Every question has the
operation its writer meant, the companies in order of mention and the metric.

Three readings of the operation are scored against those labels:

* **grammar alone** — the run-time path with the classifier switched off (``QI_OPERATION_CLASSIFIER=0``):
  ``frame.frame_operation`` (the grammar's lexicon; advice / forecast / holding questions are not comparisons);
* **classifier alone** — the classifier's most probable label (``none`` included), no grammar;
* **combined** — the run-time path with the classifier on: the grammar, the tie-breaker between matched productions
  and the fallback when the grammar reads nothing (``frame.frame_operation`` then, as in
  ``frame.resolve_frame_question``, ``operation_classifier.fallback_operation`` with the number of targets the NLU
  resolved).

It also runs the whole frame step (``frame.resolve_frame_question`` with the offline NLU's entities, no session) and
reports how often the classifier **fires** (the request records a classifier reading) and how often that **changes the
frame request** (operation differs from the classifier-off run), on this set and on the dev + ext task questions
(first turns and every turn without session context; their labels are the grammar's, so only counts are reported).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORDINGS = ROOT / "evaluation" / "operation_wordings_independent_v1.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "results"
DEV_SETS = ("agent_eval_v1.jsonl", "agent_eval_ext_v1.jsonl")


def _wilson(successes: int, total: int) -> dict[str, Any]:
    from training.operation_classifier.train import wilson

    return wilson(successes, total)


def _classifier(on: bool) -> None:
    from query_intelligence.agent import operation_classifier

    os.environ["QI_OPERATION_CLASSIFIER"] = "1" if on else "0"
    operation_classifier.reset_cache()


def runtime_operation(text: str, operands: int) -> str:
    """The operation the run-time path reads (see the module docstring), ``none`` when it reads none."""
    from query_intelligence.agent.coverage import holding_value_request
    from query_intelligence.agent.frame import _NET_MARGIN_SHARE, _NOT_A_GAP, comparison_clause, frame_operation
    from query_intelligence.agent.memory import strip_filler
    from query_intelligence.agent.operation_classifier import fallback_operation

    text = strip_filler(text).strip()
    operation = frame_operation(text)
    if (
        operation is None
        and not (_NOT_A_GAP.search(text) or _NET_MARGIN_SHARE.search(text))
        and holding_value_request(text) is None
        and comparison_clause(text) is None
    ):
        found = fallback_operation(text, operands=operands)
        operation = found.operation if found is not None else None
    return operation or "none"


def _entities(service: Any, text: str) -> list[dict[str, Any]]:
    from query_intelligence.agent.memory import listed_entities

    nlu = service.analyze_query(text)
    nlu = nlu if isinstance(nlu, dict) else nlu.model_dump()
    return [e for e in listed_entities(nlu) if "fuzzy" not in str(e.get("match_type") or "")]


def _frame_request(text: str, entities: list[dict[str, Any]]) -> dict[str, Any] | None:
    from query_intelligence.agent.frame import resolve_frame_question

    found = resolve_frame_question(text, [], entities, [])
    return found[2] if found is not None else None


def score_wordings(service: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from query_intelligence.agent.operation_classifier import load_classifier

    _classifier(True)
    model = load_classifier()
    if model is None:
        raise SystemExit("the operation classifier artifact is missing")
    tallies: Counter[str] = Counter()
    per_class: dict[str, Counter[str]] = {}
    per_language: dict[str, Counter[str]] = {}
    misses: list[dict[str, Any]] = []
    for row in rows:
        text, label = row["question"], row["operation"]
        entities = _entities(service, text)
        _classifier(False)
        grammar = runtime_operation(text, len(entities))
        request_off = _frame_request(text, entities)
        _classifier(True)
        combined = runtime_operation(text, len(entities))
        request_on = _frame_request(text, entities)
        alone = model.read(text)[0]
        frame_op = (request_on or {}).get("operation") or "none"
        frame_off = (request_off or {}).get("operation") or "none"
        names = [str(item.get("name")) for item in (request_on or {}).get("operands") or []]
        operands_ok = label == "none" or sorted(names) == sorted(row.get("companies") or [])
        hits = {
            "grammar": grammar == label,
            "classifier": alone == label,
            "combined": combined == label,
            "frame_request": frame_op == label,
            "frame_request_classifier_off": frame_off == label,
            "frame_request_and_operands": frame_op == label and operands_ok,
        }
        for key, ok in hits.items():
            tallies[key] += ok
        fired = bool((request_on or {}).get("classifier")) or combined != grammar
        tallies["fires"] += fired
        tallies["changes_operation"] += combined != grammar
        tallies["changes_frame_request"] += frame_op != frame_off
        tallies["changes_to_right"] += combined != grammar and combined == label
        tallies["changes_to_wrong"] += combined != grammar and grammar == label
        tallies["frame_changes_to_right"] += frame_op != frame_off and frame_op == label
        tallies["frame_changes_to_wrong"] += frame_op != frame_off and frame_off == label
        tallies["grammar_none_on_operation"] += grammar == "none" and label != "none"
        tallies["rescued"] += grammar == "none" and label != "none" and combined == label
        for bucket, key in ((per_class, label), (per_language, row["lang"])):
            counter = bucket.setdefault(key, Counter())
            counter["n"] += 1
            for name, ok in hits.items():
                counter[name] += ok
        if not hits["combined"] and len(misses) < 400:
            misses.append(
                {
                    "id": row["id"],
                    "expected": label,
                    "grammar": grammar,
                    "classifier": alone,
                    "combined": combined,
                    "frame_request": frame_op,
                }
            )
    total = len(rows)
    keys = (
        "grammar",
        "classifier",
        "combined",
        "frame_request_classifier_off",
        "frame_request",
        "frame_request_and_operands",
    )
    return {
        "questions": total,
        "accuracy": {key: _wilson(tallies[key], total) for key in keys},
        "classifier_use": {
            "fires": tallies["fires"],
            "changes_operation": tallies["changes_operation"],
            "changes_to_right": tallies["changes_to_right"],
            "changes_to_wrong": tallies["changes_to_wrong"],
            "changes_frame_request": tallies["changes_frame_request"],
            "frame_changes_to_right": tallies["frame_changes_to_right"],
            "frame_changes_to_wrong": tallies["frame_changes_to_wrong"],
            "grammar_none_on_an_operation": tallies["grammar_none_on_operation"],
            "rescued_by_fallback": tallies["rescued"],
        },
        "by_operation": {
            label: {key: f"{counter[key]}/{counter['n']}" for key in keys}
            for label, counter in sorted(per_class.items())
        },
        "by_language": {
            lang: {key: f"{counter[key]}/{counter['n']}" for key in keys}
            for lang, counter in sorted(per_language.items())
        },
        "combined_misses": misses,
    }


def dev_use(service: Any) -> dict[str, Any]:
    """How often the classifier fires and changes the frame request on the dev and ext task questions (each turn read
    without session context)."""
    report: dict[str, Any] = {}
    for name in DEV_SETS:
        path = ROOT / "evaluation" / "agent_eval" / "tasks" / name
        queries = [
            str(turn.get("query") or "")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for turn in json.loads(line).get("turns") or []
        ]
        fired = changed = 0
        by_use: Counter[str] = Counter()
        for query in queries:
            entities = _entities(service, query)
            _classifier(False)
            off = _frame_request(query, entities)
            _classifier(True)
            on = _frame_request(query, entities)
            reading = (on or {}).get("classifier")
            if reading:
                fired += 1
                by_use[str(reading.get("use"))] += 1
            changed += ((on or {}).get("operation") or "none") != ((off or {}).get("operation") or "none")
        report[name] = {
            "questions": len(queries),
            "classifier_fires": fired,
            "by_use": dict(by_use),
            "changes_frame_request": changed,
        }
    return report


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", required=True, help="first-run, after-exposure-round16, …")
    parser.add_argument("--commit", default=None, help="record this commit (when run from an exported tree)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    os.environ.setdefault("QI_USE_LIVE_MARKET", "0")
    from evaluation.agent_eval.runner import build_offline_service
    from query_intelligence.agent.operation_classifier import load_classifier

    service = build_offline_service(snapshot_ext=True, snapshot_fix=True)
    rows = [json.loads(line) for line in WORDINGS.read_text(encoding="utf-8").splitlines() if line.strip()]
    report: dict[str, Any] = {
        "label": args.label,
        "commit": args.commit or _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")) if not args.commit else False,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": f"python -m training.operation_classifier.evaluate_independent --label {args.label}",
        "wordings": str(WORDINGS.relative_to(ROOT)),
        "wordings_sha256": __import__("hashlib").sha256(WORDINGS.read_bytes()).hexdigest(),
        "offline_nlu": "ext + corrections snapshot (shipped default), no session",
    }
    _classifier(True)
    model = load_classifier()
    report["classifier_version"] = model.version if model else None
    report["independent"] = score_wordings(service, rows)
    report["dev_and_ext_use"] = dev_use(service)
    out = Path(args.out) if args.out else RESULTS_DIR / f"operation_classifier-independent-v1-{args.label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    summary = {key: value["rate"] for key, value in report["independent"]["accuracy"].items()}
    print(json.dumps({"label": args.label, "accuracy": summary, "use": report["independent"]["classifier_use"]}))
    print(json.dumps(report["dev_and_ext_use"]))
    return report


if __name__ == "__main__":
    main()
