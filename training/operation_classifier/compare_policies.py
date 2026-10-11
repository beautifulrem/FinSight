"""(round 17, review-13 M18) grammar-first or classifier-first: which way of combining the comparison grammar and the
operation classifier reads the operation better.

    python -m training.operation_classifier.compare_policies

The rule that decides between them was committed before this script was first run
(``evaluation/operation_wordings_independent_v1.README.md``, "Round 17"):

* **grammar-first** (shipped since round 15): the grammar's operation; the classifier only breaks a tie between two
  matched productions or reads an operation when the grammar reads none (``operation_classifier.tie_break`` /
  ``fallback_operation``);
* **classifier-first**: the classifier's top label (``none`` included) whenever its probability is at least
  ``CLASSIFIER_FIRST_MIN`` (0.6) and the label passes the fallback's structural checks; otherwise grammar-first.

Sets: the independent operation wording set (234 questions, after exposure), and a fresh held-out split of the
training templates (seed ``FRESH_SEED``; a model trained without those templates, not shipped). On the dev and ext
task questions only the number of questions whose operation changes is counted (their labels are the grammar's).

Classifier-first replaces grammar-first only if it is at least as accurate on both sets, at least one percentage point
more accurate on one of them, and changes at most 1% of the dev and ext questions; a tie keeps grammar-first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation" / "results" / "operation_classifier-policies-round17.json"
CLASSIFIER_FIRST_MIN = 0.6
FRESH_SEED = 20261017
HELD_OUT_EVERY = 5
MIN_GAIN = 0.01
MAX_DEV_CHANGES = 0.01


def _use_model(model: Any) -> None:
    """Make the run-time path (``frame.frame_operation`` and the fallback) read ``model``."""
    from query_intelligence.agent import operation_classifier as oc

    os.environ["QI_OPERATION_CLASSIFIER"] = "1"
    oc.reset_cache()
    oc._cached[str(oc._artifact_path())] = model


def grammar_first(text: str, operands: int) -> str:
    """The shipped reading (``evaluate_independent.runtime_operation``)."""
    from training.operation_classifier.evaluate_independent import runtime_operation

    return runtime_operation(text, operands)


def classifier_first(text: str, operands: int, model: Any) -> tuple[str, bool]:
    """``(operation, decided by the classifier)``: the classifier's top label when it is confident and structurally
    possible, else the grammar-first reading."""
    from query_intelligence.agent import operation_classifier as oc
    from query_intelligence.agent.memory import strip_filler

    stripped = strip_filler(text).strip()
    shipped = grammar_first(text, operands)
    label, probability = model.read(stripped)
    if probability < CLASSIFIER_FIRST_MIN:
        return shipped, False
    if label != "none":
        several = operands >= 2 or bool(oc._SEVERAL.search(stripped))
        if not (several and oc._ASKS_QUANTITY.search(stripped) and oc.compatible(label, stripped)):
            return shipped, False
    return label, label != shipped


def _score(rows: list[tuple[str, str, int]], model: Any) -> dict[str, Any]:
    from training.operation_classifier.train import wilson

    tallies: Counter[str] = Counter()
    changes: list[dict[str, str]] = []
    for text, label, operands in rows:
        shipped = grammar_first(text, operands)
        chosen, decided = classifier_first(text, operands, model)
        tallies["grammar_first"] += shipped == label
        tallies["classifier_first"] += chosen == label
        if chosen != shipped:
            tallies["changed"] += 1
            tallies["changed_to_right"] += chosen == label
            tallies["changed_to_wrong"] += shipped == label
            if len(changes) < 60:
                changes.append({"expected": label, "grammar_first": shipped, "classifier_first": chosen})
        tallies["classifier_decided"] += decided
    total = len(rows)
    return {
        "questions": total,
        "grammar_first": wilson(tallies["grammar_first"], total),
        "classifier_first": wilson(tallies["classifier_first"], total),
        "changed": tallies["changed"],
        "changed_to_right": tallies["changed_to_right"],
        "changed_to_wrong": tallies["changed_to_wrong"],
        "change_sample": changes,
    }


def independent_rows(service: Any) -> list[tuple[str, str, int]]:
    from training.operation_classifier.evaluate_independent import WORDINGS, _entities

    rows = [json.loads(line) for line in WORDINGS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(row["question"], row["operation"], len(_entities(service, row["question"]))) for row in rows]


def fresh_split() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """``(train, held_out)`` rows ``(text, label, template id)``: within each operation and language the templates are
    shuffled with ``FRESH_SEED`` and every fifth one of the shuffled order is held out."""
    from training.operation_classifier.train import PER_TEMPLATE, TEMPLATES, _fill

    rng = random.Random(FRESH_SEED)
    train: list[tuple[str, str, str]] = []
    held: list[tuple[str, str, str]] = []
    for label, languages in TEMPLATES.items():
        for language, templates in languages.items():
            order = list(range(len(templates)))
            random.Random(f"{FRESH_SEED}-{label}-{language}").shuffle(order)
            held_out = {
                index for position, index in enumerate(order) if position % HELD_OUT_EVERY == HELD_OUT_EVERY - 1
            }
            for index, template in enumerate(templates):
                target = held if index in held_out else train
                for _ in range(PER_TEMPLATE):
                    target.append((_fill(template, rng, language == "zh"), label, f"{label}/{language}/{index}"))
    return train, held


def fresh_model(train: list[tuple[str, str, str]]) -> Any:
    """The shipped pipeline (with lexicon dropout) trained on ``train`` and the dev wordings; not written anywhere."""
    from query_intelligence.agent.operation_classifier import DROPOUT_MARK, OperationClassifier, normalise
    from training.operation_classifier.train import VERSION, build_pipeline, dev_wordings

    dev = dev_wordings()
    texts = [normalise(text) for text, _label, _template in train] + [normalise(text) for text, _label in dev]
    labels = [label for _text, label, _template in train] + [label for _text, label in dev]
    texts += [DROPOUT_MARK + text for text in texts]
    labels += labels
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)
    return OperationClassifier(pipeline=pipeline, version=f"{VERSION}-fresh-{FRESH_SEED}")


def _named(text: str) -> int:
    from training.operation_classifier.train import TARGETS

    return sum(1 for zh, en in TARGETS if zh in text or en in text)


def dev_changes(service: Any, model: Any) -> dict[str, Any]:
    from training.operation_classifier.evaluate_independent import DEV_SETS, _entities

    report: dict[str, Any] = {}
    total = changed = 0
    for name in DEV_SETS:
        path = ROOT / "evaluation" / "agent_eval" / "tasks" / name
        queries = [
            str(turn.get("query") or "")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for turn in json.loads(line).get("turns") or []
        ]
        moved = 0
        for query in queries:
            operands = len(_entities(service, query))
            moved += classifier_first(query, operands, model)[0] != grammar_first(query, operands)
        report[name] = {"questions": len(queries), "operation_changed": moved}
        total += len(queries)
        changed += moved
    report["total"] = {"questions": total, "operation_changed": changed, "rate": round(changed / total, 4)}
    return report


def decide(independent: dict[str, Any], fresh: dict[str, Any], dev: dict[str, Any]) -> dict[str, Any]:
    """The committed rule (README, "Round 17")."""
    gains = [
        float(result["classifier_first"]["rate"]) - float(result["grammar_first"]["rate"])
        for result in (independent, fresh)
    ]
    not_worse = all(gain >= 0 for gain in gains)
    better = any(gain >= MIN_GAIN - 1e-9 for gain in gains)
    quiet = dev["total"]["rate"] <= MAX_DEV_CHANGES
    policy = "classifier-first" if not_worse and better and quiet else "grammar-first"
    return {
        "policy": policy,
        "gain_independent": round(gains[0], 4),
        "gain_fresh_split": round(gains[1], 4),
        "not_worse_on_both": not_worse,
        "at_least_one_point_better_on_one": better,
        "dev_changes_within_1pct": quiet,
    }


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args(argv)
    os.environ.setdefault("QI_USE_LIVE_MARKET", "0")
    from evaluation.agent_eval.runner import build_offline_service
    from query_intelligence.agent.operation_classifier import _load, _artifact_path

    service = build_offline_service(snapshot_ext=True, snapshot_fix=True)
    shipped = _load(_artifact_path())
    if shipped is None:
        raise SystemExit("the operation classifier artifact is missing")
    _use_model(shipped)
    independent = _score(independent_rows(service), shipped)
    dev = dev_changes(service, shipped)
    train, held = fresh_split()
    model = fresh_model(train)
    _use_model(model)
    fresh = _score([(text, label, _named(text)) for text, label, _template in held], model)
    fresh["templates_held_out"] = len({row[2] for row in held})
    fresh["templates_trained"] = len({row[2] for row in train})
    _use_model(shipped)
    wordings = ROOT / "evaluation" / "operation_wordings_independent_v1.jsonl"
    report = {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "python -m training.operation_classifier.compare_policies",
        "rule": "evaluation/operation_wordings_independent_v1.README.md, 'Round 17' (committed in 6db4943 before the "
        "first run)",
        "classifier_first_min": CLASSIFIER_FIRST_MIN,
        "fresh_seed": FRESH_SEED,
        "classifier_version": shipped.version,
        "wordings_sha256": hashlib.sha256(wordings.read_bytes()).hexdigest(),
        "independent_v1_after_exposure": independent,
        "fresh_held_out_split": fresh,
        "dev_and_ext_changes": dev,
    }
    report["decision"] = decide(independent, fresh, dev)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "independent": [independent["grammar_first"]["rate"], independent["classifier_first"]["rate"]],
                "fresh": [fresh["grammar_first"]["rate"], fresh["classifier_first"]["rate"]],
                "dev_changes": dev["total"],
                "decision": report["decision"],
            }
        )
    )
    return report


if __name__ == "__main__":
    main()
