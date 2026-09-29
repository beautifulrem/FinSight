"""Claim-check benchmark: verdict and per-number accuracy of ``check_claim`` on labelled claims.

Each row of ``claims_v1*.jsonl`` holds a claim, the expected verdict and one expected check per number
(metric, comparator, status), labelled from the offline tool outputs (see ``README.md``). The checker
runs on the real offline service (NLU + ``get_price_history`` / ``get_fundamentals``), no LLM.

Reported:

* verdict accuracy, with a 95% percentile bootstrap CI over claims;
* per-check accuracy: a check is correct when the status matches and the metric matches (when the label
  names one); checks are aligned by position and a missing check counts as wrong; the CI resamples
  claims, so the checks of one claim stay together;
* comparator accuracy on the aligned checks, the verdict confusion matrix, the check-status confusion
  matrix (with ``missing``/``extra`` rows), per-category and per-language verdict accuracy, and every
  error.

    python -m evaluation.claim_bench.run --set dev
    python -m evaluation.claim_bench.run --set holdout   # once, at the end
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from query_intelligence.agent.claim_check import check_claim

from ..agent_eval.metrics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, bootstrap_ci
from ..agent_eval.runner import ROOT, SNAPSHOT_NAME, _command, _display_path, _git_commit

BENCH_DIR = Path(__file__).resolve().parent
SETS = {"dev": BENCH_DIR / "claims_v1.jsonl", "holdout": BENCH_DIR / "claims_v1_holdout.jsonl"}
RESULTS_DIR = ROOT / "evaluation" / "results"
VERDICTS = ("supported", "contradicted", "partially_supported", "unverifiable")
STATUSES = ("supported", "contradicted", "unverifiable")
CI_METHOD = f"percentile bootstrap over claims, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}, 95%"


def load_claims(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _ratio_ci(pairs: list[tuple[int, int]]) -> list[float] | None:
    """Bootstrap CI of sum(correct) / sum(total), resampling claims (``(correct, total)`` per claim)."""
    if not pairs or not sum(total for _correct, total in pairs):
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    ratios = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = rng.choices(pairs, k=len(pairs))
        total = sum(item[1] for item in sample)
        ratios.append(sum(item[0] for item in sample) / total if total else 1.0)
    ratios.sort()
    low = ratios[max(0, math.floor(0.025 * BOOTSTRAP_RESAMPLES))]
    high = ratios[min(BOOTSTRAP_RESAMPLES - 1, math.ceil(0.975 * BOOTSTRAP_RESAMPLES) - 1)]
    return [round(low, 4), round(high, 4)]


def score_row(row: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Compare one checker report with its label."""
    expected = row.get("expected_checks") or []
    predicted = report.get("checks") or []
    checks = []
    for index, label in enumerate(expected):
        got = predicted[index] if index < len(predicted) else None
        status_ok = got is not None and got.get("status") == label["status"]
        metric_ok = got is not None and (label.get("metric") is None or got.get("metric") == label["metric"])
        comparator_ok = got is not None and got.get("comparator", "eq") == label.get("comparator", "eq")
        checks.append(
            {
                "expected": label,
                "predicted": None
                if got is None
                else {key: got.get(key) for key in ("metric", "comparator", "status", "claimed", "actual", "note")},
                "correct": status_ok and metric_ok,
                "comparator_correct": comparator_ok,
            }
        )
    return {
        "verdict_correct": report.get("verdict") == row["expected_verdict"],
        "checks": checks,
        "extra_checks": max(0, len(predicted) - len(expected)),
    }


def evaluate(rows: list[dict[str, Any]], *, service: Any, registry: Any) -> dict[str, Any]:
    verdicts: dict[str, Counter[str]] = {verdict: Counter() for verdict in VERDICTS}
    statuses: dict[str, Counter[str]] = {status: Counter() for status in (*STATUSES, "extra")}
    by_category: dict[str, list[bool]] = defaultdict(list)
    by_lang: dict[str, list[bool]] = defaultdict(list)
    verdict_values: list[float] = []
    check_pairs: list[tuple[int, int]] = []
    comparator_pairs: list[tuple[int, int]] = []
    errors = []
    for row in rows:
        report = check_claim(row["claim"], service=service, registry=registry, zh=row.get("lang", "zh") == "zh")
        body = report.model_dump()
        scored = score_row(row, body)
        verdicts[row["expected_verdict"]][body["verdict"]] += 1
        for check in scored["checks"]:
            predicted = check["predicted"]["status"] if check["predicted"] else "missing"
            statuses[check["expected"]["status"]][predicted] += 1
        for extra in body["checks"][len(row.get("expected_checks") or []) :]:
            statuses["extra"][extra["status"]] += 1
        correct_checks = sum(check["correct"] for check in scored["checks"])
        # Extra checks the label does not expect count against the claim's check accuracy.
        check_pairs.append((correct_checks, len(scored["checks"]) + scored["extra_checks"]))
        comparator_pairs.append(
            (sum(check["comparator_correct"] for check in scored["checks"]), len(scored["checks"]))
        )
        verdict_values.append(1.0 if scored["verdict_correct"] else 0.0)
        by_category[row.get("category", "other")].append(scored["verdict_correct"])
        by_lang[row.get("lang", "zh")].append(scored["verdict_correct"])
        wrong_checks = [check for check in scored["checks"] if not check["correct"]]
        if not scored["verdict_correct"] or wrong_checks or scored["extra_checks"]:
            errors.append(
                {
                    "id": row["id"],
                    "category": row.get("category"),
                    "claim": row["claim"],
                    "expected_verdict": row["expected_verdict"],
                    "verdict": body["verdict"],
                    "wrong_checks": wrong_checks,
                    "extra_checks": body["checks"][len(row.get("expected_checks") or []) :],
                }
            )
    total_checks = sum(total for _correct, total in check_pairs)
    total_comparators = sum(total for _correct, total in comparator_pairs)
    return {
        "claims": len(rows),
        "checks": total_checks,
        "verdict_accuracy": round(sum(verdict_values) / len(rows), 4) if rows else None,
        "verdict_accuracy_ci": bootstrap_ci(verdict_values),
        "check_accuracy": round(sum(c for c, _t in check_pairs) / total_checks, 4) if total_checks else None,
        "check_accuracy_ci": _ratio_ci(check_pairs),
        "comparator_accuracy": round(sum(c for c, _t in comparator_pairs) / total_comparators, 4)
        if total_comparators
        else None,
        "ci_method": CI_METHOD,
        "verdict_confusion": {
            expected: {predicted: verdicts[expected][predicted] for predicted in VERDICTS} for expected in VERDICTS
        },
        "check_status_confusion": {
            expected: {predicted: statuses[expected][predicted] for predicted in (*STATUSES, "missing")}
            for expected in (*STATUSES, "extra")
        },
        "by_category": {
            name: {"claims": len(values), "verdict_accuracy": round(sum(values) / len(values), 4)}
            for name, values in sorted(by_category.items())
        },
        "by_language": {
            name: {"claims": len(values), "verdict_accuracy": round(sum(values) / len(values), 4)}
            for name, values in sorted(by_lang.items())
        },
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    _git_commit()  # record the commit at start
    parser = argparse.ArgumentParser(description="Claim-check benchmark (offline data, no LLM).")
    parser.add_argument("--set", choices=sorted(SETS), default="dev")
    parser.add_argument("--claims", default=None, help="Claims file (overrides --set).")
    parser.add_argument("--out", default=None, help="Default: evaluation/results/claim_bench-<set>.json")
    args = parser.parse_args(argv)
    path = Path(args.claims) if args.claims else SETS[args.set]
    rows = load_claims(path)

    from query_intelligence.agent.tools import build_registry_for_service

    from ..agent_eval.runner import build_offline_service

    service = build_offline_service()
    report = {
        "config": {
            "claims": _display_path(path),
            "claims_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "set": None if args.claims else args.set,
            "data": SNAPSHOT_NAME,
            "model": None,  # deterministic claim checker, no LLM
            "commit": _git_commit(),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": _command("evaluation.claim_bench.run", argv),
        },
        **evaluate(rows, service=service, registry=build_registry_for_service(service)),
    }
    out = Path(args.out) if args.out else RESULTS_DIR / f"claim_bench-{args.set}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        f"verdict accuracy {report['verdict_accuracy']} {report['verdict_accuracy_ci']} over {report['claims']} "
        f"claims; check accuracy {report['check_accuracy']} {report['check_accuracy_ci']} over "
        f"{report['checks']} checks; comparator accuracy {report['comparator_accuracy']}"
    )
    print("expected \\ predicted " + " ".join(f"{verdict[:12]:>12s}" for verdict in VERDICTS))
    for verdict in VERDICTS:
        row = report["verdict_confusion"][verdict]
        print(f"{verdict:>20s} " + " ".join(f"{row[other]:12d}" for other in VERDICTS))
    for error in report["errors"]:
        print(f"  {error['id']} {error['expected_verdict']} -> {error['verdict']}  {error['claim']}")
    return report


if __name__ == "__main__":
    main()
