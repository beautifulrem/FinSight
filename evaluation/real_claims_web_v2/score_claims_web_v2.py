"""Run the project's claim checker once over ``claims_web_v2.jsonl`` (offline, no LLM) and score its verdicts.

The checker is a black box, called the way ``evaluation/real_claims_web/score_claims_web_v1.py`` calls it (and the way
``POST /agent/claim-check`` does): ``check_claim`` with the offline service and its tool registry. The service uses the
shipped default data: the v1 snapshot, the 2026-09-30 extension and the corrections layer
(``QI_OFFLINE_SNAPSHOT_EXT`` and ``QI_OFFLINE_SNAPSHOT_FIX`` on). The claim text is the published sentence with the
square brackets of an inserted subject removed (the name stays). No date is passed: a date the sentence leaves to its
context is not given to the checker, as in the first set.

A claim is correct when the checker's verdict equals ``expected_verdict`` (the definition-aware label). The accuracy
against ``expected_verdict_definition_blind`` is reported alongside. 95% percentile bootstrap CI over claims (2000
resamples, seed 20261010). Writes OUT (default ``evaluation/results/claims_web_v2-first-run.json``) and refuses to write
into a committed results folder from a dirty tree::

    python -m evaluation.real_claims_web_v2.score_claims_web_v2 [--out OUT.json]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLAIMS = HERE / "claims_web_v2.jsonl"
DEFAULT_OUT = ROOT / "evaluation" / "results" / "claims_web_v2-first-run.json"
CHECK_FIELDS = ("target", "metric", "comparator", "claimed", "actual", "status", "reason", "note")
COMMAND = "python -m evaluation.real_claims_web_v2.score_claims_web_v2"
SEED = 20261010


def checker_text(published_text: str) -> str:
    return re.sub(r"[\[\]]", "", published_text)


def bootstrap_ci(hits: list[float], *, seed: int = SEED, resamples: int = 2000) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(hits, k=len(hits))) / len(hits) for _ in range(resamples))
    return [round(means[int(resamples * 0.025) - 1], 4), round(means[int(resamples * 0.975) - 1], 4)]


def rates(results: list[dict[str, Any]], key: str, ok: str = "ok") -> dict[str, dict[str, Any]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in results:
        groups[str(row[key])].append(row[ok])
    return {
        name: {"n": len(oks), "correct": sum(oks), "accuracy": round(sum(oks) / len(oks), 3)}
        for name, oks in sorted(groups.items())
    }


def git_commit() -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    return commit.stdout.strip() + ("-dirty" if dirty else "")


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT))
    from evaluation.agent_eval.dirty_guard import add_dirty_arguments, check_dirty_write

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    add_dirty_arguments(parser)
    args = parser.parse_args(argv)
    commit = git_commit()
    dirty_note = check_dirty_write(args.out, commit, allow_dirty=args.allow_dirty, notes=args.note)

    from evaluation.agent_eval.runner import build_offline_service
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools import build_registry_for_service

    rows = [json.loads(line) for line in CLAIMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    service = build_offline_service(snapshot_ext=True, snapshot_fix=True)
    registry = build_registry_for_service(service)
    results = []
    for row in rows:
        text = checker_text(row["published_text"])
        report = check_claim(text, service=service, registry=registry, zh=row["lang"].startswith("zh")).model_dump()
        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "source_type": row["source_type"],
                "social_platform": row.get("social_platform") or "-",
                "lang": row["lang"],
                "claim": text,
                "expected": row["expected_verdict"],
                "expected_definition_blind": row["expected_verdict_definition_blind"],
                "got": report["verdict"],
                "ok": report["verdict"] == row["expected_verdict"],
                "ok_definition_blind": report["verdict"] == row["expected_verdict_definition_blind"],
                "checks": [{k: c.get(k) for k in CHECK_FIELDS} for c in report.get("checks") or []],
            }
        )
    hits = [1.0 if r["ok"] else 0.0 for r in results]
    blind_hits = [1.0 if r["ok_definition_blind"] else 0.0 for r in results]
    by_category = rates(results, "category")
    notes = [
        "engineer-collected real published claims, blind first run: collected, labelled and committed before the "
        "checker's code or tests were read; the collector had read the first set's README (its error classes) and "
        "the offline data, the search agents had not",
    ]
    if dirty_note:
        notes.append(dirty_note)
    payload = {
        "kind": "claim_slice",
        "config": {
            "commit": commit,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": COMMAND,
            "snapshot": "offline v1 + 2026-09-30 extension + corrections layer (QI_OFFLINE_SNAPSHOT_EXT and "
            "QI_OFFLINE_SNAPSHOT_FIX on, the shipped default), no LLM",
            "labels": "evaluation/real_claims_web_v2/claims_web_v2.jsonl (label_claims_web_v2, definition-aware; "
            "expected_verdict_definition_blind reported alongside)",
            "note": notes[0],
            "notes": notes,
        },
        "summary": {
            "claims": len(results),
            "verdict_accuracy": round(sum(hits) / len(hits), 4),
            "ci95": bootstrap_ci(hits),
            "by_category": {name: value["accuracy"] for name, value in by_category.items()},
            "by_category_counts": by_category,
            "by_source_type": rates(results, "source_type"),
            "by_social_platform": rates([r for r in results if r["source_type"] == "social"], "social_platform"),
            "by_lang": rates(results, "lang"),
            "by_expected_verdict": rates(results, "expected"),
            "against_definition_blind_labels": {
                "verdict_accuracy": round(sum(blind_hits) / len(blind_hits), 4),
                "ci95": bootstrap_ci(blind_hits),
            },
            "got_counts": {v: sum(1 for r in results if r["got"] == v) for v in sorted({r["got"] for r in results})},
        },
        "errors": [
            {
                k: r[k]
                for k in (
                    "id",
                    "category",
                    "source_type",
                    "claim",
                    "expected",
                    "expected_definition_blind",
                    "got",
                    "checks",
                )
            }
            for r in results
            if not r["ok"]
        ],
        "results": [{k: r[k] for k in ("id", "expected", "expected_definition_blind", "got", "ok")} for r in results],
    }
    for r in results:
        if not r["ok"]:
            print("XX", r["id"], r["expected"], "->", r["got"], "|", r["claim"])
    print(json.dumps(payload["summary"], ensure_ascii=False))
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
