"""Run the project's claim checker once over ``claims_web_v1.jsonl`` (offline, no LLM) and score its verdicts.

The checker is used as a black box: ``check_claim`` with the offline service and its tool registry, exactly as
``POST /agent/claim-check`` calls it, with the offline snapshot extension on (the product default,
``QI_OFFLINE_SNAPSHOT_EXT``), because the 2026-09-30 extension is where most of these companies' data lives.

A claim is correct when the checker's verdict equals ``expected_verdict`` or is listed in ``also_acceptable``.
95% percentile bootstrap CI over claims (2000 resamples, seed 20261001). Accuracy per category and per source
type, and every error with the checker's checks, are written to OUT (default:
``evaluation/results/claims_web_v1-first-run.json``)::

    python -m evaluation.real_claims_web.score_claims_web_v1 [OUT.json]
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLAIMS = HERE / "claims_web_v1.jsonl"
DEFAULT_OUT = ROOT / "evaluation" / "results" / "claims_web_v1-first-run.json"
CHECK_FIELDS = ("target", "metric", "comparator", "claimed", "actual", "status", "reason", "note")
COMMAND = "python -m evaluation.real_claims_web.score_claims_web_v1"


def bootstrap_ci(hits: list[float], *, seed: int = 20261001, resamples: int = 2000) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(hits, k=len(hits))) / len(hits) for _ in range(resamples))
    return [round(means[int(resamples * 0.025) - 1], 4), round(means[int(resamples * 0.975) - 1], 4)]


def rates(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in results:
        groups[row[key]].append(row["ok"])
    return {name: {"n": len(oks), "correct": sum(oks), "accuracy": round(sum(oks) / len(oks), 3)}
            for name, oks in sorted(groups.items())}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0]) if args else DEFAULT_OUT
    sys.path.insert(0, str(ROOT))
    from evaluation.agent_eval.runner import build_offline_service
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools import build_registry_for_service

    rows = [json.loads(line) for line in CLAIMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    service = build_offline_service(snapshot_ext=True)
    registry = build_registry_for_service(service)
    results = []
    for row in rows:
        report = check_claim(row["claim"], service=service, registry=registry, zh=row["lang"] == "zh").model_dump()
        ok = report["verdict"] == row["expected_verdict"] or report["verdict"] in row.get("also_acceptable", [])
        results.append({
            "id": row["id"], "category": row["category"], "source_type": row["source_type"], "claim": row["claim"],
            "expected": row["expected_verdict"], "got": report["verdict"], "ok": ok,
            "checks": [{k: c.get(k) for k in CHECK_FIELDS} for c in report.get("checks") or []],
        })
    hits = [1.0 if r["ok"] else 0.0 for r in results]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           cwd=ROOT).stdout.strip()
    by_category = rates(results, "category")
    payload = {
        "kind": "claim_slice",
        "config": {
            "commit": commit.stdout.strip() + ("-dirty" if dirty else ""),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": COMMAND,
            "snapshot": "offline v1 + 2026-09-30 extension (QI_OFFLINE_SNAPSHOT_EXT on), no LLM",
            "note": "engineer-collected real published claims, first run; not blind (the collector had read the "
            "checker); labels committed before this run",
        },
        "summary": {
            "claims": len(results),
            "verdict_accuracy": round(sum(hits) / len(hits), 4),
            "ci95": bootstrap_ci(hits),
            "by_category": {name: value["accuracy"] for name, value in by_category.items()},
            "by_category_counts": by_category,
            "by_source_type": rates(results, "source_type"),
            "by_expected_verdict": rates(results, "expected"),
        },
        "errors": [{k: r[k] for k in ("id", "category", "source_type", "claim", "expected", "got", "checks")}
                   for r in results if not r["ok"]],
        "results": [{k: r[k] for k in ("id", "expected", "got", "ok")} for r in results],
    }
    for r in results:
        if not r["ok"]:
            print("XX", r["id"], r["expected"], "->", r["got"], "|", r["claim"])
    print(json.dumps(payload["summary"], ensure_ascii=False))
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
