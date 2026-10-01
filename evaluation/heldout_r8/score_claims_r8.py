"""Score claims_r8_heldout.jsonl with the project's claim checker (offline service, no LLM).

Verdict accuracy only: a claim is correct when the checker's verdict equals ``expected_verdict`` or is listed in
``also_acceptable`` (sums, which the checker may reasonably leave unverifiable). 95% percentile bootstrap CI over
claims (2000 resamples, seed 20261001). Per-category accuracy and every error are printed and written to OUT.

    FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python score_claims_r8.py OUT.json
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_repo() -> Path:
    env = os.environ.get("FINSIGHT_REPO")
    if env and (Path(env) / "query_intelligence").is_dir():
        return Path(env).resolve()
    for start in (HERE, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "query_intelligence").is_dir():
                return candidate
    raise SystemExit("cannot find the FinSight repo: set FINSIGHT_REPO or run from inside it")


REPO = find_repo()
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from evaluation.agent_eval.runner import build_offline_service  # noqa: E402
from query_intelligence.agent.claim_check import check_claim  # noqa: E402
from query_intelligence.agent.tools import build_registry_for_service  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "claims_r8-result.json"
    rows = [json.loads(line) for line in (HERE / "claims_r8_heldout.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    service = build_offline_service()
    registry = build_registry_for_service(service)
    results = []
    for row in rows:
        report = check_claim(row["claim"], service=service, registry=registry, zh=row["lang"] == "zh").model_dump()
        ok = report["verdict"] == row["expected_verdict"] or report["verdict"] in row.get("also_acceptable", [])
        results.append({"id": row["id"], "category": row["category"], "claim": row["claim"],
                        "expected": row["expected_verdict"], "got": report["verdict"], "ok": ok,
                        "checks": [{k: c.get(k) for k in ("target", "metric", "comparator", "claimed", "status", "reason")}
                                   for c in report.get("checks") or []]})
    hits = [1.0 if r["ok"] else 0.0 for r in results]
    rng = random.Random(20261001)
    means = sorted(sum(rng.choices(hits, k=len(hits))) / len(hits) for _ in range(2000))
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["ok"])
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {"claims": len(results), "verdict_accuracy": round(sum(hits) / len(hits), 4),
               "ci95": [round(means[49], 4), round(means[1949], 4)],
               "by_category": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())},
               "commit": commit, "run_at": datetime.now(UTC).isoformat(timespec="seconds")}
    for r in results:
        if not r["ok"]:
            print("XX", r["id"], r["expected"], "->", r["got"], "|", r["claim"])
    print(json.dumps(summary, ensure_ascii=False))
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
