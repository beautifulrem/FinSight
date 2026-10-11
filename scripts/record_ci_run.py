"""Record a finished CI run of ``master`` as ``docs/results/ci/ci-run-<sha>.json`` (round-13 review M14).

The README's CI paragraph is generated from the newest record (``python -m evaluation.agent_eval.readme``), so the
link, the commit, the job count and the job closest to its timeout come from the GitHub API, not from a hand-edited
line. Run it after a push once the run has finished, then re-render the READMEs and commit both:

    python scripts/record_ci_run.py                    # the latest completed run of ci.yml on master
    python scripts/record_ci_run.py --run 38113129617  # a given run
    python -m evaluation.agent_eval.readme             # re-render README.md / README_CN.md

Needs the GitHub CLI (``gh``) with read access to the repository. A run cannot be recorded in the commit it tests, so
the record always names an earlier commit; the CI badge in the README shows the live state of ``master``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ci_budget import load_job_timeouts  # noqa: E402

REPO = "beautifulrem/FinSight"
OUT_DIR = ROOT / "docs" / "results" / "ci"


def _gh(*args: str) -> object:
    completed = subprocess.run(["gh", *args], capture_output=True, text=True, check=True, timeout=60)
    return json.loads(completed.stdout)


def _minutes(started: str, completed: str) -> float:
    start = datetime.fromisoformat(started.replace("Z", "+00:00"))
    end = datetime.fromisoformat(completed.replace("Z", "+00:00"))
    return round((end - start).total_seconds() / 60, 1)


def latest_run(branch: str = "master") -> int:
    runs = _gh(
        "run",
        "list",
        "-R",
        REPO,
        "--workflow",
        "ci.yml",
        "--branch",
        branch,
        "--status",
        "completed",
        "-L",
        "1",
        "--json",
        "databaseId",
    )
    if not runs:
        raise SystemExit(f"no completed ci.yml run on {branch}")
    return int(runs[0]["databaseId"])


def record(run_id: int) -> dict:
    run = _gh(
        "run",
        "view",
        str(run_id),
        "-R",
        REPO,
        "--json",
        "databaseId,url,headSha,headBranch,event,status,conclusion,createdAt,updatedAt,jobs",
    )
    if run["status"] != "completed":
        raise SystemExit(f"run {run_id} is {run['status']}; record it once it has finished")
    timeouts = load_job_timeouts()
    jobs = [
        {
            "name": job["name"],
            "conclusion": job["conclusion"],
            "minutes": _minutes(job["startedAt"], job["completedAt"]),
            "timeout_minutes": timeouts.get(job["name"], 360),
        }
        for job in run["jobs"]
    ]
    return {
        "run_id": run["databaseId"],
        "url": run["url"],
        "head_sha": run["headSha"],
        "branch": run["headBranch"],
        "event": run["event"],
        "conclusion": run["conclusion"],
        "created_at": run["createdAt"],
        "updated_at": run["updatedAt"],
        "jobs": sorted(jobs, key=lambda job: job["name"]),
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": f"python scripts/record_ci_run.py --run {run['databaseId']}",
        "note": "timeout_minutes from .github/workflows/ci.yml at the time of recording",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=int, help="run id (default: the latest completed ci.yml run on master)")
    parser.add_argument("--branch", default="master")
    args = parser.parse_args(argv)
    data = record(args.run or latest_run(args.branch))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"ci-run-{data['head_sha'][:7]}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}: run {data['run_id']} {data['conclusion']} at {data['head_sha'][:7]}")
    print("next: python -m evaluation.agent_eval.readme, then commit the record with the READMEs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
