"""CI time budgets for the pre-push check (round-12 review L16, Engineering to_100 8.1).

`scripts/prepush.sh` runs each CI step locally. Master CI went red in run 38095861905 because the `agent-eval-gate`
job hit its 45-minute `timeout-minutes` after the round-15 slices were added, and the pre-push check could not see it:
it only knew whether each step passed. This module maps every pre-push step to the CI job that runs it, reads each
job's `timeout-minutes` from `.github/workflows/ci.yml`, estimates the job's CI time from the local durations and
fails a step once its job's estimate exceeds a fraction of the job's budget.

Estimate of a job = its setup seconds on the hosted runner (checkout, Python, dependency install) + the sum over its
steps run so far of local seconds x the step's slowdown factor. A step is checked against the cumulative estimate of
its job, so several steps that are each fast but together too slow (the 38095861905 case) are caught at the step
that crosses the line.

The slowdown factors are hosted-runner seconds per Apple-silicon-laptop second, measured on the same code:

=================  ==========================  ==================  ======
step               CI run 38077019773 (s)      prepush 03dd10d (s)  ratio
=================  ==========================  ==================  ======
pytest             2001 (with coverage)        481                 4.2
gate-dev-holdout   347                         118                 2.9
gate-ext           73                          30                  2.4
extras             1532                        501                 3.1
claim-holdout      73                          28                  2.6
verify-slices      493                         180                 2.7
e2e                57                          23                  2.5
=================  ==========================  ==================  ======

and run 38095861905 vs prepush 7cc9a4c: pytest 2652 / 553 = 4.8, gate-dev-holdout 364 / 123 = 3.0, extras
1718 / 552 = 3.1. Defaults: 3.0 for every step, 5.0 for pytest (CI adds branch coverage). On another host set
``CI_SLOWDOWN`` (the default factor) and ``CI_SLOWDOWN_STEPS`` ("pytest=5,e2e=2.5").

Command line (used by prepush.sh)::

    python scripts/ci_budget.py check --step NAME --seconds N --state FILE   # exit 1 when over budget
    python scripts/ci_budget.py summary --state FILE                          # markdown table per job
    python scripts/ci_budget.py jobs                                          # timeout-minutes per job

Environment: ``CI_BUDGET_FRACTION`` (default 0.6; ``off`` or 0 disables the failure, the table is still written).

The estimate uses wall-clock seconds, so a busy host inflates it: in round 16, with five other evaluation processes
running on the same 4-performance-core laptop, pytest took 1186 s instead of about 550 s. Each step therefore records
the host's load average (the highest of the 1-minute value at its start and the 1-, 5- or 15-minute value at its end,
by the step's length); a step that fails its budget while the load exceeded ``CI_BUDGET_BUSY_LOAD`` (default: the
number of performance cores, else half the CPUs) still fails, and its message says that the host was busy and the
step should be re-measured on an idle host.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# GitHub's default when a job sets no timeout-minutes.
DEFAULT_TIMEOUT_MINUTES = 360
DEFAULT_FRACTION = 0.6
DEFAULT_SLOWDOWN = 3.0

# Pre-push step -> CI job that runs it (None: not run by CI, e.g. actionlint). tests/test_ci_budget.py checks that
# every step of scripts/prepush.sh is listed and every job named here exists in ci.yml.
STEP_JOBS: dict[str, str | None] = {
    "ruff": "lint",
    "actionlint": None,
    "pytest": "tests",
    "pytest-shipped": "tests-shipped",
    "fe-install": "frontend",
    "fe-typecheck": "frontend",
    "fe-lint": "frontend",
    "fe-test": "frontend",
    "fe-build": "frontend",
    "dist-diff": "frontend",
    "e2e": "tests",
    "gate-dev-holdout": "agent-eval-gate",
    "gate-ext": "agent-eval-gate",
    "gate-shipped": "agent-eval-gate",
    "extras": "agent-eval-extras",
    "claim-holdout": "agent-eval-gate",
    "verify-slices": "heldout-verify",
    "report-check": "heldout-verify",
}

# pytest-shipped runs without coverage; 4 is an assumption between the default 3 and pytest-with-coverage 5 until a CI
# run of the tests-shipped job calibrates it (round 17).
STEP_SLOWDOWN: dict[str, float] = {"pytest": 5.0, "pytest-shipped": 4.0}

# Hosted-runner seconds a job spends before its first measured step (set up, checkout, Python, dependency install;
# run 38095861905: tests 121 s, agent-eval-gate 47 s, frontend 16 s, lint 13 s in total).
JOB_SETUP_SECONDS: dict[str, int] = {
    "lint": 15,
    "frontend": 20,
    "tests": 150,
    "tests-shipped": 150,
    "agent-eval-gate": 60,
    "agent-eval-extras": 60,
    "heldout-verify": 60,
}

_JOB_KEY = re.compile(r"^  ([A-Za-z0-9_-]+):\s*(#.*)?$")
_TIMEOUT = re.compile(r"^    timeout-minutes:\s*(\d+)\s*(#.*)?$")


def parse_job_timeouts(text: str) -> dict[str, int]:
    """Job id -> timeout-minutes of a GitHub Actions workflow (job-level keys only; 360 when a job sets none).

    A line parser for the two-space layout of ci.yml, so the pre-push check needs no YAML library; the test compares
    it with yaml.safe_load.
    """
    timeouts: dict[str, int] = {}
    in_jobs = False
    job: str | None = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_jobs = line.rstrip().startswith("jobs:")
            job = None
            continue
        if not in_jobs:
            continue
        match = _JOB_KEY.match(line)
        if match:
            job = match.group(1)
            timeouts[job] = DEFAULT_TIMEOUT_MINUTES
            continue
        match = _TIMEOUT.match(line)
        if match and job is not None:
            timeouts[job] = int(match.group(1))
    return timeouts


def load_job_timeouts(path: Path = CI_WORKFLOW) -> dict[str, int]:
    return parse_job_timeouts(path.read_text(encoding="utf-8"))


def budget_fraction(value: str | None = None) -> float:
    """CI_BUDGET_FRACTION: 0.6 by default; 'off', 'no', 'false' or 0 disable the check."""
    raw = (os.environ.get("CI_BUDGET_FRACTION", "") if value is None else value).strip().lower()
    if not raw:
        return DEFAULT_FRACTION
    if raw in {"off", "no", "false", "none"}:
        return 0.0
    fraction = float(raw)
    if fraction < 0:
        raise ValueError(f"CI_BUDGET_FRACTION must be >= 0, got {raw}")
    return fraction


def slowdowns(default: str | None = None, steps: str | None = None) -> tuple[float, dict[str, float]]:
    """The default factor (CI_SLOWDOWN) and per-step factors (STEP_SLOWDOWN, then CI_SLOWDOWN_STEPS 'a=1,b=2')."""
    raw_default = os.environ.get("CI_SLOWDOWN", "") if default is None else default
    factor = float(raw_default) if raw_default.strip() else DEFAULT_SLOWDOWN
    per_step = dict(STEP_SLOWDOWN)
    raw_steps = os.environ.get("CI_SLOWDOWN_STEPS", "") if steps is None else steps
    for part in raw_steps.split(","):
        if not part.strip():
            continue
        name, _, value = part.partition("=")
        per_step[name.strip()] = float(value)
    return factor, per_step


@dataclass(frozen=True)
class JobEstimate:
    job: str
    timeout_minutes: int
    steps: tuple[tuple[str, int, float], ...]  # (step, local seconds, factor)

    @property
    def setup_seconds(self) -> int:
        return JOB_SETUP_SECONDS.get(self.job, 0)

    @property
    def local_seconds(self) -> int:
        return sum(seconds for _, seconds, _ in self.steps)

    @property
    def estimated_seconds(self) -> float:
        return self.setup_seconds + sum(seconds * factor for _, seconds, factor in self.steps)

    @property
    def share(self) -> float:
        return self.estimated_seconds / (self.timeout_minutes * 60)


def estimate(records: list[dict], timeouts: dict[str, int], default_factor: float, per_step: dict[str, float]):
    """Job -> JobEstimate over the recorded steps (records: {"step", "seconds"}; unmapped steps are ignored)."""
    jobs: dict[str, list[tuple[str, int, float]]] = {}
    for record in records:
        job = STEP_JOBS.get(record["step"])
        if job is None:
            continue
        factor = per_step.get(record["step"], default_factor)
        jobs.setdefault(job, []).append((record["step"], int(record["seconds"]), factor))
    return {
        job: JobEstimate(job, timeouts.get(job, DEFAULT_TIMEOUT_MINUTES), tuple(steps)) for job, steps in jobs.items()
    }


def busy_load_threshold() -> float:
    """CI_BUDGET_BUSY_LOAD, else the number of performance cores (macOS), else half the CPUs."""
    raw = os.environ.get("CI_BUDGET_BUSY_LOAD", "").strip()
    if raw:
        return float(raw)
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, timeout=5, check=False
            ).stdout.strip()
            if out.isdigit() and int(out) > 0:
                return float(out)
        except (OSError, subprocess.SubprocessError):
            pass
    return max(1.0, (os.cpu_count() or 2) / 2)


def step_load(seconds: int, load_start: float | None, load_end: tuple[float, float, float] | None) -> float | None:
    """The host load a step ran under: max of the 1-min load at its start and the end-of-step load average whose
    window best covers the step (1 min under 5 min, 5 min under 15 min, else 15 min)."""
    values = [] if load_start is None else [load_start]
    if load_end is not None:
        values.append(load_end[0] if seconds < 300 else load_end[1] if seconds < 900 else load_end[2])
    return round(max(values), 2) if values else None


def _read_state(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(
    step: str,
    seconds: int,
    state: Path,
    workflow: Path = CI_WORKFLOW,
    load_start: float | None = None,
    load_end: tuple[float, float, float] | None = None,
) -> tuple[bool, str]:
    """Record a step's local duration; (within budget, one-line message) for the step's job so far."""
    if step not in STEP_JOBS:
        return False, f"budget: step {step!r} is not mapped to a CI job in scripts/ci_budget.py STEP_JOBS"
    load = step_load(int(seconds), load_start, load_end)
    records = [*_read_state(state), {"step": step, "seconds": int(seconds), "load": load}]
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    job = STEP_JOBS[step]
    if job is None:
        return True, "budget: not run by CI"
    timeouts = load_job_timeouts(workflow)
    if job not in timeouts:
        return False, f"budget: CI job {job!r} (step {step}) is not in {workflow.name}"
    fraction = budget_fraction()
    default_factor, per_step = slowdowns()
    job_estimate = estimate(records, timeouts, default_factor, per_step)[job]
    message = (
        f"budget: {job} est. {job_estimate.estimated_seconds / 60:.1f} of {job_estimate.timeout_minutes} min "
        f"({job_estimate.share:.0%}; local {job_estimate.local_seconds}s so far)"
    )
    if fraction and job_estimate.share > fraction:
        message = f"{message} > {fraction:.0%} of the job's timeout-minutes"
        job_loads = [r.get("load") for r in records if STEP_JOBS.get(r["step"]) == job and r.get("load") is not None]
        threshold = busy_load_threshold()
        if job_loads and max(job_loads) > threshold:
            busy = f"; host busy (load {max(job_loads):g} > {threshold:g})"
            message += f"{busy}: local times inflated, re-measure on an idle host"
        return False, message
    return True, message


def summary(state: Path, workflow: Path = CI_WORKFLOW) -> str:
    """Markdown table: per CI job, its budget, the local seconds of its steps and the estimate."""
    timeouts = load_job_timeouts(workflow)
    fraction = budget_fraction()
    default_factor, per_step = slowdowns()
    estimates = estimate(_read_state(state), timeouts, default_factor, per_step)
    factors = ", ".join(f"{name} {value:g}" for name, value in sorted(per_step.items()))
    lines = [
        f"CI time budgets (`scripts/ci_budget.py`): estimate = runner setup + local seconds x slowdown "
        f"(default {default_factor:g}; {factors}); a step fails when its job's estimate exceeds "
        f"{f'{fraction:.0%}' if fraction else 'no limit (CI_BUDGET_FRACTION=off)'} of the job's `timeout-minutes` "
        "in `.github/workflows/ci.yml`. Skipped steps are not counted.",
        "",
        "| CI job | timeout-minutes | Steps | Local seconds | Estimated CI minutes | Share of budget | Within budget "
        "| Max host load during its steps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    loads: dict[str, list[float]] = {}
    for record in _read_state(state):
        job = STEP_JOBS.get(record["step"])
        if job and record.get("load") is not None:
            loads.setdefault(job, []).append(record["load"])
    for job in sorted(estimates, key=lambda name: list(timeouts).index(name) if name in timeouts else 99):
        item = estimates[job]
        ok = not fraction or item.share <= fraction
        lines.append(
            f"| {job} | {item.timeout_minutes} | {', '.join(step for step, _, _ in item.steps)} | "
            f"{item.local_seconds} | {item.estimated_seconds / 60:.1f} | {item.share:.0%} | "
            f"{'yes' if ok else '**no**'} | {max(loads[job]) if job in loads else '–'} |"
        )
    lines += ["", f"Busy-host threshold (`CI_BUDGET_BUSY_LOAD`): load average {busy_load_threshold():g}."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--step", required=True)
    check_parser.add_argument("--seconds", required=True, type=int)
    check_parser.add_argument("--state", required=True, type=Path)
    check_parser.add_argument("--load-start", type=float, default=None, help="1-min load average at the step's start")
    summary_parser = sub.add_parser("summary")
    summary_parser.add_argument("--state", required=True, type=Path)
    sub.add_parser("jobs")
    args = parser.parse_args(argv)
    if args.command == "check":
        ok, message = check(args.step, args.seconds, args.state, load_start=args.load_start, load_end=os.getloadavg())
        print(message)
        return 0 if ok else 1
    if args.command == "summary":
        print(summary(args.state))
        return 0
    for job, minutes in load_job_timeouts().items():
        print(f"{job}\t{minutes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
