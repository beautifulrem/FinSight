"""scripts/ci_budget.py: the pre-push check's CI time budgets (round-12 review L16; run 38095861905 was cancelled at
the agent-eval-gate job's 45-minute timeout, which scripts/prepush.sh could not see)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import ci_budget

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

SAMPLE = """name: CI
on:
  push:
    branches: [master]
env:
  timeout-minutes: 99   # top-level env, not a job
jobs:
  lint:
    runs-on: ubuntu-24.04
    steps:
      - run: ruff check .
  gate:
    # a comment between the key and its settings
    runs-on: ubuntu-24.04
    timeout-minutes: 45
    steps:
      - name: step with its own timeout
        timeout-minutes: 5
        run: python -m x
  docker:
    timeout-minutes: 30  # trailing comment
    runs-on: ubuntu-24.04
"""

# A command each mapped step's CI job must run (keeps STEP_JOBS honest when ci.yml moves a step to another job).
STEP_TOKENS = {
    "ruff": "ruff check .",
    "pytest": "python -m pytest -q tests",
    "pytest-shipped": "QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests",
    "fe-install": "pnpm install --frozen-lockfile",
    "fe-typecheck": "pnpm typecheck",
    "fe-lint": "pnpm lint",
    "fe-test": "pnpm test",
    "fe-build": "pnpm build",
    "dist-diff": "git status --porcelain query_intelligence/web/dist",
    "e2e": "pnpm run e2e",
    "gate-dev-holdout": "gate --set dev --set holdout",
    "gate-ext": "gate --set ext",
    "gate-shipped": "gate --set shipped",
    "extras": "gate --extras-only",
    "claim-holdout": "claim_bench.run --set holdout",
    "verify-slices": "verify_heldout_r11.py",
    "report-check": "report --check",
}

# Local seconds of the pre-push run at 7cc9a4c (docs/results/ci/prepush-7cc9a4c.md), whose CI run 38095861905 hit the
# agent-eval-gate timeout.
LOCAL_7CC9A4C = {
    "ruff": 0,
    "pytest": 553,
    "fe-install": 0,
    "fe-typecheck": 1,
    "fe-lint": 3,
    "fe-test": 3,
    "fe-build": 2,
    "dist-diff": 0,
    "e2e": 23,
    "gate-dev-holdout": 123,
    "gate-ext": 40,
    "gate-shipped": 23,
    "extras": 552,
    "claim-holdout": 30,
    "verify-slices": 180,
    "report-check": 2,
}


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    for name in ("CI_BUDGET_FRACTION", "CI_SLOWDOWN", "CI_SLOWDOWN_STEPS"):
        monkeypatch.delenv(name, raising=False)


def test_parse_job_timeouts_reads_job_level_keys_only():
    assert ci_budget.parse_job_timeouts(SAMPLE) == {"lint": 360, "gate": 45, "docker": 30}


def test_parse_job_timeouts_matches_yaml_on_the_real_workflow():
    yaml = pytest.importorskip("yaml")
    jobs = yaml.safe_load(WORKFLOW_TEXT)["jobs"]
    expected = {name: int(job.get("timeout-minutes", 360)) for name, job in jobs.items()}
    assert ci_budget.parse_job_timeouts(WORKFLOW_TEXT) == expected


def test_every_prepush_step_is_mapped_and_every_mapped_job_exists():
    prepush = (ROOT / "scripts" / "prepush.sh").read_text(encoding="utf-8")
    steps = re.findall(r"^run ([a-z0-9-]+) ", prepush, flags=re.MULTILINE)
    assert steps and set(steps) == set(ci_budget.STEP_JOBS)
    timeouts = ci_budget.parse_job_timeouts(WORKFLOW_TEXT)
    assert {job for job in ci_budget.STEP_JOBS.values() if job} <= set(timeouts)
    assert set(ci_budget.JOB_SETUP_SECONDS) <= set(timeouts)


def test_each_mapped_step_runs_in_its_ci_job():
    yaml = pytest.importorskip("yaml")
    jobs = yaml.safe_load(WORKFLOW_TEXT)["jobs"]
    assert set(STEP_TOKENS) == {step for step, job in ci_budget.STEP_JOBS.items() if job}
    for step, token in STEP_TOKENS.items():
        commands = " ".join(str(item.get("run", "")) for item in jobs[ci_budget.STEP_JOBS[step]]["steps"])
        assert token in " ".join(commands.split()), (step, ci_budget.STEP_JOBS[step], token)


def test_budget_fraction_and_slowdown_configuration():
    assert ci_budget.budget_fraction("") == 0.6
    assert ci_budget.budget_fraction("0.5") == 0.5
    assert ci_budget.budget_fraction("off") == 0.0
    with pytest.raises(ValueError):
        ci_budget.budget_fraction("-1")
    default, per_step = ci_budget.slowdowns("", "")
    assert default == 3.0 and per_step["pytest"] == 5.0 and per_step["pytest-shipped"] == 4.0
    default, per_step = ci_budget.slowdowns("2", "pytest=4, e2e=2.5")
    assert default == 2.0 and per_step == {"pytest": 4.0, "pytest-shipped": 4.0, "e2e": 2.5}


def test_check_accumulates_a_job_and_fails_the_step_that_crosses_the_budget(tmp_path):
    workflow = tmp_path / "ci.yml"
    text = "jobs:\n  agent-eval-gate:\n    timeout-minutes: 10\n  heldout-verify:\n    timeout-minutes: 45\n"
    workflow.write_text(text, encoding="utf-8")
    state = tmp_path / "state.jsonl"
    # setup 60 s + 60 s x 3 = 4 of 10 min: within 60%
    ok, message = ci_budget.check("gate-dev-holdout", 60, state, workflow)
    assert ok and "agent-eval-gate est. 4.0 of 10 min (40%" in message
    # another job does not count against it
    assert ci_budget.check("verify-slices", 100, state, workflow)[0]
    # + 30 s x 3: 5.5 of 10 min, still within
    assert ci_budget.check("gate-ext", 30, state, workflow)[0]
    # + 20 s x 3: 6.5 of 10 min > 60%: this step fails although it alone is short
    ok, message = ci_budget.check("gate-shipped", 20, state, workflow)
    assert not ok and "6.5 of 10 min (65%" in message and "> 60%" in message
    assert [json.loads(line)["step"] for line in state.read_text().splitlines()] == [
        "gate-dev-holdout",
        "verify-slices",
        "gate-ext",
        "gate-shipped",
    ]


def test_check_can_be_switched_off_and_rejects_unmapped_steps(tmp_path, monkeypatch):
    workflow = tmp_path / "ci.yml"
    workflow.write_text("jobs:\n  tests:\n    timeout-minutes: 1\n", encoding="utf-8")
    monkeypatch.setenv("CI_BUDGET_FRACTION", "off")
    ok, message = ci_budget.check("pytest", 600, tmp_path / "s.jsonl", workflow)
    assert ok and "tests est." in message
    assert ci_budget.check("actionlint", 5, tmp_path / "s.jsonl", workflow) == (True, "budget: not run by CI")
    ok, message = ci_budget.check("no-such-step", 1, tmp_path / "s.jsonl", workflow)
    assert not ok and "not mapped" in message
    missing = tmp_path / "other.yml"
    missing.write_text("jobs:\n  lint:\n    runs-on: x\n", encoding="utf-8")
    ok, message = ci_budget.check("pytest", 1, tmp_path / "t.jsonl", missing)
    assert not ok and "is not in other.yml" in message


def _estimates(job_of_step: dict[str, str], timeouts: dict[str, int]):
    records = [{"step": step, "seconds": seconds} for step, seconds in LOCAL_7CC9A4C.items()]
    default, per_step = ci_budget.slowdowns("", "")
    original = dict(ci_budget.STEP_JOBS)
    try:
        ci_budget.STEP_JOBS.update(job_of_step)
        return ci_budget.estimate(records, timeouts, default, per_step)
    finally:
        ci_budget.STEP_JOBS.clear()
        ci_budget.STEP_JOBS.update(original)


def test_the_38095861905_layout_would_have_failed_before_the_push():
    # bb99360's ci.yml: extras, claim-holdout, verify-slices and report-check all in agent-eval-gate (45 min), tests 60
    old_layout = {"extras": "agent-eval-gate", "verify-slices": "agent-eval-gate", "report-check": "agent-eval-gate"}
    estimates = _estimates(old_layout, {"agent-eval-gate": 45, "tests": 60, "frontend": 15, "lint": 360})
    gate = estimates["agent-eval-gate"]
    assert gate.share > 1.0  # estimated past the timeout itself (the run was cancelled at 45 min)
    assert estimates["tests"].share > 0.6  # 47 of 60 min in that run


def test_the_current_layout_is_within_budget_at_7cc9a4c_durations():
    estimates = _estimates({}, ci_budget.parse_job_timeouts(WORKFLOW_TEXT))
    for job, item in estimates.items():
        assert item.share <= ci_budget.DEFAULT_FRACTION, (job, round(item.share, 2))


def test_summary_lists_each_job(tmp_path):
    state = tmp_path / "state.jsonl"
    for step in ("ruff", "pytest", "e2e", "actionlint"):
        ci_budget.check(step, 10, state)
    text = ci_budget.summary(state)
    assert "| tests | 120 | pytest, e2e | 20 |" in text
    assert "| lint |" in text and "actionlint" not in text.split("|---|")[-1]


def test_cli_exit_status(tmp_path, capsys):
    state = tmp_path / "state.jsonl"
    assert ci_budget.main(["check", "--step", "ruff", "--seconds", "1", "--state", str(state)]) == 0
    assert ci_budget.main(["check", "--step", "pytest", "--seconds", "99999", "--state", str(state)]) == 1
    assert "> 60%" in capsys.readouterr().out
    assert ci_budget.main(["jobs"]) == 0
    assert "agent-eval-gate\t45" in capsys.readouterr().out


def test_step_load_uses_the_window_that_covers_the_step():
    assert ci_budget.step_load(60, 1.0, (2.0, 3.0, 4.0)) == 2.0
    assert ci_budget.step_load(400, 1.0, (2.0, 3.0, 4.0)) == 3.0
    assert ci_budget.step_load(1200, 5.0, (2.0, 3.0, 4.0)) == 5.0
    assert ci_budget.step_load(10, None, None) is None


def test_a_budget_failure_on_a_busy_host_says_so(tmp_path, monkeypatch):
    workflow = tmp_path / "ci.yml"
    workflow.write_text("jobs:\n  tests:\n    timeout-minutes: 90\n", encoding="utf-8")
    monkeypatch.setenv("CI_BUDGET_BUSY_LOAD", "4")
    # round 16: pytest took 1186 s under load 7 (about 550 s on an idle host)
    ok, message = ci_budget.check("pytest", 1186, tmp_path / "busy.jsonl", workflow, 7.2, (6.0, 6.5, 6.9))
    assert not ok and "host busy (load 7.2 > 4)" in message and "re-measure on an idle host" in message
    ok, message = ci_budget.check("pytest", 1186, tmp_path / "idle.jsonl", workflow, 1.0, (1.5, 1.2, 1.1))
    assert not ok and "host busy" not in message
    assert ci_budget.check("pytest", 550, tmp_path / "ok.jsonl", workflow, 1.0, (1.5, 1.2, 1.1))[0]
    text = ci_budget.summary(tmp_path / "busy.jsonl", workflow)
    assert "| tests | 90 | pytest | 1186 |" in text and text.splitlines()[4].endswith("| 7.2 |")
    assert "Busy-host threshold (`CI_BUDGET_BUSY_LOAD`): load average 4." in text
