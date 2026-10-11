# Pre-push check at `e7bf090`

Run by `scripts/prepush.sh` (2026-10-11T09:56:07Z).

| Item | Value |
|---|---|
| Commit | `e7bf090494c5aa025503504b4963327beee3d02b` — docs(readme): the CI paragraph names pre-push steps that failed only the time-budget check |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **all steps passed** |

| Step | Command | Result | Seconds | CI job budget (estimate so far) | Result line from the log |
|---|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 1 | budget: lint est. 0.3 of 10 min (3%; local 1s so far) | 207 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | budget: not run by CI | actionlint: no findings |
| pytest | `python -m pytest -q tests` | pass | 653 | budget: tests est. 56.9 of 120 min (47%; local 653s so far) | 4410 passed, 71 skipped, 5 warnings in 651.52s (0:10:51) |
| pytest-shipped | `QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests` | pass | 597 | budget: tests-shipped est. 42.3 of 120 min (35%; local 597s so far) | 4308 passed, 70 skipped, 103 deselected, 5 warnings in 594.56s (0:09:54) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | budget: frontend est. 0.3 of 15 min (2%; local 0s so far) | Done in 9ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 2 | budget: frontend est. 0.4 of 15 min (3%; local 2s so far) | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 1 | budget: frontend est. 0.5 of 15 min (3%; local 3s so far) | $ eslint . |
| fe-test | `pnpm test` | pass | 3 | budget: frontend est. 0.6 of 15 min (4%; local 6s so far) | Tests  164 passed (164) |
| fe-build | `pnpm build` | pass | 1 | budget: frontend est. 0.7 of 15 min (5%; local 7s so far) | ✓ built in 136ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | budget: frontend est. 0.7 of 15 min (5%; local 7s so far) | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 51812)` | pass | 24 | budget: tests est. 58.1 of 120 min (48%; local 677s so far) | 3 passed (23.1s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 137 | budget: agent-eval-gate est. 7.8 of 45 min (17%; local 137s so far) | result: dev task_success 1.0 (538 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 43 | budget: agent-eval-gate est. 10.0 of 45 min (22%; local 180s so far) | result: ext task_success 1.0 (99 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 23 | budget: agent-eval-gate est. 11.2 of 45 min (25%; local 203s so far) | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 609 | budget: agent-eval-extras est. 31.4 of 60 min (52%; local 609s so far) | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 29 | budget: agent-eval-gate est. 12.6 of 45 min (28%; local 232s so far) | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `11 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1, r10-r12 also FIX=1)` | pass | 257 | budget: heldout-verify est. 13.8 of 45 min (31%; local 257s so far) | 11 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 2 | budget: heldout-verify est. 13.9 of 45 min (31%; local 259s so far) | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered; the generated README regions are up to date |

CI time budgets (`scripts/ci_budget.py`): estimate = runner setup + local seconds x slowdown (default 3; pytest 5, pytest-shipped 4); a step fails when its job's estimate exceeds 60% of the job's `timeout-minutes` in `.github/workflows/ci.yml`. Skipped steps are not counted.

| CI job | timeout-minutes | Steps | Local seconds | Estimated CI minutes | Share of budget | Within budget | Max host load during its steps |
|---|---|---|---|---|---|---|---|
| lint | 10 | ruff | 1 | 0.3 | 3% | yes | 2.26 |
| frontend | 15 | fe-install, fe-typecheck, fe-lint, fe-test, fe-build, dist-diff | 7 | 0.7 | 5% | yes | 2.62 |
| tests | 120 | pytest, e2e | 677 | 58.1 | 48% | yes | 2.8 |
| tests-shipped | 120 | pytest-shipped | 597 | 42.3 | 35% | yes | 1.75 |
| agent-eval-gate | 45 | gate-dev-holdout, gate-ext, gate-shipped, claim-holdout | 232 | 12.6 | 28% | yes | 2.8 |
| agent-eval-extras | 60 | extras | 609 | 31.4 | 52% | yes | 1.99 |
| heldout-verify | 45 | verify-slices, report-check | 259 | 13.9 | 31% | yes | 2.09 |

Busy-host threshold (`CI_BUDGET_BUSY_LOAD`): load average 4.
