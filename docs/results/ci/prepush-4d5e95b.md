# Pre-push check at `4d5e95b`

Run by `scripts/prepush.sh` (2026-10-11T09:04:25Z).

| Item | Value |
|---|---|
| Commit | `4d5e95bf74b854e52e2910de6f9dd1b41abf4141` — docs(evaluation-history): how the round-17 lookalike counts were measured, now that the test samples the fuzzy pass (en/zh) |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **at least one step failed** |

| Step | Command | Result | Seconds | CI job budget (estimate so far) | Result line from the log |
|---|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 0 | budget: lint est. 0.2 of 360 min (0%; local 0s so far) | 208 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | budget: not run by CI | actionlint: no findings |
| pytest | `python -m pytest -q tests` | **FAIL** (CI time budget) | 1005 | **budget: tests est. 86.2 of 120 min (72%; local 1005s so far) > 60% of the job's timeout-minutes; host busy (load 4.08 > 4): local times inflated, re-measure on an idle host** | 4517 passed, 68 skipped, 5 warnings in 1001.79s (0:16:41) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | budget: frontend est. 0.3 of 15 min (2%; local 0s so far) | Done in 35ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 4 | budget: frontend est. 0.5 of 15 min (4%; local 4s so far) | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 3 | budget: frontend est. 0.7 of 15 min (5%; local 7s so far) | $ eslint . |
| fe-test | `pnpm test` | pass | 7 | budget: frontend est. 1.0 of 15 min (7%; local 14s so far) | Tests  164 passed (164) |
| fe-build | `pnpm build` | pass | 3 | budget: frontend est. 1.2 of 15 min (8%; local 17s so far) | ✓ built in 314ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | budget: frontend est. 1.2 of 15 min (8%; local 17s so far) | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 55855)` | **FAIL** (CI time budget) | 36 | **budget: tests est. 88.0 of 120 min (73%; local 1041s so far) > 60% of the job's timeout-minutes; host busy (load 4.65 > 4): local times inflated, re-measure on an idle host** | 2 passed (35.6s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 216 | budget: agent-eval-gate est. 11.8 of 45 min (26%; local 216s so far) | result: dev task_success 1.0 (538 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 67 | budget: agent-eval-gate est. 15.2 of 45 min (34%; local 283s so far) | result: ext task_success 1.0 (99 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 34 | budget: agent-eval-gate est. 16.9 of 45 min (37%; local 317s so far) | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | **FAIL** (CI time budget) | 1001 | **budget: agent-eval-extras est. 51.0 of 60 min (85%; local 1001s so far) > 60% of the job's timeout-minutes** | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 36 | budget: agent-eval-gate est. 18.6 of 45 min (41%; local 353s so far) | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `10 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1, r10/r11 also FIX=1)` | pass | 294 | budget: heldout-verify est. 15.7 of 45 min (35%; local 294s so far) | 10 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 2 | budget: heldout-verify est. 15.8 of 45 min (35%; local 296s so far) | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |

CI time budgets (`scripts/ci_budget.py`): estimate = runner setup + local seconds x slowdown (default 3; pytest 5); a step fails when its job's estimate exceeds 60% of the job's `timeout-minutes` in `.github/workflows/ci.yml`. Skipped steps are not counted.

| CI job | timeout-minutes | Steps | Local seconds | Estimated CI minutes | Share of budget | Within budget | Max host load during its steps |
|---|---|---|---|---|---|---|---|
| lint | 360 | ruff | 0 | 0.2 | 0% | yes | 4.05 |
| frontend | 15 | fe-install, fe-typecheck, fe-lint, fe-test, fe-build, dist-diff | 17 | 1.2 | 8% | yes | 4.65 |
| tests | 120 | pytest, e2e | 1041 | 88.0 | 73% | **no** | 4.65 |
| agent-eval-gate | 45 | gate-dev-holdout, gate-ext, gate-shipped, claim-holdout | 353 | 18.6 | 41% | yes | 4.5 |
| agent-eval-extras | 60 | extras | 1001 | 51.0 | 85% | **no** | 3.95 |
| heldout-verify | 45 | verify-slices, report-check | 296 | 15.8 | 35% | yes | 3.77 |

Busy-host threshold (`CI_BUDGET_BUSY_LOAD`): load average 4.
