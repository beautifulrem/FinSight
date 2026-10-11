# Pre-push check at `7c30f7e`

Run by `scripts/prepush.sh` (2026-10-11T08:49:19Z).

| Item | Value |
|---|---|
| Commit | `7c30f7e0965c42f61f79e9f4ec4e209dc67e8d9f` — style(holdings): wrap two round-17 regex lines (ruff E501) |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **at least one step failed** |

| Step | Command | Result | Seconds | CI job budget (estimate so far) | Result line from the log |
|---|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 0 | budget: lint est. 0.2 of 360 min (0%; local 0s so far) | 205 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | budget: not run by CI | actionlint: no findings |
| pytest | `python -m pytest -q tests` | **FAIL** (CI time budget) | 1024 | **budget: tests est. 87.8 of 120 min (73%; local 1024s so far) > 60% of the job's timeout-minutes; host busy (load 4.36 > 4): local times inflated, re-measure on an idle host** | 4445 passed, 70 skipped, 5 warnings in 1021.66s (0:17:01) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | budget: frontend est. 0.3 of 15 min (2%; local 0s so far) | Done in 32ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 3 | budget: frontend est. 0.5 of 15 min (3%; local 3s so far) | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 4 | budget: frontend est. 0.7 of 15 min (5%; local 7s so far) | $ eslint . |
| fe-test | `pnpm test` | pass | 6 | budget: frontend est. 1.0 of 15 min (7%; local 13s so far) | Tests  165 passed (165) |
| fe-build | `pnpm build` | pass | 3 | budget: frontend est. 1.1 of 15 min (8%; local 16s so far) | ✓ built in 292ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | budget: frontend est. 1.1 of 15 min (8%; local 16s so far) | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 52102)` | **FAIL** (CI time budget) | 33 | **budget: tests est. 89.5 of 120 min (75%; local 1057s so far) > 60% of the job's timeout-minutes; host busy (load 4.57 > 4): local times inflated, re-measure on an idle host** | 2 passed (33.1s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 240 | budget: agent-eval-gate est. 13.0 of 45 min (29%; local 240s so far) | result: dev task_success 1.0 (550 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 79 | budget: agent-eval-gate est. 16.9 of 45 min (38%; local 319s so far) | result: ext task_success 1.0 (111 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 35 | budget: agent-eval-gate est. 18.7 of 45 min (42%; local 354s so far) | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | **FAIL** (CI time budget) | 955 | **budget: agent-eval-extras est. 48.8 of 60 min (81%; local 955s so far) > 60% of the job's timeout-minutes; host busy (load 4.15 > 4): local times inflated, re-measure on an idle host** | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 46 | budget: agent-eval-gate est. 21.0 of 45 min (47%; local 400s so far) | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `10 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1, r10/r11 also FIX=1)` | pass | 358 | budget: heldout-verify est. 18.9 of 45 min (42%; local 358s so far) | 10 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 3 | budget: heldout-verify est. 19.1 of 45 min (42%; local 361s so far) | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |

CI time budgets (`scripts/ci_budget.py`): estimate = runner setup + local seconds x slowdown (default 3; pytest 5); a step fails when its job's estimate exceeds 60% of the job's `timeout-minutes` in `.github/workflows/ci.yml`. Skipped steps are not counted.

| CI job | timeout-minutes | Steps | Local seconds | Estimated CI minutes | Share of budget | Within budget | Max host load during its steps |
|---|---|---|---|---|---|---|---|
| lint | 360 | ruff | 0 | 0.2 | 0% | yes | 3.72 |
| frontend | 15 | fe-install, fe-typecheck, fe-lint, fe-test, fe-build, dist-diff | 16 | 1.1 | 8% | yes | 4.57 |
| tests | 120 | pytest, e2e | 1057 | 89.5 | 75% | **no** | 4.57 |
| agent-eval-gate | 45 | gate-dev-holdout, gate-ext, gate-shipped, claim-holdout | 400 | 21.0 | 47% | yes | 4.25 |
| agent-eval-extras | 60 | extras | 955 | 48.8 | 81% | **no** | 4.15 |
| heldout-verify | 45 | verify-slices, report-check | 361 | 19.1 | 42% | yes | 4.08 |

Busy-host threshold (`CI_BUDGET_BUSY_LOAD`): load average 4.
