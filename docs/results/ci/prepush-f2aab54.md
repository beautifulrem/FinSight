# Pre-push check at `f2aab54`

Run by `scripts/prepush.sh` (2026-10-11T08:18:28Z).

| Item | Value |
|---|---|
| Commit | `f2aab544673a8f39574edff000eaa5837310827b` — docs: round-17 claim rules (definitions, same-company and decline-rate comparisons, elided 扣非 vs 归母, growth multiples, the missing-corrections note) and their runs; M12 (partly fixed) and M13 (fixed) rows; L19's claim-check residual closed (en/zh) |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **at least one step failed** |

| Step | Command | Result | Seconds | CI job budget (estimate so far) | Result line from the log |
|---|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 1 | budget: lint est. 0.3 of 360 min (0%; local 1s so far) | 207 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | budget: not run by CI | actionlint: no findings |
| pytest | `python -m pytest -q tests` | **FAIL** (CI time budget) | 1206 | **budget: tests est. 103.0 of 120 min (86%; local 1206s so far) > 60% of the job's timeout-minutes; host busy (load 5.98 > 4): local times inflated, re-measure on an idle host** | 4494 passed, 68 skipped, 5 warnings in 1203.01s (0:20:03) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | budget: frontend est. 0.3 of 15 min (2%; local 0s so far) | Done in 39ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 2 | budget: frontend est. 0.4 of 15 min (3%; local 2s so far) | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 2 | budget: frontend est. 0.5 of 15 min (4%; local 4s so far) | $ eslint . |
| fe-test | `pnpm test` | pass | 4 | budget: frontend est. 0.7 of 15 min (5%; local 8s so far) | Tests  164 passed (164) |
| fe-build | `pnpm build` | pass | 3 | budget: frontend est. 0.9 of 15 min (6%; local 11s so far) | ✓ built in 248ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | budget: frontend est. 0.9 of 15 min (6%; local 11s so far) | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 60152)` | **FAIL** (CI time budget) | 24 | **budget: tests est. 104.2 of 120 min (87%; local 1230s so far) > 60% of the job's timeout-minutes; host busy (load 5.98 > 4): local times inflated, re-measure on an idle host** | 2 passed (24.2s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 211 | budget: agent-eval-gate est. 11.6 of 45 min (26%; local 211s so far) | result: dev task_success 1.0 (538 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 66 | budget: agent-eval-gate est. 14.8 of 45 min (33%; local 277s so far) | result: ext task_success 1.0 (99 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 34 | budget: agent-eval-gate est. 16.6 of 45 min (37%; local 311s so far) | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | **FAIL** (CI time budget) | 930 | **budget: agent-eval-extras est. 47.5 of 60 min (79%; local 930s so far) > 60% of the job's timeout-minutes; host busy (load 4.5 > 4): local times inflated, re-measure on an idle host** | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 45 | budget: agent-eval-gate est. 18.8 of 45 min (42%; local 356s so far) | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `10 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1, r10/r11 also FIX=1)` | pass | 437 | budget: heldout-verify est. 22.9 of 45 min (51%; local 437s so far) | 10 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 5 | budget: heldout-verify est. 23.1 of 45 min (51%; local 442s so far) | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |

CI time budgets (`scripts/ci_budget.py`): estimate = runner setup + local seconds x slowdown (default 3; pytest 5); a step fails when its job's estimate exceeds 60% of the job's `timeout-minutes` in `.github/workflows/ci.yml`. Skipped steps are not counted.

| CI job | timeout-minutes | Steps | Local seconds | Estimated CI minutes | Share of budget | Within budget | Max host load during its steps |
|---|---|---|---|---|---|---|---|
| lint | 360 | ruff | 1 | 0.3 | 0% | yes | 5.98 |
| frontend | 15 | fe-install, fe-typecheck, fe-lint, fe-test, fe-build, dist-diff | 11 | 0.9 | 6% | yes | 2.87 |
| tests | 120 | pytest, e2e | 1230 | 104.2 | 87% | **no** | 5.98 |
| agent-eval-gate | 45 | gate-dev-holdout, gate-ext, gate-shipped, claim-holdout | 356 | 18.8 | 42% | yes | 4.85 |
| agent-eval-extras | 60 | extras | 930 | 47.5 | 79% | **no** | 4.5 |
| heldout-verify | 45 | verify-slices, report-check | 442 | 23.1 | 51% | yes | 5.47 |

Busy-host threshold (`CI_BUDGET_BUSY_LOAD`): load average 4.
