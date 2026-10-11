# Pre-push check at `09da59d`

Run by `scripts/prepush.sh` (2026-10-11T10:48:57Z).

| Item | Value |
|---|---|
| Commit | `09da59de4fb90f4b848ea54f17a45af6c468d279` — eval: independent round-13 slice committed with its first run at aa73005 and one run after the round-17 fixes at 71cb928 (shipped data): chat 0.413 -> 0.978, claims 0.636 -> 1.000 (overlap tiers recorded; two exact claim copies documented); round-17 guards check slices before round 13 |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **all steps passed** |

| Step | Command | Result | Seconds | CI job budget (estimate so far) | Result line from the log |
|---|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 0 | budget: lint est. 0.2 of 10 min (2%; local 0s so far) | 215 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | budget: not run by CI | actionlint: no findings |
| pytest | `python -m pytest -q tests` | pass | 705 | budget: tests est. 61.2 of 120 min (51%; local 705s so far) | 4671 passed, 71 skipped, 5 warnings in 702.56s (0:11:42) |
| pytest-shipped | `QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests` | pass | 672 | budget: tests-shipped est. 47.3 of 120 min (39%; local 672s so far) | 4569 passed, 70 skipped, 103 deselected, 5 warnings in 669.93s (0:11:09) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | budget: frontend est. 0.3 of 15 min (2%; local 0s so far) | Done in 14ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 2 | budget: frontend est. 0.4 of 15 min (3%; local 2s so far) | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 2 | budget: frontend est. 0.5 of 15 min (4%; local 4s so far) | $ eslint . |
| fe-test | `pnpm test` | pass | 4 | budget: frontend est. 0.7 of 15 min (5%; local 8s so far) | Tests  165 passed (165) |
| fe-build | `pnpm build` | pass | 2 | budget: frontend est. 0.8 of 15 min (6%; local 10s so far) | ✓ built in 269ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | budget: frontend est. 0.8 of 15 min (6%; local 10s so far) | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 62655)` | pass | 23 | budget: tests est. 62.4 of 120 min (52%; local 728s so far) | 3 passed (23.2s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 152 | budget: agent-eval-gate est. 8.6 of 45 min (19%; local 152s so far) | result: dev task_success 1.0 (550 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 51 | budget: agent-eval-gate est. 11.2 of 45 min (25%; local 203s so far) | result: ext task_success 1.0 (111 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 23 | budget: agent-eval-gate est. 12.3 of 45 min (27%; local 226s so far) | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 665 | budget: agent-eval-extras est. 34.2 of 60 min (57%; local 665s so far) | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 29 | budget: agent-eval-gate est. 13.8 of 45 min (31%; local 255s so far) | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `11 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1, r10-r12 also FIX=1)` | pass | 260 | budget: heldout-verify est. 14.0 of 45 min (31%; local 260s so far) | 11 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 2 | budget: heldout-verify est. 14.1 of 45 min (31%; local 262s so far) | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered; the generated README regions are up to date |

CI time budgets (`scripts/ci_budget.py`): estimate = runner setup + local seconds x slowdown (default 3; pytest 5, pytest-shipped 4); a step fails when its job's estimate exceeds 60% of the job's `timeout-minutes` in `.github/workflows/ci.yml`. Skipped steps are not counted.

| CI job | timeout-minutes | Steps | Local seconds | Estimated CI minutes | Share of budget | Within budget | Max host load during its steps |
|---|---|---|---|---|---|---|---|
| lint | 10 | ruff | 0 | 0.2 | 2% | yes | 1.73 |
| frontend | 15 | fe-install, fe-typecheck, fe-lint, fe-test, fe-build, dist-diff | 10 | 0.8 | 6% | yes | 1.51 |
| tests | 120 | pytest, e2e | 728 | 62.4 | 52% | yes | 1.73 |
| tests-shipped | 120 | pytest-shipped | 672 | 47.3 | 39% | yes | 1.65 |
| agent-eval-gate | 45 | gate-dev-holdout, gate-ext, gate-shipped, claim-holdout | 255 | 13.8 | 31% | yes | 2.15 |
| agent-eval-extras | 60 | extras | 665 | 34.2 | 57% | yes | 1.88 |
| heldout-verify | 45 | verify-slices, report-check | 262 | 14.1 | 31% | yes | 1.99 |

Busy-host threshold (`CI_BUDGET_BUSY_LOAD`): load average 4.
