# Pre-push check at `cf36ec5`

Run by `scripts/prepush.sh` (2026-10-14T19:31:38Z).

| Item | Value |
|---|---|
| Commit | `cf36ec58844b02d943b478e9bbcb23d8862809ca` — fix(data): restore CRLF line endings in alias_table.csv (the round-18 merge rewrote them as LF); ci: agent-eval-extras timeout 75 min |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **all steps passed** |

| Step | Command | Result | Seconds | CI job budget (estimate so far) | Result line from the log |
|---|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 0 | budget: lint est. 0.2 of 10 min (2%; local 0s so far) | 226 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | budget: not run by CI | actionlint: no findings |
| pytest | `python -m pytest -q tests` | pass | 768 | budget: tests est. 66.5 of 120 min (55%; local 768s so far) | 4891 passed, 71 skipped, 5 warnings in 765.95s (0:12:45) |
| pytest-shipped | `QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests` | pass | 730 | budget: tests-shipped est. 51.2 of 120 min (43%; local 730s so far) | 4789 passed, 70 skipped, 103 deselected, 5 warnings in 727.87s (0:12:07) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | budget: frontend est. 0.3 of 15 min (2%; local 0s so far) | Done in 28ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 1 | budget: frontend est. 0.4 of 15 min (3%; local 1s so far) | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 2 | budget: frontend est. 0.5 of 15 min (3%; local 3s so far) | $ eslint . |
| fe-test | `pnpm test` | pass | 3 | budget: frontend est. 0.6 of 15 min (4%; local 6s so far) | Tests  175 passed (175) |
| fe-build | `pnpm build` | pass | 2 | budget: frontend est. 0.7 of 15 min (5%; local 8s so far) | ✓ built in 198ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | budget: frontend est. 0.7 of 15 min (5%; local 8s so far) | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 57569)` | pass | 26 | budget: tests est. 67.8 of 120 min (56%; local 794s so far) | 4 passed (25.7s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 168 | budget: agent-eval-gate est. 9.4 of 45 min (21%; local 168s so far) | result: dev task_success 1.0 (558 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 61 | budget: agent-eval-gate est. 12.4 of 45 min (28%; local 229s so far) | result: ext task_success 1.0 (123 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 25 | budget: agent-eval-gate est. 13.7 of 45 min (30%; local 254s so far) | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 720 | budget: agent-eval-extras est. 37.0 of 75 min (49%; local 720s so far) | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 30 | budget: agent-eval-gate est. 15.2 of 45 min (34%; local 284s so far) | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `python -m evaluation.agent_eval.verify_slices (CI's list: 12 scripts)` | pass | 308 | budget: heldout-verify est. 16.4 of 45 min (36%; local 308s so far) | 13 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 2 | budget: heldout-verify est. 16.5 of 45 min (37%; local 310s so far) | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered; the generated README regions are up to date |

CI time budgets (`scripts/ci_budget.py`): estimate = runner setup + local seconds x slowdown (default 3; pytest 5, pytest-shipped 4); a step fails when its job's estimate exceeds 60% of the job's `timeout-minutes` in `.github/workflows/ci.yml`. Skipped steps are not counted.

| CI job | timeout-minutes | Steps | Local seconds | Estimated CI minutes | Share of budget | Within budget | Max host load during its steps |
|---|---|---|---|---|---|---|---|
| lint | 10 | ruff | 0 | 0.2 | 2% | yes | 2.01 |
| frontend | 15 | fe-install, fe-typecheck, fe-lint, fe-test, fe-build, dist-diff | 8 | 0.7 | 5% | yes | 1.65 |
| tests | 120 | pytest, e2e | 794 | 67.8 | 56% | yes | 2.01 |
| tests-shipped | 120 | pytest-shipped | 730 | 51.2 | 43% | yes | 1.95 |
| agent-eval-gate | 45 | gate-dev-holdout, gate-ext, gate-shipped, claim-holdout | 284 | 15.2 | 34% | yes | 2.37 |
| agent-eval-extras | 75 | extras | 720 | 37.0 | 49% | yes | 2.05 |
| heldout-verify | 45 | verify-slices, report-check | 310 | 16.5 | 37% | yes | 1.98 |

Busy-host threshold (`CI_BUDGET_BUSY_LOAD`): load average 4.
