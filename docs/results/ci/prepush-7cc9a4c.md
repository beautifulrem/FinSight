# Pre-push check at `7cc9a4c`

Run by `scripts/prepush.sh` (2026-10-10T23:28:00Z).

| Item | Value |
|---|---|
| Commit | `7cc9a4c2006ca2e502f9f61b796e57c4bd0af2b4` — docs+test: round-11 slice in the README pitch, headline table and evaluation history (en/zh); round-15 anti-copy guards check slices committed before round 11 (the round-11 overlap is measured from its side) |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **at least one step failed** |

| Step | Command | Result | Seconds | Result line from the log |
|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 0 | 196 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | actionlint: no findings |
| pytest | `python -m pytest -q tests` | **FAIL** (exit 1) | 553 | 1 failed, 3720 passed, 70 skipped, 5 warnings in 551.02s (0:09:11) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | Done in 29ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 1 | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 3 | $ eslint . |
| fe-test | `pnpm test` | pass | 3 | Tests  161 passed (161) |
| fe-build | `pnpm build` | pass | 2 | ✓ built in 252ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 52917)` | pass | 23 | 2 passed (22.6s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 123 | result: dev task_success 1.0 (524 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 40 | result: ext task_success 1.0 (78 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 23 | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 552 | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 30 | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `8 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1)` | pass | 180 | 8 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 2 | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |

**Follow-up.** The one pytest failure above (`tests/test_alias_regression.py::test_alias_regression_at_the_agent_guard[100港元能换多少人民币]`) was a stale expectation: it passed only while the bad 人民币 → 宁德时代 alias rows existed, removed in round 15. Fixed in `a2d8f72` (the question may be out of scope, never refused as a Hong Kong line); at `a2d8f72` the full suite gives 3721 passed, 70 skipped, and the round-10 and round-11 verify scripts print OK. The other 16 steps above passed at `7cc9a4c` and `a2d8f72` changes only that test and its data row.
