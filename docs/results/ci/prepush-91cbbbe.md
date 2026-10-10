# Pre-push check at `91cbbbe`

Run by `scripts/prepush.sh` (2026-10-10T20:41:18Z).

| Item | Value |
|---|---|
| Commit | `91cbbbecf794089c95249ec7b721e5ef231e4ded` — test(ui): the web UI's copies of the server's next-question and disclaimer templates (K17 localize.ts) must match followups.py / answer_guards.py / chat/language.py |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **all steps passed** |

| Step | Command | Result | Seconds | Result line from the log |
|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 0 | 186 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | actionlint: no findings |
| pytest | `python -m pytest -q tests` | pass | 699 | 2830 passed, 70 skipped, 5 warnings in 695.96s (0:11:35) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | Done in 40ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 4 | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 5 | $ eslint . |
| fe-test | `pnpm test` | pass | 7 | Tests  161 passed (161) |
| fe-build | `pnpm build` | pass | 4 | ✓ built in 333ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 61417)` | pass | 50 | 2 passed (49.1s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 276 | result: dev task_success 1.0 (510 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 76 | result: ext task_success 1.0 (44 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 56 | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 991 | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 57 | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `8 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1)` | pass | 288 | 8 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 3 | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |
