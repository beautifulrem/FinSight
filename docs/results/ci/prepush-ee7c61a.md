# Pre-push check at `ee7c61a`

Run by `scripts/prepush.sh` (2026-10-10T22:10:16Z).

| Item | Value |
|---|---|
| Commit | `ee7c61a6486ac8f4a52527b774b152923522677e` — docs: round-15 frame rules (total vs one, mean, two operations per message, unresolved list names, dated closes, purchase lots, holdings) and the operation classifier in agent.md; K1-K4/K7-K10/K14 rows in known-issues; round-15 slice, verifier and classifier rows in the evaluation history; README gate row dev 524 / ext 68 (en/zh) |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **all steps passed** |

| Step | Command | Result | Seconds | Result line from the log |
|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 0 | 188 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | actionlint: no findings |
| pytest | `python -m pytest -q tests` | pass | 736 | 2876 passed, 70 skipped, 5 warnings in 733.51s (0:12:13) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | Done in 45ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 2 | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 2 | $ eslint . |
| fe-test | `pnpm test` | pass | 5 | Tests  161 passed (161) |
| fe-build | `pnpm build` | pass | 3 | ✓ built in 297ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 50486)` | pass | 25 | 2 passed (25.1s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 150 | result: dev task_success 1.0 (524 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 63 | result: ext task_success 1.0 (68 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 42 | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 714 | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 57 | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `8 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1)` | pass | 204 | 8 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 1 | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |
