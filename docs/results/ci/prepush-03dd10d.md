# Pre-push check at `03dd10d`

Run by `scripts/prepush.sh` (2026-10-10T18:44:25Z).

| Item | Value |
|---|---|
| Commit | `03dd10dcf9dc72e8c5e0f2eb568bdff090e415e1` — docs(development): say the committed pre-push summary at 278f7cc ran on the r14-ops branch before the round-14 merge (dev 473 / ext 24) (en/zh) |
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
| pytest | `python -m pytest -q tests` | pass | 481 | 2812 passed, 70 skipped, 5 warnings in 479.07s (0:07:59) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | Done in 139ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 2 | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 2 | $ eslint . |
| fe-test | `pnpm test` | pass | 3 | Tests  154 passed (154) |
| fe-build | `pnpm build` | pass | 2 | ✓ built in 204ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 53973)` | pass | 23 | 2 passed (22.5s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 118 | result: dev task_success 1.0 (510 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 30 | result: ext task_success 1.0 (44 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 22 | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 501 | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 28 | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `8 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1)` | pass | 180 | 8 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 1 | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |
