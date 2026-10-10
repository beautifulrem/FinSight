# Pre-push check at `fb90fe4`

Run by `scripts/prepush.sh` (2026-10-10T22:44:33Z).

| Item | Value |
|---|---|
| Commit | `fb90fe40af87eecd240cce77b7463a56e73eaf58` — docs(readme): headline rows for the third (0.540 -> 0.900 after exposure) and fourth (0.740 blind -> 0.780) real-claims sets (en/zh) |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **all steps passed** |

| Step | Command | Result | Seconds | Result line from the log |
|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 0 | 190 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | actionlint: no findings |
| pytest | `python -m pytest -q tests` | pass | 529 | 2932 passed, 68 skipped, 5 warnings in 527.82s (0:08:47) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 1 | Done in 23ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 1 | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 2 | $ eslint . |
| fe-test | `pnpm test` | pass | 3 | Tests  161 passed (161) |
| fe-build | `pnpm build` | pass | 2 | ✓ built in 248ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 59956)` | pass | 23 | 2 passed (22.7s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 119 | result: dev task_success 1.0 (510 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 28 | result: ext task_success 1.0 (44 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 23 | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 506 | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 29 | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `8 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1)` | pass | 180 | 8 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 2 | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |
