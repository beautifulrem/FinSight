# Pre-push check at `4d239d5`

Run by `scripts/prepush.sh` (2026-10-11T02:42:42Z).

| Item | Value |
|---|---|
| Commit | `4d239d564565a398c93c0c860d251de9a451653b` — fix(router): "市值是否超过…" is a present-tense comparison, not a market-value forecast (round-16 L8 follow-up) |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **all steps passed** |

| Step | Command | Result | Seconds | Result line from the log |
|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 0 | 200 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | actionlint: no findings |
| pytest | `python -m pytest -q tests` | pass | 1243 | 4172 passed, 68 skipped, 5 warnings in 1239.52s (0:20:39) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | Done in 27ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 3 | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 4 | $ eslint . |
| fe-test | `pnpm test` | pass | 7 | Tests  161 passed (161) |
| fe-build | `pnpm build` | pass | 3 | ✓ built in 370ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 58659)` | pass | 41 | 2 passed (40.2s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 245 | result: dev task_success 1.0 (524 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 87 | result: ext task_success 1.0 (87 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 47 | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 1179 | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 62 | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `8 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1)` | pass | 424 | 8 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 4 | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |
