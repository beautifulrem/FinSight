# Pre-push check at `de53f2d`

Run by `scripts/prepush.sh` (2026-10-10T21:47:07Z).

| Item | Value |
|---|---|
| Commit | `de53f2d90acb15f0a4fc239ce72a4eeaa2988381` — test(eval): the round-13 overlap test leaves the round-15 safety ext tasks (ext_r15s_*) to the round-15 test |
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
| actionlint | `actionlint .github/workflows/*.yml` | pass | 1 | actionlint: no findings |
| pytest | `python -m pytest -q tests` | pass | 730 | 3577 passed, 68 skipped, 5 warnings in 727.71s (0:12:07) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | Done in 30ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 2 | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 3 | $ eslint . |
| fe-test | `pnpm test` | pass | 6 | Tests  161 passed (161) |
| fe-build | `pnpm build` | pass | 3 | ✓ built in 270ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 60832)` | pass | 30 | 2 passed (28.8s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 159 | result: dev task_success 1.0 (510 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 42 | result: ext task_success 1.0 (54 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 28 | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 824 | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 43 | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `8 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1)` | pass | 278 | 8 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 3 | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |
