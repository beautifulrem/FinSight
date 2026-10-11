# Pre-push check at `dc6a9b6`

Run by `scripts/prepush.sh` (2026-10-11T03:57:49Z).

| Item | Value |
|---|---|
| Commit | `dc6a9b6c88def0fc8ed3b8f9f16a5456247ee90d` — docs: round-16 frame rules and the operation classifier's independent evaluation in agent.md; L1-L3, L9, L10, L12-L14, L17 rows and K1/K2/K4 corrected to partly fixed in known-issues; the independent wording set, the round-16 classifier and the r8-r11 reruns in the evaluation history (en/zh) |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **all steps passed** |

| Step | Command | Result | Seconds | Result line from the log |
|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 0 | 198 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | actionlint: no findings |
| pytest | `python -m pytest -q tests` | pass | 720 | 3772 passed, 70 skipped, 5 warnings in 718.30s (0:11:58) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | Done in 31ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 3 | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 2 | $ eslint . |
| fe-test | `pnpm test` | pass | 6 | Tests  164 passed (164) |
| fe-build | `pnpm build` | pass | 3 | ✓ built in 270ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 64804)` | pass | 28 | 2 passed (27.7s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 177 | result: dev task_success 1.0 (538 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 55 | result: ext task_success 1.0 (90 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 29 | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 714 | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 36 | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `8 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1)` | pass | 231 | 8 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 2 | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |
