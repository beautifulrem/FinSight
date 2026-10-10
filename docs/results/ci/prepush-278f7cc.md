# Pre-push check at `278f7cc`

Run by `scripts/prepush.sh` (2026-10-10T16:58:29Z).

| Item | Value |
|---|---|
| Commit | `278f7cc198e50fd56f63e564a6ceea2ac3aef822` — docs(results): kind smoke with the round-14 assertions passes (data layers loaded, 五粮液 ROE（加权） 6.89% / 营业收入 405.29 亿元 asserted) |
| Working tree clean at start (tracked files) | yes |
| Working tree clean at end (tracked files) | yes |
| Host | Darwin arm64 |
| Python | Python 3.13.13 |
| Node / pnpm | v26.10.0 / 12.6.0 |
| Skipped by request | none |
| Result | **all steps passed** |

| Step | Command | Result | Seconds | Result line from the log |
|---|---|---|---|---|
| ruff | `ruff check . && ruff format --check .` | pass | 1 | 176 files already formatted |
| actionlint | `actionlint .github/workflows/*.yml` | pass | 0 | actionlint: no findings |
| pytest | `python -m pytest -q tests` | pass | 700 | 2508 passed, 70 skipped, 5 warnings in 697.11s (0:11:37) |
| fe-install | `pnpm install --frozen-lockfile` | pass | 0 | Done in 40ms using pnpm v12.6.0 |
| fe-typecheck | `pnpm typecheck` | pass | 4 | $ tsc -b --noEmit |
| fe-lint | `pnpm lint` | pass | 4 | $ eslint . |
| fe-test | `pnpm test` | pass | 8 | Tests  152 passed (152) |
| fe-build | `pnpm build` | pass | 4 | ✓ built in 368ms |
| dist-diff | `git status --porcelain query_intelligence/web/dist (empty)` | pass | 0 | query_intelligence/web/dist matches the source |
| e2e | `pnpm run e2e (port 60953)` | pass | 48 | 2 passed (47.0s) |
| gate-dev-holdout | `python -m evaluation.agent_eval.gate --set dev --set holdout` | pass | 222 | result: dev task_success 1.0 (473 tasks); holdout task_success 0.9434 (53 tasks) |
| gate-ext | `python -m evaluation.agent_eval.gate --set ext` | pass | 42 | result: ext task_success 1.0 (24 tasks) |
| gate-shipped | `python -m evaluation.agent_eval.gate --set shipped` | pass | 44 | result: shipped task_success 1.0 (21 tasks) |
| extras | `verifier_stress; redteam; gate --extras-only` | pass | 724 | "gate": "passed", |
| claim-holdout | `claim_bench.run --set holdout (floors 0.978 / 0.98)` | pass | 50 | verdict accuracy 1.0 [1.0, 1.0] over 47 claims; check accuracy 1.0 [1.0, 1.0] over 54 checks; comparator accuracy 1.0 |
| verify-slices | `8 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1)` | pass | 286 | 8 verify scripts passed |
| report-check | `python -m evaluation.agent_eval.report --check` | pass | 3 | docs/agent-eval.md is up to date; every result the READMEs cite exists and is rendered |

Wall clock 35 min 40 s on an Apple-silicon laptop (`time scripts/prepush.sh`); the longest steps are the test suite (700 s) and the verifier stress + red team (724 s). The commit that adds this file changes docs and one comment line in `scripts/prepush.sh` only.
