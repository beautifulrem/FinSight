# Development: checks before a push

[中文](zh/development.md)

## The rule

**Run `scripts/prepush.sh` at the commit you are about to push to `master`, and push only when it passes.** A merge
commit counts: the merged code has to pass the browser e2e tests too, not only each branch.

Why: in round 13 the corrections layer became the product default, the e2e specs still asserted the v1 seed values,
and the merge reached `master` without an e2e run, so `master` CI was red at `7f4f06c` (run 38061714150) until the
spec fix `f8f5546` (run 38064588334, green). The script runs locally what CI runs, so that cannot happen silently
again (round-10 review J1).

## What it runs

```bash
scripts/prepush.sh                          # every step: 35 min 40 s on an Apple-silicon laptop at 278f7cc
PYTHON=.venv/bin/python scripts/prepush.sh  # the interpreter with requirements-dev.txt installed
SKIP=e2e,extras scripts/prepush.sh          # skip steps by name; the summary lists them as skipped
CI_BUDGET_FRACTION=0.5 scripts/prepush.sh   # CI time budget share (default 0.6; off disables the failure)
```

| Step | Command | CI job |
|---|---|---|
| `ruff` | `ruff check .` and `ruff format --check .` | lint |
| `actionlint` | `actionlint .github/workflows/*.yml` (skipped when actionlint is not installed) | – |
| `pytest` | `python -m pytest -q tests` (includes `tests/test_shipped_default.py`, below) | tests |
| `fe-install`, `fe-typecheck`, `fe-lint`, `fe-test`, `fe-build` | `pnpm install --frozen-lockfile`, `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm build` in `frontend/` | frontend |
| `dist-diff` | `git status --porcelain query_intelligence/web/dist` must be empty (the committed build matches the source) | frontend |
| `e2e` | `pnpm run e2e`: Playwright in real Chrome against the offline server | tests |
| `gate-dev-holdout` | `python -m evaluation.agent_eval.gate --set dev --set holdout` (pinned to the v1 seed) | agent-eval-gate |
| `gate-ext` | `gate --set ext` (snapshot extension on, corrections off) | agent-eval-gate |
| `gate-shipped` | `gate --set shipped` (extension **and** corrections layer on: what the product serves) | agent-eval-gate |
| `extras` | `verifier_stress`, `redteam`, `gate --extras-only` | agent-eval-extras |
| `claim-holdout` | `claim_bench.run --set holdout` with the CI floors | agent-eval-gate |
| `verify-slices` | the ten held-out verify scripts (r9 with `QI_EVAL_SNAPSHOT_EXT=1`, r10 and r11 also with `QI_EVAL_SNAPSHOT_FIX=1`) | heldout-verify |
| `report-check` | `python -m evaluation.agent_eval.report --check` | heldout-verify |

Every step runs even when an earlier one fails; the exit status is 0 only when all steps that ran passed. Live sources
and traces are off, as in CI. The e2e server is started on a free port chosen by the script: outside CI Playwright
reuses any server already listening on its port, which could be another checkout's.

Not covered locally: the gitleaks history scan, the Postgres-backed checkpointer test (it needs
`QI_TEST_POSTGRES_DSN`), the coverage floors, the Docker image and the kind cluster. For the last two:

```bash
docker build -f docker/Dockerfile -t finsight:$(git rev-parse --short=7 HEAD) .
deploy/k8s-smoke/smoke.sh        # kind + kubectl; asserts /ready, the three data layers and a corrected figure
```

## CI time budgets

A step can pass locally and still break CI by time. In run 38095861905 (`bb99360`) every step passed but the
`agent-eval-gate` job was cancelled at its 45-minute `timeout-minutes`: the round-15 slices had grown its last step,
and the job had already used 43 min 51 s of 45 at `4ad3380`. The pre-push check knew only pass or fail, so it could
not see this (round-12 review L16).

Since round 16, after each step `scripts/prepush.sh` calls [`scripts/ci_budget.py`](../scripts/ci_budget.py):

1. **Job budgets.** Each job's `timeout-minutes` is read from `.github/workflows/ci.yml`. A job without one gets
   GitHub's default of 360.
2. **Step to job.** `STEP_JOBS` names the CI job of every pre-push step (the table above). A test checks that every
   step of the script is mapped, that every mapped job exists, and that the job really runs the step's command.
3. **Estimate.** A job's CI time is estimated as the runner's setup seconds plus, for every step of that job run so
   far, local seconds × a slowdown factor. The factors are hosted-runner seconds per laptop second, measured on the
   same code (CI run 38077019773 against the pre-push run at `03dd10d`, and 38095861905 against `7cc9a4c`). The gate,
   claim, verify and e2e steps are 2.4–3.1× slower on the runner, so the default factor is 3. pytest is 4.2–4.8×
   slower, because CI adds branch coverage, so its factor is 5.
4. **Fail.** A step fails with `FAIL (CI time budget)` when its job's estimate passes `CI_BUDGET_FRACTION` (default
   0.6) of the job's `timeout-minutes`. The estimate is cumulative, so the step that crosses the line fails even
   when it is short on its own. Several steps that are each fast but too slow together, as in 38095861905, are
   caught this way.

The summary gets a column with each step's job estimate and a table per job. That table lists the timeout, the
steps, the local seconds, the estimated CI minutes and the share of the budget.

| Variable | Default | Meaning |
|---|---|---|
| `CI_BUDGET_FRACTION` | `0.6` | share of a job's `timeout-minutes` its estimate may use; `off` or `0` only reports |
| `CI_SLOWDOWN` | `3` | runner seconds per local second for every step without its own factor |
| `CI_SLOWDOWN_STEPS` | `pytest=5` (built in) | per-step factors, e.g. `pytest=5,e2e=2.5`; recalibrate on a host other than an Apple-silicon laptop |

With the local durations of `7cc9a4c`, the layout that was cancelled is estimated at 48.5 min of 45 for
`agent-eval-gate` (108%) and 49.7 of 60 for `tests` (83%), so both would have failed before the push. To keep
margin, round 16 moved the verifier stress test and the red team into their own job, `agent-eval-extras` (60 min).
It also raised the `tests` timeout to 90 min. The estimates are now: tests 49.7 / 90 (55%), agent-eval-gate
11.8 / 45 (26%), agent-eval-extras 28.6 / 60 (48%), heldout-verify 10.1 / 45 (22%). `tests/test_ci_budget.py`
checks both layouts. Skipped steps are not counted, so with `SKIP` the estimate is a lower bound.

The last run before the change, 38098598453 at `c39008b` (heldout-verify already split off), was green but used
41.8 of 45 min in `agent-eval-gate` (the verifier and red-team step alone 30.2 min) and 47.5 of 60 in `tests`.

## Output

Logs go to `outputs/prepush/<commit>/<step>.log` and the summary to `outputs/prepush/<commit>/summary.md` (or
`SUMMARY=<path>`): commit, whether the tree was clean, Python / Node / pnpm versions, and per step the command, result,
seconds, its CI job's estimated time so far against the job's budget and the result line of its log (pytest's pass count, the gate's task success, ...), then the per-job budget table. Committed summaries are in
[`docs/results/ci/`](results/ci/), named after the commit they ran at: at `278f7cc` (round 14, on the r14-ops branch before the round-14 merge, so dev 473 / ext 24) all 17 steps passed
([`prepush-278f7cc.md`](results/ci/prepush-278f7cc.md): 2508 tests passed / 70 skipped, vitest 152, e2e 2, gates dev 1.000,
holdout 0.9434, ext 1.000, shipped 1.000).

## The shipped default is tested, not only the pinned seed

The evaluation sets and most tests are pinned to the v1 seed (`tests/conftest.py` sets
`QI_OFFLINE_SNAPSHOT_EXT=false` and `QI_OFFLINE_SNAPSHOT_FIX=false`; the harness needs `QI_EVAL_SNAPSHOT_EXT=1` /
`QI_EVAL_SNAPSHOT_FIX=1` to opt in), because their labels were written against it. What a deployment serves is
covered by:

* `tests/test_shipped_default.py`: removes both variables, builds the service from the `Settings` defaults and asks
  the API for figures only the layers serve (茅台 P/E 21.44, 五粮液 ROE（加权） 6.89% and 营业收入 405.29 亿元,
  中国平安 58.03, 宁德时代 ROE（期末净资产） 21.42%, the 25.64-point gap), with the seed values forbidden. Flipping
  the corrections default off makes all 8 tests fail.
* the gate set `shipped` (`tasks/agent_eval_shipped_v1.jsonl`, 21 tasks / 22 turns, built by
  `evaluation/agent_eval/build_shipped_tasks.py`, baseline `evaluation/results/gate-shipped.json`): both layers on,
  no replay, expected values from `data/snapshot/structured_data_fix.json`. 21/21 with both layers on, 2/21 with
  the corrections layer off.
* `tests/test_data_layers.py`: the Dockerfile copies every layer file the loader reads, a switched-on layer that is
  missing logs a warning and makes `GET /ready` 503.
