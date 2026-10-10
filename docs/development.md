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
| `extras` | `verifier_stress`, `redteam`, `gate --extras-only` | agent-eval-gate |
| `claim-holdout` | `claim_bench.run --set holdout` with the CI floors | agent-eval-gate |
| `verify-slices` | the eight held-out verify scripts (r9 with `QI_EVAL_SNAPSHOT_EXT=1`) | agent-eval-gate |
| `report-check` | `python -m evaluation.agent_eval.report --check` | agent-eval-gate |

Every step runs even when an earlier one fails; the exit status is 0 only when all steps that ran passed. Live sources
and traces are off, as in CI. The e2e server is started on a free port chosen by the script: outside CI Playwright
reuses any server already listening on its port, which could be another checkout's.

Not covered locally: the gitleaks history scan, the Postgres-backed checkpointer test (it needs
`QI_TEST_POSTGRES_DSN`), the coverage floors, the Docker image and the kind cluster. For the last two:

```bash
docker build -f docker/Dockerfile -t finsight:$(git rev-parse --short=7 HEAD) .
deploy/k8s-smoke/smoke.sh        # kind + kubectl; asserts /ready, the three data layers and a corrected figure
```

## Output

Logs go to `outputs/prepush/<commit>/<step>.log` and the summary to `outputs/prepush/<commit>/summary.md` (or
`SUMMARY=<path>`): commit, whether the tree was clean, Python / Node / pnpm versions, and per step the command, result,
seconds and the result line of its log (pytest's pass count, the gate's task success, ...). Committed summaries are in
[`docs/results/ci/`](results/ci/), named after the commit they ran at: at `278f7cc` (round 14) all 17 steps passed
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
