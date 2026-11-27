# Development & Local Verification

Languages: English | [中文](zh/development.md)

Use [`scripts/prepush.sh`](../scripts/prepush.sh) to mirror GitHub Actions CI locally before pushing code changes to `master`. See [CONTRIBUTING.md](../CONTRIBUTING.md) for environment setup and architecture boundaries.

## Pre-Push Mirror (`scripts/prepush.sh`)

```bash
scripts/prepush.sh                          # full local CI mirror
PYTHON=.venv/bin/python scripts/prepush.sh  # explicit virtualenv interpreter
SKIP=e2e,extras scripts/prepush.sh          # skip selected steps by name
```

| Step | Command | CI Job |
|---|---|---|
| `ruff` | `ruff check . && ruff format --check .` | `lint` |
| `actionlint` | `actionlint .github/workflows/*.yml` (if installed) | — |
| `pytest` | `python -m pytest -q tests` (seed-pinned baseline + shipped-default unit tests) | `tests` |
| `pytest-shipped` | `QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests` (extension + corrections on) | `tests-shipped` |
| `fe-*` & `dist-diff` | `pnpm install/typecheck/lint/test/build` in `frontend/` + clean `query_intelligence/web/dist` | `frontend` |
| `e2e` | `pnpm run e2e` (Playwright against offline server on a free port) | `tests` |
| `gate-dev-holdout`, `gate-ext`, `gate-shipped` | `python -m evaluation.agent_eval.gate` across v1 seed, extension, and shipped defaults | `agent-eval-gate` |
| `claim-holdout` | `python -m evaluation.claim_bench.run --set holdout` against CI floors | `agent-eval-gate` |
| `extras` | `verifier_stress`, `redteam`, `gate --extras-only` | `agent-eval-extras` |
| `verify-slices` | `python -m evaluation.agent_eval.verify_slices` (all held-out slice verifiers) | `heldout-verify` |
| `report-check` | `python -m evaluation.agent_eval.report --check` (`docs/agent-eval.md` + README markers) | `heldout-verify` |

Logs write to `outputs/prepush/<commit>/<step>.log` with a Markdown summary in `outputs/prepush/<commit>/summary.md` (committed summaries live in `docs/results/ci/`).

Checks run in CI only (or manually locally): full-history `gitleaks`, Postgres multi-replica tests (`QI_TEST_POSTGRES_DSN`), coverage floors (`agent ≥ 88%`, `api ≥ 90%`, `answer_guards ≥ 81%`), Docker build, and `deploy/k8s-smoke/smoke.sh`.

## CI time budgets

[`scripts/ci_budget.py`](../scripts/ci_budget.py) estimates each CI job's hosted-runner duration after every `prepush.sh` step from `.github/workflows/ci.yml`'s `timeout-minutes` and fails a step (`FAIL (CI time budget)`) when cumulative estimated job runtime exceeds `CI_BUDGET_FRACTION` (default `0.6`).

| Variable | Default | Purpose |
|---|---|---|
| `CI_BUDGET_FRACTION` | `0.6` | Max fraction of a job's `timeout-minutes` allowed (`0` or `off` disables failure) |
| `CI_SLOWDOWN` | `3` | Runner-to-local wall-clock multiplier for non-coverage steps |
| `CI_SLOWDOWN_STEPS` | `pytest=5` | Per-step multipliers (pytest adds branch coverage in CI) |
| `CI_BUDGET_BUSY_LOAD` | performance cores (or half CPUs) | Flags budget failures measured on an overloaded host as `"host busy"` |

Verified by `tests/test_ci_budget.py`, which checks that every `prepush.sh` step maps to an existing CI job running the same command.

## Testing Both Pinned Seed & Shipped Default Data

Evaluation sets are pinned to the v1 seed (`tests/conftest.py` sets `QI_OFFLINE_SNAPSHOT_EXT=false` and `QI_OFFLINE_SNAPSHOT_FIX=false`) for longitudinal comparability. Production behavior (both layers enabled) is verified separately by:

- `tests/test_shipped_default.py` and `tests/test_data_layers.py` (default settings, primary-source corrected figures, Dockerfile layer parity, `/ready` 503 on missing layers);
- `QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests` (`tests-shipped` CI job, deselecting the seed-pinned assertions listed in `tests/seed_data_tests.txt`);
- `python -m evaluation.agent_eval.gate --set ext --set shipped`;
- Playwright E2E tests (`frontend/playwright.config.ts`) running with both layers enabled.

## Generated Evaluation & README Regions

Headline figures in `README.md` and `README_CN.md` live inside `<!-- generated:<region> (python -m evaluation.agent_eval.readme) -->` markers rendered by [`evaluation/agent_eval/readme.py`](../evaluation/agent_eval/readme.py) from committed files in `evaluation/results/` and `docs/results/ci/`.

```bash
python -m evaluation.agent_eval.report          # regenerate docs/agent-eval.md and README regions
python -m evaluation.agent_eval.report --check  # fail if any region or citation is stale
python scripts/record_ci_run.py                 # record latest master CI run JSON via gh CLI
```
