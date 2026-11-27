# Contributing

Thank you for your interest in contributing to FinSight.

## Local Setup

Requires **Python 3.13+** and (for frontend changes) **Node.js 24+** with **pnpm**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

For the React 19 UI in `frontend/`:

```bash
cd frontend && pnpm install --frozen-lockfile
```

## Verification Commands

| Task | Command |
|---|---|
| Format & lint | `ruff check . && ruff format --check .` |
| Offline unit & integration tests | `python -m pytest -q tests` |
| Shipped default data tests | `QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests` |
| Frontend check & build | `cd frontend && pnpm typecheck && pnpm lint && pnpm test && pnpm build` |
| Browser E2E (Playwright) | `cd frontend && FINSIGHT_PYTHON=../.venv/bin/python pnpm run e2e` |
| Offline evaluation gates | `python -m evaluation.agent_eval.gate` |
| Docs & generated README check | `python -m evaluation.agent_eval.report --check` |
| Full pre-push mirror | `scripts/prepush.sh` |

See [docs/development.md](docs/development.md) for CI time-budget calibration and test-suite layout.

## Architecture Boundaries

- **Classical Query Intelligence core**: `query_intelligence/nlu/` and `query_intelligence/retrieval/` must remain classical, explainable ML (rules, lexicons, TF-IDF, linear/tree classifiers, CRF, LTR, PostgreSQL FTS). Do not replace NLU or retrieval with an LLM or vector-DB backbone.
- **Agent layer**: `query_intelligence/agent/` orchestrates typed, read-only tools around the core. Classical NLU stays the router and pre-LLM guard; deterministic fallback must keep working without network or API keys.
- **Contracts & schemas**: Keep Pydantic models in `query_intelligence/contracts.py`, exported JSON schemas in `schemas/`, tests, and documentation in sync.
- **Frontend builds**: Any change under `frontend/src/` must rebuild and commit `query_intelligence/web/dist` (`pnpm build`).
- **Secrets & generated artifacts**: Never commit `.env`, API keys, `outputs/`, `reports/`, `data/external/`, or `data/training_assets/`.

## Pull Request Checklist

- Keep commits narrow with imperative prefixes (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Run targeted tests (and `scripts/prepush.sh` for code changes) before opening a PR.
- If evaluation result JSONs change, regenerate `docs/agent-eval.md` and README regions with `python -m evaluation.agent_eval.report`.
- Summarize changed modules, behavior impact, test commands/results, and any schema or configuration updates in the PR description.
