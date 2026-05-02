# Repository Guidelines

## Project Mission & Boundaries

ARIN Query Intelligence turns a user’s financial question into evidence artifacts that downstream systems can analyze and answer from. The owned core is NLU plus Retrieval: `nlu_result` captures normalized query, product type, intents, topics, entities, missing slots, risk flags, and source plan; `retrieval_result` captures executed sources, documents, structured data, coverage, warnings, ranking debug traces, and `analysis_summary`.

Query Intelligence does not write final natural-language investment answers, does not produce deterministic buy/sell decisions, and does not own frontend, account, or scheduling systems. Downstream consumers such as `sentiment/` and `scripts/llm_response.py` must consume upstream artifacts instead of re-inferring intent, target entities, or source plan.

## Project Structure & Module Organization

Core code is in `query_intelligence/`. `nlu/` handles normalization, entity resolution, classifiers, slots, clarification, out-of-scope detection, and source planning. `retrieval/` handles query building, document/structured retrieval, features, ranking, deduplication, packaging, and `MarketAnalyzer`. `api/` exposes FastAPI endpoints, `contracts.py` defines Pydantic schemas, and `data_loader.py` loads runtime assets.

Other important paths: `sentiment/` for downstream document sentiment, `scripts/` for operations and LLM response handoff, `training/` for model training, `tests/` for pytest coverage, `schemas/` for JSON schemas, `sql/` for DDL, `data/runtime/` for shipped runtime assets, and `models/` for shipped model artifacts.

## Architecture Rules

NLU and Retrieval must use classical, explainable ML as their main path: rules, dictionaries, TF-IDF, linear classifiers, CRF, tree models, Learning to Rank, and PostgreSQL full-text search. Do not use BERT, Transformer, LLM, or vector retrieval as the NLU/Retrieval backbone.

`sentiment/` and `scripts/llm_response.py` are downstream exceptions. They may use `torch`, `transformers`, FinBERT, or local generation models, but only over compact evidence already produced by Query Intelligence.

Every stage must remain explainable: NLU matched rules, classifier top features, entity match type, retrieval rank features, and source-planning reasons should be inspectable in outputs or debug traces.

## Upstream Additions To Know

`analysis_summary` is part of `RetrievalResult`. It is built by `query_intelligence/retrieval/market_analyzer.py` and summarizes market, fundamental, macro, and data-readiness signals. It may include technical indicators such as returns, MA5/MA20, RSI(14), MACD, volatility, Bollinger bands, and trend signal. It is evidence summarization, not investment advice.

`sentiment/` preprocesses retrieved documents, detects language, filters by entity/source type, runs per-sentence FinBERT inference, and emits document-level `SentimentItem` records. Relevant tests include `tests/test_sentiment_preprocessor.py`, `tests/test_sentiment_classifier.py`, and `tests/test_sentiment_pipeline.py`.

`scripts/llm_response.py` generates frontend-ready `answer_generation` and `next_question_prediction` JSON from compact evidence. It must strip debug/raw noise, validate JSON shape, cite `evidence_id`, include a risk disclaimer, and never invent missing market, fundamental, macro, news, sentiment, or statistical facts.

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a manual query:

```bash
python manual_test/run_manual_query.py --query "你觉得中国平安怎么样？"
```

Start the API:

```bash
uvicorn query_intelligence.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Train from an existing manifest:

```bash
python -m training.train_all data/training_assets/manifest.json
```

Run grouped tests:

```bash
python -m scripts.run_test_suite
```

## Coding Style & Naming Conventions

Use Python 3.13-compatible code, 4-space indentation, type hints, and focused functions. Module files, variables, and test functions use `snake_case`; classes use `PascalCase`. Keep public field names stable. If contracts change, update `query_intelligence/contracts.py`, JSON schemas, tests, and both README files together.

## Testing Guidelines

Tests use `pytest`; files are `tests/test_*.py`, functions are `test_*`. Use targeted suites: `tests/test_query_intelligence.py` for core NLU/Retrieval, `tests/test_analysis_summary.py` and `tests/test_market_analyzer.py` for structured analysis, sentiment tests for `sentiment/`, and `tests/test_llm_response.py` for answer handoff. Run `python -m pytest -q <paths>` before submitting related changes.

## Runtime Configuration & Security

Important environment variables include `TUSHARE_TOKEN`, `QI_POSTGRES_DSN`, `QI_USE_LIVE_MARKET`, `QI_USE_LIVE_NEWS`, `QI_USE_LIVE_ANNOUNCEMENT`, `QI_USE_LIVE_MACRO`, `QI_MODELS_DIR`, `QI_API_OUTPUT_DIR`, `QI_TRAINING_MANIFEST`, `QI_ENTITY_MASTER_PATH`, `QI_ALIAS_TABLE_PATH`, and `QI_DOCUMENTS_PATH`. LLM response also uses `LLM_MODELS_DIR`, `QI_LLM_MODELS_DIR`, `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, and `TRANSFORMERS_CACHE`.

Never commit `.env`, tokens, generated outputs, public dataset caches, reports, or local scratch files. Keep `data/external/`, `data/training_assets/`, `outputs/`, `reports/`, `manual_test/output/`, caches, and `__pycache__/` local. Runtime assets in `data/runtime/` and model artifacts in `models/*.joblib` are intentionally shipped for clone usability.

## Commit & Pull Request Guidelines

History uses short imperative messages, often `feat:`, `fix:`, `docs:`, or scoped forms such as `fix(retrieval): ...`. Keep commits narrow. Pull requests should state changed modules, behavior impact, test commands/results, schema or artifact changes, and required configuration.

## Review Checklist

Before finishing, check `git status --short`, run narrow tests for touched modules, and verify generated data was not staged. If runtime/training assets changed, confirm the repo still works without local `data/external/` or `data/training_assets/`.
