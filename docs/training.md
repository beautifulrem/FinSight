# Training & Runtime Assets

Languages: English | [中文](zh/training.md)

FinSight trains the classical, explainable NLU and retrieval models described in *FinSight: An Evidence-First Financial Analysis Chatbot with Explainable Query Intelligence* ([DOI: `10.3233/FAIA261107`](https://ebooks.iospress.nl/doi/10.3233/FAIA261107)). **Training assets** and **runtime assets** are decoupled: retraining updates `.joblib` models without modifying `data/runtime/`.

## Directory Layout

| Path | Purpose | Version Control |
|---|---|---|
| `data/external/raw/` | Downloaded public datasets | Local only (git-ignored) |
| `data/training_assets/` | Standardized task JSONL files & `manifest.json` | Local only (git-ignored) |
| `data/runtime/` | Shipped entities, aliases, dual-listing/HK-only tables, and document corpus | Committed |
| `models/` | Shipped `.joblib` models verified against `models/manifest.json` SHA-256 digests | Committed |
| `reports/` | Local evaluation reports | Local only (git-ignored) |

## Dataset Sync & Training Pipeline

Supported public datasets are registered in `query_intelligence/external_data/registry.py` (CFLUE, FiQA/BEIR, FinFE, ChnSentiCorp, MSRA/CLUENER NER, TNEWS/THUCNews/SMP2017, BBT-FinCUGE, QReCC/RiSAWOZ, T2Ranking, FinCPRG, FIR-Bench, CSPR-D, plus curated boundary cases).

```bash
# 1. Sync public datasets (optionally restrict with QI_DATASET_ALLOWLIST=finfe,t2ranking)
QI_ENABLE_EXTERNAL_DATA=1 python -m scripts.sync_public_datasets

# 2. Build standardized manifest and task JSONL assets
python -m scripts.build_training_assets

# 3. Preflight check (writes data/training_assets/preflight_report.json)
python -m training.prepare_training_run data/training_assets/manifest.json models

# 4. Train all classical models (or run sync+build+train via training.sync_and_train)
python -m training.train_all data/training_assets/manifest.json
```

### Individual Model Trainers

| Model | Module | Artifact |
|---|---|---|
| Product type / Intent / Topic | `training.train_product_type`, `train_intent`, `train_topic` | `product_type.joblib`, `intent_ovr.joblib`, `topic_ovr.joblib` |
| Question style & reranker | `training.train_question_style`, `train_question_style_reranker` | `question_style.joblib`, `question_style_reranker.joblib` |
| User sentiment & OOS detector | `training.train_sentiment`, `train_out_of_scope_detector` | `sentiment.joblib`, `out_of_scope_detector.joblib` |
| Entity CRF & typo linker | `training.train_entity_crf`, `train_typo_linker` | `entity_crf.joblib`, `typo_linker.joblib` |
| Clarification & source-plan reranker | `training.train_clarification_gate`, `train_source_plan_reranker` | `clarification_gate.joblib`, `source_plan_reranker.joblib` |
| Document LTR ranker | `training.train_ranker` | `ranker.joblib` |

## Materializing Runtime Assets

```bash
# Refresh entity master & alias tables (--no-akshare uses local seeds only)
python -m scripts.materialize_runtime_entity_assets --output-dir data/runtime

# Refresh local retrieval corpus
python -m scripts.materialize_runtime_document_assets \
  --corpus-path data/training_assets/retrieval_corpus.jsonl \
  --output-path data/runtime/documents.jsonl --max-documents 50000
```

## Evaluation & Verification

```bash
python -m pytest -q tests/test_query_intelligence.py tests/test_ml_upgrades.py tests/test_real_integrations.py
python -m scripts.run_test_suite
python -m scripts.evaluate_query_intelligence
QI_USE_LIVE_MARKET=1 QI_USE_LIVE_NEWS=1 python -m scripts.verify_live_sources --query "你觉得中国平安怎么样？" --debug
```

**Evaluation caveats**: `scripts/evaluate_query_intelligence.py` evaluates a 10,000-query mix of public splits, template boundary queries, and must-pass cases; retrieval Recall@10/MRR@10/NDCG@10 measure re-ranking against 40 sampled negatives per query rather than full-corpus recall; clarification recall (`0.75`) remains below the `0.90` target.

## Release Checklist

- Ensure `data/runtime/` and `models/*.joblib` (+ `models/manifest.json`) work cleanly without `data/external/` or `data/training_assets/`.
- Confirm `git status --short` shows no staged `.env`, tokens, `outputs/`, `reports/`, or local caches.
