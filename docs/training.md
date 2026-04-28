# Training and Runtime Assets

Languages: English | [中文](zh/training.md)

The FinSight backend trains classical, explainable models for NLU and retrieval. Training assets and runtime assets are separate:

- Training assets improve classifiers, rankers, CRF, typo linking, source planning, and out-of-scope behavior.
- Runtime assets power production entity resolution, alias matching, local document recall, and fallback structured data.

Retraining models does not automatically update runtime entities, aliases, or documents.

## Data Layout

| Path | Purpose | Commit policy |
|---|---|---|
| `data/external/raw/` | Downloaded public datasets | local only |
| `data/training_assets/` | Standardized manifest and task JSONL files | local only |
| `data/runtime/` | Shipped runtime entity, alias, and document assets | committed |
| `models/` | Shipped trained model artifacts | committed |
| `reports/` | Evaluation reports | local only |

## Public Dataset Registry

The dataset registry is in `query_intelligence/external_data/registry.py`. Adapters convert raw sources into task-specific assets through `query_intelligence/external_data/`.

Common source families:

| Source | Type | Main usage |
|---|---|---|
| CFLUE | GitHub | Finance classification and QA. |
| FiQA / BEIR | HuggingFace | Retrieval corpus, qrels, learning-to-rank. |
| FinFE | HuggingFace | Financial sentiment. |
| ChnSentiCorp | HuggingFace | Chinese sentiment. |
| Financial news sentiment | GitHub | Chinese financial sentiment. |
| MSRA NER / People's Daily NER / CLUENER | HuggingFace/GitHub | Entity boundary and CRF training. |
| TNEWS / THUCNews / SMP2017 | HuggingFace/direct/GitHub | Product type, intent, topic, OOD generalization. |
| BBT-FinCUGE, Mxode finance, BAAI finance instruction | GitHub/HuggingFace | Finance intent and topic supervision. |
| QReCC / RiSAWOZ | HuggingFace | Multi-turn and clarification behavior. |
| T2Ranking / FinCPRG / FIR-Bench / CSPR-D | HuggingFace/GitHub | Retrieval, reports, announcements, qrels, source/ranker supervision. |
| Curated boundary cases | local generated | Regression coverage for hard boundaries. |

## Sync and Build

Sync all enabled public datasets:

```bash
QI_ENABLE_EXTERNAL_DATA=1 python -m scripts.sync_public_datasets
```

Sync selected datasets:

```bash
QI_ENABLE_EXTERNAL_DATA=1 QI_DATASET_ALLOWLIST=finfe,t2ranking,fir_bench_reports \
python -m scripts.sync_public_datasets
```

Build standardized training assets:

```bash
python -m scripts.build_training_assets
```

End-to-end sync, build, preflight, and train:

```bash
QI_ENABLE_EXTERNAL_DATA=1 python -m training.sync_and_train
```

Train from an existing manifest:

```bash
python -m training.train_all data/training_assets/manifest.json
```

Use `training.sync_and_train` when the assets must be regenerated. Use `training.train_all <manifest>` when the manifest already exists and only model training is needed.

## Training Preflight

```bash
python -m training.prepare_training_run data/training_assets/manifest.json models
```

The preflight report is written to:

```text
data/training_assets/preflight_report.json
```

Check it before replacing shipped model artifacts.

## Training Scripts

| Model | Command | Output |
|---|---|---|
| Product type classifier | `python -m training.train_product_type data/training_assets/manifest.json` | `models/product_type.joblib` |
| Intent multi-label classifier | `python -m training.train_intent data/training_assets/manifest.json` | `models/intent_ovr.joblib` |
| Topic multi-label classifier | `python -m training.train_topic data/training_assets/manifest.json` | `models/topic_ovr.joblib` |
| Question style classifier | `python -m training.train_question_style data/training_assets/manifest.json` | `models/question_style.joblib` |
| User sentiment classifier | `python -m training.train_sentiment data/training_assets/manifest.json` | `models/sentiment.joblib` |
| Entity boundary CRF | `python -m training.train_entity_crf data/training_assets/manifest.json` | `models/entity_crf.joblib` |
| Clarification gate | `python -m training.train_clarification_gate data/training_assets/manifest.json` | `models/clarification_gate.joblib` |
| Question style reranker | `python -m training.train_question_style_reranker data/training_assets/manifest.json` | `models/question_style_reranker.joblib` |
| Source plan reranker | `python -m training.train_source_plan_reranker data/training_assets/manifest.json` | `models/source_plan_reranker.joblib` |
| Out-of-scope detector | `python -m training.train_out_of_scope_detector data/training_assets/manifest.json` | `models/out_of_scope_detector.joblib` |
| Document ranker | `python -m training.train_ranker data/training_assets/manifest.json` | `models/ranker.joblib` |
| Typo linker | `python -m training.train_typo_linker data/training_assets/manifest.json` | `models/typo_linker.joblib` |
| Train all | `python -m training.train_all data/training_assets/manifest.json` | all model artifacts |

## Runtime Assets

Materialize entity and alias assets after training:

```bash
python -m scripts.materialize_runtime_entity_assets
```

Common options:

```bash
python -m scripts.materialize_runtime_entity_assets \
  --seed-dir data \
  --training-assets-dir data/training_assets \
  --output-dir data/runtime \
  --max-training-pairs 80000
```

Use local assets only:

```bash
python -m scripts.materialize_runtime_entity_assets --no-akshare
```

Outputs:

```text
data/runtime/entity_master.csv
data/runtime/alias_table.csv
```

Materialize runtime documents:

```bash
python -m scripts.materialize_runtime_document_assets
```

Common options:

```bash
python -m scripts.materialize_runtime_document_assets \
  --corpus-path data/training_assets/retrieval_corpus.jsonl \
  --output-path data/runtime/documents.jsonl \
  --max-documents 50000
```

Output:

```text
data/runtime/documents.jsonl
```

## Evaluation

Recommended handoff sequence:

```bash
python -m pytest tests/test_query_intelligence.py tests/test_real_integrations.py -q
python -m pytest tests/test_manual_query.py tests/test_manual_test_runner.py -q
python -m pytest tests/test_ml_upgrades.py -q
python -m scripts.run_test_suite
python -m scripts.evaluate_query_intelligence
```

Full-stack evaluation covers Chinese/English, finance, non-finance, adversarial, and boundary queries. It reports finance-domain recall, OOD rejection, product type accuracy, question style accuracy, intent/topic F1, clarification recall, source-plan quality, retrieval recall@10, MRR@10, NDCG@10, and OOD retrieval abstention.

## Live Source Verification

```bash
QI_USE_LIVE_MARKET=1 QI_USE_LIVE_NEWS=1 QI_USE_LIVE_ANNOUNCEMENT=1 \
python -m scripts.verify_live_sources --query "What do you think about Ping An Insurance (601318.SH)?" --debug
```

Check:

- News has web URLs where providers expose them.
- Announcements have Cninfo PDF URLs.
- Market and financial rows come from live providers, not only `seed`.
- `provider_endpoint`, `query_params`, and `source_reference` are populated for structured rows.

## Release Checklist

- Keep `data/external/`, `data/training_assets/`, `outputs/`, `reports/`, and `manual_test/output/` local.
- Keep `data/runtime/` and `models/*.joblib` committed for clone usability.
- Run narrow tests for touched modules before publishing.
- Check `git status --short` and ensure generated data was not staged.
- Do not commit `.env`, tokens, caches, or local scratch files.
