# 训练和运行时资产

语言：[English](../training.md) | 中文

FinSight 后端使用传统、可解释模型训练 NLU 和 Retrieval。训练资产和运行时资产是两套东西：

- 训练资产用于提升分类器、ranker、CRF、typo linker、source planner 和 out-of-scope 行为。
- 运行时资产用于生产实体解析、alias 匹配、本地文档召回和结构化 seed fallback。

重新训练模型不会自动更新运行时实体、alias 或文档。

## 数据目录

| 路径 | 用途 | 提交策略 |
|---|---|---|
| `data/external/raw/` | 下载的公开数据集 | 本地 |
| `data/training_assets/` | 标准化 manifest 和任务 JSONL | 本地 |
| `data/runtime/` | 随仓库发布的运行时实体、alias、文档 | 提交 |
| `models/` | 随仓库发布的模型 artifact | 提交 |
| `reports/` | 评估报告 | 本地 |

## 公开数据源

数据集注册表在 `query_intelligence/external_data/registry.py`，适配器位于 `query_intelligence/external_data/`。

| 来源 | 类型 | 主要用途 |
|---|---|---|
| CFLUE | GitHub | 金融分类和问答。 |
| FiQA / BEIR | HuggingFace | 检索语料、qrels、LTR。 |
| FinFE | HuggingFace | 金融情感。 |
| ChnSentiCorp | HuggingFace | 中文情感。 |
| Financial news sentiment | GitHub | 中文金融新闻情感。 |
| MSRA NER / People's Daily NER / CLUENER | HuggingFace/GitHub | 实体边界和 CRF。 |
| TNEWS / THUCNews / SMP2017 | HuggingFace/direct/GitHub | 产品类型、意图、主题、OOD 泛化。 |
| BBT-FinCUGE、Mxode finance、BAAI finance instruction | GitHub/HuggingFace | 金融意图和主题监督。 |
| QReCC / RiSAWOZ | HuggingFace | 多轮和澄清行为。 |
| T2Ranking / FinCPRG / FIR-Bench / CSPR-D | HuggingFace/GitHub | 检索、研报、公告、qrels、source/ranker 监督。 |
| Curated boundary cases | 本地生成 | 边界和回归用例。 |

## 同步和构建

同步所有启用的数据集：

```bash
QI_ENABLE_EXTERNAL_DATA=1 python -m scripts.sync_public_datasets
```

只同步部分数据集：

```bash
QI_ENABLE_EXTERNAL_DATA=1 QI_DATASET_ALLOWLIST=finfe,t2ranking,fir_bench_reports \
python -m scripts.sync_public_datasets
```

构建标准训练资产：

```bash
python -m scripts.build_training_assets
```

一键同步、构建、预检、训练：

```bash
QI_ENABLE_EXTERNAL_DATA=1 python -m training.sync_and_train
```

从现有 manifest 训练：

```bash
python -m training.train_all data/training_assets/manifest.json
```

## 训练预检

```bash
python -m training.prepare_training_run data/training_assets/manifest.json models
```

预检报告：

```text
data/training_assets/preflight_report.json
```

替换随仓库发布的模型前先检查该报告。

## 训练脚本

| 模型 | 命令 | 输出 |
|---|---|---|
| 产品类型分类器 | `python -m training.train_product_type data/training_assets/manifest.json` | `models/product_type.joblib` |
| 意图多标签分类器 | `python -m training.train_intent data/training_assets/manifest.json` | `models/intent_ovr.joblib` |
| 主题多标签分类器 | `python -m training.train_topic data/training_assets/manifest.json` | `models/topic_ovr.joblib` |
| 问题样式分类器 | `python -m training.train_question_style data/training_assets/manifest.json` | `models/question_style.joblib` |
| 用户情绪分类器 | `python -m training.train_sentiment data/training_assets/manifest.json` | `models/sentiment.joblib` |
| 实体边界 CRF | `python -m training.train_entity_crf data/training_assets/manifest.json` | `models/entity_crf.joblib` |
| 澄清 gate | `python -m training.train_clarification_gate data/training_assets/manifest.json` | `models/clarification_gate.joblib` |
| 问题样式 reranker | `python -m training.train_question_style_reranker data/training_assets/manifest.json` | `models/question_style_reranker.joblib` |
| Source plan reranker | `python -m training.train_source_plan_reranker data/training_assets/manifest.json` | `models/source_plan_reranker.joblib` |
| 越界检测器 | `python -m training.train_out_of_scope_detector data/training_assets/manifest.json` | `models/out_of_scope_detector.joblib` |
| 文档 ranker | `python -m training.train_ranker data/training_assets/manifest.json` | `models/ranker.joblib` |
| Typo linker | `python -m training.train_typo_linker data/training_assets/manifest.json` | `models/typo_linker.joblib` |
| 全部训练 | `python -m training.train_all data/training_assets/manifest.json` | 所有模型 artifact |

## 运行时资产

训练后生成实体和 alias：

```bash
python -m scripts.materialize_runtime_entity_assets
```

常用参数：

```bash
python -m scripts.materialize_runtime_entity_assets \
  --seed-dir data \
  --training-assets-dir data/training_assets \
  --output-dir data/runtime \
  --max-training-pairs 80000
```

只使用本地资产：

```bash
python -m scripts.materialize_runtime_entity_assets --no-akshare
```

输出：

```text
data/runtime/entity_master.csv
data/runtime/alias_table.csv
```

生成运行时文档：

```bash
python -m scripts.materialize_runtime_document_assets
```

常用参数：

```bash
python -m scripts.materialize_runtime_document_assets \
  --corpus-path data/training_assets/retrieval_corpus.jsonl \
  --output-path data/runtime/documents.jsonl \
  --max-documents 50000
```

输出：

```text
data/runtime/documents.jsonl
```

## 评估

推荐交接顺序：

```bash
python -m pytest tests/test_query_intelligence.py tests/test_real_integrations.py -q
python -m pytest tests/test_manual_query.py tests/test_manual_test_runner.py -q
python -m pytest tests/test_ml_upgrades.py -q
python -m scripts.run_test_suite
python -m scripts.evaluate_query_intelligence
```

全链路评估覆盖中文/英文、金融、非金融、对抗和边界问题，并报告金融召回、OOD 拒识、产品类型准确率、问题样式准确率、意图/主题 F1、澄清召回、source-plan 质量、retrieval recall@10、MRR@10、NDCG@10 和 OOD retrieval abstention。

## Live Source 验证

```bash
QI_USE_LIVE_MARKET=1 QI_USE_LIVE_NEWS=1 QI_USE_LIVE_ANNOUNCEMENT=1 \
python -m scripts.verify_live_sources --query "你觉得中国平安怎么样？" --debug
```

检查：

- 新闻尽量有 web URL。
- 公告有巨潮 PDF URL。
- 行情和财务来自 live provider，而不是只有 `seed`。
- 结构化行填充 `provider_endpoint`、`query_params` 和 `source_reference`。

## 发布检查

- `data/external/`、`data/training_assets/`、`outputs/`、`reports/`、`manual_test/output/` 保持本地。
- `data/runtime/` 和 `models/*.joblib` 保持提交，保证 clone 后可用。
- 发布前运行涉及模块的窄测试。
- 检查 `git status --short`，不要 stage 生成数据。
- 不提交 `.env`、token、cache 或本地临时文件。
