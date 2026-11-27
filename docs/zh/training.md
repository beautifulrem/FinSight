# 模型训练与运行时资产

语言：[English](../training.md) | 中文

FinSight 训练论文中描述的经典可解释 NLU 与检索模型（[DOI: `10.3233/FAIA261107`](https://ebooks.iospress.nl/doi/10.3233/FAIA261107)）。**训练资产**与**运行时资产**严格解耦：重训仅更新 `.joblib` 模型，不会自动改写 `data/runtime/`。

## 目录与提交策略

| 路径 | 用途 | 版本控制 |
|---|---|---|
| `data/external/raw/` | 下载的公开数据集原始文件 | 仅本地（git 忽略） |
| `data/training_assets/` | 标准化任务 JSONL 与 `manifest.json` | 仅本地（git 忽略） |
| `data/runtime/` | 随仓实体表、别名表、A+H/纯港股表与本地语料库 | 提交入库 |
| `models/` | 随仓 `.joblib` 模型，加载前按 `models/manifest.json` 校验 SHA-256 | 提交入库 |
| `reports/` | 本地评测报告 | 仅本地（git 忽略） |

## 公开数据同步与训练流程

数据注册表位于 `query_intelligence/external_data/registry.py`（涵盖 CFLUE、FiQA/BEIR、FinFE、ChnSentiCorp、MSRA/CLUENER NER、TNEWS/THUCNews/SMP2017、BBT-FinCUGE、QReCC/RiSAWOZ、T2Ranking、FinCPRG、FIR-Bench、CSPR-D 及边界用例）。

```bash
# 1. 同步公开数据集（可选 QI_DATASET_ALLOWLIST=finfe,t2ranking 仅同步子集）
QI_ENABLE_EXTERNAL_DATA=1 python -m scripts.sync_public_datasets

# 2. 构建标准化训练资产与 manifest
python -m scripts.build_training_assets

# 3. 训练前预检（输出 data/training_assets/preflight_report.json）
python -m training.prepare_training_run data/training_assets/manifest.json models

# 4. 全量训练（或用 training.sync_and_train 一键串行执行）
python -m training.train_all data/training_assets/manifest.json
```

### 分模型训练入口

| 模型 | 训练模块 | 产物 |
|---|---|---|
| 产品类型 / 意图 / 主题 | `training.train_product_type`、`train_intent`、`train_topic` | `product_type.joblib`、`intent_ovr.joblib`、`topic_ovr.joblib` |
| 提问风格与重排 | `training.train_question_style`、`train_question_style_reranker` | `question_style.joblib`、`question_style_reranker.joblib` |
| 用户情绪与越界检测 | `training.train_sentiment`、`train_out_of_scope_detector` | `sentiment.joblib`、`out_of_scope_detector.joblib` |
| 实体边界 CRF 与纠错链接 | `training.train_entity_crf`、`train_typo_linker` | `entity_crf.joblib`、`typo_linker.joblib` |
| 澄清门控与 Source Plan 重排 | `training.train_clarification_gate`、`train_source_plan_reranker` | `clarification_gate.joblib`、`source_plan_reranker.joblib` |
| 文档 LTR 排序器 | `training.train_ranker` | `ranker.joblib` |

## 物化运行时资产

```bash
# 刷新实体主表与别名表（--no-akshare 仅使用本地种子）
python -m scripts.materialize_runtime_entity_assets --output-dir data/runtime

# 刷新本地文档检索语料
python -m scripts.materialize_runtime_document_assets \
  --corpus-path data/training_assets/retrieval_corpus.jsonl \
  --output-path data/runtime/documents.jsonl --max-documents 50000
```

## 评测与发布检查

```bash
python -m pytest -q tests/test_query_intelligence.py tests/test_ml_upgrades.py tests/test_real_integrations.py
python -m scripts.run_test_suite
python -m scripts.evaluate_query_intelligence
QI_USE_LIVE_MARKET=1 QI_USE_LIVE_NEWS=1 python -m scripts.verify_live_sources --query "你觉得中国平安怎么样？" --debug
```

**评测说明**：`scripts/evaluate_query_intelligence.py` 的 10,000 条查询混合公开切分与模板边界样本；检索 Recall@10/MRR@10/NDCG@10 属于 40 负样本重排指标而非全库召回；澄清召回率（`0.75`）仍低于 `0.90` 目标。发布前确认在无 `data/external/` 和 `data/training_assets/` 条件下仓库可独立运行，且未误提交 `.env`、Token 或 `outputs/`。
