# 检索输出规范（Retrieval Output Spec）

语言：[English](../retrieval_output_spec.md) | 中文

`RetrievalResult` 是 `POST /retrieval/search` 与 `POST /query/intelligence` 产出的类型化证据契约，定义于 [`query_intelligence/contracts.py`](../../query_intelligence/contracts.py)，导出 Schema 位于 [`schemas/retrieval_result.schema.json`](../../schemas/retrieval_result.schema.json)。

## 顶层字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `query_id` | string | 与上游 `NLUResult.query_id` 一致的 UUID |
| `nlu_snapshot` | object | 检索阶段消费的关键 NLU 快照字段 |
| `executed_sources` | string[] | 实际执行的数据源列表（`source_plan` 子集） |
| `documents` | `DocumentEvidence[]` | 排序后的 `news`、`announcement`、`research_note`、`faq`、`product_doc`，含稳定 `evidence_id` 与排序 `reason[]` |
| `structured_data` | `StructuredEvidence[]` | 行情、基本面、行业、基金、指数与宏观记录，含 `provider_endpoint`、`source_reference`、`as_of`、`quality_flags` 与 `payload` |
| `evidence_groups` | object[] | 去重/聚类分组 |
| `coverage`、`coverage_detail` | object | 粗粒度与细粒度覆盖度标记 |
| `warnings` | string[] | 短路或数据质量告警 |
| `retrieval_confidence` | float | 检索综合置信度 |
| `analysis_summary` | object | 预计算的技术、基本面、宏观与 `data_readiness` 摘要 |
| `debug_trace` | object | 候选数与排序调试轨迹（`debug: true` 时返回） |

## 相关参考

- 完整字段与标签枚举：[Query Intelligence — RetrievalResult](query-intelligence.md#retrievalresult)
- 技术与基本面信号定义：[数值分析](numerical-analysis.md)
- 实时/离线溯源元数据（`payload.provenance`）：[数据源架构](data-sources.md#溯源字段格式)
