# Retrieval Output Specification

Languages: English | [中文](zh/retrieval_output_spec.md)

`RetrievalResult` is the typed evidence contract produced by `POST /retrieval/search` and `POST /query/intelligence`, defined in [`query_intelligence/contracts.py`](../query_intelligence/contracts.py) and exported as [`schemas/retrieval_result.schema.json`](../schemas/retrieval_result.schema.json).

## Top-Level Fields

| Field | Type | Description |
|---|---|---|
| `query_id` | string | UUID matching upstream `NLUResult.query_id` |
| `nlu_snapshot` | object | Key NLU fields consumed by retrieval |
| `executed_sources` | string[] | Sources actually executed (subset of `source_plan`) |
| `documents` | `DocumentEvidence[]` | Ranked `news`, `announcement`, `research_note`, `faq`, `product_doc` items with stable `evidence_id` and ranking `reason[]` |
| `structured_data` | `StructuredEvidence[]` | Market, fundamental, industry, fund, index, and macro rows with `provider_endpoint`, `source_reference`, `as_of`, `quality_flags`, and `payload` |
| `evidence_groups` | object[] | Deduplication/clustering groups |
| `coverage`, `coverage_detail` | object | High-level and fine-grained coverage flags |
| `warnings` | string[] | Short-circuit or data-quality warnings |
| `retrieval_confidence` | float | Overall retrieval confidence score |
| `analysis_summary` | object | Pre-computed technical, fundamental, macro, and `data_readiness` signals |
| `debug_trace` | object | Candidate counts and ranking debug info (populated when `debug: true`) |

## Related References

- Full field & label tables: [Query Intelligence — RetrievalResult](query-intelligence.md#retrievalresult)
- Technical & fundamental indicator definitions: [Numerical Analysis](numerical-analysis.md)
- Live/offline provenance fields (`payload.provenance`): [Data Sources](data-sources.md#provenance-schema)
