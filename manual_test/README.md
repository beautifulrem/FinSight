# Manual CLI Test Runners

Command-line scripts for inspecting FinSight's classical **Query Intelligence** (`NLUResult`, `RetrievalResult`, `analysis_summary`) and **Document Sentiment** (`sentiment/`) outputs directly without running the web server.

## Query Intelligence (NLU + Retrieval)

```bash
# Run NLU + retrieval and write JSON artifacts to manual_test/output/<timestamp>-<slug>/
python manual_test/run_manual_query.py --query "中国平安最近为什么跌？"

# Also print MarketAnalyzer technical/fundamental/macro signals to stdout
python manual_test/run_retrieval_test.py --query "茅台今天股价多少"
python manual_test/run_retrieval_test.py --query "最近CPI和PMI怎么样"
```

Generated files (git-ignored): `query.txt`, `nlu_result.json`, and `retrieval_result.json` (including `analysis_summary`; see [`docs/retrieval_output_spec.md`](../docs/retrieval_output_spec.md)).

## Document Sentiment (`sentiment/`)

```bash
# Integrated run (classical/mock default; add --real-models for FinBERT weights)
python manual_test/run_manual_query.py --query "茅台最近有什么消息" --sentiment
python manual_test/run_manual_query.py --query "茅台最近有什么消息" --sentiment --real-models

# Standalone sentiment runner & test matrices
python manual_test/run_sentiment_test.py --input manual_test/output/<run-dir>
python manual_test/test_fuzz_sentiment.py --num-samples 200 --seed 1
python manual_test/test_coverage_sentiment.py --quick
```

## Environment Switches

The manual scripts default `QI_USE_LIVE_MARKET`, `QI_USE_LIVE_MACRO`, and `QI_USE_LIVE_NEWS` to `1` via `os.environ.setdefault` (set `QI_USE_LIVE_MARKET=0` etc. to run strictly offline). Configure `TUSHARE_TOKEN` to prefer Tushare market/fundamental endpoints; live announcements (`QI_USE_LIVE_ANNOUNCEMENT=1`) are opt-in and protected by source timeouts and circuit breakers.
