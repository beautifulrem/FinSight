# Numerical Analysis

Languages: English | [中文](zh/numerical-analysis.md)

The numerical analysis stage (`query_intelligence/retrieval/market_analyzer.py`) converts retrieved structured rows into compact, inspectable analytical signals exposed as `retrieval_result.analysis_summary` and invoked by the Agent tools `get_technical_indicators` and `get_fundamentals`.

It summarizes observable market, fundamental, and macro evidence; it never emits trading instructions or causal price forecasts.

## Inputs

Consumes structured rows already produced by retrieval (never re-runs NLU):

| Input | Typical Sources | Signals Derived |
|---|---|---|
| Price & quote history | `market_api` (Tushare, AKShare, efinance, snapshot) | N-day returns, `MA5`/`MA20`, `RSI(14)`, `MACD`, Bollinger bands, 20-day realized volatility, `trend_signal` |
| Fundamentals | `fundamental_sql` (Tushare, Sina/THS, snapshot) | `pe_ttm`, `pb`, `roe`, `valuation_assessment`, field completeness |
| Industry & macro | `industry_sql`, `macro_sql`, `macro_indicator`, `policy_event` | Sector context, CPI/PMI/M2/yield indicators, macro direction |
| Funds & indices | `fund_nav`, `fund_fee`, `index_daily`, `index_valuation` | NAV, fee/redemption metadata, index trend and valuation |

## Output Structure

```text
analysis_summary
├── market_signal       # trend_signal, pct_change_nd, ma5, ma20, rsi_14, macd, bollinger, volatility_20d
├── fundamental_signal  # pe_ttm, pb, roe, valuation_assessment, available_fields
├── macro_signal        # indicators[], overall
└── data_readiness      # has_price_data, has_fundamentals, has_macro, has_news,
                        # has_technical_indicators, relevant_intents, relevant_topics
```

## Indicator Semantics & Safeguards

- **`trend_signal` / `ma5` / `ma20` / `macd` / `bollinger`**: Descriptive trend and volatility-band context computed only when sufficient historical bars exist; non-finite values (`NaN`/`Inf`) are sanitized to `null`.
- **`rsi_14` / `volatility_20d`**: Momentum and realized volatility reported strictly as descriptive context, never as standalone buy/sell triggers.
- **`valuation_assessment`**: Lightweight interpretation grounded in available valuation multiples and profitability ratios.
- **`data_readiness`**: Explicit boolean readiness flags so downstream composers surface missing data rather than extrapolating.

## Verification

```bash
python -m pytest tests/test_analysis_summary.py tests/test_market_analyzer.py -q
```

See also: [Query Intelligence](query-intelligence.md) · [Agent Layer](agent.md) · [Data Sources](data-sources.md)
