# Numerical Analysis

Languages: English | [中文](zh/numerical-analysis.md)

The numerical analysis module summarizes structured evidence retrieved by FinSight. It is implemented mainly in `query_intelligence/retrieval/market_analyzer.py` and exposed through `retrieval_result.analysis_summary`.

This module supports explanation and evidence readiness. It does not produce deterministic buy/sell decisions.

## Inputs

Numerical analysis consumes structured rows already selected by retrieval:

| Input | Typical source | Use |
|---|---|---|
| Price and quote history | `market_api`, Tushare, AKShare, efinance, seed data | Returns, moving averages, volatility, RSI, MACD, Bollinger bands. |
| Fundamentals | `fundamental_sql`, Tushare, local structured rows | Profitability, valuation, ROE, PE/PB, data completeness. |
| Industry context | `industry_sql` | Sector identity and sector-level context. |
| Macro indicators | `macro_sql`, `macro_indicator`, `policy_event` | CPI, PMI, M2, treasury yields, policy-event direction. |
| Fund/index rows | `fund_nav`, `fund_fee`, `index_daily`, `index_valuation` | NAV, fee, redemption, index trend and valuation evidence. |

The module uses the resolved entity, source plan, retrieved structured rows, and NLU intent/topic hints. It does not re-run NLU.

## Output

`analysis_summary` is a compact object designed for downstream modules:

```text
analysis_summary
├── market_signal
│   ├── trend_signal
│   ├── pct_change_nd
│   ├── ma5 / ma20
│   ├── rsi_14
│   ├── macd
│   ├── bollinger
│   └── volatility_20d
├── fundamental_signal
│   ├── pe_ttm / pb / roe
│   ├── valuation_assessment
│   └── available_fields
├── macro_signal
│   ├── indicators[]
│   └── overall
└── data_readiness
    ├── has_price_data
    ├── has_fundamentals
    ├── has_macro
    ├── has_news
    ├── has_technical_indicators
    ├── relevant_intents
    └── relevant_topics
```

## Indicator Meaning

| Signal | Meaning |
|---|---|
| `trend_signal` | Directional description inferred from recent price and moving-average context. |
| `pct_change_nd` | Recent percentage return over available windows. |
| `ma5`, `ma20` | Short and medium moving averages when enough price history exists. |
| `rsi_14` | Momentum indicator. Extreme values should be treated as context, not a standalone trading rule. |
| `macd` | Trend and momentum signal derived from exponential moving averages. |
| `bollinger` | Price position relative to recent volatility bands. |
| `volatility_20d` | Recent realized volatility. |
| `valuation_assessment` | Lightweight valuation interpretation based on available PE/PB/fundamental rows. |
| `data_readiness` | Flags that tell downstream modules whether enough evidence exists for a reliable explanation. |

## Design Rules

- Use only structured evidence already retrieved by FinSight.
- Keep calculations explainable and inspectable.
- Preserve missing-data warnings instead of filling unsupported facts.
- Treat indicators as descriptive signals, not trading instructions.
- Keep downstream LLM prompts compact by passing summarized fields, not raw provider dumps.

## Related Tests

```bash
python -m pytest tests/test_analysis_summary.py tests/test_market_analyzer.py -q
```

## Related Documentation

- [Query Intelligence](query-intelligence.md)
- [LLM Response Handoff](llm-response.md)
- [Training and Runtime Assets](training.md)
