# 数值分析

语言：[English](../numerical-analysis.md) | 中文

数值分析模块（`query_intelligence/retrieval/market_analyzer.py`）将检索得到的结构化数据整理为紧凑、可审查的摘要信号，输出为 `retrieval_result.analysis_summary`，并由 Agent 的 `get_technical_indicators` 与 `get_fundamentals` 工具复用。

该模块仅做客观证据汇总与数据就绪度提示，不输出确定性买卖指令或未来价格预测。

## 输入来源

仅消费上游已检索的结构化记录（不重新运行 NLU）：

| 输入类型 | 常见来源 | 派生信号 |
|---|---|---|
| 行情与历史 K 线 | `market_api`（Tushare、AKShare、efinance、离线快照） | N 日涨跌幅、`MA5`/`MA20`、`RSI(14)`、`MACD`、布林带、20 日波动率、`trend_signal` |
| 基本面与估值 | `fundamental_sql`（Tushare、新浪/同花顺、快照） | `pe_ttm`、`pb`、`roe`、`valuation_assessment`、字段完整度 |
| 行业与宏观 | `industry_sql`、`macro_sql`、`macro_indicator`、`policy_event` | 行业背景、CPI/PMI/M2/国债收益率及宏观方向 |
| 基金与指数 | `fund_nav`、`fund_fee`、`index_daily`、`index_valuation` | 净值、费率赎回规则、指数趋势与估值 |

## 输出结构

```text
analysis_summary
├── market_signal       # trend_signal, pct_change_nd, ma5, ma20, rsi_14, macd, bollinger, volatility_20d
├── fundamental_signal  # pe_ttm, pb, roe, valuation_assessment, available_fields
├── macro_signal        # indicators[], overall
└── data_readiness      # has_price_data, has_fundamentals, has_macro, has_news,
                        # has_technical_indicators, relevant_intents, relevant_topics
```

## 指标语义与保护规则

- **`trend_signal` / `ma5` / `ma20` / `macd` / `bollinger`**：仅在历史交易日数量充足时计算；`NaN`/`Inf` 等非有限数值统一清洗为 `null`。
- **`rsi_14` / `volatility_20d`**：仅作为动量与波动率背景信号，不作为独立交易触发器。
- **`valuation_assessment`**：基于可用估值倍数与盈利指标生成轻量解释。
- **`data_readiness`**：显式暴露数据是否齐全，提示下游披露缺失项而非臆测补全。

## 验证测试

```bash
python -m pytest tests/test_analysis_summary.py tests/test_market_analyzer.py -q
```

相关文档：[Query Intelligence](query-intelligence.md) · [Agent 层](agent.md) · [数据源架构](data-sources.md)
