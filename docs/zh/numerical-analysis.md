# 数值分析

语言：[English](../numerical-analysis.md) | 中文

数值分析模块汇总 FinSight 检索到的结构化证据。主要实现位于 `query_intelligence/retrieval/market_analyzer.py`，输出在 `retrieval_result.analysis_summary` 中。

该模块用于解释和判断证据就绪程度，不输出确定性的买入/卖出决策。

## 输入

数值分析消费 Retrieval 已选出的结构化数据：

| 输入 | 常见来源 | 用途 |
|---|---|---|
| 行情和价格历史 | `market_api`、Tushare、AKShare、efinance、seed data | 收益率、均线、波动率、RSI、MACD、布林带。 |
| 基本面 | `fundamental_sql`、Tushare、本地结构化行 | 盈利能力、估值、ROE、PE/PB、字段完整度。 |
| 行业上下文 | `industry_sql` | 行业身份和行业层面背景。 |
| 宏观指标 | `macro_sql`、`macro_indicator`、`policy_event` | CPI、PMI、M2、国债收益率、政策事件方向。 |
| 基金/指数行 | `fund_nav`、`fund_fee`、`index_daily`、`index_valuation` | 净值、费用、申赎、指数趋势和估值证据。 |

模块复用已解析实体、source plan、结构化检索结果和 NLU 意图/主题，不重新执行 NLU。

## 输出

`analysis_summary` 是面向下游模块的紧凑对象：

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

## 指标含义

| 信号 | 含义 |
|---|---|
| `trend_signal` | 根据近期价格和均线上下文得到的方向描述。 |
| `pct_change_nd` | 可用窗口内的近期收益率。 |
| `ma5`, `ma20` | 短期和中期移动平均。 |
| `rsi_14` | 动量指标。极端值只能作为上下文，不应作为单独交易规则。 |
| `macd` | 基于指数移动平均的趋势和动量信号。 |
| `bollinger` | 价格相对近期波动带的位置。 |
| `volatility_20d` | 近期实现波动率。 |
| `valuation_assessment` | 基于可用 PE/PB/基本面行的轻量估值解读。 |
| `data_readiness` | 告诉下游模块证据是否足够支持可靠解释。 |

## 设计规则

- 只使用 FinSight 已检索到的结构化证据。
- 计算保持可解释、可检查。
- 保留缺失数据 warning，不填充无依据事实。
- 指标是描述性信号，不是交易指令。
- 给下游 LLM 传递摘要字段，而不是原始 provider dump。

## 相关测试

```bash
python -m pytest tests/test_analysis_summary.py tests/test_market_analyzer.py -q
```

## 相关文档

- [Query Intelligence](query-intelligence.md)
- [LLM 回答生成交接](llm-response.md)
- [训练和运行时资产](training.md)
