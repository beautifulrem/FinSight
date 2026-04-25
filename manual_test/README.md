# Manual Test

人工测试入口。脚本已内置 live 数据环境变量，直接运行即可。

## 基础测试（NLU + 检索）

```bash
python manual_test/run_manual_query.py
```

或者直接带问题：

```bash
python manual_test/run_manual_query.py --query "中国平安最近为什么跌？"
```

## 增强测试（含分析信号展示）

```bash
python manual_test/run_retrieval_test.py
```

或者直接带问题：

```bash
python manual_test/run_retrieval_test.py --query "茅台今天股价多少"
python manual_test/run_retrieval_test.py --query "最近CPI和PMI怎么样"
python manual_test/run_retrieval_test.py --query "创业板指最近走势"
```

区别：`run_retrieval_test.py` 在终端额外打印分析信号摘要（trend/RSI/MACD/Bollinger/宏观方向等），输出文件相同。

输出目录：

- `manual_test/output/<timestamp>-<slug>/query.txt`
- `manual_test/output/<timestamp>-<slug>/nlu_result.json`
- `manual_test/output/<timestamp>-<slug>/retrieval_result.json`

说明：

- `nlu_result.json` 对应第一个模块（NLU）
- `retrieval_result.json` 对应第二个模块（数据检索），含 `analysis_summary` 字段
- 每次运行都会新建一个输出目录，不会覆盖上一次结果

## 环境变量

两个测试脚本已内置以下默认值（使用 `os.environ.setdefault`，可被系统环境变量覆盖）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QI_USE_LIVE_MARKET` | `1` | 启用 live 行情/基本面数据 |
| `QI_USE_LIVE_MACRO` | `1` | 启用 live 宏观指标 |
| `QI_USE_LIVE_NEWS` | `1` | 启用 live 新闻 |

如需覆盖（例如关闭 live 数据），在运行前设置环境变量即可：

```bash
# Windows
set QI_USE_LIVE_MARKET=0
python manual_test/run_retrieval_test.py --query "茅台今天股价多少"

# Linux/macOS
QI_USE_LIVE_MARKET=0 python manual_test/run_retrieval_test.py --query "茅台今天股价多少"
```

可选环境变量：

| 变量 | 说明 |
|---|---|
| `TUSHARE_TOKEN` | Tushare API token，有则优先使用 Tushare 行情 |
| `QI_USE_LIVE_ANNOUNCEMENT` | **禁止开启**，会导致程序 hang |

## retrieval_result.json 关键字段

下游同学重点关注 `analysis_summary`：

```
analysis_summary
├── market_signal        # 市场技术信号（个股/ETF/指数时非null）
│   ├── trend_signal     # bullish / bearish / neutral
│   ├── rsi_14           # RSI相对强弱指标
│   ├── ma5 / ma20       # 移动平均线
│   ├── macd             # MACD指标
│   ├── bollinger        # 布林带
│   ├── volatility_20d   # 20日波动率
│   └── pct_change_nd    # 多日涨跌幅
├── fundamental_signal   # 基本面信号（个股时非null）
│   ├── pe_ttm / pb / roe
│   └── valuation_assessment  # 估值评估
├── macro_signal         # 宏观信号（宏观查询时非null）
│   ├── indicators[]     # 各指标方向
│   └── overall          # expansionary / contractionary / mixed
└── data_readiness       # 数据就绪状态
    ├── has_price_data / has_fundamentals / has_macro / has_news / has_technical_indicators
    ├── relevant_intents
    └── relevant_topics
```

完整字段说明见 `docs/retrieval_output_spec.md`。
