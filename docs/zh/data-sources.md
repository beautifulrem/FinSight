# 数据源架构、降级链与数据溯源

语言：[English](../data-sources.md) | 中文

FinSight 通过有序实时降级链、单源熔断器、有界调用线程池、TTL/最近可用缓存、多源财务交叉核验与三层离线快照获取中国市场行情、基本面、宏观、新闻与公告数据，并为每条记录附加不含数字的 `provenance` 溯源信息。

## 实时模式、离线快照与降级顺序

| 模式 | 行情数据 | 基本面与估值 | 宏观指标 | 新闻与公告 |
|---|---|---|---|---|
| **实时模式**（`QI_USE_LIVE_*=true`，默认） | 实时降级链（每次最新 30 个交易日；交易时段含盘中价） | 新浪 + 同花顺财务（并发核验）、东财数据中心 → 腾讯估值 | 东财数据中心（+ 中债 10 年期国债收益率） | 东财新闻；巨潮 → 东财公告 |
| **离线模式**（`QI_USE_LIVE_*=false`） | 随仓离线快照 | 随仓离线快照 | 随仓离线快照 | 本地运行时文档库（`data/runtime/documents.jsonl`） |

实时模式单条解析顺序：**实时降级链 → TTL 缓存 → 最近可用副本（≤24 小时）→ 离线快照**（行情仅在快照 ≤10 天新鲜窗口内兜底；基本面/宏观/行业快照均标注 `mode: snapshot` 与 `fallback_reason`）。

## 各类数据降级链

| 数据类型 | 主源 | 备选降级链 | 最终兜底 |
|---|---|---|---|
| A 股日线 | 东财 `stock_zh_a_hist`（配置 Token 时优先 Tushare） | 新浪 `stock_zh_a_daily` → 腾讯 fqkline → 新浪实时 → efinance | LKG → 新鲜快照 |
| ETF 日线 | 东财 `fund_etf_hist_em` | 新浪 `fund_etf_hist_sina` → 腾讯 fqkline → efinance | LKG → 新鲜快照 |
| 指数日线 | 新浪 `stock_zh_index_daily` | 东财 `index_zh_a_hist` → 腾讯 fqkline → 东财现价 | LKG → 新鲜快照 |
| 财务指标 | 新浪 `stock_financial_analysis_indicator` | 同花顺 `stock_financial_abstract_ths`（并发交叉核验） | 离线快照 |
| PE(TTM) / PB | 东财数据中心 `stock_value_em` | 腾讯行情 | 离线快照 |
| 行业背景 | 东财个股行业 / 同花顺行业指数 | 巨潮 `stock_profile_cninfo` | 离线快照（标注过期） |
| 基金费率与概况 | 东财 `fund_overview_em` | 雪球 → `fund_etf_fund_info_em` | – |
| CPI / PMI / M2 / LPR | 东财数据中心（统计局/央行口径） | – | LKG（6 h TTL / 24 h 陈旧）→ 快照 |
| 中国 10 年期国债收益率 | 东财 `bond_zh_us_rate` | 中债 `bond_china_yield` | LKG → 快照 |
| 新闻与公告 | 东财新闻；巨潮 orgId 公告 | 东财公告接口 | 本地语料库 |

所有适配器严格保留 `.SH`、`.SZ`、`.BJ` 交易所后缀，并自动将 `000xxx.SH` 与 `399xxx.SZ` 路由为指数而非个股。

## 离线三层快照与更正层

离线快照按优先级 **更正层 > v1 种子 > 扩展层** 合并，启动时校验 SHA-256 并在 `GET /ready` 展示加载详情：

1. **v1 种子**（`data/structured_data.json`，截至 `2026-04-22`）：7 只核心标的与 3 家公司，评测集默认锁定该基线以保证可比性。
2. **扩展层**（`data/snapshot/structured_data_ext.json`，截至 `2026-09-30`，`QI_OFFLINE_SNAPSHOT_EXT=true`）：新增 15 只主流 A 股、ETF 与指数的 301–307 个交易日收盘价及 FY2025 财报，离线即可计算 52 周高低点、年初至今涨跌幅（基于 `prior_year_end`）、最大回撤、EPS、总市值、同比增速与技术指标。校验命令：`python -m scripts.build_offline_snapshot verify`。
3. **更正层**（`data/snapshot/structured_data_fix.json`，`QI_OFFLINE_SNAPSHOT_FIX=true`）：依据巨潮一手 FY2025 年报及中证/新浪日线记录（登记于 `data/snapshot/corrections/sources.json`）修正 v1 种子中的五粮液/中国平安/贵州茅台财报与 2026-04-22 行情误差，并补充 `definitions` 会计口径字段（`roe_basis`、`revenue_basis`、`net_profit_basis`、`eps_basis`、`*_yoy_basis`）。若开关打开但文件缺失，`GET /ready` 返回 `503`，问答与声明核查均显式标注 `degraded: ["data_layer_missing:corrections"]`。

## 可靠性机制（`integrations/sources/`）

- **单源熔断器（`health.py`）**：连续失败 `QI_SOURCE_FAILURE_THRESHOLD`（默认 `3`）次即熔断，在 `QI_SOURCE_COOLDOWN_SECONDS`（默认 `60` 秒，指数退避上限 `600` 秒）内直接短路，随后放行单次半开探测。
- **有界线程池与硬超时（`runtime.py`）**：通过 `SourceCallPool`（默认 `QI_SOURCE_MAX_WORKERS=32`、`QI_SOURCE_CALL_TIMEOUT_SECONDS=10`）隔离不支持超时的上游库，并串行化 AKShare 的 `MiniRacer` V8 上下文避免并发崩溃；线程池打满立即快速失败（`SourcePoolSaturatedError`），不误触发上游熔断，嵌套调用走内联避免死锁。
- **TTL 与最近可用缓存（`cache.py`）**：行情 60 秒、日频宏观 30 分钟、月频宏观 6 小时；上游全挂时最多复用 24 小时内过期缓存并标记 `last_known_good`。
- **新浪/同花顺基本面交叉核验（`crosscheck.py`，`QI_SOURCE_CROSS_CHECK=1`）**：并发抓取两源，剔除越界异常值，区分累计与单季同比口径，对相差 ≥2 个百分点的同比依据绝对值重算仲裁，并将结果写入 `provenance.cross_check`。
- **盘中实时价（`intraday.py`）**：A 股交易时段（周一至周五 `09:30–11:30`、`13:00–15:00`）针对含「今天/今日/today」的提问拉取新浪/腾讯实时价（`price_basis: "intraday"`）；非交易时段或节假日自动返回最近收盘价并说明 `basis_reason`。
- **数值清洗（`values.py`）**：统一清洗 `NaN`/`Inf`、`--` 占位符、百分比字符串、`亿`/`万` 单位与成交量单位（`lot` / `share`）。

## 溯源字段格式

每条结构化记录与文档均携带仅含字符串、布尔值与 ISO 日期的 `provenance`（不含裸指标数值，防止干扰数值校验器）：

```json
{
  "source": "sina.kline",
  "source_label": "新浪财经行情",
  "endpoint": "akshare.stock_zh_a_daily",
  "is_live": true,
  "mode": "live_fallback",
  "fetched_at": "2026-09-25T09:08:12+00:00",
  "as_of": "2026-09-24",
  "freshness": "fresh",
  "fallback_reason": "eastmoney.quote:circuit_open",
  "attempts": ["eastmoney.quote:circuit_open", "sina.kline:ok"],
  "cache_hit": false,
  "note": "数据来自新浪财经行情，截至2026-09-24；因东方财富行情熔断中降级"
}
```

## 健康检查接口

- `GET /sources/health`：被动返回 19 个已登记数据源的熔断状态、调用/失败计数、延迟、`retry_in_s` 与 `worker_pool` 计数器。
- `GET /sources/health?probe=1`：可选主动探测，受全局限流间隔 `QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS`（默认 `60` 秒）保护。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QI_USE_LIVE_MARKET` / `_NEWS` / `_ANNOUNCEMENT` / `_MACRO` | `true` | 开启或关闭实时源 |
| `QI_SOURCE_CALL_TIMEOUT_SECONDS` | `10` | 单次上游调用硬超时（秒） |
| `QI_SOURCE_FAILURE_THRESHOLD` | `3` | 触发熔断的连续失败次数 |
| `QI_SOURCE_COOLDOWN_SECONDS` / `_MAX_COOLDOWN_SECONDS` | `60` / `600` | 首次熔断冷却与最大退避上限 |
| `QI_SOURCE_CACHE` / `QI_SOURCE_MAX_STALE_SECONDS` | `true` / `86400` | TTL 缓存与 LKG 最长保留时间 |
| `QI_SOURCE_MAX_WORKERS` | `32` | 有界调用线程池大小 |
| `QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS` | `60` | 主动探测 `?probe=1` 最小间隔 |
| `QI_SOURCE_CROSS_CHECK` | `true` | 开启新浪/同花顺财务交叉核验 |
| `QI_OFFLINE_SNAPSHOT_EXT` / `QI_OFFLINE_SNAPSHOT_FIX` | `true` / `true` | 离线扩展层与更正层开关 |
| `QI_EVAL_SNAPSHOT_EXT` / `QI_EVAL_SNAPSHOT_FIX` | 未设置 | 评测框架显式启用扩展/更正层的开关 |

## 测试与已知局限

```bash
python -m pytest -q tests/test_data_sources.py tests/test_source_reliability.py tests/test_intraday_quote.py tests/test_data_layers.py
QI_LIVE_TESTS=1 python -m pytest -q tests/test_data_sources_live.py
```

- **宏观备源**：CPI、PMI、M2、LPR 暂无独立实时二源，东财不可用时回退至 LKG 或离线快照。
- **不复权与窗口长度**：收盘价均为不复权；实时路径单次拉取 30 日 K 线，52 周/年初至今统计需依赖离线扩展层或更长历史窗口。
- **行业估值卡片**：同花顺实时行业刷新仅含涨跌幅，缺失行业均值估值的板块会显式披露而非估算补全。
- **历史实测审计**：原始实网源审计报告归档于 [audit-20261001T0626Z-c915aef.md](../results/data_sources/audit-20261001T0626Z-c915aef.md)。
