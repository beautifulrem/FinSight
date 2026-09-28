# 实时数据源：审计、降级链与来源标注

语言：[English](../data-sources.md) | 中文

本文记录三部分内容：

- FinSight 用到的每个实时数据源的一次真实网络审计；
- 审计暴露出的缺陷；
- 现在挡在这些数据源前面的取数设计：有序的降级链、按数据源的熔断器、带「最近一次成功数据」读取的 TTL 缓存、硬超时，以及每条返回记录上的来源标注。

下面的数字都是实测，不是估计。上游行为会变（尤其是东方财富会对来自同一 IP 的突发请求限流），引用前请重新跑一遍审计。

## 审计方法

| 项目 | 值 |
|---|---|
| 日期 | 2026-09-25（中秋节休市，最近交易日为 2026-09-24） |
| 命令 | `python -m scripts.audit_data_sources --json outputs/data_source_audit.json` |
| 代码 | commit `c1c3388` 加上本文描述的工作区改动 |
| 版本 | Python 3.13.13、akshare 1.18.97、efinance 0.5.9、requests 2.34.2、pandas 2.3.3 |
| 网络 | 中国大陆家庭宽带，环境里有本地 HTTP 代理（`127.0.0.1:6152`） |
| 方法 | 每个数据源单独调用：关闭熔断、不重试、不缓存、15 秒硬超时。然后用全新的熔断器把运行时的降级链端到端跑一遍。 |
| 标的 | 600519 贵州茅台、300750 宁德时代、601318 中国平安、ETF 510300、指数 000300，宏观序列 CPI / PMI / M2 / 10 年期国债 / LPR |

结果：64 次探测，49 次成功。每个失败都找到了根因（见下文）。

## 各数据源结果

延迟是各标的上观察到的范围，「最新数据日期」是返回数据中最新的日期。结果来自上面那次审计。

| 类型 | 数据源（函数） | 成功 | 延迟 ms | 最新数据日期 | 失败 / 说明 |
|---|---|---|---:|---|---|
| 个股日线 | 东方财富 `push2his`（`stock_zh_a_hist`） | 0/3 | 109–348 | – | 连接被断开（见根因 1） |
| 个股日线 | 新浪（`stock_zh_a_daily`） | 3/3 | 261–512 | 2026-09-24 | 成交量单位为股 |
| 个股日线 | 腾讯（`web.ifzq.gtimg.cn` fqkline，**新增**） | 3/3 | 97–104 | 2026-09-24 | 成交量单位为手 |
| 个股报价 | 新浪实时（`hq.sinajs.cn`） | 3/3 | 40–57 | 2026-09-24 | 只有一行，无历史 |
| 个股日线 | efinance（`get_quote_history`） | 0/3 | 763–1076 | – | 和东方财富是同一个 `push2his` 主机 |
| ETF 日线 | 东方财富（`fund_etf_hist_em`） | 0/1 | 123 | – | 根因 1 |
| ETF 日线 | 新浪（`fund_etf_hist_sina`） | 1/1 | 279 | 2026-09-24 | |
| ETF 日线 | 腾讯 fqkline（**新增**） | 1/1 | 98 | 2026-09-24 | |
| ETF 净值 | 东方财富基金（`fund_open_fund_info_em`） | 1/1 | 140 | 2026-09-24 | |
| ETF 概况/费率 | 东方财富基金（`fund_overview_em`，**新增**） | 1/1 | 203 | – | 费率、基金经理、跟踪指数 |
| ETF 概况/费率 | 雪球（`fund_individual_detail_info_xq`） | 0/1 | 1392 | – | `KeyError: 'data'`：现在需要登录 token（根因 4） |
| 指数日线 | 新浪（`stock_zh_index_daily`） | 1/1 | 246 | 2026-09-24 | 返回完整历史（约 6,000 行）；另一次运行用了 3,418 ms |
| 指数日线 | 东方财富（`index_zh_a_hist`） | 0/1 | 5451 | – | 根因 1（`80.push2.eastmoney.com`） |
| 指数日线 | 腾讯 fqkline（**新增**） | 1/1 | 99 | 2026-09-24 | |
| 指数估值 | 中证指数（`stock_zh_index_value_csindex`） | 1/1 | 288 | 2026-09-24 | 另一次运行约 1,600 ms |
| 财务指标 | 新浪（`stock_financial_analysis_indicator`） | 3/3 | 344–391 | 2026-06-30 | 列名改了（根因 3） |
| 财务指标 | 同花顺（`stock_financial_abstract_ths`，**新增**） | 3/3 | 215–261 | 2026-06-30 | 进程内第一次调用用了 6,197 ms |
| PE(TTM)/PB | 东方财富数据中心（`stock_value_em`，**新增**） | 3/3 | 317–431 | 2026-09-24 | 替换已被移除的 `stock_a_indicator_lg` |
| PE(TTM)/PB | 腾讯行情（`qt.gtimg.cn`，**新增**） | 3/3 | 43–81 | 2026-09-24 | 第 39/46 字段与东方财富一致（600519：18.99 / 6.15） |
| 行业 | 东方财富 `push2`（`stock_individual_info_em`） | 0/3 | 101–129 | – | 根因 1 |
| 行业 | 巨潮资讯公司概况（`stock_profile_cninfo`，**新增**） | 3/3 | 67–126 | – | 证监会行业分类 |
| 宏观 | 东方财富数据中心（`macro_china_cpi`，**新增**） | 1/1 | 140 | 2026-08 | 统计局数据；CPI 同比 0.8% |
| 宏观 | 东方财富数据中心（`macro_china_pmi`，**新增**） | 1/1 | 136 | 2026-08 | 制造业 PMI 49.8 |
| 宏观 | 东方财富数据中心（`macro_china_money_supply`） | 1/1 | 190 | 2026-08 | M2 同比 7.5% |
| 宏观 | 东方财富数据中心（`bond_zh_us_rate`） | 1/1 | 355 | 2026-09-24 | 10 年期国债 1.6738% |
| 宏观 | 中债（`bond_china_yield`，**新增**） | 1/1 | 2302 | 2026-09-24 | 10 年期 1.6738%，与东方财富一致 |
| 宏观 | 东方财富数据中心（`macro_china_lpr`，**新增**） | 1/1 | 863 | 2026-09-20 | LPR 1 年期 3.0%，5 年期 3.5% |
| 宏观（旧） | 金十（`macro_china_cpi_monthly`） | 1/1 | 23676 | **2025-09-10** | 一年前就停更了（根因 2） |
| 宏观（旧） | 金十（`macro_china_pmi_yearly`） | 1/1 | 18727 | **2025-08-31** | 一年前就停更了 |
| 宏观（旧） | `macro_china_pmi_monthly` | 0/1 | 0 | – | akshare 1.18 里已没有这个函数 |
| 新闻 | 东方财富搜索（`stock_news_em`） | 4/4 | 120–175 | 2026-09-25 | 股票和 ETF 都可用 |
| 公告 | 巨潮资讯（带 orgId 的 `hisAnnouncement/query`） | 3/4 | 191–453 | 2026-09-25 | ETF 510300 没有公告 |
| 公告 | 东方财富公告（`np-anotice-stock`，**新增**） | 3/4 | 62–144 | 2026-09-25 | ETF 510300 没有公告 |
| 宏观（尝试） | 国家统计局 `data.stats.gov.cn/easyquery.htm` | 0/2 | – | – | HTTP 403，WAF `reason:UrlACL`：拒绝脚本客户端（用 `requests` 单独测量，不在审计脚本内） |

## 失败根因

1. **东方财富行情主机断开连接**（`push2his.eastmoney.com`、`push2.eastmoney.com`、`80.push2.eastmoney.com`）。
   - **现象**：本次会话第一次调用 `stock_zh_a_hist` 成功（150 ms），之后每次都失败。用 `curl --noproxy '*'`（「Empty reply from server」）和 `trust_env=False` 的 `requests`（`RemoteDisconnected`）都能复现；经过代理时同一故障表现为 `ProxyError`。加浏览器的 `User-Agent`/`Referer` 请求头也没用。
   - **判断**：这是上游对客户端 IP 的限流，akshare 的 issue 里直到 2026 年都有人反复报告。
   - **影响范围**：efinance 用的是同一个 `push2his` 主机，所以不是独立的备源。东方财富的其他主机（`datacenter-web`、`search-api-web`、`np-anotice-stock`、`fund`/`fundf10`）不受影响。
2. **旧 provider 用的宏观序列已经停更**。
   - **CPI**：取自金十 `macro_china_cpi_monthly`，最新一行是 2025-09-10，值还是 `NaN`（尚未发布的占位符），于是 provider 返回了 `metric_value: NaN`，这不是合法 JSON。
   - **PMI**：因为 `macro_china_pmi_monthly` 已不存在，退到了金十 `macro_china_pmi_yearly`（最新一行 2025-08-31）。
   - **耗时**：整次宏观调用用了 **50.3 秒**。
3. **旧代码与库/接口的漂移**。
   - akshare 移除了 `stock_a_indicator_lg`，所以 `pe_ttm`/`pb` 始终是 `None`。
   - 新浪把 `主营业务毛利率(%)` 改名为 `销售毛利率(%)`，并去掉了 `每股收益(元)`，所以毛利率和 EPS 始终是 `None`。
   - M2 的旧匹配逻辑找不到已知列名，落到了第一个数值列 `货币和准货币(M2)-数量(亿元)`，**把货币存量（3,568,083.6 亿元）当成了百分比增速**。
   - 月度日期 `2026年08月份` 解析不了。
   - `fund_etf_fund_info_em` 用 `symbol=` 调用，而它的参数是 `fund=`。
4. **雪球现在需要登录 token**。`fund_individual_detail_info_xq` 抛出 `KeyError: 'data'`；akshare 自己的雪球公司接口也明说了这一点。
5. **巨潮资讯什么都没返回**。
   - **原因**：不提供公司 `orgId` 时，`hisAnnouncement/query` 会忽略 `stock=<code>,`，返回全市场公告流，再被 provider 的 `secCode` 过滤清空，结果每只股票都是 0 条公告。
   - **修复**：orgId 来自 `information/topSearch/query`（例如 300750 是 `GD165627`；并不总是 `gssh0<code>`）。
6. **旧的 ETF 数据包用了 14.9 秒**。大部分时间花在帮不上忙的接口上：一个在被限流主机上、耗时 5 秒的 `fund_etf_spot_em` 全市场列表，和需要 token 的雪球调用。

## 修复前后对比（端到端，同一台机器，同一天）

| 调用 | 修复前 | 修复后（审计中的降级链运行） |
|---|---|---|
| 个股数据包 600519（价格 + 基本面 + 行业） | 1,803 ms。PE/PB、毛利率和 EPS 都是 `None`。 | 2,062 ms。PE(TTM) 18.99、PB 6.15 有值；EPS 35.57 |
| 个股数据包 300750 / 601318 | 1,582 ms / 未运行 | 1,036 ms / 998 ms（东方财富被熔断器跳过） |
| ETF 数据包 510300 | 14,926 ms | 2,186 ms（进程内首次运行 5,113 ms） |
| 指数数据包 000300 | 367–1,094 ms | 5,286 ms（主要是新浪全量历史的延迟波动） |
| 宏观 CPI/PMI/M2/10 年期国债 | 50,323 ms；CPI 为 `NaN`，PMI 过期一年，M2 单位错 | 2,753 ms（含 LPR）；全部是最新数据 |
| 公告 600519 / 300750 | 0 / 0 条 | 10 / 10 条，275 / 194 ms |
| 新闻 600519 | 121 ms | 111 ms |

「修复前」是同一天早些时候用 `HEAD` 代码对真实接口的运行，「修复后」是审计报告中 `run_chains` 部分。

## 各类数据的降级链

每条链按顺序尝试，第一个返回可用数据的数据源胜出。「可用」指非空；对宏观序列还要求在时效窗口内，所以停更的数据源会被判为 `empty` 而拒绝。

| 数据类型 | 实时主源 | 实时备源（按顺序） | 再往后 |
|---|---|---|---|
| 个股日线 | 东方财富 `stock_zh_a_hist` | 新浪 `stock_zh_a_daily` → 腾讯 fqkline → 新浪实时报价 → efinance | 最近一次成功的数据包 → 仅在仍新鲜时使用随仓库提供的快照 |
| ETF 日线 | 东方财富 `fund_etf_hist_em` | 新浪 `fund_etf_hist_sina` → 腾讯 fqkline → efinance | 同上 |
| 指数日线 | 新浪 `stock_zh_index_daily` | 东方财富 `index_zh_a_hist` → 腾讯 fqkline → 东方财富实时 | 同上 |
| 财务指标 | 新浪 `stock_financial_analysis_indicator` | 同花顺 `stock_financial_abstract_ths` | 快照 |
| PE(TTM) / PB | 东方财富数据中心 `stock_value_em` | 腾讯行情 | – |
| 行业 | 东方财富 `stock_individual_info_em`（+ 板块历史） | 巨潮资讯 `stock_profile_cninfo`（只有行业名） | 快照 |
| 基金费率 / 概况 | 东方财富 `fund_overview_em` | 雪球（已知费率后跳过）→ `fund_etf_fund_info_em` | – |
| CPI / PMI / M2 / LPR | 东方财富数据中心（统计局 / 央行数据） | 没有可用备源（统计局拦截脚本，金十已停更） | 最近一次成功数据（TTL 6 小时，最多沿用 24 小时）→ 快照 |
| 10 年期国债收益率 | 东方财富 `bond_zh_us_rate` | 中债 `bond_china_yield` | 最近一次成功数据 → 快照 |
| 新闻 | 东方财富 `stock_news_em` | – | 本地文档库 |
| 公告 | 巨潮资讯（orgId 查询） | 东方财富公告接口 | 本地文档库 |

排序理由：

- 主源是最精确或最完整的数据源：东方财富的日线带 `涨跌幅` 和 `成交额`，新浪指数日线保留三位小数，腾讯只保留两位。
- 备源按「与主源不同主机」和速度来选。
- efinance 放在最后，因为它和东方财富共用被限流的主机。

## 稳健性机制

所有实时 I/O 都经过 `SourceRuntime.call`（`query_intelligence/integrations/sources/`）：

- **每个数据源一个熔断器**（`health.py`）。
  - **打开与冷却**：连续失败 `QI_SOURCE_FAILURE_THRESHOLD` 次（默认 3）后熔断打开，接下来 `QI_SOURCE_COOLDOWN_SECONDS`（默认 60 秒）内的调用直接短路，不做网络 I/O。
  - **半开试探**：之后放行一次试探，成功就关闭；失败则重新打开，冷却时间翻倍，上限 `QI_SOURCE_MAX_COOLDOWN_SECONDS`（600 秒）。
  - **审计中的效果**：第一只股票在东方财富上失败两次后，后面的股票立即跳过东方财富（`eastmoney.quote:circuit_open`），数据包耗时从 2,062 ms 降到约 1,000 ms。
- **有界线程池上的硬超时**（`runtime.py`，`SourceCallPool`）。
  - **为什么需要**：大多数 akshare 函数不接受 timeout 参数。
  - **机制**：每次受保护的调用都在固定大小线程池（`QI_SOURCE_MAX_WORKERS`，默认 32）的一个 worker 上运行。调用方在 `QI_SOURCE_CALL_TIMEOUT_SECONDS`（默认 10 秒）后不再等待，这次调用记为失败，并计为「被放弃」，直到挂住的连接最终返回、释放 worker。超时不重试。
  - **之前的问题**：以前每次调用都新开一个守护线程，上游挂住时，负载一上来线程数就会无限增长。
  - **现在**：同时最多运行 `QI_SOURCE_MAX_WORKERS` 个调用。遇到所有 worker 都忙的调用会立即被拒绝（`SourcePoolSaturatedError`，尝试标签 `<source>:saturated`），而不是排在挂住的调用后面。池满是本地背压，不计入数据源的熔断。受保护调用内部再发起的受保护调用会在外层超时下同步执行，所以嵌套不会把满池锁死。
  - **可观测**：池的计数器（`busy`、`max_busy`、`timed_out_total`、`abandoned_total`、`abandoned_running`、`rejected_total`）由 `/sources/health` 的 `worker_pool` 返回，并导出到 Prometheus（见 [A2A、容灾与可观测性](a2a-and-observability.md)）。
- **带「最近一次成功数据」读取的 TTL 缓存**（`cache.py`）。
  - **TTL**：行情数据包 60 秒，日频宏观序列 30 分钟，月度序列 6 小时。
  - **兜底**：所有实时数据源都失败时，返回不超过 `QI_SOURCE_MAX_STALE_SECONDS`（默认 24 小时）的过期缓存，标为 `last_known_good`，并加上 `market_served_last_known_good` 警告。
  - `QI_SOURCE_CACHE=0` 关闭缓存。
- **快照策略**。
  - **关闭实时数据时**：直接返回随仓库提供的 `data/structured_data.json` 记录，标为 `snapshot`。
  - **实时价格获取*失败*时**：只有快照价格仍然新鲜（不超过 10 天）才会使用。随仓库的价格是 2026-04 的，所以不会被当作今天的报价，而是以 `provider_warning` 返回失败。
  - **基本面和宏观快照**：仍然可用，但会被标为 `stale`。
- **数值清洗**（`values.py`）。
  - **规范化**：`NaN`、`--`/`---` 占位符、百分比字符串和 `亿`/`万` 量级都会被规范化，所以不会有 `NaN` 进入 JSON。
  - **成交量单位**：每个数据源都注明 `volume_unit`（`lot` 或 `share`）。以 2026-09-24 的日线核对：600519 新浪 3,123,900 股，腾讯 31,239 手。

## 每条记录的来源标注

每个结构化 payload 和每篇文档都带 `payload.provenance`，工具输出里也有同样的可选 `provenance` 字段。下面的形状只是示意，取值来自审计中 300750 那条链。

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

- `mode` 取 `live`、`live_fallback`、`last_known_good` 或 `snapshot`。
- `freshness` 按数据类型的时效窗口比较 `as_of`：日频 10 天，月度宏观 75 天，财报 200 天。
- **不含数值**：这个对象只有字符串和布尔值，紧凑日期会改写成 ISO 日期。所以来源标注永远不会让编造的数字在 Agent 的数字校验里看起来「可追溯」。
- **不影响覆盖率统计**：检索打包器把 `provenance`、`valuation_provenance` 和 `volume_unit` 当作元数据，`field_coverage` 和 `quality_flags` 不变。
- **离线记录的说明**：写的是 `数据来自离线快照，截至2026-03-31，非实时（离线快照），数据可能已过时；因未开启实时宏观数据降级`。

## 基本面的跨源核对

第一轮评审发现（缺陷 B11），000858.SZ 的实时新浪基本面给出 2026 年上半年营收同比 −46.15%、净利润同比 −55.32%，而同一个答案里检索到的一条新闻写的是 +20.87% 和 +89.30%。2026-09-26 同时取两个数据源，就能看出分歧的来源：

| 报告期 | 同花顺营收（亿元） | 同花顺营收同比 | 新浪营收同比 | 同花顺净利润同比 | 新浪净利润同比 |
|---|---:|---:|---:|---:|---:|
| 2025-06-30 | 235.10 | -53.58% | +4.19% | -75.74% | +1.56% |
| 2026-03-31 | 228.38 | +33.67% | -38.18% | +82.57% | -45.84% |
| 2026-06-30 | 284.17 | +20.87% | -46.15% | +89.30% | -55.32% |

- **同花顺的增速自洽**：与它自己报告的绝对值一致（284.17 / 235.10 − 1 = +20.87%），也与新闻引用的公司数据一致。
- **新浪的增速对不上**：与这些绝对值不一致。
- **旧逻辑的问题**：provider 用新浪，只在新浪失败时才退到同花顺，所以只要新浪有响应，给出的就是错数。

现在（`integrations/sources/crosscheck.py`，默认开启，`QI_SOURCE_CROSS_CHECK=1`）每个个股数据包都并发取两个数据源并做核对：

1. **范围检查**：超出合理范围的值（营收同比低于 −100%、ROE 超过 ±200%、毛利率超过 ±100% 等）会被丢弃并列入 `out_of_range`，绝不对外提供。
2. **报告期**：只比较或合并同一报告期。最新报告期不同时，提供较新的那个，核对结果记为 `period_mismatch`。
3. **累计口径与单季口径**：A 股报告是年初至今累计值。
   - **绝对值检查**：同花顺的绝对值要在同一财年内单调不减。
   - **增速重算**：每个报告的同比都和由绝对值重算出的同比做比较，既按累计口径，也按单季口径（例如 Q2 = H1 − Q1）。
   - **口径识别**：报告单季增速的数据源会被识别出来（`conventions: {"sina.finance:revenue_yoy": "single_quarter"}`），而不是被判为错误。
4. **裁决**：
   - 两个增速相差 2 个百分点以内视为一致。
   - 否则，采用与累计口径重算值相符（1 个百分点以内）的那个。
   - 两个都无法确认时，采用主源的值，状态记为 `disagree_unresolved`。
   - 同一报告期里一方缺的字段（例如新浪缺失的毛利率）用另一方补齐，列入 `filled_from_other`。

核对结果记录在 Agent 和前端本来就会看的地方：

- `provenance.cross_check`：`status`（`agree`、`disagree_resolved`、`disagree_unresolved`、`period_mismatch`、`single_source`）、`served_source`、`compared_with`、`disagreeing_fields`、`resolution`、`conventions`、`level_consistent`，以及一句中文 `note`。和来源标注的其他部分一样，它只含字符串、布尔值和 ISO 日期，不会让数字看起来「可追溯」。
- `provenance.note` 会补充说明，例如 `新浪财经与同花顺的营收同比、净利润同比不一致；已采用与报告期营收/净利润绝对值推算结果一致的同花顺数据，请以公司定期报告为准`。
- 一条 provider 警告，例如 `fundamentals_cross_source_disagree_resolved:000858:revenue_yoy,netprofit_yoy:served=ths.finance`，检索打包器会把它加进结果的 warnings。

代价：每个个股数据包多一次上游调用，与新浪调用并发（审计中同花顺为 215–261 ms），数据包缓存 60 秒。

## 过期的快照行业记录

问题：
- **旧的快照数据**：随仓库的快照里有按板块名存的行业数据（`白酒` 日期为 2026-04-21，`保险` 等），通过 `entity_to_industry` 查到。
- **命名对不上**：实时 provider 的行业名不一样（东方财富是 `酿酒行业`，巨潮资讯是 `酒、饮料和精制茶制造业`），所以实时记录从来替换不了快照。
- **后果**：在实时答案里，4 月的行业涨跌会出现在 9 月的价格旁边。

现在开启实时数据时：
- **刷新**：检索流水线会用同花顺行业指数（`stock_board_industry_index_ths`，数据源 `ths.industry`，2026-09-26 实测 170–440 ms；日涨跌由最近两个收盘价算出）刷新过期的快照行业记录，缓存 5 分钟。快照里的字段（PE、PB、换手率）不会带进实时记录。
- **刷新失败时**：保留快照但加以标注，快照来源的原因写 `live industry index unavailable`（`实时行业指数不可用，沿用离线快照（旧数据，勿当作今日行情）`），并加一条 `industry_snapshot_stale:<board>:<date>` 警告。

关闭实时数据时一切照旧。

## 健康检查接口

`GET /sources/health` 默认是被动的：只报告运行时已经记录下来的情况，从不调用上游。每个数据源返回：

- 状态（`up`、`degraded`、`down` 或 `unknown`）和熔断状态；
- 调用、成功、失败次数，最近一次和平均延迟，最近一次错误；
- 熔断打开时的 `retry_in_s`；
- 熔断和缓存配置，以及数据源调用池的计数器（`worker_pool`）。

**主动探测（需显式开启）。**
- **怎么探**：`GET /sources/health?probe=1` 先对每个数据源发一次轻量请求：东方财富行情/数据中心/新闻、新浪日线/报价、腾讯日线/报价、同花顺财务/行业、巨潮资讯概况。请求走同一个受保护的 `SourceRuntime.call`，所以熔断、延迟和错误的记录方式和真实流量完全一样。
- **返回什么**：报告里会多一个 `probe` 块，含每个数据源的 `ok`、`outcome`、`latency_ms`、`error`。
- **限频**：好几个上游会对同一 IP 的突发请求限流，所以探测在进程内限频：每 `QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS`（默认 60）最多一轮。
  - 窗口内再请求，返回上一轮结果，并带 `status: rate_limited` 和 `retry_in_s`；
  - 正在探测时请求，得到 `in_progress`；
  - 关闭实时行情时，探测为 `skipped`。

## 配置

| 变量 | 默认值 | 含义 |
|---|---|---|
| `QI_USE_LIVE_MARKET` / `_NEWS` / `_ANNOUNCEMENT` / `_MACRO` | `true` | 开启实时 provider（未变） |
| `QI_SOURCE_CALL_TIMEOUT_SECONDS` | `10` | 每次上游调用的硬超时（墙钟时间） |
| `QI_SOURCE_FAILURE_THRESHOLD` | `3` | 打开熔断所需的连续失败次数 |
| `QI_SOURCE_COOLDOWN_SECONDS` | `60` | 熔断第一次打开的冷却时间 |
| `QI_SOURCE_MAX_COOLDOWN_SECONDS` | `600` | 多次试探失败后冷却时间的上限 |
| `QI_SOURCE_CACHE` | `true` | TTL 缓存与「最近一次成功数据」读取 |
| `QI_SOURCE_MAX_STALE_SECONDS` | `86400` | 可以沿用的最旧「最近一次成功数据」 |
| `QI_SOURCE_MAX_WORKERS` | `32` | 有界数据源调用池的大小 |
| `QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS` | `60` | 两轮主动探测之间的最短间隔 |
| `QI_SOURCE_CROSS_CHECK` | `true` | 同时取新浪和同花顺基本面并核对 |

## 测试

- `tests/test_data_sources.py`（离线，默认运行）覆盖：
  - 熔断状态转换：打开、短路、半开和冷却翻倍；
  - 硬超时；
  - 缓存 TTL、过期读取和副本隔离；
  - 链的顺序与降级原因；
  - 腾讯备源，以及熔断器停止调用被屏蔽的数据源；
  - 新增的基本面、估值、基金和行业数据源；
  - 宏观的列名/NaN/过期处理和中债备源；
  - 巨潮资讯 orgId 查询和公告备源；
  - 流水线缓存、最近一次成功数据、快照的新鲜/过期策略，以及工具输出上的来源标注；
  - `/sources/health`。
- `tests/test_source_reliability.py`（离线）覆盖：
  - 有界线程池：被放弃调用的计数、快速拒绝且不触发熔断、上游挂住时的线程数上限、嵌套调用；
  - 限频的主动探测；
  - 用 000858 的真实数据测新浪/同花顺核对：范围、报告期、累计与单季口径、不含数字的元数据；
  - 过期行业数据的刷新与标注；
  - 熔断状态和线程池的 Prometheus 采集器。
- `tests/test_data_sources_live.py` 访问真实接口，只在 `QI_LIVE_TESTS=1`（或仓库已有的 `QI_INTEGRATION_TESTS=1`）时运行。其他测试默认关闭实时数据源（除非 `QI_TEST_LIVE=1`）。

## 已知局限

- **宏观没有备源**：CPI、PMI、M2 和 LPR 没有独立的实时备源。统计局官方接口对脚本返回 403，金十序列已停更一年。东方财富数据中心不可用时，这些指标来自最近一次成功数据或（过期且已标注的）快照。
- **毛利率可能缺失**：审计的几只股票，新浪最新半年报这一行的 `销售毛利率(%) = NaN`，所以除非这份报告由同花顺提供，否则 `grossprofit_margin` 为 `null`。它会被报告为缺失，而不是借用其他报告期的值。
- **限流因网而异**：东方财富行情主机的限流取决于客户端 IP，换一个网络主源可能就能成功；降级链和熔断两种情况都能处理。
- **Tushare 未审计**：没有 `TUSHARE_TOKEN`，所以没审计 Tushare；它的记录只带通用的实时来源标注。
- **ETF 没有公告**：两个数据源上都是 0 条（510300）。
- **指数估值覆盖有限**：中证指数估值只覆盖中证系列指数。深交所的 399006 创业板指在一次实时流水线运行中请求失败，所以 `index_valuation` 没有值，其来源说明写的是「未获取到实时数据」。
- **来源标注没进 `evidence_sources`**：Agent API 的 `evidence_sources` 列表在 `agent/graph.py` 里构建，还没有从证据 payload 复制 `provenance`；这些数据在工具结果和证据 payload 里都有。
