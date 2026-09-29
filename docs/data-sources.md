# Live Data Sources: Audit, Fallback Chains, and Provenance

Languages: English | [中文](zh/data-sources.md)

This document records a real-network audit of every live data source FinSight uses, the bugs it
exposed, and the acquisition design that now sits in front of those sources: ordered fallback chains,
per-source circuit breakers, a TTL cache with last-known-good reads, hard timeouts, and provenance
metadata on every returned record.

All numbers below were measured, not estimated. Upstream behaviour changes (Eastmoney in particular
throttles bursts from one IP), so rerun the audit before quoting them.

## How the audit was run

| Item | Value |
|---|---|
| Date | 2026-09-25 (a market holiday, Mid-Autumn Festival; the latest trading day was 2026-09-24) |
| Command | `python -m scripts.audit_data_sources --json outputs/data_source_audit.json` |
| Code | commit `c1c3388` plus the working-tree changes described here |
| Versions | Python 3.13.13, akshare 1.18.97, efinance 0.5.9, requests 2.34.2, pandas 2.3.3 |
| Network | Residential connection in mainland China with a local HTTP proxy (`127.0.0.1:6152`) in the environment |
| Method | Each source is called on its own: breaker disabled, no retries, no cache, 15 s hard timeout. The runtime chains are then run end to end with a fresh breaker. |
| Entities | 600519 贵州茅台, 300750 宁德时代, 601318 中国平安, ETF 510300, index 000300, macro series CPI / PMI / M2 / CN10Y / LPR |

Result: 64 probes, 49 succeeded. Every failure has an identified root cause (below).

**Committed re-run.** The table above comes from a run on a dirty tree whose JSON was not kept. The
audit was re-run on 2026-09-28 09:14 UTC from a clean checkout of commit `6dde495` (same network and
package versions) and the full output is committed:
[`results/data_sources/audit-20260928-6dde495.json`](results/data_sources/audit-20260928-6dde495.json)
(`commit` and `working_tree_clean: true` are recorded in the file). It gives the same totals: **49/64
probes OK, 10/10 fallback chains OK** (every market bundle served by Sina). The failures are the same
families: Eastmoney `push2`/`push2his` hosts 0/11 (proxy error, root cause 1; efinance uses the same
host), Xueqiu 0/1 (login token), the removed `macro_china_pmi_monthly`, and one empty announcement
result each for cninfo and the Eastmoney notice API (the ETF, which has no company announcements).

**Schedule.** [`.github/workflows/data-source-audit.yml`](../.github/workflows/data-source-audit.yml)
runs the audit every Monday at 01:30 UTC and on demand (`workflow_dispatch`), writes a summary table to
the job page and uploads the JSON as an artifact for 90 days. GitHub's runners are outside mainland
China, so their results are not comparable with the table above; audits quoted in the docs are run
from the deployment network and committed under `docs/results/data_sources/`.

## Per-source results

Latency is the observed range across targets. "Newest as-of" is the latest date in the returned
rows. Results come from the audit run above.

| Kind | Source (function) | OK | Latency ms | Newest as-of | Failure / note |
|---|---|---|---:|---|---|
| Stock daily | Eastmoney `push2his` (`stock_zh_a_hist`) | 0/3 | 109–348 | – | Connection dropped (see root cause 1) |
| Stock daily | Sina (`stock_zh_a_daily`) | 3/3 | 261–512 | 2026-09-24 | Volume in shares |
| Stock daily | Tencent (`web.ifzq.gtimg.cn` fqkline, **new**) | 3/3 | 97–104 | 2026-09-24 | Volume in lots (手) |
| Stock quote | Sina realtime (`hq.sinajs.cn`) | 3/3 | 40–57 | 2026-09-24 | One row, no history |
| Stock daily | efinance (`get_quote_history`) | 0/3 | 763–1076 | – | Same `push2his` host as Eastmoney |
| ETF daily | Eastmoney (`fund_etf_hist_em`) | 0/1 | 123 | – | Root cause 1 |
| ETF daily | Sina (`fund_etf_hist_sina`) | 1/1 | 279 | 2026-09-24 | |
| ETF daily | Tencent fqkline (**new**) | 1/1 | 98 | 2026-09-24 | |
| ETF NAV | Eastmoney fund (`fund_open_fund_info_em`) | 1/1 | 140 | 2026-09-24 | |
| ETF profile/fees | Eastmoney fund (`fund_overview_em`, **new**) | 1/1 | 203 | – | Fees, manager, tracking index |
| ETF profile/fees | Xueqiu (`fund_individual_detail_info_xq`) | 0/1 | 1392 | – | `KeyError: 'data'`: now needs a login token (root cause 4) |
| Index daily | Sina (`stock_zh_index_daily`) | 1/1 | 246 | 2026-09-24 | Returns full history (~6,000 rows); a later run took 3,418 ms |
| Index daily | Eastmoney (`index_zh_a_hist`) | 0/1 | 5451 | – | Root cause 1 (`80.push2.eastmoney.com`) |
| Index daily | Tencent fqkline (**new**) | 1/1 | 99 | 2026-09-24 | |
| Index valuation | CSIndex (`stock_zh_index_value_csindex`) | 1/1 | 288 | 2026-09-24 | A later run took about 1,600 ms |
| Financials | Sina (`stock_financial_analysis_indicator`) | 3/3 | 344–391 | 2026-06-30 | Column renames (root cause 3) |
| Financials | THS (`stock_financial_abstract_ths`, **new**) | 3/3 | 215–261 | 2026-06-30 | First call in a process took 6,197 ms |
| PE(TTM)/PB | Eastmoney datacenter (`stock_value_em`, **new**) | 3/3 | 317–431 | 2026-09-24 | Replaces the removed `stock_a_indicator_lg` |
| PE(TTM)/PB | Tencent quote (`qt.gtimg.cn`, **new**) | 3/3 | 43–81 | 2026-09-24 | Fields 39/46 match Eastmoney (600519: 18.99 / 6.15) |
| Industry | Eastmoney `push2` (`stock_individual_info_em`) | 0/3 | 101–129 | – | Root cause 1 |
| Industry | cninfo profile (`stock_profile_cninfo`, **new**) | 3/3 | 67–126 | – | CSRC industry classification |
| Macro | Eastmoney datacenter (`macro_china_cpi`, **new**) | 1/1 | 140 | 2026-08 | NBS figures; CPI YoY 0.8 % |
| Macro | Eastmoney datacenter (`macro_china_pmi`, **new**) | 1/1 | 136 | 2026-08 | Manufacturing PMI 49.8 |
| Macro | Eastmoney datacenter (`macro_china_money_supply`) | 1/1 | 190 | 2026-08 | M2 YoY 7.5 % |
| Macro | Eastmoney datacenter (`bond_zh_us_rate`) | 1/1 | 355 | 2026-09-24 | CN 10Y 1.6738 % |
| Macro | ChinaBond (`bond_china_yield`, **new**) | 1/1 | 2302 | 2026-09-24 | 10Y 1.6738 %, identical to Eastmoney |
| Macro | Eastmoney datacenter (`macro_china_lpr`, **new**) | 1/1 | 863 | 2026-09-20 | LPR 1Y 3.0 %, 5Y 3.5 % |
| Macro (old) | jin10 (`macro_china_cpi_monthly`) | 1/1 | 23676 | **2025-09-10** | Feed stopped a year ago (root cause 2) |
| Macro (old) | jin10 (`macro_china_pmi_yearly`) | 1/1 | 18727 | **2025-08-31** | Feed stopped a year ago |
| Macro (old) | `macro_china_pmi_monthly` | 0/1 | 0 | – | Function no longer exists in akshare 1.18 |
| News | Eastmoney search (`stock_news_em`) | 4/4 | 120–175 | 2026-09-25 | Works for stocks and the ETF |
| Announcements | cninfo (`hisAnnouncement/query` with orgId) | 3/4 | 191–453 | 2026-09-25 | Nothing for ETF 510300 |
| Announcements | Eastmoney notices (`np-anotice-stock`, **new**) | 3/4 | 62–144 | 2026-09-25 | Nothing for ETF 510300 |
| Macro (tried) | National Bureau of Statistics `data.stats.gov.cn/easyquery.htm` | 0/2 | – | – | HTTP 403 WAF `reason:UrlACL`: rejects scripted clients (measured separately with `requests`, not part of the audit script) |

## Root causes of the failures

1. **Eastmoney quote hosts drop connections** (`push2his.eastmoney.com`, `push2.eastmoney.com`,
   `80.push2.eastmoney.com`). The first `stock_zh_a_hist` call of the session succeeded (150 ms); every
   later call failed. The failure reproduced with `curl --noproxy '*'` ("Empty reply from server") and
   with `requests` using `trust_env=False` (`RemoteDisconnected`). With the proxy in the path the same
   failure surfaces as `ProxyError`. Browser `User-Agent`/`Referer` headers did not help. This is
   upstream throttling of the client IP, a pattern reported repeatedly in akshare issues through 2026.
   efinance uses the same `push2his` host, so it is not an independent fallback. Eastmoney's other
   hosts (`datacenter-web`, `search-api-web`, `np-anotice-stock`, `fund`/`fundf10`) were unaffected.
2. **Stale macro series in the old provider.** It read CPI from jin10 `macro_china_cpi_monthly`, whose
   newest row is 2025-09-10. The newest row's value was `NaN` (an unreleased placeholder), so the
   provider returned `metric_value: NaN`, which is not valid JSON. PMI fell back to jin10
   `macro_china_pmi_yearly` (newest row 2025-08-31) because `macro_china_pmi_monthly` no longer exists.
   The whole macro call took **50.3 s**.
3. **Library/API drift in the old code.**
   - `stock_a_indicator_lg` was removed from akshare, so `pe_ttm`/`pb` were always `None`.
   - Sina renamed `主营业务毛利率(%)` to `销售毛利率(%)` and dropped `每股收益(元)`, so gross margin and
     EPS were always `None`.
   - For M2 the old matcher found no known column name. It fell through to the first numeric column,
     `货币和准货币(M2)-数量(亿元)`, and **reported the money stock (3,568,083.6 亿元) as a percentage growth
     rate**.
   - The monthly dates `2026年08月份` were not parsed.
   - `fund_etf_fund_info_em` was called with `symbol=` although its parameter is `fund=`.
4. **Xueqiu now requires a login token.** `fund_individual_detail_info_xq` raises `KeyError: 'data'`;
   akshare's own Xueqiu company endpoint says so explicitly.
5. **cninfo returned nothing.** `hisAnnouncement/query` ignores `stock=<code>,` unless the company
   `orgId` is supplied. It then returns the whole-market feed, which the provider's `secCode` filter
   emptied: 0 announcements for every stock. The orgId comes from `information/topSearch/query`
   (for example `GD165627` for 300750; not always `gssh0<code>`).
6. **The old ETF bundle took 14.9 s.** Much of that was spent on endpoints that could not help: a 5 s
   `fund_etf_spot_em` whole-market list on a throttled host, and the token-gated Xueqiu call.

## Before and after (end-to-end, same machine, same day)

| Call | Before | After (audit chain run) |
|---|---|---|
| Stock bundle 600519 (price + fundamentals + industry) | 1,803 ms. PE/PB, gross margin and EPS were `None`. | 2,062 ms. PE(TTM) 18.99 and PB 6.15 filled; EPS 35.57 |
| Stock bundle 300750 / 601318 | 1,582 ms / not run | 1,036 ms / 998 ms (Eastmoney skipped by the breaker) |
| ETF bundle 510300 | 14,926 ms | 2,186 ms (first run in a process: 5,113 ms) |
| Index bundle 000300 | 367–1,094 ms | 5,286 ms (dominated by Sina full-history latency variance) |
| Macro CPI/PMI/M2/CN10Y | 50,323 ms; CPI `NaN`, PMI a year stale, M2 wrong unit | 2,753 ms including LPR; all current |
| Announcements 600519 / 300750 | 0 / 0 items | 10 / 10 items, 275 / 194 ms |
| News 600519 | 121 ms | 111 ms |

"Before" is the `HEAD` code run against live endpoints earlier the same day. "After" is the
`run_chains` section of the audit report.

## Fallback chains per data kind

Each chain is tried in order. The first source that returns usable data wins. "Usable" means non-empty,
and for macro series also within the freshness window, so a dead feed is rejected as `empty`.

| Data kind | Live primary | Live secondaries (in order) | Then |
|---|---|---|---|
| Stock daily bars | Eastmoney `stock_zh_a_hist` | Sina `stock_zh_a_daily` → Tencent fqkline → Sina realtime quote → efinance | last-known-good bundle → shipped snapshot only if still fresh |
| ETF daily bars | Eastmoney `fund_etf_hist_em` | Sina `fund_etf_hist_sina` → Tencent fqkline → efinance | same |
| Index daily bars | Sina `stock_zh_index_daily` | Eastmoney `index_zh_a_hist` → Tencent fqkline → Eastmoney spot | same |
| Financial indicators | Sina `stock_financial_analysis_indicator` | THS `stock_financial_abstract_ths` | shipped snapshot |
| PE(TTM) / PB | Eastmoney datacenter `stock_value_em` | Tencent quote | – |
| Industry | Eastmoney `stock_individual_info_em` (+ board history) | cninfo `stock_profile_cninfo` (industry name only) | shipped snapshot |
| Fund fees / profile | Eastmoney `fund_overview_em` | Xueqiu (skipped once fees are known) → `fund_etf_fund_info_em` | – |
| CPI / PMI / M2 / LPR | Eastmoney datacenter (NBS / PBoC figures) | none available (NBS blocks scripts, jin10 is stale) | last-known-good (6 h TTL, 24 h stale) → shipped snapshot |
| CN 10Y yield | Eastmoney `bond_zh_us_rate` | ChinaBond `bond_china_yield` | last-known-good → shipped snapshot |
| News | Eastmoney `stock_news_em` | – | local document corpus |
| Announcements | cninfo (orgId query) | Eastmoney notices API | local document corpus |

Ordering rationale:

- The primary is the most precise or complete source. Eastmoney bars include `涨跌幅` and `成交额`; Sina
  index bars keep three decimals, while Tencent rounds to two.
- Secondaries are chosen for independence from the primary's host and for speed.
- efinance stays last because it shares Eastmoney's throttled host.

## Robustness mechanisms

All live I/O goes through `SourceRuntime.call` (`query_intelligence/integrations/sources/`):

- **Circuit breaker per source** (`health.py`). After `QI_SOURCE_FAILURE_THRESHOLD` (default 3)
  consecutive failures the circuit opens. Calls are then short-circuited without network I/O for
  `QI_SOURCE_COOLDOWN_SECONDS` (default 60 s). Next, one half-open trial is admitted: success closes
  the circuit, while failure re-opens it with a doubled cooldown, capped by
  `QI_SOURCE_MAX_COOLDOWN_SECONDS` (600 s). The audit shows the effect: after the first stock had
  failed twice on Eastmoney, the next stocks skipped Eastmoney immediately
  (`eastmoney.quote:circuit_open`), and their bundles dropped from 2,062 ms to about 1,000 ms.
- **Hard timeouts on a bounded pool** (`runtime.py`, `SourceCallPool`). Most akshare functions accept
  no timeout, so each guarded call runs on a worker of a fixed-size pool (`QI_SOURCE_MAX_WORKERS`,
  default 32). The caller stops waiting after `QI_SOURCE_CALL_TIMEOUT_SECONDS` (default 10 s); the
  call is recorded as a failure and counted as *abandoned* until the hung socket finally returns and
  frees its worker. Timeouts are not retried. Before this change every call started a new daemon
  thread, so a hanging upstream under load grew the thread count without bound. Now at most
  `QI_SOURCE_MAX_WORKERS` calls run at once; a call that finds every worker busy is rejected at once
  (`SourcePoolSaturatedError`, attempt label `<source>:saturated`) instead of queueing behind hung
  calls. Saturation is local back-pressure, so it does not count against the source's breaker. A
  guarded call made from inside another guarded call runs inline under the outer timeout, so nesting
  cannot deadlock a full pool. Pool counters (`busy`, `max_busy`, `timed_out_total`,
  `abandoned_total`, `abandoned_running`, `rejected_total`) are served by `/sources/health` under
  `worker_pool` and exported to Prometheus (see [a2a-and-observability.md](a2a-and-observability.md)).
- **TTL cache with last-known-good reads** (`cache.py`). The TTLs are 60 s for market bundles,
  30 min for daily macro series, and 6 h for monthly ones. When every live source fails, an expired
  entry up to `QI_SOURCE_MAX_STALE_SECONDS` old (default 24 h) is served. It is marked
  `last_known_good` and a `market_served_last_known_good` warning is added. `QI_SOURCE_CACHE=0`
  disables the cache.
- **Snapshot policy.** When live sources are disabled, the shipped `data/structured_data.json` records
  are served and labelled `snapshot`. When a live price fetch *fails*, the snapshot price is used only
  if it is still fresh (at most 10 days old). The shipped prices are from 2026-04, so they are not
  presented as today's quote; the failure is returned as a `provider_warning` instead. Fundamentals
  and macro snapshots remain available but are flagged `stale`.
- **Value hygiene** (`values.py`). `NaN`, `--`/`---` placeholders, percent strings, and `亿`/`万`
  magnitudes are normalized, so no `NaN` reaches JSON. `volume_unit` (`lot` or `share`) is stated per
  source. Verified for the 2026-09-24 bars: 600519 Sina 3,123,900 shares vs Tencent 31,239 lots.

## Provenance on every record

Every structured payload and every document carries `payload.provenance`. Tool outputs expose the
same object as an optional `provenance` field. The shape below is illustrative; the values follow
the 300750 chain in the audit run.

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

- `mode` is one of `live`, `live_fallback`, `last_known_good`, or `snapshot`.
- `freshness` compares `as_of` with a per-kind window: 10 days for daily data, 75 for monthly macro,
  and 200 for financial reports.
- The object contains only strings and booleans, and compact dates are rewritten as ISO dates.
  Provenance therefore can never make an invented number look "traceable" to the agent's
  numeric-faithfulness check.
- The retrieval packager treats `provenance`, `valuation_provenance` and `volume_unit` as metadata, so
  `field_coverage` and `quality_flags` are unchanged.
- Offline records say `数据来自离线快照，截至2026-03-31，非实时（离线快照），数据可能已过时；因未开启实时宏观数据降级`.

### Intraday quotes for 今天/今日/today questions

The daily chains answer with daily bars. A price question that says 今天, 今日 or "today" asks
`get_price_history` for `intraday=true` (only when live market data is on). The tool first checks the
Beijing-time session (`query_intelligence/integrations/intraday.py`). The market is open Monday–Friday
09:30–11:30 and 13:00–15:00. The 11:30–13:00 lunch break also counts: the quote is the morning's last
trade, labelled with its time.

- **Market open.** The tool fetches a real-time quote: Sina `hq.sinajs.cn`, then Tencent `qt.gtimg.cn`.
  It rejects a quote that is not dated today. The output has these fields:
  - `price_basis: "intraday"`;
  - an `intraday` block with price, previous close, change and `quote_time` (ISO with `+08:00`);
  - its own `intraday_provenance` (source, endpoint, `fetched_at`, `as_of` = quote time, attempts).

  The evidence `as_of` is the quote time. The answer says "盘中实时价为 …（HH:MM:SS 北京时间，盘中价格，
  非收盘价）". A limitation adds that it is not a close and will change until the close.
- **Otherwise** the tool keeps the daily close (`price_basis: "daily_close"`) and records why in
  `basis_reason`:
  - `outside_trading_hours`;
  - `intraday_unavailable`: no live source, as with the offline snapshot;
  - `intraday_failed: …`: both real-time sources failed.

  The answer then says so, e.g. "当前为非交易时段，以下为最近交易日收盘价（YYYY-MM-DD）" or
  "盘中实时行情获取失败，以下为最近交易日收盘价…". On weekends and fixed-date holidays it says
  "今天不是 A 股常规交易日".
- **Holidays.** Holiday detection reuses `_is_known_non_trading_day`, which covers weekends plus the fixed
  closures (New Year, Labour Day, National Day). Movable holidays are not listed. On those days the
  real-time quote is dated an earlier day, so it is rejected and the daily close is used, with the reason
  stated.
- **Evaluation.** Offline runs never request `intraday`. The argument is omitted from the normalised call
  key when false, so recorded snapshots and the gate are unchanged.
- **Tests.** `tests/test_intraday_quote.py` injects a frozen clock (`ToolContext.clock`) and stub providers:
  - trading-hours morning → intraday quote with provenance, and the answer verifies;
  - before 09:30, after 15:00 and a weekend → daily close with the stated reason;
  - a failing real-time source → daily close, "获取失败";
  - an English "today" question;
  - a non-今天 question is unaffected;
  - Sina/Tencent parsing and rejection of stale quotes.

## Cross-source validation of fundamentals

The round-1 review found that live Sina fundamentals for 000858.SZ gave H1-2026 revenue YoY -46.15%
and net-profit YoY -55.32%, while a retrieved news item in the same answer said +20.87% and +89.30%
(bug B11). Fetching both sources on 2026-09-26 shows where the disagreement comes from:

| Report period | THS revenue (亿元) | THS revenue YoY | Sina revenue YoY | THS net profit YoY | Sina net profit YoY |
|---|---:|---:|---:|---:|---:|
| 2025-06-30 | 235.10 | -53.58% | +4.19% | -75.74% | +1.56% |
| 2026-03-31 | 228.38 | +33.67% | -38.18% | +82.57% | -45.84% |
| 2026-06-30 | 284.17 | +20.87% | -46.15% | +89.30% | -55.32% |

THS's growth rates agree with its own reported levels (284.17 / 235.10 - 1 = +20.87%), and with the
company figures quoted by the news. Sina's rates are not consistent with those levels. Before this
change the provider used Sina and only fell back to THS when Sina failed, so the wrong figures were
served whenever Sina answered.

Now (`integrations/sources/crosscheck.py`, on by default with `QI_SOURCE_CROSS_CHECK=1`) both
sources are fetched concurrently for every stock bundle and reconciled:

1. **Range checks.** Values outside plausible bounds (revenue YoY below -100%, ROE beyond ±200%,
   gross margin beyond ±100%, …) are dropped and listed in `out_of_range`, never served.
2. **Report period.** Only the same report period is compared or merged. If the latest periods differ,
   the newer one is served and the check says `period_mismatch`.
3. **Cumulative vs single-quarter convention.** A-share reports are cumulative year to date. THS
   levels are checked to be non-decreasing within each fiscal year, and each reported YoY is compared
   with the YoY recomputed from the levels both cumulatively and for the single quarter
   (e.g. Q2 = H1 - Q1). A source that reports single-quarter growth is recognised
   (`conventions: {"sina.finance:revenue_yoy": "single_quarter"}`) instead of being called wrong.
4. **Resolution.** Two growth rates within 2 percentage points agree. Otherwise the value that matches
   the level-recomputed cumulative YoY (within 1 point) is served; if neither can be confirmed, the
   primary's value is served and the status is `disagree_unresolved`. Same-period gaps (for example
   Sina's missing gross margin) are filled from the other source and listed in `filled_from_other`.

The outcome is recorded where the agent and the UI already look:

- `provenance.cross_check`: `status` (`agree`, `disagree_resolved`, `disagree_unresolved`,
  `period_mismatch`, `single_source`), `served_source`, `compared_with`, `disagreeing_fields`,
  `resolution`, `conventions`, `level_consistent`, and a Chinese `note`. Like the rest of provenance
  it contains only strings, booleans and ISO dates, so it cannot make a number look "traceable".
- `provenance.note` is extended, e.g. `新浪财经与同花顺的营收同比、净利润同比不一致；已采用与报告期营收/净利润绝对值推算结果一致的同花顺数据，请以公司定期报告为准`.
- A provider warning such as
  `fundamentals_cross_source_disagree_resolved:000858:revenue_yoy,netprofit_yoy:served=ths.finance`,
  which the retrieval packager adds to the result's warnings.

Cost: one extra upstream call per stock bundle, run concurrently with the Sina call (THS measured
215–261 ms in the audit), and bundles are cached for 60 s.

## Stale snapshot industry records

The shipped snapshot has industry tiles keyed by board name (`白酒` dated 2026-04-21, `保险`, …),
reached through `entity_to_industry`. The live provider names industries differently (Eastmoney
`酿酒行业`, cninfo `酒、饮料和精制茶制造业`), so the live record never replaced the snapshot tile, and
an April industry change sat next to September prices in live answers.

With live data enabled the pipeline now refreshes a stale snapshot industry record from the THS
industry index (`stock_board_industry_index_ths`, source `ths.industry`, 170–440 ms measured on
2026-09-26; the daily change is computed from the last two closes) and caches it for 5 minutes. No
snapshot field (PE, PB, turnover) is carried into the live record. If the refresh fails the snapshot
is kept but labelled: snapshot provenance with reason `live industry index unavailable`
(`实时行业指数不可用，沿用离线快照（旧数据，勿当作今日行情）`) and an
`industry_snapshot_stale:<board>:<date>` warning. With live data off nothing changes.

## Health endpoint

`GET /sources/health` is passive by default: it reports what the runtime has recorded and never calls
an upstream. For each source it returns status (`up`, `degraded`, `down`, or `unknown`), circuit
state, call/success/failure counts, last and average latency, last error, `retry_in_s` for open
circuits, the breaker and cache configuration, and the source-call pool counters (`worker_pool`).

**Active probe (opt-in).** `GET /sources/health?probe=1` first runs one cheap request for every
catalogued source (all 19: Eastmoney quote/datacenter/fund/news/announcements, Sina kline/quote/finance,
Tencent kline/quote, THS finance/industry, CSIndex, ChinaBond, cninfo announcements/profile, Xueqiu,
efinance, and Tushare when `TUSHARE_TOKEN` is set) through the same guarded `SourceRuntime.call`, so
breakers, latency and errors are recorded exactly as for real traffic, then returns the report with a
`probe` block (per-source `ok`, `outcome`, `latency_ms`, `error`; `catalog_total`; and `not_probed`,
each skipped source with its reason). Every source row also gets a `probe` field: the latest probe
result, or `{"probed": false, "reason": ...}` (for example Tushare without a token), so an unprobed
source is never mistaken for a healthy or merely idle one. Slow multi-page upstreams get 15 s, the rest 6 s. Several upstreams throttle bursts from one IP, so probing is rate limited
process-wide: one round per `QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS` (default 60). Inside that window
the previous round is returned with `status: rate_limited` and `retry_in_s`; a request during a
running round gets `in_progress`. With live market data off the probe is `skipped`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `QI_USE_LIVE_MARKET` / `_NEWS` / `_ANNOUNCEMENT` / `_MACRO` | `true` | Enable live providers (unchanged) |
| `QI_SOURCE_CALL_TIMEOUT_SECONDS` | `10` | Hard wall-clock timeout per upstream call |
| `QI_SOURCE_FAILURE_THRESHOLD` | `3` | Consecutive failures that open a circuit |
| `QI_SOURCE_COOLDOWN_SECONDS` | `60` | First open-circuit cooldown |
| `QI_SOURCE_MAX_COOLDOWN_SECONDS` | `600` | Cooldown cap after repeated failed trials |
| `QI_SOURCE_CACHE` | `true` | TTL cache and last-known-good reads |
| `QI_SOURCE_MAX_STALE_SECONDS` | `86400` | Oldest last-known-good value that may be served |
| `QI_SOURCE_MAX_WORKERS` | `32` | Size of the bounded source-call pool |
| `QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS` | `60` | Minimum time between two active probe rounds |
| `QI_SOURCE_CROSS_CHECK` | `true` | Fetch Sina and THS fundamentals and reconcile them |

## Tests

- `tests/test_data_sources.py` (offline, part of the default run) covers:
  - breaker transitions: open, short-circuit, half-open, and cooldown doubling;
  - hard timeouts;
  - cache TTL, stale reads, and copy isolation;
  - chain ordering and fallback reasons;
  - Tencent fallback, and the breaker stopping calls to a blocked source;
  - the new fundamentals, valuation, fund and industry sources;
  - macro column/NaN/staleness handling and the ChinaBond fallback;
  - the cninfo orgId lookup and the announcement fallback;
  - pipeline caching, last-known-good, the fresh-vs-stale snapshot policy, and provenance on tool
    outputs;
  - `/sources/health`.
- `tests/test_source_reliability.py` (offline) covers the bounded pool (abandoned-call accounting,
  fast rejection without tripping the breaker, the thread bound under a hung upstream, nested calls),
  the rate-limited active probe, the Sina/THS reconciliation on the real 000858 figures (range,
  report period, cumulative vs single quarter, digit-free metadata), the stale-industry refresh and
  labelling, and the Prometheus collector for breaker states and the pool.
- `tests/test_data_sources_live.py` hits real endpoints. It runs only with `QI_LIVE_TESTS=1` (or the
  repository's existing `QI_INTEGRATION_TESTS=1`).

## Known limitations

- CPI, PMI, M2 and LPR have no independent live secondary. The official NBS API returns 403 to
  scripts, and the jin10 series are a year stale. If the Eastmoney datacenter is down, these indicators
  come from the last-known-good cache or the (stale, labelled) snapshot.
- Sina's latest half-year report row has `销售毛利率(%) = NaN` for the audited stocks, so
  `grossprofit_margin` is `null` unless THS served the report. The value is reported as missing rather
  than borrowed from another period.
- Eastmoney quote-host throttling depends on the client IP. On another network the primary may
  succeed; the chain and breaker handle both cases.
- Tushare was not audited because no `TUSHARE_TOKEN` was available. Its records get generic live
  provenance.
- ETFs have no announcements on either source (0 items for 510300).
- CSIndex valuation only covers CSI indices. For the SZSE index 399006 创业板指 the request failed in a
  live pipeline run, so `index_valuation` has no values and its provenance note says 未获取到实时数据.
- The agent API's `evidence_sources` list is built in `agent/graph.py`; it does not yet copy
  `provenance` from the evidence payload. The data is present in tool results and evidence payloads.
