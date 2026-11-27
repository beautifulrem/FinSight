# Data Sources, Fallback Chains & Provenance

Languages: English | [中文](zh/data-sources.md)

FinSight retrieves Chinese market, fundamental, macro, news, and announcement data through ordered live fallback chains protected by per-source circuit breakers, bounded worker pools, TTL/last-known-good caching, cross-source reconciliation, and a three-layer offline snapshot. Every record carries digit-free `provenance` metadata.

## What is live, what is snapshot, and the fallback order

| Mode | Market Data | Fundamentals & Valuation | Macro | News & Announcements |
|---|---|---|---|---|
| **Live** (`QI_USE_LIVE_*=true`, default) | Live chain (30 latest daily bars; intraday during A-share hours for "today") | Sina + THS statements (cross-checked), Eastmoney datacenter → Tencent valuation | Eastmoney datacenter (+ ChinaBond 10Y yield) | Eastmoney news; Cninfo → Eastmoney notices |
| **Offline** (`QI_USE_LIVE_*=false`) | Shipped offline snapshot | Shipped offline snapshot | Shipped offline snapshot | Local runtime corpus (`data/runtime/documents.jsonl`) |

Per-record resolution order in live mode: **live chain → TTL cache → last-known-good (≤24 h) → offline snapshot** (snapshot prices stand in for failed live fetches only if ≤10 days old; fundamentals/macro/industry snapshots are served labelled `mode: snapshot` with `fallback_reason`).

## Fallback Chains

| Data Kind | Primary | Secondary Chain | Final Fallback |
|---|---|---|---|
| Stock daily bars | Eastmoney `stock_zh_a_hist` (or Tushare when token set) | Sina `stock_zh_a_daily` → Tencent fqkline → Sina realtime → efinance | LKG → fresh snapshot |
| ETF daily bars | Eastmoney `fund_etf_hist_em` | Sina `fund_etf_hist_sina` → Tencent fqkline → efinance | LKG → fresh snapshot |
| Index daily bars | Sina `stock_zh_index_daily` | Eastmoney `index_zh_a_hist` → Tencent fqkline → Eastmoney spot | LKG → fresh snapshot |
| Financial statements | Sina `stock_financial_analysis_indicator` | THS `stock_financial_abstract_ths` (reconciled concurrently) | Shipped snapshot |
| PE(TTM) / PB | Eastmoney datacenter `stock_value_em` | Tencent quote | Shipped snapshot |
| Industry context | Eastmoney `stock_individual_info_em` / THS industry index | Cninfo `stock_profile_cninfo` | Shipped snapshot (labelled stale) |
| Fund profile & fees | Eastmoney `fund_overview_em` | Xueqiu → `fund_etf_fund_info_em` | – |
| CPI / PMI / M2 / LPR | Eastmoney datacenter (NBS / PBoC) | – | LKG (6 h TTL / 24 h stale) → snapshot |
| CN 10Y yield | Eastmoney `bond_zh_us_rate` | ChinaBond `bond_china_yield` | LKG → snapshot |
| News / Announcements | Eastmoney news; Cninfo orgId notices | Eastmoney notices API | Local corpus |

All adapters preserve explicit exchange suffixes (`.SH`, `.SZ`, `.BJ`) and route SSE `000xxx.SH` and SZSE `399xxx.SZ` symbols as indices rather than stocks.

## Offline Snapshot & Corrections Layer

FinSight ships three deterministic offline layers (precedence: **corrections > v1 seed > extension**), verified by SHA-256 and surfaced on `GET /ready`:

1. **v1 Seed** (`data/structured_data.json`, as of `2026-04-22`): 7 core instruments and 3 companies, pinned by default in offline evaluation suites (`QI_OFFLINE_SNAPSHOT_EXT=false`, `QI_OFFLINE_SNAPSHOT_FIX=false` inside `tests/conftest.py`).
2. **Snapshot Extension** (`data/snapshot/structured_data_ext.json`, as of `2026-09-30`, `QI_OFFLINE_SNAPSHOT_EXT=true`): Adds 15 major A-shares, ETFs, and indices with 301–307 daily closes (enabling offline 52-week high/low, YTD return from `prior_year_end`, max drawdown, EPS, market cap, YoY growth, and technical indicators). Rebuild/verify via `python -m scripts.build_offline_snapshot verify`.
3. **Corrections Layer** (`data/snapshot/structured_data_fix.json`, `QI_OFFLINE_SNAPSHOT_FIX=true`): Replaces erroneous v1 FY2025 statements (五粮液, 中国平安, 贵州茅台) and 2026-04-22 daily bars using primary CNINFO annual reports and CSIndex/Sina daily records (`data/snapshot/corrections/sources.json`), and attaches explicit `definitions` (`roe_basis`, `revenue_basis`, `net_profit_basis`, `eps_basis`, `*_yoy_basis`). If enabled but missing at runtime, `GET /ready` returns `503` and chat/claim-check mark `degraded: ["data_layer_missing:corrections"]`.

## Reliability Mechanisms (`integrations/sources/`)

- **Per-source circuit breaker (`health.py`)**: Opens after `QI_SOURCE_FAILURE_THRESHOLD` (default `3`) consecutive failures; short-circuits for `QI_SOURCE_COOLDOWN_SECONDS` (`60 s`, exponential backoff up to `600 s`) before allowing one half-open trial.
- **Bounded call pool & hard timeouts (`runtime.py`)**: Wraps un-timeoutable third-party libraries (including serialized `MiniRacer` V8 execution for AKShare JS helpers) inside a bounded worker pool (`QI_SOURCE_MAX_WORKERS=32`, `QI_SOURCE_CALL_TIMEOUT_SECONDS=10`). Saturated pools fast-reject (`SourcePoolSaturatedError`) without tripping source breakers; nested calls run inline to prevent deadlocks.
- **TTL cache & last-known-good (`cache.py`)**: 60 s for market bundles, 30 min for daily macro, 6 h for monthly macro, with up to 24 h stale fallback on total upstream failure.
- **Sina/THS fundamental cross-check (`crosscheck.py`, `QI_SOURCE_CROSS_CHECK=1`)**: Concurrently fetches both sources, drops out-of-range anomalies, distinguishes cumulative vs single-quarter YoY conventions, resolves disagreements ≥2 pp against level-recomputed cumulative YoY, and records outcomes in `provenance.cross_check`.
- **Intraday quotes (`intraday.py`)**: Queries containing `今天`/`今日`/`today` during Beijing A-share trading hours (Mon–Fri `09:30–11:30`, `13:00–15:00`) fetch live Sina/Tencent quotes (`price_basis: "intraday"`); outside trading hours or on holidays, returns the dated daily close with explicit `basis_reason`.
- **Value hygiene (`values.py`)**: Normalizes `NaN`/`Inf`, placeholder dashes, percent strings, `亿`/`万` scales, and `volume_unit` (`lot` vs `share`).

## Provenance Schema

Every structured payload and document includes `provenance` containing only strings, booleans, and ISO dates (never raw numeric metrics, preventing false verifier matches):

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

## Health Endpoint

- `GET /sources/health`: Passive report of all 19 catalogued sources (status, circuit state, call/error counts, latencies, `retry_in_s`) and `worker_pool` counters (`busy`, `max_busy`, `timed_out_total`, `abandoned_total`, `rejected_total`).
- `GET /sources/health?probe=1`: Opt-in active probe running one lightweight check per catalogued source, rate-limited process-wide to once per `QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS` (`60 s`).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `QI_USE_LIVE_MARKET` / `_NEWS` / `_ANNOUNCEMENT` / `_MACRO` | `true` | Enable live providers |
| `QI_SOURCE_CALL_TIMEOUT_SECONDS` | `10` | Per-call hard wall-clock timeout |
| `QI_SOURCE_FAILURE_THRESHOLD` | `3` | Consecutive failures to open circuit |
| `QI_SOURCE_COOLDOWN_SECONDS` / `_MAX_COOLDOWN_SECONDS` | `60` / `600` | Circuit open cooldown and backoff cap |
| `QI_SOURCE_CACHE` / `QI_SOURCE_MAX_STALE_SECONDS` | `true` / `86400` | TTL cache and LKG retention window |
| `QI_SOURCE_MAX_WORKERS` | `32` | Bounded source worker-pool size |
| `QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS` | `60` | Minimum interval between active `?probe=1` runs |
| `QI_SOURCE_CROSS_CHECK` | `true` | Enable concurrent Sina/THS fundamental reconciliation |
| `QI_OFFLINE_SNAPSHOT_EXT` / `QI_OFFLINE_SNAPSHOT_FIX` | `true` / `true` | Toggle offline extension and corrections layers |
| `QI_EVAL_SNAPSHOT_EXT` / `QI_EVAL_SNAPSHOT_FIX` | unset | Opt-in switches for evaluation harness runs |

## Tests & Known Limitations

```bash
python -m pytest -q tests/test_data_sources.py tests/test_source_reliability.py tests/test_intraday_quote.py tests/test_data_layers.py
QI_LIVE_TESTS=1 python -m pytest -q tests/test_data_sources_live.py
```

- **Macro secondaries**: CPI, PMI, M2, and LPR rely on Eastmoney datacenter (falling back to LKG/snapshot if unavailable).
- **Unadjusted closes & window length**: Closes are unadjusted (`不复权`); live calls fetch 30 daily bars, so 52-week/YTD drawdowns require the offline extension or deeper history.
- **Industry valuation tiles**: Live THS industry refreshes provide daily change only; extension-only industries report missing industry-average valuation explicitly rather than fabricating tiles.
- **Historical audit log**: Point-in-time live audit measurements are archived in [docs/results/data_sources/audit-20261001T0626Z-c915aef.md](results/data_sources/audit-20261001T0626Z-c915aef.md).
