# Features, API and setup in full

[README](../README.md) · [中文](zh/features.md)

The full feature list, the full API table and the full quick start, moved verbatim in round 15 from `README.md` at `4ad3380` (only relative links adjusted); the README keeps a short summary of each.

- **Agent**
  - Routing with logged reasons, parallel tool calls, and step/tool/token budgets.
  - A run deadline that also bounds every LLM request, retry and failover.
  - One LLM revision after failed verification.
  - Clarification interrupts with resume.
  - Session memory: recent targets, stated constraints and holdings.
  - Follow-ups: pronouns ("它"), plurals and group references ("这两家", "三家里哪家", "前者/后者", "the latter"), elliptical questions ("ROE呢", "换成五粮液呢", "And the P/B?"), bare "why" follow-ups, and questions about a discussed stock's sector. Off-topic requests are still refused inside a finance conversation, and every rewrite is logged as a route reason.
  - Coverage: crypto and US/HK stocks get a clear "not covered" refusal. Periods or metrics the data lacks are stated rather than silently replaced ("没有2019年数据，以下为2025年报").
  - Per-node reasoning levels, and model failover with a circuit breaker.
  - Versioned prompts pinned by hash.
- **Evidence tools**
  - Entity resolution, price history, technical indicators, fundamentals, macro indicators, news, announcements, knowledge search and document sentiment.
  - Each has a Pydantic schema, a timeout, retries, a TTL cache and actionable error hints.
- **Claim check** (`POST /agent/claim-check`): paste a claim such as "茅台市盈率只有15倍，股价跌了5%". Each number is tied to a metric and compared with market and fundamental data, with no LLM involved. The result is supported, contradicted, partially supported or unverifiable, with the evidence id, source and as-of date. Comparators (超过/不到/以上/破/between), negation, ranges, YoY growth and Chinese numerals are handled, and since round 12 sums of companies ("茅台和五粮液成交额加起来不到50亿" is checked against the 52.47亿 total), stated industry averages in a comparison frame or referred to by "这一水平 / that", English "beats … by N" and one-fold differences (高出一倍多); since round 13 X倍半, 将近一倍 as (80%, 100%], "多一点" as no number, the 52-week high / low as a reference ("比52周高点低了三成多"), "N turns above", the subject carried over 但 / 却, superlatives among named companies, "a third / a quarter of", 三个多百分点 and a multiple of another sector's average; since round 14 扣非净利润 and 流通市值 as metrics of their own (never 归母 profit or total market value), two bounds split by but / 但, a superlative with its subject first, years never read as values, "up 42% year on year" as the YoY of the metric named, 不到 / 超过…的N倍 as a bound on the multiple, 两个半X, a named definition (加权平均ROE, 基本每股收益, 归母 growth, 动态市盈率) checked only against a field with that definition (else `definition_mismatch`), dated bars from the history, foreign-currency amounts (`currency`) and traditional characters. First runs on slices written by separate authors: 0.716 (round 4), 0.821 (round 5), 0.537 (round 6), 0.600 (round 8), 0.600 (round 9), 0.667 (round 10); the round-9 slice, not opened by the engineers by their statement, scores 1.000 after the round-13 rules (25/25; 20/20 without 5 claims that overlap own wording, 15/15 without content-level overlaps), and the round-10 slice 0.833 after the round-14 rules (20/24; no claim near-copies own wording, 5 repeat own content and without them it is 16/19). The older 47-claim held-out set scored 0.936 on its first run and has been exposed since. See [docs/claim-check.md](claim-check.md).
- **Live data**
  - Eastmoney → Sina → Tencent → cache → last-known-good → snapshot chains, with a circuit breaker per source.
  - An offline snapshot in two layers: v1 (as of 2026-04-22) and an extension with a year of daily bars for 15 more names (as of 2026-09-30), so 52-week high/low, maximum drawdown (52 weeks or year to date), YTD change, EPS and market cap are computed with their window and date; every answer states each target's own as-of date.
  - A corrections layer over the v1 seed: FY2025 figures transcribed from the companies' annual reports (with page and sha256) and 2026-04-22 bars from the daily record where the seed was wrong, plus field definitions for every company, which answers and tiles state ("ROE（加权）", "营业总收入", "归母净利润", basic / diluted EPS).
  - Upstream calls run on a bounded pool.
  - Sina vs THS fundamentals are cross-checked against the reported levels.
  - Every record carries its provenance. An active probe is available at `GET /sources/health?probe=1`, rate limited.
- **Protocols**: an MCP server for the tools, plus an MCP client that registers tools from external MCP servers (`QI_MCP_SERVERS`, sandboxed as untrusted data). An A2A 1.0 endpoint serves the whole agent: a clarification maps to `input-required`, streaming reports progress, and tasks are shared through Postgres. A committed a2a-sdk client demo (`scripts/a2a_client_demo.py`) exercises the endpoint.
- **Observability**
  - A trace for every run: nodes, tools, LLM calls with context composition and JSON status, tokens, cost and prompt versions.
  - A run inspector API and OpenTelemetry export.
  - Prometheus metrics, labelled by the model that actually answered.
  - A Grafana dashboard (26 panels in 5 rows, including verification failure and repair rate by prompt version, the user-feedback ratio, output-safety edits by kind and per answer), 14 alert rules, and Jaeger.
  - An audit log: one structured event per refusal and per compliance edit, with hashed ids and no user text.
  - User feedback (`POST /agent/feedback`), turned into candidate evaluation tasks by `scripts/feedback_to_tasks.py`.
- **Web UI** (React 19, TypeScript, Tailwind v4, Radix, Motion, Lightweight Charts)
  - Before the first token, a live progress panel shows the current step (plan → data → draft → verify), the tools being called with their targets, elapsed time and a Stop button. The answer then streams as it is written, next to the run timeline. Time to first token and total time appear in the run details.
  - A fact-check view for pasted claims shows each number's comparator, claimed vs actual value (a relative difference is labelled as one), source and date, under a fixed top bar. A chat message like "听说…是真的吗" offers to open it.
  - Citation chips link to an evidence ledger that shows freshness.
  - KPI tiles lead with the metric asked about (none for a question about a future day), date P/E and P/B at their valuation date and the statement figures at the report period, and use the answer's precision. News answers say what each cited document is about with fixed topic labels, never quoting a title.
  - Price charts, run cost and latency.
  - The Trace and Run tabs show every route reason in words ("Computed the ROE gap: Wuliangye vs Kweichow Moutai"), and a reload shows the restored turns under their banner.
  - Feedback buttons and Markdown export.
  - zh/en (system notices switch too), dark mode, mobile. Playwright tests in Chrome cover a three-turn gap session and a fact-check against the offline server (`cd frontend && pnpm run e2e`).
- **Security**
  - Optional API keys, with sessions and traces scoped to the key (a hash, never the key itself).
  - Rate limiting, CORS and body limits.
  - An input guard against instructions in the user turn.
  - The container runs as non-root on a read-only root filesystem. The Kubernetes manifests carry NetworkPolicies and no plaintext Secret.
  - CI scans the full git history with gitleaks, and pre-commit hooks run the same scan ([SECURITY.md](../SECURITY.md)).

## API

| Endpoint | Purpose |
|---|---|
| `POST /agent/chat`, `POST /agent/chat/stream` (SSE), `POST /agent/resume` | Agent answer with evidence, verification, trace id and cost. The stream carries node events and `answer_delta` text as it is written. Resume answers a pending clarification. |
| `POST /agent/claim-check` | Check the numbers in a pasted market claim against data (deterministic). |
| `POST /agent/feedback` | Thumbs up/down on an answer, stored with its trace. |
| `GET /agent/sessions/{id}`, `GET /agent/traces`, `GET /agent/traces/{id}` | Session memory, recent runs and a full run trace, scoped to the caller. |
| `GET /.well-known/agent-card.json`, `POST /a2a` | A2A agent card and JSON-RPC endpoint. |
| `GET /metrics`, `GET /sources/health[?probe=1]` | Prometheus metrics; live source status (active probe of all 19 sources on request, unprobed ones marked). |
| `GET /health`, `GET /ready` | Liveness; readiness (checkpoint store reachable and writable, LLM config, retrieval index, offline data layers with record counts and sha256), 503 when not ready or when a switched-on data layer is missing. |
| `POST /chat` | Original chatbot endpoint: `mode=workflow` keeps the original pipeline; `agent` and `auto` use the agent. |
| `POST /nlu/analyze`, `POST /retrieval/search`, `POST /query/intelligence` | Classical NLU and retrieval artifacts. |

Schemas are in `schemas/agent_*.schema.json`, generated from `query_intelligence/contracts.py`.

## Quick start in full

```bash
pip install -r requirements.txt
uvicorn query_intelligence.api.app:create_app --factory --port 8765   # open http://127.0.0.1:8765
```

Without a key, answers come from the deterministic path. To enable the LLM agent, point it at any OpenAI-compatible endpoint:

```bash
export DEEPSEEK_API_KEY=...                      # read from the environment, never from the config file
export DEEPSEEK_BASE_URL=https://api.deepseek.com # or a gateway, e.g. https://api.cline.bot/api/v1
export DEEPSEEK_MODEL=deepseek-v4-flash
export QI_LLM_FALLBACK_MODELS=...                 # optional failover models on the same endpoint
```

Live market, news, announcement and macro providers are on by default. For the shipped offline snapshot, set `QI_USE_LIVE_MARKET=0` (and the same for `_NEWS`, `_ANNOUNCEMENT` and `_MACRO`). The offline snapshot is v1 (`data/structured_data.json`, 7 instruments, as of 2026-04-22) plus the extension (`data/snapshot/`, 15 more, as of 2026-09-30; `QI_OFFLINE_SNAPSHOT_EXT=0` turns it off), with the corrections layer applied (`data/snapshot/structured_data_fix.json`: primary-source values where the seed was wrong, and field definitions; `QI_OFFLINE_SNAPSHOT_FIX=0` turns it off); see [data-sources.md](data-sources.md#what-is-live-what-is-snapshot-and-the-fallback-order) for what is live, what is snapshot and the fallback order.

Docker, Kubernetes and the monitoring stack are covered in [docs/deployment.md](deployment.md). On Kubernetes, replicas share sessions through Postgres and run on a read-only root filesystem. Since round 14 the image ships the corrections layer too (before, containers served the v1 seed figures, J2), and `GET /ready` lists the data layers it loaded ([deployment.md](deployment.md#docker)).

```bash
TAG=$(git rev-parse --short=7 HEAD)
docker build -f docker/Dockerfile -t finsight:$TAG .   # images are tagged by commit
(cd deploy/k8s && kustomize edit set image finsight=finsight:$TAG)   # committed tag is a placeholder (docs/deployment.md#image-tags)
kubectl apply -k deploy/k8s        # after creating the finsight-db Secret (docs/deployment.md#secrets)
docker compose -f docker/docker-compose.yml --profile monitoring up -d   # + Prometheus, Grafana, Jaeger
```
