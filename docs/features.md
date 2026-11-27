# Features, API & Setup Reference

Languages: English | [中文](zh/features.md)

Complete capability catalog, HTTP API matrix, and runtime switches for FinSight—combining the classical **Query Intelligence** core ([DOI: `10.3233/FAIA261107`](https://ebooks.iospress.nl/doi/10.3233/FAIA261107)) with the verified **Agent Layer**.

## Capabilities

### Research Agent & Orchestration
- **Explainable routing**: Classical NLU and guard rules decide refuse, clarify, deterministic workflow, or LangGraph LLM tool loop, logging every route reason.
- **Multi-turn memory & resolution**: LangGraph checkpoints track recent entities, stated constraints, and user holdings; resolves pronouns (`它`, `前者/后者`, `the latter`), group references (`这两家`), and elliptical follow-ups (`ROE呢`, `And the P/B?`) while refusing off-topic pivots.
- **Bounded execution**: Per-run wall-clock deadline (covering LLM calls, retries, and failover), step/tool/token caps, parallel tool execution, and versioned prompts pinned by hash.
- **Verification & compliance**: Sentence-level numeric and citation verification (`verifier.py`) with one automatic revision pass; output safety guards (`output_safety.py`, `text_safety.py`) strip trading advice, hedge price/earnings/market-cap forecasts, attribute single-source or disputed document figures, and reject out-of-coverage foreign listings and lookalike company names.

### Typed Read-Only Evidence Tools
Nine Pydantic-validated read-only tools cover entity resolution, price history, technical indicators, fundamentals, macro indicators, news, announcements, knowledge search, and document sentiment. Each tool enforces timeouts, retries, TTL caching, and structured error hints.

### Deterministic Claim Check
`POST /agent/claim-check` verifies pasted market claims (e.g., `"茅台市盈率只有15倍，股价跌了5%"`) against structured market and fundamental data without invoking an LLM. Supports Chinese/English comparators, negation, ranges, YoY growth, multi-company sums, 52-week references, superlatives, industry averages, explicit accounting definitions (`加权平均ROE`, `扣非净利润`, `流通市值`), dated historical bars, and currency checks. See [docs/claim-check.md](claim-check.md).

### Data & Offline Snapshot Layers
- **Live fallback chain**: Tushare / Eastmoney → Sina → Tencent → TTL cache → last-known-good → offline snapshot, with per-source circuit breakers, bounded call pools, Sina/THS fundamental cross-checking, and per-record provenance (`source_id`, `source_mode`, `fallback_reason`, `cache_hit`).
- **Offline data layers**: Bundled v1 seed (`data/structured_data.json`), extended 15-stock daily history snapshot (`data/snapshot/structured_data_ext.json`), and primary-source annual-report corrections (`data/snapshot/structured_data_fix.json`), verified by SHA-256 at startup and surfaced on `GET /ready`.

### Protocols, Observability & UI
- **Protocols**: MCP server (stdio & authenticated streamable HTTP), external MCP client (`QI_MCP_SERVERS`), and A2A 1.0 JSON-RPC + streaming server with Postgres-shared task persistence.
- **Observability**: Per-run JSON & OpenTelemetry (OTLP/Jaeger) traces, Prometheus `/metrics`, a 26-panel Grafana dashboard with 14 alert rules, structured privacy-preserving audit logs, and user feedback capture (`POST /agent/feedback`).
- **Web UI**: React 19 + TypeScript + Tailwind v4 + Lightweight Charts workspace with live step timeline, SSE answer streaming, citation ledger, KPI tiles, dedicated Claim Check tab, zh/en i18n, dark mode, Markdown export, and Playwright E2E regression tests.
- **Security**: API-key & signed-cookie tenant isolation, fail-closed `QI_PROFILE=production`, eager chunked-safe body limits (`413`), Postgres-backed rate limiting, full-history `gitleaks` CI scanning, non-root read-only containers, and Kubernetes `NetworkPolicy`. See [SECURITY.md](../SECURITY.md).

## API

| Endpoint | Purpose |
|---|---|
| `POST /agent/chat` | Verified agent answer with evidence ledger, verification report, trace ID, and token cost |
| `POST /agent/chat/stream` | SSE stream emitting node progress, tool events, `answer_delta` chunks, and final verified payload |
| `POST /agent/resume` | Resume a paused clarification turn (`interrupt` / `resume`) on the same session |
| `POST /agent/claim-check` | LLM-free numeric fact-check for a pasted financial claim |
| `POST /agent/feedback` | Record thumbs up/down feedback tied to a `trace_id` |
| `GET /agent/sessions/{id}` | Retrieve caller-scoped session memory and restored turns |
| `GET /agent/traces`, `GET /agent/traces/{id}` | List or inspect caller-scoped execution traces (`403` for anonymous callers) |
| `GET /.well-known/agent-card.json`, `POST /a2a` | A2A 1.0 discovery card and JSON-RPC / streaming endpoint |
| `GET /sources/health[?probe=1]` | Live provider circuit-breaker status and optional rate-limited active probe |
| `GET /health`, `GET /ready`, `GET /metrics` | Liveness, readiness (checkpointer, LLM config, retrieval index, offline data layers), and Prometheus metrics |
| `POST /chat` | Classic chatbot endpoint (`mode=workflow` runs the paper pipeline; `agent` / `auto` routes to the agent) |
| `POST /nlu/analyze`, `POST /retrieval/search`, `POST /query/intelligence`, `POST /query/intelligence/artifacts` | Classical Query Intelligence pipeline producing typed `NLUResult` and `RetrievalResult` artifacts |

JSON Schemas generated from `query_intelligence/contracts.py` live in `schemas/agent_*.schema.json`, `schemas/nlu_result.schema.json`, and `schemas/retrieval_result.schema.json` (`EntityItem.symbol` enforces `^[A-Za-z0-9._-]+$` up to 64 chars; `EvidenceItem` carries provenance metadata).

## Quick-Start Configuration

### Local Server & LLM Settings

```bash
pip install -r requirements.txt
uvicorn query_intelligence.api.app:create_app --factory --port 8765
```

Configure any OpenAI-compatible provider to activate the LLM agent loop (omit `DEEPSEEK_API_KEY` for deterministic mode):

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-v4-flash
export QI_LLM_FALLBACK_MODELS=...   # optional comma-separated failover models
```

### Live vs. Offline Data Switches

| Variable | Default | Effect |
|---|---|---|
| `QI_USE_LIVE_MARKET` / `_NEWS` / `_ANNOUNCEMENT` / `_MACRO` | `true` | Set to `0` to serve strictly from the offline snapshot |
| `QI_OFFLINE_SNAPSHOT_EXT` | `true` | Load extended 15-stock history (`data/snapshot/structured_data_ext.json`) |
| `QI_OFFLINE_SNAPSHOT_FIX` | `true` | Apply primary-source annual-report corrections (`data/snapshot/structured_data_fix.json`) |

### Container, Kubernetes & Monitoring

```bash
TAG=$(git rev-parse --short=7 HEAD)
docker build -f docker/Dockerfile -t finsight:$TAG .
docker compose -f docker/docker-compose.yml --profile monitoring up -d
(cd deploy/k8s && kustomize edit set image finsight=finsight:$TAG)
kubectl apply -k deploy/k8s
```

See [docs/deployment.md](deployment.md) for Secrets, NetworkPolicies, and readiness probes.
