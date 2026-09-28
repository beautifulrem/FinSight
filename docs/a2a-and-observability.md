# A2A, Model Failover and Observability

Languages: English | [中文](zh/a2a-and-observability.md)

This page covers the interoperability and operations surface of the agent API: the A2A endpoint, LLM model routing with failover, gateway cost accounting, the run inspector, and Prometheus metrics. All of it is served by the FastAPI app (`query_intelligence/api/app.py`).

## A2A (Agent2Agent)

MCP ([docs/mcp.md](mcp.md)) exposes FinSight's *tools* so another agent can call them one by one. A2A exposes the *agent* so another agent can hand over a whole research question and get back a cited answer.

Implementation: `query_intelligence/agent/a2a_server.py`, built on `a2a-sdk` 1.x (A2A protocol 1.0).

| Endpoint | Purpose |
|---|---|
| `GET /.well-known/agent-card.json` | Agent card: skills (`equity_research`, `comparison`, `macro_linkage`), JSON-RPC interface, streaming capability. Public even when `QI_API_KEYS` is set, so clients can discover the agent before they authenticate. |
| `POST /a2a` | JSON-RPC 2.0 endpoint for A2A 1.0 methods (`SendMessage`, `SendStreamingMessage`, `GetTask`, `CancelTask`, …). Requires the `A2A-Version: 1.0` header. |

How the agent maps onto A2A:

| A2A concept | FinSight behaviour |
|---|---|
| `contextId` | One agent session (`a2a<context id>`), so follow-up messages keep conversation memory and pronoun carry-over. |
| Task `input-required` | A clarification interrupt (for example "它的市盈率呢" with no prior entity). The next message on the same task resumes the paused LangGraph run through `AgentService.resume`. |
| Task `completed` | Two artifacts: `answer` (text with `[evidence_id]` citations, key points and the risk disclaimer) and `evidence` (a data part with evidence sources, verification report, route, degradations, trace id and follow-up suggestions). |
| Task `failed` | Unexpected exceptions. Tool and LLM failures do not fail the task; the graph degrades and says what is missing. |

Example:

```bash
curl -s http://127.0.0.1:8765/a2a -H 'A2A-Version: 1.0' -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
  "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "贵州茅台的市盈率是多少"}]}}
}'
```

Configuration: `QI_A2A_ENABLED=0` disables the routes; `QI_A2A_MODE` picks the agent mode (default `auto`); `QI_PUBLIC_BASE_URL` sets the URL advertised in the agent card. The task store is in memory (`InMemoryTaskStore`), which fits a single-process deployment.

Tests: `tests/test_agent_observability_a2a.py` (agent card, completed task with artifacts, input-required then resume, same-context memory, disable switch).

## LLM gateway, failover and cost

The agent's LLM client (`query_intelligence/agent/llm.py`) speaks the OpenAI-compatible Chat Completions API. Two gateway behaviours are handled explicitly:

* **Envelope.** Some gateways (for example the Cline API) wrap the completion as `{"success": true, "data": {...}}`. `unwrap_completion` accepts both shapes, for the agent client and for the legacy `/chat` client.
* **Provider-reported cost.** Per-request billing gateways report `usage.cost` (USD). It is recorded as `reported_cost_usd` on every LLM call. Run cost is resolved by `resolve_cost`: a configured price table (`QI_LLM_PRICE_*`) wins; otherwise the reported cost is used, converted to CNY when `QI_LLM_USD_CNY` is set. Responses carry `llm.cost`, `llm.currency` and `llm.cost_source` (`price_table` or `provider_reported`). Cached prompt tokens are read from either `prompt_cache_hit_tokens` (DeepSeek) or `prompt_tokens_details.cached_tokens` (OpenAI style).

**Model failover.** `QI_LLM_FALLBACK_MODELS` (comma-separated model ids on the same endpoint) wraps the primary client in `FallbackLLM`:

* Clients are tried in order; the first success answers and `AssistantTurn.model` records which model it was.
* A client's circuit opens after 3 consecutive failures and stays open for 60 s, so a failing model is skipped instead of costing a timeout on every request. After the cool-down one trial call is allowed (half-open).
* When every model fails, the error propagates and the graph degrades to the deterministic planner and template composer (the answer is still cited and verified; `degraded` lists `llm_error`).

Example (Cline gateway, DeepSeek first, GLM as backup):

```bash
export DEEPSEEK_BASE_URL=https://api.cline.bot/api/v1
export DEEPSEEK_API_KEY=...            # from the environment, never from the config file
export DEEPSEEK_MODEL=cline-pass/deepseek-v4.1-flash
export DEEPSEEK_THINKING_TYPE=         # the gateway does not take DeepSeek's `thinking` parameter
export DEEPSEEK_REASONING_EFFORT=
export QI_LLM_FALLBACK_MODELS=cline-pass/glm-5.3-flash
```

## Run inspector

Every agent run produces a trace (`query_intelligence/agent/tracing.py`): node spans, tool calls (arguments, latency, attempts, cache hits, errors), LLM calls (latency, tokens, cache hits, cost, requested tools, characters sent per context part — system, user, assistant, tool results, tool schemas — and whether the answer draft was valid JSON, repaired, or used as plain text), verification result, compliance notes and degradations.

| Endpoint | Purpose |
|---|---|
| `GET /agent/traces?limit=50&session_id=...` | Summaries of recent runs, newest first: route, answer source, duration, tool calls/errors, LLM calls, tokens, cost, verification. |
| `GET /agent/traces/{trace_id}` | The full trace. Served from an in-memory ring buffer (last 200 runs) and, for older runs, from the JSON trace files under `QI_AGENT_TRACE_DIR` (default `outputs/traces/`). |
| `POST /agent/feedback` | Thumbs up/down on a run (by `trace_id`), appended to `QI_FEEDBACK_PATH`; `scripts/feedback_to_tasks.py` turns flagged traces into candidate evaluation tasks for review. |

Traces can also be exported to any OTLP backend (Jaeger, Tempo, Langfuse) with `QI_AGENT_OTEL=1` or `OTEL_EXPORTER_OTLP_ENDPOINT`; see [agent.md](agent.md).

## Prometheus metrics

`GET /metrics` serves Prometheus text format (`prometheus-client`; the endpoint returns 503 if the package is missing). Metrics are fed by the same traces:

| Metric | Labels | Meaning |
|---|---|---|
| `finsight_agent_runs_total` | `route`, `answer_source` | Runs by route (refuse / clarify / workflow / agent) and by who wrote the answer (template, LLM compose, LLM agent, guardrail). |
| `finsight_agent_run_seconds` | `route` | End-to-end run latency histogram (P50/P95 via `histogram_quantile`). |
| `finsight_tool_calls_total` | `tool`, `outcome` | Tool calls by outcome: `ok`, `cached`, `error`. |
| `finsight_tool_seconds` | `tool` | Tool latency histogram. |
| `finsight_llm_calls_total` | `model` | LLM calls, labelled with the model that answered each call (a failover call counts against the fallback model). |
| `finsight_llm_tokens_total` | `model`, `kind` | Prompt, completion, cache-hit and reasoning tokens, per answering model. |
| `finsight_llm_cost_total` | `model`, `currency` | Accumulated LLM cost; a run's cost is split across the models it used in proportion to their prompt + completion tokens. |
| `finsight_feedback_total` | `rating` | User feedback from `POST /agent/feedback` (`up` / `down`). |
| `finsight_verification_failures_total` | — | Draft answers that failed citation or number verification (before repair). |
| `finsight_degradations_total` | `flag` | Degradations such as `llm_error` or tool failures. |

The trace-fed metrics only see finished runs. Current state is read at scrape time by
`OpsMetricsCollector` (`query_intelligence/integrations/ops_metrics.py`), registered on the same
registry by the API app:

| Metric | Labels | Meaning |
|---|---|---|
| `finsight_llm_circuit_state` | `model` | Per-model `FallbackLLM` breaker, read from `FallbackLLM.stats()`: 0 closed, 1 half-open (cool-down over, next call is a trial), 2 open. |
| `finsight_llm_client_calls_total` | `model` | Calls attempted per model, including failed ones (a failed primary call appears here but not in `finsight_llm_calls_total`). |
| `finsight_llm_consecutive_failures` | `model` | Consecutive failures per model. |
| `finsight_source_circuit_state` | `source` | Per live data source breaker (same encoding). |
| `finsight_source_calls_total` | `source`, `outcome` | `success`, `failure`, `short_circuited`. |
| `finsight_source_latency_ms` | `source` | Smoothed latency of recent calls. |
| `finsight_source_pool_workers` / `_busy` / `_abandoned_running` | — | The bounded source-call pool (see [data-sources.md](data-sources.md)). |
| `finsight_source_pool_abandoned_total` / `_rejected_total` | — | Upstream calls abandoned after their timeout; calls rejected because the pool was saturated. |

Example queries:

```promql
histogram_quantile(0.95, sum by (le) (rate(finsight_agent_run_seconds_bucket[5m])))
sum by (tool) (rate(finsight_tool_calls_total{outcome="error"}[5m])) / sum by (tool) (rate(finsight_tool_calls_total[5m]))
sum(increase(finsight_llm_cost_total[1d]))
```

When `QI_API_KEYS` is set, `/metrics` and `/agent/traces*` require an API key like every other non-public endpoint, and `/agent/traces*` only return runs of the calling key (traces carry the caller as a hash of the key, never the key itself).

## Dashboards, alerts and the monitoring stack

`docker/docker-compose.yml` has a `monitoring` profile: the app, Prometheus (scraping `/metrics` every
5 s, with the alert rules loaded), Grafana with the FinSight dashboard provisioned, and Jaeger
receiving the agent's OTLP traces. The configs are copied into the images at build time, so the
profile also works where the Docker VM cannot see the checkout.

```bash
source /tmp/llmenv.sh   # optional: DEEPSEEK_* for LLM traffic
FINSIGHT_IMAGE=finsight:merged FINSIGHT_PORT=8831 OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318 \
  QI_LLM_FALLBACK_MODELS=cline-pass/glm-5.3-flash QI_LLM_USD_CNY=6.7489 \
  FINSIGHT_EGRESS_PROXY=http://192.168.5.2:6152 \
  docker compose -p finsight-mon -f docker/docker-compose.yml --profile monitoring up -d
# Grafana http://127.0.0.1:3300 (anonymous, local use only), Prometheus :9090, Jaeger :16686
python monitoring/screenshot.py --grafana http://127.0.0.1:3300 --jaeger http://127.0.0.1:16686
```

`FINSIGHT_EGRESS_PROXY` is only needed where containers have no direct internet access (here: colima
behind the host's proxy; without it every live source failed with DNS or connection errors).

| File | Content |
|---|---|
| `monitoring/grafana/finsight-dashboard.json` | 19 panels in four rows. Traffic: requests/s by route, P50/P95 by route, answer source. Quality: verification-failure rate, degradations by flag, tool error rate by tool. LLM: cost per hour and per 24 h, calls per model (failover), per-model breaker state timeline, tokens by kind, LLM calls per answered run. Data sources: breaker state timeline per source, calls by outcome, the source-call pool. |
| `monitoring/prometheus/alerts.yml` | 10 rules: `FinSightDown`, `FinSightWorkflowP95High` (> 8 s for 10 min), `FinSightAgentP95High` (> 60 s), `FinSightVerificationFailureRateHigh` (> 20%), `FinSightToolErrorRateHigh` (> 25% per tool), `FinSightLLMModelCircuitOpen`, `FinSightAllLLMModelsDown`, `FinSightDataSourceCircuitOpen`, `FinSightSourcePoolAbandonedCalls`, `FinSightLLMCostBurnHigh` (> ¥20 per hour). `promtool check rules`: 10 rules, valid. |

Verified on 2026-09-26 (colima): all four containers up, the Prometheus target `finsight` healthy,
the rules loaded and evaluating. After the traffic below, three alerts were `pending`: data source
circuit open (Eastmoney's quote hosts throttle this IP, see [data-sources.md](data-sources.md)),
workflow P95 high (live data behind the proxy: P50 8.8 s) and agent P95 high
(`docs/results/observability/prometheus-alerts.json`). Traffic: 80 workflow requests with live data
(8 users) and 12 `auto`-mode research questions (4 users), both through `scripts/load_test.py`
(`docs/results/observability/traffic-*.json`); the active source probe
(`sources-health-probe.json`: 9 of 10 sources up, Eastmoney quote down, 1.97 s; a second call 1 s
later returned `rate_limited` with `retry_in_s` 59.6).

![Grafana dashboard](assets/ops/grafana-dashboard.png)

The Jaeger trace below is the slowest agent run of that traffic (98 s): one `finsight.agent.run` span
with node, tool and LLM child spans. The tail is visible at a glance: tools ran in parallel in 8.6 s,
the first draft failed verification, and the `llm.revise` call alone took 74 s.

![Jaeger trace](assets/ops/jaeger-trace.png)

## Chaos drill

`scripts/chaos_drill.py` injects real faults in front of a live server started by the script itself
(nothing is stubbed inside the process under test) and records latency, traces, `/sources/health` and
`/metrics` per phase. Results: `docs/results/chaos/llm/chaos-llm.json` and
`docs/results/chaos/sources/chaos-sources.json` (2026-09-26, merged build, see
[performance.md](performance.md#test-host-and-builds)).

### LLM: primary model fails, failover to GLM, breaker opens and recovers

An LLM fault proxy sits between the server and the Cline gateway. While the fault is on it rewrites the
primary model id to an invalid one, so the real gateway rejects it (HTTP 404, as a mistyped
`DEEPSEEK_MODEL` would). The proxy logs model, status and latency of every gateway call (no headers,
no prompts).

```bash
source /tmp/llmenv.sh
python -m scripts.chaos_drill --scenario llm --fallback-model cline-pass/glm-5.3-flash --usd-cny 6.7489
```

| Phase (UTC) | Requests | What the gateway saw | `finsight_llm_circuit_state` (DeepSeek / GLM) |
|---|---|---|---|
| 1 baseline 13:43 | 1 agent answer, 44.2 s, verified | 4 DeepSeek calls, 200 | 0 / 0 |
| 2 primary failing 13:44–13:48 | 3 agent requests: 80.8 s and 47.5 s (verified, answered by GLM, 3 and 5 calls); the third hit the API's 120 s timeout (504) while a GLM call took 62.8 s | DeepSeek 404 three times (0.25–1.0 s each), then the breaker opened and calls went straight to GLM (13 calls, 200). After each 60 s cool-down one half-open trial reached DeepSeek, got 404 and re-opened the breaker (3 trials) | 2 (open) / 0 |
| 3 fault healed, cool-down over 13:49 | none | none | 1 (half-open) / 0 |
| 4 recovered 13:49 | 1 agent answer, 21.0 s, verified | 3 DeepSeek calls, 200: the trial succeeded and closed the breaker | 0 / 0 |

In the traces (`trace_llm_calls` per request), the LLM spans of phase 2 carry
`gen_ai.request.model = z-ai/glm-5.3-flash`, and those of phases 1 and 4 `deepseek/deepseek-v4.1-flash`.
Latency cost of the failover: a failed primary call costs 0.25–1.0 s (404 is not retried), and an open
breaker costs nothing. The real cost is the fallback model: GLM calls took 2.5–62.8 s against
2.5–9.0 s for DeepSeek, so answers took 48–81 s instead of 21–44 s, and one request exceeded
`QI_AGENT_REQUEST_TIMEOUT_S`. Since `d1c007c` every LLM request is bounded by the run deadline (tool loop 90 s, answer-producing calls 20 s more; see [agent.md](agent.md#configuration)), so a slow fallback model now ends in the deterministic answer instead of a 504; the drill has not been rerun with it.

### Data sources: Sina, Tencent and Eastmoney blocked

The server's `HTTPS_PROXY`/`HTTP_PROXY` point at a blocking proxy that forwards traffic (to the
host's proxy) but answers `403` for `*.sina.com.cn`, `*.sinajs.cn`, `*.sina.cn`, `*.gtimg.cn` and
`*.eastmoney.com` while blocking is on. Live data on, `QI_SOURCE_COOLDOWN_SECONDS=20`,
`QI_SOURCE_MAX_STALE_SECONDS=90`, workflow mode.

```bash
python -m scripts.chaos_drill --scenario sources --source-cooldown 20 --max-stale 90
```

| Phase (UTC) | Question | Latency | What was served (from the answer's evidence provenance) |
|---|---|---:|---|
| 1 live 13:56 | 贵州茅台最新收盘价 | 4.1 s | close 1237.0 from `sina.kline`, `live_fallback` ("因东方财富行情请求失败降级": Eastmoney's quote host was already throttling this IP) |
| 1b live | 五粮液营收和净利润增长；行业表现 | 2.1 s | fundamentals from `ths.finance` with `cross_check: disagree_resolved` (Sina's YoY contradicted the reported levels); industry 白酒 live from `ths.industry` (2026-09-24) instead of the April snapshot |
| 2 blocked, within TTL | same price question | 0.13 s | the same close from the 60 s caches, no upstream call |
| 2 blocked | CPI 最新数据 | 0.18 s | never cached: straight to the offline snapshot, labelled `snapshot`, `stale`, "因实时宏观数据不可用降级" |
| 3 blocked, after TTL 13:57 | price | 1.5 s | every live candidate failed (Eastmoney, Sina, Tencent, Sina realtime, efinance); served `last_known_good`, "沿用最近一次成功获取的实时数据（获取于13:56:02）" |
| 4 blocked, after the stale window 13:58 | price | 1.8 s | no price: the shipped snapshot price (2026-04) is too old to stand in for a quote, so the answer states the limitation ("get_price_history 未返回可用数据") instead of a stale number; breakers open for `sina.kline`, `sina.quote`, `tencent.kline`, `efinance` |
| 5 unblocked, after cool-down 13:59 | price | 1.7 s | `sina.kline` half-open trial succeeded, breaker closed, live again |

Every answer was HTTP 200 and passed verification. The source-call pool never came near saturation
(`max_busy` 4 of 32, 0 abandoned, 0 rejected), because blocked hosts fail fast with 403. The pool
matters for the other failure mode, hosts that hang: `--block-mode hang` makes the proxy hold the
connection instead (not part of the recorded run); such calls end at `QI_SOURCE_CALL_TIMEOUT_SECONDS` and count as abandoned, which `tests/test_source_reliability.py` covers offline.
