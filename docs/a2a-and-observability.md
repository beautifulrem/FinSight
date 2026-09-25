# A2A, Model Failover and Observability

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

Every agent run produces a trace (`query_intelligence/agent/tracing.py`): node spans, tool calls (arguments, latency, attempts, cache hits, errors), LLM calls (latency, tokens, cache hits, cost, requested tools), verification result, compliance notes and degradations.

| Endpoint | Purpose |
|---|---|
| `GET /agent/traces?limit=50&session_id=...` | Summaries of recent runs, newest first: route, answer source, duration, tool calls/errors, LLM calls, tokens, cost, verification. |
| `GET /agent/traces/{trace_id}` | The full trace. Served from an in-memory ring buffer (last 200 runs) and, for older runs, from the JSON trace files under `QI_AGENT_TRACE_DIR` (default `outputs/traces/`). |

Traces can also be exported to any OTLP backend (Jaeger, Tempo, Langfuse) with `QI_AGENT_OTEL=1` or `OTEL_EXPORTER_OTLP_ENDPOINT`; see [agent.md](agent.md).

## Prometheus metrics

`GET /metrics` serves Prometheus text format (`prometheus-client`; the endpoint returns 503 if the package is missing). Metrics are fed by the same traces:

| Metric | Labels | Meaning |
|---|---|---|
| `finsight_agent_runs_total` | `route`, `answer_source` | Runs by route (refuse / clarify / workflow / agent) and by who wrote the answer (template, LLM compose, LLM agent, guardrail). |
| `finsight_agent_run_seconds` | `route` | End-to-end run latency histogram (P50/P95 via `histogram_quantile`). |
| `finsight_tool_calls_total` | `tool`, `outcome` | Tool calls by outcome: `ok`, `cached`, `error`. |
| `finsight_tool_seconds` | `tool` | Tool latency histogram. |
| `finsight_llm_calls_total` | `model` | LLM calls. |
| `finsight_llm_tokens_total` | `model`, `kind` | Prompt, completion, cache-hit and reasoning tokens. |
| `finsight_llm_cost_total` | `model`, `currency` | Accumulated LLM cost. |
| `finsight_verification_failures_total` | — | Draft answers that failed citation or number verification (before repair). |
| `finsight_degradations_total` | `flag` | Degradations such as `llm_error` or tool failures. |

Example queries:

```promql
histogram_quantile(0.95, sum by (le) (rate(finsight_agent_run_seconds_bucket[5m])))
sum by (tool) (rate(finsight_tool_calls_total{outcome="error"}[5m])) / sum by (tool) (rate(finsight_tool_calls_total[5m]))
sum(increase(finsight_llm_cost_total[1d]))
```

When `QI_API_KEYS` is set, `/metrics` and `/agent/traces*` require an API key like every other non-public endpoint.
