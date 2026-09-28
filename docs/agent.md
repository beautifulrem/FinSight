# Agent Layer

Languages: English | [中文](zh/agent.md)

The agent layer (`query_intelligence/agent/`) turns FinSight from a fixed "NLU → retrieval → one LLM call" pipeline into a tool-using research assistant. It sits **on top of** Query Intelligence: NLU and retrieval stay classical and explainable (see [AGENTS.md](../AGENTS.md)), and the agent only plans, calls tools, checks the draft against evidence, and applies compliance rules.

Two answer paths share one LangGraph state graph:

- **workflow**: a deterministic planner derives tool calls from `nlu_result` (style, intents, source plan, lexical cues). An LLM is used only to phrase the answer (and falls back to a template without one). This path is reproducible and works offline.
- **agent**: an LLM (DeepSeek, OpenAI-compatible tool calling) chooses tools step by step within budgets. It is used for complex questions when an LLM is configured.

`mode=auto` routes each turn; `mode=workflow` and `mode=agent` force a path. Without an LLM key, `agent` routes are downgraded to `workflow` and the downgrade is reported in `degraded`.

## Graph

```mermaid
flowchart LR
    Q[query + session] --> G[guard_in<br/>NLU, coreference,<br/>router]
    G -->|out of scope| R[refuse]
    G -->|missing target| C[clarify<br/>interrupt → /agent/resume]
    G -->|simple| P[execute_plan<br/>deterministic tools]
    G -->|complex + LLM| A[agent_llm]
    A -->|tool calls| T[agent_tools] --> A
    A -->|final draft| V
    P --> K[compose<br/>LLM or template] --> V[verify<br/>citations + numbers]
    V -->|failed, budget left| X[revise] --> V
    V --> M[compliance] --> F[finalize]
    R --> F
    C --> F
```

| Node | What it does |
|---|---|
| `guard_in` | Input guard (instruction-like spans in the user's own message are removed before NLU; a message with nothing financial left is refused), NLU with session history as `dialog_context`, follow-up resolution (pronouns, plurals and elliptical questions, see [Memory](#memory-and-sessions)), a filter for fuzzy concept matches that are not in the question, corrections of NLU out-of-scope false positives for clear finance/macro questions, and routing. Every decision is written to `route_reasons`. |
| `refuse` | Out-of-scope answer and follow-up suggestions, reusing `scripts/llm_response.py` wording. |
| `clarify` | Asks which security is meant. With a checkpointer the graph pauses with `interrupt()` and continues when `/agent/resume` supplies the reply (at most one round per turn). |
| `execute_plan` | Runs the planner's tool calls in parallel (`max_parallel_tools`). |
| `compose` | LLM composition over the collected evidence, or `compose_template` when no LLM is configured or the LLM fails. |
| `agent_llm` / `agent_tools` | The tool-calling loop. Stops on a final answer, `max_llm_steps`, `max_tool_calls`, `token_budget`, or `run_deadline_s`; hitting a limit forces a final answer from the evidence gathered so far. A call repeated with identical arguments in the same turn is not run again: the model gets a `duplicate_call` error that points to the earlier result, and the run records `repeated_tool_calls:N` in `degraded`. |
| `verify` | Every cited `evidence_id` must exist, and every number must be in the evidence cited **in its own sentence** (claim-level binding): unit scaling limited to the stated unit (亿/万/%/hundred million…), a tolerance set by the written precision, the stated direction (涨/跌, up/down) checked against the sign, and for LLM drafts market metrics (price, change, PE/PB) only from market evidence and citations required. Dates, tickers and indicator parameters are ignored. |
| `revise` | Sends the verification feedback back to the LLM (`max_revisions`). If it still fails, the answer is repaired clause by clause: unsupported clauses are dropped and `verification_failed:repaired` is recorded. |
| `compliance` | Softens judgment and causal language (conditional wording for "can I buy", caveats for "why did it rise"), removes direct trading instructions, ratings and position sizing, adds a freshness note for stale market data and the risk disclaimer. A language guard replaces an answer that is not in the question's language (e.g. hijacked by a poisoned document) with the deterministic answer. |
| `finalize` | Builds the response: answer, citations, evidence sources, tool calls, verification, LLM usage/cost, spans, sentiment, next questions. |

## Tools

All tools share one base (`tools/base.py`): Pydantic input schemas (also exported as OpenAI tool schemas and over MCP), a timeout, retries on transient errors, a TTL cache, and normalized error codes (`unknown_tool`, `invalid_arguments`, `timeout`, `upstream_error`, `not_found`, `unavailable`, `internal`). Every successful call returns `AgentEvidence` with a stable `evidence_id`, which is what answers cite.

| Tool | Backed by |
|---|---|
| `resolve_entity` | Entity resolver from NLU (aliases, tickers, fuzzy match) |
| `get_price_history` | Market fallback chain when live (Tushare with a token; otherwise Eastmoney → Sina → Tencent → Sina realtime → efinance), seed snapshot offline |
| `compute_indicators` | `MarketAnalyzer`: returns, MA5/MA20, RSI(14), MACD, volatility, Bollinger bands. Reports indicators it cannot compute from short history instead of returning empty values. |
| `get_fundamentals` | Live: Sina financial indicators → THS, PE(TTM)/PB from Eastmoney datacenter → Tencent quote; seed snapshot offline |
| `get_macro_indicators` | Live: Eastmoney datacenter (NBS CPI/PMI, M2, LPR 1Y/5Y) and the 10Y yield (Eastmoney → ChinaBond); seed snapshot offline |
| `search_news`, `search_announcements`, `search_knowledge` | The existing retrieval pipeline (PostgreSQL full-text search / TF-IDF + learning-to-rank) |
| `analyze_sentiment` | Classical sentiment model by default; FinBERT with `QI_AGENT_SENTIMENT_BACKEND=finbert` |

Data tools return an optional `provenance` object in their output and evidence payload: source, `fetched_at`, `as_of`, `is_live`, `mode` (`live` / `live_fallback` / `last_known_good` / `snapshot`), `freshness`, `fallback_reason`, and a one-line `note` such as 数据来自新浪财经行情，截至2026-09-24；因东方财富行情熔断中降级. It contains no numeric values, so it cannot make an invented number look traceable to the verifier. Chains, circuit breakers, caching, and the measured audit are described in [Live data sources](data-sources.md); `GET /sources/health` reports per-source status.

Document text is untrusted: tool output reaches the LLM inside an explicit untrusted-data envelope, and instruction-like text ("ignore previous instructions", role tags, …) is redacted when evidence is ingested. Redactions are flagged in `degraded` as `instruction_like_text_removed_from_evidence`.

The same tools are published by an MCP server; see [MCP](mcp.md).

## Memory and sessions

- Each `session_id` is a LangGraph thread. The checkpointer is in memory by default; `QI_AGENT_CHECKPOINT_DB=/path/sessions.sqlite` persists sessions across restarts, and a `postgresql://` DSN shares them between processes and replicas.
- Per-turn fields (tool log, evidence, verification, …) are reset at the start of every turn, so one turn's evidence can never be cited in the next. One checkpoint is written per run (`QI_AGENT_DURABILITY=exit`).
- Completed turns (query, answer, entities, evidence ids) are kept in `turns` and fed to NLU as dialog context.
- Requests on the same session are serialized with a per-session lock (a bounded LRU of at most 4,096 idle locks).
- **Ownership.** A session belongs to the caller that created it: with `QI_API_KEYS` set, the owner is a hash of the API key (`key:<sha256[:12]>`, never the key itself). Another key gets `404` for that session id, and `/agent/traces*` only return the caller's runs.

### Follow-up resolution

Rules in `agent/memory.py`, applied in `guard_in` only when the current question names no listed target (or only a new one). Each rewrite is visible in `route_reasons` and in the `effective_query`.

| Case | Example | Rewrite | Reason code |
|---|---|---|---|
| Pronoun, one target in the most recent turn that named any (entity-less turns in between are skipped) | 茅台市盈率 → 最新CPI → "它的市净率呢" | 贵州茅台的市净率呢 | `coreference:它->贵州茅台` |
| Plural | 茅台… → 五粮液… → "这两家哪个估值更高" / "Compare both on P/B" | 贵州茅台和五粮液哪个估值更高 | `coreference:这两家->…` |
| Ellipsis, no target | 贵州茅台的市盈率 → "ROE呢", "最近走势怎么样", "And ROE?" | 贵州茅台ROE呢 / ROE for 贵州茅台? | `ellipsis:target->贵州茅台` |
| Ellipsis, new target only | 贵州茅台的市盈率 → "换成五粮液呢", "What about BYD?" | 五粮液的市盈率呢？ / What is 比亚迪's P/E? | `ellipsis:aspect->市盈率` |

Guards against over-reach: only short questions (≤ 20 characters, or ≤ 8 English words) with an ellipsis marker (那/呢/换成/and/what about…) or a bare aspect qualify; market-wide (大盘, 行业, market, sector) and macro questions (CPI呢) are never attached to the previous company; an ambiguous pronoun (two candidates) is not guessed.

Without history the same questions are clarified, not refused: a metric with no company ("市净率是多少", "PB呢", reason `metric_without_target`; definition questions such as "什么是市净率" are exempt), and dangling references ("那家公司最近有公告吗", `dangling_reference`). Fuzzy concept hits whose name is not in the question are dropped first (`dropped_fuzzy_concept:有色金属` for "…有公告…").

### Session memory card

`session_memory(turns, query)` builds a small extractive card that the agent's user message carries as "Session memory (from earlier turns)": `recent_targets` (up to 6 distinct listed entities, newest first), `user_constraints` stated at any earlier turn (`risk:conservative` / `risk:aggressive`, `horizon:long` / `horizon:short`, `scope:a_shares_only`, `scope:etf_only`) and `stated_holdings` ("我持有招商银行", "I own …", up to 5). It is rule-based and bounded; no LLM summarisation.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/agent/chat` | One turn. Body: `AgentChatRequest`. Returns `AgentChatResponse`. |
| `POST` | `/agent/chat/stream` | Same, as Server-Sent Events (below). |
| `POST` | `/agent/resume` | Answer a pending clarification. Body: `AgentResumeRequest`. 409 if nothing is pending. |
| `GET` | `/agent/sessions/{session_id}` | Session history and any pending clarification (404 for another owner's session). |
| `POST` | `/agent/claim-check` | `{"claim": "茅台市盈率只有15倍，股价跌了5%"}` → per-number checks (`supported` within the written precision or 2%, 5% with 约/about; `contradicted` with the actual value; `unverifiable`), an overall verdict (`supported`, `contradicted`, `partially_supported`, `unverifiable`), evidence ids, sources, as-of dates and a disclaimer. Deterministic: classical NLU for targets, the verifier's number extraction, the price and fundamentals tools; no LLM. |
| `POST` | `/agent/feedback` | `{"trace_id", "rating": "up" \| "down", "comment"?, "session_id"?}`. Appended with the query and route to `QI_FEEDBACK_PATH` (default `outputs/feedback/feedback.jsonl`), counted in `finsight_feedback_total`; 404 if the trace is not the caller's. `scripts/feedback_to_tasks.py` turns flagged traces into candidate evaluation tasks for review. |
| `GET` | `/agent/traces`, `/agent/traces/{trace_id}` | Recent run summaries and full traces, the caller's own only ([details](a2a-and-observability.md#run-inspector)). |
| `GET` | `/metrics`, `/sources/health[?probe=1]` | Prometheus metrics; live source status, with an opt-in, rate-limited active probe ([details](data-sources.md#health-endpoint)). |
| `POST` | `/chat` | Existing endpoint, unchanged by default: `mode` omitted or `workflow` runs the original pipeline. `mode=auto` or `agent` hands the turn to the agent and returns `AgentChatResponse`. |

JSON Schemas, generated from `query_intelligence/contracts.py`:

- [`schemas/agent_chat_request.schema.json`](../schemas/agent_chat_request.schema.json)
- [`schemas/agent_resume_request.schema.json`](../schemas/agent_resume_request.schema.json)
- [`schemas/agent_chat_response.schema.json`](../schemas/agent_chat_response.schema.json)

Regenerate them with `python -m scripts.export_agent_schemas`. `tests/test_agent_schemas.py` fails if they are stale and validates real responses against them.

Example:

```bash
curl -s localhost:8000/agent/chat -H 'Content-Type: application/json' \
  -d '{"query": "贵州茅台的市盈率是多少", "session_id": "demo", "mode": "auto"}'
```

A response with `status: "needs_clarification"` carries `clarification.question`; answer it:

```bash
curl -s localhost:8000/agent/resume -H 'Content-Type: application/json' \
  -d '{"session_id": "demo", "reply": "贵州茅台"}'
```

SSE events from `/agent/chat/stream`: `session` first; then, as they happen, `node_start` (a node is about to run), `step` (a node finished), `tool_call`, `tool_result` and `answer_delta` (the `answer` text of the LLM's JSON draft, decoded while it streams, escapes split across chunks included); then `answer` (the verified, compliance-checked response, which replaces the streamed preview) or `clarification`; finally `done`. An `error` event is sent if the run fails. The graph runs on a worker thread that owns the session lock, so a client that disconnects does not leave the session locked.

The browser page at `/` uses these endpoints: pick a mode, watch steps stream in, answer clarifications inline, open "How this answer was produced" for tools and verification, and click suggested next questions.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` (or `deepseek.api_key` in config) | unset | Enables the LLM agent path and LLM composition. Without it everything runs on the deterministic path. |
| `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_THINKING_TYPE`, `DEEPSEEK_REASONING_EFFORT`, `DEEPSEEK_MAX_TOKENS`, `DEEPSEEK_TIMEOUT_SECONDS` | see `config/app_config.json` | LLM settings shared with `/chat`. Any OpenAI-compatible endpoint works, including gateways that wrap responses in `{"data": ...}`. |
| `DEEPSEEK_REASONING_STYLE` | `auto` | How per-node reasoning levels are sent: `deepseek` (`thinking` + `reasoning_effort`), `openrouter` (`reasoning` object, e.g. the Cline gateway) or `none`; `auto` picks from the base URL. |
| `QI_LLM_FALLBACK_MODELS` | unset | Comma-separated failover models on the same endpoint. Each model has a breaker: open after 3 consecutive failures, a trial call after 60 s (half-open), closed on success; `FallbackLLM.stats()` reports `closed` / `open` / `half_open` and `/metrics` exports it ([details](a2a-and-observability.md#llm-gateway-failover-and-cost)). |
| `QI_PROMPT_VERSION` | `v3` | Active prompt version from the registry in `agent/prompts.py` (`v1`, `v2`, `v3`). |
| `QI_LLM_PRICE_INPUT_MISS`, `QI_LLM_PRICE_INPUT_HIT`, `QI_LLM_PRICE_OUTPUT`, `QI_LLM_PRICE_CURRENCY` | unset | Price per million tokens. Without them the gateway-reported cost (`usage.cost`, USD) is used when present. |
| `QI_LLM_USD_CNY` | unset | Exchange rate to report gateway costs in CNY. |
| `QI_AGENT_CHECKPOINT_DB` | unset (memory) | Session persistence: a SQLite file path, or a `postgresql://` DSN shared by several processes or replicas. |
| `QI_AGENT_DURABILITY` | `exit` | LangGraph durability: one checkpoint per run (`exit`), or per step (`async`, `sync`); see [performance.md](performance.md). |
| `QI_A2A_ENABLED`, `QI_A2A_MODE`, `QI_PUBLIC_BASE_URL` | `1`, `auto`, `http://127.0.0.1:8765` | A2A endpoint switch, agent mode and the URL advertised in the agent card. |
| `QI_AGENT_REQUEST_TIMEOUT_S` | `120` | Per-request timeout for `/agent/chat` and `/agent/resume` (504 on expiry). |
| `QI_AGENT_SENTIMENT_BACKEND` | `classical` | `finbert` to use the FinBERT sentiment model (needs `torch`/`transformers`). |
| `QI_AGENT_TRACE_DIR` | `outputs/traces` | Where JSON traces are written; `off` disables them. |
| `QI_AGENT_OTEL`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS` | unset | Export traces as OpenTelemetry spans over OTLP/HTTP (Jaeger, Tempo, Langfuse, …). |
| `QI_API_KEYS` | unset | Comma-separated API keys; when set, all endpoints except `GET /health`, `GET /` and `/static/*` need `X-API-Key` or `Authorization: Bearer`. |
| `QI_RATE_LIMIT_PER_MINUTE` | `0` (off) | Per-client token bucket; 429 with `Retry-After`. |
| `QI_CORS_ORIGINS` | unset | Comma-separated allowed browser origins. |
| `QI_MAX_REQUEST_BYTES` | `1048576` | Larger bodies get 413. |
| `QI_SOURCE_CALL_TIMEOUT_SECONDS`, `QI_SOURCE_FAILURE_THRESHOLD`, `QI_SOURCE_COOLDOWN_SECONDS`, `QI_SOURCE_CACHE`, `QI_SOURCE_MAX_WORKERS`, `QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS`, `QI_SOURCE_CROSS_CHECK` | `10`, `3`, `60`, `true`, `32`, `60`, `true` | Live data source hard timeout, circuit breaker, TTL cache, bounded call pool, active-probe rate limit and Sina/THS cross-check ([details](data-sources.md#configuration)). |
| `QI_FEEDBACK_PATH` | `outputs/feedback/feedback.jsonl` | Where `/agent/feedback` appends records. |
| `QI_TFIDF_CACHE_DIR` | unset | Directory for the fitted TF-IDF document index (keyed by a hash of the corpus). The index is always memoised per process, which cut a rebuilt service's start from about 39 s to 6 s; with this set, a restart loads it from disk (about 4.5 s) instead of refitting. The file is about 350 MB, so it is not baked into the image. |
| `QI_TEST_LIVE` | unset | Tests only: `tests/conftest.py` sets the `QI_USE_LIVE_*` defaults to `false` unless `QI_TEST_LIVE=1`, so the suite never waits for upstream sites. |

Agent budgets (`max_llm_steps=6`, `max_tool_calls=16`, `max_parallel_tools=4`, `token_budget=80000`, `max_revisions=1`, `run_deadline_s=90`, `answer_grace_s=20`) and per-node reasoning levels (`agent_reasoning=None` i.e. the client default, `compose_reasoning`, `revise_reasoning` and `final_reasoning` = `low`) are fields of `AgentConfig` in `agent/state.py`.

**Deadlines.** `run_deadline_s` stops the tool loop; it also bounds every LLM request. Each HTTP request, including retries and failover models, gets `min(DEEPSEEK_TIMEOUT_SECONDS, time left)`: tool-loop steps until `run_deadline_s`, answer-producing calls (compose, the forced final answer, revise) until `run_deadline_s + answer_grace_s`. A retry whose back-off would overrun the deadline is not attempted, and with less than 2 s left the call fails fast with a non-retryable error, so the graph answers from its deterministic path (template or repaired draft). With the defaults a run stays under `QI_AGENT_REQUEST_TIMEOUT_S` (120 s): in the round-1 chaos drill a slow fallback model had pushed one request to a 504 ([chaos drill](a2a-and-observability.md#chaos-drill)); the bound has not yet been re-measured under load.

## Observability

Every run returns a `trace_id` and per-node `spans`. Traces are written as JSON to `outputs/traces/<date>/<trace_id>.json` (gitignored) and, when OTLP is configured, exported as spans with node, tool, and LLM timings, token usage, cost, prompt versions and errors. `docker/docker-compose.yml` has a `tracing` profile that starts Jaeger. `GET /agent/traces` and `GET /agent/traces/{trace_id}` serve recent traces (the web UI's run inspector) and `GET /metrics` exposes Prometheus counters and histograms; see [a2a-and-observability.md](a2a-and-observability.md).

## Running

```bash
pip install -r requirements.txt
uvicorn query_intelligence.api.app:create_app --factory --port 8000   # UI at http://localhost:8000/
python -m query_intelligence.agent.mcp_server --transport stdio        # MCP server (add --offline to disable live data)
docker compose -f docker/docker-compose.yml up --build                 # container (see docker/Dockerfile)
```

## Evaluation

The agent is evaluated offline on replayed tool snapshots so results are reproducible; see [Agent Evaluation](agent-eval.md) for the task sets, scoring, ablation, fault injection, and limitations. Commands:

```bash
python -m evaluation.agent_eval.runner --mode workflow      # dev set on the replay snapshot
python -m evaluation.agent_eval.ablation                    # legacy vs workflow (offline)
python -m evaluation.agent_eval.fault_injection             # graceful degradation
python -m evaluation.agent_eval.gate                        # CI thresholds
```

Online numbers (LLM agent, pass^k over repeated runs) need `DEEPSEEK_API_KEY` and are reported separately from offline numbers.

## Tests

```bash
python -m pytest -q tests/test_agent_*.py tests/test_api_security.py
python -m pytest -q tests/test_web_ui.py      # headless Chromium via Playwright
```

All agent tests run offline: `ScriptedLLM` replays fixed assistant turns and `tests/agent_fakes.py` provides stub tools.

## Limits

- The agent path is only as good as the LLM behind it. Offline evaluation measures the deterministic path and the graph's safety checks; the online evaluation in [agent-eval.md](agent-eval.md) measures two flash-class models (DeepSeek V4.1 Flash, GLM-5.3 Flash) through one gateway, and the agent loop's advantage over LLM composition is not significant with DeepSeek.
- Numeric verification is claim-level (2.6% false-accept rate on 2,314 corrupted gold answers, `evaluation/results/verifier_stress.json`), but it does not check that a number is used for the right period or metric when the cited evidence holds several, and a number planted in a document passes because it is in the evidence.
- Follow-up resolution is rule-based: it covers pronouns, plurals and short elliptical questions; longer paraphrases ("回到刚才那只股票…") and ambiguous references lead to a clarification rather than a guess.
- English aliases cover the major A-shares added in round 2 plus what `data/runtime/alias_table.csv` contains.
