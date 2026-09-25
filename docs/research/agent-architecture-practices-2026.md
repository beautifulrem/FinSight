# Agent Architecture and Prompt Engineering Practice (2025–2026), Applied to FinSight

Research date: 2026-09-25. Scope: current practice among AI-agent engineers (architecture, context engineering, tools, memory, guardrails, evaluation) and prompt engineering for production agents, compared against the FinSight agent layer as it exists in the working tree on that date (including the uncommitted `FallbackLLM`, gateway cost accounting, A2A and Prometheus changes).

How to read this document:

- A statement followed by a source tag such as **[S3]** was checked against that source on 2026-09-25. The source list with URLs and dates is at the end.
- A statement about FinSight code cites the file and function. These were read directly from the repository.
- Anything marked **Opinion** or **Assessment** is my judgement, not a sourced fact. This includes all effort estimates and all "fits / doesn't fit" verdicts.
- The project rule in `AGENTS.md` is treated as fixed: classical NLU and retrieval stay the backbone. No recommendation replaces them with an LLM router or vector retrieval.

---

## 0. Summary

FinSight already implements most of the patterns that 2025–2026 sources recommend for a single-domain, compliance-sensitive agent. These include a workflow-first hybrid with classical routing, deterministic gates, an evaluator-optimizer loop driven by code rather than by an LLM, budgets, interrupt-based human-in-the-loop, checkpointed sessions, tool schemas with retries and timeouts, prompt-injection envelopes, pass^k, dev/holdout splits and fault injection.

The gaps that matter are narrower:

1. **The LLM path has not been measured.** There are no online numbers yet (design notes §4).
2. **Numeric verification is permissive.** A number counts as supported if it appears anywhere in the run's evidence, within 13 unit scales and a 0.5% tolerance. It is not tied to the evidence the claim cites.
3. **Context handling inside the tool loop is unoptimised.** Tool payloads are duplicated, truncation can produce invalid JSON, the context has no per-call budget, and tool results are never cleared.
4. **Prompts have no versioning or registry.** Prompt versions are not recorded in traces or eval reports.
5. **Prompt-caching behaviour and per-node reasoning effort are not measured or controlled.**

Top recommendations, in priority order (details in §3):

| # | Recommendation | Value | Effort (estimate) |
|---|---|---|---|
| 1 | Run and publish the online (real-LLM) evaluation: pass^3, cost, latency, cache hit rate, first-pass verification rate | High (enabler) | 0.5–1 day + API spend |
| 2 | Claim-level citation binding in the verifier, plus a measured verifier false-accept rate | High | 2–3 days |
| 3 | Token-efficient, restorable tool observations (no duplicate payloads, valid-JSON truncation with hints, context budget, optional tool-result clearing) | High | 1–2 days |
| 4 | Prompt registry and versioning (id, version, hash in traces and eval reports) and a prompt regression gate | High | 0.5–1 day |
| 5 | Actionable tool errors (`hint` field, "when not to use" in descriptions) | Medium–High | 0.5 day |
| 6 | KV-cache-aware request layout (stable tools and `tool_choice="none"` for forced finals, deterministic JSON) and a cache-hit metric | Medium–High | 0.5–1 day |
| 7 | Restructure system prompts (tagged sections, rationale, effort scaling, 1–2 canonical examples), shipped only through an eval A/B | Medium | 1 day + eval |
| 8 | Prompt-injection red-team suite with attack-success rate | Medium | 1 day |
| 9 | Per-node reasoning effort and model routing (thinking on for the tool loop, off or low for compose and revise) | Medium | 0.5–1 day |
| 10 | Calibrated LLM-as-judge for dimensions that code cannot check (offline only) | Medium; resume padding if uncalibrated | 2 days |

Low-fit items or resume padding for this project (§4): multi-agent role play, a Skills system, LLM-based routing, vector memory, conversation summarisation/compaction, a "think" tool, dynamic tool loading and DeepSeek strict mode.

---

## 1. What current sources say

### A. Agent architecture

**Workflows vs agents.**
- Anthropic defines *workflows* as "LLMs and tools orchestrated through predefined code paths" and *agents* as systems where "LLMs dynamically direct their own processes and tool usage" [S1].
- It recommends "finding the simplest solution possible, and only increasing complexity when needed" [S1].
- It names five workflow patterns: prompt chaining (with programmatic "gates"), routing, parallelization (sectioning / voting), orchestrator-workers and evaluator-optimizer [S1].
- It says routing works when "classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm" [S1]. This directly legitimises FinSight's classical router.
- Its three principles are simplicity, transparency ("explicitly showing the agent's planning steps") and a carefully crafted agent-computer interface (ACI) [S1].

**Agent definition in 2025.** Anthropic now uses "LLMs autonomously using tools in a loop" [S3]. The loop "LLM picks next step → deterministic code executes → result appended" is also 12-factor-agents' model. 12-factor's factors include "Own your prompts", "Own your context window", "Own your control flow", "Compact errors into context window", "Small, focused agents", "Launch/Pause/Resume with simple APIs" and "Make your agent a stateless reducer" [S11]. Its author observes that most production "agents" are "mostly deterministic code, with LLM steps sprinkled in" [S11].

**Single vs multi-agent.**
- OpenAI recommends to "maximize a single agent's capabilities first". It suggests splitting into more agents when the model fails complicated instructions or consistently selects incorrect tools [S12].
- Anthropic reports that its orchestrator-worker Research system beat a single agent by 90.2% on an internal research eval [S4].
- The same source reports that agents use about 4× the tokens of chat and multi-agent systems about 15× [S4].
- It adds that domains "that require all agents to share the same context or involve many dependencies between agents are not a good fit" [S4].

**Harness.**
- Anthropic's eval glossary defines an *agent harness* (scaffold) as "the system that enables a model to act as an agent: it processes inputs, orchestrates tool calls, and returns results". It notes that evaluating "an agent" means evaluating harness plus model [S5].
- For long-running agents, Anthropic uses an initializer session plus incremental sessions that leave a progress file and git history. It prefers JSON over Markdown for the progress file because models are "less likely to inappropriately change or overwrite JSON files" [S7].
- OpenAI published "Harness engineering: leveraging Codex in an agent-first world" on 2026-02-11 [S16]. Only the title, date and premise were verified.
- In China, harness engineering, runtime engineering, evaluation and safety fallbacks, and multi-agent orchestration are described as the three new high-frequency interview areas of H2 2026 [S29].

**Context engineering.**
- Anthropic: "find the smallest possible set of high-signal tokens". System prompts should sit at the "right altitude", between brittle if-else prompts and vague guidance, and be organised into sections with XML tags or Markdown headers [S3].
- Its long-horizon techniques are compaction, structured note-taking and sub-agents. "Tool result clearing" is called "one of the safest lightest touch forms of compaction" [S3].
- For "just in time" retrieval, the agent keeps lightweight identifiers and loads data on demand. A hybrid of up-front plus just-in-time retrieval "might be better suited for contexts with less dynamic content, such as legal or finance work" [S3].
- LangChain groups the strategies as write, select, compress and isolate [S23].

**KV-cache and prompt layout.**
- Manus calls KV-cache hit rate "the single most important metric for a production-stage AI agent". It reports a ~100:1 input:output ratio and gives these rules: keep the prompt prefix stable (no timestamps at the top), keep context append-only, serialise deterministically, and "mask, don't remove" tools, because tool definitions sit near the front of the context and changing them invalidates the cache [S10].
- OpenAI's caching guide says cache reuse needs the rendered prefix to match, and that changing `tools` or `text.format` affects the prefix. It recommends keeping tool definitions stable and changing *which tools are callable* (for example `tool_choice: "none"`, `allowed_tools`) instead [S13].
- DeepSeek caching is on by default. Cache units are persisted at request boundaries, and a later request must fully match a unit. Usage reports `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` [S17].

**Restorable compression and offloading.**
- Manus designs compression to be restorable: drop page content but keep the URL. It treats the file system as external memory, keeps failed actions in context ("keep the wrong stuff in"), and recites a todo list to keep goals in recent attention [S10].
- Tencent Cloud reports, for TencentDB Agent Memory, a scheme that offloads full tool results to reference files and keeps a compact JSONL record plus a task canvas. It claims about 9.9% improvement on SWE-bench and 31–33% token savings (vendor-reported, not independently verified) [S31].

**Memory.**
- LangGraph separates short-term memory (checkpointer: thread-scoped state, human-in-the-loop, time travel, fault tolerance) from long-term memory (store: cross-thread, namespaced) [S24].
- LangChain classifies long-term memory as semantic (facts), episodic (experiences) and procedural (rules). Memory can be updated "on the hot path" or in the background [S25].

**Tool design (ACI).**
- Build few, high-impact tools rather than wrapping every endpoint [S2].
- Namespace related tools [S2].
- Return high-signal, human-readable fields [S2].
- Offer a `response_format` ("concise"/"detailed") option [S2].
- Paginate, filter or truncate with sensible defaults, and steer the agent when truncating [S2].
- Make error responses "specific and actionable … rather than opaque error codes or tracebacks" [S2].
- Write descriptions as for a new hire, with unambiguous parameter names [S2].
- Evaluate tools with realistic multi-call tasks and held-out test sets [S2].
- "Poka-yoke" the arguments. Anthropic's example is switching to absolute file paths [S1].

**Guardrails and human-in-the-loop.**
- OpenAI's Agents SDK has input, output and tool guardrails with "tripwires". Input guardrails can run in parallel with the agent (lower latency, but tokens or tools may already be spent) or blocking [S15].
- OpenAI's guide describes guardrails as layered, "from input filtering and tool use to human-in-the-loop intervention" [S12].
- The prompt-injection design-patterns paper (arXiv 2506.08837) proposes six patterns: action-selector, plan-then-execute, LLM map-reduce, dual LLM, code-then-execute and context-minimization. Each trades flexibility for security [S27]. Pattern descriptions were taken from a summary of the paper; the arXiv entry itself was confirmed.

**Durable execution.** Anthropic's Research system resumes from where it failed rather than restarting. It combines "retry logic and regular checkpoints", uses full production tracing, and uses "rainbow deployments" for prompt and tool changes to long-running agents [S4].

**Skills.** An Agent Skill is a directory with `SKILL.md` whose `name` and `description` are pre-loaded into the system prompt. The body is loaded only when relevant ("progressive disclosure") [S6]. In China, "MCP vs Skills" is among the most-asked tool-calling interview questions [S29].

**Evaluation.**
- Anthropic's vocabulary: task, trial, grader, transcript, outcome, evaluation harness [S5].
- Grader types: code-based, model-based and human, used together [S5].
- Capability evals should start at a low pass rate. Regression evals should be near 100% [S5].
- pass@k vs pass^k: pass^k matters "for customer-facing agents" [S5].
- Start with 20–50 tasks from real failures [S5].
- Build balanced sets (the behaviour should *and* should not occur) [S5].
- Grade outcomes rather than exact tool sequences [S5].
- For LLM judges: calibrate with humans, give an "Unknown" way out, and grade each dimension with an isolated judge [S5].
- Research-agent evals should combine groundedness, coverage and source-quality checks [S5].
- Anthropic's Research judge scores factual accuracy, citation accuracy, completeness, source quality and tool efficiency [S4].
- Google ADK ships `tool_trajectory_avg_score`, `hallucinations_v1` (groundedness), rubric-based response and tool-use criteria, and safety criteria [S26].

### B. Prompt engineering for production agents

**Structure.**
- Use XML tags to separate instructions, context, examples and inputs [S9].
- Give context and motivation ("explaining … why such behavior is important") rather than bare rules [S9].
- Put long documents above the query [S9].
- Examples should be relevant, diverse and wrapped in example tags [S9].
- Anthropic warns against stuffing "a laundry list of edge cases" and prefers "diverse, canonical examples" [S3].
- Manus warns that repetitive few-shot action/observation patterns make agents drift ("don't get few-shotted") [S10].

**Reasoning models.**
- OpenAI: keep prompts simple and direct, avoid "think step by step" prompts, use delimiters, "try zero shot first, then few shot if needed", and be specific about the end goal [S14].
- DeepSeek thinking mode is enabled by default with default effort `high`. It ignores `temperature`. When a request carries `tools`, the `reasoning_content` of earlier turns must be passed back or the API returns 400 [S19].

**Structured output.**
- DeepSeek JSON mode (`response_format: json_object`) requires the word "json" in the prompt plus an example of the format. It requires a sensible `max_tokens`, and "the API may return empty content" [S20].
- DeepSeek tool `strict` mode (beta) requires every property to be `required`, `additionalProperties: false`, and does not support `minLength`/`maxLength` [S18].
- DeepSeek `tool_choice` supports `none` / `auto` / `required` / a named function [S21]. Zhipu GLM's function calling supports only `auto` [S22].

**Prompt management.** 12-factor: "Own your prompts" [S11]. Langfuse recommends versioned prompts with labels and to "link prompts to traces to analyze performance by prompt version" [S28].

**Eval-driven iteration.** Anthropic recommends practising "eval-driven development", reading transcripts, and treating evals as regression tests on every change and model upgrade [S5]. Anthropic also improved tools by giving agents failure transcripts. A tool-testing agent that rewrote a flawed tool description cut task completion time by 40% for later agents [S4].

**Prompt injection.** See the design patterns above [S27]. The structural patterns (plan-then-execute, context minimization) matter more than lexical filters, because each pattern works by keeping untrusted data from choosing actions [S27].

---

## 2. Gap analysis against FinSight

Legend: **Has**, **Partial**, **Lacks** (facts about the code). **Fit** is my assessment (Opinion).

### A. Architecture

| Pattern | FinSight status (file / function) | Fit (Opinion) |
|---|---|---|
| Workflow-first, agent only when needed [S1] | **Has.** `router.decide_route` sends simple questions to the deterministic `execute_plan` and complex ones to `agent_llm`. Without an LLM it downgrades to workflow (`graph.guard_in`). | Core strength. Keep. |
| Routing with a classical classifier [S1] | **Has.** `router.decide_route` (NLU style/intents/entities plus lexical markers, with reasons recorded in `route_reasons`). `apply_finance_overrides` corrects NLU false out-of-scope decisions. | Exactly as the source allows. Keep classical. |
| Prompt chaining with programmatic gates [S1] | **Has.** `execute_plan → compose → verify → compliance`. The gate is `verifier.verify_answer`. | Keep. |
| Parallelization (sectioning) [S1] | **Has.** `AgentRuntime._run_tools` runs up to `max_parallel_tools=4`. The agent prompt asks for parallel tool calls (`prompts.AGENT_SYSTEM_PROMPT`). | Keep. Voting (n drafts) is not worth the cost. |
| Orchestrator-workers / sub-agents [S1][S4] | **Lacks** (by design; design notes §2). | Low fit. Tasks are short and share one evidence store; the token multiplier in [S4] is not justified. |
| Evaluator-optimizer [S1] | **Has.** `verify ⇄ revise` (`max_revisions=1`). The evaluator is deterministic code, not an LLM. Then `repair_answer` removes unsupported clauses. | Strong. The evaluator's *precision* is the gap (see Rec 2). |
| ReAct loop [S3][S11] | **Has.** `agent_llm ⇄ agent_tools` with step, tool-call, token and deadline budgets (`AgentConfig`). A budget stop forces a final answer (`force_final_message`). | Keep. |
| Plan-and-execute | **Has (deterministic).** `planner.plan_from_nlu` gives each planned call a reason. **Lacks** an LLM-written plan. | The deterministic plan is also the plan-then-execute injection defence [S27]. An LLM planner adds little here. |
| Reflection | **Partial.** The revision is grounded in verifier feedback (`revision_message`), not open-ended self-critique. | Grounded reflection is the better variant for finance. Keep. |
| Harness: retries, timeouts, cache, normalised errors | **Has.** `tools/base.ToolRegistry.run` (timeouts, exponential backoff, TTL cache, `ToolError` codes). LLM retries are in `DeepSeekToolClient._post_with_retries`. The working tree adds `FallbackLLM` with a per-model circuit breaker (`llm.py`). | Strong. |
| Graceful degradation | **Has.** LLM failure falls back to planner plus template. Reasons are listed in `degraded`. The fault-injection suite reports a 1.00 graceful-degradation rate (design notes §4). | Strong interview story. |
| Human-in-the-loop [S12] | **Has.** `graph.clarify` uses `interrupt()`, resumed by `/agent/resume`, at most one round. | Tools are read-only, so approval gates for tools are unnecessary. |
| Durable execution [S4][S24] | **Partial.** Checkpointer (in-memory or SQLite, `memory.make_checkpointer`), per-session lock, interrupt/resume. No resume-after-crash mid-run and no Postgres saver. | Low priority. Runs are ≤90 s and tools are idempotent reads (Opinion). |
| Observability | **Has.** JSON traces, OTLP spans with `gen_ai.*` attributes (`tracing.py`), Prometheus metrics and a recent-trace store (`telemetry.py`, working tree). | Strong. |
| Protocols | **Has.** MCP server (`mcp_server.py`, low-level `Server` sharing Pydantic schemas); A2A endpoint (`a2a_server.py`, working tree). | Enough. Don't add more protocols. |

### A. Context engineering and memory

| Pattern | FinSight status | Fit (Opinion) |
|---|---|---|
| Static prefix first (system prompt, then tools, then dynamic content) [S10][S13] | **Has.** `AGENT_SYSTEM_PROMPT` and `COMPOSE_SYSTEM_PROMPT` are constants without timestamps. Dynamic NLU context goes in the user message (`prompts.agent_user_message`). | Good. Not measured (Rec 6). |
| Append-only context [S10] | **Has, with one exception.** `agent_llm` appends. However, the budget-forced final call and `revise` call `llm.chat(messages, json_mode=True)` **without** `tools`, so the request's tool block changes at that step (`graph.agent_llm`, `graph.revise`). | Removing tools goes against [S10]'s "mask, don't remove" and [S13]'s advice. Whether DeepSeek's cache is affected is unverified (Rec 6). |
| Deterministic serialization [S10] | **Partial.** `injection.tool_message_content` and `prompts.compose_user_message` use `json.dumps` without `sort_keys`. Order is deterministic in practice (insertion order from Pydantic dumps), but not enforced. | Cheap fix (Rec 6). |
| Context budget per call | **Partial.** `token_budget=80_000` is a cumulative *spend* budget (`_total_tokens(usage)` sums all calls). The prompt size of the next call is not estimated. | Add a per-call context budget (Rec 3). |
| Token-efficient tool results [S2] | **Partial.** `ToolResult.observation()` sends both full `data` **and** `evidence[].prompt_view()`, which includes a compact `payload`, so many fields appear twice. `tool_message_content` truncates at 12,000 chars by slicing the JSON string and appending `..."}(truncated)`. That yields invalid JSON with no guidance. | Clear, cheap win (Rec 3). |
| Tool-result clearing / restorable compression [S3][S10][S31] | **Lacks** in the loop. However, every result is also stored in `state["evidence"]` keyed by `evidence_id`, and compose/verify read the store. So clearing old tool messages is *restorable by construction*. | Good fit because of the evidence store. Enable only above a threshold (Rec 3). |
| Conversation compaction / summarisation [S3] | **Partial.** `memory.turn_record` truncates answers to 600 chars. `history_messages` keeps the last 2 turns and `dialog_context_from_turns` the last 3 queries. No LLM summariser. | Low fit. Turns are short, so fixed windows are adequate (Opinion). |
| Structured notes / recitation [S3][S10] | **Lacks.** | Low fit at ≤6 LLM steps. A compact "evidence ledger" in the forced-final message is the most that is warranted (Opinion). |
| Just-in-time retrieval by reference [S3] | **Partial.** Evidence ids are stable references. Tools accept a ticker or name. No re-expand-by-id tool. | Optional `get_evidence(ids)` tool (Rec 3). |
| Short-term memory (thread) [S24] | **Has.** `turns` reducer, per-turn reset of turn-scoped fields (`state.add_or_reset`, `merge_dicts` with `__reset__`), rule-based coreference (`memory.resolve_coreference`). | Keep. The per-turn reset is a good failure-case story (design notes §3.3). |
| Long-term memory (store: semantic/episodic/procedural) [S24][S25] | **Lacks** a cross-session store. `user_profile` is per request only. Procedural "memory" lives in code (planner rules, prompts). | Limited fit, with compliance risk: remembered risk preferences edge toward suitability advice. Optional (§4). |

### A. Tools, guardrails, security

| Pattern | FinSight status | Fit (Opinion) |
|---|---|---|
| Few, high-impact tools [S2] | **Has.** 9 tools wrapping NLU, retrieval, providers, analyzer and sentiment (`tools/*.py`). | Right size. Namespacing unnecessary at 9 tools. |
| Descriptions and parameter docs [S1][S2] | **Partial.** Descriptions and field docs with examples exist (for example `market.py` `get_price_history`, `documents.py`). No "when not to use / relation to other tools" guidance. | Cheap improvement (Rec 5). |
| Poka-yoke arguments [S1] | **Has.** Tools accept a ticker *or* a name. Pydantic bounds (for example `days` 1–30). JSON repair of arguments (`base._coerce_arguments`). | Keep. |
| Actionable errors [S2][S11 factor 9] | **Partial.** Normalised codes plus messages; Pydantic `loc: msg` for invalid arguments (`_validation_message`). No hints on how to recover. Budget exhaustion returns a bare JSON error (`graph._BUDGET_EXHAUSTED`). | Rec 5. |
| `response_format` concise/detailed [S2] | **Lacks.** | Low value once Rec 3 lands (Opinion). |
| Tool-use evaluation [S2][S26] | **Has.** `required_tools` precision/recall in `evaluation/agent_eval/metrics.py`. | Keep grading outcomes, not exact trajectories [S5]. |
| Layered guardrails [S12][S15] | **Has.** Input: NLU out-of-scope and risk flags before any LLM call (`guard_in`). Tool data: injection redaction plus envelope. Output: verifier, compliance node (`compliance.apply_compliance`) and eval `forbidden_patterns`. | Strong. |
| Prompt-injection structural defences [S27] | **Partial / structurally strong.** The workflow path is plan-then-execute (the plan comes from the trusted query before any untrusted data). Tools are read-only, so no action or exfiltration channel exists. Compose sees only sanitised evidence views (context minimisation). Detection is lexical (`injection._INSTRUCTION_PATTERNS`). One fault-injection scenario (`document_injection`). | Good architecture. Measurement is thin (Rec 8). |
| Multi-agent roles [S4][S12] | **Lacks** by design. | Low fit (§4). |
| Skills / progressive disclosure [S6] | **Lacks.** | Low fit. At most "intent playbooks" chosen by NLU (§4). |

### B. Prompt engineering

| Practice | FinSight status | Fit (Opinion) |
|---|---|---|
| Sectioned system prompt with XML tags [S3][S9] | **Partial.** Plain-text sections ("How to work", "Rules you must never break") in `prompts.py`. No tags, no rationale, no effort scaling. | Rec 7. |
| Explain why a rule exists [S9] | **Lacks.** Rules are bare imperatives. | Rec 7. |
| Canonical few-shot examples [S3][S9] | **Lacks** in the agent prompts. The legacy `scripts/llm_response.py` has a few-shot bank (`load_few_shot_bank`, `select_few_shots`). | 1–2 static examples in compose. Zero-shot first for the reasoning model [S14]. Rec 7. |
| Structured output | **Partial.** `response_format: json_object` plus `composer.parse_answer` (JSON repair, fallback to raw text). No Pydantic model for the draft. Claims are not structured. | Rec 2 (claim structure). DeepSeek strict mode is only for tool calls and conflicts with `max_length` fields [S18]. |
| JSON-mode requirements [S20] | **Has.** `ANSWER_CONTRACT` includes "JSON" and a shape. **Partial** handling of empty content: `parse_answer` returns an empty answer, which then goes through verify/revise/repair instead of falling back to the template. | Small fix, can ride with Rec 2. |
| Prompt templates as code | **Has.** Builder functions in `prompts.py`, tested with `ScriptedLLM`. | Keep. |
| Prompt versioning / registry, prompt↔trace link [S11][S28] | **Lacks.** No prompt id, version or hash in `llm_log`, traces or eval reports. | Rec 4. |
| Prompt caching layout and measurement [S10][S13][S17] | **Partial.** Layout is good. `prompt_cache_hit_tokens` is recorded per call (`graph._llm_entry`) and in Prometheus (`telemetry.py`), but no hit ratio is reported in eval, and no online run exists. | Rec 6. |
| Eval-driven iteration, regression gate [S5] | **Has offline.** 207 dev + 53 holdout tasks, replay snapshots, `gate.py`, fault injection, pass^k. **Lacks online** numbers. | Rec 1. |
| LLM-as-judge with rubric [S4][S5][S26] | **Lacks.** All graders are code-based. | Rec 10, offline only. |
| Reasoning-model differences [S14][S19] | **Partial.** `reasoning_content` is passed back (`AssistantTurn.as_message`). Temperature is skipped when thinking is enabled (`DeepSeekToolClient.chat`). One global `thinking_type`/`reasoning_effort`; no per-node control. | Rec 9. |

---

## 3. Recommendations (prioritised)

Each item lists what to build, where, which interview question it answers, estimated effort (Opinion), how to measure it, and whether it is **high value** or closer to **resume padding**. Interview questions are quoted from the 2026 Chinese question bank [S29] unless noted.

### Rec 1: Run and publish the online evaluation of the LLM path (High value; enabler for everything else)

- **What.** Run the existing online modes (`python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3` and the runner) through the configured gateway. `llm.py` already handles gateway envelopes and provider-reported cost. Publish the following, separately from offline numbers as `AGENTS.md` requires, with command, date and commit:
  - dev and holdout pass^3 by route and category;
  - tool precision/recall;
  - **first-pass verification rate** (drafts that pass `verify` without `revise`/`repair`);
  - revise rate and `verification_failed:repaired` rate;
  - P50/P95 latency, reasoning tokens, cost per task;
  - cache-hit ratio.
- **Files.** `evaluation/agent_eval/runner.py`, `ablation.py`, `metrics.py` (add first-pass verification and cache-hit ratio if missing), `docs/agent-eval.md`.
- **Why / interview.**
  - "Agent 效果提升的百分比用哪个测试集算出来的", "为什么要用 Agent 而不是直接调大模型", "搭一个大模型应用，如何在单次提示词 / Workflow / Agent 三种方案之间选型".
  - Design notes §4 say explicitly that "LLM 工具循环是否优于固定 workflow" is unanswered.
  - Anthropic: run multiple trials and use pass^k for customer-facing agents [S5].
- **Effort.** 0.5–1 day plus API spend (Opinion).
- **Measure.** The metrics above. The decision rule: keep `mode=auto` routing as is only if the agent route beats workflow on the complex categories at acceptable cost per task.
- **Also do.** Read 20–30 failing transcripts before changing anything [S5].

### Rec 2: Claim-level citation binding in the verifier (High value)

- **Problem (fact from code).** `verifier.verify_answer` accepts a number if it matches **any** number in **any** evidence item of the run (`_evidence_numbers(store)`). The match allows 13 unit scales (`_SCALES`) and a tolerance of max(0.011, 0.5%). With price histories (10 closes), indicators, fundamentals and macro series in one store, chance matches are plausible, and a correct number can be attributed to the wrong security or period. This is the limitation already listed in `docs/agent.md` ("checks that numbers appear in the evidence, not that they are used correctly").
- **What.**
  1. Tighten the contract: "put the evidence id immediately after the sentence that contains the number". Templates in `composer.py` already do this.
  2. In `verify_answer`, split into sentences or clauses (reuse `_split_sentences` and the clause split in `repair_answer`). For each clause, check its numbers against the numbers of the evidence ids **cited in that clause**. Fall back to evidence of the same entity when no id is cited.
  3. Report a new `misattributed_numbers` field (number exists elsewhere in the store but not in the cited evidence) separately from `unsupported_numbers`.
  4. `repair_answer` drops misattributed clauses.
  5. Optionally, for fundamentals, require the clause to mention the evidence's report period (design notes "下一步" #4).
  6. Treat empty or unparseable LLM content as an LLM failure and fall back to the template [S20].
- **Files.** `query_intelligence/agent/verifier.py`, `prompts.py` (`ANSWER_CONTRACT`, `revision_message`), `composer.py` (`parse_answer`), `evaluation/agent_eval/metrics.py`, `build_tasks.py` / `build_holdout.py` (adversarial cases), tests.
- **Why / interview.**
  - "幻觉率怎么定义和计算？降幻觉的数字是怎么量化出来的", "如果检测 Agent 自身也会幻觉，工程上如何保证检测 Agent 可靠" (answer: it is deterministic code with a measured false-accept rate), "前序步骤产生幻觉、导致链式积累时应该如何处理".
  - Groundedness checks that claims are supported by the retrieved sources are a core research-agent grader [S5]. Anthropic added a dedicated citation stage to its Research system [S4].
- **Effort.** 2–3 days (Opinion).
- **Measure.** (a) **Verifier false-accept rate**: take gold answers, perturb each number by ±1–20%, and count how many the verifier still accepts, before and after. (b) **Misattribution recall**: swap numbers between entities in comparison answers. (c) No regression in offline dev/holdout pass^1. (d) Online first-pass verification rate (Rec 1).

### Rec 3: Token-efficient, restorable tool observations (High value; context engineering)

- **What.**
  1. `ToolResult.observation()` should send one canonical compact view per evidence item (`evidence_id` plus the fields the model needs) instead of both `data` and `evidence[].prompt_view()`.
  2. Replace character slicing in `injection.tool_message_content` with structured truncation. Cap documents to top-k and excerpts to N chars, **keep valid JSON**, and add `"truncated": {"omitted": n, "hint": "narrow the query or targets"}` [S2].
  3. Add `AgentConfig.context_budget_tokens` and estimate the next request's size in `agent_llm`.
  4. Above the threshold, replace older tool messages with a stub (`{"evidence_ids": [...], "note": "full result stored; cite by id"}`). This is tool-result clearing [S3] made restorable by the evidence store [S10]. Clear rarely, in batches, because every clearing invalidates the cache after that point (Opinion, consistent with [S10][S13]).
  5. Optional: a `get_evidence(evidence_ids)` tool for just-in-time re-expansion [S3].
- **Files.** `query_intelligence/agent/tools/base.py` (`ToolResult.observation`), `injection.py` (`tool_message_content`), `evidence.py` (`prompt_view`, `_compact_payload`), `graph.py` (`agent_llm`, `agent_tools`), `state.py` (`AgentConfig`), optional new tool in `tools/`.
- **Why / interview.** "工具调用结果怎么压缩——原地压缩还是交给另一个模型处理？对执行效果和耗时有什么影响", "除了压缩，还有哪些规避上下文窗口限制的方案？大结果卸载在工程上怎么实现", "上下文工程应该如何设计". The answer draws on [S3][S10][S31] and is backed by FinSight's own numbers.
- **Effort.** 1–2 days (Opinion).
- **Measure.** Mean and P95 prompt tokens per LLM call and per run (online). Valid-JSON rate of tool messages in the `huge_documents` fault scenario. Task success non-inferior to baseline (Rec 1). Cache-hit ratio (Rec 6).

### Rec 4: Prompt registry, versioning and a prompt regression gate (High value, cheap)

- **What.**
  - Register each prompt (`agent_system`, `compose_system`, `force_final`, `revision`, `answer_contract`) with `id`, `version` and a sha256 of the rendered static text.
  - Record `prompt_versions` in each `llm_log` entry (`graph._llm_entry`), as OTel span attributes (`tracing.py`), and in the eval summary (`runner.summarize`).
  - Add a test that fails when a prompt hash changes and the recorded eval (dev offline plus, when available, online) was not re-run for that hash, for example a `prompts.lock.json` holding hash → eval run id.
  - Keep it in-repo. Langfuse is optional because OTLP export already exists [S28].
  - Put it in traces and reports, not in `AgentChatResponse`, to avoid a public contract change (`AGENTS.md` requires schema, README and test updates for contract changes).
- **Files.** `query_intelligence/agent/prompts.py`, `graph.py`, `tracing.py`, `evaluation/agent_eval/runner.py`, `gate.py`, new `tests/test_agent_prompts.py`.
- **Why / interview.** "多步任务失败时如何区分是 Prompt、模型、工具还是逻辑问题" [S29], and the common follow-up "prompt 改了怎么保证不回归" (Opinion: a frequent follow-up; not verified as a listed question) (answer: every trace carries the prompt hash, model, tool versions and route reasons). Relevant sources: "Own your prompts" [S11]; link prompts to traces [S28]; regression evals [S5].
- **Effort.** 0.5–1 day (Opinion).
- **Measure.** 100% of traces and eval reports carry prompt versions. The gate test blocks an edited prompt without a matching eval record.

### Rec 5: Actionable tool errors and sharper tool descriptions (Medium–High value, cheap)

- **What.**
  1. Add `hint: str | None` to `ToolError`. Fill it per code:
     - `not_found`: "No security matched 'X'. Call `resolve_entity` with the company name, or use a 6-digit code with .SH/.SZ/.BJ."
     - `invalid_arguments`: field errors plus a minimal valid example generated from the input model.
     - `timeout` / `upstream_error`: "Transient. Do not call the same tool again this turn; answer with the evidence you have and list it under limitations."
     - Budget exhausted (`graph._BUDGET_EXHAUSTED`): "Stop calling tools and answer now."
  2. Add "when not to use / relation to other tools" lines to descriptions. For example, `compute_indicators` already derives from price history, so the model does not need `get_price_history` just to get MA values.
  3. Failed calls already stay in context, which matches "keep the wrong stuff in" [S10].
- **Files.** `query_intelligence/agent/tools/base.py` (`ToolError`, `_failure`, `_validation_message`), tool handlers and descriptions in `tools/*.py`, `graph.py`.
- **Why / interview.** "如果大模型返回的 Function Call 参数格式不对，工程上怎么处理", "工具调用报错、长时间无响应、连续失败时，重试、超时与异常隔离策略怎么设计". Sources: [S2] (actionable errors, new-hire descriptions), [S1] (ACI, poka-yoke), [S11] factor 9.
- **Effort.** 0.5 day (Opinion).
- **Measure.** Recovery rate in the `malformed_tool_args` and `unknown_tool` fault scenarios. Online: `invalid_arguments` rate, rate of repeated identical calls, average tool calls per run.

### Rec 6: KV-cache-aware request layout and a cache-hit metric (Medium–High value)

- **What.**
  1. For the budget-forced final call and for `revise` on the agent path, send the **same `tools` array** with `tool_choice="none"` instead of dropping `tools`. DeepSeek supports `none` [S21]. GLM supports only `auto` [S22], so gate this with a provider capability flag and keep today's behaviour as fallback.
  2. `sort_keys=True` in `tool_message_content` and `compose_user_message`.
  3. A test asserting the static system prompts contain no dates or ids.
  4. Report `cache_hit_ratio = Σ prompt_cache_hit_tokens / Σ prompt_tokens` in eval summaries and on a dashboard over the existing Prometheus counters.
- **Caveat (fact vs hypothesis).** [S10][S13] say changing tools changes the prefix. DeepSeek documents prefix-unit matching but not where tools are serialised [S17]. The expected gain is therefore a **hypothesis to verify** with Rec 1 data. Passing back `reasoning_content` stays mandatory whenever `tools` are sent [S19]; `AssistantTurn.as_message` already does this.
- **Files.** `query_intelligence/agent/graph.py` (`agent_llm` stop branch, `revise`), `llm.py` (capability flag, `tool_choice` pass-through already exists), `injection.py`, `prompts.py`, `evaluation/agent_eval/metrics.py`, `telemetry.py`.
- **Why / interview.** "KV cache 命中率为什么重要" (Manus's "single most important metric" [S10]), "为什么不建议在迭代中途动态增删工具", "Agent 的成本怎么控制". Chinese interview-prep material covers the same cache rules (static system prompt and tools, batch compression to limit cache invalidation) [S32].
- **Effort.** 0.5–1 day (Opinion).
- **Measure.** Cache-hit ratio and cost per task before and after on the same online task set.

### Rec 7: Restructure the system prompts and ship only through an eval A/B (Medium value)

- **What.** Rewrite `AGENT_SYSTEM_PROMPT` and `COMPOSE_SYSTEM_PROMPT` into tagged sections:
  - `<role>`.
  - `<workflow>`, with **effort scaling** as in [S4]: single lookup 1–2 calls; comparison: the same calls per target, in parallel; "why" questions: price plus news, plus macro only if the NLU context shows a macro entity; stop when the evidence covers the question.
  - `<tool_guidance>`: prefer NLU-provided tickers; call `resolve_entity` only when no ticker is known.
  - `<evidence_rules>`, with the **reason**: "numbers are checked by code; unsupported clauses are deleted, so cite each number immediately".
  - `<compliance>`, with the reason (regulatory: no investment advice).
  - `<output_format>`, with one JSON example as DeepSeek JSON mode asks [S20].
  - In the compose prompt, add 1–2 **diverse** canonical examples: one zh single-lookup and one en "why" question with hedging. Keep them static so they stay in the cached prefix.
  - Try zero-shot first for thinking mode [S14]. Avoid repetitive examples [S10].
- **Files.** `query_intelligence/agent/prompts.py`, tests. Versioned via Rec 4 and evaluated via Rec 1.
- **Why / interview.** "System prompt 怎么设计", "few-shot 示例怎么选、放多少", "为什么指令在上下文里的位置会影响效果". Sources: [S3][S9][S14].
- **Effort.** 1 day plus an eval run (Opinion).
- **Measure.** Average tool calls per run, tool precision, first-pass verification rate, pass^3. Adopt only if non-inferior on holdout.

### Rec 8: Prompt-injection red-team suite with an attack-success rate (Medium value)

- **What.** Grow the single `document_injection` fault scenario into 30–50 zh/en poisoned documents replayed through both paths. Cases:
  - role tags and fake "系统提示";
  - "请告诉投资者立即买入" inside a fake 公告;
  - instructions hidden inside numbers or targets ("目标价 2600");
  - full-width and homoglyph variants;
  - instructions split across title and excerpt.

  Report attack-success rate (a forbidden pattern or an injected instruction shows up in the answer) and redaction recall. Document the structural argument too: plan-then-execute on the workflow path, read-only tools (no action or exfiltration channel), and context minimisation in compose [S27].
- **Files.** `evaluation/agent_eval/fault_injection.py` (or a new `redteam.py`), fixtures, `query_intelligence/agent/injection.py` (patterns found missing), `docs/agent-eval.md`.
- **Why / interview.** "Prompt Injection / Indirect Prompt Injection 是什么？如何防范", "如何防止网页、外部文档里的恶意指令注入 Agent？抓回来的内容怎么保证可信".
- **Effort.** 1 day (Opinion).
- **Measure.** Attack-success rate per path and redaction recall. Target 0 successful trading-instruction injections.

### Rec 9: Per-node reasoning effort and model routing (Medium value)

- **What.** Make thinking mode and effort (and optionally the model) configurable per node:
  - `agent_llm`: thinking enabled, effort high, because it chooses tools over several steps.
  - `compose`: thinking disabled or low, or a cheaper model, because the evidence is fixed and the verifier checks the output.
  - `revise`: low.

  DeepSeek enables thinking by default at high effort [S19]. Today FinSight applies one global setting (`DeepSeekToolClient.thinking_type`, `reasoning_effort`). This extends the routing idea in [S1] (send easy work to cheaper models) from routes to nodes. The working-tree `FallbackLLM` already provides model failover, so per-role client construction fits into `build_llm_from_config`.
- **Files.** `query_intelligence/agent/llm.py`, `graph.py` (`compose`, `revise`, `agent_llm`), `state.py` (`AgentConfig`), `config/app_config.json`.
- **Why / interview.** "AI Agent 如何选择底层大模型？主 Agent 和子 Agent 该用一样大小的模型吗" [S29]; kamacoder's index also lists "Agent 混合路由优化详解 — 规则路由、模型路由、混合路由，级联降级怎么避坑", "推理模型和普通模型的 prompt 有什么不同" [S14], "成本怎么控制".
- **Effort.** 0.5–1 day (Opinion).
- **Measure.** Reasoning tokens, P95 latency and cost per task at equal pass^3 (Rec 1).

### Rec 10: Calibrated LLM-as-judge for dimensions code cannot check (Medium value; resume padding if uncalibrated)

- **What.** Add `evaluation/agent_eval/judge.py`, run offline only. Grade each dimension with a separate judge call returning `{score, pass, "unknown"}`:
  1. answers the question actually asked;
  2. period and metric alignment (for example 同比 vs 环比, report period);
  3. hedging adequacy and no implicit advice;
  4. honest limitations.

  Label 60–100 dev answers by hand and report judge–human agreement. Never use the judge as a runtime gate or in place of the deterministic fact checks. Prefer a judge model from a different family than the answering model (Opinion). Sources: rubric judges [S4]; isolated dimensions, an "Unknown" way out, human calibration [S5]; ADK rubric and hallucination criteria [S26].
- **Files.** New `evaluation/agent_eval/judge.py`, a runner flag, `docs/agent-eval.md`.
- **Why / interview.** "Agent 评估体系包括哪些维度", "LLM-as-judge 如何保证可靠", "如何构建 Agent 的评测体系".
- **Effort.** 2 days including labelling (Opinion).
- **Measure.** Cohen's κ between judge and human per dimension (Opinion: ≥0.6 before trusting it), then per-dimension scores for workflow vs agent.
- **Value note.** Without the calibration numbers this is resume padding. With them, it answers the hardest follow-up ("你的 judge 准不准").

---

## 4. Low fit or resume padding for FinSight (Opinion, with reasons)

| Idea | Why not (now) |
|---|---|
| Multi-agent roles (analyst/trader/critic), orchestrator-worker sub-agents | [S4]: ~15× chat tokens, and a poor fit when agents share context. FinSight tasks are short and share one evidence store. The design notes already argue this; keep the argument, not the build. |
| LLM-based router or planner replacing `router.py` / `planner.py` | Violates `AGENTS.md`. [S1] explicitly endorses classical classifiers for routing. |
| Vector retrieval or vector long-term memory | Violates `AGENTS.md`. |
| A Skills system | Nine tools and one domain give nothing to progressively disclose. If you want an interview talking point, "intent playbooks" chosen by the classical NLU and appended to the user message is the honest analogue, but mark it experimental. |
| LLM conversation compaction / summarisation | Sessions keep 2–3 short turns. Fixed windows (`memory.py`) are enough. Tool-result clearing (Rec 3) is the relevant compaction. |
| "think" tool | Anthropic reports gains in policy-heavy tool chains (τ-bench airline pass^1 0.370 → 0.570 with an optimised prompt) [S8], but DeepSeek thinking mode already reasons between tool calls [S19]. Try it only if Rec 1 transcripts show policy mistakes. |
| Dynamic tool loading / tool RAG | 9 tools. [S10] advises against changing tools mid-loop. |
| DeepSeek `strict` tool mode | Beta. Requires all fields `required` and no `maxLength` [S18], while FinSight uses `max_length` and optional fields. Pydantic validation plus the `invalid_arguments` repair loop already covers this. |
| Long-term user memory (risk preference, holdings) | Possible with a LangGraph store [S24], but remembered risk profiles push toward personalised suitability advice, which the compliance rules forbid. Only user-stated, visible, deletable facts (language, watchlist) are safe. Low priority. |
| Postgres checkpointer / crash-resume | Real for multi-instance deployment (design notes "下一步" #5), but runs are ≤90 s with idempotent read-only tools. Mention it in interviews; build it only if you deploy. |

---

## 5. Interview question → where FinSight answers it

Questions quoted from [S29] unless noted. "After" refers to the recommendation that strengthens the answer.

| Question | Current answer in FinSight | After |
|---|---|---|
| Agent 和 Workflow 有什么关系，分别适合什么场景？判断依据是什么？ | `router.decide_route` with `route_reasons`; `mode=auto/workflow/agent`; offline ablation (design notes §4) | Rec 1 (online evidence) |
| ReAct 和 Plan-Execute-Replan 的区别？ | `agent_llm ⇄ agent_tools` (ReAct) vs `planner.plan_from_nlu` (deterministic plan-execute; also an injection defence) | — |
| 如何设计机制终止无效循环，而不是只依赖最大轮次？ | `AgentConfig` step/tool/token/deadline budgets, forced final, `endless_tools` fault scenario | Rec 5 (budget hint), Rec 3 (context budget) |
| 如果大模型返回的 Function Call 参数格式不对，工程上怎么处理？ | `_coerce_arguments` JSON repair, Pydantic validation, `invalid_arguments` returned to the model | Rec 5 |
| 工具调用结果怎么压缩？大结果卸载怎么实现？ | Evidence store keyed by `evidence_id`; 12k-char truncation | Rec 3 |
| Agent 的短期记忆和长期记忆如何实现？ | Checkpointer thread plus `turns`, per-turn reset reducers, coreference rewrite | §4 (why long-term memory is limited in finance) |
| HITL：触发人工确认时 Agent Loop 处于什么状态？怎么暂停和恢复？ | `interrupt()` in `clarify`, checkpointer, `/agent/resume` | — |
| 幻觉率怎么定义和计算？ | Dealbreaker-gated facts plus numeric faithfulness in `metrics.py`; verifier | Rec 2 (false-accept rate, misattribution) |
| 如何防止外部文档里的恶意指令注入？ | Envelope plus redaction on ingestion, read-only tools, compliance node | Rec 8 |
| MCP 与 Function Calling 的区别？A2A 和 MCP 是什么关系？ | `mcp_server.py` (tools), `a2a_server.py` (whole agent) | — |
| Prompt Engineering、Context Engineering、Harness Engineering 三者有什么区别？ | Prompts in `prompts.py`; context assembly in `graph.agent_llm`; harness = `AgentRuntime` + `ToolRegistry` + budgets + fallback | Recs 3, 4, 6 give concrete numbers |
| 股票分析 Agent 全链路如何设计？ (listed in [S29] as asked by 阿里/快手/京东) | The whole project; `docs/agent.md` graph | All |
| Agent 项目应该统计哪些指标？ | pass^k, tool P/R, citation, latency, cost, degradation rate | Rec 1 (first-pass verification, cache hit), Rec 10 |

---

## 6. Sources

Dates are publication dates where the page showed one. "Accessed" means the page showed no date, or the date was not captured, and it was read on 2026-09-25.

| Tag | Source | Date |
|---|---|---|
| S1 | Anthropic, "Building effective agents". https://www.anthropic.com/engineering/building-effective-agents | 2024-12-19 (page now notes the tooling landscape has changed) |
| S2 | Anthropic, "Writing effective tools for agents — with agents". https://www.anthropic.com/engineering/writing-tools-for-agents | 2025-09-11 |
| S3 | Anthropic, "Effective context engineering for AI agents". https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | 2025-09-29 |
| S4 | Anthropic, "How we built our multi-agent research system". https://www.anthropic.com/engineering/multi-agent-research-system | 2025-06-13 |
| S5 | Anthropic, "Demystifying evals for AI agents". https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents | 2026-01-09 |
| S6 | Anthropic, "Equipping agents for the real world with Agent Skills". https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills | 2025-10-16 (date from the Anthropic engineering index) |
| S7 | Anthropic, "Effective harnesses for long-running agents". https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents | 2025-11-26 (date from the Anthropic engineering index) |
| S8 | Anthropic, "The 'think' tool". https://www.anthropic.com/engineering/claude-think-tool | 2025 (exact date not captured) |
| S9 | Claude Platform Docs, "Prompting best practices". https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices | accessed 2026-09-25 |
| S10 | Yichao "Peak" Ji (Manus), "Context Engineering for AI Agents: Lessons from Building Manus". https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus | 2025-07-18 |
| S11 | HumanLayer, "12-Factor Agents". https://github.com/humanlayer/12-factor-agents | README last commit 2025-09-21 |
| S12 | OpenAI, "A practical guide to building agents" (PDF). https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf | 2025 (date not captured) |
| S13 | OpenAI API docs, "Prompt caching". https://platform.openai.com/docs/guides/prompt-caching | accessed 2026-09-25 |
| S14 | OpenAI API docs, "Reasoning best practices". https://platform.openai.com/docs/guides/reasoning-best-practices | accessed 2026-09-25 |
| S15 | OpenAI Agents SDK docs, "Guardrails". https://openai.github.io/openai-agents-python/guardrails/ | accessed 2026-09-25 |
| S16 | OpenAI, "Harness engineering: leveraging Codex in an agent-first world". https://openai.com/index/harness-engineering | 2026-02-11 (only title, date and premise verified) |
| S17 | DeepSeek API docs, "Context Caching". https://api-docs.deepseek.com/guides/kv_cache | accessed 2026-09-25 |
| S18 | DeepSeek API docs, "Tool Calls" (strict mode, beta). https://api-docs.deepseek.com/guides/tool_calls | accessed 2026-09-25 |
| S19 | DeepSeek API docs, "Thinking Mode". https://api-docs.deepseek.com/guides/thinking_mode | accessed 2026-09-25 |
| S20 | DeepSeek API docs, "JSON Output" (zh). https://api-docs.deepseek.com/zh-cn/guides/json_mode | accessed 2026-09-25 |
| S21 | DeepSeek API reference, "Create Chat Completion" (`tool_choice`). https://api-docs.deepseek.com/api/create-chat-completion | accessed 2026-09-25 |
| S22 | 智谱AI开放文档, "工具调用". https://docs.bigmodel.cn/cn/guide/capabilities/function-calling | accessed 2026-09-25 |
| S23 | LangChain blog, "Context Engineering". https://blog.langchain.com/context-engineering-for-agents/ | 2025-07-02 |
| S24 | LangChain docs, "Persistence" (checkpointer vs store). https://docs.langchain.com/oss/python/langgraph/durable-execution | accessed 2026-09-25 |
| S25 | LangChain docs, "Memory overview". https://docs.langchain.com/oss/python/concepts/memory | accessed 2026-09-25 |
| S26 | Google ADK docs, "Why evaluate agents" (criteria). https://google.github.io/adk-docs/evaluate/ | accessed 2026-09-25 |
| S27 | Beurer-Kellner et al., "Design Patterns for Securing LLM Agents against Prompt Injections", arXiv:2506.08837. https://arxiv.org/abs/2506.08837 (pattern descriptions via https://www.themoonlight.io/en/review/design-patterns-for-securing-llm-agents-against-prompt-injections) | June 2025 |
| S28 | Langfuse docs, "Prompt Management". https://langfuse.com/docs/prompt-management/overview | accessed 2026-09-25 |
| S29 | 面灵AI, "AI Agent 面试题与八股文汇总：155 道大厂真题按主题拆解（2026 面经）". https://www.mianlingai.com/topics/ai-agent-interview-questions-2026 | covers up to 2026-09 |
| S30 | 牛客网, "2026-08-12 小红书 AI Agent 开发一面面经（含完整答案）". https://www.nowcoder.com/discuss/919617105761165312 | 2026-08-12 |
| S31 | 腾讯云, "腾讯云 Agent Memory 节省61% Token 提升52%成功率的诀窍 — Mermaid 无限画布 × 上下文卸载". https://www.tencentcloud.com/techpedia/144098?lang=zh | accessed 2026-09-25 (vendor-reported numbers) |
| S32 | bojieli, *ai-agent-book* ch. 2 "上下文工程". https://github.com/bojieli/ai-agent-book/blob/main/book/chapter2.md | accessed 2026-09-25 (secondary; used only as evidence of what Chinese prep material covers) |
| S33 | Local report, `/Users/9baka/Downloads/deep-research-report.md` ("中国大陆 AI Agent 岗位能力…研究报告") | 2026-09-24 (not re-verified here) |

Other pages consulted but not cited for specific claims: Anthropic engineering index (for publication dates of S6, S7 and "Code execution with MCP", 2025-11-04), kamacoder 大模型面经 index (https://notes.kamacoder.com/interview/llm), AWS China blog "Agentic AI 基础设施实践经验系列（九）：上下文工程" (https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-nine-context-engineering).
