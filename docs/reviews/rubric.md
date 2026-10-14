# FinSight portfolio review rubric (China AI-Agent internship / 2027 campus recruiting)

Sources the rubric is derived from:
- /Users/9baka/Downloads/deep-research-report.md (P0/P1/P2 skill tiers; recommended portfolio deliverables:
  task-success baseline → tool schema → state machine/graph → memory/context → retry/recovery → eval set →
  trace → P95 latency → token/CNY cost → permissions/sandbox → Docker deployment; interview emphasis on
  Harness, Context Engineering, Evaluation/Tracing, MCP/A2A, security).
- 2026 China interview reports (面经): interviewers probe five "accounts": failure handling, cost/latency with
  real numbers, memory/RAG details, every resume number's methodology, backend fundamentals; they prefer one
  deep, deployed, demo-able project with front end, evaluation, observability, and honest limits.

Score each dimension 0–100 against what a demanding interviewer at ByteDance/Tencent/Alibaba/Baidu would
expect from a strong intern or new-grad Agent engineer. 100 means: nothing material missing for this level,
every claim verified, no bugs found in your own testing. Deduct concretely and say exactly what would
raise the score to 100.

| # | Dimension | What 100 requires |
|---|---|---|
| 1 | Agent core architecture | Explicit loop/graph with routing, planning, tool calling, parallelism, budgets, memory, human-in-the-loop, recovery; design choices justified (workflow vs agent). |
| 2 | Context & prompt engineering | Structured, versioned prompts tied to evaluations; context budgeting/compaction; cache-aware layout; structured outputs; evidence of iterating prompts with data. |
| 3 | Evaluation & quantified results | Task sets with dev/holdout discipline, real-LLM online numbers (pass^k, success, cost, latency), ablation vs baselines, A/B, failure analysis, reproducible commands, honest caveats. |
| 4 | Reliability & fault tolerance | Timeouts, retries, circuit breakers, fallbacks at model/tool/data layers, graceful degradation measured by fault injection. |
| 5 | Cost & performance | Real per-task cost and token numbers, P50/P95, load test, identified bottleneck with measured before/after, cost optimisations with evidence. |
| 6 | Observability | Per-run traces (nodes/tools/LLM/tokens/cost/prompt versions), metrics endpoint, OTel export, a way to inspect a run. |
| 7 | Security & compliance | Prompt-injection defense measured by a red team incl. unseen attacks, auth/rate limit, secrets handling, container hardening, domain compliance guard; residual risks stated. |
| 8 | Engineering & deployment | Clean API, tests (unit/integration/browser), CI gates that actually run on the default branch, Docker, Kubernetes, reproducibility, code quality. |
| 9 | Front end & product experience | Modern polished UI that exposes the agent's internals (trace, citations, evidence, cost), zh/en, responsive, verified working in a real browser, no console errors. |
| 10 | Data acquisition | Real live sources verified by audit, fallback chains, caching, freshness/provenance on every record, health endpoint. |
| 11 | Protocols & ecosystem | MCP server, A2A endpoint, OpenAI-compatible LLM gateway support, standards-based tracing. |
| 12 | Documentation & interview readiness | README that sells the project in 30 seconds with verified numbers, design notes with trade-offs and real failure cases, bilingual, every number traceable to a command/commit. |
| 13 | Real-world value & differentiation | Clear answer to "why not just use 豆包/问财", real problem, distinctive design (e.g. classical-ML + LLM hybrid, verification), runs on live data. |
