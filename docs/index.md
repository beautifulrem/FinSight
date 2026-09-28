# Documentation

Languages: English | [中文](zh/index.md)

These pages document FinSight, an evidence-first A-share research agent. The classical NLU and retrieval backend is Query Intelligence; the agent layer orchestrates it with LangGraph.

The root README is intentionally short; use these pages when you need contracts, configuration details, training steps, or downstream handoff notes.

## Contents

| Document | Purpose |
|---|---|
| [Modules](modules.md) | Five-module map: frontend, NLU/Retrieval, numerical analysis, text analysis, and LLM summary/prediction. |
| [Agent Layer](agent.md) | Agent graph, tools, routing, verification, compliance, memory, `/agent/*` API and schemas, configuration, tracing. |
| [Claim Check](claim-check.md) | Fact-checking a pasted market claim: comparators, negation, growth, units, as-of dates, the chat hint, and the claim benchmark (dev 131, held-out 47) with bootstrap CIs and limits. |
| [Agent Evaluation](agent-eval.md) | Task sets (dev, held-out, test v2), replay snapshots, dealbreaker-gated metrics with bootstrap CIs and paired tests, online ablation with two LLM families (pass^3, latency, cost), prompt A/B, verifier stress test, prompt-injection red team, fault injection, and limitations. Generated from `evaluation/results/`. |
| [Comparison](comparison.md) | FinSight next to 问财, 豆包, Kimi, Wind Alice and 妙想: sourced capability table, where the incumbents win, and a head-to-head test design (not run). [中文](zh/comparison.md) |
| [MCP Server](mcp.md) | Serving the agent tools to MCP clients over stdio or streamable HTTP. |
| [A2A, Failover and Observability](a2a-and-observability.md) | A2A endpoint, LLM gateway handling, model failover, gateway cost, run inspector and feedback, Prometheus metrics, Grafana dashboard and alerts, chaos drill. |
| [Data Sources](data-sources.md) | Live source audit, fallback chains, circuit breakers, bounded call pool, cache, Sina/THS cross-check, provenance, `/sources/health` (with an active probe). |
| [Performance](performance.md) | Load tests of the deterministic and LLM agent paths (cost in CNY), the checkpoint fix reproduced, k3s multi-replica scaling, service start-up. |
| [Deployment](deployment.md) | Docker image, monitoring profile, Kubernetes manifests (replicas sharing sessions through Postgres), read-only root filesystem. |
| [Design notes (Chinese)](presentation/agent-design-notes.md) | Trade-offs, built vs reused, 28 real failure cases with root cause and fix, ablation conclusions with CIs, interview FAQ. |
| [Agent practices research](research/agent-architecture-practices-2026.md) | 2025–2026 agent and prompt-engineering practice from primary sources, and the gap analysis that drove the latest changes. |
| [Query Intelligence](query-intelligence.md) | Scope, architecture, API, NLU and retrieval output contracts, live providers, environment variables, troubleshooting. |
| [Local Frontend Chatbot](frontend-chatbot.md) | Browser UI, `/chat` contract, LLM API settings, real local screenshots, and troubleshooting. |
| [Numerical Analysis](numerical-analysis.md) | `analysis_summary`, technical indicators, fundamentals, macro signals, and data readiness. |
| [Presentation Materials](presentation/README.md) | Clean slide outline and links to current demo screenshots. |
| [Training](training.md) | Public dataset sync, manifest-based training, runtime asset materialization, evaluation, and release checks. |
| [LLM Response](llm-response.md) | Legacy local-transformers answer-generation JSON handoff, output contract, configuration, and safeguards. |
| [Sentiment](sentiment.md) | Downstream document sentiment pipeline, preprocessing, FinBERT routing, output fields, and test commands. |
| [Retrieval Output Spec](retrieval_output_spec.md) | Compatibility entry point for `analysis_summary` and retrieval output references. [中文](zh/retrieval_output_spec.md) |

Chinese documentation is in [docs/zh](zh/index.md).

## Documentation Style

The structure follows the common pattern used by mature open-source projects:

- The root README answers "what is this, how do I run it, where do I go next?"
- `docs/` contains stable reference pages split by topic.
- Commands are copyable from a fresh clone and avoid local-only wrappers.
- Public docs avoid generated output, private tokens, and machine-specific paths.
