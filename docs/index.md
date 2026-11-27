# Documentation

Languages: English | [中文](zh/index.md)

Reference documentation for **FinSight**, an evidence-first financial analysis chatbot and verified research agent for Chinese financial markets. The classical Query Intelligence core was published in *Frontiers in Artificial Intelligence and Applications*, Vol. 433 (*Digitalization and Management Innovation V*, DMI 2026, IOS Press; DOI: [`10.3233/FAIA261107`](https://ebooks.iospress.nl/doi/10.3233/FAIA261107), metadata in [`CITATION.cff`](../CITATION.cff)).

Start with the [README](../README.md) for the quick start and headline benchmarks, then use the guides below by topic.

## Core Architecture

| Document | Contents |
|---|---|
| [Modules](modules.md) | Six-layer system map (five paper modules + Agent layer), data flow, and extension points |
| [Query Intelligence](query-intelligence.md) | Classical NLU, hybrid retrieval, typed `NLUResult` / `RetrievalResult` contracts, and paper benchmarks |
| [Agent Layer](agent.md) | LangGraph workflow, deterministic fallback, 9 typed tools, claim verifier, safety guardrails, memory, and configuration |
| [Features & API Reference](features.md) | Complete capability list, HTTP endpoint matrix, schemas, and quick-start switches |
| [Retrieval Output Spec](retrieval_output_spec.md) | Schema reference for `RetrievalResult`, `analysis_summary`, and provenance fields |

## Analysis, Claim Verification & Data

| Document | Contents |
|---|---|
| [Claim Check](claim-check.md) | LLM-free numeric fact-checking of pasted market claims, comparator grammar, and benchmark results |
| [Data Sources](data-sources.md) | Live providers (Tushare, AKShare, Cninfo, efinance), fallback chains, circuit breakers, offline snapshot + corrections layer |
| [Numerical Analysis](numerical-analysis.md) | `MarketAnalyzer` technical indicators, fundamental/macro signals, and `data_readiness` |
| [Document Sentiment](sentiment.md) | Downstream per-sentence FinBERT and offline classical sentiment pipeline (`sentiment/`) |
| [LLM Response Handoff](llm-response.md) | Constrained local HuggingFace/transformers JSON answer and follow-up generation (`scripts/llm_response.py`) |

## Interfaces, Protocols & Operations

| Document | Contents |
|---|---|
| [Web UI & Chatbot](frontend-chatbot.md) | React 19 + TypeScript workspace, SSE streaming, claim-check tab, classic `/chat` mode, and Playwright E2E |
| [MCP Integration](mcp.md) | Serving FinSight tools over stdio / streamable HTTP MCP and attaching external MCP servers (`QI_MCP_SERVERS`) |
| [A2A & Observability](a2a-and-observability.md) | A2A 1.0 JSON-RPC/streaming server, model failover, Postgres shared stores, Prometheus, Grafana, and Jaeger |
| [Deployment](deployment.md) | Multi-stage Docker image, Compose profiles, Kubernetes kustomize manifests, image tag policy, and kind smoke test |
| [Performance & Scaling](performance.md) | Latency profiling, load tests, source chaos drills, and multi-replica k3s scaling |
| [Development](development.md) | Local pre-push mirror (`scripts/prepush.sh`), CI time budgets, and shipped-default test policy |
| [Training & Runtime Assets](training.md) | Dataset sync, manifest-driven classical model training, SHA-256 verification, and asset materialization |

## Evaluation, Governance & Presentations

| Document | Contents |
|---|---|
| [Agent Evaluation (Generated)](agent-eval.md) | Auto-generated paired tables, bootstrap CIs, ablations, verifier stress, and red-team results from `evaluation/results/` |
| [Evaluation History](evaluation-history.md) | Independent sets (first-run vs after-exposure), per-round benchmark ledger, CI workflow, and full limits |
| [Known Issues & Audit Log](known-issues.md) | Open limitations, trade-offs, and resolved review/audit findings |
| [Product Comparison](comparison.md) | Capability comparison against 问财, 豆包, Kimi, Wind Alice, and 妙想, plus head-to-head protocol |
| [Independent Reviews](reviews/README.md) | Summary index of 14 external evaluation rounds and rubric |
| [Agent Upgrade Plan](agent-upgrade/GOAL.md) · [Progress](agent-upgrade/PROGRESS.md) | Original agent-layer architecture goals and milestone log |
| [Architecture Practices Research](research/agent-architecture-practices-2026.md) | Literature synthesis of production agent and verification patterns |
| [Presentation Materials](presentation/README.md) | Slide outline, engineering design notes, and walkthrough script |
| [Security Policy](../SECURITY.md) · [Contributing](../CONTRIBUTING.md) | Vulnerability reporting, secrets/runtime security boundaries, and contributor workflow |
