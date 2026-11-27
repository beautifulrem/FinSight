# Agent Layer Upgrade Goal (Archived Plan)

> **Status: Completed.** All planned tasks (`T0.1`–`T5.3`) have been delivered and merged into `master`. See [PROGRESS.md](PROGRESS.md) for the milestone ledger and [docs/agent.md](../agent.md) for current architecture documentation.

## Mission

Extend FinSight's published classical **Query Intelligence** pipeline ([DOI: `10.3233/FAIA261107`](https://ebooks.iospress.nl/doi/10.3233/FAIA261107)) into an observable, verifiable, self-hostable financial research agent while keeping classical NLU and hybrid retrieval as the explainable routing and evidence backbone.

## Non-Negotiable Constraints

1. **Classical backbone preserved**: No LLM router or vector database replaces NLU or retrieval.
2. **Financial compliance**: Never emit buy/sell/position-size instructions; cite run `evidence_id`s; surface missing data explicitly; always attach a risk disclaimer.
3. **Offline testability & deterministic fallback**: All core tests run offline via `ScriptedLLM`; missing API keys or LLM failures fall back cleanly to the deterministic planner + template composer.
4. **Backwards compatibility**: Existing `/nlu/analyze`, `/retrieval/search`, `/query/intelligence*`, and classic `/chat` contracts remain stable.
5. **Reproducible metrics**: Every reported metric links to committed evaluation artifacts with commit hash, command, and model recorded.

## Delivered Architecture Decisions (ADRs)

- **ADR-1 Hybrid workflow + agent**: Classical NLU routes simple lookups to a deterministic workflow and complex multi-hop/comparison/causal questions to a plan-seeded LangGraph tool loop.
- **ADR-2 LangGraph runtime**: Provides thread-scoped checkpoints (memory/SQLite/Postgres), native `interrupt()/resume` clarification, and bounded retries.
- **ADR-3 OpenAI-compatible non-strict tool calling**: Supports DeepSeek and compatible gateways with local Pydantic argument validation and JSON repair.
- **ADR-4 Read-only tool boundary + MCP/A2A**: 9 typed read-only tools published via MCP and orchestrated via REST/SSE and A2A 1.0.
- **ADR-5 Code-driven verification over multi-agent role play**: Uses a deterministic sentence-level numeric verifier (`verifier.py`) and compliance guard instead of high-cost multi-agent debate.
