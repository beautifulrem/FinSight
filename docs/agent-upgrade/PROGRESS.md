# Agent Layer Upgrade Progress (Completed)

All 39 planned tasks in [GOAL.md](GOAL.md) (`T0.1`–`T5.3`) are **completed**, followed by 14 rounds of independent reviews ([docs/reviews/](../reviews/README.md)) and the full-repository P1/P2 security and correctness audit ([PR #13](https://github.com/beautifulrem/FinSight/pull/13)).

## Milestone Summary

| Phase | Tasks | Delivered Artifacts | Status |
|---|---|---|---|
| **Phase 0: Baseline & Tooling** | `T0.1`–`T0.4` | Offline test fixtures, `ruff.toml`, structured logging, alias hygiene, `evaluation/fuzz_cases.jsonl` recovery | Done |
| **Phase 1: Typed Read-Only Tools & MCP** | `T1.1`–`T1.8` | `tools/base.py`, `evidence.py`, 9 typed tools (`resolve_entity`, `get_price_history`, `compute_indicators`, `get_fundamentals`, `get_macro_indicators`, `search_news`, `search_announcements`, `search_knowledge`, `analyze_sentiment`), `mcp_server.py` | Done |
| **Phase 2: LangGraph Agent & Guardrails** | `T2.1`–`T2.10` | `llm.py` (`DeepSeekToolClient`, `FallbackLLM`, `ScriptedLLM`), `planner.py`, `router.py`, `graph.py`, `verifier.py`, `compliance.py`, `output_safety.py`, `injection.py`, `memory.py` (`interrupt/resume`), `service.py`, `/agent/*` SSE & REST APIs | Done |
| **Phase 3: Evaluation & Observability** | `T3.1`–`T3.8` | `tracing.py` (JSON + OTLP/Jaeger), dev/holdout/test/multiturn suites, replay snapshots, `metrics.py`, `ablation.py`, `verifier_stress.py`, `redteam.py`, `fault_injection.py`, `gate.py`, auto-generated `docs/agent-eval.md` | Done |
| **Phase 4: Production Engineering** | `T4.1`–`T4.6` | React 19 + TS workspace (`frontend/`), async endpoints + deadlines, hardened multi-stage Docker & Kubernetes manifests, Postgres shared stores, API-key/cookie/rate-limit/body-limit security, GitHub Actions CI | Done |
| **Phase 5: Documentation & Post-Upgrade Hardening** | `T5.1`–`T5.3` + Audits | Exported `schemas/agent_*.schema.json`, bilingual docs, A2A 1.0 + external MCP client, Claim Check (`POST /agent/claim-check`), primary-source snapshot corrections, and Issue #7 P1/P2 audit fixes | Done |

## Notable Architectural Deviation

- **Tracing backend (`T4.3`)**: Instead of bundling a heavy self-hosted Langfuse stack in Compose, FinSight exports standard OpenTelemetry spans over OTLP/HTTP (`QI_AGENT_OTEL=1`, `OTEL_EXPORTER_OTLP_ENDPOINT`) with a lightweight Jaeger profile—remaining directly compatible with Langfuse's OTLP ingest endpoint.
