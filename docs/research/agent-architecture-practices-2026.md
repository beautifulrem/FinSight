# Agent Architecture & Verification Practices Applied to FinSight

Synthesis of 2025–2026 production AI-agent and prompt-engineering practices (Anthropic *Building Effective Agents*, OpenAI practical agent guides, 12-Factor Agents, LangGraph memory/durability patterns, and DeepSeek KV-cache guidance) and how each principle maps onto FinSight's shipped implementation.

## Core Design Principles Adopted

| Industry Practice | How FinSight Implements It | Code / Evidence |
|---|---|---|
| **Workflows before agents** (use deterministic code wherever rules suffice; reserve LLM loops for genuinely multi-hop tasks) | Classical NLU + explicit router handles guardrails, single-fact lookups, and glossary definitions via deterministic `workflow`; only comparisons, causal `"why"`, and multi-hop queries enter the LangGraph loop. | `agent/router.py`, `agent/graph.py` |
| **Plan-seeded hybrid retrieval** (up-front prefetch + just-in-time tool calls) | `QI_AGENT_PREFETCH=1` executes deterministic planner tools before the first LLM turn, cutting LLM round trips by ~35% and halving TTFT (`5.5 s → 3.0 s`). | `agent/planner.py`, `docs/performance.md` |
| **Code-driven evaluator-optimizer** (deterministic verification beats uncalibrated LLM-as-judge for numeric domains) | Sentence-level numeric and `evidence_id` verifier (`verifier.py`) with `cite_repair` + single LLM revision + whole-sentence repair; `0.92%` strict false-accept across 5,631 corrupted answers. | `agent/verifier.py`, `evaluation/agent_eval/verifier_stress.py` |
| **Prompt registry & regression A/B** (versioned prompts pinned by hash in every trace) | `agent/prompts.py` pins prompt versions (`v1`–`v4`) with SHA hashes recorded in traces and Prometheus labels; `v4` shipped only after Test v3 A/B confirmed no accuracy regression. | `agent/prompts.py`, `docs/agent-eval.md` |
| **Actionable Agent-Computer Interface (ACI)** (few high-impact tools, normalized error hints, bounded outputs) | 9 typed read-only Pydantic tools with timeouts, retries, TTL caches, duplicate-call suppression, and structured error codes (`invalid_arguments`, `timeout`, `not_found`, etc.). | `agent/tools/base.py` |
| **KV-cache-friendly request layout** (stable system prefix & tool schemas, append-only messages) | Static system instructions and stable tool definitions preserve provider prompt-cache hit rates (56–93% measured on DeepSeek) and track cached tokens per call. | `agent/llm.py`, `agent/tracing.py` |
| **Layered untrusted-data security** (structural isolation > prompt-only instructions) | Read-only tools, untrusted-data envelopes, input/document injection sanitization, output safety attribution (`output_safety.py`), and 16-set red-team regression suite. | `agent/injection.py`, `agent/output_safety.py`, `SECURITY.md` |
| **Checkpointed durability & human-in-the-loop** | LangGraph checkpoints (`QI_AGENT_DURABILITY=exit`) in SQLite/Postgres enable `interrupt()/resume` clarification and stateless multi-replica scaling. | `agent/memory.py`, `docs/deployment.md` |
| **Rigorous statistical evaluation** (`pass^k`, bootstrap CIs, untouched first runs, fault injection) | Dealbreaker scoring, 3-repeat online runs with 95% bootstrap CIs and exact McNemar tests, untouched independent sets, and 11/11 fault-injection scenarios. | `evaluation/agent_eval/` |

## Deliberately Omitted Patterns (Low Fit for FinSight)

- **Role-playing multi-agent debates**: High token multiplier (~15×) without improving single-domain factual accuracy; replaced by deterministic verification + single-loop orchestration.
- **LLM-based routing or vector-DB retrieval backbone**: Violates FinSight's explainable classical Query Intelligence architecture and weakens exact financial symbol/period matching.
- **Always-on conversation summarization**: Ablated on `multiturn_v1` with zero accuracy gain on ≤5-turn financial dialogues; kept opt-in (`QI_AGENT_MEMORY_SUMMARY=1`, off by default).
- **DeepSeek strict tool mode**: Avoided in favor of standard schemas validated locally with Pydantic + JSON repair for cross-gateway compatibility.

## Primary Reference Themes

- Anthropic Engineering: *Building Effective Agents* (2024–2025), *Effective Context Engineering for AI Agents* (2025), *Writing Effective Tools for Agents* (2025), *Multi-Agent Research System* (2025).
- OpenAI: *A Practical Guide to Building Agents* (2025), *Prompt Caching Guide*, *Harness Engineering* (2026).
- Dex Horthy / HumanLayer: *12-Factor Agents* (2025); Manus Engineering: *Context Engineering for AI Agents* (2025); LangGraph Documentation (checkpointers, interrupts, durability); DeepSeek API Prompt Caching & Tool-Calling Guides.
