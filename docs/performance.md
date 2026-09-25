# Performance and Load

This page records measured service throughput and latency, the bottleneck found, and what is known about scaling. Every number comes from `scripts/load_test.py` runs listed below; LLM-mode latency and cost are reported separately in [agent-eval.md](agent-eval.md), because they are dominated by the provider.

## Setup

- Image: `docker build -f docker/Dockerfile -t finsight:dev .` (python:3.13-slim runtime, 2 GB).
- Container: `docker run -p 8001:8000 -e QI_USE_LIVE_MARKET=0 -e QI_USE_LIVE_NEWS=0 -e QI_USE_LIVE_ANNOUNCEMENT=0 -e QI_USE_LIVE_MACRO=0 finsight:dev`. One uvicorn process, sessions in SQLite (`QI_AGENT_CHECKPOINT_DB=/app/state/agent.sqlite`, the image default).
- Host: Apple-silicon Mac, Docker via colima (4 vCPU, 8 GB). The host was also running an online evaluation at the time, so absolute numbers are conservative.
- Load: `python -m scripts.load_test --base-url http://127.0.0.1:8001 --users N --requests 20`. Closed loop: N clients each send 20 requests back to back to `POST /agent/chat` in `workflow` mode (classical NLU, deterministic planner, offline tools, template answer, verifier, compliance; no LLM). Questions rotate over 8 kinds: price, valuation, comparison, "why", English, macro, out-of-scope, and a dangling-reference clarification. Every request uses a new session.
- Date: 2026-09-25.

## Results

| Build | Users | Requests | Throughput (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | Error rate |
|---|---|---|---|---|---|---|---|
| before: per-node checkpoints (SQLite) | 1 | 20 | 0.73 | 231 | 4584 | 15630 | 0.0 |
| before: per-node checkpoints (SQLite) | 8 | 160 | 0.65 | 9055 | 38416 | 68702 | 0.0 |
| before: per-node checkpoints (SQLite) | 32 | 640 | 3.02 | 7231 | 29995 | 38936 | 0.0 |
| after: `durability="exit"` + trimmed final state (SQLite) | 1 | 20 | 8.36 | 30 | 663 | 685 | 0.0 |
| after: `durability="exit"` + trimmed final state (SQLite) | 8 | 160 | 6.79 | 877 | 3203 | 5964 | 0.0 |
| after: `durability="exit"` + trimmed final state (SQLite) | 32 | 640 | 6.17 | 4658 | 8393 | 11288 | 0.0 |

A single request in isolation takes 40–120 ms (0.5 s for English questions, whose NLU path is heavier).

## The bottleneck

Single requests were fast but the service collapsed under load, and the SQLite session file had grown to 65 MB after about 800 requests. A control container with the in-memory checkpointer (`QI_AGENT_CHECKPOINT_DB=`) served 3.9 req/s for one user instead of 0.7, which located the problem:

- Without an explicit durability mode LangGraph writes a checkpoint after every step — about 11 per run here — each holding the full state, including LLM messages, tool data and the tool log.
- All writes go through one SQLite connection.

Two changes, both in `query_intelligence/agent/`:

1. `AgentService` runs the graph with `durability="exit"` (`QI_AGENT_DURABILITY`, default `exit`; `async` and `sync` are the other LangGraph modes): one checkpoint per run and one at a clarification interrupt. The trade-off, documented by LangGraph, is that a crash in the middle of a run loses that turn; runs take seconds and the next turn starts from the last completed one.
2. `finalize` clears the bulky turn-scoped working state (messages, tool log, LLM log) once the response has been assembled; the next turn resets it anyway.

Single-user throughput rose 11x (0.73 → 8.36 req/s) and P95 fell from 4.6 s to 0.66 s. The session file for the same request volume shrank from 65 MB to 9 MB.

## Scaling: what is and is not measured

- One process saturates at about 6–7 req/s in this setup. The work is CPU-bound Python (NLU, retrieval ranking, verification) under the GIL; the agent's I/O (tools, LLM) runs on threads.
- A run with `--workers 3` on the same 4-vCPU VM did not improve throughput (4.1 req/s at 8 users, 5.8 at 32) while the host was busy with the evaluation, and the three processes share one SQLite file. That measurement is inconclusive and is not claimed as a result.
- Horizontal scaling needs a checkpointer shared across processes or replicas (Postgres through `langgraph-checkpoint-postgres`, not implemented). The A2A task store and the in-memory trace buffer are also per process.
- LLM modes are bounded by the provider: agent-mode P95 is in the tens of seconds, see [agent-eval.md](agent-eval.md). The API applies `QI_AGENT_REQUEST_TIMEOUT_S` (504) and the graph applies a run deadline.

## Reproducing

```bash
docker build -f docker/Dockerfile -t finsight:dev .
docker run -d --name finsight -p 8001:8000 \
  -e QI_USE_LIVE_MARKET=0 -e QI_USE_LIVE_NEWS=0 -e QI_USE_LIVE_ANNOUNCEMENT=0 -e QI_USE_LIVE_MACRO=0 finsight:dev
for users in 1 8 32; do python -m scripts.load_test --base-url http://127.0.0.1:8001 --users $users --requests 20; done
```

`QI_AGENT_DURABILITY=async` (or `sync`) restores per-step checkpoints; the "before" rows also kept the untrimmed final state (commit `0678585`).
