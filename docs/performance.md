# Performance, Load & Scaling

Languages: English | [中文](zh/performance.md)

Committed throughput, latency, cost, and scaling measurements live under [`docs/results/perf/`](results/perf/) and [`evaluation/results/`](../evaluation/results/). Every JSON artifact records the command, commit hash, server switches, and host load average.

## 1. Deterministic Workflow Path & Checkpointing

Switching LangGraph durability from per-step (`async`) to per-run (`QI_AGENT_DURABILITY=exit`, default) eliminates redundant checkpoint writes on `POST /agent/chat` (offline snapshot, SQLite session store, `docker/perf_matrix.sh`):

| Configuration | Users | Throughput (req/s) | P50 (ms) | P95 (ms) | Errors | Session DB (820 req) |
|---|---:|---:|---:|---:|---:|---:|
| Per-step (`QI_AGENT_DURABILITY=async`) | 1 / 8 / 32 | 4.80 / 4.94 / 4.44 | 71 / 1,659 / 7,024 | 933 / 2,376 / 10,816 | 0 | 64 MB |
| **Per-run (`QI_AGENT_DURABILITY=exit`, default)** | 1 / 8 / 32 | **6.47 / 6.22 / 5.16** | **43 / 1,030 / 5,494** | 733 / 2,937 / 12,298 | 0 | **9.4 MB** (7× smaller) |

## 1b. Resilience Under Source Chaos Load (8 Users, No LLM)

`scripts/chaos_drill.py --scenario sources-load` ran 8 concurrent users × 50 requests (`400` total; [`chaos-sources-load-8users.json`](results/perf/sources-load/chaos-sources-load-8users.json)) across 5 phases—live → blocked within TTL → blocked past TTL (LKG) → blocked past stale window (explicit limitation) → unblocked recovery:
- **0 errors across all 400 requests**, every response verified.
- P50 `0.98–1.53 s`; P95 `5.1–12.4 s` (highest during cold live/recovery fetches; fastest `5.1–7.0 s` while open breakers short-circuit blocked hosts).

## 2a. Agent-Path Latency Profile & Optimizations

Profiling (`evaluation/agent_eval/profile.py`) showed **95% of agent turn latency is sequential LLM round trips** (NLU, tools, verification, and compliance together take `< 5%`). Six targeted improvements cut latency and token cost without losing accuracy (DeepSeek v4.1 Flash, 3 repeats, 0 HTTP 429s):

| Optimization | Default Switch | Effect |
|---|---|---|
| Verifier number tokenizer fix | always on | Eliminated false-positive revisions on `20.9x`, `108.5bn`, `04-16`, `10年期` |
| Pooled LLM keep-alive | `QI_LLM_KEEPALIVE=1` | Cuts per-call TLS overhead (`1.69 s → 1.32 s` P50) |
| Citation repair before LLM revise | `QI_AGENT_REVISE_POLICY=cite_repair` | Auto-attaches unambiguous single-evidence citations before calling LLM revise |
| Derived-number verification | `QI_AGENT_VERIFY_DERIVED=1` | Accepts exact sums/differences/ratios within the same cited sentence |
| Stream stall timeout | `QI_AGENT_LLM_STALL_TIMEOUT_S=20` | Retries stalled streams early (eliminates 90 s P99 outliers) |
| Planner prefetch | `QI_AGENT_PREFETCH=1` | Executes deterministic tools before the first LLM call (halves TTFT & cuts LLM calls ~35%) |

| Eval Set (DeepSeek, 3 repeats) | P50 (s) | P95 (s) | P99 (s) | TTFT P50 (s) | LLM Calls/Turn | Cost/Task (USD) | Task Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Held-out: before (`8e81f48`) → **defaults (`6050ffd`)** | 7.6 → **3.7** | 27.1 → **15.4** | 93.4 → **19.6** | 5.3 → **3.0** | 2.30 → **1.39** | 0.00122 → **0.00077** | 0.962 → **0.994** |
| Test v2: before (`8e81f48`) → **defaults (`6050ffd`)** | 8.6 → **4.7** | 24.2 → **17.3** | 37.3 → **27.5** | 5.7 → **2.8** | 2.58 → **1.71** | 0.00176 → **0.00134** | 0.898 → **0.953** |

**Retrieval contention fix (`doc_retriever.py`)**: Caching `doc_matrix.T` once per TF-IDF index eliminated per-query `csr_tocsc` transposes, dropping 4-user concurrent retrieval tool P50 from `3,941 ms` to **`265 ms`** (`P95: 7,899 ms → 1,610 ms`).

**LLM Load & Cost**:
- At 8 concurrent users (`auto`, DeepSeek, 24 multi-tool research requests, [`r13-load-8u-auto.json`](results/perf/agent/r13-load-8u-auto.json)): **0 HTTP errors**, P50 `13.6 s`, P95 `24.3 s`, **¥14.19 / 1,000 questions** (~¥850/month at 2,000 questions/day).
- Under a 30 s HTTP 503 provider outage burst at 4 users (`r12-load-4u-auto-5xx-burst.json`): **0 user-visible failures**—breakers open after 3 consecutive failures, traffic degrades seamlessly to verified template answers, and recovers automatically after the 60 s cooldown.
- Concurrency on the LLM path is bounded primarily by upstream gateway rate limits (429s), not local CPU.

## 3. Multi-Replica Scaling on k3s with Postgres Sessions

Measured on k3s (`deploy/k8s/scale_test.sh`, Postgres checkpointer/task/trace/rate-limit stores, workflow mode, offline snapshot, 20 req/user):

| Replicas | Users | Throughput (req/s) | P50 (ms) | P95 (ms) | Errors | Runs per Pod |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 / 32 | 3.04 / 3.75 | 1,971 / 7,731 | 5,912 / 13,702 | 0 | 141 / 561 |
| 2 | 8 / 32 | 4.32 / 6.07 | 694 / 4,018 | 4,654 / 13,809 | 0 | 72+69 / 275+286 |
| 3 | 8 / 32 | **10.17 / 11.72** | **275 / 2,017** | **3,081 / 6,253** | 0 | 50+46+45 / 199+166+196 |

Throughput scales roughly linearly with CPU cores (each Python worker is single-core GIL-bound). Cross-replica handoff (`results/perf/k3s/cross-replica-session.txt`) verified turn 1 on Pod A (`贵州茅台的市盈率是多少？`) seamlessly resolving turn 2 on Pod B (`它的市净率呢`) via Postgres checkpoints.

## 4. Service Start-Up

TF-IDF character n-gram fitting over the shipped 43 MB document corpus dominates cold start ([`startup.json`](results/perf/startup.json), [`startup-container.json`](results/perf/startup-container.json)):

| Mode | Startup Time |
|---|---:|
| Cold process build (in-memory TF-IDF fit) | `24.1 s` (host) / `45.7 s` median (4-vCPU container) |
| Same-process rebuild (in-memory corpus-hash memo) | `3.65 s` |
| Cold process loading cached index (`QI_TFIDF_CACHE_DIR`, ~365 MB) | **`6.81 s`** |

## Reproducing

```bash
python -m scripts.load_test --base-url http://127.0.0.1:8000 --users 8 --requests 20
python -m scripts.chaos_drill --scenario sources-load
IMAGE=finsight:$(git rev-parse --short=7 HEAD) deploy/k8s/scale_test.sh
```
