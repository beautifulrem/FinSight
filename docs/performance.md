# Performance and Load

Languages: English | [中文](zh/performance.md)

This page records measured throughput, latency and cost for the two answer paths, the checkpointing
bottleneck and its fix, and multi-replica scaling on k3s. Every number links to a committed JSON file
under [`docs/results/perf/`](results/perf/) that records the command, the run time, the load
generator's commit, the server build and the host load average at the start.

## Test host and builds

- Host: Apple-silicon Mac (10 cores), Docker and k3s v1.35 via colima (VM with 4 vCPU, 8 GB).
- **The host was not idle.** Other work (a blockchain node at 100% of one core, test suites and an
  online evaluation from other worktrees) ran during every measurement; the load averages recorded in
  the JSON files range from about 6 to 38. Absolute numbers are therefore conservative and noisy;
  compare rows measured in the same block rather than across sections.
- Server build for all "current" rows (2026-09-26): the merge of branch `r2-ops@8dc388b` with
  `round2@d04a42d` plus the ops-wiring patch `deploy/patches/app-ops-wiring.patch` (its change has been
  in `api/app.py` since `e9d9a9a`, so the file was deleted; `git show 47dd024:deploy/patches/app-ops-wiring.patch`
  shows it), built as image `finsight:merged`. Results measured since then name a single commit
  (for example [`startup-container.json`](results/perf/startup-container.json)). The "before" row is `git archive 0678585` (the commit before the
  checkpoint fix) built as `finsight:before-0678585`.
- Load generator: [`scripts/load_test.py`](../scripts/load_test.py), a closed loop: N clients each send
  their requests back to back to `POST /agent/chat`, every request in a new session. Percentiles are
  nearest-rank, so with fewer than 100 samples P99 is the maximum.

## 1. Workflow path (no LLM): the checkpointing fix, reproduced

Classical NLU, deterministic planner, offline tools, template answer, verifier and compliance; live
data off. Questions rotate over price, valuation, comparison, "why", English, macro, out-of-scope and
a dangling-reference clarification. One container per configuration with a fresh SQLite session file,
20 requests per user. Driver: [`docker/perf_matrix.sh`](../docker/perf_matrix.sh).

| Configuration | Users | Req | Throughput (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | Errors | Session file after 820 req |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before: commit `0678585` (per-step checkpoints, untrimmed state) | 1 | 20 | 3.93 | 86 | 1,169 | 1,550 | 0 | |
| | 8 | 160 | 5.01 | 1,613 | 2,501 | 2,799 | 0 | |
| | 32 | 640 | 4.51 | 6,121 | 12,816 | 14,740 | 0 | 64 MB |
| current code, `QI_AGENT_DURABILITY=async` | 1 | 20 | 4.80 | 71 | 933 | 1,319 | 0 | |
| | 8 | 160 | 4.94 | 1,659 | 2,376 | 2,574 | 0 | |
| | 32 | 640 | 4.44 | 7,024 | 10,816 | 14,901 | 0 | 64 MB |
| current code, `QI_AGENT_DURABILITY=exit` (default) | 1 | 20 | **6.47** | **43** | 733 | 1,095 | 0 | |
| | 8 | 160 | **6.22** | 1,030 | 2,937 | 4,706 | 0 | |
| | 32 | 640 | **5.16** | 5,494 | 12,298 | 18,843 | 0 | **9.4 MB** |

Files: `results/perf/workflow/load_test-{before-0678585,async,exit}-{1,8,32}.json`; container
settings and `ls -la /app/state` in `container-*.txt`. Run 2026-09-26 09:56–10:06 UTC.

What this shows, and what it does not:

- `durability="exit"` (one checkpoint per run) is the effective part of the fix. At the same commit,
  `async` vs `exit` gives 4.8 → 6.5 req/s at one user, 4.9 → 6.2 at 8 and 4.4 → 5.2 at 32, and the
  session file is 7x smaller (64 MB → 9.4 MB for the same 820 requests). The "before" build
  (per-step checkpoints plus the untrimmed final state) behaves like `async`.
- The P95/P99 tails at 8 and 32 users are **not** better with `exit` in this run. At 32 users every
  configuration is CPU-bound in one process; the tail is dominated by queueing and by the host's other
  load, which varied between runs (load average 6.4–9.3).
- **Correction of the earlier figures.** The previous version of this page (2026-09-25) reported
  0.73 req/s and P95 4.6 s "before" and 8.36 req/s "after" at one user, an 11x gain. That "before" run
  had no committed artifact and was taken while an online evaluation ran on the same VM. The
  reproduction above does not show an 11x gain: the measured effect is about 1.3–1.6x throughput at
  low concurrency and a 7x smaller session store. The earlier "after" JSONs are kept for traceability
  in `results/perf/workflow/history-2026-09-25/`.

## 2. LLM agent path: throughput, latency and cost

Server: the merged build run locally on port 8801 with `DEEPSEEK_MODEL=cline-pass/deepseek-v4.1-flash`,
`QI_LLM_FALLBACK_MODELS=cline-pass/glm-5.3-flash`, the Cline gateway, live data off (offline snapshot,
so the numbers measure the agent and the LLM rather than upstream sites), `mode=agent`, and the
`research` question set: 8 answerable multi-tool questions (comparison, why, profitability, macro
linkage, ETF, trend, English comparison, growth). Cost is the gateway-reported `usage.cost` summed per
run and converted with `QI_LLM_USD_CNY=6.7489`, the CFETS USD/CNY central parity of 2026-09-24
(中国外汇交易中心受权公布人民币汇率中间价: 1 USD = 6.7489 CNY; published on
[finance.sina.com.cn](https://finance.sina.com.cn/jjxw/2026-09-24/doc-iniswvxc5391561.shtml) and in
SAFE's central-parity table).

```bash
source /tmp/llmenv.sh   # DEEPSEEK_* from .env; the key is never printed or committed
QI_LLM_FALLBACK_MODELS=cline-pass/glm-5.3-flash QI_LLM_USD_CNY=6.7489 QI_RATE_LIMIT_PER_MINUTE=0 \
  uvicorn query_intelligence.api.app:create_app --factory --port 8801
python -m scripts.load_test --base-url http://127.0.0.1:8801 --mode agent --questions research \
  --users 4 --requests 6 --usd-cny 6.7489 --questions-per-day 2000 --timeout 240
```

| Users | Req | Throughput (req/s) | P50 (s) | P95 (s) | P99 (s) | HTTP errors | Answered by the agent LLM | Fell back to planner + template | Verified |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 24 | 0.12 | 22.0 | 41.9 | 78.8 | 0 | 24 (100%) | 0 | 79% |
| 8 | 40 | 0.42 | 18.8 | 38.1 | 43.3 | 0 | 26 (65%) | 14 | 80% |
| 16 | 64 | 0.63 | 16.8 | 66.0 | 75.4 | 0 | 23 (36%) | 41 | 88% |

Files: `results/perf/agent/load_test-agent-{4,8,16}.json` (per-request route, model, tokens, cost) and
`gateway-log-16-run2.json` (status and latency of every gateway call during the 16-user run, recorded
by the pass-through proxy from `scripts/chaos_drill.py`; no headers or prompts). Run 2026-09-26
13:25–13:43 UTC, host load average 17–38.

**The ceiling is the gateway's rate limit, not the service.** From 8 concurrent users on, the Cline
gateway answered some calls with HTTP 429 (a rate-limit page). In the 16-user run 81 of 190 gateway
calls were 429, for both the primary and the fallback model, which share the account limit (an
evaluation in another worktree used the same key at the same time). A first 16-user run got 429 on
every call (`load_test-agent-16-run1-all-llm-errors.json`: 64 requests, 0 LLM answers). There were
**no failed requests**: when both models were rate-limited the graph fell back to the deterministic
planner and template answer (`degraded: llm_error`). That is why throughput rises and P50 falls with
concurrency: a growing share of answers never waited for an LLM. The requests that did get LLM
answers had a P50 of 22.7 s at 4 users, 23.8 s at 8 and 39.3 s at 16 users (retries with backoff on
429). Scaling this path needs a higher gateway quota or several keys/providers, not more replicas.

**Cost.** The 4-user run is the clean cost measurement because every answer came from the agent loop:

| | Value |
|---|---|
| LLM calls per question | 4.1 |
| Tokens per question | 19.8k prompt (56% served from the provider's prompt cache) + 2.6k completion |
| Cost per question | $0.00291 = ¥0.0197 |
| **Cost per 1,000 agent questions** | **¥19.7** ($2.91) |
| Monthly at 2,000 questions/day, all on the agent path | **¥1,180** |
| Monthly at 10,000 questions/day | ¥5,900 |

With the router in `auto` mode, lookups, refusals and clarifications never reach the agent loop, so
these are upper bounds for the same traffic. Per LLM-answered question, the 8- and 16-user runs cost
¥15.3 and ¥13.0 per 1,000 (fewer revision loops among the answers that got through).

**Latency tail.** Traces show where the tail comes from. In the slowest agent trace of the monitoring
run (98 s), the first draft failed verification and the `llm.revise` call alone took 74 s
([Jaeger screenshot](assets/ops/jaeger-trace.png)). During the LLM chaos drill, answers on the fallback
model took 48–81 s and one hit the API's 120 s request timeout (504), because GLM-5.3-flash was 2–7x
slower per call than DeepSeek (see [a2a-and-observability.md](a2a-and-observability.md#chaos-drill)).
The fix is in the agent layer since `d1c007c`: every LLM request, including retries, revise and failover calls, gets `min(client timeout, time left before the run deadline)` (90 s for the tool loop, 20 s more for the answer), so the fallback path ends in the deterministic answer instead of a 504. It has not been re-measured under load yet.

## 3. Multi-replica scaling on k3s with Postgres sessions

Setup: `deploy/k8s/finsight.yaml` in k3s (colima), image `finsight:merged`, sessions in the Postgres
StatefulSet (`QI_AGENT_CHECKPOINT_DB=postgresql://...`), API pods limited to 2 CPU each on a 4-vCPU
node. For each replica count the HPA is pinned, every pod is warmed with the question rotation (NLU
models load lazily), and a load-generator **pod inside the cluster** (0.5 CPU) runs
`scripts/load_test.py` against `Service/finsight-api` with fresh connections, so kube-proxy spreads
requests over pods. Workflow mode, live data off, 20 requests per user. Driver:
[`deploy/k8s/scale_test.sh`](../deploy/k8s/scale_test.sh).

| Replicas | Users | Req | Throughput (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | Errors | Runs per pod |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 160 | 3.04 | 1,971 | 5,912 | 13,213 | 0 | 141 |
| 1 | 32 | 640 | 3.75 | 7,731 | 13,702 | 18,420 | 0 | 561 |
| 2 | 8 | 160 | 4.32 | 694 | 4,654 | 11,107 | 0 | 72 / 69 |
| 2 | 32 | 640 | 6.07 | 4,018 | 13,809 | 21,209 | 0 | 275 / 286 |
| 3 | 8 | 160 | 10.17 | 275 | 3,081 | 3,812 | 0 | 50 / 46 / 45 |
| 3 | 32 | 640 | 11.72 | 2,017 | 6,253 | 8,119 | 0 | 199 / 166 / 196 |

Files: `results/perf/k3s/load_test-k3s-r{1,2,3}-u{8,32}.json`; `pods-r*.txt` (`kubectl get pods -o
wide`); `top-r*-u*.txt` (`kubectl top pods` during each run: every API pod at 0.8–1.1 CPU under 32
users, one core per process); `runs-per-pod-*.txt` (completed runs per pod from each pod's `/metrics`
before and after each run; clarifications are not counted as runs). Run 2026-09-26 10:34–10:45 UTC.

Reading: one process is CPU-bound at about one core (GIL). The Service spreads load evenly (per-pod
run counts above), and at 32 users throughput grows 3.75 → 6.07 → 11.72 req/s from 1 to 3 replicas
with zero errors, while P95 falls from 13.7 s to 6.3 s. The 3-replica gain looks super-linear against
one replica; the 1-replica rows were measured while the host was busier (the same image served
5.2 req/s at 32 users in Docker earlier the same day). Read the result as "roughly linear up to the
node's cores", not as a 3.1x speed-up. An earlier attempt without per-pod warm-up is not reported:
the first English questions on a new pod take seconds and dominated its tail.

### Cross-replica session evidence

`results/perf/k3s/cross-replica-session.txt` (2026-09-26): turn 1 "贵州茅台的市盈率是多少？" was sent to
pod `…-24ktk` through a port-forward to that pod only, and turn 2 "它的市净率呢" to pod `…-6mc86`.

- Turn 2 resolved 它 → 贵州茅台 (600519.SH).
- `GET /agent/sessions/<id>` on the first pod returns both turns.
- Each pod's own trace directory holds exactly one of the turns (`cross-replica-pod-logs.txt`:
  turn_index 0 on `…-24ktk`, turn_index 1 on `…-6mc86`, plus each pod's access-log line).
- `cross-replica-postgres.txt` shows 2 checkpoints for the thread in Postgres.

Only the shared checkpointer can carry the entity from one pod to the other.

**The Postgres checkpointer is implemented.** The previous version of this page said it was "not
implemented". That was wrong: `QI_AGENT_CHECKPOINT_DB=postgresql://...` selects
`langgraph-checkpoint-postgres` with a psycopg pool (`agent/memory.py`),
`tests/test_agent_checkpoint_postgres.py` covers it, and the k3s runs above use it.

The A2A task store and the trace store behind `/agent/traces` follow the same Postgres DSN since
round 3 (see [a2a-and-observability.md](a2a-and-observability.md#shared-stores-for-several-replicas)).
Still per replica: the rate limiter and the TTL caches of tools and live sources.

## Reproducing

```bash
docker build -f docker/Dockerfile -t finsight:merged .
docker/perf_matrix.sh                                   # section 1 (also needs finsight:before-0678585)
IMAGE=finsight:merged OUT=docs/results/perf/k3s deploy/k8s/scale_test.sh   # section 3
SKIP_LOAD=1 deploy/k8s/scale_test.sh                    # only the cross-replica session check
```

`QI_AGENT_DURABILITY=async` (or `sync`) restores per-step checkpoints. The trade-off of `exit`,
documented by LangGraph, is that a crash in the middle of a run loses that turn; runs take seconds
and the next turn starts from the last completed one.

## Service start-up

Two committed measurements; every number below is copied from them.

### In-process service build: [`results/perf/startup.json`](results/perf/startup.json)

`python scripts/measure_startup.py --out docs/results/perf/startup.json` at commit `a027c7c`
(2026-09-28, Apple-silicon Mac, 10 cores, load average 4.4 / 5.6 / 10.5 at the start, live data off).
Fitting the char n-gram TF-IDF index over the 43 MB document corpus dominates the build.

| Case | Seconds |
|---|---:|
| Cold build (fresh process, index fitted) | 24.06 |
| Rebuild in the same process (index memoised by corpus hash) | 3.65 |
| Cold build that also writes the index to `QI_TFIDF_CACHE_DIR` | 29.08 |
| New process loading the index from `QI_TFIDF_CACHE_DIR` | 6.81 |

- **Per-process memo.** Since `da3ec8b` the fitted index is memoised by a hash of the corpus, and
  `clear_service_caches()` does not drop it, so the full test suite no longer refits it for every test.
- **Optional disk cache.** With `QI_TFIDF_CACHE_DIR` set, the index is written there (365 MB), and a
  restart loads it instead of refitting: 6.81 s against 24.06 s. It is not baked into the image because
  of its size; it suits a volume shared by the replicas.

(Earlier versions of this section quoted 39 s / 6 s / 4.5 s / 350 MB from the `da3ec8b` commit message
and said there was no result file; those figures did not match `startup.json` and were replaced.)

### Container cold start: [`results/perf/startup-container.json`](results/perf/startup-container.json)

`python scripts/measure_container_startup.py --image finsight:edcb442 --runs 3` with the image built
from a clean tree at commit `edcb442` (`git archive edcb442 | docker build -f docker/Dockerfile -`), the
same image the Kubernetes kustomization pins. Each run is a fresh `docker run` with the Kubernetes
hardening (read-only root filesystem, uid 10001, tmpfs for `/tmp`, `/app/state`, `/app/outputs`), live
data off, SQLite checkpointer, no LLM key; the clock runs from `docker run` until `/ready` answers 200.
colima VM with 4 vCPU / 7.7 GB on the same Mac, other work running (host load average 7–12).

| Run | `/health` 200 (s) | `/ready` 200 (s) | Service build in the log (s) |
|---|---:|---:|---:|
| 1 | 57.13 | 57.50 | 52.4 |
| 2 | 45.05 | 45.53 | 42.5 |
| 3 | 45.68 | 45.99 | 42.9 |
| **Median** | **45.68** | **45.99** | |

- `/health` only answers once the service is built (the build runs before uvicorn listens), so the
  container start is almost entirely the TF-IDF fit, slower inside the 4-vCPU VM than natively.
- The first `/ready` call builds the agent and opens the checkpointer; it adds 0.3–0.5 s.
- The Kubernetes `startupProbe` allows 5 min (60 × 5 s), comfortably above these times.
- The reviewer's round-2 run of an earlier image measured about 105–165 s to healthy on a busier host;
  that run was not committed.
