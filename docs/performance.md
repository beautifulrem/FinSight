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
The fix is in the agent layer since `d1c007c`: every LLM request, including retries, revise and failover calls, gets `min(client timeout, time left before the run deadline)` (90 s for the tool loop, 20 s more for the answer), so the fallback path ends in the deterministic answer instead of a 504. Since then a per-chunk stall timeout (20 s) also retries a stalled stream long before the deadline, and the agent path was re-measured, at 4 users as well, in [section 2a](#2a-agent-path-latency-profile-changes-and-beforeafter).

## 2a. Agent-path latency: profile, changes and before/after

The round-2 review measured an agent-path P95 of about 20–23 s on held-out/test_v2 with DeepSeek and a
first streamed token after about 14 s. This section profiles where that time goes, lists the changes made,
and gives the before/after numbers with paired confidence intervals. Every number comes from a committed
file in [`evaluation/results/`](../evaluation/results/). Each file keeps the command, the commit, the prompt
hashes, the per-task outcomes and the latency profile.

**Method.** `python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3 --workers 3 --sets holdout,test_v2 --modes agent --stream`
(DeepSeek v4.1 flash through the Cline gateway, replayed tool snapshots, the fixed evaluation date).
`--stream` sends every turn through `AgentService.stream`, the SSE path the UI uses, and records the
time to the first `answer_delta` event (TTFT). Each turn record keeps a profile: every LLM call (node, tool-loop step,
latency, prompt, cached, completion and reasoning tokens, and the context composition in characters),
every tool call and every graph node's duration. `llm_http` counts the HTTP attempts, including 429s that
a retry recovered. There were **no 429s in any run on this page**. Latency percentiles are nearest-rank
over all turns of a set (56 held-out and 174 test_v2 turns, × 3 repeats), including refusals and
clarifications; "LLM-turn" columns only count turns that called the LLM. Success differences are paired
over tasks (`metrics.paired_comparison`: bootstrap 95% CI, 2,000 resamples, and exact McNemar on pass^3).
The runs ran one after another on a shared Mac (load average 3–6) and never overlapped.

```bash
python -m evaluation.agent_eval.profile outputs/agent_eval/<run>.json        # breakdown tables below
python -m evaluation.agent_eval.profile --table evaluation/results/perf-merged-defaults-deepseek.json \
    evaluation/results/perf-merged-citerepair-stall-deepseek.json evaluation/results/perf-merged-prefetch-deepseek.json
```

### Where the time went (baseline, [`perf-baseline-deepseek.json`](../evaluation/results/perf-baseline-deepseek.json), code of `2dbb026`)

| | held-out | test_v2 |
|---|---:|---:|
| LLM calls per LLM turn | 2.7 | 3.0 |
| share of turn time in the LLM nodes (agent_llm + revise) | 95% | 95% |
| tool-loop step 0 (choose tools): P50 latency, prompt tokens | 2.3 s, 2.6k | 2.3 s, 2.7k |
| tool-loop step 1 (usually the answer): P50, prompt tokens | 3.0 s, 3.6k | 3.0 s, 3.8k |
| turns with a 2nd, 3rd … tool round | 21% | 31% |
| **revision rate** (draft failed verification → one more LLM call) | **49%** | **54%** |
| revision: P50 / P95 latency, share of all LLM time | 4.3 / 15.7 s, 32% | 4.6 / 10.2 s, 26% |
| tools: P50 / P95 latency | 0.6 / 601 ms | 0.6 / 744 ms |
| NLU and routing (`guard_in`) | 3% of turn time | 2% |
| P99 turn | 93 s | 37 s |

Reading:

- **The LLM round trips are the whole cost.** Tools, NLU, verification and compliance together take under
  5% of a turn. Prompts are small (2.6–5k tokens). Tool-loop prompts are 70–93% served from the provider's prompt
  cache, revision prompts only about 25%. Shrinking tool observations or history would save little. The number of sequential LLM calls is the lever.
- **Half of all answers made an extra revision call.** A sample of the drafts that triggered it (21 drafts
  captured with `revise` hooked) showed that most "unsupported numbers" were false positives of the
  verifier's number tokenizer, not model errors. `20.9x` and `108.5bn` were read as 20 and 108,
  month-day dates in daily series (`4.746（04-16）`) as 4 and 16, bond tenors (`10年期`) as 10 and list
  markers (`3)`) as 3. Others were derived numbers the model computed from cited values (a 17.8-point ROE
  gap, 茅台/平安 PE ratio 2.8). The rest were citation-only problems: a number without an id, or with the
  wrong or a non-existent id.
- **The tail is stalls.** The slowest turns are calls that normally take 2–5 s but hung until the 90 s run
  deadline or the client timeout (3 + 3 read timeouts in the baseline). The slowest decile also has more
  tool rounds (1.6–2.2 against 1.3–1.5) and was revised 80–91% of the time.

### Changes (each one has a switch)

| Change | Switch (default) | What it does |
|---|---|---|
| Number tokenizer fix | always on (a bug fix) | A number token ends before `x`/`bn`/`mn`/`m`/`k`/`pp`/`pct` instead of backtracking into a shorter number. Month-day dates, bond tenors and list markers are not claims. Verifier stress test on the current dev set ([`verifier_stress-perf-8a85ae5.json`](../evaluation/results/verifier_stress-perf-8a85ae5.json), 202 gold answers, 3,399 variants): true accept 1.0, false accept 1.94%. The committed `verifier_stress.json` was re-run at `9f0e46b` with the same result (202 gold answers, 1.94%); the earlier 2.1% (159 gold answers at `2494656`) is superseded. |
| Pooled LLM connections | `QI_LLM_KEEPALIVE=1` | One pooled `httpx.Client` per model instead of a new TLS connection per request. Probe ([`keepalive-probe.json`](results/perf/agent/keepalive-probe.json), 25 interleaved tiny requests): P50 1.69 s → 1.32 s per call. |
| Citation repair before revising | `QI_AGENT_REVISE_POLICY=cite_repair` | If a draft fails only on citations (uncited or misattributed numbers, invalid or missing ids), attach the id of the **single** evidence item that holds each number, re-verify, and skip the LLM revision if the result passes. Values found in several items are not guessed. |
| Derived numbers | `QI_AGENT_VERIFY_DERIVED=1` | Accept a number equal to the difference, sum, ratio or percent change of two supported numbers stated in the same cited sentence. False accept in the same stress test 1.94% → 2.03% (swapped numbers 0.53% → 0.93%). |
| Stall timeout | `QI_AGENT_LLM_STALL_TIMEOUT_S=20` | Every LLM call is streamed and httpx's read timeout (the wait for the next chunk) is capped at 20 s, so a stalled stream is retried or fails over instead of waiting for the run deadline. |
| Planner prefetch | `QI_AGENT_PREFETCH=1` | The deterministic planner's tool calls run before the first LLM call and their results go to the model with the question. A question the planner covers is answered in **one** LLM call. The model can still call other tools; repeats of a prefetched call hit the duplicate-call guard. |

Setting `QI_AGENT_PREFETCH=0 QI_AGENT_REVISE_POLICY=llm QI_AGENT_LLM_STALL_TIMEOUT_S=0 QI_AGENT_VERIFY_DERIVED=0`
restores the previous agent path. The same fields exist on `AgentConfig`, and `--agent-config FIELD=VALUE` sets them in the ablation.

### Before/after (DeepSeek v4.1 flash, 3 repeats, no 429s)

Round 2 (`df19daa`: multi-turn session rules) was merged into this branch between run 1 and run A, so
there are two paired comparisons, each made on one code base.

| Set | Run | Commit | P50 s | P95 s | P99 s | LLM-turn P95 s | TTFT P50 s | TTFT P95 s | LLM calls/turn | tokens/turn | cost/task (USD) | task success | Δ success vs reference [95% CI] | pass^3 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| held-out | baseline | `8e81f48` | 7.6 | 27.1 | 93.4 | 27.1 | 5.3 | 13.0 | 2.30 | 8,711 | 0.00122 | 0.962 | reference | 0.943 |
| held-out | run 1: tokenizer fix + keep-alive | `52e80dc` | 6.2 | 19.3 | 22.5 | 19.3 | 5.3 | 12.4 | 2.15 | 7,990 | 0.00096 | 0.994 | +0.031 [−0.006, +0.088] | 0.981 |
| test_v2 | baseline | `8e81f48` | 8.6 | 24.2 | 37.3 | 24.6 | 5.7 | 14.3 | 2.58 | 10,274 | 0.00176 | 0.898 | reference | 0.876 |
| test_v2 | run 1: tokenizer fix + keep-alive | `52e80dc` | 7.6 | 20.3 | 31.2 | 20.9 | 5.4 | 13.0 | 2.49 | 9,859 | 0.00160 | 0.923 | **+0.025 [+0.005, +0.050]** | 0.917 |
| held-out | A: merged round 2, switches off | `aae29fd` | 6.6 | 22.9 | 90.0 | 22.9 | 5.5 | 13.0 | 2.23 | 8,747 | 0.00106 | 0.987 | reference | 0.962 |
| held-out | B: A + citation repair + stall timeout + derived | `d84f4e4` | 6.5 | 17.4 | 22.4 | 18.3 | 5.3 | 11.5 | 2.09 | 7,882 | 0.00092 | 0.987 | +0.000 [−0.019, +0.019] | 0.981 |
| held-out | **C: B + planner prefetch (new default)** | `6050ffd` | **3.7** | **15.4** | **19.6** | **15.5** | **3.0** | **9.4** | **1.39** | 6,018 | 0.00077 | 0.994 | +0.006 [+0.000, +0.019] | 0.981 |
| test_v2 | A: merged round 2, switches off | `aae29fd` | 8.0 | 21.8 | 40.6 | 22.4 | 5.8 | 13.3 | 2.58 | 10,392 | 0.00166 | 0.950 | reference | 0.934 |
| test_v2 | B: A + citation repair + stall timeout + derived | `d84f4e4` | 7.7 | 21.7 | 28.8 | 23.0 | 5.8 | 14.1 | 2.50 | 10,099 | 0.00152 | 0.956 | +0.005 [−0.005, +0.017] | 0.950 |
| test_v2 | **C: B + planner prefetch (new default)** | `6050ffd` | **4.7** | **17.3** | **27.5** | **18.9** | **2.8** | **10.7** | **1.71** | 7,996 | 0.00134 | 0.953 | +0.003 [−0.005, +0.014] | 0.942 |

Files: `perf-baseline-deepseek.json`, `perf-verifierfix-deepseek.json`, `perf-merged-defaults-deepseek.json`,
`perf-merged-citerepair-stall-deepseek.json`, `perf-merged-prefetch-deepseek.json` (all in
`evaluation/results/`). The B and C switches were set with `--agent-config` at the commits shown; those
commits differ from `aae29fd` only in result files and evaluation reporting. "LLM calls/turn" averages
over all turns, refusals and clarifications included.

What each step bought:

- **Tokenizer fix + keep-alive (run 1).** Revision rate 49% → 29% (held-out) and 54% → 42% (test_v2), P95
  27.1 → 19.3 s and 24.2 → 20.3 s. Task success rose, significantly on test_v2 (+0.025
  [+0.005, +0.050]): correct drafts are no longer revised or cut by the repair step.
- **Citation repair + stall timeout + derived numbers (B vs A).** Revision rate 30% → 22% and 41% → 28%;
  citation repair replaced 10 + 52 LLM revisions. The stall timeout removed the 90 s outliers (held-out P99 90 → 22 s,
  test_v2 41 → 29 s); 4 stalled streams were retried. On its own, B does not bring the test_v2 P95 under 20 s:
  the slow turns there make 2.7 tool rounds.
- **Planner prefetch (C vs B).** 59% of held-out and 47% of test_v2 LLM turns are now answered in one LLM call,
  and the multi-round share fell. P50 drops by 40%, and TTFT P50 roughly halves (5.3–5.8 s → 2.8–3.0 s)
  because the first LLM call writes the answer. The first call's prompt grows by 600–1,000 tokens, so it takes
  2.9 s instead of 2.3 s, but a whole round trip goes. Cost per task falls with the number of calls
  (−28% held-out, −19% test_v2 vs A).
- **Against the round-2 baseline** (confounded by the merge, so not paired on one code base): held-out
  P95 27.1 → 15.4 s, test_v2 24.2 → 17.3 s, TTFT P50 5.3–5.7 → 2.8–3.0 s. Task success is +0.031 [−0.006, +0.088]
  and +0.055 [+0.022, +0.094].

**Trade-offs, stated plainly.**

- *Tool precision* (share of calls that were needed) falls slightly with prefetch: 0.684 → 0.675 on
  held-out and 0.650 → 0.616 on test_v2. The planner fetches what its rules suggest, and some of it is not
  needed. It is not part of task success, and those calls take milliseconds on the replay snapshot; with live
  sources they cost upstream requests.
- *Derived numbers* raise the verifier's false-accept rate on swapped numbers from 0.53% to 0.93%. A wrong
  number that happens to equal a difference or ratio of two other numbers in the same cited sentence now passes.
  Overall false accept is 2.03% against 1.94%. Set `QI_AGENT_VERIFY_DERIVED=0` to trade that back for more revisions.
- *Citation repair* edits the model's draft. It only adds ids (and removes non-existent ones) when exactly
  one evidence item holds the value, and the result must pass the same verifier. The binding is still to evidence
  items, not fields, the known limit documented in `verify_answer`.
- *The stall timeout* bounds the wait for the next streamed chunk, not the whole call. A model that keeps
  streaming slowly is not cut off; that is left to the run deadline (90 s + 20 s for the answer).
- The P95 target (< 20 s) is met on both sets. It is not a guarantee under load or on a slower model: see
  the load test and the GLM check below.

### Under load: 4 users, streamed ([`load_test-agent-4-*.json`](results/perf/agent/))

The same server build (commit `0309fdf`, merged code, live data off), started once with the previous path
(`QI_AGENT_PREFETCH=0 QI_AGENT_REVISE_POLICY=llm QI_AGENT_LLM_STALL_TIMEOUT_S=0 QI_AGENT_VERIFY_DERIVED=0`)
and once with the defaults. 4 users × 6 requests from the `research` set: 8 multi-tool questions,
harder than the evaluation sets. Requests go to `/agent/chat/stream` (`--stream`), so TTFT is measured as a client sees it.
Run on 2026-09-28 16:36–16:41 UTC, load average 3.0–3.3 at the start.

```bash
python -m scripts.load_test --base-url http://127.0.0.1:8811 --mode agent --questions research --users 4 \
    --requests 6 --usd-cny 6.7489 --questions-per-day 2000 --timeout 240 --stream --label defaults
```

| Server | Req | Throughput (req/s) | P50 (s) | P95 (s) | P99 (s) | TTFT P50 (s) | TTFT P95 (s) | LLM calls per question | ¥ per 1k questions | Verified | HTTP errors / LLM errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| previous path | 24 | 0.145 | 13.0 | 29.6 | 92.1 | 8.7 | 17.3 | 3.75 | 14.2 | 88% | 0 / 0 |
| defaults | 24 | 0.208 | 11.6 | 26.7 | 26.7 | 7.8 | 14.5 | 2.75 | 12.6 | 96% | 0 / 0 |

24 requests per row is a small sample: P95 is the second-slowest request and P99 the slowest. The
direction matches the evaluation runs: one LLM call fewer per question, a lower tail (the 92 s
stall is gone), 43% more throughput and 11% lower cost. **On these harder questions the P95 is still above
20 s**, and TTFT at 4 users (7.8 s) is higher than in the 3-worker evaluation (3 s). They need more tool
rounds and longer answers than most evaluation tasks. The client cannot see gateway 429s that a
retry recovered; no request failed or fell back.

### Second model: GLM-5.3 flash spot check ([`perf-glm-holdout-*.json`](../evaluation/results/))

Held-out, one repeat per run, 3 workers, streamed. The previous path and the defaults were run in the
order off, on, on, off (at `8a85ae5`), because GLM latency drifts between runs:

| Runs (pooled, 112 turns each) | P50 (s) | P95 (s) | P99 (s) | TTFT P50 (s) | TTFT P95 (s) | LLM calls/turn | revision rate (LLM turns) | task success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| previous path (off + off2) | 13.6 | 60.6 | 96.2 | 9.3 | 38.0 | 2.19 | 38% | 1.00, 1.00 |
| defaults (on + on2) | 11.0 | 68.5 | 85.3 | 8.8 | 54.7 | 1.36 | 21% | 1.00, 1.00 |

The switches cut GLM's LLM calls by 38% and halve its revisions without losing a task. **The latency
effect is not established:** two runs of the same configuration differed by 40% (P50 14.9 vs
20.8 s with the switches off, 19.1 vs 11.0 s with them on). A single GLM call ranges from 3 s to 75 s, so the
tail is set by call-to-call variance, not by the number of calls. With prefetch, GLM's first call writes
the whole answer (about 500 output tokens, 320 of them reasoning), and a slow first call now delays the first
token. The TTFT P95 is higher (38 → 55 s). The defaults are tuned for DeepSeek, which serves the
agent path. GLM remains a failover model, and its tail is bounded by the run deadline.

### What was not changed, and why

- **Reasoning effort.** Tool-loop calls use the model default and already spend few reasoning tokens (40–100
  per call on DeepSeek). The answer-writing calls are at `low`. Lowering them further was not tried:
  reasoning is not where the time goes, and the answer quality depends on it.
- **Compacting tool observations or history.** Prompts are 2.6–6k tokens and mostly cached. The first-call
  prompt with prefetched evidence grows by 600–1,000 tokens and costs about 0.6 s. The LLM call count
  dominates, not the prompt size.
- **Streaming earlier.** The answer already streams as the first JSON field (`answer`) is generated.
  TTFT is the time until the answer-writing call starts, which prefetch halves. A draft that later fails
  verification is still replaced by the verified answer in the final `answer` event.

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

Section 2a: `python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3 --workers 3 --sets holdout,test_v2 --modes agent --stream --out outputs/agent_eval/<run>.json`,
then `python -m evaluation.agent_eval.profile outputs/agent_eval/<run>.json` for the breakdown and
`python -m evaluation.agent_eval.results outputs/agent_eval/<run>.json --name <name>` for the committed file.

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
