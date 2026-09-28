# FinSight independent multi-turn evaluation set — multiturn_v1

`multiturn_v1.jsonl`: 49 conversations, 206 turns. It uses the task format read by
`evaluation/agent_eval/runner.py` and `metrics.score_turn`.

## Construction protocol

- **Author:** Claude (Opus 5.5, Claude Code subagent), 2026-09-28, written for the user of `<repo>`.
- **Commit:** FinSight-r2 at commit `47dd024`, with a clean tracked tree.
- **Independence.** The author did not open the agent's routing, memory, graph or planner code
  (`query_intelligence/agent/router.py`, `memory.py`, `graph.py`, `planner.py`). The author also did not open
  any existing task file (`evaluation/agent_eval/tasks/*.jsonl`). The conversations are modelled on how real
  users talk: elliptical follow-ups, pronouns, topic switches, bait questions and off-topic interruptions.
- **Code that was read:**
  - `evaluation/agent_eval/metrics.py`: the `score_turn` expect keys and the hedge/missing markers.
  - `evaluation/agent_eval/build_tasks.py`: only `_task`, `_turn` and `TRADING_PATTERNS`. The first ~85 lines
    were read, and they include that file's constant tables.
  - `evaluation/agent_eval/runner.py`: how tasks are loaded.
  - Tool specs in `query_intelligence/agent/tools/*.py`: parameters and evidence ids.
  - `verifier._is_supported`: numeric tolerance and scales.
  - `data/structured_data.json` and `data/entity_master.csv`: to see which symbols are covered.
- **Facts.** Every `required_facts` value was taken from the real offline tools, not guessed.
  1. Run the probe:
     ```
     cd <repo> && PYTHONPATH=. .venv/bin/python evaluation/agent_eval/tasks/ (originally written outside the repo) probe_offline_tools.py
     ```
     It calls `build_offline_service()` and `build_registry_for_service(svc)`. It then runs `reg.run(...)` for
     `get_price_history`, `compute_indicators`, `get_fundamentals`, `get_macro_indicators`, `resolve_entity`,
     `search_news` and `search_announcements`. Output goes to `tool_dump.json`, sha256
     `5135044218a31fd958cc7e58160d8c23e0715e1a0882372db47af9b14ce39a96`.
  2. Values were copied from `tool_dump.json` into `build_multiturn.py`, which writes the jsonl:
     ```
     python3 evaluation/agent_eval/tasks/ (originally written outside the repo) build_multiturn.py
     ```
  3. Validate with:
     ```
     cd <repo> && PYTHONPATH=. .venv/bin/python evaluation/agent_eval/tasks/ (originally written outside the repo) verify_multiturn.py
     ```
     The script re-runs the tools fresh; it does not read the dump. It checks:
     - every line parses and every id is unique;
     - each task has 3–6 turns;
     - expect keys are limited to the ones `score_turn` reads;
     - `forbidden_patterns` is identical to `build_tasks.TRADING_PATTERNS`;
     - all 172 fact checks pass: the evidence id is produced now and the value equals a numeric field of that payload;
     - every `required_entity` resolves to itself, and every entity mention in the queries resolves to the intended symbol;
     - every `must_state_missing` case really is absent from the data.

     Result: `OK: all checks passed`.

## Offline coverage (as found by the probe; snapshot as of 2026-04-22)

| Data | Coverage |
|---|---|
| Prices | 600519.SH, 000858.SZ, 601318.SH, 510300.SH, 159915.SZ, 512880.SH, 000300.SH. 510300.SH has 5 closes but `pct_change_1d` is null. The other symbols have 1–2 closes. |
| Fundamentals | Only 600519.SH, 000858.SZ and 601318.SH. All use report_date 2025-12-31. Industry snapshots exist for 白酒 and 保险. |
| Missing fundamental fields | No `gross_margin` for Moutai or Ping An. No dividend yield, market cap or growth for any stock. |
| Indicators | Only 510300.SH: MA5 4.7674, pct_3d 1.5193. RSI, MACD, MA20 and volatility are null. Every other symbol returns "unavailable". |
| Macro | CPI 0.8, PMI 50.6, M2 8.1, CN10Y 2.31, all as of 2026-03-31. No LPR. |
| Resolve but no data | 宁德时代 300750.SZ, 招商银行 600036.SH, 上证指数 000001.SH and others. These are used on purpose in `must_state_missing` turns. |
| Unresolved | Apple, Bitcoin, 英伟达, 狗狗币 and Amazon do not resolve. `S&P 500` resolves to `SPX.US`, but it is still expected to be refused as outside A-share scope. |

## Expectation conventions

- **Notes.** Every turn has a `note` field at turn level (next to `query` and `expect`) explaining the
  expectation. The runner ignores it.
- **Crypto and US stocks.** Expected `behavior: "refuse"`. The note records that "refuse / state not covered"
  is the intended outcome, but it is scored as refuse. If the agent answers with "not covered" text instead of
  routing to refuse, these turns will fail. Decide whether to accept that before comparing systems.
- **Advice and target-price bait.** Expected `answer` with `must_hedge`, plus the trading-instruction
  `forbidden_patterns` on every turn. A hard refusal would fail the behaviour check.
- **Missing data.** `must_state_missing` turns are `answer` turns. The agent must state that the data is
  missing and must not invent a number. Where the agent should at least try the natural tool, the turn also
  sets `required_tools`.
- **Fact values.** Facts use raw evidence values:
  - ROE 0.33 is matched against "33%" through the verifier's x0.01 scale.
  - Revenue 174120000000 is matched against "1741.2亿" through the 1e8 scale.
- **Entities in two-entity turns.** `required_entity` holds one symbol, normally the first-mentioned or
  newly introduced entity.
- **Snapshot replay.** The tasks are new, so the replay snapshot in the repo will not contain their tool calls.
  Run with `--no-replay` (offline tools directly) or `--record-missing` against a new snapshot, for example:
  ```
  python -m evaluation.agent_eval.runner --mode agent --llm deepseek --tasks evaluation/agent_eval/tasks/ (originally written outside the repo) multiturn_v1.jsonl --no-replay
  ```

## Counts

- **Totals:** 49 tasks and 206 turns.
- **Turns by behaviour:** 189 answer, 13 refuse, 4 clarify.
- **Turn flags:** 34 turns with `must_hedge`, 27 with `must_state_missing`, 172 required facts.
- **Language:**

  | Language | Tasks | Share |
  |---|---|---|
  | zh | 28 | 57% |
  | en | 18 | 37% |
  | mixed | 3 | 6% |

  A few more mixed-language turns sit inside zh tasks, for example `And ROE?` in mt-zh-14.
- **Category:**

  | Category | Tasks |
  |---|---|
  | missing_or_wrong_period | 8 |
  | coreference | 6 |
  | outside_a_share_scope | 6 |
  | macro_switch | 5 |
  | why_followup | 5 |
  | oos_in_conversation | 5 |
  | dangling_clarify | 4 |
  | cross_turn_comparison | 3 |
  | mixed_language | 3 |
  | ellipsis_followup | 2 |
  | advice_bait | 2 |

  Categories name each conversation's main feature. Most conversations also include ellipsis, pronouns,
  macro switches, bait or "why" turns. For example, 34 turns carry `must_hedge` although only 2 tasks are
  labelled `advice_bait`.

## Files

| File | Purpose |
|---|---|
| `multiturn_v1.jsonl` | The task set. |
| `build_multiturn.py` | Generates the jsonl. |
| `probe_offline_tools.py` → `tool_dump.json` | Raw tool output the facts were taken from. |
| `verify_multiturn.py` | Validation: parsing, ids, schema, fresh-tool fact re-check, entity resolution, checks that missing data is really missing. |

## SHA-256

```
f63ce0f4656faf99a09c2a4e4758b6144137eb4be34179c452f20ccb0238415c  multiturn_v1.jsonl
```
