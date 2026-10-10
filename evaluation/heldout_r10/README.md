# Round-10 independent held-out slice (reviewer-written, unopened by the engineers)

Written by the round-10 reviewer at `7f4f06c` (2026-10-10), **before any round-14 fix**, from the bug classes the
round-10 review found plus controls. The phrasings here appear nowhere in `round10.md` (its repros are reworded) or
in the reviewer's probe scripts. **The engineers must not open this folder.** Run it once after the fixes, after a
leak check (`python -m evaluation.agent_eval.slice_overlap` must also cover `tasks/agent_eval_ext_v1.jsonl`, see
round10.md J-bug on the overlap tool).

- **It runs on the SHIPPED offline default: snapshot extension + corrections layer.** Run the chat slice with
  `QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1` (the runner is pinned to the v1 seed otherwise; the facts were
  computed with `build_offline_service(snapshot_ext=True, snapshot_fix=True)`). The claim scorer and the verifier
  build the same service themselves.
- If the corrections layer changes (e.g. a corrected figure is revised again), `verify_heldout_r10.py` fails on the
  affected facts; re-derive with the build script and say so next to the result.

## Files and hashes

| file | sha256 |
|---|---|
| `chat_r10_heldout.jsonl` | `576a3861410644b44f12befc184be9ec7a32f0693900952f6be2b59e64c4d5a6` |
| `claims_r10_heldout.jsonl` | `12105276e9719fc2252b33be3eda03f228d6a8caacad40f1502dad76dff32f63` |
| `build_heldout_r10.py` | `c90b2077eaedb8e8f4dc7445b633bc1db054c1b74a0f184346a286f7cffbfaa8` |
| `verify_heldout_r10.py` | `a34b485e20b649f39206058c48b2a7326c4311b0a18d5678e57eb8dfe6ab97c2` |
| `score_claims_r10.py` | `936ef04021afd579289e7d45e45c1c4d0c4ea22287639732eab1918414baba6c` |
| `first_run/chat_r10-auto-7f4f06c.json` (pre-fix run) | `5e5093e352c7941dd855ae13d653bf5ac037949d783a18b3544c1112a28661aa` |
| `first_run/claims_r10-7f4f06c.json` (pre-fix run) | `eecfbab99d812da7ceee11438d592db0502afc6b50341e76d4971553ca942997` |

## Contents

`chat_r10_heldout.jsonl`: 48 conversations / 53 turns / 43 required facts (33 zh, 15 en), same task schema as
`evaluation/agent_eval/tasks/*.jsonl` (`required_facts` with a `derive` spec, `required_tools`,
`required_entity/entities`, `behavior`, `must_hedge`, `must_state_missing`, `required_limitations`,
`forbidden_patterns` = the four trading patterns, `language`).

| category | conversations | what it targets |
|---|---|---|
| `share_of_total` | 4 | one operand as a percent of the TOTAL of two or three (not of the other operand) |
| `sum` | 5 | totals worded with 总计 / 共 / 合在一块儿 / "in sum" / "between the two of them", one message and follow-ups |
| `verb_comparative` | 4 | 弱多少, 超出…多少亿, "top … by", "trail" |
| `extremes` | 3 | highest ÷ lowest of three, priciest − cheapest, 最赚钱 − 最不赚钱 |
| `multi_holding` | 4 | three positions; a carried position plus a named one; lots in English; ETF units plus shares |
| `cost_basis` | 3 | paper gain/loss against a stated cost price |
| `yes_no_multiple` | 3 | 有没有…两倍 / "more than three times" / 三倍不到吧: the multiple must be stated |
| `forecast_beyond` | 5 | absolute future dates, a future quarter, an earnings forecast, an opening-gap forecast, an injected wrapper: `must_hedge` |
| `foreign_line` | 4 | HK lines worded without "港股/H股/Hong Kong shares", a London GDR, an H-vs-A discount: refuse `out_of_coverage` |
| `metric_alias` | 5 | 扣非 / recurring net profit (`profit_dedt`), 流通市值 / free-float cap (`circ_mv`), 营业利润 (absent: state it missing) |
| `price_derived` | 3 | % above the 52-week low, yuan below the 52-week high, % below the 52-week high |
| `control` | 5 | single facts (ext + corrected v1), a plain gap, a single holding |

`claims_r10_heldout.jsonl`: 24 claims (扣非 thresholds, "lower than X but higher than Y", superlatives with the
subject first, published-style "同比增长 / rose N% year on year to …" sentences, "不到…的N倍", 抵得上两个多, 四倍,
五分之二 / 三分之二, 流通市值, 六成, level + multiple, corrected-data figures, controls). Verdicts come from per-part
predicates in `build_heldout_r10.py` (all true → supported, all false → contradicted, mixed → partially_supported).

## Verification

`verify_heldout_r10.py` re-runs the tools and re-derives every fact with its own implementation of every op
(`value, mul, sub, sum, div, pct, rel, lin, spread, spread_ratio, sumn, share_total, pnl`), checks operand
discrimination (a derived number must not be matched by an operand as the scorer reads it), absent fields behind
`must_state_missing`, refusals with `out_of_coverage`, a synthetic gold response per turn through
`metrics.score_turn`, and re-runs the claim predicates. At `7f4f06c`: **OK** (warnings only: four facts also match
another field of the same evidence within the scorer's tolerance; the HK names resolve to A-share evidence, which is
the behaviour under test). Mutation test (one chat value, one claim verdict changed in a copy): **3 failures**, as
expected.

```
FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python verify_heldout_r10.py      # prints OK
cd /path/to/FinSight && QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1 python -m evaluation.agent_eval.runner \
    --mode auto --no-replay --tasks /path/to/heldout_r10/chat_r10_heldout.jsonl --out <out.json>
FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python score_claims_r10.py <out.json>
```

## First run (before any round-14 fix), `7f4f06c`, deterministic path, no LLM, ext + corrections

| set | result |
|---|---|
| chat, task success | **0.250 [0.125, 0.375]** (12/48), turn 0.321, behaviour 0.925 |
| chat by category | share_of_total 0/4, sum 0/5, verb_comparative 1/4, extremes 2/3, multi_holding 1/4, cost_basis 0/3, yes_no_multiple 2/3, forecast_beyond 1/5, foreign_line 0/4, metric_alias 0/5, price_derived 0/3, control 5/5 |
| chat by language | zh 0.273, en 0.200 |
| claims, verdict accuracy | **0.667 [0.458, 0.833]** (16/24); misses: both 扣非 claims (one false accept, one false contradiction), the subject-first English superlative (unverifiable), the "rose 62% … to RMB51.8 billion" sentence (partially), both "不到…的N倍" bounds (partially / contradicted), 抵得上两个多 (unverifiable), 流通市值 (contradicted against the total market cap) |

Outputs: `first_run/chat_r10-auto-7f4f06c.json`, `first_run/claims_r10-7f4f06c.json`, logs next to them.

## Committed to the repository (after the round-14 fixes)

Copied unchanged (same hashes) after the round-14 fixes were measured on it once, at `3480a96` with
`QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`: chat 12/48 → 38/48 (`evaluation/results/chat_heldout_r10-auto-nollm-*.json`),
claims 16/24 → 20/24 (`claims_heldout_r10-*.json`). The round-14 engineers did not open this folder; before the run
`python -m evaluation.agent_eval.slice_overlap` found 3 chat tasks that near-copy own wording (no exact copies, no
claims; `slice_overlap-heldout_r10-at-3480a96.json`). Without them: chat 12/45 → 35/45. From round 15 this slice is
**after exposure**.

