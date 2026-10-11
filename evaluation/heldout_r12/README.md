# Round-12 independent held-out slice (reviewer-written, unopened by the engineers)

Written by the round-12 reviewer at `bb99360` (2026-10-11), **before any round-16 fix**, from the bug classes the
round-12 review found plus controls. The phrasings here appear nowhere in `round12.md` (its repros are reworded) and
nowhere in the reviewer's probe scripts under `round12/probes/` (checked: the closest probe string to any slice text has
difflib ratio 0.69 apart from two controls that deliberately reuse the round-11 repro shape, see below). **The engineers
must not open this folder** (do not read the jsonl files; do not point any tool other than the runner / scorer / verify
script / `slice_overlap` at it before the measured run). Run it once after the fixes, after a leak check:
`python -m evaluation.agent_eval.slice_overlap <this folder> --content` over the round-16 code, tests, task files and docs.
Report the overlap-excluded numbers too. Record the runner's `tasks_file` path in the result file and state, as the
record shows it, where the slice was read from.

- **It runs on the SHIPPED offline default: snapshot extension + corrections layer.** Run the chat slice with
  `QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`. Facts were computed with
  `build_offline_service(snapshot_ext=True, snapshot_fix=True)`; the claim scorer and the verifier build the same
  service themselves.
- If the data layers change, `verify_heldout_r12.py` fails on the affected facts; re-derive with the build script and
  say so next to the result.

## Files and hashes

| file | sha256 |
|---|---|
| `chat_r12_heldout.jsonl` | `b590ea04091e685665d6bb07c373e236b8163c28068e65856d1fb1b1c579edf6` |
| `claims_r12_heldout.jsonl` | `f39498ac3bdf344c9e016cf6b3468c25d3c41f6203e7e78d500c5c86ab75dc74` |
| `build_heldout_r12.py` | `92de2a3254a1ab5bd2d3953df35c333a0f04eb6730a4b98f3dfb8edcd2313e11` |
| `verify_heldout_r12.py` | `1d20a9988cf3766c9a9f137a1a2c1ef6e21fe28decaba73fdce4b3529af0e7a6` |
| `score_claims_r12.py` | `d78661c974bc7410a376961237c2ab3a9a8096d43960b6bdefcf5a215749572c` |
| `first_run/chat_r12-auto-bb99360.json` (pre-fix run) | `c0fe1b8ffc6366169f37e8f6e044aa0821494dcf2b3635ba2c7f449fce881878` |
| `first_run/claims_r12-bb99360.json` (pre-fix run) | `3fb631d258496ce2391d1475d64991fa7413d8810ab2949c68c7bd6dded674c2` |
| `first_run/slice_overlap-heldout_r12-at-bb99360.json` | `76fe22e5f4964a095e4e9cd22e436ea03053c91727145271d0b9abf9c76a05f2` |

## Contents

`chat_r12_heldout.jsonl`: **50 conversations / 54 turns / 46 required facts** (35 zh, 15 en), same task schema as
`evaluation/agent_eval/tasks/*.jsonl` (`required_facts` with a `derive` spec, `required_tools`, `any_of_tools`,
`required_entity/entities`, `behavior`, `must_hedge`, `must_state_missing`, `required_limitations`,
`forbidden_patterns` = the four trading patterns, `language`), plus the verify-only key `verify_date`.

| category | conversations | what it targets |
|---|---|---|
| `total_vs_one_anchor` | 5 | a total of two compared with a third whose name comes before the comparison word (同…比 / 与…相比 / 跟…比较) or an English "compared with C … combined" / "taken together … than C" |
| `leading_adverb` | 3 | an English total question that opens with an adverb and a comma or colon |
| `multi_purchase_forms` | 5 | several purchases with the name between the verb and the count, 块 / 补了 / 笔, an English "then N more at", a list "N shares at X and N shares at Y", a cost follow-up after a price turn |
| `compound_count` | 4 | counts written 两万五千 / 1万5千 / 4千2百 and an English sentence with a 万 count |
| `holding_list_carry` | 5 | a 、-list or "plus N of X" holding; a position stated in one turn (no question) and a second one added with a total question in the next |
| `two_operations` | 4 | two questions in one message separated by ？ or 再 (total + share, which + gap, ratio + total, lower + by how much) |
| `short_name` | 3 | short names in everyday use (紫金, 宁德, 长电) |
| `metric_definition` | 4 | what P/E (TTM), total vs float market value, ROE, gross margin mean — the labels the KPI tiles show (`required_tools: ["explain_concept"]`) |
| `future_market_value` | 3 | a market value reaching a level by a future year or horizon: `must_hedge` |
| `dividend_forecast` | 2 | a dividend for this year (not in the data): `must_hedge` + `must_state_missing` |
| `price_gap_yuan` | 2 | the gap to the 52-week closing high in yuan (not in percent) |
| `etf_plus_stock` | 1 | fund units plus shares in one holding (round-10 open item) |
| `dated_close` | 1 | a stated date that is a 52-week-extreme bar |
| `control` | 8 | a sum, a mean, a holding, two lots in the round-15 form, an A+H company's Hong Kong line, an earnings forecast, a ratio, a total compared with a third named after the anchor |

`claims_r12_heldout.jsonl`: **26 claims** — growth stated before the amount (至 / 达到 / "to"), a direction without a
number (同比下降 / 有所减少 / "increased"), price against book value (破净 / 低于每股净资产 / "less than book value"),
a second clause that keeps the metric and states only a direction or a comparison ("…, and X's fell too", "…，B也实现了增长",
"…，招商银行更低"), one company's growth rate against another's (增速高于 / 快过), short names as the subject, the highest /
lowest within a named group of three, an H-share line (fixed `unverifiable`), controls. Verdicts come from per-part
predicates in `build_heldout_r12.py` (all true → supported, all false → contradicted, mixed → partially_supported).
Verdicts: 15 supported, 6 contradicted, 4 partially supported, 1 unverifiable.

## Verification

`verify_heldout_r12.py` (adapted from the round-11 verifier: same checks, new categories, plus a check that every
`required_tools` name exists in the tool registry) re-runs the tools and re-derives every fact with its own implementation
of every op, checks operand discrimination, stated dates against bar dates, refusals with `out_of_coverage`, a synthetic
gold response per turn through `metrics.score_turn`, and re-runs the claim predicates. At `bb99360`: **OK** (two info
warnings: `美的集团H股` / `美的集团港股` resolve to the A-share line in `get_price_history`, which is the behaviour under
test in `r12h_ctrl_zh_05`). Mutation test (one chat value, one claim verdict changed in a copy): **3 failure messages**
(two for the value, one for the verdict), as expected.

```
FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python verify_heldout_r12.py      # prints OK
```

## Overlap with the repository at `bb99360` (before any fix)

`python -m evaluation.agent_eval.slice_overlap <this folder> --content` → `first_run/slice_overlap-heldout_r12-at-bb99360.json`:
exact 0; text near 1 (`r12h_lots_zh_03` turn 1 "…收盘价是多少", a generic first turn, 10 characters); content 4
(`r12c02`, `r12h_ctrl_zh_03`, `r12h_ctrl_zh_04`, `r12h_etfmix_zh_01`); content signature 16 (a shared company with a
holding / cost-basis operation, mostly). Two controls deliberately reuse the round-11 repro shape (`r12h_ctrl_zh_04`, the
"先在X元买了N股，后来Y元又买了N股" form the round-15 rules cover; `r12h_ctrl_zh_06`, an earnings forecast), and the control
claim `r12c23` is a prefix of the committed round-10 claim r10c18 (a control; not counted by `slice_overlap`, which compares whole texts).

## First run at `bb99360` (pre-fix, no LLM, ext + corrections)

```
cd <FinSight checkout at bb99360>
QI_USE_LIVE_MARKET=false QI_USE_LIVE_MACRO=false QI_USE_LIVE_NEWS=false QI_USE_LIVE_ANNOUNCEMENT=false \
QI_AGENT_TRACE_DIR=off QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1 \
  python -m evaluation.agent_eval.runner --mode auto --no-replay \
  --tasks <this folder>/chat_r12_heldout.jsonl --out first_run/chat_r12-auto-bb99360.json
FINSIGHT_REPO=<checkout> python score_claims_r12.py first_run/claims_r12-bb99360.json
```

- **Chat: 16/50 = 0.320 [0.20, 0.44]**, turn success 0.370; controls 7/8 (the A+H Hong Kong line of 美的集团 is answered
  with the A-share close).
  Per category: total_vs_one_anchor 1/5, leading_adverb 2/3, multi_purchase_forms 0/5, compound_count 0/4,
  holding_list_carry 1/5, two_operations 2/4, short_name 0/3, metric_definition 0/4, future_market_value 0/3,
  dividend_forecast 2/2, price_gap_yuan 0/2, etf_plus_stock 0/1, dated_close 1/1, control 7/8.
- **Claims: 11/26 = 0.423 [0.23, 0.62]**. Per category: amount_after_growth 2/4, direction_only 0/4, book_value 0/3,
  elided_second_clause 2/4 (both errors are **false "supported"** verdicts of partly false claims), growth_comparison
  0/2 (one **false "contradicted"**: a growth-rate comparison checked as a revenue-level comparison), short_name 0/2,
  group_extreme 2/2, foreign_line 1/1, control 4/4.
- Outputs: `first_run/chat_r12-auto-bb99360.json`, `first_run/claims_r12-bb99360.json`, logs next to them, and
  `first_run/first_run_digest.txt` (per-turn outcome and answer excerpt).

Note on the scorer: `metrics.py` still accepts weak hedge / missing markers (可能 / 无法 / "cannot" / 没有 / "not ").
The two dividend tasks pass on "当前数据源没有…股息率数据，无法回答" — an honest answer, so the pass is right here — but a
post-fix run should be read with that in mind for `future_market_value`.

## Committed to the repository (after the round-16 fixes)

Copied unchanged (same hashes) after the round-16 fixes were measured on it once, at `9564263` with
`QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`, reading this folder from the reviewer's directory. The engineers state
they did not open it; the record cannot show that. `python -m evaluation.agent_eval.slice_overlap --content` before the
run (`slice_overlap-heldout_r12-at-9564263.json`):

| Exclusion | chat | claims |
|---|---|---|
| none | 16/50 → 42/50 | 11/26 → 24/26 |
| text near-copies | 16/49 → 42/49 | 11/26 → 24/26 |
| + content overlaps | 14/43 → 37/43 | 9/22 → 20/22 |
| + content signatures (upper bound) | 14/35 → 31/35 | 8/18 → 16/18 |

From round 17 this slice is **after exposure**.

