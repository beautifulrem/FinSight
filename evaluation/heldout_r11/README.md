# Round-11 independent held-out slice (reviewer-written, unopened by the engineers)

Written by the round-11 reviewer at `4ad3380` (2026-10-11), **before any round-15 fix**, from the bug classes the
round-11 review found plus controls. The phrasings here appear nowhere in `round11.md` (its repros are reworded) or in
the reviewer's probe scripts (`round11/probes/`). **The engineers must not open this folder** (do not read the jsonl
files, and do not point any tool other than the runner/scorer below at it before the measured run). Run it once after
the fixes, after a leak check: `python -m evaluation.agent_eval.slice_overlap <this folder>` must scan every
`evaluation/agent_eval/tasks/*.jsonl`, `evaluation/claim_bench/*.jsonl` and test literals, **and** the round-15 code
and docs diff (round 11 found content-level near copies of round-10 slice items in fix-era tests that the 0.8-ratio
threshold did not count; report the overlap-excluded number with a looser threshold too, e.g. difflib ≥ 0.70 with the
same company and the same number).

- **It runs on the SHIPPED offline default: snapshot extension + corrections layer.** Run the chat slice with
  `QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`. The facts were computed with
  `build_offline_service(snapshot_ext=True, snapshot_fix=True)`; the claim scorer and the verifier build the same
  service themselves.
- If the data layers change, `verify_heldout_r11.py` fails on the affected facts; re-derive with the build script and
  say so next to the result.

## Files and hashes

| file | sha256 |
|---|---|
| `chat_r11_heldout.jsonl` | `42b495d9957b062a5845b0775156e1100eb3461b6669e5b44185410481209770` |
| `claims_r11_heldout.jsonl` | `5eedbc7ca2748a441a0da852edaa4eefec327b219eabfc8b1cc19b5a51311f34` |
| `build_heldout_r11.py` | `fafeaadfa4a03ad360890c243fee1ded08ef01aaa567abbc2bf2fe2d0df3ff2f` |
| `verify_heldout_r11.py` | `e0695eb7e8e6c9d9016a71dac506acca31dbf043155e8e5e3654174b0ba88e3a` |
| `score_claims_r11.py` | `89c60521be4647342d1ad91556d7197ca9d41b42abe160261f1ca165ab93dfb9` |
| `first_run/chat_r11-auto-4ad3380.json` (pre-fix run) | `7d9e5c2b9aafe0aa5b48ca9435a8ab9377aba7482b044d296998f3f19970128a` |
| `first_run/claims_r11-4ad3380.json` (pre-fix run) | `02fdea06b2ac0e22b94cb08ed6dfcf2e0b071ee0ad5d508ee03ac7aa95534b2d` |

## Contents

`chat_r11_heldout.jsonl`: **60 conversations / 73 turns / 69 required facts** (40 zh, 20 en), same task schema as
`evaluation/agent_eval/tasks/*.jsonl` (`required_facts` with a `derive` spec, `required_tools`,
`required_entity/entities`, `behavior`, `must_hedge`, `must_state_missing`, `required_limitations`,
`forbidden_patterns` = the four trading patterns, `language`), plus two verify-only keys (`verify_date`,
`verify_after_data`).

| category | conversations | what it targets |
|---|---|---|
| `sum_words` | 5 | totals worded outside the round-14 lexicon (拢共 / 统共 / 合并 / "jointly" / 一起 after an ellipsis) |
| `mean` | 5 | the mean of two or three operands, incl. "add them up then divide by 3" and a carried pair |
| `sum_vs_one` | 4 | one value against the total of two others (more or less, and by how much), zh and en |
| `share_of_total3` | 3 | one operand as a percent of a THREE-operand total (zh and en) |
| `sum_and_share` | 2 | a total and a share asked in one message after two carried operands |
| `ellipsis_two` | 2 | an ellipsis that names two new targets and keeps the metric |
| `holding_follow_up` | 3 | a holding carried into a follow-up that adds a second position |
| `holding_forms` | 4 | 万 counts, a colon/semicolon list ending in "合计市值？", Chinese numerals, "Total value?" without "worth" |
| `multi_purchase_pnl` | 3 | gain or loss over two purchases at different costs |
| `price_recovery` | 3 | the % rise needed to get back to the 52-week high; yuan above the 52-week low |
| `mv_over_profit` | 2 | market cap ÷ net profit (not the stated P/E) |
| `two_questions` | 2 | two comparisons asked in one message |
| `verb_comparative` | 3 | 多挣出, "by what margin … exceed", "eclipse … by" |
| `dated_close` | 4 | a stated past date inside the data (2025-12-31; a 52-week-low day) and a date after the last bar (must say so) |
| `foreign_line` | 5 | HK lines by stock code in brackets, venue (联交所), currency (HKD), "NNNN.HK" after a Chinese name; one A-share control |
| `unreleased_results` | 3 | results of a report not yet published (Q3 2026, FY2026): `must_hedge` |
| `news_missing` | 2 | news for a company without offline news: `must_state_missing` with `search_news` |
| `control` | 5 | single facts, a plain sum, a single holding, a drawdown % |

`claims_r11_heldout.jsonl`: **24 claims** (means of two or three, "more than A and B combined", "X trillion yuan",
English published sentences with a year and a growth rate, "但/不过" clauses with an implicit subject and a short
reference name, H-share claims (unverifiable: the data has no H-share line), 扣非 vs 归母 for one company, shares of a
three-company total, a forecast, a currency mistake, controls). Verdicts come from per-part predicates in
`build_heldout_r11.py` (all true → supported, all false → contradicted, mixed → partially_supported); three claims
carry a fixed `unverifiable` verdict (two H-share lines, one forecast) and no data predicate; one currency claim also
accepts `unverifiable`.

## Verification

`verify_heldout_r11.py` re-runs the tools and re-derives every fact with its own implementation of every op (`value,
mul, sub, sum, div, pct, rel, lin, spread, spread_ratio, sumn, share_total, pnl, mean, pnl_multi, diff_sum`), checks
operand discrimination, stated dates against the bar dates, dates after the last bar, refusals with
`out_of_coverage`, a synthetic gold response per turn through `metrics.score_turn`, and re-runs the claim predicates.
At `4ad3380`: **OK** (warnings only: two facts also match another field of the same evidence within the scorer's
tolerance; the H-share names resolve to A-share evidence, which is the behaviour under test). Mutation test (one chat
value, one claim verdict changed in a copy): **2 failures**, as expected.

```
FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python verify_heldout_r11.py      # prints OK
cd /path/to/FinSight && QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1 python -m evaluation.agent_eval.runner \
    --mode auto --no-replay --tasks /path/to/heldout_r11/chat_r11_heldout.jsonl --out <out.json>
FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python score_claims_r11.py <out.json>
```

(live sources off: `QI_USE_LIVE_MARKET/MACRO/NEWS/ANNOUNCEMENT=false`, `QI_AGENT_TRACE_DIR=off`.)

## First run (before any round-15 fix), `4ad3380`, deterministic path, no LLM, ext + corrections

| set | result |
|---|---|
| chat, task success | **0.200 [0.10, 0.30]** (12/60), turn 0.342, behaviour 0.945 |
| chat by category | sum_words 0/5, mean 0/5, sum_vs_one 0/4, share_of_total3 1/3, sum_and_share 0/2, ellipsis_two 1/2, holding_follow_up 0/3, holding_forms 1/4, multi_purchase_pnl 0/3, price_recovery 0/3, mv_over_profit 0/2, two_questions 0/2, verb_comparative 0/3, dated_close 0/4, foreign_line 1/5 (the A-share control), unreleased_results 2/3, news_missing 1/2, control 5/5 |
| chat by language | zh 0.200, en 0.200 |
| claims, verdict accuracy | **0.292 [0.125, 0.458]** (7/24); right: the forecast, the currency claim, both controls, one sum reference, one trillion, one implicit-subject clause. Wrong include false contradictions of true claims (means, "2.95 trillion yuan", "earned RMB30.1 billion, up roughly 37%", 扣非 vs 归母 difference) and a false accept (an H-share P/B claim supported from the A-share P/B) |

Note on the scorer: `must_hedge` and `must_state_missing` use the runner's marker lists (`metrics.py`: 无法 / 可能 /
没有 / "not " …), which are weak; two of the three `unreleased_results` passes and the `news_missing` pass may be
marker hits rather than real hedges or statements. Read the answers before claiming a fix in those classes.

Outputs: `first_run/chat_r11-auto-4ad3380.json`, `first_run/claims_r11-4ad3380.json`, logs next to them.

## Committed to the repository (after the round-15 fixes)

Copied unchanged (same hashes) after the round-15 fixes were measured on it once, at `c5ffb7d` with
`QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`, reading this folder from the reviewer's directory (as in rounds 9 and
10): chat 12/60 → 44/60 (`evaluation/results/chat_heldout_r11-auto-nollm-*.json`), claims 7/24 → 19/24
(`claims_heldout_r11-*.json`). The engineers state they did not open the folder; the record cannot show that.
`python -m evaluation.agent_eval.slice_overlap --content` before the run (`slice_overlap-heldout_r11-at-c5ffb7d.json`):

| Exclusion | chat | claims |
|---|---|---|
| none | 12/60 → 44/60 | 7/24 → 19/24 |
| text near-copies (2 chat, 2 claims) | 12/58 → 42/58 | 6/22 → 17/22 |
| + content overlaps | 11/55 → 39/55 | 5/20 → 15/20 |
| + content signatures (upper bound) | 9/44 → 32/44 | 4/13 → 9/13 |

One claim (r11c09) is an exact copy in `tests/test_agent_round15_claims.py:60`, and **the record does not explain
it**. (Until round 16 this section said the sentence was also in the reviewer's probe file
`round11/probes/planted11.py`. That was wrong. The round-12 review (L15) found that `planted11.py` has a different
Moutai sentence, a planted "USD 1.77 trillion" market value. A `grep -F` in round 16 confirms that the r11c09 sentence
is not in `planted11.py` and not in any other file under `round11/`, `round11.md` or `round11.json`.) The record
shows:

| When (+0800, 2026-10-11) | Event | Source |
|---|---|---|
| 03:08:24 | `claims_r11_heldout.jsonl` created in the reviewer's folder | file birth time in `finsight-review/heldout_r11/` |
| 03:12:02 / 03:12:24 | reviewer's first runs of the slice at `4ad3380` (chat / claims) | `run_at` in `chat_heldout_r11-auto-nollm-prefix.json` / `claims_heldout_r11-prefix.json` |
| 03:21–03:30 | round-11 review report written | birth and modification times of `round11.md` / `round11.json` |
| **03:42:38** | the sentence enters the repository in `138d64d` ("test(claims): round-15 K6 claim classes in own wording", parent `4ad3380`; its docstring says the review's sentences are not copied) | `git log -S` (the first commit that contains it) |
| 06:51:18 / 06:56:46 | after-fix commit `c5ffb7d` / its claims run, reading the slice from the reviewer's folder | commit time / `run_at` in `claims_heldout_r11-after-fix.json` |
| 06:58:45 | the slice enters the repository in `efa126c` | `git log -- evaluation/heldout_r11` |

So the copy was committed 34 minutes after the slice existed in the reviewer's folder and 3 h 16 min before the slice
was committed. The record cannot tell whether this is a coincidence (a natural sentence written for the round-11
review's K6 trillion-unit class) or exposure. Either way, r11c09 is excluded in every overlap tier above (it is the
exact text hit). It was right in both runs, so excluding it alone gives 6/23 → 18/23.
From round 16 this slice is **after exposure**.

