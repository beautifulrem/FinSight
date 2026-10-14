# Round-14 independent held-out slice (reviewer-written, unopened by the engineers)

Written by the round-14 reviewer at `64cfbeb` (2026-10-11), **before any round-18 fix**, from the bug classes the
round-14 review found plus controls. The phrasings here appear nowhere in `round14.md` (its repros are worded
differently and name the classes only) and are not copied from the reviewer's probe scripts under `round14/probes/`,
`round14/ui/`, `round14/sec/`, `round14/ops/` or `round14/notes.md` (difflib against every string literal and line
there: the closest is 0.83, a six-character follow-up "现在亏了多少" of the same form as a probe follow-up; every other
text is ≤ 0.82). Two turns were reworded after a draft run because one was an exact copy of a UI probe string and one
reached 0.85 against a probe (see "Draft run" below).

**The engineers must not open this folder** (do not read the jsonl files or the build script; do not point any tool
other than the runner / scorer / verify script / `slice_overlap` at it before the measured run). Run it once after the
fixes, after a leak check: `python -m evaluation.agent_eval.slice_overlap <this folder> --content` over the round-18
code, tests, task files and docs. Report the overlap-excluded numbers too. Record the runner's `tasks_file` path in the
result file and state, as the record shows it, where the slice was read from.

- **It runs on the SHIPPED offline default: snapshot extension + corrections layer.** Run the chat slice with
  `QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`. Facts were computed with
  `build_offline_service(snapshot_ext=True, snapshot_fix=True)`; the claim scorer and the verifier build the same
  service themselves.
- If the data layers change, `verify_heldout_r14.py` fails on the affected facts; re-derive with the build script and
  say so next to the result.

## Files and hashes

| file | sha256 |
|---|---|
| `chat_r14_heldout.jsonl` | `8e26761779d3d8f4479c7fd7814905bcb7cbb80630b83a4ff1beaea173028c86` |
| `claims_r14_heldout.jsonl` | `9b9d57f44cf9daf62bef916dfc1139ffbb2fc3e950e98f7d05fb54e7ddf77404` |
| `build_heldout_r14.py` | `11d0b81c567245b0f5111c5da8d8807bcf5aa9ebb70676741bd223408f3ec19c` |
| `verify_heldout_r14.py` | `349e94b59e4705d6aa1a2a82ba97df051474419ad80f4899342d66edad2f12a3` |
| `score_claims_r14.py` | `3fef795128111cfdd9baa3666abfa093d8b811b8088a7b9d2c0bd392e1f7bf54` |
| `first_run/chat_r14-auto-64cfbeb.json` (pre-fix run) | `3f51215b1f891a84e4493c55257461d23ddcda38998ecabfdbcc2200ec8bcd42` |
| `first_run/claims_r14-64cfbeb.json` (pre-fix run) | `65533db18b752b853019dcb5eb4b0ef56f1e4581174f3b7e1e5e9be4bf3a10e7` |
| `first_run/slice_overlap-heldout_r14-at-64cfbeb.json` | `2fcc666e6a9910642efb676849e2fc61c8ed86bf11e7db4a6afbed24b3176e81` |

## Contents

`chat_r14_heldout.jsonl`: **50 conversations / 62 turns / 57 required facts** (38 zh, 12 en), same task schema as
`evaluation/agent_eval/tasks/*.jsonl` (`required_facts` with a `derive` spec, `required_tools`,
`required_entity/entities`, `behavior`, `required_limitations`, `forbidden_patterns`). 15 categories:

| category | n | what it tests |
|---|---|---|
| closed_position | 5 | a position bought and then sold in full: the realised result, not a paper P&L at the close |
| comma_count | 3 | a price and a share count separated by a comma are two numbers, not one count |
| buys_then_sale | 2 | two purchases (the second with a bare price and 加 / 加仓) then a sale: the rest against the average cost |
| stated_cost_carry | 4 | a position with its cost stated in one turn, the P&L (or the rise back to cost) asked in the next |
| sale_followup | 3 | a stated position, then a sale in the next turn: the P&L of what is left |
| lookalike_unlisted | 5 | an unlisted subsidiary / business named with a covered short name and a word outside the business-word list |
| hk_terms | 4 | Hong Kong financial vocabulary in traditional characters (1 control) |
| fullwidth_latin | 3 | full-width Latin letters in a metric or a name |
| correction_turn | 3 | "I meant X" in the next turn keeps the asked metric |
| short_position | 2 | a short (融券 / shorted) gains when the price falls; a loss with that number is forbidden |
| portfolio_weight | 2 | one holding's share of a two-stock portfolio |
| range_52w | 3 | the width of the 52-week closing range and the close's place in it |
| nickname | 2 | market nicknames of covered companies |
| derived_figure | 2 | a figure derived from two fields (shares outstanding; profit per day) |
| control | 7 | forms the round-17 rules cover (partial sale, sum, average cost with 补, 52-week rise …) |

**New in round 14** (the build and verify scripts implement each independently): ops `realised`, `pnl_avg`,
`short_pnl`, `weight`, `range_pos`; `extra_forbidden` literal patterns after the wrong-value patterns (a loss stated
for a short position that gained). `wrong_values` as in round 13: the number a known wrong reading produces is a
forbidden pattern, so a turn that states it fails even when it also states the right one (paper P&L of a closed
position; the concatenated count; the first purchase lot only; the whole position after a sale).

`claims_r14_heldout.jsonl`: **24 claims** — exclusion_claim 4 (one company of a named group excluded), hk_terms_claim 5
(純利 / 按年 / 股東應佔溢利 / 市賬率), sum_vs_single 3 (two companies' total against a third), relative_change 3 (降幅 / 增幅
compared in words), unit_threshold 3 (站上万亿 / 突破八千亿), control 6. Verdicts: supported 16, contradicted 6,
partially_supported 2. Every verdict is computed by the predicates in `build_heldout_r14.py` from fresh tool output.

## Verify

```bash
cd <FinSight checkout>
FINSIGHT_REPO=$PWD .venv/bin/python /Volumes/Remi/finsight-review/heldout_r14/verify_heldout_r14.py   # -> OK
```

At `64cfbeb`: **OK** (50 conversations, 62 turns, 57 facts; 24 claims). Mutation test (one stored fact value and one
verdict changed in a temporary copy): 3 failure messages (stored vs recomputed, scorer mismatch, verdict differs).
Info lines: the price tool maps 中芯北方 / 比亚迪锂电池 / 平安理财 / "Ping An Technology" to 中芯国际 / 比亚迪 / 中国平安
(none is a `.HK` security).

## First run at `64cfbeb` (pre-fix, no LLM, ext + corrections)

```bash
QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1 python -m evaluation.agent_eval.runner --mode auto --no-replay \
  --tasks /Volumes/Remi/finsight-review/heldout_r14/chat_r14_heldout.jsonl --out first_run/chat_r14-auto-64cfbeb.json
FINSIGHT_REPO=$PWD python /Volumes/Remi/finsight-review/heldout_r14/score_claims_r14.py first_run/claims_r14-64cfbeb.json
python -m evaluation.agent_eval.slice_overlap /Volumes/Remi/finsight-review/heldout_r14 --content \
  --out first_run/slice_overlap-heldout_r14-at-64cfbeb.json
```

- **Chat: 8/50 = 0.160 [0.06, 0.28]** (task bootstrap, 2000 resamples, seed 20261014); turn success 0.323. Controls
  7/7; hk_terms 1/4 (the control); every other category 0.
- **Claims: 13/24 = 0.542 [0.33, 0.75].** Controls 6/6; exclusion_claim 1/4, hk_terms_claim 0/5, sum_vs_single 2/3,
  relative_change 2/3, unit_threshold 2/3.
- `slice_overlap --content` at `64cfbeb`: exact 0, near 2 (r14h_fix_en_01, r14c23), content 7, content signature 26.
- The runner's `tasks_file` is this folder.

**Draft run.** A run of an earlier draft (same tasks except two turns) gave the same chat 8/50 and claims 13/24. Before
the recorded run, `r14h_comma_zh_01` (an exact copy of a UI probe string) and `r14h_sale_en_02` (0.85 against a probe)
were reworded and the slice rebuilt; the recorded files above are the only kept run.

## Before the after-fix run

Run `slice_overlap --content` over the round-18 diff, report the exclusion tiers, record the runner's `tasks_file`
path, and do not copy any slice sentence into tests. The scorer counts `wrong_values` / `extra_forbidden` as failures
even when the right number is also present.

## Committed to the repository (after the round-18 fixes)

Copied unchanged (same hashes) after the round-18 fixes were measured on it once, at `f20c6c3` with
`QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`, reading this folder from the reviewer's directory. The engineers state
they did not open it; the record cannot show that. `python -m evaluation.agent_eval.slice_overlap --content` before the
run (`slice_overlap-heldout_r14-at-f20c6c3.json`):

| Exclusion | chat | claims |
|---|---|---|
| none | 8/50 → 37/50 | 13/24 → 22/24 |
| text near-copies (one exact copy: an 8-character control turn) | 7/48 → 35/48 | 12/22 → 20/22 |
| + content overlaps | 4/40 → 28/40 | 12/22 → 20/22 |
| + content signatures (upper bound) | 2/22 → 17/22 | 10/18 → 17/18 |

Classes still failing after the fixes (by_category in `chat_heldout_r14-auto-nollm-after-fix.json`): derived figures
0/2, portfolio weights 0/2, short positions 0/2, 52-week range 1/3, sale follow-ups 1/3, closed positions 3/5, comma
counts 2/3. This was the last review round; the slice is after exposure from here on.

