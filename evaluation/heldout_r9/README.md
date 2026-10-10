# FinSight held-out slice r9 (`chat_r9_heldout.jsonl`, `claims_r9_heldout.jsonl`)

An independent held-out slice for round 13 (the round after the round-9 review). It was written by the round-9
reviewer from the round-9 bug classes **before any round-13 fix**, so the fixes can be measured on phrasings nobody
tuned against. None of the phrasings appears in `round9.md`, `round9.json`, the round-9 probe scripts
(`round9/probes/*.py`), any earlier review, or any earlier slice; the review publishes the bug classes and its own
probe phrasings only.

- Date written: 2026-10-01
- Repo and commit the facts were taken from: `beautifulrem/FinSight` at `cbcf3b2` (clean worktree)
- Data: the offline snapshot **with the extension** (`build_offline_service(snapshot_ext=True)`: v1 snapshot for
  茅台/五粮液/平安 etc., `data/snapshot/structured_data_ext.json` for 宁德时代, 比亚迪, 招商银行, 工商银行, 平安银行, 美的集团,
  长江电力, 紫金矿业, 中信证券, 中芯国际 and the ETFs; market as of 2026-09-30, fundamentals FY2025, valuation 2026-09-30).
  **Run the chat slice with `QI_EVAL_SNAPSHOT_EXT=1`**: the runner is pinned to v1 otherwise and does not know these
  names.
- Nothing was committed to the repo. Nothing was written outside `/Volumes/Remi/finsight-review/heldout_r9/`.

## Files

| file | sha256 |
|---|---|
| `chat_r9_heldout.jsonl` | `73c925f85e14d507986b1bf9930c3d2a79e6f80517afb8fce248392aa7744a1e` |
| `claims_r9_heldout.jsonl` | `e289d7f68c286cc26c0466b15e2b4e70e2a156e3945f7e8c9eb2c7d0f4abea93` |
| `build_heldout_r9.py` | `9af581c127f0fb91b32dac09f80761cad68fa2c8b1e740494f278d11fbfacbf1` |
| `verify_heldout_r9.py` | `774d9a8677fd4ed3c5b68f54f0ce5b7da6e1fccb10a251c56ba965faf7e4da49` |
| `score_claims_r9.py` | `57ffc6fdddc94879ab9bd2932aeb4a3c06cc332cb1e93d6accc00be3a74f0270` |
| `first_run/chat_r9-auto-cbcf3b2.json` (pre-fix run) | `b096452410530e0fc4e7aa7b4e6b8461c0bd505d3308fdfb00c6f370e4bf7f05` |
| `first_run/claims_r9-cbcf3b2.json` (pre-fix run) | `4e7f8632f0ef9626eca4168a6051e9f584a5a3c52c8d6b4d0fd51e5fb8d047f7` |

`build_heldout_r9.py` generates both JSONL files. Every required chat number is computed from calls to the offline
tools through a `derive` spec (`value`, `sub`, `sum`, `div`, `pct` = a as a share of b, `rel` = relative difference,
`mul`, `lin` = several holdings, `spread` = highest minus lowest); dotted fields reach nested payloads
(`price_300750.SZ:range_52w.high`, `price_601899.SH:max_drawdown_1y.max_drawdown_pct`, `price_600036.SH:year_start.close`).
No number was typed in by hand. Every claim's verdict comes from predicates over tool values (one per part; all true =
supported, all false = contradicted, mixed = partially_supported). `verify_heldout_r9.py` reruns the tools with its own
implementation of every op and checks: schema, recomputed values at the stated precision, first-operand evidence ids,
scorer matching of the value and its natural rendering, **operand discrimination** (no derived value is matched by an
operand at the matcher's power-of-ten scaling; this is why the English two-holding task uses 700 shares, since
500 x 11.57 + 300 x 8.28 = 8,269 collides with 8.28 x 1000), `verify_absent` fields, `verify_no_industry` (the target's
`get_fundamentals` returns no `industry_*` evidence), refusals with `out_of_coverage`, no `.HK` security in the
snapshot, a synthetic gold response passing `metrics.score_turn`, and the claim predicates rerun from the build script.
A mutation test (one chat value and one claim verdict edited by hand) made the verify script fail with 3 failures; the
files were then regenerated.

**One edit after the first build.** `r9h_hold_zh_02` turn 2 was rephrased (to `仓里是这只的900股，外加宁德时代两手，…`,
900 x 83.31 + 200 x 291.11 = 133,201) because the original wording had also been typed into my real-Chrome UI test and
appears in a round-9 screenshot. The files were rebuilt, `verify` → OK again, and the chat first run was repeated at
`cbcf3b2` (same result, 0.346). Generic ellipsis turns (`and CMB's?`, `紫金矿业的ROE`) also occur in my probe scripts;
they carry no bug-class information. No other slice phrasing occurs in `round9.md`, `round9.json`, the notes or the
probes (checked by substring search).

## Protocol

1. Bug classes come from my own round-9 probing (`round9.md`, bugs I1..): one company **as a share** of another
   (相当于…的百分之多少 / 几成 / 占…的比例 / "what share of", "as a percentage of") computed as a relative difference;
   comparatives worded with verbs that the comparison parser still misses (领先/落后/跑输, "lead by", "outpace",
   一股能换几股, 抵得上几个); **sums** (一共/合计/再加上/combined/put together); **highest minus lowest of three**;
   derived price metrics on the snapshot extension (below the 52-week high, rebound from the 52-week low, yuan below
   the high, drawdown, YTD on the stated basis); **threshold and point price forecasts** without the hedge (跌破/突破/站上,
   break above, "your call", injection wrappers); a target against an industry the data does not have; **several
   holdings in one message**; **yes/no questions that need the ratio** ("more than twice"); controls (single facts,
   a plain gap, HK lines of CATL/BYD, a v1 name).
   Claims: X倍半 (三倍半/两倍半/一倍半), "N points higher/above" in English, 高出将近一倍/一倍多, 六成多一点 (the 一点 that
   the checker reads as a second number), compound clauses with 但/却, superlatives among three, "less than a third /
   under a quarter of", 52-week and drawdown claims (the "52" in 52周 must not be read as a value), negation, levels
   plus multiples, controls.
2. I did not reuse any phrasing from my probes (`round9/probes/*.py`) or from `round9.md`.
3. Scoring contract: `evaluation/agent_eval/metrics.py` (`score_turn`), the same expect keys as `heldout_r7`/`r8`.

## Schema

Chat: `{"id", "category", "language", "turns": [{"query", "expect", "note", "verify_absent"?, "verify_no_industry"?}]}`,
`expect` keys as read by `metrics.score_turn` (`behavior`, `required_facts` (+ `derive`), `required_tools`,
`any_of_tools`, `forbidden_patterns` = the four TRADING patterns, `must_hedge`, `must_state_missing`,
`required_limitations`, `language`, `required_entity`, `required_entities`).

Claims: `{"id", "lang", "category", "claim", "expected_verdict", "also_acceptable", "parts": [{"part", "true"}]}`.
Score verdict accuracy with `score_claims_r9.py` (a verdict in `also_acceptable` counts as correct: the three 52-week /
drawdown claims may be left `unverifiable`, but never the opposite verdict).

## Counts

Chat: 52 conversations, 79 turns, 72 required facts; 33 zh, 19 en; 37 single-turn, 4 two-turn, 10 three-turn, 1
four-turn.

| category | conversations | what it targets |
|---|---|---|
| `share_ratio` | 7 | a/b as a percent or in tenths (not the relative difference), zh/en, one message and follow-ups |
| `verb_comparative` | 7 | 领先/跑输/落后, "lead by", "outpace", 一股能换几股, 抵得上几个 |
| `sum` | 5 | two-company sums, one message and follow-ups (再加上, put together) |
| `spread` | 3 | highest minus lowest among three, after three turns or in one message |
| `price_derived` | 7 | % below the 52-week high, % above the low, yuan below the high, 1y drawdown, YTD on the first-close basis, 52-week range |
| `forecast_hedge` | 7 | point and threshold forecasts zh/en, two with injection wrappers: `must_hedge` |
| `industry_missing` | 3 | a target vs an industry average the snapshot does not have: state it missing, give the target's value |
| `multi_holding` | 3 | two positions in one message (lots and shares, zh/en) |
| `yes_no_ratio` | 3 | "more than twice / four times / at least 1.2 times": the ratio must be stated |
| `control` | 7 | single facts (ext and v1), a plain gap, a which-lower-by-how-much, HK lines of CATL and BYD |

Claims: 25 (18 supported, 7 contradicted); categories `half_multiple` 4, `points_difference` 2, `times_more` 2,
`loose_fraction` 1, `compound` 2, `superlative` 2, `fraction_of` 2, `range_52w` 3, `negation` 2,
`level_and_multiple` 3, `control` 2.

## Verify and run

```bash
FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python verify_heldout_r9.py      # prints OK
cd /path/to/FinSight && QI_EVAL_SNAPSHOT_EXT=1 python -m evaluation.agent_eval.runner --mode auto --no-replay \
    --tasks /path/to/heldout_r9/chat_r9_heldout.jsonl --out <out.json>
FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python score_claims_r9.py <out.json>
```

At `cbcf3b2` the verify output is `OK` (52 conversations, 79 turns, 72 facts, 25 claims; warnings: the
长江电力 market cap also matches the circulating market cap field, and `get_price_history('宁德时代港股')` etc. return
the A-share evidence, which is why the HK lines must be refused by the agent, not by the tool). Log:
`first_run/verify-cbcf3b2.log`.

## First run (before any round-13 fix), `cbcf3b2`, deterministic path, no LLM, `QI_EVAL_SNAPSHOT_EXT=1`

| set | result |
|---|---|
| chat, task success | **0.346 [0.231, 0.481]** (18/52), turn 0.570, behaviour 0.975 |
| chat by category | share_ratio 0/7, verb_comparative 2/7, sum 0/5, spread 0/3, price_derived 3/7, forecast_hedge 5/7, industry_missing 2/3, multi_holding 0/3, yes_no_ratio 0/3, control 6/7 |
| chat by language | zh 0.333, en 0.368 |
| claims, verdict accuracy | **0.600 [0.40, 0.80]** (15/25); compound 2/2, control 2/2, negation 2/2, level_and_multiple 3/3, range_52w 2/3, half_multiple 2/4, points_difference 1/2, times_more 1/2, loose_fraction 0/1, superlative 0/2, fraction_of 0/2 |

Files: `first_run/chat_r9-auto-cbcf3b2.json`, `first_run/claims_r9-cbcf3b2.json` (+ logs). Run the same commands after
the fixes; report both, per category, and label any run after the engineers have read this folder "after exposure".
**Round 13 had already started when this slice was finished**: branch `r13-frame` (worktree `FinSight-eval`, head
`6a069d8`, commits from 19:04 on 2026-10-01) fixes the classes the round-8 slice still failed (sums, comparative verbs,
stated multiples), which overlap several categories here. A baseline leak check at `6a069d8`
(`round9/probes/near_r9_vs_r13.py`, character 3-gram Jaccard against every string literal added on the branch): **0
slice phrasings >= 0.8**, highest 0.41. So a later improvement on those categories can be credited as out of sample,
provided the same check stays at 0 at the commit that is scored.
**Before the after-fix run, check that no slice phrasing (or a near copy, character 3-gram Jaccard >= 0.8) appears in
the engineers' tests, dev tasks or router labels**: the round-9 review found exact and near copies of four
unpublished round-8 slice phrasings in round-12 tests and dev tasks.

## Ambiguities and judgement calls

1. **Share vs relative difference.** "B 的净利润只占 A 的百分之几" is B/A x 100 (45.18%), not (A-B)/A (54.82%) or
   (B-A)/A. "大约是…的几成" is the same share (87.17% = 8.7成; the matcher accepts 8.72 for 87.17 through its power-of-ten
   scaling).
2. **Sign.** The matcher ignores sign: 跑输 2.22 points and "-2.57%" YTD pass with either sign.
3. **YTD basis.** The task states the basis explicitly (今年第一个交易日的收盘价) because the tool's `year_start` is the
   first close of the year, not the prior year-end close the usual 年初至今 convention uses (round-9 bug); the slice does
   not test the convention.
4. **52-week low rebound** is (close - low) / low = 6.53%; "below the 52-week high" is (high - close) / high.
5. **Forecast turns** use `must_hedge`, which `metrics._HEDGE_MARKERS` satisfies with weak markers (可能, 无法, 不足以). A
   threshold forecast answered with the latest close and no marker fails; a marker anywhere passes. This is the
   project's scorer; it under-counts missing hedges rather than over-counting them.
6. **Industry missing** passes on `_MISSING_MARKERS` anywhere in the answer; the same weakness applies (at `cbcf3b2`
   `r9h_ind_zh_01` passes on a judgment prefix without saying the industry average is missing).
7. **Holdings**: 3手 = 300 shares; no hedge is required; stating only one position fails.
8. **Spread**: the answer must state max - min of the three; the difference of any other pair fails.
9. **HK lines**: `refuse` + `out_of_coverage`, as in `heldout_r7/r8`; an answer with the A-share close fails.
10. **Claims.** `near()` uses 5% relative for 约/左右/大约/差不多/roughly/about; X倍半 = X.5 within 5%; 将近一倍 more = ratio - 1
    in [0.8, 1.0); 一倍多 more = ratio - 1 in [1, 2); 六成多一点 = [0.6, 0.7); "less than a third" < 1/3.

## Committed to the repository (after the round-13 fixes)

The folder was copied into the repository unchanged (same hashes) after the round-13 fixes had been measured on it
once, at `da9f6af` with `QI_EVAL_SNAPSHOT_EXT=1`: chat 18/52 → 41/52 (`evaluation/results/chat_heldout_r9-auto-nollm-*.json`),
claims 15/25 → 25/25 (`claims_heldout_r9-*.json`). The round-13 engineers did not open the folder, but before that run
`python -m evaluation.agent_eval.slice_overlap` found 4 chat tasks and 5 claims of this slice that are exact or near
copies of own wording (`slice_overlap-heldout_r9-at-da9f6af.json`): short follow-ups and claim forms that the round-9
review published as repros, and three English claim tests written from the review's class descriptions. Without those
items: chat 16/48 → 37/48, claims 15/20 → 20/20. From round 14 this slice is **after exposure**.
