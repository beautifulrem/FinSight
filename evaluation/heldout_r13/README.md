# Round-13 independent held-out slice (reviewer-written, unopened by the engineers)

Written by the round-13 reviewer at `aa73005` (2026-10-11), **before any round-17 fix**, from the bug classes the
round-13 review found plus controls. The phrasings here appear nowhere in `round13.md` (its repros are worded
differently) and not copied from the reviewer's probe scripts under `round13/probes/`, `round13/ui/`, `round13/sec/` (difflib against
every string literal there: the closest is 0.90 for the control claim `r13c20`, a growth-rate figure of the same form as
a probe; two contrast/same-company claims reach 0.74–0.77 against one probe claim of the same class; every chat turn
is ≤ 0.82).
**The engineers must not open this folder** (do not read the jsonl files or the build script; do not point any tool
other than the runner / scorer / verify script / `slice_overlap` at it before the measured run). Run it once after the
fixes, after a leak check: `python -m evaluation.agent_eval.slice_overlap <this folder> --content` over the round-17
code, tests, task files and docs. Report the overlap-excluded numbers too. Record the runner's `tasks_file` path in the
result file and state, as the record shows it, where the slice was read from.

- **It runs on the SHIPPED offline default: snapshot extension + corrections layer.** Run the chat slice with
  `QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`. Facts were computed with
  `build_offline_service(snapshot_ext=True, snapshot_fix=True)`; the claim scorer and the verifier build the same
  service themselves.
- If the data layers change, `verify_heldout_r13.py` fails on the affected facts; re-derive with the build script and
  say so next to the result.

## Files and hashes

| file | sha256 |
|---|---|
| `chat_r13_heldout.jsonl` | `1b5496f33e551c842a7001c89e658070cddfbd895072308c90def75b0d3d37af` |
| `claims_r13_heldout.jsonl` | `e18e5f20cb7133c45121b178dc069a71d64d0a696bbafe0d36724a5358f39967` |
| `build_heldout_r13.py` | `8df9bacc2a75103e9977c1f23cf3a7ce9d3d40438beb8eb6084b5abdd53f823b` |
| `verify_heldout_r13.py` | `eb388d77e1f0643dc1f44d49160b55dcc62f9628c22d53aecd68f9703f54def9` |
| `score_claims_r13.py` | `395ad6ff747e1ad7857709c1e6dbc70037c45405d241a9613afe424391599402` |
| `first_run/chat_r13-auto-aa73005.json` (pre-fix run) | `bab17e3314c22b39a66b90a86e1361b921e3a480fc7f73506ff0bd14ae073102` |
| `first_run/claims_r13-aa73005.json` (pre-fix run) | `15428cf9738a0903f5505754ac115bc2c0f56b46fde34069446cc59171e2a3a6` |
| `first_run/slice_overlap-heldout_r13-at-aa73005.json` | `164e55f03fa5450d3f334bd4b519afb23b33ad85afe8e05d86026f487408838d` |
| `first_run/first_run_digest.txt` | `c244e9a6201a5911f2c730874aecc8181c495402a30539daf16ecf3325e0678f` |

## Contents

`chat_r13_heldout.jsonl`: **46 conversations / 58 turns / 44 required facts** (35 zh, 11 en), same task schema as
`evaluation/agent_eval/tasks/*.jsonl` (`required_facts` with a `derive` spec, `required_tools`, `any_of_tools`,
`required_entity/entities`, `behavior`, `must_hedge`, `must_state_missing`, `required_limitations`,
`forbidden_patterns`, `language`).

**New in this slice: `wrong_values`.** Eight turns carry the derive spec of the number a known wrong reading produces
(the paper P&L of the whole position when part of it was sold; the total of every named company when one was
excluded). The build script turns each into a regex appended to `forbidden_patterns` after the four trading
patterns, so an answer that states the wrong number fails `no_forbidden_content` even if it also states the right
one. The verify script rebuilds the patterns with its own code, checks the gold answer does not match them and that a
gold answer with the wrong number appended fails `score_turn`.

| category | conversations | what it targets |
|---|---|---|
| `partial_sell` | 4 | part of a position sold at another price; the P&L of the shares still held (one in a two-turn session) |
| `exclusion` | 4 | a total over a named group with one member excluded (不算 / 去掉 / 排除 / "excluding") |
| `lookalike_entity` | 4 | a Hong Kong-only company whose name starts like a covered A-share company: refuse as out of coverage, never the covered company's figures |
| `breakeven_rise` | 4 | the percentage rise from the last close back to a stated cost, or to the 52-week closing high |
| `lots_in_shou` | 3 | counts in 手 (1 手 = 100 shares) in a P&L or a value |
| `statement_carry_total` | 3 | two positions stated in two turns without a question, then a bare total question |
| `definition_in_session` | 3 | how a metric is calculated, asked right after a figure of that metric for a company (`required_tools: ["explain_concept"]`) |
| `book_value_followup` | 2 | "is it below book value" about the previous turn's company: the P/B must be restated |
| `traditional_chinese` | 3 | metric words in traditional characters |
| `leading_adverb` | 2 | English totals that open with an adverb other than the round-12 ones |
| `position_sizing` | 2 | how many shares to buy with a budget: `must_hedge` |
| `fx_missing` | 2 | a value converted to US dollars (no exchange rate in the data): `must_state_missing` |
| `rank_spread` | 2 | highest minus lowest within a named group of three |
| `control` | 8 | a sum, a single-lot P&L, a holding, a ratio, a standalone definition, a price forecast, an A+H Hong Kong line, two lots in the round-15 form |

`claims_r13_heldout.jsonl`: **22 claims** — weak direction words (微增 / 基本持平 / 大增 / 小幅下滑 / 微降), two metrics of
the same company compared (净利润增速 vs 营收增速, 扣非 vs 归母, one growth rate as a multiple of another), a definition
stated next to a value (three false definitions, one true), a contrast clause about a second company (则相反 /
也是如此), controls. Verdicts come from per-part predicates in `build_heldout_r13.py` (all true → supported, all
false → contradicted, mixed → partially_supported); a definition part is a constant (true / false) with its reason
in the part text. For the three claims with a false definition next to a true value, `contradicted` is also
accepted. Verdicts: 12 supported, 6 contradicted, 4 partially supported.

## Verification

`verify_heldout_r13.py` (adapted from the round-12 verifier: same checks, new categories, the `rise_to_cost` op, and
the wrong-value checks above) re-runs the tools, re-derives every fact with its own implementation of every op,
checks operand discrimination, refusals with `out_of_coverage`, a synthetic gold response per turn through
`metrics.score_turn`, and re-runs the claim predicates. At `aa73005`: **OK**, with info lines: the price tool maps
`美的置业` and `Midea Real Estate Holding` to `price_000333.SZ`, `中信国际电讯` to `price_688981.SH`, and
`招商银行H股` to `price_600036.SH` (no `.HK` security is returned; the mapping is the behaviour under test in
`lookalike_entity`). Mutation test (one chat value and one claim verdict changed in a copy): **6 failure messages**
(five for the value, one for the verdict), as expected.

```
FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python verify_heldout_r13.py      # prints OK
```

## Overlap with the repository at `aa73005` (before any fix)

`python -m evaluation.agent_eval.slice_overlap <this folder> --content` →
`first_run/slice_overlap-heldout_r13-at-aa73005.json`: **exact 0**; text near 3 (`r13h_book_zh_01` turn 1, an
11-character P/B question; `r13c06`, a 17-character growth comparison close to a round-16 claim test; `r13c17`, a
control figure close to a round-14 grammar test); content 6; content signature 20 (a shared company with a holding /
P&L / sum operation, mostly). One claim was an exact copy of an existing round-15 claim test sentence when first
written; it was reworded **before** the measured claim run below (the chat file was not affected).

## First run at `aa73005` (pre-fix, no LLM, ext + corrections)

```
cd <FinSight checkout at aa73005>
QI_USE_LIVE_MARKET=false QI_USE_LIVE_MACRO=false QI_USE_LIVE_NEWS=false QI_USE_LIVE_ANNOUNCEMENT=false \
QI_AGENT_TRACE_DIR=off QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1 \
  python -m evaluation.agent_eval.runner --mode auto --no-replay \
  --tasks <this folder>/chat_r13_heldout.jsonl --out first_run/chat_r13-auto-aa73005.json
FINSIGHT_REPO=<checkout> python score_claims_r13.py first_run/claims_r13-aa73005.json
```

- **Chat: 19/46 = 0.413 [0.26, 0.57]**, turn success 0.534, behaviour accuracy 0.914, fact recall 0.591; controls 8/8.
  Per category: partial_sell 0/4, exclusion 1/4, lookalike_entity 0/4, breakeven_rise 0/4, lots_in_shou 1/3,
  statement_carry_total 0/3, definition_in_session 1/3, book_value_followup 1/2, traditional_chinese 1/3,
  leading_adverb 2/2, position_sizing 2/2, fx_missing 0/2, rank_spread 2/2, control 8/8.
  Wrong numbers stated (caught by `wrong_values`): 5 turns (the whole position's P&L when part was sold; the total of
  all named companies when one was excluded). `position_sizing` passes on the scorer's weak hedge markers; read it with
  that in mind.
- **Claims: 14/22 = 0.636 [0.41, 0.82]**. Per category: weak_direction 2/5, same_company_metrics 3/5,
  definition_clause 2/4 (both errors are **false "supported"** verdicts for a false definition next to a true value),
  contrast_clause 1/2, control 6/6. One more false "supported": a weak-direction clause (微降) next to a true figure.
- Outputs: `first_run/chat_r13-auto-aa73005.json`, `first_run/claims_r13-aa73005.json`, logs next to them, and
  `first_run/first_run_digest.txt` (per-turn outcome and answer excerpt).

Note on the scorer: `metrics.py` still accepts weak hedge / missing markers (可能 / 无法 / "cannot" / 没有 / "not ").
`position_sizing` and `fx_missing` (`must_state_missing`) should be read with that in mind after the fixes.

## Committed to the repository (after the round-17 fixes)

Copied unchanged (same hashes) after the round-17 fixes were measured on it once, at `71cb928` with
`QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`, reading this folder from the reviewer's directory. The engineers state
they did not open it; the record cannot show that. `python -m evaluation.agent_eval.slice_overlap --content` before the
run (`slice_overlap-heldout_r13-at-71cb928.json`):

| Exclusion | chat | claims |
|---|---|---|
| none | 19/46 → 45/46 | 14/22 → 22/22 |
| text near-copies (2 exact claim copies in round-17 claim tests) | 18/44 → 43/44 | 12/18 → 18/18 |
| + content overlaps | 15/39 → 38/39 | 11/15 → 15/15 |
| + content signatures (upper bound) | 11/27 → 26/27 | 9/12 → 12/12 |

From round 18 this slice is **after exposure**.


**The two exact copies.** Claims r13c02 and r13c06 (12 and 17 characters) equal sentences in
`tests/test_agent_round17_claim_parts.py`, committed in `2f7fbde` at 2026-10-11 14:38 (+0800). The slice's claims file
was created in the reviewer's folder at 13:16 the same day. Both are short template sentences of classes the round-13
review described in words (a weak direction word on revenue; a same-company growth comparison with 其), so the record
cannot tell coincidence from exposure; both claims are excluded from the "text" tier on.
