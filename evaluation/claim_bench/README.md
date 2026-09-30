# Claim-check benchmark (v1)

Labelled market claims for `POST /agent/claim-check` (`query_intelligence/agent/claim_check.py`), checked
against the offline snapshot (`data/structured_data.json`: prices as of 2026-04-22, reports as of
2025-12-31).

| File | Claims | Use |
| --- | --- | --- |
| `claims_v1.jsonl` | 268 (221 zh, 47 en) | dev set: the checker was developed against it (rows noted `round4` / `round5` / `round8` / `round9` / `round10` were added with later rule rounds) |
| `claims_v1_holdout.jsonl` | 47 (38 zh, 9 en) | held out: written together with the dev set, before the fixes, and run **once** at the end |

Held-out file sha256 (recorded when it was written, before any checker change):

```
a48aa59412a06f81c603604b587dec82948ae5914fdf176bdcdb7e42c0028bb8  claims_v1_holdout.jsonl
```

## Row format

```json
{"id": "d034", "lang": "zh", "category": "comparator", "claim": "贵州茅台ROE超过30%",
 "expected_verdict": "supported",
 "expected_checks": [{"metric": "roe", "comparator": "gt", "status": "supported"}],
 "basis": "M roe 33% > 30"}
```

* `expected_checks` has one entry per number in the claim, in order (a range "20到30倍" is one number; a
  number-less move such as "昨天下跌了" is one check against 0). `metric: null` means the metric is not
  scored for that check.
* Expectations were derived by hand from the tool outputs (`get_price_history`, `get_fundamentals` on the
  offline service), **not** by running the checker. `basis` names the value used.
* Categories: exact, false, approx, comparator, range, negation, direction, sign, growth, multi, no_data,
  unknown_target, opinion, wrong_unit, index_etf, chinese_numeral, forecast, period, multi_day.

## Labelling policy (written before the fixes)

* Tolerance: `eq` matches within half a unit of the last written digit or 2% (5% for `approx`).
  Bounds (`gt/ge/lt/le`, `range`) are literal: "ROE不超过15%" is contradicted by 15.2%.
* Negation turns `eq` into `ne` and flips a bound ("没有超过30%" → `le` 30; "并没有跌" → `ge` 0).
* Unverifiable: no target or no data, a metric the sources do not carry (YoY growth offline, index P/E),
  a unit that does not fit the metric (P/E in %, ROE in 倍, a price in 亿元, an amount with no unit), a
  forecast, a period other than the data's, or a multi-day move (only the daily change is available).
* Opinions without numbers have no checks and are `unverifiable`.

Round-5 rows (`note: round5`, d174-d204) were written after the independent round-4 held-out slice
(`evaluation/heldout_r4/claims_moves_heldout.jsonl`) was run once and its failure classes exposed. They are new
phrasings of those classes (bounded moves, qualitative move words, explicit dates, multiples and relations,
sector moves, several targets sharing one claim); `tests/test_agent_eval.py` checks that none copies or
near-copies a held-out claim. Labels follow this file's comparator convention: a bound after a move word is
about the size of the move ("跌超1%" is `gt` 1 with direction down), which the held-out slice writes as a bound
on the signed change (`le` −1), so its comparator accuracy measures that difference in convention too.

Round-8 rows (`note: round8`, d205-d224) were written after the round-4 review (D2-D4), as new phrasings of its
failure classes: a relation in its own clause next to a number in another clause, an industry average as the
subject of a number ("而行业平均…倍", "所属行业的平均水平约…倍", "the sector average is …"), turnover (成交额), and
ratios with a bound where the verb stands ("不足…的一半", "超过…的三倍", "至少是…的五倍", "more than twice").
`tests/test_agent_eval.py` checks that none copies or near-copies a claim of `claims_v1_holdout.jsonl` or a text of
the round-4 held-out slices.

Round-9 rows (`note: round9`, d225-d245) were written after the round-5 review (E1, E2, E8, E9) and after the
independent round-5 held-out slice (`evaluation/heldout_r5/`) was run once at `f01097a` and its failures read. They are
new phrasings of those classes: an industry average stated on the other side of a bound ("低于7倍的白酒行业平均水平",
"under the baijiu industry average of 30x": the relation and the stated average are two checks), a relation after
the number of its own clause, shares and fractions as multiples ("的三分之二", "的四成", "的九成"), "比…的1.2倍还多",
"七倍有余", "more than triple", 净赚, a derived net margin, a fund's daily change computed from its last two closes,
metrics the sources cannot give (市销率), macro series against each other, a short name after the full name, and
"更活跃" / "表现强于". `tests/test_agent_eval.py` checks that none copies or near-copies a claim of the held-out set,
the round-4 and round-5 slices or the round-5 reviewer's probes. d102 ("沪深300ETF今天涨了1%") was relabelled in round 9:
the fund's change is now computed from its closes (+0.73%, within half a unit of "1%"), so it is supported, not
unverifiable.

Round-10 rows (`note: round10`, d246-d268) were written after the round-6 review (F1, F2, F7), as new phrasings of its
failure classes: a value stated for the compared side in the metric's own unit ("比白酒行业平均的8倍低", "低于保险行业
平均的11.8倍", "比茅台的30倍低", "低于茅台的33%": the stated value is its own check next to the comparison), multiples
kept by an explicit ratio cue ("是…的1.3倍左右", "只有茅台的85%"), stated differences ("低3.6个百分点", "多赚了四百四十多
亿", "低4.3倍" of P/B, "相差约18个百分点", "低了大约两成" relative to the average, "高出600亿以上", a difference in the
wrong direction, "差了5倍多", an ambiguous "高出一倍多" of an amount, "跌幅比茅台大0.36个百分点", a later clause that belongs to the comparison's subject, not its compared side), and numerals with
多/余/出头/左右 ("一千二百多亿", "三百八十亿出头", "七成多", "三成左右", "一千余亿"). They were labelled by hand from
`data/structured_data.json` before the checker was run on them. `tests/test_agent_eval.py` checks that none copies or
near-copies a claim of the held-out set, the round-4 and round-5 slices or the round-5 and round-6 reviewer probes.
Labelling conventions added in round 10: "N多 / N余" is N < x < N + the step of N's last significant digit ("一千余亿" is
1000-2000亿, "四百四十多亿" 440-450亿); "N出头" is the lower half of that step; "约 / 左右" allows 5% or half that step.

Known judgement calls: "五粮液市盈率24.6倍，比茅台低" is two checks since round 8, the 24.6 (bound to 五粮液:
contradicted) and the relation of the second clause (五粮液 below 茅台: supported), so it is partially supported
(d082 relabelled; before round 8 the relation was dropped because the sentence stated a number). The holdout's
industry-average PE ("行业平均11.8倍") is labelled supported from the 保险 industry snapshot; the checker binds
such a number to the target's industry snapshot since round 8.

## Run

```bash
python -m evaluation.claim_bench.run --set dev       # -> evaluation/results/claim_bench-dev.json
python -m evaluation.claim_bench.run --set holdout   # once, at the end
```
