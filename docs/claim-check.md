# Claim check (fact-check a market claim)

`POST /agent/claim-check` takes a sentence from a broker note, the news or social media, for example
"听说茅台市盈率只有15倍" or "Moutai's ROE is above 30%". It checks every number in the sentence against
market and fundamental data and returns a verdict per number. It uses no LLM: the classical NLU finds
the targets, a rule-based reader handles the numbers, and the `get_price_history`,
`get_fundamentals` and `get_macro_indicators` tools supply the evidence. The code is in `query_intelligence/agent/claim_check.py`
and the UI is the "核查 / Fact-check" tab (`frontend/src/components/ClaimCheck.tsx`).

```bash
curl -s -X POST localhost:8000/agent/claim-check -H 'Content-Type: application/json' \
  -d '{"claim": "贵州茅台ROE超过30%，市盈率不是15倍"}'
```

Each check contains:

- `target`, `metric`, `claimed` and `claimed_high` (the upper bound of a range); `claimed` is signed by
  its move word ("跌超1%" is −1) and is `null` for a relation;
- `claimed_unit`, `comparator`, `negated` and `direction` (`up` / `down` when a move word states one);
- for a relation: `reference` (the other side), `reference_value` and `reference_evidence_id`;
- for a multiple ("市净率是五粮液的1.5倍"): `claimed` is the multiple and `ratio` is the target's value divided by
  the reference's;
- `actual`, `status`, and `reason` (why a check is unverifiable);
- `evidence_id`, `source`, `as_of` and `as_of_basis`;
- `note`.

The report also lists `unchecked`: the clauses that name a target or a metric but have no number, move or
comparison to check ("茅台市盈率24.6倍，ROE很高" → "ROE很高"), each with `text` and `reason: no_claim`. Every
clause of a claim therefore ends up as a check or as a "未核查 / Not checked" row in the UI; nothing is dropped
silently. The verdict is computed over `checks` only; `coverage` (`full` / `partial` / `none`, round 9) says whether
every part was checked, and the UI heads a supported claim with unchecked parts "Partly checked" (see Round 9).

The overall `verdict` is:

- `supported`: every check is supported;
- `contradicted`: no check is supported and at least one is contradicted;
- `partially_supported`: at least one check is supported and at least one is not;
- `unverifiable`: there are no checks, or all of them are unverifiable.

## Rules

### Numbers

- **What counts as a number.**
  - Arabic numerals, with full-width digits normalised.
  - Simple Chinese numerals: 十五倍 → 15倍, 二十四点六倍 → 24.6倍, 三成 → 30%, 百分之三十 → 30%, 一千七百亿 → 1700亿.
- **What is not a claimed number.** Dates, durations ("近20日"), indicator parameters (RSI(14)), tickers (600519, 600519.SH) and index names (沪深300, CSI 300).
- **Ranges.** "20到30倍", "20-30倍", "15倍至20倍" and "between 20 and 30" each become one `range` check.
- **Moves without a number.** A move word with no number in its clause ("昨天下跌了", "并没有跌", "did not fall") becomes a check of the daily change against 0.

### Metric

- **Unit-compatible match.** The metric is the nearest metric word in the number's clause whose unit fits the number:
  - `倍` or `x` → P/E or P/B;
  - `%` → ROE, margins or the daily change;
  - `元` → price;
  - `亿`, `万` or `billion` → amounts;
  - `点` → an index level.
- **"x times earnings".** "trades at 8.7 times earnings" and "below 10x earnings" are P/E claims.
- **Macro series.** CPI, PPI, manufacturing PMI, M2, the 10-year government bond yield, the 1- and
  5-year LPR and GDP are recognised by name ("CPI同比上涨0.8%", "PMI重回50以上", "10年期国债收益率低于2%",
  "China's CPI rose 0.8%"). They need no company and are checked against `get_macro_indicators`.
- **CJK acronyms (B3).** Acronyms are matched with letter look-arounds instead of `\b`, so "茅台PE为24.6倍" is recognised.
- **Growth.** A percentage whose nearest line item is revenue or net profit is YoY growth (`revenue_yoy`, `netprofit_yoy`). Examples: "营收同比增长16%" and "net profit fell 2% year on year". "净利率48.8%" is a margin, not growth.
- **Parallel clauses.** A clause with no metric word takes the previous number's metric when the unit is the same. "茅台市盈率24.6倍，五粮液20.9倍" checks two P/Es. "净利润850亿元，同比增长15%" becomes net-profit growth.
- **Unit mismatch.** When the only metric word nearby does not fit the unit, the check is unverifiable (`unit_mismatch`). Examples: "市盈率24.6%", "ROE 33倍", "股价53.61亿元".
- **Amounts without a unit.** Revenue or profit written without a unit ("营收1741") is unverifiable (`no_unit`).

### Target

- **Nearest target before the number.** Each number belongs to the nearest company, fund or index named before it.
- **Short names.** The Chinese short name is found when the NLU returns only the canonical name (茅台 → 贵州茅台).
- **分别 / respectively.** "茅台和五粮液的市盈率分别为24.6倍和20.9倍" assigns the targets in order.
- **Why binding matters.** "五粮液市盈率24.6倍" is checked against 五粮液, not against any target in the claim. Before this change, it passed with 茅台's value.

### Comparators (B4)

Comparator words are read in the text between the previous number and this one. The word closest to
the number wins. Words right after the number are read too (以上 / 以下 / 左右 / 多).

| `comparator` | Chinese | English | Supported when |
| --- | --- | --- | --- |
| `eq` | (none), 只有, 为 | is | within half a unit of the last written digit, or 2% |
| `approx` | 约, 大约, …左右 | about, around, roughly | within 5%, or half the step of the last significant digit when wider (round 10: "三成左右" is 25%-35%) |
| `approx` (from below) | 接近, 将近, 近 | nearly, almost | round 11: 0.9N ≤ actual ≤ N (+ half a unit of the last written digit): "将近900亿" accepts 823亿, "将近24倍" contradicts 24.6 |
| `gt` | 超过, 高于, 大于, 逾, 突破, 站上 | above, over, more than, exceeds | actual > claimed |
| `gt` with an upper bound | 30多倍, 八百多亿, 一千六百余亿, 七倍有余; …出头 | | round 10: N < actual < N + the step of N's last significant digit (800多亿: 800-900亿); 出头: the lower half of that step (三成出头: 30%-35%). Round 11: "N倍多" (多 after 倍) with N ≥ 10 uses min(step, 10% of N): 10倍多 is 10-11, 30倍多 30-33; 十多倍 stays 10-20 |
| `ge` | 至少, 不低于, …以上 | at least | actual ≥ claimed |
| `lt` | 低于, 小于, 不到, 不足, 跌破 | below, under, less than | actual < claimed |
| `le` | 至多, 不超过, …以下 | at most | actual ≤ claimed |
| `range` | X到Y之间, X-Y倍, X至Y | between X and Y | X ≤ actual ≤ Y |
| `ne` | negated `eq` | negated `eq` | outside the `eq` tolerance |

Bounds and ranges are literal:

- "ROE不超过15%" is contradicted by 15.2%;
- "PB小于1倍" is contradicted by 1.1.

The UI shows the comparator on each card, for example "声称 > 30%", "≠ 15 倍", "≈ 25 倍" or
"20 倍 – 30 倍". Screen readers hear it in words ("高于 30%").

### Negation (B18)

**Rule.** A negation word flips the comparator:

- Chinese: 不是, 并非, 并不是, 没有, 没, 未, and 不 before a comparator (不超过, 不低于, 不在);
- English: not, n't, never, no more/less than.

The flips are eq → ne, gt → le, ge → lt, lt → ge, le → gt. A negated range ("不在20到30倍之间") passes
when the value lies outside the range.

**Scope.** A negation applies from the previous number (or the start of the clause) up to this number.
So "茅台市盈率不是15倍而是24.6倍" gives `ne 15` (supported) and `eq 24.6` (supported), and the verdict is
supported.

**Moves.**

- "并没有跌" means the daily change is ≥ 0, so on a day when Moutai fell 0.18% it is contradicted.
- "中国平安今天没有下跌" is supported on a +0.73% day.

### Sign

A move word right before the number sets its sign: "下跌0.18%", "收跌0.53%" and "fell 0.18%" all mean
−0.18 and −0.53. A written sign ("-0.18%") is kept as is. For the daily change, the sign must match:
"上涨0.18%" on a −0.18% day is contradicted.

### Bounds on a move (C2)

A bound after a move word is about the **size of the move in the stated direction**, not about the
signed change. Direction words are 跌 / 下跌 / 跌幅 / 大跌 / fell / dropped / down and 涨 / 上涨 / 涨幅 /
rose / up (for growth and macro series also 增长 / 下降 / grew / declined).

| Claim | Meaning | Wuliangye −0.53% | Moutai −0.18% |
| --- | --- | --- | --- |
| 跌超1%, 跌了超过1%, fell more than 1% | change ≤ −1 | contradicted | contradicted |
| 跌超0.1% | change ≤ −0.1 | supported | supported |
| 跌不到1%, 跌幅不超过1%, fell less than 1% | −1 < change ≤ 0 | supported | supported |
| 涨超0.1%, 涨了不到1% | a rise | contradicted | contradicted |
| 跌了0.1%到0.3% | −0.3 ≤ change ≤ −0.1 | contradicted | supported |
| 没有跌超过1% | not a fall of more than 1% (a rise passes) | supported | supported |

A move the other way always contradicts a bound: "涨了不到1%" on a down day is not "a small rise".
Before this fix the checker negated the number and kept the comparator on the signed value, so
"五粮液昨天跌了超过1%" (−0.53%) came back supported ("> −1%") and "茅台昨天跌超0.1%" (−0.18%) came back
contradicted. The check now carries `direction`, and the UI writes the claimed side as "跌幅 > 1%" /
"Fall > 1%".

### Relations (C13)

A claim that compares two named targets, or a target with its industry, is one check with
`claimed: null`, the `reference`, and the comparison of the two values:

- "茅台的市盈率比五粮液高", "五粮液ROE低于茅台", "茅台市盈率没有五粮液高" (negated: ≤);
- "Moutai's P/E is higher than Wuliangye's", "Wuliangye has a higher ROE than Moutai";
- "茅台昨天跌得比五粮液多": a bigger fall is a lower daily change;
- "茅台市盈率高于行业平均": the target's industry snapshot from `get_fundamentals` (P/E, P/B, daily
  change);
- "高于市场平均": no source has a market average, so the check is unverifiable (`no_reference`).

A relation is checked whenever its own clause states no number, even when another clause of the sentence
does (round 8, D2): "茅台的市盈率比五粮液高，中国平安市盈率8.7倍" is two checks, and "五粮液市盈率24.6倍，比茅台低"
checks the 24.6 against 五粮液 (contradicted) and the relation of the second clause (supported), so it is partially
supported. A relation inside the clause of a number ("茅台市盈率24.6倍比五粮液的20.9倍高") is read as that clause's
numbers.

In the UI (round 12, H14) a relation row reads subject, relation, object under "claimed" ("白酒 < 贵州茅台",
"Kweichow Moutai > Wuliangye"), and the other side's value from the sources sits under "actual" ("贵州茅台 24.6 倍").
A multiple reads "贵州茅台 ≈ 1.5× 五粮液" against the actual ratio, with both values under it; a stated difference names
its two sides ("贵州茅台 − 五粮液"). Known gap, in the checker rather than the UI: when the number belongs to the
reference ("The baijiu industry's average P/E is roughly 20x, below Moutai's 24.6x", "…，低于茅台的24.6倍"), the
checker binds the comparator to the reference's own value (Moutai "< 24.6", contradicted) instead of making it a
relation of the average with Moutai's stated value; the UI shows what the checker decided.

### Round 6: moves, dates, multiples, sectors (after exposure of the round-4 held-out slice)

These rules were written after the independent round-4 claim slice (`evaluation/heldout_r4/`) was run
once and its errors were read. Every later number on that slice is labelled "after exposure".

**Bounded moves.** "跌了不到X%", "跌幅不足X%", "涨幅不到X%", "fell less than half a percent" and "rose by less
than two percent" are checks on the size of the move in the stated direction. The move must go that way:
"跌了不到1%" on a day the stock rose is contradicted, and so is "rose less than 1%" on a down day. A flat day
is not a small fall. English fractions and number words are numbers: "half a percent" is 0.5, "a quarter of
a percent" is 0.25, "three times" is 3 and "twice" is 2.

**Qualitative move words (a convention).** A move word without a number is checked against a stated
threshold. The check's `note` names the word and the threshold, for example
`convention: '大跌' means a move of at least 3% in that direction`.

| Words | Meaning | Check |
| --- | --- | --- |
| 大跌, 暴跌, 重挫, 大幅下跌, 跳水, plunged, tumbled, crashed | a fall of at least 3% | size ≥ 3, down |
| 大涨, 暴涨, 飙升, 大幅上涨, soared, surged | a rise of at least 3% | size ≥ 3, up |
| 小幅下跌, 微跌, 小跌, edged down, dipped, fell slightly | a fall of less than 1% | 0 < size < 1, down |
| 小幅上涨, 微涨, 小涨, edged up, inched up, rose slightly | a rise of less than 1% | 0 < size < 1, up |

Negation flips the check: "没有暴跌" holds for any move that is not a fall of 3% or more. With a number the
number wins: "大跌超过3%" is the bound 3. "跌幅较前一交易日扩大" compares with the previous session's move,
which the sources do not carry, so it is unverifiable (`multi_day`).

**Explicit dates.** "4月22日", "2026年4月22日", "2026-04-22", "April 22", "Apr 22nd, 2026" and "22 April" are
dates, never claimed numbers (the English forms are blanked in the verifier too). A date on a daily value
(close or daily change) is compared with the evidence's trade date: a match keeps the check; another date
is unverifiable with `period_mismatch` ("the claim is about 04-21; the data is for the trading day
2026-04-22"). "最近一个交易日", "近1个交易日", "the latest session" mean the latest trading day, not a multi-day
move; "近5个交易日" is still multi-day.

**Multiples.** "茅台的市净率大约是五粮液的1.5倍", "中国平安的净利润是五粮液的三倍多", "五粮液的跌幅大约是茅台的
三倍", "Wuliangye's P/B is roughly twice Moutai's" and "revenue is more than ten times Wuliangye's" compare the
ratio of the two values with the claimed multiple (`eq`/`approx` with the usual tolerance, `gt` for
"三倍多" / "more than"). `ratio` holds the computed ratio and `note` the arithmetic ("ratio 1.50 = 8.1 / 5.4").
For moves both must go the stated way ("跌幅是茅台的4倍" when one rose is contradicted). A multiple of a
negative or zero value is unverifiable. "的一半" is a multiple of 0.5.

**More relations.** "超过了茅台", "营收高于中国平安" and "净利润超过了贵州茅台" work on any metric the data has.
Performance words compare daily moves when no metric is named: "跑赢/跑输沪深300", "outperformed / underperformed
/ lagged / beat the CSI 300". "is higher than" is no longer read as a move.

**Sectors.** A sector named as a sector ("白酒板块", "保险行业", "券商板块", "the baijiu industry", "the insurance
sector", "brokerage stocks") is a target checked against its industry snapshot from `get_fundamentals`
(`pct_change` for the daily change, `pe`, `pb`). Its as-of is the snapshot's trade date, so "白酒板块4月22日收跌"
is unverifiable when the snapshot is dated 2026-04-21, and "白酒板块最近一个交易日跌超1%" is checked. A sector
word that only describes a company ("白酒龙头茅台") is not a target. A company against a named sector ("市净率
高于白酒板块", "P/E above the baijiu industry average", "比保险行业整体便宜") uses that sector's snapshot.

**Several targets, one claim.** "都 / 均 / 皆 / both / all" after two or more targets gives each target its own
check, in the order named: "茅台和五粮液都跌超0.5%" on 2026-04-22 is contradicted for 茅台 (−0.18%) and
supported for 五粮液 (−0.53%), so the verdict is `partially_supported`.

**PMI line.** "荣枯线" and "the boom-bust / expansion-contraction line" are 50: "3月制造业PMI跌破荣枯线" is
`pmi lt 50`, contradicted by 50.6. "上方 / 之上" after a number is `gt`, "下方 / 之下" is `lt`.

**Forecast words.** Lower-case "may" joins could / might / would ("Moutai may trade at 30 times earnings" is
a forecast); capitalised "May" is a month.

### Round 8: every clause, industry averages, turnover, bounded ratios (after the round-4 review)

These rules answer the review's D2-D4 and were written with 20 new dev claims (d205-d224, `note: round8`).
The committed claim held-out set is exposed for them: its h038 is one of these classes.

**Every clause gets a check (D2).** See Relations above; clauses with nothing to check are listed in
`unchecked`. A number whose clause names no metric takes the metric named earlier in its sentence, after the
previous number ("茅台和五粮液的市盈率，分别是24.6倍和20.9倍").

**Industry average as the subject (D3).** An industry average named after the last target and before a number
("中国平安市盈率8.7倍，而行业平均11.8倍", "所属行业的平均水平约12倍", "the sector average is 1.45x") makes the number
the industry's: it is checked against the target's industry snapshot (`target: 保险行业平均` / "insurance industry
average", `evidence_id: industry_保险`, as-of the snapshot's trade date). After a bound it was the company's
bound until round 9 ("市盈率低于行业平均11.8倍" checked the company's P/E below 11.8); since round 9 it is a relation
with the industry plus the stated average (see Round 9). "…，低于行业均值" (no number) is a relation
with the industry snapshot.

**Turnover (D4).** 成交额 / 成交金额 / 成交 (not 成交量 or 成交价) / turnover / traded value is the latest
session's amount (`metric: amount`) from the price evidence, in CNY with the claim's unit (亿, billion):
"五粮液昨天成交14.5亿元" against 14.53 亿 is supported. A multi-day turnover ("本周累计成交") is `multi_day`, a
turnover without an amount unit is `unit_mismatch`, and a reported 0 (index rows in the snapshot) is `no_data`.

**Bounded ratios (D4).** A bound may stand where the ratio verb does: "营收不到五粮液的1.5倍" (`lt`),
"超过…的三倍" (`gt`), "至少是…的五倍" (`ge`), "不足…的一半" (`lt` 0.5), "more than twice …'s" (`gt` 2).
"没有…的两倍" is `lt` 2.

**Hearsay prose.** The chat answer to a hearsay question now opens with the verdict and, per check, the claimed
number next to the data's value ("**核查结论：你听到的说法与数据不符。**“贵州茅台市盈率15倍”不符，数据为 24.6倍，截至
2025-12-31。"), built from the same report as the card (see Hearsay in the chat).

### Round 9: stated industry averages, partial coverage, fractions and ratios, own-clause binding (after the round-5 review)

These rules answer the round-5 review's E1, E2, E8 (claim side), E9 and E14, and the failure classes of the independent
round-5 slice (`evaluation/heldout_r5/`), which was run once at `f01097a` (0.821) before any of them. They were written
with 21 new dev claims (d225-d245, `note: round9`); the round-5 slice is exposed for them, and every later number on it
is labelled "after exposure".

**A stated industry average is checked (E1).** When the other side of a bound is an industry average *with its number*,
the claim states two facts: the company is below (above) the industry, and the industry average is that number. Both
are checked:

| Claim | Checks |
| --- | --- |
| 中国平安PB 1.1倍，低于3倍的行业平均水平 | P/B 1.1 (supported); 平安 P/B `lt` 保险 industry 1.45 (supported); 保险 industry average `eq` 3 (contradicted, 1.45) → partially supported |
| 五粮液PE低于白酒行业35倍的平均估值 | 五粮液 P/E `lt` 白酒 27.3 (supported); 白酒 `eq` 35 (contradicted) |
| 平安8.7倍的市盈率不到保险业均值11.8倍 | 8.7 (supported); `lt` 11.8 (supported); 保险 `eq` 11.8 (supported) |
| below the sector average of 3x / under the baijiu industry average of 30x | the same two checks |

The number may stand before the phrase ("3倍的行业平均水平", "35倍的平均估值"), after it ("行业平均11.8倍", "保险业均值11.8倍",
"sector average of 3x") or in brackets ("行业平均水平（3倍）"). Before round 9 the number was a bound on the company's own
value ("P/B < 3"), so a made-up average passed. "只有行业平均的一半" is checked as a ratio against the snapshot. (Round 9
also read "行业均值的2倍" after a bound as twice the average; since round 10 that form is the stated average, see below.)

**A relation after the number of its own clause is checked.** "Ping An's P/B of 1.1x is below the insurance-sector
average" and "五粮液市净率3.9倍低于茅台" are the number and the relation (two checks); before, the relation was dropped
because its clause stated a number. When a number stands on the compared side ("24.6倍比五粮液的20.9倍高"), the
clause is still read as its numbers.

**Partial coverage (E2).** The report carries `coverage`:

| `coverage` | When |
| --- | --- |
| `full` | every part of the claim was checked and every check decided (supported or contradicted) |
| `partial` | some part is in `unchecked`, or some check is unverifiable, and at least one check decided |
| `none` | no check decided |

The `verdict` is unchanged (over checks only), so the benchmarks keep their meaning. The UI and the inline chat verdict
use `coverage`: a supported verdict with partial coverage is headed **部分核查 / Partly checked** ("已核查的数字与数据源一致，
但说法中还有部分内容没有核查" / "The numbers that were checked match the data, but parts of the claim were not checked"),
with its own icon and tone, never "数字相符 · 说法中的数字都与数据源一致". The chat opening says "你听到的说法只核查了一部分：
已核查的数字与数据相符". `data-verdict` on the badge keeps the server's verdict and `data-headline` holds the headline.

**Fractions, shares and ratio phrasings.** A share *of another value* is a multiple: "的三分之一" → 0.3333, "的三分之二",
"的六成" / "的四成" → 0.6 / 0.4, "是茅台的64%" → 0.64, "的一半" → 0.5. "比五粮液的1.5倍还多/还高" is `gt` 1.5 and
"…还少/还低" `lt` (with 比, only when 还/更 is written or the metric is not itself a multiple: "市盈率24.6倍比五粮液的15.2倍高"
compares two P/Es). "两倍有余", "七倍有余" are `gt` like "三倍多". "more than double / triple Wuliangye's" is `gt` 2 / 3.
"三分之一的营收来自…" (no "的" before it) is not a multiple.

**Binding to the clause's own company (E9).** The NLU matches some vocabulary words to listed companies through the alias
table ("均值" is an alias of 武汉天源). An entity whose every mention lies inside an industry or market reference, an average
word ("平均值", "的均值", "中位数") or a metric word is not a target, so "茅台PB 8.1倍，高于行业均值4倍" checks 白酒's
industry P/B (6.2), not 武汉天源. A short name written after the full name in another clause ("…中国平安…，平安ROE 15.2%")
binds to its company (suffixes only, so "中国" is never 中国平安), and the longest names are placed first.

**Derived and unavailable metrics (E8, same vocabulary as the chat).** 净赚, 赚的钱, 一年赚 are net profit. Net margin
(净利率, 净利润率, 销售净利率) is derived as net profit / revenue when the source reports both (the chat's
`coverage.METRICS["net_margin"].derivable_from`), with the arithmetic in the note: "茅台净利率接近50%" → 48.76% (supported).
PEG is derived as P/E / net-profit growth when a growth rate exists; otherwise, like 市销率 (P/S), 最大回撤 (max drawdown)
and market cap, the check is unverifiable (`no_data`) with a note that says why, not "metric not recognised".

**A fund's computed daily change.** When the source leaves `pct_change_1d` empty (510300 offline) and the last two closes
end at the quoted day, the change is computed from them and the note says so: "computed from the last two closes: 4.776
(2026-04-21) → 4.811 (2026-04-22); the source reports no daily change". The chat template states it the same way
("按前一交易日收盘 4.776 元 计算的当日涨跌幅约 0.73%（数据源未提供涨跌幅）"), with both closes in the sentence so the verifier
accepts the derived percent.

**More.** One macro series against another ("M2增速高于CPI", "CPI低于10年期国债收益率") compares the latest readings. 高过 /
大过 / 强过 / 胜过 are relation words; "更活跃" compares turnover; "表现强于/弱于" with no metric compares daily moves.
English sector names may be hyphenated ("insurance-sector"). `evidence_sources` lists each evidence id once (E14), and
a sector in an English report reads "baijiu industry".

### Round 10: stated values of the compared side, stated differences, bounded numerals (after the round-6 review)

These rules answer the round-6 review's F1, F2 and F7 (claim side). They were written with 23 new dev claims (d246-d268,
`note: round10`), labelled by hand before the checker ran on them.

**Ratio or stated value: the unit of the metric decides (F1).** "茅台市盈率24.6倍，比白酒行业平均的30倍低不少" was read as
"P/E < 30 × the average" (ratio 0.90, supported), so a made-up average of 30 passed. P/E and P/B are quoted in 倍 and ROE
or margins in %, so when the number is written in the compared metric's own unit, "X的N倍" / "X的N%" is ambiguous: X's
value, or N times X's value. The checker reads it as **X's stated value** unless the claim writes an explicit ratio cue:

| Cue | Example | Reading |
| --- | --- | --- |
| a ratio verb 是 / 为 / 相当于 / 等于 / 达到 / (只)有 | "市净率是白酒行业均值的1.3倍左右", "市盈率只有茅台的85%" | multiple (8.1 / 6.2 = 1.31) |
| 还 / 更 after a 比 comparison | "市盈率比五粮液的1.5倍还高" | multiple, `gt` 1.5 |
| a fraction or share word | "不到白酒行业平均的三分之二", "只有行业平均的一半", "六成" | multiple |
| English "N times X's" | "1.5 times Wuliangye's" | multiple |
| none (a bound word or 比 … 低/高) | "比白酒行业平均的30倍低", "低于五粮液的20倍", "ROE高于五粮液的29.4%" | stated value of X |

A stated value is checked on its own (`kind: "stated_reference"`), next to the relation of the two:

| Claim | Checks |
| --- | --- |
| 茅台市盈率24.6倍，比白酒行业平均的30倍低不少 | 24.6 (supported); 茅台 P/E `lt` 白酒 27.3 (supported); **白酒行业平均 `eq` 30 (contradicted, 27.3)** → partially supported |
| 五粮液市盈率比茅台的30倍低 | 五粮液 `lt` 茅台 (supported); 茅台 P/E `eq` 30 (contradicted, 24.6) |
| 五粮液ROE低于茅台的33% | `lt` (supported); 茅台 ROE `eq` 33 (supported) |

The rule depends only on the words and the metric's unit, never on the data (a reading chosen because it makes the claim
true would hide errors). A metric not quoted in 倍 ("营收是五粮液的1.5倍", "净利润超过五粮液的三倍") is always a multiple.
In the fact-check view the stated value is its own row, marked **说法给出的数值 / Stated in the claim**, with the
average's label ("白酒行业平均") and its own verdict; the comparison row shows "< 白酒行业" with the snapshot value, never
"30× 白酒行业" (`data-kind` on each row).

**Stated differences (F2).** "茅台ROE比五粮液高出约3.6个百分点" was read as "五粮液's ROE ≈ 3.6" and contradicted a true
claim. A number after "比X + 高出 / 高 / 多 / 大 / 贵 / 多赚 / 低 / 少 / 小 / 便宜 (了)" or after "(和X)相差 / 差了 / 差距" is
the **difference** of the two values (`kind: "difference"`, `difference` = target − reference):

| Claim | Check |
| --- | --- |
| 茅台ROE比五粮液高出约3.6个百分点 | 33 − 29.4 = +3.6, `approx` 3.6 → supported |
| 五粮液ROE比茅台低3.6个百分点 | 29.4 − 33 = −3.6, claimed −3.6 → supported |
| 中国平安ROE比五粮液高出10个百分点 | 15.2 − 29.4 = −14.2: the other way → contradicted |
| 茅台和中国平安的ROE相差约18个百分点 | no direction: \|33 − 15.2\| = 17.8 → supported |
| 茅台净利润比五粮液多赚了四百四十多亿 | 823.2亿 − 378亿 = 445.2亿, within 440-450亿 → supported |
| 中国平安的市净率比五粮液低4.3倍 | P/B is quoted in 倍: 1.1 − 5.4 = −4.3 → supported |
| 五粮液市盈率比白酒行业平均低了大约两成 | "%" on a metric not quoted in % is relative: (20.9 − 27.3) / 27.3 = −23.4%, about 20% → supported |
| 茅台营收比五粮液高出一倍多 | "高出N倍" of an amount (N or N+1 times?) → unverifiable (`unit_mismatch`) |

The comparator applies to the size of the difference in the stated direction ("高出不到5个百分点" is 0 < difference < 5);
"跌幅比茅台大0.36个百分点" is a lower daily change. "%" on a metric quoted in percent (ROE) is read as percentage points.
The UI shows "差值 ≈ +3.6 个百分点", the other side's value and the actual difference ("实际 +3.6 个百分点"). A later clause
with no name of its own belongs to the comparison's subject, not to its compared side: in "茅台ROE比五粮液高出约3.6个百分点，
一年营收一千六百多亿" the revenue is 茅台's (found in the real-Chrome check; before, it bound to 五粮液, the nearest name).

**Numerals with 多 / 余 / 出头 / 左右 (F7).** "一千六百多亿", "八百余亿", "三十多倍" now parse (多 / 余 between the numeral and
its unit). They are bounded approximations, using the **step** of the number's last significant digit (800 → 100,
1600 → 100, 三成 = 30% → 10, 24.6 → 0.1):

| Written | Reading |
| --- | --- |
| N多 / N余 / N有余 ("八百多亿", "七倍有余", "三成多") | N < actual < N + step (800-900亿, 7-8, 30%-40%) |
| N出头 ("三成出头", "八百亿出头") | N < actual ≤ N + step / 2 (30%-35%) |
| N倍多 with N ≥ 10 ("10倍多", 多 after 倍; round 11) | N < actual < N + min(step, N / 10) (10-11; "十多倍", 多 before 倍, stays 10-20) |
| 约 / 左右 ("三成左右", "约30倍") | within 5%, or step / 2 when wider (三成左右: 25%-35%) |
| 接近 / 将近 / 近 / nearly (round 11) | 0.9N ≤ actual ≤ N, the r6 slice's label rule (将近900亿: 810-900亿) |

The check keeps `comparator: "gt"` with the upper bound in `claimed_high`; the UI shows "> 800 亿, < 900 亿". "近10%"
right before the number is `approx` again (the look-behind text ended before the digit, so 近 was missed).

Screenshots (real Chrome, offline server, round 10): [stated average and relation](assets/ui/chrome-r10-stated-average-zh.png),
[stated difference](assets/ui/chrome-r10-difference-zh.png).

### Round 12: sums, 破 / 不到 bounds, framed and anaphoric averages, English and one-fold differences (after the round-8 review)

These rules answer the round-8 review's H6. They were written with 23 new own-wording dev claims (d277-d299,
`note: round12`), labelled by hand from the snapshot before the checker ran on them; tests in
`tests/test_agent_round12_claims.py`. The round-8 slice (`heldout_r8`) was not opened.

| Form | Example (own wording) | Reading |
| --- | --- | --- |
| Sum of companies | "茅台和五粮液成交额加起来不到50亿", "净利润合计约1200亿", "together / combined" | a sum word (合计, 加起来, 加在一起, 总共, 之和, combined, together, in total) in the number's part of the sentence and two or more companies before it: `kind: "sum"`, `operands`, `operand_values`, `operand_evidence_ids`; the total is compared (52.47亿, contradicted). Before round 12 the number was bound to the last company (14.53亿 < 50, **supported**). A sum of P/E or rates is unverifiable |
| 破 | "五粮液营收破千亿", "股价破两千元" | `ge` (跌破 stays a fall below, 突破 stays `gt`) |
| Bound after the number | "只有五粮液的一半不到", "连五粮液的三成都不到" | `lt` on the multiple (0.416 < 0.5) |
| Stated average in a comparison frame | "比起保险业11.8倍的平均市盈率，平安的8.7倍明显偏低", "对照…", "和…相比", "相较于…", "where the average multiple is about 27x" | a P/E or P/B next to an average word with no company in its clause is the industry's (or named sector's) average, checked as `stated_reference`; an evaluation word in the sentence (偏低, 更低, 偏高, 便宜, cheaper) adds the company-vs-industry relation |
| Anaphora | "保险行业平均市盈率11.8倍，中国平安低于这一水平", "茅台的PB比它高", "Ping An trades below that" | the compared side is the latest stated value that is not the subject's own (the k17 case, previously `unchecked`) |
| Sector average vs a stated company value | "The baijiu industry's average P/E is roughly 20x, below Moutai's 24.6x" | three checks: the average (20 vs 27.3, contradicted), the relation average < Moutai (27.3 vs 24.6, contradicted) and Moutai's stated 24.6 (supported) |
| Bracketed average after a bound | "中国平安的市盈率低于保险行业均值（约20倍）" | relation (supported) + stated average (contradicted) |
| English differences | "Moutai's ROE beats Wuliangye's by about 3.6 points", "trails … by roughly 14 percentage points", "exceeded … by roughly 44.5 billion yuan", "is roughly 60 billion yuan below Moutai's" | stated difference; "points" of a percent metric are percentage points |
| One fold | "高出一倍多", "多了将近一倍", "低了将近一半" | relative difference: 一倍 = 100% (一倍多 (100%, 200%), 将近一倍 [90%, 100%]); 一半 after 比 = 50%. 两倍 and more stay unverifiable (two or three times?), so dev row d261 ("茅台营收比五粮液高出一倍多") was relabelled unverifiable → contradicted (+55.6%) |
| Numerals | "一成半", "一千四百出头", "一万二千多亿", 较 / 相较于 as 比 | 15%, (1400, 1450], (12000, 13000)亿 |

### Round 13: X倍半, one-fold bands, "a little", 52-week references, turns, contrast clauses, superlatives, English fractions (after the round-9 review)

These rules answer the round-9 review's I8 (false contradictions of true claims). Failing tests were written first,
with the engineer's own wordings and other pairings than the review's repros, mostly on the snapshot extension's
names (`tests/test_agent_round13_claims.py`, 50 cases; values from `data/snapshot/structured_data_ext.json`). The
review's held-out round-9 slice (`heldout_r9`) was not opened.

| Form | Example (own wording) | Reading |
| --- | --- | --- |
| X倍半 | "工商银行的净利润大约是招商银行的两倍半", "约为平安银行的三倍半" | X.5 times (before: the 半 was dropped, so a true 3.52 was contradicted against 3.0) |
| 将近 / 接近一倍 | "比亚迪的营收比宁德时代高出接近一倍" | a relative difference in (80%, 100%] (before: [90%, 100%], so +89.58% was contradicted). 一倍多 stays (100%, 200%) |
| "a little" | "只有招商银行的七成多一点", "比工商银行高一点", "低一点点", "多一些" | not a number (before: 一点 became an extra check "eq 1") |
| 52-week high / low | "比52周最高收盘价低了三成多", "距52周高点已经跌去超过三成", "较52周高点已经腰斩", "37% below its 52-week high" | 52 is a window, never a value; the high (low) is the target's own 52-week closing high (low) from the price history (`range_52w`), and the claim is a relative difference of the latest close against it; 腰斩 is about -50%. Unverifiable when the history does not cover 52 weeks (v1 names). Before: a check "close eq 52" |
| Turns / points above | "about 10 turns above CATL", "some 20 turns higher than CATL's", "6.5 points above BYD's" | an absolute difference of the multiple (P/E) or percentage points (ROE); before: 10 was checked as BYD's own P/E |
| Contrast clause | "贵州茅台的PE低于白酒板块平均水平，但PB高于板块平均", "…市净率却比行业平均高" | a target named as the compared side of an earlier clause is not the subject of a later one, so the subject carries over 但 / 却 (before: the second clause compared 白酒 with 行业平均) |
| Superlatives | "在招商银行、工商银行和平安银行三家中，平安银行的ROE最低", "市净率最高的是招商银行", "Of CMB, ICBC and Ping An Bank, ICBC has the highest ROE" | one `relation` check: the subject against the most extreme of the other companies the sentence names (shown as the compared side, with all values in the note). With one named company it stays unchecked; 最高价 / 最低收盘 / 最大回撤 are metrics, not superlatives |
| English fractions | "less than a third of CATL's", "about a quarter of ICBC's", "roughly two-fifths of", "less than half of" | multiples 0.3333, 0.25, 0.4, 0.5 (only before a capitalised name, so "half a percent" stays 0.5 percent) |
| Numerals before 个百分点 | "相差三个多百分点", "高约一个百分点" | (3, 4) and about 1 point; before: "三个百分点" was read as nothing |
| A sector as a ratio reference | "保险行业平均市盈率11.8倍，不到白酒行业的一半", "是保险行业的四倍多" | a multiple of the other sector's snapshot value |
| English sector average | "26% below the insurance average", "below the baijiu average" | the subject's industry average |

The chat's output-safety layer splits clauses on the same contrast words since round 13 (I19): 但 / 但是 / 却 / 不过 /
but / however / yet (not 不但), so a single-source figure after "但" is marked in its own clause.

### Round 13 on the r13-frame branch: the round-8 slice's claim classes (after exposure)

In parallel, the r13-frame branch read the five round-8 slice claims the checker still got wrong after round 12
(两倍多一点, 三个多百分点, "26% below the insurance average", "less than half of Moutai's net profit", 是保险行业的四倍多)
and wrote rules for their classes with 12 own dev claims (d300–d311, `note: round13`) and tests in
`tests/test_agent_round13.py` (`476d377`). Both branches changed the same forms, so `claim_check.py` was merged by hand
(`66fa560`). What the merged code does with the r13-frame forms:

| Form | Example (own wording) | Reading in the merged code |
| --- | --- | --- |
| A little over | "茅台净利润是五粮液的两倍多一点", "平安营收是茅台的七倍多一点", "多一些" | the "a little" rule above drops 一点 / 一些 before the numerals are read, so 两倍多一点 is read as 两倍多: over 2 and under 3 (`claimed` 2, `claimed_high` 3). Only "多些" / "多点" (without 一) keep the r13-frame band of at most 0.3 of a step (2–2.3×). On the r13-frame branch 两倍多一点 was 2–2.3×; d300 (2.18) and d301 (7.21, supported) and d302 (0.09, contradicted) get the same verdicts either way, but a true ratio of 2.5 called "两倍多一点" is now supported |
| Points with 多 | "茅台和平安的ROE相差十七个多百分点" | 17 < difference < 18 (the numeral before 个 is read, as in "Numerals before 个百分点" above) |
| Multiple of another sector | "白酒行业平均市盈率是保险行业的两倍多" | the sector name before 行业 is the compared side (27.3 / 11.8 = 2.31×) |
| A sector's average | "Wuliangye's P/E sits roughly 23% below the baijiu average" | "<sector> average" is that sector's average, a relative difference (−23.44%) |
| English shares | "Wuliangye makes less than half of Ping An's revenue", "Ping An's P/B is under a third of Wuliangye's" | half / a third / a quarter of X's is a multiple of X (0.09 < 0.5; 0.20 < 0.333) |

These rules were written after reading the round-8 slice's errors, so its 20/25 → 25/25
(`claims_heldout_r8-after-exposure-round13.json`, `b1099d8`) is **after exposure**: the classes are covered, not
shown to generalise.

### Round 14: 扣非 / 流通市值, two bounds, subject-first superlatives, years, YoY; the second real-claims set's classes (after the round-10 review)

Two lists of failures were fixed, each with failing tests written first in the engineers' own wording.

**The round-10 review's J12** (`tests/test_agent_round14_claims.py`, 40 cases; 32 of the first 39 failed before the fix):

| Form | Example (own wording) | Reading |
| --- | --- | --- |
| 扣非净利润 | "比亚迪扣非净利润超过300亿元", "扣除非经常性损益后的净利润", "net profit excluding non-recurring items" | its own metric `profit_dedt` (294.46亿, contradicted). Before: the 净利润 inside 扣非净利润 matched net profit, so the 326.19亿 归母 figure supported it. A source without the field gives `no_data` with a note; "扣非净利润同比…" is `profit_dedt_yoy`, which no source has (`growth_unavailable`), never net-profit growth |
| 流通市值 | "比亚迪的流通市值不到3000亿元", "free-float market cap" | its own metric `circ_mv` (2905.08亿, supported). Before: the total 7595.54亿 contradicted it. 总市值 / 市值 stay `market_cap` |
| Two bounds split by but / 但 | "BYD's P/B is below CATL's but above Midea's", "比亚迪的市净率高于美的集团但低于宁德时代" | the compared side of one bound is never the subject of the next, in English too and inside one clause (before: the second bound compared CATL with Midea) |
| Subject first | "紫金矿业的ROE在紫金、美的、宁德时代三家中最高", "招行是招行、工行、平安银行三家里ROE最高的" | the company named before 在 / 是 and the set is the subject (before: the last company of the set); its own short name may open the set |
| Year tokens | "for 2025", "FY2025", "fiscal 2025", "2025 年" | never a claimed value (they still set the period) |
| YoY | "up 42% year on year", "rose about 62% year over year", "同比提升2个百分点" | the year-on-year change of the metric named in the clause, else of the previous number's (before: "up 42%" was the daily change); the YoY of a ratio (ROE, margins) is `growth_unavailable`, not its level |
| 不到 / 超过…的N倍 | "招商银行市净率还不到平安银行的2倍", "超过招商银行的1.2倍" | a bound on the multiple. 不到 / 不足 / 不及 / 超过 / 超出 / 至少 / 至多 / 最多 are ratio cues now; 低于 / 高于 before "X的N倍" are still read as X's stated value. Round 10 had read "茅台市盈率不到行业均值的2倍" as "the average is 2x"; that test now expects a multiple |
| A difference without 比 | "招行ROE高出工行三个多百分点", "领先美的约1.7个百分点" | 高出 / 多出 / 领先 / 落后 right before the compared company and the number: a difference (before: 3 was checked as 工行's ROE) |
| N个半 / N个 | "工行一年赚的钱相当于两个半招行", "相当于两个比亚迪" | after 相当于 / 等于 / 抵得上: 2.5 times, 2 times ("两个半月" is a duration) |
| Colloquial names | "工行", "招行", "美的" next to the full name | surface forms of their company (the colloquial alias list), so a set written with both is read as one company each |

**The blind second real-claims set's classes** ([evaluation/real_claims_web_v2/README.md](../evaluation/real_claims_web_v2/README.md);
after exposure; `tests/test_agent_round14_claims_v2.py`, 46 cases, 34 failing before the fix; none are the set's
sentences):

| Class | Example (own wording) | Reading |
| --- | --- | --- |
| A named definition | "宁德时代加权平均净资产收益率为21.4%", "美的集团基本每股收益5.76元", "宁德时代的动态市盈率约15.9倍", "美的集团营业收入4585亿元", "比亚迪归母净利润同比下降18.8%" | the data states each field's definition (`definitions`: `roe_basis`, `eps_basis`, `net_profit_basis`, `netprofit_yoy_basis`, `revenue_basis`, `revenue_yoy_basis`, `pe_basis`). A claim that names another one is unverifiable, `reason: definition_mismatch`, with both definitions in the note; one that names the served definition is checked ("贵州茅台加权平均ROE为32.53%" is supported). The named definition carries to the growth and the parallel clause ("归母净利润518亿元，同比增长62%": the YoY is on total profit, so unverifiable). A source that states no definitions (v1) is compared, with "the source does not state its basis" in the note. A claim that names no definition is compared with the served value |
| A dated price | "截至2026年4月22日收盘，比亚迪收于101.62元", "2026年4月22日中芯国际成交量37.52万手" | `get_price_history` takes `on_date` and returns that day's bar from the history (`bar_on_date`, evidence `price_<symbol>_<yyyymmdd>`); a date outside the history stays `period_mismatch`. A date the sentence leaves to its publication day is not known |
| A valuation on its date | "截至2026年9月30日收盘，平安银行总市值约2245亿元" | P/E, P/B and market value are dated by their valuation date, not the report period |
| English | "Industrial & Commercial Bank of China …", "China Yangtze Power …", "net profit attributable to shareholders", "basic earnings per share advanced 6.6% to RMB 5.80" | names with "and" / "&" resolve before the NLU turns "and" into 和 (it used to leave "Bank of China"); 长江电力 has English aliases; a move of a reported flow (revenue, profit, EPS) is its YoY |
| Another currency | "590亿美元", "US$110 billion", "S$68.6 billion" | unverifiable, `reason: currency` (never compared with a CNY figure); "RMB 5.80" is a price in yuan |
| Traditional characters | "工商銀行年度實現淨利潤3,707.66億元", "每股盈餘", "年增3.46%" | read through a character table and a few Taiwan / Hong Kong terms (`agent/traditional.py`); the report keeps the claim as written |
| 扣非 written out | "扣除非经常性损益的净利润", "归母净利润（扣非）" | `profit_dedt` |
| Close words, volume | "报83.31元", "报收41.26元", "成交量37.52万手", "换手率1.88%" | 报 / 报收 is the close; 成交量 in 股 / 手 / 万手 / 亿股 is `volume` (shares); 换手率 is its own metric with no source (before: a daily change) |
| A base year | "和2024年相比，比亚迪营收增长约3.5%", "较 2024 年" | the year a change is measured from is not the claim's period |
| A change in amount | "工商银行净利润比上年增加约38亿元", "an increase of RMB 2.007 billion year-on-year" | derived as level − level / (1 + YoY) from the same report, with the arithmetic in the note |
| A total of several companies | "宁德时代的净利润高于比亚迪、蔚来和小鹏三家的利润之和" | not checked (`no_data`, the companies outside the data named in the note); before: compared with 比亚迪 alone and supported |

Found by the round-14 runs and still open: a definition named more than 40 characters before the number ("The net
profit attributable to the parent company was RMB 34.503 billion") is not seen; "人民币" is an alias of 宁德时代 in the
runtime alias table, so a claim that writes 人民币 gets 宁德时代 as a target; "连续 3 年" with spaces is read as a value.

### Round 15: trillions, several companies as one side, 扣非 vs 归母, foreign lines (after the round-11 review, K6)

The round-11 review's new claims scored 17/30, with false contradictions of true published-style sentences and false
accepts (K6). Failing tests were written first in the engineer's own wording (`tests/test_agent_round15_claims.py`; 34
of the first 40 cases failed at `4ad3380`, 47 cases now; none is a review sentence).

| Form | Example (own wording) | Reading |
| --- | --- | --- |
| Trillion | "CATL's market capitalisation was roughly RMB1.35 trillion", "about CNY 1.04tn" | 10^12 (the 万亿 scales). Before: the bare scale compared 1.35 with 1.347×10^12, a contradiction |
| A mean of named companies | "招商银行、工商银行和平安银行三家的市盈率平均下来约6.6倍", "ROE均值约18%", "The average P/E of CMB, ICBC and Ping An Bank is roughly 6.6x" | `kind: "mean"`: the arithmetic mean of every company named in the sentence before the number (6.86, 7.89, 5.17 → 6.64), each operand's value and evidence id in the check. Not an industry / market / peer average and not 加权平均 (a definition). Before: the last company's own P/E (5.17, contradicted) |
| A total as the compared side | "紫金矿业的总市值比美的集团和平安银行加起来还高", "more than CMB's, Ping An Bank's and CITIC Securities' combined", "the combined net profit of A, B and C" | the reference is the total of the listed companies (`reference` "美的集团+平安银行", `reference_value` 8356.84亿, operands in the check), for a relation, a stated difference ("比…三家加起来还多1450多亿") or a share. Before: the first company alone (紫金 7921.40亿 > 美的 6111.58亿, a false accept) |
| A total as the subject | "美的集团和平安银行的总市值合计超过紫金矿业" | the listed companies' total is the subject. Before: 平安银行 alone |
| A total with a company outside the data | "工行净利润比招行和蔚来加起来还多2000亿", "ICBC's net profit beat CMB and Xpeng combined by RMB200 billion" | `no_data` with the missing names; never the first operand alone, in English lists too ("and" ends a clause, so round 14's check missed them) |
| A share of an N-company total | "招商银行的净利润占招行、工行、平安银行三家合计的两成多", "在工行、招行、平安银行三家里，工行营收占比超过六成", "CMB accounts for about 27% of the combined net profit of CMB, ICBC and Ping An Bank" | a multiple of the total (1501.81 / 5613.76 = 0.2675). A share of a whole whose companies are not named ("几家大行里，工行…占几家合计的六成以上") is `no_data`, never a growth rate. Before: a sum with a unit mismatch, or the YoY of the last company |
| 扣非 vs 归母 of one company | "中国平安的扣非净利润比归母净利润高", "比亚迪扣非净利润不足300亿元，比归母净利润少30多亿", "recurring net profit was about RMB3.2 billion below its net profit attributable to shareholders" | another figure of the same target as the compared side (`reference_metric`), for a relation or a stated gap (31.73亿). A bare 归母 is net profit. Before: no check, or the gap read as a net profit of 30-40亿 |
| A year before the line item | "Midea Group's 2025 revenue of RMB458.5 billion", "CATL's 2025 net profit of …" | the year is the period, not a value. Before: "2025" was a revenue with no unit |
| A foreign line | "比亚迪H股的市净率约3.2倍", "招商银行港股市盈率不到7倍", "ICBC's Hong Kong-listed shares …", "长江电力伦敦GDR …", "…，H股约2.5倍" | `no_data`, "the data covers the A-share line only, so A-share data neither supports nor contradicts it" (the chat's own detector, `coverage.foreign_line_spans`, plus bare H股 / 港股 / ADR / GDR words, HK codes and the venue). The A-share part of the same claim is still checked. Before: the A-share P/B supported an H-share claim |
| A short name after the full name | "美的集团营收比宁德时代还多，但净利润只有宁德的六成左右" | a four-character name's two-character head ("宁德", "中芯", "紫金") is that company when the full name is in the claim, except heads several companies share (中国, 招商, 平安, 长江, …) |
| The subject across 但 | "比亚迪的营收是宁德时代的1.9倍，但净利润只有约326亿元" | the compared side of a multiple is not the subject of a later clause (as round 13 did for 比 / 高于). Before: 宁德时代's 722亿 contradicted it |
| "earned", "by RMB …" | "ICBC earned more than CMB and Ping An Bank combined", "beat … by RMB200 billion" | "earn(s/ed)" is the year's net profit; a currency may stand after "by" |
| A definition caution alone | "宁德时代2025年归母净利润722亿元，同比增长42%" | when every unverifiable part is `definition_mismatch`, the verdict is `supported` with partial coverage ("partly checked"), not `partially_supported`. Any other unverifiable part still makes it partial |

**The blind third real-claims set's classes** ([evaluation/real_claims_web_v3/README.md](../evaluation/real_claims_web_v3/README.md);
after exposure; `tests/test_agent_round15_claims_v3.py`, 32 cases at first, 26 failing before the fix, 36 now;
`tests/test_agent_round15_aliases.py`; none are the set's sentences):

| Class | Example (own wording) | Reading |
| --- | --- | --- |
| 市盈TTM | "中信证券目前的市盈TTM约为10.2倍" | P/E (the rate character left out) |
| A daily average of net profit | "中国平安2025年平均每天净赚约3.7亿元", "宁德时代2025年日赚近2亿元", "BYD earned about RMB89 million a day" | `net_profit_daily`: the year's net profit / 365 calendar days, said in the note. A day's earnings without an average word ("今天赚了2亿") are not this. Before: no metric, or "每天净赚超过1.2亿元" checked as net profit > 1.2亿 (a false accept) |
| A bare 利润 | "工商银行2025年的利润约为3686亿元", "利润增62%到518亿" | net profit (归母), said in the note; next to 扣非 / 利润总额 / 营业利润 in the same sentence it is ambiguous (`no_metric`); 利润总额 / 营业利润 / 毛利润 / 总利润 are never read as it |
| Bar words | "开盘3136.53点，最高3158.91点，最低3128.48点，收报3142.56点", "finished September at about 3,840" | `open`, `high`, `low`, `close` from the session's bar (that day's bar when dated) |
| A change in 点 / 元 | "创业板指4月22日涨了63.82点", "上证指数9月30日比前一日上涨11.74点" | `change_1d`: close − previous close (for a dated bar, close − close / (1 + pct)). Before: refused as a unit mismatch of the % change |
| An English number before its metric | "RMB 804 billion in revenue and RMB 32.6 billion in net profit", "a 6.5% year-on-year increase in net profit" | the metric after "in / of" (the growth of a flow after a YoY or move word). Before: the second number was compared with revenue |
| A previous-year value | "长江电力2025年营收862.42亿元，上年为844.9亿元", "the previous year's RMB409 billion", "较2024年的4090.84亿元" | derived as level / (1 + YoY) from the same report, with the arithmetic in the note |
| Turnover against the previous session | "成交额较前一交易日减少约52亿元", "环比增长13.75%", "缩减约两成" | `amount_change_1d` / `amount_change_pct_1d` from the session's bar and the previous session's (fetched with `on_date`) |
| A month-day without a year | "4月22日，创业板指收盘站上3750点" | the last such day on or before the latest bar (2026-04-22), and that day's bar. Before: compared with 2026-09-30 and `period_mismatch` |
| A daily figure with no date | "截至收盘，宁德时代下跌2.51%" | checked against the latest session only when it fits; otherwise `period_mismatch` ("the claim states no date …"), because a published sentence is dated by its article. A claim that names the latest session (今天, 最新, 上一个交易日, 股价, today, latest …) is still contradicted by it. Three dev rows were relabelled for this rule (d022, d216, d208's second check) |
| 人民币 | "比亚迪全年营业收入人民币8039.65亿元" | the runtime alias table listed 人民币 as an official name of 宁德时代 (scraped from "…人民币（300750）"); the builder now rejects currency words and phrases around a known name, the 11 rows those rules reject are gone, and currency words never fuzzy-match a name |
| A code in brackets | "科创50成份ETF（588000）…", "创业板ETF（159915，旗下联接基金A类代码110026）…" | a code that opens the bracket names the security (it wins over the name before it); codes after words in the bracket are asides, never the subject |
| 分别 lists and "vs" pairs | "工商银行、招商银行、宁波银行、平安银行的净利润分别为…", "平安银行与招行的ROE（7.7% vs 11.8%）预计2026年仍将承压" | element-wise in order, a name outside the data left unverifiable; "x vs y" goes to the two companies in order; a forecast after a closing bracket does not make the figures before it a forecast |
| Declines | "净利润降幅为71.9%", "营收减少了约55%" | 降幅 / 减幅 / 缩减 / 萎缩 / 缩量 make the change negative |
| Another growth rate stated | "净利润增幅36.3%，高于营收增幅的16.5%" | the number after the bound is the stated value of the other metric of the same company, next to a relation of the two (two declines compare their sizes) |
| 营业总收入 = 营业收入 | "宁德时代2025年营业总收入4237亿元" | checkable where the company's report gives one top line (`data/snapshot/corrections/sources.json`: "营业收入 = 营业总收入", or no separate 营业总收入); still `definition_mismatch` where they differ (贵州茅台) |
| 成份 | "科创50成份ETF" | "50成份" is not 50 成 (500%) |

Open after round 15: a short name with no full name in the claim ("比亚迪≈宁德1.9倍"); a total or mean whose companies
are only counted, not named ("三家银行平均市盈率"); an English share worded "X's share of …"; mean, total and daily-average
words are fixed lists; 近N keeps half a unit of slack at the top (紫金 "日赚近1.4亿元" at 1.4185亿 is supported, the
third set labels it contradicted); a date that only the article gives is not known (x04, x06 are unverifiable, not
supported); fund names whose codes collide with stock codes are still aliases of the stock in the runtime table (1,038
rows, e.g. 华夏成长混合 → 平安银行).

### Macro values (C13)

Macro claims are checked against the latest reading from `get_macro_indicators`. `as_of` is the reading's
period (`as_of_basis: indicator_date`).

- "CPI同比上涨0.8%" and "M2同比增长8%" compare the YoY value; a stated fall gives a negative claim.
- "PMI重回50以上" and "The PMI is above 50" are bounds on the level. Only the latest level is checked,
  not that it was below 50 before.
- "2月CPI同比上涨0.8%" against a March reading is unverifiable (`period_mismatch`).
- A change from the previous reading ("PMI回落了", "CPI同比回升") is unverifiable (`no_data`): only the
  latest level is served.

### Unverifiable, and why (`reason`)

| `reason` | When |
| --- | --- |
| `no_target` | no listed company, fund or index is recognised |
| `no_metric` | the number cannot be tied to a metric |
| `no_data` | the source has no value for the target and metric (a company outside the offline snapshot, an ETF with no daily change and no two closes ending at its quote date, index P/E, price-to-sales, max drawdown, PEG without a growth rate, index turnover reported as 0, dividend yield, market cap); since round 15 also a figure of an H-share / Hong Kong / ADR / GDR line, and a total or share whose companies are not all in the data or not named |
| `growth_unavailable` | a YoY growth claim, but the fundamentals payload has no YoY field |
| `unit_mismatch` | the unit does not fit the metric |
| `no_unit` | an amount with no unit |
| `forecast` | 预计, 将, 会, 明年, 目标价, will, expected, if, … |
| `period_mismatch` | the claim names a year or period other than the report's; or an amount with no period is compared with an interim (Q1/H1/Q3, year-to-date) report; since round 15 also a daily figure in a claim that states no date and names no latest session, when it does not fit the latest session |
| `multi_day` | 今年以来, 近一个月, 本周累计成交, this year, … (only the latest daily change and turnover are checked) |
| `no_reference` | a relation with something no source provides, such as the market average |
| `definition_mismatch` | (round 14) the claim names a definition other than the one the data states (加权平均 vs period-end ROE, 基本 vs diluted EPS, 归母 vs total-profit YoY, 营业总收入 vs 营业收入, 动态 vs TTM P/E) |
| `currency` | (round 14) an amount in another currency (美元, US$, S$, 港元) |

### Growth rates

**Live data.** The live fundamentals carry YoY growth: akshare (Sina / THS) provides `revenue_yoy`
and `netprofit_yoy`, and Tushare provides `netprofit_yoy`. For example, a probe on 2026-09-28 gave:

| Company | Revenue YoY | Net-profit YoY | Report |
| --- | --- | --- | --- |
| 宁德时代 | 54.8 | 41.98 | 2026-06-30 |
| 贵州茅台 | 1.47 | −2.03 | 2026-06-30 |

"宁德时代营收同比增长54.8%" is therefore supported. When the report is an interim one and the claim
names no period, the check adds a note that the growth is year to date.

**Offline data.** The snapshot (`data/structured_data.json`) has no growth fields. Growth claims are
unverifiable with `growth_unavailable`; the checker does not compare them with the level or guess.

### As-of date (B26)

- **Prices:** `as_of` is the trade date (`as_of_basis: trade_date`).
- **P/E and P/B:** `as_of` is the valuation date when the source has one (`valuation_date`):
  - the akshare valuation endpoint provides it;
  - Tushare now records the daily_basic trade date as `valuation_date`.
- **Otherwise:** `as_of` is the report period (`as_of_basis: report_date`). The offline snapshot has no
  valuation date, so offline P/E shows 2025-12-31 (报告期).

The UI shows the basis next to the date.

### Hearsay in the chat

A chat question that is a claim, such as "听说茅台市盈率只有15倍，是真的吗" or "I heard that …, is that
true?", is fact-checked inline. `query_intelligence/agent/hearsay.py` extracts the claim
("茅台市盈率只有15倍") and runs `check_claim` on it (deterministic, no LLM). The report is returned as
`fact_check` on:

- `/agent/chat`, `/agent/resume` and the SSE `answer` event (it is part of the agent result);
- workflow `/chat` (only when the message is hearsay).

The answer text opens with the verdict and each claimed number next to the actual one (round 8; built by
`fact_check_prose` from the same report, deterministic), so the prose itself answers "is it true". The answer
card renders the report as a "核查这句说法 / Fact-check of this claim" section, with a button that
opens the full fact-check view. While the answer is still running, and on a server without
`fact_check`, the "核查这句话 / Check this claim" chip under the question does the same by hand. Hearsay
with no number, move or comparison is not checked. A failed check never breaks the answer: `fact_check`
is then `null`.

Round 12 (H12): the cues are classes rather than a phrase list: a source that says something (网上 / 群里 / 博主 /
朋友…说, 告诉我, "someone told me", "I read / saw / heard"), a report word (据报道, 网传, apparently, reportedly), a request
to check (核实, 查证, "fact-check", "can you verify") or a confirmation question at the end (对吧, 是这样吗, 真的假的,
"True?", ", right?"). "I read that Ping An's P/E is 15x and Moutai's ROE is 33%. True?" now opens with "the claim you
heard partly matches the data" (15x does not match 8.7x; 33% matches). The web UI's `claimInMessage` hint still uses
the older list; the server's inline check does not depend on it.

### English names

The report's `targets` and the agent's `nlu_summary.entities` carry `name_en`, taken from the alias table
(`data/synonym_dict.json`: `display_en`, else the longest English alias; see
`query_intelligence/agent/names.py`). Structured evidence in agent answers also carries `name_en` (companies
from the alias table, industries from `INDUSTRY_EN`), and fundamentals payloads carry the company name, so a
follow-up turn that fetches only fundamentals ("那它们的ROE呢") names its tiles "贵州茅台 · ROE" / "Kweichow
Moutai · ROE" instead of "600519.SH · ROE" (round 8, D9). Industry tiles in English read "Baijiu (liquor) ·
Industry P/E". The browser's fallback industry table (`frontend/src/lib/format.ts` `INDUSTRY_EN`) is the same
table as the backend's; `tests/test_web_ui.py` checks they are equal.

Screenshots (real Chrome, offline server):

- a compare answer with the same KPI tiles for each company (C18): [zh](assets/ui/chrome-compare-kpi-zh.png), [en](assets/ui/chrome-compare-kpi-en.png);
- a hearsay question checked inside the answer: [zh](assets/ui/chrome-move-claim-inline-zh.png), [en](assets/ui/chrome-move-claim-inline-en.png);
- the move-bound card in the fact-check view ("跌幅 > 0.1%" / "Fall > 0.1%"): [zh](assets/ui/chrome-move-claim-card-zh.png), [en](assets/ui/chrome-move-claim-card-en.png).

## Benchmark

The benchmark files are:

- claims: `evaluation/claim_bench/claims_v1.jsonl` (dev) and `claims_v1_holdout.jsonl` (held out);
- labelling policy: `evaluation/claim_bench/README.md`;
- runner: `evaluation/claim_bench/run.py`.

The claims cover:

- zh and en;
- true, false and approximate values;
- comparators, ranges, negation, number-less moves and sign errors;
- growth, multi-number and multi-target claims;
- targets with no data, unknown targets and opinions;
- wrong units, index and ETF targets, Chinese numerals, forecasts, period mismatches and multi-day moves.

**How the expectations were made.** Each claim's expected verdict and per-number statuses were
derived by hand from the offline tool outputs, not by running the checker. Both files were committed
before the checker changed (`8d7d868`). The held-out file's sha256 is in the benchmark README, and a
test (`tests/test_claim_bench.py`) fails if the file changes.

**Metrics.**

- verdict accuracy;
- per-check accuracy: status and metric must match, checks are aligned by position, and missing or
  extra checks count as wrong;
- comparator accuracy;
- verdict and status confusion matrices;
- 95% percentile-bootstrap CIs over claims (2,000 resamples, seed 20260926).

```bash
python -m evaluation.claim_bench.run --set dev
python -m evaluation.claim_bench.run --set holdout
```

| Set | Commit | Claims / checks | Verdict accuracy | Per-check accuracy | Comparator |
| --- | --- | --- | --- | --- | --- |
| dev, before the fixes | `3da1a48` | 131 / 147 | 0.527 [0.443, 0.611] | 0.497 [0.404, 0.584] | 0.652 |
| dev, after the fixes | `2fcb4f0` | 131 / 138 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| **held-out, single run** | `2fcb4f0` | 47 / 54 | **0.936 [0.851, 1.000]** | **0.944 [0.880, 1.000]** | 1.000 |
| dev with the 42 round-4 rows (move bounds, relations, x earnings, macro) | `be88027` | 173 / 180 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| **independent round-4 slice, first run** | `817a2d8` | 67 / 83 | **0.716 [0.612, 0.821]** | **0.639 [0.541, 0.730]** | 0.435 |
| independent round-4 slice, **after exposure** | `c731dba` | 67 / 75 | 1.000 [1.000, 1.000] | 0.920 [0.849, 0.974] | 0.522 |
| dev with the 31 round-5 rows (d174-d204) | `c731dba` | 204 / 214 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| dev with the 20 round-8 rows (d205-d224; d082 relabelled) | `b04f364` | 224 / 245 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| held-out, **after exposure** (round 8: h038's class, industry averages, fixed) | `b04f364` | 47 / 54 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| **independent round-5 slice, first run** | `f01097a` | 56 / 86 | **0.821 [0.71, 0.91]** | **0.814 [0.72, 0.90]** | 0.835 |
| dev with the 21 round-9 rows (d225-d245; d102 relabelled) | `2be73d6` | 245 / 277 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| held-out, after exposure, at the round-9 commit | `2be73d6` | 47 / 54 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| independent round-4 slice, after exposure, at the round-9 commit | `2be73d6` | 67 / 75 | 1.000 [1.000, 1.000] | 0.920 [0.849, 0.974] | 0.522 |
| independent round-5 slice, **after exposure** (round 9) | `2be73d6` | 56 / 86 | 1.000 [1.000, 1.000] | 0.988 [0.962, 1.000] | 0.929 |
| dev with the 23 round-10 rows (d246-d268) | `f94df6f` | 268 / 307 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| held-out, after exposure, at the round-10 commit | `f94df6f` | 47 / 54 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| independent round-4 slice, after exposure, at the round-10 commit | `f94df6f` | 67 / 75 | 1.000 [1.000, 1.000] | 0.920 [0.849, 0.974] | 0.522 |
| independent round-5 slice, after exposure, at the round-10 commit | `f94df6f` | 56 / 86 | 1.000 [1.000, 1.000] | 0.988 [0.962, 1.000] | 0.929 |
| dev with the 8 round-11 rows (d269-d276: 10倍多, 将近N) | `4796e24` | 276 / 315 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| held-out, after exposure, at the round-11 commit | `4796e24` | 47 / 54 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| independent round-4 slice, after exposure, at the round-11 commit | `4796e24` | 67 / 75 | 1.000 [1.000, 1.000] | 0.920 [0.849, 0.974] | 0.522 |
| independent round-5 slice, after exposure, at the round-11 commit | `4796e24` | 56 / 86 | 1.000 [1.000, 1.000] | 0.988 [0.962, 1.000] | 0.929 |
| independent round-6 slice, **after exposure (round 11)** | `4796e24` | 67 / 106 | 0.836 [0.746, 0.925] | 0.717 [0.636, 0.802] | 0.284 |
| dev with the 23 round-12 rows (d277-d299; d261 relabelled) | `bd84003` | 299 / 347 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| held-out, after exposure, at the round-12 commit | `bd84003` | 47 / 54 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| independent round-4 slice, after exposure, at the round-12 commit | `bd84003` | 67 / 75 | 1.000 [1.000, 1.000] | 0.920 [0.849, 0.974] | 0.522 |
| independent round-5 slice, after exposure, at the round-12 commit | `bd84003` | 56 / 86 | 1.000 [1.000, 1.000] | 0.988 [0.962, 1.000] | 0.929 |
| independent round-6 slice, **after exposure (round 12)** | `bd84003` | 67 / 104 | 0.985 [0.955, 1.000] | 0.904 [0.835, 0.969] | 0.382 |
| dev with the 12 round-13 rows (d300–d311), at the merged round-13 code | `f07be9b` | 311 / 359 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| held-out, after exposure, at the merged round-13 code | `6dd4650` | 47 / 54 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| independent round-4 slice, after exposure, at the merged round-13 code | `8f7c2c8` | 67 / 75 | 1.000 [1.000, 1.000] | 0.920 [0.849, 0.974] | 0.522 |
| independent round-5 slice, after exposure, at the merged round-13 code | `f84ca52` | 56 / 86 | 1.000 [1.000, 1.000] | 0.988 [0.962, 1.000] | 0.929 |
| independent round-6 slice, **after exposure**, at the merged round-13 code | `13ece57` | 67 / 104 | 0.985 [0.955, 1.000] | 0.904 [0.835, 0.969] | 0.382 |
| independent round-8 claim slice, **after exposure (round 13)** (`score_claims_r8.py`, verdicts only) | `b1099d8` | 25 / – | 1.000 [1.00, 1.00] | – | – |
| **independent round-9 claim slice, first run** (`score_claims_r9.py`, verdicts only) | `cbcf3b2` | 25 / – | **0.600 [0.40, 0.80]** | – | – |
| **independent round-9 claim slice, after the round-13 fixes** (not opened; 5 claims overlap own wording) | `da9f6af` | 25 / – | **1.000 [1.00, 1.00]** | – | – |
| dev (311), at the merged round-14 code | `e2687bc` | 311 / 359 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| held-out, after exposure, at the merged round-14 code | `4240d3b` | 47 / 54 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| independent round-4 slice, after exposure, at the merged round-14 code | `2f45baf` | 67 / 75 | 1.000 [1.000, 1.000] | 0.920 [0.849, 0.974] | 0.522 |
| independent round-5 slice, after exposure, at the merged round-14 code | `90a8028` | 56 / 86 | 1.000 [1.000, 1.000] | 0.988 [0.962, 1.000] | 0.929 |
| independent round-6 slice, **after exposure**, at the merged round-14 code | `6cc4264` | 67 / 104 | 0.985 [0.955, 1.000] | 0.904 [0.835, 0.969] | 0.382 |
| independent round-8 claim slice, **after exposure**, at the round-14 claims code | `0dd9e1e` | 25 / – | 1.000 [1.00, 1.00] | – | – |
| independent round-9 claim slice, **after exposure (round 14)**, at the round-14 claims code | `0dd9e1e` | 25 / – | 1.000 [1.00, 1.00] | – | – |
| **independent round-10 claim slice, first run** (`score_claims_r10.py`, verdicts only) | `7f4f06c` | 24 / – | **0.667 [0.46, 0.83]** | – | – |
| **independent round-10 claim slice, after the round-14 fixes** (not opened by the engineers' statement; no claim near-copies own wording, 5 repeat own content: 16/19 without them) | `3480a96` | 24 / – | **0.833 [0.67, 0.96]** | – | – |
| dev (311; d022, d216, d208 relabelled for the round-15 no-date rule), at the round-15 claims code | `a02821d` | 311 / 359 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| held-out, after exposure, at the round-15 claims code | `a02821d` | 47 / 54 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 |
| independent round-4 slice, after exposure, at the round-15 claims code | `a02821d` | 67 / 75 | 1.000 [1.000, 1.000] | 0.907 [0.836, 0.963] | 0.522 |
| independent round-5 slice, after exposure, at the round-15 claims code | `a02821d` | 56 / 86 | 0.964 [0.911, 1.000] | 0.942 [0.884, 0.988] | 0.929 |
| independent round-6 slice, **after exposure**, at the round-15 claims code | `a02821d` | 67 / 104 | 0.970 [0.925, 1.000] | 0.894 [0.821, 0.961] | 0.382 |
| independent round-8 and round-9 claim slices, **after exposure**, at the round-15 claims code | `a02821d` | 25 / – each | 1.000 [1.00, 1.00] each | – | – |
| independent round-10 claim slice, **after exposure (round 15)** | `a02821d` | 24 / – | 0.875 [0.75, 1.00] | – | – |

The commit column is the commit each run started from (`config.commit`). The merged-code runs were committed one after
another, so each starts from the previous result commit; the code is the same in all of them.

The result files are `evaluation/results/claim_bench-dev-baseline.json`, `claim_bench-dev.json`,
`claim_bench-holdout.json` (the single first run), `claim_bench-holdout-after-round8.json` (the same file after
exposure, at the round-8 commit), `claim_bench-heldout_r4-first-run.json`, `claim_bench-heldout_r4-after-exposure.json`, and for round 9
`claim_bench-holdout-after-round9.json`, `claim_bench-heldout_r4-after-round9.json`, `claim_bench-heldout_r5-first-run.json`
and `claim_bench-heldout_r5-after-exposure.json`, and for round 10 `claim_bench-holdout-after-round10.json`,
`claim_bench-heldout_r4-after-round10.json` and `claim_bench-heldout_r5-after-round10.json`. Each records the commit, the
command and the sha256 of the claims file. The round-10 rules changed no status on the held-out set or on either slice:
none of their claims uses a stated value of the compared side, a stated difference or a bounded 多/出头 numeral that
round 9 had read differently. A fresh held-out slice is needed to measure the round-10 rules.

Round 11 (G12) re-ran every claim set at `4796e24`: `claim_bench-holdout-after-round11.json`,
`claim_bench-heldout_r4-after-round11.json`, `claim_bench-heldout_r5-after-round11.json` and
`claim_bench-heldout_r6-after-exposure-round11.json`. The round-6 slice run is labelled **after exposure (round 11)**:
the round-10 engineers never saw that slice, but the round-11 engineers could read it, so only the round-10 run
(`claim_bench-heldout_r6-after-fix.json`, 0.836) is out of sample. The new N倍多 / 将近N rules changed no status on any of
these sets (every number above equals its round-10 value); their effect shows on the 8 dev rows d269-d276.

Round 12 (H6) re-ran every claim set at `bd84003`: `claim_bench-holdout-after-round12.json`,
`claim_bench-heldout_r4-after-round12.json`, `claim_bench-heldout_r5-after-round12.json` and
`claim_bench-heldout_r6-after-exposure-round12.json`. The held-out, round-4 and round-5 numbers are unchanged. The
round-6 slice moved 0.836 → 0.985 (66 of 67; r6c10, a stated average read as a multiple, is still wrong) and its
check accuracy 0.717 → 0.904. It is **after exposure**: the slice and the reviews' per-claim findings were readable,
and several round-12 forms (English differences, 一成半, 一千四百出头, 一万二千多亿, the bracketed average) are its
classes. The round-8 reviewer's claim slice (`heldout_r8`, not opened) is the out-of-sample measure of these rules.

Round 13 (I8) re-ran every claim set at `39ba98e` on the r13-claims branch: dev 1.000 on 299 claims, held-out 1.000,
round-4 1.000 (checks 0.920), round-5 1.000 (checks 0.988) and round-6 0.985 (66 of 67, r6c10 still wrong): every
number equals its round-12 value, so the new rules changed no status on these sets. After the six round-13 branches
were merged, every set was re-run at the merged code; `claim_bench-dev.json` (311 claims with the r13-frame rows),
`claim_bench-holdout-after-round13.json`, `claim_bench-heldout_r4-after-round13.json`,
`claim_bench-heldout_r5-after-round13.json` and `claim_bench-heldout_r6-after-exposure-round13.json` now hold those runs,
with the same values (table above). The round-8 reviewer's claim slice moved 0.800 → **1.000** (25 of 25,
`claims_heldout_r8-after-exposure-round13-claims.json` at `39ba98e`, and `claims_heldout_r8-after-exposure-round13.json`
with every round-13 fix at `b1099d8`). That run is **after exposure**: the committed round-12 result
(`claims_heldout_r8-after-fix.json`) lists its five remaining errors (两倍多一点, 三个多百分点, "26% below the insurance
average", "less than half of Moutai's net profit", 是保险行业的四倍多), and the round-13 engineers read them; the rules
were tested with other wordings, but the 1.000 is not an out-of-sample estimate. The out-of-sample measure of the
round-13 rules is the round-9 reviewer's slice (`evaluation/heldout_r9/`, 25 claims, not opened by the engineers): 0.600
[0.40, 0.80] on its first run at `cbcf3b2` (by the reviewer; `claims_heldout_r9-prefix.json`, errors in X倍半, fractions
of another value, loose fractions, superlatives, point differences, "times more" and one 52-week reference) and
**1.000** (25/25) after the round-13 fixes at `da9f6af` (`claims_heldout_r9-after-fix.json`). Five of its claims (r9c01,
r9c08, r9c12–r9c14) are near copies of own test wording or of the review's published repros
(`slice_overlap-heldout_r9-at-da9f6af.json`); all five failed on the first run, and without them the slice goes 15/20
→ 20/20. From round 14 this slice is after exposure.

Round 14 re-ran every claim set once at `0dd9e1e` (clean tree, the r14-claims branch): `claim_bench-dev.json`,
`claim_bench-holdout-after-round14.json`, `claim_bench-heldout_r4-after-round14.json`,
`claim_bench-heldout_r5-after-round14.json`, `claim_bench-heldout_r6-after-exposure-round14.json`,
`claims_heldout_r8-after-exposure-round14.json` and `claims_heldout_r9-after-exposure-round14.json`; after the five
round-14 branches were merged, the first five were re-run at the merged code (`e2687bc`, `4240d3b`, `2f45baf`,
`90a8028`, `6cc4264`) and the files now hold those runs, with the same values. Every number equals
its round-13 value (dev 311 = 1.000, held-out 1.000 / 1.000, r4 1.000 / 0.920, r5 1.000 / 0.988, r6 0.985 / 0.904 with
r6c10 still wrong, r8 and r9 25/25), so the round-14 rules changed no status on these sets. The real published claims
moved (both sets after exposure): `claims_web_v1` 0.800 → 0.850 on seed data and 0.817 → 0.783 with the corrections
layer (its labels ignore a named definition; the checker no longer does), `claims_web_v2` 0.500 (blind first run) →
0.840 ([per class](../evaluation/real_claims_web/README.md#round-14-after-exposure-checker-changed),
[v2](../evaluation/real_claims_web_v2/README.md#round-14-after-exposure-checker-changed)).

The out-of-sample measure of the round-14 rules is the round-10 reviewer's claim slice (`evaluation/heldout_r10/`, 24
claims written at `7f4f06c` before any round-14 fix, on the shipped default data; not opened by the engineers by their statement, the after-fix run read it
from the reviewer's folder; no claim near-copies own wording, but the round-15 content-level scan finds 5 that repeat
own content, r10c02, r10c08, r10c09, r10c10 and r10c16, and without them it is 15/19 → 16/19,
`slice_overlap-heldout_r10-content-at-3480a96.json`): 0.667 [0.46, 0.83] on its first run (by the reviewer; `claims_heldout_r10-prefix.json`) and
**0.833 [0.67, 0.96]** (20/24) after the round-14 fixes at `3480a96` (`claims_heldout_r10-after-fix.json`). Fixed: both
扣非 claims (r10c01, a false accept, and r10c02), both 不到…的N倍 bounds (r10c10, r10c11) and 流通市值 (r10c16). Still
wrong: r10c06 ("Of ICBC, CMB and CITIC Securities, CITIC has the lowest P/E": unverifiable, labelled contradicted),
r10c08 ("rose about 62% year on year to RMB51.8 billion": partially supported, labelled supported), r10c12
("工商银行一年的净利润抵得上两个多招行": unverifiable, labelled supported), and one regression: r10c07
("2025年宁德时代实现归母净利润约722亿元，同比增长42%") was supported on the first run and is partially supported after
the round-14 fixes. From round 15 this slice is after exposure.

Round 15 re-ran every claim set once at `a02821d` (clean tree, the r15-claims branch, after round 2 was merged in):
`claim_bench-dev.json`, `claim_bench-holdout-after-round15.json`, `claim_bench-heldout_r4-after-round15.json`,
`claim_bench-heldout_r5-after-round15.json`, `claim_bench-heldout_r6-after-exposure-round15.json`,
`claims_heldout_r8-after-exposure-round15.json`, `claims_heldout_r9-after-exposure-round15.json` and
`claims_heldout_r10-after-exposure-round15.json`. Dev 311 = 1.000 / 1.000 (three rows relabelled for the no-date rule),
held-out 1.000 / 1.000, r8 and r9 25/25. The no-date rule costs the older slices: r4 1.000 / per-number 0.920 → 0.907
(r4c011's move), r5 1.000 → **0.964** (r5c024, r5c043) and per-number 0.988 → 0.942, r6 0.985 → **0.970** (r6c51) and
0.904 → 0.894. Each of those claims names no date and no latest session and does not fit the latest session, so it is
now unverifiable where the slice labels it contradicted. The round-10 slice, after exposure from round 15, moved 20/24 →
**21/24** (r10c07, the definition-caution class, is fixed; r10c06, r10c08, r10c12 still fail).

Real published claims, all **after exposure** at `a02821d`: `claims_web_v1` 0.850 (51/60, unchanged; w04 and w56, the
changes in 元 / 点, are fixed, w20 and w59 are new errors), with corrections 0.783 → **0.817** (49/60);
`claims_web_v2` **0.840** (42/50, unchanged in count: v34 and v44 fixed, v03 and v21 now `supported` under the
definition-caution rule while the set's labels say partially); `claims_web_v3` 0.540 (blind first run) → **0.900**
(45/50; 0.860 against the definition-blind labels); and a fourth blind set, `claims_web_v4` (collected on this branch by
a search agent that did not read the checker), **0.740 [0.62, 0.86]** on its blind first run at `110e656` (the K6
checker, before the third set's rules) → 0.780 (39/50) at `a02821d`
([v1](../evaluation/real_claims_web/README.md#round-15-after-exposure-checker-changed),
[v2](../evaluation/real_claims_web_v2/README.md#round-15-after-exposure-checker-changed),
[v3](../evaluation/real_claims_web_v3/README.md#round-15-after-exposure-checker-changed),
[v4](../evaluation/real_claims_web_v4/README.md)).

```bash
python -m evaluation.claim_bench.run --claims evaluation/heldout_r4/claims_moves_heldout.jsonl \
  --out evaluation/results/claim_bench-heldout_r4-after-exposure.json
FINSIGHT_REPO=. python evaluation/heldout_r8/score_claims_r8.py <out.json>
```

### How to read these numbers

- **The dev score is not an estimate of accuracy.** The checker was developed against the dev claims
  until all 131 passed. The held-out 0.936 (CI 0.85–1.00, n = 47) is the honest number, and even it
  is small and narrow in scope.
- **The held-out run found three errors:**
  - h011 "五粮液昨日收跌0.53%": "收跌" was read as an up move. This was fixed after the run in
    `d91051c`, so the held-out set is no longer untouched for that fix, and the committed held-out
    result is from before it.
  - h038 "中国平安市盈率8.7倍，而行业平均11.8倍": the industry average is checked against the company's
    P/E. Since round 8 an industry average is checked against the industry snapshot (fixed after the review
    named it, so the held-out set is exposed for this class; the committed first-run result is unchanged).
  - h039 "Kweichow Moutai trades at 24.6 times earnings": "times earnings" is not recognised as P/E.
- **The round-4 slice (written by a separate author) is exposed.** Its first run (0.716) is the honest
  number for the checker as it was. The round-6 rules above were written after reading its 30 errors, with
  new own dev rows, so its 1.000 after exposure shows that the classes are covered, not that the checker
  generalises. What is left on it:
  - 6 claims (macro and no-data) carry checks the slice does not expect (it labels CPI/PMI/10Y outside its
    metric vocabulary and no-data concepts with no checks), which is why check accuracy is 0.920 with every
    verdict right;
  - comparator accuracy 0.522 measures a difference in convention, not in status: the slice writes a bound on
    a move on the signed change ("跌超1%" as `le` −1, "跌了不到1%" as `range`), this checker on the size of the
    move with a direction (`gt` 1 down, `lt` 1 down), and the slice treats "跌超" as non-strict. The statuses
    agree.
- **The round-5 slice (a separate author, committed before its run) is exposed too.** Its first run at `f01097a`
  (0.821 [0.71, 0.91], n = 56) is the honest number for the checker before round 9; the failures were stated
  industry averages (industry_average 0.727), ratio phrasings ("1.5倍还多", "两倍有余", "六成", "more than double":
  ratio 0.6), a relation after its clause's number, "更活跃", a number bound to the wrong company ("平安ROE" after
  "中国平安"; "均值" read as 武汉天源) and the ETF daily change it labels derivable from the closes. Round 9 fixed these
  classes with new own dev rows, so its 1.000 after exposure shows the classes are covered, not that the checker
  generalises; a new slice is needed for an honest estimate. Its check accuracy is 0.988 because r5c049 (market cap)
  gets an unverifiable check the slice does not expect; comparator accuracy 0.929 is six checks whose status agrees but whose comparator does
  not: "两倍多", "两倍有余", "7倍多" read as `gt` N (the slice writes `range` [N, N+1)), "10倍以上" as `ge` (slice `gt`),
  "是茅台的三倍" as `eq` (slice `approx`), and "turnover topped CNY 2 billion" as `eq`, a real miss: "topped" is not yet a
  comparator word (the status is right only because 14.53亿 is not 20亿 either way).
- **Both sets use the same offline snapshot, companies and author.** The snapshot covers 3 stocks, 3
  ETFs and 1 index. The claims are the project author's paraphrases of common broker and social-media
  phrasings, not a sample of real posts, so accuracy on real traffic will be lower.

## Limits

- **Coverage of the data.**
  - Offline: the v1 snapshot has 3 stocks, 3 ETFs and 1 index; the snapshot extension (on by default) adds 15
    instruments as of 2026-09-30, 10 of them companies with fundamentals but **no industry snapshot**, so an industry
    relation for them is unverifiable ("no industry snapshot").
  - Offline (v1 snapshot): growth, index valuation, dividend yield, market cap, debt ratio and EPS are unavailable;
    the extension has growth, EPS, total and free-float (流通) market value and 扣非净利润 for its ten companies.
  - Definitions (round 14): a claim naming a definition the data does not serve is unverifiable, even when the
    company's report shows the two give the same figure (长江电力's 营业收入 = 营业总收入); the data does not state such
    equalities.
  - Live: coverage depends on the providers.
- **Only the daily change is checked.** Multi-day moves and comparisons with the previous session ("跌幅扩大")
  are unverifiable; "涨停" is checked only as "up".
- **Qualitative move words are a convention.** 大跌 / 大涨 ≥ 3% and 小幅 < 1% are this project's thresholds, stated
  in each check's note; a reader with other thresholds can disagree near them. Words outside the table (e.g. 下挫
  alone) are plain moves.
- **Dates.** Only explicit month-day dates are compared with the trade date; 昨天 / 今天 are not resolved against
  the snapshot date.
- **Chinese numerals.** Only simple ones before a unit are handled: 十五倍, 一点一倍, 三成, 百分之三十, and since round 10
  with 多 / 余 before the unit (一千六百多亿, 八百余亿). Since round 9 shares of another value are multiples: 的三分之一,
  的六成, 的64%, 的一半. Since round 12 also 一成半, 万 inside a numeral (一万二千多亿) and a numeral before 出头 with no
  unit (一千四百出头). Ambiguous forms are not handled: 十几倍, 上千亿.
- **Comparator reading is lexical.** Sarcasm and rhetorical questions are not understood.
- **Relations.** Two named targets, a target and its industry snapshot, or a target and a named sector with a
  snapshot (白酒, 保险, 券商 offline) are compared; peers, consensus and the market average are not. Sector
  names in English are recognised for baijiu/liquor, insurance, brokerage/securities and banking only. An
  industry average with a number ("而行业平均11.8倍") is checked against the target's industry snapshot since
  round 8; peers named without "平均/均值/中位数" ("同行11.8倍") are not.
- **Macro.** Only the latest reading of each series is available: changes from the previous reading and
  readings for other months are unverifiable.
- **Periods.** Named periods are checked (年份, 一季度, 上半年, 前三季度, FY, H1). Relative ones (去年,
  上季度) are not resolved, and an amount without a period is unverifiable when the latest report is
  an interim one.
- **Tolerance.** 2% or the written precision (for "about": 5%, or half the step of the last significant digit) is a
  policy choice. "约30倍" accepts 25-35 since round 10 (24.6 is still contradicted), and "接近9倍" against 8.7 is
  supported. The step reads trailing zeros as placeholders only for approximations, bounds with 多 / 余 / 出头; a bare
  "30倍" is exact to half a unit (or 2%). Fractions have no step: "约为三分之一" allows 5% (0.354 against 1/3 is 6% off
  and contradicted, while "三分之一左右" at 0.319 is supported).
- **Ratio or stated value.** (Round 14: the bound words 不到 / 不足 / 不及 / 超过 / 超出 / 至少 / 至多 / 最多 are ratio cues
  now, so "市盈率不到五粮液的两倍" is a multiple; the text below describes 低于 / 高于 / 比, which are unchanged.)
  For a metric quoted in 倍 or % (P/E, P/B, ROE, margins), "X的N倍" / "X的N%" after a bound or
  比 is X's stated value unless a ratio cue is written (是 / 为 / 相当于 / 只有…的, 比…的N倍还…, a fraction). "市盈率不到五粮液
  的两倍" is therefore read as "五粮液's P/E is 2x" (contradicted, next to the relation); with a cue ("市盈率只有五粮液的1.2倍")
  it is a multiple. The rule follows the unit of the metric, not the size of N, so it never depends on the data.
- **Differences.** "高出N个百分点 / 高出N倍" is a difference only when the unit fits the metric; "%" on a metric quoted in
  percent (ROE) is read as percentage points, on any other metric as a relative difference. "营收高出两倍" is
  unverifiable (two or three times?); since round 12 "高出一倍(多)" is a relative difference of 100% (and more), and
  English "beats / trails / exceeded X's by N" and "N billion yuan below X's" are read. "比保险行业平均的12倍还高" is
  read as a multiple (还 after 比 is a ratio cue); the round-6 slice labels it as a stated average (r6c10, still
  counted wrong).
- **Sums and evaluations are lexical (round 12).** A sum needs a sum word and both companies named; the evaluation word
  that turns a stated average into a relation is a short list (偏低 / 偏高 / 更低 / 便宜 / cheaper / at a discount …).
