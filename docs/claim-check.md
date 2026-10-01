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
| `no_data` | the source has no value for the target and metric (a company outside the offline snapshot, an ETF with no daily change and no two closes ending at its quote date, index P/E, price-to-sales, max drawdown, PEG without a growth rate, index turnover reported as 0, dividend yield, market cap) |
| `growth_unavailable` | a YoY growth claim, but the fundamentals payload has no YoY field |
| `unit_mismatch` | the unit does not fit the metric |
| `no_unit` | an amount with no unit |
| `forecast` | 预计, 将, 会, 明年, 目标价, will, expected, if, … |
| `period_mismatch` | the claim names a year or period other than the report's; or an amount with no period is compared with an interim (Q1/H1/Q3, year-to-date) report |
| `multi_day` | 今年以来, 近一个月, 本周累计成交, this year, … (only the latest daily change and turnover are checked) |
| `no_reference` | a relation with something no source provides, such as the market average |

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

```bash
python -m evaluation.claim_bench.run --claims evaluation/heldout_r4/claims_moves_heldout.jsonl \
  --out evaluation/results/claim_bench-heldout_r4-after-exposure.json
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
  - Offline: only 3 stocks, 3 ETFs and 1 index have data.
  - Offline: growth, index valuation, dividend yield, market cap, debt ratio and EPS are unavailable.
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
- **Ratio or stated value.** For a metric quoted in 倍 or % (P/E, P/B, ROE, margins), "X的N倍" / "X的N%" after a bound or
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
