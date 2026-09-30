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
silently. The verdict is computed over `checks` only.

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
| `approx` | 约, 大约, 接近, 将近, …左右 | about, around, roughly, nearly | within 5% |
| `gt` | 超过, 高于, 大于, 逾, 突破, 站上, 30多倍 | above, over, more than, exceeds | actual > claimed |
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
average", `evidence_id: industry_保险`, as-of the snapshot's trade date). After a bound it stays the company's
bound: "市盈率低于行业平均11.8倍" checks the company's P/E below 11.8. "…，低于行业均值" (no number) is a relation
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
| `no_data` | the source has no value for the target and metric (a company outside the offline snapshot, an ETF with no daily change, index P/E, index turnover reported as 0, dividend yield, market cap) |
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

The result files are `evaluation/results/claim_bench-dev-baseline.json`, `claim_bench-dev.json`,
`claim_bench-holdout.json` (the single first run), `claim_bench-holdout-after-round8.json` (the same file after
exposure, at the round-8 commit), `claim_bench-heldout_r4-first-run.json` and `claim_bench-heldout_r4-after-exposure.json`.
Each records the commit, the command and the sha256 of the claims file.

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
- **Chinese numerals.** Only simple ones before a unit are handled: 十五倍, 一点一倍, 三成, 百分之三十.
  Ambiguous forms are not handled: 两成多, 十几倍, 上千亿.
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
- **Tolerance.** 2% or the written precision (5% for "about") is a policy choice. "约30倍" against
  24.6 is contradicted, while "接近9倍" against 8.7 is supported.
