# Real published claims, third set, blind (`claims_web_v3.jsonl`)

Real claims as they were published about the companies, metrics and dates in FinSight's offline data. Sources are
media, brokers' public notes, company announcements and social posts (雪球, 东方财富财富号, 知乎, 搜狐号, 新浪看点,
Substack, Instagram). The set measures the claim checker (`POST /agent/claim-check`) on wording that nobody wrote for
it. Like the second set (`evaluation/real_claims_web_v2/`), it is **blind to the checker**: it was collected, labelled
and committed before the engineer who built it read the checker's code, its tests or any review report.

**What this set is not.** It is not the owner's real-claims set (`evaluation/human/real_claims/`). That set is still
owner-blocked: the owner collects 30–50 claims they actually met, labels them, and scores them with
`evaluation/human/import_real_claims.py`. This set was collected by the engineer's search agents.

## Files

| file | what |
|---|---|
| `candidates_v3.jsonl` | the 126 candidates returned by the five search agents |
| `select_claims_web_v3.py` | the fixed selection rules, candidates → 50 claims |
| `collected_v3.jsonl` | the 50 claims as collected, before any label (committed first) |
| `label_claims_web_v3.py` | computes every expected verdict from the data files (committed with the labels, before the checker was run) |
| `claims_web_v3.jsonl` | the collected fields plus labels, parts and the primary-source comparison |
| `score_claims_web_v3.py` | runs the checker once and writes `evaluation/results/claims_web_v3-first-run.json` |

## Protocol (written before labelling)

**Blind to the checker.** Before every claim was collected, labelled and committed, the engineer who built this set did
not open any of the following:

- `query_intelligence/`, in particular `query_intelligence/agent/claim_check.py`;
- the claim-bench rows or tests;
- the first two sets' labelling and scoring scripts;
- any review report.

Before collecting, the engineer read only these:

- the data files: `data/structured_data.json`, `data/snapshot/structured_data_ext.json`,
  `data/snapshot/structured_data_fix.json`, `data/snapshot/manifest.json`, `data/snapshot/manifest_fix.json` and
  `data/snapshot/corrections/sources.json`;
- the protocol, labelling-rule, selection and label-format sections of `evaluation/real_claims_web_v2/README.md`, but
  not its "First run" and "Error classes" sections or anything after them;
- `collected_v1.jsonl`, `collected_v2.jsonl` and `candidates_v2.jsonl`, for their texts and URLs (the do-not-use list).

**Exposure that remains.**

- The v2 protocol section names the first set's error classes in one sentence:
  - 分别 pairing, 元/点 changes, 报收 and "2025 年";
  - traditional characters, bracketed comparands and two-metric ratios;
  - prior-year gaps and the 2% tolerance.

  The engineer knew them. The second set's error classes were not read.
- The engineer opened the first 80 lines of `evaluation/real_claims_web_v2/score_claims_web_v2.py` (the docstring and
  the bootstrap/rate helpers) before collecting. The task allowed reading that scorer for step 4. It shows how the
  checker is called, not how it decides.
- The engineer saw the offline values. The search agents did not.

The set is therefore blind to the checker's code and to the second set's error classes. It is not blind to the first
set's error classes or to the data.

**Collection.** Done 2026-10-11 (every `retrieved` field) by five search agents with fresh contexts, using Tavily and
WebSearch/WebFetch; Brave failed.

- Each agent covered a list of names and was not allowed to open any repository file.
- The agents were told which companies, metrics and dates can be checked, and to **not select by correctness**:
  rounded, loose, comparative and wrong numbers were all wanted. They were **not** given any offline value or any
  error class.
- They were asked for relations between companies, sums, averages, shares, growth wording, loose numbers, English
  claims and a mix of source types.
- Agents A (宁德时代, 比亚迪, 美的, 紫金, 中芯), B (the banks, 中信证券, 长江电力, 中国平安) and C (茅台, 五粮液, indices and
  ETFs) ran first.
- Agent D filled gaps: 2026-04-22 ETF and index bars, 2026-09-30 stock bars, and sums and shares.
- Agent E looked only for sums, averages, shares and multi-company comparisons.
- Each agent confirmed the sentence on its page (`verified_by`: `tavily_extract`, `webfetch`, or `search_snippet`
  where the page did not extract).
- The do-not-use list held every URL and sentence of `collected_v1.jsonl`, `collected_v2.jsonl` and
  `candidates_v2.jsonl`. Agents D and E also had the earlier agents' candidates on it.

**Checkable scope (from the data files, listed before searching).**

- **FY2025 annual figures** (report date 2025-12-31): revenue, 归母净利润, 扣非净利润, revenue YoY and net-profit YoY,
  ROE, gross margin, net margin (net profit / revenue) and EPS.
  - The ten extension companies are 宁德时代, 比亚迪, 美的集团, 紫金矿业, 中芯国际, 招商银行, 工商银行, 平安银行,
    中信证券 and 长江电力. Gross margin is missing for the three banks and for 中信证券.
  - The corrections layer serves 贵州茅台, 五粮液 and 中国平安 from their annual reports.
  - Each field's basis is stated in `structured_data_fix.json` (`definitions`):
    - ROE is period-end for the ten extension companies and weighted for the three corrected ones.
    - EPS is diluted for 美的 and 紫金, basic elsewhere.
    - Revenue is 营业总收入 for 美的, 营业收入 elsewhere.
    - Net-profit YoY is on total net profit for eight extension companies, and on 归母 for 平安银行, 中芯国际 and the
      three corrected ones.
- **Valuation** (总市值, PE TTM, PB): on 2026-09-30 for the ten extension companies, and on 2026-04-22 for 茅台, 五粮液
  and 中国平安.
- **Daily bars** (close, open, high, low, % change, change in 元 / 点, 成交额, volume):
  - on 2026-09-30 and 2026-04-22 for the ten extension stocks and for 上证指数, 创业板指, 中证500ETF, 黄金ETF and
    科创50ETF. The extension's daily history runs from 2025-07-01 to 2026-09-30. The indices have no 成交额.
  - on 2026-04-22 for 茅台, 五粮液, 中国平安, the 沪深300 index, 沪深300ETF, 创业板ETF and 证券ETF.
- **Excluded while searching:** quarterly and half-year figures, forecasts and targets, prices on other dates, live
  quote widgets, market-wide turnover and northbound flows.

**Quoting.** Each claim is one sentence or clause, with source name, URL, publication date and retrieval date. No
article text beyond the claim sentence is stored.

- Shortening drops whole clauses only, and such claims are flagged `excerpt`.
- An implicit subject ("公司", "该股") is inserted in square brackets (`subject_inserted`). The checker gets the text
  without the brackets.
- A date the sentence leaves to its context is recorded in `date_implicit` and `date_source`.
- Three FY2025 claims had a publication-date estimate in that field. It was moved to `published_estimate`.

## Selection (126 candidates → 50), before any label

The five agents returned 126 candidates (`candidates_v3.jsonl`: A 31, B 28, C 28, D 23, E 16). By source: media 64,
social 42, broker 12, company 8. By language: zh 97, en 27, zh-Hant 2.

`select_claims_web_v3.py` applies fixed rules that never read an offline value. First it drops 8 candidates:

| rule | dropped |
|---|---|
| not a claim sentence (a market-review table row) | cD01 |
| a second copy of another candidate's sentence | cD05 (= cB02) |
| an intraday value, not a closing or daily-bar figure | cD02 |
| about a post-holiday session; the 09-30 figure is only implied | cD08 |
| a live valuation page ("当前市盈率"), so the date is not fixed | cB07 |
| a 雪球 stock news feed, not a stable page | cB06 |
| period not stated (the agent assumed "the latest full year") | cE14, cE15 |

No candidate repeats a v1 or v2 sentence or URL (checked by URL and by normalised-text similarity). Two candidates come
from documents that a v2 claim also used, but each quotes a different sentence:

- cD06 is a 搜狐 reprint of the 财中社 article behind v49.
- cA20 is the 中芯国际 annual-report letter behind the first set's w18.

**Grouping and form.**

- Candidates are grouped by their first-named entity: the 13 companies, plus indices and ETFs, each split by date.
- Within a group, each candidate gets a form class:
  - `rel`: a relation between companies, a sum, an average, a share or a derived ratio, read from the wording;
  - `market`: a period other than FY2025;
  - `annual`: everything else.

**Order and caps.**

- Within each class, candidates with a known publication date come first. Ties are broken by a fixed pseudo-random key
  (sha256 of URL + text).
- The groups are filled round-robin over the classes (rel, annual, market). This spreads the forms. It is blind to
  correctness.
- At most 4 claims are taken per group and 2 per URL. The two `bbntimes.com` URLs of one article count as one URL.
- The list is trimmed to 50 by dropping the last pick from the largest group, with ties broken by sha256 of the group
  name.

The engineer added the index and ETF split by date after a first dry run of the rules: that run gave only 2 claims
dated 2026-04-22. The split was made before any label and it reads no value.

## Counts

- **Claims:** 50, of which 42 are zh and 8 en. 19 of the 42 Chinese claims are 40 characters or fewer. The longest
  runs to 82 characters (x28, an announcement's run-on sentence).
- **Source types:**
  - media 26;
  - social 16 (东方财富财富号 / 股吧 6, 雪球 3, 知乎, 搜狐号, 澎湃号, 新浪财经头条, 新浪看点, Substack and Instagram 1
    each);
  - broker notes 4;
  - company announcements 4.
- **Periods:** FY2025 annual figures 31, 2026-09-30 prices and valuation 14, 2026-04-22 prices and valuation 5.
- **Verification:** 33 claims by `tavily_extract`, 15 by `search_snippet` and 2 by `webfetch`. 21 are excerpts and 17
  have an inserted subject.
- **Publication date not shown on the page:** 11 claims. Each candidate's `note` gives what is known.
- **Not found by the agents:** 2026-04-22 bars for 沪深300ETF, 证券ETF, 中证500ETF and 黄金ETF; 2026-04-22 prices
  for 五粮液 and 中国平安; any sum of two of the 13 companies that stays inside the 13.

## Labelling rules (written before labelling)

The tolerance rules are the same as v2's. Additions are marked **new** and give a reason. None of them changes a v2
rule.

Labels come from `label_claims_web_v3.py`, which reads only the data files above. A person writes each claim's
*parts*: the company, the metric, the number as written, a qualifier and the basis the sentence names. Every value a
part is compared with is computed from the files. Each stated number is one part.

**Tolerance.**

- **Exact figure** ("4237.02亿元", "1.50%", "5.8元"): true when |claimed − actual| ≤ max(half a unit of the claim's last
  written digit, **1% of the actual value**). A growth rate or a change must also have the right sign.
- **约 / 左右 / about / approximately / around / ~**: within 5% of the claimed number.
- **Bounds:**
  - **突破 / 站上 / 达到…以上** N: actual ≥ N.
  - **超 / 超过 / 逾 / 以上 / more than / over / above** N: actual > N.
  - **近 / 将近 / 接近 / nearly** N: 0.9 N ≤ actual ≤ N.
  - **不到 / 不足 / 低于 / less than** N: actual < N.
  - **N 多 / N 余**: N < actual < N + one unit of the last written digit.
  - For a signed quantity (growth, change), the bound applies to the magnitude and the sign must match.

**Relations.**

- "A 高于 / 超过 B" is true when A > B, with both values from the data. "A 仅次于 B" is true when A < B (the ranking
  among other companies is context).
- "A 是 B 的 N 倍" is the ratio A / B, under the rules above.
- A gap ("比 B 多 X") is A − B, under the rules above.

**Derived values.**

- Net margin = net profit / revenue.
- Prior-year value = value / (1 + YoY).
- Change in 元 / 点 = close − previous close in the daily history, or close − close / (1 + pct) where there is no
  previous bar.
- A fund's % change, when the bar leaves it empty, comes from its last two closes.
- **(new) Sums, averages and shares** are computed from the served values, and the claim's own wording rule applies to
  the result:
  - a sum is A + B;
  - a daily average ("日赚", "每天净赚", "平均每日") is the annual figure / 365 (2025 has 365 days);
  - a share is part / whole.

  v2 had no such claims. This set was collected to contain them.
- **(new) Bars on other dates.** A part about another trading day (for example "较上一交易日增加 8.28 亿元") is
  computed from the daily history when the bar is there, and is null otherwise. The extension's history is part of
  the offline data, and v2 had no such part.

**Definitions (as in v2).** The corrections layer states which basis each field is on.

- A sentence that uses the generic word (ROE, 净利润, 利润, 赚, 每股收益, 营收 / 收入 / 营业收入, revenue, profit,
  net income) is compared with the served value, whatever its basis.
- A sentence that **names** a basis different from the served one is **not decidable from the offline data**: the part
  is null, not false. Examples are 加权平均 or 平均净资产收益率 (ROAE) against a period-end ROE, 基本 against diluted
  EPS, 营业总收入 against 营业收入, and 归母净利润 growth against a total-net-profit YoY.
  - The exception: the annual report in `sources.json` shows that the two bases give the same figure for that company
    (for example 营业收入 = 营业总收入 for 长江电力, or basic = diluted EPS for 比亚迪).
  - "扣非 / 扣非归母 / non-recurring net profit" is the served `profit_dedt` (归母 扣非).
  - "average ROE" in English is read as naming the weighted-average basis.
- The file also carries `expected_verdict_definition_blind`, the label with every named basis ignored (the first set's
  rule), so the checker can be scored both ways.

**Not in the offline data (null).**

- Market data: 换手率, fund size (规模), USD or HKD amounts, other indices (深证成指), 成交额 of an index, valuation on
  a date other than the stated valuation date.
- Annual data: quarterly figures, prior-year margins or ROE, changes in percentage points, 扣非 YoY, dividends, R&D,
  unit sales, fee income, operating profit.
- Industry and other-company context: industry totals and shares of industry, figures for companies outside the 13,
  per-employee figures, historical years before 2024.

**Verdict.** The aggregation is the one the first set's README documents for the checker:

- every part true → supported;
- no part true but at least one false → contradicted;
- some true, others false or null → partially_supported;
- nothing decided → unverifiable.

**Context without a number of its own** (创历史新高, 位列第一, 人气排名, 高于行业中位数) is not a part. It is listed in
`context_not_checked`.

**Labels are against the offline data, not against the company's report.** For every annual-figure part, the script
also compares the offline value with the annual report's figure on the claim's basis. It records this in `primary`
(`offline_vs_primary` = agree / definition_differs / value_differs, and `claim_vs_primary`). Price parts have no
independent primary record in the data files. Where a published price differs from the offline bar, it is noted.

**Categories** (one per claim, first match wins):

1. `english_or_traditional`: an English or traditional-Chinese sentence.
2. `loose_or_relational`: a `rel` form, or any part with a loose qualifier (约, 近, 超, 突破, N多, …).
3. `market_value`: a valuation part (总市值, PE, PB).
4. `daily_price`: a daily-bar part.
5. `annual_ratios`: only ratio parts (YoY, ROE, margins).
6. `annual_figures`: everything else.

## Labels (`claims_web_v3.jsonl`), committed before the checker was run

```bash
python -m evaluation.real_claims_web_v3.select_claims_web_v3   # candidates_v3.jsonl -> collected_v3.jsonl (deterministic)
python -m evaluation.real_claims_web_v3.label_claims_web_v3    # collected_v3.jsonl -> claims_web_v3.jsonl (deterministic)
```

Each row is the collected fields plus the following:

- `category`, `expected_verdict` and `expected_verdict_definition_blind`;
- `parts`, one per stated number. Each has the reading, `holds` (true / false / null), the offline `actual`, and the
  rule applied or the reason the part is null. `definition_blind_parts` lists the parts that change when every named
  basis is ignored;
- `context_not_checked`;
- `primary`, per annual-figure part: the basis the claim's figure is on (`basis`), the served basis, the offline value,
  the report's figure, `offline_vs_primary` and `claim_vs_primary`;
- `offline_vs_primary`, a per-claim summary;
- `label_note`, where the source shows how the published figure was built.

**Reading conventions added while labelling.** None of them depends on the checker.

- `pbasis` changes only which annual-report figure the `primary` column compares with. It never changes a label. It
  is set where the sentence implies a basis without naming it:
  - a generic "ROE" is compared with the report's weighted ROE, the only ROE an annual report prints (x32);
  - a bank's own "净利润" is compared with total net profit (x28, x31);
  - "净利率" is compared with 归母 / 营业收入 (x19, x26, x44);
  - "net profit growing 1.21%" is compared with the 归母 growth (x34).
- x14 lists five insurers' profits after a sentence that names the metric (归母净利润). The 中国平安 part reads that
  metric. The other four insurers are null.
- The ten extension companies have no 扣非 figure or cost of sales in `sources.json`, so their 扣非 and gross-margin
  parts have `no_primary`.

**Counts.**

| | |
|---|---|
| expected verdicts | supported 21, partially_supported 21, contradicted 6, unverifiable 2 |
| definition-blind | supported 21, partially_supported 21, contradicted 8 |
| categories | loose_or_relational 18, daily_price 10, english_or_traditional 8, annual_figures 8, market_value 3, annual_ratios 3 |
| source types | media 26, social 16, broker notes 4, company announcements 4 |
| parts | 133: true 89, false 9, null 35 |

**Labels that change under the definition rule.** Two claims differ between the two label columns. Both go from
unverifiable to contradicted, because each names 加权平均净资产收益率 while the offline ROE is period-end:

- x11: 中信证券, 10.59% against the offline 9.40;
- x49: 长江电力, 15.90% against the offline 15.59.

Both figures match the report's weighted ROE. A third named basis (x44, 美的's 归母 growth of 14.03% against the
offline total-profit 14.87%) changes a part but not the verdict.

**The six contradicted claims.**

| claim | what the claim says | offline value | the report on the claim's own basis |
|---|---|---|---|
| x19 | 中芯国际 净利率 10.71% | 7.49% (归母 / revenue) | 归母 / revenue 7.49%; 10.71% is total net profit / revenue |
| x26 | 宁德时代 净利率 "约18.1%" | 17.04% | 归母 17.04%; 18.1% is total net profit / revenue |
| x32 | 平安银行 ROE "~9%", 招行 13.44% | period-end 7.73% and 11.80% | weighted 9.15% and 13.44%, so true |
| x41 | 紫金矿业 "日赚近1.4亿元" | 1.419亿 a day | 1.419亿 a day. "近" N needs the actual value ≤ N; it is 1.3% above N |
| x45 | 茅台 "average ROE … 34.36%" | weighted 32.53% | 32.53% |
| x47 | 茅台 "一天赚4.7个亿" | 2.26亿 a day | 2.26亿 a day; 4.7亿 is roughly revenue / 365 |

**Data vs. the published reports (from the `primary` columns).**

- **Same basis.** No annual figure in this set shows an offline value that differs from the report on the same basis.
  Every comparable part agrees. Three parts have no report figure in the data files (`no_primary`): 宁德时代's 扣非
  (x25), 紫金's 扣非 (x39) and 美的's gross margin (x43).
- **Definitions.** Seven parts in six claims are on a basis the offline field is not on:
  - period-end against weighted ROE: 中信证券 x11, 平安银行 and 招商银行 x32, 长江电力 x49;
  - total-net-profit against 归母 YoY: 招商银行 x34, 美的 x44;
  - 营业总收入 against 营业收入 as the margin's base: 美的 x44 (9.7% is true on 营业收入, at 9.63%, and false on the
    served 营业总收入, at 9.58%);
  - total against 归母 net profit: 工商银行 x28. 3,707.66亿 is total net profit; the offline 归母 is 3,685.62亿. The
    part passes through the 1% rule.

  The claim is true on its own basis in every one of these parts. x32 is contradicted and x34 partially supported
  only because of the definition.
- **Published figures on a basis the sentence does not name.** x19 and x26 quote total-net-profit margins as "净利率".
  x47 quotes revenue / 365 as "赚". x16's "日赚3.68亿" comes from 营运利润 (operating profit, 1344亿) according to the
  article. It is still within 1% of 归母 net profit / 365 (3.69亿), so it is labelled true.
- **Rounding in the corrections layer.** 中国平安's served 归母 YoY is the report's one-decimal 6.5; the report's
  figures give 6.45. Reuters' "6.45%" (x13) passes only through the 1% rule, as v2's v36 did.
- **Prices.** Every published price, % change, volume and turnover in this set agrees with the offline bars, and so
  does every valuation figure. 42 bar parts and 4 valuation parts are true, and none is false.

## First run (the only run), `7ea8f5d`

The checker was run once, on a clean tree at `7ea8f5d`, on 2026-10-10T20:32Z (2026-10-11 local time). The claims,
labels and scorer had all been committed first (`ac54321`, `0eae39b`, `61b6f6e`, `7ea8f5d`). The command was
`python -m evaluation.real_claims_web_v3.score_claims_web_v3`.

- It used the offline service with the shipped default data (v1 + 2026-09-30 extension + corrections layer) and no
  LLM.
- The checker received the published sentence with the brackets removed and no date, as in the first two sets.
- Result: `evaluation/results/claims_web_v3-first-run.json`.

| | result |
|---|---|
| verdict accuracy | **0.540 [0.40, 0.68]** (27/50; 95% percentile bootstrap over claims, 2000 resamples) |
| against the definition-blind labels | 0.500 [0.36, 0.64] (25/50) |
| by category | annual_ratios 1.000 (3/3), daily_price 0.800 (8/10), annual_figures 0.625 (5/8), loose_or_relational 0.389 (7/18), english_or_traditional 0.375 (3/8), market_value 0.333 (1/3) |
| by source type | broker notes 0.750 (3/4), company announcements 0.750 (3/4), media 0.654 (17/26), social 0.250 (4/16: 东方财富 3/6, 雪球 0/3, the seven other platforms 1/7) |
| by expected verdict | unverifiable 2/2, partially_supported 0.714 (15/21), contradicted 0.500 (3/6), supported 0.333 (7/21) |
| checker's verdicts | supported 7, partially_supported 19, contradicted 8, unverifiable 16 |

This is a blind number for the round-14 checker: neither the set nor its classes were seen by anyone who changed the
checker. It is close to the second set's blind first run (0.500 [0.36, 0.64]), which was measured on the checker
before round 14. The two intervals overlap almost entirely. Fifty claims cannot show whether round 14's rules carried
over to unseen wording.

### Error classes (not fixed; after any fix this set is "after exposure")

There are 23 errors. They are grouped below by what the checker's own `checks` show it did. Five claims whose labels
have no false part were reported **contradicted**: x04, x06, x18, x23 and x37.

1. **Metric wording not recognised** (11: x07, x08, x10, x13, x16, x29, x33, x34, x41, x47, x48). The checker reports
   `no_metric` and the claim becomes unverifiable, or only partially supported.
   - "市盈TTM" (x10, x33). "滚动市盈率" was read correctly (x24).
   - Daily averages: "日赚3.68亿", "日赚近1.4亿元", "一天赚4.7个亿" (x16, x41, x47). This set added them on purpose
     (new rule: annual / 365). Two of the three are false by the labels, so a user sees "unverifiable" for a wrong
     figure.
   - Bare "利润" for a profit figure (x29 "2025年的3686亿").
   - Abbreviated bar words: "开3839.25，最高3851.22，最低3833.09，收3842.19" (x08). In the same sentence, "涨11.74点" is
     bound to the % change and refused as a unit mismatch.
   - English phrasings: "resumes where it ended September, at 3,842" (x07), "a 6.45% year-on-year rise in net profit"
     (x13), "RMB 150.2 billion in net profit … growing 1.21% year-over-year" (x34), and "the previous year's 84.49
     billion" (x48).
2. **A number bound to the wrong metric** (3: x02, x15, x37). The result is a false `contradicted` part.
   - x37: "RMB 803.97 billion in revenue and RMB 32.62 billion in net profit". When the number comes before the metric
     word, 803.97 gets no metric and 32.62 is compared with revenue. The claim is contradicted, although both figures
     are right.
   - x15: "收入1.05万亿,利润增7%到1348亿". The 1348亿 is compared with revenue, and "利润增7%" has no metric.
   - x02: "较上一交易日增加8.28亿元" is compared with the day's turnover (68.50亿), not with the change. "环比增幅为13.75%"
     has no metric.
3. **Dates** (3: x04, x05, x06).
   - x05 says "4月22日 … 上证指数收盘指数站上4100点". The checker reports `period_mismatch`, "the data is for the
     trading day 2026-09-30", although the extension's history has 上证指数's 2026-04-22 bar. Round 14 made a stated
     date use that day's bar (v2's v17, a stock). This index sentence is not covered.
   - x04 and x06 leave the date (2026-04-22) to the article. The checker compared them with the 2026-09-30 bar and
     **contradicted** them, both the 0.52% / 1.73% moves and the 4106.26 close. A sentence with no date is checked
     against the latest bar and contradicted, not left unverifiable. This is v2's open class (v13, v46).
4. **Entity resolution** (4: x01, x14, x18, x32).
   - x18: "中芯国际全年实现销售收入人民币673.23 亿元，同比增长16.5%" is checked against **宁德时代** (revenue 4237亿,
     YoY 17.04%) and contradicted. The likely cause is v2's open v44 bug, in which the runtime alias table lists
     "人民币" as an alias of 宁德时代: this sentence contains "人民币". The checks show only the target, so this was
     not confirmed.
   - x01: the fund codes in brackets ("联接基金A/C/Y：110026/004744/022907") resolve the target to a feeder fund
     (易方达创业板ETF联接Y) rather than 159915, so the check has no data.
   - x14: "国寿、平安、太保、人保、新华分别为1541、1348、535、466、363亿元" binds all five numbers to 中国平安, and none
     has a metric. The label reads the metric from the preceding sentence, which the checker does not see. Even with
     the metric, the 分别 pairing to five names would be needed.
   - x32: "平安银行与招行的ROE差距（~9% vs 13.44%）预计2026年仍难以明显收敛" binds both ROE figures to 招商银行. The
     trailing "预计2026年" makes both parts `forecast`, so the claim is unverifiable. The label is contradicted: the
     offline ROE is period-end, while the published 9% and 13.44% are the reports' weighted ROE.
5. **"降幅" read as a positive change** (1: x23). "净利润降幅为71.9%，显著大于营收降幅的54.6%" compares +71.9 with the
   offline −71.89, and treats "大于…54.6%" as a `gt` bound on revenue YoY. The claim is contradicted, although every
   part is true.
6. **A named basis that the report shows is the same figure** (1: x50). "长江电力实现营业总收入862.42亿元，同比增长2.07%"
   is `definition_mismatch` on both parts. The data serves 营业收入, and the annual report notes in `sources.json`
   state 营业收入 = 营业总收入 for 长江电力. This is v2's open v34, unchanged. The equality is stated in the data
   files, but the checker does not use it.

**Classes already listed for the first two sets.**

- Four of the 23 errors are in classes the earlier READMEs list as still open: x04 and x06 (dates left to context, as
  in v2's v13 and v46), x50 (v2's v34), and, if the likely cause is right, x18 (the 人民币 alias of v2's v44).
- Five more are adjacent to classes round 14 wrote rules for, in wording those rules do not cover: x05 (a stated date
  on an index), x07, x13, x34 and x48 (English phrasings).
- The other fourteen (x01, x02, x08, x10, x14, x15, x16, x23, x29, x32, x33, x37, x41, x47) are in classes that are
  new for this set: 市盈TTM, daily averages, bare 利润, abbreviated bar words, the pairing
  of a number with the metric after it, relative changes, fund codes, 分别 lists, a forecast clause after reported
  figures, and 降幅.

The engineer knew the first set's classes before selecting. The search agents knew none, and the selection rules read
no class.

**What a user would see.**

- Four true published claims are marked **contradicted**: 中芯国际's revenue (x18), 五粮液's declines (x23), BYD's
  revenue and profit (x37), and 上证指数 on 2026-04-22 (x06). A fifth, partially true (x04), is also marked
  contradicted.
- Two figures that are wrong by both the data and the report are shown as unverifiable, not wrong: 茅台's
  "一天赚4.7个亿" (x47) and 紫金's "日赚近1.4亿元" (x41).
- x32's ROE pair is labelled contradicted only because the offline ROE is period-end; the reports confirm both
  figures. The checker returned unverifiable, but for another reason: a forecast clause.
- Weighted ROEs named as such (x11, x49) are correctly left unverifiable. The data's period-end ROE cannot confirm
  them, although the annual reports do.
