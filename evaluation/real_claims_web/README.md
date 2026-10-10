# Real published claims, engineer-collected (`claims_web_v1.jsonl`)

60 claims as they were published by media, brokers' public notes, company announcements and social posts (雪球, 知乎,
东方财富股吧 / 财富号, 新浪看点), about the companies, metrics and dates FinSight's offline data covers. The point is to
measure the claim checker (`POST /agent/claim-check`, `query_intelligence/agent/claim_check.py`) on wording and numbers
that nobody wrote for it, instead of on synthetic claim sets.

**What this set is not.** It is not the owner's real-claims set (`evaluation/human/real_claims/`). That one is still
owner-blocked: the owner collects 30–50 claims they actually met, labels them, and scores them with
`evaluation/human/import_real_claims.py`. This set was collected by the engineer, and the engineer
**had read the checker's code and seen the offline snapshot values before collecting**. So it is **not blind**. It
replaces synthetic phrasing with published phrasing; it does not replace an independent collector.

## Files

| file | what |
|---|---|
| `collected_v1.jsonl` | the 60 claims as collected, before any label (committed first, `9e059cf`) |
| `label_claims_web_v1.py` | computes every expected verdict from the offline tools (committed with the labels, before the checker was run) |
| `claims_web_v1.jsonl` | collected fields + `category`, `expected_verdict`, `parts` (one predicate per stated number, true / false / null = not in the offline data), `context_not_checked`, `offline_vs_primary` |
| `score_claims_web_v1.py` | runs the checker once and writes `evaluation/results/claims_web_v1-first-run.json` |

```bash
python -m evaluation.real_claims_web.label_claims_web_v1     # rewrites claims_web_v1.jsonl (deterministic)
python -m evaluation.real_claims_web.score_claims_web_v1     # writes evaluation/results/claims_web_v1-first-run.json
```

Both use the offline service with the 2026-09-30 snapshot extension on (`build_offline_service(snapshot_ext=True)`,
the product default `QI_OFFLINE_SNAPSHOT_EXT=true`). The other evaluation sets stay pinned to v1. This set needs the
extension because 10 of its 13 companies exist only there.

## Collection protocol

- Date: 2026-10-01 (every `retrieved` field). Searches ran through Tavily, Brave and the agents' web search. Three
  search agents each covered a list of names. They were told which names, metrics and dates are checkable, and to
  **not select by correctness**: rounded, loose and wrong numbers were wanted. They were **not** given the offline
  values. The agent that found a claim compared its sentence with the page text at retrieval. 雪球 / 知乎 pages
  were read through a browser or a search snippet.
- What was checkable (listed before searching):
  - FY2025 annual figures (report date 2025-12-31): revenue, net profit, their YoY growth, ROE, gross margin, net
    margin (derived as net profit / revenue), EPS.
    - The ten extension companies have all of these: 宁德时代, 比亚迪, 美的集团, 紫金矿业, 中芯国际, 招商银行,
      工商银行, 平安银行, 中信证券, 长江电力. Gross margin is missing for the banks and 中信证券.
    - The v1 snapshot's three companies have revenue, net profit and ROE only, with no YoY or EPS: 贵州茅台,
      五粮液, 中国平安. 五粮液 also has gross margin.
  - Total market value and PE / PB as of 2026-09-30, for the ten extension companies.
  - Price facts for 2026-09-30, the extension's last trading day: close, % change, turnover. Instruments: the ten
    extension companies, 上证指数, 创业板指, 中证500ETF, 黄金ETF, 科创50ETF.
  - Price facts for 2026-04-22, v1's last trading day: 茅台, 五粮液, 中国平安, 沪深300 index, 沪深300ETF, 创业板ETF,
    证券ETF.
- Excluded while searching (claims about unchecked periods or not published claims): quarterly and half-year figures,
  2024 figures, forecasts, price facts on any other date, the live quote widgets on news pages (not a published claim),
  and 深股通 (northbound) turnover presented as a stock's turnover.
- Quoting: one sentence or clause. 42 of the 59 Chinese claims are ≤ 40 characters; 17 run to 41–54 characters
  (digits and punctuation counted) because the published sentence does; the English one is 15 words. Each claim is
  quoted with its source name, URL and publication date; no article text beyond the claim sentence is stored. Shortening drops whole clauses and is flagged `excerpt`. Where the published sentence's subject
  is implicit ("公司", "该行"), the company's short name was inserted in brackets in `published_text` (`subject_inserted`).
  The checker gets the text without the brackets. A date given only by the publication day ("截至收盘，沪指涨0.31%",
  published on 2026-09-30) is recorded as `date_implicit`.

## Selection (93 candidates → 60)

The agents returned 93 candidates. 33 were dropped by these rules, before any label was computed:

| rule | dropped |
|---|---|
| not a claim sentence (two 金融界 table rows, a 股吧 string of three prices) | 3 |
| the metric is not named in the sentence ("宁德做到了722亿", "增速相对较慢，为0.74%") | 2 |
| a five-year list (知乎 ICBC ROE 2021–2025, ~70 characters) | 1 |
| figure mislabelled by its author in a way that has no single reading ("归母净利润（扣非）") | 1 |
| a fact already quoted twice, or the same sentence from a second outlet | 8 |
| cap per name, to spread the set over companies and source types | 18 |

The selection was made by the engineer, who knew the offline values for most names. Dropping was by the rules above,
but a person who knew the data applied them.

## Counts

- 60 claims: 59 zh (two in traditional characters), 1 en.
- Source types: media 37, social 14 (雪球 8, 知乎 3, 东方财富股吧 / 财富号 2, 新浪看点 1), broker notes 5, company
  announcements 4.
- Periods: FY2025 annual figures 43, 2026-09-30 prices / market value 12, 2026-04-22 prices 5.
- Categories: `annual_figures` 21, `annual_ratios` 12 (ROE, margins, EPS), `loose_or_relational` 7 (突破 / 超 / 约 / 近,
  ratios of two growth rates, a two-company comparison, a derived prior-year gap), `english_or_traditional` 3,
  `daily_price` 15, `market_value` 2.
- Expected verdicts: supported 23, partially_supported 21, contradicted 15, unverifiable 1.

## Labelling rules

The labels come from `label_claims_web_v1.py`, which reads the offline tools' output; no value is typed in. Each
stated number is one part.

- **Exact figure** ("4237.02亿元", "1.50%", "5.8元"): true when |claimed − actual| ≤ max(half a unit of the claim's last
  written digit, **1% of the actual value**). A growth rate or change must also have the right sign. Examples: "5.8元"
  vs 5.76 holds (half-step 0.05); "1.95元" vs 1.91 does not; "42.28%" vs 42.18 holds (1%); "1.21%" vs 1.05 does not.
- **约 / 左右 / about**: within 5% of the claimed number. **突破 / 站上** N: ≥ N; **超 / 超过 / 首破 / 涨超** N: > N.
  **近 / 将近 / 接近** N: 0.9 N ≤ actual ≤ N.
- **Derived values** follow the project's own derivations:
  - net margin = net profit / revenue (`coverage.METRICS["net_margin"]`);
  - prior-year profit = profit / (1 + YoY);
  - yuan or point change = close − close / (1 + pct);
  - a fund's daily change, when the source leaves it empty, from its last two closes.
  A derived value whose own rounding is coarser than the claim is treated as not in the data (招行 "同比增加0.44亿元"
  from a YoY of 0.01%).
- **Not in the offline data** (null): 换手率, 中国银行 / 深成指 / 北证50 / 科创50 index, market turnover, USD
  amounts, prior-year margins and EPS changes, dividend ratios, YoY for the v1 companies.
- **Verdict**: the checker's documented aggregation (`ClaimReport.verdict`):
  - every part true → supported;
  - no part true but at least one false → contradicted;
  - some true, others false or null → partially_supported;
  - nothing decided → unverifiable.
- **Context without a number of its own** is not a part and does not change the label. It is listed in
  `context_not_checked`. Examples: 创历史新高, 位列第一, 排名第二, 股价再创新高.
- **Labels are against the offline data, not against the company's report.** Each claim's `offline_vs_primary`
  records how the offline value compares with the primary source (from the annual reports, read at collection).
  Where they differ, a claim the company itself published can be labelled contradicted. That is the correct verdict
  for a checker that compares with this data, and a data problem the README reports separately (below).

## What the offline data gets wrong against the published reports

These are findings about the data, not the checker; the labels follow the data.

- **v1 seed fundamentals for 五粮液 and 中国平安 do not match their FY2025 reports.**
  - 五粮液: offline revenue 1085亿, net profit 378亿, ROE 29.4%, against 405.29亿, 89.54亿 and 6.89% in the report.
  - 中国平安: offline net profit 1210亿, revenue 12180亿, ROE 15.2%, against 1347.78亿, 10505.06亿 and 14.0%.
  - 贵州茅台's seed revenue and profit are right: 营业收入 1688.38亿, 823.2亿.
- **Definitions differ by field and by company.**
  - ROE is weighted for 美的, but period-end ("diluted") for the other extension companies. For example 宁德时代
    21.42% vs the reported weighted 24.91%, and 招商银行 11.8% vs 13.44%.
  - "revenue" is 营业总收入 for 美的 (4585.02亿) but 营业收入 for 茅台 (1688.38亿, against 营业总收入 1720.54亿).
  - EPS is diluted (美的 5.76 vs basic 5.80, 紫金 1.91 vs basic 1.95).
  - 招商银行's profit YoY is for total net profit (1.05% vs 归母 1.21%).
  - 中信证券's revenue YoY is against restated 2024 (28.8%); some media use 17.35%, against the original 2024 figure.
- **v1 2026-04-22 prices disagree with media for 中国平安 and the 沪深300 index.**
  - 中国平安: offline close 53.61 (+0.73%, 66.4亿) vs 58.03 / 0.0% / 33.65亿 (证券之星).
  - 沪深300: offline +0.42% vs +0.66% (华夏时报, 证券时报).
  - Not resolved against an exchange record.

In this set, 19 claims that the company's own report supports (on the basis the claim uses: weighted ROE, basic EPS,
total net profit, 营业总收入, IFRS / USD) are labelled contradicted (9) or partially_supported (10) because of these
data differences, and 2 more (the 2026-04-22 中国平安 and 沪深300 prices) are unresolved
(`label_follows_offline_difference`; the reason is in `offline_vs_primary`).

## First run (the only run), `df0d212`

Run once on a clean tree at `df0d212`, on 2026-10-01T14:24Z, after the labels had been committed:
`python -m evaluation.real_claims_web.score_claims_web_v1`. It used the offline service with the snapshot
extension and no LLM. Result: `evaluation/results/claims_web_v1-first-run.json`.

| | result |
|---|---|
| verdict accuracy | **0.800 [0.70, 0.88]** (48/60; 95% percentile bootstrap over claims, 2000 resamples) |
| by category | annual_figures 0.857 (18/21), annual_ratios 0.833 (10/12), daily_price 0.800 (12/15), english_or_traditional 0.667 (2/3), loose_or_relational 0.571 (4/7), market_value 1.000 (2/2) |
| by source type | media 0.811 (30/37), social 0.786 (11/14), broker notes 0.600 (3/5), company announcements 1.000 (4/4) |
| by expected verdict | supported 0.696 (16/23), partially_supported 0.857 (18/21), contradicted 0.867 (13/15), unverifiable 1/1 |

Every error (expected → checker):

| id | claim (as checked) | expected → got | what the checker did |
|---|---|---|---|
| w02 | 利润增速是营收增速的近 2.5 倍… | supported → unverifiable | 倍 bound to revenue (unit mismatch); a ratio of two growth rates is not read |
| w03 | …毛利率和净利率分别达到26.3%和18.1%… | partially → contradicted | both numbers checked as net margin (26.3 vs 17.04 too) |
| w04 | …收盘报291.11元，较前一交易日上涨4.31元，涨幅1.50% | supported → partially | 上涨4.31元 read as a % change in 元 (unit mismatch) |
| w07 | 但比亚迪净利润只有326亿，比2024年的403亿少了近80亿 | supported → partially | the 2024 value and the gap are "period_mismatch"; not derived from the YoY |
| w20 | 中芯國際…營收人民幣93.27億美元，年增16.2% | contradicted → unverifiable | traditional characters (營收, 年增): no metric recognised |
| w24 | 招行ROE为 13.44%，不仅远高于工商银行（9.45%）… | partially → contradicted | the comparand's stated value in brackets is read as a bound (工行 ROE > 9.45); the relation 招行 > 工行 is not checked |
| w37 | 2025 年中信证券实现营业收入748.3亿元，同比增长29% | supported → partially | "2025 年" with a space is read as a number (no metric) |
| w40 | …同比分别增长2.07%、6.17% | supported → partially | 分别 over two growth rates: both checked against net-profit growth |
| w44 | 贵州茅台实现营业总收入1720.54亿元，归母净利润823.20亿元 | partially → supported | 营业总收入 1720.54亿 accepted for 营业收入 1688.38亿 (1.9% off): the checker's exact-figure tolerance is 2% relative, the labels' 1% |
| w47 | …平均净资产收益率（ROE）保持在32.53%的高位 | contradicted → supported | 32.53 accepted for 33.0 (1.4% off): same tolerance difference |
| w56 | 上证综指9月30日…收盘上涨11.74点，涨幅：0.31%，报3842.19点 | supported → partially | 上涨11.74点 read as the close (contradicted) |
| w58 | 黄金ETF华安（518880）报收8.633元，收涨1.34%… | supported → partially | 报收 is not a close word (no metric) |

### Error classes (to be fixed later; after any fix this set is labelled "after exposure")

1. **分别 / list pairing across metrics.** "毛利率和净利率分别达到A和B" and "营收、净利同比分别增长A、B" bind every number to
   one metric (w03, w40).
2. **Absolute daily change in 元 / 点.** "上涨4.31元", "上涨11.74点" are read as a % change (unit mismatch) or as the
   close (w04, w56).
3. **Close words.** 报收 is not recognised as the close (w58).
4. **Year written with a space** ("2025 年") becomes a claimed number (w37).
5. **Traditional Chinese metric words** (營收, 年增) are not recognised (w20).
6. **A comparand's stated value in brackets.** "远高于工商银行（9.45%）" is read as a bound on 工商银行 instead of the
   stated value of the compared side plus the relation (w24).
7. **Relations between two metrics of one company.** For example, a ratio of two growth rates ("利润增速是营收增速的近2.5倍",
   w02).
8. **Prior-year values and gaps derivable from a YoY.** "比2024年的403亿少了近80亿" is left unverifiable instead of being
   derived from profit / (1 + YoY) (w07). This is a scope choice as much as a bug.
9. **Exact-figure tolerance.** The checker accepts 2% relative (`_REL_TOLERANCE`) on an exact figure; the labels accept
   the larger of 1% and the claim's own precision. A 1.4–1.9% gap therefore passes, including a different revenue
   definition (w44, w47). Under a 2% labelling rule these two would be right (50/60). The README states the rule the
   labels use; which one is correct for the product is a decision, not a measurement.

Separately from the checker, 19 of the 60 claims are true by the company's own report but labelled contradicted or
partially_supported because the offline data differs from the report (see above). The checker returned the labelled
verdict on 15 of them. It missed w03 and w24 for parsing reasons (classes 1 and 6). On w44 and w47 its 2% tolerance
happened to accept the published figure (class 9). A user pasting those 19 published claims
would see a true statement marked wrong or partly wrong. That is a data-coverage and definition problem (field
definitions, seed values), not something verdict accuracy shows.
