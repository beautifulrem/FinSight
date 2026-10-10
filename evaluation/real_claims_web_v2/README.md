# Real published claims, second set, blind (`claims_web_v2.jsonl`)

Real claims, as published by media, brokers' public notes, company announcements and social posts (雪球,
东方财富股吧, 知乎, 微博, 同花顺), about the companies, metrics and dates FinSight's offline data covers. Like the first
set (`evaluation/real_claims_web/`), the point is to measure the claim checker (`POST /agent/claim-check`) on wording
nobody wrote for it. Unlike the first set, this one is **blind to the checker**: it was collected, labelled and
committed before anyone working on it read the checker's code or its tests.

**What this set is not.** It is not the owner's real-claims set (`evaluation/human/real_claims/`), which is still
owner-blocked: the owner collects 30–50 claims they actually met, labels them, and scores them with
`evaluation/human/import_real_claims.py`. This set was collected by the engineer's search agents.

## Protocol (written before labelling)

**Blind to the checker.** The engineer who built this set did not open `query_intelligence/` (in particular
`query_intelligence/agent/claim_check.py`), the claim-bench rows or tests, or the first set's labelling and scoring
scripts, before every claim was collected, labelled and committed. Before collecting, the engineer read only:

- the data files: `data/structured_data.json`, `data/snapshot/structured_data_ext.json`,
  `data/snapshot/structured_data_fix.json`, `data/snapshot/manifest.json`, `data/snapshot/manifest_fix.json`,
  `data/snapshot/corrections/sources.json`;
- `evaluation/real_claims_web/README.md`, for the first set's format.

**Exposure that remains.** That README lists the first run's error classes (分别 pairing, 元/点 changes, 报收,
"2025 年", traditional characters, bracketed comparands, two-metric ratios, prior-year gaps, 2% tolerance). The
engineer knew them before selecting. The engineer also saw the offline values, but the search agents did not (below).
So the set is blind to the checker's code, not to its known weaknesses or to the data.

**Collection.** 2026-10-10 (every `retrieved` field). Three search agents (fresh contexts, Tavily and Brave web search)
each covered a list of names. They were not allowed to open any repository file. They were told which companies,
metrics and dates are checkable, and to **not select by correctness** (rounded, loose, comparative and wrong numbers
were wanted). They were **not** given any offline value or the checker's error classes. They were asked for
relations between companies, ratios, growth wording, loose numbers, English claims and a mix of source types. Each
agent confirmed the sentence on its page (`verified_by`: `tavily_extract`, or a search snippet where the page did
not extract). The first set's URLs were given to them as a do-not-use list.

**Checkable scope (from the data files, listed before searching).**

- FY2025 annual figures (report date 2025-12-31): revenue, 归母净利润, 扣非净利润, revenue and net-profit YoY,
  ROE, gross margin, net margin (net profit / revenue), EPS.
  - The ten extension companies: 宁德时代, 比亚迪, 美的集团, 紫金矿业, 中芯国际, 招商银行, 工商银行, 平安银行, 中信证券,
    长江电力. Gross margin is missing for the three banks and 中信证券.
  - The three companies the corrections layer replaces with their annual reports: 贵州茅台, 五粮液 (both with gross
    margin), 中国平安 (no gross margin).
  - Field definitions are stated in `structured_data_fix.json` (`definitions`). ROE is period-end for the ten
    extension companies, weighted for the three corrected ones. EPS is diluted for 美的 and 紫金, basic elsewhere.
    Revenue is 营业总收入 for 美的, 营业收入 elsewhere. Net-profit YoY is on total net profit for eight extension
    companies, 归母 for 平安银行, 中芯国际 and the three corrected ones.
- Valuation (总市值, PE TTM, PB): on 2026-09-30 for the ten extension companies; on 2026-04-22 for 茅台, 五粮液, 中国平安.
- Daily bars (close, open, high, low, % change, change in 元 / 点, 成交额, volume):
  - 2026-09-30 and 2026-04-22 for the ten extension stocks and 上证指数, 创业板指, 中证500ETF, 黄金ETF, 科创50ETF
    (the extension's daily history covers both dates; the indices have no 成交额);
  - 2026-04-22 for 茅台, 五粮液, 中国平安, 沪深300 index, 沪深300ETF, 创业板ETF, 证券ETF.
- Excluded while searching: quarterly and half-year figures, forecasts and targets, prices on other dates, live quote
  widgets, market-wide turnover, northbound turnover.

**Quoting.** One sentence or clause per claim, about 40 characters or one short English sentence, with source name,
URL, publication date and retrieval date. No article text beyond the claim sentence is stored. Shortening drops whole
clauses and is flagged `excerpt`. An implicit subject ("公司", "该行") is inserted in square brackets
(`subject_inserted`); the checker gets the text without the brackets. A date given only by the publication day is
recorded as `date_implicit`.

## Labelling rules (written before labelling)

Labels come from `label_claims_web_v2.py`, which reads only the data files above. A person writes each claim's
*parts* (company, metric, the number as written, a qualifier, the basis the sentence names); every value the parts
are compared with is computed from the files. Each stated number is one part.

- **Exact figure** ("4237.02亿元", "1.50%", "5.8元"): true when |claimed − actual| ≤ max(half a unit of the claim's last
  written digit, **1% of the actual value**). A growth rate or change must also have the right sign.
- **约 / 左右 / about / approximately**: within 5% of the claimed number.
- **突破 / 站上 / 达到…以上** N: actual ≥ N. **超 / 超过 / 逾 / 以上 / more than / over** N: actual > N.
  **近 / 将近 / 接近 / nearly** N: 0.9 N ≤ actual ≤ N. **不到 / 不足 / 低于 / less than** N: actual < N.
  **N 多 / N 余** ("两倍多", "300多亿"): N < actual < N + one unit of the last written digit.
  For a signed quantity (growth, change), the bound applies to the magnitude and the sign must match.
- **Relations**: "A 高于 / 超过 B" is true when A > B with both values from the data. "A 是 B 的 N 倍" is the ratio A / B
  under the rules above. A gap ("比 B 多 X") is A − B under the rules above.
- **Derived values**: net margin = net profit / revenue; prior-year value = value / (1 + YoY); change in 元 / 点 =
  close − previous close in the daily history (or close − close / (1 + pct) where there is no previous bar); a fund's
  % change, when the bar leaves it empty, from its last two closes.
- **Definitions (new in v2).** The corrections layer states which basis each field is on. A sentence that uses the
  generic word (ROE, 净利润, 每股收益, 营收 / 营业收入, revenue, profit) is compared with the served value whatever its
  basis. A sentence that **names** a basis different from the served one (加权平均净资产收益率 vs a period-end ROE,
  基本 vs diluted EPS, 营业总收入 vs 营业收入, 归母净利润 growth vs a total-net-profit YoY) is **not decidable from the
  offline data**: the part is null, not false, unless the annual report in `sources.json` shows the two bases give the
  same figure for that company (for example basic = diluted EPS for 比亚迪). The file also carries
  `expected_verdict_definition_blind`, the label with every named basis ignored (the first set's rule), so the
  checker can be scored both ways.
- **Not in the offline data** (null): 换手率, quarterly figures, prior-year margins or ROE, changes in percentage
  points, 扣非 YoY, dividends, R&D, unit sales, USD or HKD amounts, other indices, 成交额 of an index, valuation on a
  date other than the stated valuation date.
- **Verdict** (the aggregation the first set's README documents for the checker):
  - every part true → supported;
  - no part true but at least one false → contradicted;
  - some true, others false or null → partially_supported;
  - nothing decided → unverifiable.
- **Context without a number of its own** (创历史新高, 位列第一) is not a part. It is listed in `context_not_checked`.
- **Labels are against the offline data, not against the company's report.** For every annual-figure part the
  script also compares the offline value with the annual report's figure on the claim's basis (`primary`:
  `offline_vs_primary` = agree / definition_differs / value_differs, and `claim_vs_primary`). Price parts have no
  independent primary record in the data files; where a published price differs from the offline bar it is noted.

## Selection (104 candidates → 50), before any label

The three agents returned 104 candidates (`candidates_v2.jsonl`, agent A 33, B 33, C 38).
`select_claims_web_v2.py` picks the 50 by fixed rules that never read an offline value:

| rule | dropped |
|---|---|
| not a claim sentence (an annual-report table row, a quote-table line, a sentence with no number) | 3 |
| already in the first set by text (w27's 招商银行 sentence, w59's "沪深300指数涨0.66%") | 2 |
| a second copy of the same sentence (same URL, or the same sentence on another outlet) | 3 |
| period not stated (a 五粮液 ROE range with no year) | 1 |
| numbers about a range over several sessions, not one date | 1 |

The other 94 are grouped by first-named entity: the 13 companies, plus one group for indices and ETFs. Within a group,
candidates with a known publication date come first, then they are ordered by a fixed pseudo-random key (sha256 of URL
+ text). At most 4 are taken per group and 2 per URL, and the list is trimmed to 50 from the largest groups. The cap
spreads the set over companies and is applied blind to correctness. After selection, the subject was inserted in brackets in two claims whose sentence leaves it implicit (v10
美的集团, v34 长江电力; the agents had not bracketed it). Two long English sentences were
shortened by whole clauses (`excerpt`). Dates that a sentence leaves to its context were also recorded
(`date_implicit`, `date_source`).

## Counts

- 50 claims: 44 zh, 2 traditional Chinese, 4 en. 28 of the 46 Chinese claims are ≤ 40 characters. The rest run
  longer because the published sentence does; the longest is 60 characters (v01).
- Source types:
  - media 28;
  - social 15 (雪球 10, 东方财富股吧 / 财富号 5);
  - broker notes 3;
  - company announcements 4.
- The agents found nothing usable on 知乎, 微博 or 同花顺. Tavily often aborted and Brave failed on every call.
- Periods: FY2025 annual figures 33, 2026-09-30 prices / valuation 11, 2026-04-22 prices 6.
- Publication date not shown on the page: 2 (v11 TipRanks, v18 雪球). The candidates' `note` gives what is known.
