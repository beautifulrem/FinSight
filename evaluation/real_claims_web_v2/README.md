# Real published claims, second set, blind (`claims_web_v2.jsonl`)

Real claims, as published by media, brokers' public notes, company announcements and social posts (雪球,
东方财富股吧, 知乎, 微博, 同花顺), about the companies, metrics and dates FinSight's offline data covers. Like the first
set (`evaluation/real_claims_web/`), the point is to measure the claim checker (`POST /agent/claim-check`) on wording
nobody wrote for it. Unlike the first set, this one is **blind to the checker**: it was collected, labelled and
committed before anyone working on it read the checker's code or its tests.

**What this set is not.** It is not the owner's real-claims set (`evaluation/human/real_claims/`), which is still
owner-blocked: the owner collects 30–50 claims they actually met, labels them, and scores them with
`evaluation/human/import_real_claims.py`. This set was collected by the engineer's search agents.

## Files

| file | what |
|---|---|
| `candidates_v2.jsonl` | the 104 candidates the three search agents returned |
| `select_claims_web_v2.py` | the fixed selection rules, candidates → 50 claims |
| `collected_v2.jsonl` | the 50 claims as collected, before any label (committed first, `8f0b08b`; two subjects bracketed in `ab01c71`) |
| `label_claims_web_v2.py` | computes every expected verdict from the data files (committed with the labels, `a5378bd`, before the checker was run) |
| `claims_web_v2.jsonl` | collected fields + labels, parts, primary-source comparison |
| `score_claims_web_v2.py` | runs the checker once and writes `evaluation/results/claims_web_v2-first-run.json` |

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

## Labels (`claims_web_v2.jsonl`), committed before the checker was run

```bash
python -m evaluation.real_claims_web_v2.select_claims_web_v2   # candidates_v2.jsonl -> collected_v2.jsonl (deterministic)
python -m evaluation.real_claims_web_v2.label_claims_web_v2    # collected_v2.jsonl -> claims_web_v2.jsonl (deterministic)
```

Each row is the collected fields plus the following:

- `category` and `expected_verdict`;
- `expected_verdict_definition_blind`, the same parts with every named basis ignored;
- `parts`, one per stated number. Each has the reading, `holds` (true / false / null), the offline `actual` and the
  rule applied, or the reason it is null;
- `context_not_checked`;
- `primary`, per annual-figure part: the report's figure on the claim's basis, `offline_vs_primary` and
  `claim_vs_primary`;
- `offline_vs_primary`, a per-claim summary.

Two reading conventions were added while labelling. Neither depends on the checker.

- The v1 seed bars of 贵州茅台 and 五粮液 come from tushare. Their `amount` is read in 千元 and `volume` in 手
  (tushare's units; the file does not state them). All other bars are in 元 and shares.
- `pbasis` on a part changes only which annual-report figure the `primary` column compares with. Three parts
  (v22, v23, v34) say a basis the sentence does not name: the IFRS total profit of an H-share announcement, the
  归母 growth behind a "net income" figure, and the 营业收入 that 长江电力 reports as equal to 营业总收入.

Counts:

| | |
|---|---|
| expected verdicts | supported 19, partially_supported 25, contradicted 3, unverifiable 3 |
| definition-blind | supported 22, partially_supported 24, contradicted 3, unverifiable 1 |
| categories | annual_figures 13, annual_ratios 10, daily_price 13, market_value 4, loose_or_relational 4, english_or_traditional 6 |
| source types | media 28, social 15, company announcements 4, broker notes 3 |

**Labels that change under the definition rule.** Six claims differ between the two label columns:

- v06, v12, v33: supported → partially_supported. They name 归母 growth or a 归母 prior-year gap, but the offline YoY is
  on total net profit.
- v10, v19: partially / contradicted → unverifiable. They name 加权平均 ROE, but the offline ROE is period-end.
- v32: partially → contradicted. Its 归母 gap is no longer decidable.

**Data vs. the published reports (from the `primary` columns).** No annual figure in this set shows an offline value
that differs from the report on the same basis. The disagreements are about definitions and about prices:

- **Definitions.** Twelve claims name, or imply, a basis the offline field is not on:
  - 归母 growth against a total-net-profit YoY (比亚迪, 美的, 紫金, 工行, 长江电力);
  - weighted against period-end ROE (美的, 招行);
  - basic against diluted EPS (美的);
  - 动态 against TTM P/E (宁德时代).

  Four of these claims are true on every part by the company's report but are not labelled supported: v06, v12, v33
  (partially_supported) and v23 (Bloomberg's "net income rose 0.7 per cent" for 工行, partially_supported because the
  offline YoY is the total-profit 1.04%).
- **Rounding in the corrections layer.** 中国平安's served 归母 YoY is the report's one-decimal 6.5; the figures give
  6.454. v36's "6.45%" passes only through the 1% rule.
- **Preliminary figures.** v34 quotes 长江电力's 业绩快报 (858.82亿, +1.65%). The annual report and the offline data agree
  on 862.42亿, +2.07%.
- **Prices.** Two published prices differ from the offline bars. Neither is resolved against an exchange record.
  - v38: 中国平安 2026-04-22 low 52.23 vs 57.65. The bar comes from the corrections layer, cross-checked with Tencent.
  - v48: 黄金ETF 2026-09-30 8.339 / +0.97% vs 8.633 / +1.34%. The first set's w58 (财中社) agrees with the offline bar.
- **Wrong published figures that the offline data and the report agree against.** v32's 335.03亿 for 长江电力 (the
  same page states 345.03亿 elsewhere); v41's 五粮液 "ROE 7.47%" (weighted 6.89%); v10's 美的 "加权平均ROE 约17.3%"
  (19.70%).

## First run (the only run), `117e74b`

The run was made once, on a clean tree at `117e74b`, on 2026-10-10T16:27Z, after the claims, labels and scorer had been
committed (`8f0b08b`, `ab01c71`, `a5378bd`, `117e74b`):
`python -m evaluation.real_claims_web_v2.score_claims_web_v2`. It used the offline service with the shipped default
data (v1 + 2026-09-30 extension + corrections layer) and no LLM. The checker received the published sentence with the
brackets removed and no date, as in the first set. Result: `evaluation/results/claims_web_v2-first-run.json`.

| | result |
|---|---|
| verdict accuracy | **0.500 [0.36, 0.64]** (25/50; 95% percentile bootstrap over claims, 2000 resamples) |
| against the definition-blind labels | 0.520 [0.38, 0.66] (26/50) |
| by category | annual_ratios 0.800 (8/10), annual_figures 0.615 (8/13), market_value 0.500 (2/4), daily_price 0.462 (6/13), loose_or_relational 0.250 (1/4), english_or_traditional 0.000 (0/6) |
| by source type | broker notes 0.667 (2/3), media 0.536 (15/28), social 0.467 (7/15: 雪球 5/10, 东方财富股吧 2/5), company announcements 0.250 (1/4) |
| by expected verdict | contradicted 3/3, partially_supported 0.600 (15/25), supported 0.368 (7/19), unverifiable 0/3 |
| checker's verdicts | supported 10, partially_supported 21, contradicted 11, unverifiable 8 |

The first set scored 0.800 on its first run, but that collector had read the checker; this set is blind to it. The
two sets also differ in label rules (definitions) and in their mix of claims (6 English / traditional claims here
against 3). The gap fits the reading that the first figure was helped by a collector who knew the checker, but this
run alone cannot separate that from the other two differences.

### Error classes (not fixed; after any fix this set is "after exposure")

Every error, grouped by what the checker's own `checks` show it did:

1. **A named definition is not read** (4: v03, v10, v12, v19).
   - "动态市盈率 15.56" is accepted as the TTM P/E 15.85, inside the 2% tolerance.
   - "加权平均ROE" is compared with the period-end ROE: v10 美的, v19 招行 13.44 vs 11.8, so the company's true figure
     is contradicted.
   - v12's 归母 growth 62% is accepted against the total-profit 62.01.

   Scored against the definition-blind labels, v10, v12 and v19 are right. Under that rule v03 is still wrong, because
   of the tolerance.
2. **A dated price on a day other than the latest bar** (3: v13, v17, v46).
   - v17 says "截至2026年4月22日收盘". The checker reports `period_mismatch`, "the data is for the trading day
     2026-09-30", although the extension's daily history has the 2026-04-22 bar.
   - v13 ("最终收盘35.53元") and v46 (创业板指 "站上3700点") leave the date to the publication day or the preceding
     sentence. The checker compared them with 2026-09-30.
3. **Market value on the price date** (3: v20, v27, v31).
   - "最新总市值为2245.26亿元" next to a 2026-09-30 close is `period_mismatch`, "the data is for 2025-12-31": the
     valuation's own date is not used.
   - "以1.04万亿元居第五名…万亿市值" has no recognised metric. (v08, "以7595.37亿元市值", was read correctly.)
4. **English claims** (4: v11, v14, v23, v33).
   - "Industrial & Commercial Bank of China" resolves to 中国银行.
   - "China Yangtze Power" resolves to no company.
   - "Net profit attributable to shareholders" is not a recognised metric.
   - "up 15% year on year" and "earnings per share advanced 6.6% to RMB 5.80" are read as a daily % change.
   - US$ amounts are compared with CNY figures and contradicted.
5. **Traditional Chinese** (2: v22, v44). 淨利潤, 歸屬於上市公司股東淨利 and 增長 are not recognised. This class is in
   the first set as well.
6. **扣非 read as 归母 net profit** (2: v04, v35). "扣除非经常性损益的净利润645.08亿元" and "归母净利润（扣非）1437.73亿元"
   are compared with net profit. Their 扣非 growth is compared with net-profit growth. Both are contradicted, although
   the data has 扣非净利润 and it matches.
7. **Close words and price units** (2: v02, v49). "报291.11元/股" is bound to the % change (unit mismatch). "报收7.472元"
   has no metric (also in the first set). "成交量37.52万手" (v17) is not recognised either.
8. **A year read as a claimed number** (2: v18, v36). "2025 年" (with a space) and "和 2024 年对比" become numbers
   with no metric. v36's YoY is then dated 2024 (`period_mismatch`), and "总利润 1347 亿" is not recognised. The first
   set had the spaced-year case.
9. **Relations with companies outside the data, or between two companies** (3: v01, v05, v30).
   - v01, "超过比亚迪…零跑的利润总和", becomes a `gt` check on 宁德时代 with no value and is supported.
   - v05, "比亚迪≈宁德1.9倍", is read as revenue in 倍 (unit mismatch).
   - v30: 摩根大通's US$ revenue and profit are bound to 中信证券 and contradicted.

Five of the 25 errors (v18, v22, v36, v44, v49) are in classes the first set's README already listed: spaced or
standalone year, traditional characters, 报收. The other 20 are in classes that are new or only adjacent to the
first set's. The collector knew the first set's classes. The
search agents did not, and selection used a fixed key, not the classes. Still, the agents were asked for English,
relations and loose numbers, the same request as for the first set, and those categories score lowest here.

**What a user would see.** Several published claims are true by the company's report but marked wrong:

- v19, v10: weighted ROE;
- v04, v35: 扣非 figures;
- v22, v23, v33, v44: traditional-character and English 工行 / 长江电力 / 茅台 statements, shown as unverifiable;
- v27, v31: a correct 总市值 only partly supported.

The data problems behind the labels are listed under "Data vs. the published reports" above. The checker problems are
the classes in this list.

## Round 14: after exposure (checker changed)

The set is **after exposure** from round 14 on. The round-14 engineers read the error list above and the first run's
errors (`claims_web_v2-first-run.json`), then wrote general rules for the classes with failing tests in their own
wording (`tests/test_agent_round14_claims_v2.py`, 46 cases, 34 failing before the fix; none of this set's sentences).
The labels and the scorer's scoring are unchanged; `score_claims_web_v2.py` gained `--after-exposure`, which only
labels the output.

**Run.** Once, on a clean tree at `0dd9e1e`, 2026-10-10T17:16Z:
`python -m evaluation.real_claims_web_v2.score_claims_web_v2 --after-exposure --out
evaluation/results/claims_web_v2-after-exposure-round14.json`.

| | blind first run (`117e74b`) | round 14, after exposure (`0dd9e1e`) |
|---|---|---|
| verdict accuracy | 0.500 [0.36, 0.64], 25/50 | **0.840 [0.74, 0.94], 42/50** |
| against the definition-blind labels | 0.520 [0.38, 0.66] | 0.760 [0.64, 0.88] |
| by category | annual_ratios 0.800, annual_figures 0.615, market_value 0.500, daily_price 0.462, loose_or_relational 0.250, english_or_traditional 0.000 | annual_ratios 1.000 (10/10), annual_figures 0.846 (11/13), daily_price 0.846 (11/13), loose_or_relational 0.750 (3/4), market_value 0.750 (3/4), english_or_traditional 0.667 (4/6) |
| checker's verdicts | supported 10, partially 21, contradicted 11, unverifiable 8 | supported 14, partially 25, contradicted 5, unverifiable 6 |

The 0.840 is not an estimate of accuracy on unseen claims: the rules were written for these classes. The blind number
is the first run's 0.500.

**Per error class** (errors before → after):

| class | claims | before | after | rule (round 14) |
|---|---|---|---|---|
| 1. named definition not read | v03, v10, v12, v19 | 4 | **0** | a named definition (加权平均ROE, 基本 / 稀释每股收益, 归母 growth, 营业总收入, 动态 / 静态 / TTM 市盈率) is checked only against a field the data states with that definition, else `definition_mismatch`; it carries to the growth and the parallel clause |
| 2. dated price on another day | v13, v17, v46 | 3 | 2 | a stated date uses that day's bar (`get_price_history` `on_date`): v17 fixed. v13 and v46 leave the date to the publication day, which the checker is not given |
| 3. market value on the price date | v20, v27, v31 | 3 | 1 | a valuation is dated by its valuation date: v27, v31 fixed. v20 ("以1.04万亿元居第五名…万亿市值") names the metric in a later clause and stays unread |
| 4. English claims | v11, v14, v23, v33 | 4 | 1 | names with "and" / "&" resolve before "and" becomes 和 (Industrial & Commercial Bank of China), China Yangtze Power aliases, "net profit attributable to shareholders", "earnings per share", a move of a reported flow is its YoY, foreign-currency amounts are `currency`. v33 is still wrong: its "attributable to the parent company" lies beyond the 40-character window the definition is read in, so the 归母 growth and the derived change are compared with the total-profit YoY (supported instead of partially) |
| 5. traditional Chinese | v22, v44 | 2 | 1 | a character table (`traditional.py`): v22 fixed. v44 is still wrong: its "人幣" becomes 人民币, which the runtime alias table lists as an alias of 宁德时代, so the YoY is bound to 宁德时代 |
| 6. 扣非 read as 归母 | v04, v35 | 2 | **0** | 扣非 is its own metric (`profit_dedt`), written out (扣除非经常性损益的净利润) or in brackets (归母净利润（扣非）); its YoY is unverifiable |
| 7. close words and price units | v02, v49 (v17's volume) | 2 | **0** | 报 / 报收 is the close; 成交量 in 手 / 万手 is the volume; 换手率 is not the daily change |
| 8. a year as a claimed number | v18, v36 | 2 | 1 | a year is never a value, and the year a change is measured from (和2024年对比) is not its period: v18 fixed. v36 is partially_supported (was unverifiable): "总利润 1347 亿" is not read as net profit |
| 9. relations outside the data | v01, v05, v30 | 3 | 1 | a comparison with a total of several companies is not checked; v30's US$ amounts are `currency`. v05 ("比亚迪≈宁德1.9倍") stays unread |
| new | v34 | 0 | 1 | "营业总收入858.82亿元" for 长江电力 is now `definition_mismatch` (the data serves 营业收入). The label treats the two as one figure because the annual report gives the same value for both; the data does not state that |

Remaining errors: v05, v13, v20, v33, v34, v36, v44, v46 (8). Two of them (v33, v44) are bugs this run found (the
definition window, the 人民币 alias); they are listed as open in the main README and not fixed in round 14.
