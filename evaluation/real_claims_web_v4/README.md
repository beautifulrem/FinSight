# Real published claims, fourth set, blind (`claims_web_v4.jsonl`)

**Name.** This set was collected on the r15-claims branch under `evaluation/real_claims_web_v3/` (files `*_v3`, claim ids
`w3_01` … `w3_50`) at the same time as the round-2 branch committed a different third set under that name. It was
renamed to `real_claims_web_v4` (files `*_v4`) after its first run; the ids keep their `w3_` prefix. The two sets were
collected independently (different collectors, candidates and URLs).

Real claims, as published by media, brokers' public notes, company announcements and social posts, about the
companies, metrics and dates FinSight's offline data covers. Like the second set (`evaluation/real_claims_web_v2/`),
the point is to measure the claim checker (`POST /agent/claim-check`) on wording nobody wrote for it, and the set is
**blind to the checker**: it was collected, selected and labelled without reading the checker's code, its tests or
the claim-bench rows. Unlike the second set, the collector also saw **no offline value and no earlier set's README
before the claims were selected** (below).

This is not the owner's real-claims set (`evaluation/human/real_claims/`), which is still owner-blocked.

## Files

| file | what |
|---|---|
| `candidates_v4.jsonl` | the 107 candidates collected from the web (each with `verified_by`, `period_basis`, `is_claim`, `multi_session`) |
| `select_claims_web_v4.py` | the fixed selection rules, candidates → 50 claims (reads no offline value) |
| `collected_v4.jsonl` | the 50 claims as collected, before any label (ids `w3_01` … `w3_50`) |
| `label_claims_web_v4.py` | computes every expected verdict from the data files (v2's rules plus this set's additions below) |
| `claims_web_v4.jsonl` | collected fields + labels, parts, primary-source comparison |
| `score_claims_web_v4.py` | v2's scorer with paths and names adapted; runs the checker once and writes `evaluation/results/claims_web_v4-first-run.json` (not run) |

```bash
python evaluation/real_claims_web_v4/select_claims_web_v4.py          # candidates_v4.jsonl -> collected_v4.jsonl (deterministic)
python -m evaluation.real_claims_web_v4.label_claims_web_v4           # collected_v4.jsonl -> claims_web_v4.jsonl (deterministic)
python -m evaluation.real_claims_web_v4.score_claims_web_v4           # the first run (not yet run)
```

## Protocol

**Phase 1: collection and selection, with no data value known.** Before `collected_v4.jsonl` was written, the
collector read no repository file except, through a script that printed only the `url` field, the URLs of
`evaluation/real_claims_web/collected_v1.jsonl` and `evaluation/real_claims_web_v2/collected_v2.jsonl` (the
do-not-use list). The session's standing project instructions (the FinSight repository's `AGENTS.md`, which
describes the agent layer and its rules but contains no checker code, claim data or data values) were in context
throughout. The collector was told which
companies, metrics and dates are checkable (the scope below), and not to select by correctness.

**Phase 2: labels, after `collected_v4.jsonl` was written.** Only then did the collector read:

- the data files: `data/structured_data.json`, `data/snapshot/structured_data_ext.json`,
  `data/snapshot/structured_data_fix.json`, `data/snapshot/corrections/sources.json` (the two manifests were not
  needed and not opened);
- `evaluation/real_claims_web_v2/README.md`, `label_claims_web_v2.py`, `score_claims_web_v2.py`, the first 60 lines
  of `select_claims_web_v2.py`, and the first rows of `collected_v2.jsonl` and `claims_web_v2.jsonl` (for formats).

**Never opened or searched:** `query_intelligence/`, `tests/`, `evaluation/claim_bench/`, `docs/`, `evaluation/results/`,
the root `README.md` / `README_CN.md`, anything under `/Volumes/Remi/finsight-review/`. The checker was not imported
and the scorer was not run.

**Exposure that remains.** The v2 README lists the v2 first run's error classes (named definitions, dated prices,
English claims, 扣非, relations with companies outside the data, …) and the round-14 changes. The collector read it
after selection and before labelling. Selection could not use it; the labels are computed from the data files, but
the part readings were written by someone who knew those classes. So the set is blind to the checker's code and
tests, and its selection is blind to the data and to the known error classes; its labels are not blind to the
classes.

**Collection.** 2026-10-11 (every `retrieved` field), by one collector with web search (Tavily search and extract;
Brave was tried and failed with "fetch failed"; one Tavily search failed with a TLS error and was retried). Every
candidate was checked on its page with `tavily_extract` (`verified_by`). Several 雪球 pages returned only their
comments at the first extract and were re-extracted with `extract_depth: advanced`; five candidates whose sentence
could not be found in extracted text kept `verified_by: tavily_search snippet only` and are dropped by the selection
rules. None of the candidate URLs is in the v1 or v2 sets.

**Checkable scope (given before searching).**

- FY2025 annual figures (report date 2025-12-31): revenue, 归母净利润, 扣非净利润, revenue and net-profit YoY, ROE,
  gross margin, net margin, EPS; relations between companies (A 比 B 多, A 是 B 的 N 倍, sums, averages, shares) and
  between metrics (扣非 vs 归母); market-value comparisons. Companies: 宁德时代, 比亚迪, 美的集团, 紫金矿业, 中芯国际,
  招商银行, 工商银行, 平安银行, 中信证券, 长江电力, 贵州茅台, 五粮液, 中国平安.
- Valuation (总市值, PE TTM, PB): on 2026-09-30 for the first ten; on 2026-04-22 for 茅台, 五粮液, 中国平安.
- Daily bars (close, % change, change in 元 / 点, 成交额, volume) on 2026-09-30 and 2026-04-22 for those stocks and
  上证指数, 创业板指, 沪深300.
- Excluded: quarterly and half-year figures, forecasts, targets, prices on other dates, live quote widgets.

**Quoting.** `published_text` is the sentence or clause as published (whole clauses only; two PDF line-break spaces
inside 实现 and 管控 were removed, noted in `note`). `claim_text` is the same text with an implicit subject or period
inserted in square brackets (`subject_inserted`: 24 of the 50); the scorer gives the checker `claim_text` with the
brackets removed and no date. `period_basis` records whether the period is in the sentence (`sentence`), in the
immediately surrounding text (`context`) or only in the dateline (`dateline`).

## Selection (107 candidates → 50), before any label

`select_claims_web_v4.py` applies fixed rules that never read an offline value (the only repository data it reads is
the `url` field of the v1 and v2 collected files):

| rule | dropped |
|---|---|
| not a claim (`is_claim` false) | 0 |
| not verified on the page (`verified_by` not `tavily_extract`) | 5 (four with a search snippet only, one multi-session weekly change that was also unverified) |
| period not stated (`period_basis` dateline or none) | 1 (宁德时代 股价291.11元 / 市值1.35万亿元, dated only by the article) |
| multi-session range | 0 left after the rule above |
| URL in the v1 or v2 set | 0 |
| duplicate text (normalised) | 0 |

The other 101 are ordered by sha256(candidate_id) and taken in that order with at most 4 per group (group = the
first-named entity) and 2 per URL, up to 50: 43 were skipped by the group cap, 1 by the URL cap, and 7 remained after
the 50th.

## Counts

- 107 candidates: 90 zh, 17 en; media 56, social 40 (雪球 18, 东方财富财富号 7, Substack 4, 有驾 3, 钛媒体钛度号 2,
  腾讯新闻创作者 2, 知乎 2, X 1, 新浪看点 1), company 7, broker 4. Periods: FY2025 100, 2026-09-30 4, 2026-04-22 2,
  multi-session 1.
- 50 selected claims: **43 zh, 7 en** (w3_09, w3_19, w3_25, w3_26, w3_30, w3_32, w3_34).
  - 18 of the 43 Chinese claims are ≤ 40 characters; the longest published sentence is 130 characters (w3_39).
- Source types: media 30; social 15 (雪球 8, 东方财富财富号 3, 知乎 2, Substack 1, 腾讯新闻创作者 1); company 3 (平安银行
  annual report ×2, 工商银行 website); broker 2 (券商点评 summaries on 证券之星).
- Groups (first-named entity): 4 each for 中国平安, 中芯国际, 五粮液, 宁德时代, 平安银行, 招商银行, 比亚迪, 紫金矿业, 美的集团,
  贵州茅台; 3 工商银行; 2 each for 上证指数, 中信证券, 长江电力; 1 中国人寿 (a three-company growth claim).
- Periods: FY2025 annual figures 48, 2026-09-30 index bars 2. No 2026-04-22 claim and no valuation claim was selected:
  only six dated candidates were found (four 2026-09-30, two 2026-04-22). Of the two 2026-04-22 ones, one failed the
  verification rule and one (c105) came after the 50th claim; the one valuation candidate (c019, 宁德时代 动态市盈率 /
  市净率 on 2026-09-30) lost to the 宁德时代 group cap, and c020 (dateline only) to the period rule.
- Publication date not shown on the extracted page: 17 (`published` null; `note` gives what is known).
- English and H-share wording: three claims cite an H-share code or an HKEX filing (w3_19, w3_28, w3_39).

The collector was asked for about eight English claims, relations, ratios and loose numbers; 17 English candidates
were found, and the fixed selection kept 7. Relations among the 50: w3_11 (相差近一倍), w3_47 (three companies' growth
超40%); sums of several companies and averages were collected (two sums with companies outside the data) but not
selected.

## Labelling rules (written before labelling)

Labels come from `label_claims_web_v4.py`, which reads only the data files. Its data reader, tolerance rules,
definition rule, primary comparison and verdict aggregation are copied from `label_claims_web_v2.py`; a person wrote
each claim's parts (company, metric, the number as written, a qualifier, the basis the sentence names), and every
value they are compared with is computed from the files. The rules are v2's, unchanged, plus the additions marked
**(new in this set)**, which were written in this README before any part was labelled.

- **Exact figure**: true when |claimed − actual| ≤ max(half a unit of the claim's last written digit, 1% of the actual
  value). A growth rate or change must also have the right sign.
- **约 / 左右 / about / approximately / 约合**: within 5% of the claimed number.
- **突破 / 站上 / 达到…以上** N: actual ≥ N. **超 / 超过 / 逾 / 以上 / more than / over / exceeding** N: actual > N.
  **近 / 将近 / 接近 / nearly / almost** N: 0.9 N ≤ actual ≤ N. **不到 / 不足 / 低于 / less than** N: actual < N.
  **N 多 / N 余**: N < actual < N + one unit of the last written digit. For a signed quantity the bound applies to the
  magnitude and the sign must match.
- **Relations**: "A 高于 / 超过 B" from the two served values. "A 是 B 的 N 倍" is A / B. A gap "比 B 多 X" is A − B.
- **Multiples of a difference (new in this set)**: "差 / 相差 N 倍" (and "甩 … N 倍") is the difference in multiples of the
  smaller value, larger / smaller − 1, under the rules above ("相差近一倍": 0.9 ≤ larger / smaller − 1 ≤ 1).
- **A sum of several companies as the compared side (new in this set)**: compared with the total of the served values of all
  named companies; null if any named company is not in the offline data.
- **An average across named companies (new in this set)**: the arithmetic mean of the served values; null if any is missing.
- **A share of a total (new in this set)**: A / (sum of the named parts) × 100; null if any part is missing.
- **扣非 vs 归母 (new in this set)**: a relation or gap between 扣非净利润 and 归母净利润 is computed from the two served fields
  (`profit_dedt`, `net_profit`). 扣非 YoY stays null (not in the data).
- **H-share / ADR / GDR / foreign currency (new in this set)**: a figure about the H-share, ADR or GDR line itself (its price,
  market value, volume, P/E) or stated in HK dollars is null: the data is A-share only. Amounts stated in US dollars or
  euros are null as in v2. A company's consolidated annual figure stays a company figure when the sentence also cites
  an H-share code or an HKEX filing; it is compared on the usual rules (its basis is handled by the definition rule).
  An RMB amount given as a conversion of a foreign-currency amount ("约合1267.5亿元人民币") is an RMB claim.
- **Derived values**: net margin = net profit / revenue; prior-year value = value / (1 + YoY); change in 元 / 点 =
  close − previous close.
- **Definitions** (v2): a generic word is compared with the served value whatever its basis. A sentence that names a
  basis different from the served one (加权平均 or 平均 ROE vs period-end, 基本 vs diluted EPS, 营业总收入 vs 营业收入,
  归母 growth vs a total-net-profit YoY, 动态 vs TTM P/E) gives a null part, unless the annual report in
  `sources.json` shows the two bases give the same figure. `expected_verdict_definition_blind` ignores every named
  basis.
- **Not in the offline data** (null), as v2: 换手率, quarterly figures, prior-year margins or ROE, changes in percentage
  points, 扣非 YoY, ROA, 营运利润, segment margins, capacity utilisation, dividends, R&D, unit sales, USD / EUR / HKD
  amounts, other indices (深证成指, 科创50), other companies, valuation on a date other than the stated valuation date or
  with no stated date, per-day figures ("日赚", "每天净赚").
- **Verdict**: every part true → supported; no part true but at least one false → contradicted; some true, others
  false or null → partially_supported; nothing decided → unverifiable.
- **Context without a number of its own** (创历史新高, 稳居榜首, 盈利王, 首次) is not a part (`context_not_checked`).

Each row of `claims_web_v4.jsonl` carries `category`, `expected_verdict`, `expected_verdict_definition_blind`,
`parts` (reading, `holds`, offline `actual` and rule, or the null reason), `context_not_checked`, `primary` (the annual
report's figure on the claim's basis, `offline_vs_primary`, `claim_vs_primary`) and a per-claim `offline_vs_primary`.

## Labels

| | |
|---|---|
| expected verdicts | supported 20, partially_supported 22, contradicted 4, unverifiable 4 |
| definition-blind | supported 21, partially_supported 21, contradicted 8 |
| categories | annual_figures 28, annual_ratios 8, english_or_traditional 7, loose_or_relational 5, daily_price 2 |
| source types | media 30, social 15, company 3, broker 2 |

**Labels that change under the definition rule** (5 claims):

- w3_24 (紫金 基本每股收益1.95), w3_33 (紫金 加权平均ROE 33.04%), w3_43 (中信证券 加权平均净资产收益率10.58%), w3_29 (招行
  平均净资产收益率 13.44%): unverifiable → contradicted. The named basis (basic EPS, weighted / average ROE) is the
  report's and differs from the served one (diluted EPS 1.91; period-end ROE 27.91, 9.40, 11.80).
- w3_18 (长江电力): partially_supported → supported. Its 归母 growth 6.17% is null against the total-profit YoY 6.13
  but passes when the basis is ignored.

Other claims name 归母 growth too (w3_08, w3_14, w3_15, w3_16, w3_19, w3_23, w3_27, w3_28, w3_41); their verdict is
partially_supported either way.

**Data vs. the published reports (from the `primary` columns).** No annual figure in this set shows an offline value
that differs from the report on the same basis, except 中国平安's 归母 YoY: the corrections layer serves the report's
one-decimal 6.5, the figures give 6.454 (w3_40, `value_differs`; w3_39 and w3_40's "6.45%" pass through the 1% rule).
The other disagreements are about what the sentence quotes:

- **IFRS figures from Hong Kong.** 工商银行's 营业收入 8013.95亿 (+1.9%) in w3_27, w3_28, w3_45 is the IFRS operating
  income; the A-share report's 营业收入 is 8382.70亿 (+2.0%), which the data serves, so those parts are false. w3_28's
  "归属于母公司股东的净利润3707.66亿元" is the IFRS total net profit; it passes against the served 归母 3685.62亿 through
  the 1% rule. 中国平安's 营业收入 11403.24亿 (w3_39) is the HKEX total income against the report's 10505.06亿.
- **Preliminary figures.** w3_08 and w3_43 quote 中信证券's 业绩快报 (748.30亿, 300.51亿, +38.46%, weighted ROE 10.58%);
  the annual report gives 748.54亿, 300.76亿, +38.58%, 10.59%.
- **Round and converted numbers.** w3_05's "营收4000亿" is false as an exact figure (4237.02亿). w3_36's "约合1267.5亿元
  人民币" (a US-dollar conversion) is false against 归母 1347.78亿. w3_11's "相差近一倍" gives larger / smaller − 1 =
  0.897, just under the 0.9 bound.
- **Prices.** w3_12's "截至收盘，沪指涨0.27%…创业板指跌0.17%" sits in the page's 11:31 午评 block; against the
  2026-09-30 close (+0.31%, −0.23%) both parts are false. w3_49 states the close figures and is partially_supported
  (深证成指 is not in the data).

## First run (the only blind run), `110e656`

The checker was run once, on a clean tree at `110e656` (r15-claims, the round-15 K6 checker), on 2026-10-10T20:35Z,
after the claims, labels and scorer had been committed (`327629c`, `a4b666f`). The set then lived under
`evaluation/real_claims_web_v3/`; the command was `python -m evaluation.real_claims_web_v3.score_claims_web_v3`. It used
the offline service with the shipped default data and no LLM, the published sentence with the brackets removed (the
inserted subject kept) and no date. Result: `evaluation/results/claims_web_v4-first-run.json` (written by that run; only
the `renamed` and `claims_sha256` fields were added when the set was renamed).

**What the engineer had seen before the run.** The engineer who ran it had written the checker (round 15). The claims
were collected and labelled by a separate search agent that did not open the checker's code, tests or claim benches.
The engineer did not read the claims before the run, except three lines of one label note (w3_28's note on ICBC's IFRS
figures) that a lint command printed.

| | result |
|---|---|
| verdict accuracy | **0.740 [0.62, 0.86]** (37/50; 95% percentile bootstrap over claims, 2000 resamples) |
| against the definition-blind labels | 0.720 [0.60, 0.84] |
| by category | daily_price 1.000 (2/2), annual_ratios 0.875 (7/8), annual_figures 0.821 (23/28), loose_or_relational 0.600 (3/5), english_or_traditional 0.286 (2/7) |
| by language | zh 0.814 (35/43), en 0.286 (2/7) |
| by expected verdict | supported 0.800 (16/20), partially_supported 0.682 (15/22), contradicted 0.750 (3/4), unverifiable 0.750 (3/4) |
| checker's verdicts | supported 19, partially_supported 18, contradicted 7, unverifiable 6 |

This is a blind number for the round-15 K6 checker. It is higher than the second and third sets' first runs (0.500 and
0.540, both on the round-14 checker), but the sets differ in mix (this one is 48/50 annual figures and has no
valuation claims), so the difference is not an estimate of the K6 rules' effect.

### Error classes of the first run (13)

1. **A definition caution alone, under two verdict rules** (2: w3_08, w3_18). The claims name 归母 growth and the
   data's YoY is on total net profit; every other part is true. Round 15 calls such a claim `supported` (partial
   coverage); this set's labels (v2's rule) call it `partially_supported`.
2. **A previous-year value read as this year's** (2: w3_19, w3_20). "较2024年的4090.84亿元" is compared with 2025 revenue
   (period_mismatch); "down from 40.3 billion yuan the previous year" and the rest of w3_19 are read as daily moves.
3. **Metric wording** (4: w3_23, w3_25, w3_29, w3_47). "归属于公司拥有人的利润" (Hong Kong wording) and "业绩增长" are not
   recognised; "essentially flat (+0.01%)" has no metric; "平均净资产收益率（ROAE）13.44%" is compared with the
   period-end ROE and contradicted (labelled unverifiable: the served ROE is another definition).
4. **A daily average read as the year's net profit** (1: w3_35). "每天净赚超过1.2亿元" was checked as net profit > 1.2亿,
   which is trivially true: a false accept of the wrong quantity.
5. **人民币 as 宁德时代** (1: w3_28). The sentence's "人民币" resolved to 宁德时代 through the runtime alias table, so ICBC's
   figures were compared with CATL's.
6. **A name variant** (1: w3_30). "Kweichow Maotai" is not recognised.
7. **English moves of a reported flow** (1: w3_32). "revenues plunged to CY40.53 billion, almost 55% below sales in 2024"
   is read as daily price moves.
8. **Tolerance** (1: w3_34). The 16.2% growth against 16.49 is within the checker's 2% and outside the labels' 1%.


## Round 15: after exposure

After its first run this set is **after exposure**: the engineer read its 13 errors (above). No rule was written for
this set's sentences; the round-15 rules written for the third set's classes ([../real_claims_web_v3/README.md](../real_claims_web_v3/README.md))
overlap several of its classes. Run once on a clean tree at `a02821d`, 2026-10-10T22:07Z:
`python -m evaluation.real_claims_web_v4.score_claims_web_v4 --after-exposure --out
evaluation/results/claims_web_v4-after-exposure-round15.json`.

| | blind first run (`110e656`) | round 15, after exposure (`a02821d`) |
|---|---|---|
| verdict accuracy | 0.740 [0.62, 0.86], 37/50 | **0.780 [0.66, 0.88], 39/50** |
| against the definition-blind labels | 0.720 [0.60, 0.84] | 0.760 [0.64, 0.88] |
| by language | zh 35/43, en 2/7 | zh 36/43, en 3/7 |

The blind number is the first run's 0.740.

**Per error class** (errors before → after):

| class | claims | before | after | what changed |
|---|---|---|---|---|
| 1. a definition caution alone, under two verdict rules | w3_08, w3_18 | 2 | 3 | unchanged; w3_15 joins it ("日赚4.11亿元" is now checked and supported, leaving only the 归母-growth caution) |
| 2. a previous-year value read as this year's | w3_19, w3_20 | 2 | **0** | level / (1 + YoY), in Chinese and English |
| 3. metric wording | w3_23, w3_25, w3_29, w3_47 | 4 | 3 | "归属于公司拥有人的利润" is read (a bare 利润); "essentially flat (+0.01%)", ROAE, 业绩增长 are not |
| 4. a daily average read as the year's net profit | w3_35 | 1 | 1 | now checked as net profit / 365 (1.204亿 > 1.2亿, supported); the label treats a daily average as not checkable (partially_supported) |
| 5. 人民币 as 宁德时代 | w3_28 | 1 | **0** | the alias row is gone; ICBC's IFRS figures are compared with ICBC's |
| 6. a name variant | w3_30 | 1 | 1 | "Kweichow Maotai" is not recognised |
| 7. English moves of a reported flow | w3_32 | 1 | 1 | "plunged to CY40.53 billion, almost 55% below sales in 2024" is now unverifiable (no date; the latest session does not fit), not contradicted |
| 8. tolerance | w3_34 | 1 | 1 | unchanged (2% vs the labels' 1%) |
| new: a daily figure with no date | w3_12 | 0 | 1 | "截至收盘，沪指涨0.27%…" does not fit the latest session (+0.31%): unverifiable under the round-15 rule, labelled contradicted |

## Round 16: after exposure, per class

Re-run once on a clean tree at `c795172` (r16-claims, the round-16 claims code: the round-12 review's L4-L6 parts
without a number, growth-rate comparisons, book value and L9 short names; no rule was written for this set), 2026-10-11:
`python -m evaluation.real_claims_web_v4.score_claims_web_v4 --after-exposure --out
evaluation/results/claims_web_v4-after-exposure-round16.json`.

| | blind first run (`110e656`) | round 15 (`a02821d`) | round 16 (`c795172`) |
|---|---|---|---|
| verdict accuracy | 0.740 [0.62, 0.86], 37/50 | 0.780 [0.66, 0.88], 39/50 | **0.780 [0.66, 0.88], 39/50** |
| against the definition-blind labels | 0.720 [0.60, 0.84] | 0.760 [0.64, 0.88] | 0.760 [0.64, 0.88] |
| by language | zh 35/43, en 2/7 | zh 36/43, en 3/7 | zh 36/43, en 3/7 |

**Per category at round 16:** annual_figures 0.893 (25/28), annual_ratios 0.875 (7/8), daily_price 0.500 (1/2),
english_or_traditional 0.429 (3/7), loose_or_relational 0.600 (3/5), unchanged from round 15. One verdict moved, still
wrong: w3_32 (class 7, English moves of a reported flow) is partially supported instead of unverifiable: "net profit
nosedived by 72%" now reads as a fall (−72 against −71.89, supported) while "almost 55% below sales in 2024" is still
read as a daily move. The eleven errors are the same claims as in round 15 (classes 1, 3, 4, 6, 7, 8 and the no-date
rule); the round-16 rules changed no other verdict in this set. The blind number is the first run's 0.740.
