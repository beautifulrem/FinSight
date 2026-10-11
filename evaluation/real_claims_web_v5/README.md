# Real published claims, fifth set, blind (`claims_web_v5.jsonl`)

**Name.** Fifth set of real published claims, collected on 2026-10-11 under `evaluation/real_claims_web_v5/` (files
`*_v5`, claim ids `w5_01` … `w5_50`). It follows the fourth set's protocol (`evaluation/real_claims_web_v4/`) step by
step; the collector is a different session from the one that collected the fourth set.

Real claims, as published by media, brokers' public notes, company announcements and social posts, about the
companies, metrics and dates FinSight's offline data covers. As with the fourth set, the point is to measure the claim
checker (`POST /agent/claim-check`) on wording nobody wrote for it, and the set is **blind to the checker**: it was
collected, selected and labelled without reading the checker's code, its tests or the claim-bench rows, and the
collector saw **no offline value before the claims were selected** (below).

This is not the owner's real-claims set (`evaluation/human/real_claims/`), which is still owner-blocked.

## Files

| file | what |
|---|---|
| `candidates_v5.jsonl` | the 101 candidates collected from the web (each with `verified_by`, `period_basis`, `is_claim`, `multi_session`) |
| `select_claims_web_v5.py` | the fixed selection rules, candidates → 50 claims (reads no offline value); v4's script with paths, the do-not-use list and the id prefix changed |
| `collected_v5.jsonl` | the 50 claims as collected, before any label (ids `w5_01` … `w5_50`) |
| `label_claims_web_v5.py` | computes every expected verdict from the data files (v4's rules plus this set's additions below) |
| `claims_web_v5.jsonl` | collected fields + labels, parts, primary-source comparison |
| `score_claims_web_v5.py` | v4's scorer with paths and names adapted; runs the checker once and writes `evaluation/results/claims_web_v5-first-run.json` (not run) |

```bash
python evaluation/real_claims_web_v5/select_claims_web_v5.py          # candidates_v5.jsonl -> collected_v5.jsonl (deterministic)
python -m evaluation.real_claims_web_v5.label_claims_web_v5           # collected_v5.jsonl -> claims_web_v5.jsonl (deterministic)
python -m evaluation.real_claims_web_v5.score_claims_web_v5           # the first run (not yet run)
```

## Protocol

**Phase 1: collection and selection, with no data value known.** Before `collected_v5.jsonl` was written, the
collector read only:

- `evaluation/real_claims_web_v4/README.md`, lines 1-160 (stopping inside the labelling-rules list, well before the
  section "## First run": the name, files, protocol, collection, scope, selection, counts and the first labelling
  rules), and `evaluation/real_claims_web_v4/select_claims_web_v4.py`;
- through a one-line script that printed only the `url` field, the URLs of `evaluation/real_claims_web/collected_v1.jsonl`,
  `real_claims_web_v2/collected_v2.jsonl`, `real_claims_web_v3/collected_v3.jsonl` and `real_claims_web_v4/collected_v4.jsonl`
  (the do-not-use list, 190 URLs).

Two directory listings were also made: `evaluation/real_claims_web_v4/` (file names only, to see whether it has an
`__init__.py`; it has none, so this folder has none) and `evaluation/` (directory names only, nothing opened). The
session's standing project instructions (the FinSight repository's `AGENTS.md`, which describes the agent layer and
its rules but contains no checker code, claim data or data values) were in context throughout. The collector was told
which companies, metrics and dates are checkable (the scope below), and not to select by correctness.

Web search results inevitably show published figures (that is what was being collected); none of the repository's data
files was opened in phase 1, and no candidate was kept or left out because of its value. A few further English pages
repeating figures already collected (e.g. a second Gasgoo sentence, ecomotorsnews, Vino Joy, a second PR Newswire
sentence) were not entered, to keep English near the requested 15-20% of candidates; they were left out by language
and repetition, not by value.

**Phase 2: labels, after `collected_v5.jsonl` was written.** Only then did the collector read:

- the data files: `data/structured_data.json`, `data/snapshot/structured_data_ext.json`,
  `data/snapshot/structured_data_fix.json`, `data/snapshot/corrections/sources.json`;
- `evaluation/real_claims_web_v4/label_claims_web_v4.py`, `score_claims_web_v4.py`, and the first rows of
  `collected_v4.jsonl` and `claims_web_v4.jsonl` (for formats).

To run the linter the collector also read `ruff.toml` (its `include` list does not cover this folder, so ruff was run on
the folder and, to actually lint, on the three scripts by name), `.gitignore`, a listing of the repository root, and
`git status --short` (read-only; it shows file names only, nothing was opened).

After reading those first rows, two format-only fields of `candidates_v5.jsonl` were rewritten to v4's style
(`verified_by` as "tavily_extract 2026-10-11 (sentence found in extracted page text)", `social_platform` as slugs such
as `xueqiu`, `eastmoney_caifuhao`); the selection script was re-run and produced the identical 50 claims (same ids,
candidates and claim texts). No published text, claim text, period, basis or verification status changed.

**Never opened or searched:** `query_intelligence/`, `tests/`, `evaluation/claim_bench/`, `docs/`, `evaluation/results/`,
`evaluation/heldout_*/`, the root `README.md` / `README_CN.md`, anything under `/Volumes/Remi/finsight-review/`, and the
rest of the v4 README (from line 161, including its first-run section). The checker was not imported, the scorer was
not run, pytest was not run.

**Exposure that remains.** The collector knows the v4 labelling rules (README lines 128-160 and the v4 label script)
and, from the v4 script's part notes, the kinds of mismatch v4 met (IFRS operating income vs 营业收入, 归母 vs total
net-profit growth, weighted vs period-end ROE, preliminary vs final figures). Selection could not use any of this (it
was read after `collected_v5.jsonl` was written). The labels are computed from the data files; the part readings were
written by someone who knew those classes. So the set is blind to the checker's code and tests, and its selection is
blind to the data; its labels are not blind to the v4 classes.

**Collection.** 2026-10-11 (every `retrieved` field), by one collector with web search (Tavily search and extract,
loaded through the MCP tool search). Brave was not needed and not called. The `grok-cli` MCP server failed to connect at
session start (not used). Every candidate was checked on its page with `tavily_extract` in query mode (`verified_by`);
where the first extract returned no article text the page was re-extracted with `extract_depth: advanced` (three
candidates were found that way). Four candidates whose sentence could not be found in extracted text (two pages
returned nothing even with `advanced`, one 股吧 page returned only an avatar line, one Instagram page returned no caption)
keep `verified_by: tavily_search snippet only` and are dropped by the selection rules. One 新浪财经 broker-summary page
(`vReport_Show`) came back in a garbled encoding and was not entered. None of the candidate URLs is in the v1-v4 sets.

**Checkable scope (given before searching).**

- FY2025 annual figures (report date 2025-12-31): revenue, 归母净利润, 扣非净利润, revenue and net-profit YoY, ROE,
  gross margin, net margin, EPS; relations between companies (A 比 B 多, A 是 B 的 N 倍, sums, averages, shares, growth
  comparisons) and between metrics; market-value comparisons. Companies: 宁德时代, 比亚迪, 美的集团, 紫金矿业, 中芯国际,
  招商银行, 工商银行, 平安银行, 中信证券, 长江电力, 贵州茅台, 五粮液, 中国平安.
- Valuation (总市值, PE TTM, PB / 市净率 / 破净): on 2026-09-30 for the first ten; on 2026-04-22 for 茅台, 五粮液, 中国平安.
- Daily bars (close, % change, change in 元 / 点, 成交额, volume) on 2026-09-30 and 2026-04-22 for those stocks and
  上证指数, 创业板指, 沪深300.
- Excluded: quarterly and half-year figures, forecasts, targets, prices on other dates, live quote widgets.

**Quoting.** `published_text` is the sentence or clause as published (whole clauses only; where a clause was cut from a
longer sentence, `note` says so). `claim_text` is the same text with an implicit subject or period (in three claims also
the metric the context names: 营业收入, 总市值, 加权平均) inserted in square brackets (`subject_inserted`: 31 of the
50); the scorer gives the checker `claim_text` with the brackets removed (the inserted words stay) and no date. In w5_34 the bracket inserts
"2025年加权平均" because the paragraph the clause sits in is about 加权平均净资产收益率; this was decided in phase 1.
`period_basis` records whether the period is in the sentence (`sentence`), in the immediately surrounding text
(`context`), only in the dateline (`dateline`) or nowhere on the page (`none`).

## Selection (101 candidates → 50), before any label

`select_claims_web_v5.py` applies fixed rules that never read an offline value (the only repository data it reads is
the `url` field of the v1-v4 collected files):

| rule | dropped |
|---|---|
| not a claim (`is_claim` false) | 0 |
| not verified on the page (`verified_by` not `tavily_extract`) | 4 (two 宁德时代 snippets, the 新浪股吧 index post, the Instagram caption) |
| period not stated (`period_basis` dateline or none) | 3 (宁德时代 股价291.11元 / 市值1.35万亿元 dated only by the article; the Dow Jones "Today it is up 11.74 points" line dated only by the wire dateline; a 网易号 招商银行 close with no date on the page) |
| multi-session range | 1 (建设银行、工商银行 "自2023年起连续上涨，但至今仍未摆脱破净区间") |
| URL in the v1-v4 sets | 0 |
| duplicate text (normalised) | 0 |

The other 93 are ordered by sha256(candidate_id) and taken in that order with at most 4 per group (group = the
first-named entity) and 2 per URL, up to 50: 42 were skipped by the group cap, none by the URL cap, and 1 remained after
the 50th (c080, the only 五粮液 candidate).

## Counts

- 101 candidates: 82 zh, 19 en; media 62, social 33 (雪球 8, 东方财富财富号 5, 钛媒体钛度号 4, 搜狐号 4, 新浪财经自媒体 3,
  网易号 2, 新浪看点 2, X 1, 知乎 1, 新浪股吧 1, Facebook 1, Instagram 1), company 5, broker 1. Periods: FY2025 76,
  2026-09-30 20, 2026-04-22 3, undated 1, multi-session 1. 57 candidates have an inserted subject or period.
- First-named entities among the candidates: 宁德时代 21, 贵州茅台 12, 招商银行 11, 上证指数 11, 比亚迪 7, 中信证券 7,
  工商银行 6, 平安银行 5, 紫金矿业 4, 长江电力 4, 美的集团 3, 中国平安 3, 中芯国际 2, 沪深300 2, and 1 each for 五粮液,
  建设银行, 上证50.
- 50 selected claims: **41 zh, 9 en** (w5_01, w5_04, w5_05, w5_06, w5_17, w5_29, w5_36, w5_37, w5_40).
  - 11 of the 41 Chinese claims are ≤ 40 characters; the longest published sentence is 118 characters.
- Source types: media 29; social 16 (雪球 4, 搜狐号 3, 东方财富财富号 2, 钛媒体钛度号 2, 知乎 1, 新浪财经自媒体 1, 新浪看点 1,
  X 1, 网易号 1); company 4 (平安银行 website ×2, 美的集团 investor page, Ping An press release); broker 1 (a 国信证券
  note summarised by 格隆汇).
- Groups (first-named entity): 4 each for 上证指数, 中信证券, 宁德时代, 工商银行, 平安银行, 招商银行, 比亚迪, 紫金矿业,
  贵州茅台, 长江电力; 3 each for 中国平安, 美的集团; 2 中芯国际; 1 each for 沪深300, 上证50. No 五粮液 claim (its one
  candidate came after the 50th).
- Periods: FY2025 annual figures 39, 2026-09-30 10, 2026-04-22 1.
  - The ten 2026-09-30 claims: five index claims (closes, changes, one comparison), three stock bars (宁德时代 close,
    紫金矿业 bar, 工商银行 bar) and two market values (中信证券, 招商银行); the 2026-04-22 claim is an index change. Compared with the fourth set
    (no valuation claim, two dated index claims), more dated claims were collected (23 candidates, 20 on 2026-09-30) and
    two valuation claims were selected. No 2026-04-22 claim about 茅台, 五粮液 or 中国平安 (bar or valuation) was found
    by the searches; no PE / PB claim with a stated date was found for any company (the 破净 claim was dropped as
    multi-session).
- Publication date not shown on the extracted page: 21 of the 50 (`published` null; `note` gives what is known).

## Labelling rules (written before labelling)

Labels come from `label_claims_web_v5.py`, which reads only the data files. Its data reader, tolerance rules,
definition rule, primary comparison and verdict aggregation are copied from `label_claims_web_v4.py`; a person wrote
each claim's parts (company, metric, the number as written, a qualifier, the basis the sentence names), and every
value they are compared with is computed from the files. The rules are v4's, unchanged, plus the additions marked
**(new in this set)**, which were written in this README before any part was labelled.

- **Exact figure**: true when |claimed − actual| ≤ max(half a unit of the claim's last written digit, 1% of the actual
  value). A growth rate or change must also have the right sign.
- **约 / 左右 / about / approximately / 约合**: within 5% of the claimed number.
- **突破 / 站上 / 达到…以上** N: actual ≥ N. **超 / 超过 / 逾 / 以上 / more than / over / exceeding** N: actual > N.
  **近 / 将近 / 接近 / nearly / almost** N: 0.9 N ≤ actual ≤ N. **不到 / 不足 / 低于 / less than** N: actual < N.
  **N 多 / N 余**: N < actual < N + one unit of the last written digit. For a signed quantity the bound applies to the
  magnitude and the sign must match.
- **Relations**: "A 高于 / 超过 / 强于 B" from the two served values. "A 是 B 的 N 倍" is A / B. A gap "比 B 多 X" is A − B.
  "差 / 相差 N 倍" is larger / smaller − 1. A sum, average or share of several companies is computed from the served
  values and is null if any named company is not in the offline data (v4).
- **扣非 vs 归母** relations from the two served fields; 扣非 YoY stays null (not in the data) (v4).
- **H-share / ADR / GDR / foreign currency** (v4): figures about an H-share line or in HK dollars are null; amounts in US
  dollars or euros are null; an RMB amount given as a conversion of a foreign-currency amount is an RMB claim; a growth
  rate stated alongside a US-dollar amount is compared with the served growth rate, as v4 did.
- **Derived values** (v4): net margin = net profit / revenue; prior-year value = value / (1 + YoY); change in 元 / 点 =
  close − previous close.
- **Definitions** (v2 / v4): a generic word (净利润, net profit, 收入, ROE, 净利率) is compared with the served value
  whatever its basis. A sentence that names a basis (归母, 加权平均, 基本 / 稀释, 营业总收入 / 营业收入) different from the
  served one makes that part null in `expected_verdict`, unless the annual report shows the two bases give the same
  figure for that company; `expected_verdict_definition_blind` compares the same part with the served value anyway.
- **Valuation and bars**: a market value, PE or PB is compared only when the served valuation date equals the claim's
  date; a bar only when the served bars contain the claim's date. Market value in 亿 / 万亿 is compared with the served
  total market value (`total_mv`).
- **Not checkable** (null; v4's practice, applied to this set's entities and metrics): changes in percentage points (the prior-year ratio is not in the data), per-day
  figures (日赚), rankings and superlatives (榜首, 第五名, 历史新高, 首次), turnover rates, entities not in the data
  (深证成指, 北证50, 科创50, 上证50, 农业银行, 交通银行, 江苏银行, 宁波银行, 吉利, 上汽, 长城, 长安, 中泰证券, 国泰海通,
  Tesla), metrics not in the data (营运利润, 总资产, 毛利润, segment margins, dividends, buybacks, payout ratios), and
  ratios to a date outside the scope (a July low). Rankings and superlatives are kept in `context_not_checked`.
- **Double-digit growth (new in this set)**: "双位数增长 / 两位数增长 / double-digit growth" holds when
  10 ≤ growth < 100 (%), with a positive sign. Qualifier `double_digit` (claimed value written as "10").
- **Intraday highs and lows (new in this set)**: "盘中触及 / 盘中最高 / 盘中最低" price levels are outside the set's bar
  scope (close, % change, change, 成交额, volume) and are null, although the data files carry `high` and `low`.
- **Units (new in this set, no rule change)**: "million" (as in RMB143,773 million) is 1e6, and "万手" volume is
  compared in shares (1 手 = 100 股) with the served bar's share volume.
- **Verdict** (v2 / v4): supported if every part holds; partially_supported if at least one part holds and another
  does not or cannot be checked; contradicted if no part holds and at least one is false; unverifiable if every part is
  null.

## Label counts

`python -m evaluation.real_claims_web_v5.label_claims_web_v5` (deterministic; the selection and the labels were
re-run after `ruff format` and gave byte-identical files):

| | supported | partially_supported | contradicted | unverifiable |
|---|---|---|---|---|
| `expected_verdict` (definition-aware) | 12 | 33 | 2 | 3 |
| `expected_verdict_definition_blind` | 15 | 31 | 3 | 1 |

- Five claims differ between the two labels (w5_10, w5_20, w5_34, w5_41, w5_44): each names a basis (归母 growth or
  加权平均 ROE) that differs from the served one.
- Categories: annual_figures 24, english_or_traditional 9 (every English claim, as in v4), daily_price 8, annual_ratios
  5, market_value 2, loose_or_relational 2.
- 174 parts: 106 hold, 9 do not, 59 are null (USD amounts, entities and metrics not in the data, percentage-point
  changes, per-day figures, 扣非 YoY, the definition rule, and 沪深300 on 2026-09-30, which has no served bar).
- The new double-digit rule is used once (w5_12, two parts); the intraday rule once (w5_24); the "million" unit once
  (w5_37).

## Data vs published reports

`primary` compares each annual figure with the annual report's figure on the claim's basis (`sources.json`):

- Agree (every compared part): 18 claims. Agree, with a part whose named basis differs from the served one: 12 claims
  (11 net-profit growth rates named 归母 against a served growth of total net profit, for 招商银行, 紫金矿业, 长江电力,
  中信证券 and 美的集团; one 营业总收入 for 贵州茅台, where 营业总收入 and 营业收入 differ). w5_34 names 加权平均 ROE against
  the served period-end ROE (definition_differs only).
- One value difference: 中国平安's served 归母净利润 growth is 6.5 (the report's one-decimal figure); computed from the
  report's amounts it is 6.45% (w5_43; the claim's 6.5% holds against both).
- Notes written by the labeller (`primary_note`): net margins published on total net profit (宁德时代 18.12%, 中芯国际
  10.7%) against the rule's 归母 net margin (w5_07, w5_11, w5_18); 长江电力's 15.59% labelled 加权平均 is the period-end
  ROE the data serves, the weighted ROE being 15.90% (w5_10); 工商银行's 801.4bn / +1.9% is the IFRS operating income,
  not the A-share 营业收入 (w5_36); 中芯国际's 16.2% / 24.9% growth goes with US-dollar figures (w5_27).
- Claims whose stated figure disagrees with the annual report itself (`claim_vs_primary` false): w5_08 (ROE), w5_10
  (weighted ROE), w5_27 (both growth rates), w5_36 (operating income and its growth).
- Prices and market values have no independent record in the data files (11 claims: the ten 2026-09-30 claims and
  the 2026-04-22 claim).

## First run (the only blind run), `c795172`

The checker was run once, on a clean tree at `c795172` (r16-claims, the final round-16 claims code: review-12 L4-L6
parts without a number, growth-rate comparisons, book value, L9 short names), on 2026-10-11T02:43Z, after the set had
been committed unread (`ac7ea79`): `python -m evaluation.real_claims_web_v5.score_claims_web_v5`. It used the offline
service with the shipped default data and no LLM, the published sentence with the brackets removed (the inserted
subject kept) and no date. Result: `evaluation/results/claims_web_v5-first-run.json` (written by that run).

**What the engineer had seen before the run.** The engineer who ran it wrote the round-16 checker rules. The claims were
collected and labelled by a separate search agent that did not open the checker's code, tests, claim benches, docs or
results; its report to the engineer named counts only. The engineer did not read the claims or labels before the run.
The round-16 rules were written for the round-12 review's classes (L4-L6, L9) with the engineer's own test wordings,
not for this set.

| | result |
|---|---|
| verdict accuracy | **0.680 [0.56, 0.80]** (34/50; 95% percentile bootstrap over claims, 2000 resamples, seed 20261010) |
| against the definition-blind labels | 0.700 [0.58, 0.82] |
| by category | annual_ratios 0.800 (4/5), daily_price 0.750 (6/8), annual_figures 0.667 (16/24), english_or_traditional 0.667 (6/9), loose_or_relational 0.500 (1/2), market_value 0.500 (1/2) |
| by language | zh 0.683 (28/41), en 0.667 (6/9) |
| by expected verdict | supported 0.667 (8/12), partially_supported 0.727 (24/33), contradicted 0.000 (0/2), unverifiable 0.667 (2/3) |
| checker's verdicts | supported 15, partially_supported 26, contradicted 2, unverifiable 7 |

This is a blind number for the round-16 checker. It is lower than the fourth set's first run (0.740, round-15 K6
checker); the sets differ in mix (this one has 10 dated price and market-value claims, the fourth had 2), so the
difference is not an estimate of a change in the checker.

### Error classes of the first run (16)

Read after the run (from here on the set is after exposure):

1. **A definition caution alone, under two verdict rules** (7: w5_02, w5_19, w5_20, w5_28, w5_32, w5_41, w5_44). The
   claims name 归母 growth (w5_19 also 营业总收入) and the data's YoY is on total net profit; every other part is true.
   Round 15 calls such a claim `supported` with partial coverage; this set's labels (v2's rule) call it
   `partially_supported` (w5_20, w5_41 and w5_44 are `supported` under the definition-blind labels).
2. **A figure or move the checker does not read** (4). "rising 0.31 percent" with no metric word and "heads into …
   reopening at 3,842.19 points" read as an ADS line (the chat's foreign-line detector matches "ads" inside "heads",
   `coverage.foreign_line_spans`) (w5_01); "operating income … approximately 168.838 billion yuan" and "April 2026" read
   as a value (w5_17); "以8382.70亿元2025年营业收入蝉联榜首" (the metric after the number and a year) (w5_39); "沪深300（+0.29%）
   明显强于创业板指（-0.23%）" (a change in brackets with no move word) (w5_46).
3. **A reference price read as a hypothetical** (1: w5_03). "以…时公司股票收盘价291.11元/股作为参照" is unverifiable as a
   forecast or hypothetical; it states a close.
4. **A five-company sum with companies outside the data** (1: w5_23, a false contradiction). "比亚迪、吉利、上汽、长城、长安
   五家…净利润加在一起是735亿元" is checked as 比亚迪's own net profit (contradicted); the sum includes four companies
   the data does not have, so it should be unverifiable.
5. **An English 扣非 phrase** (1: w5_37, a false contradiction). "net profit attributable to shareholders of the parent
   company excluding non-recurring gains and losses" is read as 归母 net profit (1347.78亿 against 1437.73亿), not as
   扣非 net profit, which matches.
6. **A gap to an unnamed second place read as a bound** (1: w5_15). "超越第二名1074亿元" is checked as market value >
   1074亿 (supported); the second place (国泰海通) is not in the data, so the part is unverifiable and the claim,
   whose market value is false, is contradicted.
7. **Approximation width** (1: w5_08). "ROE约30%" against 32.53: the checker's 约 band is half the last significant
   digit's step (25-35%, round 10 F7), the labels' 5% (28.5-31.5%).
