# Real published claims, sixth set, blind (`claims_v6.jsonl`)

**Kept outside the repository** (`/Volumes/Remi/finsight-blind/v6/`, a local git repository that is never pushed) until
it is scored once, after the next round of claim-checker fixes. Nobody fixing the checker should read this folder.

Sixth set of real published claims, collected on 2026-10-11 (ids `w6_01` … `w6_60`). It measures the claim checker on
wording nobody wrote for it. The "Labelling rules" section was written and committed (local commit `bf72914`, clarified
in `bafcf32`) before any candidate was collected; `collected_v6.jsonl` was committed (`3eb5601`) before any part was
written. The checker has not been run on this set; no scorer exists here; nothing was copied into a FinSight worktree.

## Files

| file | what |
|---|---|
| `SCOPE_FOR_COLLECTORS.md` | companies, metrics and dates given to the collectors (no data value) |
| `COLLECTOR_BRIEF.md` | the collectors' instructions and record format |
| `do_not_use_urls.txt` | the 237 URLs of the v1-v5 `collected_*.jsonl` files |
| `candidates_A.jsonl` … `candidates_D.jsonl` | 110 candidates from four collectors (A: 宁德时代/比亚迪/美的/中芯; B: banks, 中信证券, 中国平安; C: 茅台/五粮液/紫金/长江电力; D: dated market facts) |
| `select_v6.py` | fixed, value-blind selection, candidates → 60 claims |
| `selection_v6.json` | what the selection dropped and skipped |
| `collected_v6.jsonl` | the 60 claims as collected, before any label |
| `claim_parts_v6.py` | the hand-written parts of every claim (numbers as written; no data value) |
| `label_claims_v6.py` | data reader, tolerance / band / definition rules, primary comparison, verdicts |
| `claims_v6.jsonl` | collected fields + `category`, `parts`, `expected_verdict`, `expected_verdict_definition_blind`, `context_not_checked`, `primary_source_agreement`, `definition_difference` |
| `recheck_v6.json` | the labeller's own re-fetch of every URL (sentence found in page text or not) |

```bash
PY=/Volumes/Remi/FinSight-r2/.venv/bin/python
$PY select_v6.py          # candidates_*.jsonl -> collected_v6.jsonl (deterministic; re-run gives the same sha256)
$PY label_claims_v6.py    # collected_v6.jsonl + claim_parts_v6.py -> claims_v6.jsonl (deterministic)
```

## Protocol

**Who read what.** The labeller (this session) read, in this order: the FinSight-r2 data files (`data/structured_data.json`,
`data/snapshot/structured_data_ext.json`, `structured_data_fix.json`, `manifest.json`, `manifest_fix.json`,
`corrections/sources.json`, and listings of `data/snapshot/raw/`), lines 1-200 of
`evaluation/real_claims_web_v5/README.md` (name, files, protocol, selection, counts, labelling rules; not "Label counts",
"Data vs published reports", the first run or its error classes), and, through a script that kept only the `url` and
text fields, the v1-v5 `collected_*.jsonl` files (do-not-use list and text repeats). Never opened: `query_intelligence/`,
tests, claim-bench rows, any `real_claims_web*/` label, score or select script, any `claims_web_*.jsonl` or
`candidates_*.jsonl` of earlier sets, `evaluation/results/`, `/Volumes/Remi/finsight-review/`. The checker was not
imported or run; no file was written in any FinSight worktree.

**Collection blind to the data.** Because the labeller had seen the offline values (step 1 of the task), collection was
delegated to four fresh sub-agents that were given only `SCOPE_FOR_COLLECTORS.md`, `COLLECTOR_BRIEF.md` and the URL list,
told not to open anything under `/Volumes/Remi/FinSight*`, and told not to keep or drop a candidate by its number. They
were given no value. Selection is a fixed script that reads no data value.

**Verification (deviation from v5).** Tavily's plan quota was exhausted for every collector after at most one call,
and Brave returned "fetch failed". Collectors searched with the built-in web search and verified each sentence by
fetching the page (curl; pypdf for one PDF) and finding the exact published text in the stripped page text; this is in
every `verified_by`. `select_v6.py` was then changed (before any claim text was read; only `verified_by` strings and
first-entity names had been counted) to accept "sentence found in fetched / extracted text" as verified, and to map
first-entity spellings (貴州茅台 / Kweichow Moutai, 紫金礦業 / Zijin Mining, 五糧液 / Wuliangye, 沪深两市 → 上证指数) to one
group. The labeller then re-fetched all 60 URLs (`recheck_v6.json`): 58 sentences found in the page text, one page
(w6_56, fxbaogao) blocked the fetch, one PDF (w6_12) not re-checked (no pypdf in the venv).

**Quoting.** `published_text` is the clause as published; `claim_text` adds an implicit subject, period or metric in
square brackets (`subject_inserted`: 49 of 60). `period_basis` is `sentence` or `context` for every selected claim.

## Selection (110 candidates → 60), before any label

| rule | dropped |
|---|---|
| not a claim | 0 |
| not verified on the page | 0 |
| period not stated (`dateline` / `none`) | 7 (B01, B09, B11, B12, D20, D21, D22) |
| multi-session | 0 |
| URL in v1-v5 | 0 |
| text repeats a v1-v5 claim (normalised, containment ≥ 15 chars) | 3 (B05, D08, D26) |
| duplicate text | 0 |

The other 100 are ordered by sha256("v6:" + candidate_id). Variety quotas are filled first in that order (collector
tags only: definition 3, weak_direction 6, same_company_comparison 4, kou_fei 3, growth_multiple 3,
cross_company_comparison 3, short_name 4, traditional 6, en 9, social 12, broker 3, company 3, 2026-09-30 6,
2026-04-22 5), then the rest in hash order, with at most 5 per group (first-named entity, normalised) and 2 per URL.
Quotas that could not be met from the pool: definition (2 candidates tagged), growth_multiple (2).

## Counts

- **60 claims**: zh 48 (simplified 36, traditional 12), en 12.
- **Source types**: media 30; social 20 (东方财富财富号 12, 网易号 2, 搜狐号 1, 新浪财经头条 1, 鉅亨號 1, 有驾 1, Substack 1,
  PandaAI community 1); broker 7 (note summaries: 长江证券, 华金证券, 中国银河, 国泰海通 on 新浪财经研报; 开源证券 on 发现报告; 国信证券 in
  English on Longbridge; a 券商研究公众号 on 紫金 reposted by 新浪财经); company 3 (Midea IR page, Zijin HKEX results announcement, ICBC annual-report
  summary in 上海证券报).
- **Periods**: FY2025 49; 2026-09-30 6 (上证指数 ×4, 创业板指, 宁德时代 bar); 2026-04-22 5 (创业板指 ×2, 三大指数, 五粮液
  close, 五粮液 PE TTM).
- **Categories** (`category`, from the parts): annual figures 22, 扣非 figures 10, daily bar / index 10, same-company
  comparison 7, weak direction word 6, definition 2, cross-company comparison 2, valuation 1.
- **Collector feature tags**: direction 44, multi_part 31, weak_direction 13, kou_fei 12, same_company_comparison 9, bar 8,
  approx 7, cross_company_comparison 5, short_name 4 (宁王-type names were rare; 平安 for 平安银行, 沪指, 茅台, CATL),
  definition 2, growth_multiple 2, valuation 1.
- **Groups** (normalised first entity): 宁德时代 5, 比亚迪 5, 平安银行 5, 招商银行 5, 上证指数 5, 紫金矿业 5, 贵州茅台 5,
  五粮液 5, 中信证券 4, 长江电力 3, 美的集团 3, 中芯国际 3, 创业板指 3, 中国平安 2, 工商银行 1, 中国移动 1 (out of scope;
  its claim also names 茅台 and 宁德时代).
- Publication date not shown on the page: 3 (`published` null).

## Labelling rules (written before collection and before labelling)

Labels come from `label_claims_v6.py`, which reads only the data files of `/Volumes/Remi/FinSight-r2` (read-only):
`data/structured_data.json` (base), `data/snapshot/structured_data_fix.json` (corrections layer, overrides the base
for the symbols it carries), `data/snapshot/structured_data_ext.json` (extension companies, indices and ETFs), and
`data/snapshot/corrections/sources.json` (annual-report transcriptions; used only for the definition rule's
"same figure" exception and for the primary-source comparison, never as the served value). Standard library + json
only; nothing from `query_intelligence` is imported and the claim checker is never run.

A person (the labeller) writes each claim's parts: company, metric, the number as written, its unit, a qualifier, the
basis the sentence names, and for definition / direction / comparison parts what the sentence asserts. Every value a
part is compared with is computed by the script from the files. **Every part is labelled**, including definition,
direction and comparison parts; a part is `true`, `false` or `null` (not checkable), and `null` parts say why.

### Tolerances (v5's, unchanged)

- **Exact figure**: true when |claimed − actual| ≤ max(half a unit of the claim's last written digit, 1% of the actual
  value). A growth rate or change must also have the right sign.
- **约 / 左右 / about / approximately / around / roughly / 约合 / 近乎**: within 5% of the claimed number.
- **突破 / 站上 / 达到…以上** N: actual ≥ N. **超 / 超过 / 逾 / 以上 / more than / over / exceeding / above** N: actual > N.
  **近 / 将近 / 接近 / nearly / almost / close to** N: 0.9 N ≤ actual ≤ N. **不到 / 不足 / 低于 / less than / under** N:
  actual < N. **N 多 / N 余**: N < actual < N + one unit of the last written digit. For a signed quantity the bound
  applies to the magnitude and the sign must match.
- **Double-digit growth** (v5): "双位数 / 两位数增长 / double-digit growth": 10 ≤ growth < 100 (%), positive.

### Weak direction words (new in this set; numeric bands fixed here, before labelling)

A plain direction verb attached to a number ("增长12%", "fell 4.5%") is the number's sign: one value part, which needs
the right sign. A direction word without a number ("营收增长", "closed higher"), and every band word below (微 / 小幅 /
大幅 / 持平 / 腰斩 / 翻倍 …) even when a number is also given, is its own part (qualifier `direction`), labelled from the
served change; the number is then a separate exact / approximate part. Bands are inclusive at the stated edges.
(Clarified on 2026-10-11 while the collectors were still running, before any candidate was seen or labelled.)

Annual YoY change g (%) of an amount (revenue, net profit, 扣非净利润, EPS):

| words | band |
|---|---|
| 增长 / 上升 / 提升 / 增加 / rose / grew / increased / up / 正增长 / 稳步(稳健)增长 | g > 0 |
| 下降 / 下滑 / 减少 / 回落 / fell / declined / dropped / down / 负增长 | g < 0 |
| 微增 / 微升 / 略增 / 小幅增长 / 小增 / 轻微增长 / edged up / inched up / slightly higher / ticked up | 0 < g ≤ 5 |
| 微降 / 略降 / 小幅下降(下滑) / 微跌 / 小降 / edged down / slipped / dipped / slightly lower | −5 ≤ g < 0 |
| 基本持平 / 持平 / 大致持平 / 基本稳定 / flat / roughly flat / little changed / broadly unchanged | \|g\| ≤ 1 |
| 大增 / 大幅增长 / 高增长 / 猛增 / 暴增 / surged / soared / jumped / leapt / strong growth | g ≥ 20 |
| 大降 / 大幅下滑(下降) / 暴跌 / 骤降 / plunged / slumped / tumbled / sank | g ≤ −20 |
| 腰斩 / halved | −60 ≤ g ≤ −40 |
| 翻倍 / 翻番 / doubled | ratio current/prior ≥ 1.9 (growth ≥ 90%) |
| 增长 N 倍 / N-fold | ratio ≈ N+1 for "增长N倍", ≈ N for "是…的N倍 / N-fold"; exact or 约 rules on the ratio |

Daily % change c of a close (stocks, ETFs, indices):

| words | band |
|---|---|
| 收涨 / 上涨 / 收高 / 走高 / 红盘 / rose / gained / closed higher / up | c > 0 |
| 收跌 / 下跌 / 收低 / 走低 / 绿盘 / fell / closed lower / down | c < 0 |
| 微涨 / 小幅上涨 / 小幅收高 / 略涨 / edged up / inched up / ticked up / slightly higher | 0 < c ≤ 1 |
| 微跌 / 小幅下跌 / 小幅收低 / 略跌 / edged down / inched down / slipped / slightly lower | −1 ≤ c < 0 |
| 平盘 / 基本持平 / 横盘 / flat / little changed / unchanged | \|c\| ≤ 0.2 |
| 大涨 / 飙升 / 强势上涨 / surged / soared / jumped / rallied strongly | c ≥ 3 (stocks, ETFs), c ≥ 1.5 (indices) |
| 大跌 / 重挫 / 暴跌 / plunged / tumbled / slumped | c ≤ −3 (stocks, ETFs), c ≤ −1.5 (indices) |
| 涨停 / 跌停 | c ≥ 9.9 (≥ 19.9 for 300xxx / 688xxx) / mirror |

Direction words about a ratio (ROE, margin) change in percentage points need the prior-year ratio, which is not served:
null (v5's "changes in percentage points" rule). A word whose sense is not in these tables is null, and the labeller
says so in the part's note; no band is invented after labelling started.

### Relations and comparisons

- (v5) "A 高于 / 超过 / 强于 B" from the two served values; "A 是 B 的 N 倍" is A / B; "比 B 多 X" is A − B; "差 / 相差 N
  倍" is larger / smaller − 1; sums, averages and shares of several companies are computed from served values and are null
  if any named company is not in the data.
- **Same-company comparisons (stated explicitly here)**: "营收增速快于(高于)净利润增速", "净利润增速跑赢营收", "扣非净利润
  低于归母净利润", "净利润降幅大于营收降幅" are compared from the company's served fields (revenue_yoy vs netprofit_yoy;
  profit_dedt vs net_profit). "降幅大于" compares magnitudes of two negative changes; if either change is not negative,
  the comparison part is false (the sentence asserts two declines). "增速快于" compares signed growth.
- **Growth multiples between metrics** ("净利润增速是营收增速的N倍") are ratio parts on the two served growth rates.

### 扣非 vs 归母 (v5)

扣非净利润 is the served `profit_dedt`; relations between 扣非 and 归母 use the two served fields. 扣非 YoY is not
served: null (the primary-source comparison still records the report's figure where sources.json has it).

### Definitions (v2 / v4 / v5, plus definition parts)

- A generic word (净利润, net profit, 营收, 收入, revenue, sales, ROE, 净资产收益率, 净利率, 每股收益, EPS, 毛利率) is
  compared with the served value whatever its basis. 营收 is treated as generic (it abbreviates 营业收入 but is used for
  both lines in practice); 营业收入 and 营业总收入 written in full name a basis.
- A sentence that names a basis (归母 / 归属于…股东, 加权平均, 基本 / 稀释, 营业总收入 / 营业收入, 总净利润 / 含少数股东)
  different from the served one makes that part null in `expected_verdict`, unless sources.json shows the two bases give
  the same figure for that company; `expected_verdict_definition_blind` compares the same part with the served value
  anyway.
- **Definition parts (new in this set)**: a definition stated in the sentence ("ROE即净利润除以净资产", "毛利率=(营收−成本)/
  营收", "扣非即扣除非经常性损益后的净利润") is its own part. It is `true` when it is the standard definition of the metric
  (textbook accounting definitions listed below) **and** consistent with the served basis; `null` when it is standard but
  names a basis other than the served one (e.g. a weighted-average ROE definition against a period-end served ROE); and
  `false` when it is not a standard definition of that metric (e.g. 毛利率 defined with net profit). Standard
  definitions: ROE = (归母)净利润 / (平均 or 期末)(归母)净资产; 毛利率 = (营业收入 − 营业成本) / 营业收入; 净利率 = 净利润 /
  营业收入; EPS = 归母净利润 / 普通股加权股数; 扣非净利润 = 归母净利润 − 非经常性损益; PE = 股价 / 每股收益 (= 市值 / 净利润);
  PB = 股价 / 每股净资产 (= 市值 / 净资产); 破净 = PB < 1. The number stated with the definition is a separate value part
  under the rules above.

### Valuation and bars (v5)

- A market value, PE or PB is compared only when the served valuation date equals the claim's date (2026-09-30 for the
  ten extension companies, 2026-04-22 for 贵州茅台, 五粮液, 中国平安). Market value in 亿 / 万亿 / billion / trillion is
  compared with `total_mv`. 破净 is PB < 1 on the served date.
- A bar is compared only when the served bars contain the claim's date. Served bars: the extension `market_api`
  histories (2025-07-01 … 2026-09-30) for the ten companies, 上证指数, 创业板指 and the ETFs 510500 / 518880 / 588000; the
  corrections layer for 沪深300, 中国平安, 创业板ETF 159915, 证券ETF 512880 (2026-04-22 and 04-21); the base for 贵州茅台,
  五粮液 and 沪深300ETF 510300 (2026-04-22). Change in 元 / 点 = close − previous served close; where no previous close
  is served (茅台, 五粮液) it is derived as close / (1 + pct/100). A null served % change is derived from the previous
  served close (510300).
- Units: 亿 = 1e8, 万亿 = 1e12, 千亿 = 1e11, billion = 1e9, trillion = 1e12, million = 1e6; 万手 / 手 volumes are compared in
  shares (1 手 = 100 股). 成交额 of an index is not served (null). Intraday highs and lows ("盘中最高 / 最低 / 触及") are
  null (v5).

### Not checkable (null; v5's practice)

Changes in percentage points, per-day figures (日赚), rankings and superlatives (榜首, 第N名, 历史新高, 首次, 最高) — kept in
`context_not_checked` —, turnover rates, entities not in the data (any company, index or ETF outside the scope list),
metrics not served (营运利润, 净资产 / 总资产 amounts, 毛利润 amounts, segment figures, dividends, buybacks, payout ratios,
cash flow, NIM, NPL ratios, 新业务价值, 销量), 两市 / market-wide turnover, H-share / ADR figures, US-dollar or HK-dollar
amounts (a growth rate stated alongside a US-dollar amount is compared with the served rate, v4/v5), prior-year
absolute values except as derived (prior = value / (1 + YoY)), quarterly and half-year figures, forecasts and targets.

### Verdict (v2 / v4 / v5)

`supported` if every part holds; `partially_supported` if at least one part holds and another does not or cannot be
checked; `contradicted` if no part holds and at least one is false; `unverifiable` if every part is null. Consequently a
claim with any false part is never `supported`. The same aggregation over the definition-blind part values gives
`expected_verdict_definition_blind`.

### Primary-source comparison

For every part whose metric is transcribed in sources.json (FY2025 annual figures), the script records the report's
figure on the basis the sentence names (or the generic basis), whether the claim holds against it with the same
tolerance (`primary_holds`), and whether the served value and the report figure agree within 1% (`served_vs_report`).
Disagreements are listed in this README after labelling.

## Decisions made while labelling (after `collected_v6.jsonl`; disclosed, not part of the pre-written rules)

- **Word senses** (mapped to a sense already in the band tables; no new band): 稳增 → 稳步增长 (g > 0, w6_05); 狂飙 →
  飙升 / surged (g ≥ 20, w6_05); 微红收官 → 微涨 (0 < c ≤ 1, w6_07); 略低于 (a year's profit below the previous year's) →
  略降 (−5 ≤ g < 0, w6_13); "was flat" → flat (|g| ≤ 1, w6_44); 大幅增长 and 小幅增长 and 微降 are in the tables.
- **Ratio wordings** use the v5 relation and qualifier rules: 不足一半 / 不足三成 → ratio < 0.5 / < 0.3 (w6_30, w6_52);
  多一倍多 → "N 多" on A/B − 1 with N = 1 (1 < A/B − 1 < 2, w6_52); 接近翻倍 → "近 N" on current / prior with N = 2
  (1.8 ≤ ratio ≤ 2, w6_53); 站上 345亿 → ≥ (w6_58); 背离 between revenue and profit changes → opposite signs (w6_35).
- **Intensity adverbs on comparisons** (显著高于, 大幅甩开, 略高的) are labelled as the plain comparison: the rules have no
  band for gaps between two growth rates (w6_21, w6_27, w6_58).
- **Readings**: w6_29 "营收同比2024年下滑了10个百分点" is read as revenue YoY −10% (an amount has no percentage-point
  change); w6_49 "集团实现净利润 … 其中归属于母公司股东净利润 …" names the total (group) basis for the first figure;
  w6_20 "total revenue" names 营业总收入; w6_22 ("for every 1,000 yuan of shareholder equity … earns 134 yuan") is a
  definition part (standard ROE) plus a value part 13.4%; w6_02's description of 扣非 is a definition part (standard).
- **Definition parts** are only 2 (w6_02, w6_22); no candidate of the "ROE即…为x%" form was found by any collector.

## Label counts

Parts: 183 (value 118, null 40, relation 12, direction 11, definition 2). Expected part values: true 117, false 14,
null 52 (40 not checkable + 12 basis differences). Definition-blind: true 125, false 18, null 40.

| `expected_verdict` | n | definition-blind | n |
|---|---|---|---|
| supported | 19 | supported | 23 |
| partially_supported | 37 | partially_supported | 33 |
| contradicted | 3 | contradicted | 4 |
| unverifiable | 1 | unverifiable | 0 |

- 12 claims have at least one false part (w6_04, 10, 21, 22, 27, 31, 38, 41, 45, 47, 52, 53); none is `supported`.
  Contradicted: w6_04 (茅台 net margin 50.53%), w6_10 (茅台 净利率 52%), w6_47 (五粮液 PE TTM 13.74 on 2026-04-22).
  Unverifiable: w6_56 (中信证券 加权平均ROE 10.59%; the served ROE is period-end).
- Expected ≠ definition-blind for 5 claims: w6_12, w6_40, w6_49, w6_58 (partially → supported) and w6_56
  (unverifiable → contradicted).
- By source type: media 8 S / 21 P / 1 C; social 9 S / 9 P / 2 C; broker 1 S / 5 P / 1 U; company 1 S / 2 P.
- By period: FY2025 13 S / 33 P / 2 C / 1 U; 2026-09-30 3 S / 3 P; 2026-04-22 3 S / 1 P / 1 C.
- By language: zh 14 S / 31 P / 2 C / 1 U; en 5 S / 6 P / 1 C.
- By category: annual figures 5 S / 14 P / 2 C / 1 U; 扣非 figures 10 P; daily bar / index 6 S / 4 P; same-company
  comparison 2 S / 5 P; weak direction word 6 S; definition 2 P; cross-company comparison 2 P; valuation 1 C.

## Data vs published reports

**Served value vs the annual report on the same basis**: every FY2025 part agrees (within 1%, or 0.01 for a printed
percentage) with sources.json. No bar or valuation has a primary transcription.

**Definition and basis differences met by the claims** (served basis ≠ the basis the published figure uses):

| claims | served | published figure | report |
|---|---|---|---|
| w6_12, w6_24, w6_27, w6_40, w6_50, w6_51, w6_53, w6_58 (and generic w6_21) | net-profit YoY on total net profit (紫金 62.01, 宁德 42.18, 中信 37.26, 比亚迪 −18.82, 招行 1.05, 美的 14.87, 长江 6.13) | 归母 YoY (61.55, 42.28, 38.58, −18.97, 1.2, 14, 6.17) | report 归母 YoY agrees with every published figure; w6_27, w6_50, w6_51 (and w6_21) fail against the served total-basis YoY |
| w6_12, w6_34 | prior 归母 profit derived as 归母 / (1 + total YoY) (紫金 319.58亿, 比亚迪 401.8亿) | report 归母 prior (320.51亿, 402.5亿) | null by basis; within tolerance either way |
| w6_22, w6_41, w6_45, w6_56 | ROE period-end (招行 11.80, 平安银行 7.73, 中信 9.40) | weighted ROE (13.4 / 13.44, 9.15, 10.59) | report weighted ROE matches the published figures; generic wordings are false against served |
| w6_04, w6_38, w6_52 (w6_10) | net margin = 归母净利润 / 营业收入 (茅台 48.76, 紫金 14.83, 宁德 17.04) | total net profit / 营业收入 (50.53, 18.28, 18.12) | published figures match total-basis margins; w6_10's 52% matches neither |
| w6_27 | 中信证券 revenue YoY 28.8 vs restated 2024 | 17.35 vs originally reported 2024 | sources.json notes both; false against served |
| w6_49 | 工商银行 net profit 归母 3685.62亿 | group net profit 3707.66亿 | report total matches; null by basis |
| w6_47 | 五粮液 PE TTM 43.63 on 2026-04-22 = close × shares / FY2025 归母净利润 (published 2026-04-30) | 13.74 (a TTM window before the FY2025 report: implied TTM profit ≈ 284亿) | not in sources.json; false against served, a timing/definition difference |
| w6_31 | 茅台 归母 823.20亿, 扣非 822.93亿 | 82.32亿, 82.29亿 | a 10× unit error in the article; false against served and report |

## Hashes (sha256)

All files of this folder except this README (a file cannot carry its own hash; the README's hash is recorded in the
local git commit message that adds this section). Data files read by the labeller (FinSight-r2, read-only):

```
762e977aad2dfebbf78547c9e7ecc31cd8bd458c3f7549963932bac069431947  data/structured_data.json
538d2bbe49646dc28c23987a564658b9412c7aaa5c9147114e1e4742e0da0e5a  data/snapshot/structured_data_ext.json
a76f2804c0c8072fdb6e414d584fb0f578c4f623ecaa2c030a4546f0853082fd  data/snapshot/structured_data_fix.json
253e7a7f3ee33954145d14c72dd50cea9819110ce79cb6c087a2ce9a7a9b5be4  data/snapshot/corrections/sources.json
```

This folder:

```
b4b8bfe069770bf7009883006ad32d93adb4ceec43598332f637c18d8f951560  COLLECTOR_BRIEF.md
f633df4e1e82587d0d7a60b8f396c0b27cce412ec7d6d143e63a1973067f96ea  SCOPE_FOR_COLLECTORS.md
ab9f12da8ea33c1a463bc039bcd2563df2949916adef27522fb3f4810c4cfc01  do_not_use_urls.txt
f8e502b3c3c0a31208b16872a358d789c657b4b87c5c5ff08b23d11f2fa9854a  candidates_A.jsonl
7617d46a201b24e28239e83c936c6d4091473af00adb5b0ef84c495ac3897c86  candidates_B.jsonl
618e29ff276fa38904db574f6ecf3bfb2dc24179bcd2ee1a5154c72779df8cb1  candidates_C.jsonl
761a12faf9a1179d1c0e065d4ad5d9fb3079a65a70f85a9257e169e2a01c1198  candidates_D.jsonl
1e4377673a4429cbebb6b0f1ea6fb23a730cabb617538c43e7769e4d0daaab13  select_v6.py
72458cba7e7c4c8ba1faa94304cd73dd668e9e44d6b4238cc6b855b54ae96ade  selection_v6.json
f2275dd2462507f8dd81a5f3cfc8670df9b466a9ecbc80136a63be985ab99f64  collected_v6.jsonl
c1c576e4a3aed158a6640c129fde153a38755afabaec7c4165315e15731afca9  claim_parts_v6.py
7a038e42ff8f280d1147561044f8d1539baf0c2e698fc832ad8eea23cbd377dd  label_claims_v6.py
7bf23130635e800f2ff7ae8476e22adb182ac6d225ea74227104f0b79b63dc75  claims_v6.jsonl
5f89b409e99c2ec6dd754a3ac3c76d3cd12a04275504a1a5f63d9de09b224899  recheck_v6.json
32dae3052f331ee34d628ef535709b301259a45df7c7522c4d35dcf49873f00b  .gitignore
```
