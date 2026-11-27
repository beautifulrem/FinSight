# Claim Check

Languages: English | [中文](zh/claim-check.md)

`POST /agent/claim-check` verifies numeric and relational financial claims pasted from broker notes, news, or social media (e.g., `"贵州茅台ROE超过30%，市盈率不是15倍"`, `"Moutai's ROE is above 30%"`) against market, fundamental, and macro tools—**without invoking an LLM**.

Implementation: `query_intelligence/agent/claim_check.py`, `query_intelligence/agent/claim_definitions.py`, `query_intelligence/agent/hearsay.py`, and `frontend/src/components/ClaimCheck.tsx`.

```bash
curl -s -X POST http://127.0.0.1:8000/agent/claim-check \
  -H 'Content-Type: application/json' \
  -d '{"claim": "贵州茅台ROE超过30%，市盈率不是15倍"}'
```

## Response Model

Each parsed check returns:
- `target`, `metric`, `claimed`, `claimed_high` (for ranges), `claimed_unit`, `comparator`, `negated`, `direction` (`up|down`)
- For relations/ratios: `reference`, `reference_value`, `reference_metric`, `reference_evidence_id`, `ratio`
- Outcome: `actual`, `status` (`supported|contradicted|unverifiable`), `reason` (when unverifiable), `evidence_id`, `source`, `as_of`, `as_of_basis`, `note`, and optional `degraded[]`
- Report-level fields: `verdict` (`supported|contradicted|partially_supported|unverifiable`), `coverage` (`full|partial|none`), `unchecked[]` (clauses with no verifiable number/comparison), `excluded[]` (group exclusions), `limitations[]`, `degraded[]`, `disclaimer`

Nothing is dropped silently: every clause produces either a check or an explicit `unchecked` item (`reason: no_claim`).

## Parsing & Comparison Rules

### Numbers, Text Normalization & Targets
- Normalizes full-width ASCII digits/letters, Traditional Chinese, and Hong Kong/Taiwan financial terms (`純利` → `净利润`, `本益比` → `市盈率`, `市賬率`/`股價淨值比` → `市净率`, `股東應佔溢利` → `归母净利润`, `按年` → `同比`, etc. in `query_intelligence/traditional.py`).
- Parses Arabic numbers, Chinese numerals (`二十四点六倍`, `三成`, `百分之三十`, `一千七百亿`, uppercase financial numerals), bare `万亿` after threshold verbs (`突破万亿`), ranges (`20到30倍`, `between 20 and 30`), and English fractions (`half a percent`, `twice`, `a quarter of`).
- Ignores dates, lookback windows (`近20日`), indicator parameters (`RSI(14)`), stock codes, and index names as claimed values.
- Binds each claim to the nearest preceding target, supports ordered pairs (`分别` / `respectively`), resolves short/colloquial names, and handles group exclusions (`除平安银行外…均同比增长`).

### Metrics, Definitions & Macro
- Matches the nearest unit-compatible metric (`倍`/`x` → P/E or P/B; `%` → ROE, margins, daily change, or YoY growth; `元` → price; `亿`/`万`/`billion` → revenue/profit/turnover/market cap; `点` → index level). Incompatible units yield `unit_mismatch`; unitless amounts yield `no_unit`.
- Distinguishes `扣非净利润` vs `归母净利润`, `流通市值` vs total market cap, `营业收入` vs `营业总收入`, weighted vs period-end ROE, basic vs diluted EPS, and TTM vs dynamic P/E—flagging explicit mismatches as `definition_mismatch`.
- Checks glossary formula definitions inline (`kind: "definition"`, `evidence_id: glossary_<term>`), including P/B < 1 ("破净") meaning price below book value per share.
- Checks macro indicators (`CPI`, `PPI`, `PMI`, `M2`, 10Y yield, 1Y/5Y LPR, `GDP`) against `get_macro_indicators` (`as_of_basis: indicator_date`).

### Comparators & Negation

| `comparator` | Examples (zh / en) | Supported Condition |
|---|---|---|
| `eq` | `为`, `只有`, `is` | Within half a unit of the last written digit, or 2% |
| `approx` | `约`, `左右`, `about`, `around` | Within 5%, or half the step of the last significant digit |
| `approx` (from below) | `接近`, `将近`, `nearly`, `almost` | `0.9N ≤ actual ≤ N` (+ rounding half-step) |
| `gt` / `ge` | `超过`, `高于`, `逾`, `站上`, `至少`, `以上` / `above`, `at least` | `actual > claimed` / `actual ≥ claimed` (with upper-step bounds for `30多倍`, `出头`) |
| `lt` / `le` | `低于`, `不到`, `不足`, `跌破`, `不超过`, `以下` / `below`, `at most` | `actual < claimed` / `actual ≤ claimed` |
| `range` | `X到Y之间`, `X-Y倍`, `between X and Y` | `X ≤ actual ≤ Y` |
| `ne` | Negated `eq` (`不是`, `并非`, `not`) | Outside `eq` tolerance |

Negation (`不是`, `没有`, `未`, `不`, `not`, `never`) applies from the previous number/clause start and flips the comparator (`eq↔ne`, `gt↔le`, `ge↔lt`).

### Directional Moves & Growth Bands

- **Daily price moves**: Bounds apply to move magnitude *in the stated direction* (`跌超1%` requires `change ≤ -1%`; `跌不到1%` requires `-1% < change ≤ 0%`; opposite direction contradicts). Qualitative daily-move thresholds: big (`大跌`/`暴跌`/`plunged`, `大涨`/`surged`) = `|move| ≥ 3%`; small (`微跌`/`小跌`/`dipped`, `微涨`/`edged up`) = `0 < |move| < 1%`.
- **Weak direction words for YoY growth** (applied when no number is stated; `g` = YoY growth % from fundamentals; negated forms flip the band):

| Band | Words (zh; en) | Holds When |
|---|---|---|
| `flat` | 基本持平, 大致持平, 持平, 变化不大, 几乎没有变化; flat, roughly/broadly/essentially/largely flat, little changed, unchanged | `−2 ≤ g ≤ 2` |
| `slight` | 微增, 微升, 微降, 略增, 略降, 略有/稍有 增长/下降; edged/inched/ticked up/down, rose/fell slightly, marginally up/down | Stated direction and `0 < |g| ≤ 5` |
| `modest` | 小幅 增长/增加/上升/下降/下滑/减少; rose/grew/fell/declined modestly, modestly higher/lower | Stated direction and `0 < |g| ≤ 10` |
| `sharp` | 大增, 暴增, 猛增, 激增, 大幅 增长/下降, 骤降, 锐减; surged, soared, plunged, slumped, rose/fell sharply | Stated direction and `|g| ≥ 20` |

### Relations, Multi-Company Sums & Superlatives
Supports two-company comparisons (`比…高/低`, `higher/lower than`), multiples (`是…的1.5倍`, `X倍半`, `twice`), stated differences (`高出N个百分点`), same-company growth-rate comparisons (`营收增速快于净利润增速`, `降幅较营收收窄`), multi-company sums (`茅台和五粮液成交额加起来不到50亿`, `A与B净利润合计不足C`), 52-week high/low references, industry-average comparisons, and superlatives among named companies.

### As-of Date, Missing Corrections Layer & Unverifiable Reasons
- `as_of_basis` distinguishes `trade_date` (prices), `valuation_date` (P/E, P/B when available), `report_date` (financial statements), and `indicator_date` (macro).
- If `QI_OFFLINE_SNAPSHOT_FIX` is enabled but `data/snapshot/structured_data_fix.json` is missing, checks evaluated against uncorrected v1 seed rows carry `degraded: ["data_layer_missing:corrections"]`, `coverage: "partial"`, and an explicit warning in `limitations` and `disclaimer`.
- Unverifiable `reason` codes: `no_target`, `no_metric`, `no_data`, `growth_unavailable`, `unit_mismatch`, `no_unit`, `forecast`, `period_mismatch`, `multi_day`, `no_reference`, `definition_mismatch`, `currency`.

### Inline Hearsay Check in Chat
When a chat message carries hearsay/verification cues (`听说…是真的吗`, `网传…`, `I read that … True?`, `核实`), `agent/hearsay.py` automatically runs `check_claim`, leads the answer with the deterministic verdict, and attaches `fact_check` to the response payload.

## Benchmark & Verification

Benchmark suites live in `evaluation/claim_bench/` (`claims_v1.jsonl` dev, `claims_v1_holdout.jsonl` held-out) and blind real-world web claim sets in `evaluation/real_claims_web*/`.

```bash
python -m evaluation.claim_bench.run --set dev
python -m evaluation.claim_bench.run --set holdout
python -m pytest -q tests/test_claim_bench.py tests/test_agent_round17_claims.py tests/test_agent_round18_claims.py
```

Latest committed offline scores: dev (311 claims) **1.000**, held-out (47 claims / 54 checks) **1.000**; blind real-world web claims (first-run untouched): v2 **0.500**, v3 **0.540**, v4 **0.740**, v5 **0.680**, v6 **0.650**. See [docs/agent-eval.md](agent-eval.md) and [docs/evaluation-history.md](evaluation-history.md).

## Known Limits

- Rule-based extraction misses phrasings outside its lexical classes (e.g., English exclusions such as `"all but X"`, multi-sentence exclusions, or arbitrary non-formula definitions).
- Offline snapshots lack prior-year ratio series (e.g., YoY gross-margin change), multi-day cumulative windows, and industry valuation tiles for extension-only industries; such claims return `unverifiable` rather than guessing.
- When a foreign currency or foreign listing (HK/US/GDR/ADR) is claimed, the check returns `unverifiable` (`currency` / `no_data`).
