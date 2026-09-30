# Round-6 independent held-out slice (claims + chat)

Written 2026-10-01 against FinSight repo commit `bc42017` (`git -C <local path> rev-parse --short HEAD`),
**before** any fix for the round-6 findings F1, F2, F4–F10. The point is to have something to run once before the fixes and once after, so the fixes can be measured honestly.

## Protocol

- **Bug classes** were taken only from `finsight-review/round6.md` (F1, F2, F4–F10). Every claim and question is a new phrasing. None of the reviewer's probes are copied, and the probe scripts in `finsight-review/round6/` were not opened.
- **Not read:** anything under `query_intelligence/`, `evaluation/agent_eval/tasks/`, `evaluation/claim_bench/`, `evaluation/heldout_r4/`, `evaluation/heldout_r5/`, `evaluation/results/`, or `finsight-review/heldout_r4|r5/`.
- **Read:** `evaluation/agent_eval/metrics.py`, for the `expect` keys, and `runner.py`, for the task shape and `build_offline_service`.
- **Two scorer functions were called, not read.** `verifier.claim_numbers` and `_is_supported` were run on a few strings. This only settled how precise the `required_facts` values need to be:
  - values are sign-insensitive;
  - 亿 and billion are scaled;
  - ratios need 2 dp, e.g. 2.83 is not matched by 2.8.
- **Every number comes from the real offline tools** (`build_offline_service()` + `build_registry_for_service`: `get_price_history`, `get_fundamentals` including its `industry_*` evidence, and `search_news`). Price data is as of 2026-04-22; fundamentals are FY2025; industry rows are dated 2026-04-21/22.
- **Nothing was run through the agent or the claim checker to pick or tune items.** No first run has been made.
- **To run once before the fixes**, copy this directory to `<repo>/evaluation/heldout_r6/`, then:
  - `python -m evaluation.agent_eval.runner --mode workflow --no-replay --tasks evaluation/heldout_r6/chat_r6_heldout.jsonl` (and `--mode auto`);
  - run the claim file through the claim-bench runner's `--claims` option.
  - Publish that first-run result before touching the code.

## Files

| File | Count | sha256 |
|---|---|---|
| `claims_r6_heldout.jsonl` | 67 claims (58 zh, 9 en) | `76e4de167ff3de83f7bc74b7329d39e17eec68326bddae0b1670321241a858de` |
| `chat_r6_heldout.jsonl` | 38 tasks, 61 turns (27 single, 2 two-turn, 6 three-turn, 3 four-turn; 32 zh, 6 en) | `03466bd4ccfa96852d9f29aa3ce5b0a6310e8b1409e48e71270610d12acc0412` |
| `verify_heldout_r6.py` | re-derives every label | `d80bcfec16293c0a1c1dbd01ccd2db55a9b7590a378485a48cc6e8ed396e0a1b` |

### Claims by category and verdict

| Category | Count | Bug class |
|---|---|---|
| `stated_industry_average` | 19 | F1 |
| `company_difference` | 20 | F2 |
| `approx_numeral` | 20 | F7 |
| `control` | 8 | none |

- **Verdicts:** 47 supported, 9 partially_supported, 9 contradicted, 2 unverifiable.
- **Industry-average word orders covered:**
  - the average stated first;
  - the average in parentheses;
  - "比…行业平均的N倍低/高" and "低于…平均的N倍", the F1 form;
  - "高于行业平均水平的N倍";
  - "不到行业平均（N倍）的一半";
  - the English "the industry averages Nx, … above/below that".
- **Traps:**
  - A comparison that is true against the fabricated average but false against the real one (r6c06, r6c11, r6c14, r6c16).
  - A difference with the right magnitude and the wrong direction (r6c22).

### Chat tasks by category

| Category | Tasks | Bug class |
|---|---|---|
| `difference_followup` | 8 | F4 |
| `two_target_compare` | 1 | F4 |
| `fair_value` | 7 | F5 |
| `out_of_coverage` | 7 | F6 |
| `derived_metric` | 6 | F7/F8 |
| `comparison_winner` | 4 | F10 |
| `news_corroborated` | 3 | F9 |
| `news_control` | 2 | none |

- Every turn carries the four TRADING forbidden patterns exactly as specified.
- The three `news_corroborated` tasks add figure-specific patterns. These fail an answer that attributes 1688.38亿 or 823.2亿 to "一篇文档" or labels them "未经其他来源证实". Those two figures appear in `aknews_600519.SH_2` and `aknews_600519.SH_4`, and they equal `fundamental_600519.SH` (revenue 168838000000, net_profit 82320000000). The verify script re-checks this.
- The same articles also carry figures that structured data does **not** corroborate: YoY −1.21%/−4.53%, EPS 65.66 and dividend 27.993. Attributing those stays legal, because the patterns only match the two corroborated numbers.

## Labelling rules (used by the verify script)

**Statuses:**
- `approx`: supported if |actual − stated| ≤ 5% of stated; otherwise contradicted. Every item was chosen to be ≤ 3.3% or ≥ 15% off; there are no borderline items.
- `range` bounds:
  - N多 / N余: [N, N + one unit of the last named digit), e.g. 四百四十多亿 = [440, 450)亿.
  - N出头: [N, 1.1N].
  - 近N / 将近N / 接近N: [0.9N, N].
  - N倍多: [N, N+1).
  - 两成多: [20%, 30%).
- `lt`/`gt`: a strict comparison against the **real** industry average from `industry_*`, never the stated one.

**Verdicts:**
- All checks supported → `supported`.
- All contradicted → `contradicted`.
- All unverifiable → `unverifiable`.
- Any other mix → `partially_supported`.

**Checks** are listed in the order the claim states them.

## Ambiguities (decided, flag if you disagree)

1. **"比白酒行业平均的N倍低" / "低于…平均的N倍"** (r6c08, r6c09, r6c10). I read N as the stated average, because N × average is an implausible PE (35 × 27.3). Under the other reading, r6c08 is still true but for the wrong reason. r6c11 ("高于行业平均水平的4.5倍") is contradicted under either reading.
2. **"市盈率相差3.7倍" and "PB差…倍"** (r6c26). I read these as a difference in PE units (24.6 − 20.9), not a ratio.
3. **Degree adverbs** (略低, 明显低于, 偏高) are not checked beyond direction. I avoided "远低于/低一大截" where the gap is small.
4. **r6c12** names no industry ("行业平均PE…平安"). The industry is inferred as 保险 from 中国平安.
5. **The r6t04 turn-3 query "低多少"** expects 15.9 (the difference), not the ratio 2.83.
6. **Required facts for comparisons** (F10) only check that both values are stated and cited. `metrics.py` has no required-text key, so "which is higher" cannot be scored directly.
7. **Out-of-coverage names** are HK/US listings (平安好医生 1833.HK, 腾讯音乐 TME/1698.HK, 美团 3690.HK, 京东健康 6618.HK, 阿里健康 0241.HK). That is world knowledge, not a tool fact. The verify script only checks that none of them is an offline-covered name. The expected limitation code is `out_of_coverage`.
8. **The English F9 patterns** (r6t36) guess the product's English attribution wording ("according to a document", "unverified/not confirmed").
9. **Derived chat facts cite one operand's evidence id**, e.g. the gap in r6t02 cites `fundamental_000858.SZ`. An answer that cites both operands passes.

## Verify

```
<repo>/.venv/bin/python <repo>/evaluation/heldout_r6/verify_heldout_r6.py
# or from elsewhere
FINSIGHT_REPO=/path/to/FinSight <repo>/.venv/bin/python verify_heldout_r6.py
```

**What the script re-runs:** the offline tools. It resolves the repo from `$FINSIGHT_REPO`, else from the parent chain of the script's own location.

**What it checks:**
- the jsonl sha256 values;
- the schema: claim keys, allowed metrics and comparators, and `expect` keys drawn from `metrics.py`;
- that each claim's check list and each status are re-derived from tool values, and that the verdict follows from them;
- that each chat `required_facts` value (tolerance max(0.006, 0.5%)) and evidence id matches the tools;
- that the TRADING patterns are exact;
- that the F9 patterns appear only on tasks whose figures the tools corroborate.

**Result at `bc42017`:** "OK: every fact and label re-derived from the offline tools". A mutation test (one verdict flipped, one fact changed by 1) made it exit 1 and name both items.
