# FinSight round-5 independent held-out slice

Written 2026-09-30 against `<local path>` at commit **`fbce040`** (`git rev-parse --short HEAD`). Nothing committed to the repo.

## Protocol

- **Not read:** anything under `query_intelligence/` (only called as a black box: the offline tool registry, plus one call to `verifier.claim_numbers` / `_is_supported` to confirm "16.7亿" matches 1.67e9 in scoring), `evaluation/agent_eval/tasks/`, `evaluation/claim_bench/`, `evaluation/heldout_r4/`, `evaluation/results/`. The round-4 slice at `<local path>` was only listed, not opened. `build_tasks.py` / `build_holdout.py` were not read, apart from a grep showing language codes are `zh` / `en`.
- **Read:** `round4.md`, for the bug classes D2–D8 only. Every phrasing here is new. None of the reviewer's probes is reused, and near-copies were reworded (e.g. "不到五粮液的1.5倍" became "不到…1.6倍", "平安PE多少" became "平安最近一天收盘在多少").
- **Also read:** `evaluation/agent_eval/metrics.py`, for the expect keys; `evaluation/agent_eval/runner.py` (loader lines); and `data/structured_data.json`, `data/entity_master.csv`, `data/alias_table.csv` for the offline universe.
- **Facts:** every number comes from `get_price_history` / `get_fundamentals` on the offline service (prices as_of 2026-04-22, fundamentals FY2025, industry rows 白酒 2026-04-21 / 保险 2026-04-22). The offline universe has 7 priced symbols (600519, 000858, 601318, 510300, 159915, 000300, 512880) and 3 with fundamentals. 平安银行 000001.SZ is in `entity_master` but has no data.
- **Run once** before the D2–D8 fixes and once after. Do not tune against it.

## Files

| file | rows | sha256 |
|---|---|---|
| `claims_r5_heldout.jsonl` | 56 claims, 85 checks | `fd58c903acd977c2e75c51d70fefd93326594fa929ef9d398a0f0e919585c4d1` |
| `chat_r5_heldout.jsonl` | 38 tasks, 41 turns, 22 required facts | `ab21d777a04f447832793013142879be06c7010b3ac360bc74da98ad0f95759c` |
| `build_heldout_r5.py` | generator (facts + labels inline) | `c540e9b4af784c66f5bee0bb191b94c3a7f3c32a40d4abe34524b45851ea453f` |
| `verify_heldout_r5.py` | re-runs tools, re-derives every label | `5c681ffb376c4bb5c20bda0570b42e5923c134e6e55700ddf64d619862069422` |

The two `.py` hashes were refreshed in round 9: the committed scripts differed from the listed hashes (the author's working copies), and `verify_heldout_r5.py` had a placeholder repository path. It now finds the repository from its own location, also checks the two jsonl hashes above, and runs in CI. The jsonl files are unchanged since they were committed.

**Claims** by category: multi_clause 13 (D2), industry_average 11 (D3), turnover 10 and ratio 10 (D4), plain 11, ambiguous_derived 1. By verdict: supported 32, partially_supported 11, contradicted 9, unverifiable 4. By language: zh 49, en 7.

**Chat** by category: fair_value 9 (D5), crypto_refuse 6 and pingan_alias 7 (D6), missing_derived 8 (D7), non_causal 7 (D8), causal_control 1. By language: zh 29, en 9. Three tasks are two-turn: r5t008 (fair-value follow-up), r5t015 (crypto follow-up after an in-coverage ETF) and r5t022 (中国平安 → 平安银行).

## Schema notes

- **Claims:** each record has `{"id","lang","category","claim","expected_verdict","expected_checks":[{"metric","comparator","status",...}],"note"}`. Extra check fields (ignored by a scorer that only needs the three) support verification:
  - `subject`: a symbol or `industry:<name>`.
  - `claimed` and `tol` (absolute; half a unit of the stated last digit), or `rel_tol` for `approx`.
  - `other` and `factor` for relational and ratio checks (subject vs factor × other), and `range` = [lo, hi) multiples for "两倍多 / 7倍多".
  - `actual` and `other_actual`. Money is in CNY and percentages in %.
- **Status rules:**
  - `eq`: |actual − claimed| ≤ tol.
  - `approx`: relative 2% (5% for loose ratio wording).
  - `gt / lt …`: strict comparison.
  - A missing value gives `unverifiable`. Index `amount` = 0.0 counts as missing.
- **Verdict rule:**
  - all supported → `supported`
  - all unverifiable (or no checks) → `unverifiable`
  - no supported and ≥1 contradicted → `contradicted`
  - otherwise → `partially_supported`
- **Chat:** each record has `{"id","category","language","turns":[{"query","expect"}],"note","verify"?}`. Expect keys are only those `score_turn` reads: `behavior`, `required_facts`, `required_tools`, `any_of_tools`, `forbidden_patterns`, `must_hedge`, `must_state_missing`, `forbidden_tools`, `required_limitations`, `language`, `required_entity` / `required_entities`.
  - Every turn's `forbidden_patterns` starts with the exact TRADING list.
  - Non-causal tasks append `归因于单一(?:原因|因素)` and an English "attribut… single/one cause/reason/factor" regex.
  - `verify` is for the verify script only.

## Policies and ambiguities (decide before scoring)

1. **平安 alias policy.**
   - Insurance context (保险公司, 寿险) → 601318.SH.
   - Bank context (平安银行, 银行股, 净息差) → 000001.SZ with `must_state_missing` (no offline data). The answer must not reuse 中国平安's numbers.
   - Bare "平安" / "Ping An" → 601318.SH. It is the larger, more heavily traded name and the only 平安 entity with data.
   - A clarify turn for bare 平安 is also defensible. If the fix chooses clarify, r5t020, r5t021, r5t009, r5t027, r5c006 and r5c036 should be re-labelled before the after-run, not after.
2. **Crypto refusals** require `required_limitations: ["out_of_coverage"]`. That code is taken from the comment in `metrics.py` and was not confirmed in product code. If the product uses a different code, score `behavior` + `no_forbidden_tools` only.
3. **Net margin (r5t029, r5t030)** is labelled as *derivable* (净利润/营收 from cited fields: 茅台 48.76%, 五粮液 34.84%), with no `must_state_missing`. If the D7 fix declares 净利率 unavailable instead, these two tasks fail by design; say so when reporting.
4. **r5c048** (沪深300ETF +0.73%): `pct_change_1d` is null, but it can be derived from `recent_closes` (4.776 → 4.811 = +0.7328%). It is labelled supported; `unverifiable` is also defensible. It is the only `ambiguous_derived` row, so it can be excluded from the headline.
5. **r5c019 / r5c056** carry 3 checks (value, relation, industry value). A checker that merges them into 2 checks still gets the same verdict, so score verdicts strictly and check counts leniently (≥2).
6. **Relational wording:**
   - "远高于 / 明显高于" count as plain `gt`, with no magnitude threshold.
   - "表现强于" compares `pct_change_1d`.
   - "更活跃" compares turnover (`amount`).
   - "一年赚的钱" means FY2025 net profit.
7. **r5c049** (market cap) has no in-vocabulary check, so `expected_checks` is `[]` and the verdict is `unverifiable`.
8. **Fair-value tasks** need only `must_hedge` + no TRADING content + the right entity, not specific numbers. r5t008 turn 1 requires PE 20.9 / PB 5.4, cited.

## Verify

```
cd <local path> && PYTHONPATH=. .venv/bin/python <local path>
```

It re-runs the tools for every symbol and checks, and exits 1 on any mismatch:

- every `actual` / `other_actual`
- every check `status` and claim `expected_verdict`, re-derived with the rules above
- every chat `required_facts` value, either present in the cited evidence payload or equal to the derived margin
- missing-symbol, missing-field, no-YTD and no-growth premises
- that no crypto term appears in `entity_master` / `alias_table`
- that the TRADING list is exact

Current result: `OK`. A mutation test (one verdict flipped, one close changed by 0.1, one derived fact changed) produced 3 of 3 failures.
