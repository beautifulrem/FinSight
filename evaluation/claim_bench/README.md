# Claim-check benchmark (v1)

Labelled market claims for `POST /agent/claim-check` (`query_intelligence/agent/claim_check.py`), checked
against the offline snapshot (`data/structured_data.json`: prices as of 2026-04-22, reports as of
2025-12-31).

| File | Claims | Use |
| --- | --- | --- |
| `claims_v1.jsonl` | 131 (115 zh, 16 en) | dev set: the checker was developed against it |
| `claims_v1_holdout.jsonl` | 47 (38 zh, 9 en) | held out: written together with the dev set, before the fixes, and run **once** at the end |

Held-out file sha256 (recorded when it was written, before any checker change):

```
a48aa59412a06f81c603604b587dec82948ae5914fdf176bdcdb7e42c0028bb8  claims_v1_holdout.jsonl
```

## Row format

```json
{"id": "d034", "lang": "zh", "category": "comparator", "claim": "贵州茅台ROE超过30%",
 "expected_verdict": "supported",
 "expected_checks": [{"metric": "roe", "comparator": "gt", "status": "supported"}],
 "basis": "M roe 33% > 30"}
```

* `expected_checks` has one entry per number in the claim, in order (a range "20到30倍" is one number; a
  number-less move such as "昨天下跌了" is one check against 0). `metric: null` means the metric is not
  scored for that check.
* Expectations were derived by hand from the tool outputs (`get_price_history`, `get_fundamentals` on the
  offline service), **not** by running the checker. `basis` names the value used.
* Categories: exact, false, approx, comparator, range, negation, direction, sign, growth, multi, no_data,
  unknown_target, opinion, wrong_unit, index_etf, chinese_numeral, forecast, period, multi_day.

## Labelling policy (written before the fixes)

* Tolerance: `eq` matches within half a unit of the last written digit or 2% (5% for `approx`).
  Bounds (`gt/ge/lt/le`, `range`) are literal: "ROE不超过15%" is contradicted by 15.2%.
* Negation turns `eq` into `ne` and flips a bound ("没有超过30%" → `le` 30; "并没有跌" → `ge` 0).
* Unverifiable: no target or no data, a metric the sources do not carry (YoY growth offline, index P/E),
  a unit that does not fit the metric (P/E in %, ROE in 倍, a price in 亿元, an amount with no unit), a
  forecast, a period other than the data's, or a multi-day move (only the daily change is available).
* Opinions without numbers have no checks and are `unverifiable`.

Known judgement calls: "五粮液市盈率24.6倍，比茅台低" is labelled contradicted (24.6 is bound to 五粮液);
the holdout's industry-average PE ("行业平均11.8倍") is labelled supported from the 保险 industry snapshot,
which the checker does not use today.

## Run

```bash
python -m evaluation.claim_bench.run --set dev       # -> evaluation/results/claim_bench-dev.json
python -m evaluation.claim_bench.run --set holdout   # once, at the end
```
