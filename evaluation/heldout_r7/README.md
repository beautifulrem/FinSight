# FinSight held-out chat slice r7 (`chat_r7_heldout.jsonl`)

This is an independent held-out slice. It was written before the round-7 fixes so that the fixes for bug classes
G1–G6 (and G11 where chat can reach it) can be measured on phrasings that nobody tuned against.

- Date written: 2026-10-01
- Repo and commit the facts were taken from: `<local path>` at `20e57a0` (`git rev-parse --short HEAD`), clean working tree
- Data: the offline runtime snapshot (`build_offline_service()`: market as of 2026-04-22, fundamentals FY2025, industry snapshot 2026-04-21/22)
- Nothing was committed. Nothing was written outside `<local path>`.

## Files

| file | sha256 |
|---|---|
| `chat_r7_heldout.jsonl` | `6551fb42cdcdda694db205de3e64bb98fdc0d0068f0b280d1a2742f08d61588d` |
| `verify_heldout_r7.py` | `f3974f4f5b73c571cf0ef8fff5248c0fefa67dd46699ef0f4c2ecee4da60ad56` |
| `build_heldout_r7.py` | `1b6bb4f974640c5258e7655ec2954441e043c456f49dcbbe404a38ad8954a9a4` |

`build_heldout_r7.py` generates the JSONL. Every required number is computed in the script from live calls to the offline tools. No number was typed in by hand. `verify_heldout_r7.py` reruns the tools and checks every fact on its own.

## Protocol

1. I read the bug classes G1–G6 and G11 in `round7.md` §8 (the table rows only) to learn what kind of failure each class is. I did not reuse the reviewer's probe phrasings. I did not open `round7/` (probe scripts, `chat_probe.txt`, `facts.json`).
2. I learned the scoring contract from `evaluation/agent_eval/metrics.py`, which defines the expect keys and gates on `_is_supported`/`claim_numbers`. I did not read the verifier source. I treated the two matcher functions as black boxes and ran them on probe numbers to measure how they behave:
   - The relative tolerance is about 0.5%.
   - The sign is ignored.
   - 亿, 万, billion and million are scaled, and so are % and fractions.
   - 万亿 and trillion are not scaled.
3. I ran the real offline tools (`get_price_history` and `get_fundamentals` on the 7 snapshot securities and the 5 industries, plus HK names) and built every fact from their payloads. The snapshot has fundamentals for only 3 companies: 600519.SH, 000858.SZ and 601318.SH. It has prices for 7 securities: those 3 plus 510300.SH, 159915.SZ, 512880.SH and 000300.SH. All conversations stay inside that universe. The only exceptions are the HK refusals.
4. Precision: each derived value is stored rounded to the precision an answer would naturally use, given by `derive.dp`. The verifier checks that the stored value equals the recomputed value at that precision. It also checks that the scorer's matcher accepts the stored value both for the exact result and for its natural rendering (亿 / billion).

### What I did not read

- No code under `query_intelligence/`. I imported only `build_registry_for_service`, `_is_supported`, `claim_numbers` and `answer_texts`, and I called them only by running them.
- Nothing under `evaluation/agent_eval/tasks/`, `evaluation/claim_bench/`, `evaluation/heldout_r4/`, `heldout_r5/`, `heldout_r6/` or `evaluation/results/`.
- Nothing in `<local path>`.

### What I did read, for full disclosure

- `evaluation/agent_eval/metrics.py`, in full.
- `evaluation/agent_eval/runner.py`, lines 1–260 and the `_turn_record`/`_task_meta` helpers.
- `evaluation/agent_eval/build_tasks.py`, lines 28–100. This covers the `TRADING_PATTERNS` constant, the `MARKET`/`FUNDAMENTALS` tables, the `_turn` helper and the first single-fact template.
- The top-level key names of `data/structured_data.json`.
- One `grep "G[0-9]+"` over `round7.md`. It also printed one-line snippets from §2/§3/§9 that mention G7–G13. I did not use them.

## Schema

Each line has this shape:

```json
{"id", "category", "language", "turns": [{"query", "expect": {...}, "note", "verify_absent"?}]}
```

The `expect` keys are those read by `metrics.score_turn`:

- `behavior`
- `required_facts`, each with `evidence_id` and `value`, plus an extra `derive` spec
- `required_tools`
- `any_of_tools`
- `forbidden_patterns`, always exactly the four TRADING patterns
- `must_hedge`
- `must_state_missing`
- `required_limitations`
- `language`
- `required_entity`
- `required_entities`

`score_turn` ignores the extra keys: `note`, `derive` and `verify_absent`.

`derive` is `{"op": value|sub|div|pct|mul|margin_gap, "a": "<evidence_id>:<field>", "b"?, "k"?, "dp"}`. `sub` and `margin_gap` are absolute differences.

## Counts

There are 53 conversations and 129 turns: 33 conversations are zh (62%) and 20 are en (38%). By turn count, 33 conversations have 3 turns, 3 have 4 turns, 1 has 2 turns and 16 have 1 turn, so 36 of 53 have 3–4 turns. There are 112 required facts.

| category | zh | en | conversations | turns | targets |
|---|---|---|---|---|---|
| `gap_followup` (a) | 14 | 9 | 23 | 72 | G1 (gap/ratio after `X的M → Y呢`), G2 (前者/后者, "the first one"), G3 (净利率 vs 净利润, 市净率 vs 市盈率, 毛利率), G5 net margin inside sessions |
| `derived_metric` (b) | 7 | 4 | 11 | 24 | G5: EPS absent (`must_state_missing`, 3), holding value N × close (5), "净利润是营收的百分之几" net margin (3) |
| `fair_value_implied` (c) | 4 | 2 | 6 | 10 | G6: implied price from sector PE/PB, `must_hedge` + no target-price pattern |
| `hk_out_of_coverage` (d) | 5 | 3 | 8 | 14 | G6: H shares (中国平安H股, "its H shares", 02318.HK), HK companies and lookalikes (腾讯控股, 比亚迪电子, BYD Electronic), refused with `out_of_coverage`, plus recovery turns |
| `control` (e) | 3 | 2 | 5 | 9 | explicit two-operand gap, explicit pair, index/ETF close, aspect switches |

Expected behaviour is `answer` on 121 turns and `refuse` on 8. `must_hedge` is set on 6 turns and `must_state_missing` on 4.

G11 is a UI-only class (KPI tiles and the English fact-check label). This chat slice can only touch it indirectly, through English answers about 白酒/baijiu. It has no dedicated scorer check.

## Verify

```bash
FINSIGHT_REPO=/path/to/FinSight /path/to/FinSight/.venv/bin/python verify_heldout_r7.py [chat_r7_heldout.jsonl]
# or, with the folder copied inside the repo (it walks up to the directory containing query_intelligence/):
cd /path/to/FinSight && .venv/bin/python evaluation/heldout_r7/verify_heldout_r7.py
```

On `20e57a0` the output is `OK`: 53 conversations, 129 turns, 112 facts checked, 0 failures. One info line notes that `get_price_history('中国平安H股')` returns `price_601318.SH`, the A-share.

The script checks:

1. **Schema:** ids are unique, the keys are right, every turn has a note, the language matches, and `forbidden_patterns` exactly equals TRADING.
2. **Facts:** each fact is recomputed from fresh tool output. The `evidence_id` must be returned by the tool and must be the first operand's id. The scorer must accept the stored value.
3. **Discrimination:** a derived fact must not be matched by its own operands. This is the "lists the two prices" failure.
4. **Wrong-metric collisions:** the script warns when a fact also matches another field of the same fundamentals/industry payload.
5. **Missing fields:** fields in `verify_absent` must really be absent (茅台 `gross_margin`, `eps` for all three companies).
6. **Refusals:** every refusal must carry `out_of_coverage`, and none of the HK names may return a `.HK` security.
7. **Scorability:** a synthetic gold response per turn must pass `metrics.score_turn`.

## Ambiguities and judgement calls

1. **Citation on gap turns.** A derived fact (gap, ratio, margin, holding value) is filed under the first operand's `evidence_id`. The `facts` check also requires that id to be in that turn's `evidence_used`. So an answer that is correct but cites only the second operand, or none, fails the strict score and passes `uncited_success`. Operands are not required on gap turns, only the derived number.
2. **Matcher tolerance is about 0.5% relative.** Ratios need two decimals in some cases:
   - 2.83: "2.8倍" fails.
   - 2.18: "2.2倍" fails.
   - 7.36: "7.4" is about 0.54% off and fails.
   - 9.93%: "约10%" fails.
   - 1.56: "1.6倍" fails.
   - 14.01: "14倍" passes.
   - 3.79: "3.8倍" passes.

   A rounded answer that a human would accept can therefore fail. That is a property of the scorer, not of the labels.
3. **万亿/trillion are not scaled by the matcher.** For that reason Ping An's revenue (1.218万亿) is not a required fact (the first turn of `r7h_gap_en_02` and of `r7h_derived_zh_06`). Ratios built from it are still required.
4. **万 scaling collision.** Holding 10,000 units of the 4.811 ETF gives 48,110, which the matcher accepts from "4.811" alone. The ETF lot is therefore 两万 (20,000 units, 96,220 元). No holding-value fact is a power-of-ten multiple of its price.
5. **EPS (`must_state_missing`).** The marker list in `metrics.py` is broad ("没有", "无法", "not ", …), so a price-only answer that happens to contain such a word would pass. Deriving price ÷ PE (茅台 ≈ 57.30, 五粮液 ≈ 4.82, 平安 ≈ 6.16) is not required. It is not penalised either.
6. **HK names are labelled `behavior: refuse` with `out_of_coverage`.** This follows the brief. An answer-route reply such as "H股不在覆盖范围（A股为53.61）" would fail `behavior` even though a human might accept it. The tool layer maps `中国平安H股` to the A-share, and maps `02318.HK`/`00700.HK` to unrelated funds (023718.SZ, 007060.SZ). A refusal has to come from coverage/NLU, not from a tool miss.
7. **Fair value.**
   - These turns have no required numbers. The implied price, for example 1409.5 × 27.3 / 24.6, may or may not be computed.
   - `must_hedge` accepts weak markers such as "可能".
   - The TRADING pattern catches only `目标价\s*\d`. A statement like "股价应为1564元" is not caught by any pattern, so it is not penalised as a target price.
8. **Holding value has no hedge requirement.** A refusal or clarification there (the G5 symptom `lexical:judgment_or_timing`) fails `behavior`.
9. **`required_entities` on gap turns** relies on `nlu_summary.entities` exposing both carried targets. A system that computes the right number without surfacing both entities fails `entities`. This matches how the project scores "这两家" follow-ups.
10. **Mixed dates.** Close and change are as of 2026-04-22. Fundamentals are FY2025. The 白酒 industry PE is from 2026-04-21. The gap in `r7h_ctrl_zh_01` (27.3 − 24.6 = 2.7) mixes these as-of dates, the same way the tools do.
11. **Signs.** 五粮液's 1-day change is −0.5337, stored as −0.53. The matcher ignores sign, so "下跌0.53%" passes. The cross-sign gap 0.73 − (−0.53) = 1.26 points is the intended answer. A system that reports 0.20 (difference of the absolute values) fails.
12. **"三家里最高的比最低的"** (`r7h_gap_zh_08`) expects 33.0 − 15.2 = 17.8 and requires all three entities.
13. **Containment pairs.** Each pair is checked by the wrong-metric warning, and the verifier found no collisions:
    - 市净率 sessions (`gap_zh_03`, `gap_zh_13`, `gap_en_03`) would fail if PE figures were answered.
    - 净利率 sessions (`gap_zh_04`, `gap_en_05`, `gap_zh_12` t3) would fail if 净利润 amounts were answered.
    - 毛利率 (`gap_zh_12`, `ctrl_en_02`): 茅台 has no gross margin in the data, so turn 2 of `gap_zh_12` must say it is missing.
14. **Running the slice.** Load it with `evaluation.agent_eval.runner.load_tasks(path)`, run it with `run_agent_tasks`, and score it with `metrics.aggregate` plus `breakdown(records, "category")`. Use the live offline registry and no replay snapshot. The facts come from the same offline assets, so a snapshot recorded from them is equivalent.
