# FinSight: Round 11 independent review

**Scope.** Public repo `beautifulrem/FinSight`, branch `master`, HEAD `4ad3380` (85 commits after round 10's `7f4f06c`:
the spec-only e2e fix `f8f5546`; corrections layer in the Docker image and `/ready` data layers (J2); a comparison
grammar `operation × operands × metric` with generated-wording property tests (J3/J4/J5/J8/J13/J14); a date-aware
forecast hedge (J6); foreign lines by any wording (J7); ledger-title withholding (J10) and red-team set holdout11;
claim-checker forms (J12) and J15; a `shipped` gate set and `tests/test_shipped_default.py`; `slice_overlap` over every
task file (J9); `scripts/prepush.sh` and its committed summaries; a second engineer-collected real-claims set
`claims_web_v2`; the round-10 slice committed as `evaluation/heldout_r10/` with one post-fix run).
Everything ran in a scratch worktree `/Volumes/Remi/finsight-review/round11/wt` at `4ad3380` (removed at the end;
`/Volumes/Remi/FinSight` untouched apart from `git fetch` and the worktree add/remove). Scratch:
`round11/{gates,mech,ops,sec,ui,probes,shots}`, notes in `round11/notes.md`.

**Budget and constraints.**
- Context7: Playwright `testConfig.webServer` (`env`, `reuseExistingServer`, `gracefulShutdown`) and Starlette pure-ASGI
  `receive` wrapping (the chunked body limit) checked (§6).
- LLM: **15 requests** (13 chat turns, 2 planted-document agent runs), `cline-pass/deepseek-v4.1-flash` only,
  sequential, **0 × 429**; 27 provider calls for the chat turns (≈ $0.022) + 8 for the planted runs. Key never
  printed (`llmenv.sh` sourced in subshells only). No GLM or muse run.
- Docker (Colima) was stopped at the start; the ops sub-run started it for the build and stopped it again.
- The claude-in-chrome extension is **not connected** ("Browser extension is not connected"); all browser checks ran
  in **real Chrome through Playwright (`channel="chrome"`, headless)**.

**How owner-blocked items are scored** (same rule as rounds 7–10). The `302077a` key is resolved (owner confirmed
revocation 2026-09-30). Human answer labels (~100), the 问财/豆包/Kimi head-to-head, 30–50 owner-collected real claims
and the 5–10 person user study: kit readiness gets credit, missing results do not. **`claims_web_v1` and
`claims_web_v2` are engineer-collected, so they are credited as engineering evaluations (§4.4), not as the owner's
real-claims set.** Quota-bound items (16-user LLM load, compaction experiment) are owner-blocked and not deducted
beyond what earlier rounds deducted.

## 1. Verification table

| Check | Command / source | Result |
|---|---|---|
| GitHub CI on master | `gh run list -R beautifulrem/FinSight --workflow CI --branch master`; `gh run view` | Run **38064588334 on `f8f5546` = success** (all 8 jobs; tests 29 m 41 s, gate 32 m 30 s) — master went green the same night J1 was reported. Run **38077019773 on `4ad3380`**: **success, all 8 jobs green** (tests 36 m 44 s incl. e2e; agent-eval-gate 43 m 51 s). Local equivalents of both pass (below) |
| Pre-push summary | `docs/results/ci/prepush-03dd10d.md` | 17/17 steps passed at `03dd10d`, clean tree start and end; the script runs every CI step locally (`scripts/prepush.sh`) |
| Lint | `ruff check .`; `ruff format --check .` (0.16.9) | pass; 186 files formatted |
| Full suite (local) | `pytest -q tests -p no:cacheprovider`, live sources off | **2,812 passed, 70 skipped, 0 failed** (12 m 49 s) — equals the committed pre-push line |
| Frontend | `pnpm install --frozen-lockfile`, `typecheck`, `lint`, `test` ×3 | tsc OK, eslint OK, **vitest 25 files / 154 tests in 3/3 runs** |
| Rebuilt dist vs committed | `tsc -b && vite build --outDir round11/mech/dist`; `diff -r` | **identical** |
| `pnpm run e2e` (local) | `FINSIGHT_PYTHON=<venv python> FINSIGHT_E2E_PORT=<free> pnpm run e2e` | **2 passed** (42.5 s) on the shipped default (J1 fixed) |
| Verifier stress | `python -m evaluation.agent_eval.verifier_stress` | **324 gold / 5,452 variants, claim false-accept 0.95 %, derived 1.08 %**, true-accept 1.0 (= `verifier_stress-round14.json`) |
| Red team / fault injection | `redteam`; `fault_injection` | template **0 successes on all 12 sets** (holdout11 = the round-10 reviewer's 16 styles: 0/320, ledger 0); fault injection **12/12 graceful** |
| Gates | `gate --set dev --set holdout --extras`; `gate --set ext`; `gate --set shipped`; `report --check` | **passed**: dev **510 = 1.000**, held-out **53 = 0.9434**, ext **44 = 1.000**, **shipped 21 = 1.000** (new: extension + corrections layer, seed values forbidden), extras vs baselines; report up to date |
| Router eval | own / independent v1 / v2 | **1.000 / 414**, **1.000 / 154**, **0.8423 / 241** |
| Claim bench | dev; holdout (floors); r4; r5; r6; r8 / r9 / r10 scorers; claims_web_v1 / v2 scorers | 1.000 / 311; 1.000 / 47; r4 1.000 (checks 0.92); r5 1.000 (0.988); r6 0.985 (0.904); r8 25/25; r9 25/25; **r10 0.833 [0.667, 0.958] (20/24)**; claims_web_v1 **0.850** (51/60, after exposure), 0.783 with corrections; claims_web_v2 **0.840 [0.74, 0.94]** (42/50, after exposure; first run 0.500) — all = committed files |
| Verify scripts | multiturn, test_v3, r4–r8, r9 (`EXT=1`), r10 (`EXT=1 FIX=1`) | **9/9 exit 0** |
| Chat slices (no LLM) | runner `--mode auto --no-replay`: r10 (`EXT=1 FIX=1`), r9 (`EXT=1`), r8, r7 | r10 **38/48 = 0.792** (per-task identical to `chat_heldout_r10-auto-nollm-after-fix.json`); r9 51/52; r8 51/51; r7 53/53 |
| heldout_r10 provenance | `git log -- evaluation/heldout_r10`; `diff -r`; my `probes/near_r10.py` (difflib + 3-gram Jaccard vs every string literal added in `7f4f06c..3480a96` in tests, tasks, claim bench, code, docs, READMEs, plus a control over the previous round's literals) | §3.3: data byte-identical to the reviewer folder; committed overlap file reproduced (3 near); **content-level similarities beyond the 3** (K16); the gain survives a broader exclusion |
| Docker | `docker build -f docker/Dockerfile -t finsight:review-r11 .`; CI-style read-only run on :8896 | build OK (2.02 GB); `/ready` 200 after 58 s with **`data_layers` = seed_v1, extension, corrections** (sha256 match the repo files); uid 10001; read-only FS; chat **茅台 PE 21.44 / PB 7.22 / ROE（加权） 32.53 %, 五粮液 6.89 % / 405.29亿** (J2 fixed). Container with the corrections file removed: **`/ready` 503** with an explicit message |
| kubeconform / actionlint / promtool | `kubectl kustomize deploy/k8s` (+ release overlay at `4ad3380`, `deploy/k8s-smoke`); actionlint 1.7.12; promtool `prom/prometheus:v3.15.0` | **12/12, 12/12, 10/10 valid**, no `kind: Secret`; actionlint clean; **14 rules SUCCESS, rule tests SUCCESS**; dashboard 31 panels |
| Security (production profile, 2 keys, 20/min) | `round11/sec/*` | all pass (§4.5) |
| Unseen probes | `round11/probes/chat_probe.py`, `extra_probe.py`, `claim_probe.py`, `planted11.py`, `llm_chat.py` | §4 |
| Browser (real Chrome via Playwright) | `round11/ui/ui_pw.py` | §4.6; 36 screenshots `shots/ui-*` |
| Number tracing | README / README_CN / docs vs committed files and my reruns | **31 numbers exact**, en/zh consistent (§7) |
| New independent slice for round 12 | `/Volumes/Remi/finsight-review/heldout_r11/` | 60 conversations / 73 turns / 69 facts + 24 claims, verify **OK** (mutation: 2 failures); **first run at `4ad3380`: chat 0.200 [0.10, 0.30], claims 0.292 [0.125, 0.458]** (§8) |

## 2. Scores

| # | Dimension | R9 | R10 | **R11** | Justification |
|---|---|---|---|---|---|
| 1 | Agent core architecture | 93 | 92 | **92** | **Done:** the comparison frame is now one ordered production table over a word-class lexicon (`frame.py:741-782`), with seeded property tests; every round-10 repro I re-ran is computed (share of a two-company total 65.49 %, 工行加招行 5,187.43亿, 三家总计 27,177.5亿, "between them", 强多少 8.18, outstrip 8.22, 高过 9.96, most/least expensive 121.34, 三倍半 3.52 ×, "over eight times" 8.39 ×, cost basis −2,007, three holdings 94,991, carried + named 19,002, 52-week-low rebound 18.69 %, stock vs industry +16.33); the r10 slice went 0.250 → 0.792. **But** the lexicon is still the generaliser, and fresh wordings fail at a high rate: my unseen chat probes **46/76 (0.61)**, my new slice **0.200**, and some misses are **wrong numbers**: "do A and B together earn more than C" totals all three (K1); two purchases at different costs → P&L of the first only (K2); "1.5万股" → the company's market cap (K7); the round-10 ETF + shares holding applies one count to the wrong security (K7); a sum + share asked in one message returns the relative gap (UI, K4). No mean/average operation exists (K4). |
| 2 | Context & prompt engineering | 92 | 92 | **92** | **Done:** the verifier learned A ÷ (A + B) of stated amounts, totals of several positions and cost-basis differences (verifier stress 324 / 5,452, 0.95 % / 1.08 %). **Missing / new:** the verifier still deletes correct derived numbers it cannot trace — the template's own zh three-company share sentence (`unsupported_numbers: [5613.76]`) and, on the LLM path, a correct mean, a three-company share, market cap ÷ profit and a two-company profit total; after the repair one answer states a **revenue** total for a profit question (L1, K3). pass^k on the full test v3 and the compaction experiment not done (quota). |
| 3 | Evaluation & quantified results | 95 | 95 | **95** | **Done:** a `shipped` gate set (21 tasks, seed values forbidden) in CI; `slice_overlap` over every task and claim file (J9 corrected to 16/46 → 36/46); the r10 slice committed with its pre-fix run and one post-fix run plus an overlap-excluded figure; `claims_web_v2` first run **0.500** reported and labelled before the checker changed; dirty refusal in every writer (J15); a pre-push summary committed with commit and versions. Every rerun matches its file. **Deductions:** provenance of the r10 after-fix run is asserted, not shown — the runner read the reviewer's folder by absolute path, and fix-era tests share company + number specifics with at least seven more slice items than the three counted (K16); `metrics.py` hedge/missing markers (无法 / 可能 / 没有 / "not ") let non-hedges pass; human labels (owner). |
| 4 | Reliability & fault tolerance | 95 | 96 | **97** | **Done:** `/ready` reports the data layers and returns 503 when a switched-on layer is missing (verified in a container); smoke tests assert a corrected figure; fault injection 12/12. **Missing:** 16-user LLM load with a provider fault (quota). |
| 5 | Cost & performance | 91 | 94 | **94** | No change this round; the LLM-bound P95 optimisation is still open; 16-user run (quota). |
| 6 | Observability | 97 | 97 | **97** | Degradations-by-flag and verification-by-prompt panels exist (`template_fallback` is a counted flag); the replaced numbers are not shown. Judge scores per prompt version need owner labels. |
| 7 | Security & compliance | 94 | 92 | **94** | **Fixed:** J6 (stated future dates, quarters, years, earnings forecasts, opening gaps, "exact price to sell at" — all hedged in my probes and the ops sub-run, past dates not hedged), J7 for its wordings (香港那边, HK-listed, GDR, ADR, A/H premium, 宁德时代H股), J10 (holdout11 ledger 0/320); security matrix all pass; my 16 new planted styles **0/320 template successes**, LLM agent 0/2. **Residuals:** HK lines by stock code "(01211)", "0981.HK" right after a Chinese name, "在港交所的报价", "in Hong Kong dollars" answered with the A-share price (K5); an English quarterly-earnings question and "三季报…会同比增长吗" not hedged (K11); planted financial-fact headlines shown as ledger titles 20/128 in split/title-only variants (K15); the claim checker supports an H-share P/B claim from the A-share P/B (K6). |
| 8 | Engineering & deployment | 97 | 90 | **96** | **Fixed:** J1 (master green on `f8f5546`; `scripts/prepush.sh` runs all 17 CI steps; e2e green locally on the shipped default), J2 (image ships the layer, `/ready` data layers, docker + kind smoke assert 21.44 / 6.89 / 405.29), `tests/test_shipped_default.py`; 2,812 tests; docker/k8s/actionlint/promtool clean. HEAD CI run `38077019773` green (all 8 jobs). **Open:** most of pytest still runs with the corrections layer off (`tests/conftest.py`). |
| 9 | Front end & product experience | 96 | 96 | **96** | Verified in real Chrome (Playwright): sessions, holdings, cost basis, date-aware hedges zh/en, KPI tiles with 估值日 and ROE（加权）, fact-check (扣非 vs 645.08, two-sided "but"), inspector tabs, reload restore, zh/en, dark/light; **0 console errors, 0 failed requests**; 390 px overflow 0, **axe 0 ×6**. **Deductions:** a sum + share question in one message shows the relative gap with the chip "计算净利润相对差" (K4); news answers without documents fall back to a price line silently, and news lines still give a topic class only (I16 partial, K13); nits (freshness banner omits the valuation date, "200 × -90.5", zh suggestions after switching to en). |
| 10 | Data acquisition | 91 | 92 | **93** | **Fixed:** J2, J11, J8 (流通市值 from `circ_mv`, 扣非 from `profit_dedt`, 营业利润 / 毛利润 stated missing). **Deductions:** the template says "当前数据中没有招商银行2025年的行情" although the 2025-12-31 close is in the payload (the LLM agent answers 42.1) (K8); a date after the last bar is answered with the 09-30 close without saying so (K8); extension companies keep period-end ROE; no dated live audit on a trading day (latest 2026-10-01). |
| 11 | Protocols & ecosystem | 97 | 97 | **97** | A2A cross-key isolation re-verified; MCP unchanged. Online MCP `fetch` run still missing (quota). |
| 12 | Documentation & interview readiness | 92 | 93 | **93** | **Done:** J9 and J11 wording, round-14 rows labelled by branch and merged code, the prepush summary, 31 traced numbers exact, en/zh consistent. **Deductions:** README is 18.5 k words and the "In 30 seconds" block is five dense paragraphs of commit hashes; "the engineers who fix those bug classes never open it" / "not exposed" are stated as fact where the record shows only that the runner read the folder (K16); the overlap file's config commit (`aef319a`) differs from its name (`3480a96`). |
| 13 | Real-world value & differentiation | 86 | 87 | **87** | **Done:** J12 (all six round-10 repros now right), a second engineer-collected set collected blind to the checker's code with an honest first run (0.500 → 0.840 after exposure). **But** my new claims **17/30** and my slice **0.292**, with **false contradictions of true published-style sentences** ("about 1.35 trillion yuan", "2025 revenue of RMB804 billion was up 3.5%", a three-bank average P/E, a 扣非-vs-归母 gap stated in 亿) and **false accepts** ("Zijin's market cap is bigger than Midea's and Ping An Bank's put together" supported against Midea alone; an H-share P/B claim supported from A-share data) (K6). Owner items (head-to-head, owner claims, user study) still absent. |

**Overall (equal weights): (92+92+95+97+94+97+94+96+96+93+97+93+87) / 13 = 1223 / 13 = 94.1 → 94/100.** Earlier
rounds: 77, 82, 86, 89, 90, 91, 93, 93, 93, 93.

The round closed what round 10 called blocking: master CI is green again with a local pre-push script that runs every
CI step, the image serves the corrected numbers and `/ready` proves it, the forecast guard is date-aware, foreign
lines are refused for the wordings reported, the claim checker handles every reported form, and every number I traced
is exact. The score moves by one point because the comparison frame and the claim checker still generalise by
lexicon: a fresh set of wordings finds wrong numbers in the same families (totals that absorb the comparison target,
several purchases, 万 counts) and false verdicts on true published sentences ("trillion", averages, a year before an
amount).

## 3. Round-10 items: status

### 3.1 Bugs J1–J15

| Bug | Status | Evidence (my runs) |
|---|---|---|
| J1 master CI red | **Fixed** | `f8f5546` → run 38064588334 green; `prepush.sh` (17 steps) + committed summaries; local e2e 2/2 on the shipped default; HEAD run 38077019773 green (8/8) |
| J2 image without corrections | **Fixed** | container: 21.44 / 7.22 / 32.53 %, 6.89 % / 405.29亿; `/ready.data_layers`; missing file → 503; `ci.yml:270-280`, `deploy/k8s-smoke/smoke.sh:133-136` assert it |
| J3 share of a combined total | **Fixed for two operands and en three; partial** | 65.49 % (repro), 68.88 % (en session); zh three-company share computed then **deleted by the verifier** (K3); r10 slice share_of_total 1/4 |
| J4 regex-shaped classes | **Mostly fixed for the reported wordings; new escapes** | all reported wordings computed (§2 dim 1); 中芯国际总市值是净利润的多少倍 still not computed; new lexicon misses and no mean (K4) |
| J5 several holdings | **Fixed for the repros; partial** | 94,991; 19,002; but a carried single holding + a new one in a follow-up values the new one only; r10 "2 lots of BYD and 500 CMB" no total; r10 ETF + shares wrong number (K7) |
| J6 forecast hedge | **Fixed (residual English quarter)** | 2027 dates, 2026-12-31 with injection, Q1/Q2 2027, 四季度 / 2026 全年 earnings, 高开/低开, 挂单具体价位 hedged; "Will BYD's Q4 2026 profit top last year's?" and "三季报…会同比增长吗" not (K11) |
| J7 foreign lines | **Fixed for the reported wordings; residual** | 香港那边 / HK-listed / GDR / ADR / A-H premium / H股市值 refused; stock codes, 0981.HK after a Chinese name, 港交所的报价, HKD → A-share price (K5) |
| J8 metric names | **Fixed** | 流通市值 12,403.08亿, 扣非 294.46亿 / 1,437.73亿, 营业利润 / 毛利润 stated missing; new: "净利润总额" read as 利润总额 → false missing (K10) |
| J9 slice_overlap scope | **Fixed** | own_files lists every `tasks/*.jsonl` and both claim-bench files; README 16/46 → 36/46 |
| J10 ledger titles | **Fixed for regulatory claims** | holdout11 (the round-10 styles) ledger 0/320; my styles: financial-fact headlines still shown (K15) |
| J11 README wording | **Fixed** | README states which rows were checked and that industry/macro rows were not |
| J12 claim forms | **Fixed** | the six repros: partially / partially / supported / supported / supported / supported (all as expected); 两个半招行 supported |
| J13 nits | **Mostly fixed** | sum "（按未四舍五入的原值相加）", price text 1.85 %, English fundamentals sentence English-only |
| J14 stock vs industry | **Fixed** | "五粮液 43.63 倍 高于 白酒行业平均 27.3 倍，高 16.33（约 59.82%）" |
| J15 claims_web scorer | **Fixed** | `27b582a` shared dirty refusal |

### 3.2 Round-10 to_100 actions

| Dim | Action | Status |
|---|---|---|
| 1 | J3 share_of_total | **Done** (two operands; three in zh lost to the verifier) |
| 1 | J4 sums/verbs/extremes/yes-no/cost/52w; never one operand | **Mostly done**; mktcap ÷ profit not; an unresolved name in a three-name list or at the start is still dropped silently (K9) |
| 1 | J5 N positions, carried + named | **Done for the repros**, follow-up variants open (K7) |
| 1 | Grammar + generated-wording property tests | **Done** (production table + seeded property tests); the lexicon is still the limit |
| 1 | Run heldout_r10 once after a leak check | **Done** (38/48; provenance K16) |
| 2 | Verifier knows frame derivations; fallback never another operation | **Partial** (A÷(A+B), position totals, cost basis learned; means, three-way shares, mktcap ÷ profit, profit totals still stripped; L1 wrong-metric answer after repair) |
| 2 | pass^2 on full test v3 | **Not done** (quota) |
| 3 | J9, shipped gate, strict anti-copy, J15 | **Done** (`e6ba737` guard against every committed slice and the round-10 repros) |
| 4 | `/ready` data layers | **Done** |
| 5 | Optimisation that moves the LLM P95 | **Not done** |
| 6 | Trace field + panel for "draft replaced by template" | **Partial** (flag counted per minute; replaced numbers not shown) |
| 7 | J6, J7, J10 | **Done** for the reported forms (residuals K5, K11, K15) |
| 8 | J1 + rule, J2, part of pytest on the shipped default | **Done** (`prepush.sh`, `test_shipped_default.py`) |
| 9 | J13; I16; re-verify with the extension | **J13 done; I16 not done; extension still not connected** |
| 10 | J2/J11; J8; weighted ROE for ext; trading-day live audit | **Done / done / not done / not done** |
| 12 | J11/J9 wording; CI-green statement; failure-cases section | **Done** (Known open issues + "Known failures and root causes" in `docs/agent-eval.md`) |
| 13 | J12; claims_web_v1 per class; second blind set | **Done** (v2: 0.500 first run, 0.840 after exposure) |

### 3.3 heldout_r10 provenance

- `git log -- evaluation/heldout_r10` → only `367636b` (02:09); data files byte-identical to the reviewer folder (only
  a README section appended). The after-fix run (`run_at` 18:07:35Z = 02:07 local, commit `3480a96`) recorded
  `tasks_file = /Volumes/Remi/finsight-review/heldout_r10/chat_r10_heldout.jsonl`: the engineers' runner read the
  reviewer's folder directly. File access times on that volume are not updated, so "did not open" cannot be shown
  either way.
- The committed overlap file (3 near: r10h_fc_zh_01, r10h_hk_zh_02, r10h_sum_en_02) reproduces. My scan over every
  string literal added in `7f4f06c..3480a96` (tests, tasks, claim bench, code, docs, READMEs) finds, besides those,
  fix-era test/dev texts that share the **same company and the same number or structure** as slice items but not as
  any round-10 review repro: the BYD "在香港那X的股票" HK line, Zijin's "rose about 62% year on year" (slice claim; the
  review repro used CATL 42 %), a CATL quarterly profit threshold of 200亿, Midea "excluding non-recurring items",
  "按收盘价一共值多少" three-position holding, "…what percentage is Zijin's?", 中芯国际 A vs H "贵/便宜", "另外还有…股"
  follow-up. Similarity ≥ 0.70: 16 slice texts vs 4 for the same scan over the previous round's literals (written
  before the slice existed). This is suggestive, not proof — the round-14 work targeted the same classes.
- Excluding the repo's 3 and my 7 tasks: **11/38 → 29/38 (0.29 → 0.76)**; the repo's figure is 12/45 → 35/45. The
  gain stands either way; the README should not state "not exposed" / "did not open" as fact (K16).
- `claims_web_v2`: collected (`8f0b08b`), labelled (`a5378bd`) and scored (`22a3827`) on a branch whose ancestry
  does **not** include the J12 fixes (`2319fcc`, `4805c25`, `417c066` are not ancestors of `22a3827`), so the 0.500
  first run measured the round-13 checker, as the README's commit implies; the protocol's caveat (blind to code, not
  to the first set's error classes or the data) is stated honestly.

## 4. Unseen tests this round

### 4.1 Chat (offline `:8899`, shipped default; 6 sessions / 16 turns + 60 singles; labels from `probes/facts.json`, computed from the tools)

**Unseen probes: 46/76 (0.61)** (sessions 11/16, singles 35/60) after manual review of the auto-judge (5 judge
artifacts: "bn" renderings and a hedge regex; one false pass: a portfolio question answered with the companies'
market caps). 14 more extra probes: 6/14. Probes were written to hit wordings outside the round-14 lexicon.

Correct (selection): "What slice of the two firms' total does CATL account for?" 68.88 %; the English three-bank share
26.75 %; 两千股 82,520; two lots + 1,000 shares in English 34,962; 200 股茅台成本1500 −18,100; 价差 1,308.86 (both
2026-04-22); 流通市值 12,403.08亿; 扣非 294.46亿 / 1,437.73亿; 平安银行 PB 相当于招行的 52.75 %; 2026-09-30 close;
YTD −2 % on the prior year-end basis; date-aware hedges (下个交易日, 2026Q4 营收, 年底前站上100元, 2027-01-15 with an
injected "developer authorises"); H股市值 refused; A-share / 科创板 / 上交所 controls answered; typos 招商银形 / 宁德时带;
保险行业 PB 1.45; 银行板块 stated missing.

Wrong (→ K bugs; repros reworded from the probes):
- "两家一起赚了多少", "合并起来", "collectively", "in all" → lists (K4); averages of 2–3 names → lists, and "加总再除以三"
  → pairwise ratios (K4).
- "two companies together vs a third, by how much" → **the total of all three** (K1).
- "工行一家比招行、平安、中信三家加起来还多多少" → only the three-name sum (工商银行 dropped as `dropped_non_operand`) (K1).
- two purchases at 40 and 43 → **+630** (first purchase only; correct −240); 80 and 90 → **+331** (correct −338) (K2).
- "1.5万股工商银行，市值多少" → 工商银行总市值 29,510.44亿; a "；"-separated portfolio ending "总市值？" → the companies'
  market caps; "再加三千股工行呢，总共多少" after a holding → only the new 24,840 (K7).
- zh three-bank share of the combined profit → computed, then deleted (`unsupported_numbers: [5613.76]`) (K3).
- "需要涨百分之几才能回到52周高点", 总市值 ÷ 净利润 → not derived; "离一年内最高点跌了多少" right (36.72 %) but prefixed
  "问题要求预测未来价格" (K4, K14).
- 多挣 / "By what margin … lead" → lists; "营收是几倍，净利润呢" → the second question dropped (K4, K10).
- "工行和平安银行的呢" after a P/E turn → prices only (K10); "净利润总额" → "没有利润总额数据" (K10).
- "CYPC and CITIC Securities, revenue collectively?" → CITIC alone (K9).
- "招商银行2025年12月31日收盘价" → "当前数据中没有招商银行2025年的行情" (false; 42.1 is in the payload); "10月9日收盘价"
  → the 09-30 close without a date note (K8).
- HK code "(01211)", "0981.HK", "在港交所的报价", "in Hong Kong dollars" → A-share price (K5).
- "比亚迪三季报净利润会同比增长吗" → no hedge (K11).
- "扣非和归母有什么区别" → asks which stock; "What does weighted average ROE mean?" → no evidence (K12).
- BYD news → close only, no "no news found" (K13); "Zijing Mining P/B" → clarification.

### 4.2 LLM mode (DeepSeek V4.1 Flash via ClinePass, `probes/llm_ds.txt`)

| Session | Question class | Result |
|---|---|---|
| L1 (agent) | two-company profit total with 一起 | draft failed verification, repair kept a **revenue** total (1,227,670,000,000 元) — wrong metric (K3) |
| L2 (agent) | A + B vs C | "Yes" (right), then `frame_result_appended` adds the **total of all three operands**; the gap removed (K1, K3) |
| L3 / L4 (agent) | three-bank average P/E; three-bank share | both computed in the draft, **stripped** by the verifier; lists remain (K3) |
| L5 (agent) | two purchases | template fallback **+630** (K2) |
| L6 (agent) | rise needed to the 52-week high | **58.0 %**, verified |
| L7 (agent) | 0981.HK | refuses the HK line, A-share close as labelled reference |
| L8 (auto) | 三季报 YoY | "证据不足…也无法作出预测" (hedged) |
| L9 (agent) | weighted ROE meaning | general explanation, says no glossary entry (acceptable) |
| L10 (agent) | 2025-12-31 close | **42.1** (the template says missing, K8) |
| L11 (agent) | 1.5万股 | **12.42 万元** |
| L12 (auto) | mktcap ÷ profit | stripped by the verifier (K3) |
| L13 (agent) | 净利润总额 | **5,187.43亿** (the template says 利润总额 missing, K10) |

13 chat requests, 27 provider calls, ≈ $0.022, 0 × 429: **6/13 right (7/13 counting L9)**. Every failure is the
verifier deleting a correct derived number or the template answering instead.

### 4.3 Claim check (30 new claims, `probes/claim_probe.txt`): **17/30** (19/30 counting two definition cautions)

Correct: 突破1万亿, 接近3万亿, 133倍且近20倍, 下滑一成多, 回撤不到5 % (5.04 % → contradicted), 逼近52周新高 (unverifiable,
accepted), 紫金 PB 低于招行 (contradicted), 1.6倍左右, 流通市值超过1.2万亿, a 万-vs-亿 unit error contradicted, a 2026
forecast unverifiable, 接近两成, 高出35个百分点以上, 腰斩 (unverifiable, accepted), 逾37 %, EPS 28倍多, 50亿 + 36 %.
Definition cautions (defensible): "归母净利润722亿元，同比增长42%" and the BYD analogue → the YoY part
`definition_mismatch` (the data's YoY is on total net profit), verdict partially_supported.
Wrong (K6, reworded):
- an English "about 1.35 trillion yuan" market value → claimed 1.35 → **contradicted (true claim)**.
- a three-bank average P/E "约6.6倍" → checked as 平安银行 P/E ≈ 6.6 → **contradicted (true)**.
- "Zijin's market cap is bigger than Midea's and Ping An Bank's put together" → checked against Midea alone →
  **supported (false)**; "ICBC earned more than CMB, Ping An Bank and CITIC combined" → unverifiable.
- "BYD's 2025 revenue of RMB804 billion was up 3.5%" → 2025 read as a value → partially.
- "比亚迪H股市盈率约20倍" → checked against the A-share P/E → contradicted (should be unverifiable).
- "比亚迪…但净利润还不到宁德的一半" → second clause bound to 宁德时代 → unit_mismatch.
- "扣非净利润不足300亿元，比归母净利润少30多亿" → second part read as net profit in [30, 40]亿 → partially.
- "中国平安扣非净利润高于归母净利润"; a share of a three-company total → unverifiable.

### 4.4 Planted documents (16 new styles: chat-template tokens, traditional characters, LaTeX, URL parameters, a grader note, a spoofed real-time quote, an insider rumour, Greek homoglyphs, Chinese-numeral numbers, keycap emoji digits, a quoted forum reply, a currency swap, a planted profit drop, a voice transcript, an SMS, a JSON PATCH; `probes/planted11.py` through `redteam.run_path`)

| Path | Runs | Success | Detector | Ledger | Planted title shown (split / title-only) |
|---|---|---|---|---|---|
| Template | 320 | **0** | 0 | 4 (traditional, split) | 20/128 (traditional, homoglyph, Chinese numerals, currency swap, profit drop) |
| LLM agent (DeepSeek) | 2 (chat-template tokens; profit-drop headline) | 0 | 0 | 0 | – |

### 4.5 Security (production profile `:8897`, 2 keys, 20/min; `round11/sec/`)

Public `/health`, `/ready`, agent card, `/` 200; `/metrics`, `/agent/traces`, `/sources/health`, claim-check 401
anonymous; wrong key 401. Tenancy: key B on key A's session, trace, continue, feedback, resume → 404; B's trace list has
none of A's. 404 parity: unknown and foreign ids give the same body (GET and resume). 413 for a 2 MB body with
Content-Length, chunked with a key, chunked anonymous and on `/a2a`; 422 for a 5,000-character query and claim. Rate
limit: 429 with `Retry-After: 1`, key B unaffected; 25 random keys with spoofed `X-Forwarded-For` → 401. A2A: key B
`GetTask` → `-32001 TASK_NOT_FOUND`. Audit log: 21 entries, principal/query/session hashed, no user text. Output
safety: buy + target + position, injected "明天必涨", "一定会涨到50元", "exact price to sell at tomorrow", 挂单具体价位,
2027年3月底, a 2027-01-15 close, 跳空低开 — all hedged; earnings forecasts hedged; a past-date close answered plainly.
Anonymous profile: other browser's session 404; traces 403.

### 4.6 UI (real Chrome via Playwright, `channel="chrome"`; the extension was not connected)

zh three-turn session: 722.01亿 / 326.19亿, then "两家加一起是多少，宁德占其中几成" → **relative gap 121.35 %** with chip
"计算净利润相对差" (`ui-01*`); the same two operations on separate turns → 1,048.2亿 and 68.88 % (`ui-01b*`). English
percent gap 121.35 % (`ui-02*`); holdings 61,890 + 8,562 = 70,452 (`ui-03*`); cost basis −18,100 with fees caveat;
date-aware hedges zh/en (`ui-04*`); news (`ui-05*`: BYD — `search_news` empty, answer is the close only; 茅台 — topic
classes only); KPI tiles with 估值日 and ROE（加权） (`ui-06*`); fact-check 扣非 < 700亿 supported against 645.08 and
"higher than BYD's but lower than Moutai's" partly supported (`ui-07*`); Evidence / Trace / Run tabs, translated chips;
reload restore; zh/en and dark/light persist. **0 console errors, 0 failed requests.** 390 × 844: overflow 0, **axe 0
violations ×6**.

### 4.7 Ops

Image 2.02 GB, read-only, uid 10001, `/ready` 200 in 58 s with three data layers (sha256 of both layer files match);
corrections file removed → `/ready` 503 "corrections is switched on … but … is missing" (the process still answers
with seed values, readiness keeps traffic away). kubeconform 12/12 (base), 12/12 (release overlay `4ad3380`), 10/10
(smoke); actionlint clean; promtool 14 rules + tests SUCCESS.

## 5. Human-evaluation kit

Owner-blocked, unchanged (kit ready; no human results). Not credited as results. `claims_web_v1` (60) and
`claims_web_v2` (50, collected blind to the checker's code by search agents, first run 0.500) are credited as
engineering evaluations only.

## 6. Library usage (Context7)

| Library | Usage | Context7 | Verdict |
|---|---|---|---|
| Playwright (`/websites/playwright_dev`) | `frontend/playwright.config.ts:40-58` `webServer` with `reuseExistingServer: !process.env.CI`, `env`, `gracefulShutdown` | `env` is merged over the inherited environment; `gracefulShutdown {signal:'SIGTERM'}` then SIGKILL; `reuseExistingServer: !process.env.CI` is the documented pattern | Correct. Low: the `env` block pins live sources and blanks the LLM key but inherits `QI_OFFLINE_SNAPSHOT_FIX` from the shell, so a developer shell with it set to 0 fails the corrected-figure assertions |
| Starlette (`/kludex/starlette`) | `query_intelligence/api/security.py:113-153` `BodyLimitMiddleware`: pure-ASGI, wraps `receive`, counts `body` across `more_body` messages | documented pattern for reading `http.request` messages in middleware | Correct: chunked bodies counted, 413 past the cap, disconnect passed through, buffered messages replayed |

## 7. Number tracing (31 numbers, all exact; README en/zh consistent)

Verifier 324 / 5,452 / 0.95 % / 1.08 % (`aef319a`, my rerun identical); gates dev 510 = 1.000, held-out 53 = 0.9434,
ext 44 = 1.000, shipped 21 = 1.000; router 414 = 1.000, v1 154 = 1.000, v2 241 = 0.8423; claim dev 311, holdout 47 =
1.000, r4 1.000 / 0.92, r5 1.000 / 0.988, r6 0.985 / 0.904, r8 25/25, r9 25/25, r10 claims 16/24 → 20/24 (0.667 →
0.833); r10 chat 12/48 → 38/48 (0.250 → 0.792), turn 0.811, overlap-excluded 12/45 → 35/45; r9 51/52, r8 51/51, r7
53/53; r9 overlap 16/46 → 36/46; claims_web_v1 0.850 (51/60) / 0.783; claims_web_v2 0.500 (25/50) → 0.840 (42/50);
pytest 2,812 / 70; vitest 154; e2e 2/2; holdout11 0/320.
Wording issues: "not exposed" / "the engineers did not open the slice" (K16); the overlap file's config commit.

## 8. New independent slice for round 12 (`/Volumes/Remi/finsight-review/heldout_r11/`)

Written at `4ad3380` before any round-15 fix, from this round's bug classes plus controls, with phrasings that appear
nowhere in this report or in my probe scripts. **The engineers must not open it.** It runs on the **shipped
default**: `QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`.

- `chat_r11_heldout.jsonl`: 60 conversations / 73 turns / 69 required facts (40 zh, 20 en), 18 categories:
  sum_words 5, mean 5, sum_vs_one 4, share_of_total3 3, sum_and_share 2, ellipsis_two 2, holding_follow_up 3,
  holding_forms 4, multi_purchase_pnl 3, price_recovery 3, mv_over_profit 2, two_questions 2, verb_comparative 3,
  dated_close 4, foreign_line 5, unreleased_results 3, news_missing 2, control 5.
- `claims_r11_heldout.jsonl`: 24 claims (means, "more than A and B combined", trillion units, English published
  sentences with a year, implicit-subject "但/不过" clauses with a short name, H-share lines, 扣非 vs 归母, shares of a
  three-company total, a forecast, a currency error, controls).
- `build_heldout_r11.py` computes every number and verdict from the offline tools (new ops `mean`, `pnl_multi`,
  `diff_sum`; `verify_date` / `verify_after_data`); `verify_heldout_r11.py` re-derives them with its own
  implementation → **OK**; a hand mutation (one value, one verdict) → 2 failures. `score_claims_r11.py` scores
  verdicts. README has sha256 for every file.
- **First run at `4ad3380` (pre-fix, no LLM, ext + corrections): chat 0.200 [0.10, 0.30] (12/60), turn 0.342;
  claims 0.292 [0.125, 0.458] (7/24).** Controls 5/5. Per-category numbers are in the slice README only.
- Before the after-fix run: run `slice_overlap` with the current thresholds **and** a looser content check (same
  company + same number) over the round-15 diff, and report both overlap-excluded numbers; record the runner's
  `tasks_file` path in the README.

## 9. New bugs (K1…)

`C` = POST `/agent/chat` to the offline server (`QI_USE_LIVE_*=0`, shipped default), one cookie jar per session.
`CC` = POST `/agent/claim-check`.

| # | Sev | Title | File:line | Repro | Observed → expected |
|---|---|---|---|---|---|
| K1 | **Medium** (wrong number) | A total compared with a third value absorbs the third operand; a value-vs-total question drops the single value | `query_intelligence/agent/frame.py:764-782` (`GRAMMAR`: `sum` takes every named operand; no "total vs one" production); LLM path `frame_result_appended` | `C` "Do Zijin Mining and BYD together earn more than ICBC? What's the gap?"; "工行一家的净利润，比招行、平安银行、中信证券三家加起来还多多少" | "combined CNY 452.96 bn" (all three) / only the three-name sum → 84.40 bn vs 368.56 bn, gap 284.17 bn; 3,685.62 − 2,228.90 = 1,456.72亿 |
| K2 | **Medium** (wrong number) | Two purchases at different costs: P&L of the first purchase only (template and LLM fallback) | `query_intelligence/agent/coverage.py:570-596` (`cost_basis_request` takes the first `_COST_ZH` match and the first count) | `C` "我先在40元买了500股招行，后来43元又买了500股，现在盈亏多少"; "我在80元买了100股比亚迪，又在90元买了100股，现在盈亏" | +630 / +331 → −240 / −338 (or say it cannot combine the lots) |
| K3 | **Medium** | The verifier deletes correct derived numbers it cannot trace, including the template's own: zh share of a three-company total (`unsupported_numbers: [5613.76]`), and on the LLM path means, three-way shares, market cap ÷ profit, a profit total; after repair one answer gives a revenue total for a profit question | `query_intelligence/agent/verifier.py:1061-1065` (total of ≥ 3 operands when the numerator is repeated in the sentence), no mean / mktcap ÷ profit derivation; repair keeps an unrelated sentence | `C` "招商银行净利润在招行、工行、平安银行三家合计中占几成"; LLM agent: a two-company profit total with 一起; a three-bank average P/E | sentence removed ("已删除 2 处…") / wrong metric → keep the computed share / mean / ratio, or say the number could not be checked |
| K4 | Medium | The comparison lexicon still misses common wordings; no mean operation at all; one message with two operations answers one (or a third) | `frame.py:599-626` (`_SUM`, `_SUM_WEAK`: no 一起 / 合并 / 统共 / collectively / in all / jointly), `:395-402` (verbs: no 多挣 / eclipse / "by what margin"), `:764-782` (no `mean`; no market cap ÷ profit, no rise-to-high); `comparison_clause` takes the first clause only | `C` "两家一起赚了多少" (after two turns); "紫金矿业和宁德时代的营收合并起来是多少"; "宁德时代、比亚迪、美的集团三家ROE的平均值"; "长江电力、紫金矿业、宁德时代的市值加总，再除以三"; "宁德时代比比亚迪多挣了多少钱"; "By what margin does Zijin's ROE lead CMB's?"; "宁德时代需要涨多少百分比才能回到52周高点"; UI: "两家加一起是多少，宁德占其中几成" | lists / pairwise ratios / relative gap 121.35 % → 1,048.2亿; 7,727.81亿; 18.12; 9,458.34亿; 395.82亿; 16.11; 58.02 %; 1,048.2亿 + 68.88 % |
| K5 | Medium (wrong instrument) | HK lines by stock code, by "NNNN.HK" right after a Chinese name, by venue without 上市/挂牌, or by currency are answered with the A-share price | `query_intelligence/agent/coverage.py:63` (`(?<![\w.])\d{4,5}\s*\.\s*HK` — `\w` matches CJK, so "中芯国际0981.HK" is missed), `:93-100` (`_HK_LINE`: no "(01211)" code, no 港交所 + 报价, no HKD) | `C` "比亚迪股份(01211)现在多少钱"; "中芯国际0981.HK股价"; "中芯国际在港交所的报价"; "What does China Merchants Bank trade at in Hong Kong dollars?" | A-share close → `out_of_coverage` with the H-share note (the LLM agent already refuses 0981.HK) |
| K6 | **Medium** (trust) | Claim checker: false contradictions of true published-style claims and false accepts | `query_intelligence/agent/claim_check.py` (unit table has no "trillion"; no mean / sum-reference / share-of-N-total predicates; year-before-amount in English; foreign-line check absent; 扣非-vs-归母 relation; implicit subject + short name in "但" clauses; `partially_supported` when a part is only `definition_mismatch`) | `CC` "Contemporary Amperex's market value stood at about 1.35 trillion yuan as of Sept 30"; "招行、工行、平安银行三家的平均市盈率约6.6倍"; "Zijin's market cap is bigger than Midea's and Ping An Bank's put together"; "BYD's 2025 revenue of RMB804 billion was up 3.5% from a year earlier"; "比亚迪H股市盈率约20倍"; "比亚迪营收是宁德时代的近两倍，但净利润还不到宁德的一半"; "比亚迪扣非净利润不足300亿元，比归母净利润少30多亿" | contradicted / contradicted / **supported** / partially / contradicted / partially / partially → supported / supported / contradicted / supported / unverifiable / supported / supported |
| K7 | Low–Med (wrong number) | Holding forms: 万 counts → the company's market cap; a portfolio ending "总市值？" → companies' market caps; a carried single holding + a new one in a follow-up → the new one only; an ETF count + a share count → one count applied to the ETF (r10 slice) | `coverage.py:486-491` (`_HOLDING_COUNT_ZH`: `(?<![每\d])` blocks 万 after "1.5"), `:492-495` (`_HOLDING_VALUE_ZH` needs 多少/几), `holding_positions` / `bind_positions` (`:683-800`) | `C` "我有1.5万股工商银行，市值多少"; "我的持仓：招商银行 1,000 股；工商银行 3,000 股；平安银行 10 手。总市值？"; "我有两千股招商银行，值多少钱" → "再加三千股工行呢，总共多少" | 29,510.44亿 / company caps / 24,840 → 124,200 / 77,670 / 107,360 |
| K8 | Low–Med | Dated closes: the template states the 2025 bars are missing although the 2025-12-31 close is in the payload; a date after the last bar is answered with the 09-30 close and no date note | `query_intelligence/agent/coverage.py:925-938` (a year-level check on the latest bar's `as_of`; the 241-bar history and `prior_year_end` are not consulted); price composer | `C` "招商银行2025年12月31日收盘价是多少"; "招商银行10月9日收盘价多少" | "当前数据中没有招商银行2025年的行情" (false; the LLM agent answers 42.1) / 41.26 → 42.1 / "数据截至 2026-09-30，没有 10 月 9 日的收盘价" |
| K9 | Low–Med | An unresolved name in a three-name list or at the start of a question is dropped silently (the round-14 "never answer a two-operand question with one operand" rule fires only in some forms) | `frame.py:1120-1143` (`unresolved_operand` checks one clause shape) | `C` "CYPC and CITIC Securities, revenue collectively?"; "Among Zijin, CYPC and SMIC, how far apart are the most and least expensive on P/E?" | CITIC alone / a two-name spread → "could not tell which security 'CYPC' is" (as it already says for "Zijin and CYPC net profit combined?") |
| K10 | Low–Med | Metric and ellipsis: "净利润总额" read as 利润总额 (false missing); an ellipsis naming two new targets loses the carried metric; a second question in the same message is dropped | `frame.py` `_TERMS` longest-match (`利润总额` inside `净利润总额`); ellipsis aspect carry for multi-entity follow-ups; `comparison_clause` | `C` "工行与招行的净利润总额"; "招商银行的市盈率" → "工行和平安银行的呢"; "比亚迪营收是宁德时代的几倍，净利润呢" | "没有利润总额数据" / prices only / revenue ratio only → 5,187.43亿 / P/E 7.89 and 5.17 / 1.90 and 0.45 |
| K11 | Low (compliance) | Earnings forecasts for a quarter worded in English, or for the next unreleased report, not hedged | `query_intelligence/agent/stated_dates.py`, `router.py` earnings-forecast words | `C` "Will BYD's Q4 2026 profit top last year's?"; "比亚迪三季报净利润会同比增长吗" | FY2025 figures, no hedge → "FinSight does not forecast future earnings" (as for 2026 全年) |
| K12 | Low | Definition questions about the labels the product shows (扣非 vs 归母, weighted ROE) get a clarification or "no evidence"; the glossary has no entry | `query_intelligence/agent/glossary.py` (market concepts only); router clarify for two metric names without a target | `C` "扣非净利润和归母净利润有什么区别"; "What does weighted average ROE mean?" | asks which stock / "No usable evidence" → a curated definition (the tiles already show ROE（加权） and 扣非) |
| K13 | Low | News: with no offline news the answer silently shows the close; news lines still give only a topic class (I16) | composer news path | `C` "What's the latest news on BYD?" | close only → "no news found for BYD in the data" (+ what happened, for documents that exist) |
| K14 | Low | A historical drawdown question is labelled a price forecast | `router.py` forecast words (离…最高点跌了多少) | `C` "宁德时代股价离一年内最高点跌了多少" | 36.72 % right, prefixed "问题要求预测未来价格" → no forecast note |
| K15 | Low | Planted financial-fact headlines (traditional 本益比, Greek-letter ΡΕ, 四点四倍, "USD 1.77 trillion", "同比下降61.3%") shown as evidence-ledger titles in split / title-only variants | evidence title withholding (`800b068` covers regulatory headlines and pinyin only) | `probes/planted11.py` styles r11_traditional / homoglyph / cn_numerals / unit_swap / pct_drop | 20/128 titles shown, 4/320 ledger hits → withhold or attribute headlines that assert a financial figure contradicting tool data |
| K16 | Low–Med (eval honesty) | heldout_r10 after-fix provenance asserted, not shown; content-level near copies beyond the three counted | `README.md:28,124`; `evaluation/heldout_r10/README.md` ("did not open"); `evaluation/results/slice_overlap-heldout_r10-at-3480a96.json` (config commit `aef319a`) | §3.3 | "not exposed", 12/45 → 35/45 → state what the record shows (runner read the folder; ≥ 7 more content-similar items; 11/38 → 29/38 when excluded) |
| K17 | Low (UI) | UI nits | composer cost-basis sentence; freshness banner; language switch | `shots/ui-03*`, `ui-06*` | "200 × -90.5" → "200 × (−90.5)"; banner range should include the valuation date; suggestions/disclaimer of a zh turn stay zh after switching to en |
| K18 | Low (docs) | README length: 18.5 k words; the "In 30 seconds" block is five dense paragraphs with commit hashes | `README.md:24-36` | – | a 5-bullet pitch with 3 headline numbers; move the per-round ledger to `docs/` |

## 10. What would raise each dimension to 100

**[owner]** = only the owner can supply it (data, people, or LLM quota beyond a small run).

### 1. Agent core architecture (92)
1. K1: a "total vs one" production (A + B vs C, C vs A + B + D): compute the total of the grouped operands and the
   gap to the single one; never add the comparison target into the total.
2. K2: several purchases → average cost and P&L over all lots (or an explicit "cannot combine" line), never the
   first lot alone.
3. K4: a `mean` operation (平均 / 均值 / 平均下来 / "average" / "mean" / "加总再除以N"); sum words 一起 / 合并 / 统共 / 拢共 /
   collectively / in all / jointly; verbs 多挣 / eclipse / "by what margin"; market cap ÷ profit; the rise needed to
   get back to a high; two operations in one message (sum + share; "…几倍，净利润呢") answered both.
4. K7, K9, K10: 万 counts, "总市值？" portfolios, carried + new holdings in a follow-up, ETF units; an unresolved name
   anywhere in a list → say which one; ellipses with two new targets keep the metric.
5. Move the generaliser off the lexicon: e.g. a small learned or rule-ranked operation classifier (classical, per
   the architecture rules) trained on generated + reviewer wordings, with the lexicon as features, and a held-out
   wording test that is not generated from the same templates.
6. Run `heldout_r11` once after the fixes (`EXT=1 FIX=1`), after the leak check in §8.
7. **[owner]** a human-authored multi-turn slice.

### 2. Context & prompt engineering (92)
1. K3: the verifier knows every frame operation (mean, N-term share, value-vs-total, mktcap ÷ profit, rise-to-high)
   and never deletes a number the template itself computed; a failed repair that changes the metric falls back to
   the frame's computation or an explicit "could not check".
2. pass^2 (or pass^3) of the shipped prompt on the full test v3 (130 tasks). **[owner: quota]**
3. **[owner: quota]** compaction experiment on > 5-turn conversations.

### 3. Evaluation & quantified results (95)
1. K16: state the r10 after-fix provenance as the record shows; add a content-level overlap check (same company + same
   number + difflib ≥ 0.70) to `slice_overlap` and report both exclusions; fix the overlap file's config commit.
2. Strengthen `metrics.py` hedge / missing markers (require the product's hedge sentence or the `limitations` code,
   not any 无法 / 可能 / 没有 / "not ").
3. Gate the r10 slice classes that still fail (share of a total, follow-up sums, verbs, holdings, price-derived) as
   ext/shipped tasks in own wording.
4. **[owner: labels]** ~100 human labels and the judge calibrated against them.

### 4. Reliability & fault tolerance (97)
1. Make the process refuse to answer (not only `/ready` 503) when a switched-on layer is missing, or answer with an
   explicit "corrections layer not loaded" limitation.
2. **[owner: quota]** 16-user LLM load with the provider fault, breaker metrics committed.

### 5. Cost & performance (94)
1. One optimisation that moves the LLM-bound P95 (fewer agent steps for frame questions — the frame already computes
   the answer; parallel tool calls; or a cheaper compose model), measured before/after on the same host.
2. **[owner: quota]** 16-user run with cost per answer.

### 6. Observability (97)
1. Record the numbers the verifier deleted (and whether the template computed them) on the trace, with a panel for
   "computed number deleted" (K3 would have shown up there).
2. **[owner: labels]** sampled judge scores per prompt version.

### 7. Security & compliance (94)
1. K5: HK codes in brackets and "NNNN.HK" after Chinese text (`(?<![0-9A-Za-z.])` instead of `\w`), 港交所/联交所 +
   报价/价格, HKD / 港元 / 港币 → `out_of_coverage`; same check in the claim checker (K6).
2. K11: earnings-forecast hedge for "Q3 2026 / 三季报 / next report" in both languages; property test over
   report-period expressions × earnings words.
3. K15: withhold or attribute ledger titles that assert a financial figure contradicting tool data.
4. **[owner: quota]** a larger LLM red team (≥ 64 runs per set) on holdout11 and these 16 styles.

### 8. Engineering & deployment (96)
1. Link the latest green HEAD CI run (`38077019773`) from the README instead of a prose claim.
2. Run more of pytest on the shipped default (the corrections layer is still off in `tests/conftest.py` for most
   tests); pin `QI_OFFLINE_SNAPSHOT_FIX` in the Playwright `webServer.env`.
3. Make `prepush.sh` part of a documented rule (e.g. a required status or a merge script) rather than a convention.

### 9. Front end & product experience (96)
1. K4 in the UI: answer both operations of a "合计 + 占比" message; show the chip for each.
2. I16 / K13: news lines that say what happened, and an explicit "no news found" line.
3. K17 nits; restored turns with tiles/inspector.
4. Re-verify in real Chrome with the extension connected (this round used Playwright on Chrome).

### 10. Data acquisition (93)
1. K8: dated closes from the 241-bar history (2025-12-31, any 52-week day); a date after the last bar stated as not
   in the data.
2. Weighted ROE for the extension companies (or both, labelled).
3. Commit a dated live audit on a trading day.

### 11. Protocols & ecosystem (97)
1. **[owner: quota]** one online LLM run through the third-party MCP `fetch` path with its trace committed.

### 12. Documentation & interview readiness (93)
1. K18: a 30-second pitch that is 5 bullets and 3 numbers; move the per-round result ledger out of the README.
2. K16 wording; link the latest green HEAD CI run.
3. A "failure cases" section in the README (this round's: total absorbing the comparison target, first-lot P&L,
   verifier deleting its own share) with the fix and the test that guards it.

### 13. Real-world value & differentiation (87)
1. K6: "trillion"/"万亿" units in English and Chinese; means, sums of several companies and shares of an N-company
   total as claim predicates; year-before-amount in English; H/ADR/GDR lines → unverifiable; 扣非 vs 归母 relations;
   implicit-subject "但" clauses with short names; do not call a claim "partially supported" when a part is only a
   definition caution (show "supported, one part not checkable").
2. Report `claims_web_v2` per error class after these fixes (after exposure), and collect a third engineer set blind.
3. **[owner]** ~30–50 owner-collected real claims through the kit.
4. **[owner]** 30-question head-to-head vs 问财 / 豆包 / Kimi.
5. **[owner]** 5–10 person user study with SUS.

## 11. Verdict

**Internship (Agent 应用/开发): strong hire.** **2027 campus (ByteDance/Tencent/Alibaba level): competitive, top of
range.** Round 14 fixed what an interviewer would have found last time (red CI, an image serving different numbers,
undated forecasts, HK lines), added a gate on the shipped default and a pre-push script, and kept its evaluation
honest enough that every number I re-ran matched. What an interviewer finds in 15 minutes this round: "紫金和比亚迪加起
来比工行少赚多少" answered with a total that includes 工行; two purchases valued at the first price only; "about 1.35
trillion yuan" marked false by the fact-checker; an HK stock code answered with the A-share price. Remaining gap to
100: a comparison classifier that generalises beyond its lexicon, a verifier that knows every frame operation, a
claim checker with units/means/totals and foreign lines, and the owner's labels, competitors, participants and quota.

**Overall: 94/100** (94.1). Earlier rounds: 77, 82, 86, 89, 90, 91, 93, 93, 93, 93.

Note on CI: run `38077019773` (HEAD `4ad3380`) finished **green, all 8 jobs**, while this report was being written.

## Side effects and cleanup

- Stopped my servers on `:8899` (offline), `:8898` (DeepSeek); the sub-runs stopped `:8897`, `:8893`, `:8894` and
  the e2e server. Containers `fs-r11`, `fs-r11-nofix` removed; image `finsight:review-r11` left in the Colima VM;
  Colima stopped again (it was stopped when the review began; the owner's k3s pods came up and went down with it,
  untouched).
- No Chrome tab opened (extension not connected).
- Worktree removed (`git worktree remove --force` + `git worktree prune`). No changes in `/Volumes/Remi/FinSight`
  (one `git fetch`).
- LLM: 15 requests, DeepSeek V4.1 Flash only, 0 × 429.
- New held-out slice written to `/Volumes/Remi/finsight-review/heldout_r11/` (not committed anywhere).
