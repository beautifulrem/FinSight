# Known Issues & Resolved Audit Summary

Languages: English | [中文](zh/known-issues.md) · [Evaluation History](evaluation-history.md) · [Independent Reviews](reviews/README.md)

This page tracks **currently open limitations** alongside a concise summary of resolved engineering reviews and security audits. Detailed verbatim transcripts of all 14 external review rounds live in [`docs/reviews/`](reviews/README.md).

## Open Limitations

| Area | Open Limitation & Impact |
|---|---|
| **LLM prompt-injection residual** | While the deterministic template path allows **0 / 3,796** successes across 16 attack sets (`redteam-offline-r18.json`), LLM paths can still mention planted narratives (0–3.4% stated as fact in the last full run `redteam-r8-llm.json`; 0/16 agent and 0/13 composition on the latest targeted slice `redteam-r16-holdout13-llm.json`). Output safety attributes recognized figures/events, which still relays marked values to the reader. |
| **Lexical generalization limits** | Comparison grammars, session coreference, lookalike continuation lists, and `claim-check` rules still rely on lexicons/regexes. Novel phrasings outside trained classes still miss (e.g., 13/50 chat tasks on the round-14 slice; English multi-company exclusions such as `"all but X"`). |
| **Blind real-world claim checks** | First-run accuracy on blind published web claims ranges from **0.500 to 0.740** (`claims_web_v2-first-run.json` … `claims_web_v6-first-run.json`). Remaining gaps include implicit publication dates, cross-clause metric bindings, non-standard definitions, and prior-year ratio changes absent from snapshots. |
| **Offline snapshot & alias coverage** | Shipped offline snapshots cover 22 priced instruments and 13 companies with fundamentals (dated `2026-04-22` / `2026-09-30`, unadjusted closes). Industry valuation tiles cover v1 industries only; `data/runtime/dual_listings.csv` covers 100 A+H pairs (88 additional A+H names rely on generic H-share rules); ~1,038 fund-code rows in `data/runtime/alias_table.csv` still share 6-digit codes with equities pending a training-asset rebuild. |
| **Live data constraints** | Free public endpoints throttle bursts; live price fetches pull the latest 30 daily bars (so YTD / 52-week / drawdown calculations require offline extension or deeper history); intraday quotes reject stale movable-holiday quotes by falling back to the daily close. |
| **Slow reasoning models** | GLM-5.3 Flash exhibited `82 s` agent P95 on Test v3 (`ablation-final4-glm-testv3.json`); mitigated in `mode=auto` via `QI_AGENT_SLOW_MODEL_POLICY=on`, which routes slow models through workflow + LLM composition. |
| **Human evaluations not yet run** | Human answer grading, 问财/豆包/Kimi head-to-head comparison, user study, and owner-collected real claims require external accounts/annotators; scripts are ready in [`evaluation/human/README.md`](../evaluation/human/README.md). |
| **Historical key in git history (`302077a`)** | Revoked at the provider (owner confirmed `2026-09-30`); history intentionally not rewritten and narrowly allowlisted in `.gitleaks.toml` ([SECURITY.md](../SECURITY.md)). |

## Resolved Audit & Review Milestones

- **Full-Repository P1/P2 Security & Correctness Audit ([PR #13](https://github.com/beautifulrem/FinSight/pull/13), Issue #7)**: Fixed 14 P1/P2 findings (`tests/test_audit_issue7_fixes.py`) covering explicit `.SH`/`.SZ`/`.BJ` suffix preservation & index routing, A2A `contextId` tenant anti-enumeration & `MAX_QUERY_LENGTH`, authenticated HTTP MCP with fail-closed non-loopback guards, credential/DSN redaction in logs/provenance/SSE errors, loopback-bound Compose ports without anonymous Grafana Admin, ReDoS hardening, Postgres connection pooling, `NaN`/`Inf` sanitization, legacy `/chat` OOS/compliance guards, pre-extraction archive validation, serialized V8 `MiniRacer` execution, and SSE timeout generator-close race handling.
- **14 Rounds of Independent Reviews (`docs/reviews/round1.md`–`round14.md`)**: Raised independent rubric scores from `77` (lowest dimension `68`) in round 1 to `94.5` (every dimension `≥ 90`) in round 14—delivering sentence-level claim verification (`0.92%` strict false-accept), primary-source FY2025 snapshot corrections & accounting definitions, server-issued anti-enumeration sessions, forecast/foreign-listing/lookalike/position-sizing guards, HK/TW terminology & full-width normalization, planner prefetch latency cuts (`P95 27.1 s → 15.4 s`), and CI time-budget enforcement.
