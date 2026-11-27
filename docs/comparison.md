# Product Comparison: FinSight vs. 问财, 豆包, Kimi, Wind Alice & 妙想

Languages: English | [中文](zh/comparison.md)

This comparison is based strictly on public vendor documentation and press reports (surveyed September 2026). **None of the commercial products was executed directly**—items lacking public verification are marked **unverified**.

## Summary

Commercial platforms (**同花顺问财**, **Wind Alice**, **东方财富妙想**, **Kimi**, **豆包**) win on licensed real-time market/research databases, breadth (A/HK/US equities, bonds, filings), mobile apps, and production user scale. FinSight does not compete with their data franchises.

FinSight is an **open, self-hostable reference implementation** focused on one measurable property none of those commercial assistants publishes publicly: **every number in every sentence is checked programmatically against the evidence cited in that same sentence**, with published verifier false-accept rates (**0.92%** strict across 5,631 corrupted answers; **100%** true-accept), hard code-enforced compliance removal of trading advice, full execution traces, and native **MCP + A2A 1.0** protocol support.

## Capability Matrix

| Dimension | FinSight | 同花顺问财 (HithinkGPT) | 豆包 (Doubao) | Kimi | Wind Alice | 东方财富妙想 |
|---|---|---|---|---|---|---|
| **Data sources** | Free public endpoints (Eastmoney → Sina → Tencent → cache → snapshot) with per-record provenance; **not licensed** | Licensed exchange/agency DB (15 domains: A/HK/US, funds, macro) | Web search + 豆包工作 finance plugins / 妙想 & 通达信 MCP | Wind, Eastmoney, S&P Global, CLS, Caixin via MCP | Wind institutional terminal DB + AIFin Market MCPs | Eastmoney Choice DB + tiered filings/EDB |
| **Citations** | Mandatory per-sentence `[evidence_id]` tied to an interactive evidence ledger | Dated announcements/reports on result pages | Mixed (unverified in default chat; cell-level in 豆包工作) | Vendor claim of traceable key figures (unverified) | Vendor claim of authoritative source attribution | Vendor claim of traceable evidence per answer |
| **Programmatic numeric verifier & published error rate** | **Yes** — sentence-level unit/precision/sign verification; published **0.92%** strict false-accept, **0%** cross-company swap | No public error rate (unverified) | No public error rate (unverified) | No public error rate (unverified) | No public error rate (unverified) | No public error rate (unverified) |
| **Investment-advice stance** | Unlicensed; code compliance node strips all buy/sell/target/position instructions | Licensed advisory (`ZX0050`); offers investment-advice skills + disclaimer | Historically gave picks in press tests; current stance unverified | Institutional compliance suite (consumer stance unverified) | Advisory/strategy skills + disclaimer | Stock suggestions + disclaimer in press tests |
| **Open protocols** | REST + SSE, **MCP server & client**, **A2A 1.0** | iFinD API/MCP, SkillHub; A2A unverified | Model API, search MCP/Skill; A2A unverified | Model API, inbound MCP; A2A unverified | AIFin Market MCP/Skills; A2A unverified | 妙想 MCP/Skills; A2A unverified |
| **Explainability & self-hosting** | MIT open source, self-hosted (Docker/k8s), logged classical NLU route reasons, offline zero-key mode | Closed SaaS / paid pro tiers | Closed SaaS / Volcano Engine | Closed SaaS / membership & enterprise tiers | Closed SaaS / Wind credit system | Closed SaaS / Choice credit system |

## Where Incumbents Win vs. Where FinSight Differentiates

- **Incumbents win** on licensed real-time multi-market coverage, full Excel/PPT/research-report generation, investment-advisory licensing, proprietary finance-tuned models, and consumer-grade apps.
- **FinSight differentiates** on open, reproducible evaluation (published task sets, bootstrap CIs, red-team suites, and explicit failure cases), code-verified numeric traceability, zero-key deterministic fallback, self-hosted tenant isolation, and full MCP + A2A 1.0 interoperability.

## Planned Head-to-Head Protocol (Not Yet Run)

Because running live commercial products requires paid accounts and human blind annotators, no head-to-head accuracy claim is made. A frozen 30-question Chinese benchmark (10 single facts, 5 comparisons, 5 macro/causal, 5 compliance traps, 3 missing-data, 2 follow-ups) with SHA-256 lock, scoring rubric (citation presence, source faithfulness, ground-truth accuracy, compliance violations, Wilson 95% CIs, Cohen's kappa), and FinSight collection script is ready in [`evaluation/human/README.md`](../evaluation/human/README.md).
