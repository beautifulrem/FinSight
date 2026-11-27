# Evaluation History & Methodology

Languages: English | [中文](zh/evaluation-history.md) · [Generated Full Report](agent-eval.md) · [Known Issues](known-issues.md)

This page summarizes FinSight's evaluation methodology, independent vs. after-exposure task sets, verifier and red-team results, full limitations, and CI / human-evaluation workflows. Every cited result file lives under [`evaluation/results/`](../evaluation/results/) and is validated automatically by `python -m evaluation.agent_eval.report --check` against [docs/agent-eval.md](agent-eval.md).

## Evaluation Protocol & Task Sets

Scoring uses strict **dealbreaker** criteria—a task succeeds only if behavior (`answer`/`clarify`/`refuse`), cited required facts, required tools, hedging on judgment queries, and absence of trading instructions all pass. Online runs repeat every task 3 times and report 95% bootstrap CIs plus paired bootstrap / exact McNemar comparisons.

| Set | Size | Authoring & Exposure Status |
|---|---|---|
| **Development (`dev`)** | 558 tasks | Project author; drives regression fixes (`gate-dev.json`: **1.000**) |
| **Held-out validation** | 53 tasks | Project author; used for prompt selection (`gate-holdout.json`: **0.943**) |
| **Extension (`ext`)** | 123 tasks | Shipped snapshot extension on, no replay (`gate-ext.json`: **1.000**) |
| **Shipped default (`shipped`)** | 21 tasks | Extension + primary-source corrections on (`gate-shipped.json`: **1.000**) |
| **Test v3** | 130 tasks / 155 turns | Independent author; untouched on its first LLM run (`9536abf`), later used once to select default prompt `v4` |
| **Test v2 / Multi-turn v1** | 121 tasks / 49 convs | First runs reported separately; **after exposure** once failure classes were addressed |
| **Reviewer held-out slices (r4–r14)** | 38–60 chat tasks + 22–67 claims per round | Written by independent reviewers before each fix round; out-of-sample for one fix round only, with text/content/signature overlap measured (`slice_overlap`) |
| **Blind real published claims (v2–v6)** | 50–60 claims each | Collected blind from public media/notes/filings before reading checker code |
| **Independent router labels (v1 / v2)** | 154 / 241 queries | Separate authors labelling against a written routing rubric |

## Final Online Ablation (`9536abf`, 3 Repeats)

Sources: [`ablation-final4-deepseek-testv3-holdout.json`](agent-eval.md#test-set-v3-independent-author-untouched-first-runs), `ablation-final4-deepseek-testv2-multiturn.json`, `ablation-final4-glm-testv3.json`, `ablation-test_v3-purellm-deepseek.json`.

| Path | Test v3 (DeepSeek, first run) | Test v3 (GLM) | Held-out (DeepSeek) | Test v2 (exposed) | Multi-turn v1 (exposed) |
|---|---|---|---|---|---|
| Legacy `/chat` (no LLM) | 0.008 | 0.008 | 0.208 | 0.223 | 0.000 |
| Pure LLM (no tools) | **0.000** (0/130) | — | — | — | — |
| Deterministic workflow | 0.769 [0.69, 0.84] | 0.769 | 0.943 | 0.901 | 1.000 |
| Workflow + LLM composition | 0.831 [0.76, 0.89] | 0.818 (P95 27 s) | 0.981 | 0.931 | 0.980 |
| **LLM Agent (plan-seeded)** | **0.869 [0.81, 0.92]** (pass^3 0.854) | 0.815 (P95 82 s) | **1.000** | **0.959** | 0.959 |

- **DeepSeek Test v3 paired diffs**: Agent vs workflow **+0.100** `[+0.049, +0.156]` (McNemar `p = 0.007`); Agent vs composition `+0.038` per run (`p = 0.29` on pass^3, not significant).
- **GLM policy**: Because GLM agent and composition tie (`-0.003`) while agent P95 hits `82 s`, `mode=auto` routes slow reasoning models through composition (`QI_AGENT_SLOW_MODEL_POLICY=on`).
- **Prompt `v4` default**: Selected via Test v3 A/B (`ablation-ab-prompt-v3-testv3.json` vs `ablation-ab-prompt-v4-testv3.json`), adding document-safety rules with no statistically significant task-success penalty.

## Independent sets: first runs vs after exposure

Every independent benchmark retains its untouched first-run score alongside any subsequent after-exposure result:

| Independent Set | First Run (Unexposed Estimate) | After Fix / After Exposure |
|---|---|---|
| Test v3 (DeepSeek, 130 tasks) | **0.869** `[0.81, 0.92]` (`ablation-final4-deepseek-testv3-holdout.json`) | Not tuned against |
| Multi-turn v1 (no LLM / DeepSeek) | 0.224 / 0.361 (`multiturn_v1-auto-nollm-first-run.json`, `ablation-multiturn_v1-deepseek-first-run.json`) | 1.000 / 0.959 (after exposure) |
| Router labels v2 (241 queries) | **0.801** (`router_eval-independent_v2-first-run.json`) | 0.842 (after exposure) |
| Reviewer slices (chat, 1 round of fixes) | r6 0.579 · r7 0.264 · r8 0.275 · r9 0.346 · r10 0.250 · r11 0.200 · r12 0.320 · r13 0.413 · **r14 0.160** | r6 0.816 · r7 0.830 · r8 0.686 · r9 0.788 · r10 0.792 · r11 0.733 · r12 0.840 · r13 0.978 · **r14 0.740** (`chat_heldout_r14-auto-nollm-after-fix.json`) |
| Blind real claims (v2 / v3 / v4 / v5 / v6) | **0.500** / **0.540** / **0.740** / **0.680** / **0.650** (`claims_web_v2-first-run.json` … `claims_web_v6-first-run.json`) | 0.840 / 0.880 / 0.780 / 0.680 / 0.650 (round-18 checker) |
| Verifier stress test (333 gold, 5,631 corrupt) | **0.92%** strict false-accept (1.05% derived), 100% true-accept, 0.0% cross-company swap (`verifier_stress-round18.json`) | CI gate baseline |
| Prompt-injection red team (16 attack sets) | Template: **0 / 3,796** successes (`redteam-offline-r18.json`); full LLM run: 0–3.4% stated as fact (`redteam-r8-llm.json`); latest targeted LLM slice (holdout13): 0/16 agent, 0/13 composition (`redteam-r16-holdout13-llm.json`) | Ongoing CI regression suite |
| Fault injection (11 scenarios) | **11/11** graceful degradation (`fault_injection.json`) | — |

**Slice provenance note**: For rounds 8–14, after-fix runs read the reviewer slice from an external directory before the slice entered git. While no fix commit references slice rows, file access is not auditable by git alone; `slice_overlap` therefore reports strict overlap-removed tiers (text, content, and content signature) alongside headline totals.

## Limits in full

- **Model scope**: Online ablations cover two flash-class models (DeepSeek V4.1 Flash and GLM-5.3 Flash) through one OpenAI-compatible gateway.
- **Small unexposed slices**: Reviewer slices have 38–60 chat tasks and remain out-of-sample for one fix round only; lexical comparison grammars still miss novel phrasing outside their classes (e.g., 13/50 failing on round-14 chat after fixes).
- **Routing generalization**: Fresh independent queries still show ~20% routing errors (`0.801` first run), concentrated on targetless advice and borderline clarify-vs-refuse turns.
- **Agent vs. composition margin**: On DeepSeek Test v3, the tool loop leads per run (`+0.038`) but not significantly on `pass^3`; on GLM they tie at higher latency and cost.
- **Traceability ≠ external truth**: The verifier proves every number comes from cited run evidence; a fake price planted inside a news excerpt still traces to that document unless contradicted by structured fundamentals or caught by output safety.
- **Residual LLM injection exposure**: Output safety attributes or removes recognized patterns, but LLM paths can still mention unattributed narrative payloads.
- **Rule-based follow-ups & real-world claim gaps**: Coreference rules and claim comparators rely on structured lexicons and miss arbitrary prose phrasing; offline snapshots cover 22 priced instruments and 13 companies with fundamentals.
- **Owner-dependent human evaluations**: Human answer grading, live competitor head-to-head (问财/豆包/Kimi), user studies, and owner-collected claims are ready in `evaluation/human/` but not yet executed.

## CI, the pre-push check and the human-evaluation kit

- **GitHub Actions CI** (`.github/workflows/ci.yml`): runs `ruff` lint/format, full-history `gitleaks`, frontend typecheck/lint/Vitest/reproducible build check, full `pytest` with Postgres + Playwright E2E + `axe-core` + branch-coverage floors (`agent ≥ 88%`, `api ≥ 90%`, `answer_guards ≥ 81%`), `tests-shipped` (`QI_TEST_SHIPPED_DEFAULT=1`), offline evaluation gates (`dev`/`holdout`/`ext`/`shipped` + claim-check holdout), `verifier_stress` + offline `redteam`, held-out slice verifiers + `report --check`, Docker read-only smoke test, Kubernetes manifest validation, and `kind` cluster smoke test.
- **Pre-push mirror**: [`scripts/prepush.sh`](../scripts/prepush.sh) runs the local CI suite with per-job time-budget guards ([docs/development.md](development.md#ci-time-budgets)).
- **Human evaluation kit**: [`evaluation/human/README.md`](../evaluation/human/README.md) provides frozen templates and SHA-256 verified scoring scripts for answer grading, 30-question competitor head-to-head, real-claim collection, and SUS usability testing.
