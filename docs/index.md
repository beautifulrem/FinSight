# Documentation

Languages: English | [中文](zh/index.md)

These pages document FinSight, the ARIN7012 Group 4.2 evidence-first financial analysis chatbot project. The technical backend remains ARIN Query Intelligence.

The root README is intentionally short; use these pages when you need contracts, configuration details, training steps, or downstream handoff notes.

## Contents

| Document | Purpose |
|---|---|
| [Query Intelligence](query-intelligence.md) | Scope, architecture, API, NLU and retrieval output contracts, live providers, environment variables, troubleshooting. |
| [Local Frontend Chatbot](frontend-chatbot.md) | Browser UI, `/chat` contract, DeepSeek v4 settings, real local screenshots, and troubleshooting. |
| [Presentation Materials](presentation/README.md) | Clean slide outline and links to current demo screenshots. |
| [Training](training.md) | Public dataset sync, manifest-based training, runtime asset materialization, evaluation, and release checks. |
| [LLM Response](llm-response.md) | Legacy local-transformers answer-generation JSON handoff, output contract, configuration, and safeguards. |
| [Sentiment](sentiment.md) | Downstream document sentiment pipeline, preprocessing, FinBERT routing, output fields, and test commands. |
| [Retrieval Output Spec](retrieval_output_spec.md) | Compatibility entry point for `analysis_summary` and retrieval output references. [中文](zh/retrieval_output_spec.md) |

Chinese documentation is in [docs/zh](zh/index.md).

## Documentation Style

The structure follows the common pattern used by mature open-source projects:

- The root README answers "what is this, how do I run it, where do I go next?"
- `docs/` contains stable reference pages split by topic.
- Commands are copyable from a fresh clone and avoid local-only wrappers.
- Public docs avoid generated output, private tokens, and machine-specific paths.
