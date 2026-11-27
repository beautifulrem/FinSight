# LLM Response Handoff

Languages: English | [中文](zh/llm-response.md)

`scripts/llm_response.py` implements the paper's constrained local-model answer generation and follow-up question prediction stage ([DOI: `10.3233/FAIA261107`](https://ebooks.iospress.nl/doi/10.3233/FAIA261107)). It transforms compact Query Intelligence artifacts into strictly validated JSON without re-inferring intent, entities, or source plans.

> For the production LangGraph agent (`/agent/chat*`) and browser workspace, see [Agent Layer](agent.md) and [Web UI & Chatbot](frontend-chatbot.md).

## Output Contract

Returns two normalized JSON blocks:

- **`answer_generation`**: `answer`, `key_points[]`, `evidence_ids_used[]` (filtered strictly to IDs present in the run), `limitations[]`, `risk_disclaimer`, and `model`.
- **`next_question_prediction`**: three follow-up questions with relevance scores and brief rationales.

## Quick Run

```bash
# Run Query Intelligence first, then generate JSON
python scripts/llm_response.py --query "你觉得中国平安（601318.SH）怎么样？"

# Consume an existing combined record ({query, nlu_result, retrieval_result, ...})
python scripts/llm_response.py --input path/to/pipeline_record.json
```

## Default Local Models & Options

| Setting / Flag | Default | Purpose |
|---|---|---|
| `--answer-model` | `instruction-pretrain/finance-Llama3-8B` | Model for `answer_generation` |
| `--next-question-model` | `Qwen/Qwen2.5-3B-Instruct` | Model for `next_question_prediction` |
| `--models-dir` | `models/llm` (`LLM_MODELS_DIR` / `QI_LLM_MODELS_DIR`) | Local model root checked before HuggingFace Hub |
| `--answer-max-new-tokens` / `--next-max-new-tokens` | `700` / `260` | Generation token caps |
| `--json-retries` | `1` | Repair retries on invalid JSON |
| `--device-map`, `--dtype`, `--temperature` | `auto` | Inference placement and decoding controls |
| `--trust-remote-code` | `false` | Opt-in only for audited HuggingFace repos |

Authenticated downloads read `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`.

## Safeguards

- Feeds compact summaries only—strips debug traces, raw long bodies, and provider dumps before prompting.
- Filters emitted `evidence_ids_used` strictly against the run's actual evidence set and strips chain-of-thought or markdown fences.
- Never invents missing market, fundamental, macro, news, or sentiment facts; always emits a risk disclaimer.
- Verified by `python -m pytest -q tests/test_llm_response.py`.
