# LLM Response Handoff

Languages: English | [中文](zh/llm-response.md)

`scripts/llm_response.py` is a downstream FinSight utility that produces frontend-ready JSON from compact Query Intelligence evidence. It can consume an existing record or run Query Intelligence first from a raw query.

It is outside the NLU/Retrieval backbone. It must not re-infer intent, targets, or source plans.

This page documents the legacy local HuggingFace/transformers handoff path in the current checkout. The browser chatbot and DeepSeek v4 API path are documented separately in [Local Frontend Chatbot](frontend-chatbot.md).

## Outputs

The script generates:

- `answer_generation`: answer text, key points, evidence IDs used, limitations, risk disclaimer, and model name.
- `next_question_prediction`: exactly three follow-up questions with scores and short reasons.

Both outputs are validated and normalized as JSON.

## Default Backend

Current script defaults:

| Setting | Default |
|---|---|
| Backend | local HuggingFace/transformers |
| Answer model | `instruction-pretrain/finance-Llama3-8B` |
| Next-question model | `Qwen/Qwen2.5-3B-Instruct` |
| Model search root | `models/llm`, then `LLM_MODELS_DIR` / `QI_LLM_MODELS_DIR` |
| Max answer tokens | `700` |
| Max next-question tokens | `260` |
| JSON repair retries | `1` |

Quick run:

```bash
python scripts/llm_response.py --query "What do you think about Ping An Insurance (601318.SH)?"
```

Use an existing combined record:

```bash
python scripts/llm_response.py --input path/to/pipeline_record.json
```

`--input` expects one JSON object with `query`, `nlu_result`, and `retrieval_result`; `statistical_result` and `sentiment_result` are optional. Query Intelligence artifact runs write separate files, so either combine them first or use `--query`.

## Safeguards

- Uses compact evidence only: query, NLU result, retrieval evidence summary, optional statistical result, and optional sentiment result.
- Strips debug traces, long raw documents, history arrays, and source-generation metadata before model input.
- Enables JSON Output by default for OpenAI-compatible APIs.
- Validates and repairs JSON when possible.
- Must not invent missing market data, fundamentals, valuation, macro data, news, sentiment, or statistical facts.
- Must include a risk disclaimer.
- Must not output chain-of-thought or markdown fences.

## Configuration

Common CLI options:

| Option | Purpose |
|---|---|
| `--answer-model` | Model for `answer_generation`. |
| `--next-question-model` | Model for follow-up questions. |
| `--models-dir` | Local model directory checked before HuggingFace model IDs. |
| `--few-shot-source` | Few-shot JSONL source used by answer and next-question prompts. |
| `--answer-max-new-tokens` | Generation limit for answer JSON. |
| `--next-max-new-tokens` | Generation limit for next-question JSON. |
| `--device-map` | Device placement passed to transformers. |
| `--dtype` | `auto`, `float16`, `bfloat16`, or `float32`. |
| `--temperature` | Sampling temperature. |
| `--json-retries` | Strict JSON retry count after parse failure. |
| `--trust-remote-code` | Enable audited HuggingFace repository code when required. |

Environment variables:

| Variable | Purpose |
|---|---|
| `LLM_MODELS_DIR` or `QI_LLM_MODELS_DIR` | Local model search root. |
| `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` | Authenticated HuggingFace access. |

Remote repository code is disabled by default. Use `--trust-remote-code` only for audited HuggingFace models that require it.
