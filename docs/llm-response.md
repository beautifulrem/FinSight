# LLM Response Handoff

Languages: English | [中文](zh/llm-response.md)

`scripts/llm_response.py` is a downstream FinSight utility that produces frontend-ready JSON from compact Query Intelligence evidence. It can consume an existing record or run Query Intelligence first from a raw query.

It is outside the NLU/Retrieval backbone. It must not re-infer intent, targets, or source plans.

## Outputs

The script generates:

- `answer_generation`: answer text, key points, evidence IDs used, limitations, risk disclaimer, and model name.
- `next_question_prediction`: exactly three follow-up questions with scores and short reasons.

Both outputs are validated and normalized as JSON.

## Default Backend

Current defaults:

| Setting | Default |
|---|---|
| Backend | `openai-compatible` |
| API base URL | `https://api.deepseek.com` |
| Answer model | `deepseek-v4-flash` |
| Next-question model | `deepseek-v4-flash` |
| Extra body | `{"reasoning_effort":"max"}` |
| JSON response format | enabled with `response_format={"type":"json_object"}` |
| Max answer tokens | `8192` |
| Max next-question tokens | `8192` |

Quick run:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
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
| `--llm-backend` | `openai-compatible`, `anthropic`, or `local-transformers`. |
| `--answer-model` | Model for `answer_generation`. |
| `--next-question-model` | Model for follow-up questions. |
| `--api-base-url` | Remote or local OpenAI-compatible/Anthropic base URL. |
| `--api-chat-url` | Full chat-completions URL, useful for Azure deployments. |
| `--api-key` | API key; defaults to env vars. |
| `--api-key-header` | Header name, default `Authorization`; Azure often uses `api-key`. |
| `--api-key-prefix` | Header prefix, default `Bearer ` for `Authorization`. |
| `--api-extra-headers-json` | JSON object merged into request headers. |
| `--api-extra-body-json` | JSON object merged into OpenAI-compatible request bodies. |
| `--no-api-response-format-json` | Disable `response_format={"type":"json_object"}` for providers that reject it. |
| `--answer-max-new-tokens` | Generation limit for answer JSON. |
| `--next-max-new-tokens` | Generation limit for next-question JSON. |
| `--temperature` | Sampling temperature. |
| `--json-retries` | Strict JSON retry count after parse failure. |

Environment variable priority for OpenAI-compatible keys:

1. `QI_LLM_API_KEY`
2. `OPENROUTER_API_KEY`
3. `DEEPSEEK_API_KEY`
4. `OPENAI_API_KEY`
5. `GEMINI_API_KEY`
6. `XAI_API_KEY`
7. `MISTRAL_API_KEY`
8. `MOONSHOT_API_KEY`
9. `DASHSCOPE_API_KEY`
10. `GROQ_API_KEY`
11. `TOGETHER_API_KEY`
12. `FIREWORKS_API_KEY`
13. `SILICONFLOW_API_KEY`
14. `AZURE_OPENAI_API_KEY`

For Anthropic, use `QI_LLM_API_KEY` or `ANTHROPIC_API_KEY`.

## Provider Examples

DeepSeek default:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
python scripts/llm_response.py --query "How are Kweichow Moutai's fundamentals recently?"
```

OpenAI-compatible endpoint:

```bash
export QI_LLM_API_KEY="your_api_key_here"
python scripts/llm_response.py \
  --api-base-url https://api.openai.com/v1 \
  --answer-model gpt-5.5 \
  --next-question-model gpt-5.4-mini \
  --api-extra-body-json '{}'
```

OpenRouter:

```bash
export OPENROUTER_API_KEY="your_openrouter_key_here"
python scripts/llm_response.py \
  --api-base-url https://openrouter.ai/api/v1 \
  --answer-model qwen/qwen3-next-80b-a3b-instruct:free \
  --next-question-model qwen/qwen3-next-80b-a3b-instruct:free \
  --api-extra-body-json '{}'
```

Anthropic:

```bash
export ANTHROPIC_API_KEY="your_anthropic_key_here"
python scripts/llm_response.py \
  --llm-backend anthropic \
  --answer-model claude-opus-4-6 \
  --next-question-model claude-sonnet-4-6
```

Local vLLM/OpenAI-compatible server:

```bash
python scripts/llm_response.py \
  --api-base-url http://127.0.0.1:8000/v1 \
  --answer-model NousResearch/Meta-Llama-3-8B-Instruct \
  --next-question-model NousResearch/Meta-Llama-3-8B-Instruct \
  --api-extra-body-json '{}' \
  --no-api-response-format-json
```

Ollama OpenAI-compatible server:

```bash
python scripts/llm_response.py \
  --api-base-url http://127.0.0.1:11434/v1 \
  --answer-model qwen3:8b \
  --next-question-model qwen3:8b \
  --api-extra-body-json '{}' \
  --no-api-response-format-json
```

Azure OpenAI deployment URL:

```bash
export AZURE_OPENAI_API_KEY="your_azure_key_here"
python scripts/llm_response.py \
  --api-chat-url "https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=<version>" \
  --api-key-header api-key \
  --api-key-prefix "" \
  --answer-model "<deployment>" \
  --next-question-model "<deployment>" \
  --api-extra-body-json '{}'
```

## Legacy Local Transformers

Local HuggingFace/transformers inference remains available:

```bash
python scripts/llm_response.py \
  --llm-backend local-transformers \
  --models-dir models/llm \
  --answer-model instruction-pretrain/finance-Llama3-8B \
  --next-question-model Qwen/Qwen2.5-3B-Instruct
```

Environment variables:

| Variable | Purpose |
|---|---|
| `LLM_MODELS_DIR` or `QI_LLM_MODELS_DIR` | Local model search root. |
| `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` | Authenticated HuggingFace access. |

Remote repository code is disabled by default. Use `--trust-remote-code` only for audited HuggingFace models that require it.

## DeepSeek JSON Output Notes

When JSON Output is enabled, prompts must include the word `json` and a JSON-format example. The script's prompts are written for strict JSON output and the max-token defaults are set high enough to avoid normal truncation. DeepSeek may still occasionally return empty content; rerun or slightly adjust prompts if that provider-side behavior appears.
