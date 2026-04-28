# LLM 回答生成交接

语言：[English](../llm-response.md) | 中文

`scripts/llm_response.py` 是 FinSight 下游工具，用紧凑的 Query Intelligence 证据生成前端可直接消费的 JSON。它可以读取已有 record，也可以先从原始 query 跑 Query Intelligence。

它不属于 NLU/Retrieval 主干，不应重新推断意图、目标实体或 source plan。

## 输出

脚本生成：

- `answer_generation`：回答文本、关键点、使用的 evidence IDs、局限、风险提示和模型名。
- `next_question_prediction`：严格 3 个追问，包含分数和简短理由。

两个输出都会做 JSON 校验和规范化。

## 默认后端

当前默认值：

| 设置 | 默认值 |
|---|---|
| Backend | `openai-compatible` |
| API base URL | `https://api.deepseek.com` |
| 回答模型 | `deepseek-v4-flash` |
| 追问模型 | `deepseek-v4-flash` |
| Extra body | `{"reasoning_effort":"max"}` |
| JSON response format | 启用 `response_format={"type":"json_object"}` |
| 回答 max tokens | `8192` |
| 追问 max tokens | `8192` |

快速运行：

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
python scripts/llm_response.py --query "你觉得中国平安怎么样？"
```

读取已有 combined record：

```bash
python scripts/llm_response.py --input path/to/pipeline_record.json
```

`--input` 需要一个 JSON 对象，包含 `query`、`nlu_result`、`retrieval_result`；`statistical_result` 和 `sentiment_result` 可选。Query Intelligence artifact 默认是分开的文件，因此可以先组合，或直接用 `--query`。

## 安全约束

- 只使用紧凑证据：query、NLU、retrieval evidence summary、可选统计结果和可选 sentiment 结果。
- 输入模型前去掉 debug trace、长原文、history arrays 和 source-generation metadata。
- OpenAI-compatible API 默认开启 JSON Output。
- 尽量校验和修复 JSON。
- 不得编造缺失行情、基本面、估值、宏观、新闻、情感或统计事实。
- 必须包含风险提示。
- 不输出 chain-of-thought 或 markdown fences。

## 配置

常用 CLI 参数：

| 参数 | 用途 |
|---|---|
| `--llm-backend` | `openai-compatible`、`anthropic` 或 `local-transformers`。 |
| `--answer-model` | `answer_generation` 使用的模型。 |
| `--next-question-model` | 追问使用的模型。 |
| `--api-base-url` | 远程或本地 OpenAI-compatible/Anthropic base URL。 |
| `--api-chat-url` | 完整 chat-completions URL，适合 Azure deployment。 |
| `--api-key` | API key，默认读环境变量。 |
| `--api-key-header` | key header，默认 `Authorization`；Azure 常用 `api-key`。 |
| `--api-key-prefix` | key 前缀，`Authorization` 默认 `Bearer `。 |
| `--api-extra-headers-json` | 合并到请求 header 的 JSON。 |
| `--api-extra-body-json` | 合并到 OpenAI-compatible 请求 body 的 JSON。 |
| `--no-api-response-format-json` | provider 不支持 JSON Output 时关闭。 |
| `--answer-max-new-tokens` | 回答 JSON token 上限。 |
| `--next-max-new-tokens` | 追问 JSON token 上限。 |
| `--temperature` | 采样温度。 |
| `--json-retries` | JSON 解析失败后的严格重试次数。 |

OpenAI-compatible key 环境变量优先级：

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

Anthropic 使用 `QI_LLM_API_KEY` 或 `ANTHROPIC_API_KEY`。

## Provider 示例

DeepSeek 默认：

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
python scripts/llm_response.py --query "贵州茅台最近基本面怎么样？"
```

OpenAI-compatible：

```bash
export QI_LLM_API_KEY="your_api_key_here"
python scripts/llm_response.py \
  --api-base-url https://api.openai.com/v1 \
  --answer-model gpt-5.5 \
  --next-question-model gpt-5.4-mini \
  --api-extra-body-json '{}'
```

OpenRouter：

```bash
export OPENROUTER_API_KEY="your_openrouter_key_here"
python scripts/llm_response.py \
  --api-base-url https://openrouter.ai/api/v1 \
  --answer-model qwen/qwen3-next-80b-a3b-instruct:free \
  --next-question-model qwen/qwen3-next-80b-a3b-instruct:free \
  --api-extra-body-json '{}'
```

Anthropic：

```bash
export ANTHROPIC_API_KEY="your_anthropic_key_here"
python scripts/llm_response.py \
  --llm-backend anthropic \
  --answer-model claude-opus-4-6 \
  --next-question-model claude-sonnet-4-6
```

本地 vLLM/OpenAI-compatible server：

```bash
python scripts/llm_response.py \
  --api-base-url http://127.0.0.1:8000/v1 \
  --answer-model NousResearch/Meta-Llama-3-8B-Instruct \
  --next-question-model NousResearch/Meta-Llama-3-8B-Instruct \
  --api-extra-body-json '{}' \
  --no-api-response-format-json
```

Ollama OpenAI-compatible server：

```bash
python scripts/llm_response.py \
  --api-base-url http://127.0.0.1:11434/v1 \
  --answer-model qwen3:8b \
  --next-question-model qwen3:8b \
  --api-extra-body-json '{}' \
  --no-api-response-format-json
```

Azure OpenAI deployment URL：

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

## 旧版本地 Transformers

仍支持本地 HuggingFace/transformers 推理：

```bash
python scripts/llm_response.py \
  --llm-backend local-transformers \
  --models-dir models/llm \
  --answer-model instruction-pretrain/finance-Llama3-8B \
  --next-question-model Qwen/Qwen2.5-3B-Instruct
```

环境变量：

| 变量 | 用途 |
|---|---|
| `LLM_MODELS_DIR` 或 `QI_LLM_MODELS_DIR` | 本地模型搜索根目录。 |
| `HF_TOKEN` 或 `HUGGINGFACE_HUB_TOKEN` | HuggingFace 鉴权。 |

默认禁用远程 repository code。只有审计过模型仓库后才使用 `--trust-remote-code`。

## DeepSeek JSON Output 注意事项

开启 JSON Output 时，prompt 中必须包含 `json` 字样和 JSON 格式示例。脚本的 prompt 已按严格 JSON 输出编写，并把 max-token 默认值调到 8192 以降低截断概率。DeepSeek 仍可能偶发返回空 content；遇到 provider 侧空响应时可以重试或微调 prompt。
