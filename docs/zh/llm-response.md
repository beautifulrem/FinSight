# LLM 回答生成交接

语言：[English](../llm-response.md) | 中文

`scripts/llm_response.py` 是 FinSight 下游工具，用紧凑的 Query Intelligence 证据生成前端可直接消费的 JSON。它可以读取已有 record，也可以先从原始 query 跑 Query Intelligence。

它不属于 NLU/Retrieval 主干，不应重新推断意图、目标实体或 source plan。

本页记录当前 checkout 中旧版的本地 HuggingFace/transformers 回答交接路径。本地网页 Chatbot 和 DeepSeek v4 API 路径见 [本地网页 Chatbot](frontend-chatbot.md)。

## 输出

脚本生成：

- `answer_generation`：回答文本、关键点、使用的 evidence IDs、局限、风险提示和模型名。
- `next_question_prediction`：严格 3 个追问，包含分数和简短理由。

两个输出都会做 JSON 校验和规范化。

## 默认后端

当前脚本默认值：

| 设置 | 默认值 |
|---|---|
| Backend | 本地 HuggingFace/transformers |
| 回答模型 | `instruction-pretrain/finance-Llama3-8B` |
| 追问模型 | `Qwen/Qwen2.5-3B-Instruct` |
| 模型搜索根目录 | `models/llm`，以及 `LLM_MODELS_DIR` / `QI_LLM_MODELS_DIR` |
| 回答 max tokens | `700` |
| 追问 max tokens | `260` |
| JSON repair retries | `1` |

快速运行：

```bash
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
| `--answer-model` | `answer_generation` 使用的模型。 |
| `--next-question-model` | 追问使用的模型。 |
| `--models-dir` | 本地模型目录，优先于 HuggingFace model ID。 |
| `--few-shot-source` | 回答和追问 prompt 使用的 few-shot JSONL 来源。 |
| `--answer-max-new-tokens` | 回答 JSON token 上限。 |
| `--next-max-new-tokens` | 追问 JSON token 上限。 |
| `--device-map` | 传给 transformers 的设备映射。 |
| `--dtype` | `auto`、`float16`、`bfloat16` 或 `float32`。 |
| `--temperature` | 采样温度。 |
| `--json-retries` | JSON 解析失败后的严格重试次数。 |
| `--trust-remote-code` | 只有已审计且必须依赖该能力的 HuggingFace 模型才开启。 |

环境变量：

| 变量 | 用途 |
|---|---|
| `LLM_MODELS_DIR` 或 `QI_LLM_MODELS_DIR` | 本地模型搜索根目录。 |
| `HF_TOKEN` 或 `HUGGINGFACE_HUB_TOKEN` | HuggingFace 鉴权。 |

默认禁用远程 repository code。只有审计过模型仓库后才使用 `--trust-remote-code`。
