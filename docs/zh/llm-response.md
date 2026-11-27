# 受约束 LLM 回答交接

语言：[English](../llm-response.md) | 中文

`scripts/llm_response.py` 实现论文中的受约束本地模型总结与追问预测阶段（[DOI: `10.3233/FAIA261107`](https://ebooks.iospress.nl/doi/10.3233/FAIA261107)），将压缩后的 Query Intelligence 证据转化为经 Schema 校验的 JSON，不重新推断意图、标的或数据源计划。

> 生产级 LangGraph Agent（`/agent/chat*`）与 React 网页前端请参见 [Agent 层](agent.md) 与 [网页前端](frontend-chatbot.md)。

## 输出契约

输出两段经校验和规范化的 JSON：

- **`answer_generation`**：`answer`、`key_points[]`、`evidence_ids_used[]`（严格过滤为本次运行实际存在的 ID）、`limitations[]`、`risk_disclaimer`、`model`。
- **`next_question_prediction`**：三条推荐追问，附相关性分数与简短理由。

## 快速运行

```bash
# 先运行 Query Intelligence，再生成 JSON 回答
python scripts/llm_response.py --query "你觉得中国平安（601318.SH）怎么样？"

# 直接消费已有合并记录（{query, nlu_result, retrieval_result, ...}）
python scripts/llm_response.py --input path/to/pipeline_record.json
```

## 默认本地模型与参数

| 参数 / 配置 | 默认值 | 说明 |
|---|---|---|
| `--answer-model` | `instruction-pretrain/finance-Llama3-8B` | 回答生成模型 |
| `--next-question-model` | `Qwen/Qwen2.5-3B-Instruct` | 追问预测模型 |
| `--models-dir` | `models/llm`（`LLM_MODELS_DIR` / `QI_LLM_MODELS_DIR`） | 优先检索的本地模型目录 |
| `--answer-max-new-tokens` / `--next-max-new-tokens` | `700` / `260` | 生成 Token 上限 |
| `--json-retries` | `1` | JSON 解析失败后的修复重试次数 |
| `--device-map`、`--dtype`、`--temperature` | `auto` | 推理设备与采样配置 |
| `--trust-remote-code` | `false` | 仅对经审计的 HuggingFace 仓库显式开启 |

需要鉴权下载权重时通过 `HF_TOKEN` 或 `HUGGINGFACE_HUB_TOKEN` 提供凭据。

## 安全与约束

- 仅输入压缩后的证据摘要，自动剥离调试轨迹、长原文与原始接口噪声。
- 严格校验并过滤 `evidence_ids_used`，剔除思维链与 Markdown 代码块围栏。
- 严禁编造缺失的行情、基本面、宏观、新闻或情感事实，强制输出风险提示。
- 回归测试：`python -m pytest -q tests/test_llm_response.py`。
