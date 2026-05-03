# 文档导航

语言：[English](../index.md) | 中文

这些文档说明 FinSight，即 ARIN7012 Group 4.2 的证据优先金融分析聊天机器人项目。技术后端仍命名为 ARIN Query Intelligence。

根目录 README 只保留项目定位、架构图、快速运行、模块和必要配置。详细说明放在本目录中。

## 内容

| 文档 | 用途 |
|---|---|
| [Query Intelligence](query-intelligence.md) | 支持范围、架构、API、NLU 和 Retrieval 输出契约、live provider、环境变量、排错。 |
| [本地网页 Chatbot](frontend-chatbot.md) | 浏览器 UI、`/chat` 契约、DeepSeek v4 配置、真实本地截图和排错。 |
| [Presentation Materials](../presentation/README.md) | 清理后的英文汇报大纲和当前演示截图入口。 |
| [训练和运行时资产](training.md) | 公开数据同步、manifest 训练、运行时资产生成、评估和发布检查。 |
| [LLM 回答生成交接](llm-response.md) | 旧版本地 transformers 回答 JSON 交接、输出契约、配置和安全约束。 |
| [文档情感分析](sentiment.md) | 下游 sentiment pipeline、预处理、FinBERT 路由、输出字段和测试命令。 |
| [Retrieval 输出兼容说明](retrieval_output_spec.md) | `analysis_summary` 和 retrieval 输出引用的兼容入口。 |

English documentation is in [docs/](../index.md).

## 文档原则

- 根 README 回答“这是什么、怎么跑、更多信息在哪”。
- `docs/` 按主题保存稳定参考文档。
- 命令从 fresh clone 可直接复制运行，不包含本机专用 wrapper。
- 公开文档不包含真实 token、生成产物、缓存或机器路径。
