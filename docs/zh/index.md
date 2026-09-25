# 文档导航

语言：[English](../index.md) | 中文

这些文档说明 FinSight：一个证据优先的 A 股研究 Agent。经典 NLU 与检索后端是 Query Intelligence，Agent 层用 LangGraph 编排它。

根目录 README 只保留项目定位、架构图、快速运行、模块和必要配置。详细说明放在本目录中。

## 内容

| 文档 | 用途 |
|---|---|
| [模块总览](modules.md) | 五个模块地图：前端、NLU/Retrieval、数值分析、文本分析、LLM 总结和预测。 |
| [Agent 层](agent.md) | Agent 状态图、工具、路由、证据校验、合规、记忆、`/agent/*` API 与 schema、配置、tracing。 |
| [Agent 评测](../agent-eval.md)（英文） | 任务集、快照回放、一票否决指标、真实 LLM 在线消融（pass^3、延迟、成本）、Prompt A/B、校验器压力测试、提示注入红队、故障注入与局限。 |
| [MCP Server](../mcp.md)（英文） | 通过 stdio 或 streamable HTTP 向 MCP 客户端提供 Agent 工具。 |
| [A2A、容灾与可观测性](../a2a-and-observability.md)（英文） | A2A 接口、LLM 网关适配、模型容灾、网关成本、运行检查器、Prometheus 指标。 |
| [数据源](../data-sources.md)（英文） | 实时数据源审计、降级链、熔断、缓存、数据来源标注、`/sources/health`。 |
| [性能](../performance.md)（英文） | 容器压测、检查点瓶颈与修复、扩展边界。 |
| [部署](../deployment.md)（英文） | Docker 镜像、Kubernetes 清单（多副本经 Postgres 共享会话）、只读根文件系统。 |
| [Agent 工程实践调研](../research/agent-architecture-practices-2026.md)（英文） | 2025–2026 Agent 与 Prompt 工程一手资料调研，以及驱动本轮改动的差距分析。 |
| [Query Intelligence](query-intelligence.md) | 支持范围、架构、API、NLU 和 Retrieval 输出契约、live provider、环境变量、排错。 |
| [本地网页 Chatbot](frontend-chatbot.md) | 浏览器 UI、`/chat` 契约、LLM API 配置、真实本地截图和排错。 |
| [数值分析](numerical-analysis.md) | `analysis_summary`、技术指标、基本面、宏观信号和数据就绪程度。 |
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
