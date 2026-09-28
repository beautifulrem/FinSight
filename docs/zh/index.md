# 文档导航

语言：[English](../index.md) | 中文

这些文档说明 FinSight：一个证据优先的 A 股研究 Agent。经典 NLU 与检索后端是 Query Intelligence，Agent 层用 LangGraph 编排它。

根目录 README 只保留项目定位、架构图、快速运行、模块和必要配置。详细说明放在本目录中。

## 内容

| 文档 | 用途 |
|---|---|
| [模块总览](modules.md) | 五个模块地图：前端、NLU/Retrieval、数值分析、文本分析、LLM 总结和预测。 |
| [Agent 层](agent.md) | Agent 状态图、工具、路由、证据校验、合规、记忆、`/agent/*` API 与 schema、配置、tracing。 |
| [说法核查](claim-check.md) | 核查一句市场说法：比较词、否定、增速、单位、数据日期、聊天提示，以及说法基准（dev 131 条、held-out 47 条）的置信区间与局限。 |
| [评测（中文摘要）](evaluation.md) | 三个任务集、一票否决指标、bootstrap 置信区间与配对检验、两个模型族的在线消融、Prompt A/B、校验器压力测试、红队、故障注入、路由评测。完整英文版见 [Agent Evaluation](../agent-eval.md)。 |
| [竞品对比](comparison.md) | 与问财、豆包、Kimi、Wind Alice、妙想的对比：带出处的能力表、它们强在哪里、一个设计好但尚未执行的对比测试。 |
| [MCP Server](../mcp.md)（英文） | 通过 stdio 或 streamable HTTP 向 MCP 客户端提供 Agent 工具。 |
| [A2A、容灾与可观测性](a2a-and-observability.md) | A2A 接口、LLM 网关适配、模型容灾、网关成本、运行查看器与用户反馈、Prometheus 指标、Grafana 看板与告警、故障演练。 |
| [实时数据源](data-sources.md) | 实时数据源审计、降级链、熔断、有界调用池、缓存、新浪/同花顺交叉核对、来源标注、`/sources/health`（含主动探测）。 |
| [性能](performance.md) | 确定性路径与 LLM Agent 路径压测（含人民币成本）、检查点修复的复现、k3s 多副本扩展、服务启动时间。 |
| [部署](deployment.md) | Docker 镜像、监控 profile、Kubernetes 清单（多副本经 Postgres 共享会话）、只读根文件系统。 |
| [设计复盘](../presentation/agent-design-notes.md) | 关键取舍、自研与复用的边界、32 个真实失败案例（根因与修复）、带置信区间的消融结论、独立任务集结果、面试问答。 |
| [面试讲稿](../presentation/interview-script.md) | 30 秒介绍、架构讲解、三个可辩护的数字、失败故事、白板提纲、AI 编程工具使用说明。 |
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
