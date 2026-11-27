# 文档目录

语言：[English](../index.md) | 中文

**FinSight** 面向中国金融市场提供证据优先的金融分析问答与可验证研究 Agent。其经典可解释 Query Intelligence 核心发表于 *Frontiers in Artificial Intelligence and Applications* 第 433 卷（*Digitalization and Management Innovation V*，DMI 2026，IOS Press；DOI：[`10.3233/FAIA261107`](https://ebooks.iospress.nl/doi/10.3233/FAIA261107)，引用元数据见 [`CITATION.cff`](../../CITATION.cff)）。

快速上手与核心指标请先阅读 [中文 README](../../README_CN.md)，按主题查阅下方参考文档。

## 核心架构

| 文档 | 内容 |
|---|---|
| [系统模块](modules.md) | 六层模块职责地图（论文五阶段 + Agent 编排层）、数据流与扩展入口 |
| [Query Intelligence](query-intelligence.md) | 经典 NLU、混合检索、`NLUResult` / `RetrievalResult` 契约与论文基准 |
| [Agent 层](agent.md) | LangGraph 工作流、确定性兜底、9 个只读工具、逐句数值校验、安全护栏与配置 |
| [完整功能、API 与配置](features.md) | 全部能力清单、HTTP 接口契约与运行开关 |
| [检索输出规范](retrieval_output_spec.md) | `RetrievalResult`、`analysis_summary` 与数据溯源字段说明 |

## 分析、声明核查与数据源

| 文档 | 内容 |
|---|---|
| [声明核查](claim-check.md) | 零 LLM 金融声明数值核查、比较语法、口径匹配与基准评测 |
| [数据源架构](data-sources.md) | 实时多源（Tushare、AKShare、巨潮、efinance）、降级链、熔断器与离线快照/更正层 |
| [数值分析](numerical-analysis.md) | `MarketAnalyzer` 技术指标、基本面/宏观信号与 `data_readiness` |
| [文档情感分析](sentiment.md) | 下游逐句 FinBERT 与离线经典词典情感流水线（`sentiment/`） |
| [受约束 LLM 回答交接](llm-response.md) | 本地 HuggingFace/transformers JSON 总结与追问生成（`scripts/llm_response.py`） |

## 界面、协议与运维

| 文档 | 内容 |
|---|---|
| [网页前端与聊天界面](frontend-chatbot.md) | React 19 + TypeScript 工作台、SSE 流式追踪、声明核查页、经典 `/chat` 与 Playwright E2E |
| [MCP 集成](mcp.md) | stdio / 鉴权 HTTP MCP 服务暴露，以及接入外部 MCP 工具（`QI_MCP_SERVERS`） |
| [A2A 与可观测性](a2a-and-observability.md) | A2A 1.0 JSON-RPC/流式协议、模型容灾、Postgres 共享存储、Prometheus、Grafana 与 Jaeger |
| [部署指南](deployment.md) | 多阶段 Docker 镜像、Compose 配置、Kubernetes kustomize 清单、镜像标签策略与 kind 冒烟测试 |
| [性能与扩展](performance.md) | 延迟剖析、并发压测、上游故障演练与 k3s 多副本扩展 |
| [开发与推送前检查](development.md) | 本地 CI 镜像脚本（`scripts/prepush.sh`）、CI 时间预算与发布默认数据测试策略 |
| [模型训练与运行时资产](training.md) | 公开数据同步、Manifest 训练、SHA-256 校验与运行时资产物化 |

## 评测、治理与宣讲材料

| 文档 | 内容 |
|---|---|
| [评测摘要（中文）](evaluation.md) | 核心评测口径、对比基线、校验器与红队结果中文总览 |
| [Agent 评测完整版（自动生成）](../agent-eval.md) | 由 `evaluation/results/` 自动渲染的配对检验、置信区间、消融与红队报告（英文） |
| [评测历史](evaluation-history.md) | 独立任务集（首次运行 vs 暴露后）、逐轮评测记录、CI 流程与完整局限 |
| [已知问题与审计记录](known-issues.md) | 当前开放局限、取舍说明，以及各轮评审与安全审计修复摘要 |
| [同类产品对比](comparison.md) | 与问财、豆包、Kimi、Wind Alice、妙想的能力对比与人工评测方案 |
| [独立评审报告](../reviews/README.md) | 14 轮外部独立评审索引与评分标准 |
| [Agent 升级目标](../agent-upgrade/GOAL.md) · [进展](../agent-upgrade/PROGRESS.md) | Agent 层架构设计目标与里程碑记录 |
| [Agent 工程实践调研](../research/agent-architecture-practices-2026.md) | 生产级 Agent 与校验架构文献调研（英文） |
| [演示与设计资料](../presentation/README.md) | PPT 大纲、架构设计复盘与口头讲解提纲 |
| [安全政策](../../SECURITY.md) · [贡献指南](../../CONTRIBUTING.md) | 漏洞私密披露、安全边界与参与贡献规范 |
