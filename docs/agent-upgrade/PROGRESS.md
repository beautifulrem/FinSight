# FinSight Agent 化升级进度

任务定义见 [GOAL.md](GOAL.md)。状态：⬜ 未开始 · 🟡 进行中 · ✅ 完成 · ⛔ 阻塞

## 任务状态

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| T0.1 | 环境与基线 | ✅ | 基线：364 passed / 53 skipped / 1 failed（49.5 分钟，live provider 默认开启导致逐个等待网络超时）；唯一失败 `test_fuzz_query_intelligence_report` 已修复（见迭代 2） |
| T0.2 | 文档纠偏 | ✅ | README 架构图按真实链路重画；删除 `submission/`；评测构造与局限写入 training 文档；UI 标题中性化；AGENTS.md 增加 Agent 层规则 |
| T0.3 | 代码质量工具 | ✅ | `ruff.toml`（仅约束 agent 相关目录）+ `requirements-dev.txt` |
| T0.4 | 结构化日志 | ✅ | `api/app.py` 改为 `logging`，保留控制台进度输出 |
| T1.1 | 工具基座 | ✅ | `agent/tools/base.py`：超时/重试/TTL 缓存/错误规范化/OpenAI schema |
| T1.2 | 证据层 | ✅ | `agent/evidence.py`：AgentEvidence/EvidenceStore/数字抽取 |
| T1.3 | resolve_entity | ✅ | `resolve_entity` |
| T1.4 | get_price_history + compute_indicators | ✅ | `get_price_history`、`compute_indicators`（离线种子数据无历史 → indicators 返回 unavailable） |
| T1.5 | get_fundamentals + get_macro_indicators | ✅ | `get_fundamentals`、`get_macro_indicators` |
| T1.6 | search_news / search_announcements | ✅ | `search_news`、`search_announcements`、额外的 `search_knowledge` |
| T1.7 | analyze_sentiment | ✅ | `analyze_sentiment`：默认经典模型，`QI_AGENT_SENTIMENT_BACKEND=finbert` 可选 |
| T1.8 | MCP Server | ✅ | `agent/mcp_server.py` + `docs/mcp.md`；stdio 子进程端到端验证通过 |
| T2.1 | LLM 层 | ✅ | `agent/llm.py`：DeepSeekToolClient（非 strict、回传 reasoning_content、429/5xx 重试）、ScriptedLLM、Pricing（仅配置时计费） |
| T2.2 | 确定性规划器 | ✅ | `agent/planner.py`，14 个测试 |
| T2.3 | 状态与图 | ✅ | `agent/state.py` + `agent/graph.py`（LangGraph 1.2.12）；13 个图测试 + 7 个真实离线服务集成测试 |
| T2.4 | 路由器 | ✅ | `agent/router.py`，31 条表驱动用例 |
| T2.5 | 证据校验器 | ✅ | `agent/verifier.py`：引用存在性 + 数值可追溯（单位换算/四舍五入容差）+ 修复 |
| T2.6 | 合规节点 | ✅ | `agent/compliance.py`：复用 llm_response 软化规则与 chatbot 新鲜度/免责声明；删除直接交易指令 |
| T2.7 | 记忆与澄清 | ✅ | `agent/memory.py`：InMemory/SQLite checkpointer、turn 级状态重置、会话历史→NLU dialog_context、代词指代改写（它/it/its）、`interrupt()` 澄清 + resume |
| T2.8 | 服务与 API | ✅ | `agent/service.py` + `/agent/chat`、`/agent/chat/stream`（SSE）、`/agent/resume`、`/agent/sessions/{id}`；`/chat` 新增 `mode`（默认 workflow 保持旧行为） |
| T2.9 | 接入情感与下一问 | ✅ | `agent/followups.py`：在线情感摘要 + 确定性下一问建议（复用 llm_response 净化规则）；README 架构图更新 |
| T2.10 | 提示注入防护 | ✅ | `agent/injection.py`：工具结果以不可信数据信封包装、指令类文本脱敏；端到端注入测试 |
| T3.1 | Tracing | ✅ | `agent/tracing.py`：节点/工具/LLM 调用 trace，JSON 落盘 + 可选 OTLP 导出；响应带 trace_id |
| T3.2 | 评测任务集 | ⬜ | |
| T3.3 | 数据快照回放 | ⬜ | |
| T3.4 | 指标 | ⬜ | |
| T3.5 | 消融 | ⬜ | |
| T3.6 | 故障注入 | ⬜ | |
| T3.7 | 评测报告 | ⬜ | |
| T3.8 | CI 评测门禁 | ⬜ | |
| T4.1 | 前端拆分 | ⬜ | |
| T4.2 | 异步与并发 | ⬜ | |
| T4.3 | Docker | ⬜ | |
| T4.4 | CI | ⬜ | |
| T4.5 | 安全 | ⬜ | |
| T4.6 | 重构 chatbot.py | ⬜ | |
| T5.1 | 文档 | ⬜ | |
| T5.2 | 设计复盘 | ⬜ | |
| T5.3 | 最终验收 | ⬜ | |

## 阻塞

（无）

## 迭代日志

<!-- 每轮追加：日期 · 任务 · 改动摘要 · 测试命令与结果 · 遗留问题 · 下一步 -->

### 2026-09-24 · 迭代 1 · T0.1–T2.1

- 环境：`uv venv .venv -p 3.13`；`download.pytorch.org` 与 `huggingface.co`、`api-docs.deepseek.com`、行情/公告数据源均被沙箱网络策略拦截；torch 从 PyPI 安装（CUDA 版，CPU 运行）。
- **发现**：`Settings` 默认 `use_live_*=True`，测试与服务在无网络环境会逐个等待 live provider 超时，导致启动/测试很慢。新增 `offline_service` 测试夹具（全部 live 关闭）。
- **发现并修复数据缺陷**：`data/runtime/alias_table.csv` 把表头 `公司名称` 作为 春秋电子 的官方别名，任何包含该词的问题都会被解析为 603890.SH；已删除该行并在 `runtime_entity_assets.py` 生成器中屏蔽表头类别名。
- `RetrievalPipeline` 新增公开方法 `retrieve_documents` / `fetch_structured`，`run()` 改为调用它们（行为不变），工具复用同一检索逻辑。
- 新依赖：`mcp>=2.2,<3`（SDK 2.x：`FastMCP` 已更名为 `MCPServer`，本项目使用低层 `Server` 以原样发布 Pydantic schema）。
- 测试：
  - `pytest tests/test_agent_tools_base.py tests/test_agent_tools.py tests/test_agent_mcp_server.py tests/test_agent_llm.py` → 全部通过（15 + 16 + 5 + 11）
  - `QI_USE_LIVE_MARKET=0 QI_USE_LIVE_MACRO=0 pytest tests/test_runtime_entity_assets.py tests/test_query_intelligence.py tests/test_analysis_summary.py tests/test_retrieval_selector.py tests/test_chatbot.py` → 64 passed, 40 skipped
  - `ruff check .` → 通过
- 下一步：T2.2 确定性规划器 → T2.3 图。

### 2026-09-24 · 迭代 2 · T0.1 收尾、T2.2–T2.6、T2.10

- 基线（T0.1）：`python -m pytest -q tests/`（live 默认开启、沙箱无外网）→ **364 passed, 53 skipped, 1 failed, 2971 s**。
  - 失败根因：`evaluation/fuzz_query_intelligence_report.py` 只读取 `evaluation/fuzz_query_intelligence_report.json`，该文件被 `.gitignore` 忽略且随 `submission/` 一起删除，任何干净 clone 都会失败。
  - 修复：从历史报告中恢复 32 个用例定义到已提交的 `evaluation/fuzz_cases.jsonl`（两个多轮用例的上下文输入为重建值，已在用例中注明），`build_fuzz_report()` 改为离线实际运行、schema 校验并打分。
  - **发现**：离线重跑结果 23/32 用例通过、331/361 检查通过（历史报告 2026-04-22 为 32/32）。差异一部分来自 live 数据（公告、告警），一部分是 NLU 漂移（question_style 4 例、错别字实体 1 例、英文板块建议 1 例），如实记录，不在本目标范围内调参。
- 新增：确定性规划器、路由器、证据校验器、合规节点、提示注入防护、模板组答器、LangGraph 图（guard_in → refuse/clarify/execute_plan/agent_llm⇄agent_tools → verify⇄revise → compliance → finalize）。
- 新依赖：`langgraph>=1.2,<2`（安装版本 1.2.12）。
- 测试：
  - `pytest tests/test_agent_planner.py tests/test_agent_router.py tests/test_agent_verifier.py tests/test_agent_compliance.py tests/test_agent_graph.py tests/test_agent_composer_injection.py` → 14 + 34 + 8 + 19 + 13 + 13 全部通过
  - `pytest tests/test_agent_graph_integration.py` → 7 passed（真实离线服务 + 真实工具）
  - `pytest tests/test_fuzz_query_intelligence_report.py` → 2 passed
  - `ruff check .` → 通过
- 下一步：T2.7 记忆与澄清中断 → T2.8 服务与 API → T2.9。

### 2026-09-24 · 迭代 3 · T2.7–T2.9、T3.1

- 记忆：checkpointer 持久化 `turns`；每轮开始用 `{__reset__}` 标记重置工具日志、证据、诊断等 turn 级字段（否则同一 thread 的上轮数据会泄漏到本轮——已用测试覆盖）。
- **发现**：真实 NLU 对「那它的市净率呢」只识别出「市净率」（财务指标实体），因此不会回退到对话上下文。新增基于规则、可解释的指代改写（代词→上一轮唯一上市实体），改写原因写入 `route_reasons`。
- 澄清：缺标的时 `interrupt()` 暂停，`/agent/resume` 注入用户回复后重新走 guard_in（每轮最多一次，避免循环）。
- API：`/agent/*` 端点与 `/chat?mode`；Agent 服务惰性构建，不影响已有测试用 stub。
- 情感与下一问：Agent 结果新增 `sentiment` 与 `next_questions`。
- Tracing：`outputs/traces/<date>/<trace_id>.json`（已 gitignore），`QI_AGENT_OTEL=1` 或 `OTEL_EXPORTER_OTLP_ENDPOINT` 时导出 OTLP；Langfuse Python SDK v3/v4 API 差异较大且无法在沙箱验证，因此统一走 OTLP（Langfuse 支持 OTLP 接入）。
- 新依赖：`langgraph-checkpoint-sqlite`、`opentelemetry-sdk`、`opentelemetry-exporter-otlp-proto-http`。
- 测试：`pytest tests/test_agent_*.py`（除集成外）全部通过；集成 `tests/test_agent_graph_integration.py` 9 passed；`tests/test_chatbot.py` 13 passed；`ruff check .` 通过。
- 下一步：T3.2 评测任务集 → T3.3 回放 → T3.4 指标 → T3.5 消融 → T3.6 故障注入 → T3.7 报告。
