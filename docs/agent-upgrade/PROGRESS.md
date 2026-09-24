# FinSight Agent 化升级进度

任务定义见 [GOAL.md](GOAL.md)。状态：⬜ 未开始 · 🟡 进行中 · ✅ 完成 · ⛔ 阻塞

## 任务状态

| ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| T0.1 | 环境与基线 | 🟡 | `.venv`（Python 3.13）已建立并安装依赖；全量基线测试运行中（live provider 默认开启且沙箱网络被拦截，耗时很长） |
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
| T2.2 | 确定性规划器 | ⬜ | |
| T2.3 | 状态与图 | ⬜ | |
| T2.4 | 路由器 | ⬜ | |
| T2.5 | 证据校验器 | ⬜ | |
| T2.6 | 合规节点 | ⬜ | |
| T2.7 | 记忆与澄清 | ⬜ | |
| T2.8 | 服务与 API | ⬜ | |
| T2.9 | 接入情感与下一问 | ⬜ | |
| T2.10 | 提示注入防护 | ⬜ | |
| T3.1 | Tracing | ⬜ | |
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
