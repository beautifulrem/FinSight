# 完整功能、API 与配置参考

语言：[English](../features.md) | 中文

本文汇总 FinSight 的完整能力清单、HTTP API 契约与常用运行开关，覆盖论文核心 **Query Intelligence**（[DOI: `10.3233/FAIA261107`](https://ebooks.iospress.nl/doi/10.3233/FAIA261107)）与可扩展的 **Agent 编排层**。

## 核心能力

### 研究 Agent 与编排
- **可解释路由**：经典 NLU 与输入守卫先完成越界拒答、缺标澄清、简单确定性流程与复杂 LangGraph 循环分流，每条路由决策均记录原因。
- **多轮会话与指代消解**：LangGraph Checkpointer 记录近期标的、约束条件与持仓；支持单/复数指代（`它`、`这两家`、`前者/后者`、`the latter`）与省略追问（`ROE呢`、`换成五粮液呢`），同时拦截对话中途的越界提问。
- **有界执行**：全链路截止时间（覆盖单次 LLM 调用、重试与模型容灾）、步数/工具/Token 预算上限、并行工具调用与哈希锁定的 Prompt 版本。
- **校验与合规护栏**：逐句数值与引用校验器（`verifier.py`）失败时触发一次定向修订；输出安全层（`output_safety.py`、`text_safety.py`）去除交易指令、对价格/业绩/市值预测加限定语、标注单源或冲突文档数字，并拒绝港美股/GDR/ADR 及仿冒公司名称。

### 9 个类型化只读工具
覆盖实体消歧、历史行情、技术指标、基本面、宏观指标、新闻、公告、知识检索与文档情感。每个工具均具备 Pydantic 输入输出契约、超时、重试、TTL 缓存与错误提示。

### 零 LLM 声明核查
`POST /agent/claim-check` 将粘贴的市场说法（如「茅台市盈率只有15倍，股价跌了5%」）对照行情与基本面数据逐项核对，不调用 LLM。支持中英比较词、否定、区间、同比增速、多公司求和、52 周高低点、极值比较、行业均值、显式会计口径（`加权平均ROE`、`扣非净利润`、`流通市值`）、指定交易日与币种校验。详见 [claim-check.md](claim-check.md)。

### 数据源与离线分层快照
- **实时降级链**：Tushare / 东方财富 → 新浪 → 腾讯 → TTL 缓存 → 最近可用值 → 离线快照，配备单源熔断器、有界调用池、新浪/同花顺基本面交叉核验与完整溯源字段（`source_id`、`source_mode`、`fallback_reason`、`cache_hit`）。
- **离线三层数据**：v1 种子（`data/structured_data.json`）、15 只标的扩展日线（`data/snapshot/structured_data_ext.json`）与一手年报更正层（`data/snapshot/structured_data_fix.json`），启动时校验 SHA-256 并在 `GET /ready` 展示加载状态。

### 协议、可观测性与前端
- **协议**：MCP 服务端（stdio 与鉴权 streamable HTTP）、外部 MCP 客户端（`QI_MCP_SERVERS`）与 A2A 1.0 JSON-RPC/流式服务（支持 Postgres 多副本共享任务）。
- **可观测性**：单次运行 JSON/OTLP（Jaeger）追踪、Prometheus `/metrics`、26 面板 Grafana 看板与 14 条告警规则、脱敏审计日志及用户反馈采集（`POST /agent/feedback`）。
- **网页前端**：React 19 + TypeScript + Tailwind v4 + Lightweight Charts 双语工作台，含实时进度面板、SSE 流式输出、证据台账、KPI 卡片、声明核查标签页、深色模式、Markdown 导出及 Playwright E2E 测试。
- **安全加固**：API Key 与签名 Cookie 租户隔离、`QI_PROFILE=production` 失败关闭、分块安全请求体上限（`413`）、Postgres 共享限流、CI 全历史 `gitleaks` 扫描、非 root 只读容器与 Kubernetes `NetworkPolicy`（见 [SECURITY.md](../../SECURITY.md)）。

## API

| 接口 | 用途 |
|---|---|
| `POST /agent/chat` | 带证据台账、校验报告、Trace ID 与成本的 Agent 回答 |
| `POST /agent/chat/stream` | SSE 流式输出节点进度、工具事件、`answer_delta` 文本与最终校验结果 |
| `POST /agent/resume` | 在同一会话中恢复暂停的澄清轮次（`interrupt` / `resume`） |
| `POST /agent/claim-check` | 零 LLM 市场声明数值核查 |
| `POST /agent/feedback` | 记录绑定 `trace_id` 的赞/踩反馈 |
| `GET /agent/sessions/{id}` | 按调用方隔离查询会话记忆与历史轮次 |
| `GET /agent/traces`、`GET /agent/traces/{id}` | 按调用方隔离查询运行 Trace（匿名访问返回 `403`） |
| `GET /.well-known/agent-card.json`、`POST /a2a` | A2A 1.0 服务发现卡片与 JSON-RPC/流式入口 |
| `GET /sources/health[?probe=1]` | 实时数据源健康度与熔断状态（`probe=1` 触发限流主动探测） |
| `GET /health`、`GET /ready`、`GET /metrics` | 存活探针、就绪探针（检查点、模型、索引、离线数据层）与 Prometheus 指标 |
| `POST /chat` | 经典聊天接口（`mode=workflow` 走论文流水线；`agent` / `auto` 路由至 Agent） |
| `POST /nlu/analyze`、`POST /retrieval/search`、`POST /query/intelligence`、`POST /query/intelligence/artifacts` | 经典 Query Intelligence 流水线，返回类型化 `NLUResult` 与 `RetrievalResult` |

由 `query_intelligence/contracts.py` 导出的 JSON Schema 位于 `schemas/agent_*.schema.json`、`schemas/nlu_result.schema.json` 与 `schemas/retrieval_result.schema.json`（`EntityItem.symbol` 限制为最长 64 字符的 `^[A-Za-z0-9._-]+$`；`EvidenceItem` 包含完整溯源字段）。

## 快速配置

### 本地服务与 LLM 接口

```bash
pip install -r requirements.txt
uvicorn query_intelligence.api.app:create_app --factory --port 8765
```

配置任意 OpenAI 兼容接口即可启用 LLM Agent（不设置 `DEEPSEEK_API_KEY` 则运行确定性兜底）：

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-v4-flash
export QI_LLM_FALLBACK_MODELS=...   # 可选，同端点逗号分隔容灾模型
```

### 实时源与离线快照开关

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `QI_USE_LIVE_MARKET` / `_NEWS` / `_ANNOUNCEMENT` / `_MACRO` | `true` | 设为 `0` 强制仅使用随仓离线快照 |
| `QI_OFFLINE_SNAPSHOT_EXT` | `true` | 加载 15 只标的扩展日线快照（`data/snapshot/structured_data_ext.json`） |
| `QI_OFFLINE_SNAPSHOT_FIX` | `true` | 应用一手年报更正层（`data/snapshot/structured_data_fix.json`） |

### 容器、Kubernetes 与监控栈

```bash
TAG=$(git rev-parse --short=7 HEAD)
docker build -f docker/Dockerfile -t finsight:$TAG .
docker compose -f docker/docker-compose.yml --profile monitoring up -d
(cd deploy/k8s && kustomize edit set image finsight=finsight:$TAG)
kubectl apply -k deploy/k8s
```

密钥创建、网络策略与探针详见 [deployment.md](deployment.md)。
