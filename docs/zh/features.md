# 完整功能、API 与安装说明

[README](../../README_CN.md) · [English](../features.md)

第 15 轮从 `4ad3380` 的 `README_CN.md` 原样移来的完整功能列表、完整 API 表和完整的快速开始（只调整了相对链接）；README 只保留简短摘要。

- **Agent**：
  - 带理由的路由；并行工具调用；步数/工具/token 预算；
  - 运行截止时间，同时约束每一次 LLM 请求、重试和模型切换；
  - 校验失败后一次 LLM 修改；澄清中断与恢复；
  - 会话记忆：近期标的、用户说过的约束和持仓；
  - 追问处理：代词（「它」）、复数与指代组（「这两家」「三家里哪家」「前者/后者」「the latter」）、省略问法（「ROE呢」「换成五粮液呢」「And the P/B?」）、单独的「为什么」追问，以及对已讨论个股所在行业的提问；金融对话中的无关请求照样拒答，每次改写都记录为路由理由；
  - 覆盖范围：加密资产和美股/港股明确告知不在覆盖范围内；数据里没有的报告期或指标会如实说明，不会悄悄换成别的期（「没有2019年数据，以下为2025年报」）；
  - 按节点设置推理强度；带熔断的模型容灾；按哈希锁定的版本化 Prompt。
- **取证工具**：
  - 实体解析、行情、技术指标、基本面、宏观指标、新闻、公告、知识检索、文档情感；
  - 每个工具都有 Pydantic schema、超时、重试、TTL 缓存和可操作的错误提示。
- **声明核查**（`POST /agent/claim-check`）：
  - 贴一句话，例如「茅台市盈率只有15倍，股价跌了5%」；
  - 把每个数字对应到指标，与行情和基本面数据比对，不用 LLM；
  - 给出支持 / 矛盾 / 部分支持 / 无法核实，并附证据 id、来源和数据日期。
  - 处理比较词（超过/不到/以上/破/之间）、否定、区间、同比增速和中文数字；第 12 轮起还处理多家公司的合计（「茅台和五粮液成交额加起来不到50亿」按 52.47 亿的合计核查）、比较框架里或用「这一水平 / that」指代的行业平均、英文「beats … by N」以及「高出一倍多」这类一倍的差；第 13 轮起还处理 X倍半、「将近一倍」按 (80%, 100%]、「多一点」不是数字、以 52 周高低点为基准（「比52周高点低了三成多」）、「N turns above」、主语延续到「但 / 却」之后、点名公司之间的最高 / 最低、「a third / a quarter of」、「三个多百分点」和以另一板块平均为倍数基准；第 14 轮起扣非净利润和流通市值是独立指标（不再用归母净利润或总市值核对）、用「但 / but」分开的两个界限、主语在前的最高 / 最低、年份不读成数值、「up 42% year on year」按所点名指标的同比、「不到 / 超过…的N倍」是倍数的界限、「两个半X」、写明的口径（加权平均ROE、基本每股收益、归母增速、动态市盈率）只与同口径字段比对（否则为 `definition_mismatch`）、按日期取历史日线、外币金额（`currency`）和繁体字；在独立作者写的切片上，首次运行分别为 0.716（第四轮）、0.821（第五轮）、0.537（第六轮）、0.600（第八轮）、0.600（第九轮）、0.667（第十轮）；工程师表示没有打开的第九轮切片在第 13 轮规则之后为 1.000（25/25；去掉与自写措辞重合的 5 条后 20/20，去掉内容层面的重合后 15/15），第十轮切片在第 14 轮规则之后为 0.833（20/24；没有说法与自写措辞近似，5 条重复了自写内容，去掉后为 16/19）；较早的 47 条 held-out 说法首次运行为 0.936，之后已暴露。见 [docs/zh/claim-check.md](claim-check.md)。
- **实时数据**：
  - 东方财富 → 新浪 → 腾讯 → 缓存 → 最近一次成功数据 → 快照的降级链，每个数据源独立熔断；
  - 离线快照分两层：v1（截至 2026-04-22）和扩展层（另 15 个标的、一年日线，截至 2026-09-30），52 周最高/最低、最大回撤（近 52 周或今年以来）、今年以来涨跌、每股收益和总市值都带窗口和日期计算；每个回答写明各标的自己的数据日期；
  - v1 种子之上的更正层：种子出错处改用公司年报中的 2025 年数字（附页码和 sha256）和 2026-04-22 的日线记录，并为每家公司标明字段口径，回答和指标卡片都会写出（「ROE（加权）」「营业总收入」「归母净利润」、基本/稀释每股收益）；
  - 上游调用走有界线程池；新浪与同花顺基本面交叉核对；
  - 每条记录标注来源与时效；`GET /sources/health?probe=1` 主动探测（限频）。
- **协议**：工具经 MCP 提供；MCP 客户端可以把外部 MCP 服务器的工具注册进来（`QI_MCP_SERVERS`，结果按不可信数据处理）。整个 Agent 经 A2A 1.0 提供：澄清对应 `input-required`，流式接口汇报进度，任务表经 Postgres 共享。仓库里附带一个 a2a-sdk 客户端示例（`scripts/a2a_client_demo.py`）。
- **可观测性**：
  - 每次运行的 trace：节点、工具、LLM 调用（含上下文构成与 JSON 解析状态）、token、成本、Prompt 版本；
  - 运行查看接口；OpenTelemetry 导出；
  - 按实际作答模型打标签的 Prometheus 指标；分 5 行、26 个面板的 Grafana 看板（含按 Prompt 版本划分的校验失败率与修复率、用户好评率、按类别和按每个回答统计的输出安全层改动）、14 条告警规则、Jaeger；
  - 审计日志：每次拒答和每次合规改写各记一条结构化事件，只含哈希 id，不含用户原文；
  - 用户反馈（`POST /agent/feedback`）由 `scripts/feedback_to_tasks.py` 转成候选评测任务。
- **网页前端**（React 19、TypeScript、Tailwind v4、Radix、Motion、Lightweight Charts）：
  - 首字出现前有实时进度面板：当前步骤（规划 → 取数 → 撰写 → 核对）、正在调用的工具及其标的、已用时间和「停止」按钮；随后答案边写边显示，旁边是执行时间线；运行详情里显示首字耗时和总耗时；
  - 声明核查页面：逐个数字显示比较关系、声称值与实际值（相对差标为「相对差」）、来源和日期，顶栏固定；聊天中出现「听说……是真的吗」时会提示一键核查；
  - 与证据面板联动的引用标签（含时效）；价格图；运行成本与延迟；
  - 指标卡片以所问指标打头（问未来某天时不标），市盈率、市净率标估值日、财报数字标报告期，精度与正文一致；新闻回答用固定的主题标签说明每条资料讲什么，从不引用标题；
  - 执行过程与运行页签把每条路由依据写成文字（“计算ROE差值：五粮液 对比 贵州茅台”），刷新后在“已恢复”横幅下显示恢复的历史；
  - 反馈按钮与 Markdown 导出；中英文（系统提示随语言切换）、深色模式、移动端；Playwright 在 Chrome 中针对离线服务测试三轮差值会话与核查（`cd frontend && pnpm run e2e`）。
- **安全**：
  - 可选 API Key，会话与 trace 按 Key 隔离（存的是哈希，不是 Key 本身）；
  - 限流、CORS、请求体上限；针对用户消息中指令的输入守卫；
  - 非 root 且根文件系统只读的容器；Kubernetes 清单带 NetworkPolicy，不含明文 Secret；
  - CI 用 gitleaks 扫描全部 git 历史，pre-commit 钩子做同样的扫描（[SECURITY.md](../../SECURITY.md)）。

## API

| 接口 | 用途 |
|---|---|
| `POST /agent/chat`、`POST /agent/chat/stream`（SSE）、`POST /agent/resume` | 带证据、校验结果、trace id 和成本的 Agent 回答；节点事件和边写边出的 `answer_delta` 文本；澄清后恢复。 |
| `POST /agent/claim-check` | 核查一句市场说法里的数字（确定性，不用 LLM）。 |
| `POST /agent/feedback` | 对回答点赞/点踩，与对应 trace 一起保存。 |
| `GET /agent/sessions/{id}`、`GET /agent/traces`、`GET /agent/traces/{id}` | 会话记忆、最近运行、完整 trace，只返回调用方自己的。 |
| `GET /.well-known/agent-card.json`、`POST /a2a` | A2A 服务卡片与 JSON-RPC 接口。 |
| `GET /metrics`、`GET /sources/health[?probe=1]` | Prometheus 指标；数据源状态（按需主动探测全部 19 个数据源，未探测的会标出）。 |
| `GET /health`、`GET /ready` | 存活检查；就绪检查（检查点存储可连接且可写、LLM 配置、检索索引、离线数据层及其记录数和 sha256），未就绪或开启的数据层缺失时返回 503。 |
| `POST /chat` | 原聊天接口（`mode=workflow` 保持原流程；`agent`/`auto` 走 Agent）。 |
| `POST /nlu/analyze`、`POST /retrieval/search`、`POST /query/intelligence` | 经典 NLU 与检索产物。 |

Schema 位于 `schemas/agent_*.schema.json`，由 `query_intelligence/contracts.py` 生成。

## 完整的快速开始

```bash
pip install -r requirements.txt
uvicorn query_intelligence.api.app:create_app --factory --port 8765   # 打开 http://127.0.0.1:8765
```

没有 Key 时由确定性路径回答。启用 LLM Agent（任何 OpenAI 兼容接口均可）：

```bash
export DEEPSEEK_API_KEY=...                      # 只从环境变量读取，不要写进配置文件
export DEEPSEEK_BASE_URL=https://api.deepseek.com # 或网关，例如 https://api.cline.bot/api/v1
export DEEPSEEK_MODEL=deepseek-v4-flash
export QI_LLM_FALLBACK_MODELS=...                 # 可选：同一接口上的备用模型
```

实时行情、新闻、公告、宏观默认开启；设置 `QI_USE_LIVE_MARKET=0`（以及 `_NEWS`、`_ANNOUNCEMENT`、`_MACRO`）即使用随仓库提供的离线快照。离线快照由 v1（`data/structured_data.json`，7 个标的，截至 2026-04-22）和扩展层（`data/snapshot/`，另 15 个，截至 2026-09-30；`QI_OFFLINE_SNAPSHOT_EXT=0` 关闭）组成，并应用更正层（`data/snapshot/structured_data_fix.json`：种子出错处的一手来源数值和字段口径；`QI_OFFLINE_SNAPSHOT_FIX=0` 关闭）；哪些是实时、哪些是快照以及降级顺序见 [docs/zh/data-sources.md](data-sources.md)。

Docker、Kubernetes（多副本经 Postgres 共享会话、只读根文件系统）与监控栈见 [docs/zh/deployment.md](deployment.md)。从第 14 轮起镜像也带上了更正层（之前容器给出的是 v1 种子数字，J2），`GET /ready` 会列出已加载的数据层（[部署](deployment.md#docker)）。

```bash
TAG=$(git rev-parse --short=7 HEAD)
docker build -f docker/Dockerfile -t finsight:$TAG .   # 镜像按 commit 打标签
(cd deploy/k8s && kustomize edit set image finsight=finsight:$TAG)   # 提交的标签是占位值（docs/zh/deployment.md#镜像标签）
kubectl apply -k deploy/k8s        # 先创建 finsight-db Secret（见 docs/zh/deployment.md#密钥）
docker compose -f docker/docker-compose.yml --profile monitoring up -d   # 加上 Prometheus、Grafana、Jaeger
```
