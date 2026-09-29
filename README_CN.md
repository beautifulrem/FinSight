<div align="center">

<h1>FinSight</h1>

<h3>证据优先的 A 股研究 Agent。</h3>

<p>
  <a href="README.md"><img alt="Language English" src="https://img.shields.io/badge/Language-English-2f80ed?style=flat&labelColor=555555"></a>
  <a href="README_CN.md"><img alt="Language Simplified Chinese" src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-d97706?style=flat&labelColor=555555"></a>
  <a href="LICENSE"><img alt="License MIT" src="https://img.shields.io/badge/License-MIT-f1c40f?style=flat&labelColor=555555"></a>
  <img alt="Python 3.13" src="https://img.shields.io/badge/Python-3.13-3776ab?style=flat&labelColor=555555">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-1.2-1c3c3c?style=flat&labelColor=555555">
  <img alt="MCP and A2A" src="https://img.shields.io/badge/MCP%20%2B%20A2A-protocols-6b46c1?style=flat&labelColor=555555">
  <img alt="React 19" src="https://img.shields.io/badge/UI-React%2019%20%2B%20TS-149eca?style=flat&labelColor=555555">
</p>

</div>

---

FinSight 回答关于 A 股上市公司、基金、指数和宏观数据的问题：

- 经典、可解释的 NLU 负责理解、守卫和路由；
- LLM 在 LangGraph 循环里编排类型化的取证工具；
- **答案里的每个数字都由代码对照同一句引用的证据逐句核对**；
- 最后由合规节点删除任何像投资建议的表述。

没有 LLM Key 时，同一张图由确定性规划器完成回答。

**30 秒看懂**

- **数字可追溯**：逐句校验器拦下了 3,399 个被篡改答案中的 98.1%，202 个正确答案全部通过。把另一家公司的数字写进来，原来的整批比对 100% 放过，现在只放过 0.5%。校验不通过的答案按整句删除修复，100% 可读；原来的按子句拼接只有 29% 可读。
- **靠的是工具**：不调工具的 LLM 在四个任务集共 511 个任务上一题都过不了（一票否决口径）。调工具的 LLM 路径在两个模型族（DeepSeek V4.1 Flash、GLM-5.3 Flash）上，保留集通过 0.95–0.96。DeepSeek Agent 在保留集上有 7.1% 的轮次（测试集 v2 上 4.4%）LLM 调用失败、由兜底路径作答，几乎都是 HTTP 429；GLM 和 LLM 组织答案都不超过 1%。
- **用不是为它写的题来测**：多轮集、测试集 v3 和两套路由标注，都由没看过代码的独立作者编写。首次运行结果照实报告：
  - 多轮集：Agent 完成 36% 的对话（0.3% 的轮次 LLM 出错，都不是 429）；
  - 独立路由标注：74%；
  - 测试集 v3：无 LLM 路径 76%。
  
  针对暴露出的问题修复之后的数字，一律标注为「暴露后」。修复后另写的一套新路由标注只跑一次，得分 80%。
- **统计如实**：每个比率都附 95% bootstrap 置信区间，路径之间做配对检验。在保留集上，无论哪个模型，LLM Agent 相对「固定流程 + LLM 组织答案」都**没有显著优势**，成本却是后者的约 1.4–6 倍。
- **按服务标准交付**：
  - 没有 LLM 也能运行；每次 LLM 调用都受运行截止时间约束；
  - DeepSeek Agent 的 P95 为 15–17 秒，首字约 3 秒；
  - 做过压测和故障演练；k3s 多副本经 Postgres 共享会话、任务和 trace；
  - 提供 MCP（服务端与客户端）和 A2A；配有 Prometheus/Grafana/Jaeger 监控。
- **为什么不直接用豆包/问财？** 见 [docs/zh/comparison.md](docs/zh/comparison.md)：它们强在哪里、哪些数据谁都没公开，以及一个设计好但尚未执行的公平对比测试。

<p align="center"><img src="docs/assets/ui/chrome-agent-trace.png" alt="Agent 回答、实时执行过程与证据面板" width="900"></p>

## 为什么这样设计

| 决策 | 理由 | 位置 |
|---|---|---|
| 经典 NLU 路由，LLM 负责编排 | 路由和守卫必须可解释、便宜；LLM 只用在它有价值的地方（选工具、组织答案）。简单问题走固定流程，对比、归因、多跳问题交给 LLM 工具循环。每个路由决定都记录理由。 | `agent/router.py`、`agent/graph.py` |
| 逐句的数字校验 | 数字必须出现在**本句**引用的证据里，按单位、写出的精度和正负号匹配。不通过时让 LLM 修改一次，仍不通过就删除该子句。 | `agent/verifier.py` |
| 处处有确定性兜底 | 没有 Key、模型故障、预算耗尽或超过截止时间时，都降级到规划器 + 模板答案，依然有引用、依然经过校验。 | `agent/planner.py`、`agent/llm.py` |
| 证据是不可信数据 | 工具结果被包装、规范化并过滤注入指令；用户消息本身也经过输入守卫；工具全部只读。 | `agent/injection.py` |

## 结果

任务成功是「一票否决」口径，以下条件缺一不可：

- 行为（回答/澄清/拒答）正确；
- 每个必需事实都陈述**且**引用；
- 调用了必需工具；
- 判断类问题有限定语；
- 全文没有交易指令。

在线结果每个任务重复 3 次，成本为网关账单口径。每个数字都来自 [`evaluation/results/`](evaluation/results/) 里已提交的文件，文件中记录了 commit、Prompt 版本和运行命令。分类明细和全部配对比较见 [docs/zh/evaluation.md](docs/zh/evaluation.md)（中文摘要）和 [docs/agent-eval.md](docs/agent-eval.md)（完整版）。

### 任务集与编写者

| 任务集 | 规模 | 编写者 | 状态 |
|---|---|---|---|
| 开发集 | 271 个任务 | 项目作者 | 用来驱动修复 |
| 保留集 | 53 个任务 | 项目作者，在规则调优之后编写 | 参与过 Prompt 选择，属于验证集 |
| 测试集 v2 | 121 个任务 / 174 轮 | 项目作者，一次性盲写 | 首次运行后读过失败类别并在开发集风格任务上修复，第二轮起属于**暴露后** |
| 多轮集 v1 | 49 段对话 / 206 轮 | 没读过路由代码和任何任务文件的独立作者（[编写说明](evaluation/agent_eval/tasks/README_multiturn_v1.md)） | 首次运行见下；之后做过修复，属于暴露后 |
| 测试集 v3 | 130 个任务 / 155 轮 | 独立作者，规则同上（[编写说明](evaluation/agent_eval/tasks/README_test_v3.md)） | **未被触碰**：没有任何修复看过它 |
| 独立路由标注 v1 / v2 | 154 / 241 条 | 独立作者按书面策略标注（[v2 说明](evaluation/agent_eval/tasks/README_router_labels_independent_v2.md)） | v1 首次运行后已暴露；v2 只跑过一次 |

### 在线 LLM 路径（DeepSeek 与 GLM，commit `d1c007c`）

任务成功率，附 95% 置信区间。来源：

- 保留集和测试集 v2：`ablation-final2-deepseek.json`、`ablation-final2-glm.json`，commit `d1c007c`；
- 开发集一列：`ablation-final.json`，commit `846bc5e`。

| 回答路径 | 开发集 · DeepSeek | 保留集 · DeepSeek | 测试集 v2 · DeepSeek | 保留集 · GLM | 测试集 v2 · GLM | 单任务成本（DeepSeek 保留集） | P95（DeepSeek 保留集） | LLM 出错轮次，保留集 / 测试集 v2（DeepSeek · GLM） |
|---|---|---|---|---|---|---|---|---|
| 原 `/chat`，无 LLM | 0.256 | 0.208 [0.11, 0.32] | 0.223 [0.15, 0.30] | 0.208 | 0.223 | – | 0.6 秒 | –（无 LLM） |
| 确定性固定流程（无 LLM） | 0.981 | 0.906 [0.83, 0.98] | 0.826 [0.76, 0.89] | 0.906 | 0.826 | $0 | 0.4 秒 | –（无 LLM） |
| 固定流程 + LLM 组织答案 | 0.979 | 0.956 [0.90, 1.00] | 0.860 [0.80, 0.92] | 0.962 [0.91, 1.00] | 0.857 [0.80, 0.91] | $0.00084 | 15.7 秒 | 0.000 / 0.000 · 0.000 / 0.000 |
| **LLM Agent（工具循环）** | **0.986** | **0.962** [0.93, 0.99] | **0.901** [0.85, 0.95] | **0.950** [0.91, 0.99] | **0.846** [0.79, 0.90] | $0.00115 | 20.4 秒 | **0.071 / 0.044**（HTTP 429：12 个出错标记中 11 个、23 个中 21 个） · 0.006 / 0.010（无 429） |

- **LLM 出错轮次**指某次 LLM 调用失败、由确定性路径兜底作答的轮次（`llm_error_rate`）。DeepSeek Agent 的出错几乎都是网关 HTTP 429：保留集 7.1%、测试集 v2 4.4% 的 Agent 轮次部分是兜底答案，所以这两格混合了 Agent 和模板路径。开发集一列（`ablation-final.json`）早于该指标，未记录。[agent-eval.md](docs/agent-eval.md) 在每张在线表格旁都给出这一比例。
- **单靠 LLM 一题都过不了**：不调工具时，在开发集、保留集和测试集 v2 共 381 个任务上一题都过不了（`ablation-final.json`、`ablation-test_v2-deepseek.json`）；在测试集 v3 的 130 个任务上也是 0（`ablation-test_v3-purellm-deepseek.json`，`3730408`）。它无法引用证据，报出的价格也无法核实，所以在要求引用的评分下它的 0 是由评分方式决定的。不要求引用、工具和免责声明字段时，它的回答里有 4–8% 的必答数字与快照一致（`fact_stated`）；按已提交的失败记录推算，任务级“不引用正确率”在开发集、保留集和测试集 v2 上介于 [0.20, 0.82]，测试集 v3 上介于 [0.00, 0.70]，此后的运行会精确记录（[agent-eval.md](docs/agent-eval.md#the-no-tools-llm-baseline-strict-scoring-vs-uncited-correctness)）。
- **Agent 与 LLM 组织答案相比**：
  - 保留集：两个模型都没有显著差异。DeepSeek +0.006 [−0.050, +0.063]；GLM −0.013 [−0.076, +0.057]。
  - 测试集 v2（暴露后）：DeepSeek Agent 单次成功率领先 +0.041 [+0.006, +0.083]，pass^3 不显著（+0.033 [−0.025, +0.091]）；GLM 没有差异，−0.011 [−0.050, +0.028]。
  - 所以 `mode=auto` 是成本与延迟上的选择，不是已被证明的质量提升。
- **GLM 的延迟**：Agent 的 P95 为 67–78 秒，GLM 组织答案为 17 秒。
- **撤回 Prompt v2/v3 提升质量的说法**：差异落在运行间波动之内。有数据支撑的是成本：v1 → v2 Agent 开发集单任务成本 −69%。
- **待重跑**：在第四轮最终 commit 上重跑这些路径（并加入测试集 v3 和多轮集 v1），因 ClinePass 周额度用完而**待定**。当时每次调用都返回 HTTP 429，测到的只是降级路径，所以没有提交。评测工具现在会把这类运行标记为无效，而不是当作结果输出。

### 独立任务集：首次运行与暴露后

| 任务集 | 首次运行（可作为估计） | 暴露后（调过，不能作为估计） |
|---|---|---|
| 多轮集 v1，确定性路径 | 任务 0.224 [0.12, 0.35]，轮次 0.709（`multiturn_v1-auto-nollm-first-run.json`，`1bd1932`） | 任务 1.000，轮次 1.000（`multiturn_v1-auto-nollm-after-fixes.json`，`7513376`） |
| 多轮集 v1，DeepSeek | Agent 任务 0.361 [0.24, 0.49]，pass^3 0.286，轮次 0.795；组织答案任务 0.286，pass^3 0.245；LLM 出错轮次：Agent 0.003（无 429），组织答案 0.000（`ablation-multiturn_v1-deepseek-first-run.json`，`527a611`） | 待重跑 |
| 独立路由标注 v1（154 条） | 0.740（`router_eval-independent_v1-first-run.json`，`882745d`） | 1.000（`router_eval-round4-independent-after-exposure.json`，`075caad`） |
| 独立路由标注 v2（241 条，新写） | **0.801**，在第四轮路由改动之后（`router_eval-independent_v2-first-run.json`，`3080bfe`） | – |
| 项目自己的路由标注（不独立） | 162 条上 0.988（`router_eval-round3b.json`） | 303 条上 1.000（`router_eval-round4-own.json`） |
| 测试集 v3，确定性路径 | 任务 0.762 [0.68, 0.83]，轮次 0.794（`test_v3-auto-nollm-first-run.json`，`882745d`） | –（未被触碰） |
| 声明核查，保留声明（47 条） | 结论准确率 **0.936 [0.851, 1.000]**，逐个数字 0.944（`claim_bench-holdout.json`，`2fcb4f0`） | 开发声明调优后 0.527 → 1.000（`claim_bench-dev-baseline.json`、`claim_bench-dev.json`） |

第三轮最有价值的发现，是作者自己写的路由标注（0.988）和第一套独立标注（0.740）之间的差距：规则贴合的是作者自己能想到的问法。之后的每个修复都归纳成一类策略，再用新的独立标注衡量。多轮集的经历也是同样的规律，见 [docs/presentation/agent-design-notes.md](docs/presentation/agent-design-notes.md)。

### 其他测量

| 测量 | 结果 | 证据 |
|---|---|---|
| 校验器压力测试：202 个正确答案、3,399 个篡改变体 | 误放率：原来的整批比对 33.3%、运行级 24.4%、逐句绑定 **1.94%**（允许派生数字时 2.03%）；公司之间互换数字 100% → **0.53%**；正确答案全部通过 | `verifier_stress.json`（`9f0e46b`） |
| 校验失败后的修复（3,333 个被拒变体） | 整句删除：100% 可读且通过校验，0% 残句，未被篡改的句子保留 98.0%，18.3% 回退到模板答案；原来的按子句拼接（`2494656` 时测得）只有 29% 可读，97% 带残句 | `verifier_stress.json`（`9f0e46b`） |
| 提示注入红队：攻击投毒到搜索结果，每种 4 种混淆 | **在线，DeepSeek，`d1c007c`**（`redteam-final2.json`）：开发攻击集三条路径都是 0/72；未见过的保留集：模板 0/64、LLM 组织答案 1/64、Agent 0/64；holdout2：0/64、1/64、0/64。该 commit 的红队没有记录 LLM 出错次数，此后每条路径都会记录（`llm_error_rate`、`llm_429_rate`）。**离线模板路径，`9f0e46b`**（`redteam-offline.json`）：holdout3（第二轮评审投放的 11 种攻击）2/88 | `redteam-final2.json`、`redteam-offline.json` |
| 故障注入：超时、5xx、空数据、超大文档、LLM 宕机、畸形工具参数、无限循环 | 11/11 个场景平稳降级 | `fault_injection.json` |
| Agent 延迟，DeepSeek，保留集 / 测试集 v2 | P95 27.1 → **15.4 秒** / 24.2 → **17.3 秒**；首字 P50 约 5.5 → **3.0 / 2.8 秒**；每轮 LLM 调用 2.30 → 1.39；成功率没有下降：与关掉开关的同一份代码配对比较，保留集 Δ +0.006 [0.000, +0.019]，测试集 v2 Δ +0.003 [−0.005, +0.014]；最终一轮 LLM 出错轮次 0.000 / 0.000（`perf-merged-prefetch-deepseek.json`），基线为 0.018 / 0.006，都不是 HTTP 429 | [performance.md §2a](docs/zh/performance.md#2a-agent-路径延迟剖析改动与前后对比)、`perf-*.json` |
| 压测：LLM Agent 路径，4 并发，流式 | 更难的多工具问题上 P95 26.7 秒（关掉开关时 29.6 秒），0 错误，24 个请求中 0 个 LLM 出错，每 1,000 次提问 ¥12.6 | `docs/results/perf/agent/load_test-agent-4-*.json` |
| 压测：确定性路径 | 每次运行只写一次检查点：单用户 4.8 → 6.5 次/秒，同样 820 次请求后会话库 64 MB → 9.4 MB。之前「11 倍」的说法复现不出来，实际约 1.3–1.6 倍 | [docs/zh/performance.md](docs/zh/performance.md) |
| 启动 | 冷启动构建服务 24.1 秒，进程内重建 3.7 秒，磁盘有索引缓存时重启 6.8 秒；容器从 `docker run` 到 `/ready` 返回 200 的中位数 46 秒 | `docs/results/perf/startup.json`、`startup-container.json` |
| k3s + Postgres 共享会话 | 1 → 3 副本：32 并发下 3.75 → 11.72 次/秒，0 错误；发往 B 副本的追问正确解析了 A 副本上一轮的「它」；A2A 任务和 trace 同样共享，在副本 1 暂停的任务能在副本 2 恢复 | [docs/zh/performance.md](docs/zh/performance.md)、`docs/results/protocols/` |
| 真实网关与数据源上的故障演练 | **LLM**：主模型失效 → 熔断打开 → GLM 接手 → 半开试探 → 关闭。**数据源**：屏蔽新浪/腾讯/东方财富 → 60 秒缓存 → 最近一次成功数据 → 过期后如实说明缺数据，而不是拿 4 月的快照价冒充行情 | [docs/zh/a2a-and-observability.md](docs/zh/a2a-and-observability.md#故障演练) |
| 实时数据源审计：64 次探测（加盘中探测后 67 次） | 49 次成功，10/10 条降级链正常；三次已提交的运行（09-28 下午、夜间、09-29 盘中：52/67、10/10）失败的都是同样 15 个探测；修复了错误的 M2 序列、停更的 CPI/PMI、始终为空的 PE/PB 和取不到的公告；新浪/同花顺增速与报告期绝对值交叉核对 | `docs/results/data_sources/audit-20260928-6dde495.json`、`audit-20260928T1955Z-4742453.json`、`audit-20260929T0257Z-5d4c192.json`、[docs/zh/data-sources.md](docs/zh/data-sources.md) |

## 架构

```mermaid
flowchart LR
  U["浏览器 (React) / API / A2A 客户端"] --> API["FastAPI"]
  API --> G["guard_in：输入守卫、经典 NLU、指代与省略补全、可解释路由"]
  G -->|超出范围 / 只有注入指令| RF["拒答"]
  G -->|缺少标的| CL["澄清（中断 / 恢复）"]
  G -->|简单| WF["确定性规划器"]
  G -->|复杂| AL["LLM 工具循环（LangGraph）"]
  WF --> T["9 个类型化工具（同时经 MCP 提供）"]
  AL <--> T
  T --> DS["实时数据源：降级链、熔断、交叉核对、来源标注"]
  WF --> C["组织答案（LLM 或模板）"]
  AL --> V["校验：逐句的引用与数字"]
  C --> V
  V -->|不通过| RV["修改一次，再修复"] --> V
  V --> K["合规 + 语言守卫"] --> F["汇总：答案、证据、trace、成本"]
  F --> U
  F -.-> O["trace：JSON / OTLP · Prometheus /metrics · 用户反馈"]
```

会话保存在 LangGraph checkpointer 中（内存、SQLite 或 Postgres），因此澄清可以暂停一次运行，任意副本都能接着对话；会话归属于创建它的 API Key。详见 [docs/zh/agent.md](docs/zh/agent.md)。

## 功能

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
  - 处理比较词（超过/不到/以上/之间）、否定、区间、同比增速和中文数字；47 条 held-out 说法上结论准确率 0.936（95% CI 0.851–1.000），见 [docs/zh/claim-check.md](docs/zh/claim-check.md)。
- **实时数据**：
  - 东方财富 → 新浪 → 腾讯 → 缓存 → 最近一次成功数据 → 快照的降级链，每个数据源独立熔断；
  - 上游调用走有界线程池；新浪与同花顺基本面交叉核对；
  - 每条记录标注来源与时效；`GET /sources/health?probe=1` 主动探测（限频）。
- **协议**：工具经 MCP 提供；MCP 客户端可以把外部 MCP 服务器的工具注册进来（`QI_MCP_SERVERS`，结果按不可信数据处理）。整个 Agent 经 A2A 1.0 提供：澄清对应 `input-required`，流式接口汇报进度，任务表经 Postgres 共享。仓库里附带一个 a2a-sdk 客户端示例（`scripts/a2a_client_demo.py`）。
- **可观测性**：
  - 每次运行的 trace：节点、工具、LLM 调用（含上下文构成与 JSON 解析状态）、token、成本、Prompt 版本；
  - 运行查看接口；OpenTelemetry 导出；
  - 按实际作答模型打标签的 Prometheus 指标；27 个面板的 Grafana 看板（含按 Prompt 版本划分的校验失败率与修复率、用户好评率）、13 条告警规则、Jaeger；
  - 审计日志：每次拒答和每次合规改写各记一条结构化事件，只含哈希 id，不含用户原文；
  - 用户反馈（`POST /agent/feedback`）由 `scripts/feedback_to_tasks.py` 转成候选评测任务。
- **网页前端**（React 19、TypeScript、Tailwind v4、Radix、Motion、Lightweight Charts）：
  - 首字出现前有实时进度面板：当前步骤（规划 → 取数 → 撰写 → 核对）、正在调用的工具及其标的、已用时间和「停止」按钮；随后答案边写边显示，旁边是执行时间线；运行详情里显示首字耗时和总耗时；
  - 声明核查页面：逐个数字显示比较关系、声称值与实际值、来源和日期；聊天中出现「听说……是真的吗」时会提示一键核查；
  - 与证据面板联动的引用标签（含时效）；价格图；运行成本与延迟；
  - 反馈按钮与 Markdown 导出；中英文、深色模式、移动端。
- **安全**：
  - 可选 API Key，会话与 trace 按 Key 隔离（存的是哈希，不是 Key 本身）；
  - 限流、CORS、请求体上限；针对用户消息中指令的输入守卫；
  - 非 root 且根文件系统只读的容器；Kubernetes 清单带 NetworkPolicy，不含明文 Secret；
  - CI 用 gitleaks 扫描全部 git 历史，pre-commit 钩子做同样的扫描（[SECURITY.md](SECURITY.md)）。

<p align="center">
  <img src="docs/assets/ui/chrome-run-details.png" alt="运行详情：token、成本、延迟" width="440">
  <img src="docs/assets/ui/chrome-mobile-dark-en.png" alt="移动端、深色、英文" width="200">
</p>

## 快速开始

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

实时行情、新闻、公告、宏观默认开启；设置 `QI_USE_LIVE_MARKET=0`（以及 `_NEWS`、`_ANNOUNCEMENT`、`_MACRO`）即使用随仓库提供的离线快照。

Docker、Kubernetes（多副本经 Postgres 共享会话、只读根文件系统）与监控栈见 [docs/zh/deployment.md](docs/zh/deployment.md)。

```bash
docker build -f docker/Dockerfile -t finsight:$(git rev-parse --short=7 HEAD) .   # 镜像按 commit 打标签
kubectl apply -k deploy/k8s        # 先创建 finsight-db Secret（见 docs/zh/deployment.md#密钥）
docker compose -f docker/docker-compose.yml --profile monitoring up -d   # 加上 Prometheus、Grafana、Jaeger
```

## API

| 接口 | 用途 |
|---|---|
| `POST /agent/chat`、`POST /agent/chat/stream`（SSE）、`POST /agent/resume` | 带证据、校验结果、trace id 和成本的 Agent 回答；节点事件和边写边出的 `answer_delta` 文本；澄清后恢复。 |
| `POST /agent/claim-check` | 核查一句市场说法里的数字（确定性，不用 LLM）。 |
| `POST /agent/feedback` | 对回答点赞/点踩，与对应 trace 一起保存。 |
| `GET /agent/sessions/{id}`、`GET /agent/traces`、`GET /agent/traces/{id}` | 会话记忆、最近运行、完整 trace，只返回调用方自己的。 |
| `GET /.well-known/agent-card.json`、`POST /a2a` | A2A 服务卡片与 JSON-RPC 接口。 |
| `GET /metrics`、`GET /sources/health[?probe=1]` | Prometheus 指标；数据源状态（按需主动探测全部 19 个数据源，未探测的会标出）。 |
| `GET /health`、`GET /ready` | 存活检查；就绪检查（检查点存储可连接且可写、LLM 配置、检索索引），未就绪返回 503。 |
| `POST /chat` | 原聊天接口（`mode=workflow` 保持原流程；`agent`/`auto` 走 Agent）。 |
| `POST /nlu/analyze`、`POST /retrieval/search`、`POST /query/intelligence` | 经典 NLU 与检索产物。 |

Schema 位于 `schemas/agent_*.schema.json`，由 `query_intelligence/contracts.py` 生成。

## 评测与测试

```bash
python -m pytest -q tests                                  # 离线；除非 QI_TEST_LIVE=1，否则不访问实时数据源
python -m evaluation.agent_eval.gate                       # 快照回放的开发集 + 保留集，与已提交基线比较
python -m evaluation.agent_eval.router_eval                # 路由准确率与混淆矩阵
python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3 --workers 3 --sets holdout,test_v2
python -m evaluation.agent_eval.verifier_stress            # 校验器误放率
python -m evaluation.agent_eval.redteam --llm deepseek     # 提示注入红队
python -m evaluation.agent_eval.fault_injection            # 平稳降级
python -m evaluation.agent_eval.report --check             # docs/agent-eval.md 与 evaluation/results/ 一致
python -m scripts.load_test --base-url http://127.0.0.1:8000 --users 8
python -m scripts.chaos_drill --scenario sources           # 对运行中的服务屏蔽上游数据源
```

`--llm deepseek` 选择的是客户端，即由 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 配置的 OpenAI 兼容客户端，不决定模型。模型依次取自 `--model`、`DEEPSEEK_MODEL`、`config/app_config.json` 中的 `deepseek.model`；因此 GLM 的运行命令也是 `--llm deepseek`，模型由 `DEEPSEEK_MODEL=cline-pass/glm-5.3-flash` 指定。每个结果文件都记录实际调用的模型（`config.model`）、客户端（`llm_client`）和模型来源（`model_source`）。

CI 包括：

- 代码检查；
- 用 gitleaks 扫描全部 git 历史中的密钥（[SECURITY.md](SECURITY.md)）；
- 前端检查（类型、lint、单测、可复现构建）；
- 带 Postgres 服务的全量测试，包括 axe 无障碍测试（在 CI 里缺依赖会失败而不是跳过），以及 `query_intelligence/agent` 88% 的覆盖率下限（实测 90.3%）；
- 与已提交基线比较的评测门禁，以及评测文档是否最新的检查；
- Docker 构建与只读根文件系统下的冒烟测试（等待 `/ready`），以及「状态目录不可写时容器不就绪」的检查；
- Kubernetes 清单校验（渲染后的 kustomization、不提交 Secret、镜像按 commit 打标签）。

`pre-commit install` 后每次提交前会跑 gitleaks、ruff 和评测文档检查。

## 文档

| 主题 | 链接 |
|---|---|
| Agent 层：图、工具、记忆、API、配置 | [docs/zh/agent.md](docs/zh/agent.md) |
| 评测（中文摘要） | [docs/zh/evaluation.md](docs/zh/evaluation.md) |
| 评测完整版：任务集、置信区间、在线消融、第二模型、Prompt A/B、红队、校验器压力测试 | [docs/agent-eval.md](docs/agent-eval.md)（英文） |
| 与问财、豆包、Kimi、Wind Alice、妙想的对比 | [docs/zh/comparison.md](docs/zh/comparison.md) |
| A2A、模型容灾、网关成本、监控、看板、故障演练 | [docs/zh/a2a-and-observability.md](docs/zh/a2a-and-observability.md) |
| 实时数据源 | [docs/zh/data-sources.md](docs/zh/data-sources.md) |
| 性能、压测与扩展 | [docs/zh/performance.md](docs/zh/performance.md) |
| 部署 | [docs/zh/deployment.md](docs/zh/deployment.md) |
| 设计复盘：取舍、自研与复用、失败案例 | [docs/presentation/agent-design-notes.md](docs/presentation/agent-design-notes.md) |
| 面试讲稿：30 秒介绍、三个可辩护的数字、失败故事、白板提纲 | [docs/presentation/interview-script.md](docs/presentation/interview-script.md) |
| 声明核查：规则、比较词、基准 | [docs/zh/claim-check.md](docs/zh/claim-check.md) |
| Agent 与 Prompt 工程实践调研 | [docs/research/agent-architecture-practices-2026.md](docs/research/agent-architecture-practices-2026.md)（英文） |
| Query Intelligence（经典 NLU 与检索） | [docs/zh/query-intelligence.md](docs/zh/query-intelligence.md) |
| 网页前端 | [docs/zh/frontend-chatbot.md](docs/zh/frontend-chatbot.md) |
| 全部文档 | [docs/zh/index.md](docs/zh/index.md) |

## 局限

- **只测了两个模型族，都是 flash 级别，都经同一个网关调用**：在线表格来自第二轮代码 `d1c007c`；在最终 commit 上的重跑因网关周额度用完而待定。
- **只有一套未被触碰的任务集**：测试集 v3 从未用于任何修复，但目前只跑了确定性路径和不调工具的 LLM。保留集参与过 Prompt 选择；测试集 v2、多轮集 v1、独立路由标注 v1 都属于暴露后。
- **路由在新问法上仍约有五分之一出错**：新写的独立路由标注得分 0.801，主要错在没有标的的建议类问题，以及澄清与拒答之间的边界情况。
- **Agent 工具循环没有被证明优于「LLM 组织答案」**：保留集上两个模型都不显著，而 Agent 成本是后者的 1.4–6 倍。
- **校验证明的是可追溯，不是真实**：校验器检查数字是否来自所引证据；一条证据含多个报告期或指标时，它无法判断选对了哪一个。投毒到新闻正文里的假价格能通过校验，因为它确实「在证据里」。
- **注入过滤器不泛化**：关键词过滤对没见过的说法不泛化（保留攻击集拦截率 0%）；防护主要来自结构：只读工具、不可信数据信封、数字校验、合规节点、语言守卫。
- **追问处理基于规则**：会话规则处理代词、复数、指代组和省略，每次改写都有记录；它们的词表来自作者本人和已暴露任务集能想到的问法。英文公司别名只覆盖主要公司。
- **延迟**：用 DeepSeek 时，Agent 的 P95 在保留集 15.4 秒、测试集 v2 17.3 秒，首个答案 token 约 3 秒出现（P50）。规划器预取、引用修复和 20 秒卡顿超时让 P95 从 22–27 秒降下来，任务成功率没有下降（[performance.md](docs/zh/performance.md#2a-agent-路径延迟剖析改动与前后对比)）。在 4 个并发用户、更难的多工具问题上，P95 为 26.7 秒。GLM 的 P95 仍约 60–70 秒，由单次调用的波动决定。每次 LLM 请求都受运行截止时间约束（90 秒 + 作答宽限 20 秒，低于接口的 120 秒超时）。
- **免费数据源会限流**：审计时东方财富拒绝了本机连接，由降级链兜底。会话存储用 Postgres 时，A2A 任务表和 trace 也由所有副本共享；限流器和缓存仍是每个副本各自一份。

## 已知未修复问题

第三轮独立评审发现的问题（编号 C1–C20）中，本分支尚未修复的如下。每一条在评审报告里都有复现步骤；修复合并后本表会更新。

| 编号 | 严重度 | 问题 | 状态 |
|---|---|---|---|
| C1 | 高 | 模板回答路径会引用攻击者控制的文档标题；同形字和改写能绕过标题黑名单（192 次投毒运行中 52 次回显了攻击内容） | 修复中（第四轮） |
| C2 | 高 | 事实核查在下跌表述上把比较方向弄反：“五粮液昨天跌了超过1%”（实际 −0.53%）被判为相符 | 修复中（第四轮） |
| C3 | 中 | 关闭 API key 时（Kubernetes 默认配置）所有调用方共用一个身份，`/agent/traces` 会列出他人的问题和会话 id | 修复中（第四轮） |
| C5 | 中 | 英文比较句里的 “it” 丢掉前文公司 | 修复中（第四轮） |
| C6 | 中 | 部分 A 股问题被拒答或无法解析（“美的和格力选哪个”“北向资金是啥”） | 修复中（第四轮） |
| C7 | 低 | 讨论两家公司后问“三家里哪家最好”只继承最后一家 | 修复中（第四轮） |
| C8 | 低 | 错别字（“贵州矛台”）能否识别取决于句子其余部分 | 修复中（第四轮） |
| C9 | 低 | 注入话术被当成实体（“…荐股机器人…” → `get_fundamentals('机器人')`） | 修复中（第四轮） |
| C10 | 低 | 板块估值问题（“半导体板块现在估值高吗”）被当成个股处理 | 修复中（第四轮） |
| C11 | 低 | 模板句子缺单位（“成交额 3793827534”“1688.38 hundred million CNY”） | 修复中（第四轮） |
| C12 | 低 | 明确的回答语言要求（“请用英文回答”）被忽略 | 修复中（第四轮） |
| C13 | 低 | 事实核查无法核对 “x earnings”、两家公司之间的比较和宏观序列 | 修复中（第四轮） |
| C14 | 低 | 输入守卫改写后仍作答的注入尝试，审计日志里没有对应事件 | 未修复 |
| C16 | 低 | “逐子句挽救只有 29% 可读”这一数字（README 校验器一行）只出现在 commit `063aeca` 的提交说明里，没有已提交的结果文件 | 未修复 |
| C17 | 低 | 无障碍：每轮回答有重复的 region 标签，页面有两个 `main` 元素 | 未修复 |
| C18 | 低 | 比较类回答只显示第一家公司的 KPI 卡片 | 未修复 |
| C19 | 低 | 网页界面把 API key 存在 `localStorage` | 未修复 |
| C20 | 低 | “美联储加息对A股有什么影响”返回空答案，也没有说明局限 | 未修复 |

本分支已修复：C4（`docs/agent-eval.md` 渲染 README 引用的每个结果，否则 `report --check` 失败，`2d368da`）；C15（每个在线主结果旁都标出 LLM 出错及 HTTP 429 比例；无 429 的重跑仍在等额度）；C16 的其余项（`--llm` 指客户端，结果配置记录 `model`；混沌演练记录 commit，`0223b25`）。新增的盘中行情路径不认识农历等浮动假日（春节、清明、端午、中秋）：这些日子里它会拒绝日期不对的实时报价，退回日线收盘价（[data-sources.md](docs/data-sources.md#intraday-quotes-for-今天今日today-questions)）。

## 安全声明

FinSight 只做证据汇总，不是投资顾问，不能作为交易决策的唯一依据。
