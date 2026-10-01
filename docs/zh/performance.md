# 性能与压测

语言：[English](../performance.md) | 中文

本页记录两条回答路径实测的吞吐、延迟和成本，检查点瓶颈及其修复，以及在 k3s 上的多副本扩展。每个数字都对应 [`docs/results/perf/`](../results/perf/) 下一个已提交的 JSON 文件，文件里记录了命令、运行时间、压测脚本的 commit、服务镜像，以及开始时主机的负载。

## 测试环境与镜像

- **主机**：Apple Silicon Mac（10 核），Docker 与 k3s v1.35 通过 colima 运行（虚拟机 4 vCPU、8 GB）。
- **主机并不空闲**。每次测量时都有其他任务在跑：一个区块链节点占满一个核，另外几个工作目录在跑测试和在线评测。JSON 里记录的负载在 6 到 38 之间，所以绝对数字偏保守、噪声较大，只应在同一组测量内比较。
- **「当前」各行的服务镜像**（2026-09-26）：分支 `r2-ops@8dc388b` 与 `round2@d04a42d` 合并，加上运维接线补丁 `deploy/patches/app-ops-wiring.patch`（其改动从 `e9d9a9a` 起已在 `api/app.py` 里，所以文件已删除；可用 `git show 47dd024:deploy/patches/app-ops-wiring.patch` 查看），镜像名 `finsight:merged`。此后的测量结果都只对应单个 commit（例如 [`startup-container.json`](../results/perf/startup-container.json)）。
- **「修复前」镜像**：`git archive 0678585`（检查点修复之前的 commit），镜像名 `finsight:before-0678585`。
- **压测脚本**：[`scripts/load_test.py`](../../scripts/load_test.py)，闭环压测：N 个客户端各自连续向 `POST /agent/chat` 发请求，每个请求一个新会话。分位数用最近秩法，所以样本少于 100 时 P99 就是最大值。

## 1. 确定性路径（无 LLM）：复现检查点修复

测的是经典 NLU、确定性规划器、离线工具、模板答案、校验器和合规节点，实时数据关闭。问题轮换覆盖以下几类：

- 价格、估值、对比、「为什么」；
- 英文、宏观、超范围；
- 一个需要澄清的悬空指代。

每种配置一个容器，会话库是新的 SQLite 文件，每个用户 20 次请求。驱动脚本：[`docker/perf_matrix.sh`](../../docker/perf_matrix.sh)。

| 配置 | 并发 | 请求数 | 吞吐（次/秒） | P50（ms） | P95（ms） | P99（ms） | 错误 | 820 次请求后的会话库 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 修复前：commit `0678585`（每步写检查点、状态未裁剪） | 1 | 20 | 3.93 | 86 | 1,169 | 1,550 | 0 | |
| | 8 | 160 | 5.01 | 1,613 | 2,501 | 2,799 | 0 | |
| | 32 | 640 | 4.51 | 6,121 | 12,816 | 14,740 | 0 | 64 MB |
| 当前代码，`QI_AGENT_DURABILITY=async` | 1 | 20 | 4.80 | 71 | 933 | 1,319 | 0 | |
| | 8 | 160 | 4.94 | 1,659 | 2,376 | 2,574 | 0 | |
| | 32 | 640 | 4.44 | 7,024 | 10,816 | 14,901 | 0 | 64 MB |
| 当前代码，`QI_AGENT_DURABILITY=exit`（默认） | 1 | 20 | **6.47** | **43** | 733 | 1,095 | 0 | |
| | 8 | 160 | **6.22** | 1,030 | 2,937 | 4,706 | 0 | |
| | 32 | 640 | **5.16** | 5,494 | 12,298 | 18,843 | 0 | **9.4 MB** |

文件：
- 压测结果：`results/perf/workflow/load_test-{before-0678585,async,exit}-{1,8,32}.json`；
- 容器设置与 `ls -la /app/state` 的输出：`container-*.txt`。

运行时间：2026-09-26 09:56–10:06 UTC。

说明了什么，没说明什么：

- **起作用的是 `durability="exit"`（每次运行只写一次检查点）**。同一 commit 上 `async` 与 `exit` 对比：
  - 吞吐：单用户 4.8 → 6.5 次/秒，8 并发 4.9 → 6.2，32 并发 4.4 → 5.2；
  - 会话库：同样 820 次请求，从 64 MB 降到 9.4 MB，小了 7 倍。

  「修复前」镜像（每步写检查点，最终状态也没裁剪）的表现和 `async` 差不多。
- **8 和 32 并发下，`exit` 的 P95/P99 尾延迟并没有更好**。32 并发时每种配置都受单进程 CPU 限制，尾延迟主要由排队和主机上的其他负载决定，而这些负载在几次运行之间有变化（负载 6.4–9.3）。
- **更正之前的数字**：本页上一版（2026-09-25）写的是单用户「修复前」0.73 次/秒、P95 4.6 秒，「修复后」8.36 次/秒，约 11 倍。那次「修复前」测量没有留下可提交的产物，而且和在线评测跑在同一台虚拟机上。上面的复现没有 11 倍：实测效果是低并发下吞吐约 1.3–1.6 倍，会话库小 7 倍。之前的「修复后」JSON 保留在 `results/perf/workflow/history-2026-09-25/` 以便追溯。

## 1b. 数据源故障演练下的确定性路径（无 LLM，8 个用户）

第七轮评审要求在数据源出故障时做一次压测，这不需要 LLM 额度。`scripts/chaos_drill.py --scenario sources-load`
在演练的拦截代理后面启动实时数据服务（`QI_USE_LIVE_*=1`，不配 LLM key），每个阶段跑一次闭环的
`scripts/load_test.py`（固定流程模式，8 个用户 × 默认轮换的 10 个问题）：实时，然后拦截新浪 / 腾讯 / 东方财富
（60 秒行情 TTL 之内、之后、90 秒过期窗口之后），再在熔断冷却后解除拦截。运行于 2026-10-01 02:51–02:57 UTC，
提交 `3641361`（工作区干净），结果见
[`results/perf/sources-load/chaos-sources-load-8users.json`](../results/perf/sources-load/chaos-sources-load-8users.json)，
服务日志在同一目录。

| 阶段 | 请求数 | 吞吐（req/s） | P50（ms） | P95（ms） | P99（ms） | 错误 | 证据由谁提供（条数） |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 实时 | 80 | 2.96 | 975 | 12,448 | 14,813 | 0 | 新浪 K 线 `live_fallback` 50（东方财富行情熔断已打开），同花顺财务 / 行业 `live` 20 / 20，东方财富宏观 `live` 10，新浪财务 `live_fallback` 10 |
| 2a 拦截，TTL 之内 | 80 | 3.82 | 1,434 | 5,123 | 6,852 | 0 | 同样的条目，来自 TTL 缓存 |
| 2b 拦截，TTL 之后 | 80 | 3.59 | 1,246 | 6,983 | 8,098 | 0 | 新浪 K 线 **`last_known_good` 50**；打开的熔断：新浪 K 线、新浪行情、腾讯 K 线、efinance、东方财富行情 |
| 3 拦截，过期窗口之后 | 80 | 3.86 | 1,258 | 6,470 | 7,959 | 0 | **没有价格**：`get_price_history` 失败（`upstream_error`，约 0.3 秒，熔断已打开），回答说明数据不可用，而不是给一个过期的数（内置快照太旧，不能替代行情） |
| 4 解除拦截，冷却之后 | 80 | 2.30 | 1,525 | 10,361 | 15,419 | 0 | 新浪 K 线 37 / 腾讯 K 线 12 / 新浪行情 1，均为 `live_fallback`；最初几次价格请求耗时 7–9 秒（半开试探和重新获取） |

这说明了什么，没说明什么：

- **400 个请求 0 错误**，每个回答都通过校验，而被拦截的行情源全部失败：降级链（TTL 缓存 → last-known-good →
  明确说明“无数据”）在每个阶段都没有出现 5xx 或超时。
- 长尾来自实时获取，而不是拦截：P95 在实时和恢复阶段最高（12.4 秒和 10.4 秒，每个数据包的首次获取），拦截期间最低
  （5–7 秒），因为被拦截的调用很快失败，打开的熔断会直接跳过。
- 宏观回答一直是 `eastmoney.datacenter/live`，因为它的 TTL 是 6 小时（缓存的实时值）；同花顺的域名不在拦截名单里，
  所以基本面一直是实时的。完整走过一遍的只有价格链。
- 主机同时在跑其他任务（开始时负载 11.6，阶段 3 时 25.9，10 个 CPU），所以绝对延迟偏悲观，不能和第 1 节（离线数据，
  负载 6–9）比较。这是一次韧性测量，不是容量测量。
- 第一次运行作废：后面的阶段用新的客户端（新的匿名身份）复用了第一阶段的会话 id，服务按租户规则返回 404。压测脚本
  现在每次运行使用唯一的会话 id（`39baa79`）。

## 2. LLM Agent 路径：吞吐、延迟与成本

服务配置：合并后的镜像在本地 8801 端口运行，环境如下：

- 模型：`DEEPSEEK_MODEL=cline-pass/deepseek-v4.1-flash`，`QI_LLM_FALLBACK_MODELS=cline-pass/glm-5.3-flash`，经 Cline 网关调用；
- 实时数据关闭（离线快照），所以测的是 Agent 和 LLM，而不是上游网站；
- `mode=agent`，问题集用 `research`：8 个需要多个工具的问题（对比、为什么、盈利能力、宏观联动、ETF、趋势、英文对比、增长）。

成本是网关上报的 `usage.cost` 按请求累加，再用 `QI_LLM_USD_CNY=6.7489` 换算：这是 2026-09-24 中国外汇交易中心受权公布的人民币汇率中间价（1 美元 = 6.7489 元；见[新浪财经](https://finance.sina.com.cn/jjxw/2026-09-24/doc-iniswvxc5391561.shtml)和外汇局中间价表）。

```bash
source /tmp/llmenv.sh   # 从 .env 读取 DEEPSEEK_*；Key 不会被打印或提交
QI_LLM_FALLBACK_MODELS=cline-pass/glm-5.3-flash QI_LLM_USD_CNY=6.7489 QI_RATE_LIMIT_PER_MINUTE=0 \
  uvicorn query_intelligence.api.app:create_app --factory --port 8801
python -m scripts.load_test --base-url http://127.0.0.1:8801 --mode agent --questions research \
  --users 4 --requests 6 --usd-cny 6.7489 --questions-per-day 2000 --timeout 240
```

| 并发 | 请求数 | 吞吐（次/秒） | P50（秒） | P95（秒） | P99（秒） | HTTP 错误 | 由 Agent 的 LLM 作答 | 降级到规划器 + 模板 | 通过校验 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 24 | 0.12 | 22.0 | 41.9 | 78.8 | 0 | 24（100%） | 0 | 79% |
| 8 | 40 | 0.42 | 18.8 | 38.1 | 43.3 | 0 | 26（65%） | 14 | 80% |
| 16 | 64 | 0.63 | 16.8 | 66.0 | 75.4 | 0 | 23（36%） | 41 | 88% |

文件：
- `results/perf/agent/load_test-agent-{4,8,16}.json`：逐请求记录路由、模型、token 和成本；
- `gateway-log-16-run2.json`：16 并发那次每个网关调用的状态和延迟，由 `scripts/chaos_drill.py` 的透传代理记录，不含请求头和提示词。

运行时间：2026-09-26 13:25–13:43 UTC，主机负载 17–38。

**上限是网关限流，不是服务本身。**
- **限流的出现**：从 8 并发开始，Cline 网关对部分调用返回 HTTP 429（限流页）。16 并发那次 190 次网关调用里有 81 次是 429，主模型和备用模型都一样，因为它们共享账号额度（另一个工作目录里的评测同时在用同一个 Key）。第一次 16 并发的尝试更是每次调用都 429（`load_test-agent-16-run1-all-llm-errors.json`：64 个请求，0 个 LLM 答案）。
- **没有失败的请求**：两个模型都被限流时，图降级到确定性规划器和模板答案（`degraded: llm_error`）。所以吞吐随并发上升、P50 下降：越来越多的答案根本没有等 LLM。
- **真正拿到 LLM 答案的请求**：P50 在 4 并发时 22.7 秒，8 并发 23.8 秒，16 并发 39.3 秒（429 之后带退避重试）。

要扩容这条路径，需要更高的网关配额或多个 Key/供应商，加副本没用。

**成本。** 4 并发那次是干净的成本测量，因为每个答案都来自 Agent 循环：

| | 数值 |
|---|---|
| 每个问题的 LLM 调用次数 | 4.1 |
| 每个问题的 token | 1.98 万 prompt（56% 命中供应商的 prompt 缓存）+ 2.6 千 completion |
| 每个问题的成本 | $0.00291 = ¥0.0197 |
| **每 1,000 次 Agent 提问** | **¥19.7**（$2.91） |
| 每天 2,000 次、全部走 Agent 路径，每月 | **¥1,180** |
| 每天 10,000 次，每月 | ¥5,900 |

在 `auto` 模式下，简单查询、拒答和澄清都不会进 Agent 循环，所以对同样的流量这些是上限。按「由 LLM 作答的问题」计，8 和 16 并发那两次每 1,000 次分别是 ¥15.3 和 ¥13.0（能通过的答案里修改循环更少）。

**尾延迟。** 从 trace 能看出尾延迟来自哪里：
- **修改调用拖了尾巴**：监控运行里最慢的 Agent trace 用了 98 秒，第一版草稿没通过校验，单是 `llm.revise` 一次调用就花了 74 秒（[Jaeger 截图](../assets/ops/jaeger-trace.png)）。
- **备用模型更慢**：LLM 故障演练中，备用模型作答需要 48–81 秒，其中一次撞上了接口的 120 秒请求超时（504），因为 GLM-5.3-flash 单次调用比 DeepSeek 慢 2–7 倍（见 [A2A、容灾与可观测性](a2a-and-observability.md#故障演练)）。
- **修复**：`d1c007c` 起，Agent 层的每次 LLM 请求（包括重试、修改和备用模型）超时都取「客户端超时」和「距运行截止的剩余时间」中较小者（工具循环 90 秒，作答再宽限 20 秒），备用路径会以确定性答案结束，而不是 504。此后又加了按数据块计的卡顿超时（20 秒），卡住的流会远早于截止时间被重试；Agent 路径也已重新测量，包括 4 个用户的压测，见[第 2a 节](#2a-agent-路径延迟剖析改动与前后对比)。

## 2a. Agent 路径延迟：剖析、改动与前后对比

第二轮评审测得：用 DeepSeek 时，held-out/test_v2 上 Agent 路径 P95 约 20–23 s，首个流式 token 约 14 s 才出现。本节剖析时间花在哪里，列出所做的改动，并给出带配对置信区间的前后对比。所有数字都来自 [`evaluation/results/`](../../evaluation/results/) 下已提交的文件，其中记录了命令、commit、prompt 哈希、逐任务结果和延迟剖析。

**方法。** `python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3 --workers 3 --sets holdout,test_v2 --modes agent --stream`（经 Cline 网关调用 DeepSeek v4.1 flash，工具结果用回放快照，评测日期固定）。`--stream` 让每一轮都走 `AgentService.stream`（即 UI 使用的 SSE 路径），并记录到第一个 `answer_delta` 事件的时间（TTFT）。每轮记录都带一份剖析：每次 LLM 调用的节点、工具循环步数、延迟，prompt、缓存命中、生成和推理 token 数，以及按字符计的上下文构成；每次工具调用；每个图节点的耗时。`llm_http` 统计 HTTP 请求次数，包括被重试救回的 429。**本页所有运行都没有出现 429。** 延迟分位数取最近秩，覆盖该集合的全部轮次（held-out 56 轮、test_v2 174 轮，各 × 3 次重复），含拒答和澄清；“LLM 轮”列只统计调用了 LLM 的轮次。成功率差异按任务配对（`metrics.paired_comparison`：2,000 次自助重采样的 95% 置信区间，pass^3 上做精确 McNemar 检验）。各次运行在一台共用的 Mac 上（负载 3–6）依次进行，互不重叠。

```bash
python -m evaluation.agent_eval.profile outputs/agent_eval/<run>.json        # 下方的剖析表
python -m evaluation.agent_eval.profile --table evaluation/results/perf-merged-defaults-deepseek.json \
    evaluation/results/perf-merged-citerepair-stall-deepseek.json evaluation/results/perf-merged-prefetch-deepseek.json
```

### 时间花在哪里（基线，[`perf-baseline-deepseek.json`](../../evaluation/results/perf-baseline-deepseek.json)，`2dbb026` 的代码）

| | held-out | test_v2 |
|---|---:|---:|
| 每个 LLM 轮的 LLM 调用数 | 2.7 | 3.0 |
| LLM 节点（agent_llm + revise）占一轮耗时的比例 | 95% | 95% |
| 工具循环第 0 步（选工具）：P50 延迟、prompt token | 2.3 s、2.6k | 2.3 s、2.7k |
| 工具循环第 1 步（通常是作答）：P50、prompt token | 3.0 s、3.6k | 3.0 s、3.8k |
| 需要第 2、3 …… 轮工具调用的轮次 | 21% | 31% |
| **修订率**（草稿未通过校验 → 再调一次 LLM） | **49%** | **54%** |
| 修订：P50 / P95 延迟、占全部 LLM 时间的比例 | 4.3 / 15.7 s、32% | 4.6 / 10.2 s、26% |
| 工具：P50 / P95 延迟 | 0.6 / 601 ms | 0.6 / 744 ms |
| NLU 与路由（`guard_in`） | 占一轮 3% | 2% |
| 最慢 1%（P99） | 93 s | 37 s |

解读：

- **成本几乎全在 LLM 往返上。** 工具、NLU、校验和合规加起来不到一轮的 5%。prompt 本身不大（2.6k–5k token），工具循环的 prompt 有 70–93% 命中提供方缓存（修订请求只有约 25%），所以压缩工具结果或历史收益很小。真正的杠杆是串行 LLM 调用的次数。
- **一半的回答多做了一次修订调用。** 抽样截获了 21 个触发修订的草稿（给 `revise` 挂钩）。多数“无法支持的数字”其实是校验器数字切分的误报，不是模型出错：`20.9x`、`108.5bn` 被读成 20 和 108，日线序列里不带年份的日期（`4.746（04-16）`）被读成 4 和 16，债券期限（`10年期`）被读成 10，列表编号（`3)`）被读成 3。另一部分是模型由所引数值推算出的数（ROE 相差 17.8 个百分点、茅台与平安的 PE 之比 2.8）。其余是纯引用问题：数字没标 id，或标了错误、不存在的 id。
- **长尾来自卡住的调用。** 最慢的轮次里，平常 2–5 s 的调用一直挂到 90 s 的运行截止或客户端超时（基线中 3 + 3 次读超时）。最慢的 10% 工具轮数也更多（1.6–2.2 轮，平均 1.3–1.5 轮），且 80–91% 做了修订。

### 改动（每项都有开关）

| 改动 | 开关（默认） | 作用 |
|---|---|---|
| 修复数字切分 | 始终生效（属于 bug 修复） | 数字 token 在 `x`/`bn`/`mn`/`m`/`k`/`pp`/`pct` 前结束，不再回退成更短的数字；不带年份的日期、债券期限和列表编号不算数值断言。在当前开发集上跑校验器压力测试（[`verifier_stress-perf-8a85ae5.json`](../../evaluation/results/verifier_stress-perf-8a85ae5.json)，202 个正确答案、3,399 个篡改变体）：正确答案全部通过，误放率 1.94%。在 `9f0e46b` 重跑结果相同（202 个正确答案，1.94%；现为 `verifier_stress-9f0e46b.json`，当前的 `verifier_stress.json` 是第 11 轮的运行：227 个正确答案，1.17%）；更早的 2.1%（`2494656`，159 个正确答案）已被取代。 |
| LLM 连接复用 | `QI_LLM_KEEPALIVE=1` | 每个模型共用一个带连接池的 `httpx.Client`，不再每次请求新建 TLS 连接。探针（[`keepalive-probe.json`](../results/perf/agent/keepalive-probe.json)，25 组交替的极小请求）：单次调用 P50 1.69 s → 1.32 s。 |
| 修订前先修引用 | `QI_AGENT_REVISE_POLICY=cite_repair` | 草稿只在引用上出错时（数字无引用或引错、id 不存在或缺失），为每个数字补上**唯一**含有该值的证据 id，重新校验；通过就跳过 LLM 修订。多个证据都含该值时不猜测。 |
| 派生数字 | `QI_AGENT_VERIFY_DERIVED=1` | 若一个数等于同一句所引证据支持的两个数之差、和、比或变化百分比，则予以接受。同一压力测试的误放率 1.94% → 2.03%（跨公司互换的数字 0.53% → 0.93%）。 |
| 卡顿超时 | `QI_AGENT_LLM_STALL_TIMEOUT_S=20` | 所有 LLM 调用都以流式发送，httpx 的读超时（等待下一个数据块的时间）上限设为 20 s；流一旦卡住就重试或切换模型，不必等到运行截止。 |
| 规划器预取 | `QI_AGENT_PREFETCH=1` | 第一次 LLM 调用前先执行确定性规划器的工具调用，把结果随问题一起交给模型。规划器能覆盖的问题**一次** LLM 调用就能答完。模型仍可调用其他工具；重复预取过的调用会被重复调用保护拦下。 |

设置 `QI_AGENT_PREFETCH=0 QI_AGENT_REVISE_POLICY=llm QI_AGENT_LLM_STALL_TIMEOUT_S=0 QI_AGENT_VERIFY_DERIVED=0` 即恢复原来的 Agent 路径。`AgentConfig` 上有同名字段，消融脚本里用 `--agent-config FIELD=VALUE` 设置。

### 前后对比（DeepSeek v4.1 flash，3 次重复，无 429）

第 1 次运行与 A 之间，本分支合并了 round 2（`df19daa`：多轮会话规则），所以下表有两组配对比较，每组都在同一份代码上完成。

| 集合 | 运行 | Commit | P50 s | P95 s | P99 s | LLM 轮 P95 s | TTFT P50 s | TTFT P95 s | LLM 调用/轮 | token/轮 | 成本/任务（USD） | 任务成功率 | 相对参照的差 [95% CI] | pass^3 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| held-out | 基线 | `8e81f48` | 7.6 | 27.1 | 93.4 | 27.1 | 5.3 | 13.0 | 2.30 | 8,711 | 0.00122 | 0.962 | 参照 | 0.943 |
| held-out | 第 1 次：数字切分修复 + 连接复用 | `52e80dc` | 6.2 | 19.3 | 22.5 | 19.3 | 5.3 | 12.4 | 2.15 | 7,990 | 0.00096 | 0.994 | +0.031 [−0.006, +0.088] | 0.981 |
| test_v2 | 基线 | `8e81f48` | 8.6 | 24.2 | 37.3 | 24.6 | 5.7 | 14.3 | 2.58 | 10,274 | 0.00176 | 0.898 | 参照 | 0.876 |
| test_v2 | 第 1 次：数字切分修复 + 连接复用 | `52e80dc` | 7.6 | 20.3 | 31.2 | 20.9 | 5.4 | 13.0 | 2.49 | 9,859 | 0.00160 | 0.923 | **+0.025 [+0.005, +0.050]** | 0.917 |
| held-out | A：合并 round 2，开关关闭 | `aae29fd` | 6.6 | 22.9 | 90.0 | 22.9 | 5.5 | 13.0 | 2.23 | 8,747 | 0.00106 | 0.987 | 参照 | 0.962 |
| held-out | B：A + 修引用 + 卡顿超时 + 派生数字 | `d84f4e4` | 6.5 | 17.4 | 22.4 | 18.3 | 5.3 | 11.5 | 2.09 | 7,882 | 0.00092 | 0.987 | +0.000 [−0.019, +0.019] | 0.981 |
| held-out | **C：B + 规划器预取（新默认）** | `6050ffd` | **3.7** | **15.4** | **19.6** | **15.5** | **3.0** | **9.4** | **1.39** | 6,018 | 0.00077 | 0.994 | +0.006 [+0.000, +0.019] | 0.981 |
| test_v2 | A：合并 round 2，开关关闭 | `aae29fd` | 8.0 | 21.8 | 40.6 | 22.4 | 5.8 | 13.3 | 2.58 | 10,392 | 0.00166 | 0.950 | 参照 | 0.934 |
| test_v2 | B：A + 修引用 + 卡顿超时 + 派生数字 | `d84f4e4` | 7.7 | 21.7 | 28.8 | 23.0 | 5.8 | 14.1 | 2.50 | 10,099 | 0.00152 | 0.956 | +0.005 [−0.005, +0.017] | 0.950 |
| test_v2 | **C：B + 规划器预取（新默认）** | `6050ffd` | **4.7** | **17.3** | **27.5** | **18.9** | **2.8** | **10.7** | **1.71** | 7,996 | 0.00134 | 0.953 | +0.003 [−0.005, +0.014] | 0.942 |

文件：`evaluation/results/` 下的 `perf-baseline-deepseek.json`、`perf-verifierfix-deepseek.json`、`perf-merged-defaults-deepseek.json`、`perf-merged-citerepair-stall-deepseek.json`、`perf-merged-prefetch-deepseek.json`。B、C 的开关是在所列 commit 上用 `--agent-config` 设置的；这些 commit 与 `aae29fd` 的差别只在结果文件和评测报表代码。“LLM 调用/轮”按全部轮次平均，含拒答和澄清。

各步的收益：

- **数字切分修复 + 连接复用（第 1 次）。** 修订率 49% → 29%（held-out）、54% → 42%（test_v2），P95 27.1 → 19.3 s、24.2 → 20.3 s。任务成功率上升，test_v2 上显著（+0.025 [+0.005, +0.050]）：正确的草稿不再被送去修订，也不再被修复步骤删句。
- **修引用 + 卡顿超时 + 派生数字（B 对 A）。** 修订率 30% → 22%、41% → 28%；修引用替代了 10 + 52 次 LLM 修订。卡顿超时去掉了 90 s 的离群值（held-out P99 90 → 22 s，test_v2 41 → 29 s），有 4 个卡住的流被重试。单靠 B，test_v2 的 P95 还压不到 20 s 以下：那里的慢轮平均要 2.7 轮工具调用。
- **规划器预取（C 对 B）。** held-out 59%、test_v2 47% 的 LLM 轮一次 LLM 调用就答完，多轮工具调用的比例也下降了。P50 下降约 40%，TTFT P50 大约减半（5.3–5.8 s → 2.8–3.0 s），因为第一次 LLM 调用就在写答案。第一次调用的 prompt 多了 600–1,000 token，耗时从 2.3 s 变成 2.9 s，但省掉了整整一次往返。成本随调用次数下降（相对 A：held-out −28%，test_v2 −19%）。
- **相对第二轮基线**（中间合并了代码，不是同一代码上的配对）：held-out P95 27.1 → 15.4 s，test_v2 24.2 → 17.3 s，TTFT P50 5.3–5.7 → 2.8–3.0 s。任务成功率差为 +0.031 [−0.006, +0.088] 和 +0.055 [+0.022, +0.094]。

**代价，如实说明。**

- 预取让*工具精度*（真正用到的调用占比）略降：held-out 0.684 → 0.675，test_v2 0.650 → 0.616。规划器按规则取数，有些用不上。这不计入任务成功率；在回放快照上这些调用只需毫秒，但接实时数据源时会多出上游请求。
- *派生数字*让校验器对互换数字的误放率从 0.53% 升到 0.93%：一个错数若恰好等于同句所引两个数的差或比，就会通过。总误放率为 2.03%（原 1.94%）。设 `QI_AGENT_VERIFY_DERIVED=0` 可换回，代价是更多修订。
- *修引用*会改动模型的草稿。只有恰好一个证据项含有该值时才补 id（并删掉不存在的 id），结果还要通过同一个校验器。绑定仍是到证据项而不是字段，这是 `verify_answer` 里已写明的已知局限。
- *卡顿超时*限制的是等下一个流式数据块的时间，不是整次调用。持续缓慢输出的模型不会被切断，那由运行截止时间处理（90 s，作答另加 20 s）。
- 两个集合都达到了 P95 < 20 s，但这不是负载下或更慢模型上的保证：见下方的压测和 GLM 抽查。

### 负载下：4 个用户，流式（[`load_test-agent-4-*.json`](../results/perf/agent/)）

同一份服务构建（commit `0309fdf`，合并后的代码，关闭实时数据）分别以原路径（`QI_AGENT_PREFETCH=0 QI_AGENT_REVISE_POLICY=llm QI_AGENT_LLM_STALL_TIMEOUT_S=0 QI_AGENT_VERIFY_DERIVED=0`）和默认配置各启动一次。4 个用户 × 6 个请求，取自 `research` 问题集：8 个需要多个工具的问题，比评测集更难。请求发往 `/agent/chat/stream`（`--stream`），因此 TTFT 就是客户端实际看到的时间。运行时间 2026-09-28 16:36–16:41 UTC，开始时负载 3.0–3.3。

```bash
python -m scripts.load_test --base-url http://127.0.0.1:8811 --mode agent --questions research --users 4 \
    --requests 6 --usd-cny 6.7489 --questions-per-day 2000 --timeout 240 --stream --label defaults
```

| 服务配置 | 请求数 | 吞吐（req/s） | P50（s） | P95（s） | P99（s） | TTFT P50（s） | TTFT P95（s） | 每问 LLM 调用 | 每千问成本（¥） | 通过校验 | HTTP 错误 / LLM 错误 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原路径 | 24 | 0.145 | 13.0 | 29.6 | 92.1 | 8.7 | 17.3 | 3.75 | 14.2 | 88% | 0 / 0 |
| 默认配置 | 24 | 0.208 | 11.6 | 26.7 | 26.7 | 7.8 | 14.5 | 2.75 | 12.6 | 96% | 0 / 0 |

每行只有 24 个请求，样本很小：P95 是第二慢的请求，P99 是最慢的。方向与评测一致：每问少一次 LLM 调用，长尾更低（92 s 的卡顿不见了），吞吐高 43%，成本低 11%。**在这批更难的问题上，P95 仍高于 20 s**；4 个用户下 TTFT（7.8 s）也高于 3 并发评测时的 3 s。这些问题比多数评测任务需要更多轮工具调用、更长的答案。被重试救回的网关 429 在客户端看不到；没有请求失败或降级。

### 第二个模型：GLM-5.3 flash 抽查（[`perf-glm-holdout-*.json`](../../evaluation/results/)）

held-out，每次运行 1 次重复，3 并发，流式。由于 GLM 的延迟在不同运行之间漂移，原路径和默认配置按“关、开、开、关”的顺序运行（`8a85ae5`）：

| 运行（合并，各 112 轮） | P50（s） | P95（s） | P99（s） | TTFT P50（s） | TTFT P95（s） | LLM 调用/轮 | 修订率（LLM 轮） | 任务成功率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原路径（off + off2） | 13.6 | 60.6 | 96.2 | 9.3 | 38.0 | 2.19 | 38% | 1.00、1.00 |
| 默认配置（on + on2） | 11.0 | 68.5 | 85.3 | 8.8 | 54.7 | 1.36 | 21% | 1.00、1.00 |

这些开关让 GLM 的 LLM 调用减少 38%、修订减半，没有丢任何任务。**但延迟收益不能确定**：同一配置的两次运行相差 40%（关闭时 P50 14.9 对 20.8 s，打开时 19.1 对 11.0 s）。GLM 单次调用从 3 s 到 75 s 不等，长尾由调用之间的波动决定，而不是调用次数。开启预取后，GLM 的第一次调用就要写出完整答案（约 500 个输出 token，其中 320 个是推理），第一次调用一慢，首个 token 就跟着晚。TTFT P95 反而更高（38 → 55 s）。默认配置是按 DeepSeek 调的，Agent 路径由 DeepSeek 服务。GLM 仍只是故障切换模型，它的长尾由运行截止时间兜底。

### 没有改的部分及原因

- **推理强度。** 工具循环的调用用模型默认值，本来推理 token 就很少（DeepSeek 每次 40–100 个）；写答案的调用已经是 `low`。没有再往下调：时间不花在推理上，而答案质量依赖推理。
- **压缩工具结果或历史。** prompt 只有 2.6k–6k token，且大多命中缓存。带预取证据的第一次调用 prompt 多了 600–1,000 token，约多花 0.6 s。决定耗时的是 LLM 调用次数，不是 prompt 长短。
- **更早开始流式输出。** 答案 JSON 的第一个字段（`answer`）一生成就开始流式输出。TTFT 就是开始写答案的那次调用启动之前的时间，预取把它减半。后来未通过校验的草稿，仍会在最终的 `answer` 事件里被校验后的答案替换。

## 3. k3s 上的多副本扩展（Postgres 会话）

部署：`deploy/k8s/finsight.yaml` 跑在 k3s（colima）里，镜像 `finsight:merged`，会话存在 Postgres StatefulSet 里（`QI_AGENT_CHECKPOINT_DB=postgresql://...`）。API Pod 每个限 2 CPU，节点共 4 vCPU。

每个副本数都这样测：
- HPA 固定住副本数；
- 每个 Pod 先用轮换问题预热（NLU 模型是懒加载的）；
- 由集群**内部**的压测 Pod（0.5 CPU）对 `Service/finsight-api` 运行 `scripts/load_test.py`，每次新建连接，让 kube-proxy 把请求分散到各个 Pod；
- 确定性路径，实时数据关闭，每个用户 20 次请求。

驱动脚本：[`deploy/k8s/scale_test.sh`](../../deploy/k8s/scale_test.sh)。

| 副本数 | 并发 | 请求数 | 吞吐（次/秒） | P50（ms） | P95（ms） | P99（ms） | 错误 | 每个 Pod 完成的运行数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 160 | 3.04 | 1,971 | 5,912 | 13,213 | 0 | 141 |
| 1 | 32 | 640 | 3.75 | 7,731 | 13,702 | 18,420 | 0 | 561 |
| 2 | 8 | 160 | 4.32 | 694 | 4,654 | 11,107 | 0 | 72 / 69 |
| 2 | 32 | 640 | 6.07 | 4,018 | 13,809 | 21,209 | 0 | 275 / 286 |
| 3 | 8 | 160 | 10.17 | 275 | 3,081 | 3,812 | 0 | 50 / 46 / 45 |
| 3 | 32 | 640 | 11.72 | 2,017 | 6,253 | 8,119 | 0 | 199 / 166 / 196 |

文件：
- `results/perf/k3s/load_test-k3s-r{1,2,3}-u{8,32}.json`：压测结果；
- `pods-r*.txt`：`kubectl get pods -o wide` 的输出；
- `top-r*-u*.txt`：每次运行期间的 `kubectl top pods`，32 并发下每个 API Pod 占 0.8–1.1 CPU，一个进程一个核；
- `runs-per-pod-*.txt`：每次运行前后从各 Pod 的 `/metrics` 读出的完成运行数，澄清不计入。

运行时间：2026-09-26 10:34–10:45 UTC。

解读：
- **单进程受 CPU 限制**：一个进程受 GIL 限制，大约只能用满一个核。Service 把负载分得很均匀（见上面每个 Pod 的运行数）。
- **扩展效果**：32 并发下，1 到 3 个副本的吞吐从 3.75 → 6.07 → 11.72 次/秒，零错误，P95 从 13.7 秒降到 6.3 秒。
- **不要读成 3.1 倍**：3 副本相对 1 副本看起来超线性，是因为 1 副本那几行测量时主机更忙（同一个镜像当天早些时候在 Docker 里 32 并发跑到了 5.2 次/秒）。应理解为「在节点核数以内大致线性」。

之前有一次没有逐 Pod 预热的尝试，没有报告：新 Pod 上的头几个英文问题要花好几秒，主导了它的尾延迟。

### 跨副本会话的证据

`results/perf/k3s/cross-replica-session.txt`（2026-09-26）：第 1 轮「贵州茅台的市盈率是多少？」只通过端口转发发给 Pod `…-24ktk`，第 2 轮「它的市净率呢」只发给 Pod `…-6mc86`。

- 第 2 轮把「它」解析成了贵州茅台（600519.SH）。
- 在第一个 Pod 上 `GET /agent/sessions/<id>` 返回了两轮。
- 每个 Pod 自己的 trace 目录里恰好有其中一轮（`cross-replica-pod-logs.txt`：`…-24ktk` 上是 turn_index 0，`…-6mc86` 上是 turn_index 1，外加各 Pod 的访问日志行）。
- `cross-replica-postgres.txt` 显示 Postgres 里这个 thread 有 2 个检查点。

能把实体从一个 Pod 带到另一个 Pod 的，只有共享的检查点。

**Postgres 检查点是实现了的。** 本页上一版写的是「未实现」，那是错的：
- `QI_AGENT_CHECKPOINT_DB=postgresql://...` 会通过 psycopg 连接池选用 `langgraph-checkpoint-postgres`（`agent/memory.py`）；
- `tests/test_agent_checkpoint_postgres.py` 覆盖了它；
- 上面的 k3s 测量用的就是它。

从第三轮起，A2A 任务表和 `/agent/traces` 背后的 trace 存储也跟随同一个 Postgres DSN（见 [A2A、容灾与可观测性](a2a-and-observability.md#多副本共享存储)）。仍然每个副本各自一份的：限流器，以及工具和数据源的 TTL 缓存。

## 复现

```bash
docker build -f docker/Dockerfile -t finsight:merged .
docker/perf_matrix.sh                                   # 第 1 节（还需要 finsight:before-0678585）
IMAGE=finsight:merged OUT=docs/results/perf/k3s deploy/k8s/scale_test.sh   # 第 3 节
SKIP_LOAD=1 deploy/k8s/scale_test.sh                    # 只做跨副本会话检查
```

第 2a 节：`python -m evaluation.agent_eval.ablation --llm deepseek --repeats 3 --workers 3 --sets holdout,test_v2 --modes agent --stream --out outputs/agent_eval/<run>.json`，然后用 `python -m evaluation.agent_eval.profile outputs/agent_eval/<run>.json` 得到剖析表，用 `python -m evaluation.agent_eval.results outputs/agent_eval/<run>.json --name <name>` 生成提交的结果文件。

`QI_AGENT_DURABILITY=async`（或 `sync`）恢复每步写检查点。LangGraph 文档说明了 `exit` 的代价：运行中途崩溃会丢掉这一轮。运行只需几秒，下一轮会从上一个完成的轮次继续。

## 服务启动时间

以下数字全部来自两个已提交的结果文件。

### 进程内构建服务：[`results/perf/startup.json`](../results/perf/startup.json)

命令 `python scripts/measure_startup.py --out docs/results/perf/startup.json`，commit `a027c7c`（2026-09-28，Apple Silicon Mac 10 核，开始时负载 4.4 / 5.6 / 10.5，关闭实时数据）。构建时间主要花在对 43 MB 文档拟合 char n-gram TF-IDF 索引上。

| 情形 | 秒 |
|---|---:|
| 冷启动构建（新进程，拟合索引） | 24.06 |
| 同一进程内重建（索引按语料哈希缓存） | 3.65 |
| 冷启动构建并把索引写到 `QI_TFIDF_CACHE_DIR` | 29.08 |
| 新进程从 `QI_TFIDF_CACHE_DIR` 加载索引 | 6.81 |

- **进程内缓存**：从 `da3ec8b` 起索引按语料哈希缓存，`clear_service_caches()` 不会清掉它，全量测试因此不再每个用例都重新拟合。
- **可选落盘**：设置 `QI_TFIDF_CACHE_DIR` 后索引写到该目录（365 MB），重启时直接加载：6.81 秒，而冷启动要 24.06 秒。文件太大，没有打进镜像；适合挂在多个副本共享的卷上。

（本节旧版本引用了 `da3ec8b` 提交说明里的 39 秒 / 6 秒 / 4.5 秒 / 350 MB，并说没有结果文件；这些数字与 `startup.json` 不符，已替换。）

### 容器冷启动：[`results/perf/startup-container.json`](../results/perf/startup-container.json)

命令 `python scripts/measure_container_startup.py --image finsight:edcb442 --runs 3`。镜像在干净工作区从 commit `edcb442` 构建（`git archive edcb442 | docker build -f docker/Dockerfile -`），也就是 Kubernetes kustomization 固定的那个镜像。每次都是全新的 `docker run`，加固方式与 Kubernetes 相同（只读根文件系统、uid 10001、`/tmp`、`/app/state`、`/app/outputs` 用 tmpfs），关闭实时数据，SQLite 检查点，不配 LLM Key；从 `docker run` 计时到 `/ready` 返回 200。colima 虚拟机 4 vCPU / 7.7 GB，同一台 Mac，同时有其他任务在跑（主机负载 7–12）。

| 次数 | `/health` 200（秒） | `/ready` 200（秒） | 日志里的服务构建（秒） |
|---|---:|---:|---:|
| 1 | 57.13 | 57.50 | 52.4 |
| 2 | 45.05 | 45.53 | 42.5 |
| 3 | 45.68 | 45.99 | 42.9 |
| **中位数** | **45.68** | **45.99** | |

- 服务构建完成后 uvicorn 才开始监听，所以 `/health` 要等构建结束；容器启动时间几乎全是 TF-IDF 拟合，在 4 vCPU 虚拟机里比本机慢。
- 第一次 `/ready` 会构建 Agent 并打开检查点，多花 0.3–0.5 秒。
- Kubernetes 的 `startupProbe` 允许 5 分钟（60 × 5 秒），远高于这些时间。
- 第二轮评审在更忙的主机上测过更早的镜像，约 105–165 秒变为 healthy；那次没有提交结果文件。
