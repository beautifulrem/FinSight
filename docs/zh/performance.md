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
- **修复**：`d1c007c` 起，Agent 层的每次 LLM 请求（包括重试、修改和备用模型）超时都取「客户端超时」和「距运行截止的剩余时间」中较小者（工具循环 90 秒，作答再宽限 20 秒），备用路径会以确定性答案结束，而不是 504。这项修复还没有在压测下重新测量。

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
