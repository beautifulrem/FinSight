# 性能、压测与扩展

语言：[English](../performance.md) | 中文

所有吞吐、延迟、成本与多副本扩展实测数据均归档于 [`docs/results/perf/`](../results/perf/) 与 [`evaluation/results/`](../../evaluation/results/)，每份 JSON 均记录运行命令、commit、服务开关与宿主机负载。

## 1. 确定性路径与检查点优化

将 LangGraph 持久化模式从逐节点写入（`async`）调整为每轮退出写一次（`QI_AGENT_DURABILITY=exit`，默认）显著降低磁盘与序列化开销（离线快照 + SQLite 会话，`docker/perf_matrix.sh`）：

| 配置 | 并发用户 | 吞吐（req/s） | P50（ms） | P95（ms） | 错误 | 820 请求后会话库体积 |
|---|---:|---:|---:|---:|---:|---:|
| 逐节点（`QI_AGENT_DURABILITY=async`） | 1 / 8 / 32 | 4.80 / 4.94 / 4.44 | 71 / 1,659 / 7,024 | 933 / 2,376 / 10,816 | 0 | 64 MB |
| **每轮一次（`QI_AGENT_DURABILITY=exit`，默认）** | 1 / 8 / 32 | **6.47 / 6.22 / 5.16** | **43 / 1,030 / 5,494** | 733 / 2,937 / 12,298 | 0 | **9.4 MB**（缩小 7 倍） |

## 1b. 数据源故障压测（8 并发，无 LLM）

`scripts/chaos_drill.py --scenario sources-load` 在 5 个阶段（正常实时 → 阻断且在 TTL 内 → 阻断超 TTL 用最近可用 → 阻断超陈旧窗口显式缺失 → 恢复实时）下运行 8 并发共 400 次请求（[`chaos-sources-load-8users.json`](../results/perf/sources-load/chaos-sources-load-8users.json)）：
- **400 请求 0 错误**，全部通过数值与引用校验；
- P50 `0.98–1.53 秒`，P95 `5.1–12.4 秒`（阻断期熔断器直接短路反而更快）。

## 2. Agent 路径延迟剖析与优化

性能剖析（`evaluation/agent_eval/profile.py`）表明 **95% 的耗时来自串行 LLM 往返**（NLU、工具、校验与合规合计 `< 5%`）。六项默认优化在不损失通过率的前提下大幅降低延迟与成本（DeepSeek v4.1 Flash，3 次重复，0 次 429）：

| 优化项 | 默认开关 | 效果 |
|---|---|---|
| 数值分词器修复 | 默认开启 | 消除 `20.9x`、`108.5bn`、`04-16`、`10年期` 导致的误报修订 |
| LLM 长连接复用 | `QI_LLM_KEEPALIVE=1` | 降低单次 TLS 建连开销（P50 `1.69 s → 1.32 s`） |
| 修订前引用自动修复 | `QI_AGENT_REVISE_POLICY=cite_repair` | 对唯一命中的缺失引用直接补齐，避免额外 LLM 修订调用 |
| 同句派生数值校验 | `QI_AGENT_VERIFY_DERIVED=1` | 允许同句内由已支持数字推导的和/差/比/变化率 |
| 流式停滞超时 | `QI_AGENT_LLM_STALL_TIMEOUT_S=20` | 提前重试卡死流，消除 90 秒级 P99 长尾 |
| 规划器预取 | `QI_AGENT_PREFETCH=1` | 首次 LLM 调用前先并行执行规划工具（TTFT 减半、LLM 调用减少 ~35%） |

| 评测集（DeepSeek，3 次重复） | P50 (s) | P95 (s) | P99 (s) | TTFT P50 (s) | LLM 调用/轮 | 单题成本 (USD) | 任务成功率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 保留集：基线 (`8e81f48`) → **当前默认 (`6050ffd`)** | 7.6 → **3.7** | 27.1 → **15.4** | 93.4 → **19.6** | 5.3 → **3.0** | 2.30 → **1.39** | 0.00122 → **0.00077** | 0.962 → **0.994** |
| test_v2：基线 (`8e81f48`) → **当前默认 (`6050ffd`)** | 8.6 → **4.7** | 24.2 → **17.3** | 37.3 → **27.5** | 5.7 → **2.8** | 2.58 → **1.71** | 0.00176 → **0.00134** | 0.898 → **0.953** |

**检索并发争用修复（`doc_retriever.py`）**：将 TF-IDF 矩阵转置缓存为一次性 CSR 操作，4 并发下检索工具 P50 从 `3,941 ms` 降至 **`265 ms`**（P95 `7,899 ms → 1,610 ms`）。

**LLM 并发负载与成本**：
- 8 并发 `auto` 混合研究请求（24 题，[`r13-load-8u-auto.json`](../results/perf/agent/r13-load-8u-auto.json)）：**0 错误**，P50 `13.6 秒`，P95 `24.3 秒`，**¥14.19 / 千题**（日均 2,000 题约 ¥850/月）。
- 4 并发下注入 30 秒上游 503 故障（`r12-load-4u-auto-5xx-burst.json`）：**0 用户可见报错**，熔断期间平滑降级至模板回答并在 60 秒冷却后自动恢复。
- LLM 路径并发上限主要受上游网关限速（429）而非本地 CPU 限制。

## 3. k3s 多副本扩展（PostgreSQL 共享会话）

在 k3s 上测试 1/2/3 副本扩展（`deploy/k8s/scale_test.sh`，共享 Postgres 检查点/任务/Trace/限流存储，离线快照）：

| 副本数 | 并发用户 | 吞吐（req/s） | P50（ms） | P95（ms） | 错误 | 各 Pod 请求分布 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 / 32 | 3.04 / 3.75 | 1,971 / 7,731 | 5,912 / 13,702 | 0 | 141 / 561 |
| 2 | 8 / 32 | 4.32 / 6.07 | 694 / 4,018 | 4,654 / 13,809 | 0 | 72+69 / 275+286 |
| 3 | 8 / 32 | **10.17 / 11.72** | **275 / 2,017** | **3,081 / 6,253** | 0 | 50+46+45 / 199+166+196 |

吞吐随可用 CPU 核心近似线性扩展，且跨副本会话验证通过（Pod A 首轮「贵州茅台市盈率」→ Pod B 次轮「它的市净率呢」准确承接实体）。

## 4. 服务冷启动耗时

启动主要开销为对随仓 43 MB 文档语料构建字符 n-gram TF-IDF 索引（[`startup.json`](../results/perf/startup.json)、[`startup-container.json`](../results/perf/startup-container.json)）：

| 启动场景 | 耗时 |
|---|---:|
| 全新进程冷启动（内存即时构建索引） | `24.1 秒`（宿主机）/ `45.7 秒` 中位数（4 vCPU 容器） |
| 同进程重建（按语料哈希内存缓存） | `3.65 秒` |
| 配置磁盘索引缓存（`QI_TFIDF_CACHE_DIR`，~365 MB） | **`6.81 秒`** |

## 复现命令

```bash
python -m scripts.load_test --base-url http://127.0.0.1:8000 --users 8 --requests 20
python -m scripts.chaos_drill --scenario sources-load
IMAGE=finsight:$(git rev-parse --short=7 HEAD) deploy/k8s/scale_test.sh
```
