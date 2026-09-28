# 部署

语言：[English](../deployment.md) | 中文

FinSight 以一个容器镜像交付，里面同时提供 API、Agent、A2A 接口和构建好的网页前端。MCP Server 是同一镜像里的另一个入口：`python -m query_intelligence.agent.mcp_server --transport http`。

会话保存在 LangGraph checkpointer 里，有三种位置：
- 内存；
- SQLite 文件；
- Postgres：多个进程或副本需要共享会话时使用。

## Docker

```bash
docker build -f docker/Dockerfile -t finsight:dev .
docker run -d --name finsight -p 8000:8000 \
  -e DEEPSEEK_API_KEY=... finsight:dev            # 不传 Key 就只走确定性路径
curl http://127.0.0.1:8000/health
```

- **多阶段构建**：wheel 在 `python:3.13` 里编译，运行镜像是不带编译器的 `python:3.13-slim`（2 GB；只有 `--build-arg WITH_TORCH=1` 时才装 torch）。
- **运行身份**：以 uid 10001 运行，Docker `HEALTHCHECK` 检查 `/health`。
- **前端**：网页用的是仓库里已提交的构建产物 `query_intelligence/web/dist`，镜像不需要 Node 工具链；CI 会检查它与 `frontend/` 源码一致。
- **可选 profile**：`docker/docker-compose.yml` 提供 `postgres`、`tracing`（通过 OTLP 接 Jaeger）和 `monitoring`（带告警规则的 Prometheus、带 FinSight 看板的 Grafana、Jaeger），见 [A2A、容灾与可观测性](a2a-and-observability.md#看板告警与监控栈)。
- **出网代理**：容器不能直接访问外网时，`FINSIGHT_EGRESS_PROXY` 把 HTTP(S) 代理传给应用。

### 运维指标与主动探测的接线

`/metrics` 在抓取时提供熔断器和线程池指标（`OpsMetricsCollector`），`/sources/health` 支持 `?probe=1`。这部分接线最初以 [`deploy/patches/app-ops-wiring.patch`](../../deploy/patches/app-ops-wiring.patch) 的形式保存（[性能](performance.md)里测量的镜像就是带着它构建的），从 `e9d9a9a` 起已直接合入 `api/app.py`。`tests/test_source_reliability.py` 覆盖了这两个接口。

2026-09-25 验证（colima、arm64）：
- 容器进入 healthy 状态，能访问网页；
- `/agent/chat` 返回经过校验的答案；
- 能访问 A2A 服务卡片和 Prometheus 指标；
- `/sources/health` 正常返回。

压测数字见[性能](performance.md)。

## Kubernetes

`deploy/k8s/finsight.yaml` 包含：

| 对象 | 说明 |
|---|---|
| `Deployment/finsight-api` | 2 个副本。启动探针（NLU 模型加载约一分钟），就绪与存活探针都查 `/health`。请求 0.5 CPU / 2 GiB，上限 2 CPU / 3 GiB。滚动更新 `maxUnavailable: 0`。 |
| Pod 安全 | `runAsNonRoot`（uid 10001）、`readOnlyRootFilesystem`、去掉全部 capability、`RuntimeDefault` seccomp；`/app/outputs`、`/app/state` 和 `/tmp` 用 `emptyDir`。 |
| `HorizontalPodAutoscaler` | 2–6 个副本，CPU 70% 时扩容。 |
| `PodDisruptionBudget` | 节点维护时至少保留一个副本。 |
| `StatefulSet/finsight-postgres` | 演示用的会话 Postgres；生产环境请用托管数据库。 |
| `ConfigMap` / `Secret` | 实时数据开关、超时、限流；检查点 DSN；可选的 `finsight-llm` Secret，内含 `DEEPSEEK_API_KEY`。 |

```bash
kubeconform -strict -summary deploy/k8s/finsight.yaml     # 9 个资源，全部有效
kubectl apply -f deploy/k8s/finsight.yaml
kubectl -n finsight create secret generic finsight-llm --from-literal=DEEPSEEK_API_KEY=...   # 可选
kubectl -n finsight rollout status deployment/finsight-api
```

### 为什么多个副本能共享一段对话

- **每一轮都是一次 LangGraph 运行**，thread 就是 `session_id`。设置 `QI_AGENT_CHECKPOINT_DB=postgresql://...` 后，检查点都在 Postgres 里：对话轮次、暂停中的澄清、上一次解析出的实体。所以下一轮落到哪个副本都可以。
- **测试覆盖**：`tests/test_agent_checkpoint_postgres.py` 用两个服务实例验证了这一点。
- **k3s 上的实测**：发给副本 B 的追问（「它的市净率呢」）从副本 A 处理过的那一轮里解析出了代词。

会话还按调用方隔离：设置 `QI_API_KEYS` 后，会话归属于 API Key 的哈希，别的 Key 访问得到 404。

同一个 DSN 也会把 A2A 任务表（`finsight_a2a_tasks`）和 `/agent/traces` 背后的 trace 表（`finsight_agent_traces`）放进 Postgres。这样任何副本都能响应 `GetTask`、接着处理 `input-required` 任务，并在运行查看器里显示任何一次运行（见 [A2A、容灾与可观测性](a2a-and-observability.md#多副本共享存储)；设置 `QI_A2A_TASK_DB` / `QI_AGENT_TRACE_DB=memory` 可退出）。仍然每个副本各自一份的：限流器，以及工具和数据源的 TTL 缓存。

### 只读根文件系统

第一次上线时 Pod 反复崩溃：efinance 在 import 时会创建 `<site-packages>/efinance/data`。现在镜像把这个目录软链到 `/tmp/efinance-data`，provider 在 import 之前先建好链接目标，所以 Pod 可以在 `readOnlyRootFilesystem: true` 下运行。

### 验证记录

2026-09-26，colima 中的 k3s v1.35（4 vCPU、8 GB）：

- **镜像 `finsight:dev`**：
  - 两个 API 副本都在 `Service/finsight-api` 后面就绪，Postgres 就绪；
  - 在 Pod 里 `touch /app/x` 报「Read-only file system」，进程以 uid 10001 运行。
- **镜像 `finsight:merged`**：证据提交在 `docs/results/perf/k3s/`，驱动脚本为 [`deploy/k8s/scale_test.sh`](../../deploy/k8s/scale_test.sh)。
  - 1、2、3 个副本挂在 Service 后面，会话在 Postgres 里；
  - 每种副本数由压测 Pod 发 800 次请求，0 错误，负载分布均匀（`runs-per-pod-*.txt`）；
  - 32 并发下吞吐 3.75 → 6.07 → 11.72 次/秒（[性能](performance.md#3-k3s-上的多副本扩展postgres-会话)）；
  - 每次运行都保存了 `kubectl get pods -o wide` 和 `kubectl top pods` 的输出。
- **跨 Pod 的会话延续**：第 2 轮（「它的市净率呢」，只发给 Pod B）从第 1 轮（只发给 Pod A）解析出了贵州茅台。每个 Pod 的 trace 目录里恰好有自己那一轮，Postgres 里这个 thread 有 2 个检查点（`cross-replica-*.txt`）。
- **日志采集的问题**：通过 colima 的 kubelet 端口执行 `kubectl logs` 时偶尔报「unexpected EOF」，脚本会退回到对应容器的 `docker logs`；在 Docker 运行时的节点上两者是同一个流。

扩展测试会为了测量改动部署（关闭实时数据、关闭限流、固定 HPA）。测完请重新 `kubectl apply -f deploy/k8s/finsight.yaml` 恢复默认配置。
