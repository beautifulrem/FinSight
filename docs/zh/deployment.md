# 部署

语言：[English](../deployment.md) | 中文

FinSight 以一个容器镜像交付，里面同时提供 API、Agent、A2A 接口和构建好的网页前端。MCP Server 是同一镜像里的另一个入口：`python -m query_intelligence.agent.mcp_server --transport http`。

会话保存在 LangGraph checkpointer 里，有三种位置：
- 内存；
- SQLite 文件；
- Postgres：多个进程或副本需要共享会话时使用。

## Docker

```bash
TAG=$(git rev-parse --short=7 HEAD)               # 镜像用构建时的 commit 打标签
docker build -f docker/Dockerfile -t finsight:$TAG .
docker run -d --name finsight -p 8000:8000 \
  -e DEEPSEEK_API_KEY=... finsight:$TAG           # 不传 Key 就只走确定性路径
curl http://127.0.0.1:8000/health                 # 存活
curl http://127.0.0.1:8000/ready                  # 就绪：检查点、模型配置、检索索引
```

- **多阶段构建**：wheel 在 `python:3.13` 里编译，运行镜像是不带编译器的 `python:3.13-slim`（2 GB；只有 `--build-arg WITH_TORCH=1` 时才装 torch）。
- **运行身份**：以 uid 10001 运行，Docker `HEALTHCHECK` 检查 `/health`。
- **构建上下文**：`.dockerignore` 排除 `.git`、`.env*`、虚拟环境、`frontend/`（含 `node_modules`）、输出目录、文档和测试；镜像只复制运行所需的目录。
- **前端**：网页用的是仓库里已提交的构建产物 `query_intelligence/web/dist`，镜像不需要 Node 工具链；CI 会检查它与 `frontend/` 源码一致。
- **可选 profile**：`docker/docker-compose.yml` 提供 `postgres`、`tracing`（通过 OTLP 接 Jaeger）和 `monitoring`（带告警规则的 Prometheus、带 FinSight 看板的 Grafana、Jaeger），见 [A2A、容灾与可观测性](a2a-and-observability.md#看板告警与监控栈)。
- **出网代理**：容器不能直接访问外网时，`FINSIGHT_EGRESS_PROXY` 把 HTTP(S) 代理传给应用。

### 运维指标与主动探测的接线

`/metrics` 在抓取时提供熔断器和线程池指标（`OpsMetricsCollector`），`/sources/health` 支持 `?probe=1`。这部分接线最初以 `deploy/patches/app-ops-wiring.patch` 的形式保存（[性能](performance.md)里测量的镜像就是带着它构建的），从 `e9d9a9a` 起已直接合入 `api/app.py`，补丁文件已删除。`tests/test_source_reliability.py` 覆盖了这两个接口。

2026-09-25 验证（colima、arm64）：
- 容器进入 healthy 状态，能访问网页；
- `/agent/chat` 返回经过校验的答案；
- 能访问 A2A 服务卡片和 Prometheus 指标；
- `/sources/health` 正常返回。

压测数字见[性能](performance.md)。

## Kubernetes

`deploy/k8s/kustomization.yaml` 应用 `finsight.yaml` 和 `networkpolicy.yaml`，并固定镜像标签。`finsight.yaml` 包含：

| 对象 | 说明 |
|---|---|
| `Deployment/finsight-api` | 2 个副本。启动与存活探针查 `/health`（NLU 模型加载约一分钟）；就绪探针查 `/ready`，它还检查 Postgres 检查点、LLM 配置和检索索引，连不上数据库的副本会停止接流量，但不会被重启。请求 0.5 CPU / 2 GiB，上限 2 CPU / 3 GiB。滚动更新 `maxUnavailable: 0`。 |
| Pod 安全 | `runAsNonRoot`（uid 10001）、`readOnlyRootFilesystem`、去掉全部 capability、`RuntimeDefault` seccomp；`/app/outputs`、`/app/state` 和 `/tmp` 用 `emptyDir`。 |
| `HorizontalPodAutoscaler` | 2–6 个副本，CPU 70% 时扩容。 |
| `PodDisruptionBudget` | 节点维护时至少保留一个副本。 |
| `StatefulSet/finsight-postgres` | 演示用的会话 Postgres；生产环境请用托管数据库。 |
| `ConfigMap` | 实时数据开关、超时、限流、LLM 端点和模型。 |
| Secret（不提交） | 按名字引用 `finsight-db`（Postgres 密码和检查点 DSN，必需）和 `finsight-llm`（`DEEPSEEK_API_KEY`，可选），见[密钥](#密钥)。 |
| `NetworkPolicy` ×2（`networkpolicy.yaml`） | 见[网络策略](#网络策略)。 |

```bash
kubectl kustomize deploy/k8s | kubeconform -strict -summary -   # 10 个资源，全部有效（CI 也会跑）
kubectl create namespace finsight
kubectl -n finsight create secret generic finsight-db \
  --from-literal=POSTGRES_PASSWORD="$PG_PASSWORD" \
  --from-literal=QI_AGENT_CHECKPOINT_DB="postgresql://postgres:$PG_PASSWORD@finsight-postgres:5432/finsight"
kubectl -n finsight create secret generic finsight-llm --from-literal=DEEPSEEK_API_KEY=...   # 可选
kubectl apply -k deploy/k8s
kubectl -n finsight rollout status deployment/finsight-api
```

### 镜像标签

镜像用构建时 commit 的短哈希打标签（`finsight:<commit>`），不用 `:dev` 或 `:latest`，这样 `kubectl get pods -o jsonpath='{..image}'` 就能对应到确切的源码。标签只写在一处：`deploy/k8s/kustomization.yaml` 的 `images[].newTag`。

```bash
TAG=$(git rev-parse --short=7 HEAD)          # 在干净的工作区里
docker build -f docker/Dockerfile -t finsight:$TAG .   # 按需推送到镜像仓库
(cd deploy/k8s && kustomize edit set image finsight=finsight:$TAG)
```

单独应用 `finsight.yaml` 时镜像名是不存在的 `finsight:set-by-kustomization`，拉取会明确失败，而不是悄悄跑一个来路不明的构建。CI 把每次推送构建为 `finsight:<github.sha>`。

### 密钥

仓库里不提交任何 Secret（之前的清单带着明文 `change-me` 密码）。三选一：

- 如上用 `kubectl create secret generic`；
- 带占位符的模板 `deploy/k8s/secret.template.yaml`：`POSTGRES_PASSWORD=$(openssl rand -hex 24) envsubst '$POSTGRES_PASSWORD' < deploy/k8s/secret.template.yaml | kubectl apply -f -`；
- 通过 [External Secrets Operator](https://external-secrets.io/) 从密钥管理服务同步，`ExternalSecret` 示例见[英文文档](../deployment.md#secrets)。

Deployment 和 StatefulSet 引用 `finsight-db` 时没有 `optional`，Secret 不存在时 Pod 会一直等待。

### 网络策略

`networkpolicy.yaml` 隔离两个工作负载（需要支持 NetworkPolicy 的 CNI，k3s 自带的控制器也支持）：

| 策略 | 入站 | 出站 |
|---|---|---|
| `finsight-api` | TCP 8000，来源限于 Ingress 控制器（ingress-nginx，或 `kube-system` 里的 Traefik）、`monitoring` 里的 Prometheus，以及带 `finsight.io/client=true` 标签的 Pod（扩展测试的压测 Pod） | DNS 到 kube-dns；TCP 5432 到 Postgres Pod；TCP 80/443 只到公网地址（排除所有私网、链路本地/元数据和 CGNAT 网段） |
| `finsight-postgres` | 只接受 API Pod 的 TCP 5432 | 无 |

kubelet 探针不受影响（节点到本机 Pod 的流量总是放行）。NetworkPolicy 按 IP 而不是主机名匹配，所以无法按名字锁定 LLM 网关和数据源。使用支持 FQDN 的 CNI（Cilium `toFQDNs`、Calico DNS 策略）或出网代理时，只需放行 `DEEPSEEK_BASE_URL` 的网关主机和 `query_intelligence/integrations/sources/catalog.py` 里的数据源主机（清单见[英文文档](../deployment.md#network-policy)）。集群外的托管 Postgres 需要把它的地址写成 `ipBlock`，替换 Postgres 的 Pod 选择器。

### 为什么多个副本能共享一段对话

- **每一轮都是一次 LangGraph 运行**，thread 就是 `session_id`。设置 `QI_AGENT_CHECKPOINT_DB=postgresql://...` 后，检查点都在 Postgres 里：对话轮次、暂停中的澄清、上一次解析出的实体。所以下一轮落到哪个副本都可以。
- **测试覆盖**：`tests/test_agent_checkpoint_postgres.py` 用两个服务实例验证了这一点（CI 里连 Postgres 服务运行；一次本地运行的输出和检查点记录见 [results/postgres/two-replica-checkpointer.md](../results/postgres/two-replica-checkpointer.md)）。
- **k3s 上的实测**：发给副本 B 的追问（「它的市净率呢」）从副本 A 处理过的那一轮里解析出了代词。

会话还按调用方隔离：设置 `QI_API_KEYS` 后，会话归属于 API Key 的哈希，别的 Key 访问得到 404。

不共享、每个副本各自一份的状态：A2A 任务表、`/agent/traces` 背后的内存 trace 缓冲（JSON trace 按 Pod 分开；需要共享视图就导出 OTLP），以及工具和数据源的 TTL 缓存。

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

扩展测试会为了测量改动部署（关闭实时数据、关闭限流、固定 HPA）。测完请重新 `kubectl apply -k deploy/k8s` 恢复默认配置。
