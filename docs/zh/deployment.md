# 部署指南

语言：[English](../deployment.md) | 中文

FinSight 将 FastAPI 接口、研究 Agent、A2A 1.0 服务与构建好的 React 19 前端打包为单个加固容器镜像（同一镜像亦可通过 `python -m query_intelligence.agent.mcp_server --transport http` 启动 MCP 服务端）。多副本部署时通过 `QI_AGENT_CHECKPOINT_DB` 共享 PostgreSQL 会话、A2A 任务、Trace 与限流状态。

## Docker

```bash
TAG=$(git rev-parse --short=7 HEAD)
docker build -f docker/Dockerfile -t finsight:$TAG .
docker run -d --name finsight -p 127.0.0.1:8000:8000 \
  -e DEEPSEEK_API_KEY=... finsight:$TAG          # 不传 Key 则走确定性路径
curl http://127.0.0.1:8000/health                # 存活检查
curl http://127.0.0.1:8000/ready                 # 就绪检查（检查点、模型、索引、离线数据层）
```

- **多阶段加固镜像**：在 `python:3.13` 编译依赖，运行于 `python:3.13-slim`（约 2 GB；需 FinBERT 时传 `--build-arg WITH_TORCH=1`），以非 root `uid 10001` 运行，兼容 `--read-only` 只读根文件系统（将 `efinance/data` 软链至 `/tmp/efinance-data`），并直接内置 `query_intelligence/web/dist` 前端产物。
- **离线三层数据与就绪探针**：镜像包含 v1 种子、扩展日线层与一手年报更正层。`GET /ready` 列出各层加载状态、记录数、版本与 SHA-256；若已开启层缺失，`/ready` 返回 `503`，对话与声明核查标记 `degraded: ["data_layer_missing:corrections"]`。
- **Docker Compose（`docker/docker-compose.yml`）**：提供 `postgres`、`tracing`（Jaeger）与 `monitoring`（Prometheus + Grafana + Jaeger）可选 profile，默认全部仅绑定 `127.0.0.1`；容器需经代理出网时可设 `FINSIGHT_EGRESS_PROXY`。

## Kubernetes（`deploy/k8s/`）

`deploy/k8s/kustomization.yaml` 应用 `finsight.yaml` 与 `networkpolicy.yaml`：

| 资源 | 配置要点 |
|---|---|
| `Deployment/finsight-api` | 2 副本、`maxUnavailable: 0`；`/health` 启动与存活探针、`/ready` 就绪探针；0.5–2 CPU、2–3 GiB 内存；`runAsNonRoot`（`uid 10001`）、`readOnlyRootFilesystem: true`、丢弃全部 Capabilities、`RuntimeDefault` seccomp，`/tmp`、`/app/state`、`/app/outputs` 挂载 `emptyDir`。 |
| `HorizontalPodAutoscaler` & `PodDisruptionBudget` | 2–6 副本（70% CPU 触发）；`minAvailable: 1`。 |
| `StatefulSet/finsight-postgres` | 演示用 Postgres（生产环境请替换为托管数据库）。 |
| `ConfigMap` & 外部 `Secret` | 启用 `QI_PROFILE=production`；强制引用外部创建的 `finsight-db` 与 `finsight-api-keys`（`finsight-llm` 可选）。 |
| `NetworkPolicy`（`networkpolicy.yaml`） | API 仅放行来自 Ingress、Prometheus 及 `finsight.io/client=true` Pod 的入站，出站仅限 DNS、Postgres `:5432` 与公网 `80/443`（拦截私网与元数据地址）；Postgres 仅接受 API Pod 访问。 |

### 镜像标签策略

仓库内清单固定使用部署期占位标签 `set-at-deploy`（绝不使用 `:latest` 或过期的 commit 哈希），避免误跑旧镜像：

```bash
TAG=$(git rev-parse --short=7 HEAD)
docker build -f docker/Dockerfile -t finsight:$TAG .
(cd deploy/k8s && kustomize edit set image finsight=finsight:$TAG)
kubectl apply -k deploy/k8s
kubectl -n finsight rollout status deployment/finsight-api
```

CI 通过 `tests/test_deploy_tag_policy.py` 与 `kubeconform -strict` 强制校验该策略。

### 密钥配置

仓库不提交任何 Secret，部署前通过 `kubectl create secret generic`、`envsubst < deploy/k8s/secret.template.yaml | kubectl apply -f -` 或 External Secrets Operator 注入：

```bash
kubectl create namespace finsight
kubectl -n finsight create secret generic finsight-db \
  --from-literal=POSTGRES_PASSWORD="$PG_PASSWORD" \
  --from-literal=QI_AGENT_CHECKPOINT_DB="postgresql://postgres:$PG_PASSWORD@finsight-postgres:5432/finsight"
kubectl -n finsight create secret generic finsight-api-keys \
  --from-literal=QI_API_KEYS="$(openssl rand -hex 24)" \
  --from-literal=QI_ANON_COOKIE_SECRET="$(openssl rand -hex 32)"
kubectl -n finsight create secret generic finsight-llm --from-literal=DEEPSEEK_API_KEY=...  # 可选
```

### 认证与请求体上限

- **生产失败关闭**：`QI_PROFILE=production` 下未设置 `QI_API_KEYS` 会直接拒绝启动（除非显式配置 `QI_ALLOW_ANONYMOUS=1`）。调用方通过 `X-API-Key` 或 `Authorization: Bearer` 鉴权，支持逗号分隔多 Key 平滑轮换。匿名模式下为每个浏览器签发 `HttpOnly; SameSite=Lax` HMAC Cookie（多副本需共享 `QI_ANON_COOKIE_SECRET`）且禁止访问 `/agent/traces*`（`403`）。详见 [SECURITY.md](../../SECURITY.md)。
- **请求体上限**：`BodyLimitMiddleware` 在鉴权与路由前主动统计字节数（含 `Transfer-Encoding: chunked`），超过 `QI_MAX_REQUEST_BYTES`（默认 `1 MiB`）在 REST、`/a2a` 与 HTTP `/mcp` 上均立即返回 `413`。

## Kind 集群冒烟测试（CI `k8s-smoke`）

`deploy/k8s-smoke/smoke.sh` 通过 `deploy/k8s-smoke` overlay（标签 `smoke`，强制 NetworkPolicy）在临时 kind 集群中验证 `/ready`（Postgres + 三层离线数据）及更正层数值回答：

```bash
IMAGE=finsight:$(git rev-parse --short=7 HEAD) deploy/k8s-smoke/smoke.sh   # KEEP=1 保留集群
```
