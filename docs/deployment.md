# Deployment

Languages: English | [中文](zh/deployment.md)

FinSight ships as one container image that serves the API, the agent, the A2A endpoint and the built web UI (the MCP server is a separate entry point in the same image: `python -m query_intelligence.agent.mcp_server --transport http`). Sessions live in a LangGraph checkpointer: in memory, in a SQLite file, or in Postgres when several processes or replicas must share them.

## Docker

```bash
TAG=$(git rev-parse --short=7 HEAD)               # images are tagged with the commit they were built from
docker build -f docker/Dockerfile -t finsight:$TAG .
docker run -d --name finsight -p 8000:8000 \
  -e DEEPSEEK_API_KEY=... finsight:$TAG           # omit the key to run on the deterministic path only
curl http://127.0.0.1:8000/health                 # liveness
curl http://127.0.0.1:8000/ready                  # readiness: checkpointer, model config, retrieval index
```

- Multi-stage build: wheels are compiled in `python:3.13`, the runtime is `python:3.13-slim` without a compiler (2 GB; torch is only added with `--build-arg WITH_TORCH=1`).
- Runs as uid 10001, with a Docker `HEALTHCHECK` on `/health`.
- `.dockerignore` keeps `.git`, `.env*`, virtualenvs, `frontend/` (including `node_modules`), outputs,
  docs and tests out of the build context; the image only copies the runtime directories.
- The web UI is the committed build in `query_intelligence/web/dist`, so the image needs no Node toolchain; CI checks that the committed build matches `frontend/`.
- `docker/docker-compose.yml` adds optional profiles: `postgres`, `tracing` (Jaeger over OTLP) and
  `monitoring` (Prometheus with the alert rules, Grafana with the FinSight dashboard, Jaeger); see
  [a2a-and-observability.md](a2a-and-observability.md#dashboards-alerts-and-the-monitoring-stack).
  `FINSIGHT_EGRESS_PROXY` passes an HTTP(S) proxy to the app for networks where containers have no
  direct internet access.

### API wiring for the ops metrics and the active source probe

`/metrics` serves the scrape-time breaker and pool metrics (`OpsMetricsCollector`), and
`/sources/health` accepts `?probe=1`. The wiring was first kept as `deploy/patches/app-ops-wiring.patch`
(the images measured in [performance.md](performance.md) were built with it); it is part of
`api/app.py` since `e9d9a9a` and the patch file has been deleted. `tests/test_source_reliability.py`
covers both endpoints.

Verified on 2026-09-25 (colima, arm64): the container becomes healthy, serves the UI, answers `/agent/chat` with a verified answer, publishes the A2A agent card and Prometheus metrics, and reports `/sources/health`. Load-test numbers are in [performance.md](performance.md).

## Kubernetes

`deploy/k8s/kustomization.yaml` applies `finsight.yaml` and `networkpolicy.yaml` and pins the image
tag. `finsight.yaml` contains:

| Object | Notes |
|---|---|
| `Deployment/finsight-api` | 2 replicas; startup and liveness probes on `/health` (NLU models load in about a minute); readiness on `/ready`, which also checks the Postgres checkpointer, the LLM config and the retrieval index, so a replica that loses its database stops taking traffic without being restarted; requests 0.5 CPU / 2 GiB, limits 2 CPU / 3 GiB; rolling update with `maxUnavailable: 0`. |
| Pod security | `runAsNonRoot` (uid 10001), `readOnlyRootFilesystem`, all capabilities dropped, `RuntimeDefault` seccomp; `emptyDir` volumes for `/app/outputs`, `/app/state` and `/tmp`. |
| `HorizontalPodAutoscaler` | 2–6 replicas at 70% CPU. |
| `PodDisruptionBudget` | At least one replica stays up during node maintenance. |
| `StatefulSet/finsight-postgres` | Demo Postgres for sessions; use a managed database in production. |
| `ConfigMap` | Live-data switches, timeouts, rate limit, LLM endpoint and model. |
| Secrets (not committed) | `finsight-db` (Postgres password and checkpoint DSN, required) and `finsight-llm` (`DEEPSEEK_API_KEY`, optional) are referenced by name; see [Secrets](#secrets). |
| `NetworkPolicy` ×2 (`networkpolicy.yaml`) | See [Network policy](#network-policy). |

```bash
kubectl kustomize deploy/k8s | kubeconform -strict -summary -   # 10 resources, all valid (also run in CI)
kubectl create namespace finsight
kubectl -n finsight create secret generic finsight-db \
  --from-literal=POSTGRES_PASSWORD="$PG_PASSWORD" \
  --from-literal=QI_AGENT_CHECKPOINT_DB="postgresql://postgres:$PG_PASSWORD@finsight-postgres:5432/finsight"
kubectl -n finsight create secret generic finsight-llm --from-literal=DEEPSEEK_API_KEY=...   # optional
kubectl apply -k deploy/k8s
kubectl -n finsight rollout status deployment/finsight-api
```

### Image tags

Images are tagged with the short hash of the commit they were built from (`finsight:<commit>`), never
`:dev` or `:latest`, so `kubectl get pods -o jsonpath='{..image}'` names the exact source. The tag lives
in one place, `images[].newTag` in `deploy/k8s/kustomization.yaml`:

```bash
TAG=$(git rev-parse --short=7 HEAD)          # from a clean working tree
docker build -f docker/Dockerfile -t finsight:$TAG .   # push to your registry as needed
(cd deploy/k8s && kustomize edit set image finsight=finsight:$TAG)
```

`finsight.yaml` alone names `finsight:set-by-kustomization`, which does not exist, so applying it without
kustomize fails visibly instead of running an unknown build. CI builds every push as `finsight:<github.sha>`.

### Secrets

No Secret is committed (the earlier manifest carried a plaintext `change-me` password). Choose one:

- `kubectl create secret generic` as above;
- `deploy/k8s/secret.template.yaml` with placeholders: `POSTGRES_PASSWORD=$(openssl rand -hex 24) envsubst '$POSTGRES_PASSWORD' < deploy/k8s/secret.template.yaml | kubectl apply -f -`;
- a secret manager through the [External Secrets Operator](https://external-secrets.io/), for example:

  ```yaml
  apiVersion: external-secrets.io/v1
  kind: ExternalSecret
  metadata: {name: finsight-db, namespace: finsight}
  spec:
    refreshInterval: 1h
    secretStoreRef: {kind: ClusterSecretStore, name: your-store}
    target: {name: finsight-db}
    data:
      - {secretKey: POSTGRES_PASSWORD, remoteRef: {key: finsight/postgres, property: password}}
      - {secretKey: QI_AGENT_CHECKPOINT_DB, remoteRef: {key: finsight/postgres, property: dsn}}
  ```

The Deployment and the StatefulSet reference `finsight-db` without `optional`, so pods stay pending
until it exists.

### Network policy

`networkpolicy.yaml` isolates both workloads (enforced by CNIs that implement NetworkPolicy, including
k3s' built-in controller):

| Policy | Ingress | Egress |
|---|---|---|
| `finsight-api` | TCP 8000 from the ingress controller (ingress-nginx, or Traefik in `kube-system`), from Prometheus in `monitoring`, and from pods labelled `finsight.io/client=true` (the scale-test load generator) | DNS to kube-dns; TCP 5432 to the Postgres pods; TCP 80/443 to public addresses only (all private, link-local/metadata and CGNAT ranges excluded) |
| `finsight-postgres` | TCP 5432 from the API pods only | none |

Kubelet probes are unaffected (node-to-pod traffic is always allowed). NetworkPolicy matches IPs, not
host names, so the LLM gateway and the data sources cannot be pinned by name. With an FQDN-aware CNI
(Cilium `toFQDNs`, Calico DNS policy) or an egress proxy, allow only: the LLM gateway host from
`DEEPSEEK_BASE_URL`, `*.eastmoney.com`, `finance.sina.com.cn`, `money.finance.sina.com.cn`,
`hq.sinajs.cn`, `web.ifzq.gtimg.cn`, `qt.gtimg.cn`, `basic.10jqka.com.cn`, `d.10jqka.com.cn`,
`q.10jqka.com.cn`, `www.csindex.com.cn`, `yield.chinabond.com.cn`, `www.cninfo.com.cn`,
`static.cninfo.com.cn`, `stock.xueqiu.com`, `api.tushare.pro` (from
`query_intelligence/integrations/sources/catalog.py`). A managed Postgres outside the cluster needs its
address as an `ipBlock` in place of the Postgres pod selector.

### Why replicas can share a conversation

Every turn is a LangGraph run on thread `session_id`. With `QI_AGENT_CHECKPOINT_DB=postgresql://...` the checkpoints (conversation turns, a paused clarification, the last resolved entity) are in Postgres, so the next turn can land on any replica. `tests/test_agent_checkpoint_postgres.py` checks this with two service instances (it runs in CI against a Postgres service; a recorded local run with the checkpoint rows is in [results/postgres/two-replica-checkpointer.md](results/postgres/two-replica-checkpointer.md)); on the k3s deployment a follow-up sent to replica B ("它的市净率呢") resolved the pronoun from a turn served by replica A.

The same DSN also moves the A2A task store (`finsight_a2a_tasks`) and the trace store behind `/agent/traces` (`finsight_agent_traces`) to Postgres, so any replica can serve `GetTask`, continue an `input-required` task, or show any run in the inspector (see [a2a-and-observability.md](a2a-and-observability.md#shared-stores-for-several-replicas); `QI_A2A_TASK_DB` / `QI_AGENT_TRACE_DB=memory` opt out). Still per replica: the rate limiter and the TTL caches of tools and live sources.

### Read-only root filesystem

The first rollout crash-looped: efinance creates `<site-packages>/efinance/data` when it is imported. The image now links that directory to `/tmp/efinance-data`, and the provider creates the link target before importing, so the pod runs with `readOnlyRootFilesystem: true`.

### Verified

On 2026-09-26, k3s v1.35 in colima (4 vCPU, 8 GB):

- Image `finsight:dev`: both API replicas ready behind `Service/finsight-api`; Postgres ready.
  `touch /app/x` inside a pod fails with "Read-only file system"; the process runs as uid 10001.
- Image `finsight:merged`, with committed evidence in `docs/results/perf/k3s/`
  ([`deploy/k8s/scale_test.sh`](../deploy/k8s/scale_test.sh)):
  - 1, 2 and 3 replicas behind the Service, sessions in Postgres, 800 requests per replica count from
    a load-generator pod, 0 errors, evenly spread (`runs-per-pod-*.txt`), 3.75 → 6.07 → 11.72 req/s at
    32 users ([performance.md](performance.md#3-multi-replica-scaling-on-k3s-with-postgres-sessions));
    `kubectl get pods -o wide` and `kubectl top pods` output for each run.
  - A session started on one pod continued on the other: turn 2 ("它的市净率呢", sent only to pod B)
    resolved 贵州茅台 from turn 1 (sent only to pod A). Each pod's trace directory holds exactly its own
    turn, and Postgres holds 2 checkpoints for the thread (`cross-replica-*.txt`).
- `kubectl logs` through colima's kubelet port intermittently failed with "unexpected EOF"; the scripts
  fall back to `docker logs` of the pod's container, which is the same stream on a Docker-runtime node.

The scale test changes the deployment for the measurement (live data off, rate limit off, HPA pinned).
Re-apply `kubectl apply -k deploy/k8s` afterwards to restore the defaults.
