# Deployment

Languages: English | [中文](zh/deployment.md)

FinSight packages the FastAPI API, research agent, A2A 1.0 server, and built React 19 UI into a single hardened container image (the MCP server runs from the same image via `python -m query_intelligence.agent.mcp_server --transport http`). Multi-replica sessions, A2A tasks, traces, and rate limits share PostgreSQL via `QI_AGENT_CHECKPOINT_DB`.

## Docker

```bash
TAG=$(git rev-parse --short=7 HEAD)
docker build -f docker/Dockerfile -t finsight:$TAG .
docker run -d --name finsight -p 127.0.0.1:8000:8000 \
  -e DEEPSEEK_API_KEY=... finsight:$TAG          # omit key for deterministic mode
curl http://127.0.0.1:8000/health                # liveness
curl http://127.0.0.1:8000/ready                 # checkpointer, LLM config, retrieval index, offline data layers
```

- **Multi-stage & hardened**: Compiles wheels in `python:3.13`, runs on `python:3.13-slim` (`~2 GB`; optional `--build-arg WITH_TORCH=1` for FinBERT), runs as non-root `uid 10001`, supports `--read-only` root filesystem (linking `efinance/data` to `/tmp/efinance-data`), and ships the prebuilt UI from `query_intelligence/web/dist`.
- **Offline data layers & readiness**: The image bundles v1 seed (`data/structured_data.json`), snapshot extension (`data/snapshot/structured_data_ext.json`), and primary-source corrections (`data/snapshot/structured_data_fix.json`). `GET /ready` reports every layer's load status, record count, version, and SHA-256; if an enabled layer is missing, `/ready` returns `503` and chat/claim-check mark `degraded: ["data_layer_missing:corrections"]`.
- **Docker Compose profiles (`docker/docker-compose.yml`)**: Optional `postgres`, `tracing` (Jaeger OTLP), and `monitoring` (Prometheus + Grafana + Jaeger) profiles—all host ports bound to `127.0.0.1`. Set `FINSIGHT_EGRESS_PROXY` when containers reach upstream APIs through an HTTP(S) proxy.

## Kubernetes (`deploy/k8s/`)

`deploy/k8s/kustomization.yaml` applies `finsight.yaml` and `networkpolicy.yaml`:

| Resource | Configuration |
|---|---|
| `Deployment/finsight-api` | 2 replicas, `maxUnavailable: 0`, `startupProbe`/`livenessProbe` on `/health`, `readinessProbe` on `/ready`; 0.5–2 CPU, 2–3 GiB RAM; `runAsNonRoot` (`uid 10001`), `readOnlyRootFilesystem: true`, dropped capabilities, `RuntimeDefault` seccomp, `emptyDir` for `/tmp`, `/app/state`, `/app/outputs`. |
| `HorizontalPodAutoscaler` & `PodDisruptionBudget` | 2–6 replicas at 70% CPU; `minAvailable: 1`. |
| `StatefulSet/finsight-postgres` | Demo Postgres for shared checkpoints, A2A tasks, traces, and rate limits (replace with managed Postgres in production). |
| `ConfigMap` & external `Secrets` | Sets `QI_PROFILE=production`, timeouts, rate limits, LLM endpoint; requires non-optional `finsight-db` and `finsight-api-keys` Secrets (plus optional `finsight-llm`). |
| `NetworkPolicy` (`networkpolicy.yaml`) | Restricts API ingress to ingress controller, Prometheus, and `finsight.io/client=true` pods; restricts egress to DNS, Postgres `:5432`, and public `80/443` (blocking private/metadata CIDRs); restricts Postgres ingress to API pods only. |

### Image tags

Committed manifests use the deploy-time placeholder `set-at-deploy` (never `:latest` or a stale commit hash) so an unconfigured apply fails fast at `ImagePullBackOff` rather than running an unknown build:

```bash
TAG=$(git rev-parse --short=7 HEAD)
docker build -f docker/Dockerfile -t finsight:$TAG .
(cd deploy/k8s && kustomize edit set image finsight=finsight:$TAG)
kubectl apply -k deploy/k8s
kubectl -n finsight rollout status deployment/finsight-api
```

CI validates this policy via `tests/test_deploy_tag_policy.py` and `kubeconform -strict`.

### Secrets

No Secret manifests are committed. Provision them before applying via `kubectl create secret generic`, `envsubst < deploy/k8s/secret.template.yaml | kubectl apply -f -`, or External Secrets Operator:

```bash
kubectl create namespace finsight
kubectl -n finsight create secret generic finsight-db \
  --from-literal=POSTGRES_PASSWORD="$PG_PASSWORD" \
  --from-literal=QI_AGENT_CHECKPOINT_DB="postgresql://postgres:$PG_PASSWORD@finsight-postgres:5432/finsight"
kubectl -n finsight create secret generic finsight-api-keys \
  --from-literal=QI_API_KEYS="$(openssl rand -hex 24)" \
  --from-literal=QI_ANON_COOKIE_SECRET="$(openssl rand -hex 32)"
kubectl -n finsight create secret generic finsight-llm --from-literal=DEEPSEEK_API_KEY=...  # optional
```

### Authentication

- `QI_PROFILE=production` fails closed (`InsecureConfigurationError`) unless `QI_API_KEYS` is set (or `QI_ALLOW_ANONYMOUS=1` is explicitly enabled).
- Clients authenticate with `X-API-Key: <key>` or `Authorization: Bearer <key>` (comma-separated keys support zero-downtime rotation).
- When anonymous access is opted in, each browser receives an HMAC-signed `HttpOnly; SameSite=Lax` cookie (`finsight_anon`, requiring a shared `QI_ANON_COOKIE_SECRET` across replicas) and receives `403` on `/agent/traces*`. See [SECURITY.md](../SECURITY.md).

### Request body limit

`BodyLimitMiddleware` (`query_intelligence/api/security.py`) enforces `QI_MAX_REQUEST_BYTES` (default `1 MiB`) eagerly before auth and routing—including `Transfer-Encoding: chunked` payloads across REST, `/a2a`, and HTTP `/mcp`—returning `413 {"detail": "request body too large"}` with `Connection: close`. (Starlette's lazy built-in middleware would otherwise be swallowed into 400 or JSON-RPC `-32603` errors by downstream wrappers; verified by `tests/test_api_security.py`.)

## Kind Smoke Test (CI Job `k8s-smoke`)

`deploy/k8s-smoke/smoke.sh` deploys the real kustomization through the `deploy/k8s-smoke` overlay (image tag `smoke`, offline data, NetworkPolicy enforced) into a throwaway kind cluster, asserting `GET /ready` (Postgres + all three offline data layers) and verified primary-source corrected figures:

```bash
IMAGE=finsight:$(git rev-parse --short=7 HEAD) deploy/k8s-smoke/smoke.sh   # KEEP=1 preserves the cluster
```
