# Deployment

FinSight ships as one container image that serves the API, the agent, the A2A endpoint and the built web UI (the MCP server is a separate entry point in the same image: `python -m query_intelligence.agent.mcp_server --transport http`). Sessions live in a LangGraph checkpointer: in memory, in a SQLite file, or in Postgres when several processes or replicas must share them.

## Docker

```bash
docker build -f docker/Dockerfile -t finsight:dev .
docker run -d --name finsight -p 8000:8000 \
  -e DEEPSEEK_API_KEY=... finsight:dev            # omit the key to run on the deterministic path only
curl http://127.0.0.1:8000/health
```

- Multi-stage build: wheels are compiled in `python:3.13`, the runtime is `python:3.13-slim` without a compiler (2 GB; torch is only added with `--build-arg WITH_TORCH=1`).
- Runs as uid 10001, with a Docker `HEALTHCHECK` on `/health`.
- The web UI is the committed build in `query_intelligence/web/dist`, so the image needs no Node toolchain; CI checks that the committed build matches `frontend/`.
- `docker/docker-compose.yml` adds optional `postgres` and `tracing` (Jaeger over OTLP) profiles.

Verified on 2026-09-25 (colima, arm64): the container becomes healthy, serves the UI, answers `/agent/chat` with a verified answer, publishes the A2A agent card and Prometheus metrics, and reports `/sources/health`. Load-test numbers are in [performance.md](performance.md).

## Kubernetes

`deploy/k8s/finsight.yaml` contains:

| Object | Notes |
|---|---|
| `Deployment/finsight-api` | 2 replicas; startup probe (NLU models load in about a minute), readiness and liveness on `/health`; requests 0.5 CPU / 2 GiB, limits 2 CPU / 3 GiB; rolling update with `maxUnavailable: 0`. |
| Pod security | `runAsNonRoot` (uid 10001), `readOnlyRootFilesystem`, all capabilities dropped, `RuntimeDefault` seccomp; `emptyDir` volumes for `/app/outputs`, `/app/state` and `/tmp`. |
| `HorizontalPodAutoscaler` | 2–6 replicas at 70% CPU. |
| `PodDisruptionBudget` | At least one replica stays up during node maintenance. |
| `StatefulSet/finsight-postgres` | Demo Postgres for sessions; use a managed database in production. |
| `ConfigMap` / `Secret` | Live-data switches, timeouts, rate limit; the checkpoint DSN; an optional `finsight-llm` secret with `DEEPSEEK_API_KEY`. |

```bash
kubeconform -strict -summary deploy/k8s/finsight.yaml     # 9 resources, all valid
kubectl apply -f deploy/k8s/finsight.yaml
kubectl -n finsight create secret generic finsight-llm --from-literal=DEEPSEEK_API_KEY=...   # optional
kubectl -n finsight rollout status deployment/finsight-api
```

### Why replicas can share a conversation

Every turn is a LangGraph run on thread `session_id`. With `QI_AGENT_CHECKPOINT_DB=postgresql://...` the checkpoints (conversation turns, a paused clarification, the last resolved entity) are in Postgres, so the next turn can land on any replica. `tests/test_agent_checkpoint_postgres.py` checks this with two service instances; on the k3s deployment a follow-up sent to replica B ("它的市净率呢") resolved the pronoun from a turn served by replica A.

Per-replica state that is not shared: the A2A task store, the in-memory trace buffer behind `/agent/traces` (JSON traces are per pod; export OTLP for a shared view) and the TTL caches of tools and live sources.

### Read-only root filesystem

The first rollout crash-looped: efinance creates `<site-packages>/efinance/data` when it is imported. The image now links that directory to `/tmp/efinance-data`, and the provider creates the link target before importing, so the pod runs with `readOnlyRootFilesystem: true`.

### Verified

On 2026-09-26, k3s v1.35 in colima (4 vCPU, 8 GB), image `finsight:dev`:

- Both API replicas ready behind `Service/finsight-api`; Postgres ready.
- `touch /app/x` inside a pod fails with "Read-only file system"; the process runs as uid 10001.
- A session started on one pod continued on the other with the correct coreference and a two-turn history.
