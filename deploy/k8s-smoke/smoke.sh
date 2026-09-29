#!/usr/bin/env bash
# Kubernetes smoke test: apply the real kustomization (through the deploy/k8s-smoke overlay) to a
# throwaway kind cluster with a test Secret, and wait until the API pod reports /ready.
#
#   deploy/k8s-smoke/smoke.sh                       # builds finsight:smoke from the working tree
#   IMAGE=finsight:abc1234 deploy/k8s-smoke/smoke.sh  # use an already built image (CI)
#   CLUSTER=finsight-smoke KEEP=1 deploy/k8s-smoke/smoke.sh   # keep the cluster afterwards
#
# Needs docker, kind and kubectl. The node pulls nothing: every image is loaded from the host. Checks:
#   1. kustomize renders; kubectl applies it (Namespace, ConfigMap, Postgres StatefulSet + PVC on kind's
#      default storage class, API Deployment, Services, HPA, PDB, NetworkPolicies);
#   2. Postgres becomes ready; the API rollout completes (its readinessProbe is GET /ready, which checks
#      the Postgres checkpointer, the model config and the retrieval index);
#   3. from a pod labelled finsight.io/client=true (the in-namespace client the NetworkPolicy admits):
#      GET /ready = 200 with checkpointer ok, and one verified /agent/chat answer on the offline path.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLUSTER="${CLUSTER:-finsight-smoke}"
NS=finsight
IMAGE="${IMAGE:-}"
TIMEOUT="${TIMEOUT:-900s}"
CTX="kind-${CLUSTER}"
POSTGRES_IMAGE=postgres:16-alpine  # as in deploy/k8s/finsight.yaml
CURL_IMAGE=curlimages/curl:8.11.1
k() { kubectl --context "$CTX" "$@"; }
log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

cleanup() {
  status=$?
  if [ "$status" != 0 ]; then
    log "FAILED (exit $status); diagnostics"
    k -n "$NS" get pods,pvc,svc -o wide || true
    k -n "$NS" describe pods || true
    k -n "$NS" logs deploy/finsight-api --tail=80 || true
  fi
  if [ -z "${KEEP:-}" ]; then kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true; fi
  exit "$status"
}
trap cleanup EXIT

log "commit $(git -C "$ROOT" rev-parse --short=7 HEAD) (working tree clean: $(test -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" && echo yes || echo no))"
log "versions: $(kind version) / kubectl $(kubectl version --client -o json | python3 -c 'import json,sys; print(json.load(sys.stdin)["clientVersion"]["gitVersion"])')"

if [ -z "$IMAGE" ]; then
  IMAGE=finsight:smoke
  log "building $IMAGE"
  docker build -q -f "$ROOT/docker/Dockerfile" -t "$IMAGE" "$ROOT"
fi
if [ "$IMAGE" != finsight:smoke ]; then docker tag "$IMAGE" finsight:smoke; fi

if ! kind get clusters | grep -qx "$CLUSTER"; then
  log "creating kind cluster $CLUSTER"
  kind create cluster --name "$CLUSTER" --wait 180s
fi
# Every image is pulled by the host's Docker and loaded into the node, so the node never pulls from a
# registry: faster, and it works when the host reaches registries through a proxy the node cannot use
# (kind copies HTTP(S)_PROXY into the node, and a 127.0.0.1 proxy is unreachable from inside it).
# Pulled images are multi-platform indexes with only one platform present locally, and
# `kind load docker-image` then fails ("content digest ... not found"), so they are loaded as
# single-platform archives.
log "loading finsight:smoke, $POSTGRES_IMAGE and $CURL_IMAGE into the cluster"
kind load docker-image finsight:smoke --name "$CLUSTER"
archive="$(mktemp -t finsight-smoke-image.XXXXXX)"
for image in "$POSTGRES_IMAGE" "$CURL_IMAGE"; do
  docker image inspect "$image" >/dev/null 2>&1 || docker pull -q "$image"
  docker save --platform "linux/$(docker image inspect "$image" --format '{{.Architecture}}')" -o "$archive" "$image"
  kind load image-archive "$archive" --name "$CLUSTER"
done
rm -f "$archive"

log "rendering and applying deploy/k8s-smoke (base deploy/k8s)"
kubectl kustomize "$ROOT/deploy/k8s-smoke" > /tmp/finsight-smoke-rendered.yaml
grep -E '^kind:' /tmp/finsight-smoke-rendered.yaml | sort | uniq -c
k apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata: {name: $NS}
EOF
# Test Secret with a throwaway password (never committed; deploy/k8s renders no Secret).
PG_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
k -n "$NS" create secret generic finsight-db \
  --from-literal=POSTGRES_PASSWORD="$PG_PASSWORD" \
  --from-literal=QI_AGENT_CHECKPOINT_DB="postgresql://postgres:${PG_PASSWORD}@finsight-postgres:5432/finsight" \
  --dry-run=client -o yaml | k apply -f -
# The production profile refuses to start without API keys (C3): a throwaway key for this run.
API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
k -n "$NS" create secret generic finsight-api-keys \
  --from-literal=QI_API_KEYS="$API_KEY" \
  --from-literal=QI_ANON_COOKIE_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" \
  --dry-run=client -o yaml | k apply -f -
k apply -f /tmp/finsight-smoke-rendered.yaml

log "waiting for Postgres"
k -n "$NS" rollout status statefulset/finsight-postgres --timeout="$TIMEOUT"
log "waiting for the API rollout (readinessProbe = GET /ready)"
started=$(date +%s)
k -n "$NS" rollout status deployment/finsight-api --timeout="$TIMEOUT"
log "API ready after $(( $(date +%s) - started )) s"
k -n "$NS" get pods -o wide

log "in-cluster checks from a finsight.io/client=true pod"
k -n "$NS" delete pod smoke-client --ignore-not-found >/dev/null
# shellcheck disable=SC2016  # expanded by the shell inside the pod
k -n "$NS" run smoke-client --restart=Never --labels=finsight.io/client=true \
  --env="API_KEY=$API_KEY" \
  --image="$CURL_IMAGE" --image-pull-policy=Never --command -- sh -c '
    set -e
    curl -fsS -o /tmp/ready.json -w "GET /ready -> %{http_code}\n" http://finsight-api/ready
    cat /tmp/ready.json; echo
    curl -fsS -X POST http://finsight-api/agent/chat -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
      -d "{\"query\":\"贵州茅台的市盈率是多少\",\"mode\":\"workflow\"}" -o /tmp/chat.json \
      -w "POST /agent/chat -> %{http_code}\n"
    head -c 400 /tmp/chat.json; echo
    echo "answer status: $(grep -o "\"status\":\"[a-z_]*\"" /tmp/chat.json | head -1)"
    echo "verification: $(grep -o "\"verification\":{\"passed\":[a-z]*" /tmp/chat.json | head -1)"'
k -n "$NS" wait pod/smoke-client --for=jsonpath='{.status.phase}'=Succeeded --timeout=180s
out="$(k -n "$NS" logs smoke-client)"
printf '%s\n' "$out"
printf '%s\n' "$out" | grep -q 'GET /ready -> 200'
printf '%s\n' "$out" | grep -q '"checkpointer":{"ok":true'
printf '%s\n' "$out" | grep -q 'POST /agent/chat -> 200'
printf '%s\n' "$out" | grep -q 'answer status: "status":"ok"'
printf '%s\n' "$out" | grep -q 'verification: "verification":{"passed":true'
log "SMOKE OK"
