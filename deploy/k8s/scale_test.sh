#!/usr/bin/env bash
# Multi-replica scaling test and cross-replica session check on the k3s deployment.
#
#   IMAGE=finsight:$(git rev-parse --short=7 HEAD) OUT=docs/results/perf/k3s deploy/k8s/scale_test.sh
#
# For each replica count N (1, 2, 3) the HPA is pinned to N, the rollout is awaited, and a load
# generator pod *inside the cluster* runs scripts/load_test.py against Service/finsight-api with fresh
# connections (kube-proxy balances per connection, so keep-alive would pin a user to one pod). The load
# generator is limited to 0.5 CPU. Every pod is warmed first (see warm_pods). Live data is switched off for the test (offline snapshot and
# documents), so the numbers measure the service rather than upstream sites, and the per-IP rate limit is
# disabled because all load comes from one pod IP. Sessions use the Postgres checkpointer from the
# manifest.
#
# Afterwards two pods are addressed directly (kubectl port-forward to each pod) to show that a
# conversation started on one replica continues on another: turn 2 ("它的市净率呢") must resolve the
# pronoun from turn 1, which only the shared Postgres checkpoint can provide.
#
# Artifacts written to $OUT: load-test JSON per (replicas, users), pods/top snapshots, the cross-replica
# transcript, pod logs for the session, and the Postgres checkpoint rows.
set -euo pipefail

CTX="${CTX:-colima}"
NS=finsight
IMAGE="${IMAGE:-finsight:ops}"
OUT="${OUT:-docs/results/perf/k3s}"
REPLICAS="${REPLICAS:-1 2 3}"
USERS="${USERS:-8 32}"
REQUESTS="${REQUESTS:-20}"
K="kubectl --context $CTX -n $NS"
mkdir -p "$OUT"

echo "== deploy $IMAGE with load-test settings"
# The manifests commit no Secret; a throwaway demo password is generated on first use.
kubectl --context "$CTX" get namespace "$NS" >/dev/null 2>&1 || kubectl --context "$CTX" create namespace "$NS" >/dev/null
if ! $K get secret finsight-db >/dev/null 2>&1; then
  PG_PASSWORD="$(openssl rand -hex 16)"
  $K create secret generic finsight-db --from-literal=POSTGRES_PASSWORD="$PG_PASSWORD" \
    --from-literal=QI_AGENT_CHECKPOINT_DB="postgresql://postgres:$PG_PASSWORD@finsight-postgres:5432/finsight" >/dev/null
fi
kubectl --context "$CTX" apply -k deploy/k8s >/dev/null
$K set image deployment/finsight-api api="$IMAGE" >/dev/null
$K set env deployment/finsight-api QI_USE_LIVE_MARKET=0 QI_USE_LIVE_NEWS=0 QI_USE_LIVE_ANNOUNCEMENT=0 \
  QI_USE_LIVE_MACRO=0 QI_RATE_LIMIT_PER_MINUTE=0 >/dev/null

pin_replicas() {
  $K patch hpa finsight-api --type merge -p "{\"spec\":{\"minReplicas\":$1,\"maxReplicas\":$1}}" >/dev/null
  $K scale deployment/finsight-api --replicas "$1" >/dev/null
  $K rollout status deployment/finsight-api --timeout=600s >/dev/null
  # Wait until exactly N ready pods remain (terminating pods from a scale-down must be gone).
  for _ in $(seq 1 120); do
    ready=$($K get pods -l app=finsight-api --no-headers 2>/dev/null | awk '$2=="1/1" && $3=="Running"' | wc -l | tr -d ' ')
    total=$($K get pods -l app=finsight-api --no-headers 2>/dev/null | wc -l | tr -d ' ')
    [ "$ready" = "$1" ] && [ "$total" = "$1" ] && return 0
    sleep 5
  done
  echo "replicas did not settle at $1" >&2
  return 1
}

# Every pod is warmed with the full question rotation before measuring: NLU components load lazily on a
# pod's first requests (an English question costs seconds the first time), and a cold replica would
# otherwise dominate the tail of the multi-replica runs.
warm_pods() {
  local pod
  for pod in $($K get pods -l app=finsight-api -o jsonpath='{.items[*].metadata.name}'); do
    $K exec "$pod" -- python -c '
import json, urllib.request
from scripts.load_test import QUESTIONS
for round_ in range(2):
    for i, q in enumerate(QUESTIONS):
        req = urllib.request.Request("http://127.0.0.1:8000/agent/chat", data=json.dumps({"query": q, "mode": "workflow", "session_id": f"warm{round_}x{i}"}).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
' >/dev/null
  done
}

# Completed runs per pod (from each pod's own /metrics), to show how the Service spread the load.
runs_per_pod() {
  local pod n
  for pod in $($K get pods -l app=finsight-api -o jsonpath='{.items[*].metadata.name}'); do
    n=$($K exec "$pod" -- python -c '
import urllib.request
text = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode()
print(int(sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines() if line.startswith("finsight_agent_runs_total{"))))
')
    echo "$pod $n"
  done
}

run_load() { # replicas users
  local name="loadgen-r$1-u$2"
  $K delete pod "$name" --ignore-not-found >/dev/null
  runs_per_pod >"$OUT/runs-per-pod-r$1-u$2.before.txt"
  # finsight.io/client=true is the label the finsight-api NetworkPolicy admits from inside the namespace.
  $K run "$name" --image="$IMAGE" --image-pull-policy=IfNotPresent --restart=Never --labels=finsight.io/client=true \
    --overrides='{"spec":{"containers":[{"name":"'"$name"'","image":"'"$IMAGE"'","resources":{"limits":{"cpu":"500m","memory":"512Mi"}},
      "command":["python","-m","scripts.load_test","--base-url","http://finsight-api.finsight.svc.cluster.local",
      "--users","'"$2"'","--requests","'"$REQUESTS"'","--fresh-connections","--label","k3s replicas='"$1"'",
      "--out","/tmp/lt.json"]}]}}' >/dev/null
  # Snapshot per-pod CPU while the test runs.
  sleep 25
  $K top pods --no-headers >"$OUT/top-r$1-u$2.txt" 2>&1 || true
  $K wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$name" --timeout=1200s >/dev/null
  # The script prints the report (without per-request rows) as its last JSON document.
  # kubectl logs goes through the kubelet port, which colima's port forward sometimes drops (unexpected
  # EOF); the container runtime is Docker, so fall back to the container's own log.
  { $K logs "$name" 2>/dev/null || docker logs "$(docker ps -aq --filter "name=k8s_${name}_${name}" | head -1)" 2>/dev/null; } | python3 -c 'import sys,json; t=sys.stdin.read(); print(json.dumps(json.loads(t[t.index("{"):]), ensure_ascii=False, indent=1))' \
    >"$OUT/load_test-k3s-r$1-u$2.json"
  $K delete pod "$name" >/dev/null
  runs_per_pod >"$OUT/runs-per-pod-r$1-u$2.after.txt"
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); l=r["latency_ms"]; print(f"replicas={sys.argv[2]} users={r["users"]} rps={r["throughput_rps"]} p50={l["p50"]} p95={l["p95"]} p99={l["p99"]} err={r["error_rate"]}")' \
    "$OUT/load_test-k3s-r$1-u$2.json" "$1"
}

# SKIP_LOAD=1 runs only the cross-replica session check.
[ "${SKIP_LOAD:-0}" = "1" ] && REPLICAS=""
for replicas in $REPLICAS; do
  echo "== replicas=$replicas"
  pin_replicas "$replicas"
  warm_pods
  $K get pods -o wide >"$OUT/pods-r$replicas.txt"
  for u in $USERS; do run_load "$replicas" "$u"; done
done

echo "== cross-replica session (2 replicas)"
pin_replicas 2
$K get pods -o wide >"$OUT/pods-session.txt"
# jsonpath output has no trailing newline, so ``read`` returns 1 at EOF: tolerate it under ``set -e``.
read -r POD_A POD_B < <($K get pods -l app=finsight-api -o jsonpath='{.items[0].metadata.name} {.items[1].metadata.name}') || true
$K port-forward "pod/$POD_A" 18803:8000 >/dev/null 2>&1 & PF_A=$!
$K port-forward "pod/$POD_B" 18804:8000 >/dev/null 2>&1 & PF_B=$!
trap 'kill $PF_A $PF_B 2>/dev/null || true' EXIT
sleep 4
SESSION="xreplica$(date +%s)"
{
  echo "session: $SESSION"
  echo "turn 1 -> $POD_A"
  curl --noproxy '*' -s http://127.0.0.1:18803/agent/chat -H 'Content-Type: application/json' \
    -d "{\"query\":\"贵州茅台的市盈率是多少？\",\"mode\":\"workflow\",\"session_id\":\"$SESSION\"}" \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); print(json.dumps({k:r.get(k) for k in ("status","session_id","trace_id","route","answer")}, ensure_ascii=False, indent=1)); print("entities:", [e for e in (r.get("nlu_summary") or {}).get("entities") or []])'
  echo "turn 2 -> $POD_B"
  curl --noproxy '*' -s http://127.0.0.1:18804/agent/chat -H 'Content-Type: application/json' \
    -d "{\"query\":\"它的市净率呢\",\"mode\":\"workflow\",\"session_id\":\"$SESSION\"}" \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); print(json.dumps({k:r.get(k) for k in ("status","session_id","trace_id","route","answer")}, ensure_ascii=False, indent=1)); print("entities:", [e for e in (r.get("nlu_summary") or {}).get("entities") or []])'
  echo "session history read back through $POD_A"
  curl --noproxy '*' -s "http://127.0.0.1:18803/agent/sessions/$SESSION" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(json.dumps(r, ensure_ascii=False)[:1500])'
} | tee "$OUT/cross-replica-session.txt"
# Which pod served which turn: each pod writes its own run traces (emptyDir), keyed by session.
{
  for pod in "$POD_A" "$POD_B"; do
    echo "--- traces for $SESSION on $pod"
    $K exec "$pod" -- sh -c "grep -l '\"session_id\": \"$SESSION\"' /app/outputs/traces/*/*.json 2>/dev/null | while read f; do python -c 'import json,sys; t=json.load(open(sys.argv[1])); print(t[\"trace_id\"], t[\"turn_index\"], t[\"query\"], t[\"route\"])' \$f; done" || true
    echo "--- last access-log lines on $pod"
    { $K logs "$pod" --tail=400 2>/dev/null || docker logs --tail 400 "$(docker ps -q --filter "name=k8s_api_${pod}_" | head -1)" 2>&1; } \
      | grep -F "POST /agent/chat" | tail -3 || true
  done
} | tee "$OUT/cross-replica-pod-logs.txt"
$K exec finsight-postgres-0 -- psql -U postgres -d finsight -c \
  "select thread_id, count(*) as checkpoints, max(checkpoint_id) as latest from checkpoints where thread_id='$SESSION' group by thread_id;" \
  >"$OUT/cross-replica-postgres.txt"
cat "$OUT/cross-replica-postgres.txt"
echo "done: $OUT"
