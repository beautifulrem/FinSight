#!/usr/bin/env bash
# Workflow-path load matrix: durability "before" vs "after", one container per configuration.
#
#   docker/perf_matrix.sh                      # writes docs/results/perf/workflow/*.json
#
# Configurations (image:durability):
#   before-0678585  finsight:before-0678585, no QI_AGENT_DURABILITY (the code before the fix: LangGraph's
#                   default per-step checkpoints and the untrimmed final state), i.e. the original "before"
#   async           current image with QI_AGENT_DURABILITY=async (per-step checkpoints, trimmed state)
#   exit            current image with QI_AGENT_DURABILITY=exit (the default: one checkpoint per run)
# Each container is fresh (empty SQLite session file) and serves 1, 8 and 32 users in that order, 20
# requests per user, workflow mode (no LLM), live data off, exactly as in the original measurement.
set -euo pipefail
IMAGE="${IMAGE:-finsight:merged}"
BEFORE_IMAGE="${BEFORE_IMAGE:-finsight:before-0678585}"
OUT="${OUT:-docs/results/perf/workflow}"
PORT="${PORT:-8802}"
PY="${PY:-.venv/bin/python}"
mkdir -p "$OUT"

run_config() { # name image durability
  local name=$1 image=$2 durability=$3
  docker rm -f finsight-perf >/dev/null 2>&1 || true
  local env_args=(-e QI_USE_LIVE_MARKET=0 -e QI_USE_LIVE_NEWS=0 -e QI_USE_LIVE_ANNOUNCEMENT=0 -e QI_USE_LIVE_MACRO=0)
  [ -n "$durability" ] && env_args+=(-e "QI_AGENT_DURABILITY=$durability")
  docker run -d --name finsight-perf -p "$PORT:8000" "${env_args[@]}" "$image" >/dev/null
  for _ in $(seq 1 120); do
    curl --noproxy '*' -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    sleep 2
  done
  docker inspect finsight-perf --format '{{.Config.Image}} {{range .Config.Env}}{{println .}}{{end}}' | grep -E "finsight|QI_AGENT_DURABILITY|QI_AGENT_CHECKPOINT_DB" >"$OUT/container-$name.txt"
  for users in 1 8 32; do
    "$PY" -m scripts.load_test --base-url "http://127.0.0.1:$PORT" --users "$users" --requests 20 \
      --label "$name ($image, durability=${durability:-langgraph-default})" --out "$OUT/load_test-$name-$users.json" \
      | "$PY" -c 'import sys,json; r=json.load(sys.stdin); l=r["latency_ms"]; print(r["label"], "users", r["users"], "rps", r["throughput_rps"], "p50", l["p50"], "p95", l["p95"], "p99", l["p99"], "err", r["error_rate"])'
  done
  docker exec finsight-perf sh -c 'ls -la /app/state/' >>"$OUT/container-$name.txt" 2>&1 || true
  docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' finsight-perf >>"$OUT/container-$name.txt"
  docker rm -f finsight-perf >/dev/null
}

run_config before-0678585 "$BEFORE_IMAGE" ""
run_config async "$IMAGE" async
run_config exit "$IMAGE" exit
