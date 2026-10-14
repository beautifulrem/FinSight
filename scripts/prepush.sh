#!/usr/bin/env bash
# shellcheck disable=SC2329  # the step functions below are invoked indirectly, through run()
# Pre-push check: run locally, before every push to master, what the CI workflow runs (round-10 review J1: the
# round-13 merge reached master without an e2e run and master CI went red). Every step runs even when an earlier one
# fails, and the script writes a summary with the commit, the versions and one result line per step.
#
#   scripts/prepush.sh                          # everything (35 min 40 s on an Apple-silicon laptop at 278f7cc)
#   PYTHON=.venv/bin/python scripts/prepush.sh  # the interpreter with requirements-dev.txt installed
#   SKIP=e2e,extras scripts/prepush.sh          # skip steps by name (they are listed as skipped in the summary)
#   SUMMARY=docs/results/ci/prepush-$(git rev-parse --short HEAD).md scripts/prepush.sh
#   CI_BUDGET_FRACTION=0.5 scripts/prepush.sh  # CI time budget share (default 0.6; "off" disables the failure)
#
# CI time budgets (round-12 review L16: run 38095861905 was cancelled at its job's 45-min timeout, which this script
# could not see): after each step, scripts/ci_budget.py adds the step's local seconds x a hosted-runner slowdown
# factor to the estimate of the CI job that runs it (job -> timeout-minutes read from .github/workflows/ci.yml) and
# fails the step when the job's estimate exceeds CI_BUDGET_FRACTION of its timeout-minutes. Factors: CI_SLOWDOWN
# (default 3), CI_SLOWDOWN_STEPS ("pytest=5,..."); see scripts/ci_budget.py for the calibration. Wall-clock times
# inflate on a busy host: each step records the load average, and a budget failure under a load above
# CI_BUDGET_BUSY_LOAD (default: the performance-core count) says so; re-measure such a job on an idle host.
#
# Steps (names for SKIP): ruff, actionlint, pytest, pytest-shipped, fe-install, fe-typecheck, fe-lint, fe-test, fe-build, dist-diff,
# e2e, gate-dev-holdout, gate-ext, gate-shipped, extras, claim-holdout, verify-slices, report-check.
# Logs: outputs/prepush/<commit>/<step>.log. Exit status: 0 only when every step that ran passed.
# Needs: Python 3.13 with requirements-dev.txt, pnpm (frontend/package.json pins the version), Google Chrome for the
# Playwright e2e (channel "chrome"); actionlint is optional (skipped when it is not installed).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python; fi
fi
PYTHON="$(command -v "$PYTHON" || echo "$PYTHON")"
SKIP=",${SKIP:-},"
COMMIT="$(git rev-parse --short=7 HEAD)"
LOG_DIR="$ROOT/outputs/prepush/$COMMIT"
SUMMARY="${SUMMARY:-$LOG_DIR/summary.md}"
mkdir -p "$LOG_DIR" "$(dirname "$SUMMARY")"
BUDGET_STATE="$LOG_DIR/ci-budget.jsonl"
rm -f "$BUDGET_STATE"
CLEAN_AT_START="$(test -z "$(git status --porcelain --untracked-files=no)" && echo yes || echo no)"

# The e2e server must be this checkout's code: Playwright reuses a server already listening on its port outside CI,
# so pick a free port instead of the default 8905 (another worktree may be serving there).
E2E_PORT="${FINSIGHT_E2E_PORT:-$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"

ROWS=()
FAILED=0

# run <name> <pattern for the result line> <what it runs, for the summary> <command...>: runs the command, logs it,
# records one summary row
run() {
  local name="$1" pattern="$2" shown="$3"
  shift 3
  if [[ "$SKIP" == *",$name,"* ]]; then
    ROWS+=("| $name | \`$shown\` | skipped (SKIP) | – | – | – |")
    printf '[prepush] %-16s skipped\n' "$name"
    return
  fi
  local log="$LOG_DIR/$name.log" started status seconds line budget budget_status load_start
  printf '[prepush] %-16s ...' "$name"
  load_start="$("$PYTHON" -c 'import os; print(os.getloadavg()[0])' 2>/dev/null || echo 0)"
  started=$(date +%s)
  "$@" >"$log" 2>&1
  status=$?
  seconds=$(($(date +%s) - started))
  # the result line: the last line matching the step's pattern (ANSI colours stripped), else the last line
  line="$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -E "$pattern" | tail -1)"
  [ -n "$line" ] || line="$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -v '^\s*$' | tail -1)"
  line="$(printf '%s' "$line" | tr '|' '/' | sed 's/^[[:space:]]*//' | cut -c1-160)"
  # the CI job's estimated time so far against its timeout-minutes (scripts/ci_budget.py)
  budget="$("$PYTHON" "$ROOT/scripts/ci_budget.py" check --step "$name" --seconds "$seconds" \
    --state "$BUDGET_STATE" --load-start "$load_start" 2>&1)"
  budget_status=$?
  budget="$(printf '%s' "$budget" | tail -1 | tr '|' '/' | cut -c1-320)"
  if [ "$status" = 0 ] && [ "$budget_status" = 0 ]; then
    ROWS+=("| $name | \`$shown\` | pass | $seconds | $budget | $line |")
    printf ' pass (%ss; %s)\n' "$seconds" "${budget#budget: }"
  elif [ "$status" = 0 ]; then
    FAILED=1
    ROWS+=("| $name | \`$shown\` | **FAIL** (CI time budget) | $seconds | **$budget** | $line |")
    printf ' FAIL (CI time budget, %ss): %s\n' "$seconds" "$budget"
  else
    FAILED=1
    ROWS+=("| $name | \`$shown\` | **FAIL** (exit $status) | $seconds | $budget | $line |")
    printf ' FAIL (exit %s, %ss): see %s\n' "$status" "$seconds" "${log#"$ROOT"/}"
  fi
}

ruff_all() { "$PYTHON" -m ruff check . && "$PYTHON" -m ruff format --check .; }
actionlint_ci() {
  if ! command -v actionlint >/dev/null; then echo "actionlint not installed: not checked"; return 0; fi
  actionlint .github/workflows/*.yml && echo "actionlint: no findings"
}
fe() { (cd frontend && pnpm "$@"); }
dist_diff() {
  git status --porcelain query_intelligence/web/dist
  test -z "$(git status --porcelain query_intelligence/web/dist)" && echo "query_intelligence/web/dist matches the source"
}
e2e() { (cd frontend && FINSIGHT_PYTHON="$PYTHON" FINSIGHT_E2E_PORT="$E2E_PORT" pnpm run e2e); }
gate() {
  "$PYTHON" -m evaluation.agent_eval.gate "$@"
  local status=$?
  # one line per gated set for the summary: task success and task count of this run
  "$PYTHON" - "$@" <<'PY'
import json, sys
from pathlib import Path
names = [sys.argv[i + 1] for i, arg in enumerate(sys.argv[1:-1], start=1) if arg == "--set"]
parts = []
for name in names:
    path = Path("outputs/agent_eval") / f"gate-{name}.json"
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"] if path.exists() else {}
    parts.append(f"{name} task_success {summary.get('task_success')} ({summary.get('tasks')} tasks)")
print("result: " + "; ".join(parts))
PY
  return "$status"
}
extras() {
  "$PYTHON" -m evaluation.agent_eval.verifier_stress >/dev/null &&
    "$PYTHON" -m evaluation.agent_eval.redteam >/dev/null &&
    "$PYTHON" -m evaluation.agent_eval.gate --extras-only
}
claim_holdout() {
  "$PYTHON" -m evaluation.claim_bench.run --set holdout --out outputs/agent_eval/claim_bench-holdout.json \
    --fail-under-verdict 0.978 --fail-under-checks 0.98
}
verify_slices() {
  # (round 18, review-14 N16) the same list CI's heldout-verify job runs (evaluation/agent_eval/verify_slices.py)
  "$PYTHON" -m evaluation.agent_eval.verify_slices
}

# CI's environment: live sources off, traces off
export QI_USE_LIVE_MARKET=false QI_USE_LIVE_MACRO=false QI_USE_LIVE_NEWS=false QI_USE_LIVE_ANNOUNCEMENT=false
export QI_AGENT_TRACE_DIR=off

echo "[prepush] commit $COMMIT, working tree clean: $CLEAN_AT_START, python $PYTHON, logs ${LOG_DIR#"$ROOT"/}"
run ruff "All checks passed|already formatted|would reformat" "ruff check . && ruff format --check ." ruff_all
run actionlint "actionlint" "actionlint .github/workflows/*.yml" actionlint_ci
run pytest "passed|failed|error" "python -m pytest -q tests" "$PYTHON" -m pytest -q tests
# (round-13 review M17) the suite again on the shipped default data, minus tests/seed_data_tests.txt
run pytest-shipped "passed|failed|error" "QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests" \
  env QI_TEST_SHIPPED_DEFAULT=1 "$PYTHON" -m pytest -q tests
run fe-install "Done in|Already up to date|Lockfile" "pnpm install --frozen-lockfile" fe install --frozen-lockfile
run fe-typecheck "error|tsc" "pnpm typecheck" fe typecheck
run fe-lint "problem|eslint" "pnpm lint" fe lint
run fe-test "Tests +[0-9]" "pnpm test" fe test
run fe-build "built in|error" "pnpm build" fe build
run dist-diff "matches the source|dist" "git status --porcelain query_intelligence/web/dist (empty)" dist_diff
run e2e "passed|failed|flaky" "pnpm run e2e (port $E2E_PORT)" e2e
run gate-dev-holdout '^result:' "python -m evaluation.agent_eval.gate --set dev --set holdout" gate --set dev --set holdout
run gate-ext '^result:' "python -m evaluation.agent_eval.gate --set ext" gate --set ext
run gate-shipped '^result:' "python -m evaluation.agent_eval.gate --set shipped" gate --set shipped
run extras '"gate"' "verifier_stress; redteam; gate --extras-only" extras
run claim-holdout "verdict|accuracy" "claim_bench.run --set holdout (floors 0.978 / 0.98)" claim_holdout
run verify-slices "verify scripts passed|FAIL|Error" "python -m evaluation.agent_eval.verify_slices (CI's list: 12 scripts)" verify_slices
run report-check "up to date|out of date|missing" "python -m evaluation.agent_eval.report --check" \
  "$PYTHON" -m evaluation.agent_eval.report --check

{
  echo "# Pre-push check at \`$COMMIT\`"
  echo
  echo "Run by \`scripts/prepush.sh\` ($(date -u +%Y-%m-%dT%H:%M:%SZ))."
  echo
  echo "| Item | Value |"
  echo "|---|---|"
  echo "| Commit | \`$(git rev-parse HEAD)\` — $(git log -1 --format=%s | tr '|' '/') |"
  echo "| Working tree clean at start (tracked files) | $CLEAN_AT_START |"
  echo "| Working tree clean at end (tracked files) | $(test -z "$(git status --porcelain --untracked-files=no)" && echo yes || echo no) |"
  echo "| Host | $(uname -sm) |"
  echo "| Python | $("$PYTHON" --version 2>&1) |"
  echo "| Node / pnpm | $(node --version 2>/dev/null || echo n/a) / $(cd frontend && pnpm --version 2>/dev/null || echo n/a) |"
  echo "| Skipped by request | $(echo "$SKIP" | sed 's/^,//; s/,$//' | sed 's/^$/none/') |"
  echo "| Result | $([ "$FAILED" = 0 ] && echo "**all steps passed**" || echo "**at least one step failed**") |"
  echo
  echo "| Step | Command | Result | Seconds | CI job budget (estimate so far) | Result line from the log |"
  echo "|---|---|---|---|---|---|"
  printf '%s\n' "${ROWS[@]}"
  echo
  "$PYTHON" "$ROOT/scripts/ci_budget.py" summary --state "$BUDGET_STATE"
} >"$SUMMARY"
echo "[prepush] summary: ${SUMMARY#"$ROOT"/}"
exit "$FAILED"
