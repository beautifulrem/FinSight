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
#
# Steps (names for SKIP): ruff, actionlint, pytest, fe-install, fe-typecheck, fe-lint, fe-test, fe-build, dist-diff,
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
    ROWS+=("| $name | \`$shown\` | skipped (SKIP) | – | – |")
    printf '[prepush] %-16s skipped\n' "$name"
    return
  fi
  local log="$LOG_DIR/$name.log" started status seconds line
  printf '[prepush] %-16s ...' "$name"
  started=$(date +%s)
  "$@" >"$log" 2>&1
  status=$?
  seconds=$(($(date +%s) - started))
  # the result line: the last line matching the step's pattern (ANSI colours stripped), else the last line
  line="$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -E "$pattern" | tail -1)"
  [ -n "$line" ] || line="$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -v '^\s*$' | tail -1)"
  line="$(printf '%s' "$line" | tr '|' '/' | sed 's/^[[:space:]]*//' | cut -c1-160)"
  if [ "$status" = 0 ]; then
    ROWS+=("| $name | \`$shown\` | pass | $seconds | $line |")
    printf ' pass (%ss)\n' "$seconds"
  else
    FAILED=1
    ROWS+=("| $name | \`$shown\` | **FAIL** (exit $status) | $seconds | $line |")
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
  local script
  for script in evaluation/agent_eval/independent/verify_multiturn.py evaluation/agent_eval/independent/verify_test_v3.py \
    evaluation/heldout_r4/verify_heldout_r4.py evaluation/heldout_r5/verify_heldout_r5.py \
    evaluation/heldout_r6/verify_heldout_r6.py evaluation/heldout_r7/verify_heldout_r7.py \
    evaluation/heldout_r8/verify_heldout_r8.py; do
    PYTHONPATH=. "$PYTHON" "$script" || return 1
  done
  QI_EVAL_SNAPSHOT_EXT=1 PYTHONPATH=. "$PYTHON" evaluation/heldout_r9/verify_heldout_r9.py || return 1
  echo "8 verify scripts passed"
}

# CI's environment: live sources off, traces off
export QI_USE_LIVE_MARKET=false QI_USE_LIVE_MACRO=false QI_USE_LIVE_NEWS=false QI_USE_LIVE_ANNOUNCEMENT=false
export QI_AGENT_TRACE_DIR=off

echo "[prepush] commit $COMMIT, working tree clean: $CLEAN_AT_START, python $PYTHON, logs ${LOG_DIR#"$ROOT"/}"
run ruff "All checks passed|already formatted|would reformat" "ruff check . && ruff format --check ." ruff_all
run actionlint "actionlint" "actionlint .github/workflows/*.yml" actionlint_ci
run pytest "passed|failed|error" "python -m pytest -q tests" "$PYTHON" -m pytest -q tests
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
run verify-slices "verify scripts passed|FAIL|Error" "8 held-out verify scripts (r9 with QI_EVAL_SNAPSHOT_EXT=1)" verify_slices
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
  echo "| Step | Command | Result | Seconds | Result line from the log |"
  echo "|---|---|---|---|---|"
  printf '%s\n' "${ROWS[@]}"
} >"$SUMMARY"
echo "[prepush] summary: ${SUMMARY#"$ROOT"/}"
exit "$FAILED"
