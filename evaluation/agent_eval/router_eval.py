"""Router evaluation: route accuracy and confusion matrix on a labelled query set.

``tasks/router_labels_v1.jsonl`` holds queries labelled with the route ``mode=auto`` should take:
``refuse`` (not a financial question, or only injected instructions), ``clarify`` (no target or a
dangling reference), ``workflow`` (one fact about one target) and ``agent`` (comparison, "why",
judgment or timing, macro-to-market links, multi-hop). The labels follow that written policy and were
written by the project author, so this is a regression and policy-consistency check, not an
independent benchmark.

The router runs on the real offline NLU (``guard_in``), with an LLM configured so that complex questions
are not downgraded to the workflow.

    python -m evaluation.agent_eval.router_eval
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM
from query_intelligence.agent.tools import build_registry_for_service

from .runner import (
    DEFAULT_OUTPUT_DIR,
    EVAL_DIR,
    EVAL_TODAY,
    _command,
    _display_path,
    _git_commit,
    build_offline_service,
)

ROUTES = ("refuse", "clarify", "workflow", "agent")
DEFAULT_LABELS = EVAL_DIR / "tasks" / "router_labels_v1.jsonl"


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    service = build_offline_service()
    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]), today=lambda: EVAL_TODAY)
    confusion: dict[str, Counter[str]] = {route: Counter() for route in ROUTES}
    errors = []
    try:
        for row in rows:
            state = runtime.initial_state(row["query"], mode="auto")
            decision = runtime.guard_in(state)
            predicted = decision["route"]
            confusion[row["expected_route"]][predicted] += 1
            if predicted != row["expected_route"]:
                errors.append(
                    {
                        "id": row["id"],
                        "query": row["query"],
                        "expected": row["expected_route"],
                        "predicted": predicted,
                        "reasons": decision.get("route_reasons") or [],
                    }
                )
    finally:
        runtime.close()
    total = len(rows)
    correct = sum(confusion[route][route] for route in ROUTES)
    per_route = {}
    for route in ROUTES:
        support = sum(confusion[route].values())
        predicted = sum(confusion[other][route] for other in ROUTES)
        per_route[route] = {
            "support": support,
            "precision": round(confusion[route][route] / predicted, 4) if predicted else None,
            "recall": round(confusion[route][route] / support, 4) if support else None,
        }
    return {
        "queries": total,
        "accuracy": round(correct / total, 4) if total else None,
        "per_route": per_route,
        "confusion": {route: {other: confusion[route][other] for other in ROUTES} for route in ROUTES},
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    _git_commit()  # record the commit at start, not when the run finishes
    parser = argparse.ArgumentParser(description="Route accuracy on a labelled query set.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR / "router_eval.json"))
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {
        "config": {
            "labels": _display_path(args.labels),
            "model": None,  # routing only, no LLM
            "commit": _git_commit(),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": _command("evaluation.agent_eval.router_eval", argv),
        },
        **evaluate(rows),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"accuracy {report['accuracy']} over {report['queries']} queries")
    print("expected \\ predicted " + " ".join(f"{route:>9s}" for route in ROUTES))
    for route in ROUTES:
        print(f"{route:>20s} " + " ".join(f"{report['confusion'][route][other]:9d}" for other in ROUTES))
    for error in report["errors"]:
        print(
            f"  {error['id']} {error['expected']:>8s} -> {error['predicted']:<8s} {error['query']}  {error['reasons']}"
        )
    return report


if __name__ == "__main__":
    main()
