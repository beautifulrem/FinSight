"""Decision regression over every evaluation query string (round 17).

A rule change (an alias guard, a new advice marker) must not change decisions it was not meant to change. This script
runs every conversation in ``evaluation/**/*.jsonl`` and ``tests/data/**/*.jsonl`` (the user turns of each task, in
order, one session per conversation) through the offline agent of a source tree (live sources off, the shipped
default snapshot layers on), and compares two runs turn by turn.

A turn's decision is (route, refusal limitation, listed symbols the NLU resolved, the advice flag, the hedge notes the
compliance guard added); answer text changes are counted separately.

    # each tree's run (``git archive <commit> | tar -x -C /tmp/base`` gives an older tree)
    python scripts/decision_regression.py run --tree /tmp/base --out /tmp/before.jsonl
    python scripts/decision_regression.py run --tree . --out /tmp/after.jsonl
    python scripts/decision_regression.py compare /tmp/before.jsonl /tmp/after.jsonl --out summary.json \
        --meta before=<commit> --meta after=<commit>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

HEDGES = (
    "conditional_prefix",
    "position_size_hedge",
    "fair_value_hedge",
    "price_forecast_hedge",
    "earnings_forecast_hedge",
    "market_value_forecast_hedge",
    "currency_conversion_note",
)
_REFUSALS = ("out_of_coverage", "out_of_scope_query", "prompt_injection_request")
_AGENT: Any = None


def collect(root: Path) -> list[dict[str, Any]]:
    """Every distinct conversation (its user turns) of the evaluation and test data files under ``root``."""
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for base in ("evaluation", "tests/data"):
        for path in sorted((root / base).rglob("*.jsonl")):
            for line in path.open(encoding="utf-8"):
                try:
                    row = json.loads(line) if line.strip() else None
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                turns = [t.get("query") for t in row.get("turns") or [] if isinstance(t, dict)]
                turns = [t for t in turns if isinstance(t, str)]
                if not turns:
                    turns = next(
                        ([row[key]] for key in ("query", "question", "text") if isinstance(row.get(key), str)), []
                    )
                if turns and tuple(turns) not in seen:
                    seen.add(tuple(turns))
                    found.append({"src": str(path.relative_to(root)), "turns": turns})
    return found


def _init(tree: str) -> None:
    global _AGENT
    os.chdir(tree)
    sys.path.insert(0, tree)
    import logging

    logging.disable(logging.WARNING)
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
        offline_snapshot_fix=True,
    )
    _AGENT = AgentService(AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([])), trace_sinks=[])


def _run(item: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
    index, conversation = item
    rows = []
    for turn, query in enumerate(conversation["turns"]):
        try:
            result = _AGENT.chat(query, session_id=f"regression-{index}")
        except Exception as exc:
            rows.append({"i": index, "t": turn, "q": query, "error": repr(exc)[:200]})
            continue
        nlu = result.get("nlu_summary") or {}
        answer = result.get("answer") or json.dumps(result.get("clarification"), ensure_ascii=False) or ""
        rows.append(
            {
                "i": index,
                "t": turn,
                "q": query,
                "src": conversation["src"],
                "route": result.get("route"),
                "reasons": result.get("route_reasons") or [],
                "symbols": sorted({e.get("symbol") for e in nlu.get("entities") or [] if e.get("symbol")}),
                "flags": sorted(nlu.get("risk_flags") or []),
                "limitations": result.get("limitations") or [],
                "notes": result.get("compliance_notes") or [],
                "answer_head": answer[:160],
                "answer_sha": hashlib.sha1(answer.encode()).hexdigest()[:12],
            }
        )
    return rows


def decision(row: dict[str, Any]) -> tuple:
    if "error" in row:
        return ("ERROR",)
    refusal = tuple(item for item in row["limitations"] if item in _REFUSALS)
    advice = "investment_advice_like" in row["flags"]
    hedges = tuple(sorted(note for note in row["notes"] if note in HEDGES))
    return (row["route"], refusal, tuple(row["symbols"]), advice, hedges)


def compare(before: Path, after: Path) -> dict[str, Any]:
    def load(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
        return {(row["i"], row["t"]): row for row in map(json.loads, path.open(encoding="utf-8"))}

    old, new = load(before), load(after)
    changed = [key for key in sorted(old) if key in new and decision(old[key]) != decision(new[key])]
    texts = [key for key in sorted(old) if key in new and old[key].get("answer_sha") != new[key].get("answer_sha")]
    return {
        "turns": len(old),
        "conversations": len({key[0] for key in old}),
        "missing_after": len([key for key in old if key not in new]),
        "decision_changes": len(changed),
        "answer_text_changes": len(texts),
        "changes": [
            {
                "src": old[key].get("src"),
                "query": old[key]["q"],
                "before": list(decision(old[key])),
                "after": list(decision(new[key])),
                "reasons_after": [reason for reason in new[key].get("reasons", []) if ":" in reason][:6],
            }
            for key in changed
        ],
        "text_changes": [
            {"src": old[key].get("src"), "query": old[key]["q"], "after": new[key].get("answer_head")} for key in texts
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--tree", default=".")
    run.add_argument("--out", required=True)
    run.add_argument("--workers", type=int, default=4)
    diff = commands.add_parser("compare")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--out", type=Path)
    diff.add_argument("--meta", action="append", default=[], help="key=value recorded in the summary (commits, date)")
    args = parser.parse_args(argv)
    if args.command == "run":
        tree = str(Path(args.tree).resolve())
        conversations = collect(Path(tree))
        with Pool(args.workers, initializer=_init, initargs=(tree,)) as pool, open(args.out, "w") as out:
            for rows in pool.imap(_run, list(enumerate(conversations)), chunksize=8):
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{len(conversations)} conversations -> {args.out}")
        return
    summary = {**dict(item.split("=", 1) for item in args.meta), **compare(args.before, args.after)}
    print(
        f"turns {summary['turns']} ({summary['conversations']} conversations): decision changes "
        f"{summary['decision_changes']}, answer text changes {summary['answer_text_changes']}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
