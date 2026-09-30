"""Fill the FinSight rows of the head-to-head answers with FinSight's own live answers at date T.

Runs the 30 frozen questions in-process with the live market, news, announcement and macro sources on,
three fresh sessions per question (a follow-up asks its ``context`` turn first in the same session), and
writes the FinSight rows of ``head_to_head/answers.csv`` (created from the template when missing; the other
products' rows are kept). Raw responses go to ``head_to_head/raw/finsight-<T>.jsonl``.

    python -m evaluation.human.fetch_finsight_answers --date 2026-10-09             # deterministic path
    source llmenv.sh && python -m evaluation.human.fetch_finsight_answers --date 2026-10-09 --llm deepseek

Run it in the same window as the other products: after the close on T and before the next open.
"""

from __future__ import annotations

import argparse
import uuid
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .common import answer_text, cited_evidence, describe_evidence, display, read_csv, write_csv, write_jsonl
from .score_head_to_head import ANSWERS_PATH, H2H_DIR, TEMPLATE_PATH, in_window, load_questions

BEIJING = timezone(timedelta(hours=8))
COLUMNS = ("question_id", "product", "run", "answer_text", "cited_sources", "screenshot_file", "asked_at")


def build_live_agent(llm: Any) -> Any:
    from query_intelligence.agent.service import AgentService
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=True, use_live_macro=True, use_live_news=True, use_live_announcement=True
    )
    return AgentService.from_service(service, llm=llm, trace_sinks=[])


def ask(agent: Any, question: dict[str, str], *, run: int, mode: str) -> dict[str, Any]:
    session = f"h2h-{question['question_id']}-{run}-{uuid.uuid4().hex[:8]}"
    if question.get("context"):
        agent.chat(question["context"], session_id=session, mode=mode)
    asked_at = datetime.now(BEIJING).strftime("%Y-%m-%d %H:%M")
    response = agent.chat(question["question"], session_id=session, mode=mode)
    return {"asked_at": asked_at, "response": response}


def finsight_row(question: dict[str, str], run: int, result: dict[str, Any]) -> dict[str, Any]:
    response = result["response"]
    return {
        "question_id": question["question_id"],
        "product": "FinSight",
        "run": str(run),
        "answer_text": answer_text(response),
        "cited_sources": "\n".join(describe_evidence(item) for item in cited_evidence(response)),
        "screenshot_file": "",
        "asked_at": result["asked_at"],
    }


def merge_rows(existing: list[dict[str, str]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace the matching FinSight rows (question_id, run) and keep every other row in place."""
    fresh = {(row["question_id"], str(row["run"])): row for row in new}
    merged = []
    for row in existing:
        key = (row.get("question_id", ""), str(row.get("run", "")))
        if row.get("product") == "FinSight" and key in fresh:
            merged.append(fresh.pop(key))
        else:
            merged.append(row)
    merged.extend(fresh.values())
    return merged


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(description="FinSight's live answers to the head-to-head questions.")
    parser.add_argument("--date", required=True, help="Trading date T (YYYY-MM-DD).")
    parser.add_argument("--mode", choices=["auto", "workflow", "agent"], default="auto")
    parser.add_argument("--llm", choices=["none", "deepseek"], default="none")
    parser.add_argument("--model", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Only the first N questions (smoke test).")
    parser.add_argument("--answers", default=str(ANSWERS_PATH))
    args = parser.parse_args(argv)
    day = date.fromisoformat(args.date)
    now = datetime.now(BEIJING).strftime("%Y-%m-%d %H:%M")
    if not in_window(now, day):
        print(f"warning: {now} (Beijing) is outside the window after the close on {day} and before the next open")

    from ..agent_eval.runner import _make_llm

    llm = _make_llm(args.llm, args.model)
    agent = build_live_agent(llm)
    questions = load_questions()[: args.limit or None]
    rows, raw = [], []
    for question in questions:
        for run in range(1, args.runs + 1):
            result = ask(agent, question, run=run, mode=args.mode)
            rows.append(finsight_row(question, run, result))
            response = result["response"]
            raw.append(
                {
                    "question_id": question["question_id"],
                    "run": run,
                    "asked_at": result["asked_at"],
                    "mode": args.mode,
                    "model": getattr(llm, "model", None),
                    "response": {
                        key: response.get(key)
                        for key in (
                            "answer",
                            "key_points",
                            "limitations",
                            "risk_disclaimer",
                            "route",
                            "answer_source",
                            "degraded",
                            "evidence_used",
                            "evidence_sources",
                            "verification",
                            "compliance_notes",
                            "status",
                            "clarification",
                        )
                    },
                }
            )
            print(f"{question['question_id']} run {run}: {response.get('route')} {answer_text(response)[:60]!r}")
    agent.close()
    answers_path = Path(args.answers)
    existing = read_csv(answers_path if answers_path.exists() else TEMPLATE_PATH)
    write_csv(answers_path, COLUMNS, merge_rows(existing, rows))
    raw_path = H2H_DIR / "raw" / f"finsight-{day.isoformat()}.jsonl"
    write_jsonl(raw_path, [{**item, "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")} for item in raw])
    print(f"wrote {len(rows)} FinSight rows to {display(answers_path)}; raw responses in {display(raw_path)}")
    return rows


if __name__ == "__main__":
    main()
