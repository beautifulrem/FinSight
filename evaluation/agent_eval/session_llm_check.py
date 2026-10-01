"""Small online checks of multi-turn sessions on the LLM agent path (round 12, H7 / H8).

``--set h7``: English sessions whose follow-ups are one or two words ("And Moutai?", "Moutai?", "PE?") and a Chinese
session with an acronym-only follow-up. Every turn records the answer language the graph chose, whether the final
answer is in that language, whether the model wrote it in that language itself or the compliance node replaced a
wrong-language draft with the template (``language_mismatch_fallback_to_template``).

``--set h8``: the round-8 reviewer's sessions where the model computed a ratio / relative difference that the
frame did not recognise (0.52 / 1.93, 17.7%). Every LLM call's raw output is recorded (``drafts``), so the drafts
can be replayed offline in tests; the final answer records whether the derived number survived verification.

Every turn runs in ``mode=agent`` with the configured LLM over the offline tools, sequentially; the run stops at the
call budget or on the first HTTP 429.

    python -m evaluation.agent_eval.session_llm_check --llm deepseek --set h7 --max-calls 20 \
        --out outputs/agent_eval/session-llm-check-h7.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from query_intelligence.agent.compliance import language_violation
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.prompts import prompt_refs
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.state import AgentConfig
from query_intelligence.agent.tools import build_registry_for_service

from .runner import EVAL_TODAY, _git_commit, _make_llm, add_llm_arguments, build_offline_service, command_fields

# (session id, turns, expected answer language per turn or a number the last answer should state)
SETS: dict[str, list[dict[str, Any]]] = {
    "h7": [
        {"id": "en-and-moutai", "turns": ["Wuliangye's P/E ratio?", "And Moutai?"], "languages": ["en", "en"]},
        {"id": "en-one-word", "turns": ["What's Ping An's ROE?", "Moutai?"], "languages": ["en", "en"]},
        {"id": "en-close", "turns": ["Moutai's latest close?", "And Wuliangye?"], "languages": ["en", "en"]},
        {"id": "en-acronym", "turns": ["What is Wuliangye's revenue?", "PB?"], "languages": ["en", "en"]},
        {"id": "zh-acronym", "turns": ["五粮液的ROE是多少", "PE?"], "languages": ["zh", "zh"]},
    ],
    "h8": [
        {"id": "l3-ratio", "turns": ["中国平安ROE多少", "五粮液呢", "二者之比是多少"], "derived": [0.52, 1.93]},
        {
            "id": "l2b-relative",
            "turns": ["Wuliangye's P/E ratio?", "And Moutai?", "By what percentage is Moutai's above Wuliangye's?"],
            "derived": [17.7],
        },
    ],
}


class RecordingLLM:
    """Passes every call to ``llm`` and keeps its raw output (content and tool calls) for offline replay."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self.model = getattr(llm, "model", None)
        self.calls: list[dict[str, Any]] = []

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        turn = self._llm.chat(*args, **kwargs)
        self.calls.append(
            {
                "content": turn.content,
                "tool_calls": [{"name": call.name, "arguments": call.arguments} for call in turn.tool_calls],
                "finish_reason": turn.finish_reason,
            }
        )
        return turn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def _language_of(text: str) -> str:
    return "en" if not language_violation(text, "", language="en") else "zh"


def _states(text: str, value: float) -> bool:
    from query_intelligence.agent.graph import _states_number

    return _states_number(text, value)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    commit = _git_commit()
    parser = argparse.ArgumentParser(description=__doc__)
    add_llm_arguments(parser)
    parser.add_argument("--set", choices=sorted(SETS), required=True)
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    llm = _make_llm(args.llm, args.model)
    if llm is None:
        raise SystemExit("an LLM is required (--llm deepseek)")
    recording = RecordingLLM(llm)
    service = build_offline_service()
    runtime = AgentRuntime(
        service, build_registry_for_service(service), recording, config=AgentConfig(), today=lambda: EVAL_TODAY
    )
    agent = AgentService(runtime, trace_sinks=[])
    calls, sessions, stopped = 0, [], None
    try:
        for session in SETS[args.set]:
            if calls >= args.max_calls - 2 * len(session["turns"]):
                stopped = f"call budget ({calls} of {args.max_calls})"
                break
            record: dict[str, Any] = {"id": session["id"], "turns": []}
            for index, query in enumerate(session["turns"]):
                first_call = len(recording.calls)
                started = time.perf_counter()
                result = agent.chat(query, session_id=f"r12-{args.set}-{session['id']}", mode="agent")
                used = int((result.get("llm") or {}).get("calls") or 0)
                calls += used
                answer = str(result.get("answer") or "")
                notes = result.get("compliance_notes") or []
                turn: dict[str, Any] = {
                    "query": query,
                    "route": result.get("route"),
                    "route_reasons": result.get("route_reasons"),
                    "answer_source": result.get("answer_source"),
                    "llm_calls": used,
                    "latency_s": round(time.perf_counter() - started, 2),
                    "verification": result.get("verification"),
                    "degraded": result.get("degraded") or [],
                    "compliance_notes": notes,
                    "answer": answer,
                    "limitations": result.get("limitations"),
                    "drafts": recording.calls[first_call:],
                }
                if "languages" in session:
                    expected = session["languages"][index]
                    drafts = [_draft_answer(d["content"]) for d in turn["drafts"] if d["content"]]
                    turn["expected_language"] = expected
                    turn["answer_language"] = _language_of(answer) if expected == "en" else _zh_or_en(answer)
                    turn["language_ok"] = turn["answer_language"] == expected
                    turn["language_fallback"] = "language_mismatch_fallback_to_template" in notes
                    turn["model_wrote_expected_language"] = bool(drafts) and all(
                        (_language_of(text) if expected == "en" else _zh_or_en(text)) == expected for text in drafts
                    )
                record["turns"].append(turn)
                if any("429" in str(item) for item in turn["degraded"]):
                    stopped = "HTTP 429"
                    break
            if "derived" in session and record["turns"]:
                last = record["turns"][-1]
                record["derived_expected"] = session["derived"]
                record["derived_stated"] = [value for value in session["derived"] if _states(last["answer"], value)]
                record["derived_in_drafts"] = [
                    value
                    for value in session["derived"]
                    if any(_states(str(d["content"] or ""), value) for d in last["drafts"])
                ]
            sessions.append(record)
            if stopped:
                break
    finally:
        runtime.close()
    turns = [turn for session in sessions for turn in session["turns"]]
    summary: dict[str, Any] = {"sessions": len(sessions), "turns": len(turns), "llm_calls": calls, "stopped": stopped}
    if args.set == "h7":
        summary.update(
            {
                "answer_language_ok": sum(bool(turn.get("language_ok")) for turn in turns),
                "language_fallback": sum(bool(turn.get("language_fallback")) for turn in turns),
                "model_wrote_expected_language": sum(bool(turn.get("model_wrote_expected_language")) for turn in turns),
            }
        )
    else:
        summary.update(
            {
                "derived_in_model_drafts": sum(bool(s.get("derived_in_drafts")) for s in sessions),
                "derived_kept_in_answer": sum(bool(s.get("derived_stated")) for s in sessions),
            }
        )
    report = {
        "config": {
            "commit": commit,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **command_fields("evaluation.agent_eval.session_llm_check", argv),
            "model": getattr(llm, "model", None),
            "prompts": prompt_refs(),
            "tools": "offline runtime assets (no replay snapshot)",
        },
        "summary": summary,
        "sessions": sessions,
    }
    path = Path(args.out or f"outputs/agent_eval/session-llm-check-{args.set}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return report


def _draft_answer(content: str) -> str:
    """The ``answer`` of a JSON draft, else the raw text."""
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", content or "", re.S)
        if not match:
            return content or ""
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return content
    return str(data.get("answer") or "") if isinstance(data, dict) else str(content)


def _zh_or_en(text: str) -> str:
    """``zh`` unless the text is long Latin-only prose (the Chinese-side rule of ``language_violation``)."""
    return "en" if language_violation(text, "", language="zh") else "zh"


if __name__ == "__main__":
    main()
