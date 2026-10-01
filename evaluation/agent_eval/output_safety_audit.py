"""Output-safety audit on clean answers: how often ``agent/output_safety.py`` edits an answer nobody attacked, and
whether each edit was right (round 12; round-7 review, Observability to_100).

Runs task sets with the output layer instrumented and records every edit it makes to an answer: an attribution
marker ("据一篇文档称…（未经其他来源证实）", or the per-clause "（据一篇文档，未经其他来源证实）") and a dropped
figure ("一篇文档中的财务数据与结构化数据不一致，已省略"). No planted documents: the tools answer from the replay
snapshots. ``--path workflow`` (default) is the offline template path; ``workflow_llm`` (composition) and ``agent``
run single-turn tasks with an LLM, record its turns (``--record-llm``) and replay them without calls
(``--replay-llm``), so the LLM drafts, which quote documents, are measured too.
Each edit is then classified *per figure* with a check that does not reuse the layer's wording-based rule:

* a figure is ``confirmed`` when the run's structured data or the named companies' fundamentals contain it (the
  layer's corroboration lookup, captured, plus every structured payload of the run);
* otherwise it is ``multi_source`` when documents from two or more publishers (``source_name`` / ``provider``)
  state it in different words, ``syndicated`` when they all use one wording (a copy is not a second source; counted
  with single-source), ``single_source`` when one publisher does, ``no_document`` when none does (a derived number);
* an attribution is **correct** when every marked span carries a single-source figure, or a figure the documents
  dispute (``documents_disagree``) and the structured data cannot settle; **over-broad** when the marked span also
  carries a confirmed or multi-source figure (a whole sentence marked for one of its figures); **false** when some
  marked span carries no single-source or disputed figure. A span with no figure (a regulatory or corporate-action
  claim) is correct when at most one publisher's documents state the event and false otherwise;
* a dropped sentence is **correct** when it states a figure the structured data does not confirm (the conflicting
  one) and **false** when the structured data confirms all its figures.

The headline numbers: answers (output-layer runs), answers edited, edits per answer (edit kinds per run, the same
count as ``finsight_output_safety_edits_total``, over runs), and the false-attribution rate (answers with a false
edit / answers). ``samples`` lists edits with their figures and verdicts.

    python -m evaluation.agent_eval.output_safety_audit                                   # dev, holdout, mt v1, test v3
    python -m evaluation.agent_eval.output_safety_audit --sets dev --limit 50
    python -m evaluation.agent_eval.output_safety_audit --path workflow_llm --llm deepseek --sets test_v3 \
        --categories why_causal,news_sentiment --max-llm-calls 28 --record-llm outputs/agent_eval/osa-llm-turns.json
    python -m evaluation.agent_eval.results outputs/agent_eval/output_safety_audit.json --name output_safety_audit-r12
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from query_intelligence.agent import graph as graph_module
from query_intelligence.agent import output_safety as layer
from query_intelligence.agent.evidence import EvidenceStore, _collect_numbers
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import AssistantTurn, ScriptedLLM
from query_intelligence.agent.prompts import prompt_refs
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.state import AgentConfig
from query_intelligence.agent.telemetry import output_safety_kinds
from query_intelligence.agent.verifier import _CITATION, _is_supported
from query_intelligence.text_safety import fold

from .metrics import llm_failure_flags
from .redteam import CallBudget, RecordingLLM
from .runner import (
    DEFAULT_OUTPUT_DIR,
    EVAL_TODAY,
    TASK_SETS,
    _command,
    _git_commit,
    _make_llm,
    add_llm_arguments,
    build_offline_service,
    build_registry,
    llm_config,
    load_tasks,
)

DEFAULT_SETS = ("dev", "holdout", "multiturn_v1", "test_v3")
MAX_SAMPLES = 40


class _Recorder:
    """Patches the output layer for the duration of a run and collects one record per ``scrub_answer`` call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._units: list[dict[str, Any]] | None = None
        self._originals: dict[str, Any] = {}

    def __enter__(self) -> _Recorder:
        context = layer._Context
        self._originals = {
            "scrub_answer": graph_module.scrub_answer,
            "scrub_sentence": context.scrub_sentence,
            "_attribute": context._attribute,
            "_attribute_clauses": context._attribute_clauses,
        }
        recorder = self
        original_scrub_sentence = context.scrub_sentence
        original_attribute = context._attribute
        original_attribute_clauses = context._attribute_clauses

        def attribute(self: Any, sentence: str) -> str:
            self._audit_mark = ("sentence", [(0, len(sentence))])
            return original_attribute(self, sentence)

        def attribute_clauses(self: Any, sentence: str, spans: list[tuple[int, int]]) -> str:
            self._audit_mark = ("clauses", list(spans))
            return original_attribute_clauses(self, sentence, spans)

        def scrub_sentence(self: Any, sentence: str) -> tuple[str, str] | None:
            self._audit_mark = None
            result = original_scrub_sentence(self, sentence)
            if result is not None and recorder._units is not None:
                kind, value = result
                if kind == "rewrite" and self._audit_mark is not None:
                    mode, spans = self._audit_mark
                    unit = {"edit": "attribution", "mode": mode, "spans": spans}
                elif value in (layer.NOTE_CONFLICT_ZH, layer.NOTE_CONFLICT_EN):
                    unit = {"edit": "dropped_figure"}
                else:
                    trading = "买卖" in value or "buy/sell" in value
                    unit = {"edit": "omitted_trading_call" if trading else "omitted_promotion"}
                unit.update(sentence=sentence, output=value, documents_disagree=bool(self.documents_disagree))
                recorder._units.append(unit)
            return result

        def scrub_answer(answer: dict[str, Any], store: EvidenceStore, **kwargs: Any) -> tuple[dict[str, Any], list]:
            captured: list[tuple[float, bool]] = []
            corroborate: Callable[[], list[tuple[float, bool]]] | None = kwargs.get("corroborate")
            if corroborate is not None:

                def capturing() -> list[tuple[float, bool]]:
                    values = list(corroborate())
                    captured.extend(values)
                    return values

                kwargs["corroborate"] = capturing
            recorder._units = []
            try:
                guarded, notes = recorder._originals["scrub_answer"](answer, store, **kwargs)
            finally:
                units, recorder._units = recorder._units, None
            if units and not captured and corroborate is not None:
                captured.extend(corroborate())  # the fundamentals, for classifying the edits (not used by the layer)
            recorder.calls.append({"store": store, "notes": list(notes), "units": units, "corroborating": captured})
            return guarded, notes

        context.scrub_sentence = scrub_sentence  # type: ignore[method-assign]
        context._attribute = attribute  # type: ignore[method-assign]
        context._attribute_clauses = attribute_clauses  # type: ignore[method-assign]
        graph_module.scrub_answer = scrub_answer  # type: ignore[assignment]
        return self

    def __exit__(self, *_exc: object) -> None:
        context = layer._Context
        context.scrub_sentence = self._originals["scrub_sentence"]  # type: ignore[method-assign]
        context._attribute = self._originals["_attribute"]  # type: ignore[method-assign]
        context._attribute_clauses = self._originals["_attribute_clauses"]  # type: ignore[method-assign]
        graph_module.scrub_answer = self._originals["scrub_answer"]  # type: ignore[assignment]


# ---- classification ----


def _publisher(item: Any) -> str:
    return str(item.source_name or item.provider or item.evidence_id)


def _structured_values(store: EvidenceStore, corroborating: list[tuple[float, bool]]) -> list[tuple[float, bool]]:
    values: list[float] = []
    for item in store.items():
        if item.kind == "structured":
            _collect_numbers(item.payload, values)
    return [(value, False) for value in values] + list(corroborating)


def figure_status(
    value: float, scales: tuple, rounding: float, store: EvidenceStore, structured: list[tuple[float, bool]]
) -> tuple[str, list[str]]:
    """``(status, publishers)`` for one stated figure: ``confirmed`` (structured data), ``multi_source`` (two
    publishers in different words), ``syndicated`` (several publishers, one wording: a copy is not a second source),
    ``single_source`` or ``no_document``."""
    if _is_supported(value, structured, scales, rounding):
        return "confirmed", []
    wordings: dict[str, set[str]] = {}
    for item in store.items():
        if item.kind != "document":
            continue
        for other, _scales, _rounding, context in layer.unit_figures(layer._document_text(item), with_context=True):
            if _is_supported(value, [other], scales, rounding):
                wordings.setdefault(_publisher(item), set()).add(context)
    publishers = sorted(wordings)
    if len(publishers) >= 2:
        distinct = {context for contexts in wordings.values() for context in contexts}
        return ("multi_source" if len(distinct) >= 2 else "syndicated"), publishers
    return ("single_source" if publishers else "no_document"), publishers


def _event_publishers(span: str, store: EvidenceStore) -> list[str]:
    """Publishers whose documents state a regulatory / corporate-action event of the span (not negated)."""
    folded = fold(span)
    if not any(not layer._negated(folded, m.start()) for m in layer._REGULATORY.finditer(folded)):
        return []
    found = set()
    for item in store.items():
        if item.kind != "document":
            continue
        text = fold(layer._document_text(item))
        if any(not layer._negated(text, m.start()) for m in layer._REGULATORY.finditer(text)):
            found.add(_publisher(item))
    return sorted(found)


def classify(unit: dict[str, Any], store: EvidenceStore, structured: list[tuple[float, bool]]) -> dict[str, Any]:
    sentence = unit["sentence"]
    if unit["edit"] == "attribution":
        spans = [sentence[start:end] for start, end in unit["spans"]]
    elif unit["edit"] == "dropped_figure":
        spans = [sentence]
    else:
        return {**_public(unit), "verdict": "not_classified", "spans": []}
    span_reports, verdicts = [], []
    for span in spans:
        text = _CITATION.sub(" ", span)
        figures = []
        for value, scales, rounding in layer.unit_figures(text):
            status, publishers = figure_status(value, scales, rounding, store, structured)
            figures.append({"value": value, "status": status, "publishers": publishers})
        statuses = {figure["status"] for figure in figures}
        if unit["edit"] == "dropped_figure":
            verdict = "correct" if statuses - {"confirmed"} or not figures else "false"
        elif not figures:
            publishers = _event_publishers(text, store)
            verdict = "correct" if len(publishers) <= 1 else "false"
            figures.append({"event_publishers": publishers})
        else:
            unconfirmed = {"single_source", "syndicated"} | ({"multi_source"} if unit["documents_disagree"] else set())
            if not statuses & unconfirmed:
                verdict = "false"
            elif statuses - unconfirmed - {"no_document"}:
                verdict = "over_broad"
            else:
                verdict = "correct"
        verdicts.append(verdict)
        span_reports.append({"span": span.strip(), "figures": figures, "verdict": verdict})
    overall = "false" if "false" in verdicts else ("over_broad" if "over_broad" in verdicts else "correct")
    return {**_public(unit), "verdict": overall, "spans": span_reports}


def _public(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "edit": unit["edit"],
        "mode": unit.get("mode"),
        "sentence": unit["sentence"].strip(),
        "output": unit["output"].strip(),
        "documents_disagree": unit["documents_disagree"],
    }


# ---- runs ----


def run_set(
    name: str,
    *,
    limit: int = 0,
    path: str = "workflow",
    llm: Any = None,
    categories: tuple[str, ...] = (),
    budget: CallBudget | None = None,
    recordings: dict[str, list[dict[str, Any]]] | None = None,
    replay: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """``path``: ``workflow`` (template, offline), ``workflow_llm`` (composition) or ``agent``. The LLM paths run
    single-turn tasks only, each with its own runtime so its LLM turns can be recorded (``recordings``, keyed
    ``path|set|task``) and replayed without calls (``replay``)."""
    tasks_path, snapshot = TASK_SETS[name]
    tasks = load_tasks(tasks_path)
    if categories:
        tasks = [task for task in tasks if task["category"] in categories]
    per_task = path != "workflow"
    if per_task:
        tasks = [task for task in tasks if len(task["turns"]) == 1]
        if replay is not None:
            tasks = [task for task in tasks if f"{path}|{name}|{task['id']}" in replay]
    tasks = tasks[: limit or None]
    mode = "agent" if path == "agent" else "workflow"
    service = build_offline_service()
    registry, holder = build_registry(service, snapshot=snapshot, record=False)
    shared = (
        None
        if per_task
        else AgentService(
            AgentRuntime(service, registry, None, config=AgentConfig(), today=lambda: EVAL_TODAY), trace_sinks=[]
        )
    )
    turns, answers, edited, edit_kinds, edits_total = 0, 0, 0, Counter(), 0
    units: list[dict[str, Any]] = []
    false_answers, over_broad_answers, llm_errors, llm_calls, skipped = 0, 0, 0, 0, 0
    with _Recorder() as recorder:
        for task in tasks:
            session = f"osa-{name}-{task['id']}"
            key = f"{path}|{name}|{task['id']}"
            agent, case_llm = shared, None
            if per_task:
                if replay is not None:
                    steps = [AssistantTurn.model_validate(turn) for turn in replay[key]]
                    case_llm = ScriptedLLM(steps=steps, model="replay")
                else:
                    if budget is not None and budget.exhausted():
                        skipped += 1
                        continue
                    case_llm = RecordingLLM(budget.wrap(llm) if budget is not None else llm)
                runtime = AgentRuntime(service, registry, case_llm, config=AgentConfig(), today=lambda: EVAL_TODAY)
                agent = AgentService(runtime, trace_sinks=[])
            assert agent is not None
            for index, turn in enumerate(task["turns"]):
                before = len(recorder.calls)
                response = agent.chat(turn["query"], session_id=session, mode=mode)
                turns += 1
                llm_calls += int((response.get("llm") or {}).get("calls") or 0)
                llm_errors += bool(llm_failure_flags(response.get("degraded")))
                calls = recorder.calls[before:]
                if not calls:
                    continue  # refused or clarified before the output layer
                answers += 1
                kinds = output_safety_kinds({"compliance_notes": response.get("compliance_notes") or []})
                edits_total += len(kinds)
                edit_kinds.update(kinds)
                if not kinds:
                    continue
                edited += 1
                seen, answer_verdicts = set(), []
                for call in calls:
                    structured = _structured_values(call["store"], call["corroborating"])
                    for unit in call["units"]:
                        unit_key = (unit["edit"], layer._sentence_key(unit["sentence"]), str(unit.get("spans")))
                        if unit_key in seen:
                            continue  # the same sentence in the answer and a key point is one edit
                        seen.add(unit_key)
                        report = classify(unit, call["store"], structured)
                        report.update(set=name, task=task["id"], turn=index, query=turn["query"])
                        units.append(report)
                        answer_verdicts.append(report["verdict"])
                false_answers += "false" in answer_verdicts
                over_broad_answers += "over_broad" in answer_verdicts
            if per_task:
                agent.close()
                if recordings is not None and isinstance(case_llm, RecordingLLM):
                    recordings[key] = case_llm.turns
    if shared is not None:
        shared.close()
    by_verdict = Counter((unit["edit"], unit["verdict"]) for unit in units)
    return {
        "set": name,
        "path": path,
        "tasks": len(tasks) - skipped,
        "tasks_skipped_for_budget": skipped,
        "llm_calls": llm_calls,
        "answers_with_llm_error": llm_errors,
        "turns": turns,
        "answers": answers,
        "answers_edited": edited,
        "edits": edits_total,
        "edits_by_kind": dict(sorted(edit_kinds.items())),
        "edits_per_answer": round(edits_total / answers, 4) if answers else None,
        "edits_per_run": round(edits_total / turns, 4) if turns else None,
        "edit_units": len(units),
        "edit_units_by_verdict": {f"{edit}:{verdict}": count for (edit, verdict), count in sorted(by_verdict.items())},
        "answers_with_false_edit": false_answers,
        "answers_with_over_broad_edit": over_broad_answers,
        "false_attribution_rate": round(false_answers / answers, 4) if answers else None,
        "over_broad_rate": round(over_broad_answers / answers, 4) if answers else None,
        "replay_misses": len(holder.misses) if holder is not None and hasattr(holder, "misses") else None,
        "units": units,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    _git_commit()  # record the commit at start
    parser = argparse.ArgumentParser(description="Output-safety edits on clean answers, classified.")
    parser.add_argument("--sets", default=",".join(DEFAULT_SETS))
    parser.add_argument("--limit", type=int, default=0, help="First N tasks of each set (0 = all).")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR / "output_safety_audit.json"))
    add_llm_arguments(parser)
    parser.add_argument("--path", choices=["workflow", "workflow_llm", "agent"], default="workflow")
    parser.add_argument("--categories", default="", help="Comma-separated task categories (default: all).")
    parser.add_argument("--max-llm-calls", type=int, default=0, help="Call budget; stops after the first 429 too.")
    parser.add_argument("--record-llm", default="", help="Write every LLM turn per task to this JSON file.")
    parser.add_argument("--replay-llm", default="", help="Replay recorded LLM turns instead of calling an LLM.")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    replay = json.loads(Path(args.replay_llm).read_text(encoding="utf-8")) if args.replay_llm else None
    llm = None if replay is not None else _make_llm(args.llm, args.model)
    if args.path != "workflow" and llm is None and replay is None:
        raise SystemExit("the LLM paths need --llm or --replay-llm")
    budget = CallBudget(args.max_llm_calls) if args.max_llm_calls and llm is not None else None
    recordings: dict[str, list[dict[str, Any]]] | None = {} if args.record_llm else None
    categories = tuple(item for item in args.categories.split(",") if item)
    sets = [
        run_set(
            name,
            limit=args.limit,
            path=args.path,
            llm=llm,
            categories=categories,
            budget=budget,
            recordings=recordings,
            replay=replay,
        )
        for name in args.sets.split(",")
        if name
    ]
    if recordings is not None:
        Path(args.record_llm).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record_llm).write_text(json.dumps(recordings, ensure_ascii=False, indent=1), encoding="utf-8")
    units = [unit for item in sets for unit in item["units"]]
    totals = {
        key: sum(item[key] for item in sets)
        for key in ("tasks", "turns", "answers", "answers_edited", "edits", "edit_units", "llm_calls")
    }
    totals["answers_with_llm_error"] = sum(item["answers_with_llm_error"] for item in sets)
    totals["answers_with_false_edit"] = sum(item["answers_with_false_edit"] for item in sets)
    totals["answers_with_over_broad_edit"] = sum(item["answers_with_over_broad_edit"] for item in sets)
    totals["edits_per_answer"] = round(totals["edits"] / totals["answers"], 4) if totals["answers"] else None
    totals["false_attribution_rate"] = (
        round(totals["answers_with_false_edit"] / totals["answers"], 4) if totals["answers"] else None
    )
    totals["over_broad_rate"] = (
        round(totals["answers_with_over_broad_edit"] / totals["answers"], 4) if totals["answers"] else None
    )
    verdicts = Counter(f"{unit['edit']}:{unit['verdict']}" for unit in units)
    totals["edit_units_by_verdict"] = dict(sorted(verdicts.items()))
    ordered = sorted(units, key=lambda unit: {"false": 0, "over_broad": 1}.get(unit["verdict"], 2))
    report = {
        "kind": "output_safety_audit",
        "config": {
            "path": args.path,
            **llm_config(args.llm, args.model, llm),
            "llm_replay": args.replay_llm or None,
            "llm_call_budget": (
                {"max_calls": budget.max_calls, "calls": budget.calls, "stopped": budget.stopped}
                if budget is not None
                else None
            ),
            "categories": list(categories) or None,
            "sets": [item["set"] for item in sets],
            "limit": args.limit or None,
            "prompts": prompt_refs(),
            "commit": _git_commit(),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "eval_today": EVAL_TODAY.isoformat(),
            "wall_seconds": round(time.perf_counter() - started, 1),
            "command": _command("evaluation.agent_eval.output_safety_audit", argv),
            "definitions": {
                "answers": "turns that reached the output layer (refusals and clarifications excluded)",
                "edits_per_answer": "output-safety edit kinds per answer (finsight_output_safety_edits_total / runs)",
                "false_attribution_rate": "answers with at least one false edit / answers",
                "verdicts": "see the module docstring of evaluation/agent_eval/output_safety_audit.py",
            },
        },
        "totals": totals,
        "by_set": [{key: value for key, value in item.items() if key != "units"} for item in sets],
        "samples": ordered[:MAX_SAMPLES],
        "units": units,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"out": str(out), **{k: v for k, v in totals.items()}}, ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main()
