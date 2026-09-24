"""LangGraph orchestration for the FinSight agent.

Graph::

    guard_in ─┬─ refuse ────────────────────────────────────────────┐
              ├─ clarify ───────────────────────────────────────────┤
              ├─ execute_plan ─ compose ─┐                          │
              └─ agent_llm ⇄ agent_tools ┴─ verify ⇄ revise         │
                     │ (LLM failure) → execute_plan / compose       │
                                            verify ─ compliance ─ finalize

* ``guard_in`` runs the classical NLU and the explainable router.
* ``execute_plan`` runs the deterministic planner (fixed workflow, and the fallback when the LLM
  is missing or fails).
* ``agent_llm``/``agent_tools`` is the LLM tool loop with step, tool-call, and token budgets.
* ``verify`` checks citations and numbers; ``revise`` gives the LLM one chance to fix its draft,
  otherwise unsupported statements are removed.
* ``compliance`` applies the financial guardrails; ``finalize`` assembles the response.
"""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..chatbot import detect_query_language
from .compliance import apply_compliance
from .composer import compose_template, parse_answer
from .evidence import AgentEvidence, EvidenceStore
from .followups import next_questions, sentiment_summary
from .injection import sanitize_untrusted_text, tool_message_content
from .llm import LLMClient, LLMError, Pricing, Usage
from .memory import (
    dialog_context_from_turns,
    history_messages,
    listed_entities,
    resolve_coreference,
    turn_record,
)
from .planner import plan_from_nlu
from .prompts import (
    AGENT_SYSTEM_PROMPT,
    COMPOSE_SYSTEM_PROMPT,
    agent_user_message,
    compose_user_message,
    force_final_message,
    revision_message,
)
from .router import apply_finance_overrides, decide_route
from .state import RESET, AgentConfig, AgentState
from .tools import ToolRegistry, ToolResult
from .verifier import cited_ids, repair_answer, verify_answer

if TYPE_CHECKING:
    from ..service import QueryIntelligenceService

_MARKET_SOURCE_TYPES = {"market_api"}
_BUDGET_EXHAUSTED = '{"ok": false, "error": {"code": "unavailable", "message": "tool-call budget exhausted"}}'


class AgentRuntime:
    def __init__(
        self,
        service: QueryIntelligenceService,
        registry: ToolRegistry,
        llm: LLMClient | None = None,
        *,
        config: AgentConfig | None = None,
        pricing: Pricing | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.service = service
        self.registry = registry
        self.llm = llm if llm is not None and getattr(llm, "configured", True) else None
        self.config = config or AgentConfig()
        self.pricing = pricing
        self.today = today
        self._pool = ThreadPoolExecutor(max_workers=self.config.max_parallel_tools, thread_name_prefix="agent-run")

    # ------------------------------------------------------------------ graph

    def build_graph(self, checkpointer: Any = None):
        graph = StateGraph(AgentState)
        for name, node in (
            ("guard_in", self.guard_in),
            ("refuse", self.refuse),
            ("clarify", functools.partial(self.clarify, interactive=checkpointer is not None)),
            ("execute_plan", self.execute_plan),
            ("compose", self.compose),
            ("agent_llm", self.agent_llm),
            ("agent_tools", self.agent_tools),
            ("verify", self.verify),
            ("revise", self.revise),
            ("compliance", self.compliance),
            ("finalize", self.finalize),
        ):
            graph.add_node(name, _timed(name, node))
        graph.add_edge(START, "guard_in")
        graph.add_conditional_edges(
            "guard_in",
            _route_after_guard,
            {"refuse": "refuse", "clarify": "clarify", "workflow": "execute_plan", "agent": "agent_llm"},
        )
        graph.add_edge("refuse", "finalize")
        graph.add_conditional_edges("clarify", _after_clarify, {"guard_in": "guard_in", "finalize": "finalize"})
        graph.add_edge("execute_plan", "compose")
        graph.add_edge("compose", "verify")
        graph.add_conditional_edges(
            "agent_llm",
            _next,
            {"agent_tools": "agent_tools", "verify": "verify", "execute_plan": "execute_plan", "compose": "compose"},
        )
        graph.add_edge("agent_tools", "agent_llm")
        graph.add_conditional_edges("verify", _next, {"revise": "revise", "compliance": "compliance"})
        graph.add_edge("revise", "verify")
        graph.add_edge("compliance", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=checkpointer)

    def initial_state(
        self,
        query: str,
        *,
        mode: str = "auto",
        user_profile: dict[str, Any] | None = None,
        dialog_context: list[dict[str, Any]] | None = None,
    ) -> AgentState:
        # Every turn-scoped field is reset explicitly: with a checkpointer, values from the previous
        # turn on the same thread would otherwise leak into this one. ``turns`` is session memory.
        return {
            "query": query,
            "mode": mode,
            "started_at": time.time(),
            "user_profile": user_profile or {},
            "dialog_context": dialog_context or [],
            "nlu": {},
            "route": "",
            "route_reasons": [],
            "messages": [],
            "llm_steps": 0,
            "llm_calls": 0,
            "usage": {},
            "next": "",
            "draft": {},
            "draft_source": "",
            "revisions": 0,
            "verification": {},
            "verification_notes": [],
            "compliance_notes": [],
            "answer": {},
            "result": {},
            "clarification_rounds": 0,
            "tool_log": {RESET: []},
            "evidence": {RESET: True},
            "degraded": {RESET: []},
            "spans": {RESET: []},
            "llm_log": {RESET: []},
        }

    def run(self, query: str, *, mode: str = "auto", **kwargs: Any) -> dict[str, Any]:
        graph = self.build_graph()
        state = graph.invoke(self.initial_state(query, mode=mode, **kwargs))
        return state["result"]

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ nodes

    def guard_in(self, state: AgentState) -> dict[str, Any]:
        dialog_context = dialog_context_from_turns(state.get("turns") or [], state.get("dialog_context") or [])
        nlu = self.service.analyze_query(
            state["query"], user_profile=state.get("user_profile") or {}, dialog_context=dialog_context
        )
        coreference_reason = None
        if not listed_entities(nlu):
            rewrite = resolve_coreference(state["query"], state.get("turns") or [])
            if rewrite is not None:
                rewritten_query, coreference_reason = rewrite
                nlu = self.service.analyze_query(
                    rewritten_query, user_profile=state.get("user_profile") or {}, dialog_context=dialog_context
                )
        nlu, override_reasons = apply_finance_overrides(nlu, state["query"])
        decision = decide_route(nlu, mode=state.get("mode", "auto"), query=state["query"])  # type: ignore[arg-type]
        reasons = [*decision.reasons, *override_reasons]
        if coreference_reason:
            reasons.append(coreference_reason)
        update: dict[str, Any] = {"nlu": nlu, "route": decision.route, "route_reasons": reasons}
        if decision.route == "agent" and self.llm is None:
            update["route"] = "workflow"
            update["degraded"] = ["no_llm_configured:agent_route_downgraded_to_workflow"]
        return update

    def refuse(self, state: AgentState) -> dict[str, Any]:
        from .compliance import _guards

        answer = _guards()._out_of_scope_answer_response(zh=self._zh(state))
        answer = {
            key: answer[key] for key in ("answer", "key_points", "evidence_used", "limitations", "risk_disclaimer")
        }
        return {"answer": answer, "draft_source": "guardrail"}

    def clarify(self, state: AgentState, *, interactive: bool = False) -> dict[str, Any]:
        zh = self._zh(state)
        question = (
            "请问您想了解哪只股票、基金、ETF 或指数？请提供名称或代码（例如 600519.SH）。"
            if zh
            else "Which stock, fund, ETF, or index do you mean? Please give a name or ticker (e.g. 600519.SH)."
        )
        if interactive and state.get("clarification_rounds", 0) < 1:
            reply = interrupt(
                {
                    "type": "clarification",
                    "question": question,
                    "missing_slots": (state.get("nlu") or {}).get("missing_slots") or [],
                    "original_query": state["query"],
                }
            )
            reply_text = str(reply.get("reply") if isinstance(reply, dict) else reply).strip()
            if reply_text:
                context = [*(state.get("dialog_context") or []), {"role": "user", "content": reply_text}]
                return {
                    "dialog_context": context,
                    "clarification_rounds": state.get("clarification_rounds", 0) + 1,
                    "next": "guard_in",
                }
        answer = {
            "answer": (
                "请问您想了解哪只股票、基金、ETF 或指数？请提供名称或代码（例如 600519.SH），我再基于证据回答。"
                if zh
                else "Which stock, fund, ETF, or index do you mean? Please give a name or ticker (e.g. 600519.SH)."
            ),
            "key_points": [],
            "evidence_used": [],
            "limitations": ["缺少明确的标的" if zh else "The target security is missing"],
        }
        return {"answer": answer, "draft_source": "clarification", "next": "finalize"}

    def execute_plan(self, state: AgentState) -> dict[str, Any]:
        plan = plan_from_nlu(state.get("nlu") or {})
        existing = {(entry["tool"], _key(entry.get("arguments"))) for entry in state.get("tool_log") or []}
        calls = [call for call in plan.calls if (call.tool, _key(call.arguments)) not in existing]
        results = self._run_tools([(call.tool, call.arguments) for call in calls])
        log = [
            _log_entry(result, source="planner", reason=call.reason, step=0)
            for call, result in zip(calls, results, strict=True)
        ]
        evidence, flagged = _evidence_update(results)
        update: dict[str, Any] = {"tool_log": log, "evidence": evidence}
        if flagged:
            update["degraded"] = ["instruction_like_text_removed_from_evidence"]
        return update

    def compose(self, state: AgentState) -> dict[str, Any]:
        zh = self._zh(state)
        tool_log = state.get("tool_log") or []
        style = str((state.get("nlu") or {}).get("question_style") or "")
        llm_failed = any(str(item).startswith("llm_error") for item in state.get("degraded") or [])
        use_llm = (
            self.llm is not None and self.config.llm_compose and state.get("next") != "template" and not llm_failed
        )
        if use_llm:
            evidence_views = [
                AgentEvidence.model_validate(item).prompt_view() for item in (state.get("evidence") or {}).values()
            ]
            messages = [
                {"role": "system", "content": COMPOSE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": compose_user_message(
                        state["query"],
                        state.get("nlu") or {},
                        evidence_views,
                        _failures(tool_log),
                        language="zh" if zh else "en",
                    ),
                },
            ]
            try:
                turn = self.llm.chat(messages, json_mode=True)  # type: ignore[union-attr]
            except LLMError as exc:
                return self._template_update(tool_log, zh, degraded=f"llm_compose_failed:{exc}", style=style)
            return {
                "draft": parse_answer(turn.content),
                "draft_source": "llm_compose",
                "messages": [*messages, turn.as_message()],
                "llm_calls": state.get("llm_calls", 0) + 1,
                "usage": _add_usage(state.get("usage"), turn.usage),
                "llm_log": [_llm_entry("compose", turn)],
            }
        return self._template_update(tool_log, zh, style=style)

    def agent_llm(self, state: AgentState) -> dict[str, Any]:
        assert self.llm is not None
        zh = self._zh(state)
        messages = list(state.get("messages") or [])
        if not messages:
            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                *history_messages(state.get("turns") or []),
                {
                    "role": "user",
                    "content": agent_user_message(
                        state["query"], state.get("nlu") or {}, language="zh" if zh else "en"
                    ),
                },
            ]
        usage = state.get("usage") or {}
        tool_calls_made = sum(1 for entry in state.get("tool_log") or [] if entry.get("source") == "llm")
        stop_reason = None
        if state.get("llm_steps", 0) >= self.config.max_llm_steps:
            stop_reason = f"step budget of {self.config.max_llm_steps} reached"
        elif tool_calls_made >= self.config.max_tool_calls:
            stop_reason = f"tool-call budget of {self.config.max_tool_calls} reached"
        elif _total_tokens(usage) >= self.config.token_budget:
            stop_reason = f"token budget of {self.config.token_budget} reached"
        elif time.time() - float(state.get("started_at") or time.time()) >= self.config.run_deadline_s:
            stop_reason = f"run deadline of {self.config.run_deadline_s:g}s reached"
        if stop_reason:
            messages.append({"role": "user", "content": force_final_message(stop_reason)})

        try:
            if stop_reason:
                turn = self.llm.chat(messages, json_mode=True)
            else:
                turn = self.llm.chat(messages, self.registry.to_openai_tools())
        except LLMError as exc:
            degraded = [f"llm_error:{exc}"]
            if state.get("evidence"):
                return {"next": "template", "degraded": degraded}
            return {"next": "execute_plan", "degraded": degraded}

        update: dict[str, Any] = {
            "messages": [*messages, turn.as_message()],
            "llm_calls": state.get("llm_calls", 0) + 1,
            "usage": _add_usage(usage, turn.usage),
            "llm_log": [_llm_entry("agent_llm", turn, step=state.get("llm_steps", 0))],
        }
        if stop_reason:
            update["degraded"] = [f"budget:{stop_reason}"]
        if turn.tool_calls and not stop_reason:
            update["next"] = "agent_tools"
            return update
        update.update({"next": "verify", "draft": parse_answer(turn.content), "draft_source": "llm_agent"})
        return update

    def agent_tools(self, state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else {}
        calls = last.get("tool_calls") or []
        used = sum(1 for entry in state.get("tool_log") or [] if entry.get("source") == "llm")
        remaining = max(self.config.max_tool_calls - used, 0)
        runnable = calls[:remaining]
        results = self._run_tools([(call["function"]["name"], call["function"].get("arguments")) for call in runnable])
        step = state.get("llm_steps", 0) + 1
        tool_messages = []
        log = []
        flagged_any = False
        for call, result in zip(runnable, results, strict=True):
            content, flagged = tool_message_content(result.tool, result.observation())
            flagged_any = flagged_any or flagged
            tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            log.append(_log_entry(result, source="llm", reason="selected by LLM", step=step, flagged=flagged))
        for call in calls[len(runnable) :]:
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": _BUDGET_EXHAUSTED,
                }
            )
        evidence_update, evidence_flagged = _evidence_update(results)
        flagged_any = flagged_any or evidence_flagged
        update: dict[str, Any] = {
            "messages": [*messages, *tool_messages],
            "llm_steps": step,
            "tool_log": log,
            "evidence": evidence_update,
        }
        if flagged_any:
            update["degraded"] = ["instruction_like_text_removed_from_tool_output"]
        return update

    def verify(self, state: AgentState) -> dict[str, Any]:
        draft = state.get("draft") or {}
        store = _store(state)
        report = verify_answer(draft, store, query=state["query"])
        update: dict[str, Any] = {"verification": report.model_dump()}
        can_revise = (
            not report.passed
            and self.llm is not None
            and state.get("draft_source") in {"llm_agent", "llm_compose"}
            and state.get("revisions", 0) < self.config.max_revisions
        )
        if can_revise:
            update["next"] = "revise"
            return update
        if not report.passed:
            repaired, notes = repair_answer(draft, report, store, zh=self._zh(state))
            update.update(
                {"draft": repaired, "verification_notes": notes, "degraded": ["verification_failed:repaired"]}
            )
        update["next"] = "compliance"
        return update

    def revise(self, state: AgentState) -> dict[str, Any]:
        assert self.llm is not None
        report_text = _feedback_text(state.get("verification") or {})
        messages = [*(state.get("messages") or []), {"role": "user", "content": revision_message(report_text)}]
        update: dict[str, Any] = {"revisions": state.get("revisions", 0) + 1}
        try:
            turn = self.llm.chat(messages, json_mode=True)
        except LLMError as exc:
            update["degraded"] = [f"llm_revision_failed:{exc}"]
            return update
        update.update(
            {
                "messages": [*messages, turn.as_message()],
                "draft": parse_answer(turn.content),
                "llm_calls": state.get("llm_calls", 0) + 1,
                "usage": _add_usage(state.get("usage"), turn.usage),
                "llm_log": [_llm_entry("revise", turn)],
            }
        )
        return update

    def compliance(self, state: AgentState) -> dict[str, Any]:
        draft = dict(state.get("draft") or {})
        limitations = list(draft.get("limitations") or [])
        limitations.extend(state.get("verification_notes") or [])
        draft["limitations"] = list(dict.fromkeys(limitations))
        market = [
            AgentEvidence.model_validate(item)
            for item in (state.get("evidence") or {}).values()
            if item.get("source_type") in _MARKET_SOURCE_TYPES
        ]
        answer, notes = apply_compliance(
            draft,
            query=state["query"],
            nlu_result=state.get("nlu") or {},
            tool_failures=_failures(state.get("tool_log") or []),
            market_evidence=market,
            today=self.today(),
        )
        return {"answer": answer, "compliance_notes": notes}

    def finalize(self, state: AgentState) -> dict[str, Any]:
        from ..chatbot import DEFAULT_RISK_DISCLAIMER_EN, DEFAULT_RISK_DISCLAIMER_ZH

        zh = self._zh(state)
        answer = dict(state.get("answer") or {})
        answer.setdefault("risk_disclaimer", DEFAULT_RISK_DISCLAIMER_ZH if zh else DEFAULT_RISK_DISCLAIMER_EN)
        evidence = state.get("evidence") or {}
        cited = [evidence_id for evidence_id in cited_ids(answer) if evidence_id in evidence]
        ordered = cited + [evidence_id for evidence_id in evidence if evidence_id not in cited]
        usage = state.get("usage") or {}
        usage_model = Usage(**usage) if usage else Usage()
        nlu = state.get("nlu") or {}
        result = {
            "run_id": uuid.uuid4().hex,
            "query": state["query"],
            "language": "zh" if zh else "en",
            "route": state.get("route"),
            "route_reasons": state.get("route_reasons") or [],
            "answer": answer.get("answer", ""),
            "key_points": answer.get("key_points") or [],
            "evidence_used": cited,
            "limitations": answer.get("limitations") or [],
            "risk_disclaimer": answer.get("risk_disclaimer"),
            "evidence_sources": [_source_view(evidence[evidence_id]) for evidence_id in ordered[:12]],
            "tool_calls": [_public_log(entry) for entry in state.get("tool_log") or []],
            "verification": state.get("verification") or {},
            "compliance_notes": state.get("compliance_notes") or [],
            "degraded": list(dict.fromkeys(state.get("degraded") or [])),
            "answer_source": state.get("draft_source"),
            "llm": {
                "model": getattr(self.llm, "model", None),
                "calls": state.get("llm_calls", 0),
                "steps": state.get("llm_steps", 0),
                "usage": usage_model.model_dump() | {"total_tokens": usage_model.total_tokens},
                "cost": None if self.pricing is None else self.pricing.cost(usage_model),
                "currency": None if self.pricing is None else self.pricing.currency,
                "log": state.get("llm_log") or [],
            },
            "nlu_summary": {
                "question_style": nlu.get("question_style"),
                "product_type": (nlu.get("product_type") or {}).get("label"),
                "entities": [
                    {"name": entity.get("canonical_name"), "symbol": entity.get("symbol")}
                    for entity in nlu.get("entities") or []
                ],
                "risk_flags": nlu.get("risk_flags") or [],
            },
            "spans": state.get("spans") or [],
            "turn_index": len(state.get("turns") or []),
            "sentiment": sentiment_summary(state.get("tool_log") or []),
            "next_questions": next_questions(
                query=state["query"],
                route=str(state.get("route") or ""),
                nlu_result=nlu,
                tool_log=state.get("tool_log") or [],
                zh=zh,
                limit=self.config.max_next_questions,
            ),
        }
        return {"result": result, "turns": [turn_record(state, result)]}

    # ---------------------------------------------------------------- helpers

    def _template_update(
        self, tool_log: list[dict[str, Any]], zh: bool, degraded: str | None = None, *, style: str = ""
    ) -> dict[str, Any]:
        update: dict[str, Any] = {
            "draft": compose_template(tool_log, zh=zh, question_style=style),
            "draft_source": "template",
        }
        if degraded:
            update["degraded"] = [degraded]
        return update

    def _run_tools(self, calls: list[tuple[str, Any]]) -> list[ToolResult]:
        if not calls:
            return []
        futures = [self._pool.submit(self.registry.run, name, arguments) for name, arguments in calls]
        return [future.result() for future in futures]

    @staticmethod
    def _zh(state: AgentState) -> bool:
        return detect_query_language(state.get("query", "")) == "zh"


def _timed(name: str, node: Callable[[AgentState], dict[str, Any]]) -> Callable[[AgentState], dict[str, Any]]:
    def wrapper(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        wall = time.time()
        update = node(state)
        span = {
            "node": name,
            "started_at": round(wall, 3),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        existing = update.get("spans") or []
        if isinstance(update.get("result"), dict):
            update["result"]["spans"] = [*update["result"].get("spans", []), span]
        return {**update, "spans": [*existing, span]}

    wrapper.__name__ = name
    return wrapper


def _after_clarify(state: AgentState) -> str:
    return "guard_in" if state.get("next") == "guard_in" else "finalize"


def _route_after_guard(state: AgentState) -> str:
    return str(state.get("route") or "workflow")


def _next(state: AgentState) -> str:
    target = state.get("next") or "compliance"
    return "compose" if target == "template" else target


def _key(arguments: Any) -> str:
    import json

    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)


def _log_entry(result: ToolResult, *, source: str, reason: str, step: int, flagged: bool = False) -> dict[str, Any]:
    return {
        "tool": result.tool,
        "arguments": result.arguments,
        "ok": result.ok,
        "data": result.data,
        "error": result.error.model_dump() if result.error else None,
        "latency_ms": result.latency_ms,
        "started_at": round(time.time() - result.latency_ms / 1000, 3),
        "attempts": result.attempts,
        "cached": result.cached,
        "evidence_ids": [item.evidence_id for item in result.evidence],
        "source": source,
        "reason": reason,
        "step": step,
        "instruction_like_text_removed": flagged,
    }


def _llm_entry(node: str, turn: Any, *, step: int | None = None) -> dict[str, Any]:
    return {
        "node": node,
        "step": step,
        "model": turn.model,
        "started_at": round(time.time() - turn.latency_ms / 1000, 3),
        "latency_ms": turn.latency_ms,
        "prompt_tokens": turn.usage.prompt_tokens,
        "completion_tokens": turn.usage.completion_tokens,
        "prompt_cache_hit_tokens": turn.usage.prompt_cache_hit_tokens,
        "reasoning_tokens": turn.usage.reasoning_tokens,
        "tool_calls": [call.name for call in turn.tool_calls],
        "finish_reason": turn.finish_reason,
    }


def _public_log(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key != "data"}


def _evidence_update(results: list[ToolResult]) -> tuple[dict[str, dict[str, Any]], bool]:
    """Evidence keyed by id, with instruction-like text in document fields redacted on ingestion."""
    update: dict[str, dict[str, Any]] = {}
    flagged_any = False
    for result in results:
        for item in result.evidence:
            dumped = item.model_dump(mode="json")
            for field in ("title", "text_excerpt"):
                if isinstance(dumped.get(field), str):
                    dumped[field], flagged = sanitize_untrusted_text(dumped[field])
                    flagged_any = flagged_any or flagged
            update[item.evidence_id] = dumped
    return update, flagged_any


def _store(state: AgentState) -> EvidenceStore:
    store = EvidenceStore()
    for item in (state.get("evidence") or {}).values():
        store.add(AgentEvidence.model_validate(item))
    return store


def _failures(tool_log: list[dict[str, Any]]) -> list[str]:
    failures = []
    for entry in tool_log:
        if not entry.get("ok"):
            error = entry.get("error") or {}
            failures.append(f"{entry.get('tool')}: {error.get('code')}")
    return list(dict.fromkeys(failures))


def _add_usage(current: dict[str, int] | None, extra: Usage) -> dict[str, int]:
    total = Usage(**(current or {})) + extra
    return total.model_dump()


def _total_tokens(usage: dict[str, int]) -> int:
    return int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))


def _feedback_text(verification: dict[str, Any]) -> str:
    from .verifier import VerificationReport

    return VerificationReport.model_validate(verification).feedback() if verification else ""


def _source_view(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in ("evidence_id", "kind", "source_type", "title", "source_name", "source_url", "as_of", "produced_by")
    }
