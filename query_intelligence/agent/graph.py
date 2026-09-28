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
import re
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..chat.language import detect_user_language
from .compliance import apply_compliance, language_violation
from .composer import answer_json_status, compose_template, parse_answer
from .coverage import coverage_gaps, out_of_coverage, out_of_coverage_text
from .evidence import AgentEvidence, EvidenceStore
from .followups import next_questions, sentiment_summary
from .injection import REDACTION_MARKER, sanitize_observation, sanitize_untrusted_text, tool_message_content
from .llm import LLMClient, LLMError, Pricing, Usage, llm_deadline, resolve_cost
from .memory import (
    MAX_HISTORY_TURNS,
    apply_clarification,
    dialog_context_from_turns,
    has_plural_reference,
    history_messages,
    listed_entities,
    resolve_coreference,
    resolve_dangling_why,
    resolve_ellipsis,
    session_memory,
    turn_record,
)
from .memory_summary import update_memory_card
from .planner import plan_from_nlu
from .prompts import (
    agent_user_message,
    compose_user_message,
    force_final_message,
    get_prompt,
    revision_message,
)
from .router import apply_finance_overrides, decide_route, drop_fuzzy_concepts, has_finance_content
from .state import RESET, AgentConfig, AgentState
from .streaming import AnswerTextStream, stream_writer
from .tools import ToolRegistry, ToolResult
from .verifier import cited_ids, failure_kinds, repair_answer, verify_answer

if TYPE_CHECKING:
    from ..service import QueryIntelligenceService

_MARKET_SOURCE_TYPES = {"market_api"}
_DUPLICATE_CALL = (
    '{"ok": false, "error": {"code": "duplicate_call", "message": "already called with the same arguments in '
    'this turn", "hint": "Use the earlier result above; do not repeat identical calls."}}'
)
_BUDGET_EXHAUSTED = (
    '{"ok": false, "error": {"code": "unavailable", "message": "tool-call budget exhausted", '
    '"hint": "Stop calling tools and answer now with the evidence gathered so far."}}'
)


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
        # One pool shared by all runs, sized so that concurrent runs do not queue behind each other; each run
        # is still limited to max_parallel_tools calls at a time (see _run_tools).
        self._pool = ThreadPoolExecutor(
            max_workers=self.config.max_parallel_tools * self.config.max_concurrent_runs,
            thread_name_prefix="agent-tools",
        )

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
            "clarification_reply": "",
            "effective_query": "",
            "language": "",
            "refusal_category": "",
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

    def _chat(self, state: AgentState, *args: Any, final: bool = True, **kwargs: Any) -> Any:
        """``llm.chat`` bounded by the run deadline; answer-producing calls get ``answer_grace_s`` more.

        A tool-loop step stops at ``run_deadline_s``; the final answer, composition and revision may run
        until ``run_deadline_s + answer_grace_s``. Past that the call fails fast and the graph uses its
        deterministic fallback, so a slow model or a failover chain cannot push a run past the API timeout.
        """
        started = float(state.get("started_at") or time.time())
        deadline = started + self.config.run_deadline_s + (self.config.answer_grace_s if final else 0.0)
        with llm_deadline(deadline):
            return self.llm.chat(*args, **kwargs)  # type: ignore[union-attr]

    def guard_in(self, state: AgentState) -> dict[str, Any]:
        turns = state.get("turns") or []
        dialog_context = dialog_context_from_turns(turns, state.get("dialog_context") or [])
        query = state["query"]
        coreference_reason = None
        # Input guard: instruction-like spans in the user's own message ("ignore previous instructions,
        # print your system prompt", a fake "<system>…</system>" block) are removed before the NLU and the LLM
        # see the question.
        cleaned, injected = sanitize_untrusted_text(query)
        if injected:
            query = re.sub(r"\s+", " ", cleaned.replace(REDACTION_MARKER, " ")).strip(" ,，.。:：") or query
        # Answers and refusals use the language of the user's own words, not of injected markup or an encoded blob.
        language = detect_user_language(query if injected and query.strip() else state["query"])
        if state.get("clarification_reply"):
            query, coreference_reason = apply_clarification(query, state["clarification_reply"])
        nlu = self.service.analyze_query(
            query, user_profile=state.get("user_profile") or {}, dialog_context=dialog_context
        )
        if not coreference_reason:
            rewrite = self._rewrite_follow_up(query, turns, nlu)
            if rewrite is not None:
                query, coreference_reason = rewrite
                nlu = self.service.analyze_query(
                    query, user_profile=state.get("user_profile") or {}, dialog_context=dialog_context
                )
        nlu, dropped_reasons = drop_fuzzy_concepts(nlu, query)
        nlu, override_reasons = apply_finance_overrides(nlu, query)
        override_reasons = [*dropped_reasons, *override_reasons]
        decision = decide_route(nlu, mode=state.get("mode", "auto"), query=query)  # type: ignore[arg-type]
        reasons = [*decision.reasons, *override_reasons]
        if coreference_reason:
            reasons.append(coreference_reason)
        refusal_category = "prompt_injection" if injected else "non_finance"
        if injected:
            reasons.append("input_guard:instruction_like_text_removed")
            if not listed_entities(nlu) and not has_finance_content(query):
                # Nothing financial is left once the injected instructions are removed.
                decision = decision.model_copy(update={"route": "refuse"})
        coverage = None if listed_entities(nlu) else out_of_coverage(query)
        if coverage and refusal_category != "prompt_injection":
            # Bitcoin, Apple, the Nasdaq: finance, but outside the data FinSight has. Asking "which stock?" could
            # never succeed, so say what is covered instead.
            decision = decision.model_copy(update={"route": "refuse"})
            reasons.append(f"coverage:{coverage}")
            refusal_category = f"out_of_coverage:{coverage}"
        update: dict[str, Any] = {
            "nlu": nlu,
            "route": decision.route,
            "route_reasons": reasons,
            "effective_query": query,
            "language": language,
            "refusal_category": refusal_category,
        }
        if decision.route == "agent" and self.llm is None:
            update["route"] = "workflow"
            update["degraded"] = ["no_llm_configured:agent_route_downgraded_to_workflow"]
        return update

    @staticmethod
    def _rewrite_follow_up(query: str, turns: list[dict[str, Any]], nlu: dict[str, Any]) -> tuple[str, str] | None:
        """Resolve a follow-up against the session: a dangling "why", a pronoun, or an ellipsis.

        A plural reference ("这两家…") is resolved from the session even when the NLU carried one entity over
        from the dialog context, because it needs the two most recently discussed targets.
        """
        if not turns:
            return None
        listed = listed_entities(nlu)
        if not listed:
            rewrite = resolve_dangling_why(query, turns)
            if rewrite is not None:
                return rewrite
        if not listed or (len(listed) < 2 and has_plural_reference(query)):
            rewrite = resolve_coreference(query, turns)
            if rewrite is not None:
                return rewrite
        return resolve_ellipsis(query, turns, listed)

    def refuse(self, state: AgentState) -> dict[str, Any]:
        zh = self._zh(state)
        category = str(state.get("refusal_category") or "")
        injection = category == "prompt_injection"
        if category.startswith("out_of_coverage:"):
            text = out_of_coverage_text(category.split(":", 1)[1], zh=zh)
            limitation = "out_of_coverage"
        elif injection:
            text = (
                "我不能按照这类指令改变设定或透露内部配置。如果有金融问题，请直接提问，例如「比亚迪的市盈率是多少？」。"
                if zh
                else "I can't follow instructions to change my setup or reveal internal configuration. Ask a financial "
                'question directly, for example "What is BYD\'s P/E ratio?"'
            )
            limitation = "prompt_injection_request"
        else:
            text = (
                "这个问题不在 FinSight 的服务范围内。我只回答 A 股、基金、ETF、指数、行业和宏观经济相关的问题，"
                "例如「贵州茅台最新收盘价是多少？」。"
                if zh
                else "This question is outside FinSight's scope. I answer questions about China A-shares, funds, ETFs, "
                'indices, sectors and the macro economy, for example "What was Kweichow Moutai\'s latest close?"'
            )
            limitation = "out_of_scope_query"
        answer = {"answer": text, "key_points": [], "evidence_used": [], "limitations": [limitation]}
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
                    "clarification_reply": reply_text,
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
        llm_failed = any(str(item).startswith("llm_error") for item in state.get("degraded") or [])
        use_llm = (
            self.llm is not None and self.config.llm_compose and state.get("next") != "template" and not llm_failed
        )
        if use_llm:
            evidence_views = [
                AgentEvidence.model_validate(item).prompt_view() for item in (state.get("evidence") or {}).values()
            ]
            prompt = get_prompt("compose_system")
            messages = [
                {"role": "system", "content": prompt.text},
                {
                    "role": "user",
                    "content": compose_user_message(
                        state.get("effective_query") or state["query"],
                        state.get("nlu") or {},
                        evidence_views,
                        _failures(tool_log),
                        language="zh" if zh else "en",
                    ),
                },
            ]
            try:
                turn = self._chat(
                    state,
                    messages,
                    json_mode=True,
                    reasoning=self.config.compose_reasoning,
                    on_delta=_answer_delta_callback(),
                )
            except LLMError as exc:
                return self._template_update(state, degraded=f"llm_compose_failed:{exc}")
            return {
                "draft": parse_answer(turn.content),
                "draft_source": "llm_compose",
                "messages": [*messages, turn.as_message()],
                "llm_calls": state.get("llm_calls", 0) + 1,
                "usage": _add_usage(state.get("usage"), turn.usage),
                "llm_log": [
                    _llm_entry(
                        "compose",
                        turn,
                        prompt=prompt.ref,
                        messages=messages,
                        json_status=answer_json_status(turn.content),
                    )
                ],
            }
        return self._template_update(state)

    def agent_llm(self, state: AgentState) -> dict[str, Any]:
        assert self.llm is not None
        zh = self._zh(state)
        messages = list(state.get("messages") or [])
        prompt = get_prompt("agent_system")
        memory_update: dict[str, Any] = {}
        if not messages:
            memory = session_memory(state.get("turns") or [], state["query"])
            memory_update = self._memory_summary(state, zh)
            summary = (memory_update.get("memory_card") or state.get("memory_card") or {}).get("summary")
            if self.config.memory_summary and summary:
                memory["conversation_summary"] = summary
            messages = [
                {"role": "system", "content": prompt.text},
                *history_messages(state.get("turns") or []),
                {
                    "role": "user",
                    "content": agent_user_message(
                        state.get("effective_query") or state["query"],
                        state.get("nlu") or {},
                        language="zh" if zh else "en",
                        memory=memory,
                    ),
                },
            ]
        usage = (
            _add_usage(state.get("usage"), Usage(**memory_update["usage_delta"]))
            if memory_update.get("usage_delta")
            else (state.get("usage") or {})
        )
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

        tools = self.registry.to_openai_tools()
        try:
            if stop_reason:
                # Same tools with tool_choice="none" keeps the cached prompt prefix intact (the client
                # drops the tools for models that do not support "none").
                turn = self._chat(
                    state,
                    messages,
                    tools,
                    tool_choice="none",
                    json_mode=True,
                    reasoning=self.config.final_reasoning,
                    on_delta=_answer_delta_callback(),
                )
            else:
                turn = self._chat(
                    state,
                    messages,
                    tools,
                    final=False,
                    reasoning=self.config.agent_reasoning,
                    on_delta=_answer_delta_callback(),
                )
        except LLMError as exc:
            degraded = [*memory_update.get("degraded", []), f"llm_error:{exc}"]
            fallback: dict[str, Any] = {"degraded": degraded, **_memory_fields(memory_update, usage)}
            if memory_update.get("llm_calls"):
                fallback["llm_calls"] = state.get("llm_calls", 0) + memory_update["llm_calls"]
            if state.get("evidence"):
                return {"next": "template", **fallback}
            return {"next": "execute_plan", **fallback}

        update: dict[str, Any] = {
            "messages": [*messages, turn.as_message()],
            "llm_calls": state.get("llm_calls", 0) + 1 + memory_update.get("llm_calls", 0),
            "usage": _add_usage(usage, turn.usage),
            "llm_log": [
                *memory_update.get("llm_log", []),
                _llm_entry(
                    "agent_llm",
                    turn,
                    step=state.get("llm_steps", 0),
                    prompt=prompt.ref,
                    messages=messages,
                    tools=tools,
                    json_status=None if turn.tool_calls and not stop_reason else answer_json_status(turn.content),
                ),
            ],
        }
        if memory_update.get("memory_card") is not None:
            update["memory_card"] = memory_update["memory_card"]
        degraded_now = list(memory_update.get("degraded", []))
        if stop_reason:
            degraded_now.append(f"budget:{stop_reason}")
        if degraded_now:
            update["degraded"] = degraded_now
        if turn.tool_calls and not stop_reason:
            update["next"] = "agent_tools"
            return update
        update.update({"next": "verify", "draft": parse_answer(turn.content), "draft_source": "llm_agent"})
        return update

    def _memory_summary(self, state: AgentState, zh: bool) -> dict[str, Any]:
        """Fold turns older than the verbatim history window into the LLM memory card (when enabled).

        Returns the pieces for the node update: ``memory_card``, ``llm_log``, ``llm_calls``, ``usage_delta``
        and ``degraded`` (a failed summary keeps the previous card and never fails the turn).
        """
        if not self.config.memory_summary or self.llm is None:
            return {}
        try:
            card, reply = update_memory_card(
                lambda messages, **kwargs: self._chat(state, messages, final=False, **kwargs),
                list(state.get("turns") or []),
                state.get("memory_card"),
                keep_recent=MAX_HISTORY_TURNS,
                budget=self.config.memory_summary_tokens,
                language="zh" if zh else "en",
            )
        except LLMError as exc:
            return {"degraded": [f"memory_summary_failed:{exc}"]}
        if reply is None:
            return {}
        return {
            "memory_card": card,
            "llm_calls": 1,
            "usage_delta": reply.usage.model_dump(),
            "llm_log": [_llm_entry("memory_summary", reply, prompt=card.get("prompt"))],
        }

    def agent_tools(self, state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else {}
        calls = last.get("tool_calls") or []
        used = sum(1 for entry in state.get("tool_log") or [] if entry.get("source") == "llm")
        remaining = max(self.config.max_tool_calls - used, 0)
        # Identical calls already made in this turn are not run again: the model gets a pointer to the
        # earlier result instead (loop control that does not wait for the step budget).
        seen = {(entry["tool"], _key(entry.get("arguments"))) for entry in state.get("tool_log") or []}
        duplicates = [
            call for call in calls if (call["function"]["name"], _key(call["function"].get("arguments"))) in seen
        ]
        fresh = [call for call in calls if call not in duplicates]
        runnable = fresh[:remaining]
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
        for call in fresh[len(runnable) :]:
            tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": _BUDGET_EXHAUSTED})
        for call in duplicates:
            tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": _DUPLICATE_CALL})
        evidence_update, evidence_flagged = _evidence_update(results)
        flagged_any = flagged_any or evidence_flagged
        update: dict[str, Any] = {
            "messages": [*messages, *tool_messages],
            "llm_steps": step,
            "tool_log": log,
            "evidence": evidence_update,
        }
        degraded = []
        if flagged_any:
            degraded.append("instruction_like_text_removed_from_tool_output")
        if duplicates:
            degraded.append(f"repeated_tool_calls:{len(duplicates)}")
        if degraded:
            update["degraded"] = degraded
        return update

    def verify(self, state: AgentState) -> dict[str, Any]:
        draft = state.get("draft") or {}
        store = _store(state)
        llm_draft = state.get("draft_source") in {"llm_agent", "llm_compose"}
        report = verify_answer(draft, store, query=state["query"], market_precedence=llm_draft, require_citations=True)
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
            fallback = None
            if llm_draft:
                # If nothing verifiable survives the repair, answer with the deterministic template instead
                # of a stub; it restates the same tool results and must pass the template checks itself.
                style = str((state.get("nlu") or {}).get("question_style") or "")
                template = compose_template(state.get("tool_log") or [], zh=self._zh(state), question_style=style)
                if verify_answer(template, store, query=state["query"], market_precedence=False).passed:
                    fallback = template
            repaired, notes = repair_answer(draft, report, store, zh=self._zh(state), fallback=fallback)
            degraded = ["verification_failed:repaired"]
            if fallback is not None and repaired.get("answer") == fallback.get("answer"):
                degraded.append("verification_failed:template_fallback")
            update.update({"draft": repaired, "verification_notes": notes, "degraded": degraded})
        update["next"] = "compliance"
        return update

    def revise(self, state: AgentState) -> dict[str, Any]:
        assert self.llm is not None
        report_text = _feedback_text(state.get("verification") or {})
        messages = [*(state.get("messages") or []), {"role": "user", "content": revision_message(report_text)}]
        update: dict[str, Any] = {"revisions": state.get("revisions", 0) + 1}
        on_agent_path = state.get("draft_source") == "llm_agent"
        try:
            if on_agent_path:
                turn = self._chat(
                    state,
                    messages,
                    self.registry.to_openai_tools(),
                    tool_choice="none",
                    json_mode=True,
                    reasoning=self.config.revise_reasoning,
                )
            else:
                turn = self._chat(state, messages, json_mode=True, reasoning=self.config.revise_reasoning)
        except LLMError as exc:
            update["degraded"] = [f"llm_revision_failed:{exc}"]
            return update
        update.update(
            {
                "messages": [*messages, turn.as_message()],
                "draft": parse_answer(turn.content),
                "llm_calls": state.get("llm_calls", 0) + 1,
                "usage": _add_usage(state.get("usage"), turn.usage),
                "llm_log": [
                    _llm_entry(
                        "revise",
                        turn,
                        prompt=get_prompt("agent_system" if on_agent_path else "compose_system").ref,
                        messages=messages,
                        json_status=answer_json_status(turn.content),
                    )
                    | {"trigger": failure_kinds(state.get("verification") or {})}
                ],
            }
        )
        return update

    def compliance(self, state: AgentState) -> dict[str, Any]:
        draft = dict(state.get("draft") or {})
        fallback_notes: list[str] = []
        llm_draft = state.get("draft_source") in {"llm_agent", "llm_compose"}
        if llm_draft and language_violation(
            str(draft.get("answer") or ""), state["query"], language="zh" if self._zh(state) else "en"
        ):
            # A poisoned document can hijack the output language; fall back to the deterministic answer.
            draft = self._template_update(state)["draft"]
            fallback_notes.append("language_mismatch_fallback_to_template")
            llm_draft = False
        limitations = list(draft.get("limitations") or [])
        if llm_draft:
            # The LLM usually says when a requested period or metric is missing; the limitation makes it explicit.
            limitations.extend(
                coverage_gaps(
                    state.get("effective_query") or state["query"], state.get("tool_log") or [], zh=self._zh(state)
                )
            )
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
            language="zh" if self._zh(state) else "en",
        )
        return {"answer": answer, "compliance_notes": [*fallback_notes, *notes]}

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
        cost, currency, cost_source = resolve_cost(usage_model, self.pricing)
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
                "cost": cost,
                "currency": currency,
                "cost_source": cost_source,
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
        # The result carries everything the response needs; drop the bulky turn-scoped working state so the
        # checkpoint stays small (it is reset at the start of the next turn anyway).
        return {
            "result": result,
            "turns": [turn_record(state, result)],
            "messages": [],
            "tool_log": {RESET: []},
            "llm_log": {RESET: []},
        }

    # ---------------------------------------------------------------- helpers

    def _template_update(self, state: AgentState, degraded: str | None = None) -> dict[str, Any]:
        nlu = state.get("nlu") or {}
        names = {
            str(entity["symbol"]): str(entity.get("canonical_name") or entity["symbol"])
            for entity in nlu.get("entities") or []
            if entity.get("symbol")
        }
        update: dict[str, Any] = {
            "draft": compose_template(
                state.get("tool_log") or [],
                zh=self._zh(state),
                question_style=str(nlu.get("question_style") or ""),
                query=state.get("effective_query") or state["query"],
                names=names,
            ),
            "draft_source": "template",
        }
        if degraded:
            update["degraded"] = [degraded]
        return update

    def _run_tools(self, calls: list[tuple[str, Any]]) -> list[ToolResult]:
        if not calls:
            return []
        results: list[ToolResult] = []
        width = max(1, self.config.max_parallel_tools)
        for start in range(0, len(calls), width):
            batch = calls[start : start + width]
            futures = [self._pool.submit(self.registry.run, name, arguments) for name, arguments in batch]
            results.extend(future.result() for future in futures)
        return results

    @staticmethod
    def _zh(state: AgentState) -> bool:
        return (state.get("language") or detect_user_language(state.get("query", ""))) == "zh"


def _context_composition(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, int]:
    """Characters sent per part of the request (system, user, assistant, tool results, tool schemas)."""
    import json

    parts = {"system": 0, "user": 0, "assistant": 0, "tool": 0, "tool_schemas": 0}
    for message in messages:
        role = str(message.get("role") or "user")
        size = len(str(message.get("content") or ""))
        if message.get("tool_calls"):
            size += len(json.dumps(message["tool_calls"], ensure_ascii=False))
        parts[role if role in parts else "user"] += size
    if tools:
        parts["tool_schemas"] = len(json.dumps(tools, ensure_ascii=False))
    return parts


def _answer_delta_callback() -> Callable[[str], None]:
    """Stream the draft answer text to SSE clients as ``answer_delta`` custom events (no-op when invoked)."""
    writer = stream_writer()
    return AnswerTextStream(lambda text: writer({"event": "answer_delta", "text": text})).feed


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
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return arguments
    return json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)


def _log_entry(result: ToolResult, *, source: str, reason: str, step: int, flagged: bool = False) -> dict[str, Any]:
    return {
        "tool": result.tool,
        "arguments": result.arguments,
        "ok": result.ok,
        # The template composer renders document titles from this data: redact it like evidence.
        "data": sanitize_observation(result.data)[0],
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


def _llm_entry(
    node: str,
    turn: Any,
    *,
    step: int | None = None,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    json_status: str | None = None,
) -> dict[str, Any]:
    return {
        "node": node,
        "step": step,
        "prompt": prompt,
        "model": turn.model,
        "started_at": round(time.time() - turn.latency_ms / 1000, 3),
        "latency_ms": turn.latency_ms,
        "prompt_tokens": turn.usage.prompt_tokens,
        "completion_tokens": turn.usage.completion_tokens,
        "prompt_cache_hit_tokens": turn.usage.prompt_cache_hit_tokens,
        "reasoning_tokens": turn.usage.reasoning_tokens,
        "reported_cost_usd": turn.usage.reported_cost_usd,
        "json_status": json_status,
        "context_chars": _context_composition(messages or [], tools),
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


def _memory_fields(memory_update: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    """State fields a memory summary produced, kept even when the following agent call fails."""
    if not memory_update.get("llm_calls"):
        return {}
    return {
        "memory_card": memory_update["memory_card"],
        "llm_log": memory_update["llm_log"],
        "usage": usage,
    }


def _add_usage(current: dict[str, Any] | None, extra: Usage) -> dict[str, Any]:
    total = Usage(**(current or {})) + extra
    return total.model_dump()


def _total_tokens(usage: dict[str, Any]) -> int:
    return int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))


def _feedback_text(verification: dict[str, Any]) -> str:
    from .verifier import VerificationReport

    return VerificationReport.model_validate(verification).feedback() if verification else ""


def _source_view(item: dict[str, Any]) -> dict[str, Any]:
    view = {
        key: item.get(key)
        for key in ("evidence_id", "kind", "source_type", "title", "source_name", "source_url", "as_of", "produced_by")
    }
    # Structured evidence carries the numbers the answer cites (and the price series the UI charts).
    # Document payloads are omitted: their text is already summarised by title/source and can be large.
    if item.get("kind") == "structured" and isinstance(item.get("payload"), dict):
        view["payload"] = item["payload"]
    return view
