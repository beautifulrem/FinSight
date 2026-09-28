"""Session-aware agent service: chat, resume after clarification, and streamed step events."""

from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from datetime import date
from typing import TYPE_CHECKING, Any

from langgraph.types import Command

from .errors import NoPendingClarificationError, SessionAccessError
from .graph import AgentRuntime
from .llm import LLMClient, Pricing, build_llm_from_config
from .memory import clarification_reply_text, is_target_only_reply, make_checkpointer
from .state import AgentConfig
from .tools import ToolRegistry, build_registry_for_service
from .tools.mcp_client import register_configured_mcp_servers
from .tracing import TraceSink, build_trace, emit, sinks_from_env

if TYPE_CHECKING:
    from ..service import QueryIntelligenceService

logger = logging.getLogger(__name__)

MAX_SESSION_LOCKS = 4096


_STEP_LABELS = {
    "guard_in": "NLU and routing",
    "refuse": "out-of-scope guard",
    "clarify": "clarification",
    "execute_plan": "deterministic tool plan",
    "compose": "answer composition",
    "agent_llm": "LLM reasoning",
    "agent_tools": "tool execution",
    "verify": "evidence verification",
    "revise": "answer revision",
    "compliance": "compliance guard",
    "finalize": "finalize",
}


class AgentService:
    def __init__(
        self, runtime: AgentRuntime, *, checkpointer: Any = None, trace_sinks: list[TraceSink] | None = None
    ) -> None:
        self.runtime = runtime
        self.trace_sinks = trace_sinks if trace_sinks is not None else sinks_from_env()
        self.checkpointer = checkpointer if checkpointer is not None else make_checkpointer()
        # "exit" writes one checkpoint per run (and at interrupts) instead of one per node: runs are short, and
        # a crash mid-run only loses that turn. QI_AGENT_DURABILITY=async|sync restores per-step checkpoints.
        self.durability = os.getenv("QI_AGENT_DURABILITY", "exit").strip() or "exit"
        self.graph = runtime.build_graph(self.checkpointer)
        self._locks: OrderedDict[str, threading.Lock] = OrderedDict()
        self._locks_guard = threading.Lock()

    @classmethod
    def from_service(
        cls,
        service: QueryIntelligenceService,
        *,
        chatbot_config: dict[str, Any] | None = None,
        llm: LLMClient | None = None,
        registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        checkpointer: Any = None,
        trace_sinks: list[TraceSink] | None = None,
    ) -> AgentService:
        if llm is None and chatbot_config is not None:
            llm = build_llm_from_config(chatbot_config)
        if registry is None:
            registry = build_registry_for_service(service)
            # External MCP servers (QI_MCP_SERVERS, off by default) add namespaced tools for the LLM agent.
            register_configured_mcp_servers(registry)
        runtime = AgentRuntime(
            service,
            registry,
            llm,
            config=config,
            pricing=Pricing.from_env(),
            today=date.today,
        )
        return cls(runtime, checkpointer=checkpointer, trace_sinks=trace_sinks)

    # ------------------------------------------------------------------ public API

    def chat(
        self,
        query: str,
        *,
        session_id: str | None = None,
        mode: str = "auto",
        user_profile: dict[str, Any] | None = None,
        dialog_context: list[dict[str, Any]] | None = None,
        owner: str = "local",
    ) -> dict[str, Any]:
        session = session_id or uuid.uuid4().hex
        state = self.runtime.initial_state(query, mode=mode, user_profile=user_profile, dialog_context=dialog_context)
        state["owner"] = owner
        with self._lock(session):
            self._check_owner(session, owner)
            if self._answers_pending_clarification(session, query):
                # "五粮液" typed into the chat box after "这个能买吗？" got "which stock?": finish the paused turn.
                output = self.graph.invoke(
                    Command(resume=query.strip()), self._config(session), durability=self.durability
                )
            else:
                output = self.graph.invoke(state, self._config(session), durability=self.durability)
        return self._response(session, output, owner)

    def _answers_pending_clarification(self, session_id: str, message: str) -> bool:
        """A chat message that only names a target, sent while a clarification is pending, is its answer."""
        if not self.pending_clarification(session_id):
            return False
        try:
            nlu = self.runtime.service.analyze_query(clarification_reply_text(message))
        except Exception:  # the NLU is best effort here; a failure just starts a new turn
            logger.exception("clarification reply check failed")
            return False
        return is_target_only_reply(message, nlu)

    def resume(self, session_id: str, reply: str, *, owner: str = "local") -> dict[str, Any]:
        """Answer a pending clarification and finish the paused turn.

        Idempotent: a repeated submission of the same reply (a double click, a client retry after a timeout)
        does not run the turn again. It returns the stored result of the turn that reply completed, marked
        ``replayed: true``, and emits no second trace. A different reply once nothing is pending raises
        ``NoPendingClarificationError`` (409 ``no_pending_clarification``). The session lock serialises
        concurrent submissions, so the second one always sees the first one's outcome.
        """
        reply = reply.strip()
        with self._lock(session_id):
            self._check_owner(session_id, owner)
            if not self.pending_clarification(session_id):
                replay = self._replayed_resume(session_id, reply)
                if replay is None:
                    raise NoPendingClarificationError(session_id)
                return replay
            output = self.graph.invoke(Command(resume=reply), self._config(session_id), durability=self.durability)
        return self._response(session_id, output, owner)

    def _replayed_resume(self, session_id: str, reply: str) -> dict[str, Any] | None:
        """The result of the last turn when it was completed by this same clarification reply."""
        values = self.graph.get_state(self._config(session_id)).values or {}
        result = values.get("result") or {}
        if not reply or not result or str(values.get("clarification_reply") or "").strip() != reply:
            return None
        return {"status": "ok", "session_id": session_id, "trace_id": result.get("run_id"), **result, "replayed": True}

    def stream(
        self,
        query: str,
        *,
        session_id: str | None = None,
        mode: str = "auto",
        user_profile: dict[str, Any] | None = None,
        dialog_context: list[dict[str, Any]] | None = None,
        owner: str = "local",
    ) -> Iterator[dict[str, Any]]:
        """Yield events: session, node_start, step, tool_call, tool_result, answer_delta, clarification, answer,
        error, done. ``answer_delta`` streams the draft answer text; the final ``answer`` event supersedes it.

        The graph runs on a worker thread that owns the session lock and pushes events into a queue. If
        the client disconnects and this generator is closed, the run still finishes (its state and
        trace are saved) and the lock is released, so the session stays usable.
        """
        session = session_id or uuid.uuid4().hex
        self._check_owner(session, owner)
        state = self.runtime.initial_state(query, mode=mode, user_profile=user_profile, dialog_context=dialog_context)
        state["owner"] = owner
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def run() -> None:
            with self._lock(session):
                try:
                    graph_input: Any = state
                    if self._answers_pending_clarification(session, query):
                        graph_input = Command(resume=query.strip())
                    for chunk in self.graph.stream(
                        graph_input,
                        self._config(session),
                        stream_mode=["updates", "tasks", "custom"],
                        version="v2",
                        durability=self.durability,
                    ):
                        for event in self._chunk_events(session, chunk, owner):
                            events.put(event)
                except Exception as exc:  # surfaced to the client as an error event
                    logger.exception("agent stream failed")
                    events.put({"event": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}})
                finally:
                    events.put(None)

        yield {"event": "session", "data": {"session_id": session}}
        threading.Thread(target=run, name=f"agent-stream-{session[:8]}", daemon=True).start()
        while (event := events.get()) is not None:
            yield event
        yield {"event": "done", "data": {"session_id": session}}

    def _chunk_events(self, session_id: str, chunk: dict[str, Any], owner: str = "local") -> Iterator[dict[str, Any]]:
        data = chunk.get("data") or {}
        if chunk.get("type") == "tasks":
            # Task-start events (they carry "input") announce a node before it runs.
            if "input" in data and data.get("name") in _STEP_LABELS:
                name = data["name"]
                yield {"event": "node_start", "data": {"node": name, "label": _STEP_LABELS[name]}}
            return
        if chunk.get("type") == "custom" and isinstance(data, dict) and data.get("event") == "answer_delta":
            yield {"event": "answer_delta", "data": {"text": data.get("text", "")}}
            return
        if chunk.get("type") == "updates":
            yield from self._events(session_id, data, owner)

    def owner_of(self, session_id: str) -> str | None:
        snapshot = self.graph.get_state(self._config(session_id))
        return (snapshot.values or {}).get("owner")

    def _check_owner(self, session_id: str, owner: str) -> None:
        existing = self.owner_of(session_id)
        if existing and existing != owner:
            raise SessionAccessError(session_id)

    def pending_clarification(self, session_id: str) -> dict[str, Any] | None:
        snapshot = self.graph.get_state(self._config(session_id))
        for task in snapshot.tasks or ():
            for pending in task.interrupts or ():
                return dict(pending.value) if isinstance(pending.value, dict) else {"question": str(pending.value)}
        return None

    def history(self, session_id: str) -> list[dict[str, Any]]:
        snapshot = self.graph.get_state(self._config(session_id))
        return list((snapshot.values or {}).get("turns") or [])

    def close(self) -> None:
        self.runtime.close()

    # ------------------------------------------------------------------ helpers

    def _config(self, session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": 60}

    def _lock(self, session_id: str) -> threading.Lock:
        """Per-session lock (turns of one session run one at a time). The map is an LRU capped at
        ``MAX_SESSION_LOCKS``; only locks that are not held are evicted."""
        with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = self._locks[session_id] = threading.Lock()
            self._locks.move_to_end(session_id)
            if len(self._locks) > MAX_SESSION_LOCKS:
                for key in list(self._locks)[: len(self._locks) - MAX_SESSION_LOCKS]:
                    if key != session_id and not self._locks[key].locked():
                        del self._locks[key]
            return lock

    def _response(self, session_id: str, output: dict[str, Any], owner: str = "local") -> dict[str, Any]:
        interrupts = output.get("__interrupt__") or []
        if interrupts:
            value = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
            payload = dict(value) if isinstance(value, dict) else {"question": str(value)}
            return {"status": "needs_clarification", "session_id": session_id, "clarification": payload}
        result = output.get("result") or {}
        self._trace(session_id, result, owner)
        return {"status": "ok", "session_id": session_id, "trace_id": result.get("run_id"), **result}

    def _trace(self, session_id: str, result: dict[str, Any], owner: str = "local") -> None:
        if result and self.trace_sinks:
            emit(self.trace_sinks, {**build_trace(result, session_id=session_id), "owner": owner})

    def _events(self, session_id: str, update: dict[str, Any], owner: str = "local") -> Iterator[dict[str, Any]]:
        for node, payload in update.items():
            if node == "__interrupt__":
                value = payload[0].value if payload and hasattr(payload[0], "value") else payload
                yield {
                    "event": "clarification",
                    "data": {
                        "session_id": session_id,
                        **(value if isinstance(value, dict) else {"question": str(value)}),
                    },
                }
                continue
            payload = payload or {}
            yield {"event": "step", "data": {"node": node, "label": _STEP_LABELS.get(node, node)}}
            if node == "agent_llm":
                for message in (payload.get("messages") or [])[-1:]:
                    for call in message.get("tool_calls") or []:
                        yield {
                            "event": "tool_call",
                            "data": {"tool": call["function"]["name"], "arguments": call["function"].get("arguments")},
                        }
            if node == "execute_plan":
                for entry in payload.get("tool_log") or []:
                    yield {"event": "tool_call", "data": {"tool": entry["tool"], "arguments": entry["arguments"]}}
            for entry in payload.get("tool_log") or [] if isinstance(payload.get("tool_log"), list) else []:
                yield {
                    "event": "tool_result",
                    "data": {
                        "tool": entry["tool"],
                        "ok": entry["ok"],
                        "latency_ms": entry["latency_ms"],
                        "evidence_ids": entry["evidence_ids"],
                        "error": entry.get("error"),
                    },
                }
            if node == "finalize" and payload.get("result"):
                result = payload["result"]
                self._trace(session_id, result, owner)
                yield {
                    "event": "answer",
                    "data": {"status": "ok", "session_id": session_id, "trace_id": result.get("run_id"), **result},
                }
