"""Session-aware agent service: chat, resume after clarification, and streamed step events."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Iterator
from datetime import date
from typing import TYPE_CHECKING, Any

from langgraph.types import Command

from .graph import AgentRuntime
from .llm import DeepSeekToolClient, LLMClient, Pricing
from .memory import make_checkpointer
from .state import AgentConfig
from .tools import ToolRegistry, build_registry_for_service

if TYPE_CHECKING:
    from ..service import QueryIntelligenceService

logger = logging.getLogger(__name__)

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
    def __init__(self, runtime: AgentRuntime, *, checkpointer: Any = None) -> None:
        self.runtime = runtime
        self.checkpointer = checkpointer if checkpointer is not None else make_checkpointer()
        self.graph = runtime.build_graph(self.checkpointer)
        self._locks: dict[str, threading.Lock] = {}
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
    ) -> AgentService:
        if llm is None and chatbot_config is not None:
            candidate = DeepSeekToolClient.from_chatbot_config(chatbot_config)
            llm = candidate if candidate.configured else None
        runtime = AgentRuntime(
            service,
            registry or build_registry_for_service(service),
            llm,
            config=config,
            pricing=Pricing.from_env(),
            today=date.today,
        )
        return cls(runtime, checkpointer=checkpointer)

    # ------------------------------------------------------------------ public API

    def chat(
        self,
        query: str,
        *,
        session_id: str | None = None,
        mode: str = "auto",
        user_profile: dict[str, Any] | None = None,
        dialog_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        session = session_id or uuid.uuid4().hex
        state = self.runtime.initial_state(query, mode=mode, user_profile=user_profile, dialog_context=dialog_context)
        with self._lock(session):
            output = self.graph.invoke(state, self._config(session))
        return self._response(session, output)

    def resume(self, session_id: str, reply: str) -> dict[str, Any]:
        with self._lock(session_id):
            if not self.pending_clarification(session_id):
                raise ValueError(f"session {session_id} has no pending clarification")
            output = self.graph.invoke(Command(resume=reply), self._config(session_id))
        return self._response(session_id, output)

    def stream(
        self,
        query: str,
        *,
        session_id: str | None = None,
        mode: str = "auto",
        user_profile: dict[str, Any] | None = None,
        dialog_context: list[dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield events: session, step, tool_call, tool_result, clarification, answer, error, done."""
        session = session_id or uuid.uuid4().hex
        state = self.runtime.initial_state(query, mode=mode, user_profile=user_profile, dialog_context=dialog_context)
        yield {"event": "session", "data": {"session_id": session}}
        with self._lock(session):
            try:
                for update in self.graph.stream(state, self._config(session), stream_mode="updates"):
                    yield from self._events(session, update)
            except Exception as exc:  # surfaced to the client as an error event
                logger.exception("agent stream failed")
                yield {"event": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}}
        yield {"event": "done", "data": {"session_id": session}}

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
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def _response(self, session_id: str, output: dict[str, Any]) -> dict[str, Any]:
        interrupts = output.get("__interrupt__") or []
        if interrupts:
            value = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
            payload = dict(value) if isinstance(value, dict) else {"question": str(value)}
            return {"status": "needs_clarification", "session_id": session_id, "clarification": payload}
        return {"status": "ok", "session_id": session_id, **(output.get("result") or {})}

    def _events(self, session_id: str, update: dict[str, Any]) -> Iterator[dict[str, Any]]:
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
                yield {"event": "answer", "data": {"status": "ok", "session_id": session_id, **payload["result"]}}
