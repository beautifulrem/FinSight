"""Expose the FinSight agent over the A2A (Agent2Agent) protocol.

MCP (``mcp_server.py``) publishes the *tools* so another agent can call them one by one; A2A publishes
the *agent* itself so another agent can delegate a whole research task and get back a cited answer.

* Agent card: ``GET /.well-known/agent-card.json`` (skills, JSON-RPC interface, streaming support).
* JSON-RPC endpoint: ``POST /a2a`` (A2A protocol 1.0 methods such as ``SendMessage``, ``GetTask``).
* An A2A ``context_id`` maps to an agent session, so follow-up messages keep conversation memory. Tasks and
  sessions are owned by the caller's API-key principal (``build_context_builder``).
* ``SendStreamingMessage`` streams ``working`` status updates for each graph node and tool call, then the
  answer artifacts.
* Tasks live in Postgres when ``QI_AGENT_CHECKPOINT_DB`` (or ``QI_A2A_TASK_DB``) is a ``postgresql://`` DSN,
  so any replica can serve ``GetTask`` and continue an ``input-required`` task (``a2a_store.py``).
* A clarification interrupt becomes an ``input-required`` task; the next message on the same task
  resumes the paused graph (``AgentService.resume``).
* The completed task carries two artifacts: the answer text (with ``[evidence_id]`` citations and the
  risk disclaimer) and a data part with evidence sources, verification, route and trace id.

``a2a-sdk`` is an optional dependency: ``install_a2a`` returns ``False`` and the rest of the API keeps
working when it is not installed or when ``QI_A2A_ENABLED=0``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

A2A_RPC_PATH = "/a2a"
_DATA_FIELDS = (
    "trace_id",
    "session_id",
    "route",
    "route_reasons",
    "evidence_used",
    "evidence_sources",
    "verification",
    "compliance_notes",
    "degraded",
    "limitations",
    "next_questions",
)


def a2a_enabled() -> bool:
    return os.getenv("QI_A2A_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def session_for_context(context_id: str | None) -> str | None:
    """A2A context ids are UUIDs; agent session ids are hex without dashes."""
    if not context_id:
        return None
    return "a2a" + "".join(ch for ch in context_id if ch.isalnum())[:60]


def answer_text(response: dict[str, Any]) -> str:
    parts = [str(response.get("answer") or "").strip()]
    points = [str(point) for point in response.get("key_points") or [] if str(point).strip()]
    if points:
        parts.append("\n".join(f"- {point}" for point in points))
    disclaimer = str(response.get("risk_disclaimer") or "").strip()
    if disclaimer:
        parts.append(disclaimer)
    return "\n\n".join(part for part in parts if part)


def answer_data(response: dict[str, Any]) -> dict[str, Any]:
    return {key: response.get(key) for key in _DATA_FIELDS if response.get(key) is not None}


def build_agent_card(base_url: str, *, api_key_required: bool = False) -> Any:
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentInterface,
        AgentProvider,
        AgentSkill,
        APIKeySecurityScheme,
        HTTPAuthSecurityScheme,
        SecurityRequirement,
        SecurityScheme,
        StringList,
    )

    security: dict[str, Any] = {}
    if api_key_required:
        # Mirrors api/security.py: an X-API-Key header or an Authorization: Bearer token.
        security = {
            "security_schemes": {
                "apiKey": SecurityScheme(
                    api_key_security_scheme=APIKeySecurityScheme(location="header", name="X-API-Key")
                ),
                "bearer": SecurityScheme(http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="bearer")),
            },
            "security_requirements": [
                SecurityRequirement(schemes={"apiKey": StringList(list=[])}),
                SecurityRequirement(schemes={"bearer": StringList(list=[])}),
            ],
        }
    return AgentCard(
        **security,
        name="FinSight",
        description=(
            "Evidence-first China A-share research agent. Classical NLU routes and guards each question; "
            "tools fetch market, fundamental, macro, news and announcement evidence; every number in the "
            "answer is verified against tool output and cited by evidence id. Never gives buy/sell instructions."
        ),
        version="1.0.0",
        provider=AgentProvider(organization="FinSight", url="https://github.com/beautifulrem/FinSight"),
        documentation_url="https://github.com/beautifulrem/FinSight/blob/master/docs/agent.md",
        supported_interfaces=[
            AgentInterface(
                url=f"{base_url.rstrip('/')}{A2A_RPC_PATH}", protocol_binding="JSONRPC", protocol_version="1.0"
            )
        ],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="equity_research",
                name="A-share equity research",
                description=(
                    "Price, valuation, fundamentals, news and announcements for listed A-shares, ETFs and indices."
                ),
                tags=["a-share", "stocks", "fundamentals", "evidence"],
                examples=["贵州茅台最近走势怎么样？", "What is CATL's latest PE ratio?"],
            ),
            AgentSkill(
                id="comparison",
                name="Company comparison",
                description="Compare two or more listed companies on the same metrics, citing each figure.",
                tags=["comparison", "valuation"],
                examples=["比较宁德时代和比亚迪的市盈率"],
            ),
            AgentSkill(
                id="macro_linkage",
                name="Macro-to-market analysis",
                description="Relate macro indicators (CPI, PMI, M2, LPR) to sectors, with hedged causal language.",
                tags=["macro", "sector"],
                examples=["CPI 对消费板块有什么影响？"],
            ),
        ],
    )


def principal_of_context(context: Any) -> str:
    """The FinSight principal (API-key hash or ``local``) that ``PrincipalContextBuilder`` put on the call."""
    call_context = getattr(context, "call_context", None)
    name = getattr(getattr(call_context, "user", None), "user_name", "") if call_context is not None else ""
    return name or "local"


def progress_text(event: dict[str, Any]) -> str | None:
    """A short progress line for a streamed agent event (node starts and tool calls), else ``None``."""
    data = event.get("data") or {}
    if event.get("event") == "node_start":
        return f"{data.get('label') or data.get('node')}…"
    if event.get("event") == "tool_call":
        return f"calling {data.get('tool')}"
    return None


def build_executor(get_agent: Callable[[], Any], *, mode: str = "auto") -> Any:
    from a2a.helpers import new_data_part, new_task_from_user_message, new_text_part
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.tasks import TaskUpdater
    from a2a.types import TaskState

    class FinSightAgentExecutor(AgentExecutor):
        async def execute(self, context: Any, event_queue: Any) -> None:
            task = context.current_task
            if task is None:
                task = new_task_from_user_message(context.message)
                await event_queue.enqueue_event(task)
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            query = context.get_user_input().strip()
            if not query:
                await updater.reject(updater.new_agent_message([new_text_part("Empty message.")]))
                return
            agent = get_agent()
            session = session_for_context(task.context_id)
            owner = principal_of_context(context)
            await updater.start_work()
            try:
                if task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED and agent.pending_clarification(session):
                    response = await asyncio.to_thread(agent.resume, session, query, owner=owner)
                else:
                    response = await self._run_streamed(agent, updater, query, session, owner)
            except Exception as exc:  # surfaced to the A2A client as a failed task
                logger.exception("A2A task failed")
                await updater.failed(updater.new_agent_message([new_text_part(f"{type(exc).__name__}: {exc}")]))
                return

            if response.get("status") == "needs_clarification":
                clarification = response.get("clarification") or {}
                question = str(clarification.get("question") or "Please clarify your question.")
                parts = [new_text_part(question), new_data_part({"clarification": clarification})]
                await updater.requires_input(updater.new_agent_message(parts))
                return

            await updater.add_artifact([new_text_part(answer_text(response))], name="answer")
            await updater.add_artifact([new_data_part(answer_data(response))], name="evidence")
            await updater.complete()

        async def _run_streamed(self, agent: Any, updater: Any, query: str, session: str | None, owner: str) -> dict:
            """Run the agent through ``AgentService.stream`` and forward node starts and tool calls as
            ``working`` status updates, so ``SendStreamingMessage`` clients see progress before the answer."""
            events = agent.stream(query, session_id=session, mode=mode, owner=owner)
            response: dict[str, Any] | None = None
            while (event := await asyncio.to_thread(next, events, None)) is not None:
                kind = event.get("event")
                text = progress_text(event)
                if text:
                    await updater.update_status(
                        TaskState.TASK_STATE_WORKING, updater.new_agent_message([new_text_part(text)])
                    )
                elif kind == "answer":
                    response = dict(event.get("data") or {})
                elif kind == "clarification":
                    data = dict(event.get("data") or {})
                    data.pop("session_id", None)
                    response = {"status": "needs_clarification", "clarification": data}
                elif kind == "error":
                    raise RuntimeError(str((event.get("data") or {}).get("message") or "agent run failed"))
            if response is None:
                raise RuntimeError("agent run ended without an answer")
            return response

        async def cancel(self, context: Any, event_queue: Any) -> None:
            task = context.current_task
            if task is not None:
                await TaskUpdater(event_queue, task.id, task.context_id).cancel()

    return FinSightAgentExecutor()


def build_context_builder() -> Any:
    """Puts the security middleware's principal on every A2A call, so tasks and agent sessions are scoped to
    the calling API key (the SDK's default builder only knows Starlette auth users)."""
    from a2a.auth.user import User
    from a2a.server.routes import DefaultServerCallContextBuilder

    class PrincipalUser(User):
        def __init__(self, principal: str) -> None:
            self._principal = principal

        @property
        def is_authenticated(self) -> bool:
            return self._principal != "local"

        @property
        def user_name(self) -> str:
            return self._principal

    class PrincipalContextBuilder(DefaultServerCallContextBuilder):
        def build_user(self, request: Any) -> User:
            return PrincipalUser(str(getattr(request.state, "principal", "local") or "local"))

    return PrincipalContextBuilder()


def build_task_store() -> Any:
    """Postgres when ``QI_A2A_TASK_DB`` (or, by default, ``QI_AGENT_CHECKPOINT_DB``) is a DSN, else in memory.
    An unreachable database falls back to memory with a warning, so the API still starts."""
    from a2a.server.tasks import InMemoryTaskStore

    from .pg import open_pool, store_dsn

    dsn = store_dsn("QI_A2A_TASK_DB")
    if dsn:
        try:
            from .a2a_store import PostgresTaskStore

            store = PostgresTaskStore(open_pool(dsn, name="a2a-tasks"))
            logger.info("[startup] A2A tasks are stored in Postgres (shared by all replicas).")
            return store
        except Exception as exc:
            logger.warning("[startup] A2A Postgres task store unavailable (%s); using the in-memory store.", exc)
    return InMemoryTaskStore()


def build_request_handler(executor: Any, task_store: Any, card: Any) -> Any:
    """The SDK's ``DefaultRequestHandler`` with spec-conformant ``SubscribeToTask`` errors.

    A2A 1.0 (§3.1.6, §9.4.6) requires ``UnsupportedOperationError`` when a client subscribes to a task in a
    terminal state. ``a2a-sdk`` 1.1's handler raises ``InvalidParamsError`` (-32602) instead, which clients
    such as ``@a2a-js/sdk`` surface as a malformed request rather than "this task has finished; call GetTask"
    (found by ``tools/a2a-js-client/interop.mjs``). The terminal check runs before the SDK's own, and the SDK's
    error for a task that finishes between the check and the subscription is mapped the same way.
    """
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.types import TaskState
    from a2a.utils.errors import InvalidParamsError, TaskNotFoundError, UnsupportedOperationError

    terminal = {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }

    def finished(task_id: str, state: Any = None) -> UnsupportedOperationError:
        suffix = f" ({TaskState.Name(state)})" if state is not None else ""
        return UnsupportedOperationError(
            message=f"Task {task_id} is in a terminal state{suffix}; SubscribeToTask only streams running "
            "tasks. Use GetTask to read the result."
        )

    class FinSightRequestHandler(DefaultRequestHandler):
        async def on_subscribe_to_task(self, params: Any, context: Any) -> Any:
            task = await self.task_store.get(params.id, context)  # owner-scoped: another caller's task is "not found"
            if task is None:
                raise TaskNotFoundError
            if task.status.state in terminal:
                raise finished(task.id, task.status.state)
            try:
                async for event in super().on_subscribe_to_task(params, context):
                    yield event
            except InvalidParamsError as exc:
                if "terminal state" in str(exc) or "already completed" in str(exc):
                    raise finished(params.id) from exc
                raise

    return FinSightRequestHandler(agent_executor=executor, task_store=task_store, agent_card=card)


def install_a2a(app: Any, get_agent: Callable[[], Any], *, base_url: str | None = None, task_store: Any = None) -> bool:
    """Mount the agent card and JSON-RPC routes on ``app``. Returns ``False`` when A2A is unavailable."""
    if not a2a_enabled():
        return False
    try:
        from a2a.server.routes import add_a2a_routes_to_fastapi, create_jsonrpc_routes
    except ImportError:
        logger.info("[startup] a2a-sdk is not installed; A2A endpoints are disabled.")
        return False

    api_key_required = bool(os.getenv("QI_API_KEYS", "").strip())
    configured_base = base_url or os.getenv("QI_PUBLIC_BASE_URL", "").strip()
    card = build_agent_card(configured_base or "http://127.0.0.1:8765", api_key_required=api_key_required)
    store = task_store if task_store is not None else build_task_store()
    app.state.a2a_task_store = store
    handler = build_request_handler(build_executor(get_agent, mode=os.getenv("QI_A2A_MODE", "auto")), store, card)
    routes = create_jsonrpc_routes(handler, rpc_url=A2A_RPC_PATH, context_builder=build_context_builder())
    add_a2a_routes_to_fastapi(app, jsonrpc_routes=routes)

    from a2a.server.request_handlers.response_helpers import agent_card_to_dict
    from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

    @app.get(AGENT_CARD_WELL_KNOWN_PATH, include_in_schema=True, tags=["A2A: Agent Card"])
    def agent_card(request: Request) -> JSONResponse:
        """The advertised JSON-RPC URL comes from QI_PUBLIC_BASE_URL or, when unset, from the request itself."""
        base = configured_base or str(request.base_url)
        return JSONResponse(agent_card_to_dict(build_agent_card(base, api_key_required=api_key_required)))

    return True
