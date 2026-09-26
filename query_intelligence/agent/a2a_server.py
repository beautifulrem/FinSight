"""Expose the FinSight agent over the A2A (Agent2Agent) protocol.

MCP (``mcp_server.py``) publishes the *tools* so another agent can call them one by one; A2A publishes
the *agent* itself so another agent can delegate a whole research task and get back a cited answer.

* Agent card: ``GET /.well-known/agent-card.json`` (skills, JSON-RPC interface, streaming support).
* JSON-RPC endpoint: ``POST /a2a`` (A2A protocol 1.0 methods such as ``SendMessage``, ``GetTask``).
* An A2A ``context_id`` maps to an agent session, so follow-up messages keep conversation memory.
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
            await updater.start_work()
            try:
                if task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED and agent.pending_clarification(session):
                    response = await asyncio.to_thread(agent.resume, session, query)
                else:
                    response = await asyncio.to_thread(agent.chat, query, session_id=session, mode=mode)
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

        async def cancel(self, context: Any, event_queue: Any) -> None:
            task = context.current_task
            if task is not None:
                await TaskUpdater(event_queue, task.id, task.context_id).cancel()

    return FinSightAgentExecutor()


def install_a2a(app: Any, get_agent: Callable[[], Any], *, base_url: str | None = None) -> bool:
    """Mount the agent card and JSON-RPC routes on ``app``. Returns ``False`` when A2A is unavailable."""
    if not a2a_enabled():
        return False
    try:
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.routes import add_a2a_routes_to_fastapi, create_jsonrpc_routes
        from a2a.server.tasks import InMemoryTaskStore
    except ImportError:
        logger.info("[startup] a2a-sdk is not installed; A2A endpoints are disabled.")
        return False

    api_key_required = bool(os.getenv("QI_API_KEYS", "").strip())
    configured_base = base_url or os.getenv("QI_PUBLIC_BASE_URL", "").strip()
    card = build_agent_card(configured_base or "http://127.0.0.1:8765", api_key_required=api_key_required)
    handler = DefaultRequestHandler(
        agent_executor=build_executor(get_agent, mode=os.getenv("QI_A2A_MODE", "auto")),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    add_a2a_routes_to_fastapi(app, jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=A2A_RPC_PATH))

    from a2a.server.request_handlers.response_helpers import agent_card_to_dict
    from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

    @app.get(AGENT_CARD_WELL_KNOWN_PATH, include_in_schema=True, tags=["A2A: Agent Card"])
    def agent_card(request: Request) -> JSONResponse:
        """The advertised JSON-RPC URL comes from QI_PUBLIC_BASE_URL or, when unset, from the request itself."""
        base = configured_base or str(request.base_url)
        return JSONResponse(agent_card_to_dict(build_agent_card(base, api_key_required=api_key_required)))

    return True
