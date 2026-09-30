from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as ApiPath
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ..agent.a2a_server import install_a2a
from ..agent.audit import AuditTraceSink
from ..agent.errors import NoPendingClarificationError, SessionAccessError
from ..agent.telemetry import PrometheusTraceSink, prompt_version_of
from ..agent.trace_store import build_trace_store
from ..agent.tracing import DEFAULT_TRACE_DIR, sinks_from_env
from ..artifacts import ArtifactWriter
from ..chat.page import DIST_DIR
from ..chatbot import (
    STATIC_DIR,
    DeepSeekClient,
    apply_live_data_env,
    build_chatbot_response,
    load_chatbot_config,
    render_index_html,
)
from ..contracts import (
    MAX_DIALOG_CONTEXT_ITEMS,
    MAX_QUERY_LENGTH,
    MAX_RETRIEVAL_TOP_K,
    MAX_USER_PROFILE_FIELDS,
    MIN_RETRIEVAL_TOP_K,
    SESSION_ID_PATTERN,
    AgentChatRequest,
    AgentMode,
    AgentResumeRequest,
    AnalyzeRequest,
    ArtifactRequest,
    ArtifactResponse,
    PipelineRequest,
    PipelineResponse,
    RetrievalRequest,
)
from ..service import QueryIntelligenceService, build_default_service
from .readiness import ReadinessChecker, check_checkpointer, check_model_config, check_retrieval_index
from .security import SecuritySettings, install_security, principal_of

logger = logging.getLogger("finsight.api")


def _ensure_console_logging() -> None:
    """Keep the launcher's console progress output when no logging is configured."""
    base = logging.getLogger("finsight")
    if base.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    base.addHandler(handler)
    base.setLevel(logging.INFO)
    base.propagate = False


def _elapsed_seconds(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.1f}s"


def _short_query(query: str, *, limit: int = 80) -> str:
    return query if len(query) <= limit else f"{query[: limit - 3]}..."


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    user_profile: dict[str, Any] = Field(default_factory=dict, max_length=MAX_USER_PROFILE_FIELDS)
    dialog_context: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_DIALOG_CONTEXT_ITEMS)
    top_k: int = Field(default=20, ge=MIN_RETRIEVAL_TOP_K, le=MAX_RETRIEVAL_TOP_K)
    debug: bool = False
    # "workflow" keeps the original /chat pipeline; "agent" and "auto" use the agent service.
    mode: AgentMode = "workflow"
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)


TRACE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,80}$"


class ClaimCheckRequest(BaseModel):
    claim: str = Field(min_length=2, max_length=MAX_QUERY_LENGTH)
    language: Literal["zh", "en"] | None = Field(default=None, description="Language of notes and disclaimer.")

    @field_validator("claim")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if len(value.strip()) < 2:
            raise ValueError("claim must contain at least 2 non-space characters")
        return value


class FeedbackRequest(BaseModel):
    trace_id: str = Field(pattern=TRACE_ID_PATTERN)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


def _session_not_found(exc: Exception) -> HTTPException:
    # 404 rather than 403, so session ids of other callers cannot be probed.
    return HTTPException(status_code=404, detail=f"session {exc} not found")


def _sse(event: dict[str, Any]) -> str:
    data = json.dumps(event["data"], ensure_ascii=False, default=str)
    return f"event: {event['event']}\ndata: {data}\n\n"


def create_app(
    service: QueryIntelligenceService | None = None,
    artifact_output_dir: str | Path | None = None,
    app_config: dict[str, Any] | None = None,
    app_config_path: str | Path | None = None,
    deepseek_client: DeepSeekClient | None = None,
    agent_service: Any = None,
    security: SecuritySettings | None = None,
) -> FastAPI:
    chatbot_config = app_config or load_chatbot_config(app_config_path, load_env_file=False)
    if service is None:
        apply_live_data_env(chatbot_config)

    _ensure_console_logging()
    app = FastAPI(title="Query Intelligence Service", version="0.1.0")
    if DIST_DIR.is_dir():
        # Built React UI (frontend/ → web/dist); mounted before /static so its prefix wins.
        app.mount("/static/app", StaticFiles(directory=DIST_DIR), name="static-app")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    security_settings = install_security(app, security)
    if security_settings.api_keys:
        logger.info("[startup] API key authentication enabled for %d key(s).", len(security_settings.api_keys))
    elif security_settings.production:
        logger.warning("[startup] QI_ALLOW_ANONYMOUS=1: production profile without API keys (anonymous callers).")
    if (
        str((chatbot_config.get("deepseek") or {}).get("api_key") or "").strip()
        and os.getenv("DEEPSEEK_API_KEY") is None
    ):
        logger.warning("[startup] An LLM API key is stored in the config file; prefer the DEEPSEEK_API_KEY variable.")
    if service is None:
        service_started_at = time.perf_counter()
        logger.info("[startup] Loading default Query Intelligence service...")
        runtime = build_default_service()
        logger.info("[startup] Query Intelligence service loaded in %s.", _elapsed_seconds(service_started_at))
    else:
        runtime = service
    artifact_writer = ArtifactWriter(
        artifact_output_dir or os.getenv("QI_API_OUTPUT_DIR", "outputs/query_intelligence")
    )
    logger.info("[startup] Preparing DeepSeek response client...")
    response_client = deepseek_client or DeepSeekClient(chatbot_config)
    agent_holder: dict[str, Any] = {"service": agent_service}
    agent_lock = threading.Lock()
    trace_dir = os.getenv("QI_AGENT_TRACE_DIR", DEFAULT_TRACE_DIR).strip()
    # In-process ring by default; a Postgres table shared by all replicas when QI_AGENT_TRACE_DB (or a Postgres
    # QI_AGENT_CHECKPOINT_DB) is set. Same interface, owner-scoped either way.
    trace_store = build_trace_store(None if trace_dir.lower() in {"", "off", "0", "false", "none"} else trace_dir)
    metrics_sink = PrometheusTraceSink()
    # Refusals and compliance edits: JSON log line, rotated JSONL file and finsight_audit_events_total.
    audit_sink = AuditTraceSink(registry=metrics_sink.registry)
    feedback_log = Path(os.getenv("QI_FEEDBACK_PATH", "outputs/feedback/feedback.jsonl"))
    feedback_lock = threading.Lock()
    if agent_service is not None and isinstance(getattr(agent_service, "trace_sinks", None), list):
        agent_service.trace_sinks.extend([trace_store, metrics_sink, audit_sink])

    def get_agent():
        # Built lazily: the agent layer is only constructed when an agent endpoint is used.
        with agent_lock:
            if agent_holder["service"] is None:
                from ..agent.service import AgentService

                agent_holder["service"] = AgentService.from_service(
                    runtime,
                    chatbot_config=chatbot_config,
                    trace_sinks=[*sinks_from_env(), trace_store, metrics_sink, audit_sink],
                )
            return agent_holder["service"]

    if metrics_sink.available:
        # Scrape-time state (source/LLM breakers, source-call pool) next to the trace-fed metrics.
        from ..integrations.ops_metrics import OpsMetricsCollector
        from ..integrations.sources.report import runtime_for

        metrics_sink.registry.register(
            OpsMetricsCollector(
                lambda: runtime_for(getattr(runtime, "retrieval_pipeline", None)),
                lambda: getattr(getattr(agent_holder["service"], "runtime", None), "llm", None),
            )
        )

    logger.info("[startup] FastAPI routes are ready.")

    readiness = ReadinessChecker(
        {
            "checkpointer": lambda: check_checkpointer(get_agent),
            "model_config": lambda: check_model_config(chatbot_config, lambda: agent_holder["service"]),
            "retrieval_index": lambda: check_retrieval_index(runtime),
        }
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        # Liveness only: the process serves HTTP. Dependencies are checked by /ready.
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        # Readiness: checkpoint store reachable and writable, model config sane, retrieval index loaded.
        ok, report = readiness.run()
        return JSONResponse(report, status_code=200 if ok else 503)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_index_html(chatbot_config)

    @app.post("/chat")
    def chat(payload: ChatRequest, request: Request) -> dict:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        if payload.mode != "workflow":
            logger.info("[chat] Routing query to the agent (mode=%s): %s", payload.mode, _short_query(query))
            try:
                return get_agent().chat(
                    query,
                    session_id=payload.session_id,
                    mode=payload.mode,
                    user_profile=payload.user_profile,
                    dialog_context=payload.dialog_context,
                    owner=principal_of(request),
                )
            except SessionAccessError as exc:
                raise _session_not_found(exc) from exc
        request_started_at = time.perf_counter()
        logger.info("[chat] Received query: %s", _short_query(query))
        logger.info("[chat] Step 1/3: running NLU, retrieval, and live data providers...")
        pipeline_started_at = time.perf_counter()
        result = runtime.run_pipeline(
            query,
            user_profile=payload.user_profile,
            dialog_context=payload.dialog_context,
            top_k=payload.top_k,
            debug=payload.debug,
        )
        retrieval = result.get("retrieval_result") or {}
        logger.info(
            "[chat] Step 1/3 complete (%s): documents=%d, structured_data=%d, warnings=%d",
            _elapsed_seconds(pipeline_started_at),
            len(retrieval.get("documents") or []),
            len(retrieval.get("structured_data") or []),
            len(retrieval.get("warnings") or []),
        )
        logger.info("[chat] Step 2/3: calling DeepSeek or fallback answer generator...")
        answer_started_at = time.perf_counter()
        response = build_chatbot_response(
            query=query,
            pipeline_result=result,
            deepseek_client=response_client,
            progress=lambda message: logger.info("[chat] %s", message),
        )
        llm_status = response.get("llm") or {}
        logger.info(
            "[chat] Step 2/3 complete (%s): llm_status=%s",
            _elapsed_seconds(answer_started_at),
            llm_status.get("status", "unknown"),
        )
        _add_english_names(response.get("nlu_result"))
        fact_check = _inline_fact_check(query)
        if fact_check:
            from ..agent.hearsay import fact_check_prose
            from ..chatbot import detect_query_language

            response["fact_check"] = fact_check
            # The answer text names the claimed number and the actual one, not only the card.
            prose = fact_check_prose(fact_check, zh=detect_query_language(query) == "zh")
            if prose and isinstance(response.get("answer"), str):
                response["answer"] = f"{prose}\n\n{response['answer']}" if response["answer"] else prose
        logger.info("[chat] Completed request in %s", _elapsed_seconds(request_started_at))
        return response

    def _add_english_names(nlu: Any) -> None:
        """The English UI shows "Kweichow Moutai" for 贵州茅台: ``name_en`` from the alias table."""
        from ..agent.names import english_name

        for entity in (nlu or {}).get("entities") or [] if isinstance(nlu, dict) else []:
            if isinstance(entity, dict) and "name_en" not in entity:
                entity["name_en"] = english_name(entity.get("canonical_name"), entity.get("symbol"))

    def _inline_fact_check(query: str) -> dict | None:
        """ "听说茅台市盈率只有15倍，是真的吗": the claim checked against the data, as the agent path does."""
        from ..agent.hearsay import claim_in_message, fact_check_for
        from ..chatbot import detect_query_language

        if claim_in_message(query) is None:
            return None
        try:
            agent = get_agent()
        except Exception:  # no agent runtime: the answer still stands
            logger.exception("[chat] inline fact check unavailable")
            return None
        zh = detect_query_language(query) == "zh"
        return fact_check_for(query, service=agent.runtime.service, registry=agent.runtime.registry, zh=zh)

    request_timeout_s = float(os.getenv("QI_AGENT_REQUEST_TIMEOUT_S", "120"))

    async def run_with_timeout(function, *args, **kwargs):
        try:
            return await asyncio.wait_for(asyncio.to_thread(function, *args, **kwargs), timeout=request_timeout_s)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=f"agent did not finish within {request_timeout_s:g}s") from exc

    @app.post("/agent/chat")
    async def agent_chat(payload: AgentChatRequest, request: Request) -> dict:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        try:
            return await run_with_timeout(
                get_agent().chat,
                query,
                session_id=payload.session_id,
                mode=payload.mode,
                user_profile=payload.user_profile,
                dialog_context=payload.dialog_context,
                owner=principal_of(request),
            )
        except SessionAccessError as exc:
            raise _session_not_found(exc) from exc

    @app.post("/agent/chat/stream")
    def agent_chat_stream(payload: AgentChatRequest, request: Request) -> StreamingResponse:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        agent = get_agent()
        owner = principal_of(request)
        if payload.session_id and agent.owner_of(payload.session_id) not in {None, owner}:
            raise _session_not_found(SessionAccessError(payload.session_id))

        def events() -> Iterator[str]:
            for event in agent.stream(
                query,
                session_id=payload.session_id,
                mode=payload.mode,
                user_profile=payload.user_profile,
                dialog_context=payload.dialog_context,
                owner=owner,
            ):
                yield _sse(event)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/agent/resume")
    async def agent_resume(payload: AgentResumeRequest, request: Request) -> dict:
        try:
            return await run_with_timeout(
                get_agent().resume, payload.session_id, payload.reply.strip(), owner=principal_of(request)
            )
        except SessionAccessError as exc:
            raise _session_not_found(exc) from exc
        except NoPendingClarificationError as exc:
            # A repeated identical reply is answered from the stored result (``replayed: true``) by the service;
            # only a reply with nothing to answer gets here.
            raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc)}) from exc

    @app.get("/agent/sessions/{session_id}")
    def agent_session(session_id: Annotated[str, ApiPath(pattern=SESSION_ID_PATTERN)], request: Request) -> dict:
        agent = get_agent()
        if agent.owner_of(session_id) not in {None, principal_of(request)}:
            raise _session_not_found(SessionAccessError(session_id))
        return {
            "session_id": session_id,
            "turns": agent.history(session_id),
            "pending_clarification": agent.pending_clarification(session_id),
        }

    @app.get("/agent/traces")
    def agent_traces(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        session_id: Annotated[str | None, Query(pattern=SESSION_ID_PATTERN)] = None,
    ) -> dict:
        # Callers only see their own runs; anonymous callers are refused (403) by the security middleware.
        return {"traces": trace_store.recent(limit, session_id=session_id, owner=principal_of(request))}

    @app.get("/agent/traces/{trace_id}")
    def agent_trace(trace_id: Annotated[str, ApiPath(pattern=TRACE_ID_PATTERN)], request: Request) -> dict:
        trace = trace_store.get(trace_id, owner=principal_of(request))
        if trace is None:
            raise HTTPException(status_code=404, detail=f"trace {trace_id} not found")
        return trace

    @app.post("/agent/claim-check")
    async def agent_claim_check(payload: ClaimCheckRequest) -> dict:
        """Check the numbers in a pasted market claim against market and fundamental data (no LLM)."""
        from ..agent.claim_check import check_claim
        from ..chatbot import detect_query_language

        agent = get_agent()
        claim = payload.claim.strip()
        report = await run_with_timeout(
            check_claim,
            claim,
            service=agent.runtime.service,
            registry=agent.runtime.registry,
            zh=(payload.language or detect_query_language(claim)) == "zh",
        )
        return report.model_dump()

    @app.post("/agent/feedback")
    def agent_feedback(payload: FeedbackRequest, request: Request) -> dict:
        """Thumbs up/down on an answer, stored with its trace so failures can become evaluation tasks
        (``scripts/feedback_to_tasks.py``)."""
        owner = principal_of(request)
        trace = trace_store.get(payload.trace_id, owner=owner)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"trace {payload.trace_id} not found")
        record = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "trace_id": payload.trace_id,
            "session_id": payload.session_id or trace.get("session_id"),
            "rating": payload.rating,
            "comment": payload.comment,
            "query": trace.get("query"),
            "route": trace.get("route"),
            "owner": owner,
        }
        feedback_log.parent.mkdir(parents=True, exist_ok=True)
        with feedback_lock, feedback_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        metrics_sink.record_feedback(payload.rating, prompt_version_of(trace))
        return {"ok": True}

    @app.get("/metrics")
    def metrics() -> Response:
        if not metrics_sink.available:
            raise HTTPException(status_code=503, detail="prometheus-client is not installed")
        body, content_type = metrics_sink.render()
        return Response(content=body, media_type=content_type)

    if install_a2a(app, get_agent):
        logger.info("[startup] A2A agent card at /.well-known/agent-card.json, JSON-RPC at /a2a.")

    app.state.trace_store = trace_store

    def close_shared_stores() -> None:
        # Postgres-backed stores own connection pools; the in-memory ones have nothing to close.
        for store in (
            trace_store,
            getattr(app.state, "a2a_task_store", None),
            audit_sink,
            getattr(app.state, "rate_limiter", None),
        ):
            close = getattr(store, "close", None)
            if callable(close):
                close()

    app.router.on_shutdown.append(close_shared_stores)

    @app.post("/nlu/analyze")
    def analyze(payload: AnalyzeRequest) -> dict:
        if not payload.query.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")
        return runtime.analyze_query(
            payload.query,
            user_profile=payload.user_profile,
            dialog_context=payload.dialog_context,
            debug=payload.debug,
        )

    @app.post("/retrieval/search")
    def retrieval(payload: RetrievalRequest) -> dict:
        return runtime.retrieve_evidence(
            payload.nlu_result.model_dump(mode="json"), top_k=payload.top_k, debug=payload.debug
        )

    @app.post("/query/intelligence", response_model=PipelineResponse)
    def pipeline(payload: PipelineRequest) -> dict:
        if not payload.query.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")
        return runtime.run_pipeline(
            payload.query,
            user_profile=payload.user_profile,
            dialog_context=payload.dialog_context,
            top_k=payload.top_k,
            debug=payload.debug,
        )

    @app.post("/query/intelligence/artifacts", response_model=ArtifactResponse)
    def pipeline_artifacts(payload: ArtifactRequest) -> dict:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        result = runtime.run_pipeline(
            query,
            user_profile=payload.user_profile,
            dialog_context=payload.dialog_context,
            top_k=payload.top_k,
            debug=payload.debug,
        )
        written = artifact_writer.write(
            query=query,
            nlu_result=result["nlu_result"],
            retrieval_result=result["retrieval_result"],
            session_id=payload.session_id,
            message_id=payload.message_id,
        )
        return {
            "query_id": result["nlu_result"]["query_id"],
            "run_id": written["run_id"],
            "status": "completed",
            "artifact_dir": written["artifact_dir"],
            "artifacts": written["artifacts"],
            "nlu_result": result["nlu_result"],
            "retrieval_result": result["retrieval_result"],
        }

    # ---- live data source health (passive: reports recorded outcomes, never calls upstreams) ----
    @app.get("/sources/health")
    def sources_health(probe: bool = False) -> dict:
        # Passive unless ?probe=1; probe rounds are rate limited (QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS).
        from ..integrations.sources.report import sources_health_report

        return sources_health_report(getattr(runtime, "retrieval_pipeline", None), probe=probe)

    return app
