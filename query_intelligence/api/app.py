from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..artifacts import ArtifactWriter
from ..chatbot import (
    DeepSeekClient,
    apply_live_data_env,
    build_chatbot_response,
    load_chatbot_config,
    render_index_html,
)
from ..contracts import (
    AnalyzeRequest,
    ArtifactRequest,
    ArtifactResponse,
    MAX_DIALOG_CONTEXT_ITEMS,
    MAX_QUERY_LENGTH,
    MAX_RETRIEVAL_TOP_K,
    MAX_USER_PROFILE_FIELDS,
    MIN_RETRIEVAL_TOP_K,
    PipelineRequest,
    PipelineResponse,
    RetrievalRequest,
)
from ..service import QueryIntelligenceService, build_default_service


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
    return query if len(query) <= limit else f"{query[:limit - 3]}..."


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    user_profile: dict[str, Any] = Field(default_factory=dict, max_length=MAX_USER_PROFILE_FIELDS)
    dialog_context: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_DIALOG_CONTEXT_ITEMS)
    top_k: int = Field(default=20, ge=MIN_RETRIEVAL_TOP_K, le=MAX_RETRIEVAL_TOP_K)
    debug: bool = False


def create_app(
    service: QueryIntelligenceService | None = None,
    artifact_output_dir: str | Path | None = None,
    app_config: dict[str, Any] | None = None,
    app_config_path: str | Path | None = None,
    deepseek_client: DeepSeekClient | None = None,
) -> FastAPI:
    chatbot_config = app_config or load_chatbot_config(app_config_path, load_env_file=False)
    if service is None:
        apply_live_data_env(chatbot_config)

    _ensure_console_logging()
    app = FastAPI(title="Query Intelligence Service", version="0.1.0")
    if service is None:
        service_started_at = time.perf_counter()
        logger.info("[startup] Loading default Query Intelligence service...")
        runtime = build_default_service()
        logger.info("[startup] Query Intelligence service loaded in %s.", _elapsed_seconds(service_started_at))
    else:
        runtime = service
    artifact_writer = ArtifactWriter(artifact_output_dir or os.getenv("QI_API_OUTPUT_DIR", "outputs/query_intelligence"))
    logger.info("[startup] Preparing DeepSeek response client...")
    response_client = deepseek_client or DeepSeekClient(chatbot_config)
    logger.info("[startup] FastAPI routes are ready.")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_index_html(chatbot_config)

    @app.post("/chat")
    def chat(payload: ChatRequest) -> dict:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
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
        logger.info("[chat] Completed request in %s", _elapsed_seconds(request_started_at))
        return response

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
        return runtime.retrieve_evidence(payload.nlu_result.model_dump(mode="json"), top_k=payload.top_k, debug=payload.debug)

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

    return app
