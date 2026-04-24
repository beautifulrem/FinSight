from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from ..artifacts import ArtifactWriter
from ..contracts import (
    AnalyzeRequest,
    ArtifactRequest,
    ArtifactResponse,
    PipelineRequest,
    PipelineResponse,
    RetrievalRequest,
)
from ..service import QueryIntelligenceService, build_default_service


def create_app(service: QueryIntelligenceService | None = None, artifact_output_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Query Intelligence Service", version="0.1.0")
    runtime = service or build_default_service()
    artifact_writer = ArtifactWriter(artifact_output_dir or os.getenv("QI_API_OUTPUT_DIR", "outputs/query_intelligence"))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

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
        return runtime.retrieve_evidence(payload.nlu_result, top_k=payload.top_k, debug=payload.debug)

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
