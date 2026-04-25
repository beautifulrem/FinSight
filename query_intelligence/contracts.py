from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


QuestionStyle = Literal["fact", "why", "compare", "advice", "forecast"]
TimeScope = Literal[
    "today",
    "recent_3d",
    "recent_1w",
    "recent_1m",
    "recent_1q",
    "long_term",
    "unspecified",
]
ActionType = Literal["buy", "sell", "hold", "reduce", "observe", "unknown"]


class LabelScore(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class ProductPrediction(LabelScore):
    pass


class EntityMatch(BaseModel):
    mention: str
    entity_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    match_type: str
    entity_id: int | None = None
    canonical_name: str | None = None
    symbol: str | None = None
    exchange: str | None = None


class Explainability(BaseModel):
    matched_rules: list[str] = Field(default_factory=list)
    top_features: list[str] = Field(default_factory=list)


class NLUResult(BaseModel):
    query_id: str
    raw_query: str
    normalized_query: str
    question_style: QuestionStyle
    product_type: ProductPrediction
    intent_labels: list[LabelScore]
    topic_labels: list[LabelScore]
    entities: list[EntityMatch]
    comparison_targets: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    time_scope: TimeScope
    forecast_horizon: str
    sentiment_of_user: str
    operation_preference: ActionType
    required_evidence_types: list[str] = Field(default_factory=list)
    source_plan: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    explainability: Explainability


class EvidenceItem(BaseModel):
    evidence_id: str
    source_type: str
    source_name: str | None = None
    source_url: str | None = None
    provider: str | None = None
    title: str | None = None
    summary: str | None = None
    text_excerpt: str | None = None
    body: str | None = None
    body_available: bool = False
    publish_time: str | None = None
    retrieved_at: str | None = None
    entity_hits: list[str] | None = None
    retrieval_score: float | None = None
    rank_score: float | None = None
    reason: list[str] | None = None
    payload: dict[str, Any] | None = None


class StructuredDataItem(BaseModel):
    evidence_id: str
    source_type: str
    source_name: str | None = None
    source_url: str | None = None
    provider: str | None = None
    provider_endpoint: str | None = None
    query_params: dict[str, Any] = Field(default_factory=dict)
    source_reference: str | None = None
    as_of: str | None = None
    period: str | None = None
    field_coverage: dict[str, Any] = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)
    retrieved_at: str | None = None
    payload: dict[str, Any]


class EvidenceGroup(BaseModel):
    group_id: str
    group_type: str
    members: list[str]


class RetrievalDebugTrace(BaseModel):
    candidate_count: int
    after_dedup: int
    top_ranked: list[str]


class RetrievalResult(BaseModel):
    query_id: str
    nlu_snapshot: dict[str, Any]
    executed_sources: list[str]
    documents: list[EvidenceItem]
    structured_data: list[StructuredDataItem]
    evidence_groups: list[EvidenceGroup]
    coverage: dict[str, bool]
    coverage_detail: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str]
    retrieval_confidence: float = Field(ge=0.0, le=1.0)
    analysis_summary: dict[str, Any] = Field(default_factory=dict)
    debug_trace: RetrievalDebugTrace


class AnalyzeRequest(BaseModel):
    query: str
    user_profile: dict[str, Any] = Field(default_factory=dict)
    dialog_context: list[dict[str, Any]] = Field(default_factory=list)
    debug: bool = False


class RetrievalRequest(BaseModel):
    nlu_result: dict[str, Any]
    top_k: int = Field(default=20, ge=1, le=100)
    debug: bool = False


class PipelineRequest(BaseModel):
    query: str
    user_profile: dict[str, Any] = Field(default_factory=dict)
    dialog_context: list[dict[str, Any]] = Field(default_factory=list)
    top_k: int = Field(default=20, ge=1, le=100)
    debug: bool = False


class PipelineResponse(BaseModel):
    nlu_result: NLUResult
    retrieval_result: RetrievalResult


class ArtifactRequest(PipelineRequest):
    session_id: str | None = None
    message_id: str | None = None


class ArtifactPaths(BaseModel):
    query_path: str
    nlu_result_path: str
    retrieval_result_path: str
    manifest_path: str


class ArtifactResponse(BaseModel):
    query_id: str
    run_id: str
    status: str
    artifact_dir: str
    artifacts: ArtifactPaths
    nlu_result: NLUResult
    retrieval_result: RetrievalResult
