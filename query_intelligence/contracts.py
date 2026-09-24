from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MIN_RETRIEVAL_TOP_K = 1
MAX_RETRIEVAL_TOP_K = 100
MAX_QUERY_LENGTH = 2000
MAX_NLU_LIST_ITEMS = 64
MAX_NLU_ENTITIES = 32
MAX_NLU_TEXT_LENGTH = 256
MAX_USER_PROFILE_FIELDS = 64
MAX_DIALOG_CONTEXT_ITEMS = 32

ShortText = Annotated[str, Field(max_length=MAX_NLU_TEXT_LENGTH)]

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
    label: ShortText
    score: float = Field(ge=0.0, le=1.0)


class ProductPrediction(LabelScore):
    pass


class EntityMatch(BaseModel):
    mention: ShortText
    entity_type: ShortText
    confidence: float = Field(ge=0.0, le=1.0)
    match_type: ShortText
    entity_id: int | None = None
    canonical_name: ShortText | None = None
    symbol: ShortText | None = None
    exchange: ShortText | None = None


class Explainability(BaseModel):
    matched_rules: list[ShortText] = Field(default_factory=list, max_length=MAX_NLU_LIST_ITEMS)
    top_features: list[ShortText] = Field(default_factory=list, max_length=MAX_NLU_LIST_ITEMS)


class NLUResult(BaseModel):
    query_id: ShortText
    raw_query: str = Field(max_length=MAX_QUERY_LENGTH)
    normalized_query: str = Field(max_length=MAX_QUERY_LENGTH)
    question_style: QuestionStyle
    product_type: ProductPrediction
    intent_labels: list[LabelScore] = Field(max_length=MAX_NLU_LIST_ITEMS)
    topic_labels: list[LabelScore] = Field(max_length=MAX_NLU_LIST_ITEMS)
    entities: list[EntityMatch] = Field(max_length=MAX_NLU_ENTITIES)
    comparison_targets: list[ShortText] = Field(default_factory=list, max_length=MAX_NLU_LIST_ITEMS)
    keywords: list[ShortText] = Field(default_factory=list, max_length=MAX_NLU_LIST_ITEMS)
    time_scope: TimeScope
    forecast_horizon: ShortText
    sentiment_of_user: ShortText
    operation_preference: ActionType
    required_evidence_types: list[ShortText] = Field(default_factory=list, max_length=MAX_NLU_LIST_ITEMS)
    source_plan: list[ShortText] = Field(default_factory=list, max_length=MAX_NLU_LIST_ITEMS)
    risk_flags: list[ShortText] = Field(default_factory=list, max_length=MAX_NLU_LIST_ITEMS)
    missing_slots: list[ShortText] = Field(default_factory=list, max_length=MAX_NLU_LIST_ITEMS)
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
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    user_profile: dict[str, Any] = Field(default_factory=dict, max_length=MAX_USER_PROFILE_FIELDS)
    dialog_context: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_DIALOG_CONTEXT_ITEMS)
    debug: bool = False


class RetrievalRequest(BaseModel):
    nlu_result: NLUResult
    top_k: int = Field(default=20, ge=MIN_RETRIEVAL_TOP_K, le=MAX_RETRIEVAL_TOP_K)
    debug: bool = False


AgentMode = Literal["auto", "workflow", "agent"]
SESSION_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,128}$"


class AgentChatRequest(BaseModel):
    """Request for ``/agent/chat`` and ``/agent/chat/stream``."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)
    mode: AgentMode = "auto"
    user_profile: dict[str, Any] = Field(default_factory=dict, max_length=MAX_USER_PROFILE_FIELDS)
    dialog_context: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_DIALOG_CONTEXT_ITEMS)


class AgentResumeRequest(BaseModel):
    """Answer to a pending clarification for ``/agent/resume``."""

    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    reply: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)


AgentRoute = Literal["refuse", "clarify", "workflow", "agent"]


class AgentToolError(BaseModel):
    code: str
    message: str


class AgentToolCall(BaseModel):
    """One executed tool call (raw tool output is omitted; it is summarized as evidence)."""

    model_config = ConfigDict(extra="allow")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    error: AgentToolError | None = None
    latency_ms: float
    attempts: int
    cached: bool
    evidence_ids: list[str] = Field(default_factory=list)
    source: str = Field(description="'plan' for the deterministic planner, 'llm' for model-chosen calls")
    reason: str = ""
    step: int | None = None


class AgentEvidenceSource(BaseModel):
    evidence_id: str
    kind: str
    source_type: str | None = None
    title: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    as_of: str | None = None
    produced_by: str | None = None


class AgentVerification(BaseModel):
    model_config = ConfigDict(extra="allow")

    passed: bool | None = None
    cited_ids: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    unsupported_numbers: list[float] = Field(default_factory=list)
    checked_numbers: int = 0
    missing_citations: bool = False


class AgentLLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class AgentLLMInfo(BaseModel):
    model: str | None = None
    calls: int = 0
    steps: int = 0
    usage: AgentLLMUsage
    cost: float | None = Field(default=None, description="Only set when QI_LLM_PRICE_* is configured")
    currency: str | None = None
    log: list[dict[str, Any]] = Field(default_factory=list)


class AgentClarification(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["clarification"] = "clarification"
    question: str
    missing_slots: list[str] = Field(default_factory=list)
    original_query: str | None = None


class AgentChatResponse(BaseModel):
    """Response of ``/agent/chat`` and ``/agent/resume``, and the ``answer`` event of the SSE stream.

    ``status == "needs_clarification"`` carries only ``session_id`` and ``clarification``; answer
    the question with ``/agent/resume``. ``status == "ok"`` carries the answer and its evidence trail.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["ok", "needs_clarification"]
    session_id: str
    clarification: AgentClarification | None = None
    trace_id: str | None = None
    run_id: str | None = None
    query: str | None = None
    language: Literal["zh", "en"] | None = None
    route: AgentRoute | None = None
    route_reasons: list[str] = Field(default_factory=list)
    answer: str | None = None
    key_points: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    risk_disclaimer: str | None = None
    evidence_sources: list[AgentEvidenceSource] = Field(default_factory=list)
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    verification: AgentVerification | None = None
    compliance_notes: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    answer_source: str | None = Field(default=None, description="template, llm_compose, llm_agent, clarification or guardrail")
    llm: AgentLLMInfo | None = None
    nlu_summary: dict[str, Any] | None = None
    spans: list[dict[str, Any]] = Field(default_factory=list)
    turn_index: int | None = None
    sentiment: dict[str, Any] | None = None
    next_questions: list[dict[str, Any]] = Field(default_factory=list)


class PipelineRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    user_profile: dict[str, Any] = Field(default_factory=dict, max_length=MAX_USER_PROFILE_FIELDS)
    dialog_context: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_DIALOG_CONTEXT_ITEMS)
    top_k: int = Field(default=20, ge=MIN_RETRIEVAL_TOP_K, le=MAX_RETRIEVAL_TOP_K)
    debug: bool = False


class PipelineResponse(BaseModel):
    nlu_result: NLUResult
    retrieval_result: RetrievalResult


class ArtifactRequest(PipelineRequest):
    session_id: ShortText | None = None
    message_id: ShortText | None = None


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
