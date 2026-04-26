from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["zh", "en", "mixed", "unknown"]

TextLevel = Literal["full", "short"]

SKIP_PRODUCT_TYPES: set[str] = {"out_of_scope", "product_info", "trading_rule_fee"}
SUPPORTED_SOURCE_TYPES: set[str] = {"news", "announcement", "research_note", "product_doc"}


class QueryInput(BaseModel):
    """Input extracted from NLU and retrieval results for a single query."""

    query_id: str
    product_type_label: str
    intent_labels: list[str]
    topic_labels: list[str]
    entities: list[dict[str, Any]]
    documents: list[dict[str, Any]]


class PreprocessedDoc(BaseModel):
    """A single document after preprocessing."""

    evidence_id: str
    source_type: str
    title: str
    publish_time: str | None = None
    source_name: str | None = None
    rank_score: float | None = None
    raw_text: str
    language: Language
    sentences: list[str]
    entity_hits: list[str] = Field(default_factory=list)
    text_level: TextLevel = "full"
    relevant_excerpt: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


class FilterMeta(BaseModel):
    """Summary statistics about what happened during preprocessing."""

    skipped_by_product_type: bool = False
    skipped_by_intent: bool = False
    skipped_docs_count: int = 0
    short_text_fallback_count: int = 0
    analyzed_docs_count: int = 0


class SentimentItem(BaseModel):
    """Sentiment analysis result for a single article."""

    evidence_id: str
    source_type: str
    title: str
    publish_time: str | None = None
    source_name: str | None = None
    entity_symbols: list[str] = Field(default_factory=list)
    sentiment_label: str = ""
    sentiment_score: float = 0.0
    confidence: float = 0.0
    text_level: TextLevel = "full"
    relevant_excerpt: str | None = None
    rank_score: float | None = None
