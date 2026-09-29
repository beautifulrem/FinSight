"""``explain_concept``: curated definitions of A-share market concepts (see ``agent/glossary.py``)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..evidence import AgentEvidence
from ..glossary import GLOSSARY, GlossaryEntry, lookup_concept
from .base import ToolFailure, ToolOutput, ToolSpec
from .context import ToolContext

GLOSSARY_SOURCE = "FinSight glossary"


class ConceptInput(BaseModel):
    term: str = Field(
        default="",
        max_length=60,
        description="The concept to explain, e.g. '北向资金', '两融', 'ST股'. Falls back to matching `query`.",
    )
    query: str = Field(default="", max_length=200, description="The user's question, used when `term` is empty.")


class ConceptOutput(BaseModel):
    term: str
    definition_zh: str
    definition_en: str
    has_data_series: bool = Field(description="Whether FinSight tracks a data series for this concept.")
    evidence_id: str
    source: str = GLOSSARY_SOURCE


def _evidence(entry: GlossaryEntry) -> AgentEvidence:
    return AgentEvidence(
        evidence_id=entry.evidence_id,
        kind="document",
        source_type="glossary",
        title=entry.term,
        source_name=GLOSSARY_SOURCE,
        text_excerpt=f"{entry.zh}\n{entry.en}",
        payload={"term": entry.term, "has_data_series": entry.has_data_series},
        produced_by="explain_concept",
    )


def build_explain_concept(_context: ToolContext) -> ToolSpec:
    def handler(args: ConceptInput) -> ToolOutput:
        text = args.term.strip() or args.query.strip()
        if not text:
            raise ToolFailure("invalid_arguments", "provide a term or a query")
        entry = lookup_concept(text)
        if entry is None:
            raise ToolFailure("not_found", f"'{text[:40]}' is not in the FinSight glossary")
        data = ConceptOutput(
            term=entry.term,
            definition_zh=entry.zh,
            definition_en=entry.en,
            has_data_series=entry.has_data_series,
            evidence_id=entry.evidence_id,
        )
        return ToolOutput(data=data, evidence=[_evidence(entry)])

    return ToolSpec(
        name="explain_concept",
        description=(
            "Explain an A-share market concept from FinSight's curated glossary (northbound funds, margin "
            "trading, price limits, ST shares, Stock Connect, the STAR Market, ...). Definitions only: it returns "
            "no market data. Covered terms: " + "、".join(entry.term for entry in GLOSSARY) + "."
        ),
        input_model=ConceptInput,
        handler=handler,
        timeout_s=5.0,
        max_retries=0,
        cache_ttl_s=3600.0,
    )
