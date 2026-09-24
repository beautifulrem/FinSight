from __future__ import annotations

from pydantic import BaseModel, Field

from .base import ToolOutput, ToolSpec
from .context import ToolContext


class ResolveEntityInput(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=200,
        description="Company, fund, ETF, index or sector name, alias, or ticker, e.g. '茅台' or '600519.SH'.",
    )


class ResolvedEntity(BaseModel):
    name: str
    symbol: str | None
    entity_type: str
    exchange: str | None
    mention: str
    match_type: str
    confidence: float


class ResolveEntityOutput(BaseModel):
    entities: list[ResolvedEntity]
    comparison_targets: list[str]
    ambiguous: bool
    found: bool


def build_resolve_entity(context: ToolContext) -> ToolSpec:
    def handler(args: ResolveEntityInput) -> ToolOutput:
        entities, comparison_targets, trace = context.resolve_entities(args.text)
        resolved = [
            ResolvedEntity(
                name=str(entity["canonical_name"]),
                symbol=entity.get("symbol"),
                entity_type=str(entity.get("entity_type") or "unknown"),
                exchange=entity.get("exchange"),
                mention=str(entity.get("mention") or ""),
                match_type=str(entity.get("match_type") or ""),
                confidence=float(entity.get("confidence") or 0.0),
            )
            for entity in entities
        ]
        output = ResolveEntityOutput(
            entities=resolved,
            comparison_targets=comparison_targets,
            ambiguous="entity_ambiguous" in trace,
            found=bool(resolved),
        )
        return ToolOutput(data=output)

    return ToolSpec(
        name="resolve_entity",
        description=(
            "Resolve a Chinese or English security/sector mention to canonical entities with tickers "
            "(A-shares, ETFs, funds, indices, sectors). Use before market/fundamental tools when the user "
            "gives a name instead of a ticker, or to check which entities a question refers to."
        ),
        input_model=ResolveEntityInput,
        handler=handler,
        timeout_s=10.0,
        max_retries=0,
        cache_ttl_s=600.0,
    )
