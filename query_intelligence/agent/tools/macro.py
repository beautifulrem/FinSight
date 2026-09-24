from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..evidence import AgentEvidence
from .base import ToolFailure, ToolOutput, ToolSpec
from .context import ToolContext

_MACRO_SOURCE_TYPES = {"macro_sql", "macro_indicator", "policy_event"}


class MacroInput(BaseModel):
    topics: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Indicators or themes, e.g. ['CPI', 'PMI', 'M2', '国债收益率', '降息']. "
            "Empty returns all tracked indicators."
        ),
    )
    query: str = Field(default="", max_length=200, description="Optional free-text context for indicator matching.")


class MacroIndicator(BaseModel):
    code: str
    name: str | None = None
    date: str | None
    value: float | None
    unit: str | None = None
    source: str | None = None
    evidence_id: str


class MacroOutput(BaseModel):
    indicators: list[MacroIndicator]
    policy_events: list[dict[str, Any]] = Field(default_factory=list)


def build_macro_tool(context: ToolContext) -> ToolSpec:
    def handler(args: MacroInput) -> ToolOutput:
        query = " ".join([*args.topics, args.query]).strip().lower()
        bundle = context.bundle(
            source_plan=["macro_sql"], query=query, keywords=[topic.lower() for topic in args.topics]
        )
        items = [item for item in context.fetch_structured(bundle) if item.get("source_type") in _MACRO_SOURCE_TYPES]
        if not items:
            raise ToolFailure("not_found", "no macro indicators available in the configured sources")

        indicators: list[MacroIndicator] = []
        events: list[dict[str, Any]] = []
        evidence: list[AgentEvidence] = []
        for item in items:
            record = AgentEvidence.from_structured(item, produced_by="get_macro_indicators")
            payload = item.get("payload") or {}
            if item.get("source_type") == "policy_event":
                record.title = str(payload.get("title") or payload.get("event_name") or "policy event")
                events.append({**_compact(payload), "evidence_id": record.evidence_id})
            else:
                code = str(payload.get("indicator_code") or record.evidence_id)
                record.title = f"{code} macro indicator"
                indicators.append(
                    MacroIndicator(
                        code=code,
                        name=payload.get("indicator_name") or payload.get("name"),
                        date=_as_str(payload.get("metric_date")),
                        value=_as_float(payload.get("metric_value")),
                        unit=payload.get("unit"),
                        source=item.get("source_name") or payload.get("source_name"),
                        evidence_id=record.evidence_id,
                    )
                )
            record.as_of = record.as_of or _as_str(payload.get("metric_date") or payload.get("publish_date"))
            evidence.append(record)
        return ToolOutput(data=MacroOutput(indicators=indicators, policy_events=events), evidence=evidence)

    return ToolSpec(
        name="get_macro_indicators",
        description=(
            "China macro indicators (CPI, PMI, M2, 10-year government bond yield) and related policy events. "
            "Pass topics to narrow the result."
        ),
        input_model=MacroInput,
        handler=handler,
        timeout_s=30.0,
        max_retries=1,
        cache_ttl_s=600.0,
    )


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if not str(key).startswith("_") and not isinstance(value, dict | list)
    }


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)
