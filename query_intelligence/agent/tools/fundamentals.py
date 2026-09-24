from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..evidence import AgentEvidence
from .base import ToolFailure, ToolOutput, ToolSpec, TransientToolError
from .context import ToolContext, provider_warnings

_METADATA_KEYS = {"symbol", "source_name", "provider", "canonical_name", "report_date", "industry_name"}


class FundamentalsInput(BaseModel):
    target: str = Field(
        min_length=1,
        max_length=64,
        description="Ticker such as '600519.SH' or a company name such as '贵州茅台'.",
    )


class IndustrySnapshot(BaseModel):
    industry_name: str
    metrics: dict[str, Any]
    evidence_id: str


class FundamentalsOutput(BaseModel):
    symbol: str
    name: str
    report_date: str | None
    metrics: dict[str, Any] = Field(
        description="Financial metrics as reported by the source, e.g. revenue, net_profit, roe, pe_ttm, pb."
    )
    source: str | None
    evidence_id: str | None
    industry: IndustrySnapshot | None = None


def build_fundamentals_tool(context: ToolContext) -> ToolSpec:
    def handler(args: FundamentalsInput) -> ToolOutput:
        resolved = context.resolve_target(args.target)
        if resolved.product_type != "stock":
            raise ToolFailure("unavailable", f"fundamentals are only available for stocks, not {resolved.product_type}")
        bundle = context.bundle(
            source_plan=["fundamental_sql", "industry_sql"],
            symbols=[resolved.symbol],
            names=[resolved.name],
            product_type="stock",
        )
        items = context.fetch_structured(bundle)
        fundamental = next(
            (
                item
                for item in items
                if item.get("source_type") == "fundamental_sql"
                and str((item.get("payload") or {}).get("symbol", "")).upper() == resolved.symbol
            ),
            None,
        )
        industry = next((item for item in items if item.get("source_type") == "industry_sql"), None)
        if fundamental is None and industry is None:
            warnings = provider_warnings(items)
            if warnings:
                raise TransientToolError("; ".join(warnings)[:300])
            raise ToolFailure(
                "not_found", f"no fundamentals for {resolved.name} ({resolved.symbol}) in the configured sources"
            )

        evidence: list[AgentEvidence] = []
        payload = (fundamental or {}).get("payload") or {}
        fundamental_evidence_id = None
        if fundamental is not None:
            item = AgentEvidence.from_structured(fundamental, produced_by="get_fundamentals")
            item.title = f"{resolved.name} ({resolved.symbol}) fundamentals"
            evidence.append(item)
            fundamental_evidence_id = item.evidence_id

        industry_snapshot = None
        if industry is not None:
            industry_evidence = AgentEvidence.from_structured(industry, produced_by="get_fundamentals")
            industry_payload = industry.get("payload") or {}
            industry_name = str(
                industry_payload.get("industry_name") or industry_evidence.evidence_id.removeprefix("industry_")
            )
            industry_evidence.title = f"{industry_name} industry snapshot"
            evidence.append(industry_evidence)
            industry_snapshot = IndustrySnapshot(
                industry_name=industry_name,
                metrics=_metrics(industry_payload),
                evidence_id=industry_evidence.evidence_id,
            )

        output = FundamentalsOutput(
            symbol=resolved.symbol,
            name=resolved.name,
            report_date=_as_str(payload.get("report_date")),
            metrics=_metrics(payload),
            source=(fundamental or {}).get("source_name") or payload.get("source_name"),
            evidence_id=fundamental_evidence_id,
            industry=industry_snapshot,
        )
        return ToolOutput(data=output, evidence=evidence)

    return ToolSpec(
        name="get_fundamentals",
        description=(
            "Latest reported fundamentals for one listed company (revenue, net profit, ROE, PE(TTM), PB, "
            "growth, when available) plus its industry snapshot. Stocks only."
        ),
        input_model=FundamentalsInput,
        handler=handler,
        timeout_s=30.0,
        max_retries=1,
        cache_ttl_s=300.0,
    )


def _metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in _METADATA_KEYS
        and not str(key).startswith("_")
        and value is not None
        and not isinstance(value, dict | list)
    }


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)
