from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...query_terms import INDUSTRY_TERMS
from ..evidence import AgentEvidence
from .base import ToolFailure, ToolOutput, ToolSpec, TransientToolError
from .context import ToolContext, provider_warnings
from .provenance import SourceProvenance, provenance_from

_METADATA_KEYS = {
    "symbol",
    "source_name",
    "provider",
    "canonical_name",
    "report_date",
    "industry_name",
    "valuation_date",
}


class FundamentalsInput(BaseModel):
    target: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Ticker such as '600519.SH', a company name such as '贵州茅台', or an industry name such as '白酒' "
            "(returns only that industry's snapshot)."
        ),
    )


class IndustrySnapshot(BaseModel):
    industry_name: str
    metrics: dict[str, Any]
    evidence_id: str
    provenance: SourceProvenance | None = None


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
    provenance: SourceProvenance | None = Field(
        default=None, description="Where the financial statements came from and how fresh they are."
    )
    valuation_provenance: SourceProvenance | None = Field(
        default=None, description="Source of pe_ttm/pb when fetched separately from the statements."
    )


def build_fundamentals_tool(context: ToolContext) -> ToolSpec:
    def industry_only(name: str) -> ToolOutput:
        """Industry snapshot for a sector question with no member stock ("白酒板块整体跌了吗")."""
        bundle = context.bundle(source_plan=["industry_sql"], query=name, product_type="stock")
        items = [
            item
            for item in context.fetch_structured(bundle)
            if item.get("source_type") == "industry_sql" and (item.get("payload") or {}).get("industry_name") == name
        ]
        if not items:
            warnings = provider_warnings(items)
            if warnings:
                raise TransientToolError("; ".join(warnings)[:300])
            raise ToolFailure("not_found", f"no industry snapshot for {name!r} in the configured sources")
        industry = items[0]
        evidence = AgentEvidence.from_structured(industry, produced_by="get_fundamentals")
        payload = industry.get("payload") or {}
        evidence.title = f"{name} industry snapshot"
        evidence.payload = _tidy_payload(evidence.payload)
        evidence.as_of = evidence.as_of or _as_str(payload.get("trade_date") or payload.get("as_of"))
        snapshot = IndustrySnapshot(
            industry_name=name,
            metrics=_metrics(payload),
            evidence_id=evidence.evidence_id,
            provenance=provenance_from(payload),
        )
        output = FundamentalsOutput(
            symbol=name, name=name, report_date=None, metrics={}, source=industry.get("source_name"),
            evidence_id=None, industry=snapshot,
        )  # fmt: skip
        return ToolOutput(data=output, evidence=[evidence])

    def handler(args: FundamentalsInput) -> ToolOutput:
        name = args.target.strip()
        if name in INDUSTRY_TERMS:
            return industry_only(name)
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
            item.payload = _tidy_payload(item.payload)
            item.as_of = item.as_of or _as_str(payload.get("report_date")) or _provenance_as_of(payload)
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
            industry_evidence.payload = _tidy_payload(industry_evidence.payload)
            industry_evidence.as_of = (
                industry_evidence.as_of
                or _as_str(industry_payload.get("trade_date") or industry_payload.get("as_of"))
                or _provenance_as_of(industry_payload)
            )
            evidence.append(industry_evidence)
            industry_snapshot = IndustrySnapshot(
                industry_name=industry_name,
                metrics=_metrics(industry_payload),
                evidence_id=industry_evidence.evidence_id,
                provenance=provenance_from(industry_payload),
            )

        output = FundamentalsOutput(
            symbol=resolved.symbol,
            name=resolved.name,
            report_date=_as_str(payload.get("report_date")),
            metrics=_metrics(payload),
            source=(fundamental or {}).get("source_name") or payload.get("source_name"),
            evidence_id=fundamental_evidence_id,
            industry=industry_snapshot,
            provenance=provenance_from(payload),
            valuation_provenance=provenance_from(payload, "valuation_provenance"),
        )
        return ToolOutput(data=output, evidence=evidence)

    return ToolSpec(
        name="get_fundamentals",
        description=(
            "Latest reported fundamentals for one listed company (revenue, net profit, ROE, PE(TTM), PB, "
            "growth, when available) plus its industry snapshot. Stocks only; pass an industry name such as '白酒' "
            "or '保险' to get just that industry's snapshot (PE, PB, daily change)."
        ),
        input_model=FundamentalsInput,
        handler=handler,
        timeout_s=30.0,
        max_retries=1,
        cache_ttl_s=300.0,
    )


# Valuation multiples are quoted to two decimals; other floats keep four (ratios such as ROE 0.1141).
_TWO_DECIMAL_KEYS = {"pe", "pe_ttm", "pb", "ps", "ps_ttm", "pcf", "dividend_yield"}


def _tidy(key: str, value: Any) -> Any:
    """Round provider floats (e.g. PE 15.97759372) so the model does not copy spurious precision."""
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    return round(value, 2 if key.lower() in _TWO_DECIMAL_KEYS else 4)


def _tidy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: _tidy(str(key), value) for key, value in (payload or {}).items()}


def _provenance_as_of(payload: dict[str, Any]) -> str | None:
    provenance = payload.get("provenance")
    return _as_str(provenance.get("as_of")) if isinstance(provenance, dict) else None


def _metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _tidy(str(key), value)
        for key, value in payload.items()
        if key not in _METADATA_KEYS
        and not str(key).startswith("_")
        and value is not None
        and not isinstance(value, dict | list)
    }


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)
