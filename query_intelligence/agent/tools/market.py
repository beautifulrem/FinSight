from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..evidence import AgentEvidence, safe_evidence_id
from .base import ToolFailure, ToolOutput, ToolSpec, TransientToolError
from .context import ResolvedTarget, ToolContext, provider_warnings

_TARGET_FIELD = Field(
    min_length=1,
    max_length=64,
    description="Ticker such as '600519.SH' or a security name/alias such as '贵州茅台'.",
)


class MarketTargetInput(BaseModel):
    target: str = _TARGET_FIELD


class PriceHistoryInput(MarketTargetInput):
    days: int = Field(default=10, ge=1, le=30, description="How many recent daily closes to return.")


class DailyClose(BaseModel):
    date: str | None
    close: float


class PriceHistoryOutput(BaseModel):
    symbol: str
    name: str
    product_type: str
    as_of: str | None
    close: float | None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    pct_change_1d: float | None = Field(default=None, description="Daily change in percent.")
    volume: float | None = None
    amount: float | None = None
    recent_closes: list[DailyClose] = Field(description="Oldest first; the last element is the latest close.")
    source: str | None
    evidence_id: str


class IndicatorsOutput(BaseModel):
    symbol: str
    name: str
    as_of: str | None
    latest_close: float | None
    ma5: float | None = None
    ma20: float | None = None
    rsi_14: float | None = None
    macd: dict[str, Any] | None = None
    volatility_20d: float | None = Field(default=None, description="Annualized 20-day volatility.")
    bollinger: dict[str, Any] | None = None
    trend_signal: str | None = None
    price_vs_ma: dict[str, Any] | None = None
    pct_change_nd: dict[str, Any] | None = Field(default=None, description="Multi-day returns in percent.")
    evidence_id: str


def build_market_tools(context: ToolContext) -> list[ToolSpec]:
    def fetch_market(target: str) -> tuple[ResolvedTarget, dict[str, Any], dict[str, Any]]:
        resolved = context.resolve_target(target)
        bundle = context.bundle(
            source_plan=["market_api"],
            symbols=[resolved.symbol],
            names=[resolved.name],
            product_type=resolved.product_type,
        )
        items = context.fetch_structured(bundle)
        for item in items:
            payload = item.get("payload") or {}
            if item.get("source_type") == "market_api" and (
                str(payload.get("symbol", "")).upper() == resolved.symbol
                or item.get("evidence_id") == f"price_{resolved.symbol}"
            ):
                return resolved, item, payload
        warnings = provider_warnings(items)
        if warnings:
            raise TransientToolError("; ".join(warnings)[:300])
        raise ToolFailure(
            "not_found", f"no market data for {resolved.name} ({resolved.symbol}) in the configured sources"
        )

    def price_history(args: PriceHistoryInput) -> ToolOutput:
        resolved, item, payload = fetch_market(args.target)
        history = [row for row in payload.get("history") or [] if row.get("close") is not None]
        # Providers return history latest-first; expose it oldest-first.
        recent = [
            DailyClose(date=_as_str(row.get("trade_date") or row.get("date")), close=float(row["close"]))
            for row in reversed(history[: args.days])
        ]
        if not recent and payload.get("close") is not None:
            recent = [DailyClose(date=_as_str(payload.get("trade_date")), close=float(payload["close"]))]
        evidence_id = safe_evidence_id(f"price_{resolved.symbol}")
        output = PriceHistoryOutput(
            symbol=resolved.symbol,
            name=resolved.name,
            product_type=resolved.product_type,
            as_of=_as_str(payload.get("trade_date")),
            close=_as_float(payload.get("close")),
            open=_as_float(payload.get("open")),
            high=_as_float(payload.get("high")),
            low=_as_float(payload.get("low")),
            pct_change_1d=_as_float(payload.get("pct_change_1d")),
            volume=_as_float(payload.get("volume")),
            amount=_as_float(payload.get("amount")),
            recent_closes=recent,
            source=item.get("source_name") or payload.get("source_name"),
            evidence_id=evidence_id,
        )
        evidence = AgentEvidence(
            evidence_id=evidence_id,
            kind="structured",
            source_type="market_api",
            title=f"{resolved.name} ({resolved.symbol}) daily market data",
            source_name=output.source,
            provider=item.get("provider"),
            as_of=output.as_of,
            payload=output.model_dump(exclude={"evidence_id"}, mode="json"),
            produced_by="get_price_history",
        )
        return ToolOutput(data=output, evidence=[evidence])

    def indicators(args: MarketTargetInput) -> ToolOutput:
        resolved, item, payload = fetch_market(args.target)
        analysis = payload.get("_market_analysis")
        if not analysis:
            raise ToolFailure(
                "unavailable",
                f"not enough price history to compute indicators for {resolved.name} ({resolved.symbol})",
            )
        evidence_id = safe_evidence_id(f"indicators_{resolved.symbol}")
        output = IndicatorsOutput(
            symbol=resolved.symbol,
            name=resolved.name,
            as_of=_as_str(payload.get("trade_date")),
            latest_close=_as_float(payload.get("close")),
            ma5=_as_float(analysis.get("ma5")),
            ma20=_as_float(analysis.get("ma20")),
            rsi_14=_as_float(analysis.get("rsi_14")),
            macd=analysis.get("macd"),
            volatility_20d=_as_float(analysis.get("volatility_20d")),
            bollinger=analysis.get("bollinger"),
            trend_signal=analysis.get("trend_signal"),
            price_vs_ma=analysis.get("price_vs_ma"),
            pct_change_nd=analysis.get("pct_change_nd"),
            evidence_id=evidence_id,
        )
        evidence = AgentEvidence(
            evidence_id=evidence_id,
            kind="structured",
            source_type="technical_indicators",
            title=f"{resolved.name} ({resolved.symbol}) technical indicators",
            source_name=item.get("source_name") or payload.get("source_name"),
            as_of=output.as_of,
            payload=output.model_dump(exclude={"evidence_id"}, mode="json"),
            produced_by="compute_indicators",
        )
        return ToolOutput(data=output, evidence=[evidence])

    return [
        ToolSpec(
            name="get_price_history",
            description=(
                "Latest daily quote (close, open/high/low, daily % change, volume) and recent closes for one "
                "A-share, ETF, fund, or index. Accepts a ticker or a name."
            ),
            input_model=PriceHistoryInput,
            handler=price_history,
            timeout_s=30.0,
            max_retries=1,
            cache_ttl_s=60.0,
        ),
        ToolSpec(
            name="compute_indicators",
            description=(
                "Technical indicators for one security computed from its recent price history: MA5, MA20, "
                "RSI(14), MACD, 20-day volatility, Bollinger bands, multi-day returns, and a trend signal."
            ),
            input_model=MarketTargetInput,
            handler=indicators,
            timeout_s=30.0,
            max_retries=1,
            cache_ttl_s=60.0,
        ),
    ]


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)
