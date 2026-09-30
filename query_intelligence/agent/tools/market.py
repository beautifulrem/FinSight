from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from ...integrations.intraday import INTRADAY_SESSIONS, market_session
from ...integrations.sources.catalog import source_label
from ..evidence import AgentEvidence, safe_evidence_id
from .base import ToolFailure, ToolOutput, ToolSpec, TransientToolError
from .context import ResolvedTarget, ToolContext, provider_warnings
from .provenance import SourceProvenance, provenance_from

_TARGET_FIELD = Field(
    min_length=1,
    max_length=64,
    description="Ticker such as '600519.SH' or a security name/alias such as '贵州茅台'.",
)


class MarketTargetInput(BaseModel):
    target: str = _TARGET_FIELD


class PriceHistoryInput(MarketTargetInput):
    days: int = Field(default=10, ge=1, le=30, description="How many recent daily closes to return.")
    intraday: bool = Field(
        default=False,
        description=(
            "Set true only for questions about today (今天/今日/today): during A-share trading hours the latest "
            "real-time quote is added and labelled intraday; otherwise the daily close is returned with the reason."
        ),
    )

    @model_serializer(mode="wrap")
    def _omit_default_intraday(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Normalised arguments key the tool cache and the evaluation snapshots: leaving the default out keeps
        # every call recorded before this field existed byte-identical.
        data = handler(self)
        if not data.get("intraday"):
            data.pop("intraday", None)
        return data


class DailyClose(BaseModel):
    date: str | None
    close: float


class IntradayQuote(BaseModel):
    price: float = Field(description="Latest traded price during the session (not a close).")
    prev_close: float | None = None
    pct_change: float | None = Field(default=None, description="Change against the previous close, in percent.")
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    volume_unit: str | None = None
    amount: float | None = Field(default=None, description="Turnover so far today, CNY.")
    quote_time: str = Field(description="Beijing time of the quote, ISO 8601 with +08:00.")
    source: str | None = None


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
    volume: float | None = Field(default=None, description="Traded volume; in shares once normalised (volume_unit).")
    amount: float | None = Field(default=None, description="Turnover; in CNY once normalised (amount_unit).")
    recent_closes: list[DailyClose] = Field(description="Oldest first; the last element is the latest close.")
    source: str | None
    evidence_id: str
    volume_unit: str | None = Field(
        default=None,
        description="'lot' (手, 100 shares) or 'share' as served; the registry converts to 'share' (tools/units.py).",
    )
    provenance: SourceProvenance | None = Field(
        default=None, description="Where the quote came from, its as-of date, and why a fallback was used."
    )
    price_basis: str = Field(
        default="daily_close", description="'intraday' when `intraday` holds a real-time quote, else 'daily_close'."
    )
    intraday: IntradayQuote | None = Field(default=None, description="Real-time quote (intraday requests only).")
    market_session: str | None = Field(
        default=None,
        description="Beijing-time session when intraday was requested: pre_open, morning, lunch_break, afternoon, "
        "closed or non_trading_day.",
    )
    basis_reason: str | None = Field(
        default=None,
        description="Why an intraday request got the daily close: outside_trading_hours, intraday_unavailable "
        "(no live source) or intraday_failed.",
    )
    intraday_provenance: SourceProvenance | None = None
    year_start: DailyClose | None = Field(
        default=None,
        description="Close of the first trading day of the latest close's year, when the history also has a close "
        "from the year before (so that close is known to be the year's first); for year-to-date changes.",
    )

    @model_serializer(mode="wrap")
    def _omit_missing_year_start(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Payloads without the field stay byte-identical to those recorded before it existed (evaluation snapshots).
        data = handler(self)
        if data.get("year_start") is None:
            data.pop("year_start", None)
        return data


def year_start_close(history: list[dict[str, Any]]) -> DailyClose | None:
    """The first close of the latest row's year, if ``history`` (any order) reaches into the year before."""
    dated = sorted(
        (date, float(row["close"]))
        for row in history
        if row.get("close") is not None and (date := _iso_date(row.get("trade_date") or row.get("date")))
    )
    if not dated:
        return None
    year = dated[-1][0][:4]
    if not any(date[:4] < year for date, _close in dated):
        return None
    date, close = next((date, close) for date, close in dated if date[:4] == year)
    return DailyClose(date=date, close=close)


def _iso_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return None


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
    history_points: int = Field(description="Daily closes available for the calculation.")
    unavailable: list[str] = Field(default_factory=list, description="Indicators not computable from the history.")
    evidence_id: str
    provenance: SourceProvenance | None = Field(default=None, description="Provenance of the underlying prices.")


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
        intraday_fields = _intraday_fields(args, resolved) if args.intraday else {}
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
            volume_unit=payload.get("volume_unit"),
            provenance=provenance_from(payload),
            year_start=year_start_close(history),
            **intraday_fields,
        )
        quote = output.intraday
        evidence = AgentEvidence(
            evidence_id=evidence_id,
            kind="structured",
            source_type="market_api",
            title=(
                f"{resolved.name} ({resolved.symbol}) intraday quote and daily market data"
                if quote
                else f"{resolved.name} ({resolved.symbol}) daily market data"
            ),
            source_name=output.source,
            provider=item.get("provider"),
            as_of=quote.quote_time if quote else output.as_of,
            payload=output.model_dump(exclude={"evidence_id"}, mode="json"),
            produced_by="get_price_history",
        )
        return ToolOutput(data=output, evidence=[evidence])

    def _intraday_fields(args: PriceHistoryInput, resolved: ResolvedTarget) -> dict[str, Any]:
        """The real-time quote during trading hours, else the reason the daily close stands."""
        now = context.clock()
        session = market_session(now)
        fields: dict[str, Any] = {"market_session": session}
        if session not in INTRADAY_SESSIONS:
            return {**fields, "basis_reason": "outside_trading_hours"}
        if context.intraday_provider is None:
            return {**fields, "basis_reason": "intraday_unavailable"}
        try:
            quote = context.intraday_provider.fetch(resolved.symbol, resolved.product_type, now=now)
        except Exception as exc:  # the daily close still answers; the reason is recorded and stated
            return {**fields, "basis_reason": f"intraday_failed: {str(exc)[:200]}"}
        provenance = SourceProvenance(
            source=quote.get("source"),
            source_label=source_label(quote.get("source")) if quote.get("source") else None,
            endpoint=quote.get("endpoint"),
            is_live=True,
            mode="live",
            fetched_at=quote.get("fetched_at"),
            as_of=quote.get("quote_time"),
            freshness="fresh",
            fallback_reason=quote.get("fallback_reason"),
            attempts=list(quote.get("attempts") or []),
            note="盘中实时行情（非收盘价） / intraday quote, not a close",
        )
        return {
            **fields,
            "price_basis": "intraday",
            "intraday": IntradayQuote.model_validate(quote),
            "intraday_provenance": provenance,
        }

    def indicators(args: MarketTargetInput) -> ToolOutput:
        resolved, item, payload = fetch_market(args.target)
        analysis = payload.get("_market_analysis")
        if not analysis:
            raise ToolFailure(
                "unavailable",
                f"not enough price history to compute indicators for {resolved.name} ({resolved.symbol})",
            )
        points = sum(1 for row in payload.get("history") or [] if row.get("close") is not None)
        core = {"ma5": analysis.get("ma5"), "ma20": analysis.get("ma20"), "rsi_14": analysis.get("rsi_14")}
        core["macd"] = analysis.get("macd")
        missing = [name for name, value in core.items() if value is None]
        if len(missing) == len(core):
            raise ToolFailure(
                "unavailable",
                f"only {points} daily closes for {resolved.name} ({resolved.symbol}); "
                "MA5 needs 5, MA20 needs 20, RSI(14) needs 15, MACD needs 26",
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
            history_points=points,
            unavailable=missing,
            evidence_id=evidence_id,
            provenance=provenance_from(payload),
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
