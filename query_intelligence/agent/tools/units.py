"""Unit normalisation for tool payloads, before they reach the LLM, the verifier or the UI.

Providers disagree on units: Tushare ``daily`` reports turnover (``amount``) in thousands of CNY and volume
in lots of 100 shares, AKShare/efinance report CNY and lots, Sina reports CNY and shares; ROE and margins
arrive in percent from every live source. A model that sees ``"amount": 3793827.534`` with no unit writes
"amount was 3793827.534, no unit specified", and the UI showed 379 万 instead of 37.9 亿.

After normalisation every tool payload states its units explicitly:

* ``get_price_history``: ``amount`` in CNY with ``amount_unit: "CNY"``, ``volume`` in shares with
  ``volume_unit: "share"``, ``pct_change_1d`` in percent (``change_unit: "%"``). The original provider unit is
  kept in ``units_source`` so the conversion is auditable.
* ``get_fundamentals``: ``metric_units`` names the unit of every metric (``%`` for ROE, margins and growth,
  ``x`` for multiples, ``CNY`` for amounts), ratios are in percent, and ``period`` labels the report
  (``FY2024``, ``2025Q3``, ...); the industry snapshot gets ``metric_units`` too.

The LLM also gets the units in words (``units_in_words``): the tool message envelope and the compose
evidence views carry a ``units`` sentence such as "amount is turnover (成交额) in CNY (yuan); revenue in CNY".

The functions are idempotent (already-normalised payloads are recognised by their unit fields), because
the evaluation replays recorded outputs through the same registry.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from .base import ToolOutput

# Provider conventions: (amount multiplier to CNY, volume multiplier to shares). ``None`` = not known.
_MARKET_UNITS: dict[str, tuple[float | None, float | None]] = {
    "tushare": (1000.0, 100.0),  # daily: amount 千元, vol 手
    "akshare": (1.0, 100.0),  # stock_zh_a_hist (Eastmoney): 成交额 元, 成交量 手
    "eastmoney": (1.0, 100.0),
    "efinance": (1.0, 100.0),
    "tencent": (1.0, 100.0),
    "sina": (1.0, 1.0),  # kline and hq quote: 元, 股
}
_VOLUME_UNIT_FACTORS = {"lot": 100.0, "share": 1.0}

# Fundamentals: every ratio is served in percent; multiples as "x"; statement amounts in CNY.
PERCENT_METRICS = frozenset(
    {
        "roe",
        "roa",
        "roic",
        "gross_margin",
        "grossprofit_margin",
        "net_margin",
        "netprofit_margin",
        "netprofit_yoy",
        "revenue_yoy",
        "profit_yoy",
        "debt_to_assets",
        "dividend_yield",
        "pct_change",
        "turnover",
        "turnover_rate",
    }
)
MULTIPLE_METRICS = frozenset({"pe", "pe_ttm", "pb", "ps", "ps_ttm", "pcf"})
CNY_METRICS = frozenset({"revenue", "net_profit", "profit_dedt", "total_assets", "total_equity", "amount"})
PER_SHARE_METRICS = frozenset({"eps", "bps", "dividend_per_share"})
# A level ratio (ROE, margins) from a source of unknown convention with 0 < |value| <= 1 is taken to be a
# fraction (0.33 -> 33%). Live sources all serve percent, so this only applies to seeds and hand-entered
# data; it is recorded in ``units_inferred`` so a reader can see where the unit was guessed. Changes and
# rates that are legitimately small (pct_change, turnover, *_yoy) are never rescaled.
_FRACTION_LIMIT = 1.0
_FRACTION_CANDIDATES = frozenset(
    {"roe", "roa", "roic", "gross_margin", "grossprofit_margin", "net_margin", "netprofit_margin", "debt_to_assets"}
)
_PERCENT_SOURCES = ("tushare", "akshare", "sina", "ths", "eastmoney", "efinance", "cninfo")


def _source_key(*candidates: Any) -> str:
    text = " ".join(str(candidate or "") for candidate in candidates).lower()
    for key in _MARKET_UNITS:
        if key in text:
            return key
    return ""


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalise_market(data: dict[str, Any], *, source: str | None = None) -> dict[str, Any]:
    """``amount`` -> CNY and ``volume`` -> shares, with explicit unit fields (idempotent)."""
    if data.get("amount_unit") == "CNY":
        return data
    out = dict(data)
    provenance = out.get("provenance") if isinstance(out.get("provenance"), dict) else {}
    key = _source_key(
        source,
        out.get("source"),
        out.get("source_name"),
        out.get("provider"),
        provenance.get("source"),
        provenance.get("original_source"),
    )
    amount_factor, table_volume_factor = _MARKET_UNITS.get(key, (None, None))
    amount, volume, close = _number(out.get("amount")), _number(out.get("volume")), _number(out.get("close"))
    if amount_factor is None and amount is not None:
        amount_factor = 1.0  # sources without a declared convention (seeds, fund NAV feeds) use CNY
    # Volume units differ between endpoints of the same provider (AKShare stocks: lots, ETFs: shares), so the
    # data decides when it can: turnover ~= close x shares. The declared unit and the provider table are
    # fallbacks for rows without turnover.
    volume_factor = None
    inferred = False
    if volume and amount and close:
        ratio = amount * (amount_factor or 1.0) / (close * volume)
        volume_factor = 100.0 if 30 <= ratio <= 300 else 1.0 if 0.3 <= ratio <= 3 else None
        inferred = volume_factor is not None
    if volume_factor is None:
        volume_factor = _VOLUME_UNIT_FACTORS.get(str(out.get("volume_unit") or ""), table_volume_factor)
    units_source = {}
    if amount is not None and amount_factor is not None:
        out["amount"] = round(amount * amount_factor, 2)
        out["amount_unit"] = "CNY"
        units_source["amount"] = {1000.0: "thousand CNY", 1.0: "CNY"}.get(amount_factor, f"x{amount_factor:g} CNY")
    if volume is not None:
        if volume_factor is not None:
            out["volume"] = round(volume * volume_factor, 2)
            out["volume_unit"] = "share"
            units_source["volume"] = ("lot (100 shares)" if volume_factor == 100.0 else "share") + (
                " (from turnover / close)" if inferred else ""
            )
        else:
            out["volume_unit"] = "unknown"
    if "pct_change_1d" in out:
        out["change_unit"] = "%"
    if amount is None:
        out.setdefault("amount_unit", "CNY")
    if units_source:
        out["units_source"] = {"provider": key or "unknown", **units_source}
    return out


def period_label(report_date: Any) -> str | None:
    """``2024-12-31`` -> ``FY2024``; ``2025-06-30`` -> ``2025H1``; ``2025-03-31``/``09-30`` -> ``2025Q1``/``Q3``."""
    match = re.match(r"^(\d{4})-?(\d{2})-?(\d{2})", str(report_date or ""))
    if not match:
        return None
    year, month = match.group(1), match.group(2)
    return {"12": f"FY{year}", "06": f"{year}H1", "03": f"{year}Q1", "09": f"{year}Q3"}.get(month, f"{year}-{month}")


def metric_unit(name: str) -> str | None:
    key = name.lower()
    if key in PERCENT_METRICS or key.endswith(("_yoy", "_margin", "_pct")):
        return "%"
    if key in MULTIPLE_METRICS:
        return "x"
    if key in CNY_METRICS:
        return "CNY"
    if key in PER_SHARE_METRICS:
        return "CNY/share"
    return None


def normalise_metrics(
    metrics: dict[str, Any], *, source: str | None = None
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """``(metrics, units, inferred)``: ratios in percent, and the unit of every recognised metric."""
    known_percent = any(name in str(source or "").lower() for name in _PERCENT_SOURCES)
    out = dict(metrics)
    units: dict[str, str] = {}
    inferred: list[str] = []
    for key, value in metrics.items():
        unit = metric_unit(str(key))
        if unit is None:
            continue
        units[str(key)] = unit
        number = _number(value)
        if (
            str(key).lower() in _FRACTION_CANDIDATES
            and number is not None
            and not known_percent
            and 0 < abs(number) <= _FRACTION_LIMIT
        ):
            out[key] = round(number * 100, 4)
            inferred.append(str(key))
    return out, units, inferred


def normalise_fundamentals(data: dict[str, Any]) -> dict[str, Any]:
    """Explicit units and a period label for ``get_fundamentals`` data (idempotent)."""
    if "metric_units" in data:
        return data
    out = dict(data)
    provenance = out.get("provenance") if isinstance(out.get("provenance"), dict) else {}
    source = " ".join(str(item or "") for item in (out.get("source"), provenance.get("source")))
    metrics, units, inferred = normalise_metrics(dict(out.get("metrics") or {}), source=source)
    out["metrics"] = metrics
    out["metric_units"] = units
    if inferred:
        out["units_inferred"] = inferred
    out["period"] = period_label(out.get("report_date"))
    industry = out.get("industry")
    if isinstance(industry, dict) and "metric_units" not in industry:
        industry_metrics, industry_units, _ = normalise_metrics(dict(industry.get("metrics") or {}))
        out["industry"] = {**industry, "metrics": industry_metrics, "metric_units": industry_units}
    return out


_UNIT_WORDS = {
    "%": "in percent (2.5 means 2.5%)",
    "x": "a multiple (times, 倍)",
    "CNY": "in CNY (yuan, not 万 or 亿)",
    "CNY/share": "in CNY per share",
}


def _metric_units_in_words(units: Any, label: str = "") -> list[str]:
    if not isinstance(units, dict):
        return []
    return [f"{label}{key} {_UNIT_WORDS.get(str(unit), f'in {unit}')}" for key, unit in units.items()]


def units_in_words(data: Any) -> str | None:
    """The units of a normalised price or fundamentals payload as one plain sentence for the LLM, else ``None``.

    The payload fields (``amount_unit``, ``metric_units``, ``units_source``) are terse codes; a model reading
    ``"units_source": {"amount": "thousand CNY"}`` next to ``"amount": 3793827534`` could take the turnover to
    be in thousands of CNY and scale it again. The sentence says which unit every value is in *now* and that
    ``units_source`` only records the provider's unit before conversion.
    """
    if not isinstance(data, dict):
        return None
    parts: list[str] = []
    if data.get("amount_unit") == "CNY" or "change_unit" in data or "volume_unit" in data:
        if any(key in data for key in ("close", "open", "high", "low")):
            parts.append("close/open/high/low are prices in CNY per share (per unit for funds)")
        if data.get("amount") is not None:
            parts.append("amount is turnover (成交额) in CNY (yuan)")
        if data.get("volume_unit") == "share":
            parts.append("volume (成交量) in shares")
        elif data.get("volume_unit") == "unknown":
            parts.append("volume in an unknown unit (do not convert or compare it)")
        if data.get("change_unit") == "%":
            parts.append("pct_change_1d in percent (-0.18 means -0.18%)")
    parts.extend(_metric_units_in_words(data.get("metric_units")))
    industry = data.get("industry")
    if isinstance(industry, dict):
        name = industry.get("industry_name") or "industry"
        parts.extend(_metric_units_in_words(industry.get("metric_units"), f"{name} industry "))
    if not parts:
        return None
    sentence = "Units: " + "; ".join(parts) + "."
    if isinstance(data.get("units_source"), dict):
        sentence += (
            " These values are already converted; units_source only records the data provider's original unit "
            "before conversion (for audit) and must not be applied again."
        )
    if data.get("units_inferred"):
        sentence += f" {', '.join(map(str, data['units_inferred']))} arrived as a fraction and is shown in percent."
    return sentence


def _normalise_evidence_payload(tool: str, evidence_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if tool == "get_price_history":
        return normalise_market(payload)
    if tool == "get_fundamentals":
        if "metric_units" in payload:
            return payload
        if evidence_id.startswith("industry_"):
            metrics, units, _ = normalise_metrics(payload)
            return {**metrics, "metric_units": units}
        source = " ".join(str(payload.get(key) or "") for key in ("source_name", "provider"))
        metrics, units, inferred = normalise_metrics(payload, source=source)
        extra: dict[str, Any] = {"metric_units": units, "period": period_label(payload.get("report_date"))}
        if inferred:
            extra["units_inferred"] = inferred
        return {**metrics, **extra}
    return payload


def normalise_sentiment(data: dict[str, Any]) -> dict[str, Any]:
    """Make ``overall_label`` agree with ``label_counts`` (also for results recorded before that rule)."""
    counts = data.get("label_counts")
    if not isinstance(counts, dict):
        return data
    from .sentiment import overall_label

    label = overall_label(counts)
    return data if data.get("overall_label") == label else {**data, "overall_label": label}


def normalise_output(tool: str, output: ToolOutput) -> ToolOutput:
    """Apply the unit rules to a tool's data and to the payloads of its structured evidence."""
    if tool == "analyze_sentiment":
        data = output.data.model_dump(mode="json") if isinstance(output.data, BaseModel) else output.data
        evidence = [
            item.model_copy(update={"payload": normalise_sentiment(item.payload)})
            if item.source_type == "sentiment_summary" and isinstance(item.payload, dict)
            else item
            for item in output.evidence
        ]
        return ToolOutput(data=normalise_sentiment(data) if isinstance(data, dict) else data, evidence=evidence)
    if tool not in {"get_price_history", "get_fundamentals"}:
        return output
    data = output.data.model_dump(mode="json") if isinstance(output.data, BaseModel) else output.data
    if isinstance(data, dict):
        data = normalise_market(data) if tool == "get_price_history" else normalise_fundamentals(data)
    evidence = []
    for item in output.evidence:
        if item.kind == "structured" and isinstance(item.payload, dict):
            payload = _normalise_evidence_payload(tool, item.evidence_id, item.payload)
            item = item.model_copy(update={"payload": payload}) if payload is not item.payload else item
        evidence.append(item)
    return ToolOutput(data=data, evidence=evidence)
