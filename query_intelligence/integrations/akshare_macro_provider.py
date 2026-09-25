"""China macro indicators with ordered live sources, caching, and provenance.

Audit 2026-09-25 (docs/data-sources.md): the jin10-backed ``macro_china_cpi_monthly`` /
``macro_china_pmi_yearly`` series stopped updating in 2025-09 and take 20-35 s, and
``macro_china_pmi_monthly`` no longer exists in akshare 1.18. The NBS-derived Eastmoney datacenter
tables (``macro_china_cpi`` / ``macro_china_pmi`` / ``macro_china_money_supply``) are current and
answer in well under a second, so they are the primary sources. ChinaBond is an independent secondary
for the 10-year yield. National Bureau of Statistics' own query API rejects scripted clients (HTTP
403), so there is no independent live secondary for CPI/PMI/M2: the chain continues with the
last-known-good cache and finally the shipped snapshot (handled by the retrieval pipeline). The
jin10 series are deliberately not used as secondaries: they are a year stale and slower than the call
timeout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .sources.catalog import source_for_endpoint
from .sources.provenance import build_provenance, freshness
from .sources.runtime import AllSourcesFailedError, Candidate, SourceRuntime, get_default_runtime
from .sources.values import to_iso_date, to_number


def _rows_to_records(rows: object) -> list[dict]:
    if rows is None:
        return []
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if hasattr(rows, "to_dict"):
        return rows.to_dict("records")
    return []


@dataclass(frozen=True)
class _Series:
    """How to read one indicator from one upstream table."""

    method: str
    date_keys: tuple[str, ...]
    value_keys: tuple[str, ...]
    kwargs: Callable[[], dict] = dict
    row_filter: Callable[[dict], bool] | None = None
    frequency: str = "monthly"


@dataclass(frozen=True)
class _Indicator:
    code: str
    name: str
    unit: str
    kind: str
    series: tuple[_Series, ...]


def _recent_start() -> dict:
    return {"start_date": (date.today() - timedelta(days=60)).strftime("%Y%m%d")}


def _chinabond_window() -> dict:
    today = date.today()
    return {"start_date": (today - timedelta(days=30)).strftime("%Y%m%d"), "end_date": today.strftime("%Y%m%d")}


INDICATORS: dict[str, _Indicator] = {
    "CPI_CN": _Indicator(
        "CPI_CN",
        "中国CPI同比",
        "%",
        "macro_monthly",
        (_Series("macro_china_cpi", ("月份",), ("全国-同比增长",)),),
    ),
    "PMI_CN": _Indicator(
        "PMI_CN",
        "中国制造业PMI",
        "",
        "macro_monthly",
        (_Series("macro_china_pmi", ("月份",), ("制造业-指数",)),),
    ),
    "M2_CN": _Indicator(
        "M2_CN",
        "中国M2同比",
        "%",
        "macro_monthly",
        (
            # Only the YoY growth column: the quantity column (亿元) must never be reported as a percent.
            _Series("macro_china_money_supply", ("月份",), ("货币和准货币(M2)-同比增长", "M2同比")),
        ),
    ),
    "CN10Y": _Indicator(
        "CN10Y",
        "中国10年期国债收益率",
        "%",
        "macro_daily",
        (
            _Series("bond_zh_us_rate", ("日期",), ("中国国债收益率10年",), kwargs=_recent_start, frequency="daily"),
            _Series(
                "bond_china_yield",
                ("日期",),
                ("10年",),
                kwargs=_chinabond_window,
                row_filter=lambda row: row.get("曲线名称") == "中债国债收益率曲线",
                frequency="daily",
            ),
        ),
    ),
    "LPR1Y_CN": _Indicator(
        "LPR1Y_CN",
        "1年期贷款市场报价利率(LPR)",
        "%",
        "macro_monthly",
        (_Series("macro_china_lpr", ("TRADE_DATE",), ("LPR1Y",)),),
    ),
    "LPR5Y_CN": _Indicator(
        "LPR5Y_CN",
        "5年期以上贷款市场报价利率(LPR)",
        "%",
        "macro_monthly",
        (_Series("macro_china_lpr", ("TRADE_DATE",), ("LPR5Y",)),),
    ),
}

_CODE_TERMS = {
    "CPI_CN": ("cpi", "通胀", "物价"),
    "PMI_CN": ("pmi", "制造业", "景气"),
    "M2_CN": ("m2", "货币", "流动性", "社融", "降准", "降息", "lpr"),
    "CN10Y": ("cn10y", "10y", "十年期", "国债", "利率", "收益率", "降息", "lpr"),
    "LPR1Y_CN": ("lpr", "贷款利率", "贷款市场报价利率", "降息", "房贷"),
    "LPR5Y_CN": ("lpr", "贷款利率", "贷款市场报价利率", "降息", "房贷"),
}

_CACHE_TTL_S = {"macro_monthly": 6 * 3600.0, "macro_daily": 1800.0}


@dataclass
class AKShareMacroProvider:
    ak_module: object
    runtime: SourceRuntime = field(default_factory=SourceRuntime)
    # Reject rows older than the freshness window for the indicator (e.g. a dead feed).
    reject_stale: bool = True

    @classmethod
    def from_import(cls, *, runtime: SourceRuntime | None = None) -> AKShareMacroProvider:
        import akshare as ak

        return cls(ak_module=ak, runtime=runtime or get_default_runtime())

    def fetch_indicators(self, query_bundle: dict) -> list[dict]:
        items, _failures = self.fetch_indicators_with_status(query_bundle)
        return items

    def fetch_indicators_with_status(self, query_bundle: dict) -> tuple[list[dict], dict[str, str]]:
        """Live items plus ``{indicator_code: reason}`` for requested indicators no live source served."""
        items: list[dict] = []
        failures: dict[str, str] = {}
        for code in self._requested_codes(query_bundle):
            payload, reason = self._fetch_payload(code)
            if reason:
                failures[code] = reason
            if payload:
                items.append(
                    {
                        "evidence_id": f"macro_live_{payload['indicator_code']}",
                        "source_type": "macro_indicator",
                        "source_name": payload["source_name"],
                        "provider": payload["provider"],
                        "payload": payload,
                    }
                )
        return items, failures

    def _requested_codes(self, query_bundle: dict) -> list[str]:
        text = self._query_text(query_bundle)
        matched = [code for code, terms in _CODE_TERMS.items() if any(term in text for term in terms)]
        if any(term in text for term in ("降准", "降息", "lpr")):
            for code in ("M2_CN", "CN10Y"):
                if code not in matched:
                    matched.append(code)
        return matched

    def _fetch_payload(self, code: str) -> tuple[dict | None, str | None]:
        indicator = INDICATORS.get(code)
        if indicator is None:
            return None, None
        candidates = [
            Candidate(
                source_id=source_for_endpoint(f"akshare.{series.method}"),
                endpoint=f"akshare.{series.method}",
                fetch=self._reader(series),
            )
            for series in indicator.series
            if callable(getattr(self.ak_module, series.method, None))
        ]
        if not candidates:
            return None, "no live source available"
        try:
            result = self.runtime.run_chain(
                "macro",
                code,
                candidates,
                ttl_s=_CACHE_TTL_S.get(indicator.kind, 3600.0),
                validate=lambda row: self._usable(indicator, row),
            )
        except AllSourcesFailedError as exc:
            return None, "; ".join(exc.attempts) or "all live sources failed"
        row = result.value
        metric_date = row["metric_date"]
        payload = {
            "indicator_code": indicator.code,
            "indicator_name": indicator.name,
            "metric_date": metric_date,
            "metric_value": row["metric_value"],
            "unit": indicator.unit,
            "source_name": "akshare",
            "provider": "akshare",
            "provider_endpoint": result.endpoint,
            "release_period": metric_date[:7] if metric_date else None,
            "frequency": row["frequency"],
            "provenance": build_provenance(
                source=result.source_id,
                kind=indicator.kind,
                as_of=metric_date,
                mode=result.mode,
                endpoint=result.endpoint,
                fetched_at=result.fetched_at,
                fallback_reason=result.fallback_reason,
                attempts=result.attempts,
                cache_hit=result.cache_hit,
            ),
        }
        return payload, None

    def _reader(self, series: _Series) -> Callable[[], dict | None]:
        def read() -> dict | None:
            rows = _rows_to_records(getattr(self.ak_module, series.method)(**series.kwargs()))
            if series.row_filter is not None:
                rows = [row for row in rows if series.row_filter(row)]
            return self._latest_value(rows, series)

        return read

    def _latest_value(self, rows: list[dict], series: _Series) -> dict | None:
        """Latest row that actually carries a value (upcoming releases appear as NaN rows)."""
        best: tuple[str, float] | None = None
        for row in rows:
            metric_date = next((to_iso_date(row[key]) for key in series.date_keys if key in row), None)
            value = next((to_number(row[key]) for key in series.value_keys if key in row), None)
            if metric_date is None or value is None:
                continue
            if best is None or metric_date > best[0]:
                best = (metric_date, value)
        if best is None:
            return None
        return {"metric_date": best[0], "metric_value": best[1], "frequency": series.frequency}

    def _usable(self, indicator: _Indicator, row: Any) -> bool:
        if not isinstance(row, dict) or row.get("metric_value") is None:
            return False
        return not self.reject_stale or freshness(indicator.kind, row.get("metric_date")) != "stale"

    def _query_text(self, query_bundle: dict) -> str:
        parts = [
            query_bundle.get("normalized_query", ""),
            *query_bundle.get("keywords", []),
            *query_bundle.get("entity_names", []),
        ]
        return " ".join(str(part) for part in parts).lower()
