from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from urllib.parse import urlencode

import requests

from .efinance_provider import EFinanceETFProvider
from .sources.catalog import source_for_endpoint
from .sources.health import CircuitOpenError
from .sources.provenance import LIVE, LIVE_FALLBACK, build_provenance
from .sources.runtime import SourceRuntime, SourceTimeoutError, fallback_reason_from, get_default_runtime
from .sources.values import is_missing, to_iso_date, to_number

logger = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)",
}
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="

# Volume units differ between upstreams (verified 2026-09-25 for 2026-09-24 bars: 600519 Sina 3,123,900
# vs Tencent 31,239; 510300 Sina 710,251,931 vs Tencent 7,102,519; 000300 Sina 15,196,578,700 vs
# Tencent 151,965,787). Eastmoney/efinance/Tencent report lots (手, 100 shares); Sina reports shares.
_VOLUME_UNITS = {
    "eastmoney.quote": "lot",
    "efinance": "lot",
    "tencent.kline": "lot",
    "sina.kline": "share",
    "sina.quote": "share",
}
_PRODUCT_KIND = {"stock": "market", "etf": "market", "fund": "fund_nav", "index": "index"}


def _rows_to_records(rows):
    if rows is None:
        return []
    if isinstance(rows, list):
        return rows
    if hasattr(rows, "to_dict"):
        return rows.to_dict("records")
    return []


def _default_http_get(url: str, headers: dict, timeout: float):
    # Resolved at call time so tests can monkeypatch ``requests.get``.
    return requests.get(url, headers=headers, timeout=timeout)


@dataclass
class AKShareMarketProvider:
    ak_module: object
    efinance_provider: EFinanceETFProvider | None = None
    timeout: int = 15
    max_retries: int = 1
    retry_backoff_seconds: float = 0.25
    # Health/circuit breaker/trace runtime. Directly constructed providers get a private runtime so
    # unit tests do not share breaker state; ``from_import`` uses the process-wide runtime.
    runtime: SourceRuntime = field(default_factory=SourceRuntime)
    # HTTP getter for the direct (non-akshare) Tencent sources. ``None`` disables them, which keeps
    # fake-module unit tests offline; ``from_import`` enables them.
    http_get: Callable | None = None

    @classmethod
    def from_import(
        cls,
        *,
        timeout: int = 15,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.25,
        runtime: SourceRuntime | None = None,
    ) -> AKShareMarketProvider:
        import akshare as ak

        return cls(
            ak_module=ak,
            efinance_provider=EFinanceETFProvider.from_import(),
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            runtime=runtime or get_default_runtime(),
            http_get=_default_http_get,
        )

    def fetch_bundle(
        self,
        symbol: str,
        canonical_name: str,
        product_type: str,
        start_date: str = "20250101",
        end_date: str = "20261231",
    ) -> dict:
        plain_symbol = symbol.split(".")[0]
        with self.runtime.trace() as market_attempts:
            market_rows, market_source_name, provider_warnings, market_endpoint = self._fetch_market_rows(
                plain_symbol, product_type, start_date, end_date
            )
        latest = market_rows[0] if market_rows else {}
        is_fund_product = product_type in {"etf", "fund"}
        is_index_product = product_type == "index"
        company_info = self._safe_fetch_company_info(plain_symbol) if product_type == "stock" else {}
        industry_name = company_info.get("industry_name")
        industry_payload = None
        if industry_name and company_info.get("industry_source") == "eastmoney":
            industry_payload = self._safe_fetch_industry_snapshot(industry_name)
        if industry_name and not industry_payload:
            industry_payload = self._identity_industry_snapshot(plain_symbol, industry_name, company_info)
        fundamental_payload = self._safe_fetch_fundamental_payload(plain_symbol) if product_type == "stock" else {}
        fund_payloads = self._safe_fetch_fund_payloads(plain_symbol) if is_fund_product else {}
        index_payloads = self._safe_fetch_index_payloads(plain_symbol, market_rows) if is_index_product else {}
        market_trace = self._market_api_trace(plain_symbol, product_type, market_endpoint, start_date, end_date)
        market_source_id = source_for_endpoint(market_endpoint)

        payload = {
            "symbol": symbol,
            "source_name": market_source_name,
            **market_trace,
            "canonical_name": canonical_name,
            "trade_date": latest.get("trade_date"),
            "open": latest.get("open"),
            "high": latest.get("high"),
            "low": latest.get("low"),
            "close": latest.get("close"),
            "pct_change_1d": latest.get("pct_change_1d"),
            "volume": latest.get("volume"),
            "amount": latest.get("amount"),
            "history": market_rows[:30],
            "industry_name": industry_name,
        }
        if product_type in {"stock", "etf", "index"} and market_rows and market_source_id in _VOLUME_UNITS:
            payload["volume_unit"] = _VOLUME_UNITS[market_source_id]
        payload["provenance"] = self._provenance(
            kind=_PRODUCT_KIND.get(product_type, "market"),
            attempts=market_attempts,
            source_id=market_source_id if market_rows else None,
            endpoint=market_endpoint if market_rows else None,
            as_of=latest.get("trade_date"),
        )
        if provider_warnings:
            payload["provider_warnings"] = provider_warnings
        if industry_payload:
            payload["industry_snapshot"] = industry_payload

        bundle = {
            "source_type": "market_api",
            "source_name": market_source_name,
            "payload": payload,
            "fundamental_payload": fundamental_payload,
            "provider_warnings": provider_warnings,
            "request_trace": {
                "symbol": symbol,
                "product_type": product_type,
                "start_date": start_date,
                "end_date": end_date,
            },
            "status": "degraded" if provider_warnings or not market_rows else "ok",
        }
        bundle.update(fund_payloads)
        bundle.update(index_payloads)
        if "index_daily_payload" in bundle:
            # Index daily bars are the market rows: same endpoint, same provenance.
            bundle["index_daily_payload"].update(market_trace)
            bundle["index_daily_payload"]["provenance"] = dict(payload["provenance"])
        return bundle

    # ---- provenance helpers -------------------------------------------------

    def _provenance(
        self,
        *,
        kind: str,
        attempts: list[str],
        source_id: str | None,
        endpoint: str | None,
        as_of,
    ) -> dict:
        if source_id is None:
            return build_provenance(
                source=None,
                kind=kind,
                as_of=None,
                mode=LIVE,
                endpoint=None,
                fallback_reason=fallback_reason_from(attempts) or "no live source returned data",
                attempts=attempts,
            )
        # Attempts before the serving source's success, excluding retries of that same source.
        ok_label = f"{source_id}:ok"
        served_at = max((i for i, item in enumerate(attempts) if item == ok_label), default=len(attempts))
        preceding = [item for item in attempts[:served_at] if not item.startswith(f"{source_id}:")]
        return build_provenance(
            source=source_id,
            kind=kind,
            as_of=as_of,
            mode=LIVE if not preceding else LIVE_FALLBACK,
            endpoint=endpoint,
            fallback_reason=fallback_reason_from(preceding),
            attempts=attempts,
        )

    def _api_trace(self, endpoint: str, query_params: dict) -> dict:
        encoded_params = urlencode(query_params, doseq=True)
        return {
            "provider": self._provider_name_for_endpoint(endpoint),
            "provider_endpoint": endpoint,
            "query_params": query_params,
            "source_reference": f"api://{endpoint}?{encoded_params}" if encoded_params else f"api://{endpoint}",
        }

    def _market_api_trace(
        self,
        symbol: str,
        product_type: str,
        endpoint: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        query_params: dict[str, str] = {"symbol": symbol}
        if endpoint in {"akshare.stock_zh_a_hist", "akshare.fund_etf_hist_em", "akshare.stock_zh_a_daily"}:
            query_params.update({"period": "daily", "start_date": start_date, "end_date": end_date, "adjust": ""})
        elif endpoint in {"akshare.stock_zh_index_daily", "akshare.fund_etf_hist_sina"}:
            query_params = {
                "symbol": self._prefixed_index_symbol(symbol) if product_type == "index" else self._prefixed(symbol)
            }
        elif endpoint == "akshare.index_zh_a_hist":
            query_params.update({"period": "daily", "start_date": start_date, "end_date": end_date})
        elif endpoint == "akshare.fund_open_fund_info_em":
            query_params["indicator"] = "单位净值走势"
        elif endpoint == "sina.hq_sinajs_cn":
            query_params = {"list": self._prefixed(symbol)}
        elif endpoint == "tencent.fqkline":
            prefixed = self._prefixed_index_symbol(symbol) if product_type == "index" else self._prefixed(symbol)
            query_params = {"param": self._tencent_param(prefixed, start_date, end_date)}
        elif endpoint in {"efinance.stock.get_quote_history", "efinance.fund.get_quote_history"}:
            query_params.update({"beg": start_date, "end": end_date})
        return self._api_trace(endpoint, query_params)

    def _default_market_endpoint(self, product_type: str) -> str:
        if product_type == "etf":
            return "akshare.fund_etf_hist_em"
        if product_type == "fund":
            return "akshare.fund_open_fund_info_em"
        if product_type == "index":
            return "akshare.stock_zh_index_daily"
        return "akshare.stock_zh_a_hist"

    def _provider_name_for_endpoint(self, endpoint: str) -> str:
        if endpoint.startswith("efinance."):
            return "efinance"
        if endpoint.startswith("sina."):
            return "sina"
        if endpoint.startswith("tencent."):
            return "tencent"
        return "akshare"

    # ---- market rows (ordered fallback chains) ------------------------------

    def _fetch_market_rows(
        self, symbol: str, product_type: str, start_date: str, end_date: str
    ) -> tuple[list[dict], str, list[str], str]:
        provider_warnings: list[str] = []
        endpoint = self._default_market_endpoint(product_type)
        if product_type == "etf":
            rows, source_name, endpoint = self._fetch_etf_rows(symbol, start_date, end_date, provider_warnings)
        elif product_type == "fund":
            rows = self._fetch_fund_rows(symbol, provider_warnings)
            source_name = "akshare"
        elif product_type == "index":
            rows, endpoint = self._fetch_index_rows(symbol, start_date, end_date, provider_warnings)
            source_name = "tencent" if endpoint == "tencent.fqkline" else "akshare"
        else:
            try:
                rows = self._call_akshare(
                    "akshare.stock_zh_a_hist",
                    "stock_zh_a_hist",
                    provider_warnings,
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="",
                )
                source_name = "akshare"
            except Exception:
                # Raises when every secondary fails; the pipeline turns that into a provider warning.
                rows, source_name, endpoint = self._fetch_stock_rows_fallback(
                    symbol, start_date, end_date, provider_warnings
                )
        records = _rows_to_records(rows)
        normalized = []
        for row in records:
            normalized.append(
                {
                    "trade_date": self._normalize_date(row.get("日期") or row.get("date") or row.get("trade_date")),
                    "open": self._first_number(row, "开盘", "open"),
                    "high": self._first_number(row, "最高", "high"),
                    "low": self._first_number(row, "最低", "low"),
                    "close": self._first_number(row, "收盘", "close"),
                    "pct_change_1d": self._first_number(row, "涨跌幅", "pct_change_1d"),
                    "volume": self._first_number(row, "成交量", "volume"),
                    "amount": self._first_number(row, "成交额", "amount"),
                }
            )
            if normalized[-1]["trade_date"] is None:
                normalized[-1]["trade_date"] = self._normalize_date(row.get("净值日期") or row.get("nav_date"))
            if normalized[-1]["close"] is None:
                normalized[-1]["close"] = self._first_number(row, "单位净值", "latest_nav")
            if normalized[-1]["pct_change_1d"] is None:
                normalized[-1]["pct_change_1d"] = self._first_number(row, "日增长率", "pct_change")
        normalized.sort(key=lambda item: item.get("trade_date") or "", reverse=True)
        self._fill_missing_pct_change(normalized)
        if not normalized:
            provider_warnings.append(f"market_provider_empty_rows:{source_name}:{symbol}")
        return normalized, source_name, list(dict.fromkeys(provider_warnings)), endpoint

    def _call_akshare(self, endpoint: str, method_name: str, provider_warnings: list[str], **kwargs):
        fn = getattr(self.ak_module, method_name)
        call_kwargs = dict(kwargs)
        if self._accepts_timeout(fn):
            call_kwargs["timeout"] = self.timeout
        return self._call_with_retry(endpoint, lambda: fn(**call_kwargs), provider_warnings)

    def _call_with_retry(self, endpoint: str, operation, provider_warnings: list[str]):
        """Run ``operation`` through the source runtime (circuit breaker + hard timeout) with retries."""
        source_id = source_for_endpoint(endpoint)
        attempts = max(1, self.max_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                result = self.runtime.call(source_id, operation)
                if attempt > 0 and last_error is not None:
                    provider_warnings.append(f"{endpoint}_retry_succeeded:{self._error_summary(last_error)}")
                return result
            except CircuitOpenError:
                provider_warnings.append(f"{endpoint}_skipped:circuit_open:{source_id}")
                raise
            except Exception as exc:
                last_error = exc
                # A timeout already consumed the full budget; retrying would double the wait.
                if attempt >= attempts - 1 or isinstance(exc, SourceTimeoutError):
                    provider_warnings.append(f"{endpoint}_failed:{self._error_summary(exc)}")
                    logger.warning("Live provider endpoint failed: endpoint=%s error=%s", endpoint, exc)
                    raise
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
        raise RuntimeError(f"{endpoint} failed without error detail")

    def _accepts_timeout(self, fn) -> bool:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        return "timeout" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )

    def _compatible_kwargs(self, fn, *candidates: dict) -> dict | None:
        """First kwargs set the function signature accepts (avoids counting TypeErrors as outages)."""
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return candidates[0] if candidates else None
        for kwargs in candidates:
            try:
                signature.bind(**kwargs)
            except TypeError:
                continue
            return kwargs
        return None

    def _error_summary(self, exc: Exception) -> str:
        text = str(exc).strip().replace("\n", " ")
        if len(text) > 180:
            text = text[:177] + "..."
        return f"{type(exc).__name__}:{text}"

    def _fetch_stock_rows_fallback(
        self, symbol: str, start_date: str, end_date: str, provider_warnings: list[str]
    ) -> tuple[object, str, str]:
        """Secondary chain after Eastmoney: Sina daily -> Tencent daily -> Sina realtime -> efinance."""
        if hasattr(self.ak_module, "stock_zh_a_daily"):
            try:
                return (
                    self._call_akshare(
                        "akshare.stock_zh_a_daily",
                        "stock_zh_a_daily",
                        provider_warnings,
                        symbol=self._prefixed(symbol),
                        start_date=start_date,
                        end_date=end_date,
                        adjust="",
                    ),
                    "akshare_sina",
                    "akshare.stock_zh_a_daily",
                )
            except Exception:
                pass

        try:
            rows = self._fetch_tencent_rows(self._prefixed(symbol), start_date, end_date, provider_warnings)
            if rows:
                return rows, "tencent", "tencent.fqkline"
        except Exception:
            pass

        try:
            return [self._fetch_sina_realtime_row(symbol, provider_warnings)], "sina_quote", "sina.hq_sinajs_cn"
        except Exception:
            pass

        if self.efinance_provider is None:
            raise RuntimeError(f"stock history fetch failed for {symbol}")
        payload = self._call_with_retry(
            "efinance.stock.get_quote_history",
            lambda: self.efinance_provider.fetch_stock_history(self._suffixed(symbol)),
            provider_warnings,
        )
        return (
            payload["payload"].get("history", []),
            payload.get("source_name", "efinance"),
            "efinance.stock.get_quote_history",
        )

    def _fetch_sina_realtime_row(self, symbol: str, provider_warnings: list[str]) -> dict:
        prefixed_symbol = self._prefixed(symbol)
        response = self._call_with_retry(
            "sina.hq_sinajs_cn",
            lambda: requests.get(
                f"https://hq.sinajs.cn/list={prefixed_symbol}",
                headers={"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"},
                timeout=self.timeout,
            ),
            provider_warnings,
        )
        response.raise_for_status()
        _, payload = response.text.split("=", 1)
        fields = payload.strip().strip('";').split(",")
        if len(fields) < 32:
            raise RuntimeError(f"unexpected sina quote payload for {symbol}")
        previous_close = self._to_float(fields[2])
        close = self._to_float(fields[3])
        pct_change = None
        if previous_close and close is not None:
            pct_change = round((close - previous_close) / previous_close * 100, 4)
        return {
            "trade_date": self._normalize_date(fields[30]),
            "open": self._to_float(fields[1]),
            "high": self._to_float(fields[4]),
            "low": self._to_float(fields[5]),
            "close": close,
            "pct_change_1d": pct_change,
            "volume": self._to_float(fields[8]),
            "amount": self._to_float(fields[9]),
        }

    def _tencent_param(self, prefixed_symbol: str, start_date: str, end_date: str) -> str:
        return f"{prefixed_symbol},day,{self._dashed(start_date)},{self._dashed(end_date)},640,"

    def _fetch_tencent_rows(
        self, prefixed_symbol: str, start_date: str, end_date: str, provider_warnings: list[str]
    ) -> list[dict]:
        """Unadjusted daily bars from Tencent (``web.ifzq.gtimg.cn``): [date, open, close, high, low, volume]."""
        if self.http_get is None:
            return []
        url = f"{TENCENT_KLINE_URL}?{urlencode({'param': self._tencent_param(prefixed_symbol, start_date, end_date)})}"
        response = self._call_with_retry(
            "tencent.fqkline",
            lambda: self.http_get(url, headers=_BROWSER_HEADERS, timeout=self.timeout),
            provider_warnings,
        )
        response.raise_for_status()
        data = (response.json().get("data") or {}).get(prefixed_symbol) or {}
        raw_rows = data.get("day") or data.get("qfqday") or []
        rows = []
        for raw in raw_rows:
            if not isinstance(raw, list) or len(raw) < 6:
                continue
            rows.append(
                {
                    "date": raw[0],
                    "open": to_number(raw[1]),
                    "close": to_number(raw[2]),
                    "high": to_number(raw[3]),
                    "low": to_number(raw[4]),
                    "volume": to_number(raw[5]),
                }
            )
        if not rows:
            provider_warnings.append(f"tencent.fqkline_empty:{prefixed_symbol}")
        return rows

    def _fetch_etf_rows(
        self, symbol: str, start_date: str, end_date: str, provider_warnings: list[str]
    ) -> tuple[object, str, str]:
        """ETF chain: Eastmoney -> Sina -> Tencent -> efinance."""
        try:
            rows = self._call_akshare(
                "akshare.fund_etf_hist_em",
                "fund_etf_hist_em",
                provider_warnings,
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="",
            )
            return rows, "akshare", "akshare.fund_etf_hist_em"
        except Exception:
            pass

        if hasattr(self.ak_module, "fund_etf_hist_sina"):
            try:
                rows = self._call_akshare(
                    "akshare.fund_etf_hist_sina", "fund_etf_hist_sina", provider_warnings, symbol=self._prefixed(symbol)
                )
                if rows is not None:
                    return rows, "akshare", "akshare.fund_etf_hist_sina"
            except Exception:
                pass

        try:
            rows = self._fetch_tencent_rows(self._prefixed(symbol), start_date, end_date, provider_warnings)
            if rows:
                return rows, "tencent", "tencent.fqkline"
        except Exception:
            pass

        if self.efinance_provider is not None:
            payload = self._call_with_retry(
                "efinance.fund.get_quote_history",
                lambda: self.efinance_provider.fetch_history(self._suffixed(symbol)),
                provider_warnings,
            )
            return payload["payload"].get("history", []), "akshare", "efinance.fund.get_quote_history"
        raise RuntimeError(f"ETF history fetch failed for {symbol}")

    def _fetch_fund_rows(self, symbol: str, provider_warnings: list[str]):
        if not hasattr(self.ak_module, "fund_open_fund_info_em"):
            return []
        fn = self.ak_module.fund_open_fund_info_em
        kwargs = self._compatible_kwargs(fn, {"symbol": symbol, "indicator": "单位净值走势"}, {"symbol": symbol})
        if kwargs is None:
            return []
        try:
            return self._call_akshare(
                "akshare.fund_open_fund_info_em", "fund_open_fund_info_em", provider_warnings, **kwargs
            )
        except Exception:
            return []

    def _prefixed(self, symbol: str) -> str:
        plain = symbol.split(".")[0]
        return f"sh{plain}" if plain.startswith(("5", "6", "9")) else f"sz{plain}"

    def _suffixed(self, symbol: str) -> str:
        plain = symbol.split(".")[0]
        return f"{plain}.SH" if plain.startswith(("5", "6", "9")) else f"{plain}.SZ"

    def _prefixed_index_symbol(self, symbol: str) -> str:
        """Return Sina-style prefixed symbol for index APIs (sh000001 / sz399001)."""
        plain = symbol.split(".")[0]
        if plain.startswith(("0", "5", "6")):
            return f"sh{plain}"
        return f"sz{plain}"

    def _fetch_index_rows(
        self, symbol: str, start_date: str, end_date: str, provider_warnings: list[str]
    ) -> tuple[list[dict], str]:
        """Index chain: Sina -> Eastmoney history -> Tencent -> Eastmoney spot."""
        prefixed = self._prefixed_index_symbol(symbol)
        steps = (
            ("stock_zh_index_daily", ({"symbol": prefixed},)),
            (
                "index_zh_a_hist",
                (
                    {"symbol": symbol, "period": "daily", "start_date": start_date, "end_date": end_date},
                    {"symbol": symbol},
                ),
            ),
            ("tencent", ()),
            ("stock_zh_index_spot_em", ({},)),
        )
        for method_name, kwargs_options in steps:
            if method_name == "tencent":
                try:
                    rows = self._fetch_tencent_rows(prefixed, start_date, end_date, provider_warnings)
                except Exception:
                    continue
                if rows:
                    return rows, "tencent.fqkline"
                continue
            if not hasattr(self.ak_module, method_name):
                continue
            endpoint = f"akshare.{method_name}"
            kwargs = self._compatible_kwargs(getattr(self.ak_module, method_name), *kwargs_options)
            if kwargs is None:
                continue
            try:
                rows = _rows_to_records(self._call_akshare(endpoint, method_name, provider_warnings, **kwargs))
            except Exception:
                continue
            if rows:
                return self._filter_index_rows(rows, symbol), endpoint
        return [], "akshare.stock_zh_index_daily"

    def _filter_index_rows(self, rows: list[dict], symbol: str) -> list[dict]:
        plain_symbol = symbol.split(".")[0]
        filtered = []
        for row in rows:
            row_code = str(row.get("代码") or row.get("指数代码") or row.get("symbol") or "")
            if row_code and row_code != plain_symbol and row_code.upper() != symbol.upper():
                continue
            filtered.append(row)
        return filtered or rows

    # ---- company profile / industry -------------------------------------------

    def _fetch_company_info(self, symbol: str) -> dict:
        """Eastmoney company info (industry = Eastmoney board name) -> cninfo profile (CSRC industry)."""
        warnings: list[str] = []
        if hasattr(self.ak_module, "stock_individual_info_em"):
            try:
                rows = _rows_to_records(
                    self._call_akshare(
                        "akshare.stock_individual_info_em", "stock_individual_info_em", warnings, symbol=symbol
                    )
                )
                mapping = {}
                for row in rows:
                    key = row.get("item") or row.get("项目")
                    value = row.get("value") or row.get("值")
                    if key:
                        mapping[key] = value
                if mapping.get("行业"):
                    return {
                        "canonical_name": mapping.get("股票简称") or mapping.get("证券简称"),
                        "industry_name": mapping.get("行业"),
                        "industry_source": "eastmoney",
                    }
            except Exception:
                pass
        if hasattr(self.ak_module, "stock_profile_cninfo"):
            rows = _rows_to_records(
                self._call_akshare("akshare.stock_profile_cninfo", "stock_profile_cninfo", warnings, symbol=symbol)
            )
            if rows and not is_missing(rows[0].get("所属行业")):
                return {
                    "canonical_name": rows[0].get("A股简称"),
                    "industry_name": rows[0].get("所属行业"),
                    "industry_source": "cninfo",
                }
        return {}

    def _safe_fetch_company_info(self, symbol: str) -> dict:
        try:
            return self._fetch_company_info(symbol)
        except Exception:
            return {}

    def _fetch_industry_snapshot(self, industry_name: str) -> dict | None:
        if not hasattr(self.ak_module, "stock_board_industry_hist_em"):
            return None
        warnings: list[str] = []
        with self.runtime.trace() as attempts:
            rows = _rows_to_records(
                self._call_akshare(
                    "akshare.stock_board_industry_hist_em",
                    "stock_board_industry_hist_em",
                    warnings,
                    symbol=industry_name,
                    start_date="20250101",
                    end_date="20261231",
                    period="日k",
                    adjust="",
                )
            )
        if not rows:
            return None
        row = rows[-1]
        trace = self._api_trace(
            "akshare.stock_board_industry_hist_em",
            {
                "symbol": industry_name,
                "start_date": "20250101",
                "end_date": "20261231",
                "period": "日k",
                "adjust": "",
            },
        )
        trade_date = self._normalize_date(row.get("日期"))
        return {
            "source_name": "akshare",
            **trace,
            "industry_name": industry_name,
            "trade_date": trade_date,
            "open": to_number(row.get("开盘")),
            "close": to_number(row.get("收盘")),
            "pct_change": to_number(row.get("涨跌幅")),
            "amount": to_number(row.get("成交额")),
            "provenance": self._provenance(
                kind="industry",
                attempts=attempts,
                source_id="eastmoney.quote",
                endpoint="akshare.stock_board_industry_hist_em",
                as_of=trade_date,
            ),
        }

    def _safe_fetch_industry_snapshot(self, industry_name: str) -> dict | None:
        try:
            return self._fetch_industry_snapshot(industry_name)
        except Exception:
            return None

    def _identity_industry_snapshot(self, symbol: str, industry_name: str, company_info: dict | None = None) -> dict:
        from_cninfo = (company_info or {}).get("industry_source") == "cninfo"
        endpoint = "akshare.stock_profile_cninfo" if from_cninfo else "akshare.stock_individual_info_em"
        trace = self._api_trace(endpoint, {"symbol": symbol})
        snapshot = {
            "source_name": "cninfo_company_profile" if from_cninfo else "akshare_company_profile",
            "provider": "akshare",
            **trace,
            "industry_name": industry_name,
            "coverage_level": "identity_only",
            "provenance": build_provenance(
                source=source_for_endpoint(endpoint),
                kind="industry",
                as_of=None,
                endpoint=endpoint,
                fallback_reason="industry board history unavailable; industry name only",
            ),
        }
        if from_cninfo:
            snapshot["industry_classification"] = "CSRC (证监会行业分类)"
        return snapshot

    # ---- fundamentals and valuation ------------------------------------------------

    def _fetch_fundamental_payload(self, symbol: str) -> dict:
        """Financial indicators: Sina -> THS; valuation (PE-TTM/PB): Eastmoney datacenter -> Tencent quote."""
        warnings: list[str] = []
        start_year = str(date.today().year - 1)
        report: dict | None = None
        endpoint = "akshare.stock_financial_analysis_indicator"
        query_params: dict = {"symbol": symbol, "start_year": start_year}
        with self.runtime.trace() as attempts:
            if hasattr(self.ak_module, "stock_financial_analysis_indicator"):
                try:
                    rows = _rows_to_records(
                        self._call_akshare(
                            endpoint,
                            "stock_financial_analysis_indicator",
                            warnings,
                            symbol=symbol,
                            start_year=start_year,
                        )
                    )
                    report = self._sina_financial_report(rows)
                except Exception:
                    report = None
            if report is None and hasattr(self.ak_module, "stock_financial_abstract_ths"):
                endpoint = "akshare.stock_financial_abstract_ths"
                query_params = {"symbol": symbol, "indicator": "按报告期"}
                try:
                    rows = _rows_to_records(
                        self._call_akshare(endpoint, "stock_financial_abstract_ths", warnings, **query_params)
                    )
                    report = self._ths_financial_report(rows)
                except Exception:
                    report = None
        if report is None:
            return {}
        valuation, valuation_provenance = self._fetch_valuation(symbol)
        trace = self._api_trace(endpoint, query_params)
        payload = {
            "source_name": "akshare",
            **trace,
            **report,
            "pe_ttm": valuation.get("pe_ttm"),
            "pb": valuation.get("pb"),
            "provenance": self._provenance(
                kind="fundamentals",
                attempts=attempts,
                source_id=source_for_endpoint(endpoint),
                endpoint=endpoint,
                as_of=report.get("report_date"),
            ),
        }
        if valuation.get("valuation_date"):
            payload["valuation_date"] = valuation["valuation_date"]
        if valuation_provenance:
            payload["valuation_provenance"] = valuation_provenance
        return payload

    def _sina_financial_report(self, rows: list[dict]) -> dict | None:
        dated = [row for row in rows if self._normalize_date(row.get("日期"))]
        if not dated:
            return None
        latest = max(dated, key=lambda row: self._normalize_date(row.get("日期")) or "")
        return {
            "report_date": self._normalize_date(latest.get("日期")),
            "roe": self._first_number(latest, "净资产收益率(%)"),
            "grossprofit_margin": self._first_number(latest, "销售毛利率(%)", "主营业务毛利率(%)"),
            "eps": self._first_number(latest, "加权每股收益(元)", "每股收益(元)", "摊薄每股收益(元)"),
            "netprofit_yoy": self._first_number(latest, "净利润增长率(%)"),
            "revenue_yoy": self._first_number(latest, "主营业务收入增长率(%)"),
            "profit_dedt": self._first_number(latest, "扣除非经常性损益后的净利润(元)"),
        }

    def _ths_financial_report(self, rows: list[dict]) -> dict | None:
        dated = [row for row in rows if self._normalize_date(row.get("报告期"))]
        if not dated:
            return None
        latest = max(dated, key=lambda row: self._normalize_date(row.get("报告期")) or "")
        return {
            "report_date": self._normalize_date(latest.get("报告期")),
            "roe": self._first_number(latest, "净资产收益率-摊薄", "净资产收益率"),
            "grossprofit_margin": self._first_number(latest, "销售毛利率"),
            "eps": self._first_number(latest, "基本每股收益"),
            "netprofit_yoy": self._first_number(latest, "净利润同比增长率"),
            "revenue_yoy": self._first_number(latest, "营业总收入同比增长率"),
            "profit_dedt": self._first_number(latest, "扣非净利润"),
            "revenue": self._first_number(latest, "营业总收入"),
            "net_profit": self._first_number(latest, "净利润"),
        }

    def _fetch_valuation(self, symbol: str) -> tuple[dict, dict | None]:
        warnings: list[str] = []
        with self.runtime.trace() as attempts:
            if hasattr(self.ak_module, "stock_value_em"):
                try:
                    rows = _rows_to_records(
                        self._call_akshare("akshare.stock_value_em", "stock_value_em", warnings, symbol=symbol)
                    )
                    dated = [row for row in rows if self._normalize_date(row.get("数据日期"))]
                    if dated:
                        latest = max(dated, key=lambda row: self._normalize_date(row.get("数据日期")) or "")
                        valuation = {
                            "pe_ttm": self._first_number(latest, "PE(TTM)"),
                            "pb": self._first_number(latest, "市净率"),
                            "valuation_date": self._normalize_date(latest.get("数据日期")),
                        }
                        if valuation["pe_ttm"] is not None or valuation["pb"] is not None:
                            return valuation, self._valuation_provenance(attempts, "akshare.stock_value_em", valuation)
                except Exception:
                    pass
            try:
                valuation = self._fetch_tencent_valuation(symbol, warnings)
            except Exception:
                valuation = {}
            if valuation:
                return valuation, self._valuation_provenance(attempts, "tencent.qt_quote", valuation)
        return {}, None

    def _valuation_provenance(self, attempts: list[str], endpoint: str, valuation: dict) -> dict:
        return self._provenance(
            kind="valuation",
            attempts=list(attempts),
            source_id=source_for_endpoint(endpoint),
            endpoint=endpoint,
            as_of=valuation.get("valuation_date"),
        )

    def _fetch_tencent_valuation(self, symbol: str, provider_warnings: list[str]) -> dict:
        """PE(TTM) and PB from the Tencent quote (fields 39 and 46, timestamp field 30)."""
        if self.http_get is None:
            return {}
        prefixed = self._prefixed(symbol)
        response = self._call_with_retry(
            "tencent.qt_quote",
            lambda: self.http_get(f"{TENCENT_QUOTE_URL}{prefixed}", headers=_BROWSER_HEADERS, timeout=self.timeout),
            provider_warnings,
        )
        response.raise_for_status()
        text = response.content.decode("gbk", errors="replace") if hasattr(response, "content") else response.text
        _, _, body = text.partition("=")
        fields = body.strip().strip('";').split("~")
        if len(fields) < 47 or fields[2] != symbol:
            return {}
        pe_ttm, pb = to_number(fields[39]), to_number(fields[46])
        if pe_ttm is None and pb is None:
            return {}
        return {"pe_ttm": pe_ttm, "pb": pb, "valuation_date": self._normalize_date(fields[30][:8])}

    def _safe_fetch_fundamental_payload(self, symbol: str) -> dict:
        try:
            return self._fetch_fundamental_payload(symbol)
        except Exception:
            return {}

    # ---- funds and indices -------------------------------------------------------

    def _safe_fetch_fund_payloads(self, symbol: str) -> dict:
        with self.runtime.trace() as detail_attempts:
            try:
                details, detail_endpoint = self._fetch_fund_detail_mapping(symbol)
            except Exception:
                details, detail_endpoint = {}, None
        try:
            nav_payload = self._fetch_fund_nav_payload(symbol)
        except Exception:
            nav_payload = self._empty_fund_nav_payload(symbol)
        detail_provenance = self._provenance(
            kind="fund_profile",
            attempts=detail_attempts,
            source_id=source_for_endpoint(detail_endpoint) if detail_endpoint else None,
            endpoint=detail_endpoint,
            as_of=None,
        )
        payloads = {
            "fund_fee_payload": self._build_fund_fee_payload(symbol, details),
            "fund_redemption_payload": self._build_fund_redemption_payload(symbol, details),
            "fund_profile_payload": self._build_fund_profile_payload(symbol, details),
        }
        for payload in payloads.values():
            payload["provenance"] = dict(detail_provenance)
        return {"fund_nav_payload": nav_payload, **payloads}

    def _safe_fetch_index_payloads(self, symbol: str, market_rows: list[dict]) -> dict:
        return {
            "index_daily_payload": self._build_index_daily_payload(symbol, market_rows),
            "index_valuation_payload": self._fetch_index_valuation_payload(symbol),
        }

    def _build_index_daily_payload(self, symbol: str, market_rows: list[dict]) -> dict:
        latest = market_rows[0] if market_rows else {}
        trace = self._api_trace("akshare.stock_zh_index_daily", {"symbol": self._prefixed_index_symbol(symbol)})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "symbol": symbol,
            "trade_date": latest.get("trade_date"),
            "open": latest.get("open"),
            "high": latest.get("high"),
            "low": latest.get("low"),
            "close": latest.get("close"),
            "pct_change_1d": latest.get("pct_change_1d"),
            "volume": latest.get("volume"),
            "amount": latest.get("amount"),
            "history": market_rows[:30],
        }

    def _fetch_index_valuation_payload(self, symbol: str) -> dict:
        rows: list[dict] = []
        used_endpoint = None
        warnings: list[str] = []
        with self.runtime.trace() as attempts:
            for method_name in ("stock_zh_index_value_csindex", "index_value_name_funddb", "index_analysis_daily_sw"):
                if not hasattr(self.ak_module, method_name):
                    continue
                kwargs = self._compatible_kwargs(getattr(self.ak_module, method_name), {"symbol": symbol}, {})
                if kwargs is None:
                    continue
                try:
                    rows = _rows_to_records(
                        self._call_akshare(f"akshare.{method_name}", method_name, warnings, **kwargs)
                    )
                except Exception:
                    continue
                rows = self._filter_index_rows(rows, symbol)
                if rows:
                    used_endpoint = f"akshare.{method_name}"
                    break
        latest = (
            max(rows, key=lambda row: self._normalize_date(row.get("日期") or row.get("date")) or "") if rows else {}
        )
        trace = self._api_trace("akshare.stock_zh_index_value_csindex", {"symbol": symbol})
        valuation_date = self._normalize_date(latest.get("日期") or latest.get("date"))
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "symbol": symbol,
            "valuation_date": valuation_date,
            "pe": self._first_present(latest, "市盈率1", "市盈率", "PE", "pe"),
            "pb": self._first_present(latest, "市净率1", "市净率", "PB", "pb"),
            "dividend_yield": self._first_present(latest, "股息率1", "股息率", "股息率(%)", "dividend_yield"),
            "percentile": self._first_present(latest, "百分位", "估值分位", "pe_percentile", "percentile"),
            "provenance": self._provenance(
                kind="index_valuation",
                attempts=attempts,
                source_id=source_for_endpoint(used_endpoint) if used_endpoint else None,
                endpoint=used_endpoint,
                as_of=valuation_date,
            ),
        }

    def _fetch_fund_nav_payload(self, symbol: str) -> dict:
        rows: list[dict] = []
        warnings: list[str] = []
        with self.runtime.trace() as attempts:
            if hasattr(self.ak_module, "fund_open_fund_info_em"):
                fn = self.ak_module.fund_open_fund_info_em
                kwargs = self._compatible_kwargs(
                    fn, {"symbol": symbol, "indicator": "单位净值走势"}, {"symbol": symbol}
                )
                if kwargs is not None:
                    rows = _rows_to_records(
                        self._call_akshare(
                            "akshare.fund_open_fund_info_em", "fund_open_fund_info_em", warnings, **kwargs
                        )
                    )
        normalized = []
        for row in rows:
            nav_date = self._normalize_date(row.get("净值日期") or row.get("日期") or row.get("nav_date"))
            normalized.append(
                {
                    "nav_date": nav_date,
                    "latest_nav": self._first_number(row, "单位净值", "latest_nav"),
                    "accumulated_nav": self._first_number(row, "累计净值", "accumulated_nav"),
                }
            )
        normalized.sort(key=lambda item: item.get("nav_date") or "", reverse=True)
        latest = normalized[0] if normalized else {}
        trace = self._api_trace("akshare.fund_open_fund_info_em", {"symbol": symbol, "indicator": "单位净值走势"})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "latest_nav": latest.get("latest_nav"),
            "accumulated_nav": latest.get("accumulated_nav"),
            "nav_date": latest.get("nav_date"),
            "history": normalized[:30],
            "provenance": self._provenance(
                kind="fund_nav",
                attempts=attempts,
                source_id="eastmoney.fund" if normalized else None,
                endpoint="akshare.fund_open_fund_info_em" if normalized else None,
                as_of=latest.get("nav_date"),
            ),
        }

    def _empty_fund_nav_payload(self, symbol: str) -> dict:
        trace = self._api_trace("akshare.fund_open_fund_info_em", {"symbol": symbol, "indicator": "单位净值走势"})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "latest_nav": None,
            "accumulated_nav": None,
            "nav_date": None,
            "history": [],
            "provenance": build_provenance(
                source=None, kind="fund_nav", as_of=None, fallback_reason="fund NAV source failed"
            ),
        }

    def _fetch_fund_detail_mapping(self, symbol: str) -> tuple[dict, str | None]:
        """Fund fees/profile: Eastmoney fund overview -> Xueqiu detail -> Eastmoney ETF NAV rows."""
        mapping: dict = {}
        first_endpoint = None
        warnings: list[str] = []
        today = date.today()
        steps = (
            ("fund_overview_em", ({"symbol": symbol},)),
            ("fund_individual_detail_info_xq", ({"symbol": symbol},)),
            (
                "fund_etf_fund_info_em",
                (
                    {
                        "fund": symbol,
                        "start_date": f"{today.year}0101",
                        "end_date": today.strftime("%Y%m%d"),
                    },
                    {"symbol": symbol},
                ),
            ),
        )
        for method_name, kwargs_options in steps:
            if not hasattr(self.ak_module, method_name):
                continue
            if method_name == "fund_individual_detail_info_xq" and mapping.get("管理费率"):
                # Xueqiu now requires a login token (audit 2026-09-25); skip it once fees are known.
                continue
            kwargs = self._compatible_kwargs(getattr(self.ak_module, method_name), *kwargs_options)
            if kwargs is None:
                continue
            try:
                rows = _rows_to_records(self._call_akshare(f"akshare.{method_name}", method_name, warnings, **kwargs))
            except Exception:
                continue
            found = self._rows_to_mapping(rows, symbol)
            if found and first_endpoint is None:
                first_endpoint = f"akshare.{method_name}"
            for key, value in found.items():
                mapping.setdefault(key, value)
        return mapping, first_endpoint

    def _rows_to_mapping(self, rows: list[dict], symbol: str) -> dict:
        mapping = {}
        plain_symbol = symbol.split(".")[0]
        for row in rows:
            row_code = str(row.get("基金代码") or row.get("代码") or row.get("symbol") or row.get("基金简称") or "")
            if row_code and row_code.isdigit() and row_code != plain_symbol:
                continue
            key = row.get("item") or row.get("项目") or row.get("key") or row.get("指标")
            value = row.get("value") or row.get("值") or row.get("数值")
            if key:
                mapping[str(key)] = value
                continue
            for item_key, item_value in row.items():
                if not is_missing(item_value):
                    mapping[str(item_key)] = item_value
        return mapping

    def _build_fund_fee_payload(self, symbol: str, details: dict) -> dict:
        trace = self._api_trace("akshare.fund_individual_detail_info_xq", {"symbol": symbol})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "management_fee": self._pick_detail(details, "管理费率", "管理费"),
            "custodian_fee": self._pick_detail(details, "托管费率", "托管费"),
            "sales_service_fee": self._pick_detail(details, "销售服务费率", "销售服务费"),
            "purchase_fee": self._pick_detail(details, "申购费率", "买入费率", "认购费率", "最高申购费率"),
            "redeem_fee": self._pick_detail(details, "赎回费率", "卖出费率", "最高赎回费率"),
        }

    def _build_fund_redemption_payload(self, symbol: str, details: dict) -> dict:
        trace = self._api_trace("akshare.fund_individual_detail_info_xq", {"symbol": symbol})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "subscription_status": self._pick_detail(details, "申购状态", "认购状态"),
            "redemption_status": self._pick_detail(details, "赎回状态"),
            "purchase_min": self._pick_detail(details, "最低申购金额", "起购金额", "最小申购单位"),
            "redemption_rule": self._pick_detail(details, "赎回规则", "赎回到账", "赎回确认"),
            "purchase_fee": self._pick_detail(details, "申购费率", "买入费率", "认购费率", "最高申购费率"),
            "redeem_fee": self._pick_detail(details, "赎回费率", "卖出费率", "最高赎回费率"),
        }

    def _build_fund_profile_payload(self, symbol: str, details: dict) -> dict:
        trace = self._api_trace("akshare.fund_individual_detail_info_xq", {"symbol": symbol})
        return {
            "source_name": "akshare",
            "provider": "akshare",
            **trace,
            "subscription_status": self._pick_detail(details, "申购状态", "认购状态"),
            "redemption_status": self._pick_detail(details, "赎回状态"),
            "purchase_min": self._pick_detail(details, "最低申购金额", "起购金额", "最小申购单位"),
            "redemption_rule": self._pick_detail(details, "赎回规则", "赎回到账", "赎回确认"),
            "trading_rule": self._pick_detail(details, "交易规则", "交易方式", "运作方式"),
            "tracking_index": self._pick_detail(details, "跟踪标的", "跟踪指数", "标的指数"),
            "fund_manager": self._pick_detail(details, "基金经理", "基金经理人", "基金管理人"),
        }

    def _pick_detail(self, details: dict, *keys: str):
        for key in keys:
            value = details.get(key)
            if not is_missing(value):
                return value
        return None

    def _first_present(self, row: dict, *keys: str):
        for key in keys:
            if key in row and not is_missing(row[key]):
                return row[key]
        return None

    def _first_number(self, row: dict, *keys: str) -> float | None:
        for key in keys:
            if key in row:
                value = to_number(row[key])
                if value is not None:
                    return value
        return None

    def _normalize_date(self, value: str | date | datetime | None) -> str | None:
        if is_missing(value):
            return None
        if isinstance(value, str) and "T" in value:
            return value
        return to_iso_date(value)

    def _dashed(self, yyyymmdd: str) -> str:
        text = str(yyyymmdd)
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) == 8 and text.isdigit() else text

    def _to_float(self, value: str | int | float | None) -> float | None:
        return to_number(value)

    def _fill_missing_pct_change(self, rows: list[dict]) -> None:
        for index, row in enumerate(rows[:-1]):
            if row.get("pct_change_1d") is not None:
                continue
            previous_close = row.get("close")
            prior_close = rows[index + 1].get("close")
            if previous_close is None or not prior_close:
                continue
            row["pct_change_1d"] = round((previous_close - prior_close) / prior_close * 100, 4)
