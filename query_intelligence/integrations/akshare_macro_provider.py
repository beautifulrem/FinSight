from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


def _rows_to_records(rows: object) -> list[dict]:
    if rows is None:
        return []
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if hasattr(rows, "to_dict"):
        return rows.to_dict("records")
    return []


@dataclass
class AKShareMacroProvider:
    ak_module: object

    @classmethod
    def from_import(cls) -> "AKShareMacroProvider":
        import akshare as ak

        return cls(ak_module=ak)

    def fetch_indicators(self, query_bundle: dict) -> list[dict]:
        requested_codes = self._requested_codes(query_bundle)
        items: list[dict] = []
        for code in requested_codes:
            payload = self._fetch_payload(code)
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
        return items

    def _requested_codes(self, query_bundle: dict) -> list[str]:
        text = self._query_text(query_bundle)
        matched: list[str] = []
        code_terms = {
            "CPI_CN": ("cpi", "通胀", "物价"),
            "PMI_CN": ("pmi", "制造业", "景气"),
            "M2_CN": ("m2", "货币", "流动性", "社融", "降准", "降息", "lpr"),
            "CN10Y": ("cn10y", "10y", "十年期", "国债", "利率", "收益率", "降息", "lpr"),
        }
        for code, terms in code_terms.items():
            if any(term in text for term in terms):
                matched.append(code)
        if any(term in text for term in ("降准", "降息", "lpr")):
            for code in ("M2_CN", "CN10Y"):
                if code not in matched:
                    matched.append(code)
        return matched

    def _fetch_payload(self, code: str) -> dict | None:
        if code == "CPI_CN":
            return self._fetch_cpi()
        if code == "PMI_CN":
            return self._fetch_pmi()
        if code == "M2_CN":
            return self._fetch_money_supply()
        if code == "CN10Y":
            return self._fetch_cn10y()
        return None

    def _fetch_cpi(self) -> dict | None:
        rows = self._safe_call("macro_china_cpi_monthly")
        row = self._latest_row(rows)
        if not row:
            return None
        return self._payload(
            indicator_code="CPI_CN",
            indicator_name="中国CPI同比",
            metric_date=self._extract_date(row),
            metric_value=self._extract_number(row, ("同比", "今值", "居民消费价格指数", "全国", "value")),
            unit="%",
            frequency="monthly",
        )

    def _fetch_pmi(self) -> dict | None:
        rows = self._safe_call("macro_china_pmi_monthly")
        if not rows:
            rows = self._safe_call("macro_china_pmi_yearly")
        row = self._latest_row(rows)
        if not row:
            return None
        frequency = "monthly" if rows and self._has_method("macro_china_pmi_monthly") else "yearly"
        return self._payload(
            indicator_code="PMI_CN",
            indicator_name="中国制造业PMI",
            metric_date=self._extract_date(row),
            metric_value=self._extract_number(row, ("制造业PMI", "今值", "pmi", "value")),
            unit="",
            frequency=frequency,
        )

    def _fetch_money_supply(self) -> dict | None:
        rows = self._safe_call("macro_china_money_supply")
        row = self._latest_row(rows)
        if not row:
            return None
        return self._payload(
            indicator_code="M2_CN",
            indicator_name="中国M2同比",
            metric_date=self._extract_date(row),
            metric_value=self._extract_number(row, ("M2同比", "货币和准货币(M2)同比增长", "M2", "今值", "value")),
            unit="%",
            frequency="monthly",
        )

    def _fetch_cn10y(self) -> dict | None:
        rows = self._safe_call("bond_zh_us_rate")
        row = self._latest_row(rows)
        if not row:
            return None
        return self._payload(
            indicator_code="CN10Y",
            indicator_name="中国10年期国债收益率",
            metric_date=self._extract_date(row),
            metric_value=self._extract_number(row, ("中国国债收益率10年", "中国10年期国债收益率", "中债国债到期收益率:10年", "CN10Y", "value")),
            unit="%",
            frequency="daily",
        )

    def _safe_call(self, method_name: str) -> list[dict]:
        if not self._has_method(method_name):
            return []
        try:
            return _rows_to_records(getattr(self.ak_module, method_name)())
        except Exception:
            return []

    def _has_method(self, method_name: str) -> bool:
        return callable(getattr(self.ak_module, method_name, None))

    def _payload(
        self,
        indicator_code: str,
        indicator_name: str,
        metric_date: str | None,
        metric_value: float | None,
        unit: str,
        frequency: str,
    ) -> dict | None:
        if metric_date is None and metric_value is None:
            return None
        release_period = metric_date[:7] if metric_date else None
        return {
            "indicator_code": indicator_code,
            "indicator_name": indicator_name,
            "metric_date": metric_date,
            "metric_value": metric_value,
            "unit": unit,
            "source_name": "akshare",
            "provider": "akshare",
            "release_period": release_period,
            "frequency": frequency,
        }

    def _latest_row(self, rows: list[dict]) -> dict | None:
        if not rows:
            return None
        return max(rows, key=lambda row: self._date_sort_key(self._extract_date(row)))

    def _extract_date(self, row: dict) -> str | None:
        for key in ("日期", "月份", "时间", "统计时间", "date", "month", "period"):
            if key in row and row[key] is not None:
                return self._normalize_date(row[key])
        return None

    def _extract_number(self, row: dict, preferred_keys: tuple[str, ...]) -> float | None:
        for key in preferred_keys:
            if key in row:
                value = self._to_float(row[key])
                if value is not None:
                    return value
        for key, raw_value in row.items():
            if any(token in str(key).lower() for token in ("date", "日期", "月份", "时间")):
                continue
            value = self._to_float(raw_value)
            if value is not None:
                return value
        return None

    def _normalize_date(self, value: object) -> str | None:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        text = str(value).strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m", "%Y/%m", "%Y年%m月"):
            try:
                parsed = datetime.strptime(text, fmt)
                if fmt in {"%Y-%m", "%Y/%m", "%Y年%m月"}:
                    return parsed.strftime("%Y-%m-01")
                return parsed.date().isoformat()
            except ValueError:
                continue
        return text[:10]

    def _date_sort_key(self, value: str | None) -> str:
        return value or ""

    def _to_float(self, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace("%", "").replace(",", "")
        if not text or text in {"-", "--", "nan", "None"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _query_text(self, query_bundle: dict) -> str:
        parts = [
            query_bundle.get("normalized_query", ""),
            *query_bundle.get("keywords", []),
            *query_bundle.get("entity_names", []),
        ]
        return " ".join(str(part) for part in parts).lower()
