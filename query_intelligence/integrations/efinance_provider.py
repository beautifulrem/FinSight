from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


def _rows_to_records(rows):
    if rows is None:
        return []
    if isinstance(rows, list):
        return rows
    if hasattr(rows, "to_dict"):
        return rows.to_dict("records")
    return []


@dataclass
class EFinanceETFProvider:
    fund_module: object
    stock_module: object | None = None

    @classmethod
    def from_import(cls) -> "EFinanceETFProvider":
        import efinance as ef

        return cls(fund_module=ef.fund, stock_module=ef.stock)

    def fetch_history(self, symbol: str, beg: str = "20250101", end: str = "20261231") -> dict:
        return self._fetch_history(self.fund_module, symbol=symbol, beg=beg, end=end, source_name="efinance")

    def fetch_stock_history(self, symbol: str, beg: str = "20250101", end: str = "20261231") -> dict:
        if self.stock_module is None:
            raise RuntimeError("efinance stock module unavailable")
        return self._fetch_history(self.stock_module, symbol=symbol, beg=beg, end=end, source_name="efinance")

    def _fetch_history(self, module: object, symbol: str, beg: str, end: str, source_name: str) -> dict:
        plain_symbol = symbol.split(".")[0]
        rows = module.get_quote_history(plain_symbol, beg=beg, end=end, klt=101, fqt=1)
        records = _rows_to_records(rows)
        normalized = []
        for row in records:
            normalized.append(
                {
                    "trade_date": self._normalize_date(row.get("日期")),
                    "open": row.get("开盘"),
                    "high": row.get("最高"),
                    "low": row.get("最低"),
                    "close": row.get("收盘"),
                    "pct_change_1d": row.get("涨跌幅"),
                    "volume": row.get("成交量"),
                    "amount": row.get("成交额"),
                }
            )
        normalized.sort(key=lambda item: item.get("trade_date") or "", reverse=True)
        latest = normalized[0] if normalized else {}
        return {
            "source_type": "market_api",
            "source_name": source_name,
            "payload": {
                "symbol": symbol,
                "source_name": source_name,
                "trade_date": latest.get("trade_date"),
                "open": latest.get("open"),
                "high": latest.get("high"),
                "low": latest.get("low"),
                "close": latest.get("close"),
                "pct_change_1d": latest.get("pct_change_1d"),
                "volume": latest.get("volume"),
                "amount": latest.get("amount"),
                "history": normalized[:5],
            },
            "status": "ok",
        }

    def _normalize_date(self, value: str | date | datetime | None) -> str | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        try:
            return datetime.fromisoformat(value).date().isoformat()
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
            except ValueError:
                return value
