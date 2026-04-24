from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class AKShareNewsProvider:
    ak_module: object

    @classmethod
    def from_import(cls) -> "AKShareNewsProvider":
        import akshare as ak

        return cls(ak_module=ak)

    def fetch_news(self, symbol: str, canonical_name: str, limit: int = 10) -> list[dict]:
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol is None:
            return []
        df = self.ak_module.stock_news_em(symbol=normalized_symbol.split(".")[0])
        rows = df if isinstance(df, list) else df.to_dict("records")
        results = []
        for row in rows[:limit]:
            publish_time = row.get("发布时间") or row.get("date") or row.get("发布时间")
            results.append(
                {
                    "evidence_id": f"aknews_{symbol}_{len(results)+1}",
                    "doc_id": None,
                    "source_type": "news",
                    "source_name": row.get("文章来源", "akshare"),
                    "source_url": row.get("新闻链接"),
                    "title": row.get("新闻标题") or row.get("title") or canonical_name,
                    "summary": row.get("新闻内容", "")[:120],
                    "body": row.get("新闻内容") or row.get("content") or "",
                    "publish_time": self._normalize_time(publish_time),
                    "product_type": "stock",
                    "credibility_score": 0.72,
                    "entity_symbols": [normalized_symbol],
                }
            )
        return results

    def _normalize_symbol(self, symbol: str) -> str | None:
        if "." in symbol:
            code, exchange = symbol.split(".", 1)
            if code.isdigit() and len(code) == 6 and exchange.upper() in {"SH", "SZ"}:
                return f"{code}.{exchange.upper()}"
            return None
        if not symbol.isdigit() or len(symbol) != 6:
            return None
        if symbol.startswith(("6", "5")):
            return f"{symbol}.SH"
        return f"{symbol}.SZ"

    def _normalize_time(self, value: str | None) -> str | None:
        if not value:
            return None
        if "T" in value:
            return value
        try:
            return datetime.fromisoformat(value).isoformat()
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").isoformat()
            except ValueError:
                return value
