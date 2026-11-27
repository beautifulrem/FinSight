from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
import threading
from typing import Any

from ..integrations.sources.values import to_number


def _iso_temporal(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Decimal, float)):
        return to_number(value)
    return value


def _normalize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_scalar(val) for key, val in mapping.items()}


@contextmanager
def _borrow_connection(connection_or_pool: Any, lock: threading.Lock):
    if hasattr(connection_or_pool, "connection") and callable(getattr(connection_or_pool, "connection", None)):
        with connection_or_pool.connection() as conn:
            yield conn
        return
    with lock:
        try:
            yield connection_or_pool
        except Exception:
            rollback = getattr(connection_or_pool, "rollback", None)
            if callable(rollback):
                try:
                    rollback()
                except Exception:
                    pass
            raise


class PostgresDocumentRepository:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self._lock = threading.Lock()

    def search(self, query_bundle: dict, top_k: int = 20) -> list[dict]:
        sql = """
        SELECT
            doc_id,
            source_type::text AS source_type,
            source_name,
            source_url,
            title,
            summary,
            body,
            publish_time,
            product_type::text AS product_type,
            credibility_score,
            COALESCE(entity_symbols, ARRAY[]::text[]) AS entity_symbols,
            retrieval_score
        FROM (
            SELECT
                ds.doc_id,
                ds.source_type,
                ds.source_name,
                ds.source_url,
                ds.title,
                ds.summary,
                ds.body,
                ds.publish_time,
                ds.product_type,
                ds.credibility_score,
                array_remove(array_agg(em.symbol), NULL) AS entity_symbols,
                ts_rank_cd(
                    ds.fts_document,
                    websearch_to_tsquery('simple', %(fts_query)s)
                ) AS retrieval_score
            FROM qi.document_store ds
            LEFT JOIN qi.document_entity_map dem ON dem.doc_id = ds.doc_id
            LEFT JOIN qi.entity_master em ON em.entity_id = dem.entity_id
            WHERE ds.source_type::text = ANY(%(allowed_sources)s)
              AND (
                    ds.fts_document @@ websearch_to_tsquery('simple', %(fts_query)s)
                 OR similarity(ds.title, %(fuzzy_query)s) > 0.2
                 OR similarity(COALESCE(ds.summary, ''), %(fuzzy_query)s) > 0.2
              )
            GROUP BY ds.doc_id
        ) ranked
        ORDER BY retrieval_score DESC, publish_time DESC NULLS LAST
        LIMIT %(limit)s
        """
        params = {
            "fts_query": self._build_fts_query(query_bundle),
            "fuzzy_query": self._build_fuzzy_query(query_bundle),
            "allowed_sources": self._allowed_sources(query_bundle),
            "limit": top_k,
        }
        with _borrow_connection(self.connection, self._lock) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        return [self._normalize_row(row) for row in rows]

    def _build_fts_query(self, query_bundle: dict) -> str:
        terms = [
            query_bundle.get("normalized_query", ""),
            *query_bundle.get("entity_names", []),
            *query_bundle.get("symbols", []),
            *query_bundle.get("keywords", []),
        ]
        return " ".join(term for term in terms if term).strip()

    def _build_fuzzy_query(self, query_bundle: dict) -> str:
        return " ".join(
            term
            for term in [
                query_bundle.get("normalized_query", ""),
                *query_bundle.get("entity_names", []),
                *query_bundle.get("keywords", []),
            ]
            if term
        ).strip()

    def _allowed_sources(self, query_bundle: dict) -> list[str]:
        source_plan = query_bundle.get("source_plan", [])
        allowed = [source for source in source_plan if source in {"news", "announcement", "faq", "product_doc", "research_note"}]
        return allowed or ["news", "announcement", "faq", "product_doc", "research_note"]

    def _normalize_row(self, row: Any) -> dict:
        item = dict(row)
        item["publish_time"] = _iso_temporal(item.get("publish_time"))
        if item.get("credibility_score") is not None:
            item["credibility_score"] = to_number(item.get("credibility_score"))
        item.setdefault("evidence_id", f"{item['source_type']}_{item['doc_id']}")
        item["retrieval_score"] = to_number(item.get("retrieval_score")) or 0.0
        return item


class PostgresStructuredRepository:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self._lock = threading.Lock()

    def fetch(self, query_bundle: dict) -> list[dict]:
        results = []
        symbols = query_bundle.get("symbols", [])
        source_plan = query_bundle.get("source_plan", [])
        for symbol in symbols:
            if "market_api" in source_plan:
                results.extend(self._fetch_market_history(symbol))
            if "fundamental_sql" in source_plan:
                results.extend(self._fetch_fundamental(symbol))
            if "industry_sql" in source_plan:
                results.extend(self._fetch_industry(symbol))
        if "macro_sql" in source_plan:
            results.extend(self._fetch_macro())
        return results

    def _fetch_market_history(self, symbol: str) -> list[dict]:
        sql = """
        SELECT symbol, trade_date, open, high, low, close, pct_change, volume, amount
        FROM qi.market_daily
        WHERE symbol = %(symbol)s
        ORDER BY trade_date DESC
        LIMIT 30
        """
        with _borrow_connection(self.connection, self._lock) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, {"symbol": symbol})
                rows = cursor.fetchall()
        if not rows:
            return []
        normalized_rows = [
            _normalize_mapping(
                dict(row)
                if isinstance(row, dict)
                else {
                    "symbol": row[0],
                    "trade_date": row[1],
                    "open": row[2],
                    "high": row[3],
                    "low": row[4],
                    "close": row[5],
                    "pct_change": row[6],
                    "volume": row[7],
                    "amount": row[8],
                }
            )
            for row in rows
        ]
        for item in normalized_rows:
            item["pct_change_1d"] = item.get("pct_change")
        latest = normalized_rows[0]
        return [
            {
                "evidence_id": f"price_{symbol}",
                "source_type": "market_api",
                "source_name": "postgres",
                "provider": "postgres",
                "payload": {
                    "symbol": symbol,
                    "trade_date": latest["trade_date"],
                    "open": latest["open"],
                    "high": latest["high"],
                    "low": latest["low"],
                    "close": latest["close"],
                    "pct_change_1d": latest["pct_change"],
                    "volume": latest["volume"],
                    "amount": latest["amount"],
                    "history": normalized_rows,
                    "data_source": "market_daily",
                    "source_name": "postgres",
                },
            }
        ]

    def _fetch_fundamental(self, symbol: str) -> list[dict]:
        sql = """
        SELECT symbol, report_date, revenue, net_profit, roe, gross_margin, pe_ttm, pb
        FROM qi.fundamental_quarterly
        WHERE symbol = %(symbol)s
        ORDER BY report_date DESC
        LIMIT 1
        """
        with _borrow_connection(self.connection, self._lock) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, {"symbol": symbol})
                rows = cursor.fetchall()
        items = []
        for row in rows:
            payload = _normalize_mapping(
                dict(row)
                if isinstance(row, dict)
                else {
                    "symbol": row[0],
                    "report_date": row[1],
                    "revenue": row[2],
                    "net_profit": row[3],
                    "roe": row[4],
                    "gross_margin": row[5],
                    "pe_ttm": row[6],
                    "pb": row[7],
                }
            )
            payload["source_name"] = "postgres"
            items.append(
                {
                    "evidence_id": f"fundamental_{payload['symbol']}",
                    "source_type": "fundamental_sql",
                    "source_name": "postgres",
                    "provider": "postgres",
                    "payload": payload,
                }
            )
        return items

    def _fetch_industry(self, symbol: str) -> list[dict]:
        sql = """
        SELECT industry_name
        FROM qi.entity_master
        WHERE symbol = %(symbol)s
        LIMIT 1
        """
        with _borrow_connection(self.connection, self._lock) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, {"symbol": symbol})
                entity_rows = cursor.fetchall()
                if not entity_rows:
                    return []
                industry_name = (
                    entity_rows[0]["industry_name"] if isinstance(entity_rows[0], dict) else entity_rows[0][0]
                )
                if not industry_name:
                    return []
                cursor.execute(
                    """
                    SELECT industry_name, trade_date, pct_change, pe, pb, turnover
                    FROM qi.industry_daily
                    WHERE industry_name = %(industry_name)s
                    ORDER BY trade_date DESC
                    LIMIT 1
                    """,
                    {"industry_name": industry_name},
                )
                rows = cursor.fetchall()
        items = []
        for row in rows:
            payload = _normalize_mapping(
                dict(row)
                if isinstance(row, dict)
                else {
                    "industry_name": row[0],
                    "trade_date": row[1],
                    "pct_change": row[2],
                    "pe": row[3],
                    "pb": row[4],
                    "turnover": row[5],
                }
            )
            payload["source_name"] = "postgres"
            items.append(
                {
                    "evidence_id": f"industry_{industry_name}",
                    "source_type": "industry_sql",
                    "source_name": "postgres",
                    "provider": "postgres",
                    "payload": payload,
                }
            )
        return items

    def _fetch_macro(self) -> list[dict]:
        sql = """
        SELECT DISTINCT ON (indicator_code) indicator_code, metric_date, metric_value, unit
        FROM qi.macro_indicator
        ORDER BY indicator_code, metric_date DESC
        """
        with _borrow_connection(self.connection, self._lock) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, {})
                rows = cursor.fetchall()
        items = []
        for row in rows:
            payload = _normalize_mapping(
                dict(row)
                if isinstance(row, dict)
                else {
                    "indicator_code": row[0],
                    "metric_date": row[1],
                    "metric_value": row[2],
                    "unit": row[3],
                }
            )
            payload["source_name"] = "postgres"
            items.append(
                {
                    "evidence_id": f"macro_{payload['indicator_code']}",
                    "source_type": "macro_sql",
                    "source_name": "postgres",
                    "provider": "postgres",
                    "payload": payload,
                }
            )
        return items
