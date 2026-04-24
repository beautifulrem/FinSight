from __future__ import annotations

from typing import Any


class PostgresDocumentRepository:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

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
        with self.connection.cursor() as cursor:
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
        if isinstance(row, dict):
            item = row
        else:
            item = dict(row)
        item.setdefault("evidence_id", f"{item['source_type']}_{item['doc_id']}")
        item["retrieval_score"] = float(item.get("retrieval_score", 0.0) or 0.0)
        return item


class PostgresStructuredRepository:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def fetch(self, query_bundle: dict) -> list[dict]:
        results = []
        symbols = query_bundle.get("symbols", [])
        source_plan = query_bundle.get("source_plan", [])
        for symbol in symbols:
            if "market_api" in source_plan:
                results.extend(self._fetch_market_history(symbol))
            if "fundamental_sql" in query_bundle.get("source_plan", []):
                results.extend(self._fetch_fundamental(symbol))
            if "industry_sql" in query_bundle.get("source_plan", []):
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
        LIMIT 5
        """
        with self.connection.cursor() as cursor:
            cursor.execute(sql, {"symbol": symbol})
            rows = cursor.fetchall()
        if not rows:
            return []
        normalized_rows = [
            dict(row) if isinstance(row, dict) else {
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
            for row in rows
        ]
        latest = normalized_rows[0]
        return [
            {
                "evidence_id": f"price_{symbol}",
                "source_type": "market_api",
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
        with self.connection.cursor() as cursor:
            cursor.execute(sql, {"symbol": symbol})
            rows = cursor.fetchall()
        return [
            {
                "evidence_id": f"fundamental_{row['symbol']}" if isinstance(row, dict) else f"fundamental_{row[0]}",
                "source_type": "fundamental_sql",
                "payload": dict(row) if isinstance(row, dict) else {
                    "symbol": row[0],
                    "report_date": row[1],
                    "revenue": row[2],
                    "net_profit": row[3],
                    "roe": row[4],
                    "gross_margin": row[5],
                    "pe_ttm": row[6],
                    "pb": row[7],
                },
            }
            for row in rows
        ]

    def _fetch_industry(self, symbol: str) -> list[dict]:
        sql = """
        SELECT industry_name
        FROM qi.entity_master
        WHERE symbol = %(symbol)s
        LIMIT 1
        """
        with self.connection.cursor() as cursor:
            cursor.execute(sql, {"symbol": symbol})
            entity_rows = cursor.fetchall()
            if not entity_rows:
                return []
            industry_name = entity_rows[0]["industry_name"] if isinstance(entity_rows[0], dict) else entity_rows[0][0]
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
        return [
            {
                "evidence_id": f"industry_{industry_name}",
                "source_type": "industry_sql",
                "payload": dict(row) if isinstance(row, dict) else {
                    "industry_name": row[0],
                    "trade_date": row[1],
                    "pct_change": row[2],
                    "pe": row[3],
                    "pb": row[4],
                    "turnover": row[5],
                },
            }
            for row in rows
        ]

    def _fetch_macro(self) -> list[dict]:
        sql = """
        SELECT indicator_code, metric_date, metric_value, unit
        FROM qi.macro_indicator
        ORDER BY metric_date DESC
        LIMIT 3
        """
        with self.connection.cursor() as cursor:
            cursor.execute(sql, {})
            rows = cursor.fetchall()
        return [
            {
                "evidence_id": f"macro_{row['indicator_code']}" if isinstance(row, dict) else f"macro_{row[0]}",
                "source_type": "macro_sql",
                "payload": dict(row) if isinstance(row, dict) else {
                    "indicator_code": row[0],
                    "metric_date": row[1],
                    "metric_value": row[2],
                    "unit": row[3],
                },
            }
            for row in rows
        ]
