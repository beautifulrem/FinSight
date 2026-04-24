from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from .data_loader import load_aliases, load_documents, load_entities, load_structured_data
from .integrations.akshare_market_provider import AKShareMarketProvider
from .integrations.akshare_provider import AKShareNewsProvider
from .integrations.cninfo_provider import CninfoAnnouncementProvider
from .integrations.tushare_provider import TushareMarketProvider, TushareNewsProvider


DEFAULT_WATCHLIST = [
    {"symbol": "600519.SH", "canonical_name": "贵州茅台", "product_type": "stock"},
    {"symbol": "000858.SZ", "canonical_name": "五粮液", "product_type": "stock"},
    {"symbol": "510300.SH", "canonical_name": "沪深300ETF", "product_type": "etf"},
]


class PublicDataBootstrapper:
    def __init__(
        self,
        output_dir: str | Path,
        ak_news_provider: AKShareNewsProvider | None = None,
        ak_market_provider: AKShareMarketProvider | None = None,
        tushare_market_provider: TushareMarketProvider | None = None,
        tushare_news_provider: TushareNewsProvider | None = None,
        cninfo_provider: CninfoAnnouncementProvider | None = None,
        existing_documents: list[dict] | None = None,
        existing_entities: list[dict] | None = None,
        existing_aliases: list[dict] | None = None,
        existing_structured_data: dict | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ak_news_provider = ak_news_provider or AKShareNewsProvider.from_import()
        self.ak_market_provider = ak_market_provider or AKShareMarketProvider.from_import()
        self.tushare_market_provider = tushare_market_provider
        self.tushare_news_provider = tushare_news_provider
        self.cninfo_provider = cninfo_provider or CninfoAnnouncementProvider()
        self.existing_documents = existing_documents if existing_documents is not None else load_documents()
        self.existing_entities = existing_entities if existing_entities is not None else load_entities()
        self.existing_aliases = existing_aliases if existing_aliases is not None else load_aliases()
        self.existing_structured_data = existing_structured_data if existing_structured_data is not None else load_structured_data()
        self.warnings: list[str] = []

    def run(self, watchlist: list[dict] | None = None) -> None:
        watched = watchlist or DEFAULT_WATCHLIST
        entity_rows = self._build_entities(watched)
        alias_rows = self._build_aliases(entity_rows)
        documents = self._build_documents(watched)
        structured_data = self._build_structured_data(watched)

        self._write_csv(self.output_dir / "entity_master.csv", entity_rows)
        self._write_csv(self.output_dir / "alias_table.csv", alias_rows)
        (self.output_dir / "documents.json").write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "structured_data.json").write_text(json.dumps(structured_data, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.warnings:
            (self.output_dir / "bootstrap_warnings.json").write_text(json.dumps(self.warnings, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_entities(self, watchlist: list[dict]) -> list[dict]:
        existing_by_symbol = {row.get("symbol"): dict(row) for row in self.existing_entities if row.get("symbol")}
        next_id = max((int(row.get("entity_id", 0)) for row in self.existing_entities), default=0) + 1
        rows = [dict(row) for row in self.existing_entities]
        for item in watchlist:
            symbol = item["symbol"]
            if symbol in existing_by_symbol:
                continue
            try:
                bundle = self._fetch_market_bundle(item)
                payload = bundle["payload"]
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"entity_enrich_failed:{symbol}:{exc}")
                payload = {}
            rows.append(
                {
                    "entity_id": str(next_id),
                    "canonical_name": item["canonical_name"],
                    "normalized_name": item["canonical_name"],
                    "symbol": symbol,
                    "market": "CN",
                    "exchange": "SSE" if symbol.endswith(".SH") else "SZSE",
                    "entity_type": item["product_type"],
                    "industry_name": payload.get("industry_name") or "",
                    "status": "active",
                }
            )
            next_id += 1
        rows.sort(key=lambda row: str(row.get("entity_id")))
        return rows

    def _build_aliases(self, entity_rows: list[dict]) -> list[dict]:
        aliases = [dict(row) for row in self.existing_aliases]
        seen = {(row.get("entity_id"), row.get("normalized_alias")) for row in aliases}
        next_id = max((int(row.get("alias_id", 0)) for row in aliases), default=0) + 1
        for row in entity_rows:
            canonical = row["canonical_name"]
            pair = (row["entity_id"], canonical)
            if pair in seen:
                continue
            aliases.append(
                {
                    "alias_id": str(next_id),
                    "entity_id": str(row["entity_id"]),
                    "alias_text": canonical,
                    "normalized_alias": canonical,
                    "alias_type": "official_name",
                    "priority": "2",
                    "is_official": "true",
                }
            )
            next_id += 1
        aliases.sort(key=lambda row: int(row["alias_id"]))
        return aliases

    def _build_documents(self, watchlist: list[dict]) -> list[dict]:
        docs = [dict(row) for row in self.existing_documents if row.get("source_type") in {"faq", "product_doc", "research_note"}]
        seen = {(doc.get("source_type"), doc.get("title")) for doc in docs}
        for item in watchlist:
            symbol = item["symbol"]
            canonical_name = item["canonical_name"]
            if item["product_type"] == "stock":
                try:
                    for news_item in self._fetch_news_items(symbol=symbol, canonical_name=canonical_name, limit=5):
                        key = (news_item["source_type"], news_item["title"])
                        if key not in seen:
                            docs.append(news_item)
                            seen.add(key)
                except Exception as exc:  # noqa: BLE001
                    self.warnings.append(f"news_fetch_failed:{symbol}:{exc}")
                try:
                    for ann_item in self.cninfo_provider.fetch_announcements(symbol=symbol, limit=5):
                        key = (ann_item["source_type"], ann_item["title"])
                        if key not in seen:
                            docs.append(ann_item)
                            seen.add(key)
                except Exception as exc:  # noqa: BLE001
                    self.warnings.append(f"announcement_fetch_failed:{symbol}:{exc}")
        return docs

    def _build_structured_data(self, watchlist: list[dict]) -> dict:
        market_api = dict(self.existing_structured_data.get("market_api", {}))
        fundamental_sql = dict(self.existing_structured_data.get("fundamental_sql", {}))
        industry_sql = dict(self.existing_structured_data.get("industry_sql", {}))
        entity_to_industry = dict(self.existing_structured_data.get("entity_to_industry", {}))
        for item in watchlist:
            try:
                bundle = self._fetch_market_bundle(item)
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"structured_fetch_failed:{item['symbol']}:{exc}")
                continue
            market_api[item["symbol"]] = {
                **bundle["payload"],
                "source_name": bundle["source_name"],
            }
            if bundle.get("fundamental_payload"):
                fundamental_sql[item["symbol"]] = {
                    "symbol": item["symbol"],
                    "source_name": bundle["source_name"],
                    **bundle["fundamental_payload"],
                }
            industry_name = bundle["payload"].get("industry_name")
            if industry_name:
                entity_to_industry[item["canonical_name"]] = industry_name
                if bundle["payload"].get("industry_snapshot"):
                    industry_sql[industry_name] = bundle["payload"]["industry_snapshot"]
        return {
            "market_api": market_api,
            "fundamental_sql": fundamental_sql,
            "industry_sql": industry_sql,
            "macro_sql": {},
            "entity_to_industry": entity_to_industry,
        }

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _fetch_market_bundle(self, item: dict) -> dict:
        providers = []
        if self.tushare_market_provider is not None:
            providers.append(("tushare", lambda: self.tushare_market_provider.fetch_bundle(item["symbol"], start_date="20250101", end_date="20261231")))
        providers.append(
            (
                "akshare",
                lambda: self.ak_market_provider.fetch_bundle(
                    symbol=item["symbol"],
                    canonical_name=item["canonical_name"],
                    product_type=item["product_type"],
                ),
            )
        )
        return self._call_with_fallback(providers, f"market:{item['symbol']}")

    def _fetch_news_items(self, symbol: str, canonical_name: str, limit: int) -> list[dict]:
        providers = []
        if self.tushare_news_provider is not None:
            providers.append(("tushare_news", lambda: self.tushare_news_provider.fetch_news(symbol=symbol, canonical_name=canonical_name, limit=limit)))
        providers.append(("akshare_news", lambda: self.ak_news_provider.fetch_news(symbol=symbol, canonical_name=canonical_name, limit=limit)))
        return self._call_with_fallback(providers, f"news:{symbol}")

    def _call_with_fallback(self, providers: list[tuple[str, callable]], label: str):
        last_exc: Exception | None = None
        for name, fn in providers:
            for attempt in range(3):
                try:
                    return fn()
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    self.warnings.append(f"{label}:{name}:attempt_{attempt+1}:{exc}")
                    time.sleep(min(0.3 * (attempt + 1), 0.8))
                    continue
        if last_exc:
            raise last_exc
        raise RuntimeError(f"{label}: no providers available")
