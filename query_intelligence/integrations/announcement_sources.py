"""Announcement sources and their ordered fallback chain.

Primary: cninfo (the official disclosure site). Secondary: Eastmoney's announcement API
(``np-anotice-stock.eastmoney.com``), which aggregates the same exchange filings from a different host
and is not affected by the Eastmoney quote-host throttling seen on ``push2his``. Final fallback is the
local document corpus, which the retrieval pipeline always searches anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

from .cninfo_provider import CninfoAnnouncementProvider
from .sources.provenance import LIVE, LIVE_FALLBACK, build_provenance
from .sources.runtime import SourceRuntime, fallback_reason_from, get_default_runtime

logger = logging.getLogger(__name__)

EASTMONEY_NOTICE_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
EASTMONEY_NOTICE_DETAIL = "https://data.eastmoney.com/notices/detail/{code}/{art_code}.html"


@dataclass
class EastmoneyAnnouncementProvider:
    session: object | None = None
    timeout: int = 15

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def fetch_announcements(self, symbol: str, limit: int = 10) -> list[dict]:
        plain_symbol = symbol.split(".")[0]
        response = self.session.get(
            EASTMONEY_NOTICE_URL,
            params={
                "sr": -1,
                "page_size": limit,
                "page_index": 1,
                "ann_type": "A",
                "client_source": "web",
                "stock_list": plain_symbol,
                "f_node": 0,
                "s_node": 0,
            },
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = ((response.json() or {}).get("data") or {}).get("list") or []
        normalized_symbol = _suffixed(plain_symbol)
        results = []
        for row in rows:
            codes = [str(item.get("stock_code")) for item in row.get("codes") or [] if isinstance(item, dict)]
            if codes and plain_symbol not in codes:
                continue
            art_code = row.get("art_code")
            title = row.get("title_ch") or row.get("title") or ""
            publish_time = _eastmoney_time(row.get("notice_date") or row.get("display_time"))
            results.append(
                {
                    "evidence_id": f"emnotice_{plain_symbol}_{len(results) + 1}",
                    "doc_id": None,
                    "source_type": "announcement",
                    "source_name": "eastmoney_notice",
                    "source_url": EASTMONEY_NOTICE_DETAIL.format(code=plain_symbol, art_code=art_code)
                    if art_code
                    else None,
                    "title": title,
                    "summary": title,
                    "body": title,
                    "publish_time": publish_time,
                    "product_type": "stock",
                    "credibility_score": 0.9,
                    "entity_symbols": [normalized_symbol],
                    "payload": {
                        "provenance": build_provenance(
                            source="eastmoney.announcement",
                            kind="announcement",
                            as_of=publish_time,
                            endpoint="eastmoney.notice_api",
                        )
                    },
                }
            )
            if len(results) >= limit:
                break
        return results


@dataclass
class FallbackAnnouncementProvider:
    """Tries each ``(source_id, provider)`` in order and returns the first non-empty result.

    Exposes ``fetch_announcements`` and ``timeout`` like a single provider, so the retrieval pipeline's
    worker-thread timeout and cooldown logic keep working unchanged.
    """

    providers: list[tuple[str, object]]
    runtime: SourceRuntime = field(default_factory=SourceRuntime)

    @classmethod
    def build_default(
        cls,
        *,
        url: str,
        static_base: str,
        timeout: int,
        runtime: SourceRuntime | None = None,
    ) -> FallbackAnnouncementProvider:
        return cls(
            providers=[
                (
                    "cninfo.announcement",
                    CninfoAnnouncementProvider(url=url, static_base=static_base, timeout=timeout, raise_errors=True),
                ),
                ("eastmoney.announcement", EastmoneyAnnouncementProvider(timeout=timeout)),
            ],
            runtime=runtime or get_default_runtime(),
        )

    @property
    def timeout(self) -> float:
        # The pipeline waits ``timeout + 5`` seconds for the whole chain.
        return float(sum(getattr(provider, "timeout", 15) for _source, provider in self.providers))

    def fetch_announcements(self, symbol: str, limit: int = 10) -> list[dict]:
        attempts: list[str] = []
        for index, (source_id, provider) in enumerate(self.providers):
            fetch = _fetcher(provider, symbol, limit)
            try:
                with self.runtime.trace() as trace:
                    docs = self.runtime.call(source_id, fetch, timeout_s=getattr(provider, "timeout", 15) + 2)
            except Exception as exc:
                attempts.extend(trace)
                logger.info("Announcement source %s failed for %s: %s", source_id, symbol, exc)
                continue
            if not docs:
                attempts.append(f"{source_id}:empty")
                continue
            attempts.extend(trace)
            reason = fallback_reason_from(attempts)
            for doc in docs:
                payload = doc.setdefault("payload", {})
                previous = payload.get("provenance") or {}
                payload["provenance"] = build_provenance(
                    source=source_id,
                    kind="announcement",
                    as_of=doc.get("publish_time"),
                    mode=LIVE if index == 0 else LIVE_FALLBACK,
                    endpoint=previous.get("endpoint"),
                    fallback_reason=reason,
                    attempts=attempts,
                )
            return docs
        logger.info("No live announcement source served %s: %s", symbol, "; ".join(attempts))
        return []


def _fetcher(provider: object, symbol: str, limit: int) -> Callable[[], list[dict]]:
    def fetch() -> list[dict]:
        return provider.fetch_announcements(symbol, limit=limit)

    return fetch


def _suffixed(plain_symbol: str) -> str:
    return f"{plain_symbol}.SH" if plain_symbol.startswith(("5", "6", "9")) else f"{plain_symbol}.SZ"


def _eastmoney_time(value: object) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    # "2026-09-24 00:00:00" or "2026-08-14 20:41:29:380"
    return text[:19].replace(" ", "T") if len(text) >= 19 else text[:10]
