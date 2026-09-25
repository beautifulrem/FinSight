from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .sources.provenance import build_provenance

logger = logging.getLogger(__name__)


CNINFO_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "Origin": "https://www.cninfo.com.cn",
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
}
CNINFO_SEARCH_URL = "https://www.cninfo.com.cn/new/information/topSearch/query"


@dataclass
class CninfoAnnouncementProvider:
    """Announcements from cninfo (巨潮资讯, the CSRC-designated disclosure site).

    Audit 2026-09-25: ``hisAnnouncement/query`` ignores ``stock=<code>,`` without the company's
    ``orgId`` and returns the whole-market feed (so filtering by ``secCode`` left nothing). The orgId
    is resolved once per code via ``topSearch/query`` and cached.
    """

    session: object | None = None
    url: str = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    static_base: str = "https://static.cninfo.com.cn/"
    timeout: int = 15
    # Fallback chains need failures to surface; the legacy default swallows network errors.
    raise_errors: bool = False
    search_url: str = CNINFO_SEARCH_URL
    _org_ids: dict[str, str | None] = field(default_factory=dict, repr=False)
    _org_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.session is None:
            session = requests.Session()
            retry = Retry(
                total=2,
                connect=2,
                read=2,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset(["POST"]),
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self.session = session

    def fetch_announcements(self, symbol: str, limit: int = 10) -> list[dict]:
        plain_symbol = symbol.split(".")[0]
        try:
            org_id = self._org_id(plain_symbol)
            response = self.session.post(
                self.url,
                data={
                    "pageNum": 1,
                    "pageSize": limit if org_id else max(limit, 30),
                    "column": "szse",
                    "tabName": "fulltext",
                    "plate": "" if org_id else "sz;sh",
                    "stock": f"{plain_symbol},{org_id}" if org_id else f"{plain_symbol},",
                    "sortName": "",
                    "sortType": "",
                    "searchkey": "",
                    "secid": "",
                    "category": "",
                    "trade": "",
                    "seDate": "",
                },
                headers=CNINFO_HEADERS,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout:
            if self.raise_errors:
                raise
            logger.warning("Cninfo announcement fetch timed out for %s (%ds), skipping", symbol, self.timeout)
            return []
        except requests.ConnectionError as exc:
            if self.raise_errors:
                raise
            logger.warning("Cninfo announcement connection error for %s: %s, skipping", symbol, exc)
            return []
        announcements = payload.get("announcements") or []
        normalized_symbol = self._normalize_symbol(symbol)
        results = []
        for row in announcements:
            sec_code = str(row.get("secCode") or "").strip()
            if sec_code and sec_code != plain_symbol:
                continue
            adjunct = row.get("adjunctUrl") or ""
            title = row.get("announcementTitle") or ""
            abstract = row.get("announcementContent") or row.get("announcementDigest") or ""
            body = abstract or title
            publish_time = self._normalize_time(row.get("announcementTime"))
            results.append(
                {
                    "evidence_id": f"cninfo_{row.get('secCode', plain_symbol)}_{len(results) + 1}",
                    "doc_id": None,
                    "source_type": "announcement",
                    "source_name": "cninfo",
                    "source_url": f"{self.static_base}{adjunct.lstrip('/')}",
                    "title": title,
                    "summary": abstract[:200] if abstract else title,
                    "body": body,
                    "publish_time": publish_time,
                    "product_type": "stock",
                    "credibility_score": 0.98,
                    "entity_symbols": [normalized_symbol],
                    "payload": {
                        "provenance": build_provenance(
                            source="cninfo.announcement",
                            kind="announcement",
                            as_of=publish_time,
                            endpoint="cninfo.his_announcement",
                        )
                    },
                }
            )
            if len(results) >= limit:
                break
        return results

    def _org_id(self, plain_symbol: str) -> str | None:
        """cninfo orgId for a security code (``None`` when the lookup fails or finds nothing)."""
        with self._org_lock:
            if plain_symbol in self._org_ids:
                return self._org_ids[plain_symbol]
        org_id = None
        try:
            response = self.session.post(
                self.search_url,
                data={"keyWord": plain_symbol, "maxNum": 10},
                headers=CNINFO_HEADERS,
                timeout=self.timeout,
            )
            response.raise_for_status()
            matches = response.json()
            if isinstance(matches, list):
                org_id = next(
                    (
                        str(item.get("orgId"))
                        for item in matches
                        if isinstance(item, dict) and str(item.get("code")) == plain_symbol and item.get("orgId")
                    ),
                    None,
                )
        except (requests.RequestException, ValueError) as exc:
            logger.info("Cninfo orgId lookup failed for %s: %s", plain_symbol, exc)
            return None
        with self._org_lock:
            self._org_ids[plain_symbol] = org_id
        return org_id

    def _normalize_symbol(self, symbol: str) -> str:
        if "." in symbol:
            return symbol
        if symbol.startswith(("6", "5")):
            return f"{symbol}.SH"
        return f"{symbol}.SZ"

    def _normalize_time(self, value: int | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, int):
            return datetime.fromtimestamp(value / 1000, tz=UTC).astimezone().isoformat()
        return value
