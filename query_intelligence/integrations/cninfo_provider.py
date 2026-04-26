from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


CNINFO_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "Origin": "https://www.cninfo.com.cn",
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
}


@dataclass
class CninfoAnnouncementProvider:
    session: object | None = None
    url: str = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    static_base: str = "https://static.cninfo.com.cn/"
    timeout: int = 15

    def __post_init__(self) -> None:
        if self.session is None:
            session = requests.Session()
            retry = Retry(
                total=3,
                connect=3,
                read=3,
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
            response = self.session.post(
                self.url,
                data={
                    "pageNum": 1,
                    "pageSize": limit,
                    "column": "szse",
                    "tabName": "fulltext",
                    "plate": "sz;sh",
                    "stock": f"{plain_symbol},",
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
            logger.warning("Cninfo announcement fetch timed out for %s (%ds), skipping", symbol, self.timeout)
            return []
        except requests.ConnectionError as exc:
            logger.warning("Cninfo announcement connection error for %s: %s, skipping", symbol, exc)
            return []
        announcements = payload.get("announcements", [])
        normalized_symbol = self._normalize_symbol(symbol)
        results = []
        for row in announcements:
            sec_code = str(row.get("secCode") or "").strip()
            if sec_code and sec_code != plain_symbol:
                continue
            adjunct = row.get("adjunctUrl", "")
            title = row.get("announcementTitle") or ""
            abstract = row.get("announcementContent") or row.get("announcementDigest") or ""
            body = abstract or title
            results.append(
                {
                    "evidence_id": f"cninfo_{row.get('secCode', plain_symbol)}_{len(results)+1}",
                    "doc_id": None,
                    "source_type": "announcement",
                    "source_name": "cninfo",
                    "source_url": f"{self.static_base}{adjunct.lstrip('/')}",
                    "title": title,
                    "summary": abstract[:200] if abstract else title,
                    "body": body,
                    "publish_time": self._normalize_time(row.get("announcementTime")),
                    "product_type": "stock",
                    "credibility_score": 0.98,
                    "entity_symbols": [normalized_symbol],
                }
            )
            if len(results) >= limit:
                break
        return results

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
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc).astimezone().isoformat()
        return value
