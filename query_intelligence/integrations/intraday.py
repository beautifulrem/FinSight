"""Intraday (盘中) quotes for 今天/今日/today price questions during A-share trading hours.

The daily chains (Eastmoney → Sina → Tencent → cache → snapshot) answer with daily bars. For a question
about *today* asked while the market is open, ``get_price_history(intraday=true)`` adds the latest
real-time quote from this module, labelled ``price_basis="intraday"`` with its quote time and
provenance; outside trading hours, when no live source is configured, or when the quote cannot be
fetched, the tool keeps the daily close and records why (``basis_reason``), and the compliance guard
says so in the answer.

Trading hours (Asia/Shanghai): Monday–Friday 09:30–11:30 and 13:00–15:00. The lunch break
(11:30–13:00) counts as intraday: the quote is the last trade of the morning session and is labelled
with its time. Holidays come from ``query_intelligence.chat.answer._is_known_non_trading_day`` (weekends
plus the fixed-date closures); movable holidays (Spring Festival, Qingming, Dragon Boat, Mid-Autumn) are
not in it, so on those days the path tries a quote, finds that its date is not today, and falls back to
the daily close.

The clock is injectable (``now=`` or ``ToolContext.clock``) so tests run with a frozen clock.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")

MORNING_OPEN, MORNING_CLOSE = time(9, 30), time(11, 30)
AFTERNOON_OPEN, AFTERNOON_CLOSE = time(13, 0), time(15, 0)

# Sessions in which a 今天 price question is answered with an intraday quote.
INTRADAY_SESSIONS = frozenset({"morning", "lunch_break", "afternoon"})

_TODAY_ZH = re.compile(r"今天|今日")
_TODAY_EN = re.compile(r"\btoday(?:'s)?\b", re.IGNORECASE)


def now_shanghai() -> datetime:
    """Current time in Beijing (the default clock)."""
    return datetime.now(SHANGHAI_TZ)


def _local(now: datetime) -> datetime:
    return now.astimezone(SHANGHAI_TZ) if now.tzinfo else now.replace(tzinfo=SHANGHAI_TZ)


def is_trading_day(day: date) -> bool:
    from ..chat.answer import _is_known_non_trading_day

    return not _is_known_non_trading_day(day)


def market_session(now: datetime | None = None) -> str:
    """``non_trading_day`` | ``pre_open`` | ``morning`` | ``lunch_break`` | ``afternoon`` | ``closed``."""
    local = _local(now or now_shanghai())
    if not is_trading_day(local.date()):
        return "non_trading_day"
    clock = local.time()
    if clock < MORNING_OPEN:
        return "pre_open"
    if clock < MORNING_CLOSE:
        return "morning"
    if clock < AFTERNOON_OPEN:
        return "lunch_break"
    if clock < AFTERNOON_CLOSE:
        return "afternoon"
    return "closed"


def is_intraday(now: datetime | None = None) -> bool:
    return market_session(now) in INTRADAY_SESSIONS


def asks_about_today(query: str) -> bool:
    """今天 / 今日 / today (not 最新 or 现在: those ask for the latest data, which may be a close)."""
    return bool(_TODAY_ZH.search(query or "") or _TODAY_EN.search(query or ""))


class IntradayQuoteError(RuntimeError):
    pass


def _prefixed(symbol: str, product_type: str) -> str:
    plain = symbol.split(".")[0]
    suffix = symbol.split(".")[1].upper() if "." in symbol else ""
    if suffix in {"SH", "SZ", "BJ"}:
        return f"{suffix.lower()}{plain}"
    if product_type == "index":
        return f"sh{plain}" if plain.startswith(("0", "5", "6")) else f"sz{plain}"
    return f"sh{plain}" if plain.startswith(("5", "6", "9")) else f"sz{plain}"


def _number(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number


def _default_http_get(url: str, headers: dict[str, str], timeout: float):
    import requests

    return requests.get(url, headers=headers, timeout=timeout)


class IntradayQuoteProvider:
    """Real-time quote from Sina (``hq.sinajs.cn``), then Tencent (``qt.gtimg.cn``).

    Both are the real-time endpoints the daily chain already uses as fallbacks. A quote whose date is not
    today (a suspended stock, a movable holiday) is rejected, so the caller falls back to the daily close.
    """

    SINA_URL = "https://hq.sinajs.cn/list="
    TENCENT_URL = "https://qt.gtimg.cn/q="
    HEADERS: ClassVar[dict[str, str]] = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}

    def __init__(self, http_get: Callable[..., Any] | None = None, *, timeout: float = 5.0) -> None:
        self.http_get = http_get or _default_http_get
        self.timeout = timeout

    def fetch(self, symbol: str, product_type: str = "stock", *, now: datetime | None = None) -> dict[str, Any]:
        today = _local(now or now_shanghai()).date()
        attempts: list[str] = []
        errors: list[str] = []
        for source, fetcher in (("sina.quote", self._sina), ("tencent.qt_quote", self._tencent)):
            attempts.append(source)
            try:
                quote = fetcher(_prefixed(symbol, product_type))
            except Exception as exc:  # network, parsing: try the next source
                errors.append(f"{source}: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            if quote["quote_time"][:10] != today.isoformat():
                errors.append(f"{source}: quote dated {quote['quote_time'][:10]}, not today")
                continue
            quote.update(
                {
                    "source": source,
                    "attempts": attempts,
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "fallback_reason": "; ".join(errors) or None,
                }
            )
            return quote
        raise IntradayQuoteError("; ".join(errors) or "no intraday source")

    def _get(self, url: str):
        response = self.http_get(url, headers=self.HEADERS, timeout=self.timeout)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        content = getattr(response, "content", None)
        if isinstance(content, bytes):
            return content.decode("gbk", errors="replace")
        return str(response.text)

    def _sina(self, prefixed: str) -> dict[str, Any]:
        text = self._get(f"{self.SINA_URL}{prefixed}")
        _, _, body = text.partition("=")
        fields = body.strip().strip('";').split(",")
        if len(fields) < 32 or _number(fields[3]) in (None, 0.0):
            raise IntradayQuoteError(f"unexpected sina quote payload for {prefixed}")
        price, previous = _number(fields[3]), _number(fields[2])
        return {
            "price": price,
            "prev_close": previous,
            "pct_change": round((price - previous) / previous * 100, 4) if previous else None,
            "open": _number(fields[1]),
            "high": _number(fields[4]),
            "low": _number(fields[5]),
            "volume": _number(fields[8]),
            "volume_unit": "share",
            "amount": _number(fields[9]),
            "quote_time": f"{fields[30].strip()}T{fields[31].strip()}+08:00",
            "endpoint": f"{self.SINA_URL}{prefixed}",
        }

    def _tencent(self, prefixed: str) -> dict[str, Any]:
        text = self._get(f"{self.TENCENT_URL}{prefixed}")
        _, _, body = text.partition("=")
        fields = body.strip().strip('";').split("~")
        if len(fields) < 38 or _number(fields[3]) in (None, 0.0):
            raise IntradayQuoteError(f"unexpected tencent quote payload for {prefixed}")
        stamp = fields[30].strip()  # yyyymmddHHMMSS
        if len(stamp) < 14:
            raise IntradayQuoteError(f"no timestamp in tencent quote for {prefixed}")
        amount_wan = _number(fields[37])
        return {
            "price": _number(fields[3]),
            "prev_close": _number(fields[4]),
            "pct_change": _number(fields[32]),
            "open": _number(fields[5]),
            "high": _number(fields[33]),
            "low": _number(fields[34]),
            "volume": _number(fields[6]),
            "volume_unit": "lot",
            "amount": amount_wan * 10_000 if amount_wan is not None else None,
            "quote_time": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:{stamp[10:12]}:{stamp[12:14]}+08:00",
            "endpoint": f"{self.TENCENT_URL}{prefixed}",
        }
