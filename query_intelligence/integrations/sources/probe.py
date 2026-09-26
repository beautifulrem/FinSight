"""Opt-in active health probes for ``GET /sources/health?probe=1``.

The health endpoint is passive by default: it reports what live traffic has recorded, so every source
reads ``unknown`` on a fresh process. A probe round sends one cheap request per source through
``SourceRuntime.call`` (so the breaker, latency and error bookkeeping are the same as for real traffic)
and returns per-source results.

Probing costs upstream requests, and several upstreams throttle bursts from one IP (Eastmoney quote
hosts in particular), so rounds are rate limited process-wide: at most one round every
``min_interval_s`` seconds (``QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS``, default 60). A request inside
that window gets the previous round's result, marked ``rate_limited`` with ``retry_in_s``; a request
while a round is running gets ``in_progress`` instead of starting a second round.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from .provenance import utc_now_iso
from .runtime import SourceRuntime, attempt_label

PROBE_SYMBOL = "600519"
PROBE_TIMEOUT_S = 6.0


@dataclass(frozen=True)
class Probe:
    source_id: str
    endpoint: str
    fetch: Callable[[], Any]


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "empty"):
        return not value.empty
    if hasattr(value, "status_code"):
        return 200 <= value.status_code < 300 and bool(getattr(value, "content", b"x"))
    try:
        return len(value) > 0
    except TypeError:
        return True


def default_probes() -> list[Probe]:
    """One inexpensive request per catalogued live source (akshare/requests imported lazily)."""
    import akshare as ak
    import requests

    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
    today = date.today()
    recent = (today - timedelta(days=14)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    def http(url: str) -> Callable[[], Any]:
        return lambda: requests.get(url, headers=headers, timeout=PROBE_TIMEOUT_S)

    return [
        Probe(
            "eastmoney.quote",
            "akshare.stock_zh_a_hist",
            lambda: ak.stock_zh_a_hist(symbol=PROBE_SYMBOL, period="daily", start_date=recent, end_date=end),
        ),
        Probe("eastmoney.datacenter", "akshare.macro_china_lpr", ak.macro_china_lpr),
        Probe("eastmoney.news", "akshare.stock_news_em", lambda: ak.stock_news_em(symbol=PROBE_SYMBOL)),
        Probe(
            "sina.kline",
            "akshare.stock_zh_a_daily",
            lambda: ak.stock_zh_a_daily(symbol=f"sh{PROBE_SYMBOL}", start_date=recent, end_date=end),
        ),
        Probe("sina.quote", "sina.hq_sinajs_cn", http(f"https://hq.sinajs.cn/list=sh{PROBE_SYMBOL}")),
        Probe(
            "tencent.kline",
            "tencent.fqkline",
            http(f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh{PROBE_SYMBOL},day,,,5,qfq"),
        ),
        Probe("tencent.quote", "tencent.qt_quote", http(f"https://qt.gtimg.cn/q=sh{PROBE_SYMBOL}")),
        Probe(
            "ths.finance",
            "akshare.stock_financial_abstract_ths",
            lambda: ak.stock_financial_abstract_ths(symbol=PROBE_SYMBOL, indicator="按报告期"),
        ),
        Probe(
            "ths.industry",
            "akshare.stock_board_industry_index_ths",
            lambda: ak.stock_board_industry_index_ths(symbol="白酒", start_date=recent, end_date=end),
        ),
        Probe("cninfo.profile", "akshare.stock_profile_cninfo", lambda: ak.stock_profile_cninfo(symbol=PROBE_SYMBOL)),
    ]


class ActiveProber:
    def __init__(
        self,
        *,
        min_interval_s: float = 60.0,
        probes_factory: Callable[[], list[Probe]] = default_probes,
        clock: Callable[[], float] = time.monotonic,
        max_parallel: int = 5,
    ) -> None:
        self.min_interval_s = max(0.0, min_interval_s)
        self._probes_factory = probes_factory
        self._clock = clock
        self._max_parallel = max(1, max_parallel)
        self._lock = threading.Lock()
        self._running = False
        self._last_started: float | None = None
        self._last_result: dict[str, Any] | None = None

    def probe(self, runtime: SourceRuntime) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            if self._running:
                return {"status": "in_progress", "last": self._last_result}
            if self._last_started is not None and now - self._last_started < self.min_interval_s:
                retry_in = round(self.min_interval_s - (now - self._last_started), 1)
                return {"status": "rate_limited", "retry_in_s": retry_in, "last": self._last_result}
            self._running = True
            self._last_started = now
        try:
            result = self._run(runtime)
        finally:
            with self._lock:
                self._running = False
        with self._lock:
            self._last_result = result
        return {"status": "completed", **result}

    def _run(self, runtime: SourceRuntime) -> dict[str, Any]:
        started = time.perf_counter()
        probes = self._probes_factory()

        def run_one(probe: Probe) -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                value = runtime.call(probe.source_id, probe.fetch, timeout_s=PROBE_TIMEOUT_S)
                ok, error = _nonempty(value), None if _nonempty(value) else "empty response"
                label = attempt_label(probe.source_id, None) if ok else f"{probe.source_id}:empty"
            except Exception as exc:
                ok, error, label = False, f"{type(exc).__name__}: {str(exc)[:120]}", attempt_label(probe.source_id, exc)
            return {
                "source": probe.source_id,
                "endpoint": probe.endpoint,
                "ok": ok,
                "outcome": label.rpartition(":")[2],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": error,
            }

        with ThreadPoolExecutor(max_workers=self._max_parallel, thread_name_prefix="source-probe") as pool:
            results = list(pool.map(run_one, probes))
        return {
            "probed_at": utc_now_iso(),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "ok": sum(1 for row in results if row["ok"]),
            "total": len(results),
            "results": results,
        }


_default_prober: ActiveProber | None = None
_default_lock = threading.Lock()


def get_default_prober() -> ActiveProber:
    global _default_prober
    with _default_lock:
        if _default_prober is None:
            from ...config import Settings

            _default_prober = ActiveProber(min_interval_s=Settings.from_env().source_probe_min_interval_seconds)
        return _default_prober
