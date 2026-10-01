"""Audit every live data source with real network calls and print a Markdown table.

Each source is called individually (circuit breaker disabled, no retries, no cache) for representative
entities, recording success, latency, row count, the newest as-of date, and the error. The fallback
chains used at runtime are then exercised end to end so the served source and provenance are visible.

    python -m scripts.audit_data_sources --json outputs/data_source_audit.json
    python -m scripts.audit_data_sources --rounds 2 --pause 1 --out-dir docs/results/data_sources   # dated audit

Requires network access; results depend on the time of day and on upstream throttling. The report records
the commit and whether the working tree was clean; committed audits live in
``docs/results/data_sources/`` (run from a clean checkout), and ``.github/workflows/data-source-audit.yml``
runs the audit weekly and uploads the JSON as an artifact.

Calls are sequential with ``--pause`` seconds after each (upstream rate limits). ``--rounds`` repeats the probe
set so every source has several samples; the per-source summary gives the success rate, latency P50/P95
(nearest rank), the freshness lag of the newest row (trading sessions behind the last trading day for daily
market data, calendar days for the rest), a schema-drift check against the columns the providers read, and
the failure reasons. A source that needs a key which is not configured (Tushare without ``TUSHARE_TOKEN``) is
listed as ``not configured`` and never called.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.integrations.akshare_macro_provider import AKShareMacroProvider  # noqa: E402
from query_intelligence.integrations.akshare_market_provider import AKShareMarketProvider  # noqa: E402
from query_intelligence.integrations.akshare_provider import AKShareNewsProvider  # noqa: E402
from query_intelligence.integrations.announcement_sources import (  # noqa: E402
    EastmoneyAnnouncementProvider,
    FallbackAnnouncementProvider,
)
from query_intelligence.integrations.cninfo_provider import CninfoAnnouncementProvider  # noqa: E402
from query_intelligence.integrations.intraday import IntradayQuoteProvider, market_session  # noqa: E402
from query_intelligence.integrations.sources import SourceCache, SourceHealthRegistry, SourceRuntime  # noqa: E402
from query_intelligence.integrations.sources.values import to_iso_date  # noqa: E402

STOCKS = [("600519", "贵州茅台"), ("300750", "宁德时代"), ("601318", "中国平安")]
ETF = ("510300", "沪深300ETF")
INDEX = ("000300", "沪深300")
DATE_KEYS = ("日期", "date", "trade_date", "净值日期", "数据日期", "月份", "TRADE_DATE", "报告期", "发布时间")
# Seconds to wait after every live call (set by --pause): sequential, gentle on upstream rate limits.
PAUSE = {"seconds": 1.0}
# Columns (or keys) each source must return because a provider reads them (schema-drift check). Sources whose
# result is a provider-built summary (dicts from our own code) are not listed.
SCHEMA: dict[str, set[str]] = {
    "eastmoney.quote (stock_zh_a_hist)": {"日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅"},
    "sina.kline (stock_zh_a_daily)": {"date", "open", "high", "low", "close", "volume", "amount"},
    "tencent.kline (fqkline)": {"date", "open", "close", "high", "low", "volume"},
    "sina.quote (hq.sinajs.cn)": {"trade_date", "open", "high", "low", "close", "volume", "amount"},
    "efinance (get_quote_history)": {"日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"},
    "eastmoney.quote (fund_etf_hist_em)": {"日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"},
    "sina.kline (fund_etf_hist_sina)": {"date", "open", "high", "low", "close", "volume"},
    "eastmoney.fund (fund_open_fund_info_em NAV)": {"净值日期", "单位净值", "日增长率"},
    "sina.kline (stock_zh_index_daily)": {"date", "open", "high", "low", "close", "volume"},
    "eastmoney.quote (index_zh_a_hist)": {"日期", "开盘", "收盘", "最高", "最低"},
    "sina.finance (stock_financial_analysis_indicator)": {
        "日期",
        "净资产收益率(%)",
        "加权每股收益(元)",
        "主营业务收入增长率(%)",
        "净利润增长率(%)",
    },
    "ths.finance (stock_financial_abstract_ths)": {
        "报告期",
        "营业总收入",
        "净利润",
        "营业总收入同比增长率",
        "净利润同比增长率",
        "基本每股收益",
        "销售毛利率",
    },
    "eastmoney.datacenter (stock_value_em)": {"数据日期", "PE(TTM)", "市净率", "总市值"},
    "eastmoney.quote (stock_individual_info_em)": {"item", "value"},
    "eastmoney.datacenter (macro_china_cpi)": {"月份", "全国-同比增长"},
    "eastmoney.datacenter (macro_china_pmi)": {"月份", "制造业-指数"},
    "eastmoney.datacenter (macro_china_money_supply)": {"月份", "货币和准货币(M2)-同比增长"},
    "eastmoney.datacenter (bond_zh_us_rate)": {"日期", "中国国债收益率10年"},
    "chinabond (bond_china_yield)": {"日期", "曲线名称", "10年"},
    "eastmoney.datacenter (macro_china_lpr)": {"TRADE_DATE", "LPR1Y", "LPR5Y"},
    "eastmoney.news (stock_news_em)": {"新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"},
}
# Groups whose newest row should be the last trading day (lag counted in trading sessions).
SESSION_GROUPS = {"market", "etf", "index", "intraday"}
# Sources that need a key: (source label, group, environment variable).
KEYED_SOURCES = [
    ("tushare.pro (daily, fina_indicator)", "market", "TUSHARE_TOKEN"),
    ("tushare.pro (news)", "news", "TUSHARE_TOKEN"),
]


def _latest(rows: Any) -> str | None:
    records = rows.to_dict("records") if hasattr(rows, "to_dict") else rows if isinstance(rows, list) else []
    dates = []
    for row in records:
        if not isinstance(row, dict):
            continue
        for key in DATE_KEYS:
            if key in row and row[key] is not None:
                iso = to_iso_date(row[key])
                if iso and iso[:4].isdigit():
                    dates.append(iso[:10])
                break
    return max(dates) if dates else None


def _count(rows: Any) -> int:
    if rows is None:
        return 0
    if hasattr(rows, "__len__"):
        return len(rows)
    return 1


def _columns(value: Any) -> set[str] | None:
    """Column names of a DataFrame, or the keys of the first record of a list of dicts."""
    if hasattr(value, "columns"):
        return {str(column) for column in value.columns}
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return {str(key) for key in value[0]}
    return None


def probe(results: list[dict], group: str, source: str, target: str, fn: Callable[[], Any], timeout: float) -> None:
    runtime = SourceRuntime(
        health=SourceHealthRegistry(failure_threshold=10_000), cache=SourceCache(), call_timeout_s=timeout
    )
    started = time.perf_counter()
    record: dict[str, Any] = {"group": group, "source": source, "target": target}
    try:
        value = runtime.call(source, fn)
        if isinstance(value, dict):
            record.update(ok=True, rows=1, as_of=value.get("valuation_date"))
            record.update(value)
            record["ok"] = bool(value.get("ok", True))
        else:
            record.update(ok=True, rows=_count(value), as_of=_latest(value))
            expected, columns = SCHEMA.get(source), _columns(value)
            if expected is not None and columns is not None:
                record["schema_missing"] = sorted(expected - columns)
        if not record.get("rows"):
            record.update(ok=False, error=record.get("error") or "empty result")
    except Exception as exc:
        text = " ".join(str(exc).split())
        record.update(ok=False, error=f"{type(exc).__name__}: {text[:140]}")
    record["latency_ms"] = round((time.perf_counter() - started) * 1000)
    results.append(record)
    status = "ok" if record["ok"] else f"FAIL {record.get('error')}"
    print(f"[{group}] {source} {target}: {status} {record['latency_ms']} ms as_of={record.get('as_of')}", flush=True)
    time.sleep(PAUSE["seconds"])


def _intraday_summary(quote: dict[str, Any]) -> dict[str, Any]:
    keys = ("source", "price", "prev_close", "pct_change", "quote_time", "fetched_at", "attempts", "fallback_reason")
    return {
        **{key: quote.get(key) for key in keys},
        "as_of": quote.get("quote_time"),
        "market_session": market_session(),
    }


def run_audit(timeout: float, include_legacy: bool, rounds: int = 1) -> dict[str, Any]:
    state = _git_state()  # at the start: later edits to the checkout are not attributed to this audit
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    calendar = trading_calendar(timeout)
    results: list[dict] = []
    for round_index in range(1, max(1, rounds) + 1):
        print(f"--- round {round_index}/{rounds}", flush=True)
        for row in run_probes(timeout, include_legacy):
            results.append({**row, "round": round_index})
    chains = run_chains(timeout)
    today = date.today()
    last_session = last_trading_day(calendar["days"], today)
    for row in results:
        row.update(freshness_lag(row, today, last_session, calendar["days"]))
    sources = summarize(results)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "started_at": started_at,
        "command": "python -m scripts.audit_data_sources",
        **state,
        "versions": _versions(),
        "network": network_note(),
        "settings": {"rounds": rounds, "pause_s": PAUSE["seconds"], "timeout_s": timeout},
        "reference": {
            "today": today.isoformat(),
            "last_trading_day": last_session.isoformat() if last_session else None,
            "calendar_source": calendar["source"],
        },
        "summary": {
            "probes": len(results),
            "probes_ok": sum(1 for row in results if row["ok"]),
            "sources": len(sources),
            "sources_all_ok": sum(1 for row in sources if row.get("status") == "ok"),
            "sources_partial": sum(1 for row in sources if row.get("status") == "partial"),
            "sources_down": sum(1 for row in sources if row.get("status") == "down"),
            "sources_not_configured": sum(1 for row in sources if row.get("status") == "not_configured"),
            "schema_drift": sorted({row["source"] for row in sources if row.get("schema_missing")}),
            "chains": len(chains),
            "chains_ok": sum(1 for row in chains if row.get("ok")),
        },
        "sources": sources,
        "probes": results,
        "chains": chains,
    }


def trading_calendar(timeout: float) -> dict[str, Any]:
    """SSE trading days (Sina calendar via AKShare); weekdays as a fallback, recorded as such."""
    import akshare as ak

    try:
        frame = ak.tool_trade_date_hist_sina()
        days = sorted({str(to_iso_date(value))[:10] for value in frame["trade_date"]})
        time.sleep(PAUSE["seconds"])
        return {"days": days, "source": "akshare.tool_trade_date_hist_sina"}
    except Exception as exc:  # the audit still runs; lags are then counted on weekdays
        return {"days": [], "source": f"weekdays (calendar unavailable: {type(exc).__name__})"}


def last_trading_day(days: list[str], today: date) -> date | None:
    if days:
        past = [day for day in days if day <= today.isoformat()]
        return date.fromisoformat(past[-1]) if past else None
    day = today
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def freshness_lag(row: dict[str, Any], today: date, last_session: date | None, days: list[str]) -> dict[str, Any]:
    """Calendar days from the newest row to today, and for daily market data the trading sessions behind."""
    as_of = str(row.get("as_of") or "")[:10]
    try:
        newest = date.fromisoformat(as_of)
    except ValueError:
        return {}
    lag: dict[str, Any] = {"lag_days": (today - newest).days}
    if row.get("group") in SESSION_GROUPS and last_session is not None:
        if days:
            lag["lag_sessions"] = sum(1 for day in days if as_of < day <= last_session.isoformat())
        else:
            lag["lag_sessions"] = sum(
                1
                for offset in range(1, (last_session - newest).days + 1)
                if (newest + timedelta(days=offset)).weekday() < 5
            )
    return lag


def _percentile(values: list[float], share: float) -> float | None:
    """Nearest-rank percentile (P50 is the median of an odd sample, the lower middle value of an even one)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def summarize(results: list[dict]) -> list[dict]:
    """One row per source: success rate, latency P50/P95, freshness lag, schema drift, failure reasons."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in results:
        grouped.setdefault((row["group"], row["source"]), []).append(row)
    summary = []
    for (group, source), rows in grouped.items():
        ok = [row for row in rows if row["ok"]]
        latencies = [float(row["latency_ms"]) for row in rows]
        lags = [row["lag_days"] for row in ok if row.get("lag_days") is not None]
        sessions = [row["lag_sessions"] for row in ok if row.get("lag_sessions") is not None]
        missing = sorted({column for row in rows for column in row.get("schema_missing") or []})
        failures = Counter(str(row.get("error") or "unknown")[:120] for row in rows if not row["ok"])
        status = "ok" if len(ok) == len(rows) else ("down" if not ok else "partial")
        summary.append(
            {
                "group": group,
                "source": source,
                "status": status,
                "calls": len(rows),
                "ok": len(ok),
                "success_rate": round(len(ok) / len(rows), 3),
                "latency_p50_ms": _percentile(latencies, 0.5),
                "latency_p95_ms": _percentile(latencies, 0.95),
                "newest_as_of": max((str(row.get("as_of"))[:10] for row in ok if row.get("as_of")), default=None),
                "lag_days_max": max(lags) if lags else None,
                "lag_days_median": statistics.median(lags) if lags else None,
                "lag_sessions_max": max(sessions) if sessions else None,
                "schema_checked": any("schema_missing" in row for row in rows),
                "schema_missing": missing,
                "failures": [{"reason": reason, "count": count} for reason, count in failures.most_common()],
            }
        )
    for source, group, variable in KEYED_SOURCES:
        if not os.getenv(variable):
            summary.append(
                {
                    "group": group,
                    "source": source,
                    "status": "not_configured",
                    "calls": 0,
                    "ok": 0,
                    "success_rate": None,
                    "note": f"{variable} is not set; the source was not called",
                }
            )
    return summary


def network_note() -> dict[str, Any]:
    """Whether HTTP(S) proxies are configured (no addresses are recorded) and where the audit ran from."""
    proxied = any(os.getenv(name) for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"))
    return {
        "proxy_env_set": proxied,
        "note": "run from the maintainer's machine; requests follow the environment's proxy settings"
        if proxied
        else "run from the maintainer's machine without a configured proxy",
    }


def run_probes(timeout: float, include_legacy: bool) -> list[dict]:
    import akshare as ak
    import efinance as ef
    import requests

    results: list[dict] = []
    today = date.today()
    start = (today - timedelta(days=60)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    provider = AKShareMarketProvider(
        ak_module=ak,
        max_retries=0,
        retry_backoff_seconds=0,
        runtime=SourceRuntime(health=SourceHealthRegistry(failure_threshold=10_000), call_timeout_s=timeout),
        http_get=lambda url, headers, timeout: requests.get(url, headers=headers, timeout=timeout),
    )

    for code, name in STOCKS:
        prefixed = provider._prefixed(code)
        target = f"{code} {name}"
        probe(
            results,
            "market",
            "eastmoney.quote (stock_zh_a_hist)",
            target,
            lambda code=code: ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start, end_date=end, adjust="", timeout=timeout
            ),
            timeout,
        )
        probe(
            results,
            "market",
            "sina.kline (stock_zh_a_daily)",
            target,
            lambda prefixed=prefixed: ak.stock_zh_a_daily(symbol=prefixed, start_date=start, end_date=end, adjust=""),
            timeout,
        )
        probe(
            results,
            "market",
            "tencent.kline (fqkline)",
            target,
            lambda prefixed=prefixed: provider._fetch_tencent_rows(prefixed, start, end, []),
            timeout,
        )
        probe(
            results,
            "market",
            "sina.quote (hq.sinajs.cn)",
            target,
            lambda code=code: [provider._fetch_sina_realtime_row(code, [])],
            timeout,
        )
        probe(
            results,
            "market",
            "efinance (get_quote_history)",
            target,
            lambda code=code: ef.stock.get_quote_history(code, beg=start, end=end, klt=101, fqt=1),
            timeout,
        )

    # The intraday path for 今天 questions (query_intelligence/integrations/intraday.py): Sina, then Tencent,
    # rejecting a quote not dated today. Outside the session the failure ("not today") is expected.
    intraday = IntradayQuoteProvider(timeout=timeout)
    for code, name in STOCKS:
        probe(
            results,
            "intraday",
            "intraday.quote (sina -> tencent)",
            f"{code} {name}",
            lambda code=code: _intraday_summary(intraday.fetch(code)),
            timeout,
        )

    code, name = ETF
    target = f"{code} {name}"
    probe(
        results,
        "etf",
        "eastmoney.quote (fund_etf_hist_em)",
        target,
        lambda: ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start, end_date=end, adjust=""),
        timeout,
    )
    probe(
        results,
        "etf",
        "sina.kline (fund_etf_hist_sina)",
        target,
        lambda: ak.fund_etf_hist_sina(symbol=f"sh{code}"),
        timeout,
    )
    probe(
        results,
        "etf",
        "tencent.kline (fqkline)",
        target,
        lambda: provider._fetch_tencent_rows(f"sh{code}", start, end, []),
        timeout,
    )
    probe(
        results,
        "etf",
        "eastmoney.fund (fund_open_fund_info_em NAV)",
        target,
        lambda: ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势"),
        timeout,
    )
    probe(
        results, "etf", "eastmoney.fund (fund_overview_em)", target, lambda: ak.fund_overview_em(symbol=code), timeout
    )
    probe(
        results,
        "etf",
        "xueqiu (fund_individual_detail_info_xq)",
        target,
        lambda: ak.fund_individual_detail_info_xq(symbol=code),
        timeout,
    )

    code, name = INDEX
    target = f"{code} {name}"
    probe(
        results,
        "index",
        "sina.kline (stock_zh_index_daily)",
        target,
        lambda: ak.stock_zh_index_daily(symbol=f"sh{code}"),
        timeout,
    )
    probe(
        results,
        "index",
        "eastmoney.quote (index_zh_a_hist)",
        target,
        lambda: ak.index_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end),
        timeout,
    )
    probe(
        results,
        "index",
        "tencent.kline (fqkline)",
        target,
        lambda: provider._fetch_tencent_rows(f"sh{code}", start, end, []),
        timeout,
    )
    probe(
        results,
        "index",
        "csindex (stock_zh_index_value_csindex)",
        target,
        lambda: ak.stock_zh_index_value_csindex(symbol=code),
        timeout,
    )

    for code, name in STOCKS:
        target = f"{code} {name}"
        probe(
            results,
            "fundamentals",
            "sina.finance (stock_financial_analysis_indicator)",
            target,
            lambda code=code: ak.stock_financial_analysis_indicator(symbol=code, start_year=str(today.year - 1)),
            timeout,
        )
        probe(
            results,
            "fundamentals",
            "ths.finance (stock_financial_abstract_ths)",
            target,
            lambda code=code: ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期"),
            timeout,
        )
        probe(
            results,
            "fundamentals",
            "eastmoney.datacenter (stock_value_em)",
            target,
            lambda code=code: ak.stock_value_em(symbol=code),
            timeout,
        )
        probe(
            results,
            "fundamentals",
            "tencent.quote (qt.gtimg.cn PE/PB)",
            target,
            lambda code=code: provider._fetch_tencent_valuation(code, []) or None,
            timeout,
        )
        probe(
            results,
            "industry",
            "eastmoney.quote (stock_individual_info_em)",
            target,
            lambda code=code: ak.stock_individual_info_em(symbol=code),
            timeout,
        )
        probe(
            results,
            "industry",
            "cninfo.profile (stock_profile_cninfo)",
            target,
            lambda code=code: ak.stock_profile_cninfo(symbol=code),
            timeout,
        )

    macro_probes = [
        ("eastmoney.datacenter (macro_china_cpi)", "CPI", lambda: ak.macro_china_cpi()),
        ("eastmoney.datacenter (macro_china_pmi)", "PMI", lambda: ak.macro_china_pmi()),
        ("eastmoney.datacenter (macro_china_money_supply)", "M2", lambda: ak.macro_china_money_supply()),
        ("eastmoney.datacenter (bond_zh_us_rate)", "CN10Y", lambda: ak.bond_zh_us_rate(start_date=start)),
        ("chinabond (bond_china_yield)", "CN10Y", lambda: ak.bond_china_yield(start_date=start, end_date=end)),
        ("eastmoney.datacenter (macro_china_lpr)", "LPR", lambda: ak.macro_china_lpr()),
    ]
    if include_legacy:
        macro_probes += [
            ("jin10 (macro_china_cpi_monthly, legacy)", "CPI", lambda: ak.macro_china_cpi_monthly()),
            ("jin10 (macro_china_pmi_yearly, legacy)", "PMI", lambda: ak.macro_china_pmi_yearly()),
            ("akshare (macro_china_pmi_monthly, legacy)", "PMI", lambda: ak.macro_china_pmi_monthly()),
        ]
    for source, target, fn in macro_probes:
        probe(results, "macro", source, target, fn, max(timeout, 45.0) if "legacy" in source else timeout)

    cninfo = CninfoAnnouncementProvider(timeout=int(timeout), raise_errors=True)
    notices = EastmoneyAnnouncementProvider(timeout=int(timeout))
    for code, name in [*STOCKS, ETF]:
        target = f"{code} {name}"
        probe(
            results,
            "news",
            "eastmoney.news (stock_news_em)",
            target,
            lambda code=code: ak.stock_news_em(symbol=code),
            timeout,
        )
        probe(
            results,
            "announcement",
            "cninfo.announcement (orgId query)",
            target,
            lambda code=code: _docs_summary(cninfo.fetch_announcements(code, limit=10)),
            timeout,
        )
        probe(
            results,
            "announcement",
            "eastmoney.announcement (notice API)",
            target,
            lambda code=code: _docs_summary(notices.fetch_announcements(code, limit=10)),
            timeout,
        )

    return results


def _git_state() -> dict[str, Any]:
    """Commit and cleanliness of the checkout the audit ran from (``"unknown"`` with the reason if git fails)."""
    from scripts.provenance import git_state

    return git_state(ROOT)


def run_chains(timeout: float) -> list[dict]:
    """End-to-end runs of the runtime fallback chains with a fresh shared runtime (breaker enabled)."""
    runtime = SourceRuntime(call_timeout_s=timeout)
    market = AKShareMarketProvider.from_import(timeout=int(timeout), runtime=runtime)
    macro = AKShareMacroProvider.from_import(runtime=runtime)
    news = AKShareNewsProvider.from_import(runtime=runtime)
    announcements = FallbackAnnouncementProvider.build_default(
        url="https://www.cninfo.com.cn/new/hisAnnouncement/query",
        static_base="https://static.cninfo.com.cn/",
        timeout=int(timeout),
        runtime=runtime,
    )
    today = date.today()
    start = (today - timedelta(days=365)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    rows: list[dict] = []

    def timed(kind: str, target: str, fn: Callable[[], dict]) -> None:
        started = time.perf_counter()
        try:
            record = fn()
        except Exception as exc:
            record = {"ok": False, "error": f"{type(exc).__name__}: {' '.join(str(exc).split())[:140]}"}
        record.update(kind=kind, target=target, latency_ms=round((time.perf_counter() - started) * 1000))
        rows.append(record)
        print(f"[chain] {kind} {target}: {json.dumps(record, ensure_ascii=False)[:300]}", flush=True)

    for symbol, name, product in [
        ("600519.SH", "贵州茅台", "stock"),
        ("300750.SZ", "宁德时代", "stock"),
        ("601318.SH", "中国平安", "stock"),
        ("510300.SH", "沪深300ETF", "etf"),
        ("000300.SH", "沪深300", "index"),
    ]:

        def market_chain(symbol=symbol, name=name, product=product) -> dict:
            bundle = market.fetch_bundle(symbol, name, product, start, end)
            payload = bundle["payload"]
            fundamentals = bundle.get("fundamental_payload") or {}
            return {
                "ok": payload.get("close") is not None,
                "served_by": payload["provenance"].get("source"),
                "mode": payload["provenance"].get("mode"),
                "as_of": payload.get("trade_date"),
                "close": payload.get("close"),
                "fallback_reason": payload["provenance"].get("fallback_reason"),
                "fundamentals_by": (fundamentals.get("provenance") or {}).get("source"),
                "valuation_by": (fundamentals.get("valuation_provenance") or {}).get("source"),
                "report_date": fundamentals.get("report_date"),
                "pe_ttm": fundamentals.get("pe_ttm"),
            }

        timed("market_bundle", symbol, market_chain)

    def macro_chain() -> dict:
        items, failures = macro.fetch_indicators_with_status({"normalized_query": "cpi pmi m2 国债 lpr"})
        return {
            "ok": bool(items),
            "indicators": {
                item["payload"]["indicator_code"]: {
                    "as_of": item["payload"]["metric_date"],
                    "value": item["payload"]["metric_value"],
                    "served_by": item["payload"]["provenance"]["source"],
                }
                for item in items
            },
            "failures": failures,
        }

    timed("macro", "CPI/PMI/M2/CN10Y/LPR", macro_chain)
    for symbol, name in [("600519.SH", "贵州茅台"), ("300750.SZ", "宁德时代")]:
        timed("news", symbol, lambda symbol=symbol, name=name: _docs_summary(news.fetch_news(symbol, name, 10)))
        timed(
            "announcements",
            symbol,
            lambda symbol=symbol: _docs_summary(announcements.fetch_announcements(symbol, 10), with_source=True),
        )
    return rows


def _docs_summary(docs: list[dict], with_source: bool = False) -> dict:
    dates = sorted(str(doc.get("publish_time") or "")[:10] for doc in docs if doc.get("publish_time"))
    summary: dict[str, Any] = {"ok": bool(docs), "rows": len(docs), "as_of": dates[-1] if dates else None}
    if with_source and docs:
        provenance = (docs[0].get("payload") or {}).get("provenance") or {}
        summary["served_by"] = provenance.get("source")
        summary["mode"] = provenance.get("mode")
    return summary


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": sys.version.split()[0]}
    for package in ("akshare", "efinance", "requests", "pandas"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _cell(value: Any) -> str:
    return "" if value is None else str(value).replace("|", "/")


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "| Group | Source (function) | Target | OK | Latency ms | Rows | Newest as-of | Error |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in report["probes"]:
        lines.append(
            f"| {row['group']} | {row['source']} | {row['target']} | {'yes' if row['ok'] else 'no'} | "
            f"{row['latency_ms']} | {row.get('rows', '')} | {row.get('as_of') or ''} | {_cell(row.get('error'))} |"
        )
    return "\n".join(lines)


def sources_markdown(report: dict[str, Any]) -> str:
    """The dated per-source report committed under docs/results/data_sources/."""
    summary, reference = report["summary"], report.get("reference") or {}
    lines = [
        f"# Live data-source audit, {report['generated_at']} (commit `{report['commit']}`)",
        "",
        f"* Command: `{report['command']} --rounds {report['settings']['rounds']} --pause "
        f"{report['settings']['pause_s']}` (timeout {report['settings']['timeout_s']} s per call), started "
        f"{report.get('started_at')}, working tree clean: {report.get('working_tree_clean')}.",
        f"* Network: {report['network']['note']}.",
        f"* Reference dates: today {reference.get('today')}, last trading day {reference.get('last_trading_day')} "
        f"(calendar: {reference.get('calendar_source')}).",
        f"* Versions: {', '.join(f'{key} {value}' for key, value in report['versions'].items())}.",
        f"* Probes: {summary['probes_ok']}/{summary['probes']} ok; sources: {summary['sources_all_ok']} all ok, "
        f"{summary['sources_partial']} partial, {summary['sources_down']} down, "
        f"{summary['sources_not_configured']} not configured; schema drift: "
        f"{', '.join(summary['schema_drift']) or 'none'}; runtime fallback chains: "
        f"{summary['chains_ok']}/{summary['chains']} served.",
        "",
        "## Per source",
        "",
        "Latency is per call (P50/P95 nearest rank over all calls of the source). Lag: calendar days from the newest "
        "row to today; sessions: trading days the newest daily bar is behind the last trading day.",
        "",
        "| Group | Source | Status | OK / calls | P50 ms | P95 ms | Newest as-of | Lag days (max) | Lag sessions "
        "(max) | Schema | Failure reasons |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for row in report["sources"]:
        if row["status"] == "not_configured":
            lines.append(
                f"| {row['group']} | {row['source']} | not configured | 0 / 0 | | | | | | | {row.get('note')} |"
            )
            continue
        schema = "not checked" if not row.get("schema_checked") else ("ok" if not row["schema_missing"] else
                 "missing " + ", ".join(row["schema_missing"]))  # fmt: skip
        reasons = "; ".join(f"{item['reason']} (x{item['count']})" for item in row["failures"])
        lines.append(
            f"| {row['group']} | {row['source']} | {row['status']} | {row['ok']} / {row['calls']} | "
            f"{_cell(row['latency_p50_ms'])} | {_cell(row['latency_p95_ms'])} | {_cell(row['newest_as_of'])} | "
            f"{_cell(row['lag_days_max'])} | {_cell(row['lag_sessions_max'])} | {schema} | {_cell(reasons)} |"
        )
    lines += [
        "",
        "## Runtime fallback chains (what the app would serve)",
        "",
        "| Kind | Target | OK | Served by | Mode | As-of | Latency ms | Fallback reason / error |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for row in report["chains"]:
        lines.append(
            f"| {row.get('kind')} | {row.get('target')} | {'yes' if row.get('ok') else 'no'} | "
            f"{_cell(row.get('served_by'))} | {_cell(row.get('mode'))} | {_cell(row.get('as_of'))} | "
            f"{row.get('latency_ms')} | {_cell(row.get('fallback_reason') or row.get('error'))} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="Write the full report to this JSON file.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per-call hard timeout in seconds.")
    parser.add_argument("--no-legacy", action="store_true", help="Skip the slow legacy jin10 macro probes.")
    parser.add_argument("--rounds", type=int, default=1, help="Repeat the probe set this many times (samples).")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to wait after every live call.")
    parser.add_argument(
        "--out-dir", help="Write audit-<UTC timestamp>-<commit>.json and .md (the per-source report) here."
    )
    args = parser.parse_args(argv)
    PAUSE["seconds"] = max(0.0, args.pause)
    report = run_audit(args.timeout, include_legacy=not args.no_legacy, rounds=args.rounds)
    print(to_markdown(report))
    summary = report["summary"]
    print(f"\n{summary['probes_ok']}/{summary['probes']} probes ok", end=", ")
    print(f"{summary['chains_ok']}/{summary['chains']} chains ok")
    paths = [Path(args.json)] if args.json else []
    if args.out_dir:
        stamp = report["generated_at"][:16].replace("-", "").replace(":", "") + "Z"
        base = Path(args.out_dir) / f"audit-{stamp}-{report['commit']}"
        paths.append(base.with_suffix(".json"))
        base.with_suffix(".md").parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".md").write_text(sources_markdown(report), encoding="utf-8")
        print(f"wrote {base.with_suffix('.md')}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
