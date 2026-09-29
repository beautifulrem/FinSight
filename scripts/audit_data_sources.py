"""Audit every live data source with real network calls and print a Markdown table.

Each source is called individually (circuit breaker disabled, no retries, no cache) for representative
entities, recording success, latency, row count, the newest as-of date, and the error. The fallback
chains used at runtime are then exercised end to end so the served source and provenance are visible.

    python -m scripts.audit_data_sources --json outputs/data_source_audit.json

Requires network access; results depend on the time of day and on upstream throttling. The report records
the commit and whether the working tree was clean; committed audits live in
``docs/results/data_sources/`` (run from a clean checkout), and ``.github/workflows/data-source-audit.yml``
runs the audit weekly and uploads the JSON as an artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
        if not record.get("rows"):
            record.update(ok=False, error=record.get("error") or "empty result")
    except Exception as exc:
        text = " ".join(str(exc).split())
        record.update(ok=False, error=f"{type(exc).__name__}: {text[:140]}")
    record["latency_ms"] = round((time.perf_counter() - started) * 1000)
    results.append(record)
    status = "ok" if record["ok"] else f"FAIL {record.get('error')}"
    print(f"[{group}] {source} {target}: {status} {record['latency_ms']} ms as_of={record.get('as_of')}", flush=True)


def _intraday_summary(quote: dict[str, Any]) -> dict[str, Any]:
    keys = ("source", "price", "prev_close", "pct_change", "quote_time", "fetched_at", "attempts", "fallback_reason")
    return {
        **{key: quote.get(key) for key in keys},
        "as_of": quote.get("quote_time"),
        "market_session": market_session(),
    }


def run_audit(timeout: float, include_legacy: bool) -> dict[str, Any]:
    import akshare as ak
    import efinance as ef
    import requests

    state = _git_state()  # at the start: later edits to the checkout are not attributed to this audit
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

    chains = run_chains(timeout)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "python -m scripts.audit_data_sources",
        **state,
        "versions": _versions(),
        "summary": {
            "probes": len(results),
            "probes_ok": sum(1 for row in results if row["ok"]),
            "chains": len(chains),
            "chains_ok": sum(1 for row in chains if row.get("ok")),
        },
        "probes": results,
        "chains": chains,
    }


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


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "| Group | Source (function) | Target | OK | Latency ms | Rows | Newest as-of | Error |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in report["probes"]:
        lines.append(
            f"| {row['group']} | {row['source']} | {row['target']} | {'yes' if row['ok'] else 'no'} | "
            f"{row['latency_ms']} | {row.get('rows', '')} | {row.get('as_of') or ''} | {row.get('error', '')} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="Write the full report to this JSON file.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per-call hard timeout in seconds.")
    parser.add_argument("--no-legacy", action="store_true", help="Skip the slow legacy jin10 macro probes.")
    args = parser.parse_args(argv)
    report = run_audit(args.timeout, include_legacy=not args.no_legacy)
    print(to_markdown(report))
    summary = report["summary"]
    print(f"\n{summary['probes_ok']}/{summary['probes']} probes ok", end=", ")
    print(f"{summary['chains_ok']}/{summary['chains']} chains ok")
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
