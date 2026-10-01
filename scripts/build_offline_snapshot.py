"""Build the offline snapshot extension (``data/snapshot/``) from the live sources the project integrates.

The shipped offline snapshot ``data/structured_data.json`` (v1) covers 7 priced instruments and 3 companies'
fundamentals, with at most 5 daily closes each. Every evaluation label in the repository was written against
it, so it is never edited. This script builds a separate, additive layer for widely asked names that v1 lacks:
~15 months of daily bars (>= 250 closes, enough for a 52-week range, a one-year maximum drawdown and a
year-to-date change) and, for stocks, the 2025 annual report (revenue, net profit, ROE, YoY growth, EPS)
plus the valuation on the snapshot date (P/E TTM, P/B, P/S, market cap).

Two steps, so the build is reproducible without network access:

    python -m scripts.build_offline_snapshot fetch    # live: data/snapshot/raw/<symbol>.json (source records)
    python -m scripts.build_offline_snapshot build    # offline, deterministic: raw -> structured_data_ext.json
    python -m scripts.build_offline_snapshot verify   # offline: hashes, rebuild == committed, no v1 overlap
    python -m scripts.build_offline_snapshot all      # fetch + build + verify

``fetch`` calls the same fallback chains as the live runtime (``AKShareMarketProvider``: Eastmoney -> Sina ->
Tencent for daily bars; Sina + THS financial indicators; Eastmoney datacenter valuation), sequentially with a
pause between calls, and stores the source records it received (window-filtered) with the endpoint, the
attempts and the fetch time. ``build`` turns them into snapshot records with the provider's own normalisation
and cross-check code, and writes ``manifest.json`` with a sha256 for every raw file and the output.
The runtime merges the extension under v1 (``query_intelligence/data_loader.py``): a symbol already in v1 is
never taken from the extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SNAPSHOT_DIR = ROOT / "data" / "snapshot"
RAW_DIR = SNAPSHOT_DIR / "raw"
EXT_PATH = SNAPSHOT_DIR / "structured_data_ext.json"
MANIFEST_PATH = SNAPSHOT_DIR / "manifest.json"
BASE_PATH = ROOT / "data" / "structured_data.json"

VERSION = "ext-2026-09-30"
# Window of daily bars: from 2025-07-01 (so the 52 weeks before the as-of date and the last 2025 close before
# the first 2026 session are both inside) to the last trading day before the build (2026-10-01 is a holiday).
START = "20250701"
AS_OF = "20260930"
# Annual statements for every stock, so companies compare on the same period as the v1 fundamentals (FY2025).
REPORT_PERIOD = "2025-12-31"
FINANCIAL_START_YEAR = "2024"  # Sina rows from 2024 on: FY2024 is the YoY base

# symbol, canonical name (entity master), product type
INSTRUMENTS: list[tuple[str, str, str]] = [
    ("300750.SZ", "宁德时代", "stock"),
    ("002594.SZ", "比亚迪", "stock"),
    ("600036.SH", "招商银行", "stock"),
    ("000001.SZ", "平安银行", "stock"),
    ("601398.SH", "工商银行", "stock"),
    ("600030.SH", "中信证券", "stock"),
    ("600900.SH", "长江电力", "stock"),
    ("601899.SH", "紫金矿业", "stock"),
    ("000333.SZ", "美的集团", "stock"),
    ("688981.SH", "中芯国际", "stock"),
    ("518880.SH", "黄金ETF华安", "etf"),
    ("510500.SH", "中证500ETF南方", "etf"),
    ("588000.SH", "科创50ETF华夏", "etf"),
    ("000001.SH", "上证指数", "index"),
    ("399006.SZ", "创业板指", "index"),
]
VALUATION_FIELDS = {  # stock_value_em column -> snapshot field
    "PE(TTM)": "pe_ttm",
    "市净率": "pb",
    "市销率": "ps_ttm",
    "总市值": "total_mv",
    "流通市值": "circ_mv",
}
AMOUNT_FIELDS = ("revenue", "net_profit", "profit_dedt", "total_mv", "circ_mv")


def _amount(field: str, value: float | None) -> float | None:
    return round(value, 2) if value is not None and field in AMOUNT_FIELDS else value


# Columns each source must still return (schema-drift guard; the build fails loudly instead of writing nulls).
REQUIRED_COLUMNS = {
    "ths": {"报告期", "营业总收入", "净利润", "营业总收入同比增长率", "净利润同比增长率", "基本每股收益"},
    "sina": {"日期", "净资产收益率(%)", "加权每股收益(元)", "主营业务收入增长率(%)", "净利润增长率(%)"},
    "valuation": {"数据日期", "当日收盘价", *VALUATION_FIELDS},
}


# --------------------------------------------------------------------------- serialisation helpers


def _plain(value: Any) -> Any:
    """JSON-safe scalar: dates as ISO strings, NaN/None as None, numpy scalars as Python numbers."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat() if not value.time() else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _records(frame: Any) -> list[dict[str, Any]]:
    rows = frame.to_dict("records") if hasattr(frame, "to_dict") else list(frame or [])
    return [{str(key): _plain(value) for key, value in row.items()} for row in rows]


def dumps(value: Any, indent: int = 0) -> str:
    """Deterministic JSON: sorted keys, one line per row for lists of flat records (readable diffs)."""
    pad, inner = " " * indent, " " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{inner}{json.dumps(str(k), ensure_ascii=False)}: {dumps(value[k], indent + 1)}" for k in sorted(value)]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(row, dict) and all(not isinstance(v, dict | list) for v in row.values()) for row in value):
            rows = [f"{inner}{json.dumps(row, ensure_ascii=False, sort_keys=True)}" for row in value]
        else:
            rows = [f"{inner}{dumps(row, indent + 1)}" for row in value]
        return "[\n" + ",\n".join(rows) + f"\n{pad}]"
    return json.dumps(value, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value) + "\n", encoding="utf-8")


def _dashed(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


# --------------------------------------------------------------------------- fetch (network)


def _provider(timeout: float):
    import akshare as ak
    import requests

    from query_intelligence.integrations.akshare_market_provider import AKShareMarketProvider
    from query_intelligence.integrations.sources import SourceHealthRegistry, SourceRuntime

    runtime = SourceRuntime(health=SourceHealthRegistry(failure_threshold=10_000), call_timeout_s=timeout)
    return AKShareMarketProvider(
        ak_module=ak,
        timeout=int(timeout),
        max_retries=0,
        retry_backoff_seconds=0,
        runtime=runtime,
        http_get=lambda url, headers, timeout: requests.get(url, headers=headers, timeout=timeout),
    ), ak


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _timed(label: str, fn, pause: float) -> dict[str, Any]:
    started, fetched_at = time.perf_counter(), _now()
    try:
        value, error = fn(), None
    except Exception as exc:  # recorded, never raised: one failed source must not stop the build
        value, error = None, f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"
    record = {"fetched_at": fetched_at, "latency_ms": round((time.perf_counter() - started) * 1000), "error": error}
    print(f"  {label}: {'ok' if error is None else 'FAIL ' + error} ({record['latency_ms']} ms)", flush=True)
    time.sleep(pause)
    return {**record, "value": value}


def fetch_instrument(symbol: str, name: str, product: str, *, timeout: float, pause: float) -> dict[str, Any]:
    provider, ak = _provider(timeout)
    code = symbol.split(".")[0]
    raw: dict[str, Any] = {"symbol": symbol, "name": name, "product_type": product, "window": [START, AS_OF]}

    def market() -> dict[str, Any]:
        # Index codes collide with stock codes (000001): the provider picks the exchange from the code, so
        # pass the prefixed symbol the index chain expects.
        with provider.runtime.trace() as attempts:
            rows, source_name, warnings, endpoint = provider._fetch_market_rows(code, product, START, AS_OF)
        first, last = _dashed(START), _dashed(AS_OF)
        rows = [row for row in rows if row.get("trade_date") and first <= row["trade_date"][:10] <= last]
        return {
            "endpoint": endpoint,
            "source_name": source_name,
            "attempts": list(attempts),
            "warnings": warnings,
            "rows": [{key: _plain(value) for key, value in row.items()} for row in rows],
        }

    result = _timed(f"{symbol} market", market, pause)
    raw["market"] = {**(result.pop("value") or {}), **result}
    if product != "stock":
        return raw

    period_floor = f"{FINANCIAL_START_YEAR}-01-01"

    def in_range(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        return [row for row in rows if str(row.get(key) or "")[:10] >= period_floor]

    sina = _timed(
        f"{symbol} sina.finance",
        lambda: in_range(_records(ak.stock_financial_analysis_indicator(symbol=code, start_year=FINANCIAL_START_YEAR)), "日期"),
        pause,
    )
    raw["sina_finance"] = {
        "endpoint": "akshare.stock_financial_analysis_indicator",
        "params": {"symbol": code, "start_year": FINANCIAL_START_YEAR},
        "rows": sina.pop("value"),
        **sina,
    }
    ths = _timed(
        f"{symbol} ths.finance",
        lambda: in_range(_records(ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")), "报告期"),
        pause,
    )
    raw["ths_finance"] = {
        "endpoint": "akshare.stock_financial_abstract_ths",
        "params": {"symbol": code, "indicator": "按报告期"},
        "rows": ths.pop("value"),
        **ths,
    }
    as_of = _dashed(AS_OF)
    valuation = _timed(
        f"{symbol} eastmoney.datacenter",
        lambda: [row for row in _records(ak.stock_value_em(symbol=code)) if str(row.get("数据日期"))[:10] == as_of],
        pause,
    )
    raw["valuation"] = {
        "endpoint": "akshare.stock_value_em",
        "params": {"symbol": code},
        "rows": valuation.pop("value"),
        **valuation,
    }
    return raw


def cmd_fetch(args: argparse.Namespace) -> int:
    selected = [item for item in INSTRUMENTS if not args.only or item[0] in args.only]
    for symbol, name, product in selected:
        print(f"{symbol} {name} ({product})", flush=True)
        raw = fetch_instrument(symbol, name, product, timeout=args.timeout, pause=args.pause)
        _write(RAW_DIR / f"{symbol}.json", raw)
    return 0


# --------------------------------------------------------------------------- build (offline, deterministic)


class BuildError(RuntimeError):
    pass


def _check_columns(kind: str, rows: list[dict[str, Any]], symbol: str) -> None:
    if not rows:
        raise BuildError(f"{symbol}: no {kind} rows in the raw file")
    missing = REQUIRED_COLUMNS[kind] - set(rows[0])
    if missing:
        raise BuildError(f"{symbol}: {kind} schema drift, missing columns {sorted(missing)}")


def build_market(raw: dict[str, Any]) -> dict[str, Any]:
    from query_intelligence.integrations.akshare_market_provider import _VOLUME_UNITS
    from query_intelligence.integrations.sources.catalog import source_for_endpoint

    symbol, market = raw["symbol"], raw.get("market") or {}
    rows = sorted(market.get("rows") or [], key=lambda row: row["trade_date"], reverse=True)
    if market.get("error") or len(rows) < 250:
        raise BuildError(f"{symbol}: {len(rows)} daily bars (need >= 250); fetch error: {market.get('error')}")
    if rows[0]["trade_date"][:10] != _dashed(AS_OF):
        raise BuildError(f"{symbol}: latest bar {rows[0]['trade_date']} is not the as-of date {_dashed(AS_OF)}")
    fields = ("trade_date", "open", "high", "low", "close", "pct_change_1d", "volume", "amount")
    history = [{key: row.get(key) for key in fields} for row in rows]
    latest = history[0]
    source_id = source_for_endpoint(market["endpoint"])
    payload: dict[str, Any] = {
        "symbol": symbol,
        "canonical_name": raw["name"],
        **latest,
        "history": history,
        "source_name": market["source_name"],
        "_snapshot": {
            "version": VERSION,
            "file": "data/snapshot/structured_data_ext.json",
            "endpoint": market["endpoint"],
            "source": source_id,
            "fetched_at": market["fetched_at"],
        },
    }
    if source_id in _VOLUME_UNITS:
        payload["volume_unit"] = _VOLUME_UNITS[source_id]
    return payload


def build_fundamentals(raw: dict[str, Any], close: float) -> dict[str, Any]:
    from query_intelligence.integrations.akshare_market_provider import AKShareMarketProvider
    from query_intelligence.integrations.sources.crosscheck import SINA, reconcile_fundamentals
    from query_intelligence.integrations.sources.values import to_number

    symbol = raw["symbol"]
    sina_rows, ths_rows = raw["sina_finance"]["rows"] or [], raw["ths_finance"]["rows"] or []
    _check_columns("sina", sina_rows, symbol)
    _check_columns("ths", ths_rows, symbol)
    parser = AKShareMarketProvider(ak_module=None)
    # the same period parsers and cross-check as the live path, restricted to periods up to FY2025
    sina = {period: row for period, row in parser._sina_periods(sina_rows).items() if period <= REPORT_PERIOD}
    ths = {period: row for period, row in parser._ths_periods(ths_rows).items() if period <= REPORT_PERIOD}
    report, check = reconcile_fundamentals(sina, ths, primary=SINA)
    if report is None or check is None or report.get("report_date") != REPORT_PERIOD:
        raise BuildError(f"{symbol}: no {REPORT_PERIOD} report (got {report and report.get('report_date')})")
    if report.get("revenue") is None or report.get("net_profit") is None:
        raise BuildError(f"{symbol}: {REPORT_PERIOD} report has no revenue/net profit level")

    valuation_rows = raw["valuation"]["rows"] or []
    _check_columns("valuation", valuation_rows, symbol)
    valuation = valuation_rows[0]
    valuation_close = to_number(valuation.get("当日收盘价"))
    if valuation_close is None or abs(valuation_close - close) > 0.011:
        raise BuildError(f"{symbol}: valuation close {valuation_close} != market close {close} on {_dashed(AS_OF)}")
    for field in AMOUNT_FIELDS:  # "4585.02亿" parses to 458502000000.00006: amounts are kept to the fen
        if report.get(field) is not None:
            report[field] = round(float(report[field]), 2)
    served = check.served_source
    finance = raw["sina_finance"] if served == SINA else raw["ths_finance"]
    payload: dict[str, Any] = {
        "symbol": symbol,
        "canonical_name": raw["name"],
        "report_date": REPORT_PERIOD,
        **{key: value for key, value in report.items() if key != "report_date"},
        **{field: _amount(field, to_number(valuation.get(column))) for column, field in VALUATION_FIELDS.items()},
        "valuation_date": _dashed(AS_OF),
        "source_name": "akshare",
        "_snapshot": {
            "version": VERSION,
            "file": "data/snapshot/structured_data_ext.json",
            "endpoint": finance["endpoint"],
            "source": served,
            "fetched_at": finance["fetched_at"],
            "cross_check": check.status,
            "valuation_endpoint": raw["valuation"]["endpoint"],
        },
    }
    return payload


def build(raw_files: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    ext: dict[str, Any] = {"version": VERSION, "as_of": _dashed(AS_OF), "market_api": {}, "fundamental_sql": {}}
    manifest: dict[str, Any] = {
        "version": VERSION,
        "as_of": _dashed(AS_OF),
        "report_period": REPORT_PERIOD,
        "window": [_dashed(START), _dashed(AS_OF)],
        "build_command": "python -m scripts.build_offline_snapshot build",
        "fetch_command": "python -m scripts.build_offline_snapshot fetch",
        "base_snapshot": {"file": "data/structured_data.json", "sha256": sha256_file(BASE_PATH)},
        "instruments": [],
    }
    for path in raw_files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        symbol = raw["symbol"]
        for section in ("market_api", "fundamental_sql"):
            if symbol in base.get(section, {}):
                raise BuildError(f"{symbol} is already in v1 {section}: the extension never replaces v1 values")
        market = build_market(raw)
        ext["market_api"][symbol] = market
        entry: dict[str, Any] = {
            "symbol": symbol,
            "name": raw["name"],
            "product_type": raw["product_type"],
            "raw_file": f"data/snapshot/raw/{path.name}",
            "raw_sha256": sha256_file(path),
            "market": {
                "endpoint": raw["market"]["endpoint"],
                "attempts": raw["market"]["attempts"],
                "fetched_at": raw["market"]["fetched_at"],
                "closes": len(market["history"]),
                "first_date": market["history"][-1]["trade_date"],
                "as_of": market["trade_date"],
            },
        }
        if raw["product_type"] == "stock":
            fundamentals = build_fundamentals(raw, float(market["close"]))
            ext["fundamental_sql"][symbol] = fundamentals
            entry["fundamentals"] = {
                "report_date": fundamentals["report_date"],
                "served_by": fundamentals["_snapshot"]["source"],
                "cross_check": fundamentals["_snapshot"]["cross_check"],
                "fetched_at": fundamentals["_snapshot"]["fetched_at"],
                "valuation_endpoint": raw["valuation"]["endpoint"],
                "valuation_date": fundamentals["valuation_date"],
            }
        manifest["instruments"].append(entry)
    return ext, manifest


def _raw_files() -> list[Path]:
    files = sorted(RAW_DIR.glob("*.json"))
    if not files:
        raise BuildError(f"no raw files in {RAW_DIR}; run the fetch step first")
    known = {symbol for symbol, _name, _product in INSTRUMENTS}
    unknown = [path.name for path in files if path.stem not in known]
    if unknown:
        raise BuildError(f"raw files for instruments not in INSTRUMENTS: {unknown}")
    return files


def cmd_build(_args: argparse.Namespace) -> int:
    ext, manifest = build(_raw_files())
    _write(EXT_PATH, ext)
    manifest["output"] = {"file": "data/snapshot/structured_data_ext.json", "sha256": sha256_file(EXT_PATH)}
    _write(MANIFEST_PATH, manifest)
    print(f"wrote {EXT_PATH.relative_to(ROOT)}: {len(ext['market_api'])} priced, {len(ext['fundamental_sql'])} fundamentals")
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    """Offline check: raw hashes match the manifest, a rebuild equals the committed output, no v1 overlap."""
    problems: list[str] = []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in manifest["instruments"]:
        path = ROOT / entry["raw_file"]
        if not path.exists() or sha256_file(path) != entry["raw_sha256"]:
            problems.append(f"raw hash mismatch: {entry['raw_file']}")
    if sha256_file(EXT_PATH) != manifest["output"]["sha256"]:
        problems.append("output hash mismatch: structured_data_ext.json")
    if sha256_file(BASE_PATH) != manifest["base_snapshot"]["sha256"]:
        problems.append("data/structured_data.json changed since the extension was built")
    ext, _ = build(_raw_files())
    if dumps(ext) + "\n" != EXT_PATH.read_text(encoding="utf-8"):
        problems.append("rebuild from raw differs from the committed structured_data_ext.json")
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print(f"OK: {len(manifest['instruments'])} instruments, hashes match, rebuild identical, no v1 overlap")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["fetch", "build", "verify", "all"])
    parser.add_argument("--only", nargs="*", help="Fetch only these symbols (e.g. 300750.SZ).")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-call hard timeout in seconds.")
    parser.add_argument("--pause", type=float, default=1.5, help="Seconds to wait after every live call.")
    args = parser.parse_args(argv)
    try:
        if args.step in {"fetch", "all"}:
            cmd_fetch(args)
        if args.step in {"build", "all"}:
            cmd_build(args)
        if args.step in {"verify", "all"}:
            return cmd_verify(args)
    except BuildError as exc:
        print(f"BUILD ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
