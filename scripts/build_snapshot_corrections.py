"""Build the offline corrections layer (``data/snapshot/structured_data_fix.json``) from primary sources.

The v1 seed (``data/structured_data.json``) is never edited: every evaluation label was computed from it. Some of
its values are contradicted by the companies' own FY2025 annual reports (五粮液 revenue 1085亿 vs 405.29亿 reported)
or by the daily record (中国平安's 2026-04-22 close 53.61 vs 58.03), and none of them says which definition a figure
uses (ROE weighted or on period-end equity, 营业收入 or 营业总收入, 归母 or total net profit, basic or diluted EPS).
This layer holds:

* ``fundamental_sql``: FY2025 statements of the v1 companies, every figure transcribed from the annual report
  (``data/snapshot/corrections/sources.json``: document, URL, sha256 of the PDF, page, the quoted line), with the
  P/E and P/B derived from those figures and the 2026-04-22 close;
* ``market_api``: the 2026-04-22 bars of the v1 instruments whose seed bars disagree with the daily record
  (index compiler CSIndex for 沪深300; Sina and Eastmoney, which agree, for the stock and the funds);
* ``definitions``: the field definitions of every company's served figures, v1 and extension.

The runtime applies it over v1 (``query_intelligence/data_loader.py``, precedence corrections > v1 seed >
extension; ``QI_OFFLINE_SNAPSHOT_FIX``, default on). Evaluation stays pinned to the seed (``QI_EVAL_SNAPSHOT_FIX``).

    python -m scripts.build_snapshot_corrections fetch     # live: daily bars -> data/snapshot/raw/corrections/
    python -m scripts.build_snapshot_corrections build     # offline, deterministic -> structured_data_fix.json
    python -m scripts.build_snapshot_corrections verify    # offline: hashes, rebuild == committed, quotes parse
    python -m scripts.build_snapshot_corrections documents --cache DIR   # live: re-download every annual report,
                                                           # check its sha256 and that each quote is on its page
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_offline_snapshot import dumps, sha256_file  # noqa: E402

SNAPSHOT_DIR = ROOT / "data" / "snapshot"
SOURCES_PATH = SNAPSHOT_DIR / "corrections" / "sources.json"
RAW_PATH = SNAPSHOT_DIR / "raw" / "corrections" / "daily_2026-04-22.json"
FIX_PATH = SNAPSHOT_DIR / "structured_data_fix.json"
MANIFEST_PATH = SNAPSHOT_DIR / "manifest_fix.json"
BASE_PATH = ROOT / "data" / "structured_data.json"
EXT_PATH = SNAPSHOT_DIR / "structured_data_ext.json"

VERSION = "fix-2026-10-10"
FILE = "data/snapshot/structured_data_fix.json"
TRADE_DATE = "2026-04-22"  # v1's last trading day
WINDOW = ("20260415", "20260422")

# The v1 instruments checked against the daily record. ``correct``: the seed bars disagree, the layer serves the
# record; ``check``: the seed bars agree (verified, unchanged). (symbol, product, primary source, cross-check source)
# The index comes from its compiler (China Securities Index Co.); stocks and funds from Sina's daily kline, checked
# against Tencent's (a separate kline service).
PRICE_CHECKS: list[tuple[str, str, str, str | None, str]] = [
    ("601318.SH", "stock", "sina.stock", "tencent.kline", "correct"),
    ("000300.SH", "index", "csindex.index", "sina.index", "correct"),
    ("159915.SZ", "etf", "sina.etf", "tencent.kline", "correct"),
    ("512880.SH", "etf", "sina.etf", "tencent.kline", "correct"),
    ("600519.SH", "stock", "sina.stock", "tencent.kline", "check"),
    ("000858.SZ", "stock", "sina.stock", "tencent.kline", "check"),
    ("510300.SH", "etf", "sina.etf", "tencent.kline", "check"),
]
ENDPOINTS = {
    "sina.stock": "akshare.stock_zh_a_daily",
    "tencent.kline": "akshare.stock_zh_a_hist_tx",
    "csindex.index": "akshare.stock_zh_index_hist_csindex",
    "sina.index": "akshare.stock_zh_index_daily",
    "sina.etf": "akshare.fund_etf_hist_sina",
}
# Source ids of the provenance catalog (query_intelligence/integrations/sources/catalog.py).
CATALOG_SOURCE = {"sina.stock": "sina.kline", "sina.etf": "sina.kline", "csindex.index": "csindex.index"}


class BuildError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dashed(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


# --------------------------------------------------------------------------- fetch (network)


def _fetch_rows(source: str, symbol: str) -> list[dict[str, Any]]:
    import akshare as ak

    from scripts.build_offline_snapshot import _records

    code, exchange = symbol.split(".")
    prefixed = f"{exchange.lower()}{code}"
    start, end = WINDOW
    if source == "sina.stock":
        frame = ak.stock_zh_a_daily(symbol=prefixed, start_date=start, end_date=end)
    elif source == "tencent.kline":
        frame = ak.stock_zh_a_hist_tx(symbol=prefixed, start_date=start, end_date=end)
    elif source == "csindex.index":
        frame = ak.stock_zh_index_hist_csindex(symbol=code, start_date=start, end_date=end)
    elif source == "sina.index":
        frame = ak.stock_zh_index_daily(symbol=prefixed)
    elif source == "sina.etf":
        frame = ak.fund_etf_hist_sina(symbol=prefixed)
    else:
        raise BuildError(f"unknown source {source}")
    rows = _records(frame)
    first, last = _dashed(start), _dashed(end)
    date_key = next((key for key in ("date", "日期") if rows and key in rows[0]), None)
    return [row for row in rows if date_key and first <= str(row[date_key])[:10] <= last]


def cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch every source's bars; a source that already has rows in the raw file is kept (``--refresh`` refetches
    all), so a rerun only retries the sources that failed (Eastmoney's kline host drops connections at times)."""
    raw: dict[str, Any] = {"trade_date": TRADE_DATE, "window": [_dashed(WINDOW[0]), _dashed(WINDOW[1])], "records": {}}
    if RAW_PATH.exists() and not args.refresh:
        raw["records"] = json.loads(RAW_PATH.read_text(encoding="utf-8")).get("records") or {}
    for symbol, _product, primary, cross, _action in PRICE_CHECKS:
        for source in (primary, cross):
            if source is None:
                continue
            kept = raw["records"].get(f"{symbol}|{source}") or {}
            if kept.get("rows") and not kept.get("error"):
                continue
            fetched_at, started = _now(), time.perf_counter()
            try:
                rows, error = _fetch_rows(source, symbol), None
            except Exception as exc:  # recorded: one failed source must not stop the fetch
                rows, error = [], f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"
            print(f"  {symbol} {source}: {'ok' if error is None else 'FAIL ' + error}", flush=True)
            raw["records"][f"{symbol}|{source}"] = {
                "symbol": symbol,
                "source": source,
                "endpoint": ENDPOINTS[source],
                "fetched_at": fetched_at,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "error": error,
                "rows": rows,
            }
            time.sleep(args.pause)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(dumps(raw) + "\n", encoding="utf-8")
    print(f"wrote {RAW_PATH.relative_to(ROOT)}")
    return 0


# --------------------------------------------------------------------------- build (offline, deterministic)

_UNIT_SCALE = {"元": 1.0, "千元": 1e3, "百万元": 1e6, "亿元": 1e8, "%": 1.0, "元/股": 1.0, "股": 1.0}


def parse_figure(figure: dict[str, Any]) -> float:
    """The number a figure states, in 元 (amounts), percent (ratios), 元/股 or shares. The written value must appear in
    the quoted line (whitespace-insensitive), so a transcription error fails the build."""
    text = str(figure["value"])
    if _squash(text) not in _squash(figure["quote"]):
        raise BuildError(f"{figure['id']}: value {text!r} is not in its quote {figure['quote']!r}")
    number = float(text.replace(",", "").replace("%", "").strip())
    unit = figure["unit"]
    if unit not in _UNIT_SCALE:
        raise BuildError(f"{figure['id']}: unknown unit {unit!r}")
    return number * _UNIT_SCALE[unit]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _figures(company: dict[str, Any]) -> dict[str, float]:
    values = {}
    for figure in company["figures"]:
        if figure["id"] in values:
            raise BuildError(f"{company['symbol']}: figure {figure['id']} twice")
        values[figure["id"]] = parse_figure(figure)
    return values


def _bar(raw: dict[str, Any], symbol: str, source: str, day: str) -> dict[str, Any] | None:
    record = raw["records"].get(f"{symbol}|{source}") or {}
    for row in record.get("rows") or []:
        if str(row.get("date") or row.get("日期"))[:10] == day:
            return row
    return None


def _num(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _sessions(raw: dict[str, Any], symbol: str, source: str) -> list[str]:
    record = raw["records"].get(f"{symbol}|{source}") or {}
    return sorted({str(row.get("date") or row.get("日期"))[:10] for row in record.get("rows") or []})


def _normalised_bar(raw: dict[str, Any], symbol: str, source: str, day: str) -> dict[str, Any]:
    """One daily bar in the seed's fields, from ``source``; the daily change from the previous session's close
    (CSIndex states its own 涨跌幅, used as published)."""
    row = _bar(raw, symbol, source, day)
    if row is None:
        raise BuildError(f"{symbol}: no {source} bar on {day}")
    sessions = _sessions(raw, symbol, source)
    previous_day = sessions[sessions.index(day) - 1] if sessions.index(day) > 0 else None
    previous = _bar(raw, symbol, source, previous_day) if previous_day else None
    if source == "csindex.index":
        bar = {
            "open": _num(row, "开盘"),
            "high": _num(row, "最高"),
            "low": _num(row, "最低"),
            "close": _num(row, "收盘"),
            "pct_change_1d": _num(row, "涨跌幅"),
            "volume": _num(row, "成交量"),
            "amount": round(_num(row, "成交金额") * 1e8, 2),  # 亿元 -> 元
        }
    else:
        bar = {key: _num(row, key) for key in ("open", "high", "low", "close", "volume", "amount")}
        prior = _num(previous, "close") if previous else None
        bar["pct_change_1d"] = round((bar["close"] / prior - 1) * 100, 2) if prior else None
    if any(value is None for key, value in bar.items() if key != "pct_change_1d"):
        raise BuildError(f"{symbol}: incomplete {source} bar on {day}: {bar}")
    bar["volume"] = round(bar["volume"])
    return {"trade_date": day, **bar}


def _cross_check(raw: dict[str, Any], symbol: str, primary: str, cross: str | None, bar: dict[str, Any]) -> str:
    """``agree`` when the second source's close (and change, when it states one) matches the primary's."""
    if cross is None:
        return "single_source"
    row = _bar(raw, symbol, cross, bar["trade_date"])
    if row is None:
        raise BuildError(f"{symbol}: cross-check source {cross} has no bar on {bar['trade_date']}")
    close = _num(row, "close", "收盘")
    if close is None or abs(close - bar["close"]) > 0.0051:
        raise BuildError(f"{symbol}: {cross} close {close} != {primary} close {bar['close']} on {bar['trade_date']}")
    change = _num(row, "涨跌幅")
    if change is not None and bar.get("pct_change_1d") is not None and abs(change - bar["pct_change_1d"]) > 0.011:
        raise BuildError(f"{symbol}: {cross} change {change} != {primary} change {bar['pct_change_1d']}")
    return "agree"


def build_market(raw: dict[str, Any], base: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from query_intelligence.integrations.akshare_market_provider import _VOLUME_UNITS

    corrected: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    for symbol, _product, primary, cross, action in PRICE_CHECKS:
        seed = base["market_api"][symbol]
        seed_days = [row["trade_date"] for row in seed.get("history") or []] or [seed["trade_date"]]
        bars = [_normalised_bar(raw, symbol, primary, day) for day in seed_days]
        latest = bars[0]
        if latest["trade_date"] != TRADE_DATE:
            raise BuildError(f"{symbol}: latest bar {latest['trade_date']} is not {TRADE_DATE}")
        agreement = _cross_check(raw, symbol, primary, cross, latest)
        seed_close = float(seed["close"])
        matches = abs(seed_close - latest["close"]) <= 0.0051
        entry = {
            "symbol": symbol,
            "action": action,
            "source": primary,
            "endpoint": ENDPOINTS[primary],
            "cross_check": f"{cross}:{agreement}" if cross else agreement,
            "seed": {"close": seed["close"], "pct_change_1d": seed.get("pct_change_1d")},
            "record": {"close": latest["close"], "pct_change_1d": latest["pct_change_1d"]},
            "seed_matches_record": matches,
        }
        if action == "check":
            if not matches:
                raise BuildError(f"{symbol}: seed close {seed_close} != record {latest['close']}; mark it 'correct'")
            checks.append(entry)
            continue
        if matches and abs((seed.get("pct_change_1d") or 0) - (latest["pct_change_1d"] or 0)) <= 0.011:
            raise BuildError(f"{symbol}: seed bar agrees with the record; nothing to correct")
        source_id = CATALOG_SOURCE.get(primary, primary)
        record = {
            "symbol": symbol,
            **({"canonical_name": seed["canonical_name"]} if seed.get("canonical_name") else {}),
            **latest,
            "history": bars,
            "source_name": "akshare",
            "_snapshot": {
                "version": VERSION,
                "file": FILE,
                "endpoint": ENDPOINTS[primary],
                "source": source_id,
                "fetched_at": raw["records"][f"{symbol}|{primary}"]["fetched_at"],
                "correction": f"replaces the v1 seed bars ({', '.join(seed_days)}), which disagree with this record",
                "cross_check": entry["cross_check"],
            },
        }
        if source_id in _VOLUME_UNITS:
            record["volume_unit"] = _VOLUME_UNITS[source_id]
        elif primary == "csindex.index":
            record["volume_unit"] = "share"  # CSIndex states 成交量 in shares (Sina's index kline agrees)
        corrected[symbol] = record
        checks.append(entry)
    return corrected, checks


def _priors(entry: dict[str, Any]) -> dict[str, float]:
    """Prior-year (FY2024) values of the figures that state one, in the figure's unit scale."""
    out = {}
    for figure in entry["figures"]:
        if figure.get("prior") is not None:
            if _squash(str(figure["prior"])) not in _squash(figure["quote"]):
                raise BuildError(f"{figure['id']}: prior {figure['prior']!r} is not in its quote")
            out[figure["id"]] = float(str(figure["prior"]).replace(",", "")) * _UNIT_SCALE[figure["unit"]]
    return out


# Which report figure each basis names, per served field: the served value must equal it (amounts within 0.05%,
# ratios within 0.011 points, EPS within half a fen), so a definition is checked against the report, not asserted.
_LEVEL_BASIS = {
    "revenue": ("revenue_basis", {"营业收入": "revenue", "营业总收入": "revenue_total"}),
    "net_profit": ("net_profit_basis", {"归母": "net_profit_parent", "total": "net_profit_total"}),
    "eps": ("eps_basis", {"basic": "eps_basic", "diluted": "eps_diluted"}),
}


def check_definitions(symbol: str, record: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    """The definitions of an extension record, checked against the figures quoted from its annual report."""
    values, priors, definitions = _figures(entry), _priors(entry), entry["definitions"]
    problems = []

    def near(served: Any, expected: float, tolerance: float, relative: bool = False) -> bool:
        if served is None:
            return False
        gap = abs(float(served) - expected)
        if relative:
            return gap <= abs(expected) * tolerance
        # a figure the report prints to one decimal (中国平安 YoY 6.5, 中芯国际 weighted ROE 3.4) holds to half a unit
        decimals = len(repr(float(served)).split(".")[1])
        return gap <= max(tolerance, 0.5 * 10**-decimals + 1e-9)

    for field, (basis_key, figures) in _LEVEL_BASIS.items():
        figure = figures.get(definitions.get(basis_key, ""))
        if figure is None or figure not in values:
            problems.append(f"{symbol}: {basis_key}={definitions.get(basis_key)!r} has no quoted figure")
            continue
        tolerance, relative = (0.0051, False) if field == "eps" else (0.0005, True)
        if not near(record.get(field), values[figure], tolerance, relative):
            problems.append(f"{symbol}: {field} {record.get(field)} != {figure} {values[figure]}")
    roe_basis = definitions.get("roe_basis")
    if roe_basis == "weighted":
        expected = values["roe_weighted"]
    else:
        expected = values["net_profit_parent"] / values["equity_parent"] * 100
    # a weighted ROE printed to one decimal (中芯国际 3.4) is checked at that precision
    if roe_basis not in {"weighted", "period_end"} or not near(record.get("roe"), expected, 0.011):
        problems.append(f"{symbol}: roe {record.get('roe')} is not {roe_basis} ({expected:.4f})")
    yoy_bases = {"归母": "net_profit_parent", "total": "net_profit_total"}
    yoy_figure = yoy_bases.get(definitions.get("netprofit_yoy_basis", ""))
    if yoy_figure and record.get("netprofit_yoy") is not None:
        expected = (values[yoy_figure] / priors[yoy_figure] - 1) * 100
        if not near(record["netprofit_yoy"], expected, 0.011):
            served = record["netprofit_yoy"]
            problems.append(f"{symbol}: netprofit_yoy {served} != {yoy_figure} growth {expected:.4f}")
    revenue_yoy_figure = "revenue_total" if definitions.get("revenue_yoy_basis") == "营业总收入" else "revenue"
    if record.get("revenue_yoy") is not None and revenue_yoy_figure in priors:
        expected = (values[revenue_yoy_figure] / priors[revenue_yoy_figure] - 1) * 100
        if not near(record["revenue_yoy"], expected, 0.011):
            served = record["revenue_yoy"]
            problems.append(f"{symbol}: revenue_yoy {served} != {revenue_yoy_figure} growth {expected:.4f}")
    return problems


def build_fundamentals(
    sources: dict[str, Any], market: dict[str, Any], base: dict[str, Any], ext: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """FY2025 statements of the v1 companies from their annual reports, and every company's definitions."""
    statements: dict[str, Any] = {}
    definitions: dict[str, Any] = {}
    for company in sources["companies"]:
        symbol = company["symbol"]
        values = _figures(company)
        document = sources["documents"][company["document"]]
        serve = company["serve"]
        record: dict[str, Any] = {"symbol": symbol, "canonical_name": company["name"], "report_date": "2025-12-31"}
        for field, figure_id in serve.items():
            if figure_id not in values:
                raise BuildError(f"{symbol}: {field} refers to missing figure {figure_id}")
            record[field] = round(values[figure_id], 4)
        derived = company.get("derived") or {}
        if "gross_margin" in derived:
            revenue, cost = values[derived["gross_margin"][0]], values[derived["gross_margin"][1]]
            record["gross_margin"] = round((revenue - cost) / revenue * 100, 2)
        if "valuation" in derived:
            spec = derived["valuation"]
            bar = market.get(symbol) or base["market_api"][symbol]
            if bar["trade_date"] != spec["date"]:
                raise BuildError(f"{symbol}: no close on {spec['date']} for the valuation")
            close, shares = float(bar["close"]), values[spec["shares"]]
            market_cap = close * shares
            record["pe_ttm"] = round(market_cap / values[spec["net_profit"]], 2)
            record["pb"] = round(market_cap / values[spec["equity"]], 2)
            record["total_mv"] = round(market_cap, 2)
            record["valuation_date"] = spec["date"]
        record["source_name"] = "annual_report"
        record["definitions"] = dict(company["definitions"])
        record["_snapshot"] = {
            "version": VERSION,
            "file": FILE,
            "endpoint": "cninfo.annual_report",
            "source": "cninfo.annual_report",
            "fetched_at": sources["retrieved"],
            "document": document["title"],
            "document_url": document["url"],
            "correction": "replaces the v1 seed statement, which the annual report contradicts",
        }
        statements[symbol] = record
    problems = []
    for symbol, entry in (sources.get("extension_definitions") or {}).items():
        record = (ext.get("fundamental_sql") or {}).get(symbol)
        if record is None:
            raise BuildError(f"{symbol}: extension definitions for a symbol the extension does not serve")
        problems.extend(check_definitions(symbol, record, entry))
        definitions[symbol] = dict(entry["definitions"])
    for company in sources["companies"]:  # the v1 corrections serve what their definitions say, by construction
        problems.extend(check_definitions(company["symbol"], statements[company["symbol"]], company))
    if problems:
        raise BuildError("definitions disagree with the quoted figures: " + "; ".join(problems))
    return statements, definitions


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    ext = json.loads(EXT_PATH.read_text(encoding="utf-8"))
    market, price_checks = build_market(raw, base)
    statements, definitions = build_fundamentals(sources, market, base, ext)
    for symbol in statements:
        if symbol not in base["fundamental_sql"]:
            raise BuildError(f"{symbol}: the layer corrects v1 records only")
    fix = {
        "version": VERSION,
        "retrieved": sources["retrieved"],
        "fundamental_sql": statements,
        "market_api": market,
        "definitions": definitions,
    }
    manifest = {
        "version": VERSION,
        "retrieved": sources["retrieved"],
        "build_command": "python -m scripts.build_snapshot_corrections build",
        "fetch_command": "python -m scripts.build_snapshot_corrections fetch",
        "base_snapshot": {"file": "data/structured_data.json", "sha256": sha256_file(BASE_PATH)},
        "extension": {"file": "data/snapshot/structured_data_ext.json", "sha256": sha256_file(EXT_PATH)},
        "sources": {"file": "data/snapshot/corrections/sources.json", "sha256": sha256_file(SOURCES_PATH)},
        "raw": {"file": "data/snapshot/raw/corrections/daily_2026-04-22.json", "sha256": sha256_file(RAW_PATH)},
        "documents": [
            {"symbol": key, **{k: v for k, v in value.items() if k in {"title", "url", "sha256", "published"}}}
            for key, value in sorted(sources["documents"].items())
        ],
        "statements": sorted(statements),
        "prices": price_checks,
        "definitions": sorted(definitions),
    }
    return fix, manifest


def cmd_build(_args: argparse.Namespace) -> int:
    fix, manifest = build()
    FIX_PATH.write_text(dumps(fix) + "\n", encoding="utf-8")
    manifest["output"] = {"file": FILE, "sha256": sha256_file(FIX_PATH)}
    MANIFEST_PATH.write_text(dumps(manifest) + "\n", encoding="utf-8")
    print(
        f"wrote {FILE}: {len(fix['fundamental_sql'])} statements, {len(fix['market_api'])} bars corrected, "
        f"{len(fix['definitions'])} extension definitions"
    )
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    """Offline: input hashes match the manifest, a rebuild equals the committed output (which re-parses every
    quoted figure and re-checks every price against its cross-check source)."""
    problems: list[str] = []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inputs = (("sources", SOURCES_PATH), ("raw", RAW_PATH), ("base_snapshot", BASE_PATH), ("extension", EXT_PATH))
    for key, path in inputs:
        if sha256_file(path) != manifest[key]["sha256"]:
            problems.append(f"hash mismatch: {manifest[key]['file']}")
    if sha256_file(FIX_PATH) != manifest["output"]["sha256"]:
        problems.append(f"output hash mismatch: {FILE}")
    try:
        fix, _ = build()
        if dumps(fix) + "\n" != FIX_PATH.read_text(encoding="utf-8"):
            problems.append(f"rebuild differs from the committed {FILE}")
    except BuildError as exc:
        problems.append(f"rebuild failed: {exc}")
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print(
            f"OK: {len(manifest['statements'])} statements from {len(manifest['documents'])} annual reports, "
            f"{sum(1 for p in manifest['prices'] if p['action'] == 'correct')} bars corrected and "
            f"{sum(1 for p in manifest['prices'] if p['action'] == 'check')} checked, "
            f"{len(manifest['definitions'])} extension definitions; hashes match, rebuild identical"
        )
    return 1 if problems else 0


def cmd_documents(args: argparse.Namespace) -> int:
    """Live: download every cited annual report, check its sha256, and that each quoted line is on its page (needs
    ``pypdf``; not a runtime dependency)."""
    import requests

    try:
        import pypdf
    except ImportError:
        print("pypdf is not installed: pip install pypdf (only this check needs it)")
        return 2
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    quotes: dict[str, list[dict[str, Any]]] = {}
    for company in sources["companies"]:
        quotes.setdefault(company["document"], []).extend(company["figures"])
    for entry in (sources.get("extension_definitions") or {}).values():
        quotes.setdefault(entry["document"], []).extend(entry.get("figures") or [])
    problems = []
    for key, document in sorted(sources["documents"].items()):
        path = cache / f"{document['announcement_id']}.pdf"
        if not path.exists():
            response = requests.get(document["url"], timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        if sha256_file(path) != document["sha256"]:
            problems.append(f"{key}: sha256 differs from the recorded {document['sha256'][:12]}")
            continue
        reader = pypdf.PdfReader(str(path))
        for figure in quotes.get(key, []):
            page = reader.pages[figure["page"] - 1].extract_text() or ""
            if _squash(figure["quote"]) not in _squash(page):
                problems.append(f"{key}: quote of {figure['id']} not found on PDF page {figure['page']}")
        print(f"  {key}: {document['title']} sha256 ok, {len(quotes.get(key, []))} quotes checked", flush=True)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print(f"OK: {len(sources['documents'])} documents, every quote found on its page")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["fetch", "build", "verify", "documents"])
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to wait after every live call.")
    parser.add_argument("--refresh", action="store_true", help="fetch: refetch sources that already have rows.")
    parser.add_argument("--cache", default=".cache/annual_reports", help="Download folder for 'documents'.")
    args = parser.parse_args(argv)
    try:
        if args.step == "fetch":
            return cmd_fetch(args)
        if args.step == "build":
            return cmd_build(args)
        if args.step == "documents":
            return cmd_documents(args)
        return cmd_verify(args)
    except BuildError as exc:
        print(f"BUILD ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
