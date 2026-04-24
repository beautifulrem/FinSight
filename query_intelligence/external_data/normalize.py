from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .adapters import (
    adapt_baai_finance_instruction_rows,
    adapt_cflue_rows,
    adapt_dailydialog_rows,
    adapt_chnsenticorp_rows,
    adapt_cluener_rows,
    adapt_csprd_rows,
    adapt_financial_news_rows,
    adapt_finfe_rows,
    adapt_fiqa_rows,
    adapt_fincprg_rows,
    adapt_finnl_rows,
    adapt_fir_bench_announcement_rows,
    adapt_fir_bench_report_rows,
    adapt_msra_rows,
    adapt_mxode_finance_rows,
    adapt_naturalconv_rows,
    adapt_peoples_daily_rows,
    adapt_qrecc_rows,
    adapt_risawoz_rows,
    adapt_smp2017_rows,
    adapt_t2ranking_rows,
    adapt_thucnews_rows,
    adapt_tnews_rows,
)

_RAW_FILE_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".txt", ".parquet", ".train", ".dev", ".test"}
_DIRECTORY_ADAPTER_SOURCES = {"fiqa", "t2ranking", "smp2017", "thucnews", "fincprg", "naturalconv", "dailydialog"}
_SOURCE_PLAN_CACHE_SOURCES = {
    "cflue",
    "finnl",
    "tnews",
    "thucnews",
    "smp2017",
    "fin_news_sentiment",
    "mxode_finance",
    "baai_finance_instruction",
}
_ALIAS_CACHE_SOURCES = {"fin_news_sentiment"}

_ADAPTER_REGISTRY: dict[str, Callable[[Any], list[dict]]] = {
    "cflue": adapt_cflue_rows,
    "finfe": adapt_finfe_rows,
    "chnsenticorp": adapt_chnsenticorp_rows,
    "fin_news_sentiment": adapt_financial_news_rows,
    "fiqa": adapt_fiqa_rows,
    "msra_ner": adapt_msra_rows,
    "peoples_daily_ner": adapt_peoples_daily_rows,
    "cluener": adapt_cluener_rows,
    "tnews": adapt_tnews_rows,
    "thucnews": adapt_thucnews_rows,
    "finnl": adapt_finnl_rows,
    "t2ranking": adapt_t2ranking_rows,
    "csprd": adapt_csprd_rows,
    "smp2017": adapt_smp2017_rows,
    "mxode_finance": adapt_mxode_finance_rows,
    "baai_finance_instruction": adapt_baai_finance_instruction_rows,
    "qrecc": adapt_qrecc_rows,
    "risawoz": adapt_risawoz_rows,
    "naturalconv": adapt_naturalconv_rows,
    "dailydialog": adapt_dailydialog_rows,
    "fincprg": adapt_fincprg_rows,
    "fir_bench_reports": adapt_fir_bench_report_rows,
    "fir_bench_announcements": adapt_fir_bench_announcement_rows,
}


def route_source_to_adapter(source_id: str) -> Callable[[Any], list[dict]]:
    adapter = _ADAPTER_REGISTRY.get(source_id)
    if adapter is None:
        raise ValueError(f"Unknown source: {source_id}")
    return adapter


def load_standardized_rows(raw_root: Path) -> list[dict]:
    root = Path(raw_root)
    if not root.exists():
        return []

    rows: list[dict] = []
    for source_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for version_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            rows.extend(_load_rows_for_version(source_dir.name, version_dir))
    return rows


def _load_rows_for_version(source_id: str, version_dir: Path) -> list[dict]:
    cache_path = version_dir / "records.jsonl"
    if cache_path.exists():
        cached_rows = _load_rows_from_records_file(cache_path, source_id, version_dir.name)
        if not _cache_requires_refresh(source_id, cached_rows):
            return cached_rows

    try:
        adapter = route_source_to_adapter(source_id)
    except ValueError:
        return []

    if source_id in _DIRECTORY_ADAPTER_SOURCES:
        try:
            rows = [_standardize_row(row, source_id, version_dir.name) for row in adapter(version_dir)]
        except Exception:
            rows = []
        if rows:
            _write_records_cache(cache_path, rows)
        return rows

    rows: list[dict] = []
    for raw_path in _discover_raw_files(version_dir):
        try:
            payloads = _adapter_payloads_for_file(raw_path, source_id)
        except OSError:
            continue
        for payload in payloads:
            try:
                adapted_rows = adapter(payload)
            except Exception:
                continue
            for row in adapted_rows:
                rows.append(_standardize_row(row, source_id, version_dir.name))

    if rows:
        _write_records_cache(cache_path, rows)
    return rows


def _cache_requires_refresh(source_id: str, rows: list[dict]) -> bool:
    if not rows:
        return True

    if source_id in _SOURCE_PLAN_CACHE_SOURCES:
        has_source_plan_fields = any(
            row.get("sample_family") == "classification"
            and (row.get("expected_document_sources") or row.get("expected_structured_sources"))
            for row in rows
        )
        if not has_source_plan_fields:
            return True

    if source_id in _ALIAS_CACHE_SOURCES:
        has_alias_rows = any(row.get("sample_family") == "alias" for row in rows)
        if not has_alias_rows:
            return True

    return False


def _load_rows_from_records_file(path: Path, source_id: str, source_version: str) -> list[dict]:
    rows: list[dict] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return []
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            rows.append(_standardize_row(payload, source_id, source_version))
    return rows


def _discover_raw_files(version_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(version_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "records.jsonl":
            continue
        if path.suffix.lower() in _RAW_FILE_SUFFIXES:
            files.append(path)
    return files


def _adapter_payloads_for_file(path: Path, source_id: str) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        if source_id in {"msra_ner", "peoples_daily_ner"}:
            return [path]
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return []
        return [{"text": line} for line in lines]
    if suffix == ".parquet":
        return _parquet_payloads(path)
    return [path]


def _parquet_payloads(path: Path) -> list[Any]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return []

    try:
        frame = pd.read_parquet(path)
    except Exception:
        return []

    records = frame.to_dict(orient="records")
    return [record for record in records if isinstance(record, dict)]


def _standardize_row(row: dict, source_id: str, source_version: str) -> dict:
    standardized = dict(row)
    standardized.setdefault("source_id", source_id)
    standardized.setdefault("source_version", source_version)
    return standardized


def _write_records_cache(path: Path, rows: list[dict]) -> None:
    try:
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
            encoding="utf-8",
        )
    except OSError:
        return


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        fingerprint = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def assign_split_groups(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _split_lock_key(row)
        grouped.setdefault(key, []).append(dict(row))

    assigned: list[dict] = []
    for key in sorted(grouped):
        split = _split_for_key(key)
        for row in grouped[key]:
            row["split"] = split
            assigned.append(row)
    return assigned


def _split_lock_key(row: dict) -> str:
    for field in ("split_lock_key", "sample_id", "query_id", "text"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _split_for_key(key: str) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 10
    if bucket == 8:
        return "valid"
    if bucket == 9:
        return "test"
    return "train"
