from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from ..label_maps import CHNSENTICORP_SENTIMENT_MAP, FINFE_SENTIMENT_MAP
from .classification import _product_type_from_text, _source_labels_from_text


_ALIAS_CODE_RE = re.compile(r"\b(?P<code>\d{6})\.(?P<exchange>SH|SZ|HK)\b", re.IGNORECASE)
_TITLE_ALIAS_SPLIT_RE = re.compile(r"^[^A-Za-z0-9\u4e00-\u9fff]{0,2}(?P<name>[^：:，,;；()（）\s]{2,20})[：:]", re.UNICODE)
_GENERIC_TITLE_STOPWORDS = {"公告", "新闻", "快讯", "研报", "点评", "收评", "午评", "早评", "晚间", "财经", "市场"}


def _load_records(raw: Any) -> list[Mapping[str, Any]]:
    if isinstance(raw, Path):
        return _load_records_from_path(raw)
    if isinstance(raw, str):
        path = Path(raw)
        if path.exists():
            return _load_records_from_path(path)
        return [{"text": raw}]
    if isinstance(raw, Mapping):
        for key in ("records", "data", "items", "examples", "samples", "rows"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, Mapping)]
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, Mapping)]
    return []


def _load_records_from_path(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson", ".json"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            payload = json.loads(text)
            return _load_records(payload)
        rows: list[Mapping[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, Mapping):
                rows.append(item)
        return rows
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            return [row for row in reader]
    return _load_records(path.read_text(encoding="utf-8").splitlines())


def _first_text(record: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_present(record: Mapping[str, Any], fields: tuple[str, ...]) -> Any | None:
    for field in fields:
        if field in record and record.get(field) is not None:
            return record.get(field)
    return None


def _stable_key(*parts: Any) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _dedupe_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for label in labels:
        value = str(label).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _extend_available_labels(row: dict[str, Any], labels: list[str]) -> None:
    available = list(row.get("available_labels") or [])
    for label in labels:
        if label not in available:
            available.append(label)
    row["available_labels"] = available


def _attach_source_supervision(
    row: dict[str, Any],
    *,
    expected_document_sources: list[str] | None = None,
    expected_structured_sources: list[str] | None = None,
) -> dict[str, Any]:
    doc_sources = _dedupe_labels(expected_document_sources or [])
    structured_sources = _dedupe_labels(expected_structured_sources or [])
    if doc_sources:
        row["expected_document_sources"] = doc_sources
    if structured_sources:
        row["expected_structured_sources"] = structured_sources
    if doc_sources or structured_sources:
        _extend_available_labels(row, ["expected_document_sources", "expected_structured_sources"])
    return row


def _normalize_label(raw_label: Any, label_map: Mapping[int, str]) -> str | None:
    if raw_label is None:
        return None
    if isinstance(raw_label, str):
        text = raw_label.strip().lower()
        if text in {"negative", "neutral", "positive"}:
            return text
        if text.isdigit():
            raw_label = int(text)
        else:
            return text or None
    if isinstance(raw_label, (int, float)):
        return label_map.get(int(raw_label), str(int(raw_label)))
    return str(raw_label).strip() or None


def _build_sentiment_row(
    record: Mapping[str, Any],
    *,
    source_id: str,
    label_map: Mapping[int, str],
    text_fields: tuple[str, ...],
    label_fields: tuple[str, ...],
) -> dict[str, Any]:
    text = _first_text(record, text_fields)
    raw_label = next((record.get(field) for field in label_fields if record.get(field) is not None), None)
    sentiment_label = _normalize_label(raw_label, label_map)
    if not text or sentiment_label is None:
        return {}
    row = {
        "sample_family": "classification",
        "query": text,
        "text": text,
        "product_type": "",
        "intent_labels": [],
        "topic_labels": [],
        "question_style": "",
        "sentiment_label": sentiment_label,
        "split_lock_key": _stable_key(source_id, text, sentiment_label),
        "source_id": source_id,
        "source_row_id": _first_present(record, ("id", "uid", "doc_id", "sample_id")),
        "raw_label": raw_label,
        "available_labels": ["sentiment_label"],
    }
    doc_sources, structured_sources = _source_labels_from_text(text, _product_type_from_text(text), raw_label)
    return _attach_source_supervision(row, expected_document_sources=doc_sources, expected_structured_sources=structured_sources)


def adapt_finfe_rows(raw: Any, source_id: str = "finfe") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        row = _build_sentiment_row(
            record,
            source_id=source_id,
            label_map=FINFE_SENTIMENT_MAP,
            text_fields=("text", "sentence", "content", "review", "title"),
            label_fields=("label", "sentiment_label", "sentiment", "polarity", "rating"),
        )
        if row:
            rows.append(row)
    return rows


def adapt_chnsenticorp_rows(raw: Any, source_id: str = "chnsenticorp") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        row = _build_sentiment_row(
            record,
            source_id=source_id,
            label_map=CHNSENTICORP_SENTIMENT_MAP,
            text_fields=("text", "sentence", "content", "review", "title"),
            label_fields=("label", "sentiment_label", "sentiment", "polarity", "rating"),
        )
        if row:
            rows.append(row)
    return rows


def _alias_rows_for_financial_news(record: Mapping[str, Any], *, source_id: str, source_row_id: Any | None) -> list[dict[str, Any]]:
    title = _first_text(record, ("title", "headline", "news_title", "标题"))
    content = _first_text(record, ("content", "article", "text", "body", "sentence", "正文"))
    combined = " ".join(part for part in (title, content) if part).strip()
    candidates: list[tuple[str, str, str | None]] = []
    seen: set[tuple[str, str]] = set()

    def add_candidate(alias_text: str, alias_type: str, symbol: str | None = None, company_name: str | None = None) -> None:
        normalized = str(alias_text).strip()
        if not normalized:
            return
        key = (normalized, alias_type)
        if key in seen:
            return
        seen.add(key)
        candidates.append((normalized, alias_type, symbol or company_name))

    for field in ("company_name", "company", "name", "stock_name", "证券简称", "简称", "alias", "alias_text"):
        value = _first_text(record, (field,))
        if value:
            add_candidate(value, "company_name", company_name=value)

    for field in ("symbol", "code", "ticker", "stock_code", "security_code", "证券代码"):
        value = _first_text(record, (field,))
        if not value:
            continue
        code = value.upper().replace(" ", "")
        if code.isdigit() and len(code) == 6:
            add_candidate(code, "company_code", symbol=code)
        elif _ALIAS_CODE_RE.fullmatch(code):
            add_candidate(code, "company_code", symbol=code)

    for match in _ALIAS_CODE_RE.finditer(combined):
        code = f"{match.group('code')}.{match.group('exchange').upper()}"
        add_candidate(code, "company_code", symbol=code)

    title_match = _TITLE_ALIAS_SPLIT_RE.search(title)
    if title_match:
        leading = title_match.group("name").strip()
        if leading and leading not in _GENERIC_TITLE_STOPWORDS:
            add_candidate(leading, "company_name", company_name=leading)

    rows: list[dict[str, Any]] = []
    raw_label = next((record.get(field) for field in ("label", "sentiment_label", "sentiment", "polarity", "正负面") if record.get(field) is not None), None)
    for alias_text, alias_type, entity_hint in candidates:
        row = {
            "sample_family": "alias",
            "alias_text": alias_text,
            "normalized_alias": alias_text,
            "alias_type": alias_type,
            "company_name": alias_text if alias_type == "company_name" else (entity_hint or ""),
            "symbol": alias_text if alias_type == "company_code" else "",
            "source_id": source_id,
            "source_row_id": source_row_id,
            "raw_label": raw_label,
            "split_lock_key": _stable_key(source_id, source_row_id or combined, alias_text, alias_type),
        }
        rows.append(row)
    return rows


def adapt_financial_news_rows(raw: Any, source_id: str = "fin_news_sentiment") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        title = _first_text(record, ("title", "headline", "news_title", "标题"))
        content = _first_text(record, ("content", "article", "text", "body", "sentence", "正文"))
        text = "\n".join(part for part in (title, content) if part)
        raw_label = next((record.get(field) for field in ("label", "sentiment_label", "sentiment", "polarity", "正负面") if record.get(field) is not None), None)
        if isinstance(raw_label, str):
            lower = raw_label.strip().lower()
            if lower in {"positive", "negative"}:
                sentiment_label = lower
            elif lower in {"1", "0"}:
                sentiment_label = "positive" if lower == "1" else "negative"
            else:
                sentiment_label = lower or None
        elif raw_label is not None:
            sentiment_label = "positive" if int(raw_label) > 0 else "negative"
        else:
            sentiment_label = None
        if not text or sentiment_label is None:
            continue

        row = {
            "sample_family": "classification",
            "query": text,
            "text": text,
            "product_type": "",
            "intent_labels": [],
            "topic_labels": [],
            "question_style": "",
            "sentiment_label": sentiment_label,
            "split_lock_key": _stable_key(source_id, text, sentiment_label),
            "source_id": source_id,
            "source_row_id": _first_present(record, ("id", "uid", "news_id")),
            "raw_label": raw_label,
            "available_labels": ["sentiment_label"],
        }
        doc_sources, structured_sources = _source_labels_from_text(text, _product_type_from_text(text), raw_label)
        rows.append(_attach_source_supervision(row, expected_document_sources=doc_sources, expected_structured_sources=structured_sources))
        rows.extend(_alias_rows_for_financial_news(record, source_id=source_id, source_row_id=row["source_row_id"]))
    return rows
