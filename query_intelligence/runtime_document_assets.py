from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RUNTIME_SOURCE_TYPES = {"announcement", "research_note", "news", "faq", "product_doc"}
DEFAULT_MAX_DOCUMENTS = 50_000

_SOURCE_ALIASES = {
    "announcement": "announcement",
    "announcements": "announcement",
    "filing": "announcement",
    "research": "research_note",
    "research_report": "research_note",
    "research_note": "research_note",
    "report": "research_note",
    "news": "news",
    "article": "news",
    "faq": "faq",
    "qa": "faq",
    "q&a": "faq",
    "product_doc": "product_doc",
    "product": "product_doc",
    "doc": "product_doc",
}

_TYPE_WEIGHTS = {
    "announcement": 0.40,
    "research_note": 0.20,
    "product_doc": 0.15,
    "faq": 0.15,
    "news": 0.10,
}

_SYMBOL_RE = re.compile(r"\b(?P<code>\d{6})(?:\.(?P<exchange>SH|SZ|BJ)|\s*CH)?\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RuntimeDocumentSummary:
    document_count: int
    skipped_count: int
    duplicate_count: int
    source_type_counts: dict[str, int]
    output_path: str


class RuntimeDocumentAssetBuilder:
    """Materialize retrieval corpus rows into DocumentRetriever runtime documents."""

    def __init__(self, corpus_path: str | Path, max_documents: int = DEFAULT_MAX_DOCUMENTS) -> None:
        self.corpus_path = Path(corpus_path)
        self.max_documents = max_documents

    def build(self) -> tuple[list[dict], RuntimeDocumentSummary]:
        documents: list[dict] = []
        seen_content: set[str] = set()
        seen_evidence_ids: Counter[str] = Counter()
        source_type_counts: Counter[str] = Counter()
        quotas = _source_type_quotas(self.max_documents)
        skipped_count = 0
        duplicate_count = 0

        for row in _iter_jsonl(self.corpus_path):
            source_type = _normalize_source_type(row)
            if source_type is None:
                skipped_count += 1
                continue
            if source_type_counts[source_type] >= quotas[source_type]:
                skipped_count += 1
                continue
            document = _materialize_row(row, source_type)
            if document is None:
                skipped_count += 1
                continue
            content_key = _content_key(document)
            if content_key in seen_content:
                duplicate_count += 1
                continue
            seen_content.add(content_key)
            evidence_id = document["evidence_id"]
            seen_evidence_ids[evidence_id] += 1
            if seen_evidence_ids[evidence_id] > 1:
                document["evidence_id"] = f"{evidence_id}_{seen_evidence_ids[evidence_id]}"
            documents.append(document)
            source_type_counts[source_type] += 1
            if len(documents) >= self.max_documents:
                break

        summary = RuntimeDocumentSummary(
            document_count=len(documents),
            skipped_count=skipped_count,
            duplicate_count=duplicate_count,
            source_type_counts=dict(source_type_counts),
            output_path="",
        )
        return documents, summary

    def write(self, output_path: str | Path) -> RuntimeDocumentSummary:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        documents, summary = self.build()
        with path.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
        return RuntimeDocumentSummary(
            document_count=summary.document_count,
            skipped_count=summary.skipped_count,
            duplicate_count=summary.duplicate_count,
            source_type_counts=summary.source_type_counts,
            output_path=str(path),
        )


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _normalize_source_type(row: dict) -> str | None:
    raw_type = str(row.get("source_type") or "").strip().lower()
    if raw_type in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[raw_type]
    return _infer_unknown_source_type(row)


def _infer_unknown_source_type(row: dict) -> str | None:
    text = f"{row.get('title') or ''}\n{row.get('body') or row.get('text') or ''}".lower()
    if any(token in text for token in ("faq", "q&a", "常见问题", "如何", "怎么", "开户", "开通")):
        return "faq"
    if any(token in text for token in ("产品说明", "交易规则", "费率", "规则", "机制", "指南", "手册", "product", "guide", "rule")):
        return "product_doc"
    if any(token in text for token in ("证券研究报告", "研究报告", "研报", "评级", "目标价", "research", "analyst", "rating")):
        return "research_note"
    return None


def _materialize_row(row: dict, source_type: str) -> dict | None:
    title = _clean_text(row.get("title"))
    body = _clean_text(row.get("body") or row.get("text"))
    if not title and not body:
        return None
    summary = _clean_text(row.get("summary")) or _summarize(title, body)
    source_name = _clean_text(row.get("source_name") or row.get("source_id")) or "training_retrieval_corpus"
    provider = _clean_text(row.get("provider") or row.get("source_id")) or "training_assets"
    raw_id = _clean_text(row.get("doc_id") or row.get("source_row_id")) or _short_hash(f"{title}\n{body}")
    return {
        "evidence_id": _safe_evidence_id(f"{source_type}_{raw_id}"),
        "doc_id": raw_id,
        "source_type": source_type,
        "source_name": source_name,
        "provider": provider,
        "source_url": _clean_text(row.get("source_url") or row.get("url")),
        "title": title or summary,
        "summary": summary,
        "body": body,
        "publish_time": _clean_text(row.get("publish_time") or row.get("published_at") or row.get("date")),
        "entity_symbols": _extract_symbols(row, f"{title}\n{body}"),
        "product_type": _clean_text(row.get("product_type")) or _infer_product_type(f"{title}\n{body}"),
        "credibility_score": _credibility_score(source_type),
    }


def _clean_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "").strip())


def _summarize(title: str, body: str, limit: int = 160) -> str:
    source = title or body
    return source[:limit]


def _extract_symbols(row: dict, text: str) -> list[str]:
    symbols = row.get("entity_symbols")
    if isinstance(symbols, list):
        normalized = [_normalize_symbol(str(symbol)) for symbol in symbols]
        return sorted({symbol for symbol in normalized if symbol})
    found = {_normalize_symbol(match.group(0)) for match in _SYMBOL_RE.finditer(text)}
    return sorted(symbol for symbol in found if symbol)


def _normalize_symbol(value: str) -> str:
    text = value.strip().upper().replace(" ", "")
    match = _SYMBOL_RE.search(text)
    if not match:
        return ""
    code = match.group("code")
    exchange = match.group("exchange")
    if not exchange:
        if code.startswith(("6", "9")):
            exchange = "SH"
        elif code.startswith(("0", "2", "3")):
            exchange = "SZ"
        elif code.startswith(("4", "8")):
            exchange = "BJ"
    return f"{code}.{exchange}" if exchange else code


def _infer_product_type(text: str) -> str:
    lowered = text.lower()
    if "etf" in lowered:
        return "etf"
    if "基金" in text or "fund" in lowered:
        return "fund"
    if "指数" in text or "index" in lowered:
        return "index"
    return "stock"


def _credibility_score(source_type: str) -> float:
    return {
        "announcement": 0.97,
        "research_note": 0.86,
        "news": 0.78,
        "faq": 0.82,
        "product_doc": 0.88,
    }[source_type]


def _content_key(document: dict) -> str:
    return _short_hash(
        "\n".join(
            [
                document["source_type"],
                document["title"].lower(),
                document["body"].lower(),
            ]
        )
    )


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _safe_evidence_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value).strip("_")[:160]


def _source_type_quotas(max_documents: int) -> dict[str, int]:
    quotas = {source_type: max(1, int(max_documents * weight)) for source_type, weight in _TYPE_WEIGHTS.items()}
    remainder = max_documents - sum(quotas.values())
    for source_type in ("announcement", "research_note", "product_doc", "faq", "news"):
        if remainder <= 0:
            break
        quotas[source_type] += 1
        remainder -= 1
    return quotas
