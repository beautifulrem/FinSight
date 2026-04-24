from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _load_records(raw: Any) -> list[Mapping[str, Any]]:
    if isinstance(raw, Path):
        return _load_records_from_path(raw)
    if isinstance(raw, str):
        path = Path(raw)
        if path.exists():
            return _load_records_from_path(path)
        return [{"query": raw}]
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
    return []


def _stable_key(*parts: Any) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


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


def _as_relevance(value: Any, *, collapse_three_level: bool = False) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        relevance = int(value)
    else:
        text = str(value).strip()
        if not text or not text.lstrip("-").isdigit():
            return None
        relevance = int(text)
    if collapse_three_level and relevance >= 3:
        return 2
    return relevance


def _build_row(record: Mapping[str, Any], *, source_id: str, collapse_three_level: bool) -> dict[str, Any]:
    query_id = record.get("query_id") or record.get("qid") or record.get("question_id") or record.get("q_id")
    doc_id = record.get("doc_id") or record.get("did") or record.get("pid") or record.get("passage_id") or record.get("docid")
    query = _first_text(record, ("query", "question", "text", "title", "query_text"))
    doc_text = _first_text(record, ("doc_text", "doc", "passage", "content", "body", "document"))
    relevance = _as_relevance(_first_present(record, ("relevance", "label", "score")), collapse_three_level=collapse_three_level)
    if relevance is None:
        return {}
    return {
        "sample_family": "qrel",
        "query_id": query_id,
        "query": query,
        "doc_id": doc_id,
        "doc_text": doc_text or None,
        "relevance": relevance,
        "split_lock_key": _stable_key(source_id, query_id, doc_id, relevance),
        "source_id": source_id,
        "source_row_id": record.get("id") or record.get("uid") or record.get("sample_id"),
    }


def _build_corpus_row(
    *,
    source_id: str,
    doc_id: str,
    text: str,
    title: str = "",
    source_type: str = "research_note",
) -> dict[str, Any]:
    return {
        "sample_family": "retrieval_corpus",
        "doc_id": doc_id,
        "title": title,
        "body": text,
        "text": text,
        "source_type": source_type,
        "split_lock_key": _stable_key(source_id, doc_id),
        "source_id": source_id,
        "source_row_id": doc_id,
    }


def _split_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _build_csprd_qrels(record: Mapping[str, Any], *, source_id: str) -> list[dict[str, Any]]:
    query_id = str(record.get("id") or record.get("query_id") or "").strip()
    query = _first_text(record, ("passage", "query", "question", "text", "title"))
    if not query_id or not query:
        return []

    rows: list[dict[str, Any]] = []
    for doc_id in _split_id_list(record.get("pos_ctxs")):
        rows.append(
            {
                "sample_family": "qrel",
                "query_id": query_id,
                "query": query,
                "doc_id": doc_id,
                "relevance": 2,
                "split_lock_key": _stable_key(source_id, query_id),
                "source_id": source_id,
                "source_row_id": query_id,
            }
        )
    for doc_id in _split_id_list(record.get("neg_ctxs")):
        rows.append(
            {
                "sample_family": "qrel",
                "query_id": query_id,
                "query": query,
                "doc_id": doc_id,
                "relevance": 0,
                "split_lock_key": _stable_key(source_id, query_id),
                "source_id": source_id,
                "source_row_id": query_id,
            }
        )
    return rows


def _build_csprd_corpus_row(record: Mapping[str, Any], *, source_id: str) -> dict[str, Any]:
    doc_id = str(record.get("id") or record.get("doc_id") or "").strip()
    text = _first_text(record, ("content", "text", "passage", "body"))
    title = _first_text(record, ("title", "name"))
    if not doc_id or not text:
        return {}
    return {
        "sample_family": "retrieval_corpus",
        "doc_id": doc_id,
        "title": title,
        "body": text,
        "text": text,
        "split_lock_key": _stable_key(source_id, doc_id),
        "source_id": source_id,
        "source_row_id": doc_id,
    }


def adapt_t2ranking_rows(raw: Any, source_id: str = "t2ranking") -> list[dict[str, Any]]:
    if isinstance(raw, Path) and raw.is_dir():
        return _adapt_t2ranking_directory(raw, source_id=source_id)
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        row = _build_row(record, source_id=source_id, collapse_three_level=True)
        if row:
            rows.append(row)
    return rows


def adapt_csprd_rows(raw: Any, source_id: str = "csprd") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        if record.get("pos_ctxs") is not None or record.get("neg_ctxs") is not None:
            rows.extend(_build_csprd_qrels(record, source_id=source_id))
            continue
        row = _build_csprd_corpus_row(record, source_id=source_id)
        if row:
            rows.append(row)
            continue
        row = _build_row(record, source_id=source_id, collapse_three_level=False)
        if row:
            rows.append(row)
    return rows


def adapt_fiqa_rows(raw: Any, source_id: str = "fiqa") -> list[dict[str, Any]]:
    if isinstance(raw, Path) and raw.is_dir():
        return _adapt_fiqa_directory(raw, source_id=source_id)
    return []


def adapt_fincprg_rows(raw: Any, source_id: str = "fincprg") -> list[dict[str, Any]]:
    if isinstance(raw, Path) and raw.is_dir():
        return _adapt_fincprg_directory(raw, source_id=source_id)
    return []


def adapt_fir_bench_report_rows(raw: Any, source_id: str = "fir_bench_reports") -> list[dict[str, Any]]:
    return _adapt_fir_bench_rows(raw, source_id=source_id, source_type="research_note")


def adapt_fir_bench_announcement_rows(raw: Any, source_id: str = "fir_bench_announcements") -> list[dict[str, Any]]:
    return _adapt_fir_bench_rows(raw, source_id=source_id, source_type="announcement")


def _adapt_fiqa_directory(raw_dir: Path, *, source_id: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    queries: dict[str, str] = {}
    for path in sorted(raw_dir.rglob("queries*.parquet")):
        frame = pd.read_parquet(path)
        for record in frame.to_dict(orient="records"):
            query_id = str(record.get("_id") or record.get("query_id") or "").strip()
            query = _first_text(record, ("text", "query", "title"))
            if query_id and query:
                queries[query_id] = query

    for path in sorted(raw_dir.rglob("corpus*.parquet")):
        frame = pd.read_parquet(path)
        for record in frame.to_dict(orient="records"):
            doc_id = str(record.get("_id") or record.get("doc_id") or "").strip()
            text = _first_text(record, ("text", "body", "content"))
            title = _first_text(record, ("title",))
            if doc_id and text:
                rows.append(
                    {
                        "sample_family": "retrieval_corpus",
                        "doc_id": doc_id,
                        "title": title,
                        "body": text,
                        "text": text,
                        "split_lock_key": _stable_key(source_id, doc_id),
                        "source_id": source_id,
                        "source_row_id": doc_id,
                    }
                )

    for path in sorted(raw_dir.rglob("*.tsv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for record in reader:
                query_id = str(record.get("query-id") or record.get("query_id") or record.get("qid") or "").strip()
                doc_id = str(record.get("corpus-id") or record.get("doc_id") or record.get("pid") or "").strip()
                query = queries.get(query_id, "")
                if not query_id or not doc_id or not query:
                    continue
                rows.append(
                    {
                        "sample_family": "qrel",
                        "query_id": query_id,
                        "query": query,
                        "doc_id": doc_id,
                        "relevance": _as_relevance(record.get("score"), collapse_three_level=False) or 0,
                        "split_lock_key": _stable_key(source_id, query_id),
                        "source_id": source_id,
                        "source_row_id": f"{query_id}:{doc_id}",
                    }
                )
    return rows


def _adapt_fincprg_directory(raw_dir: Path, *, source_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    corpus_path = raw_dir / "corpus.jsonl"
    query_path = raw_dir / "queries.jsonl"
    qrels_path = raw_dir / "qrels" / "all.tsv"
    if not corpus_path.exists() or not query_path.exists() or not qrels_path.exists():
        return rows

    queries: dict[str, str] = {}
    for record in _load_records(corpus_path):
        doc_id = str(record.get("_id") or record.get("doc_id") or "").strip()
        text = _first_text(record, ("text", "body", "content", "passage"))
        if not doc_id or not text:
            continue
        rows.append(_build_corpus_row(source_id=source_id, doc_id=doc_id, text=text, source_type="research_note"))

    for record in _load_records(query_path):
        query_id = str(record.get("_id") or record.get("query_id") or "").strip()
        query = _first_text(record, ("text", "query"))
        if query_id and query:
            queries[query_id] = query

    with qrels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for record in reader:
            query_id = str(record.get("query-id") or record.get("query_id") or record.get("queryid") or "").strip()
            doc_id = str(record.get("corpus-id") or record.get("doc_id") or record.get("docid") or "").strip()
            query = queries.get(query_id, "")
            if not query_id or not doc_id or not query:
                continue
            rows.append(
                {
                    "sample_family": "qrel",
                    "query_id": query_id,
                    "query": query,
                    "doc_id": doc_id,
                    "relevance": _as_relevance(record.get("score"), collapse_three_level=False) or 0,
                    "split_lock_key": _stable_key(source_id, query_id),
                    "source_id": source_id,
                    "source_row_id": f"{query_id}:{doc_id}",
                }
            )
    return rows


def _adapt_t2ranking_directory(raw_dir: Path, *, source_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data_dir = raw_dir / "data"
    if not data_dir.exists():
        return rows

    queries: dict[str, str] = {}
    for path in sorted(data_dir.glob("queries.*.tsv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for record in reader:
                query_id = str(record.get("qid") or "").strip()
                query = _first_text(record, ("text", "query"))
                if query_id and query:
                    queries[query_id] = query

    referenced_doc_ids: set[str] = set()
    for path in sorted(data_dir.glob("qrels.retrieval*.tsv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for record in reader:
                query_id = str(record.get("qid") or "").strip()
                doc_id = str(record.get("pid") or "").strip()
                query = queries.get(query_id, "")
                if not query_id or not doc_id or not query:
                    continue
                referenced_doc_ids.add(doc_id)
                rows.append(
                    {
                        "sample_family": "qrel",
                        "query_id": query_id,
                        "query": query,
                        "doc_id": doc_id,
                        "relevance": 1,
                        "split_lock_key": _stable_key(source_id, query_id),
                        "source_id": source_id,
                        "source_row_id": f"{query_id}:{doc_id}",
                    }
                )

    collection_path = data_dir / "collection.tsv"
    if collection_path.exists() and referenced_doc_ids:
        with collection_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for record in reader:
                if len(record) < 2:
                    continue
                doc_id, text = record[0].strip(), record[1].strip()
                if doc_id not in referenced_doc_ids or not text:
                    continue
                rows.append(
                    {
                        "sample_family": "retrieval_corpus",
                        "doc_id": doc_id,
                        "title": "",
                        "body": text,
                        "text": text,
                        "split_lock_key": _stable_key(source_id, doc_id),
                        "source_id": source_id,
                        "source_row_id": doc_id,
                    }
                )
    return rows


def _adapt_fir_bench_rows(raw: Any, *, source_id: str, source_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        query = _first_text(record, ("query", "question"))
        passage = _first_text(record, ("passage", "text", "body", "content"))
        title = _first_text(record, ("title",))
        relevance = _as_relevance(_first_present(record, ("label", "score")), collapse_three_level=False)
        if not query or not passage or relevance is None:
            continue
        doc_id = hashlib.sha1(f"{source_id}|{title}|{passage}".encode("utf-8")).hexdigest()
        rows.append(_build_corpus_row(source_id=source_id, doc_id=doc_id, text=passage, title=title, source_type=source_type))
        rows.append(
            {
                "sample_family": "qrel",
                "query_id": hashlib.sha1(f"{source_id}|{query}".encode("utf-8")).hexdigest(),
                "query": query,
                "doc_id": doc_id,
                "doc_text": passage,
                "relevance": relevance,
                "split_lock_key": _stable_key(source_id, query, doc_id),
                "source_id": source_id,
                "source_row_id": f"{source_id}:{doc_id}",
            }
        )
    return rows
