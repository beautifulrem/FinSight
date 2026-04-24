from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..label_maps import map_ner_tag_to_ent


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
    if suffix in {".txt", ".train", ".dev", ".test"}:
        return _load_bio_txt_records(path)
    return []


def _load_bio_txt_records(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    tokens: list[str] = []
    tags: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                if tokens:
                    records.append({"tokens": list(tokens), "tags": list(tags)})
                    tokens = []
                    tags = []
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split()
            token = parts[0]
            tag = parts[1] if len(parts) > 1 else "O"
            tokens.append(token)
            tags.append(tag)
    if tokens:
        records.append({"tokens": list(tokens), "tags": list(tags)})
    return records


def _stable_key(*parts: Any) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _as_tokens(value: Any) -> list[str]:
    if isinstance(value, str):
        return [token for token in value.split() if token]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value]
    return []


def _as_tags(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [map_ner_tag_to_ent(str(item)) for item in value]
    return []


def _labels_to_tags(tags: list[str], labels: Mapping[str, Any], text_len: int) -> list[str]:
    for ent_type, spans in labels.items():
        if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes, bytearray)):
            continue
        ent_type_text = str(ent_type).lower()
        keep_entity = any(part in ent_type_text for part in ("org", "company", "organization", "agency", "institution"))
        if not keep_entity:
            continue
        for span in spans:
            start: int | None = None
            end: int | None = None
            if isinstance(span, Mapping):
                start = span.get("start") if isinstance(span.get("start"), int) else span.get("begin") if isinstance(span.get("begin"), int) else span.get("offset")
                end = span.get("end") if isinstance(span.get("end"), int) else span.get("stop") if isinstance(span.get("stop"), int) else span.get("offset_end")
                if isinstance(start, str) and start.isdigit():
                    start = int(start)
                if isinstance(end, str) and end.isdigit():
                    end = int(end)
            elif isinstance(span, Sequence) and len(span) >= 2:
                start = int(span[0])
                end = int(span[1])
            if start is None or end is None:
                continue
            start = max(0, min(int(start), text_len))
            end = max(start, min(int(end), text_len))
            if start >= end:
                continue
            tags[start] = "B-ENT"
            for index in range(start + 1, end):
                tags[index] = "I-ENT"
    return tags


def _build_row(record: Mapping[str, Any], *, source_id: str) -> dict[str, Any]:
    text = str(record.get("text") or record.get("sentence") or record.get("content") or "").strip()
    tokens = _as_tokens(record.get("tokens") or record.get("words") or record.get("chars"))
    tags = _as_tags(record.get("tags") or record.get("ner_tags") or record.get("labels"))

    if not tags and isinstance(record.get("label"), Mapping):
        # CLUENER-style span annotations: keep only organization/company spans.
        chars = list(text)
        tags = ["O"] * len(chars)
        tags = _labels_to_tags(tags, record["label"], len(chars))
        tokens = chars
    elif tokens and not tags and isinstance(record.get("label"), Sequence):
        tags = _as_tags(record.get("label"))
    elif text and not tokens:
        tokens = list(text)
        if len(tags) != len(tokens):
            tags = ["O"] * len(tokens)

    if not text:
        text = "".join(tokens)
    if not tokens:
        tokens = list(text)
    if not tags:
        tags = ["O"] * len(tokens)
    if len(tags) != len(tokens):
        # Best-effort alignment: truncate to shared length rather than fail.
        size = min(len(tags), len(tokens))
        tokens = tokens[:size]
        tags = tags[:size]

    if not text:
        return {}

    return {
        "sample_family": "entity",
        "text": text,
        "tokens": tokens,
        "tags": tags,
        "split_lock_key": _stable_key(source_id, text, " ".join(tokens), " ".join(tags)),
        "source_id": source_id,
        "source_row_id": record.get("id") or record.get("uid") or record.get("sample_id"),
    }


def adapt_msra_rows(raw: Any, source_id: str = "msra_ner") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        row = _build_row(record, source_id=source_id)
        if row:
            rows.append(row)
    return rows


def adapt_peoples_daily_rows(raw: Any, source_id: str = "peoples_daily_ner") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        row = _build_row(record, source_id=source_id)
        if row:
            rows.append(row)
    return rows


def adapt_cluener_rows(raw: Any, source_id: str = "cluener") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in _load_records(raw):
        row = _build_row(record, source_id=source_id)
        if row:
            rows.append(row)
    return rows
