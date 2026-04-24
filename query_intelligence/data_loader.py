from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNTIME_DATA_DIR = DATA_DIR / "runtime"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_document_file(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Document file must contain a list: {path}")
    return data


@lru_cache(maxsize=1)
def load_entities() -> list[dict[str, str]]:
    rows = _read_csv(_resolve_data_path("QI_ENTITY_MASTER_PATH", "entity_master.csv"))
    for row in rows:
        if row.get("entity_id"):
            row["entity_id"] = str(int(row["entity_id"]))
    return rows


@lru_cache(maxsize=1)
def load_aliases() -> list[dict[str, str]]:
    return _read_csv(_resolve_data_path("QI_ALIAS_TABLE_PATH", "alias_table.csv"))


@lru_cache(maxsize=1)
def load_seed_entities() -> list[dict[str, str]]:
    rows = _read_csv(DATA_DIR / "entity_master.csv")
    for row in rows:
        if row.get("entity_id"):
            row["entity_id"] = str(int(row["entity_id"]))
    return rows


@lru_cache(maxsize=1)
def load_seed_aliases() -> list[dict[str, str]]:
    return _read_csv(DATA_DIR / "alias_table.csv")


@lru_cache(maxsize=1)
def load_synonyms() -> dict:
    return _read_json(DATA_DIR / "synonym_dict.json")


@lru_cache(maxsize=1)
def load_documents() -> list[dict]:
    override = os.getenv("QI_DOCUMENTS_PATH")
    if override:
        return _read_document_file(Path(override))

    documents_path = _resolve_documents_path()
    documents = _read_document_file(documents_path)
    seed_path = DATA_DIR / "documents.json"
    if documents_path == seed_path or not seed_path.exists():
        return documents

    # Runtime corpora are large but may lack small curated fallbacks such as
    # seed news/FAQ docs. Keep seed docs as a safety net without duplicating ids.
    seed_documents = _read_document_file(seed_path)
    seen_ids = {str(doc.get("evidence_id", "")) for doc in documents}
    merged = list(documents)
    for doc in seed_documents:
        evidence_id = str(doc.get("evidence_id", ""))
        if evidence_id and evidence_id in seen_ids:
            continue
        merged.append(doc)
        if evidence_id:
            seen_ids.add(evidence_id)
    return merged


@lru_cache(maxsize=1)
def load_structured_data() -> dict:
    return _read_json(DATA_DIR / "structured_data.json")


def _resolve_data_path(env_var: str, filename: str) -> Path:
    override = os.getenv(env_var)
    if override:
        return Path(override)
    runtime_path = DATA_DIR / "runtime" / filename
    if runtime_path.exists():
        return runtime_path
    return DATA_DIR / filename


def _resolve_documents_path() -> Path:
    override = os.getenv("QI_DOCUMENTS_PATH")
    if override:
        return Path(override)
    for path in (
        DATA_DIR / "runtime" / "documents.jsonl",
        DATA_DIR / "runtime" / "documents.json",
        DATA_DIR / "documents.json",
    ):
        if path.exists():
            return path
    return DATA_DIR / "documents.json"


def clear_data_caches() -> None:
    load_entities.cache_clear()
    load_aliases.cache_clear()
    load_seed_entities.cache_clear()
    load_seed_aliases.cache_clear()
    load_synonyms.cache_clear()
    load_documents.cache_clear()
    load_structured_data.cache_clear()
