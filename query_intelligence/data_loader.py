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


SNAPSHOT_EXT_PATH = DATA_DIR / "snapshot" / "structured_data_ext.json"
# Sections of the extension merged under the v1 snapshot.
_SNAPSHOT_EXT_SECTIONS = ("market_api", "fundamental_sql", "industry_sql", "macro_sql", "entity_to_industry")


def snapshot_ext_enabled() -> bool:
    """``QI_OFFLINE_SNAPSHOT_EXT`` (default on): merge ``data/snapshot/structured_data_ext.json`` under v1."""
    return os.getenv("QI_OFFLINE_SNAPSHOT_EXT", "true").strip().lower() in {"1", "true", "yes"}


SNAPSHOT_FIX_PATH = DATA_DIR / "snapshot" / "structured_data_fix.json"
# Sections of the corrections layer whose records replace the v1 record of the same key.
_SNAPSHOT_FIX_SECTIONS = ("market_api", "fundamental_sql")


def snapshot_fix_enabled() -> bool:
    """``QI_OFFLINE_SNAPSHOT_FIX`` (default on): apply ``data/snapshot/structured_data_fix.json`` over v1."""
    return os.getenv("QI_OFFLINE_SNAPSHOT_FIX", "true").strip().lower() in {"1", "true", "yes"}


def load_structured_data(extended: bool | None = None, corrected: bool | None = None) -> dict:
    """The offline structured snapshot: v1 (``data/structured_data.json``), plus the extension when ``extended``,
    with the corrections layer applied when ``corrected``.

    ``extended=None`` follows ``QI_OFFLINE_SNAPSHOT_EXT``; ``corrected=None`` follows ``QI_OFFLINE_SNAPSHOT_FIX``.
    The evaluation harness pins both off: its labels were written against v1 (for example "宁德时代 has no
    offline price", or 五粮液's v1 seed revenue), see docs/data-sources.md.
    """
    return _load_structured_data(
        snapshot_ext_enabled() if extended is None else bool(extended),
        snapshot_fix_enabled() if corrected is None else bool(corrected),
    )


@lru_cache(maxsize=4)
def _load_structured_data(extended: bool, corrected: bool = False) -> dict:
    data = _read_json(DATA_DIR / "structured_data.json")
    if extended and SNAPSHOT_EXT_PATH.exists():
        data = merge_snapshot_extension(data, _read_json(SNAPSHOT_EXT_PATH))
    if corrected and SNAPSHOT_FIX_PATH.exists():
        data = apply_snapshot_corrections(data, _read_json(SNAPSHOT_FIX_PATH))
    return data


def apply_snapshot_corrections(base: dict, corrections: dict) -> dict:
    """``base`` with the corrections layer applied (precedence: corrections > v1 seed > extension).

    * ``market_api`` / ``fundamental_sql``: a corrected record replaces the record of the same key (the v1 seed
      values that a primary source contradicts: an annual report, an exchange or index-compiler daily record).
    * ``definitions``: field definitions (``roe_basis``, ``revenue_basis``, ``net_profit_basis``, ``eps_basis``,
      ...) added as ``definitions`` to the fundamentals record that serves the symbol, when there is one and it
      does not carry its own (the extension's records: their values are kept, only their definitions are stated).
    """
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for section in _SNAPSHOT_FIX_SECTIONS:
        target = merged.setdefault(section, {})
        for key, record in (corrections.get(section) or {}).items():
            target[key] = record
    statements = merged.setdefault("fundamental_sql", {})
    for symbol, definitions in (corrections.get("definitions") or {}).items():
        record = statements.get(symbol)
        if isinstance(record, dict) and "definitions" not in record:
            statements[symbol] = {**record, "definitions": dict(definitions)}
    return merged


def merge_snapshot_extension(base: dict, extension: dict) -> dict:
    """``base`` with the extension's records added. A key already in ``base`` always keeps the base record, so
    the extension can add instruments but never change a value v1 already serves."""
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for section in _SNAPSHOT_EXT_SECTIONS:
        target = merged.setdefault(section, {})
        for key, record in (extension.get(section) or {}).items():
            target.setdefault(key, record)
    return merged


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
    _load_structured_data.cache_clear()
