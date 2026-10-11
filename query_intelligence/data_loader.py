from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
def load_dual_listings() -> list[dict[str, str]]:
    """(round 16, review-12 L7) A-share companies with a Hong Kong (H-share) line: ``a_symbol``, ``h_code`` (HKEX,
    five digits), ``name_zh`` (the entity master's name), ``short_zh`` / ``short_en`` (``|``-separated), ``name_en``,
    ``hkex_name`` and ``source`` (the HKEX list of securities that confirms the code). ``QI_DUAL_LISTINGS_PATH``
    overrides the file. The agent's foreign-line check (``agent/coverage.py``) joins it with the alias table."""
    return _read_csv(_resolve_data_path("QI_DUAL_LISTINGS_PATH", "dual_listings.csv"))


@lru_cache(maxsize=1)
def load_hk_only_names() -> list[dict[str, str]]:
    """(round 17, review-13 M3) Hong Kong-listed equities with no A-share line: ``h_code``, ``name_zh`` (simplified)
    and ``hkex_name`` (as HKEX lists it), from the HKEX list of securities (``scripts/build_hk_only_names.py``; the
    first line of the file records the list's date and sha256). ``QI_HK_ONLY_NAMES_PATH`` overrides the file; a
    missing file gives an empty list."""
    path = _resolve_data_path("QI_HK_ONLY_NAMES_PATH", "hk_only_names.csv")
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


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
    if extended:
        if SNAPSHOT_EXT_PATH.exists():
            data = merge_snapshot_extension(data, _read_json(SNAPSHOT_EXT_PATH))
        else:
            _warn_missing_layer("extension", "QI_OFFLINE_SNAPSHOT_EXT", SNAPSHOT_EXT_PATH)
    if corrected:
        if SNAPSHOT_FIX_PATH.exists():
            data = apply_snapshot_corrections(data, _read_json(SNAPSHOT_FIX_PATH))
        else:
            _warn_missing_layer("corrections", "QI_OFFLINE_SNAPSHOT_FIX", SNAPSHOT_FIX_PATH)
    return data


def _warn_missing_layer(name: str, env_var: str, path: Path) -> None:
    # (round-10 review J2) a layer that is switched on but absent (an image built without it) is never skipped
    # silently: the log says so once per configuration, and GET /ready reports it as not loaded (503).
    logger.warning(
        "offline snapshot layer %r is switched on (%s) but %s is missing; serving without it", name, env_var, path
    )


def _snapshot_layer_specs() -> tuple[tuple[str, Path, str | None, tuple[str, ...]], ...]:
    """name, file, the env var that switches it (None: always on), the sections whose records it serves."""
    return (
        ("seed_v1", DATA_DIR / "structured_data.json", None, _SNAPSHOT_EXT_SECTIONS),
        ("extension", SNAPSHOT_EXT_PATH, "QI_OFFLINE_SNAPSHOT_EXT", _SNAPSHOT_EXT_SECTIONS),
        ("corrections", SNAPSHOT_FIX_PATH, "QI_OFFLINE_SNAPSHOT_FIX", (*_SNAPSHOT_FIX_SECTIONS, "definitions")),
    )


@lru_cache(maxsize=16)
def _layer_file_summary(path: str, mtime_ns: int, size: int, sections: tuple[str, ...]) -> dict[str, Any]:
    """sha256, keyed records per section and version of one layer file (cached per file state)."""
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    data = data if isinstance(data, dict) else {}
    records = {section: len(data[section]) for section in sections if isinstance(data.get(section), (dict, list))}
    return {
        "records": records,
        "record_count": sum(records.values()),
        "version": data.get("version"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def snapshot_layers(extended: bool | None = None, corrected: bool | None = None) -> list[dict[str, Any]]:
    """The offline structured-data layers in load order, for ``GET /ready`` and the ops docs.

    One row per layer: ``name``, ``file`` (relative to the repository root), ``enabled`` (switched on by its setting;
    the v1 seed always is), ``present``, ``loaded`` (enabled and present), ``records`` (keyed records per section),
    ``record_count`` (their sum), ``version`` and ``sha256`` of the file (compare with ``data/snapshot/manifest*.json``).
    ``extended`` / ``corrected`` = ``None`` follow the environment, like :func:`load_structured_data`.
    """
    switches = {
        "extension": snapshot_ext_enabled() if extended is None else bool(extended),
        "corrections": snapshot_fix_enabled() if corrected is None else bool(corrected),
    }
    rows: list[dict[str, Any]] = []
    for name, path, env_var, sections in _snapshot_layer_specs():
        enabled = switches.get(name, True)
        row: dict[str, Any] = {"name": name, "file": _relative(path), "enabled": enabled, "present": path.exists()}
        if env_var:
            row["setting"] = env_var
        row["loaded"] = bool(enabled and row["present"])
        if row["present"]:
            stat = path.stat()
            row.update(_layer_file_summary(str(path), stat.st_mtime_ns, stat.st_size, sections))
        rows.append(row)
    return rows


def missing_snapshot_layers(extended: bool | None = None, corrected: bool | None = None) -> list[str]:
    """(round 16, review-12 L19) Names of the layers that are switched on but whose file is missing (``/ready`` 503);
    the agent states this in every answer that uses offline snapshot figures."""
    return [row["name"] for row in snapshot_layers(extended, corrected) if row["enabled"] and not row["present"]]


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
    load_dual_listings.cache_clear()
    load_synonyms.cache_clear()
    load_documents.cache_clear()
    _load_structured_data.cache_clear()
    _layer_file_summary.cache_clear()
