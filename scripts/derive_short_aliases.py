"""(round 16, review-12 L9) Derive short names for the covered securities from the entity table and add them to the
runtime alias table (idempotent).

    python scripts/derive_short_aliases.py            # add the accepted rows, write the derivation report
    python scripts/derive_short_aliases.py --check    # exit 1 when the alias table or the report is out of date

The rules and the collision checks are in ``query_intelligence.runtime_entity_assets.derive_short_aliases``. The
covered securities are those with structured data offline: the seed snapshot (``data/structured_data.json``) and the
snapshot extension (``data/snapshot/structured_data_ext.json``). Every candidate, accepted or rejected, is written to
``data/runtime/derived_short_aliases.json`` with its rule and its evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.runtime_entity_assets import (  # noqa: E402
    ALIAS_FIELDS,
    SHORT_ALIAS_DOMINANCE,
    SHORT_ALIAS_MAX_RIVALS,
    SHORT_ALIAS_MAX_WORD_FREQ,
    SHORT_ALIAS_MIN_DOCUMENTS,
    SHORT_ALIAS_MIN_SUFFIX_NAMES,
    derive_short_aliases,
    document_name_counts,
    short_alias_candidates,
    short_alias_rows,
)

RUNTIME = ROOT / "data" / "runtime"
REPORT = RUNTIME / "derived_short_aliases.json"


def covered_symbols() -> set[str]:
    symbols: set[str] = set()
    for path in (ROOT / "data" / "structured_data.json", ROOT / "data" / "snapshot" / "structured_data_ext.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        symbols.update(str(symbol) for symbol in (data.get("fundamental_sql") or {}))
    return symbols


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def derive() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    import jieba

    jieba.setLogLevel(60)
    jieba.initialize()
    entities = _read(RUNTIME / "entity_master.csv")
    for row in entities:
        row["entity_id"] = str(int(row["entity_id"])) if str(row.get("entity_id") or "").isdigit() else row["entity_id"]
    aliases = [row for row in _read(RUNTIME / "alias_table.csv") if row.get("alias_type") != "derived_short_alias"]
    scope = covered_symbols()
    candidates = short_alias_candidates(entities, scope)
    names = {row["entity_id"]: row["canonical_name"] for row in entities}
    listed = [row["canonical_name"] for row in entities if row.get("entity_type") in {"stock", "etf", "index", "fund"}]
    wanted = {names[entity_id] for claims in candidates.values() for entity_id, _rule in claims}
    wanted |= {name for name in listed if any(name.startswith(form) for form in candidates)}
    with (RUNTIME / "documents.jsonl").open(encoding="utf-8") as handle:
        documents = (json.loads(line) for line in handle if line.strip())
        counts = document_name_counts(documents, wanted, candidates)
    # the rows this script added before are not "another entity's alias" of themselves
    previous = {(row["entity_id"], row["normalized_alias"]) for row in aliases if _is_derived(row)}
    base = [row for row in aliases if (row["entity_id"], row["normalized_alias"]) not in previous]
    accepted, rejected = derive_short_aliases(
        entities, base, scope=scope, document_counts=counts, word_frequency=dict(jieba.dt.FREQ)
    )
    parameters = {
        "scope": sorted(scope),
        "min_suffix_names": SHORT_ALIAS_MIN_SUFFIX_NAMES,
        "max_word_frequency": SHORT_ALIAS_MAX_WORD_FREQ,
        "max_rivals": SHORT_ALIAS_MAX_RIVALS,
        "dominance": SHORT_ALIAS_DOMINANCE,
        "min_documents": SHORT_ALIAS_MIN_DOCUMENTS,
    }
    return accepted, rejected, parameters


_DERIVED_MARK = "derived"


def _is_derived(row: dict[str, str]) -> bool:
    return str(row.get("is_official") or "").startswith(_DERIVED_MARK)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    accepted, rejected, parameters = derive()
    report = {"parameters": parameters, "accepted": accepted, "rejected": rejected}
    rendered = json.dumps(report, ensure_ascii=False, indent=1) + "\n"
    path = RUNTIME / "alias_table.csv"
    rows = _read(path)
    kept = [row for row in rows if not _is_derived(row)]
    next_id = max(int(row["alias_id"]) for row in kept) + 1
    added = short_alias_rows(accepted, next_id)
    wanted = kept + added
    if args.check:
        same_rows = [(r["entity_id"], r["alias_text"], r["is_official"]) for r in rows if _is_derived(r)] == [
            (r["entity_id"], r["alias_text"], r["is_official"]) for r in added
        ]
        same_report = REPORT.exists() and REPORT.read_text(encoding="utf-8") == rendered
        print(json.dumps({"rows_up_to_date": same_rows, "report_up_to_date": same_report}))
        return 0 if same_rows and same_report else 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALIAS_FIELDS)
        writer.writeheader()
        writer.writerows(wanted)
    REPORT.write_text(rendered, encoding="utf-8")
    print(json.dumps({"accepted": [(a["alias"], a["canonical_name"]) for a in accepted], "rejected": len(rejected)},
                     ensure_ascii=False))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
