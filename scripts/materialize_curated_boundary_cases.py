from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.external_data.build_assets import _build_curated_boundary_classification_rows
from query_intelligence.training_data import (
    build_clarification_supervision_rows_from_records,
    build_source_plan_supervision_rows_from_records,
)


ASSETS_DIR = ROOT / "data" / "training_assets"
CURATED_SOURCE_ID = "curated_boundary_cases"
OOD_REPEAT = 12


def main() -> None:
    curated_rows = _with_training_split(_build_curated_boundary_classification_rows())
    _replace_source_rows(ASSETS_DIR / "classification.jsonl", curated_rows)
    _replace_source_rows(ASSETS_DIR / "source_plan_supervision.jsonl", build_source_plan_supervision_rows_from_records(curated_rows))
    _replace_source_rows(ASSETS_DIR / "clarification_supervision.jsonl", build_clarification_supervision_rows_from_records(curated_rows))
    _replace_source_rows(ASSETS_DIR / "out_of_scope_supervision.jsonl", _build_ood_rows(curated_rows))
    _update_training_report(len(curated_rows))
    print(json.dumps({"materialized_curated_boundary_rows": len(curated_rows)}, ensure_ascii=False), flush=True)


def _with_training_split(rows: list[dict]) -> list[dict]:
    assigned: list[dict] = []
    for index, row in enumerate(rows):
        cloned = dict(row)
        cloned["split"] = "test" if index % 10 == 0 else ("valid" if index % 10 == 1 else "train")
        cloned["source_row_id"] = f"{CURATED_SOURCE_ID}:{index}"
        assigned.append(cloned)
    return assigned


def _build_ood_rows(rows: list[dict]) -> list[dict]:
    ood_rows: list[dict] = []
    for index, row in enumerate(rows):
        for repeat in range(OOD_REPEAT):
            ood_rows.append(
                {
                    "query": row["query"],
                    "product_type": row["product_type"],
                    "domain": "finance",
                    "label": 0,
                    "out_of_scope": 0,
                    "source_family": "curated_boundary_positive",
                    "is_in_domain": True,
                    "source_id": CURATED_SOURCE_ID,
                    "source_row_id": f"{CURATED_SOURCE_ID}:ood:{index}:{repeat}",
                }
            )
    return ood_rows


def _replace_source_rows(path: Path, new_rows: list[dict]) -> None:
    existing = _read_jsonl(path)
    kept = [row for row in existing if str(row.get("source_id") or "") != CURATED_SOURCE_ID]
    _write_jsonl(path, [*kept, *new_rows])


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def _update_training_report(curated_count: int) -> None:
    report_path = ASSETS_DIR / "training_report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sources = [source for source in report.get("sources", []) if source.get("source_id") != CURATED_SOURCE_ID]
    sources.append({"source_id": CURATED_SOURCE_ID, "record_count": curated_count, "status": "ready"})
    report["sources"] = sorted(sources, key=lambda source: str(source.get("source_id") or ""))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
