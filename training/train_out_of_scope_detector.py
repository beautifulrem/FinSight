from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from joblib import dump

from training.progress import ProgressBar, StageProgress, call_with_optional_fit_progress

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.nlu.out_of_scope_detector import OutOfScopeDetector, build_out_of_scope_training_rows, estimate_out_of_scope_fit_steps

MAX_OUT_OF_SCOPE_ROWS = 120000


def _coerce_label(row: dict) -> int:
    value = row.get("out_of_scope", row.get("label", row.get("is_out_of_scope", 0)))
    return int(value or 0)


def _stable_hash(row: dict) -> str:
    return hashlib.sha1(str(row.get("query") or "").encode("utf-8")).hexdigest()


def _sample_balanced_by_source(rows: list[dict], limit: int) -> list[dict]:
    if len(rows) <= limit:
        return list(rows)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "unknown")
        grouped.setdefault(source_id, []).append(row)
    if not grouped:
        return sorted(rows, key=_stable_hash)[:limit]

    per_source = max(limit // len(grouped), 1)
    sampled: list[dict] = []
    for source_id in sorted(grouped):
        sampled.extend(sorted(grouped[source_id], key=_stable_hash)[:per_source])
    if len(sampled) < limit:
        seen_ids = {id(row) for row in sampled}
        remainder = [row for row in rows if id(row) not in seen_ids]
        sampled.extend(sorted(remainder, key=_stable_hash)[: limit - len(sampled)])
    return sampled[:limit]


def _cap_rows(rows: list[dict], limit: int) -> list[dict]:
    if len(rows) <= limit:
        return rows
    positives = [row for row in rows if _coerce_label(row) == 1]
    negatives = [row for row in rows if _coerce_label(row) == 0]
    per_label = max(limit // 2, 1)
    capped = _sample_balanced_by_source(positives, per_label) + _sample_balanced_by_source(negatives, per_label)
    if len(capped) < limit:
        capped_ids = {id(row) for row in capped}
        remainder = sorted((row for row in rows if id(row) not in capped_ids), key=_stable_hash)
        capped.extend(remainder[: limit - len(capped)])
    return capped[:limit]


def main() -> None:
    progress = StageProgress("out_of_scope_detector", 3)
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/query_labels.csv")
    progress.advance("loading training rows")
    rows = _cap_rows(build_out_of_scope_training_rows(dataset_path), MAX_OUT_OF_SCOPE_ROWS)
    progress.note(f"loaded {len(rows)} rows")
    progress.advance("fitting model")
    fit_progress = ProgressBar("out_of_scope_detector:fit", estimate_out_of_scope_fit_steps(len(rows)), every=1)
    fit_progress.update(0, "starting", force=True)

    def on_fit_batch(completed: int, total: int) -> None:
        fit_progress.update(completed, "partial_fit batches", force=completed == total)

    detector = call_with_optional_fit_progress(OutOfScopeDetector.build_from_rows, rows, on_fit_batch)
    fit_progress.finish("fit complete")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    output_path = model_dir / "out_of_scope_detector.joblib"
    progress.advance("writing artifact")
    dump(detector.model, output_path)
    print(json.dumps({"trained_samples": len(rows), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
