from __future__ import annotations

import json
import sys
import hashlib
from pathlib import Path

from joblib import dump

from training.progress import ProgressBar, StageProgress

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.nlu.classifiers import SingleLabelTextClassifier, estimate_text_classifier_fit_steps
from query_intelligence.training_data import sample_training_rows_for_label

MAX_QUESTION_STYLE_ROWS = 20000


def _cap_rows(rows: list[dict], limit: int) -> list[dict]:
    if len(rows) <= limit:
        return rows
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha1(str(row.get("split_lock_key") or row.get("query") or "").encode("utf-8")).hexdigest(),
    )
    return ordered[:limit]


def main() -> None:
    progress = StageProgress("question_style", 3)
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/query_labels.csv")
    progress.advance("loading training rows")
    rows = sample_training_rows_for_label(dataset_path, "question_style", MAX_QUESTION_STYLE_ROWS)
    progress.note(f"loaded {len(rows)} rows")
    progress.advance("fitting model")
    fit_progress = ProgressBar("question_style:fit", estimate_text_classifier_fit_steps(len(rows)), every=1)
    fit_progress.update(0, "starting", force=True)

    def on_fit_batch(completed: int, total: int) -> None:
        fit_progress.update(completed, "partial_fit batches", force=completed == total)

    classifier = SingleLabelTextClassifier.build_from_records(rows, "question_style", fit_progress_callback=on_fit_batch)
    fit_progress.finish("fit complete")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    progress.advance("writing artifact")
    dump(classifier.model, model_dir / "question_style.joblib")
    print(json.dumps({"trained_samples": len(rows), "output": str(model_dir / "question_style.joblib")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
