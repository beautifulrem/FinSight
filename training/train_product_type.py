from __future__ import annotations

import json
import sys
from pathlib import Path

from joblib import dump

from training.progress import ProgressBar, StageProgress

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.nlu.classifiers import PRODUCT_SAMPLES, ProductTypeClassifier, estimate_text_classifier_fit_steps
from query_intelligence.training_data import sample_training_rows_per_value

MAX_PRODUCT_TYPE_ROWS_PER_LABEL = 3000


def main() -> None:
    progress = StageProgress("product_type", 3)
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/query_labels.csv")
    progress.advance("loading training rows")
    if dataset_path and dataset_path.exists():
        rows = sample_training_rows_per_value(dataset_path, "product_type", MAX_PRODUCT_TYPE_ROWS_PER_LABEL)
        progress.note(f"loaded {len(rows)} rows")
        trained_samples = len(rows)
    else:
        rows = [{"query": text, "product_type": label} for text, label in PRODUCT_SAMPLES]
        progress.note(f"using fallback samples: {len(rows)}")
        trained_samples = len(rows)
    progress.advance("fitting model")
    fit_progress = ProgressBar("product_type:fit", estimate_text_classifier_fit_steps(len(rows)), every=1)
    fit_progress.update(0, "starting", force=True)

    def on_fit_batch(completed: int, total: int) -> None:
        fit_progress.update(completed, "partial_fit batches", force=completed == total)

    model = ProductTypeClassifier.build_from_records(rows, fit_progress_callback=on_fit_batch).model
    fit_progress.finish("fit complete")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    output_path = model_dir / "product_type.joblib"
    progress.advance("writing artifact")
    dump(model, output_path)
    print(json.dumps({"trained_samples": trained_samples, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
