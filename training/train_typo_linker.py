from __future__ import annotations

import json
import sys
from pathlib import Path

from joblib import dump

from training.progress import ProgressBar, StageProgress, call_with_optional_fit_progress

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.nlu.typo_linker import TypoLinker
from query_intelligence.nlu.typo_linker import build_typo_training_rows_from_dataset
from query_intelligence.nlu.batch_linear import estimate_batch_fit_steps


def main() -> None:
    progress = StageProgress("typo_linker", 3)
    alias_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/alias_table.csv")
    progress.advance("loading training rows")
    rows = build_typo_training_rows_from_dataset(alias_path)
    progress.note(f"loaded {len(rows)} rows")
    progress.advance("fitting model")
    fit_progress = ProgressBar("typo_linker:fit", estimate_batch_fit_steps(len(rows)), every=1)
    fit_progress.update(0, "starting", force=True)

    def on_fit_batch(completed: int, total: int) -> None:
        fit_progress.update(completed, "partial_fit batches", force=completed == total)

    linker = call_with_optional_fit_progress(TypoLinker.build_from_rows, rows, on_fit_batch)
    fit_progress.finish("fit complete")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    output_path = model_dir / "typo_linker.joblib"
    progress.advance("writing artifact")
    dump({"vectorizer": linker.vectorizer, "model": linker.model}, output_path)
    print(json.dumps({"trained_samples": len(rows), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
