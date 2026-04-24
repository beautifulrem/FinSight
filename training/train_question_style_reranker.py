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

from query_intelligence.nlu.question_style_reranker import QuestionStyleReranker, build_question_style_reranker_rows
from query_intelligence.nlu.batch_linear import estimate_batch_fit_steps

MAX_QUESTION_STYLE_RERANKER_ROWS = 30000


def _cap_rows(rows: list[dict], limit: int) -> list[dict]:
    if len(rows) <= limit:
        return rows
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha1(str(row.get("query") or "").encode("utf-8")).hexdigest(),
    )
    return ordered[:limit]


def main() -> None:
    progress = StageProgress("question_style_reranker", 3)
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/query_labels.csv")
    progress.advance("loading training rows")
    rows = _cap_rows(build_question_style_reranker_rows(dataset_path), MAX_QUESTION_STYLE_RERANKER_ROWS)
    progress.note(f"loaded {len(rows)} rows")
    progress.advance("fitting model")
    fit_progress = ProgressBar("question_style_reranker:fit", estimate_batch_fit_steps(len(rows)), every=1)
    fit_progress.update(0, "starting", force=True)

    def on_fit_batch(completed: int, total: int) -> None:
        fit_progress.update(completed, "partial_fit batches", force=completed == total)

    reranker = QuestionStyleReranker.build_from_rows(rows, fit_progress_callback=on_fit_batch)
    fit_progress.finish("fit complete")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    output_path = model_dir / "question_style_reranker.joblib"
    progress.advance("writing artifact")
    dump({"vectorizer": reranker.vectorizer, "model": reranker.model}, output_path)
    print(json.dumps({"trained_samples": len(rows), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
