from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from training.progress import StageProgress, module_label


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TRAINING_MODULES = (
    "training.train_product_type",
    "training.train_intent",
    "training.train_topic",
    "training.train_question_style",
    "training.train_sentiment",
    "training.train_entity_crf",
    "training.train_clarification_gate",
    "training.train_question_style_reranker",
    "training.train_source_plan_reranker",
    "training.train_out_of_scope_detector",
    "training.train_ranker",
    "training.train_typo_linker",
)


def train_all_models(manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    progress = StageProgress("train_all", len(TRAINING_MODULES))
    for module in TRAINING_MODULES:
        progress.note(f"starting {module_label(module)}")
        subprocess.run([sys.executable, "-m", module, str(manifest_path)], check=True)
        progress.advance(f"completed {module_label(module)}")


def main() -> None:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/training_assets/manifest.json")
    train_all_models(manifest_path)


if __name__ == "__main__":
    main()
