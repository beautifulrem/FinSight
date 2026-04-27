from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from training.progress import StageProgress


ROOT = Path(__file__).resolve().parents[1]

TEST_GROUPS = (
    ("data_assets", ["tests/test_data_assets.py"]),
    ("adapters_assets", ["tests/test_dataset_adapters.py", "tests/test_training_assets.py"]),
    ("sentiment", ["tests/test_sentiment_preprocessor.py", "tests/test_sentiment_pipeline.py"]),
    ("core_nlu_retrieval", ["tests/test_query_intelligence.py", "tests/test_ml_upgrades.py"]),
    ("manual_and_fuzz", ["tests/test_manual_test_runner.py", "tests/test_fuzz_query_intelligence_report.py"]),
    ("real_integrations", ["tests/test_real_integrations.py"]),
)


def main() -> None:
    progress = StageProgress("run_test_suite", len(TEST_GROUPS))
    for name, paths in TEST_GROUPS:
        progress.note(f"starting {name}")
        subprocess.run([sys.executable, "-m", "pytest", "-q", *paths], cwd=ROOT, check=True)
        progress.advance(f"completed {name}")


if __name__ == "__main__":
    main()
