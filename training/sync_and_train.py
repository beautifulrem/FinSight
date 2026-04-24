from __future__ import annotations

import subprocess
import sys

from training.progress import StageProgress


def main() -> None:
    progress = StageProgress("sync_and_train", 4)
    progress.advance("syncing public datasets")
    subprocess.run([sys.executable, "-m", "scripts.sync_public_datasets"], check=True)
    progress.advance("building training assets")
    subprocess.run([sys.executable, "-m", "scripts.build_training_assets"], check=True)
    progress.advance("running training preflight")
    subprocess.run([sys.executable, "-m", "training.prepare_training_run"], check=True)
    progress.advance("training all models")
    subprocess.run([sys.executable, "-m", "training.train_all"], check=True)


if __name__ == "__main__":
    main()
