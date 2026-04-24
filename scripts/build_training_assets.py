from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.config import Settings
from query_intelligence.external_data.build_assets import build_training_assets
from training.progress import StageProgress


def main() -> None:
    progress = StageProgress("build_training_assets_cli", 1)
    settings = Settings.from_env()
    progress.advance("building assets from raw datasets")
    manifest = build_training_assets(
        raw_root=ROOT / "data" / "external" / "raw",
        output_root=ROOT / "data" / "training_assets",
        enable_translation=settings.enable_translation,
    )
    print(json.dumps({"manifest": str(manifest.root_dir / "manifest.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
