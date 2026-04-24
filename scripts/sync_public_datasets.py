from __future__ import annotations

import json
from pathlib import Path

from query_intelligence.config import Settings
from query_intelligence.external_data.models import SyncResult
from query_intelligence.external_data.registry import build_default_dataset_registry, resolve_enabled_sources
from query_intelligence.external_data.sync import sync_source
from training.progress import StageProgress


def main() -> None:
    settings = Settings.from_env()
    registry = build_default_dataset_registry()
    sources = resolve_enabled_sources(registry, settings)
    if not settings.enable_external_data:
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "QI_ENABLE_EXTERNAL_DATA is not enabled",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    raw_root = Path("data/external/raw")
    progress = StageProgress("sync_public_datasets", max(len(sources), 1))
    if not sources:
        progress.note("no enabled dataset sources")
        return

    results: list[SyncResult] = []
    for source in sources:
        progress.note(f"starting {source.source_id}")
        try:
            result = sync_source(source, raw_root=raw_root)
        except Exception as exc:  # pragma: no cover - integration guard
            result = SyncResult(
                source_id=source.source_id,
                status="failed",
                output_dir=str(raw_root / source.source_id / source.version),
                error=str(exc),
            )
        results.append(result)
        progress.advance(f"{source.source_id}:{result.status}")
    for result in results:
        print(
            json.dumps(
                {
                    "source_id": result.source_id,
                    "status": result.status,
                    "error": result.error,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
