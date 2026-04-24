from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    source_type: str
    entrypoint: str
    version: str
    license: str
    language: str
    market_scope: str
    task_types: tuple[str, ...]
    quality_score: float
    allow_training: bool
    allow_translation: bool
    enabled_by_default: bool


@dataclass(frozen=True)
class SyncResult:
    source_id: str
    status: str
    output_dir: str | None = None
    error: str | None = None
