from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingSourceUsage:
    source_id: str
    record_count: int
    status: str


@dataclass(frozen=True)
class TrainingManifest:
    root_dir: Path
    classification_path: Path
    entity_annotations_path: Path
    retrieval_corpus_path: Path
    qrels_path: Path
    alias_catalog_path: Path
    sources: tuple[TrainingSourceUsage, ...]
    source_plan_supervision_path: Path | None = None
    clarification_supervision_path: Path | None = None
    out_of_scope_supervision_path: Path | None = None
    typo_supervision_path: Path | None = None


def load_training_manifest(path: Path) -> TrainingManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    return TrainingManifest(
        root_dir=root,
        classification_path=root / payload["classification_path"],
        entity_annotations_path=root / payload["entity_annotations_path"],
        retrieval_corpus_path=root / payload["retrieval_corpus_path"],
        qrels_path=root / payload["qrels_path"],
        alias_catalog_path=root / payload["alias_catalog_path"],
        source_plan_supervision_path=_resolve_optional_path(root, payload, "source_plan_supervision_path"),
        clarification_supervision_path=_resolve_optional_path(root, payload, "clarification_supervision_path"),
        out_of_scope_supervision_path=_resolve_optional_path(root, payload, "out_of_scope_supervision_path"),
        typo_supervision_path=_resolve_optional_path(root, payload, "typo_supervision_path"),
        sources=tuple(
            TrainingSourceUsage(
                source_id=item["source_id"],
                record_count=int(item["record_count"]),
                status=item["status"],
            )
            for item in payload.get("sources", [])
        ),
    )


def _resolve_optional_path(root: Path, payload: dict, key: str) -> Path | None:
    relative_path = payload.get(key)
    if not relative_path:
        return None
    return root / str(relative_path)
