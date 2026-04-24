from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.training_manifest import TrainingManifest, load_training_manifest
from training.train_all import TRAINING_MODULES


MIN_RECOMMENDED_FREE_BYTES = 8 * 1024 * 1024 * 1024


def prepare_training_run(manifest_path: str | Path, models_dir: str | Path = "models") -> dict:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"training manifest not found: {manifest_file}")

    manifest = load_training_manifest(manifest_file)
    model_root = Path(models_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    _assert_models_dir_writable(model_root)

    errors: list[str] = []
    warnings: list[str] = []

    artifact_entries = _collect_artifact_entries(manifest, errors)
    training_report = _load_training_report(manifest.root_dir / "training_report.json", warnings)
    module_entries = _collect_training_modules(errors)
    source_entries = _collect_source_entries(manifest, training_report, errors)
    model_entries = _collect_existing_models(model_root)

    total_artifact_bytes = sum(int(item["size_bytes"]) for item in artifact_entries)
    total_model_bytes = sum(int(item["size_bytes"]) for item in model_entries)
    disk_usage = shutil.disk_usage(model_root.resolve())
    recommended_free_bytes = max(
        MIN_RECOMMENDED_FREE_BYTES,
        total_model_bytes * 2,
        total_artifact_bytes // 8,
    )
    if disk_usage.free < recommended_free_bytes:
        warnings.append(
            "available disk space is below the recommended threshold for a full retrain: "
            f"{disk_usage.free} < {recommended_free_bytes}"
        )

    record_counts = training_report.get("record_counts", {})
    for key in (
        "classification",
        "entity",
        "retrieval_corpus",
        "qrel",
        "alias",
        "source_plan_supervision",
        "clarification_supervision",
        "out_of_scope_supervision",
        "typo_supervision",
    ):
        value = int(record_counts.get(key, 0))
        if value <= 0:
            errors.append(f"training_report record_counts[{key!r}] is missing or zero")

    report_path = manifest.root_dir / "preflight_report.json"
    report = {
        "status": "ready" if not errors else "blocked",
        "manifest_path": str(manifest_file.resolve()),
        "preflight_report_path": str(report_path.resolve()),
        "models_dir": str(model_root.resolve()),
        "artifact_files": artifact_entries,
        "training_modules": module_entries,
        "sources": source_entries,
        "record_counts": record_counts,
        "existing_models": model_entries,
        "disk": {
            "total_bytes": disk_usage.total,
            "used_bytes": disk_usage.used,
            "free_bytes": disk_usage.free,
            "recommended_free_bytes": recommended_free_bytes,
        },
        "warnings": warnings,
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(f"training preflight failed; see {report_path}")
    return report


def _collect_artifact_entries(manifest: TrainingManifest, errors: list[str]) -> list[dict]:
    entries: list[dict] = []
    required_paths = (
        ("classification", manifest.classification_path),
        ("entity_annotations", manifest.entity_annotations_path),
        ("retrieval_corpus", manifest.retrieval_corpus_path),
        ("qrels", manifest.qrels_path),
        ("alias_catalog", manifest.alias_catalog_path),
        ("source_plan_supervision", manifest.source_plan_supervision_path),
        ("clarification_supervision", manifest.clarification_supervision_path),
        ("out_of_scope_supervision", manifest.out_of_scope_supervision_path),
        ("typo_supervision", manifest.typo_supervision_path),
    )
    for name, path in required_paths:
        if path is None:
            errors.append(f"{name} artifact path is missing from manifest")
            entries.append({"name": name, "path": None, "exists": False, "size_bytes": 0})
            continue
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        if not exists:
            errors.append(f"{name} artifact file is missing: {path}")
        elif size_bytes <= 0:
            errors.append(f"{name} artifact file is empty: {path}")
        entries.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "exists": exists,
                "size_bytes": size_bytes,
            }
        )
    return entries


def _load_training_report(path: Path, warnings: list[str]) -> dict:
    if not path.exists():
        warnings.append(f"training report not found: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_training_modules(errors: list[str]) -> list[dict]:
    entries: list[dict] = []
    for module_name in TRAINING_MODULES:
        try:
            imported = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
            errors.append(f"failed to import training module {module_name}: {exc}")
            entries.append({"module": module_name, "import_ok": False, "has_main": False})
            continue
        has_main = hasattr(imported, "main")
        if not has_main:
            errors.append(f"training module {module_name} does not expose main()")
        entries.append({"module": module_name, "import_ok": True, "has_main": has_main})
    return entries


def _collect_source_entries(manifest: TrainingManifest, training_report: dict, errors: list[str]) -> list[dict]:
    report_sources = {
        str(item["source_id"]): item
        for item in training_report.get("sources", [])
        if isinstance(item, dict) and item.get("source_id")
    }
    entries: list[dict] = []
    for source in manifest.sources:
        report_item = report_sources.get(source.source_id, {})
        status = str(report_item.get("status", source.status))
        record_count = int(report_item.get("record_count", source.record_count))
        if status != "ready":
            errors.append(f"dataset source {source.source_id} is not ready: {status}")
        if record_count <= 0:
            errors.append(f"dataset source {source.source_id} has no usable records")
        entries.append(
            {
                "source_id": source.source_id,
                "status": status,
                "record_count": record_count,
            }
        )
    if not entries:
        errors.append("manifest contains no dataset sources")
    return entries


def _collect_existing_models(models_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for path in sorted(models_dir.glob("*.joblib")):
        stat = path.stat()
        entries.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "modified_at_epoch": int(stat.st_mtime),
            }
        )
    return entries


def _assert_models_dir_writable(models_dir: Path) -> None:
    probe_path = models_dir / ".preflight-write-test"
    probe_path.write_text("ready\n", encoding="utf-8")
    probe_path.unlink()


def main() -> None:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/training_assets/manifest.json")
    models_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models")
    report = prepare_training_run(manifest_path=manifest_path, models_dir=models_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "manifest": report["manifest_path"],
                "preflight_report": report["preflight_report_path"],
                "models_dir": report["models_dir"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
