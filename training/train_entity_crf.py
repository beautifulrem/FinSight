from __future__ import annotations

import json
import sys
from pathlib import Path

from joblib import dump

from training.progress import ProgressBar, StageProgress

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.data_loader import load_aliases
from query_intelligence.nlu.entity_boundary_crf import EntityBoundaryCRF
from query_intelligence.training_data import load_training_rows
from query_intelligence.training_manifest import load_training_manifest


def _spans_from_tags(text: str, tags: list[str]) -> list[dict[str, int]]:
    spans: list[dict[str, int]] = []
    start: int | None = None
    for index, tag in enumerate(tags):
        if tag == "B-ENT":
            if start is not None:
                spans.append({"start": start, "end": index})
            start = index
            continue
        if tag != "I-ENT" and start is not None:
            spans.append({"start": start, "end": index})
            start = None
    if start is not None:
        spans.append({"start": start, "end": len(text)})
    return spans


def _normalize_annotations(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        query = str(row.get("query") or row.get("text") or "").strip()
        if not query:
            tokens = row.get("tokens") or []
            if isinstance(tokens, list):
                query = "".join(str(token) for token in tokens)
        spans = row.get("spans")
        if not spans:
            tags = row.get("tags") or row.get("bio_tags") or []
            if isinstance(tags, list):
                spans = _spans_from_tags(query, [str(tag) for tag in tags])
        if query and spans:
            normalized.append({"query": query, "spans": spans})
    return normalized


def main() -> None:
    progress = StageProgress("entity_crf", 3)
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/query_labels.csv")
    manifest = None
    if dataset_path.suffix.lower() == ".json":
        try:
            manifest = load_training_manifest(dataset_path)
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
            manifest = None

    annotations_path = manifest.entity_annotations_path if manifest else dataset_path.with_name("entity_bio_annotations.jsonl")
    progress.advance("loading training rows")
    if annotations_path.exists():
        with annotations_path.open("r", encoding="utf-8") as handle:
            annotations = _normalize_annotations([json.loads(line) for line in handle if line.strip()])
        progress.note(f"loaded {len(annotations)} annotation rows")
        progress.advance("fitting model")
        feature_progress = ProgressBar("entity_crf:features", len(annotations), every=max(len(annotations) // 20, 1))
        feature_progress.update(0, "building char features", force=True)

        def on_features(completed: int, total: int) -> None:
            feature_progress.update(completed, "building char features", force=completed == total)

        progress.note("CRF fitting starts after feature build; sklearn-crfsuite reports optimizer iterations")
        model = EntityBoundaryCRF.build_from_annotations(annotations, feature_progress_callback=on_features, verbose=True)
        trained_samples = len(annotations)
    else:
        rows = load_training_rows(manifest.classification_path if manifest else dataset_path)
        aliases = load_aliases()
        progress.note(f"loaded {len(rows)} fallback rows")
        progress.advance("fitting model")
        queries = [row["query"] for row in rows]
        alias_values = [alias["normalized_alias"] for alias in aliases]
        feature_progress = ProgressBar("entity_crf:features", len(queries), every=max(len(queries) // 20, 1))
        feature_progress.update(0, "building char features", force=True)

        def on_features(completed: int, total: int) -> None:
            feature_progress.update(completed, "building char features", force=completed == total)

        progress.note("CRF fitting starts after feature build; sklearn-crfsuite reports optimizer iterations")
        model = EntityBoundaryCRF.build_from_queries(
            queries,
            alias_values,
            feature_progress_callback=on_features,
            verbose=True,
        )
        trained_samples = len(rows)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    output_path = model_dir / "entity_crf.joblib"
    progress.advance("writing artifact")
    dump(model, output_path)
    print(json.dumps({"trained_samples": trained_samples, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
