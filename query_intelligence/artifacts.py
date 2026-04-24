from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "1.0"


@dataclass
class ArtifactWriter:
    output_dir: Path

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()

    def write(
        self,
        *,
        query: str,
        nlu_result: dict,
        retrieval_result: dict,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> dict:
        query_id = str(nlu_result["query_id"])
        created_at = datetime.now(timezone.utc).isoformat()
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{query_id[:8]}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        query_path = run_dir / "query.txt"
        nlu_path = run_dir / "nlu_result.json"
        retrieval_path = run_dir / "retrieval_result.json"
        manifest_path = run_dir / "manifest.json"

        query_path.write_text(query + "\n", encoding="utf-8")
        self._write_json(nlu_path, nlu_result)
        self._write_json(retrieval_path, retrieval_result)

        artifacts = {
            "query_path": str(query_path),
            "nlu_result_path": str(nlu_path),
            "retrieval_result_path": str(retrieval_path),
            "manifest_path": str(manifest_path),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "created_at": created_at,
            "query_id": query_id,
            "run_id": run_id,
            "session_id": session_id,
            "message_id": message_id,
            "query": query,
            "artifacts": artifacts,
            "modules": {
                "nlu": "query_intelligence.nlu",
                "retrieval": "query_intelligence.retrieval",
            },
        }
        self._write_json(manifest_path, manifest)
        return {
            "run_id": run_id,
            "artifact_dir": str(run_dir),
            "artifacts": artifacts,
            "manifest": manifest,
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
