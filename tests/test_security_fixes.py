from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from query_intelligence.api.app import create_app
from query_intelligence.contracts import (
    AnalyzeRequest,
    MAX_DIALOG_CONTEXT_ITEMS,
    MAX_NLU_LIST_ITEMS,
    MAX_QUERY_LENGTH,
    PipelineRequest,
    RetrievalRequest,
)
from query_intelligence.external_data import sync
from query_intelligence.service import QueryIntelligenceService
from scripts.llm_response import coerce_query


class _NoopNLUPipeline:
    def run(self, **kwargs) -> dict:
        return _valid_nlu(raw_query=kwargs["query"], normalized_query=kwargs["query"])


class _NoopRetrievalPipeline:
    def run(self, *, nlu_result: dict, top_k: int, debug: bool) -> dict:
        return {
            "query_id": nlu_result["query_id"],
            "nlu_snapshot": nlu_result,
            "executed_sources": [],
            "documents": [],
            "structured_data": [],
            "evidence_groups": [],
            "coverage": {},
            "coverage_detail": {},
            "warnings": [],
            "retrieval_confidence": 1.0,
            "analysis_summary": {},
            "debug_trace": {"candidate_count": top_k, "after_dedup": 0, "top_ranked": []},
        }


def _client() -> TestClient:
    service = QueryIntelligenceService(_NoopNLUPipeline(), _NoopRetrievalPipeline())
    return TestClient(create_app(service=service))


def _valid_nlu(**overrides) -> dict:
    payload = {
        "query_id": "Q-test",
        "raw_query": "茅台今天为什么跌",
        "normalized_query": "茅台今天为什么跌",
        "question_style": "why",
        "product_type": {"label": "stock", "score": 1.0},
        "intent_labels": [],
        "topic_labels": [],
        "entities": [],
        "comparison_targets": [],
        "keywords": [],
        "time_scope": "today",
        "forecast_horizon": "unspecified",
        "sentiment_of_user": "neutral",
        "operation_preference": "unknown",
        "required_evidence_types": [],
        "source_plan": ["news"],
        "risk_flags": [],
        "missing_slots": [],
        "confidence": 1.0,
        "explainability": {"matched_rules": [], "top_features": []},
    }
    payload.update(overrides)
    return payload


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buf.getvalue()


def test_safe_zip_extract_blocks_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    zip_path.write_bytes(_zip_bytes({"../../evil.txt": b"pwned"}))
    extract_dir = tmp_path / "out"

    with zipfile.ZipFile(zip_path) as archive:
        with pytest.raises(ValueError, match="unsafe archive member"):
            sync._safe_zip_extractall(archive, extract_dir)

    assert not (tmp_path / "evil.txt").exists()


def test_safe_zip_extract_normal_members(tmp_path: Path) -> None:
    zip_path = tmp_path / "ok.zip"
    zip_path.write_bytes(_zip_bytes({"nested/file.txt": b"ok"}))
    extract_dir = tmp_path / "out"

    with zipfile.ZipFile(zip_path) as archive:
        sync._safe_zip_extractall(archive, extract_dir)

    assert (extract_dir / "nested" / "file.txt").read_text() == "ok"


def test_safe_zip_extract_blocks_symlink_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(info, "../outside")

    with zipfile.ZipFile(zip_path) as archive:
        with pytest.raises(ValueError, match="symlink"):
            sync._safe_zip_extractall(archive, tmp_path / "out")


def test_http_download_rejects_declared_oversize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Response:
        headers = {"Content-Length": "5"}

        def raise_for_status(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(sync.requests, "get", lambda *args, **kwargs: _Response())

    with pytest.raises(ValueError, match="download too large"):
        sync.download_http_file("https://example.test/data.bin", tmp_path, max_bytes=4)


def test_http_download_rejects_streamed_oversize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Response:
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int):
            yield b"abc"
            yield b"de"

        def close(self) -> None:
            pass

    monkeypatch.setattr(sync.requests, "get", lambda *args, **kwargs: _Response())

    with pytest.raises(ValueError, match="download too large"):
        sync.download_http_file("https://example.test/data.bin", tmp_path, max_bytes=4)

    assert not (tmp_path / "data.bin").exists()
    assert not (tmp_path / ".data.bin.part").exists()


def test_query_contracts_reject_oversized_queries() -> None:
    oversized = "x" * (MAX_QUERY_LENGTH + 1)

    with pytest.raises(ValidationError):
        AnalyzeRequest(query=oversized)
    with pytest.raises(ValidationError):
        PipelineRequest(query=oversized)
    with pytest.raises(ValueError, match="query must be less than or equal"):
        coerce_query(oversized)


def test_request_contracts_reject_large_contexts() -> None:
    with pytest.raises(ValidationError):
        PipelineRequest(query="hello", dialog_context=[{}] * (MAX_DIALOG_CONTEXT_ITEMS + 1))


def test_retrieval_request_rejects_oversized_direct_nlu_payload() -> None:
    bad_nlu = _valid_nlu(keywords=["x"] * (MAX_NLU_LIST_ITEMS + 1))

    with pytest.raises(ValidationError):
        RetrievalRequest(nlu_result=bad_nlu)


def test_retrieval_endpoint_rejects_oversized_direct_nlu_payload() -> None:
    bad_nlu = _valid_nlu(source_plan=["news"] * (MAX_NLU_LIST_ITEMS + 1))

    response = _client().post("/retrieval/search", json={"nlu_result": bad_nlu, "top_k": 5})

    assert response.status_code == 422
