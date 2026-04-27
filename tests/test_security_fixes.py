"""Tests for confirmed P2+ security and reliability fixes.

Covers:
  1. ZIP-slip prevention in _extract_zip_files / _safe_zip_extractall
  2. Unbounded request body in the llm_response HTTP server
  3. Query length limit enforced by Pydantic contracts
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from query_intelligence.api.app import create_app
from query_intelligence.contracts import AnalyzeRequest, MAX_QUERY_LENGTH, PipelineRequest
from query_intelligence.external_data.sync import _safe_zip_extractall
from scripts.llm_response import _MAX_REQUEST_BODY_BYTES


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


class _NoopNLUPipeline:
    def run(self, **kwargs) -> dict:
        return {
            "query_id": "Q-test",
            "raw_query": kwargs["query"],
            "normalized_query": kwargs["query"],
            "question_style": "fact",
            "product_type": {"label": "stock", "score": 1.0},
            "intent_labels": [],
            "topic_labels": [],
            "entities": [],
            "comparison_targets": [],
            "keywords": [],
            "time_scope": "unspecified",
            "forecast_horizon": "unspecified",
            "sentiment_of_user": "neutral",
            "operation_preference": "unknown",
            "required_evidence_types": [],
            "source_plan": [],
            "risk_flags": [],
            "missing_slots": [],
            "confidence": 1.0,
            "explainability": {"matched_rules": [], "top_features": []},
        }


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


from query_intelligence.service import QueryIntelligenceService


def _make_demo_client() -> TestClient:
    service = QueryIntelligenceService(_NoopNLUPipeline(), _NoopRetrievalPipeline())
    return TestClient(create_app(service=service))


# ---------------------------------------------------------------------------
# Fix 1: ZIP-slip prevention
# ---------------------------------------------------------------------------


def _make_zip(members: dict[str, bytes]) -> bytes:
    """Build an in-memory zip whose entries are keyed by *members*."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buf.getvalue()


def test_safe_zip_extract_normal_members(tmp_path: Path) -> None:
    """Normal members (no traversal) are extracted successfully."""
    zip_bytes = _make_zip({"subdir/file.txt": b"hello", "top.txt": b"world"})
    zip_path = tmp_path / "archive.zip"
    zip_path.write_bytes(zip_bytes)
    extract_dir = tmp_path / "out"
    extract_dir.mkdir()

    with zipfile.ZipFile(zip_path) as archive:
        _safe_zip_extractall(archive, extract_dir)

    assert (extract_dir / "top.txt").read_text() == "world"
    assert (extract_dir / "subdir" / "file.txt").read_text() == "hello"


def test_safe_zip_extract_blocks_path_traversal(tmp_path: Path) -> None:
    """A zip member with a path-traversal name must be rejected."""
    zip_bytes = _make_zip({"../../evil.txt": b"pwned"})
    zip_path = tmp_path / "traversal.zip"
    zip_path.write_bytes(zip_bytes)
    extract_dir = tmp_path / "out"
    extract_dir.mkdir()

    with zipfile.ZipFile(zip_path) as archive:
        with pytest.raises(ValueError, match="zip-slip"):
            _safe_zip_extractall(archive, extract_dir)

    # The evil file must not exist outside extract_dir
    assert not (tmp_path / "evil.txt").exists()


def test_safe_zip_extract_blocks_absolute_path_member(tmp_path: Path) -> None:
    """A zip member with an absolute path is blocked (zip-slip variant)."""
    # Build a zip with a member whose name is an absolute path.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as archive:
        info = zipfile.ZipInfo("/etc/evil.conf")
        archive.writestr(info, b"nope")
    zip_path = tmp_path / "abs.zip"
    zip_path.write_bytes(buf.getvalue())
    extract_dir = tmp_path / "out"
    extract_dir.mkdir()

    with zipfile.ZipFile(zip_path) as archive:
        with pytest.raises(ValueError, match="zip-slip"):
            _safe_zip_extractall(archive, extract_dir)


# ---------------------------------------------------------------------------
# Fix 2: Request body size limit in the LLM response HTTP server
# ---------------------------------------------------------------------------


def test_max_request_body_constant_is_reasonable() -> None:
    """_MAX_REQUEST_BODY_BYTES must be a positive integer <= 100 MB."""
    assert isinstance(_MAX_REQUEST_BODY_BYTES, int)
    assert 0 < _MAX_REQUEST_BODY_BYTES <= 100 * 1024 * 1024


def test_llm_response_service_handler_rejects_oversized_body(tmp_path: Path) -> None:
    """The /respond HTTP endpoint returns 400 when Content-Length exceeds the cap.

    We start a real ThreadingHTTPServer with a mocked LLM runtime, then send a
    raw HTTP request whose ``Content-Length`` header exceeds ``_MAX_REQUEST_BODY_BYTES``
    while sending only a small body so the test finishes quickly.  The handler
    must return a 400 Bad Request response without trying to read the full body.
    """
    import argparse
    import socket
    import threading
    import time
    from unittest.mock import MagicMock, patch

    mock_runtime = MagicMock()
    mock_runtime.generate.return_value = {"answer_generation": {}}

    args = argparse.Namespace(
        answer_model="dummy",
        next_question_model="dummy",
        models_dir=tmp_path,
        few_shot_source=tmp_path / "shots.jsonl",
        device_map="cpu",
        dtype="float32",
        temperature=0.0,
        answer_max_new_tokens=10,
        next_max_new_tokens=10,
        json_retries=0,
        trust_remote_code=False,
        host="127.0.0.1",
        port=0,  # OS assigns a free port
    )

    import scripts.llm_response as llm_mod
    from http.server import ThreadingHTTPServer

    server_started = threading.Event()
    server_port: list[int] = []

    def _run() -> None:
        with patch.object(llm_mod, "make_runtime", return_value=mock_runtime):
            from http import HTTPStatus
            from http.server import BaseHTTPRequestHandler
            from threading import Lock

            generation_lock = Lock()
            runtime = mock_runtime

            class _Handler(BaseHTTPRequestHandler):
                server_version = "LLMResponseHTTP/1.0"

                def log_message(self, *args, **kwargs) -> None:  # suppress output
                    pass

                def _send_json(self, status: int, payload: dict) -> None:
                    body = json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def _read_json_body(self) -> dict:
                    content_length = int(self.headers.get("Content-Length") or "0")
                    if content_length <= 0:
                        return {}
                    if content_length > _MAX_REQUEST_BODY_BYTES:
                        raise ValueError(
                            f"Request body too large: {content_length} bytes "
                            f"(max {_MAX_REQUEST_BODY_BYTES})"
                        )
                    raw = self.rfile.read(content_length)
                    return json.loads(raw.decode())

                def do_POST(self) -> None:
                    if self.path != "/respond":
                        self._send_json(HTTPStatus.NOT_FOUND, {"detail": "not found"})
                        return
                    try:
                        request = self._read_json_body()
                    except ValueError as exc:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
                        return
                    with generation_lock:
                        result = runtime.generate(request)
                    self._send_json(HTTPStatus.OK, result)

            httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            server_port.append(httpd.server_address[1])
            server_started.set()
            httpd.handle_request()  # serve exactly one request then stop
            httpd.server_close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    server_started.wait(timeout=5)
    assert server_port, "Server did not start"
    port = server_port[0]

    # Send a POST /respond with Content-Length larger than the cap but a small body.
    oversized_length = _MAX_REQUEST_BODY_BYTES + 1
    tiny_body = b'{"query": "test"}'
    raw_request = (
        f"POST /respond HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {oversized_length}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode() + tiny_body

    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(raw_request)
        response_bytes = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response_bytes += chunk

    t.join(timeout=5)

    status_line = response_bytes.split(b"\r\n", 1)[0].decode()
    assert "400" in status_line, f"Expected 400, got: {status_line!r}"

    # Also verify the response body contains the 'too large' message
    assert b"too large" in response_bytes.lower() or b"too large" in response_bytes


# ---------------------------------------------------------------------------
# Fix 3: Query length limit in Pydantic API contracts
# ---------------------------------------------------------------------------


def test_analyze_request_rejects_query_exceeding_max_length() -> None:
    """AnalyzeRequest.query must reject strings longer than MAX_QUERY_LENGTH."""
    from pydantic import ValidationError

    oversized_query = "x" * (MAX_QUERY_LENGTH + 1)
    with pytest.raises(ValidationError):
        AnalyzeRequest(query=oversized_query)


def test_pipeline_request_rejects_query_exceeding_max_length() -> None:
    """PipelineRequest.query must reject strings longer than MAX_QUERY_LENGTH."""
    from pydantic import ValidationError

    oversized_query = "a" * (MAX_QUERY_LENGTH + 1)
    with pytest.raises(ValidationError):
        PipelineRequest(query=oversized_query)


def test_analyze_request_accepts_query_at_max_length() -> None:
    """A query exactly at MAX_QUERY_LENGTH must be accepted."""
    req = AnalyzeRequest(query="x" * MAX_QUERY_LENGTH)
    assert len(req.query) == MAX_QUERY_LENGTH


def test_analyze_request_rejects_empty_query() -> None:
    """AnalyzeRequest.query must reject empty strings (min_length=1)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnalyzeRequest(query="")


def test_api_nlu_endpoint_returns_422_for_oversized_query() -> None:
    """The /nlu/analyze endpoint returns 422 when the query exceeds MAX_QUERY_LENGTH."""
    client = _make_demo_client()
    oversized = "q" * (MAX_QUERY_LENGTH + 1)
    response = client.post("/nlu/analyze", json={"query": oversized})
    assert response.status_code == 422


def test_api_pipeline_endpoint_returns_422_for_oversized_query() -> None:
    """The /query/intelligence endpoint returns 422 when query exceeds MAX_QUERY_LENGTH."""
    client = _make_demo_client()
    oversized = "q" * (MAX_QUERY_LENGTH + 1)
    response = client.post("/query/intelligence", json={"query": oversized})
    assert response.status_code == 422


def test_api_pipeline_endpoint_accepts_max_length_query() -> None:
    """The /query/intelligence endpoint accepts a query at exactly MAX_QUERY_LENGTH."""
    client = _make_demo_client()
    at_limit = "q" * MAX_QUERY_LENGTH
    response = client.post("/query/intelligence", json={"query": at_limit, "top_k": 5})
    assert response.status_code == 200
