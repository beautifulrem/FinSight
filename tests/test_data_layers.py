"""Offline data layers are shipped, reported and never skipped silently (round-10 review J2).

The corrections layer (``data/snapshot/structured_data_fix.json``) became the product default in round 13, but the
Docker image copied only the extension, and ``load_structured_data`` skipped a missing layer without a word, so the
container served the v1 seed figures the README calls wrong. These tests pin the three fixes: the image copies every
layer file the loader reads, a configured-but-missing layer logs a warning, and ``GET /ready`` reports each layer
(record count, sha256) and is 503 when a switched-on layer is missing.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from agent_fakes import StubService
from fastapi.testclient import TestClient

from query_intelligence import data_loader
from query_intelligence.api import readiness
from query_intelligence.api.app import create_app
from query_intelligence.api.security import SecuritySettings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_caches():
    data_loader.clear_data_caches()
    yield
    data_loader.clear_data_caches()


def test_every_layer_is_reported_with_records_and_the_manifest_hash():
    layers = {row["name"]: row for row in data_loader.snapshot_layers(True, True)}

    assert list(layers) == ["seed_v1", "extension", "corrections"]
    assert all(row["loaded"] for row in layers.values())
    assert layers["corrections"]["records"] == {"market_api": 4, "fundamental_sql": 3, "definitions": 10}
    assert layers["extension"]["records"] == {"market_api": 15, "fundamental_sql": 10}
    # the hashes are the ones the build scripts recorded, so ops can match a running pod to a committed build
    for name, manifest in (("extension", "manifest.json"), ("corrections", "manifest_fix.json")):
        recorded = json.loads((ROOT / "data" / "snapshot" / manifest).read_text(encoding="utf-8"))["output"]
        assert recorded["file"] == layers[name]["file"]
        assert recorded["sha256"] == layers[name]["sha256"]


def test_a_switched_off_layer_is_reported_as_not_loaded():
    layers = {row["name"]: row for row in data_loader.snapshot_layers(True, False)}

    assert layers["corrections"]["enabled"] is False and layers["corrections"]["loaded"] is False
    assert layers["corrections"]["present"] is True
    assert layers["extension"]["loaded"] is True


def test_a_configured_layer_that_is_missing_logs_a_warning_and_is_not_applied(monkeypatch, tmp_path, caplog):
    missing = tmp_path / "structured_data_fix.json"
    monkeypatch.setattr(data_loader, "SNAPSHOT_FIX_PATH", missing)

    with caplog.at_level(logging.WARNING, logger="query_intelligence.data_loader"):
        corrected = data_loader.load_structured_data(True, True)

    assert any("corrections" in r.message and "QI_OFFLINE_SNAPSHOT_FIX" in r.message for r in caplog.records)
    assert corrected == data_loader.load_structured_data(True, False)
    row = {r["name"]: r for r in data_loader.snapshot_layers(True, True)}["corrections"]
    assert row["enabled"] is True and row["present"] is False and row["loaded"] is False


def test_the_shipped_layers_serve_the_corrected_figures():
    data = data_loader.load_structured_data(True, True)

    wuliangye = data["fundamental_sql"]["000858.SZ"]
    assert (wuliangye["roe"], round(wuliangye["revenue"] / 1e8, 2)) == (6.89, 405.29)
    assert data["fundamental_sql"]["600519.SH"]["pe_ttm"] == 21.44
    assert data["market_api"]["601318.SH"]["close"] == 58.03


def _client_with_layers(extended: bool, corrected: bool) -> TestClient:
    service = StubService()
    service.retrieval_pipeline = SimpleNamespace(snapshot_layer_settings=(extended, corrected))
    app = create_app(service=service, app_config={"deepseek": {"api_key": ""}}, security=SecuritySettings())
    return TestClient(app)


def test_ready_reports_the_loaded_layers():
    response = _client_with_layers(True, True).get("/ready")

    check = response.json()["checks"]["data_layers"]
    assert check["ok"] is True
    assert check["loaded"] == ["seed_v1", "extension", "corrections"]
    corrections = check["layers"][2]
    assert corrections["record_count"] == 17 and len(corrections["sha256"]) == 64
    assert corrections["version"].startswith("fix-")


def test_ready_is_503_when_a_switched_on_layer_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader, "SNAPSHOT_FIX_PATH", tmp_path / "absent.json")

    response = _client_with_layers(True, True).get("/ready")

    assert response.status_code == 503
    check = response.json()["checks"]["data_layers"]
    assert check["ok"] is False
    assert "corrections is switched on (QI_OFFLINE_SNAPSHOT_FIX)" in check["error"]
    assert check["loaded"] == ["seed_v1", "extension"]


def test_a_missing_layer_that_is_switched_off_does_not_block_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader, "SNAPSHOT_FIX_PATH", tmp_path / "absent.json")
    service = SimpleNamespace(retrieval_pipeline=SimpleNamespace(snapshot_layer_settings=(True, False)))

    detail = readiness.check_data_layers(service)

    assert detail["ok"] is True and detail["loaded"] == ["seed_v1", "extension"]


def test_the_real_pipeline_records_its_layer_settings(offline_service):
    # tests/conftest.py switches both layers off for services built from the environment
    pipeline = offline_service.retrieval_pipeline

    assert pipeline.snapshot_layer_settings == (False, False)
    assert readiness.check_data_layers(offline_service)["loaded"] == ["seed_v1"]


def _docker_copied_paths() -> set[str]:
    text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
    copied: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "COPY" and not any(p.startswith("--from") for p in parts):
            copied.update(p for p in parts[1:-1] if not p.startswith("--"))
    return copied


def _dockerignored(path: str) -> bool:
    patterns = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for pattern in patterns:
        regex = re.escape(pattern.rstrip("/")).replace(r"\*\*/", "(?:.*/)?").replace(r"\*", "[^/]*")
        if re.fullmatch(f"{regex}(?:/.*)?", path):
            return True
    return False


@pytest.mark.parametrize("name", ["extension", "corrections"])
def test_the_docker_image_ships_every_layer_the_loader_reads(name):
    # J2: the image copied structured_data_ext.json but not structured_data_fix.json
    layer = {row["name"]: row for row in data_loader.snapshot_layers(True, True)}[name]
    copied = _docker_copied_paths()

    assert layer["file"] in copied or any(layer["file"].startswith(f"{p.rstrip('/')}/") for p in copied)
    assert not _dockerignored(layer["file"])


def test_the_docker_image_ships_the_corrections_sources_and_manifests():
    copied = _docker_copied_paths()

    assert {"data/snapshot/manifest.json", "data/snapshot/manifest_fix.json", "data/snapshot/corrections"} <= copied
    assert not _dockerignored("data/snapshot/corrections/sources.json")
