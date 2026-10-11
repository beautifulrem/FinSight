"""The shipped default, end to end (round-10 review J1/J2).

``tests/conftest.py`` pins every other module to the v1 seed (``QI_OFFLINE_SNAPSHOT_EXT=false``,
``QI_OFFLINE_SNAPSHOT_FIX=false``) because their assertions were written against it. This module removes both
variables, so the service is built exactly as a deployment builds it (``Settings.from_env()`` defaults), and asks
the API for figures that only the corrections layer and the extension serve. A regression in the shipped data path
(a default flipped off, a layer that stops loading, a loader or template change that drops a corrected value or its
definition) fails here, not only in the gate set ``shipped``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from query_intelligence.data_loader import clear_data_caches

_LAYER_SWITCHES = ("QI_OFFLINE_SNAPSHOT_EXT", "QI_OFFLINE_SNAPSHOT_FIX")


@pytest.fixture(scope="module")
def shipped_service():
    patch = pytest.MonkeyPatch()
    for name in _LAYER_SWITCHES:
        patch.delenv(name, raising=False)
    clear_data_caches()
    from query_intelligence.service import build_default_service

    # live sources off (conftest), every other setting from the environment: the layers follow their defaults
    service = build_default_service(
        use_live_market=False, use_live_macro=False, use_live_news=False, use_live_announcement=False
    )
    yield service
    patch.undo()
    clear_data_caches()


@pytest.fixture(scope="module")
def client(shipped_service):
    from query_intelligence.api.app import create_app
    from query_intelligence.api.security import SecuritySettings

    app = create_app(service=shipped_service, app_config={"deepseek": {"api_key": ""}}, security=SecuritySettings())
    return TestClient(app)


def test_both_layers_are_on_by_default(shipped_service):
    from query_intelligence.config import Settings
    from query_intelligence.data_loader import snapshot_ext_enabled, snapshot_fix_enabled

    settings = Settings.from_env()
    assert (settings.offline_snapshot_ext, settings.offline_snapshot_fix) == (True, True)
    assert snapshot_ext_enabled() and snapshot_fix_enabled()
    assert shipped_service.retrieval_pipeline.snapshot_layer_settings == (True, True)


def test_ready_reports_the_three_layers(client):
    response = client.get("/ready")

    assert response.status_code == 200
    check = response.json()["checks"]["data_layers"]
    assert check["ok"] and check["loaded"] == ["seed_v1", "extension", "corrections"]


def _answer(client: TestClient, query: str) -> str:
    response = client.post("/agent/chat", json={"query": query, "mode": "workflow"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["verification"]["passed"], body
    return body["answer"]


@pytest.mark.parametrize(
    ("query", "expected", "seed_value"),
    [
        # FY2025 annual reports (corrections layer); the v1 seed had 茅台 P/E 24.6, 五粮液 ROE 29.4 / revenue 1,085 亿
        ("贵州茅台的市盈率是多少", ["PE(TTM) 21.44 倍", "ROE（加权） 32.53%"], "24.6 倍"),
        ("五粮液2025年的加权ROE是多少", ["ROE（加权） 6.89%", "营业收入 405.29 亿元"], "29.4%"),
        ("What is Kweichow Moutai's P/E ratio?", ["PE(TTM) 21.44x", "ROE (weighted average) 32.53%"], "24.6x"),
        # the 2026-04-22 bar (corrections layer); the seed had 53.61
        ("中国平安最新收盘价", ["58.03 元（2026-04-22）"], "53.61"),
        # an extension company: its own values, period-end ROE named as such (definitions from the corrections layer)
        ("宁德时代的ROE是多少", ["ROE（期末净资产） 21.42%", "PE(TTM) 15.85 倍"], "ROE（加权）"),
    ],
)
def test_answers_carry_the_shipped_figures(client, query, expected, seed_value):
    answer = _answer(client, query)

    for text in expected:
        assert text in answer, answer
    assert seed_value not in answer, answer


def test_a_two_company_gap_uses_the_corrected_figures(client):
    answer = _answer(client, "茅台和五粮液ROE差多少")

    assert "相差 25.64 个百分点" in answer, answer


# QI_TEST_SHIPPED_DEFAULT=1 (round-13 review M17): the whole suite on the shipped default, minus the seed-pinned list.


def test_the_seed_pinned_list_names_existing_tests():
    import re
    from pathlib import Path

    from tests.conftest import ROOT, seed_data_tests

    listed = seed_data_tests()
    assert listed and len(listed) < 200, len(listed)  # a short list of exceptions, not a second suite
    for node in sorted(listed):
        path, *_, name = node.split("::")
        source = ROOT / path
        assert Path(path).parts[0] == "tests" and source.is_file(), node
        function = re.sub(r"\[.*\]$", "", name)
        assert re.search(rf"^\s*def {re.escape(function)}\(", source.read_text(encoding="utf-8"), re.M), node


def test_a_shipped_default_run_has_both_layers_on():
    import os

    from query_intelligence.data_loader import snapshot_ext_enabled, snapshot_fix_enabled
    from tests.conftest import SHIPPED_DEFAULT

    if not SHIPPED_DEFAULT:
        pytest.skip("the shipped-default run (QI_TEST_SHIPPED_DEFAULT=1) checks this")
    assert os.environ["QI_OFFLINE_SNAPSHOT_EXT"] == os.environ["QI_OFFLINE_SNAPSHOT_FIX"] == "true"
    assert snapshot_ext_enabled() and snapshot_fix_enabled()
