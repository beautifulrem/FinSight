"""Round-16 rule L19 (round-12 review): with the corrections layer switched on but its file missing, ``GET /ready``
is 503 and the chat used to serve the v1 seed values with no word about it. Chosen behaviour: the answer is given,
opens with "数据更正层未加载，以下为未经更正的原始数据" (en: "The data corrections layer is not loaded…"), carries the
same limitation and ``degraded: data_layer_missing:corrections``. With the layer present, or with the layer switched
off on purpose, no note. Figures from live providers are not affected (``data_layer_note`` reads the provenance).
"""

from __future__ import annotations

import pytest

from query_intelligence import data_loader
from query_intelligence.agent.compliance import DATA_LAYER_NOTE_EN, DATA_LAYER_NOTE_ZH, data_layer_note


def _agent(**settings):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=False, use_live_macro=False, use_live_news=False, use_live_announcement=False, **settings
    )
    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]))
    return AgentService(runtime, trace_sinks=[]), runtime


@pytest.fixture
def without_corrections_file(monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader, "SNAPSHOT_FIX_PATH", tmp_path / "structured_data_fix.json")
    data_loader.clear_data_caches()
    yield
    monkeypatch.undo()
    data_loader.clear_data_caches()


def test_missing_layers_are_the_switched_on_ones_without_a_file(without_corrections_file):
    assert data_loader.missing_snapshot_layers(True, True) == ["corrections"]
    assert data_loader.missing_snapshot_layers(True, False) == []


def test_answers_from_uncorrected_seed_figures_say_so(without_corrections_file):
    agent, runtime = _agent(offline_snapshot_ext=True, offline_snapshot_fix=True)
    try:
        zh = agent.chat("五粮液的市盈率和净资产收益率", session_id="r16-l19-zh")
        en = agent.chat("What is Wuliangye's P/E ratio?", session_id="r16-l19-en")
    finally:
        runtime.close()
    assert zh["answer"].startswith(DATA_LAYER_NOTE_ZH) and DATA_LAYER_NOTE_ZH in zh["limitations"]
    assert "data_layer_missing:corrections" in zh["degraded"]
    assert en["answer"].startswith(DATA_LAYER_NOTE_EN) and DATA_LAYER_NOTE_EN in en["limitations"]


def test_no_note_with_the_layer_present_or_switched_off():
    for settings in ({"offline_snapshot_ext": True, "offline_snapshot_fix": True}, {"offline_snapshot_fix": False}):
        agent, runtime = _agent(**settings)
        try:
            result = agent.chat("五粮液的市盈率是多少", session_id=f"r16-l19-ok-{len(settings)}")
        finally:
            runtime.close()
        assert DATA_LAYER_NOTE_ZH not in result["answer"]
        assert not any(item.startswith("data_layer_missing") for item in result["degraded"])


def test_the_note_needs_offline_snapshot_market_or_fundamental_figures():
    snapshot = {
        "kind": "structured",
        "source_type": "fundamental_sql",
        "payload": {"provenance": {"source": "offline_snapshot"}},
    }
    live = {**snapshot, "payload": {"provenance": {"source": "tushare.daily_basic", "is_live": True}}}
    document = {"kind": "document", "source_type": "news", "payload": {}}
    assert data_layer_note([snapshot], ["corrections"], zh=True) == (
        DATA_LAYER_NOTE_ZH,
        ["data_layer_missing:corrections"],
    )
    assert data_layer_note([live, document], ["corrections"], zh=True) == (None, [])
    assert data_layer_note([snapshot], ["extension"], zh=False) == (None, [])
    assert data_layer_note([snapshot], [], zh=False) == (None, [])
