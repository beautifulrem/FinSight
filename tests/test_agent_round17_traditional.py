"""Round-17 rule M9 (round-13 review): traditional characters are read as simplified ones before any NLU rule or
model. Before round 17 "長江電力的市淨率是多少" was refused as out of scope, "招商銀行的淨利潤是多少" answered with
the close only and "比亞迪和寧德時代的營收合計" with no total. The character table merges the hand-written one (round
14, claims) with the one-to-one rows of OpenCC's ``TSCharacters.txt`` (Apache-2.0,
``data/runtime/opencc_ts_characters.tsv``, built by ``scripts/build_opencc_table.py``); the NLU pipeline and the
agent convert the question first.
"""

from __future__ import annotations

import itertools

import pytest

from query_intelligence.traditional import has_traditional, to_simplified


@pytest.fixture(scope="module")
def service(shipped_offline_service):
    return shipped_offline_service


@pytest.fixture(scope="module")
def agent(service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]))
    yield AgentService(runtime, trace_sinks=[])
    runtime.close()


def _chat(agent, query: str, session: str | None = None) -> dict:
    return agent.chat(query, session_id=session or f"r17-{abs(hash(query))}")


# ---------------------------------------------------------------- M9 traditional characters


def test_the_opencc_table_is_shipped_with_its_source_and_licence():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    head = (root / "data/runtime/opencc_ts_characters.tsv").read_text(encoding="utf-8").splitlines()[:6]
    assert any("OpenCC ver.1.4.2" in line and "sha256" in line for line in head)
    assert any("Apache License, Version 2.0" in line for line in head)
    assert "Apache License" in (root / "data/runtime/LICENSE.OpenCC").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("traditional", "simplified"),
    [
        ("長江電力的市淨率是多少", "长江电力的市净率是多少"),
        ("寧德時代營業收入", "宁德时代营业收入"),
        ("貴州茅台總市值", "贵州茅台总市值"),
        ("鋰電龍頭贛鋒鋰業", "锂电龙头赣锋锂业"),  # 贛 is only in the OpenCC rows
        ("紫金礦業淨利潤增長", "紫金矿业净利润增长"),
    ],
)
def test_traditional_text_is_converted(traditional, simplified):
    assert has_traditional(traditional)
    assert to_simplified(traditional) == simplified


@pytest.mark.parametrize("text", ["乾照光电的股价", "贵州茅台的市盈率", "What is BYD's P/E?", "ROE 13.24%", ""])
def test_simplified_and_ambiguous_characters_are_kept(text):
    # 乾 is a simplified character too (乾照光电): OpenCC lists it with itself as a candidate, so it is not in the table
    assert to_simplified(text) == text


_TRAD_COMPANIES = {
    "長江電力": "600900.SH",
    "招商銀行": "600036.SH",
    "寧德時代": "300750.SZ",
    "紫金礦業": "601899.SH",
    "貴州茅台": "600519.SH",
    "比亞迪": "002594.SZ",
    "工商銀行": "601398.SH",
    "中國平安": "601318.SH",
}
_TRAD_FRAMES = ("{c}的市淨率是多少", "{c}的淨利潤是多少", "{c}最新收盤價", "{c}營業收入多少", "{c}總市值多少")


@pytest.mark.parametrize(("company", "frame"), list(itertools.product(_TRAD_COMPANIES, _TRAD_FRAMES)))
def test_traditional_questions_resolve_like_simplified_ones(service, company, frame):
    result = service.analyze_query(frame.format(c=company))
    symbols = {entity.get("symbol") for entity in result["entities"]}
    assert _TRAD_COMPANIES[company] in symbols, result["entities"]
    assert "out_of_scope_query" not in result["risk_flags"]
    assert result["raw_query"] == frame.format(c=company)  # what the user wrote is kept
    assert "traditional_to_simplified" in result["explainability"]["matched_rules"]


def test_traditional_questions_are_answered_end_to_end(agent):
    pb = _chat(agent, "長江電力的市淨率是多少")
    assert pb["route"] != "refuse" and "3.23" in pb["answer"]
    profit = _chat(agent, "招商銀行的淨利潤是多少")
    assert "1501.81" in profit["answer"]
    total = _chat(agent, "比亞迪和寧德時代的營收合計")
    assert "12276.67" in total["answer"]
