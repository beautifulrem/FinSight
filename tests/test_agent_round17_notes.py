"""Round-17 rule M15 (round-13 review).

M15: an H-share question about a covered company with no Hong Kong line says the company has no H shares (the covered
companies are audited against the HKEX list in ``data/runtime/hong_kong_line_audit.csv``; a company outside the audit
keeps the neutral note, because the dual-listing table does not list every A+H company).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def service():
    from query_intelligence.service import build_default_service

    return build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
        offline_snapshot_fix=True,
    )


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


# ---------------------------------------------------------------- M15 no Hong Kong line


def test_the_audit_agrees_with_the_dual_listing_table():
    from query_intelligence.data_loader import load_dual_listings, load_hong_kong_line_audit

    audit = {row["a_symbol"]: row["h_code"] for row in load_hong_kong_line_audit()}
    assert len(audit) == 13
    dual = {row["a_symbol"]: row["h_code"] for row in load_dual_listings()}
    for symbol, code in audit.items():
        assert dual.get(symbol, "") == code, symbol
    assert {symbol for symbol, code in audit.items() if not code} == {
        "000001.SZ",
        "000858.SZ",
        "600519.SH",
        "600900.SH",
    }


def test_has_hong_kong_line_only_claims_none_for_audited_companies():
    from query_intelligence.agent.coverage import has_hong_kong_line

    assert has_hong_kong_line(["贵州茅台"]) is False
    assert has_hong_kong_line(["五粮液", "长江电力"]) is False
    assert has_hong_kong_line(["美的集团"]) is True
    assert has_hong_kong_line(["贵州茅台", "美的集团"]) is True
    # not audited: the dual-listing table does not list every A+H company, so nothing is claimed
    assert has_hong_kong_line(["立讯精密"]) is None


@pytest.mark.parametrize(
    ("query", "company"),
    [("贵州茅台的港股报价", "贵州茅台"), ("五粮液H股多少钱", "五粮液"), ("平安银行港股收盘价", "平安银行")],
)
def test_an_h_share_question_about_a_company_without_one_says_so(agent, query, company):
    result = _chat(agent, query)
    assert result["route"] == "refuse"
    assert f"{company}没有在香港上市的股份" in result["answer"]


def test_an_h_share_question_about_a_dual_listed_company_keeps_the_note(agent):
    result = _chat(agent, "比亚迪H股报价多少")
    assert result["route"] == "refuse"
    assert "比亚迪在 A 股上市的股份在覆盖范围内" in result["answer"] and "没有在香港上市" not in result["answer"]


def test_the_english_note_says_so_too(agent):
    result = _chat(agent, "What are Kweichow Moutai's H shares trading at?")
    assert result["route"] == "refuse" and "has no shares listed in Hong Kong" in result["answer"]


def test_an_index_picked_up_next_to_a_hong_kong_line_is_not_called_an_a_share():
    from query_intelligence.agent.coverage import a_share_stocks

    assert a_share_stocks(["恒生指数", "贵州茅台"]) == ["贵州茅台"]
