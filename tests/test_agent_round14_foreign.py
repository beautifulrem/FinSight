"""Round-14 rule J7 (round-10 review), written with the engineer's own wording: a company's Hong Kong line, London /
Swiss GDR, US ADR or A/H premium, by any wording, is out of coverage and refused with the note that the company's A
shares are covered; "A股" elsewhere in the question does not switch the check off. A concept question, the A-share
line and A-share funds named after a foreign market stay in scope.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.coverage import foreign_line_kind, foreign_line_spans, out_of_coverage


@pytest.mark.parametrize(
    ("query", "kind"),
    [
        ("比亚迪香港那边收盘多少", "h"),
        ("宁德时代在港交所交易的那部分涨了吗", "h"),
        ("招行H股今天怎么样，A股呢", "h"),
        ("Where is SMIC's HK-traded stock now?", "h"),
        ("How did CMB shares listed in Hong Kong do?", "h"),
        ("华泰证券伦敦GDR报价", "gdr"),
        ("长江电力的全球存托凭证收在多少", "gdr"),
        ("And the London-listed GDRs?", "gdr"),
        ("中国石油美国存托凭证多少钱", "adr"),
        ("What's the NYSE-listed ADR of China Life at?", "adr"),
        ("Sinopec ADS price", "adr"),
        ("工商银行A股相对H股溢价多少", "ah"),
        ("中国平安AH比价是多少", "ah"),
        ("What's the A/H premium on ICBC?", "ah"),
        ("招商银行A+H两地的股价", "h"),
    ],
)
def test_a_foreign_line_by_any_wording_is_out_of_coverage(query, kind):
    assert foreign_line_spans(query)
    assert out_of_coverage(query) == "foreign_equity"
    assert foreign_line_kind(query) == kind


@pytest.mark.parametrize(
    "query",
    [
        "A股和H股有什么区别",
        "What is an ADR?",
        "中国平安A股收盘多少",
        "美股大跌对A股有什么影响",
        "香港的利率政策对A股有什么影响",
        "伦敦金价格多少",
        "嘉实H股指数基金净值",
        "What did BYD's A shares close at?",
    ],
)
def test_concepts_markets_and_a_share_lines_are_not_foreign_lines(query):
    assert not foreign_line_spans(query)
    assert out_of_coverage(query) is None


def test_the_line_span_includes_the_company_name():
    for query, name in [
        ("中芯国际香港那边的股价多少", "中芯国际"),
        ("ICBC ADR price?", "ICBC"),
        ("招商银行A股相对H股溢价", "招商银行"),
    ]:
        (lo, hi), *_ = foreign_line_spans(query)
        assert name in query[lo:hi]


# ---- end to end: refused with the A-share-covered note; the A-share line and funds still answered ----


@pytest.fixture(scope="module")
def ext_agent():
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
    )
    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]))
    agent = AgentService(runtime, trace_sinks=[])
    yield agent
    runtime.close()


@pytest.mark.parametrize(
    ("query", "close", "note"),
    [
        ("比亚迪香港那边收盘多少", "83.31", "所问的是 H 股（香港上市的股份）；比亚迪在 A 股上市的股份在覆盖范围内"),
        ("长江电力伦敦上市的GDR报价多少", "28.54", "所问的是在伦敦或瑞士上市的 GDR（全球存托凭证）；长江电力在 A 股"),
        ("工商银行A股相对H股溢价多少", "8.28", "所问的是 A 股相对 H 股的溢价"),
        ("招商银行H股价格，A股也说一下", "41.26", "招商银行在 A 股上市的股份在覆盖范围内"),
    ],
)
def test_a_chinese_foreign_line_is_refused_with_the_note(ext_agent, query, close, note):
    result = ext_agent.chat(query, session_id=f"r14-foreign-{abs(hash(query))}")
    assert result["route"] == "refuse" and result["limitations"] == ["out_of_coverage"]
    assert close not in result["answer"]
    assert note in result["answer"]


@pytest.mark.parametrize(
    ("query", "close", "note"),
    [
        (
            "Where's BYD's Hong Kong-traded stock?",
            "83.31",
            "the H shares (the Hong Kong line); BYD's A shares are covered",
        ),
        ("ICBC ADR quote?", "8.28", "the ADRs (American depositary receipts listed in the US)"),
    ],
)
def test_an_english_foreign_line_is_refused_with_the_note(ext_agent, query, close, note):
    result = ext_agent.chat(query, session_id=f"r14-foreign-en-{abs(hash(query))}")
    assert result["route"] == "refuse" and close not in result["answer"]
    assert note in result["answer"]


def test_the_a_share_line_and_a_fund_named_after_hong_kong_are_answered(ext_agent):
    result = ext_agent.chat("招商银行A股最新收盘价", session_id="r14-a-line")
    assert result["route"] != "refuse" and "41.26" in result["answer"]
    fund = ext_agent.chat("港股通科技ETF华夏收盘多少", session_id="r14-hk-fund")
    assert fund["route"] != "refuse"


def test_a_follow_up_asking_for_the_london_line_is_refused(ext_agent):
    first = ext_agent.chat("长江电力最新收盘价", session_id="r14-gdr-session")
    assert "28.54" in first["answer"]
    follow = ext_agent.chat("那它在伦敦交易的GDR呢", session_id="r14-gdr-session")
    assert follow["route"] == "refuse" and "28.54" not in follow["answer"]
