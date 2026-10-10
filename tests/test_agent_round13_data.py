"""Round-13 data items of the round-9 review, written with the engineer's own wording.

* I11: a year-to-date change is measured from the previous year's last close (the 年初至今 basis of quote vendors),
  not from the year's first close; a payload recorded before round 13 (no ``prior_year_end``) keeps the first close
  and says that this is its base.
* I10: a question about a company's industry average, when the data has no industry snapshot for the company (the
  2026-09-30 snapshot extension carries none), says the industry average is missing.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.tools.market import max_drawdown_ytd, prior_year_end_close, year_start_close


def _history(closes: list[tuple[str, float]]) -> list[dict]:
    return [{"trade_date": day, "close": close} for day, close in reversed(closes)]  # providers: latest first


def test_the_prior_year_end_close_is_the_last_close_before_the_latest_year():
    history = _history([("2025-12-29", 9.0), ("2025-12-31", 10.0), ("2026-01-05", 12.0), ("2026-03-02", 11.0)])
    base = prior_year_end_close(history)
    assert base is not None and (base.date, base.close) == ("2025-12-31", 10.0)
    first = year_start_close(history)
    assert first is not None and (first.date, first.close) == ("2026-01-05", 12.0)
    # the drawdown window is unchanged: from the year's first close
    drawdown = max_drawdown_ytd(history)
    assert drawdown is not None and drawdown.start_date == "2026-01-05"


def test_no_prior_year_end_without_a_close_from_the_year_before():
    assert prior_year_end_close(_history([("2026-01-05", 12.0), ("2026-03-02", 11.0)])) is None
    assert prior_year_end_close([]) is None


def _ytd(data: dict, zh: bool) -> list[str]:
    from query_intelligence.agent.composer import _year_to_date

    return _year_to_date(data, zh)


_PAYLOAD = {
    "name": "宁德时代",
    "evidence_id": "price_300750.SZ",
    "as_of": "2026-09-30",
    "close": 291.11,
    "product_type": "stock",
    "year_start": {"date": "2026-01-05", "close": 377.5},
}


def test_the_year_to_date_sentence_uses_the_prior_year_end_close():
    data = {**_PAYLOAD, "prior_year_end": {"date": "2025-12-31", "close": 367.26}}
    (zh,) = _ytd(data, True)
    assert "-20.73%" in zh and "2025-12-31" in zh and "367.26" in zh and "上年最后一个交易日" in zh
    assert "377.5" not in zh
    (en,) = _ytd(data, False)
    assert "-20.73%" in en and "previous year's last close" in en


def test_a_payload_without_the_prior_year_end_keeps_the_labelled_first_close():
    (zh,) = _ytd(_PAYLOAD, True)
    assert "今年首个交易日（2026-01-05）" in zh and "-22.88%" in zh
    (en,) = _ytd(_PAYLOAD, False)
    assert "first close of the year (2026-01-05)" in en


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


def test_a_year_to_date_question_on_an_extension_name_is_measured_from_the_prior_year_end(ext_agent):
    result = ext_agent.chat("宁德时代年初至今涨跌多少", session_id="r13-ytd-zh")
    assert "-20.73%" in result["answer"] and "2025-12-31" in result["answer"]
    assert "-22.88" not in result["answer"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("美的集团的市净率和所在行业平均比怎么样", "当前数据没有美的集团所属行业的平均市净率"),
        ("工商银行ROE跟行业平均相比高还是低", "当前数据没有工商银行所属行业的平均ROE"),
        (
            "Is Zijin Mining's P/E above its industry average?",
            "The industry average P/E of Zijin Mining's industry is not available",
        ),
    ],
)
def test_a_missing_industry_snapshot_is_stated(ext_agent, query, expected):
    result = ext_agent.chat(query, session_id=f"r13-industry-{abs(hash(query))}")
    assert expected in result["answer"], result["answer"]


def test_a_company_with_an_industry_snapshot_gets_it_and_no_missing_statement(ext_agent):
    result = ext_agent.chat("五粮液的市盈率比行业平均高还是低", session_id="r13-industry-v1")
    assert "所属行业 白酒" in result["answer"] and "所属行业的平均" not in result["answer"]


@pytest.mark.parametrize(
    "query",
    [
        "从今年首个交易日的收盘算起，宁德时代涨跌幅是多少",
        "按年内第一个交易日收盘计，宁德时代今年以来表现如何",
        "How has CATL done since the first close of the year?",
    ],
)
def test_a_named_first_close_base_is_kept_for_year_to_date(query):
    """(round 14) A user who names the year's first close as the base gets that base, not the prior year-end close
    (round 13 made the prior year-end close the default, which overrode an explicit request)."""
    from query_intelligence.agent.composer import _year_to_date
    from query_intelligence.agent.coverage import requested_price_fields

    request = requested_price_fields(query)
    assert request.ytd_first_close
    data = {**_PAYLOAD, "prior_year_end": {"date": "2025-12-31", "close": 367.26}}
    (sentence,) = _year_to_date(data, "CATL" not in query, first_close_base=request.ytd_first_close)
    assert "-22.88%" in sentence and "2026-01-05" in sentence and "367.26" not in sentence


def test_a_plain_year_to_date_question_keeps_the_prior_year_end_base():
    from query_intelligence.agent.coverage import requested_price_fields

    assert not requested_price_fields("宁德时代今年以来涨了多少").ytd_first_close
