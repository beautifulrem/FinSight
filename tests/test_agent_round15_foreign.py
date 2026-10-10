"""Round-15 rule K5 (round-11 review), written with the engineer's own wording: a company's Hong Kong line asked by
its stock code ("(01398)", "02318", "3968" right after 招行, "00981.HK" right after Chinese text), by its venue
without 上市/挂牌 ("联交所的…", "…在港交所的成交量") or by its currency ("港币计价", "in HKD") is out of coverage and
refused with the note that the company's A shares are covered. A six-digit A-share code, share counts, prices,
years and currency exchange questions stay in scope.

The property test crosses every code, venue and currency template with A+H names (zh and en), so a new template or
name is checked against every counterpart.
"""

from __future__ import annotations

import itertools

import pytest

from query_intelligence.agent.coverage import (
    _AH_CODES,
    foreign_line_kind,
    foreign_line_spans,
    out_of_coverage,
)

# (zh name, en name, Hong Kong code) for A+H companies; the code is the one in ``_AH_CODES`` or a leading-zero code
_DUAL = [
    ("中国平安", "Ping An", "2318"),
    ("招商银行", "China Merchants Bank", "3968"),
    ("工商银行", "ICBC", "1398"),
    ("比亚迪", "BYD", "1211"),
    ("宁德时代", "CATL", "3750"),
    ("紫金矿业", "Zijin Mining", "2899"),
    ("中信证券", "CITIC Sec", "6030"),
    ("中芯国际", "SMIC", "0981"),
    ("建设银行", "CCB", "0939"),
    ("中国石化", "Sinopec", "0386"),
]
_ZH_CODE = [
    "{n}（0{c}）最新价是多少",
    "{n}(0{c})收在哪里",
    "{n}0{c}的股价",
    "{n} 0{c} 今天涨了吗",
    "{n}{c}.HK最近走势",
    "{n}（{c}.HK）的市值",
    "{n}{c}现在多少钱",
    "{n}的{c}收盘",
]
_ZH_VENUE = [
    "联交所的{n}今天怎么样",
    "港交所上的{n}收盘多少",
    "香港交易所的{n}最新价",
    "{n}在联交所的成交量",
    "{n}在香港的收盘价",
    "{n}于港交所的市净率",
]
_ZH_CURRENCY = [
    "{n}用港币计价的收盘价",
    "{n}港元股价",
    "{n}的HKD报价",
    "{n}收盘是多少港元",
    "{n}现在值多少港币",
]
_EN = [
    "Where did {e} close in Hong Kong dollars?",
    "How much is {e} quoted at in HKD?",
    "{e}'s HKD share price?",
    "What is {e} (0{c}) at now?",
    "What is {e} trading at in HK dollars today?",
]


def _cases() -> list[tuple[str, str]]:
    cases = []
    for (zh, _en, code), template in itertools.product(_DUAL, _ZH_CODE + _ZH_VENUE + _ZH_CURRENCY):
        if code.startswith("0") and "0{c}" in template:
            continue  # a leading-zero code is written once
        cases.append((template.format(n=zh, c=code), zh))
    for (_zh, en, code), template in itertools.product(_DUAL, _EN):
        if code.startswith("0") and "0{c}" in template:
            continue
        cases.append((template.format(e=en, c=code), en))
    return cases


def _name_in_or_before_a_span(query: str, name: str) -> bool:
    """The name is inside a line span, or right before one (only 的 or spaces between: the agent joins them)."""
    start = query.find(name)
    end = start + len(name)
    return any((lo < end and start < hi) or query[end:lo].strip() in {"", "的"} for lo, hi in foreign_line_spans(query))


@pytest.mark.parametrize(("query", "name"), _cases())
def test_a_hong_kong_line_by_code_venue_or_currency_is_out_of_coverage(query, name):
    assert out_of_coverage(query) == "foreign_equity"
    assert foreign_line_kind(query) == "h"
    assert _name_in_or_before_a_span(query, name)


def test_the_property_set_covers_every_template_and_name():
    cases = _cases()
    assert len(cases) >= 220
    assert {name for _, name in cases} >= {zh for zh, _, _ in _DUAL} | {en for _, en, _ in _DUAL}


@pytest.mark.parametrize(
    "query",
    [
        "工商银行601398今天收盘多少",  # the A-share code
        "中芯国际(688981)的市盈率",
        "比亚迪1000股现在市值多少",  # a share count
        "招行40元附近能买吗",  # a price
        "中国平安1500手",
        "宁德时代2025年报净利润",  # a year
        "比亚迪（2025）营收",
        "茅台1880附近还能买吗",  # an A+H code after another company
        "100港元能换多少人民币",  # currency exchange
        "港元兑人民币汇率是多少",
        "How much is 200 HKD in RMB?",
        "港交所的新规对A股有什么影响",  # the venue as a regulator
        "香港市场的估值比A股低吗",
        "招行3968点",  # not a code
        "What did ICBC's A shares close at?",
    ],
)
def test_a_share_codes_counts_prices_years_and_fx_are_not_hong_kong_lines(query):
    assert not foreign_line_spans(query)
    assert out_of_coverage(query) is None


def test_bare_codes_need_their_own_company():
    assert set(_AH_CODES) >= {"2318", "3968", "1398", "1211", "3750"}
    assert all(code.isdigit() and len(code) == 4 and not code.startswith("0") for code in _AH_CODES)
    assert foreign_line_spans("平安2318收盘")
    assert not foreign_line_spans("招行2318收盘")  # 2318 is Ping An's code, not CMB's


# ---- end to end: refused with the A-share-covered note; the A-share code still answered ----


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
        ("比亚迪（01211）收在哪里", "83.31", "比亚迪在 A 股上市的股份在覆盖范围内"),
        ("招商银行03968最新价", "41.26", "招商银行在 A 股上市的股份在覆盖范围内"),
        ("联交所的招行收盘", "41.26", "所问的是 H 股（香港上市的股份）"),
        ("工商银行港币计价的收盘价", "8.28", "工商银行在 A 股上市的股份在覆盖范围内"),
    ],
)
def test_a_chinese_hong_kong_line_by_code_venue_or_currency_is_refused(ext_agent, query, close, note):
    result = ext_agent.chat(query, session_id=f"r15-hk-{abs(hash(query))}")
    assert result["route"] == "refuse" and result["limitations"] == ["out_of_coverage"]
    assert close not in result["answer"]
    assert note in result["answer"]


def test_an_english_hong_kong_dollar_question_is_refused(ext_agent):
    result = ext_agent.chat("What is BYD trading at in HKD right now?", session_id="r15-hk-en")
    assert result["route"] == "refuse" and "83.31" not in result["answer"]
    assert "the H shares (the Hong Kong line); BYD's A shares are covered" in result["answer"]


def test_the_a_share_code_is_answered(ext_agent):
    result = ext_agent.chat("招商银行600036最新收盘价", session_id="r15-a-code")
    assert result["route"] != "refuse" and "41.26" in result["answer"]
