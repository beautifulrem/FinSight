"""Round 18 (review 14, N11 and N12): text read the same way before any NLU rule, the agent or the claim checker.

* N11: Hong Kong / Taiwan financial vocabulary (純利, 股東應佔溢利, 股本回報率, 按年, 市賬率, 本益比, 營業額, 每股盈利,
  收市價 ...) is rewritten to the mainland term by a small curated table (``REGIONAL_TERMS`` in
  ``query_intelligence/traditional.py``, each row with its region and a note). Before round 18 only the characters
  were converted, so "股东应占溢利" or "市账率" named no metric (chat: close only; claims: unverifiable).
* N12: full-width Latin letters and digits (ＢＹＤ, ＰＥ, ２０２５) are read as ASCII; full-width Chinese punctuation
  (，。；：（）？！) is kept, because the parsers split clauses on it.

The wordings and company pairings are the engineer's own (the guard in tests/test_agent_eval.py checks that none
copies a slice, a real claim or a reviewer probe). Values: the shipped default data (v1 + the 2026-09-30 extension +
the corrections layer), FY2025 report; see the table in tests/test_agent_round17_claims.py.
"""

from __future__ import annotations

import pytest

from query_intelligence.traditional import REGIONAL_TERMS, fold_width, normalise_text


# ---------------------------------------------------------------- N12 width
@pytest.mark.parametrize(
    ("written", "read"),
    [
        ("ＢＹＤ的最新股价", "BYD的最新股价"),
        ("五粮液的ＰＢ是多少", "五粮液的PB是多少"),
        ("Ｍｉｄｅａ的ＲＯＥ", "Midea的ROE"),
        ("２０２５年营收３．２倍", "2025年营收3.2倍"),
        ("净利率１７％", "净利率17%"),
        ("ＣＭＢ的Ｐ／Ｅ", "CMB的P/E"),
    ],
)
def test_full_width_letters_and_digits_are_read_as_ascii(written, read):
    assert fold_width(written) == read


@pytest.mark.parametrize(
    "text",
    [
        "茅台市盈率21倍，五粮液呢？",  # the clause comma and the question mark stay full-width
        "营收（不含利息收入）增长；净利润下降：",
        "涨了！",
        "比亚迪．",  # a full stop-like dot not between digits is kept
        "①茅台②五粮液",  # circled numbers are list markers, not digits
    ],
)
def test_chinese_punctuation_is_kept(text):
    assert fold_width(text) == text


# ---------------------------------------------------------------- N11 vocabulary
@pytest.mark.parametrize(
    ("written", "read"),
    [
        ("比亞迪股東應佔溢利", "比亚迪归母净利润"),
        ("本公司擁有人應佔溢利", "归母净利润"),
        ("長江電力股本回報率", "长江电力净资产收益率"),
        ("工商銀行市賬率", "工商银行市净率"),
        ("美的集團本益比", "美的集团市盈率"),
        ("比亞迪營業額", "比亚迪营业收入"),
        ("中芯國際每股盈利", "中芯国际每股收益"),
        ("招商銀行收市價", "招商银行收盘价"),
        ("紫金礦業純利按年上升62%", "紫金矿业净利润同比上升62%"),
        ("五粮液纯利按年下跌七成", "五粮液净利润同比下跌七成"),  # simplified HK press wording too
        ("台股股價淨值比", "台股市净率"),
        ("營收按季增長", "营收环比增长"),
    ],
)
def test_regional_vocabulary_is_rewritten(written, read):
    assert normalise_text(written) == read


@pytest.mark.parametrize(
    ("text", "read"),
    [
        ("按年计算的收益率", "按年计算的收益率"),  # 按年 not followed by a change word: not a year-on-year
        ("按季度披露", "按季度披露"),
        ("贵州茅台的市盈率", "贵州茅台的市盈率"),
        ("纯利润和毛利润", "净利润和毛利润"),  # colloquial 纯利润 is the net profit, never "净利润润"
        ("纯利率超过一成", "净利率超过一成"),
    ],
)
def test_mainland_text_keeps_its_meaning(text, read):
    assert normalise_text(text) == read


def test_every_regional_term_states_its_region_and_a_note():
    assert len(REGIONAL_TERMS) >= 10
    for row in REGIONAL_TERMS:
        assert row.region in {"HK", "TW", "HK/TW"}, row
        assert row.note and row.term and row.pattern.pattern, row


# ---------------------------------------------------------------- NLU and chat
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


@pytest.mark.parametrize(
    ("query", "symbol"),
    [
        ("ＢＹＤ最新的股价是多少", "002594.SZ"),
        ("ＣＭＢ的Ｐ／Ｅ", "600036.SH"),
        ("比亞迪股東應佔溢利有多少", "002594.SZ"),
        ("長江電力股本回報率是多少", "600900.SH"),
    ],
)
def test_the_nlu_reads_width_and_vocabulary(service, query, symbol):
    result = service.analyze_query(query)
    assert symbol in {entity.get("symbol") for entity in result["entities"]}, result["entities"]
    assert result["raw_query"] == query
    assert "out_of_scope_query" not in result["risk_flags"]


@pytest.mark.parametrize(
    ("query", "number"),
    [
        ("比亞迪股東應佔溢利有多少", "326.19"),
        ("長江電力股本回報率是多少", "15.59"),
        ("工商銀行市賬率幾多", "0.74"),
        ("美的集團本益比是多少", "13.77"),
        ("五粮液的ＰＢ是多少", "3.26"),
        ("中芯国际的ＲＯＥ", "3.34"),
    ],
)
def test_chat_answers_regional_and_full_width_questions(agent, query, number):
    response = agent.chat(query, session_id=f"r18-{abs(hash(query))}")
    assert response["route"] != "refuse" and number in response["answer"], response["answer"]


# ---------------------------------------------------------------- claims
@pytest.fixture(scope="module")
def checker(service):
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    registry = build_registry_for_service(service)
    return lambda claim: check_claim(claim, service=service, registry=registry, zh=not claim.isascii())


@pytest.mark.parametrize(
    ("claim", "verdict", "metrics"),
    [
        ("美的集團去年純利約439億元，按年上升近15%", "supported", ["net_profit", "netprofit_yoy"]),
        ("比亞迪股東應佔溢利326億元人民幣", "supported", ["net_profit"]),
        ("紫金礦業股本回報率接近28%", "supported", ["roe"]),
        ("工商銀行市賬率僅0.74倍", "supported", ["pb"]),
        ("五糧液本益比低於20倍", "contradicted", ["pe_ttm"]),
        ("比亞迪營業額逾8000億元", "supported", ["revenue"]),
        ("中芯國際純利按年下跌", "contradicted", ["netprofit_yoy"]),
        ("ＢＹＤ的ＰＥ约25.8倍", "supported", ["pe_ttm"]),
        ("Ｍｏｕｔａｉ's ROE is about 32.5%", "supported", ["roe"]),
    ],
)
def test_claims_in_regional_vocabulary_or_full_width_are_checked(checker, claim, verdict, metrics):
    report = checker(claim)
    assert report.verdict == verdict and [check.metric for check in report.checks] == metrics, report.model_dump()
    assert report.claim == claim  # the report keeps the claim as written
