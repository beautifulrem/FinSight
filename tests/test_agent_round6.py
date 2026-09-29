"""Round-6 rules, written after the independent round-4 held-out slices were run once (evaluation/heldout_r4).

Each test uses new wording for a failure class of those slices; none repeats a held-out claim or question.
Claim values come from the offline snapshot: 贵州茅台 -0.1778% / 1409.5, 五粮液 -0.5337% / 100.64, 中国平安
+0.73% / 53.61 (2026-04-22); FY2025 P/B 8.1 / 5.4 / 1.1; industry 白酒 -1.05% (2026-04-21), 保险 +0.68%.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.claim_check import check_claim, normalise
from query_intelligence.agent.tools.defaults import build_registry_for_service


@pytest.fixture(scope="module")
def registry(offline_service):
    return build_registry_for_service(offline_service)


def _check(claim: str, offline_service, registry):
    return check_claim(claim, service=offline_service, registry=registry, zh=not claim.isascii())


def _summary(report) -> list[tuple]:
    return [(check.target, check.metric, check.comparator, check.status) for check in report.checks]


# --- class 1: bounded moves ("跌了不到X%", "fell less than half a percent") -------------------------------------
@pytest.mark.parametrize(
    ("claim", "status"),
    [
        ("五粮液上一个交易日跌了不到0.8%", "supported"),  # a fall of 0.53
        ("茅台上一个交易日跌幅不足0.1%", "contradicted"),  # a fall of 0.18
        ("中国平安上一个交易日跌了不到2%", "contradicted"),  # it rose: not a small fall
        ("Wuliangye lost less than half a percent in the latest session", "contradicted"),
        ("Moutai rose by less than two percent in the latest session", "contradicted"),  # it fell
        ("Ping An rose by less than two percent in the latest session", "supported"),
    ],
)
def test_bounded_moves_are_checked_not_unverifiable(claim, status, offline_service, registry):
    report = _check(claim, offline_service, registry)

    assert [check.status for check in report.checks] == [status]
    assert report.checks[0].metric == "pct_change_1d" and report.checks[0].comparator == "lt"


def test_english_fractions_and_number_words_become_numbers():
    assert "0.5 percent" in normalise("fell less than half a percent")
    assert "0.25 percent" in normalise("slipped by a quarter of a percent")
    assert "3 times" in normalise("three times Wuliangye's")
    assert "2 times" in normalise("twice Moutai's")
    assert "twice a year" in normalise("twice a year")


# --- class 2: qualitative move words (a documented convention) --------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "status", "comparator", "threshold"),
    [
        ("五粮液上一个交易日重挫", "contradicted", "ge", "3%"),
        ("中国平安上一个交易日小幅上涨", "supported", "lt", "1%"),
        ("茅台上一个交易日微跌", "supported", "lt", "1%"),
        ("Ping An plunged in the latest session", "contradicted", "ge", "3%"),
        ("五粮液上一个交易日没有暴跌", "supported", "lt", "3%"),  # negated: not a fall of 3% or more
    ],
)
def test_qualitative_moves_use_a_stated_threshold(claim, status, comparator, threshold, offline_service, registry):
    check = _check(claim, offline_service, registry).checks[0]

    assert (check.metric, check.comparator, check.status) == ("pct_change_1d", comparator, status)
    assert "convention" in check.note and threshold in check.note


# --- class 3: explicit dates -------------------------------------------------------------------------------------
def test_a_date_matching_the_trade_date_is_not_a_number_or_a_downgrade(offline_service, registry):
    report = _check("Moutai closed at 1409.5 yuan on Apr 22nd, 2026", offline_service, registry)

    assert _summary(report) == [("贵州茅台", "close", "eq", "supported")]
    assert report.checks[0].as_of == "2026-04-22"


@pytest.mark.parametrize(
    "claim", ["茅台4月20日收于1409.5元", "Moutai fell on 20 April", "2025年4月22日茅台收于1409.5元"]
)
def test_a_date_other_than_the_trade_date_is_a_period_mismatch(claim, offline_service, registry):
    check = _check(claim, offline_service, registry).checks[0]

    assert (check.status, check.reason) == ("unverifiable", "period_mismatch")


def test_latest_session_wording_is_not_a_multi_day_move(offline_service, registry):
    for claim in ("五粮液最近一个交易日收跌", "五粮液近1个交易日收跌"):
        assert _summary(_check(claim, offline_service, registry)) == [("五粮液", "pct_change_1d", "lt", "supported")]
    assert _check("五粮液近5个交易日下跌", offline_service, registry).checks[0].reason == "multi_day"


def test_may_as_a_verb_is_not_a_date(offline_service, registry):
    report = _check("Moutai may trade at 30 times earnings", offline_service, registry)

    assert report.checks[0].claimed == 30.0 and report.checks[0].reason == "forecast"


# --- class 4: multiples, relations, sectors, macro thresholds ------------------------------------------------------
def test_a_multiple_of_another_target_compares_the_ratio(offline_service, registry):
    check = _check("五粮液的市净率大概是中国平安的5倍", offline_service, registry).checks[0]

    assert (check.target, check.metric, check.reference, check.comparator) == ("五粮液", "pb", "中国平安", "approx")
    assert check.claimed == 5.0 and check.ratio == pytest.approx(5.4 / 1.1, rel=1e-3)
    assert check.status == "supported" and check.reference_value == 1.1
    english = _check("Wuliangye's P/B is roughly twice Moutai's", offline_service, registry).checks[0]
    assert (english.target, english.reference, english.status) == ("五粮液", "贵州茅台", "contradicted")


def test_a_multiple_of_moves_needs_both_moves_in_the_stated_direction(offline_service, registry):
    check = _check("中国平安的跌幅是茅台的4倍", offline_service, registry).checks[0]

    assert check.status == "contradicted" and "direction" in check.note  # Ping An rose


def test_outperformance_compares_daily_moves(offline_service, registry):
    report = _check("中国平安上一个交易日跑输沪深300", offline_service, registry)

    assert _summary(report) == [("中国平安", "pct_change_1d", "lt", "contradicted")]  # +0.73 vs +0.42


def test_relational_claims_on_amounts(offline_service, registry):
    report = _check("五粮液的净利润高于中国平安", offline_service, registry)

    assert _summary(report) == [("五粮液", "net_profit", "gt", "contradicted")]


def test_company_against_a_named_sector(offline_service, registry):
    report = _check("茅台的市净率高于白酒板块", offline_service, registry)

    assert _summary(report) == [("贵州茅台", "pb", "gt", "supported")]  # 8.1 vs 6.2
    assert report.checks[0].reference == "白酒行业" and report.checks[0].reference_value == 6.2


def test_sector_moves_use_the_industry_snapshot_and_its_date(offline_service, registry):
    insurance = _check("保险板块上一个交易日收涨", offline_service, registry).checks[0]
    assert (insurance.target, insurance.status, insurance.actual) == ("保险", "supported", 0.68)
    assert insurance.as_of_basis == "trade_date"
    baijiu = _check("白酒板块4月22日跌了", offline_service, registry).checks[0]
    assert (baijiu.status, baijiu.reason) == ("unverifiable", "period_mismatch")  # the snapshot is 2026-04-21


def test_a_sector_word_describing_a_company_is_not_a_target(offline_service, registry):
    report = _check("白酒龙头茅台市净率8.1倍", offline_service, registry)

    assert _summary(report) == [("贵州茅台", "pb", "eq", "supported")]


def test_the_pmi_line_is_fifty(offline_service, registry):
    check = _check("PMI已经回到荣枯线之上", offline_service, registry).checks[0]

    assert (check.metric, check.comparator, check.claimed, check.status) == ("pmi", "gt", 50.0, "supported")


def test_is_higher_than_is_not_read_as_a_move(offline_service, registry):
    report = _check("Moutai's P/B is higher than Ping An's", offline_service, registry)

    assert _summary(report) == [("贵州茅台", "pb", "gt", "supported")]


# --- class 5: several targets sharing one claim -----------------------------------------------------------------
def test_shared_claims_get_one_check_per_target(offline_service, registry):
    report = _check("五粮液和中国平安上一个交易日都涨了", offline_service, registry)

    assert _summary(report) == [
        ("五粮液", "pct_change_1d", "gt", "contradicted"),
        ("中国平安", "pct_change_1d", "gt", "supported"),
    ]
    assert report.verdict == "partially_supported"


def test_a_list_without_a_shared_word_keeps_the_nearest_target(offline_service, registry):
    report = _check("茅台和五粮液里，五粮液市净率5.4倍", offline_service, registry)

    assert _summary(report) == [("五粮液", "pb", "eq", "supported")]


# --- class 6: comparison follow-ups keep every earlier target ----------------------------------------------------
def _turn(query: str, *entities: tuple[str, str], named: bool = True) -> dict:
    return {
        "query": query,
        "effective_query": query,
        "entities": [{"name": name, "symbol": symbol} for name, symbol in entities],
        "named": named,
    }


MOUTAI, WULIANGYE, PINGAN = ("贵州茅台", "600519.SH"), ("五粮液", "000858.SZ"), ("中国平安", "601318.SH")


@pytest.mark.parametrize(
    "query", ["Stack it side by side with Wuliangye", "Line that up next to Wuliangye", "把它跟五粮液放在一起看"]
)
def test_comparison_verbs_anchor_the_earlier_target(query):
    from query_intelligence.agent.memory import resolve_comparison_anchor

    turns = [_turn("茅台的市净率", MOUTAI)]
    rewritten, reason = resolve_comparison_anchor(query, turns, [{"canonical_name": "五粮液", "symbol": "000858.SZ"}])
    assert reason == "comparison_anchor:+贵州茅台" and "贵州茅台" in rewritten


def test_group_pronouns_keep_the_whole_last_group_and_dual_words_keep_two():
    from query_intelligence.agent.memory import has_plural_reference, resolve_coreference

    turns = [_turn("三家的市盈率", WULIANGYE, PINGAN, MOUTAI)]
    rewritten, _ = resolve_coreference("And the lowest P/B among those?", turns)
    assert all(name in rewritten for name in ("五粮液", "中国平安", "贵州茅台"))
    rewritten, _ = resolve_coreference("它们谁的ROE最高", turns)
    assert all(name in rewritten for name in ("五粮液", "中国平安", "贵州茅台"))
    rewritten, _ = resolve_coreference("这两家谁更便宜", [_turn("茅台", MOUTAI), _turn("平安", PINGAN)])
    assert rewritten == "贵州茅台和中国平安谁更便宜"
    assert not has_plural_reference("How did the market do these days?")


@pytest.fixture
def agent(offline_service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService

    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), ScriptedLLM([]))
    service = AgentService(runtime, trace_sinks=[])
    yield service
    runtime.close()


def _targets(result: dict) -> set[str]:
    return {call["arguments"].get("target") for call in result.get("tool_calls") or []}


def test_a_count_above_the_discussed_targets_is_clarified_and_the_reply_joins_them(agent):
    session = "r6-three-of-two"
    agent.chat("五粮液跟茅台的ROE各多少", session_id=session)
    asked = agent.chat("这三家谁的市净率最高", session_id=session)
    assert asked.get("status") == "needs_clarification"
    assert "只讨论过五粮液和贵州茅台" in asked["clarification"]["question"]
    answered = agent.chat("中国平安", session_id=session)
    assert {"000858.SZ", "600519.SH", "601318.SH"} <= _targets(answered)


# --- class 7: English typos of company names -----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "symbol"),
    [("What's the ROE of Wulaingye?", "000858.SZ"), ("Kweichow Mouati closing price", "600519.SH")],
)
def test_english_typos_resolve(query, symbol, offline_service):
    entities = offline_service.analyze_query(query)["entities"]
    assert symbol in {entity.get("symbol") for entity in entities}


@pytest.mark.parametrize(
    "query",
    [
        "Why are insured deposits rising?",  # "insured" vs the alias "insurers"
        "Is the mountain region economy growing?",
        "Industrial banks lend to factories",  # a plural of the alias word is not a typo
        "The Cambrian explosion and ping pong",
        "Sinologists study Chinese history",
    ],
)
def test_english_words_are_not_read_as_typod_names(query, offline_service):
    corrected, trace = offline_service.nlu_pipeline.normalizer._correct_english_typos(query)

    assert (corrected, trace) == (query, [])


def test_dictionary_words_rarely_look_like_typod_names(offline_service):
    """Every word of the system dictionary (when present) through the typo corrector: only a handful of rare words
    are one edit from a security's English name."""
    from pathlib import Path

    words_file = Path("/usr/share/dict/words")
    if not words_file.exists():
        pytest.skip("no system word list")
    normalizer = offline_service.nlu_pipeline.normalizer
    words = sorted({line.strip().lower() for line in words_file.read_text().splitlines() if line.strip().isalpha()})
    hits = [word for word in words if normalizer._correct_english_typos(word)[1]]
    assert len(hits) <= 5, hits


# --- class 8: a persisted answer language --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "requested", "persistent"),
    [
        ("接下来都用英文回答", "en", "en"),
        ("继续用英文，市盈率呢", "en", "en"),
        ("Keep answering in Chinese", "zh", "zh"),
        ("From now on, reply in English", "en", "en"),
        ("请用英文回答：平安的ROE", "en", None),  # one-off
        ("市盈率的英文说法是什么", None, None),
    ],
)
def test_persistent_language_instructions(text, requested, persistent):
    from query_intelligence.chat.language import persistent_answer_language, requested_answer_language

    assert (requested_answer_language(text), persistent_answer_language(text)) == (requested, persistent)


def test_a_persisted_language_holds_until_a_new_instruction(agent):
    session = "r6-language"
    first = agent.chat("茅台的市盈率？以后都用英文回答", session_id=session)
    later = agent.chat("那它的市净率呢", session_id=session)
    one_off = agent.chat("用中文说一下它的ROE", session_id=session)
    after = agent.chat("它的净利润呢", session_id=session)
    assert [detect(item) for item in (first, later, one_off, after)] == ["en", "en", "zh", "en"]


def detect(result: dict) -> str:
    from query_intelligence.chat.language import detect_query_language

    return detect_query_language(str(result.get("answer") or ""))


# --- class 9: holdings and fund flows ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "stated"),
    [
        ("社保基金最近有没有买入茅台", True),
        ("Are foreign investors dumping Wuliangye?", True),
        ("茅台最近主力资金流入吗", True),
        ("机构给了茅台买入评级吗", False),  # a rating, not holdings
        ("茅台大股东最近增持了吗", False),  # disclosed in announcements
        ("北向资金是什么意思", False),  # a definition
    ],
)
def test_flow_questions_state_the_missing_data(query, stated):
    from query_intelligence.agent.coverage import flow_gaps

    assert bool(flow_gaps(query, [], zh=not query.isascii())) is stated


def test_a_flow_question_about_a_stock_says_so_in_the_answer(agent):
    result = agent.chat("险资这阵子在加仓五粮液吗", session_id="r6-flows")
    assert "没有险资的持仓或资金流向数据" in str(result["answer"])
    assert "000858.SZ" in _targets(result)
