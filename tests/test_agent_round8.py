"""Round-8 rules, written from the round-4 review (D5-D8), with new wording (none repeats a reviewer probe).

* D5: fair-value / "what is it worth" questions are judgment: hedged, with a limitation, never one number as the value.
* D6: crypto funds are out of coverage (never fuzzy-matched to an A-share ETF); one policy for the short name 平安.
* D7: net margin is derived from the cited fundamentals; PEG and year-to-date returns are derived only when the
  evidence has what they need, and otherwise stated as unavailable.
* D8: the "no single cause" caveat only on causal (why) questions.

Offline snapshot values: 贵州茅台 revenue 1688.38 亿 / net profit 823.2 亿 (FY2025), 五粮液 1085 亿 / 378 亿,
中国平安 PE 8.7; 沪深300 has two closes (2026-04-21, 2026-04-22).
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.tools.defaults import build_registry_for_service


@pytest.fixture(scope="module")
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


def _entities(result: dict) -> set[str]:
    return {entity.get("symbol") for entity in (result.get("nlu_summary") or {}).get("entities") or []}


# --- D5: fair value / "what is it worth" is a judgment -------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "五粮液一股合理价位大概在哪",
        "按现在的业绩，平安估值应该是多少才合理",
        "你觉得茅台的内在价值有多少",
        "五粮液到底值多少钱",
        "How much is Wuliangye really worth?",
        "What's a fair value for Ping An shares?",
        "Estimate the intrinsic value of Moutai",
    ],
)
def test_fair_value_questions_are_judgments(query):
    from query_intelligence.agent.router import FAIR_VALUE_MARKERS, decide_route

    assert FAIR_VALUE_MARKERS.search(query)
    nlu = {"entities": [{"symbol": "600519.SH", "entity_type": "stock"}], "question_style": "fact"}
    assert "lexical:judgment_or_timing" in decide_route(nlu, query=query).reasons


@pytest.mark.parametrize(
    "query",
    [
        "五粮液现在多少钱一股",  # a price, not a value judgment
        "茅台总市值多少钱",
        "这只基金净值多少钱",
        "公允价值变动收益是多少",
        "Is Moutai worth buying?",  # already a judgment through "worth buying", not a fair-value request
        "茅台的估值多少倍",
    ],
)
def test_prices_and_reported_values_are_not_fair_value_requests(query):
    from query_intelligence.agent.router import FAIR_VALUE_MARKERS

    assert not FAIR_VALUE_MARKERS.search(query)


def test_a_fair_value_answer_is_hedged_with_a_limitation_and_no_value_of_its_own(agent):
    result = agent.chat("五粮液一股合理价位大概在哪", session_id="r8-fair-value")
    answer = str(result["answer"])
    assert {"conditional_prefix", "fair_value_hedge"} <= set(result["compliance_notes"])
    assert answer.startswith("基于当前证据只能做条件性判断")
    assert "不给出合理估值" in answer
    assert any("合理估值" in item for item in result["limitations"])
    assert "000858.SZ" in _targets(result)


def test_an_english_worth_question_is_hedged(agent):
    result = agent.chat("How much is Wuliangye really worth?", session_id="r8-worth-en")
    assert "fair_value_hedge" in result["compliance_notes"]
    assert "does not give a fair value" in str(result["answer"])


@pytest.mark.parametrize(
    "sentence",
    [
        "综合来看，茅台的合理估值约为1500元。",
        "我们测算每股内在价值在1320元左右。",
        "五粮液合理股价区间应在120元附近。",
        "Its fair value is about CNY 1,500 per share.",
        "We think the stock is worth about CNY 120.",
        "The intrinsic value of Moutai is around 1320.",
    ],
)
def test_a_single_number_presented_as_the_fair_value_is_removed(sentence):
    from query_intelligence.agent.compliance import apply_compliance, contains_trading_instruction

    assert contains_trading_instruction(sentence)
    guarded, notes = apply_compliance(
        {"answer": f"贵州茅台最新收盘价为 1409.5 元 [price_600519.SH]。{sentence}", "limitations": []},
        query="茅台合理估值多少",
        nlu_result={"question_style": "fact"},
    )
    assert "removed_trading_instruction" in notes
    assert "1500" not in guarded["answer"] and "1320" not in guarded["answer"]


@pytest.mark.parametrize(
    "sentence",
    [
        "贵州茅台 PE(TTM) 24.6 倍，行业 PE 27.3 倍 [fundamental_600519.SH]。",
        "公允价值变动收益为 3.2 亿元 [fundamental_601318.SH]。",
        "Net worth of the fund rose 2% [price_510300.SH].",
    ],
)
def test_valuation_facts_are_not_fair_value_claims(sentence):
    from query_intelligence.agent.compliance import contains_trading_instruction

    assert not contains_trading_instruction(sentence)


# --- D8: the "no single cause" caveat only on causal questions ------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "causal"),
    [
        ("黄金ETF近来表现如何", False),
        ("有白酒相关的ETF吗", False),
        ("市场上有哪些黄金ETF", False),
        ("五粮液这阵子行情咋样", False),
        ("沪深300ETF最近走势怎么样", False),
        ("How has the gold ETF performed lately?", False),
        ("黄金ETF最近为啥涨这么多", True),
        ("五粮液怎么突然跌了", True),
        ("茅台下跌是什么原因", True),
        ("降准对银行股有什么影响", True),
        ("M2增速回落说明了什么", True),
        ("Why did Wuliangye drop?", True),
        ("What drove the rally in baijiu stocks?", True),
        ("How does a rate cut affect insurers?", True),
    ],
)
def test_causal_questions_are_recognised_lexically(query, causal):
    from query_intelligence.agent.router import is_causal_question

    assert is_causal_question(query) is causal


def test_a_why_style_without_causal_wording_becomes_a_fact_question():
    from query_intelligence.agent.router import correct_question_style

    nlu = {"question_style": "why", "entities": []}
    assert correct_question_style(nlu, "市场上有哪些黄金ETF") == (
        {"question_style": "fact", "entities": []},
        ["override:why_style_without_causal_cue"],
    )
    assert correct_question_style(nlu, "黄金ETF最近为啥涨这么多") == (nlu, [])
    assert correct_question_style({"question_style": "fact"}, "有哪些ETF")[1] == []


@pytest.mark.parametrize("query", ["沪深300ETF最近走势怎么样", "五粮液这阵子行情咋样"])
def test_a_plain_performance_question_gets_no_causal_caveat(agent, query):
    result = agent.chat(query, session_id=f"r8-no-cause-{query}")
    answer = str(result["answer"])
    assert "override:why_style_without_causal_cue" in result["route_reasons"]
    assert "单一原因" not in answer and "因果" not in answer
    assert not any("因果解释" in item for item in result["limitations"])
    assert result["evidence_used"]


def test_a_why_question_keeps_the_caveat(agent):
    result = agent.chat("五粮液最近为啥跌了", session_id="r8-cause")
    assert "override:why_style_without_causal_cue" not in result["route_reasons"]
    assert "不能据此确定" in str(result["answer"])


def test_the_template_adds_the_caveat_only_for_why_style():
    from query_intelligence.agent.composer import compose_template

    log = [
        {
            "tool": "get_price_history",
            "ok": True,
            "data": {"symbol": "510300.SH", "name": "沪深300ETF", "close": 4.811, "as_of": "2026-04-22",
                     "evidence_id": "price_510300.SH", "product_type": "etf"},
            "evidence_ids": ["price_510300.SH"],
        }
    ]  # fmt: skip
    assert "单一原因" not in compose_template(log, zh=True, question_style="fact", query="走势怎么样")["answer"]
    assert "单一原因" in compose_template(log, zh=True, question_style="why", query="为什么跌")["answer"]


# --- D6: crypto funds are out of coverage ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    ["比特币ETF现在值多少", "我想买点比特币ETF，行吗", "以太坊基金最近收益怎么样", "Should I buy a Bitcoin ETF now?"],
)
def test_crypto_funds_are_refused_as_out_of_coverage(agent, query):
    result = agent.chat(query, session_id=f"r8-crypto-{query}")
    assert result["route"] == "refuse"
    assert "coverage:crypto" in result["route_reasons"]
    assert result["limitations"] == ["out_of_coverage"]
    assert not result.get("tool_calls")
    assert not _entities(result)


def test_a_crypto_etf_is_not_a_typo_of_an_a_share_etf(offline_service):
    symbols = {entity.get("symbol") for entity in offline_service.analyze_query("比特币ETF现在值多少")["entities"]}
    assert "512690.SH" not in symbols


def test_a_misspelt_etf_name_still_resolves(offline_service):
    symbols = {entity.get("symbol") for entity in offline_service.analyze_query("黄今ETF最近表现怎么样")["entities"]}
    assert "518880.SH" in symbols


def test_an_advice_phrase_that_is_a_company_alias_does_not_veto_the_refusal(agent):
    result = agent.chat("以太坊基金现在值得买吗", session_id="r8-crypto-advice")
    assert result["route"] == "refuse"
    assert any(reason.startswith("dropped_unnamed_target_out_of_coverage:") for reason in result["route_reasons"])


def test_an_a_share_named_next_to_a_crypto_word_stays_in_scope(agent):
    result = agent.chat("比特币大跌那天贵州茅台收盘多少", session_id="r8-crypto-a-share")
    assert result["route"] != "refuse"
    assert "600519.SH" in _targets(result)


# --- D6: one policy for the short name 平安 ------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "symbol", "match_type"),
    [
        ("平安的赔付率怎么样", "601318.SH", "linked_context"),  # insurance words
        ("平安放贷规模有多大", "000001.SZ", "linked_context"),  # bank words
        ("平安跟行业平均比估值如何", "601318.SH", "linked_default"),  # 行业 is no bank cue
        ("平安和建设银行哪个市净率低", "601318.SH", "linked_default"),  # 银行 of another name is masked
        ("平安PB多少", "601318.SH", "linked_default"),
    ],
)
def test_the_short_name_pingan_follows_one_policy(offline_service, query, symbol, match_type):
    entities = [e for e in offline_service.analyze_query(query)["entities"] if e.get("mention") == "平安"]
    assert [(e["symbol"], e["match_type"]) for e in entities] == [(symbol, match_type)]


def test_an_assumed_reading_is_stated_in_the_answer(agent):
    result = agent.chat("平安跟行业平均比估值如何", session_id="r8-pingan-default")
    assert "alias_default:平安->中国平安|平安银行" in result["route_reasons"]
    assert "alias_assumption_stated" in result["compliance_notes"]
    assert "本次按中国平安回答" in str(result["answer"])
    assert "601318.SH" in _targets(result)


def test_the_conversation_decides_before_the_default(agent):
    session = "r8-pingan-session"
    agent.chat("平安银行最新收盘价多少", session_id=session)
    result = agent.chat("平安PB多少", session_id=session)
    assert "session_disambiguation:平安->平安银行" in result["route_reasons"]
    assert "000001.SZ" in _targets(result)
    assert "alias_assumption_stated" not in result["compliance_notes"]


def test_industry_words_decide_before_the_conversation(agent):
    session = "r8-pingan-context"
    agent.chat("中国平安的市净率", session_id=session)
    result = agent.chat("平安放贷规模有多大", session_id=session)
    assert "alias_context:平安->平安银行" in result["route_reasons"]
    assert "000001.SZ" in _entities(result)


def test_a_discussed_default_needs_no_note(agent):
    session = "r8-pingan-discussed"
    agent.chat("中国平安的市净率", session_id=session)
    result = agent.chat("平安PB多少", session_id=session)
    assert not any(reason.startswith("alias_default:") for reason in result["route_reasons"])
    assert "601318.SH" in _targets(result)


# --- D7: derived metrics or a stated gap, never unrelated raw fields ------------------------------------------------
def test_net_margin_is_derived_from_the_cited_fundamentals(agent):
    result = agent.chat("茅台卖一百块钱能净赚多少，净利润率多高", session_id="r8-margin")
    answer = str(result["answer"])
    assert "823.2 亿元 ÷ 1688.38 亿元 ≈ 48.76% [fundamental_600519.SH]" in answer
    assert result["verification"]["passed"]


def test_net_margins_are_compared_in_words(agent):
    result = agent.chat("五粮液和茅台，哪个净利润率高一些", session_id="r8-margin-compare")
    answer = str(result["answer"])
    assert "≈ 34.84%" in answer and "≈ 48.76%" in answer
    assert "净利率由高到低为：贵州茅台、五粮液" in answer
    assert result["verification"]["passed"]


def test_peg_without_a_growth_rate_is_stated_as_not_computable(agent):
    result = agent.chat("算算中国平安的PEG", session_id="r8-peg")
    answer = str(result["answer"])
    assert "无法计算中国平安的PEG" in answer and "净利润增速" in answer
    assert "601318.SH" in _targets(result)


def test_peg_is_derived_when_the_source_reports_growth():
    from query_intelligence.agent.composer import compose_template

    log = [
        {
            "tool": "get_fundamentals",
            "ok": True,
            "data": {"symbol": "600519.SH", "name": "贵州茅台", "evidence_id": "fundamental_600519.SH",
                     "report_date": "2025-12-31", "metrics": {"pe_ttm": 24.6, "netprofit_yoy": 15.0}},
            "evidence_ids": ["fundamental_600519.SH"],
        }
    ]  # fmt: skip
    answer = compose_template(log, zh=True, query="茅台PEG多少")["answer"]
    assert "PEG（市盈率 ÷ 净利润增速）：24.6 ÷ 15 ≈ 1.64 [fundamental_600519.SH]" in answer
    assert "无法计算" not in answer


@pytest.mark.parametrize(
    ("query", "ytd"),
    [
        ("上证50今年以来累计涨了多少", True),
        ("五粮液年初到现在跌了多少", True),
        ("How much is Moutai up year to date?", True),
        ("What's the YTD return of the CSI 300?", True),
        ("茅台今年营收多少", False),  # this year's statements, not a move
        ("五粮液今天涨了多少", False),
    ],
)
def test_year_to_date_questions_are_recognised(query, ytd):
    from query_intelligence.agent.coverage import asks_year_to_date

    assert asks_year_to_date(query) is ytd


def test_year_to_date_without_the_first_close_of_the_year_is_stated_unavailable(agent):
    result = agent.chat("中国平安年初到现在涨了多少", session_id="r8-ytd")
    answer = str(result["answer"])
    assert answer.startswith("当前数据中没有中国平安今年首个交易日的收盘价，无法计算今年以来的涨跌幅")
    assert "601318.SH" in _targets(result)


def test_year_start_needs_a_close_from_the_year_before():
    from query_intelligence.agent.tools.market import year_start_close

    history = [
        {"trade_date": "20260105", "close": 10.0},
        {"trade_date": "20251231", "close": 9.5},
        {"trade_date": "2026-04-22", "close": 12.0},
        {"trade_date": "2026-01-06", "close": 10.2},
    ]
    assert year_start_close(history).model_dump() == {"date": "2026-01-05", "close": 10.0}
    assert year_start_close(history[:1] + history[2:]) is None  # the first 2026 close might not be the year's first


def test_year_to_date_change_is_derived_with_both_closes_and_verifies(offline_service):
    from query_intelligence.agent.composer import compose_template
    from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore
    from query_intelligence.agent.verifier import verify_answer

    data = {"symbol": "510300.SH", "name": "沪深300ETF", "close": 4.811, "as_of": "2026-04-22",
            "evidence_id": "price_510300.SH", "product_type": "etf",
            "year_start": {"date": "2026-01-05", "close": 4.5}}  # fmt: skip
    log = [{"tool": "get_price_history", "ok": True, "data": data, "evidence_ids": ["price_510300.SH"]}]
    answer = compose_template(log, zh=True, query="沪深300ETF今年以来涨了多少")
    assert "今年首个交易日（2026-01-05）收盘 4.5 元，最新（2026-04-22）收盘 4.811 元，涨跌幅 6.91%" in answer["answer"]
    store = EvidenceStore()
    store.add(AgentEvidence(evidence_id="price_510300.SH", kind="structured", source_type="market_api",
                            title="t", payload={**data, "year_start": data["year_start"]}))  # fmt: skip
    assert verify_answer(answer, store, query="沪深300ETF今年以来涨了多少", allow_derived=True).passed


def test_a_percent_share_is_derived_only_from_two_amounts():
    from query_intelligence.agent.verifier import _is_amount, _is_derived, claim_values

    claims = claim_values("823.2 亿元 ÷ 1688.38 亿元 ≈ 48.76% [fundamental_600519.SH]。")
    amounts = [value for value, scales, _rounding, _sign in claims if _is_amount(scales)]
    assert _is_derived(48.76, 0.005, [823.2, 1688.38], shares=amounts)
    # PB / PE * 100 = 32.93 is not a figure anyone states: multiples are not amounts, so it is not "derived"
    multiples = claim_values("PE 24.6 倍，PB 8.1 倍，ROE 32.93%")
    assert not [value for value, scales, _rounding, _sign in multiples if _is_amount(scales)]
    assert not _is_derived(32.93, 0.005, [24.6, 8.1], shares=[])
