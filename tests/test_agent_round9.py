"""Round-9 rules, written from the round-5 review (E3-E8) with the author's own wording (none repeats a reviewer probe
or a round-5 held-out text; ``tests/test_agent_eval.py`` checks the dev tasks and router labels for that).

* E6: fair value asked per share, through a valuation model or with a verdict word ("每股值多少", "按DCF…", "估值应该
  给到…", "多少元比较公道") is a judgment: hedged with a limitation.
* E7: a sector word in the question ("这只银行股") decides the short name 平安; crypto funds named by token (BTC, ETH,
  Solana, "<X>币ETF") are out of coverage.
* E8: net margin however phrased is derived from the cited revenue and net profit; P/S and a maximum drawdown are
  stated as not computable.
* E5: "这个行业/该板块" resolves to the discussed target's industry; "相差多少" after a comparison derives the
  difference.
* E3: a figure one document states and the structured data does not is attributed with the layer's marker, in the
  answer and in the key points. E4: a ledger headline with such a figure, or with an unconfirmed-source shape, is
  hidden.

Offline snapshot values: 贵州茅台 PE 24.6 / ROE 33% / revenue 1688.38 亿 / net profit 823.2 亿; 五粮液 PE 20.9 / PB 5.4
/ ROE 29.4% / 1085 亿 / 378 亿, daily change -0.5337%; 中国平安 PE 8.7 / PB 1.1 / 12180 亿, daily change 0.73%;
保险 industry PE 11.8 / PB 1.45; 白酒 industry PE 27.3.
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


# --- E6: fair value per share, by a model, or with a verdict word ---------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "用贴现现金流模型算，五粮液一股值多少",
        "平安估值应该给到多少倍才公道",
        "五粮液每股大概值几块钱",
        "茅台给到多少元一股算公允",
        "What is Wuliangye worth on a DCF basis?",
    ],
)
def test_fair_value_by_share_model_or_verdict_word_is_a_judgment(query):
    from query_intelligence.agent.router import FAIR_VALUE_MARKERS, decide_route

    assert FAIR_VALUE_MARKERS.search(query)
    nlu = {"entities": [{"symbol": "000858.SZ", "entity_type": "stock"}], "question_style": "fact"}
    assert "lexical:judgment_or_timing" in decide_route(nlu, query=query).reasons


@pytest.mark.parametrize(
    "query",
    ["DCF估值法是什么意思", "现金流折现怎么理解", "五粮液现在每股多少钱", "五粮液的市盈率多少倍", "手续费多少比较合理"],
)
def test_the_model_itself_prices_and_multiples_are_not_fair_value_requests(query):
    from query_intelligence.agent.router import FAIR_VALUE_MARKERS

    assert not FAIR_VALUE_MARKERS.search(query)


def test_a_model_based_value_question_is_hedged_and_fetches_the_multiples(agent):
    result = agent.chat("用贴现现金流模型算，五粮液一股值多少", session_id="r9-dcf")
    assert "fair_value_hedge" in result["compliance_notes"]
    assert "不给出合理估值" in str(result["answer"])
    assert any("合理估值" in item for item in result["limitations"])
    assert any(call["tool"] == "get_fundamentals" for call in result["tool_calls"])


# --- E7: bank context for 平安; crypto funds by token ---------------------------------------------------------------
@pytest.mark.parametrize("query", ["作为银行股，平安的市净率高吗", "平安这家银行的市盈率多少"])
def test_a_sector_word_in_the_question_decides_pingan(offline_service, query):
    entities = [e for e in offline_service.analyze_query(query)["entities"] if e.get("mention") == "平安"]
    assert [(e["symbol"], e["match_type"]) for e in entities] == [("000001.SZ", "linked_context")]


def test_another_bank_name_is_still_masked(offline_service):
    # "平安和建设银行…": the 银行 of 建设银行 is part of another name, not context for 平安 (round 8, unchanged)
    entities = [
        e for e in offline_service.analyze_query("平安和建设银行谁的PE低")["entities"] if e.get("mention") == "平安"
    ]
    assert [e["symbol"] for e in entities] == ["601318.SH"]


def test_bank_context_pingan_states_the_missing_data(agent):
    result = agent.chat("作为银行股，平安的市净率高吗", session_id="r9-pingan-bank")
    assert "alias_context:平安->平安银行" in result["route_reasons"]
    assert "000001.SZ" in _targets(result)
    assert "平安银行（000001.SZ）" in str(result["answer"])


@pytest.mark.parametrize(
    "query",
    [
        "SOL现货基金能买吗",
        "索拉纳ETF表现如何",
        "Is a Solana fund worth a look?",
        "莱特币基金收益高吗",
        "某某币ETF能配吗",
    ],
)
def test_crypto_funds_named_by_token_are_out_of_coverage(query):
    from query_intelligence.agent.coverage import out_of_coverage

    assert out_of_coverage(query) == "crypto"


@pytest.mark.parametrize(
    "query",
    ["货币ETF收益怎么样", "货币基金年化多少", "以太网概念股有哪些", "LINK这个词什么意思", "人民币升值利好哪些板块"],
)
def test_money_market_funds_and_lookalikes_are_not_crypto(query):
    from query_intelligence.agent.coverage import out_of_coverage

    assert out_of_coverage(query) is None


def test_a_token_fund_question_is_refused(agent):
    result = agent.chat("SOL现货基金能买吗", session_id="r9-sol")
    assert result["route"] == "refuse" and result["limitations"] == ["out_of_coverage"]


# --- E8: net margin however phrased; P/S and drawdown stated as not computable ------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "五粮液净利润在营业收入里占几成",
        "中国平安的销售利润率",
        "五粮液每卖100元能落下多少净利润",
        "What is Wuliangye's profit as a percentage of revenue?",
    ],
)
def test_net_margin_phrasings_are_recognised(query):
    from query_intelligence.agent.coverage import requested_metrics

    assert [metric.key for metric in requested_metrics(query)] == ["net_margin"]


@pytest.mark.parametrize("query", ["五粮液的毛利润率", "五粮液的毛利率"])
def test_gross_margin_is_not_net_margin(query):
    from query_intelligence.agent.coverage import requested_metrics

    assert [metric.key for metric in requested_metrics(query)] == ["gross_margin"]


def test_a_share_of_revenue_question_is_derived(agent):
    result = agent.chat("五粮液净利润在营业收入里占几成", session_id="r9-margin")
    assert "378 亿元 ÷ 1085 亿元 ≈ 34.84% [fundamental_000858.SZ]" in str(result["answer"])
    assert result["verification"]["passed"]


def test_price_to_sales_is_stated_as_not_computable(agent):
    result = agent.chat("五粮液市销率多少", session_id="r9-ps")
    answer = str(result["answer"])
    assert answer.startswith("无法计算五粮液的市销率") and "总市值" in answer
    assert "000858.SZ" in _targets(result)


def test_a_drawdown_is_stated_as_not_computable(agent):
    result = agent.chat("中国平安过去半年的最大回撤", session_id="r9-drawdown")
    answer = str(result["answer"])
    assert answer.startswith("当前数据只有中国平安") and "无法计算所问期间的最大回撤" in answer


# --- E5: the discussed target's industry; the difference after a comparison -----------------------------------------
def test_an_industry_reference_resolves_to_the_discussed_targets_industry(agent):
    session = "r9-industry"
    agent.chat("五粮液PE多少", session_id=session)
    result = agent.chat("该板块的平均市盈率呢", session_id=session)
    assert "industry_reference:该板块->白酒" in result["route_reasons"]
    assert result["route"] != "clarify"
    assert str(result["answer"]).startswith("根据本次检索到的证据：所属行业 白酒：PE 27.3 倍")


def test_an_industry_reference_without_a_target_is_left_alone():
    from query_intelligence.agent.memory import resolve_industry_reference

    assert resolve_industry_reference("这个行业的平均PE呢", [], lambda _symbol: "白酒") is None
    two = [{"entities": [{"symbol": "600519.SH", "name": "贵州茅台"}, {"symbol": "601318.SH", "name": "中国平安"}]}]
    industries = {"600519.SH": "白酒", "601318.SH": "保险"}
    assert resolve_industry_reference("这个行业的平均PE呢", two, industries.get) is None  # two industries: ambiguous
    one = [{"entities": [{"symbol": "601318.SH", "name": "中国平安"}]}]
    assert resolve_industry_reference("它属于哪个行业", one, industries.get) is None  # asks for the industry


@pytest.mark.parametrize(
    ("query", "follow_up"),
    [("差了多少个百分点", True), ("相差多少", True), ("两者差距多大", True), ("How big is the gap?", True),
     ("差不多吧", False), ("茅台差多少到2000元", False), ("为什么", False)],
)  # fmt: skip
def test_difference_follow_ups_are_recognised(query, follow_up):
    from query_intelligence.agent.memory import is_difference_follow_up

    assert is_difference_follow_up(query) is follow_up


def test_a_difference_follow_up_derives_the_gap_from_the_previous_comparison(agent):
    session = "r9-difference"
    agent.chat("中国平安和五粮液今天哪个涨得多", session_id=session)
    result = agent.chat("相差多少", session_id=session)
    answer = str(result["answer"])
    assert any(reason.startswith("difference_follow_up:") for reason in result["route_reasons"])
    assert "两者相差 1.26 个百分点（中国平安更高） [price_601318.SH][price_000858.SZ]" in answer
    assert result["verification"]["passed"]


def test_a_same_turn_difference_and_a_ratio_to_the_industry_are_derived(agent):
    result = agent.chat("贵州茅台和五粮液的净利润相差多少亿", session_id="r9-gap-amount")
    assert "两者相差 445.2 亿元（贵州茅台更高）" in str(result["answer"])
    assert result["verification"]["passed"]
    ratio = agent.chat("中国平安市净率是保险行业的多少倍", session_id="r9-ratio")
    assert "中国平安市净率 1.1 倍，保险行业 1.45 倍，前者约为后者的 0.76 倍" in str(ratio["answer"])
    assert ratio["verification"]["passed"]
