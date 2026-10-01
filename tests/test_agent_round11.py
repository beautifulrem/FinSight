"""Round-11 rules, written from the round-7 review (G1-G6, G11) with the author's own wording.

* G1-G3: the session comparison frame (``agent/frame.py``): after every turn the session keeps the last metric
  discussed and its operands (targets, an industry average) in order, with values and evidence ids. "X呢" adds an
  operand; a gap / ratio / relative / which-is-higher question is computed from the frame; "前者/后者" follow its order;
  metric names are matched longest first (净利率 ≠ 净利润 ≠ 净利, 市净率 ≠ 市盈率, 毛利率).
* G4: the LLM agent sees the frame in its session memory; a rule in prompts v3/v4 says to fetch a missing operand; a
  draft that does not state the computed result gets it appended.
"""

from __future__ import annotations

import pytest

# ---- G3: metric names, longest first ----


@pytest.mark.parametrize(
    ("text", "metric"),
    [
        ("五粮液的净利率有多高", "net_margin"),
        ("五粮液的净利润率", "net_margin"),
        ("五粮液的净利润", "net_profit"),
        ("五粮液净利多少", "net_profit"),
        ("中国平安的市净率", "pb"),
        ("中国平安的市盈率", "pe"),
        ("五粮液毛利率", "gross_margin"),
        ("贵州茅台净利润占营业收入的比重是多少", "net_margin"),
        ("贵州茅台的净利润是营收的百分之多少", "net_margin"),
        ("Wuliangye's net profit margin", "net_margin"),
        ("Wuliangye's net income", "net_profit"),
        ("Moutai earnings per share", "eps"),
        ("Ping An price-to-book", "pb"),
        ("成交额", "amount"),
        ("证券ETF今天收盘", "close"),
    ],
)
def test_frame_metrics_are_matched_longest_first(text, metric):
    from query_intelligence.agent.frame import metric_of

    assert metric_of(text) == metric


def test_the_ellipsis_aspect_keeps_net_margin_apart_from_net_profit():
    from query_intelligence.agent.memory import resolve_ellipsis

    turns = [
        {
            "query": "贵州茅台净利率多少",
            "effective_query": "贵州茅台净利率多少",
            "entities": [{"name": "贵州茅台", "symbol": "600519.SH"}],
        }
    ]
    rewritten, reason = resolve_ellipsis("那五粮液呢", turns, [{"canonical_name": "五粮液", "symbol": "000858.SZ"}])
    assert reason == "ellipsis:aspect->净利率"
    assert "净利率" in rewritten


# ---- G1/G2: which questions read the frame ----


@pytest.mark.parametrize(
    ("text", "operation"),
    [
        ("那两个差了几个百分点", "difference"),
        ("差距有多大", "difference"),
        ("大了多少", "difference"),
        ("后一个是前一个的多少倍", "ratio"),
        ("比值是多少", "ratio"),
        ("贵了百分之多少", "relative"),
        ("算溢价吗，溢价多少", "relative"),
        ("谁的更低", "which"),
        ("哪一只高一些", "which"),
        ("By how much?", "difference"),
        ("What's the ratio of the two?", "ratio"),
        ("how many percent lower is it?", "relative"),
        ("Which of them is bigger?", "which"),
        # not comparisons
        ("五粮液市盈率是多少倍", None),
        ("贵州茅台的净利润是营收的百分之多少", None),
        ("哪个更值得买", None),
        ("成交额大概多少", None),
        ("为什么差这么多", None),
    ],
)
def test_frame_operations(text, operation):
    from query_intelligence.agent.frame import frame_operation

    assert frame_operation(text) == operation


def _frame_turn(metric, *operands):
    return {
        "query": "x",
        "entities": [],
        "frame": {"metric": metric, "operands": [dict(item) for item in operands]},
    }


_WLY = {
    "kind": "target",
    "name": "五粮液",
    "symbol": "000858.SZ",
    "value": 29.4,
    "evidence_id": "fundamental_000858.SZ",
}
_MT = {
    "kind": "target",
    "name": "贵州茅台",
    "symbol": "600519.SH",
    "value": 33.0,
    "evidence_id": "fundamental_600519.SH",
}


def test_former_and_latter_follow_the_frame_order():
    from query_intelligence.agent.frame import resolve_frame_question

    turns = [_frame_turn("revenue", _WLY, _MT)]
    rewritten, reason, request = resolve_frame_question("后者是前者的几倍", turns, [])
    assert request["operation"] == "ratio" and request["metric"] == "revenue"
    assert [item["symbol"] for item in request["operands"]] == ["600519.SH", "000858.SZ"]
    assert reason == "frame:ratio:revenue:贵州茅台|五粮液"
    assert rewritten == "贵州茅台和五粮液的营业收入，贵州茅台是五粮液的几倍"


def test_a_named_target_joins_the_frame_and_a_named_metric_replaces_its_metric():
    from query_intelligence.agent.frame import resolve_frame_question

    turns = [_frame_turn("roe", _WLY)]
    named = [{"canonical_name": "中国平安", "symbol": "601318.SH"}]
    _rewritten, _reason, request = resolve_frame_question("那中国平安呢，相差几个点", turns, named)
    assert [item["symbol"] for item in request["operands"]] == ["000858.SZ", "601318.SH"]
    _rewritten, _reason, request = resolve_frame_question("那市净率差多少", [_frame_turn("roe", _WLY, _MT)], [])
    assert request["metric"] == "pb"


def test_no_frame_question_without_two_operands_and_a_metric():
    from query_intelligence.agent.frame import resolve_frame_question

    assert resolve_frame_question("差了多少", [_frame_turn("roe", _WLY)], []) is None
    assert resolve_frame_question("差了多少", [_frame_turn(None, _WLY, _MT)], []) is None
    assert resolve_frame_question("差了多少", [], []) is None
    # two named targets and a metric: since round 12 (H2) computed from the operands the question names, with its own
    # metric, not the frame's
    named = [{"canonical_name": "五粮液", "symbol": "000858.SZ"}, {"canonical_name": "贵州茅台", "symbol": "600519.SH"}]
    _rewritten, reason, _request = resolve_frame_question(
        "五粮液和茅台的市盈率差多少", [_frame_turn("roe", _WLY, _MT)], named
    )
    assert reason == "frame:difference:pe:五粮液|贵州茅台"


def test_the_frame_follows_the_turns():
    from query_intelligence.agent.frame import next_frame

    tool_log = [
        {
            "tool": "get_fundamentals",
            "ok": True,
            "data": {
                "symbol": "000858.SZ",
                "name": "五粮液",
                "evidence_id": "fundamental_000858.SZ",
                "metrics": {"roe": 29.4, "pe_ttm": 20.9},
                "industry": {"industry_name": "白酒", "evidence_id": "industry_白酒", "metrics": {"pe": 27.3}},
            },
        }
    ]
    first = next_frame(
        None, query="五粮液的净资产收益率", route="workflow", targets=[{"name": "五粮液", "symbol": "000858.SZ"}],
        tool_log=tool_log,
    )  # fmt: skip
    assert first == {
        "metric": "roe",
        "operands": [
            {
                "kind": "target",
                "name": "五粮液",
                "symbol": "000858.SZ",
                "value": 29.4,
                "evidence_id": "fundamental_000858.SZ",
            }
        ],
    }
    # the same metric for another target adds an operand and keeps the earlier value
    second = next_frame(
        first, query="贵州茅台的ROE呢？", route="workflow", targets=[{"name": "贵州茅台", "symbol": "600519.SH"}],
        tool_log=[],
    )  # fmt: skip
    assert [item["symbol"] for item in second["operands"]] == ["000858.SZ", "600519.SH"]
    assert second["operands"][0]["value"] == 29.4
    # a refusal keeps the frame; a turn about something else with targets of its own starts a frame without a metric
    assert next_frame(second, query="写首诗", route="refuse", targets=[], tool_log=[]) == second
    trend = next_frame(
        second, query="贵州茅台最近走势怎么样", route="workflow", targets=[{"name": "贵州茅台", "symbol": "600519.SH"}],
        tool_log=[],
    )  # fmt: skip
    assert trend["metric"] is None
    # a short question about the industry average carries the metric and adds the industry
    pe = next_frame(
        None, query="五粮液市盈率", route="workflow", targets=[{"name": "五粮液", "symbol": "000858.SZ"}],
        tool_log=tool_log,
    )  # fmt: skip
    industry = next_frame(
        pe, query="同行平均是多少", route="workflow", targets=[{"name": "五粮液", "symbol": "000858.SZ"}],
        tool_log=tool_log,
    )  # fmt: skip
    assert industry["metric"] == "pe"
    assert [item.get("industry") for item in industry["operands"]] == [None, "白酒"]
    assert industry["operands"][1]["value"] == 27.3


# ---- G1-G3 end to end (offline template path) ----


@pytest.fixture(scope="module")
def agent(offline_service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), ScriptedLLM([]))
    service = AgentService(runtime, trace_sinks=[])
    yield service
    runtime.close()


def _session(agent, name: str, *queries: str) -> list[dict]:
    return [agent.chat(query, session_id=f"r11-{name}") for query in queries]


def test_an_ellipsis_chain_then_a_gap_in_percentage_points(agent):
    *_, gap = _session(agent, "roe-chain", "五粮液净资产收益率多高", "贵州茅台的呢", "那相差几个百分点")
    assert "frame:difference:roe:五粮液|贵州茅台" in gap["route_reasons"]
    assert "两者相差 3.6 个百分点（贵州茅台更高） [fundamental_000858.SZ][fundamental_600519.SH]" in gap["answer"]
    assert gap["verification"]["passed"]
    assert "最新可用收盘价" not in gap["answer"].split("。")[0]


def test_an_ellipsis_and_a_gap_in_one_message(agent):
    _first, gap = _session(agent, "one-message", "五粮液ROE", "那茅台呢，两个差多少")
    assert "两者相差 3.6 个百分点" in gap["answer"]


def test_the_latter_as_a_multiple_of_the_former(agent):
    *_, ratio = _session(agent, "ratio", "五粮液去年营收多少", "那茅台的呢", "后者是前者的几倍")
    assert "frame:ratio:revenue:贵州茅台|五粮液" in ratio["route_reasons"]
    assert "前者约为后者的 1.56 倍" in ratio["answer"]
    assert ratio["route"] != "refuse" and ratio["verification"]["passed"]


def test_a_discount_to_the_industry_average_in_percent(agent):
    *_, relative = _session(agent, "discount", "平安的市净率是多少", "保险业平均水平呢", "比行业便宜百分之几")
    assert "中国平安相对保险行业平均折价约 24.14%" in relative["answer"]
    assert relative["verification"]["passed"]


def test_an_english_ratio_between_the_two(agent):
    *_, ratio = _session(agent, "ratio-en", "What's Wuliangye's P/B?", "and Ping An's?", "what's the ratio of the two?")
    assert "the former is about 4.91 times the latter" in ratio["answer"]


def test_a_turnover_gap_states_which_is_larger(agent):
    *_, gap = _session(agent, "turnover", "创业板ETF的成交额", "证券ETF的呢", "谁大，大了多少")
    assert "两者相差 12.29 亿元（创业板ETF更高）" in gap["answer"]


def test_net_margin_is_carried_as_net_margin_not_as_a_price(agent):
    _first, carried, gap = _session(agent, "net-margin", "五粮液的净利润率", "茅台呢", "差距是多少")
    assert "ellipsis:aspect->净利润率" in carried["route_reasons"]
    assert "48.76%" in carried["answer"]
    assert "两者相差 13.92 个百分点" in gap["answer"]


def test_three_operands_are_ranked(agent):
    *_, ranked = _session(agent, "three", "五粮液的市盈率", "茅台呢", "那平安呢", "这三家谁最低")
    assert "市盈率由高到低：贵州茅台 24.6 倍、五粮液 20.9 倍、中国平安 8.7 倍" in ranked["answer"]


def test_a_frame_without_a_metric_is_clarified_not_refused(agent):
    *_, asked = _session(agent, "no-metric", "茅台最近走势", "五粮液呢", "差了多少")
    assert asked.get("status") == "needs_clarification"
    assert "哪项指标" in asked["clarification"]["question"]


def test_a_ratio_with_one_target_is_never_refused_as_out_of_scope(agent):
    _first, asked = _session(agent, "one-target", "五粮液市净率", "前者是后者的几倍")
    assert asked.get("status") == "needs_clarification" or asked.get("route") != "refuse"


# ---- G4: the LLM path ----


def test_the_frame_is_in_the_llm_memory_and_the_prompt_rule_is_in_v3_and_v4():
    from query_intelligence.agent.memory import session_memory
    from query_intelligence.agent.prompts import agent_user_message, get_prompt

    memory = session_memory([_frame_turn("roe", _WLY, _MT)], "两者相差多少")
    frame = memory["comparison_frame"]
    assert frame["metric"] == "ROE"
    assert frame["operands_in_order"][0]["value_in_earlier_turn"] == 29.4
    assert "comparison_frame" in agent_user_message("q", {}, language="zh", memory=memory)
    for version in ("v3", "v4"):
        assert "comparison_frame" in get_prompt("agent_system", version).text


def test_a_declined_llm_gap_gets_the_computed_gap_appended(offline_service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM, final_turn, tool_call_turn
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.state import AgentConfig
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    llm = ScriptedLLM(
        [
            tool_call_turn(("get_fundamentals", {"target": "000858.SZ"})),
            final_turn({"answer": "五粮液 ROE 为 29.4% [fundamental_000858.SZ]。", "evidence_used": []}),
            tool_call_turn(("get_fundamentals", {"target": "600519.SH"})),
            final_turn({"answer": "贵州茅台 ROE 为 33% [fundamental_600519.SH]。", "evidence_used": []}),
            # the third turn declines without fetching: the fallback fetches both operands and appends the gap
            final_turn({"answer": "本轮工具结果中没有五粮液的数据，无法核实两者的差值。", "evidence_used": []}),
        ]
    )
    # the shipped defaults (tests/conftest.py turns prefetch and derived numbers off for older tests)
    config = AgentConfig(planner_prefetch=False, revise_policy="cite_repair", verify_derived=True)
    runtime = AgentRuntime(offline_service, build_registry_for_service(offline_service), llm, config=config)
    service = AgentService(runtime, trace_sinks=[])
    try:
        for query in ("五粮液ROE是多少", "茅台的呢"):
            service.chat(query, session_id="r11-llm-decline", mode="agent")
        gap = service.chat("两家差几个点", session_id="r11-llm-decline", mode="agent")
    finally:
        runtime.close()
    assert "frame_result_appended" in gap["degraded"]
    assert "两者相差 3.6 个百分点" in gap["answer"]
    assert gap["verification"]["passed"]
    last_request = "\n".join(str(message.get("content")) for message in llm.requests[-1]["messages"])
    assert "comparison_frame" in last_request


# ---- G5: derived chat metrics ----


@pytest.mark.parametrize(
    ("query", "shares"),
    [
        ("我手里有1500股五粮液，现在市值多少", 1500),
        ("持有三千股中国平安，合计值多少钱", 3000),
        ("I own 250 shares of Wuliangye, how much are they worth?", 250),
        ("五粮液一股值多少钱", None),
        ("每10股派现多少", None),
        ("五粮液股价多少", None),
    ],
)
def test_holding_value_requests(query, shares):
    from query_intelligence.agent.coverage import holding_value_request

    found = holding_value_request(query)
    assert (found[0] if found else None) == shares


def test_a_holding_is_valued_at_the_close_without_a_fair_value_hedge(agent):
    result = agent.chat("我手里有1500股五粮液，现在市值多少", session_id="r11-holding")
    assert "1500 股五粮液的市值约为 1500 × 100.64 = 150960 元 [price_000858.SZ]" in result["answer"]
    assert "不是可成交价格" in result["answer"]
    assert "fair_value_hedge" not in result["compliance_notes"]
    assert result["verification"]["passed"]
    english = agent.chat("I own 250 shares of Wuliangye, how much are they worth?", session_id="r11-holding-en")
    assert "250 × 100.64 = CNY 25160" in english["answer"]


def test_net_profit_as_a_share_of_revenue_is_the_net_margin(agent):
    result = agent.chat("五粮液净利润为营收的百分之多少", session_id="r11-share")
    assert "378 亿元 ÷ 1085 亿元 ≈ 34.84%" in result["answer"]


def test_eps_is_named_missing_and_the_implied_value_is_labelled(agent):
    result = agent.chat("中国平安的每股收益是多少", session_id="r11-eps")
    answer = result["answer"]
    assert "当前数据源没有中国平安的每股收益数据" in answer
    assert "53.61 元 ÷ 8.7 ≈ 6.16 元（推算值，不是公司披露的每股收益" in answer
    assert result["verification"]["passed"]


# ---- G6: an implied price is hedged; H shares and Hong Kong lookalikes are out of coverage ----


@pytest.mark.parametrize(
    "query",
    [
        "参照保险行业的平均市净率给中国平安估个股价，应该是多少",
        "用白酒同行的PE算，五粮液每股该值多少钱",
        "五粮液理论股价是多少",
        "Using the industry average P/E, what would Moutai trade at?",
        "What's Wuliangye's implied share price at the sector multiple?",
    ],
)
def test_an_implied_price_from_a_multiple_is_a_fair_value_question(query):
    from query_intelligence.agent.router import FAIR_VALUE_MARKERS

    assert FAIR_VALUE_MARKERS.search(query)


@pytest.mark.parametrize("query", ["五粮液的市盈率是多少", "中国平安的市净率比行业低多少", "What is Moutai's P/E?"])
def test_plain_multiples_are_not_fair_value_questions(query):
    from query_intelligence.agent.router import FAIR_VALUE_MARKERS

    assert not FAIR_VALUE_MARKERS.search(query)


def test_an_implied_price_is_hedged_without_a_single_price(agent):
    result = agent.chat("用白酒同行的PE算，五粮液每股该值多少钱", session_id="r11-implied")
    assert "fair_value_hedge" in result["compliance_notes"]
    assert "FinSight 不给出合理估值" in result["answer"]


def test_an_h_share_question_is_out_of_coverage_and_names_no_a_share_price(agent):
    result = agent.chat("平安的H股收盘多少", session_id="r11-h-share")
    assert result["route"] == "refuse" and result["limitations"] == ["out_of_coverage"]
    assert "53.61" not in result["answer"] and "H 股" in result["answer"]
    mixed = agent.chat("中国平安和比亚迪电子的市盈率", session_id="r11-mixed")
    assert "未用名称相近的 A 股代替" in mixed["answer"]
    assert "002594.SZ" not in {call["arguments"].get("target") for call in mixed["tool_calls"]}


# ---- G11: English labels for the fact-check rows; the asked metric for the KPI tiles ----


def test_claim_reports_carry_english_labels_for_industries():
    from query_intelligence.agent.claim_check import english_label

    assert english_label("白酒行业平均") == "baijiu (liquor) industry average"
    assert english_label("保险行业") == "insurance industry"
    assert english_label("贵州茅台") == "Kweichow Moutai"


def test_the_response_lists_the_asked_metrics(agent):
    result = agent.chat("五粮液和中国平安的净资产收益率谁高", session_id="r11-asked")
    assert result["nlu_summary"]["asked_metrics"][0] == "roe"
