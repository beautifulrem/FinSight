"""Round-12 rules: the chat classes the round-7 held-out slice still failed after round 11, in the author's own
wording (other targets and phrasings than the slice; the slice is after exposure from round 12 on).

* Turnover as a frame metric however it is asked ("成交了多少钱", "交易额", "交易更活跃", "trading value", "value
  traded"): one vocabulary for the frame, the price answer's details, the template's comparisons and the ellipsis
  aspect, so the answer states the turnover and a later gap / ratio turn has a metric to compute.
* Gap wording: "多跌了多少", "少涨了几个点", "by how many points", "how many points apart"; a gap asked together with
  "which one is higher" is a gap (the higher side is named in the same sentence) and a factual "which one is higher" is
  not hedged as a judgment.
* Holding values: carried to another target ("换成同样数量的X呢"), stated in English across a sentence boundary, and
  for fund units ("三万份证券ETF", "4000 units").
* Net profit as a share of revenue in English ("What share of that revenue is left as net profit?"), and "net-margin"
  with a hyphen.
* A Hong Kong listing described in words ("它在港交所挂牌的那部分股票") is out of coverage, never the A share.
"""

from __future__ import annotations

import pytest

# ---- turnover vocabulary ----


@pytest.mark.parametrize(
    ("text", "metric"),
    [
        ("贵州茅台今天成交了多少钱", "amount"),
        ("中国平安昨天交易额多大", "amount"),
        ("这两个哪个交易更活跃", "amount"),
        ("五粮液成交得活跃吗", "amount"),
        ("What was Ping An's value traded on the last session?", "amount"),
        ("Moutai's trading value", "amount"),
        ("五粮液成交量多少", "volume"),
        ("How much did the securities ETF move yesterday?", "pct_change"),
        ("Wuliangye fell today, by how much?", "pct_change"),
        ("What's the net-margin gap between Wuliangye and Ping An?", "net_margin"),
        ("What share of that revenue is left as net profit?", "net_margin"),
        ("What percent of its sales turns into net income?", "net_margin"),
    ],
)
def test_round12_metric_vocabulary(text, metric):
    from query_intelligence.agent.frame import metric_of

    assert metric_of(text) == metric


@pytest.mark.parametrize(
    ("text", "operation"),
    [
        ("多跌了多少", "difference"),
        ("少涨了几个点", "difference"),
        ("茅台少跌了多少", "difference"),
        ("Which is larger, and by how many points?", "difference"),
        ("so how many points apart are they?", "difference"),
        ("哪只交易更活跃", "which"),
        # not comparisons
        ("今天跌了多少", None),
        ("What share of that revenue is left as net profit?", None),
    ],
)
def test_round12_frame_operations(text, operation):
    from query_intelligence.agent.frame import frame_operation

    assert frame_operation(text) == operation


@pytest.mark.parametrize(
    ("query", "amount"),
    [("沪深300ETF今天成交了多少钱", True), ("Ping An's trading value?", True), ("五粮液收盘价", False)],
)
def test_turnover_wording_asks_for_the_turnover_detail(query, amount):
    from query_intelligence.agent.coverage import requested_price_fields

    assert requested_price_fields(query).amount is amount


# ---- end to end on the offline template path ----


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
    return [agent.chat(query, session_id=f"r12-{name}") for query in queries]


def test_a_turnover_asked_in_words_is_stated_and_its_ratio_computed(agent):
    first, second, ratio = _session(
        agent, "turnover-ratio", "沪深300ETF今天成交了多少钱", "中国平安的呢", "前者是后者的几倍"
    )
    assert "成交额 48.52 亿元 [price_510300.SH]" in first["answer"]
    assert "成交额 66.4 亿元 [price_601318.SH]" in second["answer"]
    assert "frame:ratio:amount:沪深300ETF|中国平安" in ratio["route_reasons"]
    assert "前者约为后者的 0.73 倍 [price_510300.SH][price_601318.SH]" in ratio["answer"]
    assert ratio["verification"]["passed"]


def test_an_english_trading_value_is_stated_carried_and_compared(agent):
    first, second, gap = _session(
        agent,
        "turnover-en",
        "What was Moutai's trading value on the last session?",
        "and the CSI 300 ETF's?",
        "how many times bigger is the latter?",
    )
    assert "turnover CNY 3.79 bn [price_600519.SH]" in first["answer"]
    assert "turnover CNY 4.85 bn [price_510300.SH]" in second["answer"]
    assert "frame:ratio:amount:" in " ".join(gap["route_reasons"])
    assert "1.28 times" in gap["answer"]


def test_which_traded_more_actively_compares_turnover(agent):
    *_, verdict = _session(agent, "active", "五粮液成交额多少", "证券ETF呢", "两个里哪个交易更活跃")
    assert "成交额：五粮液 14.53 亿元 高于 证券ETF 4.41 亿元 [price_000858.SZ][price_512880.SH]" in verdict["answer"]


def test_a_smaller_fall_is_a_gap_with_the_side_that_fell_more(agent):
    *_, gap = _session(agent, "fell-less", "五粮液今天涨跌幅", "茅台呢", "茅台少跌了多少")
    assert "frame:difference:pct_change:" in " ".join(gap["route_reasons"])
    assert "两者相差 0.36 个百分点" in gap["answer"]
    assert "五粮液跌得更多" in gap["answer"]


def test_an_english_gap_in_points_after_a_move(agent):
    *_, gap = _session(
        agent, "move-en", "How much did Wuliangye move yesterday?", "what about Ping An?", "so how many points apart?"
    )
    assert "a difference of 1.26 percentage points" in gap["answer"]
    assert gap.get("status") != "needs_clarification"


def test_which_is_higher_and_by_how_many_points_states_the_gap_without_a_judgment_hedge(agent):
    *_, gap = _session(
        agent, "which-gap", "Moutai's return on equity?", "Ping An's?", "Which is larger, and by how many points?"
    )
    assert "frame:difference:roe:" in " ".join(gap["route_reasons"])
    assert "a difference of 17.8 percentage points (Kweichow Moutai is higher)" in gap["answer"]
    assert "which one is better" not in gap["answer"]


def test_a_factual_which_one_is_not_a_judgment():
    from query_intelligence.answer_guards import _is_comparison_judgment_context

    nlu = {"question_style": "compare"}
    assert not _is_comparison_judgment_context(nlu, "Which one has the higher ROE?")
    assert _is_comparison_judgment_context(nlu, "Which one should I pick?")


# ---- holding values ----


@pytest.mark.parametrize(
    ("query", "shares"),
    [
        ("I hold 1,000 shares of Wuliangye. What's that worth at the last close?", 1000),
        ("手里有三万份证券ETF，按最新收盘价算值多少钱", 30000),
        ("I have 4000 units of the CSI 300 ETF; what are they worth at the close?", 4000),
        ("Wuliangye has 3.88 billion shares. What is its P/E?", None),
    ],
)
def test_round12_holding_value_requests(query, shares):
    from query_intelligence.agent.coverage import holding_value_request

    found = holding_value_request(query)
    assert (found[0] if found else None) == shares


def test_a_holding_is_carried_to_another_target(agent):
    first, carried, recounted = _session(
        agent,
        "holding-carry",
        "我手上有200股五粮液，按最近收盘算值多少",
        "要是换成同样数量的中国平安呢",
        "如果是500股茅台呢",
    )
    assert "200 × 100.64 = 20128 元" in first["answer"]
    assert any(reason.startswith("holding_follow_up:") for reason in carried["route_reasons"])
    assert "200 × 53.61 = 10722 元 [price_601318.SH]" in carried["answer"]
    assert "500 × 1409.5 = 704750 元 [price_600519.SH]" in recounted["answer"]
    for result in (carried, recounted):
        assert "fair_value_hedge" not in result["compliance_notes"]
        assert result["verification"]["passed"]


def test_an_english_holding_across_a_sentence_boundary(agent):
    result = agent.chat(
        "I hold 1,000 shares of Wuliangye. What's that worth at the last close?", session_id="r12-hold-en"
    )
    assert "1000 × 100.64 = CNY 100640 [price_000858.SZ]" in result["answer"]
    assert "fair value" not in result["answer"].lower().split("not a tradable price")[0]
    assert result["verification"]["passed"]


def test_fund_units_are_valued_as_units(agent):
    result = agent.chat("手里有三万份证券ETF，按最新收盘价算值多少钱", session_id="r12-hold-etf")
    assert "30000 份证券ETF的市值约为 30000 × 1.021 = 30630 元 [price_512880.SH]" in result["answer"]
    assert "fair_value_hedge" not in result["compliance_notes"]
    english = agent.chat(
        "I have 4000 units of the CSI 300 ETF; what are they worth at the close?", session_id="r12-hold-etf-en"
    )
    assert "4000 units of 沪深300ETF are worth 4000 × 4.811 = CNY 19244" in english["answer"]


# ---- net margin as a share of revenue (English), net-margin with a hyphen ----


def test_an_english_share_of_revenue_kept_as_net_profit(agent):
    *_, share = _session(
        agent,
        "share-en",
        "How much revenue did Wuliangye book last year?",
        "and its net profit?",
        "What share of that revenue is left as net profit?",
    )
    assert "≈ 34.84% [fundamental_000858.SZ]" in share["answer"]


def test_a_hyphenated_net_margin_gap_between_two_named_targets(agent):
    result = agent.chat("What's the net-margin gap between Wuliangye and Ping An?", session_id="r12-margin-en")
    assert "a gap of 24.91 percentage points (Wuliangye is higher)" in result["answer"]
    assert result["verification"]["passed"]


# ---- a Hong Kong listing described in words ----


@pytest.mark.parametrize(
    ("query", "h_share"),
    [
        ("它在港交所挂牌的那部分股票呢", True),
        ("中国平安在香港上市的股份怎么样", True),
        ("and its shares listed in Hong Kong?", True),
        ("中国平安A股收盘多少", False),
        ("香港的利率政策对A股有什么影响", False),
    ],
)
def test_hong_kong_listing_descriptions(query, h_share):
    from query_intelligence.agent.coverage import asks_h_share

    assert asks_h_share(query) is h_share


def test_a_hong_kong_listing_by_description_is_out_of_coverage(agent):
    _first, hk, back = _session(
        agent, "hk-words", "中国平安今天收盘多少", "它在港交所挂牌的那部分股票呢", "回到A股，它的市净率呢"
    )
    assert hk["route"] == "refuse" and hk["limitations"] == ["out_of_coverage"]
    assert "53.61" not in hk["answer"]
    assert "1.1 倍" in back["answer"]
    english = _session(agent, "hk-words-en", "What's Ping An's P/E?", "and its shares listed in Hong Kong?")[-1]
    assert english["route"] == "refuse" and "8.7" not in english["answer"]


# ---- round-8 review H1 / H2 / H11: the comparison parser, one-message comparisons, longer gap questions ----


@pytest.mark.parametrize(
    ("text", "operation"),
    [
        # the review's repros as regression rows
        ("前者比后者多成交了多少钱", "difference"),
        ("二者之比是多少", "ratio"),
        ("By what percentage is Wuliangye's price above Ping An's?", "relative"),
        ("Is the latter more than 1.2x the former?", "ratio"),
        ("PB差了多少倍", "ratio"),
        ("五粮液的PE比茅台低百分之多少", "relative"),
        ("how big is the discount? Someone told me Ping An trades at a 40% discount to peers", "relative"),
        # own wording
        ("后一只比前一只少赚了多少", "difference"),
        ("两者的比值大概多少", "ratio"),
        ("What multiple of Ping An's is Moutai's?", "ratio"),
        # one operand, no anchor: not a comparison
        ("五粮液市盈率是多少倍", None),
        ("茅台的ROE是百分之多少", None),
    ],
)
def test_comparison_parser(text, operation):
    from query_intelligence.agent.frame import frame_operation

    assert frame_operation(text) == operation


def test_ratio_direction_follows_the_question():
    from query_intelligence.agent.frame import resolve_frame_question

    entities = [
        {"canonical_name": "中国平安", "symbol": "601318.SH", "mention": "中国平安"},
        {"canonical_name": "贵州茅台", "symbol": "600519.SH", "mention": "贵州茅台"},
    ]
    turns = [{"query": "x", "entities": [], "frame": {"metric": "roe", "operands": []}}]
    *_, request = resolve_frame_question("how many times Ping An's is Moutai's?", turns, entities)
    assert [item["symbol"] for item in request["operands"]] == ["600519.SH", "601318.SH"]
    *_, request = resolve_frame_question("茅台的ROE是平安的几倍", [], entities)
    assert [item["symbol"] for item in request["operands"]] == ["600519.SH", "601318.SH"]


def test_one_message_comparisons_use_the_frame_computation(agent):
    relative = agent.chat("贵州茅台的市净率比五粮液高百分之多少", session_id="r12-h2-relative")
    assert "frame:relative:pb:贵州茅台|五粮液" in relative["route_reasons"]
    assert "贵州茅台比五粮液高约 50%（以五粮液为基数）" in relative["answer"]
    english = agent.chat("Between Ping An and Wuliangye, whose ROE is higher, and by how much?", session_id="r12-h2-en")
    assert "a difference of 14.2 percentage points (Wuliangye is higher)" in english["answer"]
    industry = agent.chat("五粮液的市净率比白酒行业平均低百分之多少", session_id="r12-h2-industry")
    assert "五粮液相对白酒行业平均折价约 12.9%" in industry["answer"]
    sectors = agent.chat("白酒行业和保险行业的平均市净率差多少", session_id="r12-h2-sectors")
    assert "白酒行业平均市净率 6.2 倍，保险行业平均 1.45 倍，两者相差 4.75" in sectors["answer"]


def test_a_metric_switch_keeps_the_pair_and_a_longer_gap_question_computes(agent):
    *_, gap = _session(agent, "h11-switch", "中国平安市盈率", "贵州茅台的呢", "那ROE呢", "相差几个点")
    assert "frame:difference:roe:中国平安|贵州茅台" in gap["route_reasons"]
    assert "两者相差 17.8 个百分点" in gap["answer"]
    *_, longer = _session(
        agent,
        "h11-long",
        "Wuliangye's P/E?",
        "and the baijiu sector average?",
        "what's the discount in percent? A friend said Wuliangye is far cheaper than its peers, maybe 30%",
    )
    assert "about 23.44% below" in longer["answer"]
