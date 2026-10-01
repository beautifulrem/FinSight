"""Round-13 rules: the classes the independent round-8 slice still failed after round 12, in the author's own wording
(other targets, metrics and phrasings than the slice; the slice is after exposure from round 13 on, and
``tests/test_agent_eval.py`` checks that no dev task copies it).

* Comparative verbs as a class (领先 / 落后 / 跑赢 / 跑输 / 超出 / 输给, "outperform", "trail", "out-earn", "out-sell")
  with a unit word or a quantity ask a difference; with "by what multiple" a ratio; a colloquial count of one operand in
  the other ("顶得上几个", "能抵几个") and imperatives ("divide … by …", "除以") a ratio; "subtract" / "减去" a
  difference. A resolvable comparison is computed, never refused as out of scope.
* A ratio question that states the multiple it checks ("大概是…的1.5倍吗") gets the ratio and whether it holds.
* Two metrics of one target ("这两项差几个点" after its ROE and gross margin) are compared with each other.
* Two industries named in one question, the NLU tagging only one of them; "closed higher / lower"; "percentage-wise".
* A total of two operands ("加在一起", "combined", "in total") for amounts.
* The cost of buying N shares or lots at the carried close ("照这个价买五手需要多少钱").
* An industry's turnover rate from its snapshot; a change in yuan derived from the close and the % change.
* Claims: "N倍多一点" (a little over N: up to 0.3 of a step), "N个多百分点", a sector's average written without
  "industry" ("the baijiu average"), English "less than half of X's", a multiple of another sector
  ("是保险行业的N倍多").
"""

from __future__ import annotations

import pytest

# ---- the comparison parser ----


@pytest.mark.parametrize(
    ("text", "operation"),
    [
        ("后一只超出前一只几个百分点", "difference"),
        ("沪深300ETF跑输创业板ETF几个点", "difference"),
        ("中国平安的市盈率落后五粮液多少", "difference"),
        ("贵州茅台输给五粮液多少个点", "difference"),
        ("按营收算，五粮液顶得上几个平安", "ratio"),
        ("论成交额，前者能抵几个后者", "ratio"),
        ("后者除以前者等于多少", "ratio"),
        ("Moutai out-sells Wuliangye by what multiple?", "ratio"),
        ("Does Wuliangye outperform Ping An by how many points?", "difference"),
        ("Take the first and divide it by the second.", "ratio"),
        ("Subtract the second from the first.", "difference"),
        ("茅台的市净率大概是五粮液的1.5倍吗", "ratio"),
        ("Is Moutai's revenue about twice Wuliangye's?", "ratio"),
        ("How much cheaper than the sector is it, percentage-wise?", "relative"),
        ("中国平安和五粮液的净利润加在一起是多少", "sum"),
        ("What do Ping An and Moutai earn in total?", "sum"),
        # not comparisons
        ("茅台市盈率是20倍吗", None),
        ("我手上有800股五粮液，合计值多少", None),
        ("五粮液今天领先大盘吗", None),
    ],
)
def test_round13_comparison_parser(text, operation):
    from query_intelligence.agent.frame import frame_operation

    assert frame_operation(text) == operation


@pytest.mark.parametrize(
    ("text", "metric"),
    [
        ("Moutai out-sells Wuliangye by what multiple?", "revenue"),
        ("What do Ping An and Moutai earn in total?", "net_profit"),
        ("Which of Moutai and Ping An closed lower?", "close"),
    ],
)
def test_round13_metric_words(text, metric):
    from query_intelligence.agent.frame import metric_of

    assert metric_of(text) == metric


def test_claimed_multiples_are_read_with_their_wording():
    from query_intelligence.agent.frame import claimed_multiple

    assert claimed_multiple("茅台的市净率大概是五粮液的1.5倍吗") == (1.5, "1.5", True)
    assert claimed_multiple("Is Moutai's revenue about twice Wuliangye's?") == (2.0, "twice", True)
    assert claimed_multiple("中国平安的营收是五粮液的十倍吗") == (10.0, "十", False)
    assert claimed_multiple("茅台市盈率是20倍吗") is None  # one operand: a value, not a multiple of another


def test_ordinals_without_a_noun_follow_the_frame_order():
    from query_intelligence.agent.frame import resolve_frame_question

    turns = [
        {
            "query": "x",
            "frame": {
                "metric": "pb",
                "operands": [
                    {"kind": "target", "name": "中国平安", "symbol": "601318.SH"},
                    {"kind": "target", "name": "五粮液", "symbol": "000858.SZ"},
                ],
            },
        }
    ]
    *_, request = resolve_frame_question("Take the first and divide it by the second.", turns, [])
    assert [item["symbol"] for item in request["operands"]] == ["601318.SH", "000858.SZ"]
    *_, request = resolve_frame_question("Subtract the first from the second.", turns, [])
    assert request["operation"] == "difference"


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
    return [agent.chat(query, session_id=f"r13-{name}") for query in queries]


def test_a_verb_comparative_follow_up_is_computed_not_refused(agent):
    *_, gap = _session(agent, "verb-zh", "贵州茅台的净资产收益率", "换五粮液看看", "后一只落后前一只几个百分点")
    assert gap["route"] != "refuse"
    assert "frame:difference:roe:五粮液|贵州茅台" in gap["route_reasons"]
    assert "两者相差 3.6 个百分点" in gap["answer"]
    assert gap["verification"]["passed"]


def test_an_english_verb_ratio_and_an_imperative_division(agent):
    *_, ratio = _session(
        agent, "verb-en", "Moutai's revenue?", "And for Wuliangye?", "Moutai out-sells it by what multiple?"
    )
    assert "frame:ratio:revenue:" in " ".join(ratio["route_reasons"])
    assert "1.56 times" in ratio["answer"]
    *_, divided = _session(
        agent, "divide-en", "Moutai's price-to-book?", "Ping An?", "Take the first and divide it by the second."
    )
    assert divided["route"] != "refuse"
    assert "7.36 times" in divided["answer"]


def test_a_colloquial_count_of_one_operand_in_the_other_is_a_ratio(agent):
    answer = agent.chat("按营收算，平安顶得上几个五粮液", session_id="r13-count")["answer"]
    assert "前者约为后者的 11.23 倍" in answer


def test_a_stated_multiple_is_checked(agent):
    checked = agent.chat("茅台的市净率大概是五粮液的1.5倍吗", session_id="r13-claimed")
    assert "前者约为后者的 1.5 倍" in checked["answer"]
    assert "是的，大约是1.5倍。" in checked["answer"]
    assert checked["verification"]["passed"]
    english = agent.chat("Is Ping An's revenue roughly 10 times Wuliangye's?", session_id="r13-claimed-en")
    assert "11.23 times" in english["answer"] and "So no: it is more than 10 times." in english["answer"]


def test_two_metrics_of_one_target_are_compared(agent):
    *_, gap = _session(agent, "two-rates", "五粮液净资产收益率是多少", "它毛利率是多少", "这两项差了几个百分点")
    assert "frame:difference:roe-gross_margin:五粮液" in gap["route_reasons"]
    assert "两者相差 46.7 个百分点" in gap["answer"]
    *_, margins = _session(agent, "two-margins", "贵州茅台的净利率", "那ROE呢", "两个比率差多少个点")
    assert "frame:difference:net_margin-roe:贵州茅台" in margins["route_reasons"]
    assert "823.2 亿元 ÷ 1688.38 亿元 ≈ 48.76%" in margins["answer"]
    assert "两者相差 15.76 个百分点" in margins["answer"] and margins["verification"]["passed"]


def test_two_industries_closed_lower_and_percentage_wise(agent):
    sectors = agent.chat("保险和白酒两个板块的平均市盈率相差多少", session_id="r13-sectors")
    assert "frame:difference:pe:保险行业平均|白酒行业平均" in sectors["route_reasons"]
    assert "两者相差 15.5" in sectors["answer"]
    lower = agent.chat("Which of Moutai and Ping An closed lower, and by how much?", session_id="r13-closed")
    assert "a difference of CNY 1355.89" in lower["answer"]
    *_, relative = _session(
        agent,
        "pct-wise",
        "Ping An's P/E?",
        "and its industry's?",
        "How much cheaper than the sector is it, percentage-wise?",
    )
    assert "about 26.27% below" in relative["answer"]


def test_a_total_of_two_amounts(agent):
    total = agent.chat("中国平安和五粮液的净利润加在一起是多少", session_id="r13-sum")
    assert "合计 1588 亿元" in total["answer"] and total["verification"]["passed"]
    english = agent.chat("What do Ping An and Moutai earn in total?", session_id="r13-sum-en")
    assert "combined CNY 203.32 bn" in english["answer"] and english["verification"]["passed"]
    multiples = agent.chat("茅台和五粮液的市盈率加起来多少", session_id="r13-sum-pe")
    assert "合计" not in multiples["answer"]  # a total of P/Es is not a value


def test_the_cost_of_buying_lots_at_the_carried_close(agent):
    _first, cost = _session(agent, "buy-lots", "五粮液最近收在多少", "照这个价买五手需要多少钱")
    assert cost["route"] != "refuse"
    assert "买入 5 手（500 股）五粮液约需 500 × 100.64 = 50320 元" in cost["answer"]
    assert "未含佣金" in cost["answer"] and cost["verification"]["passed"]
    english = agent.chat("How much would it cost to buy 400 Ping An shares at the last close?", session_id="r13-buy")
    assert "buying 400 shares of Ping An Insurance takes about 400 × 53.61 = CNY 21444" in english["answer"]


def test_holding_purchase_wording():
    from query_intelligence.agent.coverage import holding_lots, holding_purchase, holding_value_request

    assert holding_value_request("照这个价买五手需要多少钱")[0] == 500
    assert holding_lots("照这个价买五手需要多少钱") == 5
    assert holding_purchase("照这个价买五手需要多少钱")
    assert not holding_purchase("我有5手五粮液，按最新收盘值多少")
    assert holding_value_request("买五粮液划算吗") is None


def test_an_industry_turnover_rate_and_a_yuan_change_from_the_percent(agent):
    rate = agent.chat("保险行业的换手率是多少", session_id="r13-industry-rate")
    assert "换手率 0.43% [industry_保险]" in rate["answer"]
    assert "glossary" not in " ".join(rate["route_reasons"])
    concept = agent.chat("换手率是什么意思", session_id="r13-rate-concept")
    assert "glossary_换手率" in concept["answer"]
    yuan = agent.chat("五粮液最新一个交易日每股下跌了多少元", session_id="r13-yuan")
    assert "100.64 ÷ (1 − 0.5337%) ≈ 101.18 元，每股下跌约 0.54 元" in yuan["answer"]
    assert yuan["verification"]["passed"]


# ---- the verifier: the derivations the round-13 sentences write out ----


def test_the_verifier_accepts_written_out_round13_derivations():
    from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore
    from query_intelligence.agent.verifier import verify_answer

    store = EvidenceStore()
    price = {"close": 100.64, "pct_change_1d": -0.5337}
    store.add(AgentEvidence(evidence_id="price_x", kind="structured", source_type="market_api", payload=price))
    fundamentals = {"gross_margin": 76.1, "roe": 29.4, "net_profit": 37800000000, "revenue": 108500000000}
    store.add(
        AgentEvidence(evidence_id="fund_x", kind="structured", source_type="fundamental_sql", payload=fundamentals)
    )
    implied = (
        "收盘 100.64 元，涨跌幅 -0.5337%，前收约 100.64 ÷ (1 − 0.5337%) ≈ 101.18 元，每股下跌约 0.54 元 [price_x]。"
    )
    assert verify_answer({"answer": implied}, store, allow_derived=True).passed
    assert not verify_answer({"answer": implied.replace("0.54 元", "0.91 元")}, store, allow_derived=True).passed
    gap = "毛利率 76.1%，净利率 378 亿元 ÷ 1085 亿元 ≈ 34.84%，两者相差 41.26 个百分点 [fund_x]。"
    assert verify_answer({"answer": gap}, store, allow_derived=True).passed
    assert not verify_answer({"answer": gap.replace("41.26", "43.1")}, store, allow_derived=True).passed


# ---- claims ----


@pytest.fixture(scope="module")
def registry(offline_service):
    from query_intelligence.agent.tools import build_registry_for_service

    return build_registry_for_service(offline_service)


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("茅台净利润是五粮液的两倍多一点", "supported"),  # 2.18
        ("平安营收是茅台的七倍多一点", "supported"),  # 7.21
        ("五粮液营收是平安的两倍多一点", "contradicted"),
        ("茅台和平安的ROE相差十七个多百分点", "supported"),  # 17.8
        ("茅台和平安的ROE相差十八个多百分点", "contradicted"),
        ("Wuliangye's P/E sits roughly 23% below the baijiu average", "supported"),  # -23.4%
        ("Wuliangye's revenue is less than half of Moutai's", "contradicted"),  # 0.64
        ("Wuliangye makes less than half of Ping An's revenue", "supported"),  # 0.09
        ("白酒行业平均市盈率是保险行业的两倍多", "supported"),  # 2.31
        ("白酒行业平均市盈率是保险行业的三倍多", "contradicted"),
    ],
)
def test_round13_claim_rules(offline_service, registry, claim, verdict):
    from query_intelligence.agent.claim_check import check_claim

    zh = any("一" <= char <= "鿿" for char in claim)
    assert check_claim(claim, service=offline_service, registry=registry, zh=zh).verdict == verdict
