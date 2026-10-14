"""Round-18 rules for the round-14 review's N1, N2, N3, N4, N6 and N14, in the author's own wording (the review's
repro strings and its hidden slice are not used; ``tests/test_agent_eval.py`` checks that no string here copies a
committed slice, a quoted review probe or the independent operation wording set).

* N1: a position sold in full (全部卖掉 / 清仓 / 都卖了 / sold all / closed out …) is closed: the realised result is
  shares × (sale price − cost), nothing remains, and no paper P&L or holding value at the latest close is stated.
* N2: a comma is a thousands separator only as ``\\d{1,3}(,\\d{3})+``; a full-width "，" always ends the number, so
  "买入价38，1500股" is 1,500 shares at 38, never 381,500 shares.
* N3: a purchase written with a bare price before 加 / 加仓 / 补 / 又买 is a lot.
* N4: a cost stated with a holding in one turn carries into a P&L or break-even question in the next one; an English
  sale follow-up is a holding turn, not out of scope.
* N6: a threshold read from the question in tenths ("不到三成吗") is registered with the verifier, so the yes/no verdict
  is never deleted; on the LLM path "could not verify" is never followed by the number.
"""

from __future__ import annotations

import random
import re

import pytest

SEED = 18

# (symbol, Chinese name, English name, close on the shipped data at 2026-09-30)
STOCKS = [
    ("601899.SH", "紫金矿业", "Zijin Mining", 29.79),
    ("300750.SZ", "宁德时代", "CATL", 291.11),
    ("002594.SZ", "比亚迪", "BYD", 83.31),
    ("600036.SH", "招商银行", "China Merchants Bank", 41.26),
    ("601398.SH", "工商银行", "ICBC", 8.28),
    ("000333.SZ", "美的集团", "Midea Group", 80.1),
    ("600030.SH", "中信证券", "CITIC Securities Ltd", 25.86),
]


def _price_log(symbol: str, name: str, close: float) -> list[dict]:
    return [
        {
            "tool": "get_price_history",
            "ok": True,
            "arguments": {"target": symbol},
            "data": {
                "symbol": symbol,
                "name": name,
                "close": close,
                "as_of": "2026-09-30",
                "evidence_id": f"price_{symbol}",
            },
        }
    ]


def _store(tool_log: list[dict]):
    from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore

    store = EvidenceStore()
    for entry in tool_log:
        data = entry["data"]
        store.add(
            AgentEvidence(
                evidence_id=data["evidence_id"],
                source_type="fundamental" if entry["tool"] == "get_fundamentals" else "market",
                kind="structured",
                produced_by=entry["tool"],
                title=str(data.get("name")),
                payload={**(data.get("metrics") or {}), **{k: v for k, v in data.items() if k != "metrics"}},
            )
        )
    return store


def _price(rng: random.Random, close: float) -> float:
    return round(close * rng.uniform(0.75, 1.3), rng.choice([0, 1, 2]))


# ---- N1: a full sale closes the position ----

FULL_SALE_ZH = [
    "{name}当初{c}元建仓{n}股，后来以{p}元全部卖掉了，这笔交易赚了还是亏了，多少",
    "我{c}块买进{name}{n}股，前段时间{p}块全都卖了，算下来盈亏多少",
    "{name}{n}股成本{c}元，上个月在{p}元清仓了，一共赚了多少钱",
    "{name}的{n}股我已经清仓，清仓价{p}元，当时买入成本{c}元，赚了多少",
    "{name}：{c}元进了{n}股，{p}元一股都卖光了，到底亏了多少",
    "前年{c}元入手的{name}{n}股，今年{p}元全部抛了，收益多少",
    "{name}买入价{c}元，共{n}股，现在已经全部出掉了，卖出价{p}元，盈亏是多少",
]
FULL_SALE_EN = [
    "I picked up {n} {en} shares at {c} and sold the whole lot at {p}. What did I make?",
    "Bought {n} shares of {en} at {c}; closed out the position at {p}. Profit or loss?",
    "My {n} {en} shares, bought at {c}, were all sold at {p} last month. What was my gain?",
    "I exited my {en} position entirely at {p}; I had {n} shares at a cost of {c}. How much did I make or lose?",
    "Sold all {n} of my {en} shares at {p} after buying at {c}. What's the realised P&L?",
    "I had {n} {en} shares bought at {c} and closed the entire position at {p}. How much did I make?",
]


@pytest.mark.parametrize("language", ["zh", "en"])
def test_a_full_sale_closes_the_position_with_the_realised_result(language):
    from query_intelligence.agent.composer import cost_basis_sentences, cost_basis_values, sale_numbers
    from query_intelligence.agent.coverage import cost_basis_request
    from query_intelligence.agent.verifier import verify_answer

    zh = language == "zh"
    rng = random.Random(f"{SEED}-n1-{language}")
    for _ in range(70):
        symbol, name, english, close = rng.choice(STOCKS)
        bought = rng.choice(range(200, 6000, 100))
        cost, price = _price(rng, close), _price(rng, close)
        question = rng.choice(FULL_SALE_ZH if zh else FULL_SALE_EN).format(
            name=name, en=english, c=f"{cost:g}", n=bought, p=f"{price:g}"
        )
        asked = cost_basis_request(question)
        assert asked is not None, question
        assert asked["bought"] == bought and asked["shares"] == 0, (question, asked)
        assert asked["sales"] == [{"shares": bought, "price": price}], (question, asked)
        found = sale_numbers(asked, close)
        realised = round(bought * (price - cost), 2)
        assert found["sales"][0]["realised"] == realised and found["unrealised"] is None, question
        tool_log = _price_log(symbol, name, close)
        sentences = cost_basis_sentences(tool_log, question, zh)
        text = " ".join(sentences)
        assert f"{realised:g}" in text, (question, text)
        # nothing is valued at the latest close: no paper result, no remaining shares
        assert f"{close:g}" not in sentences[0], (question, sentences[0])
        assert ("没有剩余持仓" in text) if zh else ("nothing remains" in text), text
        assert not re.search(r"浮盈|浮亏|paper (?:gain|loss)", sentences[0]), sentences[0]
        report = verify_answer(
            {"answer": sentences[0]}, _store(tool_log), query=question, allow_derived=True,
            computed=cost_basis_values(question, tool_log),
        )  # fmt: skip
        assert report.passed, (question, report.unsupported_numbers, report.uncited_numbers)


@pytest.mark.parametrize(
    ("question", "zh"),
    [
        ("宁德时代我以280元买的，后来310元全部卖了，每股赚多少", True),
        ("I closed my CATL position at 310; my cost was 280. How much did I make per share?", False),
    ],
)
def test_a_full_sale_without_a_count_is_answered_per_share(question, zh):
    from query_intelligence.agent.composer import cost_basis_sentences
    from query_intelligence.agent.coverage import cost_basis_request

    asked = cost_basis_request(question)
    assert asked is not None and asked.get("closed") and asked["sales"][0]["price"] == 310.0, asked
    sentences = cost_basis_sentences(_price_log("300750.SZ", "宁德时代", 291.11), question, zh)
    text = " ".join(sentences)
    assert "30" in text and "291.11" not in sentences[0], text
    assert not re.search(r"浮盈|浮亏|paper (?:gain|loss)", sentences[0]), sentences[0]


@pytest.mark.parametrize(
    ("question", "zh"),
    [
        ("比亚迪我90元买了500股，上礼拜全部卖掉了，赚了多少", True),
        ("I bought 500 BYD at 90 and have since sold them all. How much did I make?", False),
    ],
)
def test_a_full_sale_without_its_price_says_the_price_is_missing(question, zh):
    from query_intelligence.agent.composer import cost_basis_sentences
    from query_intelligence.agent.coverage import cost_basis_request

    asked = cost_basis_request(question)
    assert asked is not None and asked["shares"] == 0 and asked["sales"] == [{"shares": 500, "price": None}], asked
    sentences = cost_basis_sentences(_price_log("002594.SZ", "比亚迪", 83.31), question, zh)
    assert sentences and "83.31" not in sentences[0], sentences
    assert ("没有给出卖出价格" in sentences[0]) if zh else ("no sale price" in sentences[0]), sentences[0]


@pytest.mark.parametrize(
    "question",
    [
        "我想清仓美的集团，现在卖的话每股能拿多少钱",  # a sale considered, not made
        "要不要把招商银行全部卖掉",  # advice, not a trade
        "如果明天全卖了能赚多少",  # hypothetical without a cost
        "美的集团我成本80元有1000股，要是现在全部卖掉能赚多少",  # hypothetical: the paper result at the close
        "I hold 700 CATL at 300. If I sold them all today, what would I make?",
    ],
)
def test_a_sale_that_is_not_a_closed_trade_is_not_a_closed_position(question):
    from query_intelligence.agent.coverage import cost_basis_request

    asked = cost_basis_request(question)
    assert asked is None or not asked.get("closed"), (question, asked)


# ---- N2: a comma between a price and a count ----

COMMA_FORMS = [
    "{name}买入价{c}，{n}股，现在值多少",
    "{name}成本{c}，{n}股，按收盘价算一共值多少",
    "我的{name}：均价{c}，持有{n}股，市值多少",
    "{name}进价{c}，{n}股，现在赚了多少",
]


def test_a_comma_between_a_price_and_a_count_never_merges_them():
    from query_intelligence.agent.coverage import cost_basis_request, count_value, holding_value_request

    rng = random.Random(f"{SEED}-n2")
    for _ in range(60):
        _symbol, name, _english, close = rng.choice(STOCKS)
        cost = round(close * rng.uniform(0.8, 1.2), rng.choice([0, 1, 2]))
        shares = rng.choice(range(100, 9000, 100))
        template = rng.choice(COMMA_FORMS)
        question = template.format(name=name, c=f"{cost:g}", n=shares)
        held = holding_value_request(question)
        asked = cost_basis_request(question)
        if held is not None:
            assert held[0] == shares, (question, held)
        if asked is not None:
            assert asked["shares"] == shares and asked["cost"] == cost, (question, asked)
        assert held is not None or asked is not None, question
    # ASCII thousands separators still read as one number; anything else is not a separator
    assert holding_value_request("我有12,500股工商银行，值多少")[0] == 12500
    assert holding_value_request("I hold 2,400 shares of BYD. What are they worth?")[0] == 2400
    assert count_value("3,000") == 3000


@pytest.mark.parametrize(
    ("question", "shares"),
    [
        ("美的集团买在76，2200股，现在值多少钱", 2200),
        ("招商银行买入价38,1500股，市值多少", 1500),  # an ASCII comma that is not a thousands group
        ("宁德时代成本265，1,200股，现在值多少", 1200),
    ],
)
def test_comma_forms_keep_the_count(question, shares):
    from query_intelligence.agent.coverage import holding_value_request

    held = holding_value_request(question)
    assert held is not None and held[0] == shares, (question, held)


# ---- N3: a bare price before 加 / 加仓 / 补 is a lot ----

SECOND_BUY = ["又加了", "加仓了", "又加仓", "补了", "又补了", "又买了", "再买了", "加了"]
PRICE_UNIT = ["元", "块", ""]


def test_a_second_purchase_with_a_bare_price_is_a_lot():
    from query_intelligence.agent.composer import sale_numbers
    from query_intelligence.agent.coverage import cost_basis_request, cost_lots

    rng = random.Random(f"{SEED}-n3")
    for verb in SECOND_BUY:
        for unit in PRICE_UNIT:
            _symbol, name, _english, close = rng.choice(STOCKS)
            first, second = rng.choice(range(200, 1500, 100)), rng.choice(range(100, 1000, 100))
            c1, c2 = _price(rng, close), _price(rng, close)
            sold = rng.choice(range(100, first + second, 100))
            price = _price(rng, close)
            question = (
                f"{name}我起初{c1:g}{unit}买了{first}股，之后{c2:g}{unit}{verb}{second}股，"
                f"最近{price:g}元卖出{sold}股，余下的现在是赚是亏，多少"
            )
            lots = cost_lots(question)
            assert lots == [{"cost": c1, "shares": first}, {"cost": c2, "shares": second}], (question, lots)
            asked = cost_basis_request(question)
            assert asked is not None and asked["bought"] == first + second, (question, asked)
            assert asked["shares"] == first + second - sold, (question, asked)
            average = (c1 * first + c2 * second) / (first + second)
            found = sale_numbers(asked, close)
            assert found["sales"][0]["realised"] == round(sold * (price - round(average, 4)), 2), question


def test_a_number_before_a_non_purchase_jia_is_not_a_lot():
    from query_intelligence.agent.coverage import cost_lots

    assert cost_lots("2024加了一倍的仓位，300股之后又是多少") == []
    assert cost_lots("比亚迪和宁德时代加起来300股，值多少") == []


# ---- N4: a stated cost carries into the next turn ----


def _turn(query: str, stock: tuple) -> dict:
    return {"query": query, "entities": [{"symbol": stock[0], "name": stock[1], "canonical_name": stock[1]}]}


@pytest.mark.parametrize(
    ("statement", "follow_up", "stock", "expected"),
    [
        ("美的集团我拿着1200股，成本价86元", "那按收盘价我现在亏多少", STOCKS[5], {"cost": 86.0, "shares": 1200}),
        ("中信证券有3500股在手上，均价29", "回本还得涨多少", STOCKS[6], {"cost": 29.0, "shares": 3500}),
        ("I'm holding 900 BYD shares bought at 95.", "How much am I down at the moment?", STOCKS[2],
         {"cost": 95.0, "shares": 900}),
        ("I own 4,000 ICBC; my cost was 7.1.", "I sold a quarter at 8.6. How am I doing on the remainder?",
         STOCKS[4], {"cost": 7.1, "bought": 4000, "shares": 3000}),
    ],
)  # fmt: skip
def test_a_stated_cost_carries_into_a_pnl_follow_up(statement, follow_up, stock, expected):
    from query_intelligence.agent.coverage import cost_basis_request
    from query_intelligence.agent.memory import resolve_holding_follow_up

    found = resolve_holding_follow_up(follow_up, [_turn(statement, stock)], [])
    assert found is not None and found[1].startswith("cost_carried:"), found
    asked = cost_basis_request(found[0])
    assert asked is not None, found
    for key, value in expected.items():
        assert asked.get(key) == value, (found[0], asked)


@pytest.mark.parametrize(
    ("statement", "follow_up"),
    [
        ("美的集团的收盘价多少", "那我亏多少"),  # no cost stated before
        ("我持有美的集团1200股，成本价86元", "比亚迪的市盈率呢"),  # another question
    ],
)
def test_no_cost_carry_without_a_stated_cost_or_a_pnl_question(statement, follow_up):
    from query_intelligence.agent.memory import resolve_holding_follow_up

    found = resolve_holding_follow_up(follow_up, [_turn(statement, STOCKS[5])], [])
    assert found is None or not found[1].startswith("cost_carried:"), found


# ---- N6: a threshold in tenths is registered with the verifier ----


@pytest.mark.parametrize(
    ("question", "threshold"),
    [
        ("招商银行的总市值是不是不到工商银行的四成", 40.0),
        ("比亚迪净利润有没有宁德时代的一半", 50.0),
        ("美的集团营收超过比亚迪的五成了吗", 50.0),
    ],
)
def test_a_threshold_in_tenths_is_a_question_number(question, threshold):
    from query_intelligence.agent.verifier import question_thresholds

    assert threshold in question_thresholds(question), question


def test_the_verdict_on_a_tenths_threshold_survives_the_verifier():
    from query_intelligence.agent.verifier import verify_answer

    tool_log = [
        {"tool": "get_fundamentals", "ok": True,
         "data": {"symbol": s, "name": n, "evidence_id": f"fundamental_{s}", "metrics": {"total_mv": mv}}}
        for s, n, mv in (("600036.SH", "招商银行", 1040570829497.26), ("601398.SH", "工商银行", 2951043808696.92))
    ]  # fmt: skip
    answer = (
        "招商银行总市值 10405.71 亿元，工商银行 29510.44 亿元，10405.71 亿元 ÷ 29510.44 亿元 ≈ 35.26%，即招商银行约为"
        "工商银行的 35.26% [fundamental_600036.SH][fundamental_601398.SH]。是的，不到四成。"
    )
    report = verify_answer(
        {"answer": answer}, _store(tool_log), query="招商银行的总市值不到工商银行的四成吗", allow_derived=True
    )
    assert report.passed, (report.unsupported_numbers, report.uncited_numbers)


# ---- end to end: deterministic path and LLM path (shipped default, offline) ----


@pytest.fixture(scope="module")
def agent(shipped_offline_service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    runtime = AgentRuntime(shipped_offline_service, build_registry_for_service(shipped_offline_service), None)
    yield AgentService(runtime, trace_sinks=[])
    runtime.close()


def _chat(agent, turns: list[str], session: str) -> dict:
    result: dict = {}
    for query in turns:
        result = agent.chat(query, session_id=session, mode="auto")
    return result


@pytest.mark.parametrize(
    ("turns", "present", "absent"),
    [
        (
            ["紫金矿业我当时26.5元建仓了3000股，前几天以31元全部卖掉了，这笔一共赚了多少"],
            ["3000 × (31 − 26.5) = 13500", "没有剩余持仓"],
            ["浮盈", "浮亏", "的市值约为"],
        ),
        (
            ["I bought 800 Midea Group shares at 72 and closed out the whole position at 79.5. What did I make?"],
            ["800 × (79.5 − 72) = CNY 6000", "nothing remains"],
            ["paper gain", "paper loss", "worth"],
        ),
        (
            ["招商银行买入价39，2600股，按最新收盘价值多少"],
            ["2600 × 41.26 = 107276"],
            ["392600"],
        ),
        (
            ["比亚迪收盘价是多少", "我先95买了600股，后来88又补了400股，最近90卖出500股，剩下的现在赚还是亏"],
            ["共买入 1000 股", "剩余 500 股"],
            ["共买入 600 股"],
        ),
        (
            ["美的集团我手上有1500股，成本价是84元", "照最新价格算我亏了多少"],
            ["1500 × (−3.9) = -5850"],
            [],
        ),
        (
            [
                "I have 5,000 shares of ICBC at a cost of 7.4",
                "I sold 2,000 of them at 8.2. How am I doing on the rest?",
            ],
            ["2000 × (8.2 − 7.4) = CNY 1600", "3000 × 0.88 = CNY 2640"],
            ["outside FinSight's scope"],
        ),
    ],
)
def test_end_to_end_holding_answers(agent, turns, present, absent):
    result = _chat(agent, turns, f"r18-{abs(hash(tuple(turns)))}")
    answer = str(result["answer"])
    for text in present:
        assert text in answer, (turns, answer)
    for text in absent:
        assert text not in answer, (turns, answer)


def test_end_to_end_a_tenths_threshold_keeps_its_verdict(agent):
    result = _chat(agent, ["比亚迪的总市值不到宁德时代的六成吗"], "r18-tenths")
    answer = str(result["answer"])
    assert "是的，不到六成" in answer, answer
    assert result["verification"]["passed"], result["verification"]


def _llm_agent(service, steps):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.state import AgentConfig
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    config = AgentConfig(planner_prefetch=True, revise_policy="cite_repair", verify_derived=True)
    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM(steps), config=config)
    return runtime, AgentService(runtime, trace_sinks=[])


def test_llm_a_refused_full_sale_gets_the_realised_result_not_a_paper_figure(shipped_offline_service):
    from query_intelligence.agent.llm import final_turn

    refusal = "买入价与卖出价是你的交易记录，行情数据中没有，无法核实；紫金矿业最新收盘价 29.79 元 [price_601899.SH]。"
    runtime, service = _llm_agent(shipped_offline_service, [final_turn({"answer": refusal, "evidence_used": []})])
    try:
        result = service.chat(
            "紫金矿业我24元买了2000股，上周按33元全卖了，总共赚了多少", session_id="r18-llm-sale", mode="agent"
        )
    finally:
        runtime.close()
    answer = str(result["answer"])
    assert "2000 × (33 − 24) = 18000" in answer, answer
    assert "浮亏" not in answer and "浮盈" not in answer and "的市值约为" not in answer, answer
    assert "cost_basis_result_appended" in (result.get("degraded") or [])


def test_llm_a_stated_cost_reaches_the_next_turn(shipped_offline_service):
    from query_intelligence.agent.llm import final_turn

    statement = "已记录你的持仓 [price_000333.SZ]。"
    refusal = "你没有给出持仓成本价和股数，无法计算盈亏；美的集团最新收盘价 80.1 元 [price_000333.SZ]。"
    steps = [final_turn({"answer": text, "evidence_used": []}) for text in (statement, refusal)]
    runtime, service = _llm_agent(shipped_offline_service, steps)
    try:
        service.chat("美的集团我拿了900股，均价83元", session_id="r18-llm-carry", mode="agent")
        result = service.chat("那现在浮亏多少", session_id="r18-llm-carry", mode="agent")
    finally:
        runtime.close()
    answer = str(result["answer"])
    assert "900 × (−2.9) = -2610" in answer, answer


def test_llm_could_not_verify_is_never_followed_by_the_number(shipped_offline_service, monkeypatch):
    """The template is replaced by one that does not state the asked share, so the failed draft gets the "could not
    verify" note; the frame's verified sentence appended afterwards replaces the note instead of following it."""
    from query_intelligence.agent.llm import final_turn

    wrong = "比亚迪总市值约为宁德时代的 71.3%，不到六成的说法不成立 [fundamental_002594.SZ]。"
    steps = [final_turn({"answer": wrong, "evidence_used": []}) for _ in range(3)]
    runtime, service = _llm_agent(shipped_offline_service, steps)
    other = "比亚迪最新收盘价 83.31 元 [price_002594.SZ]。"
    monkeypatch.setattr(
        runtime, "_template_update", lambda state: {"draft": {"answer": other, "evidence_used": ["price_002594.SZ"]}}
    )
    try:
        result = service.chat("比亚迪的总市值不到宁德时代的六成吗", session_id="r18-llm-cnv", mode="agent")
    finally:
        runtime.close()
    answer = str(result["answer"])
    assert "71.3" not in answer, answer
    share = re.search(r"≈\s*56\.39%", answer)
    if share is not None:
        assert "未能核实" not in answer, answer
    else:
        assert "未能核实" in answer and not re.search(r"未能核实.*\d", answer.split("。")[0]), answer


# ---- N14: correction turns, nicknames, short positions, weights, the 52-week spread and position, total shares ----


@pytest.mark.parametrize(
    ("turns", "present", "absent"),
    [
        (["美的集团的ROE有多高", "哦不对，我是说比亚迪"], ["13.24%"], ["19.69%"]),
        (["紫金矿业的市盈率", "说错了，应该是中信证券"], ["10.19 倍"], ["11.71 倍"]),
        (["招商银行的市净率", "不是问招商银行，是问工商银行的市盈率"], ["7.89 倍"], ["6.86 倍"]),
        (["What is BYD's P/B?", "Sorry, I meant Midea Group"], ["2.89x"], ["3.17x"]),
    ],
)
def test_a_correction_turn_keeps_the_metric_for_the_corrected_target(agent, turns, present, absent):
    result = _chat(agent, turns, f"r18-fix-{abs(hash(tuple(turns)))}")
    answer = str(result["answer"])
    assert any(reason.startswith("correction:") for reason in result.get("route_reasons") or []), result
    for text in present:
        assert text in answer, (turns, answer)
    for text in absent:
        assert text not in answer, (turns, answer)


def test_the_nickname_rows_collide_with_no_other_alias_or_name():
    import csv
    from pathlib import Path

    from query_intelligence.runtime_entity_assets import COLLOQUIAL_ALIASES

    root = Path(__file__).resolve().parents[1] / "data" / "runtime"
    with (root / "alias_table.csv").open(encoding="utf-8", newline="") as handle:
        aliases = list(csv.DictReader(handle))
    with (root / "entity_master.csv").open(encoding="utf-8", newline="") as handle:
        entities = list(csv.DictReader(handle))
    by_symbol = {row["symbol"]: row["entity_id"] for row in entities}
    for symbol, nickname in (("601398.SH", "宇宙行"), ("601318.SH", "平保")):
        assert nickname in COLLOQUIAL_ALIASES[symbol][1]
        owners = {row["entity_id"] for row in aliases if row["normalized_alias"] == nickname}
        assert owners == {by_symbol[symbol]}, (nickname, owners)
        assert not [row for row in entities if nickname in row["canonical_name"]], nickname
        longer = [row for row in aliases if nickname in row["normalized_alias"] != nickname]
        assert not longer, nickname


@pytest.mark.parametrize(
    ("question", "symbol"),
    [("宇宙行最近一个交易日收盘多少", "601398.SH"), ("平保的市净率有多高", "601318.SH")],
)
def test_nicknames_name_their_company(shipped_offline_service, question, symbol):
    nlu = shipped_offline_service.analyze_query(question)
    assert symbol in {entity.get("symbol") for entity in nlu["entities"]}, nlu["entities"]


@pytest.mark.parametrize(
    ("turns", "present"),
    [
        (["我做空了300股美的集团，卖空价格85元，现在浮盈多少"], ["85 − 80.1 = 4.9", "300 × 4.9 = 1470"]),
        (["招商银行我融券卖空了1000股，融券卖出价40元", "现在盈亏怎么样"],
         ["40 − 41.26 = -1.26", "1000 × (−1.26) = -1260"]),
        (["I shorted 2,000 ICBC at 7.9. What's my P&L at the close?"], ["7.9 − 8.28 = CNY -0.38", "2000 × (−0.38)"]),
    ],
)  # fmt: skip
def test_a_short_position_gains_when_the_price_falls(agent, turns, present):
    result = _chat(agent, turns, f"r18-short-{abs(hash(tuple(turns)))}")
    answer = str(result["answer"])
    for text in present:
        assert text in answer, (turns, answer)
    assert result["verification"]["passed"], result["verification"]


def test_a_short_is_never_read_as_a_purchase_cost():
    from query_intelligence.agent.coverage import cost_basis_request, short_position_request

    question = "比亚迪我卖空了500股，卖空价格92元，现在赚了多少"
    assert cost_basis_request(question) is None
    assert short_position_request(question) == {"price": 92.0, "shares": 500}


@pytest.mark.parametrize(
    ("question", "present"),
    [
        ("我持有美的集团500股和比亚迪600股，两只各占我仓位的比例是多少",
         ["40050 ÷ 90036 ≈ 44.48%", "49986 ÷ 90036 ≈ 55.52%"]),
        ("I own 300 shares of CATL and 2,000 shares of Zijin Mining; what are the portfolio weights?",
         ["87333 ÷ 146913 ≈ 59.45%", "59580 ÷ 146913 ≈ 40.55%"]),
    ],
)  # fmt: skip
def test_portfolio_weights_of_stated_positions(agent, question, present):
    result = _chat(agent, [question], f"r18-weight-{abs(hash(question))}")
    answer = str(result["answer"])
    for text in present:
        assert text in answer, answer
    assert result["verification"]["passed"], result["verification"]


@pytest.mark.parametrize(
    ("question", "present"),
    [
        ("中信证券近52周最高收盘和最低收盘相差多少", ["30.3 − 23.92 = 6.38 元"]),
        ("长江电力眼下处在52周股价区间的哪个位置", ["(28.54 − 25.65) ÷ (29.49 − 25.65) ≈ 75.26%"]),
        ("What's the gap between CMB's 52-week high and low?", ["43.45 − 35.5 = CNY 7.95"]),
        ("Where is the stock price of Midea Group trading within its 52-week range?", ["≈ 48.37%"]),
    ],
)
def test_the_52_week_spread_and_position(agent, question, present):
    result = _chat(agent, [question], f"r18-52w-{abs(hash(question))}")
    answer = str(result["answer"])
    for text in present:
        assert text in answer, answer
    assert result["verification"]["passed"], result["verification"]


def test_the_total_share_count_is_implied_and_labelled(agent):
    result = _chat(agent, ["长江电力的总股本有多少"], "r18-shares")
    answer = str(result["answer"])
    assert "÷ 28.54 元 ≈ 244.68 亿股" in answer and "推算值" in answer, answer
    assert result["verification"]["passed"], result["verification"]


def test_the_users_own_trade_arithmetic_is_accepted_without_market_data():
    """(N1 on the LLM path) A draft that computes the realised result from the user's own buy and sell figures is not
    an unverifiable market claim: it survives the verifier even in a sentence citing no evidence."""
    from query_intelligence.agent.verifier import verify_answer

    question = "中信证券我当初24.5元买了4000股，上个月27元全部抛了，到底赚了多少"
    tool_log = _price_log("600030.SH", "中信证券", 25.86)
    draft = (
        "按你给出的买入价 24.5 元和卖出价 27 元，4000 股全部卖出的已实现盈利为 4000 × (27 − 24.5) = 10000 元。"
        "中信证券最新收盘价 25.86 元 [price_600030.SH]。"
    )
    report = verify_answer({"answer": draft}, _store(tool_log), query=question, allow_derived=True)
    assert report.passed, (report.unsupported_numbers, report.uncited_numbers)
    wrong = draft.replace("10000", "13440")
    assert not verify_answer({"answer": wrong}, _store(tool_log), query=question, allow_derived=True).passed


@pytest.mark.parametrize(
    ("question", "cost", "shares"),
    [
        ("I'm sitting on 250 Moutai shares with an average cost of 1,480; how far down am I?", 1480.0, 250),
        ("贵州茅台成本价1,455元，持有120股，现在浮亏多少", 1455.0, 120),
        ("Bought 40 Moutai at 1,390 and sold them all at 1,425. What did I make?", 1390.0, 0),
    ],
)
def test_a_cost_with_a_thousands_separator_is_one_number(question, cost, shares):
    from query_intelligence.agent.coverage import cost_basis_request

    asked = cost_basis_request(question)
    assert asked is not None and asked["cost"] == cost and asked["shares"] == shares, (question, asked)


def test_a_correction_opening_a_conversation_asks_only_the_corrected_target(agent):
    result = _chat(agent, ["不是问工商银行，是问招商银行的市净率"], "r18-fix-open")
    answer = str(result["answer"])
    assert "0.91 倍" in answer and "0.74 倍" not in answer, answer
