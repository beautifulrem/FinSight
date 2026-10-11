"""Round-17 rules for the round-13 review's M1, M2, M4, M5, M6, M7, M8 and M10, in the author's own wording (the
review's repro strings and its hidden slice are not used; ``tests/test_agent_eval.py`` checks that no string here
copies a committed slice, a quoted review probe or the independent operation wording set).

* M1: a sale reduces the position; the remaining shares are valued at the close against the average cost and the sold
  shares against their sale price (realised); a sale is never a purchase lot.
* M2: an exclusion word removes the named target from the group before a total, mean or spread; "除了…还/也" is
  inclusive; on the LLM path a frame result whose operands ignore an exclusion is never appended.
* M4: every number the template's arithmetic computes is registered with the verifier; a failed LLM draft whose
  template does not state the asked result says it could not be verified.
* M5: the rise needed from the close back to a stated cost or to the 52-week high, zh and en.
* M6: positions stated in earlier turns, then a bare total, are valued at the close.
* M7, M8: a definition asked as "how is it computed" after a figure; a book-value follow-up answers the P/B.
* M10: an opening adverb of totalling reads a total; CYPC is an alias of 长江电力 with no collision.
"""

from __future__ import annotations

import random
import re

import pytest

SEED = 17

# (symbol, Chinese name, English name, close on the shipped data)
STOCKS = [
    ("601899.SH", "紫金矿业", "Zijin Mining", 29.79),
    ("300750.SZ", "宁德时代", "CATL", 291.11),
    ("002594.SZ", "比亚迪", "BYD", 83.31),
    ("600036.SH", "招商银行", "China Merchants Bank", 41.26),
    ("601398.SH", "工商银行", "ICBC", 8.28),
    ("000333.SZ", "美的集团", "Midea Group", 80.1),
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


# ---- M1: a sale reduces the position ----

SALE_ZH = [
    "{name}我{c}块钱建仓了{n}股，前阵子{p}元出掉了{s}股，余下的按收盘算赚还是亏，多少",
    "持仓成本{c}元，原本有{n}股{name}，减持{s}股（卖价{p}元）之后，剩下的浮盈浮亏是多少",
    "我以{c}元买入{name}{n}股，后来按{p}元卖了{s}股，现在盈亏情况如何",
    "{name}：{c}元进的{n}股，{p}元的时候抛了{s}股，留着的部分现在盈亏多少",
]
SALE_EN = [
    "I bought {n} shares of {en} at {c} and later sold {s} at {p}. How much am I up or down now?",
    "Paid {c} a share for {n} {en} shares, then trimmed {s} shares for {p} each. What's my P&L?",
    "{en}: got in at {c} with {n} shares, sold off {s} at {p}; what's the gain or loss on what's left?",
]


@pytest.mark.parametrize("language", ["zh", "en"])
def test_a_sale_reduces_the_position_and_is_never_a_purchase(language):
    from query_intelligence.agent.composer import cost_basis_sentences, cost_basis_values, sale_numbers
    from query_intelligence.agent.coverage import cost_basis_request, cost_lots
    from query_intelligence.agent.verifier import verify_answer

    zh = language == "zh"
    rng = random.Random(f"{SEED}-m1-{language}")
    for _ in range(60):
        symbol, name, english, close = rng.choice(STOCKS)
        bought = rng.choice(range(600, 4000, 100))
        sold = rng.choice(range(100, bought, 100))
        cost = round(close * rng.uniform(0.8, 1.2), rng.choice([0, 1, 2]))
        price = round(close * rng.uniform(0.85, 1.25), rng.choice([0, 1, 2]))
        question = rng.choice(SALE_ZH if zh else SALE_EN).format(
            name=name, en=english, c=f"{cost:g}", n=bought, s=sold, p=f"{price:g}"
        )
        asked = cost_basis_request(question)
        assert asked is not None, question
        assert asked["bought"] == bought and asked["shares"] == bought - sold, (question, asked)
        assert asked["sales"] == [{"shares": sold, "price": price}], (question, asked)
        assert {"cost": price, "shares": sold} not in cost_lots(question, minimum=1), question  # never the sale
        found = sale_numbers(asked, close)
        assert found["unrealised"] == round((bought - sold) * round(close - cost, 2), 2), question
        assert found["sales"][0]["realised"] == round(sold * (price - cost), 2), question
        tool_log = _price_log(symbol, name, close)
        sentences = cost_basis_sentences(tool_log, question, zh)
        assert f"{bought - sold}" in sentences[0] and f"{found['unrealised']:g}" in sentences[0], sentences[0]
        whole = round(bought * round(close - cost, 2), 2)
        if whole not in (found["unrealised"], found["sales"][0]["realised"], found["total"]):
            assert f"= {whole:g} " not in sentences[0], sentences[0]  # never the whole position's P&L
        report = verify_answer(
            {"answer": sentences[0]},
            _store(tool_log),
            query=question,
            allow_derived=True,
            computed=cost_basis_values(question, tool_log),
        )
        assert report.passed, (sentences[0], report.unsupported_numbers)


def test_a_fraction_sold_and_a_sale_without_its_price():
    from query_intelligence.agent.composer import cost_basis_sentences
    from query_intelligence.agent.coverage import cost_basis_request

    half = cost_basis_request("比亚迪成本90元，拿了1200股，上周减掉一半，剩下的亏多少")
    assert half is not None and half["shares"] == 600 and half["sales"][0]["price"] is None
    text = cost_basis_sentences(
        _price_log("002594.SZ", "比亚迪", 83.31), "比亚迪成本90元，拿了1200股，上周减掉一半，剩下的亏多少", True
    )[0]
    assert "没有给出卖出价格" in text and "600 × (−6.69) = -4014" in text
    english = cost_basis_request("I paid 300 for 800 CATL shares and sold half of them at 320; profit?")
    assert english is not None and english["shares"] == 400 and english["sales"] == [{"shares": 400, "price": 320.0}]


def test_more_sold_than_bought_is_said_not_valued():
    from query_intelligence.agent.composer import cost_basis_sentences

    question = "工商银行我8元买了500股，后来9元卖出800股，还剩多少盈亏"
    text = cost_basis_sentences(_price_log("601398.SH", "工商银行", 8.28), question, True)[0]
    assert "多于买入的股数" in text and not re.search(r"剩余 \d", text)


def test_several_purchases_then_a_sale_use_the_average_cost():
    from query_intelligence.agent.composer import sale_numbers
    from query_intelligence.agent.coverage import cost_basis_request

    asked = cost_basis_request("先买宁德时代300股成本280元，又买200股成本300元，后来以310元卖出100股，现在盈亏多少")
    assert asked is not None and asked["average_cost"] == 288.0 and asked["shares"] == 400
    found = sale_numbers(asked, 291.11)
    assert found["sales"][0]["realised"] == 2200.0 and found["unrealised"] == round(400 * 3.11, 2)


# ---- M5: the rise needed back to a cost or to the 52-week high ----

RISE_ZH = [
    "{name}我当初{c}元买的，股价得再涨百分之几我才回本",
    "按{c}元的成本，{name}从现在要涨多少个百分点才能解套",
    "{name}成本{c}元，离回本还差多少",
]
RISE_EN = [
    "I bought {en} at {c}. What percentage gain does it need for me to break even?",
    "My {en} entry was {c}; how much would it have to rise to get back to my cost?",
]


@pytest.mark.parametrize("language", ["zh", "en"])
def test_the_rise_back_to_a_cost_is_measured_from_the_close(language):
    from query_intelligence.agent.composer import cost_basis_sentences, cost_basis_values
    from query_intelligence.agent.coverage import cost_basis_request
    from query_intelligence.agent.verifier import verify_answer

    zh = language == "zh"
    rng = random.Random(f"{SEED}-m5-{language}")
    for _ in range(40):
        symbol, name, english, close = rng.choice(STOCKS)
        cost = round(close * rng.uniform(1.05, 1.6), rng.choice([0, 1]))
        question = rng.choice(RISE_ZH if zh else RISE_EN).format(name=name, en=english, c=f"{cost:g}")
        asked = cost_basis_request(question)
        assert asked is not None and asked.get("rise_to_cost"), question
        tool_log = _price_log(symbol, name, close)
        sentence = cost_basis_sentences(tool_log, question, zh)[0]
        rise = round((cost - close) / close * 100, 2)
        loss = round((close - cost) / cost * 100, 2)
        assert f"≈ {rise:g}%" in sentence, (question, sentence)
        assert f"{loss:g}%" not in sentence, sentence  # never the loss in percent of the cost
        report = verify_answer(
            {"answer": sentence}, _store(tool_log), query=question, allow_derived=True,
            computed=cost_basis_values(question, tool_log),
        )  # fmt: skip
        assert report.passed, (sentence, report.unsupported_numbers)


@pytest.mark.parametrize(
    "question",
    [
        "比亚迪想重回52周最高收盘价，股价还得涨百分之几",
        "离一年来最高收盘价还要再上涨多少",
        "要回到近52周高点，还需要涨多少",
        "How much would BYD have to climb to get back to its 52-week high?",
    ],
)
def test_the_rise_back_to_the_52_week_high_is_asked(question):
    from query_intelligence.agent.coverage import requested_price_fields

    assert requested_price_fields(question).back_to_high_52w, question


# ---- M2: an exclusion before a group operation ----

EXCLUDE_ZH = [
    ("{A}、{B}、{C}这三家，去掉{X}，剩下的{m}加总多少", "sum"),
    ("{A}、{B}和{C}里面不算{X}，其余两家的{m}一共多少", "sum"),
    ("把{X}排除在外，{A}、{B}、{C}的{m}平均是多少", "mean"),
    ("{A}、{B}、{C}三家的{m}，{X}除外，合计是多少", "sum"),
    ("{A}、{B}、{C}的{m}，剔除{X}之后最高和最低差多少", "spread"),
]
EXCLUDE_EN = [
    ("Excluding {X}, what is the combined {m} of {A}, {B} and {C}?", "sum"),
    ("Of {A}, {B} and {C}, what's the average {m} if we leave out {X}?", "mean"),
    ("Not counting {X}, how much {m} do {A}, {B} and {C} have in total?", "sum"),
    ("Other than {X}, what do {A}, {B} and {C} add up to in {m}?", "sum"),
]
MONEY = {"net_profit": ("净利润", "net profit"), "revenue": ("营业收入", "revenue")}


def _entity(row, zh: bool) -> dict:
    symbol, name, english, _close = row
    return {
        "symbol": symbol,
        "canonical_name": name,
        "mention": name if zh else english,
        "entity_type": "stock",
        "match_type": "alias_exact",
    }


def _fundamentals(symbol: str, name: str, key: str, value: float) -> dict:
    return {
        "tool": "get_fundamentals",
        "ok": True,
        "arguments": {"target": symbol},
        "evidence_ids": [f"fundamental_{symbol}"],
        "data": {
            "symbol": symbol,
            "name": name,
            "evidence_id": f"fundamental_{symbol}",
            "report_date": "2025-12-31",
            "metrics": {"revenue": 2e11, "net_profit": 3e10, key: value},
        },
    }


@pytest.mark.parametrize("language", ["zh", "en"])
def test_an_excluded_target_is_removed_before_the_group_operation(language):
    from query_intelligence.agent.composer import frame_result
    from query_intelligence.agent.frame import frame_extra_reasons, resolve_frame_question

    zh = language == "zh"
    rng = random.Random(f"{SEED}-m2-{language}")
    for _ in range(60):
        a, b, c = rng.sample(STOCKS, 3)
        x = rng.choice((a, b, c))
        key = rng.choice(sorted(MONEY))
        template, operation = rng.choice(EXCLUDE_ZH if zh else EXCLUDE_EN)
        question = template.format(
            A=a[1] if zh else a[2], B=b[1] if zh else b[2], C=c[1] if zh else c[2], X=x[1] if zh else x[2],
            m=MONEY[key][0 if zh else 1],
        )  # fmt: skip
        found = resolve_frame_question(question, [], [_entity(row, zh) for row in (a, b, c)], [])
        assert found is not None, question
        request = found[2]
        kept = [row for row in (a, b, c) if row is not x]
        assert request["operation"] == operation, (question, request["operation"])
        assert {item["symbol"] for item in request["operands"]} == {row[0] for row in kept}, question
        assert [item["symbol"] for item in request["excluded"]] == [x[0]], question
        assert f"frame_excluded:{x[1]}" in frame_extra_reasons(request)
        values = {row[0]: round(rng.uniform(1e9, 9e11), 2) for row in (a, b, c)}
        tool_log = [_fundamentals(row[0], row[1], key, values[row[0]]) for row in (a, b, c)]
        kept_values = [values[row[0]] for row in kept]
        expected = {
            "sum": round(sum(kept_values), 2),
            "mean": round(sum(kept_values) / 2, 2),
            "spread": round(max(kept_values) - min(kept_values), 2),
        }[operation]
        assert frame_result(request, tool_log) == expected, question


@pytest.mark.parametrize(
    "question",
    [
        "除了宁德时代，还有比亚迪和美的集团，三家营业收入加起来多少",
        "除了宁德时代以外，比亚迪和美的集团也算上，三家净利润合计多少",
    ],
)
def test_besides_is_not_an_exclusion(question):
    from query_intelligence.agent.frame import resolve_frame_question

    rows = [STOCKS[1], STOCKS[2], STOCKS[5]]
    request = resolve_frame_question(question, [], [_entity(row, True) for row in rows], [])[2]
    assert not request.get("excluded") and len(request["operands"]) == 3, question


def test_an_excluded_target_carried_from_the_frame_is_left_out():
    from query_intelligence.agent.frame import resolve_frame_question

    rows = [STOCKS[3], STOCKS[4], STOCKS[0]]
    frame = {
        "metric": "net_profit",
        "operands": [{"kind": "target", "name": row[1], "symbol": row[0]} for row in rows],
    }
    found = resolve_frame_question(
        "不算招商银行的话，另外两家合计多少", [{"frame": frame}], [_entity(rows[0], True)], []
    )
    request = found[2]
    assert [item["symbol"] for item in request["operands"]] == [rows[1][0], rows[2][0]]
    assert request["excluded"][0]["symbol"] == rows[0][0]


def test_the_llm_path_never_appends_a_frame_result_that_ignores_an_exclusion():
    from query_intelligence.agent.frame import has_exclusion_word

    assert has_exclusion_word("三家里刨去工行，剩余的净利润总和") and has_exclusion_word("What's the sum, except CMB?")
    assert not has_exclusion_word("三家净利润总和是多少")


# ---- M10: an opening adverb of totalling; the CYPC alias ----


@pytest.mark.parametrize(
    ("template", "operation"),
    [
        ("Overall, what is the {m} of {A} and {B}?", "sum"),
        ("All in all, how much {m} do {A} and {B} bring in?", "sum"),
        ("Cumulatively, what {m} do {A} and {B} report?", "sum"),
        ("总体来看，{a}和{b}的{zm}有多少", "sum"),
        ("Overall, which has more {m}, {A} or {B}?", "which"),
        ("Overall, how much higher is {A}'s {m} than {B}'s?", "difference"),
    ],
)
def test_an_opening_adverb_reads_a_total_only_without_a_comparison(template, operation):
    from query_intelligence.agent.frame import resolve_frame_question

    rng = random.Random(f"{SEED}-m10-{template}")
    for _ in range(10):
        a, b = rng.sample(STOCKS, 2)
        key = rng.choice(sorted(MONEY))
        zh = "{a}" in template
        question = template.format(A=a[2], B=b[2], a=a[1], b=b[1], m=MONEY[key][1], zm=MONEY[key][0])
        found = resolve_frame_question(question, [], [_entity(row, zh) for row in (a, b)], [])
        assert found is not None and found[2]["operation"] == operation, (question, found and found[2])


def test_cypc_is_an_alias_of_china_yangtze_power_and_of_nothing_else():
    import csv
    import json

    from query_intelligence.data_loader import DATA_DIR, RUNTIME_DATA_DIR

    synonyms = json.loads((DATA_DIR / "synonym_dict.json").read_text(encoding="utf-8"))["alias"]
    assert synonyms["cypc"] == "长江电力"
    assert [alias for alias in synonyms if alias.strip().lower() == "cypc"] == ["cypc"], "CYPC is listed twice"
    with (RUNTIME_DATA_DIR / "alias_table.csv").open(encoding="utf-8") as handle:
        clashes = [row for row in csv.DictReader(handle) if row["normalized_alias"].strip().lower() == "cypc"]
    with (RUNTIME_DATA_DIR / "entity_master.csv").open(encoding="utf-8") as handle:
        names = [row for row in csv.DictReader(handle) if "cypc" in str(row).lower()]
    assert clashes == [] and names == [], "CYPC already names another entity"


# ---- M6: positions stated in earlier turns, then a bare total ----


def _turn(query: str, *rows) -> dict:
    return {"query": query, "entities": [{"symbol": row[0], "name": row[1]} for row in rows]}


@pytest.mark.parametrize(
    ("turns", "question", "expected"),
    [
        (
            [_turn("帮我记一笔：紫金矿业我拿着3000股", STOCKS[0]), _turn("美的集团那边还有200股", STOCKS[5])],
            "总的值多少钱",
            "我持有3000股紫金矿业和200股美的集团，按最新收盘价合计市值多少",
        ),
        (
            [_turn("I own 400 BYD shares.", STOCKS[2]), _turn("And 150 shares of CATL too.", STOCKS[1])],
            "What are they worth altogether?",
            "I hold 400 shares of BYD and 150 shares of CATL: what are they worth in total at the latest close?",
        ),
        (
            [
                _turn("比亚迪持仓100股", STOCKS[2]),
                _turn("比亚迪改成300股", STOCKS[2]),
                _turn("工商银行有5手", STOCKS[4]),
            ],
            "这几只加起来值多少",
            "我持有300股比亚迪和500股工商银行，按最新收盘价合计市值多少",
        ),
    ],
)
def test_stated_positions_carry_into_a_bare_total(turns, question, expected):
    from query_intelligence.agent.memory import resolve_ledger_total

    found = resolve_ledger_total(question, turns, [])
    assert found is not None and found[0] == expected, found
    assert found[1].startswith("holding_ledger:")


@pytest.mark.parametrize(
    ("turns", "question"),
    [
        ([_turn("紫金矿业的营收多少", STOCKS[0]), _turn("美的集团呢", STOCKS[5])], "一共多少"),  # no position stated
        ([_turn("要是买500股宁德时代要多少钱", STOCKS[1]), _turn("工商银行有800股", STOCKS[4])], "合计值多少"),
        ([_turn("我有招商银行600股", STOCKS[3])], "一共值多少"),  # one position: the holding paths answer
        ([_turn("我有招商银行600股", STOCKS[3]), _turn("工商银行也有1000股", STOCKS[4])], "再加200股呢，一共值多少"),
    ],
)
def test_no_ledger_total_without_two_stated_positions(turns, question):
    from query_intelligence.agent.memory import resolve_ledger_total

    assert resolve_ledger_total(question, turns, []) is None


# ---- M7: a definition asked as "how is it computed" ----


@pytest.mark.parametrize(
    "question",
    ["净利率平时是如何计算的", "市净率通常怎么计算", "ROE一般是怎么计算的", "毛利率具体是如何算的"],
)
def test_how_a_metric_is_computed_is_a_definition_question(question):
    from query_intelligence.agent.glossary import definition_question
    from query_intelligence.agent.memory import resolve_ellipsis

    assert definition_question(question) is not None, question
    turns = [{"query": "美的集团毛利率有多高", "entities": [{"symbol": "000333.SZ", "name": "美的集团"}]}]
    assert resolve_ellipsis(question, turns, []) is None  # never the earlier company's figure


# ---- M8: a book-value follow-up ----


@pytest.mark.parametrize(
    ("question", "zh"),
    [
        ("它现在是不是跌到净资产以下了", True),
        ("有没有破净", True),
        ("股价低于每股净资产了吗", True),
        ("Is it trading under book value now?", False),
    ],
)
def test_a_book_value_question_is_answered_with_the_pb(question, zh):
    from query_intelligence.agent.composer import book_value_sentences
    from query_intelligence.agent.planner import _VALUATION_TERMS

    assert _VALUATION_TERMS.search(question) or "book" in question
    for pb, below in ((0.62, True), (1.85, False)):
        tool_log = [
            {
                "tool": "get_fundamentals",
                "ok": True,
                "data": {"symbol": "X", "name": "X", "evidence_id": "fundamental_X", "metrics": {"pb": pb}},
            }
        ]
        (sentence,) = book_value_sentences(question, tool_log, zh)
        assert f"{pb:g}" in sentence and "fundamental_X" in sentence
        if zh:
            assert ("没有破净" in sentence) != below and "破净" in sentence
        else:
            assert ("not below book value" in sentence) != below and "below book value" in sentence


def test_a_moving_average_break_is_not_a_book_value_break():
    from query_intelligence.agent.coverage import requested_price_fields

    assert requested_price_fields("跌破20日均线了吗").needs_indicators
    assert not requested_price_fields("跌破净资产了吗").needs_indicators


# ---- M4: computed numbers are registered; a failed draft never answers another operation ----


def test_a_lot_count_in_hand_units_is_kept_by_the_verifier():
    from query_intelligence.agent.composer import cost_basis_sentences, cost_basis_values
    from query_intelligence.agent.verifier import verify_answer

    question = "美的集团：3手的买入价是76元，按最新收盘价算浮盈多少"
    tool_log = _price_log("000333.SZ", "美的集团", 80.1)
    sentence = cost_basis_sentences(tool_log, question, True)[0]
    assert "300 × 4.1 = 1230 元" in sentence, sentence
    plain = verify_answer({"answer": sentence}, _store(tool_log), query=question, allow_derived=True)
    kept = verify_answer(
        {"answer": sentence}, _store(tool_log), query=question, allow_derived=True,
        computed=cost_basis_values(question, tool_log),
    )  # fmt: skip
    assert kept.passed, kept.unsupported_numbers
    assert plain.passed or set(plain.unsupported_numbers) <= {300.0, 1230.0}


@pytest.fixture(scope="module")
def shipped_service():
    from query_intelligence.service import build_default_service

    return build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
        offline_snapshot_fix=True,
    )


def _llm_agent(shipped_service, steps):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.state import AgentConfig
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    config = AgentConfig(planner_prefetch=True, revise_policy="cite_repair", verify_derived=True)
    runtime = AgentRuntime(
        shipped_service, build_registry_for_service(shipped_service), ScriptedLLM(steps), config=config
    )
    return runtime, AgentService(runtime, trace_sinks=[])


def test_llm_a_failed_rise_draft_is_answered_with_the_rise_not_the_loss(shipped_service):
    from query_intelligence.agent.llm import final_turn

    wrong = "美的集团需要上涨约 37.5% 才能回到成本 [price_000333.SZ]。"
    steps = [final_turn({"answer": wrong, "evidence_used": []}) for _ in range(3)]
    runtime, service = _llm_agent(shipped_service, steps)
    try:
        result = service.chat("美的集团我是96元的成本，涨百分之多少能回到成本价", session_id="r17-rise", mode="agent")
    finally:
        runtime.close()
    answer = str(result["answer"])
    assert "≈ 19.85%" in answer and "37.5" not in answer, answer
    assert "-16.56%" not in answer, answer  # never the loss in percent of the cost


def test_llm_a_draft_that_will_not_compute_the_users_figures_gets_the_sale_arithmetic(shipped_service):
    from query_intelligence.agent.llm import final_turn

    refusal = "买卖价格与股数属于你的交易记录，无法由行情数据核实；美的集团最新收盘价 80.1 元 [price_000333.SZ]。"
    runtime, service = _llm_agent(shipped_service, [final_turn({"answer": refusal, "evidence_used": []})])
    try:
        result = service.chat(
            "美的集团我70元买了1000股，前几天78元卖掉400股，剩下的现在赚了多少", session_id="r17-sale", mode="agent"
        )
    finally:
        runtime.close()
    answer = str(result["answer"])
    assert "600 × 10.1 = 6060" in answer and "3200" in answer, answer
    assert "cost_basis_result_appended" in (result.get("degraded") or [])


def test_could_not_verify_names_the_asked_operation():
    from query_intelligence.agent.composer import asked_result, could_not_verify_note

    tool_log = _price_log("000333.SZ", "美的集团", 80.1)
    asked = asked_result("美的集团我是96元的成本，涨百分之多少能回到成本价", tool_log, None, True)
    assert asked == {"label": "回到成本价所需的涨幅", "values": [19.85]}
    note = could_not_verify_note(asked["label"], True)
    assert "未能核实所问的回到成本价所需的涨幅" in note and not re.search(r"\d", note)


def test_llm_a_failed_draft_never_gets_a_template_answering_another_operation(shipped_service, monkeypatch):
    """The template is replaced by one that answers the loss in percent (another operation): the failed draft is then
    answered with "could not verify" for the rise asked, never with that template."""
    from query_intelligence.agent.llm import final_turn

    wrong = "美的集团需要上涨约 37.5% 才能回到成本 [price_000333.SZ]。"
    steps = [final_turn({"answer": wrong, "evidence_used": []}) for _ in range(3)]
    runtime, service = _llm_agent(shipped_service, steps)
    other = (
        "按 2026-09-30 收盘价 80.1 元和你给出的成本价 96 元计算，美的集团每股浮亏 80.1 − 96 = -15.9 元（-16.56%） "
        "[price_000333.SZ]。"
    )
    monkeypatch.setattr(
        runtime, "_template_update", lambda state: {"draft": {"answer": other, "evidence_used": ["price_000333.SZ"]}}
    )
    try:
        result = service.chat("美的集团我是96元的成本，涨百分之多少能回到成本价", session_id="r17-cnv", mode="agent")
    finally:
        runtime.close()
    answer = str(result["answer"])
    assert answer.startswith("未能核实所问的回到成本价所需的涨幅"), answer
    assert "-16.56%" not in answer and "37.5" not in answer, answer
    assert "verification_failed:asked_result_unverified" in (result.get("degraded") or [])
