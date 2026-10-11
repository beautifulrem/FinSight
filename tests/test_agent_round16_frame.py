"""Round-16 rules for the round-12 review's L1, L2, L3, L10, L12, L13, L14 and L17, in the author's own wording (the
review's repro strings and its hidden slice are not used; ``tests/test_agent_eval.py`` checks that no string here
copies a committed slice, a quoted review probe or the independent operation wording set).

* L1: a total compared with a value named with the comparison word ("…合计，跟C比", "与C相比", "compared with C, …")
  is (A + B) vs C wherever C stands; generated wordings × entity triples, both languages.
* L2: several purchases written naturally (the name between the verb and the count, 块 / 补了, "another N at P").
* L3: mixed-unit counts, a 、-list valued at the close, a second English position, a stated position carried.
* L10: a leading English adverb is never an unresolved security.
* L12: two operations across two sentences; a share asked right after a total.
* L13: the yuan gap to the 52-week high; "worth" with a 万 count; no advice prefix on stated-cost arithmetic.
* L14: the classifier's reading and further operations are reasons of their own.
* L17: the classifier's structural features and the artifact's provenance.
"""

from __future__ import annotations

import random

import pytest

SEED = 16

# (symbol, Chinese name, English name)
ENTITIES = [
    ("600519.SH", "贵州茅台", "Kweichow Moutai"),
    ("000858.SZ", "五粮液", "Wuliangye"),
    ("601318.SH", "中国平安", "Ping An Insurance"),
    ("300750.SZ", "宁德时代", "CATL"),
    ("002594.SZ", "比亚迪", "BYD"),
    ("600036.SH", "招商银行", "China Merchants Bank"),
    ("601398.SH", "工商银行", "ICBC"),
    ("601899.SH", "紫金矿业", "Zijin Mining"),
    ("000333.SZ", "美的集团", "Midea Group"),
    ("600900.SH", "长江电力", "China Yangtze Power"),
]
MONEY = {
    "net_profit": ("净利润", "net profit"),
    "revenue": ("营业收入", "revenue"),
    "market_cap": ("总市值", "market cap"),
}
FIELDS = {"net_profit": "net_profit", "revenue": "revenue", "market_cap": "total_mv"}


def _entity(row, zh: bool) -> dict:
    symbol, name, english = row
    return {
        "symbol": symbol,
        "canonical_name": name,
        "mention": name if zh else english,
        "entity_type": "stock",
        "match_type": "alias_exact",
    }


def _fundamentals(symbol: str, name: str, field: str, value: float) -> dict:
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
            "metrics": {"revenue": 2e11, "net_profit": 3e10, field: value},
        },
    }


# ---- L1: the single value named with the comparison word ----

COMPARAND_ZH = [
    "{A}和{B}的{m}凑在一块，跟{C}比起来哪边大",
    "{A}与{B}两家{m}合计，同{C}比较的话差多少",
    "{A}、{B}的{m}加总以后与{C}相比如何",
    "和{C}比，{A}加{B}的{m}总共是多还是少",
    "相比{C}，{A}和{B}的{m}合在一起够不够",
]
COMPARAND_EN = [
    "Measured against {C}, do {A} and {B} combined have more {m}?",
    "Compared to {C}, how big is the {m} of {A} and {B} put together?",
    "Set {A} and {B}'s combined {m} against {C}: which is larger?",
    "Relative to {C}, what do {A} and {B} make in {m} together?",
]


@pytest.mark.parametrize("language", ["zh", "en"])
def test_the_single_value_named_with_the_comparison_word_is_never_added_into_the_total(language):
    from query_intelligence.agent.composer import frame_result
    from query_intelligence.agent.frame import resolve_frame_question

    zh = language == "zh"
    rng = random.Random(f"{SEED}-l1-{language}")
    for _ in range(60):
        a, b, c = rng.sample(ENTITIES, 3)
        key = rng.choice(sorted(MONEY))
        question = rng.choice(COMPARAND_ZH if zh else COMPARAND_EN).format(
            A=a[1] if zh else a[2], B=b[1] if zh else b[2], C=c[1] if zh else c[2], m=MONEY[key][0 if zh else 1]
        )
        found = resolve_frame_question(question, [], [_entity(row, zh) for row in (a, b, c)], [])
        assert found is not None, question
        request = found[2]
        assert request["operation"] == "total_vs_one", question
        symbols = [item["symbol"] for item in request["operands"]]
        assert symbols[request["split"] :] == [c[0]], question  # the single side
        assert set(symbols[: request["split"]]) == {a[0], b[0]}, question
        values = {row[0]: round(rng.uniform(1e9, 9e11), 2) for row in (a, b, c)}
        tool_log = [_fundamentals(row[0], row[1], FIELDS[key], values[row[0]]) for row in (a, b, c)]
        assert frame_result(request, tool_log) == round(abs(values[a[0]] + values[b[0]] - values[c[0]]), 2)


def test_a_total_and_the_value_after_it_keep_their_order():
    """ "A与B合计比C多吗": C is the single side (the connector before B is not a comparison)."""
    from query_intelligence.agent.frame import resolve_frame_question

    a, b, c = ENTITIES[3], ENTITIES[4], ENTITIES[5]
    request = resolve_frame_question(
        f"{a[1]}与{b[1]}的净利润合计比{c[1]}多吗", [], [_entity(row, True) for row in (a, b, c)], []
    )[2]
    assert request["operation"] == "total_vs_one" and request["split"] == 2
    assert [item["symbol"] for item in request["operands"]][2] == c[0]


# ---- L10: a leading adverb ----


@pytest.mark.parametrize("adverb", ["Together", "Jointly", "Combined", "Altogether", "In total", "Collectively"])
def test_a_leading_adverb_is_never_an_unresolved_security(adverb):
    from query_intelligence.agent.frame import resolve_frame_question, unresolved_in_list

    a, b = ENTITIES[6], ENTITIES[9]
    question = f"{adverb}, how much net profit do {a[2]} and {b[2]} make?"
    listed = [_entity(row, False) for row in (a, b)]
    assert unresolved_in_list(question, listed) == []
    request = resolve_frame_question(question, [], listed, [])[2]
    assert request["operation"] == "sum"
    assert all(item.get("kind") != "unresolved" for item in request["operands"])


def test_a_real_unknown_name_in_the_list_is_still_named():
    from query_intelligence.agent.frame import unresolved_in_list

    listed = [_entity(ENTITIES[6], False)]
    assert unresolved_in_list("Overall, what do ICBC and Foobarco earn?", listed) == [("Foobarco", 26)]


# ---- L12: two operations in one message ----


def test_a_ratio_then_a_total_in_the_next_sentence_answers_both():
    from query_intelligence.agent.frame import frame_extra_reasons, resolve_frame_question

    a, b = ENTITIES[9], ENTITIES[7]
    found = resolve_frame_question(f"{a[1]}的营业收入是{b[1]}的几倍？顺手把两家加总也算出来", [], [
        _entity(row, True) for row in (a, b)
    ], [])  # fmt: skip
    request = found[2]
    assert request["operation"] == "ratio"
    assert [item["operation"] for item in request["also"]] == ["sum"]
    assert "frame_also:sum:revenue" in frame_extra_reasons(request)
    assert "+also" not in found[1]


def test_a_share_asked_right_after_a_total_takes_the_named_target_as_numerator():
    from query_intelligence.agent.composer import frame_sentences
    from query_intelligence.agent.frame import resolve_frame_question

    a, b = ENTITIES[5], ENTITIES[6]
    request = resolve_frame_question(f"{a[1]}和{b[1]}的营业收入一共多少？工行占比是多少", [], [
        _entity(row, True) for row in (a, b)
    ], [])[2]  # fmt: skip
    assert request["operation"] == "sum"
    extra = request["also"][0]
    assert extra["operation"] == "share_total" and extra["operands"][0]["symbol"] == b[0]
    tool_log = [_fundamentals(a[0], a[1], "revenue", 3e11), _fundamentals(b[0], b[1], "revenue", 9e11)]
    sentences, _gaps = frame_sentences(request, tool_log, True)
    assert any("75%" in sentence and "工商银行约占" in sentence for sentence in sentences), sentences


def test_which_and_by_how_much_in_two_sentences_is_one_difference():
    from query_intelligence.agent.frame import frame_operation

    assert frame_operation("五粮液和贵州茅台，谁的ROE更高？高出几个点") == "difference"
    assert frame_operation("Which of Moutai and Wuliangye has the bigger P/E? By how much?") == "difference"


# ---- L2 / L3 / L13: holdings and lots ----


@pytest.mark.parametrize(
    ("question", "lots"),
    [
        ("我先以50元买进招商银行800股，后来45块又补了400股，现在浮盈浮亏多少", [(50.0, 800), (45.0, 400)]),
        ("12块钱入手工商银行2万股，10块又加了1万股，亏了没有", [(12.0, 20000), (10.0, 10000)]),
        ("I picked up 600 Midea at 82 and another 400 at 76; gain or loss now?", [(82.0, 600), (76.0, 400)]),
    ],
)
def test_every_lot_is_counted_in_natural_forms(question, lots):
    from query_intelligence.agent.coverage import cost_basis_request, cost_lots

    assert [(lot["cost"], lot["shares"]) for lot in cost_lots(question)] == lots
    assert cost_basis_request(question)["shares"] == sum(count for _cost, count in lots)


@pytest.mark.parametrize(
    ("raw", "count"),
    [("1万2千", 12000), ("3千5百", 3500), ("2万3千", 23000), ("1.2万", 12000), ("1万5", 15000), ("8千", 8000)],
)
def test_mixed_unit_counts(raw, count):
    from query_intelligence.agent.coverage import count_value, holding_value_request

    assert count_value(raw) == count
    assert holding_value_request(f"{raw}股招商银行按收盘价算值多少")[0] == count


def test_a_list_valued_at_the_close_and_a_second_english_position():
    from query_intelligence.agent.coverage import holding_positions, holding_value_request

    zh = "账户里放着工行3000股、美的集团200股，拿最新收盘价算一下总市值"
    assert holding_value_request(zh) is not None
    assert [item["shares"] for item in holding_positions(zh)] == [3000, 200]
    en = "I own 300 shares of BYD plus 2,500 of ICBC. What are they worth together?"
    assert [item["shares"] for item in holding_positions(en)] == [300, 2500]


def test_worth_with_a_wan_count_in_english_is_a_holding():
    from query_intelligence.agent.coverage import holding_value_request

    assert holding_value_request("What are 1.5万 shares of Midea worth at the last close?")[0] == 15000
    assert holding_value_request("How much would 30k shares of ICBC be worth?")[0] == 30000


def test_a_stated_position_is_carried_into_a_later_total():
    from query_intelligence.agent.memory import _added_holding

    turns = [{"query": "我手里有招商银行1500股", "entities": [{"symbol": "600036.SH", "name": "招商银行"}]}]
    found = _added_holding("再算上工商银行2000股，加起来一共值多少", turns, [_entity(ENTITIES[6], True)])
    assert found is not None
    assert found[0] == "我持有1500股招商银行和2000股工商银行，合计市值多少"


def test_the_gap_to_the_52_week_high_is_given_in_yuan_and_percent():
    from query_intelligence.agent.composer import _from_52w_extremes
    from query_intelligence.agent.coverage import requested_price_fields

    data = {
        "close": 30.0,
        "name": "测试",
        "evidence_id": "price_X",
        "range_52w": {"high": 40.0, "high_date": "2026-01-05", "low": 20.0, "low_date": "2026-06-01"},
    }
    request = requested_price_fields("测试现在离52周最高收盘还差多少钱")
    sentence = " ".join(_from_52w_extremes(data, True, request))
    assert "40 − 30 = 10 元" in sentence and "25%" in sentence


# ---- L14 / L17: reasons and the classifier ----


def test_the_classifier_reading_is_its_own_reason():
    from query_intelligence.agent.frame import frame_extra_reasons

    request = {
        "operation": "sum",
        "metric": "net_profit",
        "operands": [],
        "classifier": {"operation": "sum", "probability": 0.734, "top_features": ["lex=JOINED", "chars:一块"]},
    }
    assert frame_extra_reasons(request) == ["clf:sum:0.73:lex=JOINED+chars:一块"]


def test_structure_features():
    from query_intelligence.agent.operation_classifier import structure_hits

    rows = structure_hits(["甲和乙合计比丙多吗", "跟丙相比，甲、乙、丁加起来多少", "What is X's P/E?"])
    assert rows[0] == {"order=sum_then_versus": 1.0, "yes_no": 1.0}
    assert rows[1]["order=versus_then_sum"] == 1.0 and rows[1]["list>=3"] == 1.0
    assert rows[2] == {}


def test_the_shipped_classifier_records_where_it_was_trained():
    import joblib

    from query_intelligence.agent.operation_classifier import _artifact_path

    payload = joblib.load(_artifact_path())
    assert payload["version"] == "operation-clf-r16"
    assert payload["trained_at_commit"] and len(payload["dev_wordings_sha256"]) == 64


# ---- after the round-11 slice rerun (after exposure): the remaining forms of the same classes, own wording ----


def test_a_group_led_by_a_connector_after_a_comma_is_the_totalled_side():
    from query_intelligence.agent.frame import resolve_frame_question

    a, b, c = ENTITIES[9], ENTITIES[7], ENTITIES[2]
    question = f"{a[1]}单独的净利润，与{b[1]}、{c[1]}两家加起来比较，哪边多"
    request = resolve_frame_question(question, [], [_entity(row, True) for row in (a, b, c)], [])[2]
    assert request["operation"] == "total_vs_one" and request["split"] == 1
    assert request["operands"][0]["symbol"] == a[0]


def test_a_total_without_its_own_question_then_a_share_of_it():
    from query_intelligence.agent.frame import operation_pieces

    pieces = operation_pieces("Sum of the pair, and Moutai's share of that sum?")
    assert [operation for operation, _piece in pieces] == ["sum", "share_total"]


def test_lots_written_as_count_then_cost_after_an_ordinal():
    from query_intelligence.agent.coverage import cost_basis_request

    asked = cost_basis_request("工行我买过两回：第一回？不，第一次1000股6元，第二次500股7元，现在盈亏多少")
    assert asked is not None
    assert [(lot["cost"], lot["shares"]) for lot in asked["lots_detail"]] == [(6.0, 1000), (7.0, 500)]


def test_an_english_list_with_the_unit_written_once():
    from query_intelligence.agent.coverage import holding_positions

    text = "In my account: 600 Zijin and 900 ICBC shares. Total value?"
    assert [item["shares"] for item in holding_positions(text)] == [600, 900]


@pytest.mark.parametrize(
    ("question", "field"),
    [
        ("宁德时代要回到52周最高收盘价还需要涨百分之多少", "back_to_high_52w"),
        ("比亚迪现在比一年来的最低收盘价高了多少钱", "rebound_52w"),
    ],
)
def test_the_rise_back_to_the_high_and_the_distance_above_the_low(question, field):
    from query_intelligence.agent.coverage import requested_price_fields
    from query_intelligence.agent.frame import resolve_frame_question

    assert getattr(requested_price_fields(question), field)
    # a price-derived question of one target: the time window is not a second security, so no comparison frame
    target = ENTITIES[3] if question.startswith("宁德") else ENTITIES[4]
    assert resolve_frame_question(question, [], [_entity(target, True)], []) is None


def test_market_cap_as_years_of_net_profit_is_a_ratio_of_two_metrics():
    from query_intelligence.agent.frame import resolve_frame_question

    request = resolve_frame_question("美的集团的总市值相当于多少年的净利润", [], [_entity(ENTITIES[8], True)], [])[2]
    assert request["operation"] == "ratio"
    assert [item["metric"] for item in request["operands"]] == ["market_cap", "net_profit"]


def test_the_last_session_of_a_stated_year():
    from query_intelligence.agent.composer import dated_close_sentences

    tool_log = [
        {
            "tool": "get_price_history",
            "ok": True,
            "data": {
                "symbol": "600036.SH",
                "name": "招商银行",
                "evidence_id": "price_600036.SH",
                "as_of": "2026-09-30",
                "close": 41.26,
                "prior_year_end": {"date": "2025-12-31", "close": 42.1},
            },
        }
    ]
    facts, _gaps = dated_close_sentences("招商银行2025年最后一天收盘价是多少", tool_log, True)
    assert facts and "2025-12-31" in facts[0] and "42.1" in facts[0]
