"""Round-15 rules for the round-11 review's K1, K2, K3, K4, K7, K8, K9, K10 and K14, in the author's own wording (the
review's repro strings and its hidden slice are not used; ``tests/test_agent_eval.py`` checks that no string here
copies a committed slice or a quoted review probe).

* K1: a total compared with another value ("A和B合起来比C高吗", "is C bigger than A and B combined?") is
  (A + B) vs C — the other side is never added into the total; generated wordings × entity triples, both orders.
* K2: several purchases at different costs: Σ shares × (close − cost) over every lot, each lot shown.
* K3: the verifier re-derives every operation the frame computes (mean, N-way share, total vs one, cost lots) and
  accepts the numbers the frame computed; a draft whose repair loses the asked result falls back to the frame.
* K4: a mean operation; two operations in one message; two metrics of one target; the operation classifier
  (fallback / tie-breaker, explainable, measured on held-out templates).
* K7: 万 / 千 counts; a portfolio listed in clauses with a bare value question; a position added to the last one.
* K8: a dated close the payload carries is answered; a day after the last bar is said to be after the data.
* K9: a name in a list that matches no security is named, never dropped.
* K10: 净利润总额 is net profit (a total), not 利润总额; an ellipsis naming two targets keeps the metric.
* K14: a past one-year window is not a forecast.
"""

from __future__ import annotations

import random
import re

import pytest

SEED = 15

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
FIELDS = {"net_profit": "net_profit", "revenue": "revenue", "market_cap": "total_mv", "roe": "roe", "pe": "pe_ttm"}


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


# ---- K1: a total compared with another value ----

GROUP_VS_ONE_ZH = [
    ("{A}和{B}的{m}合起来比{C}高吗", 2),
    ("{A}与{B}两家{m}总共，比{C}少多少", 2),
    ("{C}一家的{m}比{A}、{B}合在一起还多吗", 1),
    ("{C}的{m}，比{A}加{B}合计多出多少", 1),
]
GROUP_VS_ONE_EN = [
    ("Taken together, do {A} and {B} beat {C} on {m}?", 2),
    ("Is {C}'s {m} larger than {A} and {B} combined?", 1),
    ("How far is the combined {m} of {A} and {B} below {C}'s?", 2),
]


@pytest.mark.parametrize("language", ["zh", "en"])
def test_a_total_against_one_value_never_absorbs_the_other_side(language):
    """Generated wordings: the two sides are split at the comparison anchor, the total is of the grouped side only,
    and the composed sentence states the gap between the two sides (checked against an independent computation)."""
    from query_intelligence.agent.composer import frame_result, frame_sentences
    from query_intelligence.agent.frame import resolve_frame_question

    zh = language == "zh"
    rng = random.Random(f"{SEED}-k1-{language}")
    templates = GROUP_VS_ONE_ZH if zh else GROUP_VS_ONE_EN
    for _ in range(60):
        a, b, c = rng.sample(ENTITIES, 3)
        key = rng.choice(sorted(MONEY))
        template, split = rng.choice(templates)
        question = template.format(
            A=a[1] if zh else a[2], B=b[1] if zh else b[2], C=c[1] if zh else c[2], m=MONEY[key][0 if zh else 1]
        )
        listed = [_entity(row, zh) for row in (a, b, c)]
        found = resolve_frame_question(question, [], listed, [])
        assert found is not None, question
        request = found[2]
        assert request["operation"] == "total_vs_one", question
        assert request["split"] == split, question
        symbols = [item["symbol"] for item in request["operands"]]
        group, single = {a[0], b[0]}, c[0]
        sides = (symbols[:split], symbols[split:])
        assert {*sides[0], *sides[1]} == {*group, single}
        assert [single] in (sides[0], sides[1]), question
        values = {row[0]: round(rng.uniform(1e9, 9e11), 2) for row in (a, b, c)}
        tool_log = [_fundamentals(row[0], row[1], FIELDS[key], values[row[0]]) for row in (a, b, c)]
        expected = round(abs(values[a[0]] + values[b[0]] - values[c[0]]), 2)
        assert frame_result(request, tool_log) == expected, question
        sentences, gaps = frame_sentences(request, tool_log, zh)
        assert sentences and not gaps
        total_shown = re.sub(r"\s", "", str(round((values[a[0]] + values[b[0]] + values[c[0]]) / 1e8, 2)))
        assert f"合计 {total_shown}" not in sentences[0], "the other side was added into the total"


def test_a_yes_no_total_question_answers_yes_or_no_first():
    from query_intelligence.agent.composer import frame_sentences
    from query_intelligence.agent.frame import resolve_frame_question

    a, b, c = ENTITIES[0], ENTITIES[1], ENTITIES[6]
    request = resolve_frame_question(
        f"{a[1]}和{b[1]}的营业收入加起来比{c[1]}多吗", [], [_entity(r, True) for r in (a, b, c)]
    )[2]
    assert request["asked"] == "more"
    tool_log = [_fundamentals(a[0], a[1], "revenue", 1.5e11), _fundamentals(b[0], b[1], "revenue", 0.8e11)]
    tool_log.append(_fundamentals(c[0], c[1], "revenue", 8e11))
    sentence = frame_sentences(request, tool_log, True)[0][0]
    assert sentence.startswith("不是，") and "合计 2300 亿元" in sentence and "少 5700 亿元" in sentence


# ---- K4: the mean ----

MEAN_ZH = ["{A}、{B}、{C}三家{m}的均值是多少", "{A}、{B}和{C}的{m}平均下来多少", "{A}、{B}、{C}的{m}加总以后除以三"]
MEAN_EN = ["What do {A}, {B} and {C} average on {m}?", "Mean {m} across {A}, {B} and {C}?"]
MEAN_METRICS = {"roe": ("ROE", "ROE"), "pe": ("市盈率", "P/E"), "net_profit": ("净利润", "net profit")}


@pytest.mark.parametrize("language", ["zh", "en"])
def test_the_mean_of_several_operands(language):
    from query_intelligence.agent.composer import frame_result, frame_sentences
    from query_intelligence.agent.frame import resolve_frame_question
    from query_intelligence.agent.verifier import verify_answer

    zh = language == "zh"
    rng = random.Random(f"{SEED}-mean-{language}")
    for _ in range(40):
        rows = rng.sample(ENTITIES, 3)
        key = rng.choice(sorted(MEAN_METRICS))
        question = rng.choice(MEAN_ZH if zh else MEAN_EN).format(
            A=rows[0][1 if zh else 2],
            B=rows[1][1 if zh else 2],
            C=rows[2][1 if zh else 2],
            m=MEAN_METRICS[key][0 if zh else 1],
        )
        request = resolve_frame_question(question, [], [_entity(row, zh) for row in rows], [])[2]
        assert request["operation"] == "mean" and len(request["operands"]) == 3, question
        values = [
            round(rng.uniform(5, 40), 2) if key != "net_profit" else round(rng.uniform(1e9, 9e10), 2) for _r in rows
        ]
        tool_log = [_fundamentals(row[0], row[1], FIELDS[key], value) for row, value in zip(rows, values, strict=True)]
        assert frame_result(request, tool_log) == round(sum(values) / 3, 2)
        sentences, _gaps = frame_sentences(request, tool_log, zh)
        # the template's own sentence passes the verifier with the generic mean rule alone (no computed values)
        assert verify_answer({"answer": sentences[0]}, _store(tool_log), allow_derived=True).passed, sentences[0]


def test_an_industry_average_or_a_weighted_roe_is_not_a_mean():
    from query_intelligence.agent.frame import frame_operation

    assert frame_operation("贵州茅台的市盈率比白酒行业平均高多少") == "difference"
    assert frame_operation("Is Wuliangye's P/E above the sector average?") != "mean"
    assert frame_operation("加权平均净资产收益率怎么算") is None
    assert frame_operation("五粮液近20日平均成交额是多少") is None


# ---- K4 / K10: two operations in one message; two metrics of one target ----


def test_a_total_and_a_share_of_it_in_one_message():
    from query_intelligence.agent.composer import frame_sentences
    from query_intelligence.agent.frame import resolve_frame_question

    mt, wly = ENTITIES[0], ENTITIES[1]
    frame = {
        "metric": "net_profit",
        "operands": [
            {"kind": "target", "name": mt[1], "symbol": mt[0]},
            {"kind": "target", "name": wly[1], "symbol": wly[0]},
        ],
    }
    request = resolve_frame_question(
        "这两家合在一起赚多少，五粮液占里面几成", [{"frame": frame}], [_entity(wly, True)]
    )[2]
    assert request["operation"] == "sum"
    assert [(item["operation"], item["operands"][0]["symbol"]) for item in request["also"]] == [("share_total", wly[0])]
    tool_log = [_fundamentals(mt[0], mt[1], "net_profit", 8.6e10), _fundamentals(wly[0], wly[1], "net_profit", 3.4e10)]
    text = " ".join(frame_sentences(request, tool_log, True)[0])
    assert "合计 1200 亿元" in text and "≈ 28.33%" in text


@pytest.mark.parametrize(
    ("question", "metrics"),
    [
        ("贵州茅台的营业收入是五粮液的几倍，归母净利润呢", ["revenue", "net_profit"]),
        ("How many times Wuliangye's revenue is Moutai's, and what about net profit?", ["revenue", "net_profit"]),
    ],
)
def test_a_second_metric_in_the_same_message_is_answered(question, metrics):
    from query_intelligence.agent.frame import resolve_frame_question

    zh = bool(re.search(r"[一-鿿]", question))
    listed = [_entity(ENTITIES[0], zh), _entity(ENTITIES[1], zh)]
    request = resolve_frame_question(question, [], listed, [])[2]
    assert [request["metric"], *(item["metric"] for item in request.get("also") or [])] == metrics
    assert request["operation"] == "ratio" and request["also"][0]["operation"] == "ratio"


def test_who_is_higher_and_by_how_much_stays_one_comparison():
    from query_intelligence.agent.frame import resolve_frame_question

    listed = [_entity(ENTITIES[0], True), _entity(ENTITIES[1], True)]
    request = resolve_frame_question("贵州茅台和五粮液比，市净率哪个高，高多少", [], listed, [])[2]
    assert "also" not in request


def test_two_metrics_of_one_target_in_one_message():
    from query_intelligence.agent.composer import frame_result
    from query_intelligence.agent.frame import resolve_frame_question

    row = ENTITIES[7]
    request = resolve_frame_question(f"{row[1]}的总市值相当于它净利润的多少倍", [], [_entity(row, True)], [])[2]
    assert [item["metric"] for item in request["operands"]] == ["market_cap", "net_profit"]
    data = _fundamentals(row[0], row[1], "total_mv", 7.2e11)
    data["data"]["metrics"]["net_profit"] = 4.8e10
    assert frame_result(request, [data]) == 15.0


@pytest.mark.parametrize(
    ("question", "operation"),
    [
        ("比亚迪和宁德时代的营业收入合并起来多少", "sum"),
        ("比亚迪和宁德时代的净利润统共有多少", "sum"),
        ("What is the revenue of BYD and CATL collectively?", "sum"),
        ("比亚迪的净利润比宁德时代少挣多少", "difference"),
        ("By what margin does CATL's ROE lead BYD's?", "difference"),
        ("How wide is the margin between CATL and BYD on ROE?", "difference"),
    ],
)
def test_more_total_and_verb_words(question, operation):
    from query_intelligence.agent.frame import frame_operation

    assert frame_operation(question) == operation


# ---- K4: the operation classifier ----


def test_the_classifier_ships_and_is_measured_on_held_out_templates():
    import json
    from pathlib import Path

    from query_intelligence.agent.operation_classifier import load_classifier

    model = load_classifier()
    assert model is not None and model.version == "operation-clf-r15"
    results = json.loads(
        (Path(__file__).parents[1] / "evaluation" / "results" / "operation_classifier-r15.json").read_text("utf-8")
    )
    held = results["held_out"]
    assert held["templates"] >= 50 and held["classifier_accuracy"]["rate"] >= 0.85
    # the grammar with the classifier is at least as good as the grammar alone on wordings neither has seen
    assert held["grammar_plus_classifier_accuracy"]["rate"] >= held["grammar_accuracy"]["rate"]
    assert all(item["fallback_fires"] <= 2 for item in results["fallback_on_evaluation_sets"].values())


def test_a_fallback_reading_is_explained_in_the_reason():
    from query_intelligence.agent.frame import frame_extra_reasons, resolve_frame_question
    from query_intelligence.agent.operation_classifier import fallback_operation

    # "凑一块" is in no lexicon class; the wording is one of the classifier's training templates (in-sample here: this
    # test checks the explanation, the held-out accuracy is measured above)
    question = "宁德时代和比亚迪的净利润凑一块是多少"
    reading = fallback_operation(question, operands=2)
    assert reading is not None and reading.operation == "sum" and reading.features
    listed = [_entity(ENTITIES[3], True), _entity(ENTITIES[4], True)]
    _rewritten, reason, request = resolve_frame_question(question, [], listed, [])
    assert request["operation"] == "sum" and request["classifier"]["use"] == "fallback"
    # (round 16, review-12 L14) the classifier's reading is a route reason of its own, never an operand of the frame
    assert reason == "frame:sum:net_profit:宁德时代|比亚迪"
    assert any(re.fullmatch(r"clf:sum:0\.\d\d:\S+", item) for item in frame_extra_reasons(request))


def test_the_fallback_needs_several_operands_and_a_question():
    from query_intelligence.agent.operation_classifier import compatible, fallback_operation

    assert fallback_operation("五粮液最近怎么样", operands=1) is None
    assert not compatible("spread", "平安银行当天的最高价和最低价呢")
    assert not compatible("total_vs_one", "Are both cheaper than the sector average?")


# ---- K2: several purchases ----


@pytest.mark.parametrize(
    ("question", "lots"),
    [
        ("我先以36元买了800股平安银行，后来30元又补仓了400股，现在算下来盈亏多少", [(36.0, 800), (30.0, 400)]),
        ("先买了2手招商银行成本44元，又买了300股成本39元，浮盈多少", [(44.0, 200), (39.0, 300)]),
        ("I bought 300 Midea shares at 75 and another 200 at 82. Am I up or down?", [(75.0, 300), (82.0, 200)]),
    ],
)
def test_every_purchase_counts(question, lots):
    from query_intelligence.agent.composer import cost_basis_sentences
    from query_intelligence.agent.coverage import cost_basis_request
    from query_intelligence.agent.verifier import verify_answer

    asked = cost_basis_request(question)
    assert [(lot["cost"], lot["shares"]) for lot in asked["lots_detail"]] == lots
    close = 40.0
    tool_log = [
        {
            "tool": "get_price_history",
            "ok": True,
            "data": {"symbol": "X", "name": "X", "close": close, "as_of": "2026-09-30", "evidence_id": "price_X"},
        }
    ]
    zh = bool(re.search(r"[一-鿿]", question))
    sentences = cost_basis_sentences(tool_log, question, zh)
    expected = round(sum(count * (close - cost) for cost, count in lots), 2)
    shown = f"{expected:g}"
    assert shown in sentences[0], (sentences[0], shown)
    for cost, count in lots:
        assert f"{count} " in sentences[0] and f"{cost:g}" in sentences[0]
    from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore

    store = EvidenceStore()
    store.add(
        AgentEvidence(
            evidence_id="price_X",
            source_type="market",
            kind="structured",
            produced_by="get_price_history",
            title="X",
            payload={"close": close, "as_of": "2026-09-30"},
        )
    )
    assert verify_answer({"answer": sentences[0]}, store, query=question, allow_derived=True).passed


# ---- K3: the verifier keeps what the frame computed ----


def test_a_share_of_a_three_company_total_with_its_numerator_repeated_is_kept():
    from query_intelligence.agent.composer import frame_sentences
    from query_intelligence.agent.verifier import verify_answer

    rows = ENTITIES[5:8]
    values = [1.2e11, 3.1e11, 0.5e11]
    tool_log = [_fundamentals(r[0], r[1], "net_profit", v) for r, v in zip(rows, values, strict=True)]
    operands = [{"kind": "target", "name": r[1], "symbol": r[0]} for r in rows]
    sentences, _ = frame_sentences(
        {"operation": "share_total", "metric": "net_profit", "operands": operands}, tool_log, True
    )
    assert "合计 4800 亿元" in sentences[0] and "≈ 25%" in sentences[0]
    assert verify_answer({"answer": sentences[0]}, _store(tool_log), allow_derived=True).passed


def test_a_scripted_draft_with_a_frame_result_the_generic_rules_cannot_derive_is_kept():
    """An LLM-style draft states the mean ROE without listing every operand next to it: the generic rule cannot
    derive it, the frame's computed value can."""
    from query_intelligence.agent.composer import frame_values
    from query_intelligence.agent.verifier import verify_answer

    rows = ENTITIES[3:6]
    values = [22.4, 12.1, 10.9]
    tool_log = [_fundamentals(r[0], r[1], "roe", v) for r, v in zip(rows, values, strict=True)]
    request = {
        "operation": "mean",
        "metric": "roe",
        "operands": [{"kind": "target", "name": r[1], "symbol": r[0]} for r in rows],
    }
    draft = {"answer": "三家公司ROE的简单平均约为 15.13% [fundamental_300750.SZ][fundamental_002594.SZ]。"}
    store = _store(tool_log)
    assert not verify_answer(draft, store, allow_derived=True).passed
    assert verify_answer(draft, store, allow_derived=True, computed=frame_values(request, tool_log)).passed
    # a number the frame did not compute is still rejected
    wrong = {"answer": "三家公司ROE的简单平均约为 16.4% [fundamental_300750.SZ][fundamental_002594.SZ]。"}
    assert not verify_answer(wrong, store, allow_derived=True, computed=frame_values(request, tool_log)).passed


def test_a_gap_between_a_total_and_one_value_is_derived_from_the_stated_operands():
    from query_intelligence.agent.verifier import verify_answer

    rows = ENTITIES[:3]
    tool_log = [_fundamentals(r[0], r[1], "net_profit", v) for r, v in zip(rows, (5e10, 3e10, 1.2e11), strict=True)]
    text = (
        "净利润：贵州茅台 500 亿元，五粮液 300 亿元，中国平安 1200 亿元；"
        "贵州茅台、五粮液合计 800 亿元，比中国平安少 400 亿元 "
        "[fundamental_600519.SH][fundamental_000858.SZ][fundamental_601318.SH]。"
    )
    assert verify_answer({"answer": text}, _store(tool_log), allow_derived=True).passed
    bad = text.replace("少 400 亿元", "少 450 亿元")
    assert not verify_answer({"answer": bad}, _store(tool_log), allow_derived=True).passed


@pytest.fixture(scope="module")
def v1_agent_factory():
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=False, use_live_macro=False, use_live_news=False, use_live_announcement=False
    )

    def make(steps):
        from query_intelligence.agent.graph import AgentRuntime
        from query_intelligence.agent.llm import ScriptedLLM
        from query_intelligence.agent.service import AgentService
        from query_intelligence.agent.state import AgentConfig
        from query_intelligence.agent.tools.defaults import build_registry_for_service

        config = AgentConfig(planner_prefetch=True, revise_policy="cite_repair", verify_derived=True, max_revisions=0)
        runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM(steps), config=config)
        return runtime, AgentService(runtime, trace_sinks=[])

    return make


def test_a_repair_that_keeps_another_metric_falls_back_to_the_frame(v1_agent_factory):
    """A scripted draft for a total of net profits states a revenue total (a figure no tool gives) next to a valid
    sentence: pruning would leave an answer without the asked total, so the frame's computation answers."""
    from query_intelligence.agent.llm import final_turn

    draft = {
        "answer": "贵州茅台营业收入合计为 9999.99 亿元 [fundamental_600519.SH]。"
        "五粮液的数据见工具结果 [fundamental_000858.SZ]。",
        "evidence_used": [],
    }
    runtime, agent = v1_agent_factory([final_turn(draft)])
    try:
        result = agent.chat("贵州茅台跟五粮液的归母净利润合到一块是多少", session_id="r15-k3-metric", mode="agent")
    finally:
        runtime.close()
    assert "9999.99" not in result["answer"]
    assert "净利润" in result["answer"] and "合计" in result["answer"]
    assert "verification_failed:repaired" in result["degraded"]


# ---- K7: holdings as stated ----


@pytest.mark.parametrize(
    ("question", "shares"),
    [
        ("我手里有2.5万股工商银行，按收盘价值多少", 25000),
        ("持有3万股长江电力，市值是多少", 30000),
        ("我有8千股平安银行，值多少钱", 8000),
    ],
)
def test_a_count_with_wan_or_qian_is_a_holding(question, shares):
    from query_intelligence.agent.coverage import holding_value_request, requested_metrics, without_holding_value

    assert holding_value_request(question)[0] == shares
    assert "market_cap" not in [metric.key for metric in requested_metrics(without_holding_value(question))]


def test_a_portfolio_listed_in_clauses_with_a_bare_value_question():
    from query_intelligence.agent.coverage import holding_positions

    question = "仓位如下：长江电力 2,000 股；紫金矿业 5 手；美的集团 300 股。合计值多少？"
    assert [item["shares"] for item in holding_positions(question)] == [2000, 500, 300]
    assert holding_positions("长江电力 2,000 股；紫金矿业 5 手。紫金矿业的总市值？") == []


def test_a_position_added_to_the_last_one_is_valued_together():
    from query_intelligence.agent.memory import resolve_holding_follow_up

    turns = [
        {
            "query": "我有1500股美的集团，值多少钱",
            "effective_query": "我有1500股美的集团，值多少钱",
            "entities": [{"symbol": "000333.SZ", "name": "美的集团"}],
        }
    ]
    found = resolve_holding_follow_up(
        "另外还有600股长江电力，一共多少", turns, [{"symbol": "600900.SH", "canonical_name": "长江电力"}]
    )
    assert found is not None and "1500股美的集团" in found[0] and "600股长江电力" in found[0]


# ---- K8: dated closes ----


def _price(day_close: list[tuple[str, float]], **extra) -> list[dict]:
    data = {
        "symbol": "600900.SH",
        "name": "长江电力",
        "close": day_close[-1][1],
        "as_of": day_close[-1][0],
        "evidence_id": "price_600900.SH",
        "recent_closes": [{"date": d, "close": c} for d, c in day_close],
        **extra,
    }
    return [{"tool": "get_price_history", "ok": True, "data": data}]


def test_a_dated_close_in_the_payload_is_answered():
    from query_intelligence.agent.composer import dated_close_sentences
    from query_intelligence.agent.coverage import coverage_gaps

    log = _price([("2026-09-29", 28.4), ("2026-09-30", 28.54)], prior_year_end={"date": "2025-12-31", "close": 27.75})
    facts, gaps = dated_close_sentences("长江电力去年12月31日，也就是2025年12月31日的收盘价", log, True)
    assert facts == ["长江电力 2025-12-31 的收盘价为 27.75 元 [price_600900.SH]。"] and not gaps
    assert not coverage_gaps("长江电力2025年12月31日收盘价", log, zh=True)
    facts, gaps = dated_close_sentences("What did China Yangtze Power close at on September 29, 2026?", log, False)
    assert "28.4" in facts[0]


def test_a_day_after_the_last_bar_is_after_the_data():
    from query_intelligence.agent.composer import dated_close_sentences

    log = _price([("2026-09-29", 28.4), ("2026-09-30", 28.54)])
    facts, gaps = dated_close_sentences("长江电力10月12日的收盘价", log, True)
    assert not facts and "当前数据截至 2026-09-30，没有长江电力 2026-10-12 的收盘价" in gaps[0]
    facts, gaps = dated_close_sentences("长江电力9月26日收盘价", log, True)
    assert not facts and "没有长江电力 2026-09-26 的收盘价" in gaps[0]


# ---- K9: names in a list that match no security ----


@pytest.mark.parametrize(
    ("question", "language", "stray"),
    [
        ("XYZW and Midea Group: what's their revenue in all?", "en", "XYZW"),
        ("Among CATL, Qorvex and BYD, how far apart are the highest and lowest P/E?", "en", "Qorvex"),
        ("宁德时代、某某新材和比亚迪三家的营业收入加起来多少", "zh", "某某新材"),
        ("某某新材和比亚迪的净利润一共多少", "zh", "某某新材"),
    ],
)
def test_an_unknown_name_in_a_list_is_named(question, language, stray):
    from query_intelligence.agent.composer import frame_sentences
    from query_intelligence.agent.frame import resolve_frame_question

    zh = language == "zh"
    known = [row for row in ENTITIES if (row[1] if zh else row[2]) in question]
    found = resolve_frame_question(question, [], [_entity(row, zh) for row in known], [])
    assert found is not None, question
    request = found[2]
    assert any(item["kind"] == "unresolved" and item["name"] == stray for item in request["operands"]), request
    sentences, gaps = frame_sentences(request, [], zh)
    assert not sentences and stray in gaps[0]


def test_a_list_of_known_names_has_no_stray():
    from query_intelligence.agent.frame import unresolved_in_list

    rows = [_entity(row, True) for row in ENTITIES[:3]]
    assert unresolved_in_list("贵州茅台、五粮液和中国平安的营收之和里茅台占比", rows) == []
    assert unresolved_in_list("茅台、五粮液和平安三家的净利润加起来", rows) == []


# ---- K10: metric names and ellipsis ----


def test_a_total_of_net_profits_is_net_profit():
    from query_intelligence.agent.coverage import requested_metrics
    from query_intelligence.agent.frame import frame_operation, metric_of

    assert metric_of("长江电力和紫金矿业的归母净利润总额") == "net_profit"
    assert frame_operation("长江电力和紫金矿业的归母净利润总额") == "sum"
    assert "total_profit" not in [m.key for m in requested_metrics("长江电力和紫金矿业的净利润总额")]
    assert metric_of("长江电力的利润总额") == "total_profit"


def test_an_ellipsis_naming_two_targets_keeps_the_metric():
    from query_intelligence.agent.memory import resolve_ellipsis

    turns = [{"query": "紫金矿业的市净率", "entities": [{"symbol": "601899.SH", "name": "紫金矿业"}]}]
    targets = [
        {"symbol": "600900.SH", "canonical_name": "长江电力", "mention": "长江电力"},
        {"symbol": "000333.SZ", "canonical_name": "美的集团", "mention": "美的"},
    ]
    rewritten, reason = resolve_ellipsis("那长江电力跟美的的呢", turns, targets)
    assert "长江电力和美的集团的市净率" in rewritten and reason.startswith("ellipsis:aspect")


# ---- K14: a past window is not a forecast ----


@pytest.mark.parametrize(
    ("question", "forecast"),
    [
        ("长江电力距离一年内的最高点回落了多少", False),
        ("美的集团离年内高点还差几个百分点", False),
        ("长江电力年内能不能站上35元", True),
        ("美的集团年内会涨到100元吗", True),
    ],
)
def test_a_past_one_year_window_is_not_a_forecast(question, forecast):
    from datetime import date

    from query_intelligence.agent.router import asks_prediction

    assert asks_prediction(question, as_of=date(2026, 9, 30)) is forecast
