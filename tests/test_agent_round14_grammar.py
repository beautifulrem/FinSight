"""Round-14 rules for the round-10 review's J3, J4, J5, J8, J13 and J14, in the author's own wording (the review's repro
strings and its hidden slice are not used; ``tests/test_agent_eval.py`` checks that no string here copies a committed
slice or a quoted review probe).

The comparison frame is a small grammar now (``frame.GRAMMAR``: OPERATION × OPERANDS × METRIC). Instead of a list of
hand-picked sentences, the property tests below GENERATE wordings from templates — operation phrases × entity pairs ×
metric names, Chinese and English, with a fixed seed — and check for every generated question the operation the
grammar reads, the operands in order, and the value the composer computes from synthetic tool results (checked
against an independent computation here).

* J3: a share of a combined total ("A占两家合计的多少", "A as a percentage of A and B combined") is A ÷ (A + B); the
  verifier traces A ÷ (A + B) and totals of three or more positions, so a correct LLM draft is kept.
* J4: totals with 加 / 总计 / 共计 / 合起来 / "between them" / "add … to it"; 强 / 高过 / outstrip; the most and least
  expensive (a P/E spread, not a judgment); "有没有…N倍半" / "over N times"; a gain or loss against a stated cost; the
  move from the 52-week low; and a two-operand question is never answered with one operand.
* J5: several positions in one message (short names such as 招行 / 工行), and a carried position plus a named one.
* J8: 流通市值 and 扣非净利润 are their own metrics; 营业利润 / 毛利润 are named missing, never a neighbour metric.
* J13 / J14: totals say when the rounded figures shown do not add up; daily change to two decimals; English labels
  without Chinese; a stock against its industry average says higher or lower and by how much.
"""

from __future__ import annotations

import random
import re

import pytest

SEED = 14

# (symbol, Chinese name, English name) of v1 and extension targets
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
# metric key → (Chinese names, English names, the field the synthetic fundamentals carry)
METRICS = {
    "net_profit": (("净利润", "归母净利润"), ("net profit",), "net_profit"),
    "revenue": (("营收", "营业收入"), ("revenue",), "revenue"),
    "roe": (("ROE", "净资产收益率"), ("ROE", "return on equity"), "roe"),
    "pe": (("市盈率",), ("P/E",), "pe_ttm"),
    "pb": (("市净率",), ("P/B",), "pb"),
    "market_cap": (("总市值",), ("market cap",), "total_mv"),
    "float_market_cap": (("流通市值",), ("free-float market cap",), "circ_mv"),
    "profit_dedt": (("扣非净利润",), ("recurring net profit",), "profit_dedt"),
}
MONEY = {"net_profit", "revenue", "market_cap", "float_market_cap", "profit_dedt"}

# operation → templates; {A} and {B} are the operands, {m} the metric name, {k} a stated multiple
TEMPLATES_ZH = {
    "difference": [
        "{A}的{m}比{B}高多少",
        "{A}的{m}比{B}强多少",
        "{A}的{m}高过{B}多少",
        "{A}和{B}的{m}相差多少",
        "{A}跟{B}比，{m}差了多少",
    ],
    "ratio": ["{A}的{m}是{B}的几倍", "{A}的{m}相当于几个{B}"],
    "share": ["{A}的{m}相当于{B}的百分之多少", "{A}的{m}只有{B}的几成"],
    "share_total": [
        "{A}的{m}占{A}和{B}两家合计的多少",
        "{A}和{B}的{m}合计中，{A}占几成",
        "{A}的{m}在{A}与{B}之和里占比多少",
    ],
    "sum": [
        "{A}加{B}的{m}是多少",
        "{A}和{B}的{m}总计多少",
        "{A}和{B}的{m}共计多少",
        "{A}和{B}的{m}合起来是多少",
        "{A}与{B}的{m}加在一起有多少",
    ],
    "relative": ["{A}的{m}比{B}高百分之多少", "{A}的{m}比{B}低多少个百分比"],
    "which": ["{A}和{B}谁的{m}更高", "{A}的{m}比{B}高还是低"],
}
TEMPLATES_EN = {
    "difference": [
        "By how much does {A}'s {m} outstrip {B}'s?",
        "How much higher is {A}'s {m} than {B}'s?",
        "What's the gap between {A}'s {m} and {B}'s?",
        "By how much does {A}'s {m} beat {B}'s?",
    ],
    "ratio": ["What's the ratio of {A}'s {m} to {B}'s?", "By what multiple does {A}'s {m} exceed {B}'s?"],
    "share": ["{A}'s {m} is what percentage of {B}'s?", "As a percentage of {B}'s, how big is {A}'s {m}?"],
    "share_total": [
        "What is {A}'s {m} as a percentage of {A} and {B} combined?",
        "What share of the combined {m} of {A} and {B} does {A} account for?",
    ],
    "sum": [
        "What is {A}'s {m} plus {B}'s?",
        "What is the combined {m} of {A} and {B}?",
        "How much {m} do {A} and {B} make between them?",
    ],
    "relative": ["By what percentage is {A}'s {m} above {B}'s?"],
    "which": ["Which is higher, {A}'s {m} or {B}'s?", "Is {A}'s {m} higher or lower than {B}'s?"],
}
TWO_OPERAND = ("difference", "ratio", "share", "share_total", "sum", "relative", "which")


def _entity(row: tuple[str, str, str], zh: bool) -> dict:
    symbol, name, english = row
    return {
        "symbol": symbol,
        "canonical_name": name,
        "mention": name if zh else english,
        "entity_type": "stock",
        "match_type": "alias_exact",
    }


def _values(rng: random.Random, key: str) -> float:
    if key in MONEY:
        return round(rng.uniform(5e9, 9e11), 2)
    if key == "roe":
        return round(rng.uniform(3, 35), 2)
    return round(rng.uniform(3, 60), 2)


def _fundamentals(symbol: str, name: str, field: str, value: float) -> dict:
    metrics = {"revenue": 1e11, "net_profit": 1e10, field: value}
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
            "metrics": metrics,
        },
    }


def _expected(operation: str, a: float, b: float) -> float | None:
    if operation == "difference":
        return round(abs(a - b), 2)
    if operation == "ratio":
        return round(a / b, 2)
    if operation == "share":
        return round(a / b * 100, 2)
    if operation == "share_total":
        return round(a / (a + b) * 100, 2)
    if operation == "sum":
        return round(a + b, 2)
    if operation == "relative":
        return round(abs(a - b) / abs(b) * 100, 2)
    return None  # which: both values, no single number


def _generated(language: str, count: int) -> list[tuple[str, str, tuple, tuple, str]]:
    """``(question, operation, A, B, metric)`` generated from the templates (seeded)."""
    rng = random.Random(f"{SEED}-{language}")
    templates = TEMPLATES_ZH if language == "zh" else TEMPLATES_EN
    cases = []
    while len(cases) < count:
        operation = rng.choice(TWO_OPERAND)
        key = rng.choice(sorted(MONEY if operation in {"sum", "share_total"} else METRICS))
        a, b = rng.sample(ENTITIES, 2)
        names_zh, names_en, _field = METRICS[key]
        metric_name = rng.choice(names_zh if language == "zh" else names_en)
        if operation == "ratio" and language == "zh" and "几个" in templates[operation][-1] and key not in MONEY:
            template = templates[operation][0]  # "相当于几个X" counts amounts, not rates
        else:
            template = rng.choice(templates[operation])
        if language == "zh":
            question = template.format(A=a[1], B=b[1], m=metric_name)
        else:
            question = template.format(A=a[2], B=b[2], m=metric_name)
        cases.append((question, operation, a, b, key))
    return cases


@pytest.mark.parametrize("language", ["zh", "en"])
def test_generated_wordings_read_the_operation_operands_and_value(language):
    from query_intelligence.agent.composer import frame_result
    from query_intelligence.agent.frame import resolve_frame_question

    rng = random.Random(f"{SEED}-values-{language}")
    failures = []
    cases = _generated(language, 240)
    for question, operation, a, b, key in cases:
        listed = [_entity(a, language == "zh"), _entity(b, language == "zh")]
        found = resolve_frame_question(question, [], listed, [])
        if found is None:
            failures.append((question, "not read"))
            continue
        _rewritten, _reason, request = found
        got_operation = request["operation"]
        symbols = [item.get("symbol") for item in request["operands"]]
        if got_operation != operation or request["metric"] != key or symbols != [a[0], b[0]]:
            failures.append((question, got_operation, request["metric"], symbols))
            continue
        assert request["operand_form"] == "named"
        value_a, value_b = _values(rng, key), _values(rng, key)
        field = METRICS[key][2]
        tool_log = [_fundamentals(a[0], a[1], field, value_a), _fundamentals(b[0], b[1], field, value_b)]
        expected = _expected(operation, value_a, value_b)
        assert frame_result(request, tool_log) == expected, question
    assert not failures, failures[:10]
    assert len(cases) == 240 and len({case[0] for case in cases}) > 120  # many distinct wordings


@pytest.mark.parametrize("language", ["zh", "en"])
def test_generated_share_and_ratio_sentences_state_the_computed_number(language):
    """The composer's sentence states the number the grammar's operation asks for (A ÷ (A + B) for a share of the
    total, never A ÷ B), with the division written out."""
    from query_intelligence.agent.composer import frame_sentences

    rng = random.Random(f"{SEED}-sentences-{language}")
    zh = language == "zh"
    for _ in range(60):
        a, b = rng.sample(ENTITIES, 2)
        key = rng.choice(sorted(MONEY))
        value_a, value_b = _values(rng, key), _values(rng, key)
        field = METRICS[key][2]
        tool_log = [_fundamentals(a[0], a[1], field, value_a), _fundamentals(b[0], b[1], field, value_b)]
        operands = [{"kind": "target", "name": a[1], "symbol": a[0]}, {"kind": "target", "name": b[1], "symbol": b[0]}]
        sentences, gaps = frame_sentences(
            {"operation": "share_total", "metric": key, "operands": operands}, tool_log, zh
        )
        share = round(value_a / (value_a + value_b) * 100, 2)
        wrong = round(value_a / value_b * 100, 2)
        text = " ".join(sentences)
        assert not gaps and "÷" in text and f"{share:g}%" in text.replace(",", "")
        assert f"≈ {wrong:g}%" not in text


@pytest.mark.parametrize("language", ["zh", "en"])
def test_generated_follow_ups_read_the_frame_by_order_and_reference(language):
    """Ordinal and anaphoric references (前者 / 两者 / the former / both) take the frame's operands in order."""
    from query_intelligence.agent.frame import resolve_frame_question

    zh = language == "zh"
    templates = (
        [
            ("前者的{m}占两者合计的几成", "share_total", [0, 1]),
            ("后者的{m}占两者合计的多少", "share_total", [1, 0]),
            ("两家的{m}加起来多少", "sum", [0, 1]),
            ("两者的{m}共计多少", "sum", [0, 1]),
            ("后者比前者的{m}高多少", "difference", [1, 0]),
        ]
        if zh
        else [
            ("What share of the combined {m} is the former's?", "share_total", [0, 1]),
            ("How much {m} do they make between them?", "sum", [0, 1]),
            ("What is the {m} of the two put together?", "sum", [0, 1]),
            ("By how much does the latter's {m} outstrip the former's?", "difference", [1, 0]),
        ]
    )
    rng = random.Random(f"{SEED}-frame-{language}")
    for _ in range(40):
        a, b = rng.sample(ENTITIES, 2)
        key = rng.choice(sorted(MONEY))
        names_zh, names_en, _field = METRICS[key]
        template, operation, order = rng.choice(templates)
        question = template.format(m=rng.choice(names_zh if zh else names_en))
        frame = {
            "metric": key,
            "operands": [
                {"kind": "target", "name": a[1], "symbol": a[0]},
                {"kind": "target", "name": b[1], "symbol": b[0]},
            ],
        }
        found = resolve_frame_question(question, [{"frame": frame}], [], [])
        assert found is not None, question
        request = found[2]
        pair = [a[0], b[0]]
        assert request["operation"] == operation, question
        assert [item["symbol"] for item in request["operands"]] == [pair[i] for i in order], question
        assert request["operand_form"] in {"ordinal", "anaphoric"}


@pytest.mark.parametrize(
    ("template", "language"),
    [
        ("By how much does {A}'s ROE outstrip Acmeco's?", "en"),
        ("Is {A}'s P/E higher or lower than Globex's?", "en"),
        ("{A}的市盈率比某某科技高多少", "zh"),
        ("{A}和某某科技的净利润加起来多少", "zh"),
    ],
)
def test_a_two_operand_question_never_answers_with_one_operand(template, language):
    """An operand the question names but the resolver does not know is reported, not dropped."""
    from query_intelligence.agent.composer import frame_sentences
    from query_intelligence.agent.frame import resolve_frame_question

    rng = random.Random(f"{SEED}-{template}")
    for row in rng.sample(ENTITIES, 4):
        zh = language == "zh"
        question = template.format(A=row[1] if zh else row[2])
        found = resolve_frame_question(question, [], [_entity(row, zh)], [])
        assert found is not None, question
        request = found[2]
        kinds = [item["kind"] for item in request["operands"]]
        assert "unresolved" in kinds and "target" in kinds, question
        sentences, gaps = frame_sentences(request, [], zh)
        assert not sentences and gaps and ("未能识别" in gaps[0] if zh else "Could not tell" in gaps[0])


@pytest.mark.parametrize(
    ("text", "multiple", "bound"),
    [
        ("贵州茅台的营收有没有五粮液的四倍半", 4.5, "gt"),
        ("中国平安的净利润是不是五粮液的两倍半左右", 2.5, "eq"),
        ("Is Moutai's revenue over three times Wuliangye's?", 3.0, "gt"),
        ("Is Ping An's net profit more than five and a half times Wuliangye's?", 5.5, "gt"),
    ],
)
def test_yes_no_multiples_with_a_half_and_spelled_numbers(text, multiple, bound):
    from query_intelligence.agent.frame import claimed_check, frame_operation

    assert frame_operation(text) == "ratio"
    found = claimed_check(text)
    assert found["multiple"] == multiple and found["bound"] == bound


@pytest.mark.parametrize(
    "text",
    [
        "宁德时代、比亚迪、美的集团这三家，市盈率最贵的比最便宜的高多少",
        "Among CATL, BYD and Midea, what separates the most and least expensive on P/E?",
        "Between the cheapest and the priciest of Zijin, Midea and CATL, how wide is the P/E gap?",
    ],
)
def test_the_most_and_least_expensive_is_a_spread_not_a_judgment(text):
    from query_intelligence.agent.compliance import _JUDGMENT_TRIGGER
    from query_intelligence.agent.frame import frame_operation, without_valuation_extremes

    assert frame_operation(text) == "spread"
    assert not _JUDGMENT_TRIGGER.search(without_valuation_extremes(text))
    # a judgment next to it is still one
    assert _JUDGMENT_TRIGGER.search(without_valuation_extremes(text + " 值得买吗"))


# ---- J8: metrics as their own fields ----


@pytest.mark.parametrize(
    ("text", "metric"),
    [
        ("招商银行的流通市值是多少", "float_market_cap"),
        ("What is ICBC's free-float market cap?", "float_market_cap"),
        ("宁德时代的总市值", "market_cap"),
        ("比亚迪扣非后净利润多少", "profit_dedt"),
        ("What were Midea's earnings excluding non-recurring items?", "profit_dedt"),
        ("美的集团的营业利润", "operating_profit"),
        ("What's BYD's operating profit?", "operating_profit"),
        ("紫金矿业的毛利润", "gross_profit"),
        ("紫金矿业的毛利率", "gross_margin"),
        ("长江电力的利润总额", "total_profit"),
        ("长江电力的净利润", "net_profit"),
    ],
)
def test_metric_names_map_to_their_own_metric(text, metric):
    from query_intelligence.agent.frame import metric_of

    assert metric_of(text) == metric


def test_metric_requests_read_their_own_fields():
    from query_intelligence.agent.coverage import requested_metrics

    assert [m.key for m in requested_metrics("工商银行的流通盘市值")] == ["float_market_cap"]
    assert [m.fields for m in requested_metrics("工商银行的流通盘市值")] == [("circ_mv",)]
    assert "circ_mv" not in requested_metrics("招商银行总市值")[0].fields
    assert [m.key for m in requested_metrics("紫金矿业扣非以后的净利润")] == ["profit_dedt"]
    assert [m.key for m in requested_metrics("比亚迪的营业利润和毛利润")] == ["operating_profit", "gross_profit"]


# ---- J5: several positions; J4: cost basis ----


@pytest.mark.parametrize("seed", range(6))
def test_generated_holdings_value_every_position(seed):
    from query_intelligence.agent.composer import _several_holdings
    from query_intelligence.agent.coverage import holding_positions

    rng = random.Random(f"{SEED}-holdings-{seed}")
    short = {"招商银行": "招行", "工商银行": "工行", "贵州茅台": "茅台"}
    rows = rng.sample(ENTITIES, rng.choice([2, 3, 4]))
    parts, expected = [], 0.0
    closes = {}
    for index, (symbol, name, _english) in enumerate(rows):
        close = round(rng.uniform(5, 300), 2)
        closes[symbol] = (name, close)
        written = short.get(name, name)
        if rng.random() < 0.5:
            lots = rng.randint(1, 30)
            parts.append(f"{written}{lots}手" if index % 2 else f"{lots}手{written}")
            expected += lots * 100 * close
        else:
            count = rng.randint(1, 50) * 100
            parts.append(f"{written}{count}股" if index % 2 else f"{count}股{written}")
            expected += count * close
    question = f"我持有{'、'.join(parts[:-1])}和{parts[-1]}，合计市值多少"
    positions = holding_positions(question)
    assert len(positions) == len(rows), question
    tool_log = [
        {
            "tool": "get_price_history",
            "ok": True,
            "data": {"symbol": s, "name": n, "close": c, "as_of": "2026-09-30", "evidence_id": f"price_{s}"},
        }
        for s, (n, c) in closes.items()
    ]
    sentences = _several_holdings(positions, tool_log, question, True)
    assert sentences, question
    total = re.search(r"合计市值约为 ([\d.]+) 元", sentences[0])
    assert total and abs(float(total.group(1)) - round(expected, 2)) < 0.011, (question, sentences[0])


@pytest.mark.parametrize(
    ("text", "cost", "shares"),
    [
        ("我手里600股宁德时代，买入价320元，现在亏了多少", 320.0, 600),
        ("我2手美的成本价72.5，是赚是亏", 72.5, 200),
        ("我25块钱买的，现在盈亏多少", 25.0, None),
        ("I paid 300 for 400 shares of CATL. Am I up or down?", 300.0, 400),
        ("My average cost is 60 on 3 lots of Midea; what's my P&L?", 60.0, 300),
    ],
)
def test_cost_basis_questions(text, cost, shares):
    from query_intelligence.agent.coverage import cost_basis_request

    found = cost_basis_request(text)
    assert found is not None and found["cost"] == cost and found["shares"] == shares


@pytest.mark.parametrize("zh", [True, False])
def test_cost_basis_sentence_computes_and_verifies(zh):
    from query_intelligence.agent.composer import cost_basis_sentences
    from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore
    from query_intelligence.agent.verifier import verify_answer

    question = (
        "我持有400股宁德时代，成本价300元，浮盈多少" if zh else "I bought 400 shares of CATL at 300; what's my profit?"
    )
    tool_log = [
        {
            "tool": "get_price_history",
            "ok": True,
            "data": {
                "symbol": "300750.SZ",
                "name": "宁德时代",
                "close": 291.11,
                "as_of": "2026-09-30",
                "evidence_id": "price_300750.SZ",
            },
        }
    ]
    sentences = cost_basis_sentences(tool_log, question, zh)
    assert re.search(r"291\.11 − 300 = (?:CNY )?-8\.89", sentences[0]), sentences[0]
    assert re.search(r"400 × \(−8\.89\) = (?:CNY )?-3556", sentences[0]) and "-2.96%" in sentences[0]
    store = EvidenceStore()
    store.add(
        AgentEvidence(
            evidence_id="price_300750.SZ",
            kind="structured",
            source_type="market",
            payload={"close": 291.11, "symbol": "300750.SZ"},
        )
    )
    report = verify_answer({"answer": sentences[0]}, store, query=question, allow_derived=True)
    assert report.passed, report


# ---- J3: the verifier traces a share of a total and a total of several positions ----


def _store(**closes_and_amounts):
    from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore

    store = EvidenceStore()
    for evidence_id, payload in closes_and_amounts.items():
        store.add(AgentEvidence(evidence_id=evidence_id, kind="structured", source_type="market", payload=payload))
    return store


def test_the_verifier_keeps_a_share_of_a_combined_total_and_rejects_the_share_of_the_other():
    from query_intelligence.agent.verifier import verify_answer

    store = _store(f_a={"revenue": 803965000000.0}, f_b={"revenue": 423702000000.0})
    question = "比亚迪的营收在比亚迪与宁德时代之和里占比多少"
    good = "比亚迪营收 8039.65 亿元，宁德时代 4237.02 亿元，比亚迪约占两家合计的 65.49% [f_a][f_b]。"
    english = "BYD's revenue (CNY 803.97 bn) is about 65.49% of the combined total with CATL (CNY 423.7 bn) [f_a][f_b]."
    wrong = "比亚迪营收 8039.65 亿元，宁德时代 4237.02 亿元，比亚迪约占两家合计的 189.75% [f_a][f_b]。"
    assert verify_answer({"answer": good}, store, query=question, allow_derived=True).passed
    assert verify_answer({"answer": english}, store, query=question, allow_derived=True).passed
    assert not verify_answer({"answer": wrong}, store, query=question, allow_derived=True).passed


def test_the_verifier_keeps_a_total_of_three_positions_written_any_way():
    from query_intelligence.agent.verifier import verify_answer

    store = _store(p_a={"close": 41.26}, p_b={"close": 8.28}, p_c={"close": 83.31})
    question = "我有招行3000股、工行8手、比亚迪200股，一共值多少"
    expected = 3000 * 41.26 + 800 * 8.28 + 200 * 83.31  # 147066
    drafts = [
        f"这些持仓按最新收盘价合计约 {expected:.0f} 元 [p_a][p_b][p_c]。",
        f"按最新收盘价，这些持仓总共约 {expected / 1e4:.2f} 万元 [p_a][p_b][p_c]。",
        f"The three positions total about CNY {expected:,.0f} at the latest closes [p_a][p_b][p_c].",
    ]
    for draft in drafts:
        assert verify_answer({"answer": draft}, store, query=question, allow_derived=True).passed, draft
    wrong = f"这些持仓按最新收盘价合计约 {3000 * 41.26:.0f} 元 [p_a][p_b][p_c]。"  # the first position only
    assert not verify_answer({"answer": wrong}, store, query=question, allow_derived=True).passed


# ---- J13 / J14 ----


def test_a_total_says_when_the_rounded_figures_do_not_add_up():
    from query_intelligence.agent.composer import frame_sentences

    tool_log = [
        _fundamentals("600519.SH", "贵州茅台", "revenue", 3793827534.0),
        _fundamentals("000858.SZ", "五粮液", "revenue", 1452833705.0),
    ]
    operands = [
        {"kind": "target", "name": "贵州茅台", "symbol": "600519.SH"},
        {"kind": "target", "name": "五粮液", "symbol": "000858.SZ"},
    ]
    zh, _gaps = frame_sentences({"operation": "sum", "metric": "revenue", "operands": operands}, tool_log, True)
    en, _gaps = frame_sentences({"operation": "sum", "metric": "revenue", "operands": operands}, tool_log, False)
    assert "37.94 亿元" in zh[0] and "14.53 亿元" in zh[0] and "合计 52.47 亿元" in zh[0]
    assert "（按未四舍五入的原值相加）" not in zh[0]  # 37.94 + 14.53 = 52.47
    assert "combined CNY 5.25 bn (summed from the unrounded figures)" in en[0]  # 3.79 + 1.45 = 5.24


def test_english_fundamentals_labels_have_no_chinese():
    from query_intelligence.agent.composer import basis_label

    data = {"definitions": {"revenue_basis": "营业总收入"}}
    assert basis_label(data, "revenue", False, "revenue") == "total operating revenue"
    assert basis_label({"definitions": {"revenue_basis": "营业收入"}}, "revenue", False, "x") == "operating revenue"


def test_a_stock_against_its_industry_average_says_higher_or_lower_and_by_how_much():
    from query_intelligence.agent.composer import frame_sentences

    tool_log = [_fundamentals("000858.SZ", "五粮液", "pe_ttm", 18.4)]
    tool_log[0]["data"]["industry"] = {
        "industry_name": "白酒",
        "evidence_id": "industry_白酒",
        "metrics": {"pe": 23.0},
    }
    request = {
        "operation": "which",
        "metric": "pe",
        "operands": [
            {"kind": "target", "name": "五粮液", "symbol": "000858.SZ"},
            {"kind": "industry", "member": "000858.SZ"},
        ],
    }
    zh, _gaps = frame_sentences(request, tool_log, True)
    en, _gaps = frame_sentences(request, tool_log, False)
    assert "低于" in zh[0] and "低 4.6（约 20%，以白酒行业平均为基数）" in zh[0]
    assert "is lower than" in en[0] and "4.6 below it (about 20% of" in en[0]


# ---- end to end on the shipped default data (extension + corrections layer), template and scripted LLM ----


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


@pytest.fixture(scope="module")
def shipped_agent(shipped_service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    runtime = AgentRuntime(shipped_service, build_registry_for_service(shipped_service), ScriptedLLM([]))
    yield AgentService(runtime, trace_sinks=[])
    runtime.close()


def _chat(agent, name: str, *queries: str) -> list[dict]:
    return [agent.chat(query, session_id=f"r14-{name}") for query in queries]


def test_e2e_share_of_a_combined_total_zh_and_en(shipped_agent):
    (zh,) = _chat(shipped_agent, "share-total-zh", "宁德时代和比亚迪的营收合计中，宁德时代占几成")
    (en,) = _chat(shipped_agent, "share-total-en", "What share of BYD and CATL's combined revenue comes from BYD?")
    assert "frame:share_total:revenue:宁德时代|比亚迪" in zh["route_reasons"]
    assert "4237.02 亿元 ÷ 12276.67 亿元 ≈ 34.51%" in zh["answer"] and zh["verification"]["passed"]
    assert "frame:share_total:revenue:比亚迪|宁德时代" in en["route_reasons"]
    assert "≈ 65.49%" in en["answer"] and "189.75" not in en["answer"] and en["verification"]["passed"]


def test_e2e_totals_in_other_words(shipped_agent):
    plus, total = _chat(shipped_agent, "plus", "紫金矿业加美的集团的营业收入是多少", "那这两家的净利润共计多少")
    assert "frame:sum:revenue:紫金矿业|美的集团" in plus["route_reasons"]
    assert "合计 8075.81 亿元" in plus["answer"] and plus["verification"]["passed"]
    assert "frame:sum:net_profit:紫金矿业|美的集团" in total["route_reasons"]
    assert "合计 957.22 亿元" in total["answer"]
    (english,) = _chat(shipped_agent, "between", "How much net profit do Zijin Mining and Midea make between them?")
    assert "combined CNY 95.72 bn" in english["answer"]


def test_e2e_a_comparative_verb_keeps_both_operands(shipped_agent):
    (zh,) = _chat(shipped_agent, "verb-zh", "长江电力的ROE高过美的集团多少")
    (en,) = _chat(shipped_agent, "verb-en", "By how many points does Midea's ROE outpace Yangtze Power's?")
    assert "两者相差 4.1 个百分点（美的集团更高）" in zh["answer"]
    assert "a difference of 4.1 percentage points (Midea Group is higher)" in en["answer"]


def test_e2e_several_positions_and_a_carried_one(shipped_agent):
    (three,) = _chat(shipped_agent, "three", "我手里有工行10手、招行1000股，还有紫金矿业500股，合计市值多少")
    assert "合计市值约为 64435 元" in three["answer"] and three["verification"]["passed"]
    _first, carried = _chat(
        shipped_agent, "carried", "紫金矿业今天收盘价多少", "我有2手，另外还有300股美的，一共值多少"
    )
    assert "2 手（200 股）紫金矿业 200 × 29.79 = 5958 元" in carried["answer"]
    assert "300 股美的集团 300 × 80.1 = 24030 元" in carried["answer"]
    assert "合计市值约为 29988 元" in carried["answer"]


def test_e2e_cost_basis_and_the_rebound_from_the_low(shipped_agent):
    _first, cost = _chat(shipped_agent, "cost", "美的集团最新股价", "我是按85块买进的，现在亏了多少")
    assert "美的集团每股浮亏 80.1 − 85 = -4.9 元（-5.76%）" in cost["answer"] and cost["verification"]["passed"]
    (rebound,) = _chat(shipped_agent, "rebound", "宁德时代从一年内最低点反弹了多少")
    assert "(291.11 − 286.8) ÷ 286.8 ≈ 1.5%" in rebound["answer"] and rebound["verification"]["passed"]


def test_e2e_metrics_from_their_own_fields(shipped_agent):
    (circ,) = _chat(shipped_agent, "circ", "工商银行流通市值多少")
    (dedt,) = _chat(shipped_agent, "dedt", "宁德时代扣非净利润是多少")
    (oper,) = _chat(shipped_agent, "oper", "宁德时代营业利润是多少")
    assert "流通市值" in circ["answer"] and circ["verification"]["passed"]
    assert "扣非净利润" in dedt["answer"] and "722.01" not in dedt["answer"].split("扣非净利润")[1][:12]
    assert "没有宁德时代的营业利润" in oper["answer"]


def test_e2e_a_stock_against_its_industry_average(shipped_agent):
    (verdict,) = _chat(shipped_agent, "industry", "贵州茅台的市净率跟白酒行业平均比是高还是低")
    assert "frame:which:pb:" in " ".join(verdict["route_reasons"])
    assert "高于" in verdict["answer"] and "以白酒行业平均为基数" in verdict["answer"]


def _shipped_llm(shipped_service, steps):
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


def test_llm_a_correct_share_of_the_total_is_kept(shipped_service):
    from query_intelligence.agent.llm import final_turn

    draft = (
        "比亚迪营业收入 8039.65 亿元，宁德时代 4237.02 亿元，比亚迪约占两家合计的 65.49% "
        "[fundamental_002594.SZ][fundamental_300750.SZ]。"
    )
    runtime, service = _shipped_llm(shipped_service, [final_turn({"answer": draft, "evidence_used": []})])
    try:
        result = service.chat("比亚迪的营收在它和宁德时代两家之和中占多大比重", session_id="r14-llm-keep", mode="agent")
    finally:
        runtime.close()
    assert result["answer"].startswith("比亚迪营业收入 8039.65 亿元") and "65.49%" in result["answer"]
    assert not any("template_fallback" in flag for flag in result.get("degraded") or [])
    assert result["verification"]["passed"]


def test_llm_a_failed_draft_falls_back_to_the_correct_computation(shipped_service):
    from query_intelligence.agent.llm import final_turn

    wrong = "三笔持仓合计约 41260 元 [price_600036.SH]。"  # the first position only, and a wrong total
    steps = [final_turn({"answer": wrong, "evidence_used": []}) for _ in range(3)]
    runtime, service = _shipped_llm(shipped_service, steps)
    try:
        result = service.chat(
            "我有招行1000股、工行10手和紫金矿业500股，按收盘价一共值多少", session_id="r14-llm-fb", mode="agent"
        )
    finally:
        runtime.close()
    assert "合计市值约为 64435 元" in result["answer"]
    assert "合计约 41260 元" not in result["answer"]
    assert "verification_failed:template_fallback" in (result.get("degraded") or [])


@pytest.mark.parametrize(
    ("text", "operation"),
    [
        ("Now add Wuliangye's to Ping An's. What do you get?", "sum"),
        ("And combined?", "sum"),
        ("那合计呢", "sum"),
        ("Should I add more to Moutai?", None),
        ("I'd like to add to my Ping An position; how much?", None),
        ("五粮液的加权ROE比茅台低多少", "difference"),
        ("我打算加仓五粮液", None),
    ],
)
def test_more_total_wordings_and_their_negatives(text, operation):
    from query_intelligence.agent.frame import frame_operation

    assert frame_operation(text) == operation


# ---- classes the round-9 slice still failed at the first round-14 run (after exposure; own wording) ----


def test_after_exposure_classes_in_own_wording():
    from query_intelligence.agent.coverage import holding_positions, holding_value_request, requested_price_fields
    from query_intelligence.agent.frame import claimed_check, frame_operation, metric_of
    from query_intelligence.agent.planner import plan_from_nlu

    # a total of what two companies earned
    assert frame_operation("紫金矿业和美的集团两家一共赚了多少") == "sum"
    assert metric_of("紫金矿业和美的集团两家一共赚了多少") == "net_profit"
    assert metric_of("贵州茅台一股赚多少") == "eps"  # still EPS
    # market value is the market cap; a bare follow-up total refers to "that"
    assert metric_of("What is the combined market value of Midea and Zijin?") == "market_cap"
    assert frame_operation("All told, how much is that?") == "sum"
    # "at least N times what B earns"
    assert claimed_check("Does Zijin earn at least twice what Midea earns?")["multiple"] == 2.0
    # an English holding total asked in the next sentence
    text = "I hold 400 Midea shares and 900 Zijin Mining shares. Total value at the last close?"
    assert holding_value_request(text) is not None and len(holding_positions(text)) == 2
    # "现在比52周最高收盘价低了…" is a distance below the high; a name with 比 in it is not
    assert requested_price_fields("紫金矿业现在比52周最高收盘价低了多少").off_high_52w
    assert not requested_price_fields("比亚迪52周最高和最低").off_high_52w
    # a 52-week question fetches the price history
    nlu = {
        "entities": [{"symbol": "600036.SH", "canonical_name": "招商银行", "entity_type": "stock"}],
        "source_plan": ["fundamental_sql"],
        "question_style": "fact",
        "raw_query": "How far below its one-year high is China Merchants Bank now?",
    }
    plan = plan_from_nlu(nlu)
    assert "get_price_history" in [call.tool for call in plan.calls]
