"""Round-3b multi-turn rules: entity-less follow-ups, group references, sector questions, price details,
hedging on follow-ups, stated gaps, English aliases, and clarification replies typed into the chat box.

Written from the failure classes of the independent multi-turn set (``multiturn_v1``) with new examples; no query
here repeats one of that set."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest
from agent_fakes import StubService, build_fake_registry

from query_intelligence.agent.compliance import apply_compliance
from query_intelligence.agent.composer import compose_template
from query_intelligence.agent.coverage import (
    industry_gaps,
    macro_gaps,
    non_stock_fundamental_gaps,
    requested_metrics,
    requested_price_fields,
    requested_quarter,
)
from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore
from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.memory import (
    clarification_reply_text,
    discussed_targets,
    inherit_session_context,
    is_target_only_reply,
    resolve_comparison_anchor,
    resolve_ellipsis,
    resolve_group_reference,
    strip_filler,
)
from query_intelligence.agent.planner import plan_from_nlu
from query_intelligence.agent.router import drop_fuzzy_concepts, has_follow_up_cue, has_macro_content, off_topic_request
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.verifier import verify_answer

ROOT = Path(__file__).resolve().parents[1]

MOUTAI = {"name": "贵州茅台", "symbol": "600519.SH"}
WULIANGYE = {"name": "五粮液", "symbol": "000858.SZ"}
PINGAN = {"name": "中国平安", "symbol": "601318.SH"}
ETF = {"name": "沪深300ETF", "symbol": "510300.SH"}


def _turn(query: str, *entities: dict, named: bool = True, macro: list[str] | None = None) -> dict:
    return {
        "query": query,
        "effective_query": query,
        "entities": list(entities),
        "named": named,
        "macro_topics": macro or [],
        "route": "workflow",
    }


# --------------------------------------------------------------------------- class 1: entity-less follow-ups


def test_off_topic_tasks_are_detected_without_catching_finance_questions():
    for query, label in [
        ("能不能用Python帮我抓一下五粮液的历史行情", "coding"),
        ("Write me some code that pulls Moutai's quotes", "coding"),
        ("把这段话翻译成英文", "translation"),
        ("帮我订个明天去北京的酒店", "travel"),
        ("明天北京天气怎么样", "weather"),
        ("写一首关于股市的诗", "writing"),
    ]:
        assert off_topic_request(query) == label, query
    for query in ("天气转暖对白酒消费有影响吗", "贵州茅台的股票代码是多少", "中国平安的市盈率是多少"):
        assert off_topic_request(query) is None, query


def test_follow_up_cues_mark_finance_follow_ups_only():
    for query in ("怎么又跌了", "增长快吗", "The 5-day return?", "What about the trend?", "舆情怎么样", "哪家更好"):
        assert has_follow_up_cue(query), query
    for query in ("你好", "谢谢", "你是谁", "What's your name?"):
        assert not has_follow_up_cue(query), query


def test_macro_anchor_does_not_fire_inside_company_margins():
    assert not has_macro_content("那它的毛利率呢") and not has_macro_content("净利率多少")
    assert has_macro_content("存款利率会降吗") and has_macro_content("What's the CGB yield now?")


def test_session_inheritance_takes_the_latest_target_or_macro_topic():
    turns = [_turn("五粮液的市盈率", WULIANGYE), _turn("那沪深300ETF呢", ETF)]
    assert inherit_session_context("怎么又跌了", turns) == (
        "沪深300ETF怎么又跌了",
        "session_inherit:target->沪深300ETF",
    )
    assert inherit_session_context("And the 10-day return?", turns)[0] == "And the 10-day return for 沪深300ETF?"

    macro = [*turns, _turn("M2最新多少", macro=["M2"])]
    assert inherit_session_context("说明货币偏宽松吗？", macro) == (
        "M2：说明货币偏宽松吗？",
        "session_inherit:macro->M2",
    )
    # no finance cue, too long, or nothing discussed: no inheritance
    assert inherit_session_context("谢谢", turns) is None
    assert (
        inherit_session_context("请详细讲讲你对今后十年整个资本市场改革方向和监管框架演变趋势的看法吧", turns) is None
    )
    assert inherit_session_context("怎么又跌了", []) is None


# --------------------------------------------------------------------------- class 2: group references


def test_ordinal_references_follow_the_order_the_user_named_them():
    turns = [
        _turn("茅台和平安的ROE", MOUTAI, PINGAN),
        _turn("这两家哪个便宜", PINGAN, MOUTAI, named=False),  # a rewrite has no order of its own
    ]
    assert resolve_group_reference("后者的毛利率呢", turns) == ("中国平安的毛利率呢", "group_reference:后者->中国平安")
    assert resolve_group_reference("What about the former's P/B?", turns)[0] == "What about 贵州茅台's P/B?"
    assert resolve_group_reference("后者呢", turns[:0]) is None


def test_triple_and_bare_which_references():
    turns = [_turn("茅台PB", MOUTAI), _turn("五粮液PB", WULIANGYE), _turn("平安PB", PINGAN)]
    assert resolve_group_reference("这三家谁更贵", turns)[0] == "贵州茅台和五粮液和中国平安谁更贵"
    assert resolve_group_reference("这三家谁更贵", turns[:2]) is None

    pair = [_turn("平安和五粮液的营收", PINGAN, WULIANGYE)]
    assert resolve_group_reference("哪家利润更多", pair)[0] == "中国平安和五粮液哪家利润更多"
    assert resolve_group_reference("哪家利润更多", turns[:1]) is None  # one target: nothing to choose from


def test_discussed_targets_keep_the_order_of_mention():
    turns = [_turn("平安和五粮液", PINGAN, WULIANGYE), _turn("茅台", MOUTAI), _turn("平安", PINGAN)]
    assert [item["name"] for item in discussed_targets(turns, limit=3)] == ["五粮液", "贵州茅台", "中国平安"]


def test_comparison_naming_one_side_adds_the_earlier_target():
    turns = [_turn("And what was its net profit?", WULIANGYE)]
    turns[0]["effective_query"] = "And 五粮液's net profit?"
    rewritten, reason = resolve_comparison_anchor("Is it higher than Ping An's?", turns, [{"symbol": "601318.SH"}])
    assert rewritten == "Is 五粮液's net profit higher than Ping An's?" and reason == "comparison_anchor:+五粮液"
    zh = resolve_comparison_anchor(
        "和证券ETF相比呢",
        [_turn("创业板ETF收盘", {"name": "创业板ETF", "symbol": "159915.SZ"})],
        [{"symbol": "512880.SH"}],
    )
    assert zh[0].startswith("创业板ETF")
    # a comparison with the same target, or without a comparison word, is left alone
    assert resolve_comparison_anchor("五粮液呢", turns, [{"symbol": "600519.SH"}]) is None


def test_fillers_and_period_only_follow_ups_keep_the_metric():
    turns = [_turn("贵州茅台2025年的营收是多少", MOUTAI)]
    assert strip_filler("OK. 那 PB 呢") == "那 PB 呢"
    assert resolve_ellipsis("2023年的呢？", turns, [])[0] == "贵州茅台2023年的营收呢？"
    assert resolve_ellipsis("And for 2021?", turns, [])[0] == "营收 for 2021 for 贵州茅台?"
    assert resolve_ellipsis("Fine. What about Wuliangye?", turns, [{"canonical_name": "五粮液"}])[0] == (
        "What is 五粮液's 营收?"
    )


# --------------------------------------------------------------------------- class 4 / 6: requested details and gaps


def test_requested_price_fields():
    request = requested_price_fields("最近三个交易日的收盘价和成交量")
    assert request.closes == 3 and request.volume and not request.high
    assert requested_price_fields("Give me the last four closes").closes == 4
    english = requested_price_fields("What was the session high and low, and the 10-day return?")
    assert english.high and english.low and english.return_days == (10,)
    assert requested_price_fields("收盘价在20日均线下方吗").moving_averages == (20,)
    assert requested_price_fields("收盘价在20日均线下方吗").above_ma
    # "谁的ROE最高" and "is the P/B high or low" are not requests for the day's high/low
    assert not requested_price_fields("三家里谁的ROE最高").high
    assert not requested_price_fields("Is the P/B high or low?").low


def test_generic_growth_and_market_cap_are_requested_metrics_but_not_for_macro():
    assert [metric.key for metric in requested_metrics("净利润增速是多少")] == ["profit_growth"]
    assert [metric.key for metric in requested_metrics("增长率多少")] == ["growth"]
    assert [metric.key for metric in requested_metrics("五粮液总市值多少")] == ["market_cap"]


def test_quarter_requests():
    assert requested_quarter("今年三季度的营收") == ("三季度", 9)
    assert requested_quarter("Q2 net profit?") == ("Q2", 6)
    assert requested_quarter("上半年利润") == ("上半年", 6)
    assert requested_quarter("2025年营收") is None


FUNDAMENTALS_WITH_INDUSTRY = {
    "tool": "get_fundamentals",
    "ok": True,
    "arguments": {"target": "601318.SH"},
    "data": {
        "symbol": "601318.SH",
        "name": "中国平安",
        "report_date": "2025-12-31",
        "metrics": {"pe_ttm": 8.7, "pb": 1.1, "roe": 15.2, "revenue": 1218000000000, "net_profit": 121000000000},
        "evidence_id": "fundamental_601318.SH",
        "industry": {
            "industry_name": "保险",
            "metrics": {"pe": 11.8, "pb": 1.45, "pct_change": 0.68},
            "evidence_id": "industry_保险",
        },
    },
    "evidence_ids": ["fundamental_601318.SH", "industry_保险"],
}


def test_industry_metric_gaps_and_quarter_gaps_are_stated():
    assert industry_gaps("净利润跟保险同业比呢", [FUNDAMENTALS_WITH_INDUSTRY], zh=True) == [
        "当前数据源的保险行业快照没有净利润，无法给出行业层面的这一项。"
    ]
    assert industry_gaps("保险行业市净率", [FUNDAMENTALS_WITH_INDUSTRY], zh=True) == []
    draft = compose_template([FUNDAMENTALS_WITH_INDUSTRY], zh=True, query="中国平安三季度的营收")
    assert draft["answer"].startswith("当前数据中没有所问的三季度数据")


def test_etf_fundamentals_and_missing_macro_indicators_are_named():
    gaps = non_stock_fundamental_gaps(
        "它的ROE多少", [], zh=True, names={"159915.SZ": "创业板ETF"}, types={"159915.SZ": "etf"}
    )
    assert gaps and "创业板ETF（159915.SZ）是ETF" in gaps[0]
    macro = {
        "tool": "get_macro_indicators",
        "ok": True,
        "data": {"indicators": [{"code": "M2_CN", "value": 8.1, "evidence_id": "macro_M2_CN"}]},
    }
    assert macro_gaps("What are GDP growth and M2?", [macro], zh=False) == [
        "The current data sources do not include GDP, so that part cannot be answered; only the available macro "
        "indicators are listed below."
    ]
    assert macro_gaps("M2多少", [macro], zh=True) == []


PRICE_ENTRY = {
    "tool": "get_price_history",
    "ok": True,
    "arguments": {"target": "159915.SZ"},
    "data": {
        "symbol": "159915.SZ",
        "name": "创业板ETF",
        "as_of": "2026-04-22",
        "close": 2.465,
        "open": 2.438,
        "high": 2.471,
        "low": 2.431,
        "pct_change_1d": 0.86,
        "volume": 0,
        "recent_closes": [{"date": "2026-04-21", "close": 2.444}, {"date": "2026-04-22", "close": 2.465}],
        "evidence_id": "price_159915.SZ",
    },
    "evidence_ids": ["price_159915.SZ"],
}


def test_template_states_requested_price_details_names_missing_ones_and_verifies():
    draft = compose_template([PRICE_ENTRY], zh=True, query="创业板ETF近三个交易日收盘价、开盘价和成交量")
    store = EvidenceStore()
    store.add(
        AgentEvidence(
            evidence_id="price_159915.SZ", kind="structured", source_type="market_api", payload=PRICE_ENTRY["data"]
        )
    )

    assert "只有最近2个交易日的收盘价（所问为3个）" in draft["answer"]
    assert "2026-04-21 2.444" in draft["answer"] and "开盘价 2.438" in draft["answer"]
    assert "当前数据中没有创业板ETF的成交量" in draft["answer"]  # a zero volume is a missing value, not 0 shares
    assert verify_answer(draft, store, market_precedence=False, require_citations=False).passed


def test_template_compares_close_with_the_requested_moving_average():
    entry = {
        "tool": "compute_indicators",
        "ok": True,
        "arguments": {"target": "510300.SH"},
        "data": {
            "name": "沪深300ETF",
            "latest_close": 4.811,
            "ma5": 4.7674,
            "pct_change_nd": {"pct_3d": 1.5193},
            "unavailable": ["ma20"],
            "evidence_id": "indicators_510300.SH",
        },
        "evidence_ids": ["indicators_510300.SH"],
    }
    english = compose_template([entry], zh=False, query="Is it above the 5-day moving average? And MA20?")

    assert "latest close 4.811 is above its MA5 of 4.7674" in english["answer"]
    assert "The current data has no MA20" in english["answer"]


# --------------------------------------------------------------------------- class 5: hedging


def test_judgment_is_detected_on_the_effective_question_too():
    answer = {"answer": "五粮液最新收盘价 100.64 [price_000858.SZ]。", "key_points": [], "limitations": []}
    raw_only, notes = apply_compliance(answer, query="五粮液", nlu_result={}, language="zh")
    guarded, notes = apply_compliance(
        answer, query="五粮液", nlu_result={}, language="zh", effective_query="五粮液这个能买吗？"
    )

    assert "conditional_prefix" not in apply_compliance(answer, query="五粮液", nlu_result={}, language="zh")[1]
    assert raw_only["answer"] == answer["answer"]
    assert "conditional_prefix" in notes and guarded["answer"].startswith("基于当前证据只能做条件性判断")


@pytest.mark.parametrize(
    "query",
    ["现在是不是该上车了", "它俩谁更便宜", "估值偏贵还是便宜", "Is this a bear market signal?", "Is it an uptrend?"],
)
def test_new_judgment_markers_get_the_conditional_prefix(query):
    answer = {"answer": "Close 1409.5 [price_600519.SH].", "key_points": [], "limitations": []}
    _, notes = apply_compliance(answer, query=query, nlu_result={}, language="en")
    assert "conditional_prefix" in notes


def test_interpretation_questions_get_the_causal_caveat():
    answer = {"answer": "PMI 50.6 [macro_PMI_CN].", "key_points": [], "limitations": []}
    for query in ("这说明经济转好了吗", "What does that signal?", "股价反映了这些消息吗"):
        _, notes = apply_compliance(answer, query=query, nlu_result={}, language="en")
        assert "causal_caveat" in notes or "conditional_prefix" in notes, query


# --------------------------------------------------------------------------- clarification replies in the chat box


def test_target_only_reply_detection():
    moutai_nlu = {
        "normalized_query": "the 贵州茅台",
        "entities": [
            {"canonical_name": "贵州茅台", "symbol": "600519.SH", "entity_type": "stock", "mention": "贵州茅台"}
        ],
    }
    assert clarification_reply_text("I mean the Moutai one.") == "Moutai one"
    assert is_target_only_reply("I mean the Moutai", moutai_nlu)
    full_question = {**moutai_nlu, "normalized_query": "贵州茅台昨天收在多少点"}
    assert not is_target_only_reply("贵州茅台昨天收在多少点", full_question)
    assert not is_target_only_reply("谢谢", {"normalized_query": "谢谢", "entities": []})


def _stub_service() -> AgentService:
    runtime = AgentRuntime(StubService(), build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    return AgentService(runtime, trace_sinks=[])


def test_a_target_typed_into_chat_answers_the_pending_clarification():
    service = _stub_service()
    first = service.chat("这只股票能买吗", session_id="typed")
    answered = service.chat("贵州茅台", session_id="typed")

    assert first["status"] == "needs_clarification"
    assert answered["status"] == "ok" and "clarified:贵州茅台" in answered["route_reasons"]
    assert "conditional_prefix" in answered["compliance_notes"]  # the pending question was a judgment
    assert service.pending_clarification("typed") is None


def test_a_new_question_while_a_clarification_is_pending_starts_a_new_turn():
    service = _stub_service()
    service.chat("这只股票能买吗", session_id="new-turn")
    result = service.chat("今天天气怎么样", session_id="new-turn")

    assert result["status"] == "ok" and result["route"] == "refuse"


# --------------------------------------------------------------------------- planner


def _nlu(query: str, entities: list[dict], **extra) -> dict:
    return {"raw_query": query, "normalized_query": query, "entities": entities, "source_plan": [], **extra}


STOCK = {"canonical_name": "贵州茅台", "symbol": "600519.SH", "entity_type": "stock"}


def test_planner_calls_the_tool_that_has_the_requested_price_detail():
    closes = plan_from_nlu(_nlu("贵州茅台最近三个交易日的收盘价", [STOCK]))
    returns = plan_from_nlu(_nlu("What is the 5-day return for 贵州茅台?", [STOCK]))
    change = plan_from_nlu(_nlu("贵州茅台's percentage change?", [STOCK]))

    assert [call.tool for call in closes.calls] == ["get_price_history"]
    assert "compute_indicators" in [call.tool for call in returns.calls]
    assert [call.tool for call in change.calls] == ["get_price_history"]


def test_planner_sector_question_uses_the_industry_snapshot_and_margin_is_not_macro():
    sector = plan_from_nlu(
        _nlu("白酒板块估值高吗", [{"canonical_name": "白酒", "entity_type": "sector"}], source_plan=["macro_sql"])
    )
    margin = plan_from_nlu(_nlu("贵州茅台的毛利率", [STOCK], source_plan=["fundamental_sql"]))

    assert [(call.tool, call.arguments) for call in sector.calls] == [("get_fundamentals", {"target": "白酒"})]
    assert "get_macro_indicators" not in [call.tool for call in margin.calls]


def test_fundamentals_tool_returns_an_industry_snapshot_for_an_industry_name(offline_service):
    from query_intelligence.agent.tools import build_registry_for_service

    registry = build_registry_for_service(offline_service)
    industry = registry.run("get_fundamentals", {"target": "保险"})
    unknown = registry.run("get_fundamentals", {"target": "地产"})

    assert industry.ok and [item.evidence_id for item in industry.evidence] == ["industry_保险"]
    assert industry.data["industry"]["metrics"]["pb"] == 1.45 and industry.data["metrics"] == {}
    assert not unknown.ok and unknown.error.code == "not_found"


# --------------------------------------------------------------------------- guard_in with the offline NLU


def _guard(offline_service, query: str, turns: list[dict]) -> dict:
    runtime = AgentRuntime(offline_service, build_fake_registry(), None)
    try:
        state = runtime.initial_state(query, mode="auto")
        state["turns"] = turns
        return runtime.guard_in(state)
    finally:
        runtime.close()


def _symbols(decision: dict) -> set:
    return {entity.get("symbol") for entity in decision["nlu"]["entities"] if entity.get("symbol")}


def test_entity_less_follow_up_inherits_the_target_instead_of_being_refused(offline_service):
    decision = _guard(offline_service, "怎么又跌了？", [_turn("五粮液的收盘价是多少", WULIANGYE)])

    assert decision["route"] != "refuse" and "000858.SZ" in _symbols(decision)
    assert "session_inherit:target->五粮液" in decision["route_reasons"]


def test_off_topic_and_out_of_coverage_requests_do_not_inherit(offline_service):
    turns = [_turn("贵州茅台的市盈率", MOUTAI)]
    coding = _guard(offline_service, "用Python写个脚本把股价存到Excel", turns)
    foreign = _guard(offline_service, "微软呢？", turns)

    assert coding["route"] == "refuse" and "off_topic_request:coding" in coding["route_reasons"]
    assert not any(reason.startswith(("session_inherit", "ellipsis")) for reason in coding["route_reasons"])
    assert foreign["route"] == "refuse" and foreign["refusal_category"] == "out_of_coverage:foreign_equity"


def test_standalone_macro_question_in_a_stock_conversation_stays_macro(offline_service):
    decision = _guard(offline_service, "What is the CGB yield now?", [_turn("沪深300ETF最新价", ETF)])

    assert decision["route"] != "refuse" and not _symbols(decision)
    assert not any(reason.startswith("session_inherit") for reason in decision["route_reasons"])


def test_sector_question_keeps_the_discussed_member(offline_service):
    decision = _guard(offline_service, "白酒行业估值现在高不高", [_turn("五粮液收盘价", WULIANGYE)])

    assert "000858.SZ" in _symbols(decision) and "sector_member:白酒->五粮液" in decision["route_reasons"]


def test_ambiguous_abbreviation_resolves_to_the_discussed_company(offline_service):
    # Standalone, "平安" links to 中国平安; in a conversation about 平安银行 it means 平安银行.
    bank = {"name": "平安银行", "symbol": "000001.SZ"}
    decision = _guard(offline_service, "平安的分红多少", [_turn("平安银行的市净率", bank)])

    assert "000001.SZ" in _symbols(decision) and "601318.SH" not in _symbols(decision)
    assert "session_disambiguation:平安->平安银行" in decision["route_reasons"]
    assert "601318.SH" in _symbols(_guard(offline_service, "平安的分红多少", [_turn("中国平安的市净率", PINGAN)]))


def test_pronoun_question_drops_fuzzy_company_noise():
    nlu = {
        "entities": [
            {"canonical_name": "长江投资", "symbol": "600119.SH", "entity_type": "stock", "match_type": "alias_fuzzy"}
        ]
    }
    kept, reasons = drop_fuzzy_concepts(nlu, "它适合长期持有吗")
    assert kept["entities"] == [] and reasons == ["dropped_fuzzy_concept:长江投资"]
    assert drop_fuzzy_concepts(nlu, "长江投资适合长期持有吗")[0]["entities"]  # named in the question: kept


def test_elliptical_opening_without_antecedent_is_clarified(offline_service):
    assert _guard(offline_service, "And what about the dividend yield?", [])["route"] == "clarify"


# --------------------------------------------------------------------------- alias assets


def _alias_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("path", [ROOT / "data" / "alias_table.csv", ROOT / "data" / "runtime" / "alias_table.csv"])
def test_alias_tables_keep_crlf_and_unique_ids(path):
    raw = path.read_bytes()
    rows = _alias_rows(path)

    assert raw.count(b"\r\n") == raw.count(b"\n")  # every line ends with CRLF
    assert len({row["alias_id"] for row in rows}) == len(rows)
    assert any(row["normalized_alias"] == "csi 300 index" for row in rows)


@pytest.mark.parametrize(
    ("query", "symbol_or_type"),
    [
        ("How did the CSI 300 index do?", "000300.SH"),
        ("CSI 300 ETF close?", "510300.SH"),
        ("Where does the 10-year CGB yield stand today?", "macro"),
    ],
)
def test_english_aliases_resolve(offline_service, query, symbol_or_type):
    nlu = offline_service.analyze_query(query)
    if symbol_or_type == "macro":
        assert "十年期国债收益率" in nlu["normalized_query"]
    else:
        assert symbol_or_type in {entity.get("symbol") for entity in nlu["entities"]}


@pytest.mark.parametrize("query", ["How is the CSI 3000 doing?", "cbg yield", "baiju prices", "CSI 500 news"])
def test_short_english_aliases_cause_no_fuzzy_false_hits(offline_service, query):
    nlu = offline_service.analyze_query(query)
    assert not {entity.get("symbol") for entity in nlu["entities"]} & {"000300.SH", "510300.SH"}
    assert "十年期国债收益率" not in nlu["normalized_query"] and "白酒" not in nlu["normalized_query"]
