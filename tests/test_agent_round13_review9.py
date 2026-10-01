"""Round-13 rules for the round-9 review's I1, I3, I4, I5, I6, I7 and I9, in the author's own wording (v1 snapshot
names; the review's repro strings and its hidden slice are not used, and ``tests/test_agent_eval.py`` checks that no
test or dev string copies a committed slice or a quoted review probe).

* I1: a share ("A相当于B的百分之几", "只有…的几成", "what percentage / fraction of B") is A ÷ B × 100, written out with
  the division, not the relative difference (A − B) ÷ B.
* I3: a total of two or more operands ("合在一起", "put together", "add up"), opening a conversation clarified.
* I4: the spread of three or more operands ("最高减最低", "highest and lowest"), and a two-operand comparison of three
  named targets computed for every pair, said as such (no operand dropped).
* I5: several positions in one holding question, each with its own count and unit, and their total.
* I6: verb comparatives and "换几股" (one share of A buys how many of B).
* I7: a factual "more than double / 一半以上 / 两倍还多" question answered with the ratio and yes or no, not hedged.
* I9: "<company>'s Hong Kong stock" with a possessive is the Hong Kong line, out of coverage.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("text", "operation"),
    [
        ("后者占前者的百分之几", "share"),
        ("五粮液的净资产收益率只有茅台的几成", "share"),
        ("Ping An's P/B is what percentage of Wuliangye's?", "share"),
        ("What fraction of Moutai's turnover does Wuliangye trade?", "share"),
        ("五粮液比茅台低百分之多少", "relative"),  # still a relative difference
        ("两者合在一起有多少", "sum"),
        ("Their turnover put together?", "sum"),
        ("这三只里头最高减最低是多少个点", "spread"),
        ("Of Moutai, Wuliangye and Ping An, how far apart are the highest and lowest P/E?", "spread"),
        ("一股茅台能换几股五粮液", "ratio"),
        ("Is Moutai's net profit more than double Wuliangye's?", "ratio"),
        ("中国平安的市净率是不是五粮液的一半以上", "ratio"),
    ],
)
def test_review9_operations(text, operation):
    from query_intelligence.agent.frame import frame_operation

    assert frame_operation(text) == operation


def test_review9_bounds_of_a_stated_multiple():
    from query_intelligence.agent.frame import claimed_check

    assert claimed_check("Is Moutai's net profit more than double Wuliangye's?")["bound"] == "gt"
    assert claimed_check("中国平安的市净率是不是五粮液的一半以上")["multiple"] == 0.5
    assert claimed_check("茅台的市净率大概是五粮液的1.5倍吗")["bound"] == "eq"


def test_review9_double_as_a_forecast_still_hedges():
    from query_intelligence.agent.compliance import _JUDGMENT_TRIGGER

    assert _JUDGMENT_TRIGGER.search("Could Moutai double from here?")
    assert not _JUDGMENT_TRIGGER.search("Is Moutai's net profit more than double Wuliangye's?")


def test_review9_hong_kong_line_with_a_possessive():
    from query_intelligence.agent.coverage import asks_h_share

    assert asks_h_share("What's Ping An’s Hong Kong stock trading at?")
    assert asks_h_share("Where did China Ping An's Hong Kong shares end the day?")
    assert not asks_h_share("How did Ping An's A shares close?")


def test_review9_holding_positions():
    from query_intelligence.agent.coverage import bind_positions, holding_positions

    text = "我持有300股五粮液和2手中国平安，一共值多少钱"
    positions = holding_positions(text)
    assert [(p["shares"], p["lots"]) for p in positions] == [(300, 0), (200, 2)]
    candidates = {"000858.SZ": ["五粮液"], "601318.SH": ["中国平安", "平安"]}
    assert bind_positions(text, positions, candidates) == ["000858.SZ", "601318.SH"]
    assert holding_positions("我有200股五粮液，值多少") == []  # one position: the single-holding path


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
    return [agent.chat(query, session_id=f"r13r9-{name}") for query in queries]


def test_i1_a_share_is_computed_as_a_share(agent):
    *_, share = _session(agent, "share", "贵州茅台的净利润", "那五粮液的呢", "后者占前者的百分之几")
    assert "frame:share:net_profit:五粮液|贵州茅台" in share["route_reasons"]
    assert "378 亿元 ÷ 823.2 亿元 ≈ 45.92%" in share["answer"]
    assert "低约" not in share["answer"] and share["verification"]["passed"]
    rate = agent.chat("五粮液的净资产收益率只有茅台的几成", session_id="r13r9-share-rate")
    assert "29.4% ÷ 33% ≈ 89.09%" in rate["answer"] and rate["verification"]["passed"]
    english = agent.chat("What fraction of Moutai's turnover does Wuliangye trade?", session_id="r13r9-share-en")
    assert "≈ 38.29%" in english["answer"] and english["verification"]["passed"]


def test_i3_totals_and_an_opening_total(agent):
    *_, total = _session(agent, "sum", "贵州茅台的净利润", "五粮液的呢", "两者合在一起有多少")
    assert "合计 1201.2 亿元" in total["answer"] and total["verification"]["passed"]
    *_, english = _session(agent, "sum-en", "Moutai's turnover?", "And for Wuliangye?", "Their turnover put together?")
    assert "combined CNY 5.25 bn" in english["answer"]
    opening = agent.chat("这两家合计多少", session_id="r13r9-sum-open")
    assert opening.get("status") == "needs_clarification" or opening.get("route") == "clarify"


def test_i4_spread_and_every_pair(agent):
    *_, spread = _session(agent, "spread", "茅台ROE", "五粮液呢", "中国平安呢", "这三只里头最高减最低是多少个点")
    assert "最高的贵州茅台比最低的中国平安高 17.8 个百分点" in spread["answer"]
    pairs = agent.chat("茅台、五粮液、平安三家的ROE各差多少", session_id="r13r9-pairs")
    assert "问题涉及 3 个标的，逐对计算" in pairs["answer"]
    assert "五粮液与中国平安相差 14.2 个百分点" in pairs["answer"]
    assert not any(reason.startswith("frame:dropped_non_operand") for reason in pairs["route_reasons"])


def test_i5_several_positions(agent):
    held = agent.chat("我持有300股五粮液和2手中国平安，一共值多少钱", session_id="r13r9-hold")
    assert "300 × 100.64 = 30192 元" in held["answer"] and "200 × 53.61 = 10722 元" in held["answer"]
    assert "合计市值约为 40914 元" in held["answer"] and held["verification"]["passed"]
    _first, carried = _session(agent, "hold-carried", "中国平安收盘价", "我还有5手，再加上800股五粮液，总共值多少")
    assert "5 手（500 股）中国平安 500 × 53.61 = 26805 元" in carried["answer"]
    assert "合计市值约为 107317 元" in carried["answer"]
    english = agent.chat("I own 2 lots of Moutai plus 600 Ping An shares, total value?", session_id="r13r9-hold-en")
    assert "CNY 314066 in total" in english["answer"] and english["verification"]["passed"]


def test_i6_swap_shares(agent):
    swap = agent.chat("一股茅台能换几股五粮液", session_id="r13r9-swap")
    assert "前者约为后者的 14.01 倍" in swap["answer"]


def test_i7_a_factual_more_than_double_is_answered(agent):
    english = agent.chat("Is Moutai's net profit more than double Wuliangye's?", session_id="r13r9-double")
    assert "2.18 times" in english["answer"] and "So yes: it is more than double." in english["answer"]
    assert "conditional assessment" not in english["answer"]
    half = agent.chat("中国平安的市净率是不是五粮液的一半以上", session_id="r13r9-half")
    assert "不是，不到一半。" in half["answer"]


def test_i9_hong_kong_stock_with_a_possessive(agent):
    refused = agent.chat("What's Ping An’s Hong Kong stock trading at?", session_id="r13r9-hk")
    assert refused["route"] == "refuse" and refused["limitations"] == ["out_of_coverage"]
    assert "53.61" not in refused["answer"]
