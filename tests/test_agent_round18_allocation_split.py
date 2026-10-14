"""Round-18 rule N8 (round-14 review): splitting money between several stocks is allocation advice.

Round 17 (M11) hedged position-size questions about one stock. A split across several ("茅台、招行、长江电力各放多少
比例", "茅台招行长江电力怎么分配比例", "How should I split my money between BYD and CATL?") still got three fact sheets
with no risk flag and no hedge. Now ``query_terms.POSITION_SIZING`` also reads the split wordings, so the NLU flags
them, the router reads them as judgment and the answer gets the conditional prefix and the "FinSight 不提供仓位或买入
数量建议" hedge with no allocation figure.

Property test: zh (2 or 3 names x separators x split verbs x 比例 / 多少 / 几成 / 百分之几 asks) and en (2 or 3 names x
split wordings); factual controls that also name several stocks with 各 / 分配 / split (each one's P/E, index weights,
a payout ratio, revenue split by segment) are not flagged.
"""

from __future__ import annotations

import itertools
import re

import pytest

from query_intelligence.query_terms import asks_position_size

_ZH_GROUPS = (("茅台", "招行"), ("贵州茅台", "招商银行", "长江电力"), ("比亚迪", "宁德时代", "紫金矿业"))
_ZH_SEPARATORS = ("、", "和", "")
_ZH_SPLITS = (
    "{names}各放多少比例",
    "帮我配一下：{names}各放多少比例",
    "{names}各配多少",
    "{names}各买多少比较好",
    "{names}各投几成",
    "{names}各占百分之几合适",
    "{names}各占多少仓位",
    "{names}怎么分配比例",
    "{names}如何分配仓位",
    "{names}的仓位怎么分",
    "10万块在{names}之间怎么分",
    "{names}按什么比例配置",
    "{names}配置比例多少合适",
    "{names}各自配多少钱",
)
_EN_GROUPS = (("BYD", "CATL"), ("Kweichow Moutai", "China Merchants Bank", "China Yangtze Power"))
_EN_SPLITS = (
    "How should I split my money between {names}?",
    "How should I split 100k between {names}?",
    "How do I divide my portfolio between {names}?",
    "How much should go into each of {names}?",
    "What weights should I give {names}?",
    "What allocation between {names} makes sense?",
    "What mix of {names} should I hold?",
    "How would you allocate between {names}?",
)
_ZH_CONTROLS = (
    "{names}各自的市盈率是多少",
    "{names}各占沪深300多少比例",
    "{names}的营收各多少",
    "{names}的分红比例各是多少",
    "{names}的利润怎么分配",
    "{names}的利润分配比例",
    "{names}最新收盘价各是多少",
    "我持有{names}各1000股，现在各值多少钱",
)
_EN_CONTROLS = (
    "What are the P/E ratios of {names}?",
    "How is revenue split between segments at {names}?",
    "What weight do {names} have in the CSI 300?",
    "What is the dividend payout ratio of {names}?",
)


def _zh_names(group: tuple[str, ...], separator: str) -> str:
    return separator.join(group)


def _en_names(group: tuple[str, ...]) -> str:
    return " and ".join(group) if len(group) == 2 else ", ".join(group[:-1]) + " and " + group[-1]


def _zh_cases(frames: tuple[str, ...]) -> list[str]:
    return [
        frame.format(names=_zh_names(group, separator))
        for frame, group, separator in itertools.product(frames, _ZH_GROUPS, _ZH_SEPARATORS)
    ]


def _en_cases(frames: tuple[str, ...]) -> list[str]:
    return [frame.format(names=_en_names(group)) for frame, group in itertools.product(frames, _EN_GROUPS)]


def test_split_wordings_across_several_stocks_are_allocation_advice():
    from query_intelligence.agent.router import asks_prediction

    cases = _zh_cases(_ZH_SPLITS) + _en_cases(_EN_SPLITS)
    assert len(cases) == 14 * 9 + 8 * 2
    missed = [case for case in cases if not (asks_position_size(case) and asks_prediction(case))]
    assert not missed, missed


def test_factual_questions_about_several_stocks_are_not_allocation_requests():
    cases = _zh_cases(_ZH_CONTROLS) + _en_cases(_EN_CONTROLS)
    flagged = [case for case in cases if asks_position_size(case)]
    assert not flagged, flagged


@pytest.fixture(scope="module")
def agent(shipped_offline_service):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    runtime = AgentRuntime(shipped_offline_service, build_registry_for_service(shipped_offline_service), ScriptedLLM([]))
    yield AgentService(runtime, trace_sinks=[])
    runtime.close()


# an allocation figure: a percentage, 几成 / 三成, or a share of the money attached to a name
_ALLOCATION_FIGURE = re.compile(
    r"\d+(?:\.\d+)?\s*[%％]\s*(?:的)?(?:仓位|资金|比例|配置)|[一二两三四五六七八九十]\s*成(?:仓位|资金)?|"
    r"\b\d+(?:\.\d+)?\s*%\s+(?:in|into|to|for)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    ("query", "zh"),
    [
        ("茅台、招行、长江电力各放多少比例", True),
        ("帮我配一下：茅台、招行、长江电力各放多少比例", True),
        ("茅台招行长江电力怎么分配比例", True),
        ("How should I split my money between BYD and CATL?", False),
    ],
)
def test_split_requests_are_flagged_hedged_and_carry_no_allocation_figure(agent, query, zh):
    result = agent.chat(query)
    assert "investment_advice_like" in result["nlu_summary"]["risk_flags"]
    hedge = "FinSight 不提供仓位或买入数量建议" if zh else "FinSight does not give position-size or allocation advice"
    assert hedge in result["answer"]
    assert "position_size_hedge" in result["compliance_notes"]
    assert not _ALLOCATION_FIGURE.search(result["answer"]), result["answer"]
