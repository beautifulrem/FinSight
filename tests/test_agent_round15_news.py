"""Round-15 rule K13 (round-11 review), written with the engineer's own wording: a news or announcement question
whose document search found nothing says so before anything else (template and LLM paths); a news line quotes the
headline, attributed as an unverified third-party headline, only when it passes every evidence-list check, states no
figure, names a target of the question and is in the answer's language.
"""

from __future__ import annotations

from datetime import date

import pytest

from query_intelligence.agent.composer import _documents, news_gaps, quotable_headline

# ---- K13: no document found ----


def _search(tool: str, documents: list[dict], targets: list[str], ok: bool = True) -> dict:
    return {"tool": tool, "ok": ok, "data": {"documents": documents, "targets": targets}}


def test_an_empty_news_search_is_stated_by_kind_and_target():
    log = [_search("search_news", [], ["宁德时代"]), _search("search_announcements", [], ["宁德时代"])]
    assert news_gaps("宁德时代近期有哪些消息", log, zh=True) == [
        "当前数据中没有找到宁德时代的相关新闻。",
        "当前数据中没有找到宁德时代的相关公告。",
    ]
    assert news_gaps("Any headlines on CATL lately?", log[:1], zh=False) == [
        "No news about 宁德时代 was found in the data."  # names are made English by the composer afterwards
    ]


def test_no_gap_without_a_news_question_with_documents_or_after_a_failed_search():
    empty = [_search("search_news", [], ["招商银行"])]
    assert news_gaps("招商银行眼下市盈率几倍", empty, zh=True) == []
    found = [_search("search_news", [{"evidence_id": "n1", "title": "x"}], ["招商银行"])]
    assert news_gaps("招商银行最近有什么新闻", found, zh=True) == []
    failed = [_search("search_news", [], ["招商银行"], ok=False)]
    assert news_gaps("招商银行最近有什么新闻", failed, zh=True) == []


# ---- K13: which headlines are quoted ----


def _doc(title: str, *, kind: str = "news", source: str = "证券时报", eid: str = "news_9") -> dict:
    when = "2026-09-28"
    return {"evidence_id": eid, "source_type": kind, "source_name": source, "publish_time": when, "title": title}


@pytest.mark.parametrize(
    "title",
    [
        "长江电力完成年度检修计划",
        "招商银行召开年度股东大会",
        "紫金矿业海外铜矿项目投产",
    ],
)
def test_a_safe_figure_free_headline_naming_the_target_is_quoted(title):
    targets = [title[:4]]
    assert quotable_headline(_doc(title), targets, zh=True) == title
    (line,) = _documents({"targets": targets, "documents": [_doc(title)]}, zh=True)
    assert f"标题为「{title}」（第三方标题，未经核实）" in line and line.endswith("[news_9]。")


@pytest.mark.parametrize(
    ("title", "targets", "kind", "zh"),
    [
        ("招商银行净利润增长百分之八", ["招商银行"], "news", True),  # a figure in Chinese numerals
        ("招商银行三季度营收1200亿元", ["招商银行"], "news", True),  # a figure
        ("招商银行被证监会立案调查", ["招商银行"], "news", True),  # regulatory action, not an official source
        ("招商银行目标价上调，建议买入", ["招商银行"], "news", True),  # advice wording
        ("白酒板块午后走弱", ["五粮液"], "news", True),  # does not name the target
        ("五粮液发布投资者关系活动记录", ["五粮液"], "research_note", True),  # not news or an announcement
        ("五粮液召开经销商大会", ["五粮液"], "news", False),  # a Chinese headline in an English answer
        ("Wuliangye holds dealer conference", ["Wuliangye"], "news", True),  # an English headline in a Chinese one
        ("券商结论：长江电力进入配置窗口", ["长江电力"], "news", True),  # a colon lead-in
        ("分析人士看好美的集团海外渠道扩张", ["美的集团"], "news", True),  # a judgment word
        ("紫金矿业海外项目有望提前投产", ["紫金矿业"], "news", True),  # a forecast word
        ("机构普遍给出宁德时代新的", ["宁德时代"], "news", True),  # cut on a dangling word
        ("比亚迪一季度海外销量同比增长三", ["比亚迪"], "news", True),  # cut on a Chinese numeral
        ("招商银行快讯", ["招商银行"], "news", True),  # a label, not a headline
        ("Midea plans new plant in", ["Midea"], "news", False),  # cut on a preposition
    ],
)
def test_headlines_that_are_not_quoted(title, targets, kind, zh):
    assert quotable_headline(_doc(title, kind=kind), targets, zh=zh) is None


def test_an_official_announcement_with_regulatory_wording_is_quoted_and_a_rumour_is_not():
    filing = "关于招商银行收到监管函的公告"
    official = _doc(filing, kind="announcement", source="巨潮资讯")
    assert quotable_headline(official, ["招商银行"], zh=True) == filing
    assert quotable_headline(_doc("网传招商银行高管离任"), ["招商银行"], zh=True) is None


# ---- end to end ----


@pytest.fixture(scope="module")
def ext_runtime():
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.tools.defaults import build_registry_for_service
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
    )
    registry = build_registry_for_service(service)
    runtime = AgentRuntime(service, registry, ScriptedLLM([]), today=lambda: date(2026, 10, 11))
    yield service, registry, runtime
    runtime.close()


def test_a_news_question_without_documents_says_so_before_the_close(ext_runtime):
    from query_intelligence.agent.service import AgentService

    agent = AgentService(ext_runtime[2], trace_sinks=[])
    result = agent.chat("比亚迪近期有哪些新闻报道", session_id="r15-news-none")
    answer = result["answer"]
    assert answer.startswith("当前数据中没有找到比亚迪的相关新闻。根据本次检索到的证据："), answer
    assert "当前数据中没有找到比亚迪的相关新闻。" in result["limitations"]
    result = agent.chat("美的集团近期有哪些新闻", session_id="r15-news-only")
    assert "当前数据中没有找到美的集团的相关新闻。" in result["answer"]


def test_the_llm_path_carries_the_no_news_limitation(ext_runtime):
    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM, final_turn
    from query_intelligence.agent.service import AgentService

    service, registry, _ = ext_runtime
    draft = {"answer": "美的集团最新可用收盘价为 80.1 元 [price_000333.SZ]。", "evidence_used": ["price_000333.SZ"]}
    runtime = AgentRuntime(service, registry, ScriptedLLM([final_turn(draft)]), today=lambda: date(2026, 10, 11))
    try:
        result = AgentService(runtime, trace_sinks=[]).chat("美的集团近期有哪些新闻", session_id="r15-news-llm")
    finally:
        runtime.close()
    assert "当前数据中没有找到美的集团的相关新闻。" in result["limitations"]
