"""Round 12 (round-7 review §4.3 / to_100, round-8 review H10 and H15): rules tested on the author's own examples."""

from __future__ import annotations

import pytest

from query_intelligence.agent.output_safety import _clause_spans, scrub_answer
from tests.test_agent_round10 import _news_only_store


def _corroborate():
    return [(1.085e11, False), (3.78e10, False)]


# ---- H15: a conjunction right after a figure joins two clauses ----


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            "五粮液营业收入1085亿元以及二季度提价7.5% [news_7]。",
            "五粮液营业收入1085亿元以及二季度提价7.5%（据一篇文档，未经其他来源证实） [news_7]。",
        ),
        (
            "五粮液归母净利润378亿元，同时二季度提价7.5% [news_7]。",
            "五粮液归母净利润378亿元，同时二季度提价7.5%（据一篇文档，未经其他来源证实） [news_7]。",
        ),
        (
            "五粮液营业收入1085亿元而二季度提价7.5% [news_7]。",
            "五粮液营业收入1085亿元而二季度提价7.5%（据一篇文档，未经其他来源证实） [news_7]。",
        ),
    ],
)
def test_a_corroborated_figure_joined_by_a_chinese_conjunction_is_not_marked(answer, expected):
    guarded, notes = scrub_answer(
        {"answer": answer, "key_points": []}, _news_only_store(), zh=True, corroborate=_corroborate
    )
    assert notes == ["attributed_document_claim"] and guarded["answer"] == expected


def test_a_corroborated_figure_joined_by_while_or_and_is_not_marked():
    marker = " (according to one document; not confirmed by other sources)"
    answer = "Wuliangye lifted prices 7.5% whereas its net profit was 37.8 billion yuan [news_7]."
    guarded, _ = scrub_answer(
        {"answer": answer, "key_points": []}, _news_only_store(), zh=False, corroborate=_corroborate
    )
    assert (
        guarded["answer"]
        == f"Wuliangye lifted prices 7.5%{marker} whereas its net profit was 37.8 billion yuan [news_7]."
    )
    answer = "Net profit came in at 37.8 billion yuan and prices went up 7.5% [news_7]."
    guarded, _ = scrub_answer(
        {"answer": answer, "key_points": []}, _news_only_store(), zh=False, corroborate=_corroborate
    )
    assert guarded["answer"] == f"Net profit came in at 37.8 billion yuan and prices went up 7.5%{marker} [news_7]."


def test_conjunctions_split_only_after_a_figure():
    # "合并" is a word, and "revenue and net profit" lists two metrics: neither is a clause break
    assert len(_clause_spans("五粮液吸收合并二季度提价7.5%")) == 1
    assert len(_clause_spans("revenue and net profit grew 7.5%")) == 1
    assert len(_clause_spans("营业收入1085亿元及二季度提价7.5%")) == 2
    assert len(_clause_spans("prices rose 7.5% while revenue reached 108.5 billion")) == 2
