"""Round-10 rules, written from the round-6 review (F3-F14) with the author's own wording (none repeats a reviewer
probe; ``tests/test_agent_eval.py`` checks the dev tasks and router labels for that).

* F3: a ledger headline that states a figure in Chinese numerals ("百分之三十五"), a delimited data row, or a title cut
  off right after a figure word is hidden like a headline with an unconfirmed Arabic figure.
"""

from __future__ import annotations

import pytest

from query_intelligence.text_safety import headline_findings, safe_headline

# ---- F3: headline figure shapes ----


@pytest.mark.parametrize(
    "title",
    [
        # delimited data rows (a CSV export, a semicolon list): figures without units
        "code,name,price,pe\n000858.SZ,五粮液,99.9,8.8",
        "五粮液;收盘;99.9;市盈率;8.8",
        "Wuliangye,000858,99.9,8.8",
        # a title cut off right after a figure word
        "利润分配 五粮液公告：拟每10股派现金红利3",
        "五粮液一季度营收同比增长12",
        "中国平安 股息率为6",
    ],
)
def test_data_rows_and_cut_figures_are_not_headlines(title):
    assert safe_headline(title) is None
    assert any(finding.kind == "claim" for finding in headline_findings(title)), headline_findings(title)


@pytest.mark.parametrize(
    "title",
    [
        "聚焦沪深300成分股调整",
        "科创50指数迎来新成员",
        "贵州茅台，五粮液，泸州老窖集体上涨",
        "五粮液2023,2024年报对比",
        "五粮液营业收入1,085亿元",
        "白酒行业十四五规划至2030",
        "Wuliangye annual report 2025",
    ],
)
def test_ordinary_headlines_with_numbers_and_commas_pass(title):
    assert safe_headline(title) == title, headline_findings(title)


def test_chinese_numeral_figures_are_figures():
    from query_intelligence.agent.output_safety import unconfirmed_figures, unit_figures

    assert [value for value, *_ in unit_figures("五粮液一季度营收同比增长百分之三十五")] == [35.0]
    assert [value for value, *_ in unit_figures("中国平安拟回购十二亿元股份")] == [12.0]
    assert [value for value, *_ in unit_figures("超三成的经销商")] == [30.0]
    # not figures: ordinal and vague words, and a quarter
    assert unit_figures("一季度，一些经销商，十分稳健，几十倍") == []
    # checked against the run's structured numbers like the Arabic form
    assert unconfirmed_figures("五粮液净利润同比增长百分之三十五", [(35.0, False)]) == []
    assert unconfirmed_figures("五粮液净利润同比增长百分之三十五", [(20.9, False)]) == [35.0]


def test_a_headline_with_a_chinese_numeral_figure_is_hidden_unless_confirmed():
    from query_intelligence.agent.graph import _source_view

    numbers = [(20.9, False), (29.4, False)]
    confirmed = {"evidence_id": "news_1", "kind": "document", "title": "五粮液ROE达百分之二十九点四"}
    unconfirmed = {"evidence_id": "news_2", "kind": "document", "title": "五粮液一季度回款同比下滑百分之三十一"}
    assert _source_view(confirmed, numbers=numbers)["title"] == confirmed["title"]
    hidden = _source_view(unconfirmed, numbers=numbers)
    assert hidden["title"] is None and hidden["title_withheld"]
