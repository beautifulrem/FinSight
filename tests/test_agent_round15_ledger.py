"""Round-15 rule K15 (round-11 review), written with the engineer's own wording: headlines cut at a limiter or at a
metric's change word, written in traditional characters with Taiwanese terms (本益比), or with Latin look-alike words in
Greek / Cyrillic letters are not shown as headlines; ordinary headlines with the same words stay shown.
"""

from __future__ import annotations

import pytest

from query_intelligence.text_safety import headline_findings, lookalike_word, safe_headline

# ---- K15: planted-fact headline shapes (the author's own examples) ----


@pytest.mark.parametrize(
    "title",
    [
        "財經快訊 招商銀行本益比僅 4.8",  # traditional, Taiwanese term, limiter + bare figure
        "港媒報導 長江電力股價淨值比只有零點九",  # traditional, Chinese numerals
        "法人建議逢低加碼紫金礦業",  # traditional advice wording
        "盘后速递 招行市盈率现在只有",  # a limiter at the end
        "速评 美的市净率仅为",
        "数据 宁德时代市盈率高达 95",
        "财报速读 长江电力归母净利润同",  # a metric cut at its change word
        "盘中 招商银行营收环",
        "Midea Group's net profit has risen",  # an English metric and verb with no value
        "BYD's market value now stands",
        "CATL P/E ratio has dropp",
        "招商银行 ΡΒ 跌破净资产",  # Greek capitals that read as "PB"
        "工商银行 ЕРЅ 创新高",  # Cyrillic capitals that read as "EPS"
    ],
)
def test_planted_fact_shapes_are_not_headlines(title):
    assert safe_headline(title) is None, title


@pytest.mark.parametrize(
    "title",
    [
        "招商银行毛利率提升",  # a whole change word
        "紫金矿业引进高端人才",  # 才 in 人才
        "宁德时代与车企签约",  # 约 in 签约
        "长江电力：补齐短板 弥补不足",
        "臺灣半導體產業景氣回升",  # traditional, no claim shape
        "美的集团 β 系数解读",  # a single Greek letter
        "BYD shares rise after strong sales",
        "Midea Group's revenue rises on export demand",
    ],
)
def test_ordinary_headlines_stay_shown(title):
    assert safe_headline(title) == title, headline_findings(title)


def test_lookalike_words_need_two_confusable_letters():
    assert lookalike_word("市净率 ΡΒ") == "ΡΒ"
    assert lookalike_word("高 β 股") is None
    assert lookalike_word("Ελλάδα") is None  # Greek with letters that are not Latin look-alikes
