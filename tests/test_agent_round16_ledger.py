"""Round-16 rule L18 (round-12 review), written with the engineer's own wording: a planted headline that writes its
figure in English words ("only six point one times"), with spaced-out digits ("3 . 8"), with soft hyphens inside the
digits, cut right after a decimal point or a limiter, or as a year and a metric with no value, is not shown in the
evidence ledger; "fell by half" is the figure-free "halved". Ordinary headlines with number words or years stay
shown.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.output_safety import unconfirmed_figures, unit_figures
from query_intelligence.text_safety import joined_digits, safe_headline, spelled_numbers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("thirty-five percent", "35 percent"),
        ("two point five times", "2.5 times"),
        ("one hundred and twenty billion yuan", "120 billion yuan"),
        ("three thousand points", "3000 points"),
        ("seven point", "7."),
        ("ninety nine", "99"),
    ],
)
def test_english_numbers_in_words_read_as_digits(text, expected):
    assert spelled_numbers(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("市净率 3 . 8 倍", "市净率 3.8 倍"), ("净利润 4 5 6 亿元", "净利润 456 亿元"), ("Top 20 2026", "Top 20 2026")],
)
def test_spaced_digits_are_joined(text, expected):
    assert joined_digits(text) == expected


def test_a_figure_in_words_is_a_figure_the_ledger_checks():
    title = "Wuliangye now trades at only six point one times earnings"
    assert [value for value, *_ in unit_figures(title, spelled=True)] == [6.1]
    assert unit_figures(title) == []  # the answer-side default is unchanged
    assert unconfirmed_figures(title, [(18.4, False)]) == [6.1]
    assert unconfirmed_figures(title, [(6.1, False)]) == []


@pytest.mark.parametrize(
    "title",
    [
        "Midea's P/B is now only one point",  # an English limiter cut at a figure in words
        "CATL's P/E is just 3.",  # cut after the decimal point
        "Analyst note: CMB P/E of 4.4 times",  # an English multiple with its value
        "五粮液市净率 3 . 8",  # spaced digits after a metric, cut
        "招商银行市盈率仅\u00ad6\u00ad.",  # soft hyphens inside the digits, cut after the point
        "宁德时代2024年度营业收入",  # a year and a metric with no value
        "比亚迪年报归母净利润",
        "BYD's quarterly profit fell by half",  # "halved" in words
        "Ping An's earnings cut in half, says blogger",
    ],
)
def test_planted_figure_shapes_are_not_shown(title):
    assert safe_headline(title) is None


@pytest.mark.parametrize(
    "title",
    [
        "One year after the merger, Ping An expands its health services",
        "Top 10 brokers by revenue in 2026",
        "中国平安发布2025年年度报告",
        "贵州茅台召开2025年度股东大会",
        "Two new energy funds launch in Shenzhen",
        "招商银行披露半年度报告",
    ],
)
def test_ordinary_headlines_with_number_words_or_years_stay_shown(title):
    assert safe_headline(title) == title
    assert unconfirmed_figures(title, []) == []
