"""Round 16 (review 12, L4-L6): claim parts without a number, growth-rate comparisons, book value and "增长X%至Y亿元".

* L4: a clause with a direction or comparison word and no number is a claim part ("…，平安银行也实现了增长",
  "…，工商银行则更高", "…and Moutai's increased too"); an elided metric is carried from the previous clause, never read
  as the day's price move; a claim is never "supported" when such a part is false.
* L5: "A的营收增速高于B" compares the growth rates, not the levels.
* L6: direction-only changes ("净利润同比下降"), price against book value (破净, "below book value", "N times book
  value") and an amount after a growth ("增长约15%，至3491亿元") are checked.

The wordings, companies and pairings are the engineer's own. Values from the shipped default data (v1 + the 2026-09-30
extension + the corrections layer), FY2025 report, valuation 2026-09-30 (茅台 / 五粮液 / 中国平安: 2026-04-22):

=========  ========  =========  ========  =========  =====  =====  =====
company    revenue   rev YoY %  NP 亿     NP YoY %   PE     PB     ROE
=========  ========  =========  ========  =========  =====  =====  =====
宁德时代   4237.02   17.04      722.01    42.18      15.85  3.55   21.42
比亚迪     8039.65    3.46      326.19   -18.82      25.81  3.17   13.24
美的集团   4585.02   12.11      439.45    14.87      13.77  2.89   19.69
紫金矿业   3490.79   14.96      517.77    62.01      11.71  3.92   27.91
中芯国际    673.23   16.49       50.41    36.29     133.05  5.59    3.34
招商银行   3375.32    0.01     1501.81     1.05       6.86  0.91   11.80
工商银行   8382.70    2.00     3685.62     1.04       7.89  0.74    8.68
平安银行   1314.42  -10.40      426.33    -4.21       5.17  0.48    7.73
中信证券    748.54   28.80      300.76    37.26      10.19  1.37    9.40
长江电力    862.42    2.07      345.03     6.13      19.29  3.23   15.59
贵州茅台   1688.38   -1.21      823.20    -4.53      21.44  7.22   32.53
五粮液      405.29  -54.55       89.54   -71.89      43.63  3.26    6.89
中国平安  10505.06    2.10     1347.78     6.50       7.80  1.05   14.00
=========  ========  =========  ========  =========  =====  =====  =====
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def checker():
    from evaluation.agent_eval.runner import build_offline_service
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    service = build_offline_service(snapshot_ext=True, snapshot_fix=True)
    registry = build_registry_for_service(service)

    def run(claim: str):
        return check_claim(claim, service=service, registry=registry, zh=not claim.isascii())

    return run


def _rows(report):
    return [
        (check.target, check.kind, check.metric, check.comparator, check.claimed, check.actual, check.reference,
         check.status, check.reason, check.note[:60])
        for check in report.checks
    ] + [("unchecked", item.text) for item in report.unchecked]  # fmt: skip


# ---- L4: a true number next to a false clause without a number ---------------------------------------------------------
# Each claim has one true part with a number and one false part (the value that makes it false is in the comment). None
# may be "supported": the false part is checked (partially_supported), or both are false (contradicted).
TRUE_NUMBER_FALSE_CLAUSE = [
    # an elided metric carried into the next clause, with a direction word
    "CATL's revenue grew 17% and Ping An Bank's grew as well",  # PAB revenue YoY -10.40
    "招商银行净利润同比增长1.05%，平安银行也实现了增长",  # PAB net profit YoY -4.21
    "紫金矿业营收增长约15%，贵州茅台同样有所增长",  # 茅台 revenue YoY -1.21
    "Zijin Mining's net profit jumped 62%, and Moutai's increased too",  # 茅台 NP YoY -4.53
    "长江电力营收同比增长2.07%，五粮液亦有增长",  # 五粮液 revenue YoY -54.55
    "中信证券净利润增长37%，比亚迪的也在增长",  # BYD NP YoY -18.82
    "Midea's revenue was up 12% while BYD's declined",  # BYD revenue YoY +3.46
    "宁德时代净利润增长42%，中芯国际则出现下滑",  # 中芯 NP YoY +36.29
    "工商银行营收增长2%，招商银行的营收反而减少了",  # CMB revenue YoY +0.01
    "China Merchants Bank's net profit rose about 1%, but ICBC's fell",  # ICBC NP YoY +1.04
    # the same company against the previous year, no number
    "宁德时代2025年净利润722亿元，较上年下降",  # NP YoY +42.18
    "平安银行营收1314亿元，比上年有所增长",  # revenue YoY -10.40
    "贵州茅台2025年营收1688亿元，同比增加",  # revenue YoY -1.21
    "BYD's net profit was RMB32.6 billion in 2025, up from the year before",  # NP YoY -18.82
    "五粮液净利润89.5亿元，比去年增加",  # NP YoY -71.89
    # another metric named, direction only
    "中芯国际营收673亿元，净利润同比下降",  # NP YoY +36.29
    "美的集团净利润439亿元，营收同比减少",  # revenue YoY +12.11
    "Ping An Bank's ROE was 7.7% and its revenue increased year on year",  # revenue YoY -10.40
    "长江电力2025年净利润345亿元，营业收入同比下滑",  # revenue YoY +2.07
    "五粮液市盈率约43.6倍，净利润同比增长",  # NP YoY -71.89
    # an elided comparison: "则更高", "lower still"
    "招商银行的市净率比平安银行高，工商银行的还要更高",  # ICBC PB 0.74 < CMB 0.91
    "宁德时代净利润高于比亚迪，紫金矿业更高",  # 紫金 517.77 < 宁德 722.01
    "比亚迪市盈率25.8倍，中芯国际更低",  # 中芯 PE 133.05
    "Midea's P/E is lower than BYD's, and CATL's is lower still",  # CATL 15.85 > Midea 13.77
    "工商银行ROE为8.68%，中国平安的更低",  # 平安 ROE 14.00
    # an explicit comparison in a clause of its own
    "紫金矿业净利润517.8亿元，高于宁德时代",  # 722.01
    "中信证券市盈率10.2倍，比招商银行还低",  # CMB PE 6.86
    "Ping An Bank's P/B is 0.48x, above ICBC's",  # ICBC PB 0.74
    # a growth-rate comparison next to a level
    "比亚迪营收8040亿元，营收增速高于宁德时代",  # 3.46 < 17.04
    "美的集团净利润增长14.9%，增速比紫金矿业快",  # 14.87 < 62.01
    "CATL's revenue grew 17%, slower than BYD's revenue growth",  # 17.04 > 3.46
    # book value
    "中国平安市盈率7.8倍，目前已经破净",  # PB 1.05
    "CATL trades at 15.9 times earnings and below book value",  # PB 3.55
    "招商银行市净率0.91倍，没有破净",  # PB 0.91 < 1
    # an amount after a growth
    "中芯国际净利润同比增长约36%，至60亿元",  # NP 50.41亿
    "Zijin Mining's revenue rose 15% to RMB400 billion",  # revenue 3490.79亿
]


def test_the_false_clause_set_has_at_least_thirty_claims():
    assert len(TRUE_NUMBER_FALSE_CLAUSE) >= 30
    assert len(set(TRUE_NUMBER_FALSE_CLAUSE)) == len(TRUE_NUMBER_FALSE_CLAUSE)


@pytest.mark.parametrize("claim", TRUE_NUMBER_FALSE_CLAUSE)
def test_a_true_number_next_to_a_false_clause_is_never_supported(checker, claim):
    report = checker(claim)
    assert report.verdict in {"partially_supported", "contradicted"}, _rows(report)
    assert any(check.status == "contradicted" for check in report.checks), _rows(report)


# The same forms with a true clause stay supported (the clause is checked, not just flagged).
@pytest.mark.parametrize(
    "claim",
    [
        "比亚迪营收增长3.5%，宁德时代也有所增长",  # +17.04
        "CATL's net profit rose 42% and Zijin's rose too",  # +62.01
        "宁德时代2025年净利润722亿元，比上年增长",  # +42.18
        "平安银行营收1314亿元，较上年下降",  # -10.40
        "工商银行市净率低于招商银行，平安银行更低",  # 0.74 < 0.91; 0.48 < 0.74
        "中国平安ROE为14%，工商银行的更低",  # 8.68 < 14
        "中芯国际营收673亿元，净利润同比增长",  # +36.29
        "Midea's revenue was up 12% and BYD's grew too",  # +3.46
        "紫金矿业营收同比增长约15%，至3491亿元",  # 14.96; 3490.79
        "招商银行市盈率6.9倍，已经破净",  # PB 0.91
    ],
)
def test_the_same_forms_with_a_true_clause_are_supported(checker, claim):
    report = checker(claim)
    assert report.verdict == "supported", _rows(report)
    assert len(report.checks) >= 2, _rows(report)


def test_an_elided_metric_is_the_previous_clauses_not_the_days_price_move(checker):
    report = checker("Zijin Mining's net profit jumped 62%, and Moutai's increased too")
    metrics = [(check.target, check.metric) for check in report.checks]
    assert ("贵州茅台", "netprofit_yoy") in metrics, _rows(report)
    assert all(check.metric != "pct_change_1d" for check in report.checks), _rows(report)


def test_an_elided_comparison_compares_with_the_previous_subject(checker):
    report = checker("宁德时代净利润高于比亚迪，紫金矿业更高")
    rows = [(check.target, check.reference, check.metric, check.comparator, check.status) for check in report.checks]
    assert ("宁德时代", "比亚迪", "net_profit", "gt", "supported") in rows, _rows(report)
    assert ("紫金矿业", "宁德时代", "net_profit", "gt", "contradicted") in rows, _rows(report)


def test_a_direction_part_against_the_previous_year_is_the_yoy_of_the_named_figure(checker):
    report = checker("宁德时代2025年净利润722亿元，较上年下降")
    (_level, change) = report.checks
    assert (change.target, change.metric, change.comparator, change.status) == (
        "宁德时代", "netprofit_yoy", "lt", "contradicted"
    ), _rows(report)  # fmt: skip


# ---- L5: growth-rate comparisons --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "metric", "verdict"),
    [
        ("比亚迪的营收增速低于美的集团", "revenue_yoy", "supported"),  # 3.46 < 12.11 (levels: 8039.65 > 4585.02)
        ("紫金矿业的净利润增速比宁德时代快", "netprofit_yoy", "supported"),  # 62.01 > 42.18
        ("中信证券净利润的增长速度不如长江电力", "netprofit_yoy", "contradicted"),  # 37.26 > 6.13
        ("工商银行营收增幅高于招商银行", "revenue_yoy", "supported"),  # 2.00 > 0.01 (levels would agree too)
        ("平安银行的营收增长率比贵州茅台高", "revenue_yoy", "contradicted"),  # -10.40 < -1.21
        ("美的集团的营收增速比比亚迪慢", "revenue_yoy", "contradicted"),  # 12.11 > 3.46
        ("Midea's revenue grew faster than BYD's", "revenue_yoy", "supported"),
        ("CMB's net profit growth was higher than CITIC Securities'", "netprofit_yoy", "contradicted"),  # 1.05 < 37.26
        ("BYD's revenue growth lagged CATL's", "revenue_yoy", "supported"),  # 3.46 < 17.04
    ],
)
def test_a_growth_rate_comparison_compares_growth_rates(checker, claim, metric, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    assert [check.metric for check in report.checks] == [metric], _rows(report)


def test_a_level_comparison_stays_a_level_comparison(checker):
    report = checker("比亚迪的营收高于美的集团")  # 8039.65 > 4585.02
    assert report.verdict == "supported" and [check.metric for check in report.checks] == ["revenue"], _rows(report)


# ---- L6: direction only, book value, amount after a growth ------------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "metric", "comparator", "verdict"),
    [
        ("平安银行2025年净利润同比下降", "netprofit_yoy", "lt", "supported"),  # -4.21
        ("长江电力净利润同比减少", "netprofit_yoy", "lt", "contradicted"),  # +6.13
        ("贵州茅台营收同比下滑", "revenue_yoy", "lt", "supported"),  # -1.21
        ("紫金矿业2025年营业收入较上年增长", "revenue_yoy", "gt", "supported"),  # +14.96
        ("五粮液的净利润比上年增加了", "netprofit_yoy", "gt", "contradicted"),  # -71.89
        ("BYD's net profit declined year on year", "netprofit_yoy", "lt", "supported"),  # -18.82
        ("CITIC Securities' revenue fell year on year in 2025", "revenue_yoy", "lt", "contradicted"),  # +28.80
        ("中芯国际净利润没有同比下降", "netprofit_yoy", "ge", "supported"),  # +36.29
    ],
)
def test_a_direction_only_change_is_checked_against_its_growth(checker, claim, metric, comparator, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    (check,) = report.checks
    assert (check.metric, check.comparator) == (metric, comparator), _rows(report)


@pytest.mark.parametrize(
    ("claim", "comparator", "verdict"),
    [
        ("平安银行处于破净状态", "lt", "supported"),  # PB 0.48
        ("中信证券目前已经破净", "lt", "contradicted"),  # 1.37
        ("招商银行的股价跌破了每股净资产", "lt", "supported"),  # 0.91
        ("工商银行没有破净", "ge", "contradicted"),  # 0.74
        ("ICBC shares trade below book value", "lt", "supported"),  # 0.74
        ("Midea trades below its book value", "lt", "contradicted"),  # 2.89
        ("Ping An Insurance is trading above book value", "gt", "supported"),  # 1.05
    ],
)
def test_price_against_book_value_is_a_pb_bound_of_one(checker, claim, comparator, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    (check,) = report.checks
    assert (check.metric, check.comparator, check.claimed) == ("pb", comparator, 1.0), _rows(report)
    assert "book" in check.note, _rows(report)


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("Moutai trades at about 7.2 times book value", "supported"),  # 7.22
        ("Zijin Mining is valued at roughly 2 times book", "contradicted"),  # 3.92
        ("China Merchants Bank trades at 0.9x book value", "supported"),  # 0.91
    ],
)
def test_times_book_value_is_the_pb(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    assert [check.metric for check in report.checks] == ["pb"], _rows(report)


@pytest.mark.parametrize(
    ("claim", "level", "verdict"),
    [
        ("紫金矿业营收同比增长约15%，至3491亿元", "revenue", "supported"),
        ("宁德时代2025年净利润增长42%至722亿元", "net_profit", "supported"),
        ("比亚迪净利润同比下降约19%，降至326亿元", "net_profit", "supported"),
        ("美的集团营收增长12%，达到4585亿元", "revenue", "supported"),
        ("中信证券净利润同比增长37%，至350亿元", "net_profit", "partially_supported"),  # 300.76
        ("CATL's net profit rose 42% to RMB72.2 billion", "net_profit", "supported"),
        ("Midea's revenue increased 12% to RMB500 billion", "revenue", "partially_supported"),  # 4585.02亿
    ],
)
def test_an_amount_after_a_growth_is_the_level_it_reached(checker, claim, level, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    assert level in [check.metric for check in report.checks], _rows(report)
    assert all(check.reason != "no_metric" for check in report.checks), _rows(report)
