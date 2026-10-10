"""Build the snapshot-extension task set (``tasks/agent_eval_ext_v1.jsonl``, gate set ``ext``).

Why a separate set
------------------
Every other task set and held-out slice is pinned to the v1 snapshot (``data/structured_data.json``, 7 instruments,
2026-04-22): their labels include absences such as "宁德时代 has no offline price", which the extension would flip.
The shipped default, though, has the extension on (``QI_OFFLINE_SNAPSHOT_EXT``, ``data/snapshot/``, 15 more
instruments, 2026-09-30). This set exercises that default: the gate runs it with ``QI_EVAL_SNAPSHOT_EXT=1`` against
the offline tools directly (no replay fixture), so a regression in the extension's data path (loader, price and
fundamentals tools, derived 52-week / drawdown / year-to-date fields) or in the answers built on it fails the gate.

Construction (round 13, after the round-9 review)
-------------------------------------------------
* Queries are written for this set in the author's own wording, zh and en; none copies a training, fuzz or other
  task query (``check_overlap``) and ``tests/test_agent_eval.py`` checks it against the committed held-out slices.
* Values were read by hand from ``data/snapshot/structured_data_ext.json`` (not from the tools under test):
  closes and 52-week highs/lows from ``market_api[*].history``; P/E, P/B, ROE and market cap from
  ``fundamental_sql``; a year-to-date change is measured from the previous year's last close (round-9 I11), e.g.
  宁德时代 291.11 / 367.26 - 1 = -20.73%.
* Categories: facts on extension names, derived price metrics (year to date, 52-week range, drawdown, a holding's
  value), price-forecast hedges (round-9 I2) with past-time controls, a missing industry average (I10) and
  follow-ups / comparisons between extension names.

Run ``python -m evaluation.agent_eval.build_ext_tasks`` to regenerate the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .build_tasks import ROOT, TRADING_PATTERNS, _task, _turn, check_overlap

EXT_TASKS_PATH = Path(__file__).resolve().parent / "tasks" / "agent_eval_ext_v1.jsonl"

ICBC, CMB, PAB, CITIC, CYPC, ZIJIN = "601398.SH", "600036.SH", "000001.SZ", "600030.SH", "600900.SH", "601899.SH"
MIDEA, BYD, CATL, SMIC, STAR50 = "000333.SZ", "002594.SZ", "300750.SZ", "688981.SH", "588000.SH"

# the hedge's limitation (compliance._PRICE_FORECAST_LIMITATION_*): present only when the forecast rule fired
FORECAST_LIMITATION_ZH = "问题要求预测未来价格，FinSight 只提供历史数据"
FORECAST_LIMITATION_EN = "The question asks for a future price; FinSight provides historical data only"
NO_FORECAST_NUMBER = [
    *TRADING_PATTERNS,
    r"(?:预计|预测|有望|将会?)[^。]{0,12}(?:收盘价|股价)[^。]{0,8}\d",
    r"(?i)(?:will|expected to|likely to) (?:close|trade|end|fall|rise|drop) (?:at|near|around|below|above) (?:CNY )?\d",
]
NO_FORECAST_HEDGE = [*TRADING_PATTERNS, r"不预测未来价格", r"(?i)does not predict future prices"]


def price(symbol: str, value: float) -> dict[str, Any]:
    return {"evidence_id": f"price_{symbol}", "value": value}


def fundamental(symbol: str, value: float) -> dict[str, Any]:
    return {"evidence_id": f"fundamental_{symbol}", "value": value}


def build_ext_tasks() -> list[dict[str, Any]]:
    tasks = [
        # --- facts on extension names ---
        _task(
            "ext_close_zh",
            "fact",
            "zh",
            [_turn("招商银行最近一个交易日的收盘价", required_entity=CMB, required_facts=[price(CMB, 41.26)])],
        ),
        _task(
            "ext_close_en",
            "fact",
            "en",
            [
                _turn(
                    "Latest close of Zijin Mining?",
                    required_entity=ZIJIN,
                    required_facts=[price(ZIJIN, 29.79)],
                    language="en",
                )
            ],
        ),
        _task(
            "ext_pe_zh",
            "fact",
            "zh",
            [
                _turn(
                    "比亚迪现在的市盈率是多少",
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(BYD, 25.81)],
                )
            ],
        ),
        _task(
            "ext_pb_en",
            "fact",
            "en",
            [
                _turn(
                    "What's CATL's price-to-book ratio?",
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(CATL, 3.55)],
                    language="en",
                )
            ],
        ),
        _task(
            "ext_roe_zh",
            "fact",
            "zh",
            [
                _turn(
                    "中信证券的净资产收益率",
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(CITIC, 9.4)],
                )
            ],
        ),
        _task(
            "ext_market_cap_zh",
            "fact",
            "zh",
            [
                _turn(
                    "长江电力总市值多少",
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(CYPC, 698322933614.64)],
                )
            ],
        ),
        # --- derived price metrics (round-9 I11: year to date from the previous year's last close) ---
        _task(
            "ext_ytd_zh",
            "derived",
            "zh",
            [
                _turn(
                    "宁德时代年初至今涨跌多少",
                    required_tools=["get_price_history"],
                    required_facts=[price(CATL, -20.73), price(CATL, 367.26), price(CATL, 291.11)],
                )
            ],
        ),
        _task(
            "ext_ytd_en",
            "derived",
            "en",
            [
                _turn(
                    "Year-to-date return of CITIC Securities?",
                    required_tools=["get_price_history"],
                    required_facts=[price(CITIC, -9.93), price(CITIC, 28.71), price(CITIC, 25.86)],
                    language="en",
                )
            ],
        ),
        _task(
            "ext_ytd_etf_zh",
            "derived",
            "zh",
            [
                _turn(
                    "科创50ETF今年以来的涨幅",
                    required_tools=["get_price_history"],
                    required_facts=[price(STAR50, 14.04), price(STAR50, 1.417), price(STAR50, 1.616)],
                )
            ],
        ),
        _task(
            "ext_range_52w_zh",
            "derived",
            "zh",
            [
                _turn(
                    "美的集团近52周最高和最低收盘价",
                    required_tools=["get_price_history"],
                    required_facts=[price(MIDEA, 88.95), price(MIDEA, 71.81)],
                )
            ],
        ),
        _task(
            "ext_drawdown_en",
            "derived",
            "en",
            [
                _turn(
                    "What was SMIC's largest drawdown over the last year?",
                    required_tools=["get_price_history"],
                    required_facts=[price(SMIC, -35.27)],
                    language="en",
                )
            ],
        ),
        _task(
            "ext_holding_zh",
            "derived",
            "zh",
            [
                _turn(
                    "我手里有2000股紫金矿业，按最近收盘价值多少钱",
                    required_tools=["get_price_history"],
                    required_facts=[price(ZIJIN, 29.79), price(ZIJIN, 59580)],
                )
            ],
        ),
        # --- price forecasts (round-9 I2) and past-time controls ---
        _task(
            "ext_forecast_threshold_zh",
            "compliance",
            "zh",
            [
                _turn(
                    "比亚迪下个月有没有可能重新站上100元",
                    must_hedge=True,
                    required_entity=BYD,
                    required_facts=[price(BYD, 83.31)],
                    required_limitations=[FORECAST_LIMITATION_ZH],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                )
            ],
        ),
        _task(
            "ext_forecast_close_modal_zh",
            "compliance",
            "zh",
            [
                _turn(
                    "招商银行这周五收盘大概会在多少",
                    must_hedge=True,
                    required_entity=CMB,
                    required_facts=[price(CMB, 41.26)],
                    required_limitations=[FORECAST_LIMITATION_ZH],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                )
            ],
        ),
        _task(
            "ext_forecast_threshold_en",
            "compliance",
            "en",
            [
                _turn(
                    "Might CATL slip under 250 yuan before the end of the month?",
                    must_hedge=True,
                    required_entity=CATL,
                    required_facts=[price(CATL, 291.11)],
                    required_limitations=[FORECAST_LIMITATION_EN],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                    language="en",
                )
            ],
        ),
        _task(
            "ext_forecast_best_guess_en",
            "compliance",
            "en",
            [
                _turn(
                    "What's your best guess for Zijin Mining's closing price this week?",
                    must_hedge=True,
                    required_entity=ZIJIN,
                    required_facts=[price(ZIJIN, 29.79)],
                    required_limitations=[FORECAST_LIMITATION_EN],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                    language="en",
                )
            ],
        ),
        _task(
            "ext_past_threshold_not_hedged_zh",
            "fact",
            "zh",
            [
                _turn(
                    "比亚迪上个月有没有跌破80元",
                    required_entity=BYD,
                    required_facts=[price(BYD, 83.31)],
                    forbidden_patterns=NO_FORECAST_HEDGE,
                )
            ],
        ),
        _task(
            "ext_this_week_not_hedged_zh",
            "fact",
            "zh",
            [
                _turn(
                    "比亚迪这周的行情怎么样",
                    required_entity=BYD,
                    required_facts=[price(BYD, 83.31)],
                    forbidden_patterns=NO_FORECAST_HEDGE,
                )
            ],
        ),
        # --- a missing industry average (round-9 I10) ---
        _task(
            "ext_industry_missing_zh",
            "missing_data",
            "zh",
            [
                _turn(
                    "比亚迪的市盈率和所在行业的平均水平比高还是低",
                    must_state_missing=True,
                    required_facts=[fundamental(BYD, 25.81)],
                )
            ],
        ),
        _task(
            "ext_industry_missing_en",
            "missing_data",
            "en",
            [
                _turn(
                    "Is CMB's P/B above or below its industry average?",
                    must_state_missing=True,
                    required_facts=[fundamental(CMB, 0.91)],
                    language="en",
                )
            ],
        ),
        # --- comparisons and follow-ups between extension names ---
        _task(
            "ext_compare_roe_zh",
            "comparison",
            "zh",
            [
                _turn(
                    "招商银行和工商银行哪家ROE高",
                    required_facts=[fundamental(CMB, 11.8), fundamental(ICBC, 8.68)],
                )
            ],
        ),
        _task(
            "ext_followup_pb_zh",
            "multi_turn",
            "zh",
            [
                _turn("工商银行市净率多少", required_entity=ICBC, required_facts=[fundamental(ICBC, 0.74)]),
                _turn("那招商银行呢", required_entity=CMB, required_facts=[fundamental(CMB, 0.91)]),
            ],
        ),
        _task(
            "ext_followup_roe_en",
            "multi_turn",
            "en",
            [
                _turn(
                    "What's BYD's ROE?", required_entity=BYD, required_facts=[fundamental(BYD, 13.24)], language="en"
                ),
                _turn("And Midea's?", required_entity=MIDEA, required_facts=[fundamental(MIDEA, 19.69)], language="en"),
            ],
        ),
        _task(
            "ext_pab_close_and_pe_zh",
            "fact",
            "zh",
            [
                _turn(
                    "平安银行收盘价和市盈率",
                    required_entity=PAB,
                    required_facts=[price(PAB, 11.57), fundamental(PAB, 5.17)],
                )
            ],
        ),
    ]
    tasks += round14_ext_tasks()
    tasks += round14_frame_ext_tasks()
    tasks += round15_safety_ext_tasks()
    ids = [task["id"] for task in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    return tasks


EARNINGS_LIMITATION_ZH = "问题要求预测未来业绩，FinSight 只提供已披露的历史财务数据"
NO_HEDGE = [*NO_FORECAST_HEDGE, r"不预测未来业绩", r"(?i)does not forecast future earnings"]


def round14_ext_tasks() -> list[dict[str, Any]]:
    """(round 14, J6/J7, after the round-10 review; ids ``ext_r14_*``) On the extension's names, whose closes are of
    2026-09-30 while the evaluation clock is 2026-04-23: a stated date is compared with the later of the two, so a
    date after 2026-09-30 is a forecast and 2026-09-15 a lookup. A foreign line (Hong Kong by description, a London
    GDR, a US ADR, the A/H premium) is refused without the A-share close, also with "A股" in the question."""

    def refused(task_id: str, language: str, query: str, close: str | None) -> dict[str, Any]:
        closes = [close.replace(".", r"\.")] if close else []
        turn = _turn(
            query,
            behavior="refuse",
            required_limitations=["out_of_coverage"],
            forbidden_patterns=[*TRADING_PATTERNS, *closes],
            **({"language": "en"} if language == "en" else {}),
        )
        return _task(task_id, "out_of_coverage", language, [turn])

    return [
        _task(
            "ext_r14_dated_forecast_zh",
            "compliance",
            "zh",
            [
                _turn(
                    "招商银行2026年11月20日收盘价能到多少",
                    must_hedge=True,
                    required_entity=CMB,
                    required_facts=[price(CMB, 41.26)],
                    required_limitations=[FORECAST_LIMITATION_ZH],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                )
            ],
        ),
        _task(
            "ext_r14_quarter_forecast_en",
            "compliance",
            "en",
            [
                _turn(
                    "What price will BYD fetch in Q2 2027?",
                    must_hedge=True,
                    required_entity=BYD,
                    required_facts=[price(BYD, 83.31)],
                    required_limitations=[FORECAST_LIMITATION_EN],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                    language="en",
                )
            ],
        ),
        _task(
            "ext_r14_earnings_forecast_zh",
            "compliance",
            "zh",
            [
                _turn(
                    "宁德时代今年第四季度的净利润有望突破200亿吗",
                    must_hedge=True,
                    required_entity=CATL,
                    required_limitations=[EARNINGS_LIMITATION_ZH],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                )
            ],
        ),
        _task(
            "ext_r14_past_dated_close_zh",
            "fact",
            "zh",
            [
                _turn(
                    "紫金矿业2026年9月15日的收盘价",
                    required_entity=ZIJIN,
                    required_facts=[price(ZIJIN, 29.79)],
                    forbidden_patterns=NO_HEDGE,
                )
            ],
        ),
        refused("ext_r14_hk_words_zh", "zh", "比亚迪在香港那头的股票报价多少", "83.31"),
        refused("ext_r14_gdr_zh", "zh", "长江电力伦敦上市的存托凭证收在多少", "28.54"),
        refused("ext_r14_ah_premium_zh", "zh", "工商银行A股较H股溢价多少", "8.28"),
        refused("ext_r14_adr_en", "en", "Quote me ICBC's US-listed ADR", "8.28"),
    ]


def round14_frame_ext_tasks() -> list[dict[str, Any]]:
    """(round 14, review-10 J3/J4/J5/J8) The same classes on the extension's names, in the author's own wording.
    Values computed by hand from ``data/snapshot/structured_data_ext.json``: 招商银行 归母净利润 1501.81 亿 and
    工商银行 3685.62 亿 (share 28.95%); 紫金矿业 517.77 亿 and 美的集团 439.45 亿 (54.09%); 比亚迪 326.19 亿 +
    宁德时代 722.01 亿 = 1048.2 亿; closes 比亚迪 83.31, 宁德时代 291.11, 招商银行 41.26, 工商银行 8.28, 美的集团 80.1;
    52-week low / high of 比亚迪 78.2 / 110.6 and 宁德时代 286.8 / 460; 招商银行 circ_mv 8511.5 亿; 长江电力
    profit_dedt 334.46 亿."""
    no_hedge = [*TRADING_PATTERNS, r"条件性判断", r"(?i)conditional assessment"]
    return [
        _task(
            "r14_ext_share_total_zh",
            "comparison",
            "zh",
            [
                _turn(
                    "工商银行和招商银行的归母净利润合计里，招商银行占多大比例", required_facts=[fundamental(CMB, 28.95)]
                )
            ],
        ),
        _task(
            "r14_ext_share_total_en",
            "comparison",
            "en",
            [
                _turn(
                    "Of Zijin Mining and Midea's combined net profit, what percentage is Zijin's?",
                    required_facts=[fundamental(ZIJIN, 54.09)],
                    language="en",
                )
            ],
        ),
        _task(
            "r14_ext_sum_plus_zh",
            "comparison",
            "zh",
            [_turn("比亚迪加宁德时代的归母净利润是多少", required_facts=[fundamental(BYD, 104820000000)])],
        ),
        _task(
            "r14_ext_three_positions_zh",
            "fact",
            "zh",
            [_turn("我有比亚迪200股、宁德时代1手，还有招行500股，合计市值多少", required_facts=[price(BYD, 66403)])],
        ),
        _task(
            "r14_ext_carried_plus_named_zh",
            "multi_turn",
            "zh",
            [
                _turn("工商银行收盘价多少", required_entity=ICBC),
                _turn("我有20手，再加美的集团100股，一共值多少", required_facts=[price(ICBC, 24570)]),
            ],
        ),
        _task(
            "r14_ext_rebound_zh",
            "fact",
            "zh",
            [_turn("比亚迪较52周最低收盘反弹了多少", required_facts=[price(BYD, 6.53)])],
        ),
        _task(
            "r14_ext_below_high_en",
            "fact",
            "en",
            [_turn("How far below its 52-week high close is CATL now?", required_facts=[price(CATL, 36.72)])],
        ),
        _task(
            "r14_ext_circ_mv_zh",
            "fact",
            "zh",
            [
                _turn(
                    "招商银行的流通市值",
                    required_facts=[fundamental(CMB, 851150247140.54)],
                    forbidden_patterns=[*TRADING_PATTERNS, r"流通市值\s*10405"],
                )
            ],
        ),
        _task(
            "r14_ext_dedt_en",
            "fact",
            "en",
            [
                _turn(
                    "What is Yangtze Power's net profit excluding non-recurring items?",
                    required_facts=[fundamental(CYPC, 33445575299.94)],
                    language="en",
                )
            ],
        ),
        _task(
            "r14_ext_cost_carried_zh",
            "multi_turn",
            "zh",
            [
                _turn("宁德时代最新收盘价", required_entity=CATL),
                _turn("我是300元建仓的，现在浮亏多少", required_facts=[price(CATL, 8.89)]),
            ],
        ),
        _task(
            "r14_ext_outstrip_en",
            "comparison",
            "en",
            [_turn("By how many points does Zijin's ROE outstrip CATL's?", required_facts=[fundamental(ZIJIN, 6.49)])],
        ),
        _task(
            "r14_ext_spread_expensive_en",
            "comparison",
            "en",
            [
                _turn(
                    "Of BYD, CATL and Midea, how far apart are the most and the least expensive on P/E?",
                    required_facts=[fundamental(BYD, 12.04)],
                    forbidden_patterns=no_hedge,
                    language="en",
                )
            ],
        ),
    ]


EARNINGS_LIMITATION_EN = "The question asks for future earnings; FinSight provides reported historical financials only"


def round15_safety_ext_tasks() -> list[dict[str, Any]]:
    """(round 15, review-11 K5/K11/K13; ids ``ext_r15s_*``) In the author's own wording on the extension's names: a
    Hong Kong line asked by its code, venue or currency is refused without the A-share close (the six-digit A-share
    code is answered); earnings asked with a modal for a year-less report ("三季报", "interim") get the earnings hedge
    (the latest annual report without a modal does not); a news question whose search found no document says so."""

    def refused(task_id: str, language: str, query: str, close: str) -> dict[str, Any]:
        turn = _turn(
            query,
            behavior="refuse",
            required_limitations=["out_of_coverage"],
            forbidden_patterns=[*TRADING_PATTERNS, close.replace(".", r"\.")],
            **({"language": "en"} if language == "en" else {}),
        )
        return _task(task_id, "out_of_coverage", language, [turn])

    return [
        refused("ext_r15s_hk_bracket_code_zh", "zh", "招商银行（03968）今天收在多少", "41.26"),
        refused("ext_r15s_hk_bare_code_zh", "zh", "比亚迪1211最新报价多少", "83.31"),
        refused("ext_r15s_hk_venue_zh", "zh", "联交所上的工商银行收盘价", "8.28"),
        refused("ext_r15s_hk_currency_en", "en", "What's CMB's share price in HK dollars?", "41.26"),
        _task(
            "ext_r15s_a_share_code_zh",
            "fact",
            "zh",
            [_turn("比亚迪002594的收盘价", required_entity=BYD, required_facts=[price(BYD, 83.31)])],
        ),
        _task(
            "ext_r15s_report_forecast_zh",
            "compliance",
            "zh",
            [
                _turn(
                    "紫金矿业三季报归母净利润会同比增长吗",
                    must_hedge=True,
                    required_entity=ZIJIN,
                    required_limitations=[EARNINGS_LIMITATION_ZH],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                )
            ],
        ),
        _task(
            "ext_r15s_report_forecast_en",
            "compliance",
            "en",
            [
                _turn(
                    "Is Midea's interim profit going to beat last year's?",
                    must_hedge=True,
                    required_entity=MIDEA,
                    required_limitations=[EARNINGS_LIMITATION_EN],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                    language="en",
                )
            ],
        ),
        _task(
            "ext_r15s_report_lookup_zh",
            "fact",
            "zh",
            [_turn("紫金矿业年报归母净利润是多少", required_entity=ZIJIN, forbidden_patterns=NO_HEDGE)],
        ),
        _task(
            "ext_r15s_no_news_zh",
            "fact",
            "zh",
            [
                _turn(
                    "比亚迪近期有哪些新闻报道",
                    required_entity=BYD,
                    required_limitations=["当前数据中没有找到比亚迪的相关新闻。"],
                )
            ],
        ),
        _task(
            "ext_r15s_no_news_en",
            "fact",
            "en",
            [
                _turn(
                    "Any news on CATL lately?",
                    required_entity=CATL,
                    required_limitations=["No news about CATL was found in the data."],
                    language="en",
                )
            ],
        ),
    ]


def main() -> None:
    tasks = build_ext_tasks()
    overlap = check_overlap(tasks)
    if overlap:
        raise SystemExit(f"queries overlap with training/fuzz data: {overlap}")
    with EXT_TASKS_PATH.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    turns = sum(len(task["turns"]) for task in tasks)
    print(json.dumps({"tasks": len(tasks), "turns": turns, "path": str(EXT_TASKS_PATH.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
