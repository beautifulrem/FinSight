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
    ids = [task["id"] for task in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    return tasks


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
