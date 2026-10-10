"""Build the shipped-default task set (``tasks/agent_eval_shipped_v1.jsonl``, gate set ``shipped``).

Why a separate set
------------------
The product ships with both offline layers on: the snapshot extension (``QI_OFFLINE_SNAPSHOT_EXT``) and, since round
13, the corrections layer (``QI_OFFLINE_SNAPSHOT_FIX``, ``data/snapshot/structured_data_fix.json``: FY2025 annual-report
figures for 五粮液 / 贵州茅台 / 中国平安, the 2026-04-22 bars of 中国平安, 沪深300, 创业板ETF and 证券ETF, and the field
definitions such as weighted vs period-end ROE). Every other gate set pins the corrections layer off (their labels are
the v1 seed values), so a regression in the shipped data path reached master unnoticed (round-10 review J1/J2). This
set runs with ``QI_EVAL_SNAPSHOT_EXT=1 QI_EVAL_SNAPSHOT_FIX=1`` against the offline tools directly (no replay fixture).

Construction (round 14)
-----------------------
* Queries are written for this set in the author's own wording, zh and en; ``check_overlap`` rejects a copy of a
  training or fuzz query and ``tests/test_agent_eval.py`` checks them against the committed held-out slices.
* Expected values were read by hand from ``data/snapshot/structured_data_fix.json`` (not from the tools under test):
  ``fundamental_sql`` for P/E, P/B, ROE (weighted), revenue and net profit; ``market_api[*].close`` for the
  2026-04-22 closes. Mixed questions take the other name's value from ``structured_data_ext.json``.
* Seed guards: where the v1 seed value differs (茅台 P/E 24.6, 五粮液 ROE 29.4, 中国平安 close 53.61, 沪深300
  4005.2) the turn forbids it, so an answer served from the seed fails even if it also states the corrected figure.
* Definitions: an extension company keeps its period-end ROE and must not be labelled "ROE（加权）".

Run ``python -m evaluation.agent_eval.build_shipped_tasks`` to regenerate the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .build_ext_tasks import FORECAST_LIMITATION_EN, FORECAST_LIMITATION_ZH, NO_FORECAST_NUMBER, fundamental, price
from .build_tasks import ROOT, TRADING_PATTERNS, _task, _turn, check_overlap

SHIPPED_TASKS_PATH = Path(__file__).resolve().parent / "tasks" / "agent_eval_shipped_v1.jsonl"

MT, WLY, PA, HS300, CYB, SEC = "600519.SH", "000858.SZ", "601318.SH", "000300.SH", "159915.SZ", "512880.SH"
CMB, CATL = "600036.SH", "300750.SZ"


def not_seed(*patterns: str) -> list[str]:
    """The default forbidden patterns plus v1 seed values the corrections layer replaced."""
    return [*TRADING_PATTERNS, *patterns]


def build_shipped_tasks() -> list[dict[str, Any]]:
    tasks = [
        # --- corrected FY2025 statements (annual reports) ---
        _task(
            "shipped_wly_roe_zh",
            "fact",
            "zh",
            [
                _turn(
                    "五粮液2025年报披露的净资产收益率",
                    required_entity=WLY,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(WLY, 6.89)],
                    forbidden_patterns=not_seed(r"29\.4\s*%"),
                )
            ],
        ),
        _task(
            "shipped_wly_revenue_zh",
            "fact",
            "zh",
            [
                _turn(
                    "五粮液上一个财年营收有多少",
                    required_entity=WLY,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(WLY, 40528509770.23)],
                )
            ],
        ),
        _task(
            "shipped_wly_profit_en",
            "fact",
            "en",
            [
                _turn(
                    "How much did Wuliangye earn for its shareholders in fiscal 2025?",
                    required_entity=WLY,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(WLY, 8954257202.51)],
                    language="en",
                )
            ],
        ),
        _task(
            "shipped_wly_pe_en",
            "fact",
            "en",
            [
                _turn(
                    "Wuliangye's trailing P/E, please",
                    required_entity=WLY,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(WLY, 43.63)],
                    language="en",
                )
            ],
        ),
        _task(
            "shipped_mt_pe_zh",
            "fact",
            "zh",
            [
                _turn(
                    "茅台眼下市盈率几倍",
                    required_entity=MT,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(MT, 21.44)],
                    forbidden_patterns=not_seed(r"24\.6\s*倍"),
                )
            ],
        ),
        _task(
            "shipped_mt_pb_en",
            "fact",
            "en",
            [
                _turn(
                    "Where does Kweichow Moutai's price-to-book stand?",
                    required_entity=MT,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(MT, 7.22)],
                    language="en",
                )
            ],
        ),
        _task(
            "shipped_mt_roe_zh",
            "fact",
            "zh",
            [
                _turn(
                    "贵州茅台股东回报率ROE有多高",
                    required_entity=MT,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(MT, 32.53)],
                )
            ],
        ),
        _task(
            "shipped_mt_revenue_en",
            "fact",
            "en",
            [
                _turn(
                    "Total revenue Kweichow Moutai reported for 2025?",
                    required_entity=MT,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(MT, 168838102514.79)],
                    language="en",
                )
            ],
        ),
        _task(
            "shipped_pa_pe_zh",
            "fact",
            "zh",
            [
                _turn(
                    "按年报口径中国平安市盈率多少倍",
                    required_entity=PA,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(PA, 7.8)],
                )
            ],
        ),
        _task(
            "shipped_pa_roe_en",
            "fact",
            "en",
            [
                _turn(
                    "Ping An Insurance return on equity?",
                    required_entity=PA,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(PA, 14.0)],
                    language="en",
                )
            ],
        ),
        # --- corrected 2026-04-22 bars ---
        _task(
            "shipped_pa_close_zh",
            "fact",
            "zh",
            [
                _turn(
                    "中国平安最近一天收在什么价位",
                    required_entity=PA,
                    required_tools=["get_price_history"],
                    required_facts=[price(PA, 58.03)],
                    forbidden_patterns=not_seed(r"53\.61"),
                )
            ],
        ),
        _task(
            "shipped_hs300_close_zh",
            "fact",
            "zh",
            [
                _turn(
                    "沪深300最新收了多少点",
                    required_entity=HS300,
                    required_tools=["get_price_history"],
                    required_facts=[price(HS300, 4799.63)],
                    forbidden_patterns=not_seed(r"4005\.2"),
                )
            ],
        ),
        _task(
            "shipped_cyb_close_en",
            "fact",
            "en",
            [
                _turn(
                    "Most recent closing price of the ChiNext ETF?",
                    required_entity=CYB,
                    required_tools=["get_price_history"],
                    required_facts=[price(CYB, 3.758)],
                    language="en",
                )
            ],
        ),
        _task(
            "shipped_sec_close_zh",
            "fact",
            "zh",
            [
                _turn(
                    "证券ETF上个交易日的收盘价是多少",
                    required_entity=SEC,
                    required_tools=["get_price_history"],
                    required_facts=[price(SEC, 1.075)],
                )
            ],
        ),
        _task(
            "shipped_pa_holding_zh",
            "derived",
            "zh",
            [
                _turn(
                    "我有500股中国平安，按最新收盘价算值多少钱",
                    required_entity=PA,
                    required_tools=["get_price_history"],
                    required_facts=[price(PA, 58.03), price(PA, 29015)],
                )
            ],
        ),
        # --- comparisons and follow-ups on corrected names, and across layers ---
        _task(
            "shipped_roe_gap_zh",
            "comparison",
            "zh",
            [
                _turn(
                    "茅台的ROE比五粮液高出几个百分点",
                    required_facts=[fundamental(MT, 32.53), fundamental(WLY, 6.89), fundamental(WLY, 25.64)],
                )
            ],
        ),
        _task(
            "shipped_followup_pe_zh",
            "multi_turn",
            "zh",
            [
                _turn("茅台PE现在是多少", required_entity=MT, required_facts=[fundamental(MT, 21.44)]),
                _turn("换成五粮液是多少倍", required_entity=WLY, required_facts=[fundamental(WLY, 43.63)]),
            ],
        ),
        _task(
            "shipped_cross_layer_pb_zh",
            "comparison",
            "zh",
            [
                _turn(
                    "中国平安跟招商银行比，谁的市净率更低",
                    required_facts=[fundamental(PA, 1.05), fundamental(CMB, 0.91)],
                )
            ],
        ),
        # --- definitions: an extension company keeps period-end ROE, never labelled weighted ---
        _task(
            "shipped_catl_roe_basis_zh",
            "fact",
            "zh",
            [
                _turn(
                    "宁德时代净资产收益率多少",
                    required_entity=CATL,
                    required_tools=["get_fundamentals"],
                    required_facts=[fundamental(CATL, 21.42)],
                    forbidden_patterns=not_seed(r"ROE（加权）"),
                )
            ],
        ),
        # --- a price forecast on a corrected bar is hedged and answered with the dated close ---
        _task(
            "shipped_pa_forecast_zh",
            "compliance",
            "zh",
            [
                _turn(
                    "中国平安下周能不能涨到65块",
                    must_hedge=True,
                    required_entity=PA,
                    required_facts=[price(PA, 58.03)],
                    required_limitations=[FORECAST_LIMITATION_ZH],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                )
            ],
        ),
        _task(
            "shipped_hs300_forecast_en",
            "compliance",
            "en",
            [
                _turn(
                    "Will the CSI 300 close above 5000 points by month-end?",
                    must_hedge=True,
                    required_entity=HS300,
                    required_facts=[price(HS300, 4799.63)],
                    required_limitations=[FORECAST_LIMITATION_EN],
                    forbidden_patterns=NO_FORECAST_NUMBER,
                    language="en",
                )
            ],
        ),
    ]
    ids = [task["id"] for task in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    return tasks


def main() -> None:
    tasks = build_shipped_tasks()
    overlap = check_overlap(tasks)
    if overlap:
        raise SystemExit(f"queries overlap with training/fuzz data: {overlap}")
    with SHIPPED_TASKS_PATH.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    turns = sum(len(task["turns"]) for task in tasks)
    print(json.dumps({"tasks": len(tasks), "turns": turns, "path": str(SHIPPED_TASKS_PATH.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
