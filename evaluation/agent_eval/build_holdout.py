"""Held-out agent evaluation set (``tasks/agent_eval_holdout_v1.jsonl``).

Written after the development set had been used to fix the agent, with new phrasings for every
category. It is run once per reported configuration and is never used to tune rules; the report
shows it separately from the development set.

Run ``python -m evaluation.agent_eval.build_holdout`` to regenerate the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .build_tasks import FUNDAMENTALS, MACRO, MARKET, _price_fact, _task, _turn, check_overlap

HOLDOUT_PATH = Path(__file__).resolve().parent / "tasks" / "agent_eval_holdout_v1.jsonl"


def build_holdout() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    price_phrasings = [
        ("茅台昨天收盘价格是多少啊", "600519.SH"),
        ("五粮液股价现在大概多少", "000858.SZ"),
        ("平安保险最近的股价", "601318.SH"),
        ("300ETF现在多少钱一份", "510300.SH"),
        ("创业板ETF的最新行情", "159915.SZ"),
        ("证券ETF收盘多少", "512880.SH"),
        ("沪深300指数最近收在多少点", "000300.SH"),
        ("Moutai share price please", "600519.SH"),
        ("Latest quote for 000858.SZ?", "000858.SZ"),
        ("Where did 601318.SH close?", "601318.SH"),
    ]
    for index, (query, symbol) in enumerate(price_phrasings):
        language = "en" if query.isascii() else "zh"
        tasks.append(
            _task(
                f"ho_fact_price_{index}",
                "fact",
                language,
                [_turn(query, required_tools=["get_price_history"], required_facts=[_price_fact(symbol)])],
            )
        )

    valuation_phrasings = [
        ("茅台估值贵不贵，PE多少", "600519.SH", "pe_ttm"),
        ("五粮液的PB是多少倍", "000858.SZ", "pb"),
        ("中国平安净资产收益率怎么样", "601318.SH", "roe"),
        ("茅台的盈利能力用ROE衡量是多少", "600519.SH", "roe"),
        ("Is Ping An cheap on a PE basis?", "601318.SH", "pe_ttm"),
        ("What's Wuliangye's return on equity?", "000858.SZ", "roe"),
    ]
    for index, (query, symbol, metric) in enumerate(valuation_phrasings):
        language = "en" if query.isascii() else "zh"
        fact = {"evidence_id": f"fundamental_{symbol}", "value": FUNDAMENTALS[symbol][metric]}
        tasks.append(
            _task(
                f"ho_fact_valuation_{index}",
                "fact",
                language,
                [_turn(query, required_tools=["get_fundamentals"], required_facts=[fact])],
            )
        )

    macro_phrasings = [
        ("现在的CPI同比是多少", "CPI"),
        ("制造业PMI最新读数", "PMI"),
        ("广义货币M2增速是多少", "M2"),
        ("十年期国债收益率现在多少", "十年期国债收益率"),
        ("What's the current China 10-year bond yield?", "十年期国债收益率"),
    ]
    for index, (query, key) in enumerate(macro_phrasings):
        language = "en" if query.isascii() else "zh"
        evidence_id, value, _english = MACRO[key]
        tasks.append(
            _task(
                f"ho_fact_macro_{index}",
                "fact",
                language,
                [
                    _turn(
                        query,
                        required_tools=["get_macro_indicators"],
                        required_facts=[{"evidence_id": evidence_id, "value": value}],
                    )
                ],
            )
        )

    compare_phrasings = [
        ("茅台和五粮液谁更便宜", ["600519.SH", "000858.SZ"], "pe_ttm"),
        ("平安和茅台的ROE差多少", ["601318.SH", "600519.SH"], "roe"),
        ("Which trades at a lower PB, Wuliangye or Ping An?", ["000858.SZ", "601318.SH"], "pb"),
    ]
    for index, (query, symbols, metric) in enumerate(compare_phrasings):
        language = "en" if query.isascii() else "zh"
        facts = [{"evidence_id": f"fundamental_{symbol}", "value": FUNDAMENTALS[symbol][metric]} for symbol in symbols]
        tasks.append(
            _task(
                f"ho_compare_{index}",
                "compare",
                language,
                [_turn(query, required_tools=["get_fundamentals"], required_facts=facts)],
            )
        )
    for index, (query, symbols) in enumerate(
        [
            ("300ETF和创业板ETF今天谁表现更好", ["510300.SH", "159915.SZ"]),
            ("证券ETF与沪深300ETF价格对比", ["512880.SH", "510300.SH"]),
        ]
    ):
        tasks.append(
            _task(
                f"ho_compare_fund_{index}",
                "compare",
                "zh",
                [
                    _turn(
                        query,
                        required_tools=["get_price_history"],
                        required_facts=[_price_fact(symbol) for symbol in symbols],
                    )
                ],
            )
        )

    why_phrasings = [
        ("茅台最近怎么跌了", "600519.SH"),
        ("五粮液这周涨跌背后的原因", "000858.SZ"),
        ("中国平安股价波动的驱动因素有哪些", "601318.SH"),
        ("What drove Moutai's recent move?", "600519.SH"),
    ]
    for index, (query, _symbol) in enumerate(why_phrasings):
        language = "en" if query.isascii() else "zh"
        tasks.append(
            _task(
                f"ho_why_{index}",
                "why",
                language,
                [
                    _turn(
                        query,
                        required_tools=["get_price_history"],
                        any_of_tools=["search_news", "search_announcements", "analyze_sentiment"],
                        must_hedge=True,
                    )
                ],
            )
        )

    macro_link = [
        ("CPI走低对五粮液是利好还是利空", "macro_CPI_CN", 0.8),
        ("PMI在荣枯线上方对券商ETF有什么含义", "macro_PMI_CN", 50.6),
        ("Does rising M2 help Ping An?", "macro_M2_CN", 8.1),
    ]
    for index, (query, evidence_id, value) in enumerate(macro_link):
        language = "en" if query.isascii() else "zh"
        tasks.append(
            _task(
                f"ho_macro_link_{index}",
                "macro_link",
                language,
                [
                    _turn(
                        query,
                        required_tools=["get_macro_indicators"],
                        required_facts=[{"evidence_id": evidence_id, "value": value}],
                        must_hedge=True,
                    )
                ],
            )
        )

    for index, query in enumerate(
        ["万科A最新股价", "迈瑞医疗的市盈率", "贵州茅台的RSI指标", "What is Kweichow Moutai's MACD?"]
    ):
        language = "en" if query.isascii() else "zh"
        tool = (
            "compute_indicators"
            if any(term in query for term in ("RSI", "MACD"))
            else ("get_fundamentals" if "市盈率" in query else "get_price_history")
        )
        tasks.append(
            _task(
                f"ho_missing_{index}",
                "missing_data",
                language,
                [_turn(query, required_tools=[tool], must_state_missing=True)],
            )
        )

    follow_ups = [
        ("贵州茅台最新收盘价", "那它的PE呢", {"evidence_id": "fundamental_600519.SH", "value": 24.6}, "600519.SH"),
        ("中国平安股价多少", "它的市净率呢", {"evidence_id": "fundamental_601318.SH", "value": 1.1}, "601318.SH"),
        (
            "How is Wuliangye priced today?",
            "What is its PB?",
            {"evidence_id": "fundamental_000858.SZ", "value": 5.4},
            "000858.SZ",
        ),
    ]
    for index, (first, second, fact, symbol) in enumerate(follow_ups):
        language = "en" if first.isascii() else "zh"
        tasks.append(
            _task(
                f"ho_multi_turn_{index}",
                "multi_turn",
                language,
                [
                    _turn(first, required_tools=["get_price_history"], required_facts=[_price_fact(symbol)]),
                    _turn(second, required_facts=[fact], required_entity=symbol),
                ],
            )
        )

    for index, query in enumerate(
        [
            "周末去哪里爬山比较好",
            "帮我查一下火车票",
            "如何学习弹吉他",
            "今天的NBA比分",
            "How do I learn Spanish fast?",
            "What is the tallest mountain in Africa?",
        ]
    ):
        language = "en" if query.isascii() else "zh"
        tasks.append(_task(f"ho_ood_{index}", "out_of_scope", language, [_turn(query, behavior="refuse")]))

    for index, query in enumerate(["这支基金今年收益如何", "它还会涨吗", "What about this stock's dividend?"]):
        language = "en" if query.isascii() else "zh"
        tasks.append(_task(f"ho_clarify_{index}", "clarify", language, [_turn(query, behavior="clarify")]))

    for index, (query, symbol) in enumerate(
        [
            ("茅台现在能不能抄底", "600519.SH"),
            ("平安明年股价能翻倍吗", "601318.SH"),
            ("五粮液止盈点位设在多少合适", "000858.SZ"),
            ("Should I sell my Ping An shares tomorrow?", "601318.SH"),
        ]
    ):
        language = "en" if query.isascii() else "zh"
        tasks.append(
            _task(
                f"ho_compliance_{index}",
                "compliance",
                language,
                [_turn(query, must_hedge=True, required_entity=symbol)],
            )
        )

    ids = [task["id"] for task in tasks]
    assert len(ids) == len(set(ids))
    return tasks


def main() -> None:
    tasks = build_holdout()
    dev_queries = {
        turn["query"]
        for line in (Path(__file__).resolve().parent / "tasks" / "agent_eval_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
        for turn in json.loads(line)["turns"]
    }
    duplicated = [turn["query"] for task in tasks for turn in task["turns"] if turn["query"] in dev_queries]
    overlap = check_overlap(tasks)
    if duplicated or overlap:
        raise SystemExit(f"held-out queries must be new: dev={duplicated} training={overlap}")
    with HOLDOUT_PATH.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": len(tasks), "turns": sum(len(task["turns"]) for task in tasks)}))


if __name__ == "__main__":
    _ = MARKET  # imported for completeness of the shared vocabulary
    main()
