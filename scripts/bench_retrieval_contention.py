"""Tool-stage benchmark for the P95 driver of the 4-user LLM load run (round 13, round-9 review Cost item 1).

The slowest request of ``docs/results/perf/agent/r12-load-4u-auto.json`` (53.8 s, "CPI 和 PMI 最近的变化对消费板块有
什么影响？", agent path) spent 21.3 s in six LLM calls, **20.0 s in tools** and 5.9 s in ``guard_in``; its
``get_fundamentals`` calls took 9.5 s each although the call itself takes ~1 ms, and ``search_news`` /
``search_knowledge`` / ``analyze_sentiment`` took 3.4–6.2 s against ~0.3 s alone. A profile of one ``search_news`` call put ~87% of it in
``scipy.sparse`` ``csr_tocsc``: ``linear_kernel(query, doc_matrix)`` transposes the whole TF-IDF matrix on every
search. Under four concurrent users those CPU-bound conversions queue in the shared tool pool (8 workers) and starve
the cheap lookups behind them.

This script replays the tool calls of the twelve traced requests of that run, in process, on the offline service, and
times them with the scoring done the old way (``linear_kernel``, transposing per call) and the new way
(``DocumentRetriever._scores``, the transposed matrix computed once). Rounds alternate old/new so host-load drift hits
both alike. Two scenarios:

* ``sequential``: every retrieval-backed call once per round, cache cleared before each call;
* ``concurrent``: four "users", each sending its three requests in order, each request running its tool calls with up to
  four in parallel (the agent's ``max_parallel_tools``) through one shared registry, cache cleared per request; reports
  the per-request tool-stage wall time and every call's latency.

    python -m scripts.bench_retrieval_contention --rounds 6 --out docs/results/perf/agent/r13-retrieval-bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The tool calls of the twelve requests in r12-load-4u-auto.json, from that run's traces (trace-file order, which is not
# the original per-user order; the benchmark splits them into four users x three requests). Agent-path calls are the
# prefetch and loop calls as recorded.
REQUESTS: list[dict[str, Any]] = [
    {"query": "对比一下贵州茅台和五粮液的估值，并说明差异的原因", "calls": [
        ["get_price_history", {"target": "600519.SH", "days": 10}], ["get_fundamentals", {"target": "600519.SH"}],
        ["search_news", {"query": "原因 估值", "targets": ["600519.SH"], "top_k": 5}],
        ["get_price_history", {"target": "000858.SZ", "days": 10}], ["get_fundamentals", {"target": "000858.SZ"}],
        ["search_news", {"query": "原因 估值", "targets": ["000858.SZ"], "top_k": 5}]]},
    {"query": "中国平安最近为什么下跌？结合基本面和新闻分析", "calls": [
        ["get_price_history", {"target": "601318.SH", "days": 10}], ["get_fundamentals", {"target": "601318.SH"}],
        ["search_news", {"query": "下跌 新闻", "targets": ["601318.SH"], "top_k": 5}],
        ["compute_indicators", {"target": "601318.SH"}],
        ["analyze_sentiment", {"targets": ["601318.SH"], "query": "", "top_k": 6}],
        ["search_announcements", {"query": "业绩 保费 分红", "targets": ["601318.SH"], "top_k": 5}],
        ["get_fundamentals", {"target": "保险"}]]},
    {"query": "沪深300ETF的费率和最近走势如何？", "calls": [
        ["get_price_history", {"target": "510300.SH", "days": 10}], ["compute_indicators", {"target": "510300.SH"}],
        ["analyze_sentiment", {"targets": ["510300.SH"], "query": "", "top_k": 6}],
        ["search_knowledge", {"query": "沪深300ETF的费率和最近走势如何", "targets": [], "top_k": 5}]]},
    {"query": "宁德时代的盈利能力怎么样？看看ROE和毛利率", "calls": [["get_fundamentals", {"target": "300750.SZ"}]]},
    {"query": "贵州茅台最近一个月的走势和成交量有什么特点？", "calls": [
        ["get_price_history", {"target": "600519.SH", "days": 10}], ["compute_indicators", {"target": "600519.SH"}],
        ["get_fundamentals", {"target": "600519.SH"}],
        ["analyze_sentiment", {"targets": ["600519.SH"], "query": "", "top_k": 6}]]},
    {"query": "CPI 和 PMI 最近的变化对消费板块有什么影响？", "calls": [
        ["get_macro_indicators", {"topics": ["CPI", "PMI"], "query": "CPI 和 PMI 最近的变化对消费板块有什么影响"}],
        ["get_fundamentals", {"target": "消费"}],
        ["search_news", {"query": "CPI PMI 消费板块 影响", "targets": ["消费"], "top_k": 6}],
        ["resolve_entity", {"text": "消费"}], ["resolve_entity", {"text": "消费ETF"}],
        ["resolve_entity", {"text": "中证消费"}], ["get_fundamentals", {"target": "食品饮料"}],
        ["get_fundamentals", {"target": "白酒"}], ["get_price_history", {"target": "008975.SZ", "days": 15}],
        ["resolve_entity", {"text": "食品饮料"}],
        ["search_news", {"query": "CPI PMI 消费 板块 影响 走势", "targets": [], "top_k": 8}],
        ["get_fundamentals", {"target": "家用电器"}], ["get_fundamentals", {"target": "社会服务"}],
        ["get_macro_indicators", {"topics": ["CPI", "PMI", "社会消费品零售", "M2"],
                                  "query": "消费 社会消费品零售 社零 CPI PMI"}]]},
    {"query": "CPI 和 PMI 最近的变化对消费板块有什么影响？", "calls": [
        ["get_macro_indicators", {"topics": ["CPI", "PMI"], "query": "CPI 和 PMI 最近的变化对消费板块有什么影响"}],
        ["get_fundamentals", {"target": "消费"}],
        ["search_news", {"query": "CPI PMI 消费板块 影响", "targets": ["消费"], "top_k": 5}],
        ["get_macro_indicators", {"topics": ["CPI", "PMI", "M2", "降息"], "query": "消费 社零 政策"}],
        ["resolve_entity", {"text": "消费板块"}],
        ["search_news", {"query": "CPI PMI 消费板块 影响 白酒 家电", "targets": [], "top_k": 8}],
        ["get_fundamentals", {"target": "白酒"}], ["get_fundamentals", {"target": "食品饮料"}]]},
    {"query": "中国平安最近为什么下跌？结合基本面和新闻分析", "calls": [
        ["get_price_history", {"target": "601318.SH", "days": 10}], ["get_fundamentals", {"target": "601318.SH"}],
        ["search_news", {"query": "下跌 新闻", "targets": ["601318.SH"], "top_k": 5}],
        ["compute_indicators", {"target": "601318.SH"}],
        ["analyze_sentiment", {"targets": ["601318.SH"], "query": "下跌 业绩 保险", "top_k": 6}]]},
    {"query": "CPI 和 PMI 最近的变化对消费板块有什么影响？", "calls": [
        ["get_macro_indicators", {"topics": ["CPI", "PMI"], "query": "CPI 和 PMI 最近的变化对消费板块有什么影响"}],
        ["resolve_entity", {"text": "消费板块"}], ["get_fundamentals", {"target": "食品饮料"}],
        ["search_news", {"query": "CPI PMI 消费板块 影响", "targets": ["消费"], "top_k": 5}],
        ["search_news", {"query": "CPI PMI 消费板块 影响", "targets": [], "top_k": 6}],
        ["search_knowledge", {"query": "CPI PMI 对消费板块的影响 传导机制", "targets": [], "top_k": 6}],
        ["get_fundamentals", {"target": "白酒"}], ["resolve_entity", {"text": "消费ETF"}],
        ["get_price_history", {"target": "000932.SH", "days": 10}]]},
    {"query": "沪深300ETF的费率和最近走势如何？", "calls": [
        ["get_price_history", {"target": "510300.SH", "days": 10}], ["compute_indicators", {"target": "510300.SH"}],
        ["analyze_sentiment", {"targets": ["510300.SH"], "query": "", "top_k": 6}],
        ["search_knowledge", {"query": "沪深300ETF的费率和最近走势如何", "targets": [], "top_k": 5}]]},
    {"query": "宁德时代的盈利能力怎么样？看看ROE和毛利率", "calls": [["get_fundamentals", {"target": "300750.SZ"}]]},
    {"query": "宁德时代的盈利能力怎么样？看看ROE和毛利率", "calls": [["get_fundamentals", {"target": "300750.SZ"}]]},
]  # fmt: skip
RETRIEVAL_TOOLS = frozenset({"search_news", "search_knowledge", "search_announcements", "analyze_sentiment"})


def _legacy_scores(self: Any, query_vector: Any) -> Any:
    from sklearn.metrics.pairwise import linear_kernel

    return linear_kernel(query_vector, self.doc_matrix).flatten()


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))], 1)


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": _pct(values, 0.5),
        "p95": _pct(values, 0.95),
        "max": round(max(values), 1) if values else None,
        "mean": round(statistics.fmean(values), 1) if values else None,
    }


def run_request(
    registry: Any, calls: list[list[Any]], latencies: dict[str, list[float]], lock: threading.Lock
) -> float:
    registry.clear_cache()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:  # AgentConfig.max_parallel_tools
        results = list(pool.map(lambda call: registry.run(call[0], call[1]), calls))
    wall = (time.perf_counter() - started) * 1000
    with lock:
        for call, result in zip(calls, results, strict=True):
            latencies.setdefault(call[0], []).append(result.latency_ms)
    return wall


def concurrent_round(registry: Any) -> tuple[list[float], dict[str, list[float]]]:
    latencies: dict[str, list[float]] = {}
    walls: list[float] = []
    lock = threading.Lock()

    def user(index: int) -> None:
        for request in REQUESTS[index * 3 : index * 3 + 3]:
            wall = run_request(registry, request["calls"], latencies, lock)
            with lock:
                walls.append(wall)

    threads = [threading.Thread(target=user, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return walls, latencies


def sequential_round(registry: Any) -> dict[str, list[float]]:
    latencies: dict[str, list[float]] = {}
    seen = set()
    for request in REQUESTS:
        for name, args in request["calls"]:
            key = json.dumps([name, args], ensure_ascii=False, sort_keys=True)
            if name not in RETRIEVAL_TOOLS or key in seen:
                continue
            seen.add(key)
            registry.clear_cache()
            latencies.setdefault(name, []).append(registry.run(name, args).latency_ms)
    return latencies


def main(argv: list[str] | None = None) -> dict[str, Any]:
    from scripts.provenance import env_switches, git_state

    state = git_state()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rounds", type=int, default=6, help="Rounds per variant (alternating).")
    parser.add_argument("--out", default="outputs/perf/r13-retrieval-bench.json")
    args = parser.parse_args(argv)

    from evaluation.agent_eval.runner import build_offline_service
    from query_intelligence.agent.tools import build_registry_for_service
    from query_intelligence.retrieval.doc_retriever import DocumentRetriever

    current_scores = DocumentRetriever._scores
    service = build_offline_service()
    registry = build_registry_for_service(service)
    retriever = service.retrieval_pipeline.doc_retriever
    probe = retriever.vectorizer.transform(["中国平安 下跌 原因"])
    identical = bool((_legacy_scores(retriever, probe) == current_scores(retriever, probe)).all())
    variants = {"legacy_linear_kernel": _legacy_scores, "transposed_once": current_scores}
    collected: dict[str, dict[str, Any]] = {
        name: {"request_wall": [], "concurrent": {}, "sequential": {}} for name in variants
    }
    loads = []
    concurrent_round(registry)  # warm-up (model loads, first-call caches), not recorded
    for _round in range(args.rounds):
        for name, scores in variants.items():
            DocumentRetriever._scores = scores
            loads.append(os.getloadavg()[0])
            for tool, values in sequential_round(registry).items():
                collected[name]["sequential"].setdefault(tool, []).extend(values)
            walls, latencies = concurrent_round(registry)
            collected[name]["request_wall"].extend(walls)
            for tool, values in latencies.items():
                collected[name]["concurrent"].setdefault(tool, []).extend(values)
    DocumentRetriever._scores = current_scores
    registry.shutdown()

    def summarise(data: dict[str, Any]) -> dict[str, Any]:
        retrieval = [v for tool, values in data["concurrent"].items() if tool in RETRIEVAL_TOOLS for v in values]
        lookups = [v for tool, values in data["concurrent"].items() if tool not in RETRIEVAL_TOOLS for v in values]
        return {
            "concurrent_request_tool_stage_ms": _stats(data["request_wall"]),
            "concurrent_retrieval_call_ms": _stats(retrieval),
            "concurrent_lookup_call_ms": _stats(lookups),
            "concurrent_by_tool_ms": {tool: _stats(values) for tool, values in sorted(data["concurrent"].items())},
            "sequential_by_tool_ms": {tool: _stats(values) for tool, values in sorted(data["sequential"].items())},
        }

    report = {
        "kind": "retrieval_contention_bench",
        "config": {
            **state,
            "model": None,
            "rounds_per_variant": args.rounds,
            "requests_per_round": len(REQUESTS),
            "users": 4,
            "parallel_tools_per_request": 4,
            "scores_identical_on_probe": identical,
            "loadavg_1m_per_round": [round(value, 1) for value in loads],
            "cpu_count": os.cpu_count(),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": "python -m scripts.bench_retrieval_contention " + " ".join(argv or sys.argv[1:]),
            "env": env_switches(),
        },
        "variants": {name: summarise(data) for name, data in collected.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, data in report["variants"].items():
        print(
            name,
            "request tool stage",
            data["concurrent_request_tool_stage_ms"],
            "retrieval calls",
            data["concurrent_retrieval_call_ms"],
            "lookups",
            data["concurrent_lookup_call_ms"],
        )
    return report


if __name__ == "__main__":
    main()
