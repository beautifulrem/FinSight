"""Closed-loop load test against a running FinSight server.

``--users`` concurrent clients each send ``--requests`` questions back to back to ``/agent/chat`` and
record latency and status. Reports throughput, P50/P95/P99 latency and the error rate. Use the
``workflow`` mode (no LLM) to measure the service itself; LLM modes are dominated by provider latency.

    python -m scripts.load_test --base-url http://127.0.0.1:8001 --users 16 --requests 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

QUESTIONS = [
    "贵州茅台最新收盘价是多少？",
    "五粮液的市盈率是多少？",
    "对比一下贵州茅台和五粮液的估值",
    "中国平安最近为什么下跌？",
    "What was the latest close of Kweichow Moutai (600519.SH)?",
    "CPI 对消费板块有什么影响？",
    "今天天气怎么样？",
    "它的市盈率呢",
]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)], 1)


async def user(client: httpx.AsyncClient, index: int, requests: int, mode: str, results: list[dict]) -> None:
    for step in range(requests):
        query = QUESTIONS[(index + step) % len(QUESTIONS)]
        started = time.perf_counter()
        try:
            response = await client.post(
                "/agent/chat", json={"query": query, "mode": mode, "session_id": f"load{index}x{step}"}
            )
            ok = response.status_code == 200 and response.json().get("status") in {"ok", "needs_clarification"}
            status = response.status_code
        except httpx.HTTPError as exc:
            ok, status = False, type(exc).__name__
        results.append({"ok": ok, "status": status, "latency_ms": (time.perf_counter() - started) * 1000})


async def run(base_url: str, users: int, requests: int, mode: str) -> dict:
    results: list[dict] = []
    limits = httpx.Limits(max_connections=users, max_keepalive_connections=users)
    async with httpx.AsyncClient(base_url=base_url, timeout=120, limits=limits, trust_env=False) as client:
        await client.post("/agent/chat", json={"query": "贵州茅台最新收盘价是多少？", "mode": mode})  # warm-up
        started = time.perf_counter()
        await asyncio.gather(*(user(client, index, requests, mode, results) for index in range(users)))
        wall = time.perf_counter() - started
    latencies = [item["latency_ms"] for item in results if item["ok"]]
    return {
        "base_url": base_url,
        "mode": mode,
        "users": users,
        "requests": len(results),
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(len(results) / wall, 2),
        "error_rate": round(sum(1 for item in results if not item["ok"]) / len(results), 4),
        "latency_ms": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "mean": round(statistics.fmean(latencies), 1),
        }
        if latencies
        else None,
        "statuses": sorted({str(item["status"]) for item in results}),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Closed-loop load test for /agent/chat.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=16)
    parser.add_argument("--requests", type=int, default=25, help="Requests per user.")
    parser.add_argument("--mode", default="workflow", choices=["workflow", "auto", "agent"])
    parser.add_argument("--out", default="outputs/load_test.json")
    args = parser.parse_args(argv)
    report = asyncio.run(run(args.base_url, args.users, args.requests, args.mode))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main()
