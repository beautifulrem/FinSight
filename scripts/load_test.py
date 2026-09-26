"""Closed-loop load test against a running FinSight server.

``--users`` concurrent clients each send ``--requests`` questions back to back to ``/agent/chat`` and
record latency and status. Reports throughput, P50/P95/P99 latency and the error rate, and for LLM
modes (``auto``/``agent``) also the routes taken, LLM calls, tokens and the billed cost aggregated
from each response's ``llm`` block, converted to CNY, per 1,000 questions and per month.

    # service only (no LLM): classical NLU, deterministic planner, template answer
    python -m scripts.load_test --base-url http://127.0.0.1:8001 --users 16 --requests 25
    # LLM agent path with cost accounting
    python -m scripts.load_test --base-url http://127.0.0.1:8801 --mode agent --questions research \\
        --users 8 --requests 5 --usd-cny 6.7489 --questions-per-day 2000

Cost: the gateway reports ``usage.cost`` in USD per call; the server sums it per run as
``llm.usage.reported_cost_usd``. When the server is configured with a price table instead, ``llm.cost``
is used in its currency. ``--usd-cny`` (default ``QI_LLM_USD_CNY``) converts USD to CNY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

# Rotation for the service test: price, valuation, comparison, why, English, macro, out-of-scope and a
# dangling reference (clarification). Unchanged since the first load test so runs stay comparable.
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

# Answerable research questions for the LLM path (comparison, why, fundamentals, macro linkage, ETF,
# trend, English), so the measured cost reflects real agent traffic instead of being diluted by cheap
# refusals and clarifications.
RESEARCH_QUESTIONS = [
    "对比一下贵州茅台和五粮液的估值，并说明差异的原因",
    "中国平安最近为什么下跌？结合基本面和新闻分析",
    "宁德时代的盈利能力怎么样？看看ROE和毛利率",
    "CPI 和 PMI 最近的变化对消费板块有什么影响？",
    "沪深300ETF的费率和最近走势如何？",
    "贵州茅台最近一个月的走势和成交量有什么特点？",
    "Compare the valuation of Kweichow Moutai and Wuliangye and explain the gap.",
    "五粮液最新的营收和净利润增长情况如何？",
]

QUESTION_SETS = {"default": QUESTIONS, "research": RESEARCH_QUESTIONS}


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (no interpolation): with fewer than 100 samples P99 is the maximum."""
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)], 1)


def _llm_fields(body: dict) -> dict:
    llm = body.get("llm") or {}
    usage = llm.get("usage") or {}
    log = llm.get("log") or []
    return {
        "route": body.get("route"),
        "answer_source": body.get("answer_source"),
        "verified": (body.get("verification") or {}).get("passed"),
        "degraded": body.get("degraded") or [],
        "llm_calls": llm.get("calls") or len(log),
        "models": sorted({str(call.get("model")) for call in log if call.get("model")}),
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "cache_hit_tokens": usage.get("prompt_cache_hit_tokens") or 0,
        "reported_cost_usd": usage.get("reported_cost_usd") or 0.0,
        "cost": llm.get("cost"),
        "currency": llm.get("currency"),
        "cost_source": llm.get("cost_source"),
    }


async def user(
    client: httpx.AsyncClient, index: int, requests: int, mode: str, questions: list[str], results: list[dict]
) -> None:
    for step in range(requests):
        query = questions[(index + step) % len(questions)]
        started = time.perf_counter()
        record: dict = {"user": index, "step": step, "query": query}
        try:
            response = await client.post(
                "/agent/chat", json={"query": query, "mode": mode, "session_id": f"load{index}x{step}"}
            )
            status = response.status_code
            body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            ok = status == 200 and body.get("status") in {"ok", "needs_clarification"}
            if status == 200:
                record.update(_llm_fields(body))
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            ok, status = False, type(exc).__name__
        record.update({"ok": ok, "status": status, "latency_ms": round((time.perf_counter() - started) * 1000, 1)})
        results.append(record)


def _cost_cny(record: dict, usd_cny: float | None) -> float | None:
    """CNY cost of one response; ``None`` when a USD cost cannot be converted."""
    if record.get("cost_source") == "price_table" and record.get("currency") == "CNY":
        return float(record.get("cost") or 0.0)
    usd = float(record.get("reported_cost_usd") or 0.0)
    if not usd:
        return 0.0
    return usd * usd_cny if usd_cny else None


def cost_summary(results: list[dict], usd_cny: float | None, questions_per_day: int) -> dict | None:
    answered = [item for item in results if item.get("status") == 200]
    if not answered or not any("reported_cost_usd" in item for item in answered):
        return None
    usd = sum(float(item.get("reported_cost_usd") or 0.0) for item in answered)
    cny_values = [_cost_cny(item, usd_cny) for item in answered]
    cny = None if any(value is None for value in cny_values) else sum(v for v in cny_values if v is not None)
    per_question_cny = cny / len(answered) if cny is not None else None
    return {
        "requests_counted": len(answered),
        "requests_with_llm": sum(1 for item in answered if (item.get("llm_calls") or 0) > 0),
        "total_usd": round(usd, 6),
        "usd_cny": usd_cny,
        "total_cny": round(cny, 4) if cny is not None else None,
        "per_question_cny": round(per_question_cny, 6) if per_question_cny is not None else None,
        "per_1k_questions_cny": round(per_question_cny * 1000, 2) if per_question_cny is not None else None,
        "per_1k_questions_usd": round(usd / len(answered) * 1000, 3),
        "questions_per_day": questions_per_day,
        "monthly_cny_at_questions_per_day": round(per_question_cny * questions_per_day * 30, 2)
        if per_question_cny is not None
        else None,
        "tokens": {
            "prompt": sum(int(item.get("prompt_tokens") or 0) for item in answered),
            "completion": sum(int(item.get("completion_tokens") or 0) for item in answered),
            "cache_hit": sum(int(item.get("cache_hit_tokens") or 0) for item in answered),
        },
        "llm_calls": sum(int(item.get("llm_calls") or 0) for item in answered),
        "cost_sources": sorted({str(item.get("cost_source")) for item in answered if item.get("cost_source")}),
    }


def _rate(values: list) -> float | None:
    known = [value for value in values if value is not None]
    return round(sum(1 for value in known if value) / len(known), 4) if known else None


def _git_commit() -> str | None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def _environment() -> dict:
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {
        "host": platform.platform(),
        "cpu_count": os.cpu_count(),
        "loadavg_at_start": [round(value, 2) for value in load] if load else None,
        "python": sys.version.split()[0],
    }


async def run(
    base_url: str,
    users: int,
    requests: int,
    mode: str,
    *,
    question_set: str = "default",
    timeout_s: float = 180.0,
    usd_cny: float | None = None,
    questions_per_day: int = 1000,
    label: str | None = None,
    warmup: bool = True,
    fresh_connections: bool = False,
) -> dict:
    questions = QUESTION_SETS[question_set]
    results: list[dict] = []
    environment = _environment()
    # Fresh connections spread requests over replicas behind a Service (kube-proxy balances per
    # connection); keep-alive pins each simulated user to one replica.
    limits = httpx.Limits(max_connections=users + 1, max_keepalive_connections=0 if fresh_connections else users + 1)
    warmup_record: dict | None = None
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s, limits=limits, trust_env=False) as client:
        if warmup:
            started = time.perf_counter()
            response = await client.post("/agent/chat", json={"query": questions[0], "mode": mode})
            warmup_record = {
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            if response.status_code == 200:
                warmup_record.update(_llm_fields(response.json()))
        started = time.perf_counter()
        await asyncio.gather(*(user(client, index, requests, mode, questions, results) for index in range(users)))
        wall = time.perf_counter() - started
    latencies = [item["latency_ms"] for item in results if item["ok"]]
    answered = [item for item in results if item.get("status") == 200]
    keys = (
        "user",
        "step",
        "query",
        "ok",
        "status",
        "latency_ms",
        "route",
        "answer_source",
        "verified",
        "llm_calls",
        "models",
        "prompt_tokens",
        "completion_tokens",
        "reported_cost_usd",
        "degraded",
    )
    return {
        "label": label,
        "base_url": base_url,
        "mode": mode,
        "question_set": question_set,
        "fresh_connections": fresh_connections,
        "users": users,
        "requests_per_user": requests,
        "requests": len(results),
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(len(results) / wall, 3),
        "error_rate": round(sum(1 for item in results if not item["ok"]) / len(results), 4),
        "latency_ms": {
            "n": len(latencies),
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "mean": round(statistics.fmean(latencies), 1),
            "max": round(max(latencies), 1),
        }
        if latencies
        else None,
        "statuses": dict(Counter(str(item["status"]) for item in results)),
        "routes": dict(Counter(str(item.get("route")) for item in answered)),
        "answer_sources": dict(Counter(str(item.get("answer_source")) for item in answered)),
        "verified_rate": _rate([item.get("verified") for item in answered]),
        "degraded_rate": _rate([bool(item.get("degraded")) for item in answered]),
        "models": dict(Counter(model for item in answered for model in item.get("models") or [])),
        "cost": cost_summary(results, usd_cny, questions_per_day),
        "warmup": warmup_record,
        "environment": environment,
        "commit": _git_commit(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "per_request": [{key: item[key] for key in keys if key in item} for item in results],
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Closed-loop load test for /agent/chat.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=16)
    parser.add_argument("--requests", type=int, default=25, help="Requests per user.")
    parser.add_argument("--mode", default="workflow", choices=["workflow", "auto", "agent"])
    parser.add_argument("--questions", default="default", choices=sorted(QUESTION_SETS), help="Question rotation.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request client timeout (s).")
    parser.add_argument(
        "--usd-cny",
        type=float,
        default=float(os.getenv("QI_LLM_USD_CNY") or 0) or None,
        help="USD->CNY rate for gateway-reported costs (default: QI_LLM_USD_CNY).",
    )
    parser.add_argument("--questions-per-day", type=int, default=1000, help="For the monthly cost estimate.")
    parser.add_argument("--label", default=None, help="Free-form label stored in the report.")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--fresh-connections", action="store_true", help="No keep-alive (balance over replicas).")
    parser.add_argument("--out", default="outputs/load_test.json")
    args = parser.parse_args(argv)
    report = asyncio.run(
        run(
            args.base_url,
            args.users,
            args.requests,
            args.mode,
            question_set=args.questions,
            timeout_s=args.timeout,
            usd_cny=args.usd_cny,
            questions_per_day=args.questions_per_day,
            label=args.label,
            warmup=not args.no_warmup,
            fresh_connections=args.fresh_connections,
        )
    )
    report["command"] = "python -m scripts.load_test " + " ".join(sys.argv[1:] if argv is None else argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        json.dumps({key: value for key, value in report.items() if key != "per_request"}, ensure_ascii=False, indent=1)
    )
    return report


if __name__ == "__main__":
    main()
