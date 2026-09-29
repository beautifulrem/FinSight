"""Isolate per-request overhead of the LLM gateway: keep-alive vs a new connection per request.

Sends ``--requests`` tiny sequential chat requests (one short user message, ``max_tokens`` 8, reasoning
off where the model allows it) with a pooled client and with a fresh connection per request, interleaved
(A, B, A, B, ...) so both see the same gateway conditions. Reports P50/P95 latency per variant and the
HTTP counters (429s included). No prompt or key is printed.

    source /tmp/llmenv.sh
    python -m scripts.llm_latency_probe --requests 20 --out docs/results/perf/agent/keepalive-probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from query_intelligence.agent.llm import DeepSeekToolClient, LLMError
from query_intelligence.chatbot import load_chatbot_config
try:
    from scripts.provenance import commit_label, git_state
except ModuleNotFoundError:  # run as a file (python scripts/x.py): scripts/ itself is on sys.path
    from provenance import commit_label, git_state  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)], 1)


def _commit() -> str:
    return commit_label(git_state(ROOT))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=20, help="Requests per variant.")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    config = load_chatbot_config()
    variants = {
        "keepalive": DeepSeekToolClient.from_chatbot_config(config, keepalive=True),
        "new_connection": DeepSeekToolClient.from_chatbot_config(config, keepalive=False),
    }
    messages = [{"role": "user", "content": "Reply with the single word: ok"}]
    latencies: dict[str, list[float]] = {name: [] for name in variants}
    errors: dict[str, int] = {name: 0 for name in variants}
    for _ in range(args.requests):
        for name, client in variants.items():
            started = time.perf_counter()
            try:
                client.chat(messages, max_tokens=8, reasoning="off")
            except LLMError:
                errors[name] += 1
                continue
            latencies[name].append((time.perf_counter() - started) * 1000)
    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _commit(),
        "model": variants["keepalive"].model,
        "host": platform.platform(),
        "load_average": list(os.getloadavg()),
        "requests_per_variant": args.requests,
        "command": "python -m scripts.llm_latency_probe " + " ".join(argv if argv is not None else sys.argv[1:]),
        "variants": {
            name: {
                "ok": len(latencies[name]),
                "errors": errors[name],
                "latency_ms_p50": _percentile(latencies[name], 0.5),
                "latency_ms_p95": _percentile(latencies[name], 0.95),
                "latency_ms_mean": round(statistics.fmean(latencies[name]), 1) if latencies[name] else None,
                "first_request_ms": round(latencies[name][0], 1) if latencies[name] else None,
                "http": client.http_stats(),
            }
            for name, client in variants.items()
        },
    }
    variants["keepalive"].close()
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
