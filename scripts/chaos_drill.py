"""Chaos drill against a live FinSight server: real faults, real network, real metrics.

Two scenarios, each starting its own uvicorn server on ``--port`` (default 8821) with fault-injecting proxies in front
of its dependencies (nothing is stubbed inside the process under test):

``llm`` – model failover and breaker recovery.
    An LLM fault proxy sits between the server and the OpenAI-compatible gateway. While the fault is on,
    every request for the primary model is forwarded with its model id rewritten to an invalid one, so
    the *real* gateway rejects it (the same failure as a mistyped ``DEEPSEEK_MODEL``). Phases:
    baseline -> fault on (agent requests; the primary fails, ``FallbackLLM`` fails over to the fallback
    model and opens the primary's breaker after 3 failures) -> fault healed, wait for the cool-down
    (``/metrics`` shows the primary half-open) -> one request (the trial call reaches the primary,
    succeeds, and the breaker closes). Requires the gateway credentials in the environment
    (``source /tmp/llmenv.sh``) and ``--fallback-model``.

``sources-load`` – the ``sources`` setup under load (no LLM).
    Same blocking proxy and server, but each phase is a closed-loop load test (``scripts/load_test.py``,
    workflow mode, ``--load-users`` users x ``--load-requests`` questions of its default rotation): live ->
    blocked within the 60 s TTL (cache) -> blocked after the TTL (last-known-good) -> blocked after the stale
    window (snapshot policy) -> unblocked after the breaker cool-down. Every phase
    records P50/P95/P99, the error and degraded rates, which ``source/mode`` served each evidence item, and
    ``/sources/health`` plus the breaker metrics at its end.

``sources`` – live data fallback chain.
    A blocking HTTP(S) proxy is set as ``HTTPS_PROXY``/``HTTP_PROXY`` for the server. It forwards
    traffic (to the machine's own upstream proxy if one is configured) but refuses Sina, Tencent and
    Eastmoney hosts while blocking is on. Phases: live -> blocked while the market bundle is still in its
    60 s TTL (cache hit) and a never-cached macro question (straight to the snapshot) -> blocked after
    the TTL (last-known-good) -> blocked after the stale window (snapshot policy) -> unblocked after the
    breaker cool-down (recovery). Every phase records latency and the provenance of the price,
    fundamentals and macro evidence, plus ``/sources/health`` and the breaker metrics.

    source /tmp/llmenv.sh
    python -m scripts.chaos_drill --scenario llm --fallback-model cline-pass/glm-5.3-flash
    python -m scripts.chaos_drill --scenario sources
    python -m scripts.chaos_drill --scenario sources-load --load-users 8 --load-requests 10

Results go to ``--out`` (JSON), with the server log and the traces next to it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

try:
    from scripts.load_test import run as load_test_run
    from scripts.provenance import commit_label, env_switches, git_state
except ModuleNotFoundError:  # run as a file (python scripts/x.py): scripts/ itself is on sys.path
    from load_test import run as load_test_run  # type: ignore[no-redef]
    from provenance import commit_label, env_switches, git_state  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
INVALID_MODEL = "cline-pass/chaos-invalid-model"
DEFAULT_BLOCKED = ("sina.com.cn", "sinajs.cn", "sina.cn", "gtimg.cn", "eastmoney.com")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---- LLM fault proxy ------------------------------------------------------------------------------


class LLMFaultProxy:
    """Forwards OpenAI-compatible requests to ``upstream``; rewrites the primary model id while faulty.

    Never records request headers (the API key) or message contents; only model, fault flag, status and
    latency per request.
    """

    def __init__(self, upstream: str, fail_models: set[str], fault_status: int | None = None) -> None:
        self.upstream = upstream.rstrip("/")
        self.fail_models = fail_models
        # (round 12) with a fault status the proxy answers it itself while the fault is on (a provider 5xx burst):
        # nothing reaches the gateway, so the burst costs no quota
        self.fault_status = fault_status
        self.fault_on = False
        self.log: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=180, trust_env=True)
        self.port = _free_port()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                started = time.perf_counter()
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    body = {}
                model = str(body.get("model"))
                faulty = proxy.fault_on and model in proxy.fail_models
                if faulty and proxy.fault_status:
                    payload = json.dumps(
                        {"error": {"message": f"injected HTTP {proxy.fault_status} (chaos drill)", "type": "chaos"}}
                    ).encode()
                    with proxy._lock:
                        proxy.log.append(
                            {
                                "t": round(time.time(), 3),
                                "at": _now(),
                                "model": model,
                                "fault_applied": True,
                                "injected": True,
                                "status": proxy.fault_status,
                                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                                "error": None,
                            }
                        )
                    self.send_response(proxy.fault_status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if faulty:
                    body["model"] = INVALID_MODEL
                    raw = json.dumps(body).encode()
                headers = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() in {"authorization", "content-type", "accept"}
                }
                try:
                    response = proxy._client.post(f"{proxy.upstream}{self.path}", content=raw, headers=headers)
                    status, payload = response.status_code, response.content
                    content_type = response.headers.get("content-type", "application/json")
                except httpx.HTTPError as exc:
                    status, payload, content_type = (
                        502,
                        json.dumps({"error": type(exc).__name__}).encode(),
                        "application/json",
                    )
                with proxy._lock:
                    proxy.log.append(
                        {
                            "t": round(time.time(), 3),
                            "at": _now(),
                            "model": model,
                            "fault_applied": faulty,
                            "status": status,
                            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                            "error": payload[:160].decode("utf-8", "replace") if status >= 400 else None,
                        }
                    )
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self._client.close()

    def entries_since(self, t: float) -> list[dict[str, Any]]:
        with self._lock:
            return [entry for entry in self.log if entry["t"] >= t]


# ---- blocking HTTP(S) proxy ---------------------------------------------------------------------


class BlockingProxy:
    """HTTP proxy (CONNECT + absolute-form) that refuses blocked hosts and forwards everything else."""

    def __init__(self, blocked_suffixes: tuple[str, ...], upstream_proxy: str | None, mode: str = "reject") -> None:
        self.blocked_suffixes = blocked_suffixes
        self.blocking = False
        self.mode = mode
        self.upstream = urlsplit(upstream_proxy) if upstream_proxy else None
        self.port = _free_port()
        self.counts: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(10)

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)

    def is_blocked(self, host: str) -> bool:
        host = host.lower().strip(".")
        return self.blocking and any(host == s or host.endswith("." + s) for s in self.blocked_suffixes)

    def _count(self, host: str, outcome: str) -> None:
        with self._lock:
            bucket = self.counts.setdefault(host, {})
            bucket[outcome] = bucket.get(outcome, 0) + 1

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        server = self._loop.run_until_complete(asyncio.start_server(self._handle, "127.0.0.1", self.port))
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            server.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            writer.close()
            return
        first, _, rest = head.partition(b"\r\n")
        try:
            method, target, version = first.decode("latin-1").split(" ", 2)
        except ValueError:
            writer.close()
            return
        if method == "CONNECT":
            host, _, port = target.rpartition(":")
            port_number = int(port or 443)
        else:
            parts = urlsplit(target)
            host, port_number = parts.hostname or "", parts.port or 80
            # One request per connection, so a kept-alive connection cannot carry a blocked host later.
            lines = [
                line
                for line in rest.split(b"\r\n")
                if line and not line.lower().startswith((b"connection:", b"proxy-connection:"))
            ]
            head = first + b"\r\n" + b"\r\n".join([*lines, b"Connection: close"]) + b"\r\n\r\n"
        if self.is_blocked(host):
            self._count(host, "blocked")
            if self.mode == "hang":
                with contextlib.suppress(Exception):
                    await asyncio.sleep(3600)
            else:
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                with contextlib.suppress(Exception):
                    await writer.drain()
            writer.close()
            return
        self._count(host, "forwarded")
        try:
            if self.upstream is not None:
                up_reader, up_writer = await asyncio.open_connection(self.upstream.hostname, self.upstream.port)
                up_writer.write(head)
            elif method == "CONNECT":
                up_reader, up_writer = await asyncio.open_connection(host, port_number)
                writer.write(f"{version} 200 Connection established\r\n\r\n".encode())
            else:
                up_reader, up_writer = await asyncio.open_connection(host, port_number)
                up_writer.write(head)
        except OSError:
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            writer.close()
            return
        await asyncio.gather(self._pipe(reader, up_writer), self._pipe(up_reader, writer))

    async def _pipe(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()


# ---- server under test ----------------------------------------------------------------------------


class Server:
    def __init__(self, port: int, env: dict[str, str], log_path: Path) -> None:
        self.port = port
        self.env = env
        self.log_path = log_path
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout_s: float = 300) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "query_intelligence.api.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        self.process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout_s
        with httpx.Client(trust_env=False, timeout=5) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"server exited early; see {self.log_path}")
                with contextlib.suppress(httpx.HTTPError):
                    if client.get(f"{self.base_url}/health").status_code == 200:
                        return
                time.sleep(2)
        raise TimeoutError("server did not become healthy")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(20)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=240, trust_env=False)


def _metric_lines(text: str, prefixes: tuple[str, ...]) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(prefixes)]


def _ask(client: httpx.Client, query: str, *, mode: str, session: str) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post("/agent/chat", json={"query": query, "mode": mode, "session_id": session})
    latency = round((time.perf_counter() - started) * 1000, 1)
    body = response.json() if response.status_code == 200 else {"error": response.text[:300]}
    return {"query": query, "status": response.status_code, "latency_ms": latency, "body": body}


# ---- scenario: LLM failover -----------------------------------------------------------------------


def run_llm(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    upstream = os.environ.get("DEEPSEEK_BASE_URL")
    primary = os.environ.get("DEEPSEEK_MODEL")
    if not (upstream and primary and os.environ.get("DEEPSEEK_API_KEY")):
        raise SystemExit(
            "llm scenario needs DEEPSEEK_BASE_URL, DEEPSEEK_MODEL and DEEPSEEK_API_KEY (source /tmp/llmenv.sh)"
        )
    proxy = LLMFaultProxy(upstream, {primary})
    proxy.start()
    trace_dir = out_dir / "traces"
    env = {
        **os.environ,
        "DEEPSEEK_BASE_URL": proxy.url,
        "QI_LLM_FALLBACK_MODELS": args.fallback_model,
        "QI_AGENT_TRACE_DIR": str(trace_dir),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    if args.usd_cny:
        env["QI_LLM_USD_CNY"] = str(args.usd_cny)
    server = Server(args.port, env, out_dir / "server-llm.log")
    report: dict[str, Any] = {"scenario": "llm", "primary": primary, "fallback": args.fallback_model, "phases": []}
    questions = [
        "对比一下贵州茅台和五粮液的估值，并说明差异的原因",
        "中国平安最近为什么下跌？结合基本面和新闻分析",
        "宁德时代的盈利能力怎么样？看看ROE和毛利率",
        "CPI 和 PMI 最近的变化对消费板块有什么影响？",
    ]
    prefixes = ("finsight_llm_circuit_state", "finsight_llm_client_calls_total", "finsight_llm_consecutive_failures")
    try:
        server.start()
        with _client(server.base_url) as client:

            def phase(name: str, queries: list[str]) -> dict[str, Any]:
                t0 = time.time()
                runs = []
                for index, query in enumerate(queries):
                    result = _ask(client, query, mode="agent", session=f"chaos-{name}-{index}")
                    body = result.pop("body")
                    llm = body.get("llm") or {}
                    trace_id = body.get("trace_id")
                    trace = client.get(f"/agent/traces/{trace_id}").json() if trace_id else None
                    runs.append(
                        {
                            **result,
                            "route": body.get("route"),
                            "answer_source": body.get("answer_source"),
                            "verified": (body.get("verification") or {}).get("passed"),
                            "degraded": body.get("degraded"),
                            "llm_calls_by_model": _count_models(llm.get("log") or []),
                            "cost": llm.get("cost"),
                            "currency": llm.get("currency"),
                            "trace_id": trace_id,
                            "trace_llm_calls": [
                                {
                                    k: call.get(k)
                                    for k in ("node", "model", "latency_ms", "prompt_tokens", "completion_tokens")
                                }
                                for call in (trace or {}).get("llm_calls") or []
                            ],
                        }
                    )
                metrics = client.get("/metrics").text
                entry = {
                    "phase": name,
                    "started_at": datetime.fromtimestamp(t0, UTC).isoformat(timespec="seconds"),
                    "requests": runs,
                    "gateway_requests": proxy.entries_since(t0),
                    "metrics": _metric_lines(metrics, prefixes),
                }
                report["phases"].append(entry)
                print(json.dumps({"phase": name, "latency_ms": [r["latency_ms"] for r in runs]}, ensure_ascii=False))
                return entry

            phase("1_baseline", questions[:1])
            proxy.fault_on = True
            fault = phase("2_primary_failing", questions[1:4])
            proxy.fault_on = False
            failures = [entry for entry in fault["gateway_requests"] if entry["fault_applied"]]
            opened_at = failures[-1]["t"] if failures else time.time()
            wait = max(0.0, opened_at + args.llm_cooldown + 2 - time.time())
            print(f"fault healed; waiting {wait:.0f}s for the breaker cool-down")
            time.sleep(wait)
            report["phases"].append(
                {
                    "phase": "3_after_cooldown_before_trial",
                    "at": _now(),
                    "metrics": _metric_lines(client.get("/metrics").text, prefixes),
                }
            )
            phase("4_recovered", questions[:1])
    finally:
        server.stop()
        proxy.stop()
    return report


# ---- scenario: LLM load with a provider 5xx burst --------------------------------------------------------------


_STATE_NAMES = {0.0: "closed", 1.0: "half_open", 2.0: "open"}


def _breaker_states(lines: list[str]) -> dict[str, str]:
    states = {}
    for line in lines:
        if line.startswith("finsight_llm_circuit_state{"):
            model = line.split('model="', 1)[1].split('"', 1)[0]
            states[model] = _STATE_NAMES.get(float(line.rsplit(" ", 1)[1]), line.rsplit(" ", 1)[1])
    return states


_LLM_FAILURE_MARKERS = ("llm_error", "llm_compose_failed", "llm_revision_failed")


def run_llm_load(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """``--load-users`` closed-loop users on the LLM path (``--load-mode``, research questions), optionally with a
    provider-wide HTTP 5xx burst (``--burst-seconds`` > 0) injected by the LLM proxy ``--burst-start`` seconds into
    the run, for the primary and the failover model alike. ``/metrics`` is sampled every second for the breaker
    state of each model; the report has the load summary (latency, user-visible errors, routes, cost), how many
    answered requests had an LLM failure and fell back to the deterministic answer, the gateway traffic by status
    (injected, forwarded, 429) and the breaker timeline."""
    upstream = os.environ.get("DEEPSEEK_BASE_URL")
    primary = os.environ.get("DEEPSEEK_MODEL")
    if not (upstream and primary and os.environ.get("DEEPSEEK_API_KEY")):
        raise SystemExit("llm-load needs DEEPSEEK_BASE_URL, DEEPSEEK_MODEL and DEEPSEEK_API_KEY")
    models = {primary, *([args.fallback_model] if args.fallback_model else [])}
    proxy = LLMFaultProxy(upstream, models, fault_status=args.burst_status)
    proxy.start()
    env = {
        **os.environ,
        "DEEPSEEK_BASE_URL": proxy.url,
        "QI_LLM_FALLBACK_MODELS": args.fallback_model or "",
        "QI_AGENT_TRACE_DIR": str(out_dir / "traces"),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    if args.usd_cny:
        env["QI_LLM_USD_CNY"] = str(args.usd_cny)
    server = Server(args.port, env, out_dir / "server-llm-load.log")
    prefixes = ("finsight_llm_circuit_state", "finsight_llm_client_calls", "finsight_llm_consecutive_failures")
    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    burst: dict[str, Any] = {"status": args.burst_status if args.burst_seconds else None}
    try:
        server.start()
        started = time.time()

        def sample() -> None:
            with _client(server.base_url) as client:
                while not stop.is_set():
                    with contextlib.suppress(httpx.HTTPError):
                        lines = _metric_lines(client.get("/metrics").text, prefixes)
                        samples.append(
                            {
                                "t": round(time.time() - started, 1),
                                "fault_on": proxy.fault_on,
                                "breaker": _breaker_states(lines),
                            }
                        )
                    stop.wait(1.0)

        def inject() -> None:
            if stop.wait(args.burst_start):
                return
            proxy.fault_on = True
            burst["started_s"] = round(time.time() - started, 1)
            stop.wait(args.burst_seconds)
            proxy.fault_on = False
            burst["ended_s"] = round(time.time() - started, 1)

        threads = [threading.Thread(target=sample, daemon=True)]
        if args.burst_seconds:
            threads.append(threading.Thread(target=inject, daemon=True))
        for thread in threads:
            thread.start()
        result = asyncio.run(
            load_test_run(
                server.base_url,
                args.load_users,
                args.load_requests,
                args.load_mode,
                question_set="research",
                timeout_s=240.0,
                usd_cny=args.usd_cny,
                warmup=False,
                label=f"chaos llm-load burst={args.burst_seconds}s",
            )
        )
        recovery: dict[str, Any] | None = None
        if args.burst_seconds and args.recovery_requests:
            # After the burst the breakers stay open for their cool-down (60 s): wait until no model is open, then
            # a short load shows the half-open trial calls and the breakers closing again.
            deadline = time.time() + args.burst_start + args.burst_seconds + 180
            while time.time() < deadline:
                current = samples[-1]["breaker"] if samples else {}
                if burst.get("ended_s") is not None and current and "open" not in current.values():
                    break
                time.sleep(1)
            recovery_started = round(time.time() - started, 1)
            recovery = asyncio.run(
                load_test_run(
                    server.base_url,
                    args.load_users,
                    args.recovery_requests,
                    args.load_mode,
                    question_set="research",
                    timeout_s=240.0,
                    usd_cny=args.usd_cny,
                    warmup=False,
                    label="chaos llm-load recovery",
                )
            )
            recovery["started_s"] = recovery_started
            time.sleep(3)  # one more breaker sample after the last answer
        stop.set()
        for thread in threads:
            thread.join(5)
        with _client(server.base_url) as client:
            final_metrics = _metric_lines(client.get("/metrics").text, prefixes)
    finally:
        stop.set()
        server.stop()
        proxy.stop()
    gateway = proxy.log
    for item in [*result["per_request"], *((recovery or {}).get("per_request") or [])]:
        item["t_start_s"] = round(item.get("started_at_unix", started) - started, 1)
        item["t_end_s"] = round(item["t_start_s"] + item["latency_ms"] / 1000, 1)
        item["during_burst"] = bool(
            burst.get("started_s") is not None
            and item["t_start_s"] < burst.get("ended_s", float("inf"))
            and item["t_end_s"] > burst["started_s"]
        )
    answered = [item for item in result["per_request"] if item.get("status") == 200]
    llm_failed = [
        item
        for item in answered
        if any(str(flag).startswith(_LLM_FAILURE_MARKERS) for flag in item.get("degraded") or [])
    ]
    transitions: list[dict[str, Any]] = []
    last: dict[str, str] = {}
    for entry in samples:
        for model, state in entry["breaker"].items():
            if last.get(model) != state:
                transitions.append({"t": entry["t"], "model": model, "state": state, "fault_on": entry["fault_on"]})
                last[model] = state
    return {
        "scenario": "llm-load",
        "server_env": env_switches(env),
        "primary": primary,
        "fallback": args.fallback_model or None,
        "load": {"users": args.load_users, "requests_per_user": args.load_requests, "mode": args.load_mode},
        "burst": {**burst, "configured_start_s": args.burst_start, "configured_seconds": args.burst_seconds},
        "summary": {
            "requests": result["requests"],
            "user_visible_error_rate": result["error_rate"],
            "statuses": result["statuses"],
            "latency_ms": result["latency_ms"],
            "wall_seconds": result["wall_seconds"],
            "routes": result["routes"],
            "answer_sources": result["answer_sources"],
            "verified_rate": result["verified_rate"],
            "requests_overlapping_burst": sum(1 for item in result["per_request"] if item["during_burst"]),
            "llm_failure_requests": len(llm_failed),
            "llm_failure_fell_back_to_template": sum(
                1 for item in llm_failed if item.get("answer_source") == "template"
            ),
            "gateway_requests": len(gateway),
            "gateway_injected": sum(1 for entry in gateway if entry.get("injected")),
            "gateway_forwarded": sum(1 for entry in gateway if not entry.get("injected")),
            "gateway_forwarded_by_status": dict(
                Counter(str(entry["status"]) for entry in gateway if not entry.get("injected"))
            ),
            "gateway_429": sum(1 for entry in gateway if entry.get("status") == 429),
            "rate_429_of_forwarded": round(
                sum(1 for entry in gateway if entry.get("status") == 429)
                / max(1, sum(1 for entry in gateway if not entry.get("injected"))),
                4,
            ),
            "llm_calls_reported": sum(int(item.get("llm_calls") or 0) for item in answered),
            "cost": result["cost"],
            "breaker_transitions": transitions,
        },
        "load_test": {key: value for key, value in result.items() if key != "per_request"},
        "per_request": result["per_request"],
        "recovery": None
        if recovery is None
        else {
            "started_s": recovery["started_s"],
            "requests": recovery["requests"],
            "user_visible_error_rate": recovery["error_rate"],
            "latency_ms": recovery["latency_ms"],
            "answer_sources": recovery["answer_sources"],
            "llm_failure_requests": sum(
                1
                for item in recovery["per_request"]
                if any(str(flag).startswith(_LLM_FAILURE_MARKERS) for flag in item.get("degraded") or [])
            ),
            "cost": recovery["cost"],
            "per_request": recovery["per_request"],
        },
        "gateway_log": gateway,
        "breaker_samples": samples,
        "final_metrics": final_metrics,
    }


def _count_models(log: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for call in log:
        model = str(call.get("model"))
        counts[model] = counts.get(model, 0) + 1
    return counts


# ---- scenario: data source fallback --------------------------------------------------------------


def _evidence_provenance(body: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in body.get("evidence_sources") or []:
        payload = source.get("payload") or {}
        provenance = payload.get("provenance") if isinstance(payload, dict) else None
        if not isinstance(provenance, dict):
            continue
        rows.append(
            {
                "evidence_id": source.get("evidence_id"),
                "source_type": source.get("source_type"),
                "close": payload.get("close"),
                "trade_date": payload.get("trade_date"),
                **{
                    key: provenance.get(key)
                    for key in ("source", "mode", "freshness", "as_of", "cache_hit", "fallback_reason", "note")
                },
                "provider_warnings": payload.get("provider_warnings"),
            }
        )
    return rows


def _sources_env(proxy: BlockingProxy, args: argparse.Namespace, out_dir: Path) -> dict[str, str]:
    return {
        **os.environ,
        "HTTPS_PROXY": proxy.url,
        "HTTP_PROXY": proxy.url,
        "https_proxy": proxy.url,
        "http_proxy": proxy.url,
        "ALL_PROXY": "",
        "all_proxy": "",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "QI_USE_LIVE_MARKET": "1",
        "QI_USE_LIVE_NEWS": "1",
        "QI_USE_LIVE_ANNOUNCEMENT": "1",
        "QI_USE_LIVE_MACRO": "1",
        "QI_SOURCE_COOLDOWN_SECONDS": str(args.source_cooldown),
        "QI_SOURCE_MAX_STALE_SECONDS": str(args.max_stale),
        "QI_AGENT_TRACE_DIR": str(out_dir / "traces"),
        "DEEPSEEK_API_KEY": "",
    }


def run_sources_load(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """The ``sources`` drill with a closed-loop load test (no LLM) in every phase instead of single questions."""
    upstream_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    proxy = BlockingProxy(tuple(args.block), upstream_proxy, mode=args.block_mode)
    proxy.start()
    server = Server(args.port, _sources_env(proxy, args, out_dir), out_dir / "server-sources-load.log")
    prefixes = ("finsight_source_circuit_state", "finsight_source_pool_", "finsight_source_calls_total")
    report: dict[str, Any] = {
        "scenario": "sources-load",
        "blocked_hosts": list(args.block),
        "block_mode": args.block_mode,
        "upstream_proxy_used": bool(upstream_proxy),
        "load": {"users": args.load_users, "requests_per_user": args.load_requests, "mode": "workflow", "llm": None},
        "settings": {
            "market_bundle_ttl_s": 60,
            "max_stale_s": args.max_stale,
            "source_cooldown_s": args.source_cooldown,
        },
        "phases": [],
    }
    summary_keys = (
        "requests",
        "wall_seconds",
        "throughput_rps",
        "error_rate",
        "latency_ms",
        "statuses",
        "routes",
        "answer_sources",
        "verified_rate",
        "degraded_rate",
        "sources_served",
        "environment",
        "per_request",
    )
    try:
        server.start()
        with _client(server.base_url) as client:

            def phase(name: str, *, warmup: bool = False) -> dict[str, Any]:
                started = time.monotonic()
                result = asyncio.run(
                    load_test_run(
                        server.base_url,
                        args.load_users,
                        args.load_requests,
                        "workflow",
                        warmup=warmup,
                        label=f"chaos sources-load {name}",
                    )
                )
                health = client.get("/sources/health").json()
                entry = {
                    "phase": name,
                    "at": _now(),
                    "blocking": proxy.blocking,
                    "seconds": round(time.monotonic() - started, 1),
                    **{key: result.get(key) for key in summary_keys},
                    "sources": {
                        row["source"]: {k: row.get(k) for k in ("status", "circuit", "calls", "failures")}
                        for row in health["sources"]
                        if row.get("calls")
                    },
                    "worker_pool": health.get("worker_pool"),
                    "metrics": _metric_lines(client.get("/metrics").text, prefixes),
                    "proxy_counts": json.loads(json.dumps(proxy.counts)),
                }
                report["phases"].append(entry)
                print(
                    json.dumps(
                        {
                            "phase": name,
                            "latency_ms": result.get("latency_ms"),
                            "error_rate": result.get("error_rate"),
                            "sources_served": result.get("sources_served"),
                        },
                        ensure_ascii=False,
                    )
                )
                return entry

            phase("1_live", warmup=True)
            # The last live fetches happened during phase 1: the 60 s TTL and the stale window count from here.
            fetched = time.monotonic()
            proxy.blocking = True
            phase("2a_blocked_within_ttl")  # cache hits
            time.sleep(max(0.0, fetched + 75 - time.monotonic()))
            phase("2b_blocked_after_ttl")  # last-known-good inside the stale window
            time.sleep(max(0.0, fetched + 60 + args.max_stale + 15 - time.monotonic()))
            phase("3_blocked_after_stale_window")  # snapshot policy
            proxy.blocking = False
            time.sleep(args.source_cooldown * 2 + 5)  # a failed half-open trial doubles the cool-down
            phase("4_unblocked_recovered")
    finally:
        server.stop()
        proxy.stop()
    return report


def run_sources(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    upstream_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    proxy = BlockingProxy(tuple(args.block), upstream_proxy, mode=args.block_mode)
    proxy.start()
    server = Server(args.port, _sources_env(proxy, args, out_dir), out_dir / "server-sources.log")
    price_q = "贵州茅台最新收盘价是多少？"
    macro_q = "CPI 最新数据是多少？"
    fund_q = "五粮液最新的营收和净利润增长情况如何？所在行业最近表现怎样？"
    prefixes = ("finsight_source_circuit_state", "finsight_source_pool_", "finsight_source_calls_total")
    report: dict[str, Any] = {
        "scenario": "sources",
        "blocked_hosts": list(args.block),
        "block_mode": args.block_mode,
        "upstream_proxy_used": bool(upstream_proxy),
        "settings": {
            "market_bundle_ttl_s": 60,
            "max_stale_s": args.max_stale,
            "source_cooldown_s": args.source_cooldown,
        },
        "phases": [],
    }
    try:
        server.start()
        with _client(server.base_url) as client:

            def phase(name: str, queries: list[str]) -> None:
                runs = []
                for index, query in enumerate(queries):
                    result = _ask(client, query, mode="workflow", session=f"chaos-src-{name}-{index}")
                    body = result.pop("body")
                    runs.append(
                        {
                            **result,
                            "route": body.get("route"),
                            "verified": (body.get("verification") or {}).get("passed"),
                            "degraded": body.get("degraded"),
                            "answer_excerpt": (body.get("answer") or "")[:300],
                            "limitations": body.get("limitations"),
                            "evidence": _evidence_provenance(body),
                        }
                    )
                health = client.get("/sources/health").json()
                report["phases"].append(
                    {
                        "phase": name,
                        "at": _now(),
                        "blocking": proxy.blocking,
                        "requests": runs,
                        "sources": {
                            row["source"]: {
                                k: row.get(k) for k in ("status", "circuit", "calls", "failures", "last_error")
                            }
                            for row in health["sources"]
                            if row.get("calls")
                        },
                        "worker_pool": health.get("worker_pool"),
                        "metrics": _metric_lines(client.get("/metrics").text, prefixes),
                        "proxy_counts": json.loads(json.dumps(proxy.counts)),
                    }
                )
                print(json.dumps({"phase": name, "latency_ms": [r["latency_ms"] for r in runs]}, ensure_ascii=False))

            phase("1_live", [price_q])
            # Every cache layer holds the price for 60 s from here: the agent tool cache, the structured
            # context cache and the pipeline's market-bundle TTL cache.
            fetched = time.monotonic()
            phase("1b_live_fundamentals_and_industry", [fund_q])
            proxy.blocking = True
            phase("2_blocked_within_ttl", [price_q, macro_q])
            time.sleep(max(0.0, fetched + 75 - time.monotonic()))
            phase("3_blocked_after_ttl", [price_q])
            time.sleep(max(0.0, fetched + 60 + args.max_stale + 15 - time.monotonic()))
            phase("4_blocked_after_stale_window", [price_q])
            proxy.blocking = False
            time.sleep(args.source_cooldown * 2 + 5)  # a failed half-open trial doubles the cool-down
            phase("5_unblocked_recovered", [price_q])
    finally:
        server.stop()
        proxy.stop()
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Chaos drill against a live FinSight server.")
    parser.add_argument("--scenario", choices=["llm", "llm-load", "sources", "sources-load"], required=True)
    parser.add_argument("--port", type=int, default=8821)
    parser.add_argument("--fallback-model", default="cline-pass/glm-5.3-flash")
    parser.add_argument("--llm-cooldown", type=float, default=60.0, help="FallbackLLM cool-down (s).")
    parser.add_argument("--usd-cny", type=float, default=float(os.getenv("QI_LLM_USD_CNY") or 0) or None)
    parser.add_argument("--block", nargs="*", default=list(DEFAULT_BLOCKED), help="Blocked host suffixes.")
    parser.add_argument("--block-mode", choices=["reject", "hang"], default="reject")
    parser.add_argument("--source-cooldown", type=float, default=20.0)
    parser.add_argument("--max-stale", type=float, default=90.0)
    parser.add_argument("--load-users", type=int, default=8, help="sources-load: concurrent users.")
    parser.add_argument("--load-requests", type=int, default=10, help="sources-load: questions per user and phase.")
    parser.add_argument("--load-mode", choices=["auto", "agent", "workflow"], default="auto", help="llm-load: mode.")
    parser.add_argument("--burst-start", type=float, default=15.0, help="llm-load: seconds before the 5xx burst.")
    parser.add_argument("--burst-seconds", type=float, default=0.0, help="llm-load: burst length (0 = no burst).")
    parser.add_argument("--burst-status", type=int, default=503, help="llm-load: injected HTTP status.")
    parser.add_argument(
        "--recovery-requests", type=int, default=1, help="llm-load: requests per user after the breakers' cool-down."
    )
    parser.add_argument("--out", default="outputs/chaos/chaos.json")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out_dir = out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    state = git_state(ROOT)  # before the run: a commit made during a long drill is not attributed to it
    scenarios = {"llm": run_llm, "llm-load": run_llm_load, "sources": run_sources, "sources-load": run_sources_load}
    report = scenarios[args.scenario](args, out_dir)
    report.update(
        {
            "command": "python -m scripts.chaos_drill " + " ".join(sys.argv[1:] if argv is None else argv),
            "run_at": datetime.fromtimestamp(started, UTC).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - started, 1),
            "commit": commit_label(state),
            "working_tree_clean": state["working_tree_clean"],
            "env": env_switches(),
            **({"commit_error": state["commit_error"]} if state.get("commit_error") else {}),
        }
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
