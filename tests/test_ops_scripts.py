"""Offline tests for the ops scripts: load-test cost aggregation and the chaos-drill proxies."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import httpx
import pytest

from scripts.chaos_drill import INVALID_MODEL, BlockingProxy, LLMFaultProxy
from scripts.load_test import cost_summary, percentile


def test_percentile_is_nearest_rank():
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 0.5) == 10.0
    assert percentile(values, 0.95) == 19.0
    assert percentile(values, 0.99) == 20.0


def test_cost_summary_converts_gateway_usd_to_cny_and_projects_monthly_cost():
    results = [
        {"status": 200, "reported_cost_usd": 0.002, "llm_calls": 3, "prompt_tokens": 9000, "completion_tokens": 500},
        {"status": 200, "reported_cost_usd": 0.001, "llm_calls": 2, "prompt_tokens": 6000, "completion_tokens": 300},
        {"status": 200, "reported_cost_usd": 0.0, "llm_calls": 0},  # refusal: no LLM call, no cost
        {"status": 504},  # not counted
    ]
    summary = cost_summary(results, usd_cny=7.0, questions_per_day=1000)
    assert summary["requests_counted"] == 3 and summary["requests_with_llm"] == 2
    assert summary["total_usd"] == pytest.approx(0.003)
    assert summary["total_cny"] == pytest.approx(0.021)
    assert summary["per_1k_questions_cny"] == pytest.approx(7.0)
    assert summary["monthly_cny_at_questions_per_day"] == pytest.approx(210.0)
    assert summary["tokens"] == {"prompt": 15000, "completion": 800, "cache_hit": 0}


def test_cost_summary_without_rate_keeps_usd_and_leaves_cny_unknown():
    summary = cost_summary([{"status": 200, "reported_cost_usd": 0.004, "llm_calls": 1}], None, 100)
    assert summary["total_usd"] == 0.004 and summary["total_cny"] is None
    assert summary["per_1k_questions_usd"] == 4.0
    assert cost_summary([{"status": 200}], 7.0, 100) is None  # workflow mode: no LLM block


class _Upstream(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict]] = []

    def log_message(self, *args):
        return

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Upstream.seen.append(body)
        status = 400 if body["model"] == INVALID_MODEL else 200
        payload = json.dumps({"model": body["model"], "ok": status == 200}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        payload = b"hello"
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_llm_fault_proxy_rewrites_only_the_failing_model(upstream):
    proxy = LLMFaultProxy(upstream, {"primary"})
    proxy.start()
    try:
        with httpx.Client(trust_env=False) as client:
            assert client.post(f"{proxy.url}/chat/completions", json={"model": "primary"}).status_code == 200
            proxy.fault_on = True
            assert client.post(f"{proxy.url}/chat/completions", json={"model": "primary"}).status_code == 400
            assert client.post(f"{proxy.url}/chat/completions", json={"model": "backup"}).status_code == 200
    finally:
        proxy.stop()
    assert [entry["fault_applied"] for entry in proxy.log] == [False, True, False]
    assert [entry["status"] for entry in proxy.log] == [200, 400, 200]
    assert "authorization" not in json.dumps(proxy.log).lower()


def test_blocking_proxy_refuses_blocked_hosts_and_forwards_the_rest(upstream):
    proxy = BlockingProxy(("blocked.example",), upstream_proxy=None)
    proxy.start()
    try:
        port = int(upstream.rsplit(":", 1)[1])
        with httpx.Client(proxy=proxy.url, trust_env=False, timeout=5) as client:
            assert client.get(f"http://127.0.0.1:{port}/x").text == "hello"
        proxy.blocking = True
        with socket.create_connection(("127.0.0.1", proxy.port), timeout=5) as sock:
            sock.sendall(b"CONNECT hq.blocked.example:443 HTTP/1.1\r\nHost: hq.blocked.example:443\r\n\r\n")
            assert sock.recv(64).startswith(b"HTTP/1.1 403")
        with httpx.Client(proxy=proxy.url, trust_env=False, timeout=5) as client:
            assert client.get(f"http://127.0.0.1:{port}/y").text == "hello"  # other hosts still pass
    finally:
        proxy.stop()
    assert proxy.counts["hq.blocked.example"] == {"blocked": 1}
    assert proxy.counts["127.0.0.1"]["forwarded"] == 2


def test_load_test_stream_records_time_to_first_answer_token():
    import asyncio
    import json as _json
    import time as _time

    import httpx

    from scripts.load_test import _streamed

    answer = {"status": "ok", "route": "agent", "llm": {"calls": 2, "usage": {"prompt_tokens": 10}}}
    sse = (
        'event: node_start\ndata: {"node": "agent_llm"}\n\n'
        'event: answer_delta\ndata: {"text": "茅台"}\n\n'
        f"event: answer\ndata: {_json.dumps(answer)}\n\n"
    )

    async def go():
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, text=sse))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await _streamed(client, {"query": "q"}, _time.perf_counter())

    status, body, ttft_ms = asyncio.run(go())
    assert status == 200 and body["route"] == "agent"
    assert ttft_ms is not None and ttft_ms >= 0


def test_load_test_records_which_source_and_fallback_served_the_evidence():
    from scripts.load_test import _sources_served

    body = {
        "evidence_sources": [
            {"payload": {"provenance": {"source": "sina.kline", "mode": "last_known_good"}}},
            {"payload": {"provenance": {"source": "offline_snapshot", "mode": "snapshot"}}},
            {"kind": "structured", "source_name": "seed", "payload": {"close": 1}},
            {"kind": "document", "payload": None},
        ]
    }
    assert _sources_served(body) == ["sina.kline/last_known_good", "offline_snapshot/snapshot", "seed/unlabelled"]
    assert _sources_served({}) == []
