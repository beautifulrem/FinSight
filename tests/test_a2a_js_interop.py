"""A2A interop with another framework: the official JavaScript SDK client (@a2a-js/sdk) against FinSight.

``tools/a2a-js-client/interop.mjs`` runs discovery, SendMessage, GetTask, input-required + reply,
SendStreamingMessage, SubscribeToTask (running and finished tasks) and CancelTask (running and finished
tasks) against a real HTTP server: the FastAPI app with the stub agent, served by uvicorn on a free port.
Skipped when ``node`` or ``tools/a2a-js-client/node_modules`` is missing (``cd tools/a2a-js-client && npm ci``).
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from agent_fakes import StubService, build_fake_registry

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app

CLIENT_DIR = Path(__file__).resolve().parents[1] / "tools" / "a2a-js-client"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    NODE is None or not (CLIENT_DIR / "node_modules" / "@a2a-js" / "sdk").is_dir(),
    reason="node or tools/a2a-js-client/node_modules missing (run: cd tools/a2a-js-client && npm ci)",
)


class SlowAgentService(AgentService):
    """Paces streamed events so a task is still running when the client subscribes to or cancels it
    (the stub agent otherwise finishes in milliseconds)."""

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        for event in super().stream(*args, **kwargs):
            time.sleep(0.15)
            yield event


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    import uvicorn

    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=SlowAgentService(runtime))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws="none"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_js_sdk_client_interoperates_with_finsight(server_url, tmp_path, monkeypatch):
    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    summary_path = tmp_path / "summary.json"
    completed = subprocess.run(
        [NODE, "interop.mjs", "--url", server_url, "--json", str(summary_path), "--settle-ms", "2500"],
        cwd=CLIENT_DIR,
        capture_output=True,
        text=True,
        timeout=180,
        env={"PATH": str(Path(NODE).parent), "NO_PROXY": "*"},  # no proxy variables: talk to 127.0.0.1 directly
    )
    assert summary_path.exists(), completed.stdout + completed.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed = [check for check in summary["checks"] if not check["ok"]]
    assert completed.returncode == 0 and not failed, (failed, completed.stdout[-3000:], completed.stderr[-2000:])

    assert summary["send"]["evidence_used"] == ["fundamental_600519.SH"]
    assert summary["resumed"]["task_id"] == summary["clarification"]["task_id"]
    assert summary["resubscribe_completed"]["name"] == "JsonRpcUnsupportedOperationError"
    assert summary["cancel"] == {"state": "TASK_STATE_CANCELED", "after": "TASK_STATE_CANCELED", "artifacts": 0}
    assert summary["cancel_completed"]["name"] == "JsonRpcTaskNotCancelableError"
    methods = {entry["request"] for entry in summary["wire"]}
    assert {
        "GET /.well-known/agent-card.json",
        "POST SendMessage /a2a",
        "POST SendStreamingMessage /a2a",
        "POST GetTask /a2a",
        "POST SubscribeToTask /a2a",
        "POST CancelTask /a2a",
    } <= methods
    assert {entry["a2a_version"] for entry in summary["wire"] if entry["request"].startswith("POST")} == {"1.0"}
