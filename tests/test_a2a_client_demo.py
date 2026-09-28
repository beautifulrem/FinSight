"""The committed A2A client demo (scripts/a2a_client_demo.py) runs against the app in-process."""

from __future__ import annotations

from datetime import date

import anyio
import httpx
from agent_fakes import StubService, build_fake_registry

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app
from scripts.a2a_client_demo import run_demo


def _app(monkeypatch, **kwargs):
    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    monkeypatch.delenv("QI_AGENT_CHECKPOINT_DB", raising=False)
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    return create_app(
        service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime), **kwargs
    )


def _run(app, **kwargs) -> tuple[dict, list[str]]:
    lines: list[str] = []

    async def go() -> dict:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=60) as http:
            return await run_demo("http://testserver", http_client=http, out=lines.append, **kwargs)

    return anyio.run(go), lines


def test_demo_covers_card_send_clarification_and_streaming(monkeypatch):
    summary, lines = _run(_app(monkeypatch), stream_question="贵州茅台的市净率是多少")

    assert summary["card"]["name"] == "FinSight" and "equity_research" in summary["card"]["skills"]

    answer = summary["answer"]
    assert answer["state"] == "TASK_STATE_COMPLETED"
    assert answer["evidence_used"] == ["fundamental_600519.SH"] and answer["trace_id"]
    assert "[fundamental_600519.SH]" in answer["answer"]

    assert summary["clarification"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert "哪只" in summary["clarification"]["status_message"]
    resumed = summary["resumed"]
    assert resumed["state"] == "TASK_STATE_COMPLETED"
    assert resumed["task_id"] == summary["clarification"]["task_id"]
    assert "600519" in resumed["answer"]

    stream = summary["stream"]
    assert stream["kinds"][0] == "task" and stream["final_state"] == "TASK_STATE_COMPLETED"
    assert stream["kinds"].count("artifact_update") == 2
    # Progress: one working status update per graph node / tool call before the answer.
    assert stream["kinds"].count("status_update") >= 5
    assert any("calling get_fundamentals" in line for line in lines)
    assert any("evidence verification" in line for line in lines)


def test_demo_sends_the_api_key_and_tasks_are_scoped_to_it(monkeypatch):
    from query_intelligence.api.security import SecuritySettings

    app = _app(monkeypatch, security=SecuritySettings(api_keys=("key-a", "key-b")))
    summary, _ = _run(app, api_key="key-a")
    assert summary["answer"]["state"] == "TASK_STATE_COMPLETED"

    async def get_task(key: str) -> dict:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
            response = await http.post(
                "/a2a",
                headers={"X-API-Key": key, "A2A-Version": "1.0"},
                json={"jsonrpc": "2.0", "id": 1, "method": "GetTask", "params": {"id": summary["answer"]["task_id"]}},
            )
            return response.json()

    assert anyio.run(get_task, "key-a")["result"]["id"] == summary["answer"]["task_id"]
    other = anyio.run(get_task, "key-b")
    assert "error" in other and "result" not in other  # another caller cannot read the task
