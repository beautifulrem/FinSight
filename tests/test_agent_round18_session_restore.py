"""Round-18 rule N15 (round-14 review), server side: a restored session shows the question as the user typed it.

Every stage reads a traditional-character question in simplified characters (round 17, M9), and the session's turn
record kept only that reading, so a reload showed "长江电力最新收盘价" for "長江電力最新收盤價". The turn record
now also keeps ``typed_query``; the front end shows it (and re-renders the full answers it kept for the session).
"""

from __future__ import annotations

from datetime import date

from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app


def test_the_restored_turn_keeps_the_question_as_typed():
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    client = TestClient(
        create_app(
            service=stub, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime, trace_sinks=[])
        )
    )
    typed = "貴州茅臺的市盈率是多少"
    session = client.post("/agent/chat", json={"query": typed}).json()["session_id"]
    client.post("/agent/chat", json={"query": "它的市净率呢", "session_id": session})

    turns = client.get(f"/agent/sessions/{session}").json()["turns"]
    assert [turn["typed_query"] for turn in turns] == [typed, "它的市净率呢"]
    assert turns[0]["query"] == "贵州茅台的市盈率是多少"  # every stage still reads the simplified form
