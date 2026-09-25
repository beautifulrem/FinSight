"""Sessions shared across replicas through the Postgres checkpointer.

Opt-in integration test: needs a Postgres server, e.g.

    docker run -d -e POSTGRES_PASSWORD=finsight -e POSTGRES_DB=finsight -p 55432:5432 postgres:16-alpine
    export QI_TEST_POSTGRES_DSN=postgresql://postgres:finsight@127.0.0.1:55432/finsight
    pytest tests/test_agent_checkpoint_postgres.py
"""

from __future__ import annotations

import os
import uuid
from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.memory import make_checkpointer
from query_intelligence.agent.service import AgentService

DSN = os.getenv("QI_TEST_POSTGRES_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="set QI_TEST_POSTGRES_DSN to run the Postgres checkpointer test")


def _replica() -> AgentService:
    runtime = AgentRuntime(StubService(), build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    return AgentService(runtime, checkpointer=make_checkpointer(DSN), trace_sinks=[])


def test_two_replicas_share_session_memory_and_clarifications():
    replica_a, replica_b = _replica(), _replica()
    session = uuid.uuid4().hex

    first = replica_a.chat("贵州茅台的市盈率是多少", session_id=session, mode="workflow")
    follow_up = replica_b.chat("它的市净率呢", session_id=session, mode="workflow")

    assert first["status"] == "ok" and follow_up["status"] == "ok"
    assert any("coreference" in reason for reason in follow_up["route_reasons"])
    assert [turn["query"] for turn in replica_a.history(session)] == ["贵州茅台的市盈率是多少", "它的市净率呢"]

    pending_session = uuid.uuid4().hex
    assert replica_a.chat("它的市盈率呢", session_id=pending_session)["status"] == "needs_clarification"
    assert replica_b.pending_clarification(pending_session) is not None
    resumed = replica_b.resume(pending_session, "贵州茅台")
    assert resumed["status"] == "ok" and "clarified:贵州茅台" in resumed["route_reasons"]
