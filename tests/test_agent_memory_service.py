from __future__ import annotations

from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM, final_turn, tool_call_turn
from query_intelligence.agent.memory import (
    dialog_context_from_turns,
    history_messages,
    make_checkpointer,
)
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.state import RESET, add_or_reset, merge_dicts


def _service(llm=None, *, checkpointer=None, stub=None) -> tuple[AgentService, StubService]:
    stub = stub or StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), llm, today=lambda: date(2026, 9, 24))
    return AgentService(runtime, checkpointer=checkpointer), stub


def test_reset_reducers():
    assert add_or_reset([1, 2], [3]) == [1, 2, 3]
    assert add_or_reset([1, 2], {RESET: []}) == []
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dicts({"a": 1}, {RESET: True}) == {}


def test_memory_helpers():
    turns = [
        {"query": "q1", "route": "workflow", "answer": "a1"},
        {"query": "weather", "route": "refuse", "answer": "no"},
        {"query": "q3", "route": "agent", "answer": "a3"},
    ]

    assert dialog_context_from_turns(turns, [{"role": "user", "content": "x"}]) == [
        {"role": "user", "content": "q1"},
        {"role": "user", "content": "weather"},
        {"role": "user", "content": "q3"},
        {"role": "user", "content": "x"},
    ]
    history = history_messages(turns)
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert "q3" in history[0]["content"]


def test_sqlite_checkpointer_persists_sessions(tmp_path):
    path = tmp_path / "agent.sqlite"
    service, _ = _service(checkpointer=make_checkpointer(str(path)))
    service.chat("贵州茅台的市盈率是多少", session_id="s1")

    reopened, _ = _service(checkpointer=make_checkpointer(str(path)))

    assert [turn["query"] for turn in reopened.history("s1")] == ["贵州茅台的市盈率是多少"]


def test_turn_state_is_reset_but_history_is_kept():
    service, stub = _service()

    first = service.chat("贵州茅台的市盈率是多少", session_id="s")
    second = service.chat("今天天气怎么样", session_id="s")

    assert first["status"] == "ok" and first["tool_calls"]
    # The refusal turn must not inherit the previous turn's tools, evidence, or verification.
    assert second["route"] == "refuse"
    assert second["tool_calls"] == [] and second["evidence_sources"] == [] and second["verification"] == {}
    assert second["turn_index"] == 1
    assert [turn["query"] for turn in service.history("s")] == ["贵州茅台的市盈率是多少", "今天天气怎么样"]
    # Previous questions are passed to the NLU as dialog context for entity carry-over.
    assert stub.calls[1]["dialog_context"] == [{"role": "user", "content": "贵州茅台的市盈率是多少"}]


def test_sessions_are_isolated():
    service, stub = _service()

    service.chat("贵州茅台的市盈率是多少", session_id="a")
    service.chat("今天天气怎么样", session_id="b")

    assert stub.calls[1]["dialog_context"] == []


def test_clarification_interrupt_and_resume():
    service, stub = _service()

    pending = service.chat("这只股票能买吗", session_id="c")

    assert pending["status"] == "needs_clarification"
    assert pending["clarification"]["type"] == "clarification"
    assert "600519.SH" in pending["clarification"]["question"]
    assert service.pending_clarification("c")["original_query"] == "这只股票能买吗"

    # The stub NLU keeps flagging the original query as missing an entity, so after one clarification round
    # the agent answers with a clarification message instead of looping.
    resumed = service.resume("c", "贵州茅台")

    assert resumed["status"] == "ok"
    assert stub.calls[-1]["dialog_context"][-1] == {"role": "user", "content": "贵州茅台"}
    assert service.pending_clarification("c") is None


def test_resume_without_pending_clarification_raises():
    service, _ = _service()
    service.chat("贵州茅台的市盈率是多少", session_id="d")

    with pytest.raises(ValueError):
        service.resume("d", "anything")


def test_agent_llm_sees_previous_turns():
    llm = ScriptedLLM(
        [
            # turn 1 (workflow): one compose call
            final_turn({"answer": "PE(TTM) 为 24.6 [fundamental_600519.SH]。", "evidence_used": []}),
            # turn 2 (agent): tool call, then final answer
            tool_call_turn(("get_price_history", {"target": "600519.SH"})),
            final_turn({"answer": "最新收盘价 1409.5 元 [price_600519.SH]。", "evidence_used": ["price_600519.SH"]}),
        ]
    )
    service, _ = _service(llm)

    service.chat("贵州茅台的市盈率是多少", session_id="h", mode="workflow")
    service.chat("茅台为什么跌了", session_id="h")

    agent_request = llm.requests[1]["messages"]
    assert any("(earlier question) 贵州茅台的市盈率是多少" in message["content"] for message in agent_request)


def test_stream_emits_steps_tools_and_answer():
    service, _ = _service()

    events = list(service.stream("贵州茅台的市盈率是多少", session_id="st"))

    names = [event["event"] for event in events]
    assert names[0] == "session" and names[-1] == "done"
    assert "tool_call" in names and "tool_result" in names
    steps = [event["data"]["node"] for event in events if event["event"] == "step"]
    assert steps[:2] == ["guard_in", "execute_plan"] and steps[-1] == "finalize"
    answer = next(event for event in events if event["event"] == "answer")
    assert answer["data"]["verification"]["passed"] is True


def test_stream_reports_clarification():
    service, _ = _service()

    events = list(service.stream("这只股票能买吗", session_id="sc"))

    clarification = next(event for event in events if event["event"] == "clarification")
    assert clarification["data"]["session_id"] == "sc"
    assert "answer" not in [event["event"] for event in events]


def test_resolve_coreference_rules():
    from query_intelligence.agent.memory import resolve_coreference

    turns = [{"query": "q", "entities": [{"name": "贵州茅台", "symbol": "600519.SH"}]}]

    assert resolve_coreference("那它的市净率呢", turns) == ("那贵州茅台的市净率呢", "coreference:它->贵州茅台")
    moutai = [{"entities": [{"name": "Moutai", "symbol": "600519.SH"}]}]
    assert resolve_coreference("What about its PE?", moutai) == ("What about Moutai's PE?", "coreference:its->Moutai")
    assert resolve_coreference("Is it expensive?", moutai) == (
        "Is Moutai expensive?",
        "coreference:it->Moutai",
    )
    assert resolve_coreference("那它呢", []) is None
    two = [{"entities": [{"name": "A", "symbol": "1"}, {"name": "B", "symbol": "2"}]}]
    assert resolve_coreference("那它呢", two) is None


def test_clarification_reply_is_folded_into_the_question():
    service, stub = _service()
    service.chat("它的市盈率呢", session_id="fold")

    resumed = service.resume("fold", "贵州茅台")

    assert stub.calls[-1]["query"] == "贵州茅台的市盈率呢"
    assert "clarified:贵州茅台" in resumed["route_reasons"]
    assert resumed["status"] == "ok" and resumed["route"] != "clarify"


def test_apply_clarification_rewrites_or_prepends():
    from query_intelligence.agent.memory import apply_clarification

    assert apply_clarification("它的市盈率呢", "宁德时代")[0] == "宁德时代的市盈率呢"
    assert apply_clarification("What is its PE?", "CATL")[0] == "What is CATL's PE?"
    assert apply_clarification("能买吗", "宁德时代")[0] == "宁德时代能买吗"
    assert apply_clarification("buy now?", "BYD") == ("BYD buy now?", "clarified:BYD")


def test_session_lock_map_is_bounded(monkeypatch):
    import query_intelligence.agent.service as service_module

    monkeypatch.setattr(service_module, "MAX_SESSION_LOCKS", 3)
    service, _ = _service()
    held = service._lock("keep")
    held.acquire()
    try:
        for index in range(10):
            service._lock(f"s{index}")
        assert len(service._locks) <= 4 and "keep" in service._locks  # a held lock is never evicted
    finally:
        held.release()


def test_plural_and_skip_back_coreference():
    from query_intelligence.agent.memory import resolve_coreference

    turns = [
        {"query": "茅台市盈率", "entities": [{"name": "贵州茅台", "symbol": "600519.SH"}]},
        {"query": "五粮液市盈率", "entities": [{"name": "五粮液", "symbol": "000858.SZ"}]},
        {"query": "最新CPI", "entities": []},
    ]

    assert resolve_coreference("这两家哪个估值更高", turns)[0] == "贵州茅台和五粮液哪个估值更高"
    assert resolve_coreference("Compare both on P/B", turns)[0] == "Compare 贵州茅台 and 五粮液 on P/B"
    # the macro turn in between has no entity: "它" resolves to the last named company
    assert resolve_coreference("它的市净率呢", turns)[0] == "五粮液的市净率呢"
    assert resolve_coreference("这两家怎么样", turns[:1]) is None


def test_session_memory_carries_constraints_and_holdings():
    from query_intelligence.agent.memory import session_memory

    turns = [
        {"query": "我是保守型投资者，只看A股，我持有招商银行", "entities": []},
        {"query": "茅台市盈率", "entities": [{"name": "贵州茅台", "symbol": "600519.SH"}]},
    ]

    memory = session_memory(turns, "它能长期持有吗")

    assert memory["user_constraints"] == ["risk:conservative", "scope:a_shares_only", "horizon:long"]
    assert memory["stated_holdings"] == ["招商银行"]
    assert memory["recent_targets"] == [{"name": "贵州茅台", "symbol": "600519.SH"}]


def test_agent_prompt_includes_session_memory():
    llm = ScriptedLLM(
        [
            final_turn({"answer": "PE(TTM) 为 24.6 [fundamental_600519.SH]。", "evidence_used": []}),
            final_turn({"answer": "最新收盘价 1409.5 元 [price_600519.SH]。", "evidence_used": ["price_600519.SH"]}),
        ]
    )
    service, _ = _service(llm)

    service.chat("我是保守型投资者，贵州茅台的市盈率是多少", session_id="mem", mode="workflow")
    service.chat("茅台为什么跌了", session_id="mem", mode="agent")

    user_messages = [
        message["content"]
        for request in llm.requests
        for message in request["messages"]
        if message["role"] == "user" and "Session memory" in str(message["content"])
    ]
    assert user_messages and "risk:conservative" in user_messages[0]
