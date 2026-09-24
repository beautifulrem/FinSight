from __future__ import annotations

import json
from datetime import date

from agent_fakes import StubService, build_fake_registry
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.llm import ScriptedLLM, Usage, final_turn, tool_call_turn
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.tracing import JsonFileTraceSink, OTelTraceSink, build_trace, sinks_from_env


class ListSink:
    def __init__(self) -> None:
        self.traces: list[dict] = []

    def emit(self, trace):
        self.traces.append(trace)


class BrokenSink:
    def emit(self, trace):
        raise RuntimeError("sink down")


def _agent_service(sinks, llm=None):
    runtime = AgentRuntime(StubService(), build_fake_registry(), llm, today=lambda: date(2026, 9, 24))
    return AgentService(runtime, trace_sinks=sinks)


def _scripted():
    return ScriptedLLM(
        [
            tool_call_turn(
                ("get_price_history", {"target": "600519.SH"}), usage=Usage(prompt_tokens=300, completion_tokens=20)
            ),
            final_turn(
                {"answer": "最新收盘价 1409.5 元 [price_600519.SH]。", "evidence_used": ["price_600519.SH"]},
                usage=Usage(prompt_tokens=500, completion_tokens=60, prompt_cache_hit_tokens=200),
            ),
        ]
    )


def test_trace_contains_nodes_tools_and_llm_calls():
    sink = ListSink()
    service = _agent_service([sink], _scripted())

    response = service.chat("茅台为什么跌了", session_id="t1")

    [trace] = sink.traces
    assert trace["trace_id"] == response["trace_id"] == response["run_id"]
    assert trace["session_id"] == "t1" and trace["route"] == "agent"
    assert trace["nodes"][0]["node"] == "guard_in"
    assert trace["nodes"][-1]["node"] == "finalize"
    assert [tool["tool"] for tool in trace["tools"]] == ["get_price_history"]
    assert [call["node"] for call in trace["llm_calls"]] == ["agent_llm", "agent_llm"]
    assert trace["llm_calls"][0]["tool_calls"] == ["get_price_history"]
    assert trace["llm_calls"][1]["prompt_cache_hit_tokens"] == 200
    assert trace["usage"]["total_tokens"] == 880
    assert trace["verification_passed"] is True
    assert trace["duration_ms"] >= 0


def test_json_file_sink_writes_trace(tmp_path):
    service = _agent_service([JsonFileTraceSink(tmp_path)])

    response = service.chat("贵州茅台的市盈率是多少")

    [path] = list(tmp_path.glob("*/*.json"))
    assert path.stem == response["trace_id"]
    assert json.loads(path.read_text(encoding="utf-8"))["route"] == "workflow"


def test_stream_also_emits_trace():
    sink = ListSink()
    service = _agent_service([sink])

    events = list(service.stream("贵州茅台的市盈率是多少"))

    answer = next(event for event in events if event["event"] == "answer")
    assert sink.traces[0]["trace_id"] == answer["data"]["trace_id"]


def test_broken_sink_does_not_break_answers():
    sink = ListSink()
    service = _agent_service([BrokenSink(), sink])

    assert service.chat("贵州茅台的市盈率是多少")["status"] == "ok"
    assert len(sink.traces) == 1


def test_clarification_turns_are_not_traced_until_answered():
    sink = ListSink()
    service = _agent_service([sink])

    service.chat("这只股票能买吗", session_id="c")
    assert sink.traces == []
    service.resume("c", "贵州茅台")
    assert len(sink.traces) == 1


def test_otel_sink_exports_spans_with_recorded_timing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    service = _agent_service([OTelTraceSink(provider)], _scripted())

    response = service.chat("茅台为什么跌了")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    root = spans["finsight.agent.run"]
    assert root.attributes["finsight.trace_id"] == response["trace_id"]
    assert root.attributes["gen_ai.usage.input_tokens"] == 800
    assert "node.guard_in" in spans and "tool.get_price_history" in spans and "llm.agent_llm" in spans
    tool_span = spans["tool.get_price_history"]
    assert tool_span.parent.span_id == root.context.span_id
    assert tool_span.attributes["finsight.tool.ok"] is True
    assert root.start_time <= spans["node.guard_in"].start_time
    assert root.end_time >= root.start_time


def test_sinks_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    monkeypatch.delenv("QI_AGENT_OTEL", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert sinks_from_env() == []

    monkeypatch.setenv("QI_AGENT_TRACE_DIR", str(tmp_path))
    [sink] = sinks_from_env()
    assert isinstance(sink, JsonFileTraceSink) and sink.directory == tmp_path

    monkeypatch.setenv("QI_AGENT_OTEL", "1")
    assert any(isinstance(item, OTelTraceSink) for item in sinks_from_env())


def test_build_trace_handles_minimal_result():
    trace = build_trace({"run_id": "r", "spans": []})

    assert trace["trace_id"] == "r" and trace["started_at"] is None and trace["duration_ms"] is None
