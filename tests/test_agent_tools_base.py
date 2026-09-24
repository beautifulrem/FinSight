from __future__ import annotations

import threading
import time

import pytest
from pydantic import BaseModel, Field

from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore, extract_numbers, safe_evidence_id
from query_intelligence.agent.tools import (
    ToolFailure,
    ToolOutput,
    ToolRegistry,
    ToolSpec,
    TransientToolError,
)


class EchoInput(BaseModel):
    symbol: str = Field(min_length=1)
    limit: int = Field(default=3, ge=1, le=10)


def _registry() -> ToolRegistry:
    return ToolRegistry(sleep=lambda _seconds: None)


def _echo_spec(handler, **overrides) -> ToolSpec:
    return ToolSpec(name="echo", description="Echo a symbol.", input_model=EchoInput, handler=handler, **overrides)


def test_run_success_returns_data_and_evidence():
    registry = _registry()
    evidence = AgentEvidence(evidence_id="price_600519.SH", kind="structured", source_type="market_api")
    registry.register(_echo_spec(lambda args: ToolOutput(data={"symbol": args.symbol}, evidence=[evidence])))

    result = registry.run("echo", {"symbol": "600519.SH"})

    assert result.ok
    assert result.data == {"symbol": "600519.SH"}
    assert result.arguments == {"symbol": "600519.SH", "limit": 3}
    assert [item.evidence_id for item in result.evidence] == ["price_600519.SH"]
    assert result.attempts == 1
    assert result.observation()["evidence"][0]["evidence_id"] == "price_600519.SH"


def test_run_accepts_json_string_and_repairs_malformed_json():
    registry = _registry()
    registry.register(_echo_spec(lambda args: ToolOutput(data=args.symbol)))

    assert registry.run("echo", '{"symbol": "000001.SZ"}').data == "000001.SZ"
    # Missing closing quote on the key, as seen with some tool-calling providers.
    assert registry.run("echo", '{"symbol: "000001.SZ"}').ok


def test_run_rejects_invalid_arguments_without_calling_handler():
    calls = []
    registry = _registry()
    registry.register(_echo_spec(lambda args: calls.append(args) or ToolOutput(data=None)))

    result = registry.run("echo", {"symbol": "", "limit": 99})

    assert not result.ok
    assert result.error.code == "invalid_arguments"
    assert "symbol" in result.error.message and "limit" in result.error.message
    assert calls == []


def test_run_unknown_tool():
    result = _registry().run("missing", {})
    assert not result.ok and result.error.code == "unknown_tool"


def test_run_retries_transient_errors_then_succeeds():
    attempts = {"count": 0}

    def flaky(args):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TransientToolError("upstream 502")
        return ToolOutput(data="ok")

    registry = _registry()
    registry.register(_echo_spec(flaky, max_retries=2))

    result = registry.run("echo", {"symbol": "x"})

    assert result.ok and result.attempts == 3


def test_run_gives_up_after_max_retries():
    registry = _registry()
    registry.register(_echo_spec(lambda args: (_ for _ in ()).throw(ConnectionError("reset")), max_retries=1))

    result = registry.run("echo", {"symbol": "x"})

    assert not result.ok
    assert result.error.code == "upstream_error" and result.error.retryable
    assert result.attempts == 2


def test_run_does_not_retry_expected_failures_or_bugs():
    registry = _registry()
    registry.register(_echo_spec(lambda args: (_ for _ in ()).throw(ToolFailure("not_found", "no such symbol"))))
    registry.register(
        ToolSpec(
            name="buggy",
            description="Raises a programming error.",
            input_model=EchoInput,
            handler=lambda args: 1 / 0,
            max_retries=3,
        )
    )

    not_found = registry.run("echo", {"symbol": "x"})
    buggy = registry.run("buggy", {"symbol": "x"})

    assert not_found.error.code == "not_found" and not_found.attempts == 1
    assert buggy.error.code == "internal" and buggy.attempts == 1
    assert "ZeroDivisionError" in buggy.error.message


def test_run_times_out_slow_handlers():
    release = threading.Event()

    def slow(args):
        release.wait(2)
        return ToolOutput(data="late")

    registry = _registry()
    registry.register(_echo_spec(slow, timeout_s=0.05, max_retries=0))

    started = time.perf_counter()
    result = registry.run("echo", {"symbol": "x"})
    release.set()

    assert not result.ok and result.error.code == "timeout"
    assert time.perf_counter() - started < 1.0


def test_run_caches_successful_results_only():
    calls = {"count": 0}

    def counted(args):
        calls["count"] += 1
        return ToolOutput(data=calls["count"])

    registry = _registry()
    registry.register(_echo_spec(counted, cache_ttl_s=60))

    first = registry.run("echo", {"symbol": "x"})
    second = registry.run("echo", {"symbol": "x", "limit": 3})
    other = registry.run("echo", {"symbol": "y"})

    assert first.data == 1 and not first.cached
    assert second.data == 1 and second.cached
    assert other.data == 2


def test_openai_schema_uses_input_model():
    registry = _registry()
    registry.register(_echo_spec(lambda args: ToolOutput(data=None)))

    [schema] = registry.to_openai_tools()

    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "echo"
    assert function["parameters"]["required"] == ["symbol"]
    assert function["parameters"]["properties"]["limit"]["maximum"] == 10


def test_duplicate_registration_is_rejected():
    registry = _registry()
    registry.register(_echo_spec(lambda args: ToolOutput(data=None)))
    with pytest.raises(ValueError):
        registry.register(_echo_spec(lambda args: ToolOutput(data=None)))


def test_evidence_from_document_and_structured():
    document = AgentEvidence.from_document(
        {
            "evidence_id": "news_abc",
            "source_type": "news",
            "title": "贵州茅台发布公告",
            "summary": "营收 1,234.5 亿元，同比增长 15.2%",
            "publish_time": "2026-09-01",
            "rank_score": 0.8,
        },
        produced_by="search_news",
    )
    structured = AgentEvidence.from_structured(
        {
            "evidence_id": "price_600519.SH",
            "source_type": "market_api",
            "payload": {"close": 1500.5, "pct_change_1d": "-1.2%"},
        }
    )

    assert document.kind == "document" and document.as_of == "2026-09-01"
    assert document.payload == {"rank_score": 0.8}
    assert 1234.5 in document.numbers() and 15.2 in document.numbers()
    assert structured.numbers() == [1500.5, -1.2]
    assert structured.prompt_view()["payload"] == {"close": 1500.5, "pct_change_1d": "-1.2%"}


def test_prompt_view_drops_raw_history():
    item = AgentEvidence(
        evidence_id="price_x",
        kind="structured",
        source_type="market_api",
        payload={"history": [1, 2, 3], "closes": list(range(20)), "close": 19},
    )

    view = item.prompt_view()["payload"]

    assert "history" not in view
    assert view["closes"] == [15, 16, 17, 18, 19] and view["closes_count"] == 20


def test_evidence_store_dedupes_identical_and_renames_conflicts():
    store = EvidenceStore()
    first = AgentEvidence(evidence_id="price_x", kind="structured", source_type="market_api", payload={"close": 1})
    same = first.model_copy(update={"produced_by": "other_tool"})
    conflict = first.model_copy(update={"payload": {"close": 2}})

    assert store.add(first).evidence_id == "price_x"
    assert store.add(same).evidence_id == "price_x"
    assert store.add(conflict).evidence_id == "price_x_2"
    assert store.ids() == ["price_x", "price_x_2"] and "price_x_2" in store and len(store) == 2


def test_safe_evidence_id_and_number_extraction():
    assert safe_evidence_id("news/贵州茅台 2026") == "news_2026"
    assert safe_evidence_id("///") == "evidence"
    assert extract_numbers("MA5 为 12.3，RSI14=45, 成交 1,200 手") == [12.3, 45.0, 1200.0]
    assert extract_numbers("涨幅 -3.5%，600519.SH 收盘 1500") == [-3.5, 1500.0]
