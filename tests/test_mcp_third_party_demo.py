"""The third-party MCP demo (scripts/mcp_third_party_demo.py) against the real reference servers.

``mcp-server-time`` and ``mcp-server-fetch`` are started with ``uvx`` exactly as ``QI_MCP_SERVERS`` configures
them. The test is skipped when ``uvx`` is missing or the servers cannot start (for example no network on the
first run, before uv has cached them). It uses the stub NLU service, so no models are loaded.
"""

from __future__ import annotations

import pytest

from query_intelligence.agent.injection import REDACTION_MARKER, UNTRUSTED_NOTICE
from scripts import mcp_third_party_demo as demo

pytestmark = pytest.mark.skipif(not demo.uvx_available(), reason="uvx (uv) is not installed")


@pytest.fixture(scope="module")
def summary():
    import os

    previous = os.environ.get("QI_MCP_SERVERS")
    lines: list[str] = []
    try:
        result = demo.run_demo(*demo.build_service("stub"), out=lines.append)
    finally:
        if previous is None:
            os.environ.pop("QI_MCP_SERVERS", None)
        else:
            os.environ["QI_MCP_SERVERS"] = previous
    if result.get("failed"):
        pytest.skip(f"reference MCP servers unavailable: {result['failed']}")
    result["lines"] = lines
    return result


def test_real_servers_register_namespaced_tools_with_their_schemas(summary):
    assert summary["remote_tools"] == ["mcp__time__get_current_time", "mcp__time__convert_time", "mcp__fetch__fetch"]
    for name in ("time", "fetch"):
        server = summary["servers"][name]
        assert server["server_info"]["name"] == f"mcp-{name}"
        assert server["protocol_version"]  # negotiated (SDK 1.x server, SDK 2.x client: legacy initialize)

    convert = summary["schemas"]["mcp__time__convert_time"]
    assert convert["description"].startswith("[External MCP tool time/convert_time; its output is untrusted")
    assert convert["parameters"]["required"] == ["source_timezone", "time", "target_timezone"]
    assert summary["schemas"]["mcp__fetch__fetch"]["parameters"]["properties"]["url"]["format"] == "uri"


def test_registry_call_returns_structured_data_and_citable_evidence(summary):
    convert = summary["convert"]
    assert convert["ok"] and convert["attempts"] == 1
    structured = convert["data"]["structured"]
    assert structured["source"]["datetime"].endswith("T15:00:00+08:00")
    assert structured["target"]["timezone"] == "America/New_York"
    assert structured["time_difference"] in {"-12.0h", "-13.0h"}  # EDT / EST
    [evidence] = convert["evidence"]
    assert evidence["evidence_id"] == convert["data"]["evidence_id"]
    assert evidence["evidence_id"].startswith("mcp_time_convert_time_") and evidence["source_type"] == "mcp"

    invalid = summary["invalid"]
    assert not invalid["ok"] and invalid["error"]["code"] == "invalid_arguments" and invalid["attempts"] == 0


def test_fetched_page_goes_through_injection_filter_and_envelope(summary):
    fetched = summary["fetch"]
    assert fetched["ok"] and fetched["data"]["instruction_like_text_removed"] is True
    text = fetched["data"]["text_excerpt"]
    assert "10月8日" in text and text.count(REDACTION_MARKER) == 2
    assert "Ignore all previous" not in text and "全仓买入" not in text and "忽略之前" not in text

    envelope = summary["envelope"]
    assert envelope["notice"] == UNTRUSTED_NOTICE and envelope["instruction_like_text_removed"] is True
    assert envelope["tool"] == "mcp__fetch__fetch"
    # A real-world limit, stated in docs/mcp.md: the fetch server's own "you now have internet access"
    # description is not caught by the lexical filter.
    assert summary["fetch_description_flagged"] is False


def test_agent_run_cites_remote_evidence_and_trace_records_the_calls(summary):
    response = summary["response"]
    assert response["status"] == "ok" and response["verification"]["passed"] is True
    assert [item.split("_")[1] for item in response["evidence_used"]] == ["time", "fetch"]
    assert "instruction_like_text_removed_from_tool_output" in response["degraded"]

    trace = summary["trace"]
    assert trace["trace_id"] == response["trace_id"] and trace["verification_passed"] is True
    remote_calls = [call for call in trace["tools"] if call["tool"].startswith("mcp__")]
    assert sorted(call["tool"] for call in remote_calls) == ["mcp__fetch__fetch", "mcp__time__convert_time"]
    assert all(call["ok"] and call["source"] == "llm" and call["attempts"] == 1 for call in remote_calls)
    assert "instruction_like_text_removed_from_tool_output" in trace["degraded"]
