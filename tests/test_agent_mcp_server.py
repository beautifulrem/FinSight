from __future__ import annotations

import anyio
import pytest
from mcp import Client
from pydantic import BaseModel, Field

from query_intelligence.agent.evidence import AgentEvidence
from query_intelligence.agent.mcp_server import build_mcp_server
from query_intelligence.agent.tools import (
    DEFAULT_TOOL_NAMES,
    ToolFailure,
    ToolOutput,
    ToolRegistry,
    ToolSpec,
    build_registry_for_service,
)


class QuoteInput(BaseModel):
    target: str = Field(min_length=1)


def _fake_registry() -> ToolRegistry:
    def quote(args: QuoteInput) -> ToolOutput:
        if args.target == "missing":
            raise ToolFailure("not_found", "no such security")
        evidence = AgentEvidence(
            evidence_id=f"price_{args.target}", kind="structured", source_type="market_api", payload={"close": 10.5}
        )
        return ToolOutput(data={"target": args.target, "close": 10.5}, evidence=[evidence])

    registry = ToolRegistry()
    registry.register(ToolSpec(name="get_quote", description="Latest quote.", input_model=QuoteInput, handler=quote))
    return registry


async def _with_client(registry: ToolRegistry, callback):
    async with Client(build_mcp_server(registry), raise_exceptions=True) as client:
        return await callback(client)


def test_mcp_lists_tools_with_pydantic_schemas():
    async def run(client):
        return await client.list_tools()

    result = anyio.run(_with_client, _fake_registry(), run)

    [tool] = result.tools
    assert tool.name == "get_quote"
    assert tool.input_schema["required"] == ["target"]
    assert tool.annotations.read_only_hint is True


def test_mcp_call_returns_structured_evidence():
    async def run(client):
        return await client.call_tool("get_quote", {"target": "600519.SH"})

    result = anyio.run(_with_client, _fake_registry(), run)

    assert result.is_error is False
    assert result.structured_content["ok"] is True
    assert result.structured_content["data"] == {"target": "600519.SH", "close": 10.5}
    assert result.structured_content["evidence"][0]["evidence_id"] == "price_600519.SH"


def test_mcp_call_errors_are_flagged_not_raised():
    async def run(client):
        missing = await client.call_tool("get_quote", {"target": "missing"})
        invalid = await client.call_tool("get_quote", {"target": ""})
        unknown = await client.call_tool("nope", {})
        return missing, invalid, unknown

    missing, invalid, unknown = anyio.run(_with_client, _fake_registry(), run)

    assert missing.is_error and missing.structured_content["error"]["code"] == "not_found"
    assert invalid.is_error and invalid.structured_content["error"]["code"] == "invalid_arguments"
    assert unknown.is_error and unknown.structured_content["error"]["code"] == "unknown_tool"


@pytest.mark.parametrize("tool_name", ["resolve_entity", "get_fundamentals"])
def test_mcp_serves_default_registry(offline_service, tool_name):
    registry = build_registry_for_service(offline_service)
    arguments = {"text": "贵州茅台"} if tool_name == "resolve_entity" else {"target": "600519.SH"}

    async def run(client):
        tools = await client.list_tools()
        call = await client.call_tool(tool_name, arguments)
        return tools, call

    tools, call = anyio.run(_with_client, registry, run)

    assert [tool.name for tool in tools.tools] == list(DEFAULT_TOOL_NAMES)
    assert call.is_error is False
    registry.shutdown()
