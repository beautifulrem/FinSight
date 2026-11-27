# MCP Integration (Server & Client)

Languages: English | [中文](zh/mcp.md)

FinSight implements both sides of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/): an **MCP Server** exposing FinSight's read-only financial evidence tools to external MCP hosts (desktop/IDE MCP clients and custom agents), and an **MCP Client** allowing FinSight's LangGraph agent to call external MCP servers safely.

## MCP Server (`query_intelligence/agent/mcp_server.py`)

Publishes all 9 read-only tools through the shared `ToolRegistry` (identical Pydantic schemas, timeouts, retries, caching, and error normalization):

| Tool | Purpose | Evidence IDs |
|---|---|---|
| `resolve_entity` | Resolve Chinese/English security or sector mentions to tickers | — |
| `get_price_history` | Daily bars and intraday quote (during trading hours) | `price_<symbol>` |
| `compute_indicators` | MA5/MA20, RSI-14, MACD, Bollinger bands, 20d volatility, returns | `indicators_<symbol>` |
| `get_fundamentals` | Financial indicators, valuation multiples, and industry context | `fundamental_<symbol>`, `industry_<name>` |
| `get_macro_indicators` | CPI, PMI, M2, LPR, 10Y China yield, and policy events | `macro_<code>` |
| `search_news`, `search_announcements`, `search_knowledge` | Ranked news, company filings, research notes, and FAQs | `<source>_<doc_id>` |
| `analyze_sentiment` | Per-document financial sentiment (`classical` default; `QI_AGENT_SENTIMENT_BACKEND=finbert` optional) | `sentiment_<symbols>` |

### Running the Server

```bash
# stdio transport (for desktop MCP clients; logs go to stderr)
python -m query_intelligence.agent.mcp_server --offline

# Authenticated streamable HTTP transport
python -m query_intelligence.agent.mcp_server --transport http --host 127.0.0.1 --port 8001
# Endpoint: http://127.0.0.1:8001/mcp
```

**HTTP Security**: `build_mcp_http_app()` enforces `SecuritySettings.validate()`, `BodyLimitMiddleware`, rate limiting, and `QI_API_KEYS` authentication. Binding to a non-loopback host or running under `QI_PROFILE=production` fails closed unless `QI_API_KEYS` is configured (or `QI_ALLOW_ANONYMOUS=1` is set explicitly). Optionally restrict host headers with `QI_MCP_ALLOWED_HOSTS`.

### Client Configuration Example (`mcp.json`)

```json
{
  "mcpServers": {
    "finsight": {
      "command": "/absolute/path/to/FinSight/.venv/bin/python",
      "args": ["-m", "query_intelligence.agent.mcp_server", "--offline"],
      "env": {"PYTHONPATH": "/absolute/path/to/FinSight"}
    }
  }
}
```

## External MCP Client (`query_intelligence/agent/tools/mcp_client.py`)

Disabled by default. Setting `QI_MCP_SERVERS` (inline JSON or path to a JSON file) connects to stdio or streamable HTTP MCP servers at startup and registers allowed tools under `mcp__<server>__<tool>` for the LLM agent loop (the deterministic planner never invokes remote tools).

```json
{
  "calendar": {
    "command": "python",
    "args": ["tests/fixtures/mcp_trading_calendar_server.py"],
    "timeout_s": 5
  },
  "filings": {
    "url": "http://127.0.0.1:9000/mcp",
    "headers": {"Authorization": "Bearer ${FILINGS_TOKEN}"},
    "tools": ["search_filings"],
    "timeout_s": 10
  }
}
```

`${VAR}` expressions inside `env` and `headers` expand from environment variables (and are redacted from provenance snapshots).

### Remote-Tool Safety Controls

- **Local schema validation**: Arguments validate locally via `jsonschema` before contacting remote servers (`invalid_arguments` on mismatch).
- **Bounded calls**: Enforces per-server `timeout_s` (default `15 s`), `connect_timeout_s` (`30 s`), `max_text_chars` (`4000`), and list caps without blind retries.
- **Untrusted output & description sanitization**: Tool descriptions, schema text, keys, and returned strings pass through `injection.sanitize_untrusted_text`, wrap in `UNTRUSTED TOOL DATA`, emit `mcp_<server>_<tool>_<hash>` evidence IDs, and set `instruction_like_text_removed_from_tool_output` when redacted.
- **Graceful isolation**: Unreachable servers are skipped at startup without blocking local tools; tool name collisions are rejected.

## The fixture server and tests

`tests/fixtures/mcp_trading_calendar_server.py` provides a stdio A-share trading calendar (`is_trading_day`, `next_trading_day`, `count_trading_days`) plus probe tools (`exchange_notice` with embedded prompt injection, `slow_lookup`, `broken_lookup`).

```bash
python -m pytest -q tests/test_agent_mcp_server.py tests/test_agent_mcp_client.py
```

## Real third-party servers

`scripts/mcp_third_party_demo.py` verifies interoperability with unmodified upstream reference servers (`mcp-server-time==2026.8.18` and `mcp-server-fetch==2026.8.18` via `uvx`), exercising schema validation, indirect prompt-injection redaction, and verified citation of MCP evidence IDs (transcript in [`docs/results/protocols/mcp-third-party-demo.txt`](results/protocols/mcp-third-party-demo.txt)):

```bash
QI_USE_LIVE_MARKET=0 QI_USE_LIVE_MACRO=0 QI_USE_LIVE_NEWS=0 QI_USE_LIVE_ANNOUNCEMENT=0 \
  python scripts/mcp_third_party_demo.py 2>/dev/null
python -m pytest -q tests/test_mcp_third_party_demo.py
```

Always review third-party tool descriptions before adding them to the `tools` allowlist.
