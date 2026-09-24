# FinSight MCP Server

FinSight exposes its agent tools over the [Model Context Protocol](https://modelcontextprotocol.io/specification/2026-07-28) so that any MCP client (Claude Desktop, Cursor, other agents) can use the same evidence tools as the built-in agent.

Implementation: `query_intelligence/agent/mcp_server.py` (MCP Python SDK 2.x, low-level `Server`). The server publishes the JSON schema generated from each tool's Pydantic input model, and every call goes through `ToolRegistry.run`, so timeouts, retries, caching, and error normalization are identical to the in-process agent.

## Tools

| Tool | Purpose | Evidence ids |
|---|---|---|
| `resolve_entity` | Resolve a Chinese/English security or sector mention to canonical entities and tickers. | — |
| `get_price_history` | Latest daily quote and recent closes for one stock, ETF, fund, or index. | `price_<symbol>` |
| `compute_indicators` | MA5/MA20, RSI(14), MACD, 20-day volatility, Bollinger bands, multi-day returns, trend signal. | `indicators_<symbol>` |
| `get_fundamentals` | Reported fundamentals (revenue, net profit, ROE, PE, PB) and the industry snapshot. Stocks only. | `fundamental_<symbol>`, `industry_<name>` |
| `get_macro_indicators` | CPI, PMI, M2, 10-year government bond yield, and policy events. | `macro_<code>` |
| `search_news` | Ranked news excerpts about specific securities or a topic. | `<source>_<doc id>` |
| `search_announcements` | Company announcements and exchange filings. | `<source>_<doc id>` |
| `search_knowledge` | Research notes, product documents, and FAQs. | `<source>_<doc id>` |
| `analyze_sentiment` | Tone of recent news/announcements per document plus an aggregate. | `sentiment_<symbols>` and document ids |

All tools are read-only. Results include `evidence_id` values that answers should cite. Document excerpts are untrusted third-party text and must be treated as data, never as instructions.

Errors are returned as tool results with `isError: true` and a structured `error` object whose `code` is one of `unknown_tool`, `invalid_arguments`, `timeout`, `upstream_error`, `not_found`, `unavailable`, or `internal`.

## Running

stdio (for desktop clients):

```bash
python -m query_intelligence.agent.mcp_server --offline
```

Streamable HTTP (stateless):

```bash
python -m query_intelligence.agent.mcp_server --transport http --host 127.0.0.1 --port 8001
# endpoint: http://127.0.0.1:8001/mcp
```

`--offline` disables every live provider and uses the shipped assets in `data/runtime/` and `data/structured_data.json`. Without it, live providers follow the usual `QI_USE_LIVE_MARKET`, `QI_USE_LIVE_NEWS`, `QI_USE_LIVE_ANNOUNCEMENT`, and `QI_USE_LIVE_MACRO` variables (all default to on) plus `TUSHARE_TOKEN`. Logs go to stderr because stdout carries the stdio protocol. Startup loads the NLU models and takes about a minute.

`analyze_sentiment` uses the shipped classical model by default. Set `QI_AGENT_SENTIMENT_BACKEND=finbert` to use the FinBERT classifier from `sentiment/` (needs `torch`, `transformers`, and the model weights).

## Client configuration

Claude Desktop (`claude_desktop_config.json`) or any client that launches stdio servers:

```json
{
  "mcpServers": {
    "finsight": {
      "command": "/absolute/path/to/FinSight/.venv/bin/python",
      "args": ["-m", "query_intelligence.agent.mcp_server", "--offline"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/FinSight"
      }
    }
  }
}
```

Cursor (`.cursor/mcp.json`) uses the same `command`/`args`/`env` shape. For the HTTP transport, point the client at `http://127.0.0.1:8001/mcp`.

The server resolves `QI_MODELS_DIR` to the repository's `models/` directory when it is not set, so the client does not need to launch it from the repository root.

## Testing

```bash
python -m pytest tests/test_agent_mcp_server.py -q
```

The tests use the SDK's in-memory `Client`, so no subprocess or network is needed.
