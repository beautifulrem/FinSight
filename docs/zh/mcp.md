# MCP 集成（服务端与客户端）

语言：[English](../mcp.md) | 中文

FinSight 双向支持 [Model Context Protocol（MCP）](https://modelcontextprotocol.io/)：既可作为 **MCP 服务端** 向桌面/IDE MCP 客户端或外部 Agent 暴露 9 个只读金融取证工具，也可作为 **MCP 客户端** 安全挂载外部 MCP 工具。

## MCP 服务端（`query_intelligence/agent/mcp_server.py`）

通过共享 `ToolRegistry` 发布全部 9 个只读工具（与进程内 Agent 共用 Pydantic Schema、超时、重试、缓存与统一错误码）：

| 工具 | 功能 | 证据 ID |
|---|---|---|
| `resolve_entity` | 中英文证券与板块实体消歧 | — |
| `get_price_history` | 个股/ETF/基金/指数日线及交易时段盘中实时价 | `price_<symbol>` |
| `compute_indicators` | MA5/MA20、RSI(14)、MACD、布林带、20 日波动率、多日收益 | `indicators_<symbol>` |
| `get_fundamentals` | 财务指标、估值倍数与所属行业快照 | `fundamental_<symbol>`、`industry_<name>` |
| `get_macro_indicators` | CPI、PMI、M2、LPR、10 年期国债收益率与政策事件 | `macro_<code>` |
| `search_news`、`search_announcements`、`search_knowledge` | 排序后的新闻、公告、研报与产品知识 | `<source>_<doc_id>` |
| `analyze_sentiment` | 文档金融情感分析（默认 `classical`；可选 `QI_AGENT_SENTIMENT_BACKEND=finbert`） | `sentiment_<symbols>` |

### 启动服务端

```bash
# stdio 传输（供桌面 MCP 客户端使用；日志输出至 stderr）
python -m query_intelligence.agent.mcp_server --offline

# 鉴权 streamable HTTP 传输
python -m query_intelligence.agent.mcp_server --transport http --host 127.0.0.1 --port 8001
# 访问地址：http://127.0.0.1:8001/mcp
```

**HTTP 安全边界**：`build_mcp_http_app()` 强制启用 `SecuritySettings.validate()`、`BodyLimitMiddleware`、限流与 `QI_API_KEYS` 鉴权。在 `QI_PROFILE=production` 或绑定非回环地址时，未配置 `QI_API_KEYS` 会拒绝启动（除非显式设置 `QI_ALLOW_ANONYMOUS=1`）；可通过 `QI_MCP_ALLOWED_HOSTS` 限制允许的 Host 头。

### 客户端配置示例（`mcp.json`）

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

## 外部 MCP 客户端（`query_intelligence/agent/tools/mcp_client.py`）

默认关闭。通过 `QI_MCP_SERVERS`（内联 JSON 或文件路径）配置 stdio / streamable HTTP 外部服务，允许的工具将以 `mcp__<server>__<tool>` 命名注册进 LLM 工具循环（确定性规划器不调用外部工具）。

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

`env` 与 `headers` 中的 `${VAR}` 自动从环境变量展开，且在评测溯源快照中被脱敏。

### 外部工具安全隔离

- **本地 Schema 校验**：调用前使用 `jsonschema` 校验参数，非法入参直接返回 `invalid_arguments`。
- **有界超时与大小限制**：强制单次 `timeout_s`（默认 15 秒）、连接超时 `30 秒`、单字符串长度 `max_text_chars=4000` 及列表长度上限，不盲目重试。
- **不可信数据与描述清洗**：工具描述、Schema 文本与返回内容均经过 `injection.sanitize_untrusted_text` 过滤并封装于 `UNTRUSTED TOOL DATA` 信封，产出 `mcp_<server>_<tool>_<hash>` 证据 ID，脱敏时触发 `instruction_like_text_removed_from_tool_output`。
- **故障隔离**：启动时不可达的外部服务自动跳过，不影响本地 9 个工具正常运行。

## 测试与第三方互操作验证

```bash
python -m pytest -q tests/test_agent_mcp_server.py tests/test_agent_mcp_client.py

# 官方第三方参考服务互操作（mcp-server-time & mcp-server-fetch，需 uvx）
QI_USE_LIVE_MARKET=0 QI_USE_LIVE_MACRO=0 QI_USE_LIVE_NEWS=0 QI_USE_LIVE_ANNOUNCEMENT=0 \
  python scripts/mcp_third_party_demo.py 2>/dev/null
python -m pytest -q tests/test_mcp_third_party_demo.py
```

互操作实测日志归档于 [`docs/results/protocols/mcp-third-party-demo.txt`](../results/protocols/mcp-third-party-demo.txt)。加入白名单前请人工审阅第三方服务的工具描述。
