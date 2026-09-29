# Protocol results: shared stores, A2A client, MCP client

Date: 2026-09-28. Machine: MacBook (Apple silicon, macOS), Python 3.13. Postgres 16.15 (`postgres:16-alpine`) ran in a colima/Docker container. All runs used offline data and no LLM, so answers are template answers and no gateway calls were made.

## 1. Two replicas sharing one Postgres (pytest)

At commit `e01cc16`, two `create_app` instances in one process. They share only the database: each has its own agent service, trace store and A2A task store.

```bash
docker run -d --name fs-r3-pg -e POSTGRES_PASSWORD=finsight -e POSTGRES_DB=finsight -p 55433:5432 postgres:16-alpine
export QI_TEST_POSTGRES_DSN=postgresql://postgres:finsight@127.0.0.1:55433/finsight
python -m pytest tests/test_agent_shared_stores_postgres.py tests/test_agent_checkpoint_postgres.py -v
```

```text
test_agent_shared_stores_postgres.py::test_a2a_task_started_on_one_replica_is_served_and_resumed_by_the_other PASSED
test_agent_shared_stores_postgres.py::test_traces_written_by_one_replica_are_listed_and_served_by_the_other PASSED
test_agent_shared_stores_postgres.py::test_trace_retention_by_count_and_age PASSED
test_agent_shared_stores_postgres.py::test_control_in_memory_stores_are_not_shared PASSED
test_agent_checkpoint_postgres.py::test_two_replicas_share_session_memory_and_clarifications PASSED
5 passed in 4.32s
```

What these tests show:

- **Task continuation across replicas.** A task paused at `input-required` on replica 1 is returned by `GetTask` on replica 2. The reply sent to replica 2 resumes it to `completed` with the same task id. Replica 1 then reads the completed task and its `answer` and `evidence` artifacts, and `ListTasks` by context finds it.
- **Traces across replicas.** A trace written on replica 1 is listed and served by `/agent/traces` on replica 2, and `/agent/feedback` on replica 2 finds it.
- **Owner scoping.** A second API key gets `TaskNotFoundError` for the task, a 404 for the trace, and an empty trace list.
- **Retention.** With `max_rows=3` and a 1-day age limit, 6 emitted traces (one 3 days old) leave the 3 newest.
- **Negative control.** With `QI_A2A_TASK_DB=memory` and `QI_AGENT_TRACE_DB=memory`, replica 2 cannot see replica 1's task or trace. This confirms the Postgres stores are what makes sharing work.

## 2. Two real API processes (`uvicorn`) sharing one Postgres

Each process loaded the full offline Query Intelligence service. Both were started at commit `a1ddfca`; the probe ran at `ca5b9c1`, which only adds the probe script.

```bash
export QI_AGENT_CHECKPOINT_DB=postgresql://postgres:finsight@127.0.0.1:55433/finsight QI_API_KEYS=<a>,<b>
export DEEPSEEK_API_KEY= CHATBOT_LIVE_DATA=0 QI_USE_LIVE_MARKET=0 QI_USE_LIVE_NEWS=0 QI_USE_LIVE_ANNOUNCEMENT=0 QI_USE_LIVE_MACRO=0
uvicorn query_intelligence.api.app:create_app --factory --host 127.0.0.1 --port 8851 &
uvicorn query_intelligence.api.app:create_app --factory --host 127.0.0.1 --port 8852 &
python scripts/shared_store_probe.py --replica http://127.0.0.1:8851 --replica http://127.0.0.1:8852 \
    --key <a> --other-key <b> --out docs/results/protocols/shared-stores-two-replicas.json
```

Result: [`shared-stores-two-replicas.json`](shared-stores-two-replicas.json), `"passed": true`, with every check true:

- input-required on replica 1;
- `GetTask` on replica 2 returns `TASK_STATE_INPUT_REQUIRED`;
- the other key is refused with `TaskNotFoundError`;
- the resume on replica 2 reaches `TASK_STATE_COMPLETED` with the same task id;
- replica 1 reads the completed task with the `answer` and `evidence` artifacts;
- the trace is listed and served on replica 2 and hidden from the other key.

Timings: SendMessage 0.07 s on replica 1, resume 0.83 s on replica 2.

After the runs, the database held two FinSight tables next to the LangGraph checkpoint tables: `finsight_a2a_tasks` (12 rows) and `finsight_agent_traces` (17 rows). The rows were grouped under two owners, one per API-key hash.

## 3. A2A client demo against a real replica

The output is in [`a2a-client-demo.txt`](a2a-client-demo.txt), from `python scripts/a2a_client_demo.py --url http://127.0.0.1:8851 --api-key <a>`, exit code 0. The four steps:

- the agent card: JSON-RPC interface, streaming, 3 skills;
- `SendMessage`: completed, citing `fundamental_600519.SH` and `industry_白酒`;
- `它的市盈率呢`: `TASK_STATE_INPUT_REQUIRED`, with the clarification question; the reply `贵州茅台` on the same task: completed;
- `SendStreamingMessage`: 14 events. First the `submitted` task, then 10 `working` status updates (the start of work, 6 graph-node starts and 3 tool calls), then the `answer` and `evidence` artifact updates, then `completed`.

## 4. MCP client

`python -m pytest tests/test_agent_mcp_client.py -q`: 12 passed in about 30 s. This spawns `tests/fixtures/mcp_trading_calendar_server.py` over stdio. See [docs/mcp.md](../../mcp.md#the-fixture-server-and-tests).

## 5. A2A interop with the JavaScript SDK client

Date: 2026-09-28. `node tools/a2a-js-client/interop.mjs --url http://127.0.0.1:8861` (@a2a-js/sdk 1.2.1, Node v26.9.0) against an offline uvicorn server with no LLM key:

- [`a2a-js-interop-before-fix.txt`](a2a-js-interop-before-fix.txt): server at `6ae375f`, 15/16. `SubscribeToTask` on a finished task returned -32602 (`InvalidParams`), which the JS SDK reports as `JsonRpcRequestMalformedError`.
- [`a2a-js-interop.txt`](a2a-js-interop.txt): server at `1535922`, 16/16. The same call now returns `UnsupportedOperationError` (-32004), as A2A 1.0 §9.4.6 requires.

See [docs/a2a-and-observability.md](../../a2a-and-observability.md#interop-with-another-framework-the-javascript-sdk-client).

## 6. MCP client against real third-party servers

[`mcp-third-party-demo.txt`](mcp-third-party-demo.txt), from `python scripts/mcp_third_party_demo.py 2>/dev/null` at `1535922` (uvx 0.12.18): the official `mcp-server-time` and `mcp-server-fetch` (2026.8.18, MCP SDK 1.x) attached through `QI_MCP_SERVERS`. Namespaced tools and schemas, a structured call with evidence, local schema rejection, an injected page redacted and wrapped as untrusted, and a scripted-LLM agent run citing both MCP evidence ids with verification passed. See [docs/mcp.md](../../mcp.md#real-third-party-servers).

`python -m pytest tests/test_a2a_js_interop.py tests/test_mcp_third_party_demo.py -q`: 5 passed in 27 s.
