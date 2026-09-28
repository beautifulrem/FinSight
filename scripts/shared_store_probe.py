"""Cross-replica probe: two running FinSight processes that share one Postgres.

Checks, with the ``a2a-sdk`` client and plain HTTP:

1. an A2A task that stops at ``input-required`` on replica 1 is visible on replica 2 (``GetTask``), not to a
   second API key, and is resumed to ``completed`` by a reply sent to replica 2;
2. replica 1 then reads the completed task, with its artifacts;
3. a trace written by ``/agent/chat`` on replica 1 is listed and served by ``/agent/traces`` on replica 2, and
   hidden from the second API key.

Start the replicas with the same ``QI_AGENT_CHECKPOINT_DB=postgresql://…`` and ``QI_API_KEYS=<a>,<b>``, then:

    python scripts/shared_store_probe.py --replica http://127.0.0.1:8851 --replica http://127.0.0.1:8852 \
        --key "$KEY_A" --other-key "$KEY_B" --out docs/results/protocols/shared-stores-two-replicas.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from a2a.client import A2ACardResolver, A2AClientError, ClientConfig, ClientFactory
from a2a.helpers import get_artifact_text, new_text_message
from a2a.types import GetTaskRequest, Role, SendMessageRequest, TaskState
from a2a.utils.errors import A2AError


async def _client(base: str, http: httpx.AsyncClient) -> Any:
    card = await A2ACardResolver(http, base).get_agent_card()
    return ClientFactory(ClientConfig(streaming=False, httpx_client=http)).create(card)


async def _send(client: Any, text: str, *, task_id: str | None = None, context_id: str | None = None) -> Any:
    message = new_text_message(text, role=Role.ROLE_USER, task_id=task_id, context_id=context_id)
    async for event in client.send_message(SendMessageRequest(message=message)):
        if event.HasField("task"):
            return event.task
    raise RuntimeError("no task returned")


async def probe(replicas: list[str], key: str, other_key: str) -> dict[str, Any]:
    first, second = replicas
    checks: dict[str, Any] = {}
    timings: dict[str, float] = {}
    async with (
        httpx.AsyncClient(timeout=120, headers={"X-API-Key": key}) as http_a,
        httpx.AsyncClient(timeout=120, headers={"X-API-Key": other_key}) as http_b,
    ):
        a1, a2 = await _client(first, http_a), await _client(second, http_a)
        b2 = await _client(second, http_b)

        started = time.perf_counter()
        pending = await _send(a1, "它的市盈率呢")
        timings["send_on_replica_1_s"] = round(time.perf_counter() - started, 3)
        checks["replica_1_input_required"] = pending.status.state == TaskState.TASK_STATE_INPUT_REQUIRED

        seen = await a2.get_task(GetTaskRequest(id=pending.id))
        checks["replica_2_get_task_state"] = TaskState.Name(seen.status.state)
        try:
            await b2.get_task(GetTaskRequest(id=pending.id))
            checks["other_key_denied"] = False
        except (A2AClientError, A2AError) as exc:  # the server answers TaskNotFoundError, not 403
            checks["other_key_denied"] = True
            checks["other_key_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"

        started = time.perf_counter()
        resumed = await _send(a2, "贵州茅台", task_id=pending.id, context_id=pending.context_id)
        timings["resume_on_replica_2_s"] = round(time.perf_counter() - started, 3)
        checks["replica_2_resumed_state"] = TaskState.Name(resumed.status.state)
        checks["same_task_id"] = resumed.id == pending.id

        final = await a1.get_task(GetTaskRequest(id=pending.id))
        artifacts = {artifact.name: artifact for artifact in final.artifacts}
        checks["replica_1_final_state"] = TaskState.Name(final.status.state)
        checks["replica_1_artifacts"] = sorted(artifacts)
        checks["answer_cites"] = [
            token for token in ("fundamental_600519.SH", "600519") if token in get_artifact_text(artifacts["answer"])
        ]

        session = f"probe{uuid.uuid4().hex[:10]}"
        answer = (
            await http_a.post(f"{first}/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": session})
        ).json()
        listing = (await http_a.get(f"{second}/agent/traces", params={"session_id": session})).json()["traces"]
        checks["trace_listed_on_replica_2"] = [item["trace_id"] for item in listing] == [answer["trace_id"]]
        detail = await http_a.get(f"{second}/agent/traces/{answer['trace_id']}")
        checks["trace_served_on_replica_2"] = detail.status_code == 200 and detail.json()["query"] == answer["query"]
        hidden = await http_b.get(f"{second}/agent/traces/{answer['trace_id']}")
        checks["trace_hidden_from_other_key"] = hidden.status_code == 404
        for client in (a1, a2, b2):
            await client.close()

    passed = (
        checks["replica_1_input_required"]
        and checks["replica_2_get_task_state"] == "TASK_STATE_INPUT_REQUIRED"
        and checks["other_key_denied"]
        and checks["replica_2_resumed_state"] == "TASK_STATE_COMPLETED"
        and checks["same_task_id"]
        and checks["replica_1_final_state"] == "TASK_STATE_COMPLETED"
        and checks["trace_listed_on_replica_2"]
        and checks["trace_served_on_replica_2"]
        and checks["trace_hidden_from_other_key"]
    )
    return {"passed": bool(passed), "checks": checks, "timings": timings}


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replica", action="append", required=True, help="base URL; pass exactly two")
    parser.add_argument("--key", required=True)
    parser.add_argument("--other-key", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if len(args.replica) != 2:
        parser.error("pass --replica twice")
    result = asyncio.run(probe(args.replica, args.key, args.other_key))
    record = {
        "probe": "shared_store_probe",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _commit(),
        "command": "python scripts/shared_store_probe.py --replica <r1> --replica <r2> --key <a> --other-key <b>",
        "replicas": args.replica,
        **result,
    }
    text = json.dumps(record, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
