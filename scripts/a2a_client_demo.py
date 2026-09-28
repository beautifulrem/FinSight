"""A2A client demo: another agent delegating research to FinSight with the official ``a2a-sdk`` client.

It walks through the four interactions an orchestrating agent needs:

1. discovery: fetch and print the agent card (``/.well-known/agent-card.json``);
2. ``SendMessage``: a normal question, returning a completed task with the answer and evidence artifacts;
3. ``input-required``: a question without a target gets a clarification; the reply is sent on the same task
   (``task_id`` + ``context_id``) and resumes the paused run;
4. ``SendStreamingMessage``: prints each streamed event (status updates per graph node and tool call, then
   the artifacts and the final status).

Run it against a running API (offline data and no LLM are fine):

    uvicorn query_intelligence.api.app:create_app --factory --host 127.0.0.1 --port 8000
    python scripts/a2a_client_demo.py --url http://127.0.0.1:8000
    # with QI_API_KEYS set on the server:
    python scripts/a2a_client_demo.py --url http://127.0.0.1:8000 --api-key "$FINSIGHT_API_KEY"

``tests/test_a2a_client_demo.py`` runs ``run_demo`` in-process against the app (httpx ASGI transport).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, StreamResponse, Task, TaskState

DEFAULT_QUESTION = "贵州茅台的市盈率是多少"
DEFAULT_VAGUE_QUESTION = "它的市盈率呢"
DEFAULT_CLARIFICATION_REPLY = "贵州茅台"
DEFAULT_STREAM_QUESTION = "贵州茅台最近走势怎么样"


def state_name(state: int) -> str:
    return TaskState.Name(state)


def describe_task(task: Task) -> dict[str, Any]:
    """The parts of a task an orchestrator uses: state, answer text, evidence data, clarification text."""
    artifacts = {artifact.name: artifact for artifact in task.artifacts}
    evidence: dict[str, Any] = {}
    if "evidence" in artifacts:
        from google.protobuf.json_format import MessageToDict

        for part in artifacts["evidence"].parts:
            if part.HasField("data"):
                evidence = MessageToDict(part.data)
    return {
        "task_id": task.id,
        "context_id": task.context_id,
        "state": state_name(task.status.state),
        "answer": get_artifact_text(artifacts["answer"]) if "answer" in artifacts else "",
        "evidence_used": evidence.get("evidence_used") or [],
        "trace_id": evidence.get("trace_id"),
        "status_message": get_message_text(task.status.message) if task.status.HasField("message") else "",
    }


def event_line(event: StreamResponse) -> tuple[str, str]:
    """(kind, one-line description) for a streamed event."""
    if event.HasField("task"):
        return "task", f"task {event.task.id[:8]} {state_name(event.task.status.state)}"
    if event.HasField("status_update"):
        update = event.status_update
        text = get_message_text(update.status.message) if update.status.HasField("message") else ""
        return "status_update", f"{state_name(update.status.state)} {text}".strip()
    if event.HasField("artifact_update"):
        artifact = event.artifact_update.artifact
        preview = get_artifact_text(artifact).replace("\n", " ")[:80] or "(structured data part)"
        return "artifact_update", f"artifact {artifact.name}: {preview}"
    if event.HasField("message"):
        return "message", get_message_text(event.message)[:80]
    return "unknown", ""


async def _send(client: Any, text: str, *, task_id: str | None = None, context_id: str | None = None) -> Task:
    message = new_text_message(text, role=Role.ROLE_USER, task_id=task_id, context_id=context_id)
    async for event in client.send_message(SendMessageRequest(message=message)):
        if event.HasField("task"):
            return event.task
    raise RuntimeError("SendMessage returned no task")


async def run_demo(
    base_url: str,
    *,
    http_client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
    question: str = DEFAULT_QUESTION,
    vague_question: str = DEFAULT_VAGUE_QUESTION,
    clarification_reply: str = DEFAULT_CLARIFICATION_REPLY,
    stream_question: str = DEFAULT_STREAM_QUESTION,
    out: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the four steps and return a summary (used by the test)."""
    owns_client = http_client is None
    headers = {"X-API-Key": api_key} if api_key else {}
    http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(180.0), headers=headers)
    if http_client is not None and headers:
        http.headers.update(headers)
    summary: dict[str, Any] = {}
    try:
        # 1. Discovery
        card = await A2ACardResolver(http, base_url).get_agent_card()
        out(f"== Agent card: {card.name} {card.version}")
        out(f"   interface: {card.supported_interfaces[0].url} ({card.supported_interfaces[0].protocol_binding})")
        out(f"   streaming: {card.capabilities.streaming}; skills: {', '.join(skill.id for skill in card.skills)}")
        summary["card"] = {"name": card.name, "skills": [skill.id for skill in card.skills]}

        blocking = ClientFactory(ClientConfig(streaming=False, httpx_client=http)).create(card)
        streaming = ClientFactory(ClientConfig(streaming=True, httpx_client=http)).create(card)

        # 2. SendMessage
        out(f"\n== SendMessage: {question}")
        answered = describe_task(await _send(blocking, question))
        out(f"   state: {answered['state']}  trace: {answered['trace_id']}")
        out(f"   evidence: {', '.join(answered['evidence_used'])}")
        out("   " + answered["answer"].replace("\n", "\n   "))
        summary["answer"] = answered

        # 3. input-required, then the reply on the same task
        out(f"\n== SendMessage (needs clarification): {vague_question}")
        pending = describe_task(await _send(blocking, vague_question))
        out(f"   state: {pending['state']}  question: {pending['status_message']}")
        summary["clarification"] = pending
        if pending["state"] == "TASK_STATE_INPUT_REQUIRED":
            out(f"== Reply on task {pending['task_id'][:8]}: {clarification_reply}")
            resumed = describe_task(
                await _send(
                    blocking, clarification_reply, task_id=pending["task_id"], context_id=pending["context_id"]
                )
            )
            out(f"   state: {resumed['state']}  evidence: {', '.join(resumed['evidence_used'])}")
            out("   " + resumed["answer"].split("\n")[0])
            summary["resumed"] = resumed

        # 4. SendStreamingMessage
        out(f"\n== SendStreamingMessage: {stream_question}")
        kinds: list[str] = []
        final_state = ""
        message = new_text_message(stream_question, role=Role.ROLE_USER)
        async for event in streaming.send_message(SendMessageRequest(message=message)):
            kind, line = event_line(event)
            kinds.append(kind)
            out(f"   [{kind}] {line}")
            if event.HasField("status_update"):
                final_state = state_name(event.status_update.status.state)
        summary["stream"] = {"kinds": kinds, "final_state": final_state}
        await blocking.close()
        await streaming.close()
    finally:
        if owns_client:
            await http.aclose()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="FinSight base URL (agent card host)")
    parser.add_argument("--api-key", default=None, help="X-API-Key when the server sets QI_API_KEYS")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--vague-question", default=DEFAULT_VAGUE_QUESTION)
    parser.add_argument("--reply", default=DEFAULT_CLARIFICATION_REPLY)
    parser.add_argument("--stream-question", default=DEFAULT_STREAM_QUESTION)
    parser.add_argument("--json", action="store_true", help="also print the summary as JSON")
    args = parser.parse_args(argv)
    summary = asyncio.run(
        run_demo(
            args.url,
            api_key=args.api_key,
            question=args.question,
            vague_question=args.vague_question,
            clarification_reply=args.reply,
            stream_question=args.stream_question,
        )
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = summary["answer"]["state"] == "TASK_STATE_COMPLETED" and summary["stream"]["final_state"] == (
        "TASK_STATE_COMPLETED"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
