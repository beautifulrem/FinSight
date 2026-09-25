"""Session memory for the agent.

* Checkpointer: in-memory by default; ``QI_AGENT_CHECKPOINT_DB=/path/agent.sqlite`` persists
  sessions in SQLite so conversations survive restarts; ``QI_AGENT_CHECKPOINT_DB=postgresql://...``
  shares them across processes and replicas.
* Each finished turn is appended to ``state["turns"]``. The previous user questions are handed to
  the classical NLU as ``dialog_context`` (so "它/那它的市盈率呢/that stock" resolve to the last
  single entity), and short summaries of recent turns are given to the LLM.
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

MAX_CONTEXT_TURNS = 3
MAX_HISTORY_TURNS = 2
_HISTORY_ANSWER_CHARS = 600


def make_checkpointer(path: str | None = None) -> Any:
    """In-memory by default; a SQLite file path; or a ``postgresql://`` DSN for sessions shared by several
    processes or replicas (``langgraph-checkpoint-postgres`` with a psycopg connection pool)."""
    target = path if path is not None else os.getenv("QI_AGENT_CHECKPOINT_DB", "").strip()
    if not target:
        return InMemorySaver()
    if target.startswith(("postgres://", "postgresql://")):
        return _postgres_checkpointer(target)
    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(target, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def _postgres_checkpointer(dsn: str) -> Any:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=int(os.getenv("QI_AGENT_CHECKPOINT_POOL", "10")),
        # settings required by PostgresSaver (see PostgresSaver.from_conn_string)
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


def turn_record(state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": state.get("query", ""),
        "route": result.get("route"),
        "answer": str(result.get("answer") or "")[:_HISTORY_ANSWER_CHARS],
        "entities": result.get("nlu_summary", {}).get("entities", []),
        "evidence_used": result.get("evidence_used", []),
    }


def dialog_context_from_turns(turns: list[dict[str, Any]], explicit: list[dict[str, Any]] | None = None) -> list[dict]:
    """Previous user questions (oldest first) followed by any context supplied with the request."""
    context = [{"role": "user", "content": turn["query"]} for turn in turns[-MAX_CONTEXT_TURNS:] if turn.get("query")]
    return [*context, *(explicit or [])]


def history_messages(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns[-MAX_HISTORY_TURNS:]:
        if turn.get("route") in {"refuse", "clarify"}:
            continue
        messages.append({"role": "user", "content": f"(earlier question) {turn['query']}"})
        messages.append({"role": "assistant", "content": f"(earlier answer summary) {turn['answer']}"})
    return messages


_PRONOUN_ZH = re.compile(r"这只股票|这支股票|这只基金|这个标的|这家公司|该公司|该股|这只|它")
_PRONOUN_EN = re.compile(r"\b(?:this stock|that stock|the stock|this company|it)\b", re.IGNORECASE)
_POSSESSIVE_EN = re.compile(r"\bits\b", re.IGNORECASE)
_LISTED_TYPES = {"stock", "etf", "fund", "index"}


def listed_entities(nlu_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        entity
        for entity in nlu_result.get("entities") or []
        if entity.get("symbol") and entity.get("entity_type") in _LISTED_TYPES
    ]


def resolve_coreference(query: str, turns: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Rewrite a pronoun to the last turn's single listed entity: ``(rewritten_query, reason)``."""
    if not turns:
        return None
    previous = [entity for entity in turns[-1].get("entities") or [] if entity.get("symbol")]
    if len(previous) != 1:
        return None
    name = str(previous[0].get("name") or previous[0]["symbol"])
    possessive = _POSSESSIVE_EN.search(query)
    if possessive:
        rewritten = f"{query[: possessive.start()]}{name}'s{query[possessive.end() :]}"
        return rewritten, f"coreference:{possessive.group(0)}->{name}"
    for pattern in (_PRONOUN_ZH, _PRONOUN_EN):
        match = pattern.search(query)
        if match:
            rewritten = f"{query[: match.start()]}{name}{query[match.end() :]}"
            return rewritten, f"coreference:{match.group(0)}->{name}"
    return None


def apply_clarification(query: str, reply: str) -> tuple[str, str]:
    """Fold a clarification reply (e.g. "宁德时代") into the original question.

    The reply replaces the dangling pronoun when there is one ("它的市盈率呢" -> "宁德时代的市盈率呢");
    otherwise it is prepended so the NLU sees the entity. Returns ``(query, reason)``.
    """
    reply = reply.strip()
    possessive = _POSSESSIVE_EN.search(query)
    if possessive:
        return f"{query[: possessive.start()]}{reply}'s{query[possessive.end() :]}", f"clarified:{reply}"
    for pattern in (_PRONOUN_ZH, _PRONOUN_EN):
        match = pattern.search(query)
        if match:
            return f"{query[: match.start()]}{reply}{query[match.end() :]}", f"clarified:{reply}"
    separator = "" if re.search(r"[\u4e00-\u9fff]$", reply) else " "
    return f"{reply}{separator}{query}", f"clarified:{reply}"
