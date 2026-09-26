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


_PLURAL_ZH = re.compile(r"这两家公司|这两家|这两只|这两个|两者|它们|他们俩|二者")
_PLURAL_EN = re.compile(r"\b(?:both of them|both|them|these two|the two)\b", re.IGNORECASE)


def recent_entities(turns: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    """Distinct listed entities mentioned in the session, most recent first."""
    seen: dict[str, dict[str, Any]] = {}
    for turn in reversed(turns):
        for entity in turn.get("entities") or []:
            symbol = entity.get("symbol")
            if symbol and symbol not in seen:
                seen[symbol] = {"name": entity.get("name") or symbol, "symbol": symbol}
    return list(seen.values())[:limit]


def resolve_coreference(query: str, turns: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Rewrite a pronoun to entities from earlier turns: ``(rewritten_query, reason)``.

    "它/it/its" resolves to the most recent turn that named exactly one listed entity (skipping turns
    without entities, e.g. a macro question in between); "这两家/它们/both/them" resolves to the two most
    recently named entities. Ambiguous cases return ``None`` and the router asks for clarification.
    """
    if not turns:
        return None
    plural = _PLURAL_ZH.search(query) or _PLURAL_EN.search(query)
    if plural:
        entities = recent_entities(turns, limit=2)
        if len(entities) != 2:
            return None
        zh = bool(_PLURAL_ZH.search(query))
        joined = (
            f"{entities[1]['name']}和{entities[0]['name']}"
            if zh
            else f"{entities[1]['name']} and {entities[0]['name']}"
        )
        rewritten = f"{query[: plural.start()]}{joined}{query[plural.end() :]}"
        return rewritten, f"coreference:{plural.group(0)}->{joined}"
    single = None
    for turn in reversed(turns[-MAX_CONTEXT_TURNS:]):
        listed = [entity for entity in turn.get("entities") or [] if entity.get("symbol")]
        if listed:
            single = listed[0] if len(listed) == 1 else None
            break
    if single is None:
        return None
    name = str(single.get("name") or single["symbol"])
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


_CONSTRAINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("risk:conservative", re.compile(r"保守|稳健|低风险|风险承受能力(?:较)?低|\bconservative\b|\blow risk\b", re.I)),
    ("risk:aggressive", re.compile(r"激进|高风险|风险承受能力(?:较)?高|\baggressive\b|\bhigh risk\b", re.I)),
    ("horizon:long", re.compile(r"长期|长线|中长期|\blong[- ]term\b", re.I)),
    ("horizon:short", re.compile(r"短期|短线|\bshort[- ]term\b", re.I)),
    ("scope:a_shares_only", re.compile(r"只看A股|只关注A股|\bonly A-?shares\b", re.I)),
    ("scope:etf_only", re.compile(r"只看ETF|只买ETF|\bonly ETFs?\b", re.I)),
)
_HOLDING = re.compile(
    r"我(?:持有|买了|拿着|手里有)(?P<what>[^，,。.？?！!]{2,12})|\bI (?:own|hold|bought) (?P<en>[A-Za-z .]{2,30})"
)


def session_memory(turns: list[dict[str, Any]], current_query: str = "") -> dict[str, Any]:
    """Compact memory of the session for the LLM: recent targets and constraints the user stated.

    Extractive and rule-based (no LLM), bounded in size; constraints are carried forward from any earlier
    question so "我是保守型投资者" still applies five turns later.
    """
    queries = [str(turn.get("query") or "") for turn in turns] + ([current_query] if current_query else [])
    constraints: list[str] = []
    holdings: list[str] = []
    for query in queries:
        for label, pattern in _CONSTRAINTS:
            if pattern.search(query) and label not in constraints:
                constraints.append(label)
        for match in _HOLDING.finditer(query):
            what = (match.group("what") or match.group("en") or "").strip()
            if what and what not in holdings:
                holdings.append(what)
    return {
        "turns_so_far": len(turns),
        "recent_targets": recent_entities(turns),
        "user_constraints": constraints,
        "stated_holdings": holdings[:5],
    }


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
