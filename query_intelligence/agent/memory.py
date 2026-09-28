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

from .router import has_macro_content

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


_PRONOUN_ZH = re.compile(r"这只股票|这支股票|这只基金|这个标的|这家公司|那家公司|该公司|该股|这只|那只|这家|那家|它")
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


# Elliptical follow-ups: "ROE呢", "那市净率呢", "最近走势怎么样", "And ROE?", "换成五粮液呢", "What about BYD?".
_ELLIPSIS_ZH = re.compile(r"^(?:那么|那就|那|再看看|再看|还有|换成|换个|改成|看看)|呢[？?。.!！]?$")
_ELLIPSIS_EN = re.compile(r"^(?:and|what about|how about|same for|now for|and for)\b", re.IGNORECASE)
_LEADING_ZH = re.compile(r"^(?:那么|那就|那|再看看|再看|还有|看看)")
_MARKET_WIDE = re.compile(
    r"大盘|市场|A股|沪指|深指|创业板|行业|板块|宏观|\bmarket\b|\bsector\b|\bindex\b", re.IGNORECASE
)
_ASPECT = re.compile(
    r"市盈率|市净率|净资产收益率|营收|营业收入|净利润|净利|毛利率|股息率|市值|负债率|收盘价?|股价|"
    r"走势|涨跌幅?|估值|公告|新闻|分红|业绩|财报|(?<![A-Za-z])(?:P/?E|P/?B|ROE)(?![A-Za-z])|"
    r"revenue|net (?:profit|income)|"
    r"dividend|market cap|valuation|\bprice\b|announcements?|news|trend",
    re.IGNORECASE,
)


def _is_short(query: str) -> bool:
    words = query.split()
    return len(query) <= 20 if not re.search(r"[A-Za-z]{3,}", query) else len(words) <= 8


def _last_targets(turns: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    for turn in reversed(turns[-MAX_CONTEXT_TURNS:]):
        listed = [entity for entity in turn.get("entities") or [] if entity.get("symbol")]
        if listed:
            return listed[:limit]
    return []


def _last_aspects(turns: list[dict[str, Any]]) -> list[str]:
    for turn in reversed(turns[-MAX_CONTEXT_TURNS:]):
        aspects = list(dict.fromkeys(match.group(0) for match in _ASPECT.finditer(str(turn.get("query") or ""))))
        if aspects:
            return aspects[:3]
    return []


def resolve_ellipsis(
    query: str, turns: list[dict[str, Any]], current_targets: list[dict[str, Any]]
) -> tuple[str, str] | None:
    """Complete a short follow-up that leaves out the target or the question: ``(rewritten, reason)``.

    * No target named ("ROE呢", "最近走势怎么样", "And ROE?"): the targets of the most recent turn that
      named any are carried over, unless the question is market-wide or macro.
    * Only a new target named ("换成五粮液呢", "What about BYD?"): the previous question's aspects
      (市盈率, 走势, ...) are carried over to the new target.

    Only short questions with an ellipsis marker or a bare aspect qualify; anything else returns ``None``.
    """
    if not turns or not _is_short(query.strip()):
        return None
    text = query.strip()
    zh = bool(re.search(r"[\u4e00-\u9fff]", text))
    marker = bool(_ELLIPSIS_ZH.search(text) or _ELLIPSIS_EN.search(text))
    aspects_now = [match.group(0) for match in _ASPECT.finditer(text)]
    if not current_targets:
        if _MARKET_WIDE.search(text) or has_macro_content(text) or not (marker or aspects_now):
            return None
        targets = _last_targets(turns)
        if not targets:
            return None
        names = [str(entity.get("name") or entity["symbol"]) for entity in targets]
        if zh:
            joined = "和".join(names)
            rewritten = f"{joined}{_LEADING_ZH.sub('', text)}"
        else:
            joined = " and ".join(names)
            rest = _ELLIPSIS_EN.sub("", text).strip(" ,?.!") or text.rstrip("?.! ")
            rewritten = f"{rest} for {joined}?"
        return rewritten, f"ellipsis:target->{joined}"
    if len(current_targets) == 1 and marker and not aspects_now:
        aspects = _last_aspects(turns)
        if not aspects:
            return None
        name = str(current_targets[0].get("canonical_name") or current_targets[0].get("symbol"))
        rewritten = f"{name}的{'、'.join(aspects)}呢？" if zh else f"What is {name}'s {', '.join(aspects)}?"
        return rewritten, f"ellipsis:aspect->{'+'.join(aspects)}"
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
