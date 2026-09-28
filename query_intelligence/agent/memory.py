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

from .router import has_macro_content, is_dangling_why

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
    """One finished turn. ``entities`` are those of the *effective* question (after coreference/ellipsis
    rewrites), so a target introduced by "换成比亚迪呢" or carried by "ROE呢" counts as discussed.
    ``macro_topics`` are the macro indicators the turn was about (so "这说明什么？" after a CPI question stays on
    CPI); ``named`` marks turns whose targets the user typed (not carried by a rewrite), for "前者/后者"."""
    query = state.get("query", "")
    effective = state.get("effective_query") or query
    return {
        "query": query,
        "effective_query": effective,
        "route": result.get("route"),
        "answer": str(result.get("answer") or "")[:_HISTORY_ANSWER_CHARS],
        "entities": result.get("nlu_summary", {}).get("entities", []),
        "macro_topics": macro_topics_of(state.get("nlu") or {}, effective) if result.get("route") != "refuse" else [],
        "named": effective == query,
        "evidence_used": result.get("evidence_used", []),
    }


_MACRO_TOPIC_TERMS = (
    (re.compile(r"cpi|通胀|inflation|通缩|deflation", re.I), "CPI"),
    (re.compile(r"ppi", re.I), "PPI"),
    (re.compile(r"pmi", re.I), "PMI"),
    (re.compile(r"(?<![A-Za-z])m2(?![A-Za-z0-9])|货币供应|money supply", re.I), "M2"),
    (re.compile(r"lpr|贷款市场报价利率", re.I), "LPR"),
    (re.compile(r"国债|cgb|government bond|treasury|bond yield", re.I), "十年期国债收益率"),
    (re.compile(r"社融", re.I), "社融"),
    (re.compile(r"gdp", re.I), "GDP"),
)


def macro_topics_of(nlu: dict[str, Any], text: str) -> list[str]:
    """Macro indicators a question is about: NLU macro/policy entities, else lexical terms in the question."""
    topics = [
        str(entity.get("canonical_name"))
        for entity in nlu.get("entities") or []
        if entity.get("entity_type") in {"macro_indicator", "policy"} and entity.get("canonical_name")
    ]
    if not topics:
        topics = [topic for pattern, topic in _MACRO_TOPIC_TERMS if pattern.search(text or "")]
    return list(dict.fromkeys(topics))[:3]


def dialog_context_from_turns(turns: list[dict[str, Any]], explicit: list[dict[str, Any]] | None = None) -> list[dict]:
    """Previous user questions (oldest first) followed by any context supplied with the request."""
    context = [{"role": "user", "content": _effective(turn)} for turn in turns[-MAX_CONTEXT_TURNS:] if _effective(turn)]
    return [*context, *(explicit or [])]


def _effective(turn: dict[str, Any]) -> str:
    return str(turn.get("effective_query") or turn.get("query") or "")


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


_PLURAL_ZH = re.compile(r"这两家公司|这两家|这两只|这两个|两家公司|两只股票|两家|两只|两者|它们|他们俩|二者|俩")
_PLURAL_EN = re.compile(r"\b(?:both of them|both|them|these two|the two)\b", re.IGNORECASE)
# "三家里面哪家最便宜", "all three": the three most recently discussed targets.
_TRIPLE = re.compile(r"这三家|这三只|这三个|三家|三只|三者|\ball three\b|\bthe three\b|\bthese three\b", re.IGNORECASE)
# "哪家赚得多" with no count: the targets of the last turn that compared several.
_WHICH = re.compile(r"哪家|哪一家|哪只|哪一只|哪个|哪一个|\bwhich (?:one|company|stock|fund|of them)\b", re.IGNORECASE)
# "前者/后者", "the former/the latter": by the order in which the user named them.
_ORDINAL = re.compile(
    r"(?P<first>前者|前一个|前一家|第一个|第一家|\bthe former\b|\bthe first (?:one|company|stock)\b)|"
    r"(?P<last>后者|后一个|后一家|第二个|第二家|\bthe latter\b|\bthe second (?:one|company|stock)\b)",
    re.IGNORECASE,
)
# Leading discourse words that do not change the question ("Fine. What about Moutai's?", "OK 那 ROE 呢").
_FILLER = re.compile(
    r"^(?:ok(?:ay)?|fine|alright|all right|well|so|then|sure|嗯+|哦+|好的?|行吧?|算了|那好)(?:[\s,.，。!！:：]+|$)",
    re.IGNORECASE,
)


def strip_filler(query: str) -> str:
    text = (query or "").strip()
    for _ in range(2):
        stripped = _FILLER.sub("", text, count=1).strip()
        if stripped == text or not stripped:
            break
        text = stripped
    return text


def has_plural_reference(query: str) -> bool:
    return bool(_PLURAL_ZH.search(query) or _PLURAL_EN.search(query) or _TRIPLE.search(query))


def recent_entities(turns: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    """Distinct listed entities mentioned in the session, most recent first."""
    seen: dict[str, dict[str, Any]] = {}
    for turn in reversed(turns):
        for entity in turn.get("entities") or []:
            symbol = entity.get("symbol")
            if symbol and symbol not in seen:
                seen[symbol] = {"name": entity.get("name") or symbol, "symbol": symbol}
    return list(seen.values())[:limit]


def discussed_targets(turns: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """The ``limit`` most recently discussed distinct targets, oldest first.

    A target mentioned again moves to the end; within one turn the order is the order of mention, so
    "五粮液和中国平安" stays in that order (``recent_entities`` would reverse it).
    """
    order: dict[str, dict[str, Any]] = {}
    for turn in turns:
        for entity in turn.get("entities") or []:
            symbol = entity.get("symbol")
            if symbol:
                order.pop(symbol, None)
                order[symbol] = {"name": entity.get("name") or symbol, "symbol": symbol}
    return list(order.values())[-limit:]


def _last_group(turns: list[dict[str, Any]], *, named_only: bool = False) -> list[dict[str, Any]]:
    """Targets of the most recent turn (in the context window) that named two or more, in order of mention."""
    for turn in reversed(turns[-MAX_CONTEXT_TURNS:]):
        listed = [entity for entity in turn.get("entities") or [] if entity.get("symbol")]
        if len(listed) >= 2 and (turn.get("named", True) or not named_only):
            return [{"name": entity.get("name") or entity["symbol"], "symbol": entity["symbol"]} for entity in listed]
    return []


def _join(names: list[str], zh: bool) -> str:
    return "和".join(names) if zh else " and ".join(names)


def resolve_group_reference(query: str, turns: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Resolve references to a group of earlier targets: ``(rewritten, reason)``.

    * "前者/后者", "the former/the latter" pick one target of the last turn in which the user *named* two or
      more targets (a turn that only said "the two" has no order of its own).
    * "三家/all three" -> the three most recently discussed targets; "这两家/both/the two" -> two (see
      ``resolve_coreference``); a bare "哪家/which one" -> the targets of the last turn that named several.
    """
    if not turns:
        return None
    zh = bool(re.search(r"[\u4e00-\u9fff]", query))
    ordinal = _ORDINAL.search(query)
    if ordinal:
        group = _last_group(turns, named_only=True) or _last_group(turns)
        if len(group) < 2:
            return None
        chosen = group[0] if ordinal.group("first") else group[-1]
        word = ordinal.group(0)
        end = ordinal.end()
        possessive = not zh and query[end : end + 2] in {"'s", "’s"}
        replacement = f"{chosen['name']}'s" if possessive else chosen["name"]
        rewritten = f"{query[: ordinal.start()]}{replacement}{query[end + (2 if possessive else 0) :]}"
        return rewritten, f"group_reference:{word}->{chosen['name']}"
    triple = _TRIPLE.search(query)
    if triple:
        group = discussed_targets(turns, limit=3)
        if len(group) != 3:
            return None
        joined = _join([item["name"] for item in group], zh)
        rewritten = f"{query[: triple.start()]}{joined}{query[triple.end() :]}"
        return rewritten, f"group_reference:{triple.group(0)}->{joined}"
    if _WHICH.search(query) and not (_PLURAL_ZH.search(query) or _PLURAL_EN.search(query)):
        group = _last_group(turns)
        if len(group) < 2:
            return None
        joined = _join([item["name"] for item in group], zh)
        rewritten = f"{joined}{'' if zh else ': '}{query}"
        return rewritten, f"group_reference:which->{joined}"
    return None


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
        entities = discussed_targets(turns, limit=2)
        if len(entities) != 2:
            return None
        zh = bool(_PLURAL_ZH.search(query))
        joined = _join([entity["name"] for entity in entities], zh)
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
    r"市盈率|市净率|净资产收益率|营收|营业收入|净利润|净利|毛利率|股息率|市值|负债率|收盘价?|股价|走势|最高价?|最低价?|"
    r"开盘价?|成交量|成交额|涨跌幅?|估值|公告|新闻|分红|业绩|财报|舆情|均线|波动率|"
    r"(?<![A-Za-z])(?:P/?E|P/?B|ROE|RSI|MACD|MA\d+)(?![A-Za-z])|"
    r"revenue|net (?:profit|income)|gross margin|net margin|"
    r"dividend|market cap|valuation|\bprice\b|\bclos(?:e|es|ing price)\b|\bhigh\b|\blow\b|\bvolume\b|"
    r"percentage change|\breturn\b|\bgrowth\b|volatility|moving average|announcements?|news|trend",
    re.IGNORECASE,
)
# A follow-up that only changes the period ("2024年的呢", "And in 2022?") keeps the previous question's metric.
_PERIOD_ONLY = re.compile(
    r"^(?:(?:19|20)\d{2}\s*(?:年|财年|年度)?|今年|去年|前年|上一?年|[一二三四1-4]季度|第[一二三四]季度|上半年|下半年|"
    r"半年报?|年报|季报|in|for|fy|q[1-4]|the|first|second|third|fourth|quarter|half|last|this|year|的|\s)+$",
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
        aspects = list(dict.fromkeys(match.group(0) for match in _ASPECT.finditer(_effective(turn))))
        if aspects:
            return aspects[:3]
    return []


def resolve_ellipsis(
    query: str, turns: list[dict[str, Any]], current_targets: list[dict[str, Any]]
) -> tuple[str, str] | None:
    """Complete a short follow-up that leaves out the target or the question: ``(rewritten, reason)``.

    * No target named ("ROE呢", "最近走势怎么样", "And ROE?"): the targets of the most recent turn that
      named any are carried over, unless the question is market-wide or macro. When the follow-up only
      changes the period ("2024年的呢", "And in 2022?"), the previous question's metric is carried too.
    * Only a new target named ("换成五粮液呢", "What about BYD?"): the previous question's aspects
      (市盈率, 走势, ...) are carried over to the new target.

    Only short questions with an ellipsis marker or a bare aspect qualify; anything else returns ``None``.
    Leading discourse words ("Fine.", "OK", "算了") are ignored.
    """
    text = strip_filler(query)
    if not turns or not _is_short(text):
        return None
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
        rest_zh = _LEADING_ZH.sub("", text)
        rest_en = _ELLIPSIS_EN.sub("", text).strip(" ,?.!") or text.rstrip("?.! ")
        remainder = re.sub(r"[呢吗？?。.!！,，\s]", "", rest_zh if zh else rest_en)
        carried = _last_aspects(turns) if not aspects_now and (not remainder or _PERIOD_ONLY.match(remainder)) else []
        if zh:
            joined = "和".join(names)
            if carried:
                rewritten = f"{joined}{rest_zh.rstrip('呢吗？?。.!！ ')}{'、'.join(carried)}呢？"
            else:
                rewritten = f"{joined}{rest_zh}"
        else:
            joined = " and ".join(names)
            rewritten = f"{', '.join(carried)} {rest_en} for {joined}?" if carried else f"{rest_en} for {joined}?"
        reason = f"ellipsis:target->{joined}"
        return rewritten, reason + (f"+aspect->{'+'.join(carried)}" if carried else "")
    if len(current_targets) == 1 and marker and not aspects_now:
        aspects = _last_aspects(turns)
        if not aspects:
            return None
        name = str(current_targets[0].get("canonical_name") or current_targets[0].get("symbol"))
        rewritten = f"{name}的{'、'.join(aspects)}呢？" if zh else f"What is {name}'s {', '.join(aspects)}?"
        return rewritten, f"ellipsis:aspect->{'+'.join(aspects)}"
    return None


# "跟沪深300ETF比…", "Is that bigger than Moutai's?": a comparison that names only the new side.
_COMPARE_ZH = re.compile(
    r"(?:跟|与|和|同)\S{1,16}?(?:比|相比|对比)|比\S{1,12}?(?:高|低|大|小|多|少|贵|便宜|强|弱|好)|相比|对比"
)
_COMPARE_EN = re.compile(
    r"\bthan\b|\bcompared? (?:with|to)\b|\bversus\b|\bvs\.?(?=\s)|\brelative to\b|\bstack up against\b",
    re.IGNORECASE,
)
_BACK_REFERENCE = re.compile(r"^(?P<lead>.*?)\b(?P<ref>that|this|it)\b", re.IGNORECASE)


def resolve_comparison_anchor(
    query: str, turns: list[dict[str, Any]], current_targets: list[dict[str, Any]]
) -> tuple[str, str] | None:
    """A comparison naming one new target is a comparison with the target discussed before.

    "跟沪深300ETF比，最新收盘价分别是多少？" after a 创业板ETF turn compares both; "Is that bigger than Moutai's?"
    after "And its revenue?" (五粮液) compares 五粮液's revenue with Moutai's. The earlier target is added to the
    question, with the earlier aspect when the question names none. ``None`` without such a comparison.
    """
    text = strip_filler(query)
    if not turns or len(current_targets) != 1 or not (_COMPARE_ZH.search(text) or _COMPARE_EN.search(text)):
        return None
    new_symbol = current_targets[0].get("symbol")
    previous = [entity for entity in _last_targets(turns) if entity.get("symbol") != new_symbol]
    if not previous:
        return None
    zh = not re.search(r"[A-Za-z]{3,}", re.sub(r"(?i)ETF|ROE|P/?E|P/?B|MA\d+", "", text))
    names = [str(entity.get("name") or entity["symbol"]) for entity in previous]
    aspects = [] if _ASPECT.search(text) else _last_aspects(turns)
    joined = _join(names, zh)
    if zh:
        anchor = f"{joined}的{'、'.join(aspects)}" if aspects else joined
        rewritten = f"{anchor}{text}"
    else:
        anchor = f"{joined}'s {', '.join(aspects)}" if aspects else joined
        back = _BACK_REFERENCE.match(text)
        rewritten = f"{text[: back.start('ref')]}{anchor}{text[back.end('ref') :]}" if back else f"{anchor}: {text}"
    return rewritten, f"comparison_anchor:+{joined}"


_SESSION_FOLLOW_UP_CHARS = 30
_SESSION_FOLLOW_UP_WORDS = 12


def inherit_session_context(query: str, turns: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Anchor an entity-less follow-up to what the conversation is about: ``(rewritten, reason)``.

    Used only when the question names no target and would otherwise be refused or clarified. The question
    must be short, carry a finance cue ("为什么涨？", "增速是多少？", "What's the 3-day return?") and must not
    ask for an off-topic task (checked by the caller). It inherits whichever came last in the recent turns:
    the discussed targets, or the macro topic ("这说明什么？通缩压力大吗？" after a CPI question stays on CPI).
    """
    from .router import has_follow_up_cue

    text = strip_filler(query)
    if not turns or not text or not has_follow_up_cue(text):
        return None
    if re.search(r"[A-Za-z]{3,}", text):
        if len(text.split()) > _SESSION_FOLLOW_UP_WORDS:
            return None
    elif len(text) > _SESSION_FOLLOW_UP_CHARS:
        return None
    zh = bool(re.search(r"[\u4e00-\u9fff]", text))
    for turn in reversed(turns[-MAX_CONTEXT_TURNS:]):
        listed = [entity for entity in turn.get("entities") or [] if entity.get("symbol")]
        topics = [str(topic) for topic in turn.get("macro_topics") or []]
        if listed:
            names = [str(entity.get("name") or entity["symbol"]) for entity in listed[:3]]
            joined = _join(names, zh)
            rewritten = f"{joined}{text}" if zh else f"{text.rstrip('?.! ')} for {joined}?"
            return rewritten, f"session_inherit:target->{joined}"
        if topics:
            joined = "、".join(topics) if zh else ", ".join(topics)
            rewritten = f"{joined}：{text}" if zh else f"{text.rstrip('?.! ')} ({joined})?"
            return rewritten, f"session_inherit:macro->{joined}"
    return None


def resolve_dangling_why(query: str, turns: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Anchor "为什么会这样" / "why did that happen?" to the targets (and aspect) of the previous turn.

    "五粮液的营收增速呢" → "为什么会这样" becomes "五粮液的营收为什么会这样"; the rewritten question keeps its
    why-marker, so it is routed as a causal question about that stock. ``None`` when nothing was discussed.
    """
    text = query.strip()
    if not turns or not is_dangling_why(text):
        return None
    targets = _last_targets(turns)
    if not targets:
        return None
    names = [str(entity.get("name") or entity["symbol"]) for entity in targets]
    aspects = _last_aspects(turns[-1:])
    zh = bool(re.search(r"[\u4e00-\u9fff]", text))
    if zh:
        joined = "和".join(names)
        aspect = f"的{'、'.join(aspects)}" if aspects else ""
        rewritten = f"{joined}{aspect}{text}"
    else:
        joined = " and ".join(names)
        aspect = f" ({', '.join(aspects)})" if aspects else ""
        rewritten = f"{text.rstrip('?.! ')} for {joined}{aspect}?"
    return rewritten, f"dangling_why:target->{joined}"


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


_REPLY_FILLER = re.compile(
    r"^(?:我说的是|我是说|我指的是|我问的是|说的是|指的是|就是|是|i mean[t]?|i'm asking about|i am asking about|"
    r"i'm talking about|for|about|the)\s*",
    re.IGNORECASE,
)


def clarification_reply_text(reply: str) -> str:
    """The target a clarification reply names.

    "I mean the CSI 300 ETF." -> "CSI 300 ETF"; "我说的是五粮液" -> "五粮液".
    """
    text = reply.strip()
    for _ in range(3):
        stripped = _REPLY_FILLER.sub("", text, count=1).strip()
        if stripped == text or not stripped:
            break
        text = stripped
    return text.strip(" ,，.。!！?？:：") or reply.strip()


def apply_clarification(query: str, reply: str) -> tuple[str, str]:
    """Fold a clarification reply (e.g. "宁德时代") into the original question.

    The reply replaces the dangling pronoun when there is one ("它的市盈率呢" -> "宁德时代的市盈率呢");
    otherwise it is prepended so the NLU sees the entity. Returns ``(query, reason)``.
    """
    reply = clarification_reply_text(reply)
    possessive = _POSSESSIVE_EN.search(query)
    if possessive:
        return f"{query[: possessive.start()]}{reply}'s{query[possessive.end() :]}", f"clarified:{reply}"
    for pattern in (_PRONOUN_ZH, _PRONOUN_EN):
        match = pattern.search(query)
        if match:
            return f"{query[: match.start()]}{reply}{query[match.end() :]}", f"clarified:{reply}"
    separator = "" if re.search(r"[\u4e00-\u9fff]$", reply) else " "
    return f"{reply}{separator}{query}", f"clarified:{reply}"


def is_target_only_reply(reply: str, nlu: dict[str, Any]) -> bool:
    """True when a message only names a security ("五粮液", "I mean the CSI 300 ETF.", "For Wuliangye.").

    Such a message, sent while a clarification question is pending, answers that question rather than
    starting a new one. Anything left after removing the named targets must be filler, not a new question.
    """
    listed = listed_entities(nlu)
    if not listed or len(clarification_reply_text(reply)) > 40:
        return False
    rest = str(nlu.get("normalized_query") or reply)
    for entity in listed:
        for name in (entity.get("mention"), entity.get("canonical_name"), entity.get("symbol")):
            if name:
                rest = rest.replace(str(name), " ")
    rest = clarification_reply_text(rest) if rest.strip() else ""
    rest = re.sub(
        r"(?i)\b(?:the|a|an|one|stock|etf|fund|index|please|thanks?)\b|股票|基金|指数|吧|呀|啊|呢|哦", " ", rest
    )
    return not re.sub(r"[\s,，.。!！?？:：'’]", "", rest)
