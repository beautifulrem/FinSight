"""Shared Postgres plumbing for state that every replica must see: A2A tasks and agent traces.

Sessions already live in Postgres when ``QI_AGENT_CHECKPOINT_DB`` is a ``postgresql://`` DSN (``memory.py``).
The A2A task store (``a2a_store.py``) and the trace store (``trace_store.py``) follow the same switch by
default, so one setting makes a multi-replica deployment consistent. Each can be overridden:

* ``QI_A2A_TASK_DB`` / ``QI_AGENT_TRACE_DB``: a ``postgresql://`` DSN, or ``memory`` to keep the store in
  process even when sessions are in Postgres.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

_MEMORY = {"memory", "off", "none", "0", "false", "local"}
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def is_postgres_dsn(value: str | None) -> bool:
    return bool(value) and str(value).startswith(("postgres://", "postgresql://"))


def store_dsn(env_name: str) -> str | None:
    """The DSN for a shared store: ``env_name`` if set, else the checkpointer's DSN when it is Postgres."""
    explicit = os.getenv(env_name, "").strip()
    if explicit:
        if explicit.lower() in _MEMORY:
            return None
        if is_postgres_dsn(explicit):
            return explicit
        logger.warning("%s is not a postgresql:// DSN or 'memory'; using the in-process store", env_name)
        return None
    checkpoint = os.getenv("QI_AGENT_CHECKPOINT_DB", "").strip()
    return checkpoint if is_postgres_dsn(checkpoint) else None


def check_identifier(name: str) -> str:
    """Table names come from code or env; only plain lower-case identifiers are accepted."""
    if not _IDENTIFIER.match(name):
        raise ValueError(f"invalid table name: {name!r}")
    return name


def open_pool(dsn: str, *, name: str, max_size: int = 4, connect_timeout_s: float | None = None):
    """A small autocommit pool, opened eagerly so an unreachable database fails fast (``PoolTimeout`` after
    ``QI_STORE_CONNECT_TIMEOUT_S``, default 5 s)."""
    from psycopg_pool import ConnectionPool

    if connect_timeout_s is None:
        connect_timeout_s = float(os.getenv("QI_STORE_CONNECT_TIMEOUT_S", "5"))

    pool = ConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=max_size,
        kwargs={"autocommit": True},
        timeout=connect_timeout_s,
        name=name,
        open=False,
    )
    try:
        pool.open(wait=True, timeout=connect_timeout_s)
    except Exception:
        pool.close()
        raise
    return pool
