"""A2A task store in Postgres, so any replica can serve ``GetTask``/``ListTasks`` and continue a task.

The SDK ships ``DatabaseTaskStore`` on SQLAlchemy + asyncpg; FinSight already depends on psycopg (the
LangGraph checkpointer uses it), so this store implements the same ``TaskStore`` contract on a small psycopg
pool instead of adding a second database stack. Semantics follow the SDK's stores:

* tasks are scoped by owner (``ServerCallContext.user.user_name``; FinSight sets it to the API-key principal),
  so one caller never sees another caller's task;
* the task is stored whole (``Task`` protobuf as JSON), with ``context_id``, ``state`` and ``last_updated``
  columns for filtering and ordering;
* ``list`` orders by status timestamp (newest first, then id) and pages with the SDK's page tokens.

Blocking psycopg calls run in a worker thread (``asyncio.to_thread``), so the store is not tied to one event
loop. Tasks untouched for ``QI_A2A_TASK_RETENTION_DAYS`` (default 7) are pruned at most every 10 minutes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import UTC
from typing import Any

from a2a.server.context import ServerCallContext
from a2a.server.owner_resolver import OwnerResolver, resolve_user_scope
from a2a.server.tasks.task_store import TaskStore
from a2a.types import a2a_pb2
from a2a.types.a2a_pb2 import Task
from a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
from a2a.utils.errors import InvalidParamsError
from a2a.utils.task import decode_page_token, encode_page_token
from google.protobuf.json_format import MessageToDict, ParseDict

from .pg import check_identifier

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "finsight_a2a_tasks"
_PRUNE_INTERVAL_S = 600.0


class PostgresTaskStore(TaskStore):
    def __init__(
        self,
        pool: Any,
        *,
        table: str = DEFAULT_TABLE,
        owner_resolver: OwnerResolver = resolve_user_scope,
        retention_days: float | None = None,
    ) -> None:
        self.pool = pool
        self.table = check_identifier(table)
        self.owner_resolver = owner_resolver
        days = retention_days if retention_days is not None else float(os.getenv("QI_A2A_TASK_RETENTION_DAYS", "7"))
        self.retention_s = max(days, 0.0) * 86400
        self._last_prune = 0.0
        self._prune_lock = threading.Lock()
        self.setup()

    def setup(self) -> None:
        t = self.table
        with self.pool.connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    owner TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    last_updated TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    task JSONB NOT NULL,
                    PRIMARY KEY (owner, task_id)
                )"""
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_order ON {t} (owner, last_updated DESC NULLS LAST, task_id DESC)"
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_context ON {t} (owner, context_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_updated ON {t} (updated_at)")

    # ------------------------------------------------------------------------------------------ TaskStore

    async def save(self, task: Task, context: ServerCallContext) -> None:
        await asyncio.to_thread(self._save, task, self.owner_resolver(context))

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        return await asyncio.to_thread(self._get, task_id, self.owner_resolver(context))

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        await asyncio.to_thread(self._delete, task_id, self.owner_resolver(context))

    async def list(self, params: a2a_pb2.ListTasksRequest, context: ServerCallContext) -> a2a_pb2.ListTasksResponse:
        return await asyncio.to_thread(self._list, params, self.owner_resolver(context))

    # ------------------------------------------------------------------------------------------ blocking

    def _save(self, task: Task, owner: str) -> None:
        from psycopg.types.json import Jsonb

        last_updated = task.status.timestamp.ToDatetime(tzinfo=UTC) if task.status.HasField("timestamp") else None
        with self.pool.connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table} (owner, task_id, context_id, state, last_updated, updated_at, task)
                VALUES (%s, %s, %s, %s, %s, now(), %s)
                ON CONFLICT (owner, task_id) DO UPDATE SET
                    context_id = EXCLUDED.context_id, state = EXCLUDED.state,
                    last_updated = EXCLUDED.last_updated, updated_at = now(), task = EXCLUDED.task
                """,
                (
                    owner,
                    task.id,
                    task.context_id,
                    a2a_pb2.TaskState.Name(task.status.state),
                    last_updated,
                    Jsonb(MessageToDict(task)),
                ),
            )
        self._maybe_prune()

    def _get(self, task_id: str, owner: str) -> Task | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                f"SELECT task FROM {self.table} WHERE owner = %s AND task_id = %s", (owner, task_id)
            ).fetchone()
        return _task(row[0]) if row else None

    def _delete(self, task_id: str, owner: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(f"DELETE FROM {self.table} WHERE owner = %s AND task_id = %s", (owner, task_id))

    def _list(self, params: a2a_pb2.ListTasksRequest, owner: str) -> a2a_pb2.ListTasksResponse:
        where = ["owner = %s"]
        args: list[Any] = [owner]
        if params.context_id:
            where.append("context_id = %s")
            args.append(params.context_id)
        if params.status:
            where.append("state = %s")
            args.append(a2a_pb2.TaskState.Name(params.status))
        if params.HasField("status_timestamp_after"):
            where.append("last_updated >= %s")
            args.append(params.status_timestamp_after.ToDatetime(tzinfo=UTC))
        page_size = params.page_size or DEFAULT_LIST_TASKS_PAGE_SIZE
        with self.pool.connection() as conn:
            total = conn.execute(f"SELECT count(*) FROM {self.table} WHERE {' AND '.join(where)}", args).fetchone()[0]
            page_where, page_args = list(where), list(args)
            if params.page_token:
                start_id = decode_page_token(params.page_token)
                start = conn.execute(
                    f"SELECT last_updated FROM {self.table} WHERE owner = %s AND task_id = %s", (owner, start_id)
                ).fetchone()
                if start is None:
                    raise InvalidParamsError(f"Invalid page token: {params.page_token}")
                if start[0] is not None:
                    page_where.append(
                        "((last_updated = %s AND task_id <= %s) OR last_updated < %s OR last_updated IS NULL)"
                    )
                    page_args.extend([start[0], start_id, start[0]])
                else:
                    page_where.append("(last_updated IS NULL AND task_id <= %s)")
                    page_args.append(start_id)
            rows = conn.execute(
                f"SELECT task FROM {self.table} WHERE {' AND '.join(page_where)} "
                "ORDER BY last_updated DESC NULLS LAST, task_id DESC LIMIT %s",
                [*page_args, page_size + 1],
            ).fetchall()
        tasks = [_task(row[0]) for row in rows]
        next_token = encode_page_token(tasks[page_size].id) if len(tasks) > page_size else None
        return a2a_pb2.ListTasksResponse(
            tasks=tasks[:page_size], total_size=total, next_page_token=next_token, page_size=page_size
        )

    def _maybe_prune(self) -> None:
        if not self.retention_s or time.monotonic() - self._last_prune < _PRUNE_INTERVAL_S:
            return
        if not self._prune_lock.acquire(blocking=False):
            return
        try:
            self._last_prune = time.monotonic()
            with self.pool.connection() as conn:
                deleted = conn.execute(
                    f"DELETE FROM {self.table} WHERE updated_at < now() - make_interval(secs => %s)",
                    (self.retention_s,),
                ).rowcount
            if deleted:
                logger.info("pruned %d A2A task(s) older than %g days", deleted, self.retention_s / 86400)
        except Exception:  # pruning is housekeeping; never fail a save for it
            logger.warning("A2A task pruning failed", exc_info=True)
        finally:
            self._prune_lock.release()

    def close(self) -> None:
        self.pool.close()


def _task(data: dict[str, Any]) -> Task:
    task = Task()
    ParseDict(data, task, ignore_unknown_fields=True)
    return task
