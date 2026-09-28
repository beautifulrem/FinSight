# Postgres checkpointer: two replicas sharing sessions (local run)

- Commit: `651721d` (only local change: an unrelated `ruff.toml` lint include)
- Date: 2026-09-28T09:15:48Z
- Postgres: `postgres:16-alpine` container (server 16.15) under colima on Apple silicon, fresh database `r3_record`
- DSN: `QI_TEST_POSTGRES_DSN=postgresql://postgres:***@127.0.0.1:55432/r3_record` (throwaway container password, redacted)
- Command: `python -m pytest -v -rA tests/test_agent_checkpoint_postgres.py tests/test_api_readiness.py::test_live_postgres_checkpointer_is_ready`
- Packages: langgraph-checkpoint-postgres 3.1.2, psycopg 3.3.6, psycopg-pool 3.3.3

What the test does: two `AgentService` instances ("replicas"), each with its own `PostgresSaver` and
connection pool on the same database. Replica A answers "贵州茅台的市盈率是多少"; replica B answers the
follow-up "它的市净率呢" in the same session and must resolve the pronoun from A's turn (route reason
`coreference`); A then reads back both turns. In a second session A pauses on a clarification
("它的市盈率呢"), B sees the pending interrupt and resumes it with "贵州茅台". The readiness test checks
that `/ready`'s checkpointer probe answers `SELECT 1` through the pool.

## pytest output

```text
============================= test session starts ==============================
platform darwin -- Python 3.13.13, pytest-9.1.1, pluggy-1.6.0 -- python
collecting ... collected 2 items

tests/test_agent_checkpoint_postgres.py::test_two_replicas_share_session_memory_and_clarifications PASSED [ 50%]
tests/test_api_readiness.py::test_live_postgres_checkpointer_is_ready PASSED [100%]

==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED tests/test_agent_checkpoint_postgres.py::test_two_replicas_share_session_memory_and_clarifications
PASSED tests/test_api_readiness.py::test_live_postgres_checkpointer_is_ready
============================== 2 passed in 2.81s ===============================
```

## Checkpoint rows written (fresh database, after the run)

```text
            thread_id             | checkpoint_ns | checkpoints |                latest                
----------------------------------+---------------+-------------+--------------------------------------
 16b62ff30fa341dfa32562f44a0cfa2e |               |           2 | 1f1bb1d2-9454-6aee-800e-8f5cd5758a1a
 e90e8c1951494729b64b071198c72fa9 |               |           2 | 1f1bb1d2-94a5-66d8-8008-cc2375d7f373
(2 rows)

 writes 
--------
      1
(1 row)
```

Two threads (the shared session and the clarification session), two checkpoints each: with
`QI_AGENT_DURABILITY=exit` (the default) one checkpoint is written per finished turn or interrupt.
The same test also passed in the local runs of the CI `tests` job steps against this container.
