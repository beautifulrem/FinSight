# Shared rate limiter across replicas (round 4)

**Question.** With `QI_RATE_LIMIT_PER_MINUTE=N` and R replicas, does a client get N requests per minute, or N×R?

**Before.** The token bucket lived in process (`TokenBucket`), so each replica had its own bucket: up to N×R.

**Change.** `query_intelligence/api/rate_limit.py`. When `QI_RATE_LIMIT_DB` is a Postgres DSN, or it is unset and `QI_AGENT_CHECKPOINT_DB` is one, every replica uses `PostgresTokenBucket`:

- one row per client in `finsight_rate_buckets`;
- one atomic `INSERT … ON CONFLICT DO UPDATE … RETURNING` per request. The row lock serialises concurrent requests for the same client, and the time comes from the database clock (`clock_timestamp()`), so clock skew between replicas does not matter;
- idle rows (older than 60 s, when the bucket would be full anyway) are deleted at most once a minute.

Without a DSN, or with `QI_RATE_LIMIT_DB=memory`, the in-process bucket is used as before. If the database cannot be reached at start-up, or a call fails later, the replica limits with its own in-process bucket and logs a warning at most once a minute. Rate limiting degrades; the API stays up.

**Test.** Local `postgres:16-alpine` (PostgreSQL 16.15), `QI_TEST_POSTGRES_DSN=postgresql://postgres:finsight@127.0.0.1:55439/finsight`, 2026-09-29:

```
pytest tests/test_rate_limit.py tests/test_rate_limit_postgres.py tests/test_api_security.py
26 passed
```

| Test | Setup | Result |
|---|---|---|
| `test_replicas_share_one_bucket_per_client` | Two `create_app` instances, one API key, limit 4/min, 6 requests alternating between replicas | `200 200 200 200 429 429`, plus `Retry-After ≥ 1`. In-process buckets would give 6 × 200. |
| `test_concurrent_takes_across_two_limiters_never_overspend` | Two limiters with separate pools, capacity 30, frozen clock, 6 threads × 20 takes | Exactly 30 of 120 allowed. After 2 s, one more token: the next take is allowed and the one after is limited. |
| `tests/test_rate_limit.py` (no database) | Unreachable DSN at start-up; failing pool after start-up | Falls back to the in-process bucket. It still limits, and the warning is logged once. |

The test skips when `QI_TEST_POSTGRES_DSN` is unset.

**Limits.**
- One extra database round trip per rate-limited request (a single-row upsert).
- During a database outage, the limit is per replica again.
- Clients are the validated key's principal, or the remote address for anonymous and invalid-key callers. Behind a proxy that address is the proxy's; `X-Forwarded-For` is deliberately not trusted.
