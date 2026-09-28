from __future__ import annotations

from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app
from query_intelligence.api.security import SecuritySettings, TokenBucket


def _client(settings: SecuritySettings) -> TestClient:
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(
        service=stub,
        app_config={"deepseek": {"api_key": ""}},
        agent_service=AgentService(runtime, trace_sinks=[]),
        security=settings,
    )
    return TestClient(app)


def test_defaults_leave_api_open():
    client = _client(SecuritySettings())

    assert client.post("/agent/chat", json={"query": "今天天气怎么样"}).status_code == 200


def test_api_key_required_except_public_paths():
    client = _client(SecuritySettings(api_keys=("secret-1", "secret-2")))

    assert client.get("/health").status_code == 200
    missing = client.post("/agent/chat", json={"query": "今天天气怎么样"})
    wrong = client.post("/agent/chat", json={"query": "今天天气怎么样"}, headers={"X-API-Key": "nope"})
    header = client.post("/agent/chat", json={"query": "今天天气怎么样"}, headers={"X-API-Key": "secret-2"})
    bearer = client.post("/agent/chat", json={"query": "今天天气怎么样"}, headers={"Authorization": "Bearer secret-1"})

    assert missing.status_code == 401 and missing.headers["www-authenticate"] == "Bearer"
    assert wrong.status_code == 401
    assert header.status_code == 200 and bearer.status_code == 200


def test_rate_limit_returns_429_with_retry_after():
    client = _client(SecuritySettings(rate_limit_per_minute=2))

    statuses = [client.post("/agent/chat", json={"query": "今天天气怎么样"}).status_code for _ in range(3)]
    limited = client.post("/agent/chat", json={"query": "今天天气怎么样"})

    assert statuses == [200, 200, 429]
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
    assert client.get("/health").status_code == 200


def test_request_size_limit():
    client = _client(SecuritySettings(max_request_bytes=200))

    response = client.post(
        "/agent/chat", content=b'{"query": "' + b"x" * 500 + b'"}', headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 413


def test_cors_only_when_configured():
    open_client = _client(SecuritySettings())
    cors_client = _client(SecuritySettings(cors_origins=("https://app.example.com",)))
    preflight = {"Origin": "https://app.example.com", "Access-Control-Request-Method": "POST"}

    assert "access-control-allow-origin" not in open_client.options("/agent/chat", headers=preflight).headers
    allowed = cors_client.options("/agent/chat", headers=preflight)
    assert allowed.headers["access-control-allow-origin"] == "https://app.example.com"


def test_token_bucket_refills_over_time():
    now = {"t": 0.0}
    bucket = TokenBucket(60, clock=lambda: now["t"])

    assert all(bucket.take("c") == 0 for _ in range(60))
    assert bucket.take("c") == pytest.approx(1.0)
    now["t"] = 1.0
    assert bucket.take("c") == 0
    assert bucket.take("other") == 0


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("QI_API_KEYS", " a, b ,")
    monkeypatch.setenv("QI_RATE_LIMIT_PER_MINUTE", "30")
    monkeypatch.setenv("QI_CORS_ORIGINS", "*")
    monkeypatch.setenv("QI_MAX_REQUEST_BYTES", "2048")

    assert SecuritySettings.from_env() == SecuritySettings(
        api_keys=("a", "b"), rate_limit_per_minute=30, cors_origins=("*",), max_request_bytes=2048
    )


def test_chunked_body_over_the_limit_is_rejected_while_streaming():
    """B11 (round-2 review): a 2 MiB chunked POST bypassed the Content-Length check and got 200."""
    client = _client(SecuritySettings())  # default limit: 1 MiB
    sent = {"chunks": 0}

    def chunks():
        yield b'{"query": "'
        for _ in range(32):  # 32 x 64 KiB = 2 MiB, no Content-Length header
            sent["chunks"] += 1
            yield b"x" * 65536
        yield b'"}'

    response = client.post("/agent/chat", content=chunks(), headers={"Content-Type": "application/json"})

    assert response.status_code == 413
    assert response.json() == {"detail": "request body too large"}


def test_chunked_body_under_the_limit_is_replayed_intact():
    client = _client(SecuritySettings(max_request_bytes=4096))

    def chunks():
        yield b'{"query": '
        yield '"今天天气怎么样"}'.encode()

    response = client.post("/agent/chat", content=chunks(), headers={"Content-Type": "application/json"})

    assert response.status_code == 200 and response.json()["route"]


def test_made_up_keys_share_the_address_bucket_when_keys_are_off():
    """B23: rotating random X-API-Key values used to get a fresh bucket each time."""
    client = _client(SecuritySettings(rate_limit_per_minute=2))

    statuses = [
        client.post("/agent/chat", json={"query": "今天天气怎么样"}, headers={"X-API-Key": f"fake-{i}"}).status_code
        for i in range(4)
    ]

    assert statuses == [200, 200, 429, 429]


def test_valid_keys_get_their_own_bucket_and_invalid_keys_are_not_counted_per_key():
    client = _client(SecuritySettings(api_keys=("alpha", "beta"), rate_limit_per_minute=1))

    def status(key: str) -> int:
        return client.post("/agent/chat", json={"query": "今天天气怎么样"}, headers={"X-API-Key": key}).status_code

    assert [status("alpha"), status("alpha"), status("beta"), status("nope")] == [200, 429, 200, 401]


def test_token_bucket_state_is_bounded_and_idle_buckets_expire():
    now = {"t": 0.0}
    bucket = TokenBucket(60, clock=lambda: now["t"], max_clients=100)

    for i in range(1000):
        bucket.take(f"ip:{i}")
    assert len(bucket) == 100  # LRU bound

    bucket.take("busy")
    bucket.take("busy")
    now["t"] = 61.0
    bucket.take("fresh")
    assert len(bucket) == 1  # everything idle for a minute had refilled and was dropped
    for _ in range(59):
        bucket.take("fresh")
    assert bucket.take("fresh") > 0  # eviction never resets an active client
