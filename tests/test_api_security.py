from __future__ import annotations

from datetime import date

import pytest
from agent_fakes import StubService, build_fake_registry
from fastapi.testclient import TestClient

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app
from query_intelligence.api.security import (
    ANON_COOKIE,
    AnonymousIdentity,
    InsecureConfigurationError,
    SecuritySettings,
    TokenBucket,
)


def _app(settings: SecuritySettings):
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    return create_app(
        service=stub,
        app_config={"deepseek": {"api_key": ""}},
        agent_service=AgentService(runtime, trace_sinks=[]),
        security=settings,
    )


def _client(settings: SecuritySettings) -> TestClient:
    return TestClient(_app(settings))


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
    monkeypatch.setenv("QI_PROFILE", "production")
    monkeypatch.setenv("QI_ALLOW_ANONYMOUS", "1")
    monkeypatch.setenv("QI_ANON_COOKIE_SECRET", "s3cret")

    assert SecuritySettings.from_env() == SecuritySettings(
        api_keys=("a", "b"),
        rate_limit_per_minute=30,
        cors_origins=("*",),
        max_request_bytes=2048,
        profile="production",
        allow_anonymous=True,
        anon_cookie_secret="s3cret",
    )
    for name in ("QI_PROFILE", "QI_ALLOW_ANONYMOUS", "QI_ANON_COOKIE_SECRET"):
        monkeypatch.delenv(name)
    settings = SecuritySettings.from_env()
    assert settings.profile == "development" and not settings.production and not settings.allow_anonymous


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


# ---- C3 (round-3 review): no shared anonymous principal, production refuses to run without keys ----


def test_production_profile_refuses_to_start_without_api_keys():
    with pytest.raises(InsecureConfigurationError, match="QI_API_KEYS"):
        _app(SecuritySettings(profile="production"))
    with pytest.raises(InsecureConfigurationError):
        _app(SecuritySettings(profile="prod", allow_anonymous=False))

    assert _app(SecuritySettings(profile="production", api_keys=("k",)))  # keys: starts
    assert _app(SecuritySettings(profile="production", allow_anonymous=True))  # explicit opt-in: starts
    assert _app(SecuritySettings())  # development default: starts (local use)


def test_anonymous_callers_are_scoped_per_browser():
    app = _app(SecuritySettings())
    alice, bob = TestClient(app), TestClient(app)

    first = alice.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": "anon-own-1"})
    assert first.status_code == 200
    cookie = first.cookies.get(ANON_COOKIE)
    assert cookie and "httponly" in first.headers["set-cookie"].lower()
    assert "samesite=lax" in first.headers["set-cookie"].lower()

    # another browser cannot read or continue that session
    assert bob.get("/agent/sessions/anon-own-1").status_code == 404
    follow_up = {"query": "它的市净率呢", "session_id": "anon-own-1"}
    assert bob.post("/agent/chat", json=follow_up).status_code == 404
    # the owner can, and keeps the same identity (no new cookie)
    own = alice.get("/agent/sessions/anon-own-1")
    assert own.status_code == 200 and own.json()["turns"]
    assert ANON_COOKIE not in own.cookies


def test_unknown_and_foreign_sessions_are_indistinguishable():
    """No session-existence oracle: an id nobody used and another caller's id give the same 404 body
    on read and on resume, after the same checkpointer reads."""
    stub = StubService()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    agent = AgentService(runtime, trace_sinks=[])
    app = create_app(
        service=stub,
        app_config={"deepseek": {"api_key": ""}},
        agent_service=agent,
        security=SecuritySettings(api_keys=("key-a", "key-b")),
    )
    client = TestClient(app)
    a, b = {"X-API-Key": "key-a"}, {"X-API-Key": "key-b"}
    first = client.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": "oracle-a"}, headers=a)
    assert first.status_code == 200
    reads: list[str] = []
    for name in ("owner_of", "history", "pending_clarification"):
        original = getattr(agent, name)
        setattr(agent, name, lambda sid, _o=original, _n=name: reads.append(_n) or _o(sid))

    foreign = client.get("/agent/sessions/oracle-a", headers=b)
    foreign_reads, reads[:] = list(reads), []
    unknown = client.get("/agent/sessions/oracle-nobody", headers=b)
    unknown_reads = list(reads)

    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == {"detail": "session oracle-a not found"}
    assert unknown.json() == {"detail": "session oracle-nobody not found"}
    assert foreign_reads == unknown_reads
    for sid in ("oracle-a", "oracle-nobody"):
        resumed = client.post("/agent/resume", json={"session_id": sid, "reply": "贵州茅台"}, headers=b)
        assert resumed.status_code == 404 and resumed.json() == {"detail": f"session {sid} not found"}
    # the owner still reads its session; a known session with nothing pending stays a 409 for its owner
    assert client.get("/agent/sessions/oracle-a", headers=a).json()["turns"]
    assert client.post("/agent/resume", json={"session_id": "oracle-a", "reply": "x"}, headers=a).status_code == 409


def test_a_forged_or_tampered_anonymous_cookie_gets_a_fresh_identity():
    app = _app(SecuritySettings(anon_cookie_secret="server-secret"))
    alice = TestClient(app)
    alice.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少", "session_id": "anon-own-2"})
    ident = alice.cookies.get(ANON_COOKIE).split(".")[0]

    for forged in (f"{ident}.{'0' * 32}", f"{ident}.", ident, AnonymousIdentity("other").issue()[1]):
        mallory = TestClient(app, cookies={ANON_COOKIE: forged})
        assert mallory.get("/agent/sessions/anon-own-2").status_code == 404

    # the same secret on another replica accepts the cookie (QI_ANON_COOKIE_SECRET shared across pods)
    replica = AnonymousIdentity("server-secret")
    assert replica.verify(alice.cookies.get(ANON_COOKIE)) == ident


def test_anonymous_callers_never_see_traces():
    app = _app(SecuritySettings(api_keys=(), allow_anonymous=True))
    client = TestClient(app)
    answer = client.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少"}).json()

    assert client.get("/agent/traces").status_code == 403
    assert client.get(f"/agent/traces/{answer['trace_id']}").status_code == 403
    assert TestClient(app).get("/agent/traces").status_code == 403


def test_keyed_callers_still_list_their_own_traces():
    client = _client(SecuritySettings(api_keys=("k1",)))
    headers = {"X-API-Key": "k1"}
    answer = client.post("/agent/chat", json={"query": "贵州茅台的市盈率是多少"}, headers=headers).json()

    listing = client.get("/agent/traces", headers=headers)
    assert listing.status_code == 200
    assert [item["trace_id"] for item in listing.json()["traces"]] == [answer["trace_id"]]
    assert ANON_COOKIE not in listing.cookies


def test_probes_and_static_files_do_not_set_the_anonymous_cookie():
    client = _client(SecuritySettings())

    assert ANON_COOKIE not in client.get("/health").cookies
    assert ANON_COOKIE in client.get("/").cookies  # the page itself starts the browser's identity


def test_kubernetes_manifest_requires_api_keys_from_a_secret():
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    documents = [doc for doc in yaml.safe_load_all((root / "deploy/k8s/finsight.yaml").read_text()) if doc]
    config = next(doc for doc in documents if doc["kind"] == "ConfigMap")
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container.get("env") or []}

    assert config["data"]["QI_PROFILE"] == "production"
    assert "QI_ALLOW_ANONYMOUS" not in config["data"] and "QI_ALLOW_ANONYMOUS" not in env
    assert "QI_API_KEYS" not in config["data"]
    keys = env["QI_API_KEYS"]["valueFrom"]["secretKeyRef"]
    assert keys["name"] == "finsight-api-keys" and keys["key"] == "QI_API_KEYS"
    assert keys.get("optional", False) is False  # no Secret, no pod: never anonymous by accident
    assert env["QI_ANON_COOKIE_SECRET"]["valueFrom"]["secretKeyRef"]["name"] == "finsight-api-keys"
    template = [doc for doc in yaml.safe_load_all((root / "deploy/k8s/secret.template.yaml").read_text()) if doc]
    assert {doc["metadata"]["name"] for doc in template} == {"finsight-db", "finsight-api-keys"}
