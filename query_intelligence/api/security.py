"""Optional API hardening, configured only through environment variables.

* ``QI_API_KEYS``: comma-separated keys. When set, every endpoint except the probes ``GET /health`` and
  ``GET /ready``, the agent card and the browser page ``GET /`` requires ``X-API-Key: <key>`` or
  ``Authorization: Bearer <key>``.
* ``QI_RATE_LIMIT_PER_MINUTE``: per-client token bucket (client = the validated API key's principal,
  else the remote address, so made-up keys do not get fresh buckets); ``0`` disables it. Exceeding it
  returns 429 with ``Retry-After``. The bucket table is bounded (LRU, idle buckets dropped).
* ``QI_CORS_ORIGINS``: comma-separated allowed origins for browsers (``*`` allows any origin).
* ``QI_MAX_REQUEST_BYTES``: reject request bodies larger than this (default 1 MiB) with 413, whether
  the size is declared in ``Content-Length`` or only known while a chunked body streams in.

All are off by default so the local chatbot keeps working without configuration.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PUBLIC_PATHS = {
    ("GET", "/health"),
    ("GET", "/"),
    ("HEAD", "/health"),
    # Kubernetes readiness probes carry no credentials.
    ("GET", "/ready"),
    # A2A discovery: the agent card must be readable before a client can authenticate.
    ("GET", "/.well-known/agent-card.json"),
}
DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024
DEFAULT_MAX_RATE_CLIENTS = 10_000


@dataclass(frozen=True)
class SecuritySettings:
    api_keys: tuple[str, ...] = ()
    rate_limit_per_minute: int = 0
    cors_origins: tuple[str, ...] = ()
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES

    @classmethod
    def from_env(cls) -> SecuritySettings:
        def split(name: str) -> tuple[str, ...]:
            return tuple(item.strip() for item in os.getenv(name, "").split(",") if item.strip())

        return cls(
            api_keys=split("QI_API_KEYS"),
            rate_limit_per_minute=max(int(os.getenv("QI_RATE_LIMIT_PER_MINUTE", "0") or 0), 0),
            cors_origins=split("QI_CORS_ORIGINS"),
            max_request_bytes=int(os.getenv("QI_MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES))),
        )


@dataclass
class TokenBucket:
    """Per-client token buckets with bounded state.

    At most ``max_clients`` buckets are kept, least recently used first out. A bucket untouched for a full
    minute has refilled to capacity, which is the same as having no entry, so idle buckets are dropped
    first; evicting a still-draining bucket only happens under more than ``max_clients`` active clients.
    """

    rate_per_minute: int
    clock: callable = time.monotonic  # type: ignore[valid-type]
    max_clients: int = DEFAULT_MAX_RATE_CLIENTS
    _state: OrderedDict[str, tuple[float, float]] = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, client: str) -> float:
        """Consume one token; return 0 when allowed, else seconds until the next token."""
        capacity = float(self.rate_per_minute)
        refill_per_second = capacity / 60.0
        now = self.clock()
        with self._lock:
            tokens, updated = self._state.get(client, (capacity, now))
            tokens = min(capacity, tokens + (now - updated) * refill_per_second)
            if tokens >= 1.0:
                self._state[client] = (tokens - 1.0, now)
                wait = 0.0
            else:
                self._state[client] = (tokens, now)
                wait = (1.0 - tokens) / refill_per_second
            self._state.move_to_end(client)
            self._evict(now)
            return wait

    def _evict(self, now: float) -> None:
        while self._state:
            oldest, (_tokens, updated) = next(iter(self._state.items()))
            if len(self._state) > self.max_clients or now - updated >= 60.0:
                del self._state[oldest]
                continue
            break

    def __len__(self) -> int:
        return len(self._state)


Message = MutableMapping[str, Any]


class BodyLimitMiddleware:
    """Pure ASGI middleware that caps the request body while it streams in.

    ``Content-Length`` can be absent (``Transfer-Encoding: chunked``) or wrong, so the body is read here,
    counting bytes as they arrive, and the request is answered with 413 as soon as the cap is passed,
    without reading the rest. Accepted bodies (at most ``max_bytes``) are replayed to the application.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Message, receive: Callable[[], Awaitable[Message]], send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers") or []).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            await _too_large(scope, receive, send)
            return
        buffered: list[Message] = []
        size = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break  # client disconnected: let the application see it
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                await _too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            return buffered.pop(0) if buffered else await receive()

        await self.app(scope, replay, send)


async def _too_large(scope: Message, receive: Callable, send: Callable) -> None:
    response = JSONResponse({"detail": "request body too large"}, status_code=413, headers={"Connection": "close"})
    await response(scope, receive, send)


def _presented_key(request: Request) -> str | None:
    header = request.headers.get("x-api-key")
    if header:
        return header.strip()
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def install_security(app: FastAPI, settings: SecuritySettings | None = None) -> SecuritySettings:
    settings = settings or SecuritySettings.from_env()
    bucket = TokenBucket(settings.rate_limit_per_minute) if settings.rate_limit_per_minute else None

    @app.middleware("http")
    async def guard(request: Request, call_next):
        public = (
            (request.method, request.url.path) in PUBLIC_PATHS
            or request.method == "OPTIONS"
            or (request.method == "GET" and request.url.path.startswith("/static/"))
        )
        key = _presented_key(request)
        key_valid = bool(key) and any(hmac.compare_digest(key, allowed) for allowed in settings.api_keys)
        # Who is calling: sessions and traces are scoped to it (a hash, never the key itself).
        request.state.principal = f"key:{hashlib.sha256(key.encode()).hexdigest()[:12]}" if key_valid else "local"
        if settings.api_keys and not public and not key_valid:
            return JSONResponse(
                {"detail": "missing or invalid API key"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if bucket is not None and not public:
            # Only a validated key identifies a client; a missing or made-up key is limited by address, so
            # rotating random keys does not buy fresh buckets.
            address = request.client.host if request.client else "unknown"
            client = request.state.principal if key_valid else f"ip:{address}"
            wait = bucket.take(client)
            if wait > 0:
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(math.ceil(wait))},
                )
        return await call_next(request)

    # added after the http middleware above, so it wraps it: oversized bodies never reach auth or routing
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_request_bytes)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-API-Key"],
        )
    return settings


def principal_of(request: Request) -> str:
    """The caller identity set by the security middleware (``local`` when API keys are off)."""
    return str(getattr(request.state, "principal", "local"))
