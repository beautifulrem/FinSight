"""Optional API hardening, configured only through environment variables.

* ``QI_API_KEYS``: comma-separated keys. When set, every endpoint except ``GET /health`` and the
  browser page ``GET /`` requires ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
* ``QI_RATE_LIMIT_PER_MINUTE``: per-client token bucket (client = API key, else remote address);
  ``0`` disables it. Exceeding it returns 429 with ``Retry-After``.
* ``QI_CORS_ORIGINS``: comma-separated allowed origins for browsers (``*`` allows any origin).
* ``QI_MAX_REQUEST_BYTES``: reject request bodies larger than this (default 1 MiB) with 413.

All are off by default so the local chatbot keeps working without configuration.
"""

from __future__ import annotations

import hmac
import math
import os
import threading
import time
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PUBLIC_PATHS = {("GET", "/health"), ("GET", "/"), ("HEAD", "/health")}
DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024


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
    rate_per_minute: int
    clock: callable = time.monotonic  # type: ignore[valid-type]
    _state: dict[str, tuple[float, float]] = field(default_factory=dict)
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
                return 0.0
            self._state[client] = (tokens, now)
            return (1.0 - tokens) / refill_per_second


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
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_request_bytes:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        key = _presented_key(request)
        key_valid = bool(key) and any(hmac.compare_digest(key, allowed) for allowed in settings.api_keys)
        if settings.api_keys and not public and not key_valid:
            return JSONResponse(
                {"detail": "missing or invalid API key"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if bucket is not None and not public:
            client = f"key:{key}" if key else f"ip:{request.client.host if request.client else 'unknown'}"
            wait = bucket.take(client)
            if wait > 0:
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(math.ceil(wait))},
                )
        return await call_next(request)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-API-Key"],
        )
    return settings
