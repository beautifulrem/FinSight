"""Optional API hardening, configured only through environment variables.

* ``QI_API_KEYS``: comma-separated keys. When set, every endpoint except the probes ``GET /health`` and
  ``GET /ready``, the agent card and the browser page ``GET /`` requires ``X-API-Key: <key>`` or
  ``Authorization: Bearer <key>``.
* ``QI_RATE_LIMIT_PER_MINUTE``: per-client token bucket (client = the validated API key's principal,
  else the remote address, so made-up keys do not get fresh buckets); ``0`` disables it. Exceeding it
  returns 429 with ``Retry-After``. Shared across replicas through Postgres when ``QI_RATE_LIMIT_DB`` (or the
  checkpointer's ``QI_AGENT_CHECKPOINT_DB``) is a Postgres DSN, else per replica in process with a bounded
  table (LRU, idle buckets dropped); see ``rate_limit.py``.
* ``QI_CORS_ORIGINS``: comma-separated allowed origins for browsers (``*`` allows any origin).
* ``QI_MAX_REQUEST_BYTES``: reject request bodies larger than this (default 1 MiB) with 413, whether
  the size is declared in ``Content-Length`` or only known while a chunked body streams in.

All are off by default so the local chatbot keeps working without configuration.

Anonymous callers (C3, round-3 review)
--------------------------------------
* ``QI_PROFILE=production`` (set by the Kubernetes manifest): the app refuses to start without ``QI_API_KEYS``
  unless ``QI_ALLOW_ANONYMOUS=1`` explicitly opts in to anonymous access.
* A caller without a valid key is never the shared ``local`` principal. Each browser gets its own anonymous
  principal from an HMAC-signed, HttpOnly cookie (``finsight_anon``), so sessions it creates are invisible to
  every other caller. The signing secret is ``QI_ANON_COOKIE_SECRET`` (set it to the same value on every
  replica); without it a random per-process secret is used, and a browser that lands on another replica starts
  a fresh anonymous identity. A client that does not keep cookies gets a new identity on every request.
* Anonymous principals cannot list or read traces (``/agent/traces*`` answer 403): traces carry queries and
  session ids, so they need an API key.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import secrets
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .rate_limit import DEFAULT_MAX_RATE_CLIENTS, TokenBucket, build_rate_limiter, rate_limit_dsn

__all__ = ["DEFAULT_MAX_RATE_CLIENTS", "TokenBucket"]  # re-exported: TokenBucket moved to rate_limit.py

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
ANON_COOKIE = "finsight_anon"
ANON_PREFIX = "anon:"
ANON_COOKIE_MAX_AGE = 30 * 24 * 3600
PRODUCTION_PROFILES = {"production", "prod"}
_TRUE = {"1", "true", "yes", "on"}


class InsecureConfigurationError(RuntimeError):
    """Raised at start-up when a production deployment would run without authentication."""


@dataclass(frozen=True)
class SecuritySettings:
    api_keys: tuple[str, ...] = ()
    rate_limit_per_minute: int = 0
    cors_origins: tuple[str, ...] = ()
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    profile: str = "development"
    allow_anonymous: bool = False
    anon_cookie_secret: str = ""
    # Postgres DSN for the shared rate limiter ("" = per replica, in process); see rate_limit.py.
    rate_limit_db: str = ""

    @property
    def production(self) -> bool:
        return self.profile.strip().lower() in PRODUCTION_PROFILES

    def validate(self) -> None:
        """Refuse an unauthenticated production deployment unless anonymous access is an explicit choice."""
        if self.production and not self.api_keys and not self.allow_anonymous:
            raise InsecureConfigurationError(
                "QI_PROFILE=production requires QI_API_KEYS (or QI_ALLOW_ANONYMOUS=1 to accept anonymous callers); "
                "refusing to start"
            )

    @classmethod
    def from_env(cls) -> SecuritySettings:
        def split(name: str) -> tuple[str, ...]:
            return tuple(item.strip() for item in os.getenv(name, "").split(",") if item.strip())

        return cls(
            api_keys=split("QI_API_KEYS"),
            rate_limit_per_minute=max(int(os.getenv("QI_RATE_LIMIT_PER_MINUTE", "0") or 0), 0),
            cors_origins=split("QI_CORS_ORIGINS"),
            max_request_bytes=int(os.getenv("QI_MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES))),
            profile=os.getenv("QI_PROFILE", "development").strip() or "development",
            allow_anonymous=os.getenv("QI_ALLOW_ANONYMOUS", "").strip().lower() in _TRUE,
            anon_cookie_secret=os.getenv("QI_ANON_COOKIE_SECRET", ""),
            rate_limit_db=rate_limit_dsn() or "",
        )


Message = MutableMapping[str, Any]


class BodyLimitMiddleware:
    """Pure ASGI middleware that caps the request body while it streams in.

    ``Content-Length`` can be absent (``Transfer-Encoding: chunked``) or wrong, so the body is read here,
    counting bytes as they arrive, and the request is answered with 413 as soon as the cap is passed,
    without reading the rest. Accepted bodies (at most ``max_bytes``) are replayed to the application.

    Starlette's own ``RequestBodyLimitMiddleware`` is lazy (it raises when the endpoint reads the body); behind the
    ``BaseHTTPMiddleware`` security layer that surfaced as 400 / a JSON-RPC 200 instead of 413, so this one stays
    (docs/deployment.md, "Request body limit").
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


class AnonymousIdentity:
    """Per-browser anonymous principals from an HMAC-signed cookie ``<id>.<signature>``."""

    def __init__(self, secret: str = "") -> None:
        self._secret = (secret or secrets.token_hex(32)).encode()

    def _sign(self, ident: str) -> str:
        return hmac.new(self._secret, ident.encode(), hashlib.sha256).hexdigest()[:32]

    def issue(self) -> tuple[str, str]:
        """A new ``(id, cookie value)``."""
        ident = secrets.token_hex(16)
        return ident, f"{ident}.{self._sign(ident)}"

    def verify(self, cookie: str | None) -> str | None:
        """The id of a correctly signed cookie, else ``None``."""
        ident, _, signature = (cookie or "").partition(".")
        if len(ident) != 32 or not signature or not hmac.compare_digest(signature, self._sign(ident)):
            return None
        return ident

    @staticmethod
    def principal(ident: str) -> str:
        return f"{ANON_PREFIX}{hashlib.sha256(ident.encode()).hexdigest()[:12]}"


def is_anonymous(principal: str) -> bool:
    return principal.startswith(ANON_PREFIX)


def install_security(app: FastAPI, settings: SecuritySettings | None = None) -> SecuritySettings:
    settings = settings or SecuritySettings.from_env()
    settings.validate()
    bucket = build_rate_limiter(settings.rate_limit_per_minute, dsn=settings.rate_limit_db)
    app.state.rate_limiter = bucket  # closed with the other shared stores at shutdown
    anonymous = AnonymousIdentity(settings.anon_cookie_secret)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        public = (
            (request.method, request.url.path) in PUBLIC_PATHS
            or request.method == "OPTIONS"
            or (request.method == "GET" and request.url.path.startswith("/static/"))
        )
        key = _presented_key(request)
        key_valid = bool(key) and any(hmac.compare_digest(key, allowed) for allowed in settings.api_keys)
        # Who is calling: sessions and traces are scoped to it (a hash, never the key itself). Without a valid
        # key the caller is an anonymous principal of its own (signed cookie), never a shared one.
        new_cookie = None
        if key_valid:
            request.state.principal = f"key:{hashlib.sha256(key.encode()).hexdigest()[:12]}"
        else:
            ident = anonymous.verify(request.cookies.get(ANON_COOKIE))
            if ident is None:
                ident, new_cookie = anonymous.issue()
            request.state.principal = anonymous.principal(ident)
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
        if is_anonymous(request.state.principal) and request.url.path.startswith("/agent/traces"):
            return JSONResponse({"detail": "traces require an API key"}, status_code=403)
        response = await call_next(request)
        path = request.url.path
        if new_cookie is not None and path not in {"/health", "/ready"} and not path.startswith("/static/"):
            response.set_cookie(
                ANON_COOKIE,
                new_cookie,
                max_age=ANON_COOKIE_MAX_AGE,
                httponly=True,
                samesite="lax",
                secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
                path="/",
            )
        return response

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
    """The caller identity set by the security middleware: ``key:<hash>`` or ``anon:<hash>`` (``local`` only
    for callers that bypass the middleware, such as in-process use)."""
    return str(getattr(request.state, "principal", "local"))
