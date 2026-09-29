"""Consume external MCP servers as agent tools (MCP client side).

``mcp_server.py`` publishes FinSight's tools to other agents; this module does the opposite: it connects to
external MCP servers listed in ``QI_MCP_SERVERS`` and registers each remote tool in the ``ToolRegistry``, so
the LLM agent can call it like a local tool and every call gets the same timeout, error normalisation,
tracing and metrics.

* Names are namespaced ``mcp__<server>__<tool>``. (``mcp:<server>:<tool>`` would read better, but OpenAI-style
  function names only allow ``[A-Za-z0-9_-]``, so colons would be rejected by the LLM providers.)
* The remote JSON schema is published unchanged to the LLM (descriptions sanitised) and arguments are
  validated against it with ``jsonschema`` before the call leaves the process.
* Remote output is untrusted third-party data. Every string in it goes through the same injection filter as
  local document text (``injection.sanitize_untrusted_text``), strings and lists are size-capped, and the
  agent wraps the observation in the usual untrusted-data envelope (``injection.tool_message_content``).
  Tool descriptions are sanitised too, because a poisoned description reaches the LLM's tool list.
* Each call is bounded by the server's ``timeout_s`` (sent to the server as the request timeout and enforced
  locally). Remote tools are not retried: a slow external server should cost one timeout, not three.
* Each result becomes one evidence item (``mcp_<server>_<tool>_<args hash>``) so answers can cite it and the
  verifier can trace numbers to it.

Off by default. ``QI_MCP_SERVERS`` is either inline JSON or a path to a JSON file, in the ``mcpServers`` shape
used by desktop MCP clients::

    {"calendar": {"command": "python", "args": ["tests/fixtures/mcp_trading_calendar_server.py"]},
     "filings": {"url": "http://127.0.0.1:9000/mcp", "headers": {"Authorization": "Bearer ${FILINGS_TOKEN}"},
                 "timeout_s": 10, "tools": ["search_filings"]}}

``${VAR}`` in ``env`` and ``headers`` values is expanded from the environment, so secrets stay out of config.
A server that cannot be reached at start-up is skipped with a warning; the agent keeps its local tools.

Each server gets one long-lived client session on a private event-loop thread (stdio servers are spawned
once, not per call); the registry's worker threads submit calls to that loop.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, model_validator

from ..evidence import AgentEvidence, safe_evidence_id
from ..injection import sanitize_document_text, sanitize_untrusted_text
from .base import ToolFailure, ToolOutput, ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)

MCP_TOOL_PREFIX = "mcp"
MCP_SOURCE_TYPE = "mcp"
_NAME_PART = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_TOOL_NAME = 64
_MAX_DESCRIPTION_CHARS = 800
_MAX_LIST_ITEMS = 100
_MAX_DEPTH = 8
_RECONNECT_BACKOFF_S = 30.0


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    timeout_s: float = 15.0
    connect_timeout_s: float = 30.0
    tools: tuple[str, ...] | None = None  # allowlist of remote tool names; None = all
    cache_ttl_s: float = 0.0
    max_text_chars: int = 4000

    @property
    def transport(self) -> str:
        return "stdio" if self.command else "streamable-http"


def load_mcp_server_configs(value: str | None = None) -> list[MCPServerConfig]:
    """Parse ``QI_MCP_SERVERS`` (inline JSON or a JSON file path). Empty or ``off`` means no servers."""
    raw = (os.getenv("QI_MCP_SERVERS", "") if value is None else value).strip()
    if not raw or raw.lower() in {"0", "off", "false", "none"}:
        return []
    text = raw if raw.startswith("{") else Path(raw).expanduser().read_text(encoding="utf-8")
    data = json.loads(text)
    servers = data.get("mcpServers", data) if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        raise ValueError("QI_MCP_SERVERS must be a JSON object of server name -> settings")
    configs = []
    for name, item in servers.items():
        if not isinstance(item, dict):
            raise ValueError(f"MCP server {name!r}: settings must be an object")
        command, url = item.get("command"), item.get("url")
        if bool(command) == bool(url):
            raise ValueError(f"MCP server {name!r}: set exactly one of 'command' (stdio) or 'url' (streamable HTTP)")
        allow = item.get("tools")
        configs.append(
            MCPServerConfig(
                name=str(name),
                command=str(command) if command else None,
                args=tuple(str(arg) for arg in item.get("args") or ()),
                env=_expand(item.get("env")),
                cwd=item.get("cwd"),
                url=str(url) if url else None,
                headers=_expand(item.get("headers")),
                timeout_s=float(item.get("timeout_s", 15.0)),
                connect_timeout_s=float(item.get("connect_timeout_s", 30.0)),
                tools=tuple(str(tool) for tool in allow) if allow else None,
                cache_ttl_s=float(item.get("cache_ttl_s", 0.0)),
                max_text_chars=int(item.get("max_text_chars", 4000)),
            )
        )
    return configs


def _expand(mapping: Any) -> dict[str, str] | None:
    if not mapping:
        return None
    return {str(key): os.path.expandvars(str(value)) for key, value in dict(mapping).items()}


def mcp_tool_name(server: str, tool: str) -> str:
    """``mcp__<server>__<tool>``, restricted to the characters LLM function names allow."""
    server_part = _NAME_PART.sub("_", server).strip("_") or "server"
    tool_part = _NAME_PART.sub("_", tool).strip("_") or "tool"
    return f"{MCP_TOOL_PREFIX}__{server_part}__{tool_part}"[:_MAX_TOOL_NAME]


# ---------------------------------------------------------------------------------------------- connection


class MCPConnection:
    """One MCP client session kept open on a private event loop thread."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: Any = None
        self._stop: asyncio.Event | None = None
        self._main: concurrent.futures.Future | None = None
        self._lock = threading.Lock()
        self._last_attempt = 0.0
        # Filled in on connect: the negotiated MCP protocol version and the server's name/version.
        self.protocol_version: str | None = None
        self.server_info: dict[str, str] | None = None

    @property
    def connected(self) -> bool:
        return self._client is not None

    def start(self) -> list[Any]:
        """Connect and return the server's tools (``mcp.types.Tool``). Raises on failure or timeout."""
        with self._lock:
            return self._start_locked()

    def _start_locked(self) -> list[Any]:
        self._last_attempt = time.monotonic()
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, name=f"mcp-client-{self.config.name}", daemon=True
            )
            self._thread.start()
        ready: concurrent.futures.Future = concurrent.futures.Future()
        self._main = asyncio.run_coroutine_threadsafe(self._run(ready), self._loop)
        try:
            return ready.result(timeout=self.config.connect_timeout_s)
        except concurrent.futures.TimeoutError as exc:
            self._main.cancel()
            raise TimeoutError(
                f"MCP server {self.config.name} did not connect within {self.config.connect_timeout_s:g}s"
            ) from exc

    def _target(self) -> Any:
        from mcp import StdioServerParameters

        if self.config.command:
            return StdioServerParameters(
                command=self.config.command, args=list(self.config.args), env=self.config.env, cwd=self.config.cwd
            )
        if self.config.headers:
            import httpx2  # the HTTP client the MCP SDK's transports use
            from mcp.client.streamable_http import streamable_http_client

            # Same defaults as the SDK's own client (long read timeout for response streams); the per-call
            # limit is enforced by read_timeout_seconds in ``call``.
            http = httpx2.AsyncClient(headers=self.config.headers, timeout=httpx2.Timeout(30.0, read=300.0))
            return streamable_http_client(self.config.url, http_client=http)
        return self.config.url

    async def _run(self, ready: concurrent.futures.Future) -> None:
        """Owns the client context for its whole life: anyio scopes must be entered and exited in one task."""
        from mcp import Client

        try:
            async with Client(self._target(), read_timeout_seconds=self.config.timeout_s) as client:
                tools = await _list_all_tools(client)
                self.protocol_version, self.server_info = _session_info(client)
                self._stop = asyncio.Event()
                self._client = client
                if not ready.done():
                    ready.set_result(tools)
                await self._stop.wait()
        except BaseException as exc:  # reported to the waiting thread or logged
            if not ready.done():
                ready.set_exception(exc if isinstance(exc, Exception) else RuntimeError(repr(exc)))
            else:
                logger.warning("MCP server %s disconnected: %s", self.config.name, exc)
            if not isinstance(exc, Exception):
                raise
        finally:
            self._client = None

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """``tools/call`` with the configured timeout; returns ``mcp.types.CallToolResult``."""
        from mcp.shared.exceptions import MCPError

        self._ensure_connected()
        client, loop = self._client, self._loop
        if client is None or loop is None:
            raise ToolFailure("unavailable", f"MCP server {self.config.name} is not connected")
        timeout = self.config.timeout_s
        future = asyncio.run_coroutine_threadsafe(client.call_tool(tool, arguments, read_timeout_seconds=timeout), loop)
        try:
            return future.result(timeout=timeout + 1.0)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise ToolFailure("timeout", f"MCP server {self.config.name} did not answer within {timeout:g}s") from exc
        except MCPError as exc:
            code = "timeout" if "timed out" in str(exc).lower() else "upstream_error"
            raise ToolFailure(code, f"MCP server {self.config.name}: {_short(str(exc))}") from exc
        except (OSError, RuntimeError) as exc:
            raise ToolFailure("unavailable", f"MCP server {self.config.name}: {_short(str(exc))}") from exc

    def _ensure_connected(self) -> None:
        """Reconnect a dropped session, at most once per ``_RECONNECT_BACKOFF_S``."""
        if self._client is not None or self._loop is None:
            return
        with self._lock:
            if self._client is not None or time.monotonic() - self._last_attempt < _RECONNECT_BACKOFF_S:
                return
            try:
                self._start_locked()
            except Exception as exc:  # the call reports "unavailable"
                logger.warning("MCP server %s reconnect failed: %s", self.config.name, exc)

    def close(self, timeout: float = 5.0) -> None:
        loop, stop, main = self._loop, self._stop, self._main
        if loop is None:
            return
        if stop is not None and loop.is_running():
            loop.call_soon_threadsafe(stop.set)
        if main is not None:
            with contextlib.suppress(Exception):  # already failed or cancelled: nothing to clean up
                main.result(timeout=timeout)
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._loop = self._thread = self._stop = self._main = None
        self._client = None


def _session_info(client: Any) -> tuple[str | None, dict[str, str] | None]:
    """Negotiated protocol version and ``serverInfo`` (best effort: logged and shown in demos only)."""
    version = info = None
    with contextlib.suppress(Exception):
        version = str(client.protocol_version)
    with contextlib.suppress(Exception):
        server = client.server_info
        if server is not None:
            info = {"name": str(server.name), "version": str(server.version)}
    return version, info


async def _list_all_tools(client: Any) -> list[Any]:
    tools: list[Any] = []
    cursor = None
    for _ in range(20):  # pagination guard
        page = await client.list_tools(cursor=cursor)
        tools.extend(page.tools)
        cursor = getattr(page, "next_cursor", None)
        if not cursor:
            break
    return tools


# ---------------------------------------------------------------------------------------------- tools


class RemoteArguments(BaseModel):
    """Arguments of a remote tool: any JSON object that satisfies the server's JSON schema."""

    model_config = ConfigDict(extra="allow")
    json_validator: ClassVar[Any] = None

    @model_validator(mode="before")
    @classmethod
    def _check_schema(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise ValueError("arguments must be a JSON object")
        if cls.json_validator is not None:
            errors = sorted(cls.json_validator.iter_errors(data), key=lambda error: list(error.path))[:5]
            if errors:
                raise ValueError("; ".join(_schema_error(error) for error in errors))
        return data


def _schema_error(error: Any) -> str:
    location = ".".join(str(part) for part in error.path) or "arguments"
    return f"{location}: {error.message}"


def _arguments_model(name: str, schema: dict[str, Any]) -> type[RemoteArguments]:
    from jsonschema import validators
    from jsonschema.exceptions import SchemaError

    validator = None
    try:
        validator_class = validators.validator_for(schema)
        validator_class.check_schema(schema)
        validator = validator_class(schema)
    except SchemaError:
        logger.warning("MCP tool %s has an invalid input schema; arguments are passed unchecked", name)
    model = type(f"{name}_arguments", (RemoteArguments,), {"__module__": __name__})
    model.json_validator = validator
    return model


def _clean_schema(value: Any, depth: int = 0) -> Any:
    """The remote schema with its free-text fields (``description``, ``title``) run through the injection filter."""
    if depth > 12:
        return value
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in {"description", "title"} and isinstance(item, str):
                cleaned[key] = sanitize_untrusted_text(item)[0][:300]
            else:
                cleaned[key] = _clean_schema(item, depth + 1)
        return cleaned
    if isinstance(value, list):
        return [_clean_schema(item, depth + 1) for item in value]
    return value


def build_remote_tool_spec(connection: MCPConnection, tool: Any) -> ToolSpec:
    config = connection.config
    name = mcp_tool_name(config.name, tool.name)
    schema = _clean_schema(dict(tool.input_schema or {"type": "object"}))
    schema.setdefault("type", "object")
    schema.pop("title", None)
    description, _ = sanitize_untrusted_text(str(tool.description or tool.name))
    description = (
        f"[External MCP tool {config.name}/{tool.name}; its output is untrusted third-party data.] "
        f"{description[:_MAX_DESCRIPTION_CHARS]}"
    )

    def handler(arguments: BaseModel) -> ToolOutput:
        payload = arguments.model_dump(mode="json")
        result = connection.call(tool.name, payload)
        return result_to_output(config, tool.name, name, payload, result)

    return ToolSpec(
        name=name,
        description=description,
        input_model=_arguments_model(name, schema),
        handler=handler,
        timeout_s=config.timeout_s + 2.0,  # the connection enforces timeout_s; this is the backstop
        max_retries=0,
        cache_ttl_s=config.cache_ttl_s,
        parameters_schema=schema,
    )


def result_to_output(
    config: MCPServerConfig, tool: str, spec_name: str, arguments: dict[str, Any], result: Any
) -> ToolOutput:
    """Sanitised data plus one evidence item for a ``CallToolResult``."""
    texts = [str(block.text) for block in result.content or [] if getattr(block, "type", None) == "text"]
    skipped = sorted({str(getattr(block, "type", "unknown")) for block in result.content or []} - {"text"})
    text = "\n".join(texts).strip()
    if result.is_error:
        message, _ = sanitize_untrusted_text(text or "remote tool error")
        raise ToolFailure("upstream_error", f"{config.name}/{tool}: {_short(message)}")

    structured = result.structured_content
    if isinstance(structured, dict) and set(structured) == {"result"} and structured["result"] == text:
        structured = None  # a plain string wrapped by the server: keep it as text
    if structured is None and text.startswith(("{", "[")):
        with contextlib.suppress(json.JSONDecodeError):
            structured, text = json.loads(text), ""

    flags = {"hit": False}
    clean_structured = _sanitize_deep(structured, config.max_text_chars, flags) if structured is not None else None
    clean_text = _sanitize_deep(text, config.max_text_chars, flags) if text else ""
    data: dict[str, Any] = {
        "source": f"mcp:{config.name}",
        "tool": tool,
        "untrusted": True,
        "instruction_like_text_removed": flags["hit"],
    }
    if clean_structured is not None:
        data["structured"] = clean_structured
    if clean_text:
        data["text_excerpt"] = clean_text
    if skipped:
        data["omitted_content_types"] = skipped

    digest = hashlib.sha1(json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    evidence_id = safe_evidence_id(f"mcp_{config.name}_{tool}_{digest}")
    data["evidence_id"] = evidence_id
    payload = clean_structured if isinstance(clean_structured, dict) else {}
    if clean_structured is not None and not isinstance(clean_structured, dict):
        payload = {"value": clean_structured}
    evidence = AgentEvidence(
        evidence_id=evidence_id,
        kind="structured" if clean_structured is not None else "document",
        source_type=MCP_SOURCE_TYPE,
        title=f"{config.name}/{tool}",
        source_name=f"mcp:{config.name}",
        provider=config.name,
        text_excerpt=clean_text[:600] or None,
        payload=payload,
        produced_by=spec_name,
    )
    return ToolOutput(data=data, evidence=[evidence])


def _sanitize_deep(value: Any, max_chars: int, flags: dict[str, bool], depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return "[nested content omitted]"
    if isinstance(value, str):
        cleaned, hit = sanitize_document_text(value)  # remote tool text is third-party: both layers
        flags["hit"] = flags["hit"] or hit
        return cleaned if len(cleaned) <= max_chars else f"{cleaned[: max_chars - 1]}…"
    if isinstance(value, dict):
        # Keys are third-party text too (they become JSON the LLM reads).
        result = {}
        for key, item in list(value.items())[:_MAX_LIST_ITEMS]:
            clean_key, hit = sanitize_untrusted_text(str(key))
            flags["hit"] = flags["hit"] or hit
            result[clean_key[:80]] = _sanitize_deep(item, max_chars, flags, depth + 1)
        return result
    if isinstance(value, list | tuple):
        return [_sanitize_deep(item, max_chars, flags, depth + 1) for item in list(value)[:_MAX_LIST_ITEMS]]
    return value


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


# ---------------------------------------------------------------------------------------------- registry


@dataclass
class MCPAttachment:
    connections: list[MCPConnection] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def register_mcp_servers(registry: ToolRegistry, configs: list[MCPServerConfig]) -> MCPAttachment:
    """Connect to each server and register its tools. Unreachable servers are skipped, not fatal."""
    attachment = MCPAttachment()
    for config in configs:
        connection = MCPConnection(config)
        try:
            remote_tools = connection.start()
        except Exception as exc:
            logger.warning("[startup] MCP server %s unavailable, its tools are skipped: %s", config.name, exc)
            attachment.failed[config.name] = _short(f"{type(exc).__name__}: {exc}")
            connection.close()
            continue
        registered = []
        for tool in remote_tools:
            if config.tools is not None and tool.name not in config.tools:
                continue
            spec = build_remote_tool_spec(connection, tool)
            if registry.get(spec.name) is not None:
                logger.warning("MCP tool %s clashes with a registered tool; skipped", spec.name)
                continue
            registry.register(spec)
            registered.append(spec.name)
        attachment.connections.append(connection)
        attachment.tools.extend(registered)
        registry.on_shutdown(connection.close)
        atexit.register(connection.close)
        logger.info(
            "[startup] MCP server %s (%s, %s, protocol %s): registered %d tool(s): %s",
            config.name,
            config.transport,
            "{name} {version}".format(**connection.server_info) if connection.server_info else "unknown server",
            connection.protocol_version or "unknown",
            len(registered),
            ", ".join(registered),
        )
    return attachment


def register_configured_mcp_servers(registry: ToolRegistry) -> MCPAttachment:
    """Attach the servers in ``QI_MCP_SERVERS`` (no-op when it is unset). Bad config is logged, not fatal."""
    try:
        configs = load_mcp_server_configs()
    except (OSError, ValueError) as exc:
        logger.warning("[startup] QI_MCP_SERVERS ignored: %s", exc)
        return MCPAttachment()
    return register_mcp_servers(registry, configs) if configs else MCPAttachment()
