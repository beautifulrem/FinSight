"""Point-in-time tool snapshots for reproducible agent evaluation.

``RecordingRegistry`` runs the real tools and stores every result (data, evidence, or error) keyed by
tool name and validated arguments. ``ReplayRegistry`` serves those results back with the same tool
schemas, so evaluation does not depend on live providers or on later changes to runtime assets.
Calls that were not recorded return an ``unavailable`` error (or fall through to a live registry when
one is supplied, which is how new online LLM trajectories are recorded).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from query_intelligence.agent.evidence import AgentEvidence
from query_intelligence.agent.tools import ToolFailure, ToolOutput, ToolRegistry, ToolResult, ToolSpec

FIXTURE_VERSION = 1


def call_key(tool: str, arguments: dict[str, Any]) -> str:
    return f"{tool}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def _stored(result: ToolResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "data": result.data,
        "evidence": [item.model_dump(mode="json") for item in result.evidence],
        "error": result.error.model_dump() if result.error else None,
    }


class RecordingRegistry(ToolRegistry):
    """Delegates to a live registry and records every normalized call."""

    def __init__(self, live: ToolRegistry, calls: dict[str, dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.live = live
        self.calls: dict[str, dict[str, Any]] = calls if calls is not None else {}
        for spec in live.specs():
            self.register(spec)

    def run(self, name: str, arguments: dict[str, Any] | str | None = None) -> ToolResult:
        result = self.live.run(name, arguments)
        if result.error is None or result.error.code not in {"unknown_tool", "invalid_arguments"}:
            self.calls[call_key(name, result.arguments)] = _stored(result)
        return result

    def save(self, path: str | Path, *, snapshot: str) -> None:
        payload = {
            "version": FIXTURE_VERSION,
            "snapshot": snapshot,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "calls": dict(sorted(self.calls.items())),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


class ReplayRegistry(ToolRegistry):
    """Serves recorded tool results using the original tool schemas."""

    def __init__(
        self,
        specs: list[ToolSpec],
        calls: dict[str, dict[str, Any]],
        *,
        fallback: ToolRegistry | None = None,
    ) -> None:
        super().__init__()
        self.calls = calls
        self.fallback = fallback
        self.misses: list[str] = []
        for spec in specs:
            self.register(
                ToolSpec(
                    name=spec.name,
                    description=spec.description,
                    input_model=spec.input_model,
                    handler=self._handler(spec.name),
                    timeout_s=spec.timeout_s,
                    max_retries=0,
                    cache_ttl_s=0,
                )
            )

    @classmethod
    def from_file(
        cls, path: str | Path, specs: list[ToolSpec], *, fallback: ToolRegistry | None = None
    ) -> ReplayRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != FIXTURE_VERSION:
            raise ValueError(f"unsupported fixture version: {payload.get('version')}")
        return cls(specs, payload["calls"], fallback=fallback)

    def _handler(self, name: str):
        def handler(args: BaseModel) -> ToolOutput:
            arguments = args.model_dump(mode="json")
            stored = self.calls.get(call_key(name, arguments))
            if stored is None:
                self.misses.append(call_key(name, arguments))
                if self.fallback is not None:
                    live = self.fallback.run(name, arguments)
                    self.calls[call_key(name, live.arguments)] = _stored(live)
                    stored = self.calls[call_key(name, live.arguments)]
                else:
                    raise ToolFailure("unavailable", "not recorded in the evaluation snapshot")
            if not stored["ok"]:
                error = stored.get("error") or {}
                raise ToolFailure(error.get("code") or "unavailable", str(error.get("message") or "recorded failure"))
            evidence = [AgentEvidence.model_validate(item) for item in stored.get("evidence") or []]
            return ToolOutput(data=stored.get("data"), evidence=evidence)

        return handler
