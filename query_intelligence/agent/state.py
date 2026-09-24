"""Agent graph state and configuration."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class AgentState(TypedDict, total=False):
    # request
    query: str
    mode: str
    user_profile: dict[str, Any]
    dialog_context: list[dict[str, Any]]
    # classical front end
    nlu: dict[str, Any]
    route: str
    route_reasons: list[str]
    # agent loop
    messages: list[dict[str, Any]]
    llm_steps: int
    llm_calls: int
    usage: dict[str, int]
    next: str
    # evidence
    tool_log: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[dict[str, dict[str, Any]], merge_dicts]
    # answer
    draft: dict[str, Any]
    draft_source: str
    revisions: int
    verification: dict[str, Any]
    verification_notes: list[str]
    compliance_notes: list[str]
    answer: dict[str, Any]
    result: dict[str, Any]
    # diagnostics
    degraded: Annotated[list[str], operator.add]
    spans: Annotated[list[dict[str, Any]], operator.add]


@dataclass(frozen=True)
class AgentConfig:
    max_llm_steps: int = 6
    max_tool_calls: int = 16
    max_parallel_tools: int = 4
    token_budget: int = 80_000
    max_revisions: int = 1
    llm_compose: bool = True
    max_next_questions: int = 3
