"""Agent graph state and configuration."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

RESET = "__reset__"


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Merge reducer; ``{RESET: True}`` clears the value (used at the start of each turn)."""
    if isinstance(right, dict) and right.get(RESET) is True:
        return {}
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def add_or_reset(left: list[Any] | None, right: Any) -> list[Any]:
    """Append reducer; ``{RESET: [...]}`` replaces the value (used at the start of each turn)."""
    if isinstance(right, dict) and RESET in right:
        return list(right[RESET])
    return [*(left or []), *(right or [])]


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
    tool_log: Annotated[list[dict[str, Any]], add_or_reset]
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
    degraded: Annotated[list[str], add_or_reset]
    spans: Annotated[list[dict[str, Any]], add_or_reset]
    # session memory (persists across turns on the same thread)
    turns: Annotated[list[dict[str, Any]], operator.add]
    clarification_rounds: int


@dataclass(frozen=True)
class AgentConfig:
    max_llm_steps: int = 6
    max_tool_calls: int = 16
    max_parallel_tools: int = 4
    token_budget: int = 80_000
    max_revisions: int = 1
    llm_compose: bool = True
    max_next_questions: int = 3
