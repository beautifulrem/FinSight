"""Agent graph state and configuration."""

from __future__ import annotations

import operator
import os
from dataclasses import dataclass, field
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
    started_at: float
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
    prefetched: list[dict[str, Any]]  # planner results given to the first LLM call (planner_prefetch)
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
    llm_log: Annotated[list[dict[str, Any]], add_or_reset]
    # session memory (persists across turns on the same thread)
    turns: Annotated[list[dict[str, Any]], operator.add]
    memory_card: dict[str, Any]  # optional LLM summary of older turns (see memory_summary.py)
    clarification_rounds: int
    clarification_reply: str
    clarification_base: str  # the rewritten question a clarification reply is folded into ("三家" -> two names)
    answer_language: str  # "zh"/"en" persisted by "继续用英文" / "keep answering in English" ("" = none)
    effective_query: str
    # (round 11) a gap / ratio / which question read against the session's comparison frame (frame.py): operation,
    # metric and operands for the planner and the composer; {} otherwise
    frame_request: dict[str, Any]
    language: str  # "zh"/"en": the language of the user's own words (markup and encoded blobs ignored)
    refusal_category: str
    owner: str


@dataclass(frozen=True)
class AgentConfig:
    max_llm_steps: int = 6
    max_tool_calls: int = 16
    max_parallel_tools: int = 4  # per run
    max_concurrent_runs: int = 8  # sizes the shared tool pool: runs * parallel tools
    token_budget: int = 80_000
    max_revisions: int = 1
    llm_compose: bool = True
    max_next_questions: int = 3
    run_deadline_s: float = 90.0
    answer_grace_s: float = 20.0  # extra time for the final answer / revision after the tool loop's deadline
    # Reasoning level per LLM node (None = the client's configured default). The tool loop needs
    # multi-step reasoning; composing from fixed evidence and revising are checked by the verifier.
    agent_reasoning: str | None = None
    compose_reasoning: str | None = "low"
    revise_reasoning: str | None = "low"
    final_reasoning: str | None = "low"
    # Optional LLM summary of turns older than the verbatim history window, under a token budget. Off by
    # default: the rule-based memory card is the default until the summary is ablated (docs/agent.md).
    memory_summary: bool = field(
        default_factory=lambda: os.getenv("QI_AGENT_MEMORY_SUMMARY", "").strip().lower() in {"1", "true", "on"}
    )
    memory_summary_tokens: int = field(
        default_factory=lambda: int(os.getenv("QI_AGENT_MEMORY_SUMMARY_TOKENS", "300") or 300)
    )
    # Latency switches. The defaults are the configuration measured as run C in docs/performance.md (section
    # "Agent-path latency"): P95 22.9/21.8 s -> 15.4/17.3 s on held-out/test_v2 with task success not lower.
    # Each can be turned off by its environment variable (or an AgentConfig field) to restore the earlier path.
    # revise_policy: "llm" sends every failed LLM draft back to the model once; "cite_repair" first tries a
    # deterministic citation fix (uncited / misattributed numbers, invalid ids) and calls the LLM only if
    # the fixed draft still fails verification.
    revise_policy: str = field(default_factory=lambda: _env("QI_AGENT_REVISE_POLICY", "cite_repair"))
    # Run the deterministic planner's tool calls before the first LLM call and give the results to the model,
    # so a question the planner covers can be answered in one LLM round trip (the model may still call tools).
    planner_prefetch: bool = field(default_factory=lambda: _env("QI_AGENT_PREFETCH", "1") in {"1", "true", "on"})
    # Per-call stall timeout (s) for LLM requests: every call is streamed and fails when no chunk arrives for
    # this long, then is retried (or fails over) instead of hanging until the run deadline. 0 = off.
    llm_stall_timeout_s: float = field(default_factory=lambda: float(_env("QI_AGENT_LLM_STALL_TIMEOUT_S", "20") or 0))
    # Accept derived numbers (difference / sum / ratio / percent change of two supported numbers stated in the
    # same cited sentence) in LLM drafts instead of sending them back for revision (see verify_answer).
    verify_derived: bool = field(default_factory=lambda: _env("QI_AGENT_VERIFY_DERIVED", "1") in {"1", "true", "on"})


def _env(name: str, default: str) -> str:
    return (os.getenv(name, default) or default).strip().lower()
