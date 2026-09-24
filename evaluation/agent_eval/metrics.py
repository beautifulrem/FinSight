"""Scoring for agent evaluation turns and aggregate metrics.

Task success uses "dealbreaker" gating (as in finance-agent benchmarks): a turn only succeeds when
the behaviour is right (answer / clarify / refuse), every required fact is stated *and* cited,
required tools were used, no forbidden (trading-instruction) content appears, and the hedging,
missing-data, and entity-carry-over expectations hold.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from typing import Any

from query_intelligence.agent.verifier import _is_supported, answer_texts, claim_numbers

_HEDGE_MARKERS = (
    "不能据此",
    "条件性",
    "可能",
    "不构成",
    "不足以",
    "仅为证据",
    "无法",
    "not a ",
    "not investment advice",
    "possible",
    "cannot",
    "conditional",
    "not enough",
    "does not establish",
)
_MISSING_MARKERS = ("未返回", "没有", "未获取", "缺少", "不足", "无法", "not ", "no usable", "unavailable", "missing")


def behavior_of(response: dict[str, Any]) -> str:
    if response.get("status") == "needs_clarification" or response.get("route") == "clarify":
        return "clarify"
    if response.get("route") == "refuse":
        return "refuse"
    return "answer"


def score_turn(response: dict[str, Any], expect: dict[str, Any]) -> dict[str, Any]:
    behavior = behavior_of(response)
    expected_behavior = expect.get("behavior", "answer")
    texts = answer_texts(response) if behavior == "answer" else []
    joined = " ".join(texts)
    claims = [value for text in texts for value in claim_numbers(text)]
    cited = set(response.get("evidence_used") or [])
    used = [call.get("tool") for call in response.get("tool_calls") or []]
    used_set = set(used)

    facts = []
    for fact in expect.get("required_facts") or []:
        stated = _is_supported(float(fact["value"]), claims) if claims else False
        facts.append(
            {
                "evidence_id": fact["evidence_id"],
                "value": fact["value"],
                "stated": stated,
                "cited": fact["evidence_id"] in cited,
            }
        )
    required = set(expect.get("required_tools") or [])
    any_of = set(expect.get("any_of_tools") or [])
    allowed = required | any_of | {"resolve_entity"}
    forbidden = [pattern for pattern in expect.get("forbidden_patterns") or [] if re.search(pattern, joined)]
    verification = response.get("verification") or {}
    limitations = " ".join(str(item) for item in response.get("limitations") or [])

    checks = {
        "behavior": behavior == expected_behavior,
        "facts": all(item["stated"] and item["cited"] for item in facts),
        "required_tools": required <= used_set,
        "any_of_tools": not any_of or bool(any_of & used_set),
        "no_forbidden_content": not forbidden,
    }
    if expected_behavior == "answer":
        checks["disclaimer"] = bool(str(response.get("risk_disclaimer") or "").strip())
    if expect.get("must_hedge"):
        lowered = joined.lower()
        checks["hedged"] = any(marker in lowered for marker in _HEDGE_MARKERS) or bool(
            {"conditional_prefix", "removed_trading_instruction"} & set(response.get("compliance_notes") or [])
        )
    if expect.get("must_state_missing"):
        lowered = f"{joined} {limitations}".lower()
        checks["states_missing"] = any(marker in lowered for marker in _MISSING_MARKERS)
    if expect.get("required_entity"):
        symbols = {entity.get("symbol") for entity in (response.get("nlu_summary") or {}).get("entities") or []}
        checks["entity"] = expect["required_entity"] in symbols

    llm = response.get("llm") or {}
    usage = llm.get("usage") or {}
    return {
        "success": all(checks.values()),
        "checks": checks,
        "behavior": behavior,
        "expected_behavior": expected_behavior,
        "facts": facts,
        "forbidden_hits": forbidden,
        "tools_used": used,
        "tool_recall": (len(required & used_set) / len(required)) if required else None,
        "tool_precision": (len([tool for tool in used if tool in allowed]) / len(used)) if used and allowed else None,
        "tool_errors": sum(1 for call in response.get("tool_calls") or [] if not call.get("ok")),
        "citations": len(cited),
        "invalid_citations": len(verification.get("invalid_citations") or []),
        "unsupported_numbers": len(verification.get("unsupported_numbers") or []),
        "draft_verified": verification.get("passed") if verification else None,
        "degraded": response.get("degraded") or [],
        "route": response.get("route"),
        "answer_source": response.get("answer_source"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cost": llm.get("cost"),
        "llm_calls": llm.get("calls", 0),
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return round(ordered[index], 2)


def aggregate(records: list[dict[str, Any]], *, repeats: int = 1) -> dict[str, Any]:
    """``records``: one per task run with ``task``, ``repeat``, ``turns`` (scored turns + latency)."""
    turns = [turn for record in records for turn in record["turns"]]
    task_runs: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        task_runs[record["task"]["id"]].append(all(turn["score"]["success"] for turn in record["turns"]))

    def mean(values: list[float]) -> float | None:
        clean = [value for value in values if value is not None]
        return round(statistics.fmean(clean), 4) if clean else None

    scores = [turn["score"] for turn in turns]
    facts = [fact for score in scores for fact in score["facts"]]
    latencies = [turn["latency_ms"] for turn in turns]
    costs = [score["cost"] for score in scores if score["cost"] is not None]
    summary = {
        "tasks": len(task_runs),
        "turns": len(turns),
        "repeats": repeats,
        "task_success": mean([sum(runs) / len(runs) for runs in task_runs.values()]),
        f"pass^{repeats}": mean([1.0 if all(runs) else 0.0 for runs in task_runs.values()]),
        "turn_success": mean([1.0 if score["success"] else 0.0 for score in scores]),
        "behavior_accuracy": mean([1.0 if score["checks"]["behavior"] else 0.0 for score in scores]),
        "fact_recall": mean([1.0 if fact["stated"] and fact["cited"] else 0.0 for fact in facts]),
        "fact_stated": mean([1.0 if fact["stated"] else 0.0 for fact in facts]),
        "tool_recall": mean([score["tool_recall"] for score in scores]),
        "tool_precision": mean([score["tool_precision"] for score in scores]),
        "compliance_clean": mean([1.0 if score["checks"]["no_forbidden_content"] else 0.0 for score in scores]),
        "hedged_when_required": mean(
            [1.0 if score["checks"]["hedged"] else 0.0 for score in scores if "hedged" in score["checks"]]
        ),
        "states_missing_when_required": mean(
            [
                1.0 if score["checks"]["states_missing"] else 0.0
                for score in scores
                if "states_missing" in score["checks"]
            ]
        ),
        "draft_verification_pass": mean(
            [1.0 if score["draft_verified"] else 0.0 for score in scores if score["draft_verified"] is not None]
        ),
        "unsupported_numbers_per_answer": mean(
            [float(score["unsupported_numbers"]) for score in scores if score["behavior"] == "answer"]
        ),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "llm_calls_per_turn": mean([float(score["llm_calls"]) for score in scores]),
        "tokens_per_turn": mean([float(score["prompt_tokens"] + score["completion_tokens"]) for score in scores]),
        "cost_per_turn": round(statistics.fmean(costs), 6) if costs else None,
    }
    return summary


def breakdown(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["task"].get(key))].append(record)
    result = {}
    for name, items in sorted(groups.items()):
        turns = [turn for record in items for turn in record["turns"]]
        result[name] = {
            "tasks": len({record["task"]["id"] for record in items}),
            "task_success": round(
                statistics.fmean(
                    [1.0 if all(turn["score"]["success"] for turn in record["turns"]) else 0.0 for record in items]
                ),
                4,
            ),
            "latency_ms_p95": percentile([turn["latency_ms"] for turn in turns], 0.95),
        }
    return result


def failed_checks(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        for turn in record["turns"]:
            for name, passed in turn["score"]["checks"].items():
                if not passed:
                    counts[name] += 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))
