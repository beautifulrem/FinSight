"""Scoring for agent evaluation turns and aggregate metrics.

Task success uses "dealbreaker" gating (as in finance-agent benchmarks): a turn only succeeds when
the behaviour is right (answer / clarify / refuse), every required fact is stated *and* cited,
required tools were used, no forbidden (trading-instruction) content appears, and the hedging,
missing-data, and entity-carry-over expectations hold.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
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
_LLM_FAILURE_FLAGS = ("llm_error", "llm_compose_failed", "llm_revision_failed")
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
    if expect.get("forbidden_tools"):
        # e.g. a carried-over "And the P/B?" must not be answered with macro indicators
        checks["no_forbidden_tools"] = not (set(expect["forbidden_tools"]) & used_set)
    if expect.get("required_limitations"):
        # machine codes such as out_of_coverage: the refusal must give the specific reason
        checks["limitations"] = set(expect["required_limitations"]) <= set(response.get("limitations") or [])
    if expect.get("language"):
        checks["language"] = response.get("language") == expect["language"]
    symbols = {entity.get("symbol") for entity in (response.get("nlu_summary") or {}).get("entities") or []}
    if expect.get("required_entity"):
        checks["entity"] = expect["required_entity"] in symbols
    if expect.get("required_entities"):
        # plural follow-ups ("这两家…") must carry every earlier target, not just the last one
        checks["entities"] = set(expect["required_entities"]) <= symbols

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
        "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
        "reasoning_tokens": usage.get("reasoning_tokens", 0),
        "revisions": sum(1 for entry in llm.get("log") or [] if entry.get("node") == "revise"),
        "first_pass_verified": _first_pass_verified(response, llm),
        "cost": llm.get("cost"),
        "cost_currency": llm.get("currency"),
        "llm_calls": llm.get("calls", 0),
        # JSON status of each answer-producing LLM call (compose, final agent turn, revise): ok/repaired/failed.
        "json_statuses": [entry["json_status"] for entry in llm.get("log") or [] if entry.get("json_status")],
    }


def _status_rate(scores: list[dict[str, Any]], status: str) -> float | None:
    statuses = [value for score in scores for value in score.get("json_statuses") or []]
    return round(statuses.count(status) / len(statuses), 4) if statuses else None


def _first_pass_verified(response: dict[str, Any], llm: dict[str, Any]) -> bool | None:
    """Draft passed verification with no LLM revision and no deterministic repair (LLM answers only)."""
    if response.get("answer_source") not in {"llm_agent", "llm_compose"}:
        return None
    revised = any(entry.get("node") == "revise" for entry in llm.get("log") or [])
    repaired = any(str(flag).startswith("verification_failed") for flag in response.get("degraded") or [])
    return bool((response.get("verification") or {}).get("passed")) and not revised and not repaired


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return round(ordered[index], 2)


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


# Answer paths compared on every task set (a minus b), when both ran.
COMPARISONS = (("agent", "workflow_llm"), ("agent", "workflow"), ("workflow_llm", "workflow"))
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260926
CI_METHOD = (
    f"percentile bootstrap over tasks, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}, 95%; "
    "repeats of a task stay together"
)


def task_outcomes(records: list[dict[str, Any]]) -> dict[str, list[bool]]:
    """Task id -> success of each repeat (a run succeeds only when every turn succeeds), in run order."""
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item.get("repeat", 0)):
        outcomes[record["task"]["id"]].append(all(turn["score"]["success"] for turn in record["turns"]))
    return dict(outcomes)


def task_success_value(runs: Sequence[bool]) -> float:
    return sum(runs) / len(runs)


def pass_all_value(runs: Sequence[bool]) -> float:
    return 1.0 if all(runs) else 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> list[float] | None:
    """Percentile bootstrap CI of the mean of per-task ``values`` (tasks are the resampling unit)."""
    if not values:
        return None
    rng = random.Random(seed)
    size = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=size)) for _ in range(resamples))
    tail = (1.0 - level) / 2
    low = means[max(0, math.floor(tail * resamples))]
    high = means[min(resamples - 1, math.ceil((1.0 - tail) * resamples) - 1)]
    return [round(low, 4), round(high, 4)]


def outcome_cis(outcomes: Mapping[str, Sequence[bool]]) -> dict[str, list[float] | None]:
    """Bootstrap CIs for task success (mean over repeats) and pass^k (all repeats succeed)."""
    runs = list(outcomes.values())
    k = min((len(item) for item in runs), default=1)
    return {
        "task_success": bootstrap_ci([task_success_value(item) for item in runs]),
        f"pass^{k}": bootstrap_ci([pass_all_value(item) for item in runs]),
    }


def _binomial_two_sided(successes: int, trials: int) -> float:
    """Exact two-sided binomial test against p = 0.5 (the exact McNemar test on discordant pairs)."""
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, i) for i in range(0, min(successes, trials - successes) + 1)) / 2**trials
    return min(1.0, 2 * tail)


def paired_comparison(
    a: Mapping[str, Sequence[bool]],
    b: Mapping[str, Sequence[bool]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare two answer paths on the same tasks (``a`` minus ``b``).

    * Paired bootstrap over tasks for the difference in task success and in pass^k: each resample
      draws tasks with replacement and uses both paths' results on the drawn tasks. A difference is
      called significant when the 95% interval excludes 0.
    * Exact McNemar test on the per-task pass^k outcome (discordant tasks only).
    """
    common = sorted(set(a) & set(b))
    if not common:
        return {"tasks": 0}
    metrics: dict[str, Callable[[Sequence[bool]], float]] = {
        "task_success": task_success_value,
        "pass^k": pass_all_value,
    }
    rng = random.Random(seed)
    draws = [rng.choices(range(len(common)), k=len(common)) for _ in range(resamples)]
    result: dict[str, Any] = {"tasks": len(common), "method": "paired bootstrap over tasks + exact McNemar"}
    for name, fn in metrics.items():
        diffs = [fn(a[task]) - fn(b[task]) for task in common]
        boot = sorted(statistics.fmean(diffs[index] for index in draw) for draw in draws)
        low = boot[max(0, math.floor(alpha / 2 * resamples))]
        high = boot[min(resamples - 1, math.ceil((1 - alpha / 2) * resamples) - 1)]
        result[name] = {
            "a": round(statistics.fmean(fn(a[task]) for task in common), 4),
            "b": round(statistics.fmean(fn(b[task]) for task in common), 4),
            "diff": round(statistics.fmean(diffs), 4),
            "ci": [round(low, 4), round(high, 4)],
            "significant": low > 0 or high < 0,
        }
    only_a = sum(1 for task in common if all(a[task]) and not all(b[task]))
    only_b = sum(1 for task in common if all(b[task]) and not all(a[task]))
    p_value = _binomial_two_sided(only_a, only_a + only_b)
    result["mcnemar"] = {
        "a_only_pass": only_a,
        "b_only_pass": only_b,
        "p_value": round(p_value, 4),
        "significant": p_value < alpha,
    }
    return result


def aggregate(records: list[dict[str, Any]], *, repeats: int = 1) -> dict[str, Any]:
    """``records``: one per task run with ``task``, ``repeat``, ``turns`` (scored turns + latency).

    ``pass^k`` uses the number of runs actually present per task (so a path that ran once is reported
    as pass^1 even when ``repeats`` says otherwise), and carries a bootstrap 95% CI.
    """
    turns = [turn for record in records for turn in record["turns"]]
    task_runs = task_outcomes(records)
    k = min((len(runs) for runs in task_runs.values()), default=repeats)

    def mean(values: list[float]) -> float | None:
        clean = [value for value in values if value is not None]
        return round(statistics.fmean(clean), 4) if clean else None

    scores = [turn["score"] for turn in turns]
    facts = [fact for score in scores for fact in score["facts"]]
    latencies = [turn["latency_ms"] for turn in turns]
    llm_latencies = [turn["latency_ms"] for turn in turns if turn["score"].get("llm_calls")]
    ttfts = [turn["ttft_ms"] for turn in turns if turn.get("ttft_ms") is not None]
    costs = [score["cost"] for score in scores if score["cost"] is not None]
    currencies = sorted({score.get("cost_currency") for score in scores if score.get("cost_currency")})
    task_costs = [
        sum(turn["score"]["cost"] for turn in record["turns"])
        for record in records
        if record["turns"] and all(turn["score"]["cost"] is not None for turn in record["turns"])
    ]
    summary = {
        "tasks": len(task_runs),
        "turns": len(turns),
        "repeats": k,
        "task_success": mean([task_success_value(runs) for runs in task_runs.values()]),
        f"pass^{k}": mean([pass_all_value(runs) for runs in task_runs.values()]),
        "ci": outcome_cis(task_runs),
        "ci_method": CI_METHOD,
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
        "latency_ms_p99": percentile(latencies, 0.99),
        # Turns that made at least one LLM call (refusals, clarifications and template answers excluded).
        "llm_turns": len(llm_latencies),
        "llm_turn_latency_ms_p50": percentile(llm_latencies, 0.5),
        "llm_turn_latency_ms_p95": percentile(llm_latencies, 0.95),
        "llm_turn_latency_ms_p99": percentile(llm_latencies, 0.99),
        # Time to the first streamed answer character (runs with --stream only; turns that streamed text).
        "ttft_turns": len(ttfts),
        "ttft_ms_p50": percentile(ttfts, 0.5),
        "ttft_ms_p95": percentile(ttfts, 0.95),
        "ttft_ms_p99": percentile(ttfts, 0.99),
        "llm_calls_per_turn": mean([float(score["llm_calls"]) for score in scores]),
        "tokens_per_turn": mean([float(score["prompt_tokens"] + score["completion_tokens"]) for score in scores]),
        "reasoning_tokens_per_turn": mean([float(score.get("reasoning_tokens") or 0) for score in scores]),
        "cache_hit_ratio": _ratio(
            sum(score.get("cache_hit_tokens") or 0 for score in scores), sum(score["prompt_tokens"] for score in scores)
        ),
        "first_pass_verification": mean(
            [
                1.0 if score["first_pass_verified"] else 0.0
                for score in scores
                if score.get("first_pass_verified") is not None
            ]
        ),
        "revise_rate": mean(
            [1.0 if score.get("revisions") else 0.0 for score in scores if score.get("first_pass_verified") is not None]
        ),
        # Turns where an LLM call failed and the run fell back (planner, template, or unrevised draft).
        "llm_error_rate": mean(
            [
                1.0 if any(str(flag).startswith(_LLM_FAILURE_FLAGS) for flag in score.get("degraded") or []) else 0.0
                for score in scores
            ]
        ),
        "llm_error_kinds": dict(
            Counter(
                str(flag)[:80]
                for score in scores
                for flag in score.get("degraded") or []
                if str(flag).startswith(_LLM_FAILURE_FLAGS)
            ).most_common(5)
        ),
        # Share of answer drafts that were not valid JSON: repaired with json_repair, or used as plain text.
        "json_repair_rate": _status_rate(scores, "repaired"),
        "json_failure_rate": _status_rate(scores, "failed"),
        "cost_per_turn": round(statistics.fmean(costs), 6) if costs else None,
        "cost_per_task": round(statistics.fmean(task_costs), 6) if task_costs else None,
        "cost_currency": "/".join(currencies) if currencies else None,
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
