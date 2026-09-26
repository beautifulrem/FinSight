"""Check a pasted market claim against live (or snapshot) data: "茅台市盈率只有15倍，股价跌了5%".

The claim is parsed with the same classical NLU (targets) and the verifier's number extraction (values,
units, stated direction). Each number is tied to a metric by the words around it, the price and
fundamentals tools fetch evidence for every target, and each claimed value is compared with the
evidence value:

* ``supported``: within the stated precision, 2% relative (5% when the claim says 约/about);
* ``contradicted``: the evidence has that metric and the value differs (the actual value is returned);
* ``unverifiable``: no target, no metric recognised, or no evidence for it.

The verdict is deterministic (no LLM) and every check cites the evidence id, source and as-of date,
which is what a broker note or a social-media claim is usually missing.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from .evidence import _NUMBER as _NUMBER_TOKEN
from .evidence import AgentEvidence
from .tools import ToolRegistry
from .verifier import _cleaned, _stated_sign, claim_values

Status = Literal["supported", "contradicted", "unverifiable"]

# metric -> (payload keys, words that name it); the nearest metric word to a number wins.
_METRICS: dict[str, tuple[tuple[str, ...], re.Pattern[str]]] = {
    "close": (("close",), re.compile(r"收盘价?|股价|现价|价格|报价|\bclos(?:e|ed|ing)\b|\bprice\b", re.I)),
    "pct_change_1d": (
        ("pct_change_1d",),
        re.compile(r"涨跌幅|涨了|跌了|上涨|下跌|大涨|大跌|\bup\b|\bdown\b|rose|fell", re.I),
    ),
    "pe_ttm": (("pe_ttm", "pe"), re.compile(r"市盈率|\bP/?E\b|\bPE\(TTM\)", re.I)),
    "pb": (("pb",), re.compile(r"市净率|\bP/?B\b", re.I)),
    "roe": (("roe",), re.compile(r"ROE|净资产收益率", re.I)),
    "revenue": (("revenue",), re.compile(r"营收|营业收入|收入|\brevenue\b", re.I)),
    "net_profit": (("net_profit",), re.compile(r"净利润|净利|\bnet (?:profit|income)\b", re.I)),
}
_APPROXIMATE = re.compile(r"约|大约|左右|将近|接近|超过|不到|\babout\b|\baround\b|\broughly\b|\bnearly\b", re.I)
_FUNDAMENTAL_METRICS = {"pe_ttm", "pb", "roe", "revenue", "net_profit"}


class ClaimCheck(BaseModel):
    target: str | None = None
    metric: str | None = None
    claimed: float
    actual: float | None = None
    status: Status
    evidence_id: str | None = None
    source: str | None = None
    as_of: str | None = None
    note: str = ""


class ClaimReport(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "partially_supported", "unverifiable"]
    checks: list[ClaimCheck] = Field(default_factory=list)
    targets: list[dict[str, Any]] = Field(default_factory=list)
    evidence_sources: list[dict[str, Any]] = Field(default_factory=list)
    disclaimer: str


def check_claim(claim: str, *, service: Any, registry: ToolRegistry, zh: bool = True) -> ClaimReport:
    nlu = service.analyze_query(claim)
    targets = [
        {"name": entity.get("canonical_name"), "symbol": entity.get("symbol")}
        for entity in nlu.get("entities") or []
        if entity.get("symbol") and entity.get("entity_type") in {"stock", "etf", "fund", "index"}
    ]
    numbers = _metric_numbers(claim)
    evidence = _fetch(targets, {metric for _value, metric, *_ in numbers}, registry)
    checks = [
        _check(value, metric, scales, rounding, approximate, targets, evidence)
        for value, metric, scales, rounding, approximate in numbers
    ]
    statuses = {check.status for check in checks}
    if not checks or statuses == {"unverifiable"}:
        verdict = "unverifiable"
    elif statuses == {"supported"}:
        verdict = "supported"
    elif "contradicted" in statuses and "supported" not in statuses:
        verdict = "contradicted"
    else:
        verdict = "partially_supported" if "supported" in statuses else "contradicted"
    disclaimer = (
        "核查只比对声明中的数字与所列数据源，不评价观点本身，也不构成投资建议。"
        if zh
        else "This check compares the claim's numbers with the listed data sources; it is not investment advice."
    )
    return ClaimReport(
        claim=claim,
        verdict=verdict,
        checks=checks,
        targets=targets,
        evidence_sources=[
            {"evidence_id": item.evidence_id, "source_name": item.source_name, "as_of": item.as_of, "title": item.title}
            for items in evidence.values()
            for item in items
        ],
        disclaimer=disclaimer,
    )


def _metric_numbers(claim: str) -> list[tuple[float, str | None, tuple[float, ...], float, bool]]:
    """``(value, metric, scales, rounding, approximate)`` for each number, signed by the stated direction."""
    cleaned = _cleaned(claim)
    out = []
    for match in _NUMBER_TOKEN.finditer(cleaned):
        token = match.group(0).replace(",", "")
        (value, scales, rounding, sign), *_ = claim_values(token + cleaned[match.end() : match.end() + 24]) or [
            (0.0, (1.0,), 0.5, None)
        ]
        before = cleaned[max(0, match.start() - 12) : match.start()]
        sign = sign if token.startswith("-") else _stated_direction(before)
        metric = _nearest_metric(before, cleaned[match.end() : match.end() + 6])
        signed = -abs(value) if sign == -1 else abs(value)
        out.append((signed, metric, scales, rounding, bool(_APPROXIMATE.search(before))))
    return out


def _stated_direction(before: str) -> int | None:
    return _stated_sign("", before)


def _nearest_metric(before: str, after: str) -> str | None:
    best: tuple[int, str] | None = None
    for metric, (_keys, pattern) in _METRICS.items():
        for match in pattern.finditer(before):
            distance = len(before) - match.end()
            if best is None or distance < best[0]:
                best = (distance, metric)
        match = pattern.search(after)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), metric)
    return best[1] if best else None


def _fetch(
    targets: list[dict[str, Any]], metrics: set[str | None], registry: ToolRegistry
) -> dict[str, list[AgentEvidence]]:
    evidence: dict[str, list[AgentEvidence]] = {}
    for target in targets[:3]:
        items: list[AgentEvidence] = []
        if metrics & {"close", "pct_change_1d", None}:
            result = registry.run("get_price_history", {"target": target["symbol"]})
            items.extend(result.evidence if result.ok else [])
        if metrics & (_FUNDAMENTAL_METRICS | {None}):
            result = registry.run("get_fundamentals", {"target": target["symbol"]})
            items.extend(
                item for item in (result.evidence if result.ok else []) if item.evidence_id.startswith("fundamental_")
            )
        evidence[target["symbol"]] = items
    return evidence


def _check(
    value: float,
    metric: str | None,
    scales: tuple[float, ...],
    rounding: float,
    approximate: bool,
    targets: list[dict[str, Any]],
    evidence: dict[str, list[AgentEvidence]],
) -> ClaimCheck:
    if not targets:
        return ClaimCheck(claimed=value, metric=metric, status="unverifiable", note="no listed company, fund or index")
    if metric is None:
        return ClaimCheck(claimed=value, target=targets[0]["name"], status="unverifiable", note="metric not recognised")
    keys = _METRICS[metric][0]
    relative = 0.05 if approximate else 0.02
    candidates = []
    for target in targets:
        for item in evidence.get(target["symbol"]) or []:
            for key in keys:
                actual = item.payload.get(key)
                if isinstance(actual, int | float) and not isinstance(actual, bool):
                    candidates.append((target, item, float(actual)))
    if not candidates:
        return ClaimCheck(
            claimed=value,
            metric=metric,
            target=targets[0]["name"],
            status="unverifiable",
            note="no data for this metric",
        )
    for target, item, actual in candidates:
        for scale in scales:
            if scale == 100.0 and abs(actual) > 1.5:
                continue
            expected = actual * scale
            tolerance = max(rounding, abs(expected) * relative) + 1e-9
            same_direction = metric != "pct_change_1d" or (value >= 0) == (expected >= 0) or expected == 0
            if same_direction and abs(value - expected) <= tolerance:
                return _result(value, metric, target, item, actual, "supported")
    target, item, actual = candidates[0]
    return _result(value, metric, target, item, actual, "contradicted")


def _result(
    value: float, metric: str, target: dict[str, Any], item: AgentEvidence, actual: float, status: Status
) -> ClaimCheck:
    return ClaimCheck(
        claimed=value,
        metric=metric,
        target=target["name"],
        actual=actual,
        status=status,
        evidence_id=item.evidence_id,
        source=item.source_name,
        as_of=item.as_of,
    )
