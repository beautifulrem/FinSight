"""Evidence verifier: citations must exist and numbers must be traceable to tool outputs.

Checks are claim-level. The answer and each key point are split into sentences; a number in a
sentence that cites evidence ids must be found in *those* evidence items, not merely somewhere in the
run. A number found only in other evidence is reported as ``misattributed`` (for example a PE ratio of
one company cited with another company's evidence id); a number found nowhere is ``unsupported``.
Sentences without a citation fall back to the whole evidence store. Matching allows common unit
conversions (percent <-> ratio, 万/亿, thousand/million/billion) and rounding. Dates, evidence ids,
ticker codes, and window parameters such as ``RSI(14)`` or ``近5日`` are not treated as factual claims.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .evidence import _NUMBER as _NUMBER_TOKEN
from .evidence import EvidenceStore, extract_numbers

_SCALES = (1.0, 100.0, 0.01, 1e-4, 1e-8, 1e4, 1e8, 1e-3, 1e-6, 1e-9, 1e3, 1e6, 1e9)
# Scale factors (evidence value -> stated value) allowed for the unit written after a number. Evidence
# amounts may be stored in 元, 千元 (Tushare), 万元 or 亿元, so each unit lists the conversions that can
# legitimately produce it; a bare number must match as is. Restricting scales by unit keeps a wrong
# number from matching an unrelated value by an arbitrary power of ten.
_UNIT_SCALES: tuple[tuple[re.Pattern[str], tuple[float, ...]], ...] = (
    (re.compile(r"^\s*(?:%|％|个百分点|百分点|pct|percentage points?|bp)", re.IGNORECASE), (1.0, 100.0, 0.01)),
    (re.compile(r"^\s*万亿"), (1e-12, 1e-9, 1e-8, 1e-4, 1.0)),
    (re.compile(r"^\s*(?:亿|hundred million)", re.IGNORECASE), (1e-8, 1e-5, 1e-4, 1.0)),
    (re.compile(r"^\s*千万"), (1e-7, 1e-4, 1e-3, 1.0)),
    (re.compile(r"^\s*百万"), (1e-6, 1e-3, 1e-2, 1.0)),
    (re.compile(r"^\s*万"), (1e-4, 0.1, 1.0)),
    (re.compile(r"^\s*千(?!元)"), (1e-3, 1.0)),
    (re.compile(r"^\s*(?:billion|bn)\b", re.IGNORECASE), (1e-9, 1e-6, 1e-5, 0.1, 1.0)),
    (re.compile(r"^\s*(?:million|mn|m)\b", re.IGNORECASE), (1e-6, 1e-3, 1e-2, 100.0, 1.0)),
    (re.compile(r"^\s*(?:thousand|k)\b", re.IGNORECASE), (1e-3, 1.0)),
)
_BARE_SCALES = (1.0,)
_CITATION = re.compile(r"\[([^\[\]\s]{2,160})\]")
_DATE_PATTERNS = (
    re.compile(r"\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?"),
    re.compile(r"\d{4}/\d{1,2}/\d{1,2}"),
    re.compile(r"\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?"),
    re.compile(r"\d{1,2}月\d{1,2}日"),
    re.compile(r"(?:\bin|\bsince|\bby|\bFY|财年)\s*(?:19|20)\d{2}\b", re.IGNORECASE),
    re.compile(r"(?:19|20)\d{2}\s*(?:年报|年度|annual|fiscal|full[- ]year)", re.IGNORECASE),
    re.compile(r"\bQ[1-4]\b", re.IGNORECASE),
)
_PARAMETER_PATTERNS = (
    re.compile(r"(?:RSI|MA|EMA|SMA|MACD|BOLL)\s*[\(（]?\s*\d+(?:\s*[,，]\s*\d+)*\s*[\)）]?", re.IGNORECASE),
    re.compile(r"(?:近|过去|最近|前|后|未来)\s*\d+\s*(?:个)?(?:交易日|日|天|周|个月|月|年|季度)"),
    re.compile(r"\d+\s*(?:个)?(?:交易日|日均线|日线|篇|条|家|只|个|项|名|位)"),
    re.compile(
        r"\b\d+[- ]?(?:day|days|week|weeks|month|months|year|years|articles?|items?|documents?)\b", re.IGNORECASE
    ),
    re.compile(r"\d{6}\.(?:SH|SZ|BJ)", re.IGNORECASE),
)


class VerificationReport(BaseModel):
    passed: bool
    cited_ids: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    unsupported_numbers: list[float] = Field(default_factory=list)
    misattributed_numbers: list[float] = Field(
        default_factory=list,
        description="Numbers present in the run's evidence but not in the evidence cited next to them",
    )
    checked_numbers: int = 0
    missing_citations: bool = False

    def feedback(self) -> str:
        problems = []
        if self.invalid_citations:
            problems.append(f"These evidence ids do not exist in this run: {', '.join(self.invalid_citations)}.")
        if self.unsupported_numbers:
            values = ", ".join(_format_number(value) for value in self.unsupported_numbers)
            problems.append(f"These numbers are not found in any tool output: {values}.")
        if self.misattributed_numbers:
            values = ", ".join(_format_number(value) for value in self.misattributed_numbers)
            problems.append(
                f"These numbers do not appear in the evidence cited in the same sentence: {values}. "
                "Cite the evidence id that actually contains each number."
            )
        if self.missing_citations:
            problems.append("The answer cites no evidence ids although evidence is available.")
        return " ".join(problems)


def answer_texts(answer: dict[str, Any]) -> list[str]:
    texts = [str(answer.get("answer") or "")]
    texts.extend(str(point) for point in answer.get("key_points") or [])
    return [text for text in texts if text.strip()]


def cited_ids(answer: dict[str, Any]) -> list[str]:
    ids: list[str] = [str(item) for item in answer.get("evidence_used") or [] if str(item).strip()]
    for text in answer_texts(answer):
        ids.extend(match.group(1) for match in _CITATION.finditer(text))
    return list(dict.fromkeys(ids))


def _cleaned(text: str) -> str:
    cleaned = _CITATION.sub(" ", text)
    for pattern in (*_DATE_PATTERNS, *_PARAMETER_PATTERNS):
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


def claim_numbers(text: str) -> list[float]:
    return extract_numbers(_cleaned(text))


def claim_values(text: str) -> list[tuple[float, tuple[float, ...], float]]:
    """Claimed numbers with the scale factors their unit allows and the rounding tolerance of their precision.

    A number written with ``d`` decimals can differ from the evidence by at most half a unit in its
    last place (``0.5 * 10**-d``) plus 0.05% for binary rounding; "24.6" matches 24.63 but not 24.8.
    """
    cleaned = _cleaned(text)
    values = []
    for match in _NUMBER_TOKEN.finditer(cleaned):
        token = match.group(0).replace(",", "")
        try:
            value = float(token)
        except ValueError:
            continue
        tail = cleaned[match.end() : match.end() + 24]
        scales = next((scales for pattern, scales in _UNIT_SCALES if pattern.search(tail)), _BARE_SCALES)
        decimals = len(token.split(".")[1]) if "." in token else 0
        values.append((value, scales, 0.5 * 10**-decimals))
    return values


def claim_units(answer: dict[str, Any]) -> list[str]:
    """Sentences of the answer and of each key point: the unit a citation applies to."""
    units: list[str] = []
    for text in answer_texts(answer):
        units.extend(sentence for sentence in _split_sentences(text) if sentence.strip())
    return units


def verify_answer(
    answer: dict[str, Any], store: EvidenceStore, *, query: str = "", binding: str = "claim"
) -> VerificationReport:
    """``binding="claim"`` (default) checks each number against the evidence cited in its sentence with a
    unit- and precision-aware tolerance. ``"run"`` checks against all evidence of the run, and ``"legacy"``
    also uses the original loose matching (any of 13 scales, ±max(0.011, 0.5%)); both are kept only so
    ``evaluation/agent_eval/verifier_stress.py`` can measure the improvement."""
    ids = cited_ids(answer)
    invalid = [evidence_id for evidence_id in ids if evidence_id not in store]
    known = _evidence_numbers(store)
    query_numbers = claim_numbers(query)
    unsupported: list[float] = []
    misattributed: list[float] = []
    checked = 0
    for unit in claim_units(answer):
        unit_ids = (
            [match.group(1) for match in _CITATION.finditer(unit) if match.group(1) in store]
            if binding == "claim"
            else []
        )
        scope = _evidence_numbers(store, unit_ids) if unit_ids else known
        for value, scales, rounding in claim_values(unit):
            if binding == "legacy":
                scales, rounding = _SCALES, None
            if value == 0 and binding != "legacy":
                continue  # zero counts ("0 negative") carry no checkable magnitude
            checked += 1
            if _is_supported(value, scope, scales, rounding) or _is_supported(value, query_numbers, _BARE_SCALES):
                continue
            if unit_ids and _is_supported(value, known, scales, rounding):
                if value not in misattributed:
                    misattributed.append(value)
            elif value not in unsupported:
                unsupported.append(value)
    valid_cited = [evidence_id for evidence_id in ids if evidence_id in store]
    missing = len(store) > 0 and not valid_cited
    return VerificationReport(
        passed=not invalid and not unsupported and not misattributed and not missing,
        cited_ids=valid_cited,
        invalid_citations=invalid,
        unsupported_numbers=unsupported,
        misattributed_numbers=misattributed,
        checked_numbers=checked,
        missing_citations=missing,
    )


def repair_answer(
    answer: dict[str, Any], report: VerificationReport, store: EvidenceStore, *, zh: bool
) -> tuple[dict[str, Any], list[str]]:
    """Remove unsupported statements and invalid citations. Returns ``(answer, notes)``."""
    repaired = dict(answer)
    notes: list[str] = []
    invalid = set(report.invalid_citations)
    unsupported = [*report.unsupported_numbers, *report.misattributed_numbers]

    def has_unsupported(text: str) -> bool:
        return bool(unsupported) and any(
            _is_supported(value, unsupported, _BARE_SCALES) for value in claim_numbers(text)
        )

    salvaged = 0

    def clean_sentence(sentence: str) -> str | None:
        nonlocal salvaged
        stripped = _CITATION.sub(lambda match: "" if match.group(1) in invalid else match.group(0), sentence)
        if not has_unsupported(stripped):
            return stripped
        # Salvage the clauses that only contain supported numbers.
        clauses = [clause for clause in re.split(r"(?<=[，,；;])", stripped) if clause]
        kept_clauses = [clause for clause in clauses if not has_unsupported(clause)]
        if not kept_clauses or not claim_numbers("".join(kept_clauses)):
            return None
        text = "".join(kept_clauses).rstrip("，,；; ")
        salvaged += 1
        return text + ("。" if re.search(r"[\u4e00-\u9fff]", text) else ".")

    sentences = [clean_sentence(sentence) for sentence in _split_sentences(str(answer.get("answer") or ""))]
    kept = [sentence for sentence in sentences if sentence and sentence.strip()]
    removed = len(sentences) - len(kept)
    repaired["answer"] = "".join(kept).strip()
    points = [clean_sentence(str(point)) for point in answer.get("key_points") or []]
    kept_points = [point.strip() for point in points if point and point.strip()]
    removed += len(points) - len(kept_points) + salvaged
    repaired["key_points"] = kept_points

    valid = [evidence_id for evidence_id in cited_ids(repaired) if evidence_id in store]
    if not valid and len(store):
        valid = store.ids()[:5]
    repaired["evidence_used"] = valid

    if removed:
        notes.append(
            f"已删除 {removed} 处无法由证据核实的表述。"
            if zh
            else f"Removed {removed} statement(s) not supported by evidence."
        )
    if invalid:
        notes.append("已移除不存在的证据引用。" if zh else "Removed citations to non-existent evidence.")
    if not repaired["answer"]:
        repaired["answer"] = (
            "现有证据不足以支持完整回答，以下仅列出可核实的要点。"
            if zh
            else "The available evidence is not enough for a complete answer; only verifiable points are listed."
        )
    return repaired, notes


def _evidence_numbers(store: EvidenceStore, evidence_ids: list[str] | None = None) -> list[float]:
    values: list[float] = []
    items = store.items() if evidence_ids is None else [store.get(evidence_id) for evidence_id in evidence_ids]
    for item in items:
        if item is not None:
            values.extend(item.numbers())
    return values


def _is_supported(
    value: float, known: list[float], scales: tuple[float, ...] = _SCALES, rounding: float | None = None
) -> bool:
    """``rounding`` is the stated number's precision tolerance; ``None`` keeps the legacy loose tolerance."""
    for base in known:
        for scale in scales:
            target = base * scale
            loose = max(0.011, abs(target) * 0.005)
            tolerance = loose if rounding is None else rounding + abs(target) * 0.0005 + 1e-9
            # Signs are compared loosely: "下跌 1.2%" legitimately restates a change of -1.2.
            if abs(abs(value) - abs(target)) <= tolerance:
                return True
    return False


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])|(?<=\.)\s+", text)
    return [part for part in parts if part]


def _format_number(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:g}"
