"""Evidence verifier: citations must exist and numbers must be traceable to tool outputs.

Numbers are matched against every numeric value carried by the run's evidence, allowing common
unit conversions (percent <-> ratio, 万/亿, thousand/million/billion) and rounding. Dates, evidence
ids, ticker codes, and window parameters such as ``RSI(14)`` or ``近5日`` are not treated as factual
claims.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .evidence import EvidenceStore, extract_numbers

_SCALES = (1.0, 100.0, 0.01, 1e-4, 1e-8, 1e4, 1e8, 1e-3, 1e-6, 1e-9, 1e3, 1e6, 1e9)
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
    checked_numbers: int = 0
    missing_citations: bool = False

    def feedback(self) -> str:
        problems = []
        if self.invalid_citations:
            problems.append(f"These evidence ids do not exist in this run: {', '.join(self.invalid_citations)}.")
        if self.unsupported_numbers:
            values = ", ".join(_format_number(value) for value in self.unsupported_numbers)
            problems.append(f"These numbers are not found in any tool output: {values}.")
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


def claim_numbers(text: str) -> list[float]:
    cleaned = _CITATION.sub(" ", text)
    for pattern in (*_DATE_PATTERNS, *_PARAMETER_PATTERNS):
        cleaned = pattern.sub(" ", cleaned)
    return extract_numbers(cleaned)


def verify_answer(answer: dict[str, Any], store: EvidenceStore, *, query: str = "") -> VerificationReport:
    ids = cited_ids(answer)
    invalid = [evidence_id for evidence_id in ids if evidence_id not in store]
    known = _evidence_numbers(store)
    query_numbers = claim_numbers(query)
    unsupported: list[float] = []
    checked = 0
    for text in answer_texts(answer):
        for value in claim_numbers(text):
            checked += 1
            if _is_supported(value, known) or _is_supported(value, query_numbers):
                continue
            if value not in unsupported:
                unsupported.append(value)
    valid_cited = [evidence_id for evidence_id in ids if evidence_id in store]
    missing = len(store) > 0 and not valid_cited
    return VerificationReport(
        passed=not invalid and not unsupported and not missing,
        cited_ids=valid_cited,
        invalid_citations=invalid,
        unsupported_numbers=unsupported,
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
    unsupported = report.unsupported_numbers

    def clean_sentence(sentence: str) -> str | None:
        stripped = _CITATION.sub(lambda match: "" if match.group(1) in invalid else match.group(0), sentence)
        if unsupported and any(_is_supported(value, unsupported) for value in claim_numbers(stripped)):
            return None
        return stripped

    sentences = [clean_sentence(sentence) for sentence in _split_sentences(str(answer.get("answer") or ""))]
    kept = [sentence for sentence in sentences if sentence and sentence.strip()]
    removed = len(sentences) - len(kept)
    repaired["answer"] = "".join(kept).strip()
    points = [clean_sentence(str(point)) for point in answer.get("key_points") or []]
    kept_points = [point.strip() for point in points if point and point.strip()]
    removed += len(points) - len(kept_points)
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


def _evidence_numbers(store: EvidenceStore) -> list[float]:
    values: list[float] = []
    for item in store.items():
        values.extend(item.numbers())
    return values


def _is_supported(value: float, known: list[float]) -> bool:
    for base in known:
        for scale in _SCALES:
            target = base * scale
            tolerance = max(0.011, abs(target) * 0.005)
            # Signs are compared loosely: "下跌 1.2%" legitimately restates a change of -1.2.
            if abs(abs(value) - abs(target)) <= tolerance:
                return True
    return False


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])|(?<=\.)\s+", text)
    return [part for part in parts if part]


def _format_number(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:g}"
