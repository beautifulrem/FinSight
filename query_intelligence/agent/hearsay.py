"""Hearsay in the chat: "听说茅台市盈率只有15倍，是真的吗" is a claim to check, not only a question.

``claim_in_message`` finds the claim inside such a message (the same rule as the web UI's ``claimInMessage``);
``fact_check_for`` runs the deterministic claim check on it (no LLM) and returns the report as a plain dict,
which ``/chat`` (workflow) and the agent's result carry as ``fact_check`` for the UI to render inline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .claim_check import check_claim

logger = logging.getLogger(__name__)

_CUE = re.compile(
    r"听说|据说|传言|传闻|有人说|网上说|听人说|据传|号称|是真的吗|真的吗|是真的么|对吗|对不对|是不是真的|属实|靠谱吗|"
    r"\bis it true\b|\bi heard\b|\bsomeone said\b|\brumou?r\b|\bis (?:that|this) (?:true|right)\b",
    re.I,
)
_CONTENT = re.compile(
    r"\d|[一二两三四五六七八九十]+(?:点[〇零一二三四五六七八九]+)?(?:倍|成|%|元|亿)|涨了|跌了|大涨|大跌|涨停|跌停|"
    r"比.{1,12}(?:高|低|贵|便宜|多|少)|高于|低于|\b(?:rose|fell|higher than|lower than)\b",
    re.I,
)
_LEADING = re.compile(
    r"^\s*(?:我)?(?:听说|据说|传言|传闻|有人说|网上说|听人说|据传|I heard(?: that)?|someone said(?: that)?|"
    r"is it true(?: that)?)[，,：:\s]*",
    re.I,
)
_TRAILING = re.compile(
    r"[，,。\s]*(?:这|这个|这话|这是)?(?:是真的吗|真的吗|是真的么|对吗|对不对|是不是真的|属实吗?|靠谱吗|"
    r",?\s*is (?:that|this|it) (?:true|right))?\s*[？?！!。.]*\s*$",
    re.I,
)


def claim_in_message(message: str) -> str | None:
    """The claim inside a hearsay chat message, or None. "听说茅台市盈率只有15倍，是真的吗？" → "茅台市盈率只有15倍"."""
    text = (message or "").strip()
    if len(text) < 4 or not _CUE.search(text) or not _CONTENT.search(text):
        return None
    claim = _TRAILING.sub("", _LEADING.sub("", text)).strip()
    return claim if len(claim) >= 2 else None


def fact_check_for(message: str, *, service: Any, registry: Any, zh: bool = True) -> dict[str, Any] | None:
    """The claim-check report for a hearsay message, or None when the message is not hearsay or the check found
    no number, move or relation to compare. Best effort: a failure never breaks the chat answer."""
    claim = claim_in_message(message)
    if claim is None:
        return None
    try:
        report = check_claim(claim, service=service, registry=registry, zh=zh)
    except Exception:  # the answer does not depend on it
        logger.exception("inline fact check failed")
        return None
    if not report.checks:
        return None
    return report.model_dump()
