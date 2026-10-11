"""(round 17, review-13 M3) A short name of a covered company followed by a business-line word names another company.

"美的置业" (a Hong Kong-listed property developer), "中芯集成" (another STAR-market company), "平安证券" (an unlisted
subsidiary) and "Midea Real Estate" are absent from the entity master, but each starts with a short name of a covered
company (美的 / 中芯 / 平安 / Midea). Before round 17 the short name matched exactly and the answer gave the covered
company's figures. A short name (not the official name) followed by one of the words below is a *lookalike name*:
it is not resolved to the covered company, and the question is answered as naming a company FinSight does not cover.

Exceptions, so a descriptive word keeps its company:
* the extended name is the covered company's own name or starts it (中芯 + 国际 = 中芯国际, 中信 + 证券 = 中信证券);
* the word names the company's own industry ("平安保险" is 中国平安, an insurer; "招行银行" is 招商银行): the word's
  family (保险 / 人寿 / 寿险 / 财险, 银行, 证券 / 券商) appears in the company's name or industry;
* legal-form words (集团, 股份, 有限公司) are not business-line words: "比亚迪股份有限公司" is 比亚迪.
"""

from __future__ import annotations

import re

# Business-line words that make a short name another company's name (longest first in the pattern).
ZH_SUFFIXES = (
    "置业", "置地", "房地产", "地产", "物业", "集成", "证券", "银行", "保险", "人寿", "财险", "寿险", "信托", "期货", "基金",
    "资本", "资管", "租赁", "控股", "投资", "国际", "电讯", "电信", "基建", "实业", "发展", "健康", "医疗", "电子", "物流",
    "金融", "好医生",
)  # fmt: skip
# Words that may follow a business-line word inside the same name ("美的置业控股", "中信国际电讯集团").
_ZH_TAIL = ("控股", "集团", "股份有限公司", "有限公司", "股份", "国际", "电讯", "医疗", "科技", "公司")
EN_SUFFIXES = (
    r"real\s+estate", r"propert(?:y|ies)", r"securities", r"bank(?:ing)?", r"insurance", r"life", r"holdings?",
    r"international", r"telecom(?:munications)?", r"electronics?", r"capital", r"land", r"development",
    r"leasing", r"trust", r"investments?", r"asset\s+management", r"futures", r"logistics", r"health(?:care)?",
    r"good\s+doctor", r"financial",
)  # fmt: skip
_ZH_PATTERN = re.compile(
    "(?:" + "|".join(sorted(ZH_SUFFIXES, key=len, reverse=True)) + ")(?:" + "|".join(_ZH_TAIL) + ")*"
)
_EN_PATTERN = re.compile(r"\s+(?:" + "|".join(EN_SUFFIXES) + r")\b(?:\s+(?:holdings?|group|limited|ltd\.?))?", re.I)
# A word that names the company's own industry is a descriptor, not another company.
_FAMILIES = (
    ("保险", "人寿", "寿险", "财险", "insurance", "life"),
    ("银行", "bank", "banking"),
    ("证券", "券商", "securities"),
)


def _family(word: str) -> tuple[str, ...] | None:
    lowered = word.lower()
    return next((family for family in _FAMILIES if any(lowered.startswith(term) for term in family)), None)


def _describes(word: str, owner: str, industry: str) -> bool:
    """The business-line word names the owner's own industry, or extends into the owner's own name."""
    family = _family(word)
    return bool(family) and any(term in owner or term in industry for term in family if not term.isascii())


def lookalike_end(text: str, end: int, *, alias: str, owner: str, industry: str = "") -> int | None:
    """Where a lookalike name that starts with ``alias`` (ending at ``end`` in ``text``) ends, or ``None``.

    ``owner`` is the covered company's canonical (Chinese) name, ``industry`` its industry name."""
    rest = text[end:]
    if alias.isascii():
        match = _EN_PATTERN.match(rest)
        if match is None:
            return None
        word = match.group(0).strip().split()[0]
        if word[:1].islower() and not text.islower():
            # a name is capitalised ("Midea Real Estate"); "where does BYD land?" and "CATL international sales"
            # use the word as an ordinary word (an all-lower-case question is read as written)
            return None
        return None if _describes(word, owner, industry) else end + match.end()
    match = _ZH_PATTERN.match(rest)
    if match is None:
        return None
    word = next(suffix for suffix in sorted(ZH_SUFFIXES, key=len, reverse=True) if rest.startswith(suffix))
    if owner.startswith(alias + word) or (alias + word) in owner or _describes(word, owner, industry):
        return None
    return end + match.end()
