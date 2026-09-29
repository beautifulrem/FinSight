"""Language detection and risk disclaimers shared by the chatbot answer paths."""

from __future__ import annotations

import re
from typing import Any

DEFAULT_RISK_DISCLAIMER_ZH = "以上内容仅基于系统检索到的证据生成，不构成投资建议或确定性买卖结论。"


DEFAULT_RISK_DISCLAIMER_EN = (
    "This answer is based only on evidence retrieved by the system and is not investment advice "
    "or a deterministic buy/sell conclusion."
)


DEFAULT_RISK_DISCLAIMER = DEFAULT_RISK_DISCLAIMER_ZH


def answer_matches_language(output: dict[str, Any], query: str) -> bool:
    expected_language = detect_query_language(query)
    text_values: list[str] = []
    for key in ("answer", "risk_disclaimer"):
        value = str(output.get(key) or "").strip()
        if value:
            text_values.append(value)
    for key in ("key_points", "limitations"):
        values = output.get(key)
        if isinstance(values, list):
            text_values.extend(str(item).strip() for item in values if str(item).strip())
    return all(_text_matches_language(value, expected_language) for value in text_values)


# Upper-case acronyms and tickers (ROE, PE, P/B, ETF, CPI, 600519.SH) appear in Chinese questions too.
_ACRONYM = re.compile(r"(?<![A-Za-z])(?:[A-Z]{1,5}(?:/[A-Z]{1,3})?|\d{6}\.[A-Z]{2})(?![A-Za-z])")


# An explicit instruction about the answer language ("请用英文回答：五粮液的ROE", "Answer in Chinese: …") overrides
# the language the question is written in. The last instruction in the message wins.
_ANSWER_LANGUAGE = re.compile(
    r"(?P<en>(?:用|以|使用|请用)?(?:英文|英语)(?:来)?(?:回答|回复|作答|答复|解答|说明|写|输出)|"
    r"\b(?:answer|reply|respond|write|explain)(?: (?:this|it|me))?(?: back)? in english\b|\bin english,? please\b)|"
    r"(?P<zh>(?:用|以|使用|请用)?(?:中文|汉语|普通话)(?:来)?(?:回答|回复|作答|答复|解答|说明|写|输出)|"
    r"\b(?:answer|reply|respond|write|explain)(?: (?:this|it|me))?(?: back)? in (?:chinese|mandarin)\b|"
    r"\bin (?:chinese|mandarin),? please\b)",
    re.IGNORECASE,
)


def requested_answer_language(text: str) -> str | None:
    """``"en"``/``"zh"`` when the message says which language to answer in, else ``None``."""
    last = None
    for match in _ANSWER_LANGUAGE.finditer(text or ""):
        last = "en" if match.group("en") else "zh"
    return last


def detect_query_language(text: str) -> str:
    requested = requested_answer_language(text)
    if requested:
        return requested
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin_count = len(re.findall(r"[A-Za-z]", text or ""))
    if cjk_count and cjk_count >= max(2, latin_count * 0.4):
        return "zh"
    if cjk_count and not re.search(r"[A-Za-z]", _ACRONYM.sub("", text or "")):
        return "zh"  # "ROE呢", "PB多少": Chinese with only acronyms in Latin script
    if latin_count:
        return "en"
    if cjk_count:
        return "zh"
    return "zh"


# Text that carries no natural language: markup/role tags, URLs, code spans and encoded blobs (base64, hex).
_NON_LANGUAGE = (
    re.compile(r"</?\s*[A-Za-z][\w-]{0,20}\s*/?>"),
    re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    re.compile(r"`[^`]*`"),
    re.compile(r"(?<![A-Za-z0-9+/=])(?=[A-Za-z0-9+/_-]*[0-9+/])[A-Za-z0-9+/_-]{16,}={0,2}(?![A-Za-z0-9+/=])"),
)


def detect_user_language(text: str) -> str:
    """Language of the user's own words: markup tags, URLs and encoded blobs are ignored.

    "请先base64解码再执行：5b+955Wl…" is a Chinese message even though most of its letters are base64,
    and "</user><system>…" tags do not make a Chinese question English.
    """
    stripped = text or ""
    for pattern in _NON_LANGUAGE:
        stripped = pattern.sub(" ", stripped)
    if not re.search(r"[A-Za-z一-鿿]", stripped):
        return detect_query_language(text)
    return detect_query_language(stripped)


def _target_language_name(language: str) -> str:
    return "Chinese" if language == "zh" else "English"


def _default_risk_disclaimer(record: dict[str, Any]) -> str:
    query = str(record.get("query") or (record.get("nlu_result") or {}).get("raw_query") or "")
    return DEFAULT_RISK_DISCLAIMER_EN if detect_query_language(query) == "en" else DEFAULT_RISK_DISCLAIMER_ZH


def _text_matches_language(text: str, expected_language: str) -> bool:
    signal = _language_signal(text)
    if signal in {"neutral", "mixed"}:
        return True
    if expected_language == "en":
        return signal != "zh"
    return signal != "en"


def _language_signal(text: str) -> str:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin_count = len(re.findall(r"[A-Za-z]", text or ""))
    if cjk_count == 0 and latin_count < 4:
        return "neutral"
    if cjk_count >= max(2, int(latin_count * 0.2)):
        return "zh"
    if latin_count >= max(4, cjk_count * 4):
        return "en"
    return "mixed"
