from __future__ import annotations

import re

ENGLISH_PATTERN_REWRITES: list[tuple[str, str]] = [
    (r"(?i)\bwhat(?:'s| is) the difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", r"\1和\2有什么区别"),
    (r"(?i)\bwhich is better[, ]+\s*(.+?)\s+or\s+(.+?)(?:\?|$)", r"\1和\2哪个好"),
    (r"(?i)\bwhich is better[, ]+\s*(.+?)\s+and\s+(.+?)(?:\?|$)", r"\1和\2哪个好"),
    (r"(?i)\bany recent announcements from\s+(.+?)(?:\?|$)", r"\1最近有什么公告"),
    (r"(?i)\bany recent news about\s+(.+?)(?:\?|$)", r"最近有哪些\1新闻"),
    (r"(?i)\bany recent news on\s+(.+?)(?:\?|$)", r"最近有哪些\1新闻"),
    (r"(?i)\bdoes\s+(.+?)\s+affect\s+(.+?)(?:\?|$)", r"\1会影响\2吗"),
    (r"(?i)\bis the\s+(.+?)\s+sector\s+still buyable(?:\?|$)", r"\1行业最近还能买吗"),
    (r"(?i)\bis\s+(.+?)\s+still worth holding(?:\?|$)", r"\1还值得持有吗"),
]

ENGLISH_TERM_REWRITES: list[tuple[str, str]] = [
    (r"(?i)\bwhy did\b", "为什么"),
    (r"(?i)\bwhy is\b", "为什么"),
    (r"(?i)\bwhy\b", "为什么"),
    (r"(?i)\bfall\b", "跌"),
    (r"(?i)\bfell\b", "跌"),
    (r"(?i)\bdrop(?:ped)?\b", "跌"),
    (r"(?i)\brise\b", "涨"),
    (r"(?i)\bup\b", "涨"),
    (r"(?i)\bnews\b", "新闻"),
    (r"(?i)\bannouncements?\b", "公告"),
    (r"(?i)\bfundamentals?\b", "基本面"),
    (r"(?i)\bvaluation\b", "估值"),
    (r"(?i)\brisk\b", "风险"),
    (r"(?i)\bvolatility\b", "波动"),
    (r"(?i)\bvolatile\b", "波动"),
    (r"(?i)\bdrawdown\b", "回撤"),
    (r"(?i)macro policy", "宏观政策"),
    (r"(?i)liquor sector", "白酒行业"),
    (r"(?i)liquor", "白酒"),
    (r"(?i)sector", "行业"),
    (r"(?i)industry", "行业"),
    (r"(?i)\bstill worth holding\b", "还值得持有"),
    (r"(?i)\bworth holding\b", "值得持有"),
    (r"(?i)\bstill buyable\b", "还能买"),
    (r"(?i)\bbuyable\b", "能买"),
    (r"(?i)\bcompare\b", "比较"),
    (r"(?i)\bdifference\b", "区别"),
]


class QueryNormalizer:
    def __init__(self, synonyms: dict) -> None:
        self.synonyms = synonyms

    def normalize(self, raw_query: str) -> tuple[str, list[str]]:
        text = raw_query.strip()
        text = text.replace("？", "?").replace("，", " ").replace("。", " ").replace("！", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(?i)e\s*t\s*f", "ETF", text)
        text = re.sub(r"(?i)l\s*o\s*f", "LOF", text)
        text = re.sub(r"(?i)etf", "ETF", text)
        text = re.sub(r"(?i)lof", "LOF", text)
        trace: list[str] = []

        text, english_trace = self._rewrite_english_patterns(text)
        trace.extend(english_trace)

        for source, target in sorted(
            self.synonyms.get("alias", {}).items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            updated_text, replacements = self._replace_alias_mentions(text, source, target)
            if replacements:
                text = updated_text
                trace.extend([f"alias_exact: {source}->{target}"] * replacements)

        normalized = re.sub(r"([?])", " ", text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized, trace

    def _replace_alias_mentions(self, text: str, source: str, target: str) -> tuple[str, int]:
        if not source or source == target:
            return text, 0
        is_ascii_source = source.isascii()
        if is_ascii_source:
            pattern = re.compile(re.escape(source), flags=re.IGNORECASE)
            if not pattern.search(text):
                return text, 0
        elif source not in text:
            return text, 0

        pieces: list[str] = []
        last_end = 0
        replacements = 0
        protected_prefix = target[: max(len(target) - len(source), 0)]

        iterator = re.finditer(re.escape(source), text, flags=re.IGNORECASE) if is_ascii_source else re.finditer(re.escape(source), text)
        for match in iterator:
            start, end = match.span()
            current_slice = text[max(0, start - len(protected_prefix)) : end]
            if protected_prefix and current_slice.lower() == target.lower():
                continue

            pieces.append(text[last_end:start])
            pieces.append(target)
            last_end = end
            replacements += 1

        if replacements == 0:
            return text, 0

        pieces.append(text[last_end:])
        return "".join(pieces), replacements

    def detect_time_scope(self, query: str) -> str:
        for phrase, value in self.synonyms.get("time_scope", {}).items():
            if self._phrase_in_query(query, phrase):
                return value
        return "unspecified"

    def detect_operation(self, query: str) -> str:
        for phrase, value in self.synonyms.get("operation", {}).items():
            if self._phrase_in_query(query, phrase):
                return value
        return "unknown"

    def _rewrite_english_patterns(self, text: str) -> tuple[str, list[str]]:
        rewritten = text
        trace: list[str] = []
        for pattern, replacement in ENGLISH_PATTERN_REWRITES:
            updated, count = re.subn(pattern, replacement, rewritten)
            if count:
                rewritten = updated
                trace.extend([f"english_pattern:{pattern}->{replacement}"] * count)
        for pattern, replacement in ENGLISH_TERM_REWRITES:
            updated, count = re.subn(pattern, replacement, rewritten)
            if count:
                rewritten = updated
                trace.extend([f"english_term:{pattern}->{replacement}"] * count)
        rewritten = re.sub(r"(?i)\bor\b", "和", rewritten)
        rewritten = re.sub(r"(?i)\band\b", "和", rewritten)
        rewritten = re.sub(r"(?i)\bvs\.?\b", "和", rewritten)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        return rewritten, trace

    def _phrase_in_query(self, query: str, phrase: str) -> bool:
        if phrase.isascii():
            return phrase.lower() in query.lower()
        return phrase in query
