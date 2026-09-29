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
    (r"(?i)baijiu sector", "白酒行业"),
    (r"(?i)insurance sector", "保险行业"),
    (r"(?i)liquor", "白酒"),
    (r"(?i)sector", "行业"),
    (r"(?i)industry", "行业"),
    (r"(?i)\bstill worth holding\b", "还值得持有"),
    (r"(?i)\bworth holding\b", "值得持有"),
    (r"(?i)\bstill buyable\b", "还能买"),
    (r"(?i)\bbuyable\b", "能买"),
    (r"(?i)\bcompare\b", "比较"),
    (r"(?i)\bdifference\b", "区别"),
    # English metric names the alias table only knows as abbreviations ("return on equity" -> ROE)
    (r"(?i)\breturns? on (?:shareholders'? )?equity\b", "ROE"),
    (r"(?i)\bprice[- ]to[- ]book(?: (?:ratio|multiple|value))?\b|\bbook(?:[- ]value)? multiple\b", "P/B"),
    (r"(?i)\bprice[- ]to[- ]earnings(?: (?:ratio|multiple))?\b|\bearnings multiple\b|\bP/E ratio\b", "P/E"),
]


_LATIN_WORD = re.compile(r"[A-Za-z]+")
_MIN_TYPO_TOKEN = 6  # shorter names ("BYD", "Gree") differ from ordinary words by one letter too often


class QueryNormalizer:
    def __init__(self, synonyms: dict, security_names: set[str] | frozenset[str] | None = None) -> None:
        self.synonyms = synonyms
        # English aliases of listed securities, for typo correction ("Moutia" -> "moutai"); off without the list.
        self._typo_aliases: list[tuple[str, tuple[str, ...]]] = sorted(
            (
                (alias, tuple(alias.lower().split()))
                for alias, target in (synonyms.get("alias") or {}).items()
                if security_names
                and target in security_names
                and alias.isascii()
                and all(part.isalpha() for part in alias.split())
                and any(len(part) >= _MIN_TYPO_TOKEN for part in alias.split())
            ),
            key=lambda item: len(item[1]),
            reverse=True,
        )

    def normalize(self, raw_query: str) -> tuple[str, list[str]]:
        text = raw_query.strip()
        text = text.replace("？", "?").replace("，", " ").replace("。", " ").replace("！", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(?i)e\s*t\s*f", "ETF", text)
        text = re.sub(r"(?i)l\s*o\s*f", "LOF", text)
        text = re.sub(r"(?i)etf", "ETF", text)
        text = re.sub(r"(?i)lof", "LOF", text)
        trace: list[str] = []

        text, typo_trace = self._correct_english_typos(text)
        trace.extend(typo_trace)
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

    def _correct_english_typos(self, text: str) -> tuple[str, list[str]]:
        """English names of listed securities with one typo ("Kweichow Moutia", "Wuliangey's") -> the alias.

        The same idea as the Chinese typo matching ("贵州矛台"), with a stricter rule because short English words
        are often one letter apart: the words must line up with the alias's words, exactly one word may differ, that
        word has at least six letters, the same first letter, and one edit (a swap of two neighbouring letters
        counts as one; two edits from ten letters on); a plural or "-ed" form of the alias word is not a typo.
        """
        if not self._typo_aliases or not _LATIN_WORD.search(text):
            return text, []
        from rapidfuzz.distance import OSA

        words = list(_LATIN_WORD.finditer(text))
        taken: list[tuple[int, int, str]] = []
        for alias, parts in self._typo_aliases:
            size = len(parts)
            for index in range(len(words) - size + 1):
                window = words[index : index + size]
                start, end = window[0].start(), window[-1].end()
                if any(not (end <= a or start >= b) for a, b, _alias in taken):
                    continue
                if any(text[left.end() : right.start()].strip() for left, right in zip(window, window[1:], strict=False)):
                    continue  # the alias's words are separated by spaces only
                written = [match.group(0).lower() for match in window]
                differing = [(mine, theirs) for mine, theirs in zip(written, parts, strict=True) if mine != theirs]
                if len(differing) != 1:
                    continue
                mine, theirs = differing[0]
                limit = 2 if len(theirs) >= 10 else 1
                if (
                    len(theirs) < _MIN_TYPO_TOKEN
                    or mine[0] != theirs[0]
                    or mine in {f"{theirs}s", f"{theirs}es", f"{theirs}ed", f"{theirs}d"}
                    or theirs in {f"{mine}s", f"{mine}es"}
                    or OSA.distance(mine, theirs) > limit
                ):
                    continue
                taken.append((start, end, alias))
        if not taken:
            return text, []
        trace = []
        for start, end, alias in sorted(taken, reverse=True):
            trace.append(f"alias_typo_en: {text[start:end]}->{alias}")
            text = f"{text[:start]}{alias}{text[end:]}"
        return text, trace[::-1]

    def _replace_alias_mentions(self, text: str, source: str, target: str) -> tuple[str, int]:
        if not source or source == target:
            return text, 0
        is_ascii_source = source.isascii()
        # English aliases match whole words only ("zte" must not fire inside "ztest").
        pattern = (
            re.compile(rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])", flags=re.IGNORECASE)
            if is_ascii_source
            else None
        )
        if pattern is not None:
            if not pattern.search(text):
                return text, 0
        elif source not in text:
            return text, 0

        pieces: list[str] = []
        last_end = 0
        replacements = 0
        protected_prefix = target[: max(len(target) - len(source), 0)]

        iterator = pattern.finditer(text) if pattern is not None else re.finditer(re.escape(source), text)
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
