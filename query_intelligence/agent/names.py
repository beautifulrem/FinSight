"""English display names for listed targets, from the alias table (``data/synonym_dict.json``).

``display_en`` holds the written form ("Kweichow Moutai", "CATL"); a name with English aliases but no display
entry falls back to its longest alias, title-cased. The English UI shows these next to (or instead of) the
Chinese canonical name; the API adds them as ``name_en`` so the browser never keeps its own table.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..data_loader import load_entities, load_seed_entities, load_synonyms

_LATIN = re.compile(r"[A-Za-z]{2}")


@lru_cache(maxsize=1)
def _english_by_name() -> dict[str, str]:
    synonyms = load_synonyms()
    names: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}
    for alias, canonical in (synonyms.get("alias") or {}).items():
        if _LATIN.search(alias) and alias.isascii():
            aliases.setdefault(str(canonical), []).append(alias)
    for canonical, found in aliases.items():
        names[canonical] = max(found, key=len).title()
    names.update({str(key): str(value) for key, value in (synonyms.get("display_en") or {}).items()})
    return names


@lru_cache(maxsize=1)
def _aliases_by_name() -> dict[str, tuple[str, ...]]:
    found: dict[str, list[str]] = {}
    for alias, canonical in (load_synonyms().get("alias") or {}).items():
        if _LATIN.search(alias) and alias.isascii():
            found.setdefault(str(canonical), []).append(alias)
    for canonical, english in (load_synonyms().get("display_en") or {}).items():
        found.setdefault(str(canonical), []).append(str(english))
    return {name: tuple(sorted(set(items), key=len, reverse=True)) for name, items in found.items()}


def english_aliases(name: str | None) -> tuple[str, ...]:
    """How a target may be written in English ("moutai", "kweichow moutai"), longest first."""
    return _aliases_by_name().get(name or "", ())


@lru_cache(maxsize=1)
def _name_by_symbol() -> dict[str, str]:
    names: dict[str, str] = {}
    for rows in (load_seed_entities(), load_entities()):
        for row in rows:
            symbol, name = str(row.get("symbol") or "").upper(), str(row.get("canonical_name") or "")
            if symbol and name and name in _english_by_name():
                names.setdefault(symbol, name)
    return names


def english_name(name: str | None, symbol: str | None = None) -> str | None:
    """The English name of a target ("贵州茅台" → "Kweichow Moutai"), by name or else by symbol; None when
    the alias table has none."""
    english = _english_by_name()
    if name and name in english:
        return english[name]
    if symbol:
        canonical = _name_by_symbol().get(symbol.upper())
        if canonical:
            return english.get(canonical)
    if name and name.isascii() and _LATIN.search(name):
        return name  # already English
    return None


# Industry names used by the offline and live industry snapshots.
INDUSTRY_EN = {
    "白酒": "baijiu (liquor)",
    "保险": "insurance",
    "券商": "brokerage",
    "证券": "securities",
    "银行": "banking",
    "宽基指数": "broad-based index",
    "成长指数": "growth index",
    "新能源": "new energy",
    "医药": "pharmaceuticals",
    "半导体": "semiconductors",
}
_BRACKETED = re.compile(r"(\[[^\]]*\])")


def english_display(text: str) -> str:
    """Replace Chinese names of listed targets and industries in English text with their English names.

    Citation brackets ("[industry_白酒]") are left untouched, so evidence ids keep matching.
    """
    if not text or not re.search(r"[\u4e00-\u9fff]", text):
        return text
    # English names with digits ("CSI 300 ETF") are skipped: the verifier would read the digits as a claimed number.
    table = {
        **{
            name: english
            for name, english in _english_by_name().items()
            if not name.isascii() and not re.search(r"\d", english)
        },
        **INDUSTRY_EN,
    }
    names = sorted((name for name in table if name in text), key=len, reverse=True)
    if not names:
        return text
    pattern = re.compile("|".join(re.escape(name) for name in names))
    parts = _BRACKETED.split(text)
    return "".join(part if part.startswith("[") else pattern.sub(lambda m: table[m.group(0)], part) for part in parts)
