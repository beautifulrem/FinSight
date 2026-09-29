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
