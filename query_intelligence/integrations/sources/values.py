"""Value normalization shared by live providers.

Upstream tables mix ``NaN`` placeholders (for unreleased periods), ``--``/``---`` markers, percent
strings, and Chinese magnitude suffixes (``445.17亿``). Returning ``NaN`` would also break JSON
responses, so every numeric field goes through ``to_number``.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

_MAGNITUDES = (("万亿", 1e12), ("亿", 1e8), ("万", 1e4))


def to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    text = str(value).strip().replace(",", "")
    if not text or text.startswith("--") or text.lower() in {"nan", "none", "null", "false"}:
        return None
    multiplier = 1.0
    for suffix, factor in _MAGNITUDES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            multiplier = factor
            break
    text = text.rstrip("%").rstrip("元").strip()
    try:
        number = float(text) * multiplier
    except ValueError:
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and (not value.strip() or value.strip().startswith("---"))


def to_iso_date(value: Any) -> str | None:
    """Normalize dates such as ``2026-09-24``, ``20260924``, ``2026年08月份`` or pandas timestamps."""
    if is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date().isoformat()
    text = str(value).strip()
    if "年" in text:
        digits = text.replace("年", "-").replace("月份", "").replace("月", "").replace("日", "")
        parts = [part for part in digits.split("-") if part]
        if len(parts) >= 2 and all(part.isdigit() for part in parts[:3]):
            day = int(parts[2]) if len(parts) > 2 else 1
            return date(int(parts[0]), int(parts[1]), day).isoformat()
    for fmt, length in (("%Y-%m-%d", 10), ("%Y/%m/%d", 10), ("%Y%m%d", 8), ("%Y-%m", 7)):
        try:
            return datetime.strptime(text[:length], fmt).date().isoformat()
        except ValueError:
            continue
    return text
