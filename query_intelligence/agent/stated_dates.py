"""(round 14, J6) Dates and periods a question names, placed relative to a reference date.

"茅台2027年1月4日的收盘价", "CMB's close on 2026-12-31", "Where do you see CATL trading in Q1 2027?" and "比亚迪
2027年一季度净利润" name an absolute date or period; before round 14 the forecast rules only knew relative words
("明天", "next week"), so such questions were answered with the latest close and no hedge. Each stated date or period
becomes a ``StatedPeriod`` (first and last day) and is compared with a reference date — the later of today and the
data's latest market date (``reference_date``): a period that starts after it has not happened (``future``), one
that ends on or before it has (``past``: a lookup), and one that contains it is ``in_progress`` ("Q4 2026" in
October, "2026年" in 2026).

Recognised forms: ``2027年1月4日`` / ``2027-01-04`` / ``2027/1/4`` / ``2027.1.4``, ``11月3日`` / ``十一月三号``
(the reference year), ``2026年12月``, ``2026年底`` (31 December), ``2027年一季度`` / ``2027年第1季度`` / ``2027Q1`` /
``明年一季度`` / ``四季度`` (the reference year), ``2027年上半年``, ``2027年``; ``Jan 4, 2027`` / ``4 January 2027``
/ ``December 31``, ``May 2027``, ``Q1 2027`` / ``the first quarter of 2027`` / ``1Q27``, ``H1 2027``, ``in 2027`` /
``FY2027`` / ``2027E``. A bare number is never a year ("能到2027吗" is a price).
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Literal

Status = Literal["past", "in_progress", "future"]

_CN_DIGITS = {
    "〇": 0,
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN = r"[〇零一二两三四五六七八九十]{1,3}"


def _int(text: str) -> int | None:
    """``12`` or a Chinese numeral up to 99 (``十一``, ``二十三``, ``三十一``)."""
    text = text.strip()
    if text.isdigit():
        return int(text)
    if not text or any(ch not in _CN_DIGITS and ch != "十" for ch in text):
        return None
    if "十" not in text:
        return _CN_DIGITS[text] if len(text) == 1 else None
    tens, _, units = text.partition("十")
    value = (_CN_DIGITS.get(tens, 0) if tens else 1) * 10 + (_CN_DIGITS.get(units, 0) if units else 0)
    return value if (not tens or tens in _CN_DIGITS) and (not units or units in _CN_DIGITS) else None


@dataclass(frozen=True)
class StatedPeriod:
    start: date
    end: date
    kind: Literal["day", "month", "quarter", "half", "year"]
    text: str
    explicit_year: bool  # the year was written ("2027年一季度") or relative ("明年"), not assumed ("四季度")

    def status(self, reference: date) -> Status:
        if self.start > reference:
            return "future"
        if self.end <= reference:
            return "past"
        return "in_progress"


def _day(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _month(year: int, month: int) -> tuple[date, date] | None:
    if not 1 <= month <= 12:
        return None
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _quarter(year: int, quarter: int) -> tuple[date, date] | None:
    if not 1 <= quarter <= 4:
        return None
    first = 3 * quarter - 2
    return date(year, first, 1), date(year, first + 2, calendar.monthrange(year, first + 2)[1])


def _half(year: int, half: int) -> tuple[date, date]:
    return (date(year, 1, 1), date(year, 6, 30)) if half == 1 else (date(year, 7, 1), date(year, 12, 31))


_YEAR = r"(?<![\d.])((?:19|20)\d{2})"
_RELATIVE_YEAR = r"(明年|后年|今年|去年|前年|本年|来年)"
_RELATIVE_OFFSET = {"明年": 1, "来年": 1, "后年": 2, "今年": 0, "本年": 0, "去年": -1, "前年": -2}

_ZH_YMD = re.compile(_YEAR + r"\s*(?:年|[-/.])\s*(\d{1,2}|" + _CN + r")\s*(?:月|[-/.])\s*(\d{1,2}|" + _CN + r")"
                     r"\s*(?:日|号)?(?![\d.])")  # fmt: skip
# no digit, 年 or Chinese numeral right before the month ("十一月" is one month, not "一月")
_ZH_MD = re.compile(
    r"(?<![\d年./〇零一二两三四五六七八九十-])(\d{1,2}|" + _CN + r")\s*月\s*(\d{1,2}|" + _CN + r")\s*(?:日|号)"
)
_ZH_YM = re.compile(_YEAR + r"\s*年\s*(\d{1,2}|" + _CN + r")\s*(?:月份?)(?!\s*(?:\d{1,2}|" + _CN + r")\s*(?:日|号))")
_ZH_YEAR_END = re.compile(_YEAR + r"\s*年\s*(?:底|末|年底|年末)")
_ZH_QUARTER = re.compile(
    r"(?:" + _YEAR + r"\s*年?\s*|" + _RELATIVE_YEAR + r"\s*)?(?<!前)(?<!前\s)(?:第\s*)?([一二三四1-4])\s*季度"
)
# cumulative: "2026年前三季度" is January to September
_ZH_CUMULATIVE = re.compile(r"(?:" + _YEAR + r"\s*年\s*|" + _RELATIVE_YEAR + r"\s*)?前\s*([一二三1-3])\s*(?:个)?季度")
_ZH_Q = re.compile(_YEAR + r"\s*年?\s*[Qq]\s*([1-4])(?!\d)")
_ZH_HALF = re.compile(r"(?:" + _YEAR + r"\s*年\s*|" + _RELATIVE_YEAR + r"\s*)([上下])半年")
_ZH_YEAR = re.compile(
    _YEAR + r"\s*(?:年|年度|财年)(?!\s*(?:\d{1,2}|" + _CN + r")\s*月|\s*(?:第\s*)?[一二三四1-4]\s*季度|\s*[Qq][1-4]|"
    r"\s*[上下]半年|\s*前[一二三]季度|\s*(?:底|末|初|以来|至今|迄今|年初至今))"
)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
    "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH_NAME = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
)
_EN_MDY = re.compile(
    r"\b" + _MONTH_NAME + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+((?:19|20)\d{2})\b)?", re.I | re.A
)
_EN_DMY = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH_NAME + r"\b\.?(?:,?\s+((?:19|20)\d{2})\b)?", re.I | re.A
)
_EN_MY = re.compile(r"\b" + _MONTH_NAME + r"\.?\s+((?:19|20)\d{2})\b", re.I | re.A)
_ORDINAL_QUARTER = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
_EN_QUARTER = re.compile(
    r"\bQ([1-4])\s*(?:of\s+)?(?:FY\s?)?'?((?:19|20)\d{2}|\d{2})\b|\b((?:19|20)\d{2})\s*Q([1-4])\b|"
    r"\b([1-4])Q\s*'?(\d{2}|(?:19|20)\d{2})\b|"
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?(?:(?:fiscal|FY)\s*)?((?:19|20)\d{2})\b|"
    r"\b(?:the\s+)?(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b"
    r"(?!\s+(?:of\s+)?(?:(?:fiscal|FY)\s*)?(?:19|20)\d{2})|\bQ([1-4])\b(?!\s*(?:of\s+)?'?\d)",
    re.I | re.A,
)
_EN_HALF = re.compile(
    r"\bH([12])\s*'?((?:19|20)\d{2}|\d{2})\b|\b(first|second)\s+half\s+(?:of\s+)?((?:19|20)\d{2})\b", re.I | re.A
)
_EN_YEAR = re.compile(
    r"\b(?:in|for|by|during|through|throughout|until|till|before)\s+(?:the\s+)?(?:year\s+)?"
    r"((?:19|20)\d{2})\b(?!\s*(?:yuan|CNY|RMB|points?|元|块|点|%))|"
    r"\bFY\s?'?((?:19|20)\d{2}|\d{2})\b|\b((?:19|20)\d{2})\s*(?:E|F)\b|\b((?:19|20)\d{2})(?:'s)?\s+"
    r"(?:full[- ]year|annual|earnings|profits?|net\s+(?:profit|income)|revenue|sales|EPS|results)\b",
    re.I | re.A,
)


_DATE_PREPOSITION = re.compile(r"(?:^|\b(?:on|by|before|after|until|till|from|since|for|of|in))\s*$", re.I)


def _year2(text: str) -> int:
    value = int(text)
    return value + 2000 if value < 100 else value


def stated_periods(text: str, reference: date) -> list[StatedPeriod]:
    """Every absolute date or period ``text`` names; a year left out is the reference year."""
    text = str(text or "")
    found: list[StatedPeriod] = []
    taken: list[tuple[int, int]] = []

    def add(match: re.Match[str], span: tuple[date, date] | None, kind: str, explicit: bool) -> None:
        if span is None or any(lo < match.end() and match.start() < hi for lo, hi in taken):
            return
        taken.append(match.span())
        found.append(StatedPeriod(span[0], span[1], kind, match.group(0), explicit))  # type: ignore[arg-type]

    for match in _ZH_YMD.finditer(text):
        month, day = _int(match.group(2)), _int(match.group(3))
        point = _day(int(match.group(1)), month or 0, day or 0) if month and day else None
        add(match, (point, point) if point else None, "day", True)
    for match in _ZH_YEAR_END.finditer(text):
        point = date(int(match.group(1)), 12, 31)
        add(match, (point, point), "day", True)
    for match in _ZH_YM.finditer(text):
        month = _int(match.group(2))
        add(match, _month(int(match.group(1)), month) if month else None, "month", True)
    for match in _ZH_MD.finditer(text):
        month, day = _int(match.group(1)), _int(match.group(2))
        point = _day(reference.year, month, day) if month and day else None
        add(match, (point, point) if point else None, "day", False)
    for match in _ZH_CUMULATIVE.finditer(text):
        quarters = _int(match.group(3)) or 0
        year = int(match.group(1)) if match.group(1) else reference.year + _RELATIVE_OFFSET.get(match.group(2) or "", 0)
        span = _quarter(year, quarters)
        add(match, (date(year, 1, 1), span[1]) if span else None, "quarter", bool(match.group(1) or match.group(2)))
    for match in _ZH_Q.finditer(text):
        add(match, _quarter(int(match.group(1)), int(match.group(2))), "quarter", True)
    for match in _ZH_QUARTER.finditer(text):
        quarter = _int(match.group(3))
        if match.group(1):
            year, explicit = int(match.group(1)), True
        elif match.group(2):
            year, explicit = reference.year + _RELATIVE_OFFSET[match.group(2)], True
        else:
            year, explicit = reference.year, False
        add(match, _quarter(year, quarter) if quarter else None, "quarter", explicit)
    for match in _ZH_HALF.finditer(text):
        year = int(match.group(1)) if match.group(1) else reference.year + _RELATIVE_OFFSET[match.group(2)]
        add(match, _half(year, 1 if match.group(3) == "上" else 2), "half", True)
    for match in _ZH_YEAR.finditer(text):
        year = int(match.group(1))
        add(match, (date(year, 1, 1), date(year, 12, 31)), "year", True)

    for match in _EN_MDY.finditer(text):
        if (
            match.group(1).lower() == "may"
            and not match.group(3)
            and not _DATE_PREPOSITION.search(text[: match.start()])
        ):
            continue  # the modal ("BYD may 3 times…"); "on May 5" or "May 5, 2027" is a date
        month = _MONTHS[match.group(1).lower()[:3]]
        year = int(match.group(3)) if match.group(3) else reference.year
        point = _day(year, month, int(match.group(2)))
        add(match, (point, point) if point else None, "day", bool(match.group(3)))
    for match in _EN_DMY.finditer(text):
        if match.group(2).lower() == "may" and not match.group(3):
            continue  # "the top 5 may outperform"
        month = _MONTHS[match.group(2).lower()[:3]]
        year = int(match.group(3)) if match.group(3) else reference.year
        point = _day(year, month, int(match.group(1)))
        add(match, (point, point) if point else None, "day", bool(match.group(3)))
    for match in _EN_MY.finditer(text):
        add(match, _month(int(match.group(2)), _MONTHS[match.group(1).lower()[:3]]), "month", True)
    for match in _EN_QUARTER.finditer(text):
        groups = match.groups()
        if groups[0]:
            quarter, year, explicit = int(groups[0]), _year2(groups[1]), True
        elif groups[2]:
            quarter, year, explicit = int(groups[3]), int(groups[2]), True
        elif groups[4]:
            quarter, year, explicit = int(groups[4]), _year2(groups[5]), True
        elif groups[6]:
            quarter, year, explicit = _ORDINAL_QUARTER[groups[6].lower()], int(groups[7]), True
        elif groups[8]:
            quarter, year, explicit = _ORDINAL_QUARTER[groups[8].lower()], reference.year, False
        else:
            quarter, year, explicit = int(groups[9]), reference.year, False
        add(match, _quarter(year, quarter), "quarter", explicit)
    for match in _EN_HALF.finditer(text):
        if match.group(1):
            half, year = int(match.group(1)), _year2(match.group(2))
        else:
            half, year = (1 if match.group(3).lower() == "first" else 2), int(match.group(4))
        add(match, _half(year, half), "half", True)
    for match in _EN_YEAR.finditer(text):
        raw = next(group for group in match.groups() if group)
        year = _year2(raw)
        add(match, (date(year, 1, 1), date(year, 12, 31)), "year", True)
    return found


def statuses(periods: Iterable[StatedPeriod], reference: date) -> set[Status]:
    return {period.status(reference) for period in periods}


def reference_date(today: date, data_dates: Iterable[str | date | None] = ()) -> date:
    """The later of ``today`` and the latest data date: a date after it has not happened yet."""
    latest = today
    for value in data_dates:
        if not value:
            continue
        try:
            point = value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
        except ValueError:
            continue
        latest = max(latest, point)
    return latest
