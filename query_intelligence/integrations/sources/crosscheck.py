"""Cross-source validation of live fundamentals (Sina financial indicators vs THS financial abstract).

Why: the two free sources disagree for some stocks. In the round-1 review, Sina reported 000858.SZ
H1-2026 revenue YoY -46.15% and net-profit YoY -55.32%, while THS (and the company's own release
quoted in the news) gave +20.87% and +89.30%. A single-source pipeline passes whichever it fetched
first into the answer.

What is checked, per report period:

1. **Range.** Values outside plausible bounds (for example revenue YoY below -100%, ROE beyond ±200%)
   are dropped and flagged, never served.
2. **Period alignment.** Only values for the same report period are compared or merged. When the
   latest periods differ, the newer period wins and the mismatch is recorded.
3. **Cumulative vs quarter convention.** A-share reports are cumulative year-to-date (Q1, H1, Q1-Q3,
   annual). THS levels are checked to be non-decreasing within a fiscal year. From the levels the YoY
   is recomputed both cumulatively and for the single quarter, so a source that reports single-quarter
   growth is recognised instead of being called wrong.
4. **Consistency.** A reported YoY that matches the YoY recomputed from reported levels (revenue and
   net profit of the same period one year earlier) is *level-consistent*. When Sina and THS disagree
   by more than ``YOY_TOLERANCE_PP`` points, the level-consistent value is served; if neither can be
   confirmed, the primary's value is served and the disagreement is flagged as unresolved.

The output metadata contains strings, booleans and ISO dates only (like provenance), so it cannot make
an unsupported number look "traceable" to the agent's numeric-faithfulness check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Two reported growth rates closer than this (percentage points) agree.
YOY_TOLERANCE_PP = 2.0
# A reported YoY within this distance of the level-recomputed YoY is level-consistent.
LEVEL_TOLERANCE_PP = 1.0

# Plausible bounds per field. Net-profit YoY can legitimately fall below -100% (profit to loss).
RANGES: dict[str, tuple[float, float]] = {
    "revenue_yoy": (-100.0, 10_000.0),
    "netprofit_yoy": (-100_000.0, 100_000.0),
    "roe": (-200.0, 200.0),
    "grossprofit_margin": (-100.0, 100.0),
    "eps": (-1_000.0, 1_000.0),
}

YOY_FIELDS = (("revenue_yoy", "revenue"), ("netprofit_yoy", "net_profit"))
MERGE_FIELDS = ("roe", "grossprofit_margin", "eps", "profit_dedt", "revenue", "net_profit")

SINA = "sina.finance"
THS = "ths.finance"

_FIELD_ZH = {"revenue_yoy": "营收同比", "netprofit_yoy": "净利润同比"}
_SOURCE_ZH = {SINA: "新浪财经", THS: "同花顺"}


@dataclass
class CrossCheck:
    status: str  # agree | disagree_resolved | disagree_unresolved | single_source | period_mismatch
    served_source: str
    compared_with: str | None = None
    report_period: str | None = None
    other_period: str | None = None
    disagreeing_fields: list[str] = field(default_factory=list)
    resolution: dict[str, str] = field(default_factory=dict)
    conventions: dict[str, str] = field(default_factory=dict)
    out_of_range: list[str] = field(default_factory=list)
    level_consistent: dict[str, bool] = field(default_factory=dict)
    filled_from_other: list[str] = field(default_factory=list)

    @property
    def disagrees(self) -> bool:
        return self.status in {"disagree_resolved", "disagree_unresolved"}

    def as_metadata(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "status": self.status,
            "served_source": self.served_source,
            "compared_with": self.compared_with,
            "report_period": self.report_period,
        }
        if self.other_period and self.other_period != self.report_period:
            record["other_period"] = self.other_period
        if self.disagreeing_fields:
            record["disagreeing_fields"] = list(self.disagreeing_fields)
            record["resolution"] = dict(self.resolution)
        if self.conventions:
            record["conventions"] = dict(self.conventions)
        if self.out_of_range:
            record["out_of_range"] = list(self.out_of_range)
        if self.level_consistent:
            record["level_consistent"] = dict(self.level_consistent)
        if self.filled_from_other:
            record["filled_from_other"] = list(self.filled_from_other)
        record["note"] = self.note()
        return record

    def note(self) -> str:
        """Chinese note without digits (dates are left out on purpose)."""
        other = _SOURCE_ZH.get(self.compared_with or "", self.compared_with or "")
        served = _SOURCE_ZH.get(self.served_source, self.served_source)
        if self.status == "agree":
            return f"{served}与{other}的同比增速一致（同一报告期交叉校验通过）"
        if self.status == "single_source":
            return f"仅{served}可用，未能与第二数据源交叉校验"
        if self.status == "period_mismatch":
            return f"{served}与{other}的最新报告期不同，采用较新报告期的{served}数据，未做同期比对"
        fields = "、".join(_FIELD_ZH.get(name, name) for name in self.disagreeing_fields)
        if self.status == "disagree_resolved":
            return (
                f"{served}与{other}的{fields}不一致；已采用与报告期营收/净利润绝对值推算结果一致的{served}数据，"
                f"请以公司定期报告为准"
            )
        return f"{served}与{other}的{fields}不一致且无法用绝对值核验，数值存疑，请以公司定期报告为准"


def _in_range(name: str, value: float | None) -> bool:
    if value is None:
        return True
    low, high = RANGES.get(name, (float("-inf"), float("inf")))
    return low <= value <= high


def sanitize(record: dict[str, Any], out_of_range: list[str], source: str) -> dict[str, Any]:
    """Drop out-of-range values (recorded as ``source:field``)."""
    clean = dict(record)
    for name in RANGES:
        if not _in_range(name, clean.get(name)):
            out_of_range.append(f"{source}:{name}")
            clean[name] = None
    return clean


def _shift_year(period: str, years: int = -1) -> str:
    return f"{int(period[:4]) + years:04d}{period[4:]}"


def _previous_quarter_end(period: str) -> str | None:
    """Previous cumulative period in the same fiscal year (``None`` for Q1)."""
    month_day = period[5:]
    order = ["03-31", "06-30", "09-30", "12-31"]
    if month_day not in order:
        return None
    index = order.index(month_day)
    return None if index == 0 else f"{period[:4]}-{order[index - 1]}"


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return (current / previous - 1.0) * 100.0 if previous > 0 else None


def is_cumulative(periods: dict[str, dict[str, Any]], level: str = "revenue") -> bool | None:
    """Whether levels are non-decreasing within each fiscal year (year-to-date convention)."""
    verdict: bool | None = None
    for period in sorted(periods):
        previous = _previous_quarter_end(period)
        if previous is None or previous not in periods:
            continue
        current_value, previous_value = periods[period].get(level), periods[previous].get(level)
        if current_value is None or previous_value is None or previous_value <= 0:
            continue
        if current_value < previous_value:
            return False
        verdict = True
    return verdict


def recomputed_yoy(periods: dict[str, dict[str, Any]], period: str, level: str) -> dict[str, float]:
    """YoY recomputed from levels: ``cumulative`` and, where derivable, ``single_quarter``."""
    out: dict[str, float] = {}
    current = periods.get(period, {}).get(level)
    prior = periods.get(_shift_year(period), {}).get(level)
    cumulative = _growth(current, prior)
    if cumulative is not None:
        out["cumulative"] = cumulative
    previous = _previous_quarter_end(period)
    if previous is not None:
        current_prev = periods.get(previous, {}).get(level)
        prior_prev = periods.get(_shift_year(previous), {}).get(level)
        if None not in (current, current_prev, prior, prior_prev):
            single = _growth(current - current_prev, prior - prior_prev)  # type: ignore[operator]
            if single is not None:
                out["single_quarter"] = single
    return out


def _convention(value: float | None, recomputed: dict[str, float]) -> str | None:
    if value is None:
        return None
    for name in ("cumulative", "single_quarter"):
        if name in recomputed and abs(value - recomputed[name]) <= LEVEL_TOLERANCE_PP:
            return name
    return None


def reconcile_fundamentals(
    sina_periods: dict[str, dict[str, Any]] | None,
    ths_periods: dict[str, dict[str, Any]] | None,
    *,
    primary: str = SINA,
) -> tuple[dict[str, Any] | None, CrossCheck | None]:
    """Return ``(report, check)`` for the latest period, or ``(None, None)`` without data.

    ``*_periods`` map ISO report dates to records with ``revenue_yoy``, ``netprofit_yoy``, ``roe``,
    ``grossprofit_margin``, ``eps``, ``profit_dedt`` and, for THS, the ``revenue``/``net_profit`` levels.
    """
    sources = {SINA: sina_periods or {}, THS: ths_periods or {}}
    available = [name for name in (primary, THS if primary == SINA else SINA) if sources[name]]
    if not available:
        return None, None
    out_of_range: list[str] = []
    latest = {name: max(sources[name]) for name in available}

    if len(available) == 1:
        name = available[0]
        report = sanitize(sources[name][latest[name]], out_of_range, name)
        check = CrossCheck(status="single_source", served_source=name, report_period=latest[name])
        check.out_of_range = out_of_range
        return {**report, "report_date": latest[name]}, check

    first, second = available
    if latest[first] != latest[second]:
        newer = first if latest[first] > latest[second] else second
        older = second if newer == first else first
        report = sanitize(sources[newer][latest[newer]], out_of_range, newer)
        check = CrossCheck(
            status="period_mismatch",
            served_source=newer,
            compared_with=older,
            report_period=latest[newer],
            other_period=latest[older],
            out_of_range=out_of_range,
        )
        return {**report, "report_date": latest[newer]}, check

    period = latest[first]
    records = {name: sanitize(sources[name][period], out_of_range, name) for name in available}
    levels = sources[THS]  # only THS carries absolute levels
    check = CrossCheck(status="agree", served_source=first, compared_with=second, report_period=period)
    check.out_of_range = out_of_range
    cumulative = is_cumulative(levels)
    if cumulative is False:
        check.conventions[f"{THS}:levels"] = "not_cumulative"

    served_votes: dict[str, str] = {}
    for yoy_field, level in YOY_FIELDS:
        recomputed = recomputed_yoy(levels, period, level) if cumulative is not False else {}
        for name in available:
            convention = _convention(records[name].get(yoy_field), recomputed)
            if convention:
                check.conventions[f"{name}:{yoy_field}"] = convention
            if recomputed.get("cumulative") is not None and records[name].get(yoy_field) is not None:
                check.level_consistent[f"{name}:{yoy_field}"] = convention == "cumulative"
        values = {name: records[name].get(yoy_field) for name in available}
        if None in values.values() or abs(values[first] - values[second]) <= YOY_TOLERANCE_PP:  # type: ignore[operator]
            continue
        check.disagreeing_fields.append(yoy_field)
        confirmed = [name for name in available if check.conventions.get(f"{name}:{yoy_field}") == "cumulative"]
        if len(confirmed) == 1:
            served_votes[yoy_field] = confirmed[0]
            check.resolution[yoy_field] = f"{confirmed[0]}:matches_reported_levels"
        else:
            served_votes[yoy_field] = first
            check.resolution[yoy_field] = f"{first}:unverified"

    if check.disagreeing_fields:
        resolved = all(value.endswith(":matches_reported_levels") for value in check.resolution.values())
        check.status = "disagree_resolved" if resolved else "disagree_unresolved"
        winners = set(served_votes.values())
        if len(winners) == 1:
            check.served_source = winners.pop()
    served = check.served_source
    other = second if served == first else first
    report = dict(records[served])
    for name in (*(yoy for yoy, _level in YOY_FIELDS), *MERGE_FIELDS):
        if name in served_votes:
            report[name] = records[served_votes[name]].get(name)
        elif report.get(name) is None and records[other].get(name) is not None and name not in check.disagreeing_fields:
            # Same report period: fill a gap (e.g. Sina's missing gross margin) from the other source.
            report[name] = records[other][name]
            check.filled_from_other.append(name)
    check.compared_with = other
    return {**report, "report_date": period}, check
