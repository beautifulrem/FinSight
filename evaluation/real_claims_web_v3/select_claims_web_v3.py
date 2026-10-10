"""Select the third blind set's 50 claims from the search agents' candidates (deterministic, blind to correctness).

The rules never read an offline value or a checker output. They use only each candidate's text, URL, period,
companies, publication date and the fixed per-candidate decisions below (made from the sentence's form, before any
label existed)::

    python -m evaluation.real_claims_web_v3.select_claims_web_v3   # candidates_v3.jsonl -> collected_v3.jsonl
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates_v3.jsonl"
OUT = HERE / "collected_v3.jsonl"
TARGET = 50
PER_GROUP = 4
PER_URL = 2

# Fixed drops, decided from the sentence's form and provenance only.
DROP: dict[str, str] = {
    "cD01": "not a claim sentence (a market-review table row)",
    "cD05": "second copy of cB02's sentence (same 数据宝 sentence on another outlet)",
    "cD02": "intraday value ('盘中市值'), not a daily-bar or closing figure",
    "cD08": "about a post-holiday session; the 2026-09-30 figure is only implied by a relative change",
    "cB07": "live valuation page ('当前市盈率'), the date the figure refers to is not fixed",
    "cB06": "雪球 stock news feed, not a stable page",
    "cE14": "period not stated (the agent assumed the latest full year)",
    "cE15": "period not stated (the agent assumed the latest full year)",
}

# Same article / page for the per-URL cap (one article at two URLs).
BBN = "bbntimes:sse-composite-shut-for-golden-week"
URL_ALIASES: dict[str, str] = {
    "https://www.bbntimes.com/financial/sse-composite-shut-for-golden-week-": BBN,
    "https://www.bbntimes.com/global-economy/sse-composite-shut-for-golden-week": BBN,
}

# Form class "rel": a relation between companies, a sum, an average, a share or a derived ratio, read from the
# sentence's wording. Market-data claims (period other than FY2025) are "market"; the rest "annual".
REL = set(
    [
        "cA21",
        "cA24",
        "cA26",
        "cA27",
        "cA28",
        "cB16",
        "cB17",
        "cB23",
        "cB24",
        "cC03",
        "cC04",
        "cC12",
        "cC22",
        "cD18",
        "cD19",
        "cD20",
        "cD21",
        "cD22",
        "cE00",
        "cE01",
        "cE02",
        "cE03",
        "cE04",
        "cE05",
        "cE06",
        "cE07",
        "cE08",
        "cE09",
        "cE10",
        "cE11",
        "cE12",
        "cE13",
    ]
)

INDICES = {"上证指数", "创业板指", "沪深300指数"}
ETFS = {"创业板ETF", "科创50ETF", "中证500ETF", "黄金ETF", "沪深300ETF", "证券ETF"}

# date_implicit holds the date a claim refers to. Where an agent wrote a publication-date estimate there for an
# FY2025 claim, it is moved to published_estimate.
PUBLISHED_ESTIMATE = {"cE03", "cE04", "cE11"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def url_key(url: str) -> str:
    for prefix, alias in URL_ALIASES.items():
        if url.startswith(prefix):
            return alias
    return url


def group_of(row: dict[str, Any]) -> str:
    first = row["companies"][0]
    if first in INDICES:
        return f"indices {row['period']}"
    if first in ETFS:
        return f"etfs {row['period']}"
    return first


def form_of(row: dict[str, Any]) -> str:
    if row["candidate_id"] in REL:
        return "rel"
    return "annual" if row["period"] == "FY2025" else "market"


def order_key(row: dict[str, Any]) -> tuple[int, str]:
    return (0 if row.get("published") else 1, sha(row["url"] + row["published_text"]))


def interleave(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_form: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=order_key):
        by_form[form_of(row)].append(row)
    out: list[dict[str, Any]] = []
    while any(by_form.values()):
        for form in ("rel", "annual", "market"):
            if by_form[form]:
                out.append(by_form[form].pop(0))
    return out


def select(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = [row for row in candidates if row["candidate_id"] not in DROP]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        groups[group_of(row)].append(row)
    per_url: dict[str, int] = defaultdict(int)
    chosen: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(groups, key=sha):
        picks: list[dict[str, Any]] = []
        for row in interleave(groups[name]):
            if len(picks) == PER_GROUP:
                break
            key = url_key(row["url"])
            if per_url[key] >= PER_URL:
                continue
            per_url[key] += 1
            picks.append(row)
        chosen[name] = picks
    while sum(len(v) for v in chosen.values()) > TARGET:
        largest = max(chosen, key=lambda g: (len(chosen[g]), sha(g)))
        chosen[largest].pop()
    rows = [row for name in sorted(chosen, key=sha) for row in chosen[name]]
    return sorted(rows, key=lambda r: (group_of(r), r["candidate_id"]))


def main() -> int:
    candidates = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = select(candidates)
    with OUT.open("w", encoding="utf-8") as fh:
        for index, row in enumerate(rows, 1):
            out = {"id": f"x{index:02d}", **row}
            if row["candidate_id"] in PUBLISHED_ESTIMATE:
                out["published_estimate"] = out.pop("date_implicit")
                out["date_source"] = None
            for key in ("date_implicit", "date_source"):
                if out.get(key) is None:
                    out.pop(key, None)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"{len(candidates)} candidates, {len(DROP)} dropped by rule, {len(rows)} selected -> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
