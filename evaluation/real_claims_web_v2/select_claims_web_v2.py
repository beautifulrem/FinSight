"""Select the 50 claims of ``collected_v2.jsonl`` from the agents' candidates, by fixed rules, before any label.

The rules use only what a candidate says and where it was published, never whether it is right (no offline value is
read here):

1. drop by the listed reasons (not a claim sentence, already in the first set, a second copy of the same sentence,
   period not stated, numbers spread over several sessions);
2. group the rest by their first-named entity (the 13 companies, and one group for indices and ETFs);
3. order each group: a known publication date first, then a fixed pseudo-random key (sha256 of URL + text);
4. take up to ``CAP`` per group and two per URL; if more than ``TARGET`` remain, remove from the largest groups, last
   in order first; if fewer, take further candidates in global key order with the cap raised by one.

    python -m evaluation.real_claims_web_v2.select_claims_web_v2      # rewrites collected_v2.jsonl
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates_v2.jsonl"
COLLECTED = HERE / "collected_v2.jsonl"
TARGET = 50
CAP = 4
PER_URL = 2

INDEX_GROUP = {"上证指数", "创业板指", "沪深300指数", "科创50ETF华夏", "黄金ETF华安", "中证500ETF"}

DROPPED = {
    "cC13": "not a claim sentence: an annual-report table row (加权平均净资产收益率 6.89% 23.35%)",
    "cC27": "not a claim sentence: a quote-table line (上证指数 3842.19 +0.31% 成交 6794.0 亿)",
    "cC37": "no number (招商银行，平安银行小幅度上涨)",
    "cB03": "already in the first set: the same sentence is w27 (证券之星, 招商银行 2026-09-30)",
    "cC22": "already in the first set: the text is part of w59 (沪深300指数涨0.66%)",
    "cC35": "second copy of cB20 (same sentence, same URL)",
    "cC32": "second copy of cB09 (the same sentence on another outlet)",
    "cC33": "second copy of cB04 (the same sentence on another outlet)",
    "cC10": "period not stated (五粮液 ROE range without a year)",
    "cC24": "numbers about a range over several sessions, not one date (BBN Times, SSE Composite)",
}

# Applied after selection (the selection key uses the text as collected). Trims drop whole clauses of a long English
# sentence (flagged ``excerpt``); dates that the sentence leaves to its context are recorded, as in the first set.
EDITS: dict[str, dict] = {
    "cA19": {
        "published_text": "Net profit attributable to shareholders [of Midea] grew 14.0% to RMB 43.9 billion, "
        "while basic earnings per share advanced 6.6% to RMB 5.80",
        "excerpt": True,
    },
    "cA25": {
        "published_text": "[Zijin Mining] achieving revenue of US$49.7 billion, up 15% year on year, and net "
        "profit attributable to shareholders of the parent company of US$7.4 billion, up 62%.",
        "excerpt": True,
        "subject_inserted": True,
    },
    "cA02": {"date_implicit": "2026-09-30", "date_source": "preceding sentence (截至9月30日收盘)"},
    "cA13": {"date_implicit": "2026-09-30", "date_source": "preceding sentence (截至2026年9月30日收盘)"},
    "cA24": {"date_source": "publication day (今日)"},
    "cB04": {"date_implicit": "2026-09-30", "date_source": "preceding sentence (截至2026年9月30日收盘)"},
    "cB09": {"date_implicit": "2026-09-30", "date_source": "same sentence, dropped lead (截至2026年9月30日收盘)"},
    "cB32": {"date_implicit": "2026-04-22", "date_source": "same sentence, dropped lead (2026年4月22日…今日)"},
    "cC11": {"date_source": "publication day (今日)"},
    "cC12": {"date_source": "publication day (今日)"},
    "cC21": {"date_implicit": "2026-04-22", "date_source": "preceding sentence (4月22日)"},
    "cC26": {"date_source": "publication day (截至收盘)"},
    "cC34": {"date_source": "publication day (今日)"},
}


def _key(row: dict) -> str:
    return hashlib.sha256((row["url"] + "\n" + row["published_text"]).encode("utf-8")).hexdigest()


def _group(row: dict) -> str:
    first = row["companies"][0]
    return "指数/ETF" if first in INDEX_GROUP else first


def select(rows: list[dict]) -> list[dict]:
    kept = [r for r in rows if r["candidate_id"] not in DROPPED]
    order = sorted(kept, key=lambda r: (r.get("published") is None, _key(r)))
    picked: list[dict] = []
    per_group: Counter[str] = Counter()
    per_url: Counter[str] = Counter()
    for r in order:
        if per_group[_group(r)] < CAP and per_url[r["url"]] < PER_URL:
            picked.append(r)
            per_group[_group(r)] += 1
            per_url[r["url"]] += 1
    while len(picked) > TARGET:
        biggest = max(per_group.values())
        victim = [r for r in picked if per_group[_group(r)] == biggest][-1]
        picked.remove(victim)
        per_group[_group(victim)] -= 1
    for r in order:
        if len(picked) >= TARGET:
            break
        if r in picked:
            continue
        if per_group[_group(r)] < CAP + 1 and per_url[r["url"]] < PER_URL:
            picked.append(r)
            per_group[_group(r)] += 1
            per_url[r["url"]] += 1
    picked.sort(key=lambda r: r["candidate_id"])
    return picked


def main() -> None:
    rows = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = select(rows)
    out = []
    for i, r in enumerate(picked, 1):
        rec = {"id": f"v{i:02d}"}
        rec.update(r)
        rec.update(EDITS.get(r["candidate_id"], {}))
        out.append(rec)
    COLLECTED.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"{len(rows)} candidates, {len([r for r in rows if r['candidate_id'] in DROPPED])} dropped by rule, "
          f"{len(out)} selected")
    print("by group:", dict(Counter(_group(r) for r in out)))
    print("by source type:", dict(Counter(r["source_type"] for r in out)))
    print("by lang:", dict(Counter(r["lang"] for r in out)))
    print("by period:", dict(Counter(r["period"] for r in out)))


if __name__ == "__main__":
    main()
