"""Fixed, value-blind selection: candidates_*.jsonl -> collected_v6.jsonl (60 claims).

Reads no offline data value. The only repository files it reads are the `url` and text fields
(published_text / claim_text / excerpt) of the v1-v5 collected files, to drop repeats.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/Volumes/Remi/FinSight-r2")
TARGET = 60
GROUP_CAP = 5
URL_CAP = 2
# Variety quotas, filled first in hash order (value-blind: they look only at collector tags and source fields).
QUOTAS = [
    ("feature", "definition", 3),
    ("feature", "weak_direction", 6),
    ("feature", "same_company_comparison", 4),
    ("feature", "kou_fei", 3),
    ("feature", "growth_multiple", 3),
    ("feature", "cross_company_comparison", 3),
    ("feature", "short_name", 4),
    ("script", "traditional", 6),
    ("lang", "en", 9),
    ("source_type", "social", 12),
    ("source_type", "broker", 3),
    ("source_type", "company", 3),
    ("period", "2026-09-30", 6),
    ("period", "2026-04-22", 5),
]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"\[[^\]]*\]", "", text)
    return re.sub(r"[\s\W_]+", "", text)


def prior_sets() -> tuple[set[str], list[str]]:
    urls: set[str] = set()
    texts: list[str] = []
    for path in sorted(glob.glob(str(REPO / "evaluation/real_claims_web*/collected_*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("url"):
                urls.add(row["url"].rstrip("/"))
            for key in ("published_text", "claim_text", "excerpt"):
                if isinstance(row.get(key), str) and row[key]:
                    texts.append(norm(row[key]))
    return urls, texts


def is_repeat(text: str, prior: list[str]) -> bool:
    n = norm(text)
    for p in prior:
        if n == p or (len(n) >= 15 and (n in p)) or (len(p) >= 15 and p in n):
            return True
    return False


# Tavily's quota was exhausted for every collector, so a sentence counts as verified when the collector found it in
# the text of the page it fetched (tavily_extract, curl + text stripping, or pypdf for a PDF); search snippets do not.
def verified(c: dict) -> bool:
    v = str(c.get("verified_by", ""))
    return v.startswith("tavily_extract") or "sentence found in" in v


# First-named entity, normalised to one name per company / index (scripts and languages differ between collectors).
GROUP_ALIASES = {
    "貴州茅台": "贵州茅台", "Kweichow Moutai": "贵州茅台", "紫金礦業": "紫金矿业", "Zijin Mining": "紫金矿业",
    "五糧液": "五粮液", "Wuliangye": "五粮液", "创业板ETF(159915)": "创业板ETF", "沪深两市": "上证指数",
}


def group(c: dict) -> str:
    g = (c.get("entities") or ["?"])[0]
    return GROUP_ALIASES.get(g, g)


def has(c: dict, kind: str, value: str) -> bool:
    if kind == "feature":
        return value in (c.get("features") or [])
    return c.get(kind) == value


def main() -> None:
    cands = []
    for path in sorted(HERE.glob("candidates_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                cands.append(json.loads(line))
    prior_urls, prior_texts = prior_sets()
    dropped: dict[str, list[str]] = {k: [] for k in ("not_claim", "not_verified", "period_unstated", "multi_session",
                                                      "prior_url", "prior_text", "duplicate", "out_of_scope_period")}
    seen: set[str] = set()
    pool = []
    for c in cands:
        cid = c["candidate_id"]
        if not c.get("is_claim", True):
            dropped["not_claim"].append(cid)
        elif not verified(c):
            dropped["not_verified"].append(cid)
        elif c.get("period") not in ("FY2025", "2026-09-30", "2026-04-22"):
            dropped["out_of_scope_period"].append(cid)
        elif c.get("period_basis") not in ("sentence", "context"):
            dropped["period_unstated"].append(cid)
        elif c.get("multi_session"):
            dropped["multi_session"].append(cid)
        elif (c.get("url") or "").rstrip("/") in prior_urls:
            dropped["prior_url"].append(cid)
        elif is_repeat(c["published_text"], prior_texts):
            dropped["prior_text"].append(cid)
        elif norm(c["published_text"]) in seen:
            dropped["duplicate"].append(cid)
        else:
            seen.add(norm(c["published_text"]))
            pool.append(c)
    pool.sort(key=lambda c: hashlib.sha256(("v6:" + c["candidate_id"]).encode()).hexdigest())

    chosen: list[dict] = []
    groups: dict[str, int] = {}
    urls: dict[str, int] = {}
    skipped_cap: set[str] = set()

    def take(c: dict) -> bool:
        g = group(c)
        u = c.get("url", "")
        if c in chosen:
            return False
        if groups.get(g, 0) >= GROUP_CAP or urls.get(u, 0) >= URL_CAP:
            skipped_cap.add(c["candidate_id"])
            return False
        chosen.append(c)
        groups[g] = groups.get(g, 0) + 1
        urls[u] = urls.get(u, 0) + 1
        return True

    for kind, value, need in QUOTAS:
        have = sum(1 for c in chosen if has(c, kind, value))
        for c in pool:
            if have >= need or len(chosen) >= TARGET:
                break
            if c not in chosen and has(c, kind, value) and take(c):
                have += 1
    for c in pool:
        if len(chosen) >= TARGET:
            break
        if c not in chosen:
            take(c)

    order = {c["candidate_id"]: i for i, c in enumerate(pool)}
    chosen.sort(key=lambda c: order[c["candidate_id"]])
    with open(HERE / "collected_v6.jsonl", "w", encoding="utf-8") as fh:
        for i, c in enumerate(chosen, 1):
            row = {"id": f"w6_{i:02d}", **c}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "candidates": len(cands),
        "pool": len(pool),
        "selected": len(chosen),
        "dropped": {k: v for k, v in dropped.items() if v},
        "skipped_by_cap_at_some_point": sorted(skipped_cap - {c["candidate_id"] for c in chosen}),
        "not_selected": sorted({c["candidate_id"] for c in pool} - {c["candidate_id"] for c in chosen}),
    }
    (HERE / "selection_v6.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
