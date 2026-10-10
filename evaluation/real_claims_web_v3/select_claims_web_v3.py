"""Select the v3 web-claim set from candidates_v3.jsonl.

Fixed, deterministic rules. The script never reads an offline data value
(no structured_data*.json, no snapshot, no checker); it reads only the
candidate records and, from the v1/v2 collected files, the ``url`` field.

Rules, applied in this order (the first rule that fires is the drop reason):

1. not_a_claim        is_claim is false (no stated in-scope number or comparison).
2. not_verified       verified_by does not start with "tavily_extract" (the
                      sentence was not found in extracted page text).
3. unstated_period    period_basis is "dateline" or "none" (the period is not
                      stated in the sentence or its immediate context).
4. multi_session      the claim covers a multi-session range.
5. prior_url          the URL appears in collected_v1.jsonl or collected_v2.jsonl.
6. duplicate          same normalised published_text as an earlier candidate
                      (normalisation: drop whitespace and punctuation, lower-case).

Survivors are ordered by sha256(candidate_id) (hex, ascending) and taken in
that order subject to caps: at most 4 claims per group (group = first-named
entity, companies[0]) and at most 2 claims per URL. The set is trimmed to 50,
or to what is available if fewer than 50 qualify. Selected claims get ids
w3_01, w3_02, ... in selection order.

Usage (from the repo root):
    python evaluation/real_claims_web_v3/select_claims_web_v3.py
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CANDIDATES = HERE / "candidates_v3.jsonl"
OUT = HERE / "collected_v3.jsonl"
PRIOR = [
    ROOT / "evaluation" / "real_claims_web" / "collected_v1.jsonl",
    ROOT / "evaluation" / "real_claims_web_v2" / "collected_v2.jsonl",
]
TARGET = 50
PER_GROUP = 4
PER_URL = 2

KEEP_FIELDS = [
    "candidate_id", "published_text", "claim_text", "subject_inserted", "companies",
    "period", "lang", "source_name", "source_type", "social_platform", "url",
    "published", "retrieved", "verified_by", "note",
]


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _prior_urls() -> set[str]:
    urls: set[str] = set()
    for path in PRIOR:
        for row in _load(path):
            if row.get("url"):
                urls.add(row["url"].strip())
    return urls


def _norm(text: str) -> str:
    return "".join(
        ch.lower() for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def _order_key(candidate_id: str) -> str:
    return hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()


def drop_reason(row: dict, prior_urls: set[str], seen: set[str]) -> str | None:
    if not row.get("is_claim", True):
        return "not_a_claim"
    if not str(row.get("verified_by", "")).startswith("tavily_extract"):
        return "not_verified"
    if row.get("period_basis") in ("dateline", "none"):
        return "unstated_period"
    if row.get("multi_session"):
        return "multi_session"
    if row["url"].strip() in prior_urls:
        return "prior_url"
    key = _norm(row["published_text"])
    if key in seen:
        return "duplicate"
    seen.add(key)
    return None


def main() -> None:
    candidates = _load(CANDIDATES)
    prior_urls = _prior_urls()
    seen: set[str] = set()
    reasons: Counter = Counter()
    qualified = []
    for row in candidates:  # candidate order decides which duplicate survives
        reason = drop_reason(row, prior_urls, seen)
        if reason:
            reasons[reason] += 1
        else:
            qualified.append(row)

    qualified.sort(key=lambda r: _order_key(r["candidate_id"]))
    per_group: Counter = Counter()
    per_url: Counter = Counter()
    selected = []
    for row in qualified:
        group = row["companies"][0]
        if per_group[group] >= PER_GROUP:
            reasons["group_cap"] += 1
            continue
        if per_url[row["url"]] >= PER_URL:
            reasons["url_cap"] += 1
            continue
        if len(selected) >= TARGET:
            reasons["over_target"] += 1
            continue
        per_group[group] += 1
        per_url[row["url"]] += 1
        selected.append(row)

    with OUT.open("w", encoding="utf-8") as f:
        for i, row in enumerate(selected, 1):
            rec = {"id": f"w3_{i:02d}"}
            rec.update({k: row.get(k) for k in KEEP_FIELDS})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"candidates: {len(candidates)}")
    print(f"qualified after drop rules: {len(qualified)}")
    print(f"selected: {len(selected)} -> {OUT.relative_to(ROOT)}")
    for reason, n in sorted(reasons.items()):
        print(f"  dropped {reason}: {n}")
    print("by group:", dict(sorted(Counter(r['companies'][0] for r in selected).items())))
    print("by lang:", dict(Counter(r['lang'] for r in selected)))
    print("by source_type:", dict(Counter(r['source_type'] for r in selected)))
    print("by period:", dict(Counter(r['period'] for r in selected)))
    print("subject_inserted:", sum(1 for r in selected if r['claim_text'] != r['published_text']))


if __name__ == "__main__":
    main()
