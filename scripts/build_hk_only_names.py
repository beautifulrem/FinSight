"""Build ``data/runtime/hk_only_names.csv``: Hong Kong-listed companies with no A-share line (round 17, review-13 M3).

Before round 17 a Hong Kong-only company whose name starts like a covered A-share ("美的置业" ~ 美的集团, "中信国际电讯"
~ 中工国际 by one character) was answered with the A-share company's figures. The entity master lists A-shares only,
so these names are taken from the HKEX list of securities (Chinese edition, ``ListOfSecurities_c.xlsx``) and shipped as
explicit out-of-coverage names: the NLU masks them before it resolves entities (``nlu/entity_resolver.py``) and the
agent refuses them as Hong Kong equities.

A row is kept when the security is an equity (股本), its name (converted to simplified characters, a trailing -B / -W /
-S / -SW marker removed) has at least four characters, and it is not an A-share name: it is not an alias of the entity
master, not part of one, does not start with the official name of an A-share stock (比亚迪股份 is 比亚迪's own H line), and
its code is not in ``data/runtime/dual_listings.csv``.

usage: python scripts/build_hk_only_names.py ListOfSecurities_c.xlsx
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "runtime" / "hk_only_names.csv"
_MARKER = re.compile(r"\s*[-－]\s*[A-ZＡ-Ｚ]{1,2}\s*$")


def main() -> None:
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    from query_intelligence.traditional import to_simplified

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    raw = pd.read_excel(args.source, header=None)
    as_at = str(raw.iloc[1, 0]).replace(" ", "").replace("截至", "")
    table = pd.read_excel(args.source, header=2)
    equities = table[table["分類"].astype(str) == "股本"]

    entities = {row["entity_id"]: row for row in csv.DictReader(open(ROOT / "data/runtime/entity_master.csv"))}
    aliases = list(csv.DictReader(open(ROOT / "data/runtime/alias_table.csv")))
    alias_names = {row["normalized_alias"] for row in aliases if row.get("normalized_alias")}
    long_aliases = [name for name in alias_names if len(name) >= 4 and re.search(r"[一-鿿]", name)]
    stock_names = {
        row["canonical_name"] for row in entities.values() if row.get("entity_type") == "stock" and row["canonical_name"]
    }
    dual_codes = {row["h_code"].lstrip("0") for row in csv.DictReader(open(ROOT / "data/runtime/dual_listings.csv"))}

    rows = []
    for code, listed in zip(equities["股份代號"], equities["股份名稱"], strict=True):
        name = _MARKER.sub("", to_simplified(str(listed))).strip()
        if len(name) < 4 or not re.fullmatch(r"[一-鿿]+", name):
            continue
        if str(int(code)) in dual_codes or name in alias_names:
            continue
        if any(name.startswith(stock) for stock in stock_names if len(stock) >= 2):
            continue
        if any(name in other for other in long_aliases if other != name):
            continue
        rows.append((f"{int(code):05d}", name, str(listed)))
    rows.sort()
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            f"# Hong Kong-listed equities with no A-share line, from HKEX ListOfSecurities_c.xlsx as at {as_at} "
            f"(sha256 {digest}); built by scripts/build_hk_only_names.py\n"
        )
        writer = csv.writer(handle)
        writer.writerow(["h_code", "name_zh", "hkex_name"])
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
