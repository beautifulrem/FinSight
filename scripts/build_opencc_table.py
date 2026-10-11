"""Build ``data/runtime/opencc_ts_characters.tsv`` from OpenCC's ``TSCharacters.txt`` (round 17, review-13 M9).

OpenCC (https://github.com/BYVoid/OpenCC, Apache-2.0) maps traditional to simplified characters. FinSight does not
depend on the OpenCC library; it ships the one-to-one part of the character table as data:

* a row is kept only when the traditional character is never itself a simplified one (its candidate list does not
  contain it: 乾 -> 干 / 乾, 於 -> 于 / 於, 徵 -> 征 / 徵 are dropped, so 乾照光电 is never rewritten), and
* the first candidate is a character of the Basic Multilingual Plane (the rare inferred forms OpenCC marks
  ``@tofu-risk`` keep the original character as their first candidate and are dropped by the first rule).

usage: python scripts/build_opencc_table.py TSCharacters.txt [--tag ver.1.4.2]
The input is ``data/dictionary/TSCharacters.txt`` of the OpenCC release named by ``--tag``; its sha256 is recorded in
the output header. The licence of the source is in ``data/runtime/LICENSE.OpenCC``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "runtime" / "opencc_ts_characters.tsv"


def build(source: Path, tag: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        traditional, candidates = line.split("\t", 1)
        values = candidates.split()
        if len(traditional) != 1 or not values or traditional in values:
            continue
        simplified = values[0]
        if len(simplified) != 1 or ord(simplified) > 0xFFFF or ord(traditional) > 0xFFFF:
            continue
        rows.append((traditional, simplified))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--tag", default="ver.1.4.2")
    args = parser.parse_args()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    rows = build(args.source, args.tag)
    header = [
        "# Traditional -> simplified Chinese characters, one-to-one rows only (see scripts/build_opencc_table.py).",
        f"# Source: OpenCC {args.tag}, data/dictionary/TSCharacters.txt (sha256 {digest}),",
        "# https://github.com/BYVoid/OpenCC. Copyright (c) BYVoid and OpenCC contributors.",
        "# Licensed under the Apache License, Version 2.0: data/runtime/LICENSE.OpenCC.",
        "# Modification: rows whose traditional character is also a simplified one, and multi-character or",
        "# non-BMP rows, are dropped; only the first candidate is kept.",
        "traditional\tsimplified",
    ]
    OUT.write_text("\n".join([*header, *(f"{t}\t{s}" for t, s in rows)]) + "\n", encoding="utf-8")
    print(f"{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
