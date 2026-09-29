"""Alias regression: colloquial short names, typo'd names and fuzzy false hits (round-5 C6/C8).

Each line of ``tests/data/alias_regression.jsonl`` names the symbols a question must resolve to and the symbols it
must not resolve to. The queries are new wording, not copied from any evaluation set.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from query_intelligence.runtime_entity_assets import COLLOQUIAL_ALIAS_TYPE, COLLOQUIAL_ALIASES, colloquial_alias_rows

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    json.loads(line)
    for line in (ROOT / "tests" / "data" / "alias_regression.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


@pytest.mark.parametrize("case", CASES, ids=[case["query"] for case in CASES])
def test_alias_regression(offline_service, case):
    nlu = offline_service.analyze_query(case["query"])
    symbols = {entity.get("symbol") for entity in nlu["entities"] if entity.get("symbol")}
    assert set(case["expect_symbols"]) <= symbols, (case["kind"], symbols)
    assert not symbols & set(case["forbid_symbols"]), (case["kind"], symbols)


def test_colloquial_aliases_are_in_the_runtime_table_once_each():
    path = ROOT / "data" / "runtime" / "alias_table.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    colloquial = [row for row in rows if row["alias_type"] == COLLOQUIAL_ALIAS_TYPE]
    expected = {text for _name, texts in COLLOQUIAL_ALIASES.values() for text in texts}
    assert {row["normalized_alias"] for row in colloquial} == expected
    assert len(colloquial) == len(expected)
    ids = [int(row["alias_id"]) for row in rows]
    assert len(set(ids)) == len(ids)


def test_colloquial_alias_rows_are_idempotent_and_skip_reused_symbols():
    entities = [
        {"entity_id": "1", "symbol": "600519.SH", "canonical_name": "贵州茅台"},
        {"entity_id": "2", "symbol": "000333.SZ", "canonical_name": "某新公司"},  # the symbol was reused
    ]
    rows = colloquial_alias_rows(entities, [], 100)
    assert [(row["entity_id"], row["alias_text"], row["alias_id"]) for row in rows] == [("1", "茅子", "100")]
    aliases = [{"entity_id": "1", "normalized_alias": "茅子"}]
    assert colloquial_alias_rows(entities, aliases, 101) == []
