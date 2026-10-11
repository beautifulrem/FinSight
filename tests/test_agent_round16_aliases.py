"""Round 16 (review-12 L9): short names of the covered securities are derived from the entity table by rules with
collision checks (``runtime_entity_assets.derive_short_aliases``, ``scripts/derive_short_aliases.py``), not listed by
hand. These tests check the rules on a small table, that no derived alias in the shipped table collides with another
entity, and that the NLU resolves the derived forms in a question (own wording)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from query_intelligence.runtime_entity_assets import (
    SHORT_ALIAS_MAX_RIVALS,
    SHORT_ALIAS_MAX_WORD_FREQ,
    derive_short_aliases,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "data" / "runtime"


def _stock(entity_id: str, name: str, symbol: str, kind: str = "stock") -> dict[str, str]:
    return {"entity_id": entity_id, "canonical_name": name, "symbol": symbol, "entity_type": kind}


def _table() -> list[dict[str, str]]:
    # six "矿业" and six "电力" names make both common company-type suffixes
    rows = [_stock(str(10 + i), f"{a}甲矿业", f"60{i:04d}.SH") for i, a in enumerate("甲乙丙丁戊己")]
    rows += [_stock(str(30 + i), f"{a}乙电力", f"00{i:04d}.SZ") for i, a in enumerate("庚辛壬癸子丑")]
    rows += [
        _stock("1", "紫金矿业", "601899.SH"),
        _stock("2", "紫金银行", "601860.SH"),
        _stock("3", "长江电力", "600900.SH"),
        _stock("4", "长电科技", "600584.SH"),
        _stock("5", "宁德矿业", "300750.SZ"),
    ]
    return rows


def test_a_prefix_shared_with_another_listed_name_needs_the_corpus_to_dominate():
    entities = _table()
    scope = {"601899.SH", "600900.SH", "300750.SZ"}
    # no corpus counts: 紫金 (also 紫金银行) and 长电 (also 长电科技) are ambiguous, 宁德 is unique
    accepted, rejected = derive_short_aliases(entities, [], scope=scope, document_counts={})
    taken = {item["alias"]: item["canonical_name"] for item in accepted}
    assert taken.get("宁德") == "宁德矿业"
    assert "紫金" not in taken and "长电" not in taken
    assert any(item["alias"] == "紫金" and item["evidence"].startswith("ambiguous") for item in rejected)
    # the corpus names 紫金矿业 nine times as often as 紫金银行: the short name is 紫金矿业's
    counts = {"紫金矿业": 9, "紫金银行": 1, "长江电力": 14, "长电科技": 1, "short:长电": 3}
    accepted, _rejected = derive_short_aliases(entities, [], scope=scope, document_counts=counts)
    taken = {item["alias"]: item["canonical_name"] for item in accepted}
    assert taken["紫金"] == "紫金矿业" and taken["长电"] == "长江电力"


def test_common_words_owned_aliases_and_unwritten_abbreviations_are_dropped():
    entities = _table()
    scope = {"601899.SH", "600900.SH"}
    owned = [{"entity_id": "99", "alias_text": "紫金", "normalized_alias": "紫金", "alias_type": "common_alias"}]
    accepted, rejected = derive_short_aliases(
        entities, owned, scope=scope, document_counts={"长江电力": 14}, word_frequency={"长江": 18930}
    )
    taken = {item["alias"] for item in accepted}
    reasons = {item["alias"]: item["evidence"] for item in rejected}
    assert "紫金" not in taken and reasons["紫金"] == "alias of another entity"
    assert "长江" not in taken and reasons["长江"].startswith("common word")
    # R2 initials nobody writes ("紫矿", "长电" without a corpus attestation) are not aliases
    assert reasons["紫矿"] == "abbreviation never written in the corpus"
    assert "长电" not in taken


def test_a_prefix_of_many_names_is_never_an_alias():
    entities = _table() + [_stock(str(200 + i), f"紫金{i}号基金", f"9{i:05d}.OF", "fund") for i in range(5)]
    counts = {"紫金矿业": 90}
    accepted, rejected = derive_short_aliases(entities, [], scope={"601899.SH"}, document_counts=counts)
    assert "紫金" not in {item["alias"] for item in accepted}
    assert any(item["alias"] == "紫金" and "other names" in item["evidence"] for item in rejected)
    assert SHORT_ALIAS_MAX_RIVALS < 6


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_no_derived_alias_in_the_shipped_table_collides_with_another_entity():
    entities = _read(RUNTIME / "entity_master.csv")
    for row in entities:
        row["entity_id"] = str(int(row["entity_id"])) if row["entity_id"].isdigit() else row["entity_id"]
    aliases = _read(RUNTIME / "alias_table.csv")
    report = json.loads((RUNTIME / "derived_short_aliases.json").read_text(encoding="utf-8"))
    derived = [row for row in aliases if row.get("is_official") == "derived"]
    accepted = {(item["entity_id"], item["alias"]): item for item in report["accepted"]}
    assert {(row["entity_id"], row["alias_text"]) for row in derived} == set(accepted)
    assert {item["alias"] for item in report["accepted"]} >= {"紫金", "宁德", "平银", "长电"}
    names = {row["entity_id"]: row["canonical_name"] for row in entities}
    listed = [row["canonical_name"] for row in entities if row["entity_type"] in {"stock", "etf", "index", "fund"}]
    for row in derived:
        alias, owner = row["alias_text"], row["entity_id"]
        assert alias not in names.values(), alias
        others = {r["entity_id"] for r in aliases if r["normalized_alias"] == alias} - {owner}
        assert not others, (alias, others)
        rivals = [name for name in listed if name.startswith(alias) and name != names[owner]]
        evidence = accepted[(owner, alias)]["evidence"]
        # another listed name starting with the alias is allowed only with the corpus evidence recorded
        assert not rivals or (len(rivals) <= SHORT_ALIAS_MAX_RIVALS and evidence.startswith("documents")), (
            alias,
            rivals,
        )
    assert all(int(item["evidence"].split()[2]) >= 8 for item in report["accepted"] if item["evidence"] != "unique")
    assert report["parameters"]["max_word_frequency"] == SHORT_ALIAS_MAX_WORD_FREQ


@pytest.mark.parametrize(
    ("question", "symbol"),
    [
        ("紫金今年的ROE是多少", "601899.SH"),
        ("宁德的市盈率现在多少", "300750.SZ"),
        ("长电去年赚了多少钱", "600900.SH"),
        ("平银的市净率", "000001.SZ"),
    ],
)
def test_the_nlu_resolves_a_derived_short_name(offline_service, question, symbol):
    nlu = offline_service.analyze_query(question)
    nlu = nlu if isinstance(nlu, dict) else nlu.model_dump()
    assert symbol in {entity.get("symbol") for entity in nlu.get("entities") or []}, question
