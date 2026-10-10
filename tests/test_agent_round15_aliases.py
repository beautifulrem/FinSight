"""Round 15 (the third real-claims set's entity class): "人民币" was an official name of 宁德时代 in the runtime alias
table, scraped from text where a name precedes a bracketed code, so a claim quoting an amount in 人民币 was checked
against 宁德时代. The builder now rejects currency words and phrases around a known name, and the shipped table no
longer has those rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from query_intelligence.runtime_entity_assets import (
    RuntimeEntityAssetBuilder,
    _is_valid_security_name,
    _plausible_extra_name,
)

ROOT = Path(__file__).resolve().parents[1]
CURRENCIES = {"人民币", "美元", "港元", "港币", "欧元", "日元", "英镑"}


def test_currency_words_are_not_security_names():
    assert not any(_is_valid_security_name(word) for word in CURRENCIES)


def test_a_scraped_name_next_to_a_known_code_must_be_a_name_of_that_company():
    assert _plausible_extra_name("宁德时代", "宁德时代")
    assert _plausible_extra_name("宁德", "宁德时代")
    assert not _plausible_extra_name("人民币", "宁德时代")  # nothing in common
    assert not _plausible_extra_name("蔚来与宁德时代", "宁德时代")  # a phrase around the name
    assert not _plausible_extra_name("和招商南油", "招商南油")


def test_the_builder_drops_scraped_currency_words_and_phrases(tmp_path: Path):
    seed_entities = [
        {"entity_id": "1", "canonical_name": "宁德时代", "normalized_name": "宁德时代", "symbol": "300750.SZ",
         "market": "CN", "exchange": "SZSE", "entity_type": "stock", "industry_name": "", "status": "active"},
    ]  # fmt: skip
    seed_aliases = [
        {"alias_id": "1", "entity_id": "1", "alias_text": "宁德时代", "normalized_alias": "宁德时代",
         "alias_type": "official_name", "priority": "1", "is_official": "true"},
    ]  # fmt: skip
    assets = tmp_path / "training_assets"
    assets.mkdir()
    lines = [
        {"title": "x", "body": "营收以人民币（300750）计价"},
        {"title": "y", "body": "蔚来与宁德时代（300750）签约"},
    ]
    (assets / "retrieval_corpus.jsonl").write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines), encoding="utf-8"
    )
    (assets / "alias_catalog.jsonl").write_text("", encoding="utf-8")
    _entities, aliases = RuntimeEntityAssetBuilder(
        seed_entities=seed_entities, seed_aliases=seed_aliases, training_assets_dir=assets
    ).build()
    texts = {row["alias_text"] for row in aliases if row["entity_id"] == "1"}
    assert "蔚来与宁德时代" not in texts, texts
    # every name alias shares a character with the company's name ("营收以人民币" scraped before the code does not)
    assert all(set(text) & set("宁德时代") for text in texts if not text[:6].isdigit()), texts


def test_the_shipped_alias_table_has_no_currency_alias_of_a_company():
    with (ROOT / "data" / "runtime" / "alias_table.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    bad = [row for row in rows if row["alias_text"] in CURRENCIES or row["alias_text"] == "蔚来与宁德时代"]
    assert not bad, bad
