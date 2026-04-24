from __future__ import annotations

import csv
import json
from pathlib import Path

from query_intelligence.runtime_entity_assets import RuntimeEntityAssetBuilder
from query_intelligence.nlu.entity_resolver import EntityResolver


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_runtime_entity_builder_extracts_stock_symbols_from_training_corpus(tmp_path: Path) -> None:
    seed_entities = [
        {
            "entity_id": "1",
            "canonical_name": "贵州茅台",
            "normalized_name": "贵州茅台",
            "symbol": "600519.SH",
            "market": "CN",
            "exchange": "SSE",
            "entity_type": "stock",
            "industry_name": "白酒",
            "status": "active",
        }
    ]
    seed_aliases = [
        {
            "alias_id": "1",
            "entity_id": "1",
            "alias_text": "贵州茅台",
            "normalized_alias": "贵州茅台",
            "alias_type": "official_name",
            "priority": "1",
            "is_official": "true",
        }
    ]
    training_assets_dir = tmp_path / "training_assets"
    training_assets_dir.mkdir()
    (training_assets_dir / "retrieval_corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "title": "1Q24：供应链问题尚待解决",
                        "body": "证券研究报告 寒武纪 (688256 CH) 1Q24：供应链问题尚待解决",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "title": "基金重仓股",
                        "body": "股票代码 688256.SH 股票简称 寒武纪-U 占比 3.57",
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    (training_assets_dir / "alias_catalog.jsonl").write_text("", encoding="utf-8")

    builder = RuntimeEntityAssetBuilder(
        seed_entities=seed_entities,
        seed_aliases=seed_aliases,
        training_assets_dir=training_assets_dir,
    )

    entities, aliases = builder.build()

    hanwuji = next(row for row in entities if row["symbol"] == "688256.SH")
    assert hanwuji["canonical_name"] == "寒武纪"
    assert hanwuji["entity_type"] == "stock"
    alias_texts = {row["alias_text"] for row in aliases if row["entity_id"] == hanwuji["entity_id"]}
    assert {"寒武纪", "寒武纪-U", "688256", "688256.SH"} <= alias_texts


def test_materialized_runtime_assets_are_preferred_by_data_loader(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    runtime_dir = data_dir / "runtime"
    runtime_dir.mkdir(parents=True)
    _write_csv(
        data_dir / "entity_master.csv",
        [
            {
                "entity_id": "1",
                "canonical_name": "贵州茅台",
                "normalized_name": "贵州茅台",
                "symbol": "600519.SH",
                "market": "CN",
                "exchange": "SSE",
                "entity_type": "stock",
                "industry_name": "白酒",
                "status": "active",
            }
        ],
    )
    _write_csv(
        data_dir / "alias_table.csv",
        [
            {
                "alias_id": "1",
                "entity_id": "1",
                "alias_text": "贵州茅台",
                "normalized_alias": "贵州茅台",
                "alias_type": "official_name",
                "priority": "1",
                "is_official": "true",
            }
        ],
    )
    _write_csv(
        runtime_dir / "entity_master.csv",
        [
            {
                "entity_id": "2",
                "canonical_name": "寒武纪",
                "normalized_name": "寒武纪",
                "symbol": "688256.SH",
                "market": "CN",
                "exchange": "SSE",
                "entity_type": "stock",
                "industry_name": "",
                "status": "active",
            }
        ],
    )
    _write_csv(
        runtime_dir / "alias_table.csv",
        [
            {
                "alias_id": "2",
                "entity_id": "2",
                "alias_text": "寒武纪",
                "normalized_alias": "寒武纪",
                "alias_type": "official_name",
                "priority": "1",
                "is_official": "true",
            }
        ],
    )
    monkeypatch.setattr("query_intelligence.data_loader.DATA_DIR", data_dir)

    from query_intelligence.data_loader import clear_data_caches, load_aliases, load_entities

    clear_data_caches()
    try:
        assert load_entities()[0]["canonical_name"] == "寒武纪"
        assert load_aliases()[0]["alias_text"] == "寒武纪"
    finally:
        clear_data_caches()


def test_runtime_builder_adds_curated_scope_assets_and_keeps_aliases_integral(tmp_path: Path) -> None:
    seed_entities = [
        {
            "entity_id": "1",
            "canonical_name": "贵州茅台",
            "normalized_name": "贵州茅台",
            "symbol": "600519.SH",
            "market": "CN",
            "exchange": "SSE",
            "entity_type": "stock",
            "industry_name": "白酒",
            "status": "active",
        }
    ]
    seed_aliases = [
        {
            "alias_id": "1",
            "entity_id": "1",
            "alias_text": "贵州茅台",
            "normalized_alias": "贵州茅台",
            "alias_type": "official_name",
            "priority": "1",
            "is_official": "true",
        },
        {
            "alias_id": "2",
            "entity_id": "999",
            "alias_text": "CPI",
            "normalized_alias": "CPI",
            "alias_type": "official_name",
            "priority": "1",
            "is_official": "true",
        },
    ]
    training_assets_dir = tmp_path / "training_assets"
    training_assets_dir.mkdir()
    (training_assets_dir / "retrieval_corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"body": "证券研究报告 寒武纪 (688256 CH) 业绩改善"}, ensure_ascii=False),
                json.dumps({"body": "证券研究报告 中华企业 (600675 CH) 地产跟踪"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )
    (training_assets_dir / "alias_catalog.jsonl").write_text("", encoding="utf-8")

    builder = RuntimeEntityAssetBuilder(
        seed_entities=seed_entities,
        seed_aliases=seed_aliases,
        training_assets_dir=training_assets_dir,
    )

    entities, aliases = builder.build()

    entity_ids = {row["entity_id"] for row in entities}
    assert {row["entity_id"] for row in aliases} <= entity_ids

    aliases_by_name = {row["alias_text"]: row for row in aliases}
    expected_aliases = {"深证成指", "创业板指", "中证500", "银行板块", "白酒板块", "CPI", "PMI", "降准"}
    assert expected_aliases <= set(aliases_by_name)

    by_alias = {
        row["alias_text"]: next(entity for entity in entities if entity["entity_id"] == row["entity_id"])
        for row in aliases
        if row["alias_text"] in expected_aliases
    }
    assert by_alias["深证成指"]["entity_type"] == "index"
    assert by_alias["创业板指"]["entity_type"] == "index"
    assert by_alias["中证500"]["entity_type"] == "index"
    assert by_alias["银行板块"]["entity_type"] == "sector"
    assert by_alias["白酒板块"]["entity_type"] == "sector"
    assert by_alias["CPI"]["entity_type"] == "macro_indicator"
    assert by_alias["PMI"]["entity_type"] == "macro_indicator"
    assert by_alias["降准"]["entity_type"] == "policy"

    resolver = EntityResolver(entities, aliases)
    hanwuji_entities, _, _ = resolver.resolve_exact("寒武纪怎么看")
    assert any(entity["canonical_name"] == "寒武纪" and entity["symbol"] == "688256.SH" for entity in hanwuji_entities)

    zhonghua_entities, _, zhonghua_trace = resolver.resolve("你觉得中华汽车怎么样")
    assert zhonghua_entities == []
    assert "entity_not_found" in zhonghua_trace
