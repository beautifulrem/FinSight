from __future__ import annotations

import json
from pathlib import Path

from query_intelligence.config import Settings
from query_intelligence.external_data.registry import build_default_dataset_registry, resolve_enabled_sources
from query_intelligence.nlu.question_style_reranker import build_question_style_reranker_rows
from query_intelligence.training_data import (
    build_out_of_scope_supervision_rows_from_records,
    load_clarification_supervision_rows,
    load_out_of_scope_supervision_rows,
    load_source_plan_supervision_rows,
    load_typo_supervision_rows,
    filter_rows_for_label,
    load_entity_annotation_rows,
    load_qrel_rows,
    load_training_rows,
)
from query_intelligence.training_manifest import load_training_manifest


def test_load_training_manifest_round_trip(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classification_path": "classification.jsonl",
                "entity_annotations_path": "entity_annotations.jsonl",
                "retrieval_corpus_path": "retrieval_corpus.jsonl",
                "qrels_path": "qrels.jsonl",
                "alias_catalog_path": "alias_catalog.jsonl",
                "source_plan_supervision_path": "source_plan_supervision.jsonl",
                "clarification_supervision_path": "clarification_supervision.jsonl",
                "out_of_scope_supervision_path": "out_of_scope_supervision.jsonl",
                "typo_supervision_path": "typo_supervision.jsonl",
                "sources": [{"source_id": "local_seed", "record_count": 223, "status": "ready"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = load_training_manifest(manifest_path)

    assert manifest.classification_path == tmp_path / "classification.jsonl"
    assert manifest.qrels_path == tmp_path / "qrels.jsonl"
    assert manifest.source_plan_supervision_path == tmp_path / "source_plan_supervision.jsonl"
    assert manifest.clarification_supervision_path == tmp_path / "clarification_supervision.jsonl"
    assert manifest.out_of_scope_supervision_path == tmp_path / "out_of_scope_supervision.jsonl"
    assert manifest.typo_supervision_path == tmp_path / "typo_supervision.jsonl"
    assert manifest.sources[0].source_id == "local_seed"


def test_load_training_manifest_missing_optional_paths_defaults_to_none(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classification_path": "classification.jsonl",
                "entity_annotations_path": "entity_annotations.jsonl",
                "retrieval_corpus_path": "retrieval_corpus.jsonl",
                "qrels_path": "qrels.jsonl",
                "alias_catalog_path": "alias_catalog.jsonl",
                "sources": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = load_training_manifest(manifest_path)

    assert manifest.source_plan_supervision_path is None
    assert manifest.clarification_supervision_path is None
    assert manifest.out_of_scope_supervision_path is None
    assert manifest.typo_supervision_path is None


def test_resolve_enabled_sources_honors_allowlist() -> None:
    settings = Settings(
        enable_external_data=True,
        dataset_allowlist=("cflue", "fiqa"),
        enable_translation=True,
    )

    enabled = resolve_enabled_sources(build_default_dataset_registry(), settings)

    assert [item.source_id for item in enabled] == ["cflue", "fiqa"]
    assert all(item.allow_training for item in enabled)


def test_sync_result_round_trip() -> None:
    from query_intelligence.external_data.models import SyncResult

    result = SyncResult(source_id="demo_cache", status="cached", output_dir="/tmp/demo")
    assert result.source_id == "demo_cache"
    assert result.status == "cached"
    assert result.output_dir == "/tmp/demo"


def test_sync_skips_cached_sources(tmp_path: Path) -> None:
    from query_intelligence.external_data.sync import sync_source
    from query_intelligence.external_data.models import DatasetSource

    source = DatasetSource(
        source_id="demo_cache",
        source_type="local_seed",
        entrypoint=str(tmp_path / "seed"),
        version="v1",
        license="test",
        language="zh",
        market_scope="test",
        task_types=("classification",),
        quality_score=1.0,
        allow_training=True,
        allow_translation=False,
        enabled_by_default=True,
    )

    cache_dir = tmp_path / "raw" / "demo_cache" / "v1"
    cache_dir.mkdir(parents=True)
    (cache_dir / "data.jsonl").write_text("{}\n", encoding="utf-8")

    result = sync_source(source, raw_root=tmp_path / "raw")

    assert result.status == "cached"


def test_build_training_assets_emits_manifest_and_task_views(tmp_path: Path) -> None:
    from query_intelligence.external_data.build_assets import build_training_assets

    raw_dir = tmp_path / "raw" / "demo" / "v1"
    raw_dir.mkdir(parents=True)
    (raw_dir / "records.jsonl").write_text(
        "\n".join([
            json.dumps(
                {
                    "sample_family": "classification",
                    "text": "贵州茅台今天为什么跌",
                    "query": "贵州茅台今天为什么跌",
                    "product_type": "stock",
                    "intent_labels": ["market_explanation"],
                    "topic_labels": ["price", "news"],
                    "expected_document_sources": ["news", "announcement"],
                    "expected_structured_sources": ["market_api"],
                    "primary_symbol": "贵州茅台",
                    "available_labels": [
                        "product_type",
                        "intent_labels",
                        "topic_labels",
                        "expected_document_sources",
                        "expected_structured_sources",
                        "question_style",
                    ],
                    "split_lock_key": "q1",
                },
                ensure_ascii=False,
            ),
            json.dumps({"sample_family": "qrel", "query_id": "q1", "query": "贵州茅台", "doc_id": "doc-1", "relevance": 2, "split_lock_key": "q1"}, ensure_ascii=False),
            json.dumps({"sample_family": "alias", "alias_text": "茅台", "normalized_alias": "贵州茅台", "canonical_name": "贵州茅台", "entity_id": "1", "split_lock_key": "a1"}, ensure_ascii=False),
            json.dumps({"sample_family": "classification", "text": "帮我写个Python冒泡排序", "query": "帮我写个Python冒泡排序", "product_type": "software", "intent_labels": [], "topic_labels": [], "split_lock_key": "q2"}, ensure_ascii=False),
        ]),
        encoding="utf-8",
    )

    manifest = build_training_assets(raw_root=tmp_path / "raw", output_root=tmp_path / "training_assets", enable_translation=False)

    assert manifest.classification_path.exists()
    assert manifest.qrels_path.exists()
    assert manifest.source_plan_supervision_path and manifest.source_plan_supervision_path.exists()
    assert manifest.clarification_supervision_path and manifest.clarification_supervision_path.exists()
    assert manifest.out_of_scope_supervision_path and manifest.out_of_scope_supervision_path.exists()
    assert manifest.typo_supervision_path and manifest.typo_supervision_path.exists()


def test_build_training_assets_writes_training_report_and_entity_rows(tmp_path: Path) -> None:
    from query_intelligence.external_data.build_assets import build_training_assets

    raw_dir = tmp_path / "raw" / "demo" / "v1"
    raw_dir.mkdir(parents=True)
    (raw_dir / "records.jsonl").write_text(
        "\n".join([
            json.dumps(
                {
                    "sample_family": "classification",
                    "text": "中国平安今天为什么跌",
                    "query": "中国平安今天为什么跌",
                    "product_type": "stock",
                    "intent_labels": ["market_explanation"],
                    "topic_labels": ["price", "news"],
                    "expected_document_sources": ["news"],
                    "expected_structured_sources": ["market_api"],
                    "primary_symbol": "中国平安",
                    "available_labels": [
                        "product_type",
                        "intent_labels",
                        "topic_labels",
                        "expected_document_sources",
                        "expected_structured_sources",
                        "question_style",
                    ],
                    "split_lock_key": "q1",
                },
                ensure_ascii=False,
            ),
            json.dumps({"sample_family": "entity_annotation", "text": "中国平安", "tokens": ["中", "国", "平", "安"], "tags": ["B-ENT", "I-ENT", "I-ENT", "I-ENT"], "split_lock_key": "e1"}, ensure_ascii=False),
            json.dumps({"sample_family": "qrel", "query_id": "q1", "query": "中国平安", "doc_id": "doc-1", "doc_text": "中国平安公告", "relevance": 2, "split_lock_key": "q1"}, ensure_ascii=False),
            json.dumps({"sample_family": "alias", "alias_text": "平安", "normalized_alias": "中国平安", "canonical_name": "中国平安", "entity_id": "2", "split_lock_key": "a2"}, ensure_ascii=False),
        ]),
        encoding="utf-8",
    )

    manifest = build_training_assets(raw_root=tmp_path / "raw", output_root=tmp_path / "training_assets", enable_translation=False)

    entity_lines = manifest.entity_annotations_path.read_text(encoding="utf-8").splitlines()
    report_payload = json.loads((tmp_path / "training_assets" / "training_report.json").read_text(encoding="utf-8"))
    source_plan_rows = load_source_plan_supervision_rows(manifest.root_dir / "manifest.json")
    clarification_rows = load_clarification_supervision_rows(manifest.root_dir / "manifest.json")
    out_of_scope_rows = load_out_of_scope_supervision_rows(manifest.root_dir / "manifest.json")
    typo_rows = load_typo_supervision_rows(manifest.root_dir / "manifest.json")
    sources_by_id = {item["source_id"]: item for item in report_payload["sources"]}

    assert len(entity_lines) == 1
    assert sources_by_id["demo"]["record_count"] == 4
    assert source_plan_rows[0]["source"] in {"market_api", "news", "announcement"}
    assert any(row["needs_clarification"] for row in clarification_rows)
    assert {row["label"] for row in out_of_scope_rows} <= {0, 1}
    assert typo_rows[0]["alias"] == "中国平安"


def test_out_of_scope_supervision_adds_synthetic_noise_rows() -> None:
    rows = build_out_of_scope_supervision_rows_from_records(
        [
            {
                "sample_family": "classification",
                "query": "帮我写个Python冒泡排序",
                "product_type": "unknown",
                "available_labels": ["out_of_scope_only"],
            },
            {
                "sample_family": "classification",
                "query": "讲个笑话给我听",
                "product_type": "unknown",
                "available_labels": ["out_of_scope_only"],
            },
        ]
    )

    synthetic_rows = [row for row in rows if row["source_family"] == "synthetic_noise"]
    assert synthetic_rows
    assert all(row["label"] == 1 for row in synthetic_rows)
    assert all(row["source_id"] == "synthetic_ood_noise" for row in synthetic_rows)


def test_split_assignment_keeps_lock_groups_together() -> None:
    from query_intelligence.external_data.normalize import assign_split_groups

    rows = [
        {"sample_id": "a", "split_lock_key": "group-1"},
        {"sample_id": "b", "split_lock_key": "group-1"},
        {"sample_id": "c", "split_lock_key": "group-2"},
    ]

    assigned = assign_split_groups(rows)

    split_by_group: dict[str, set[str]] = {}
    for row in assigned:
        split_by_group.setdefault(row["split_lock_key"], set()).add(row["split"])
    assert split_by_group["group-1"] == {next(iter(split_by_group["group-1"]))}


def test_load_training_rows_from_manifest_returns_classification_rows(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "classification.jsonl").write_text(
        json.dumps(
            {
                "query": "贵州茅台今天为什么跌",
                "product_type": "stock",
                "intent_labels": ["market_explanation"],
                "topic_labels": ["price", "news"],
                "question_style": "why",
                "sentiment_label": "neutral",
                "split": "train",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (assets_dir / "entity_annotations.jsonl").write_text(
        json.dumps(
            {
                "text": "贵州茅台",
                "tokens": ["贵", "州", "茅", "台"],
                "tags": ["B-ENT", "I-ENT", "I-ENT", "I-ENT"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (assets_dir / "qrels.jsonl").write_text(
        json.dumps({"query": "贵州茅台今天为什么跌", "doc_id": "doc-1", "relevance": 2}, ensure_ascii=False),
        encoding="utf-8",
    )
    (assets_dir / "manifest.json").write_text(
        json.dumps(
            {
                "classification_path": "classification.jsonl",
                "entity_annotations_path": "entity_annotations.jsonl",
                "retrieval_corpus_path": "retrieval_corpus.jsonl",
                "qrels_path": "qrels.jsonl",
                "alias_catalog_path": "alias_catalog.jsonl",
                "source_plan_supervision_path": "source_plan_supervision.jsonl",
                "clarification_supervision_path": "clarification_supervision.jsonl",
                "out_of_scope_supervision_path": "out_of_scope_supervision.jsonl",
                "typo_supervision_path": "typo_supervision.jsonl",
                "sources": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (assets_dir / "source_plan_supervision.jsonl").write_text(
        json.dumps({"query": "贵州茅台今天为什么跌", "source": "news", "label": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (assets_dir / "clarification_supervision.jsonl").write_text(
        json.dumps({"query": "这个标的今天为什么跌", "base_query": "贵州茅台今天为什么跌", "needs_clarification": True}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (assets_dir / "out_of_scope_supervision.jsonl").write_text(
        json.dumps({"query": "帮我写个Python冒泡排序", "label": 1, "domain": "general"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (assets_dir / "typo_supervision.jsonl").write_text(
        json.dumps({"query": "茅台", "mention": "茅台", "alias": "贵州茅台", "label": 1}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows = load_training_rows(assets_dir / "manifest.json")
    qrels = load_qrel_rows(assets_dir / "manifest.json")
    entity_rows = load_entity_annotation_rows(assets_dir / "manifest.json")
    source_plan_rows = load_source_plan_supervision_rows(assets_dir / "manifest.json")
    clarification_rows = load_clarification_supervision_rows(assets_dir / "manifest.json")
    out_of_scope_rows = load_out_of_scope_supervision_rows(assets_dir / "manifest.json")
    typo_rows = load_typo_supervision_rows(assets_dir / "manifest.json")

    assert rows[0]["query"] == "贵州茅台今天为什么跌"
    assert qrels[0]["doc_id"] == "doc-1"
    assert entity_rows[0]["tags"][0] == "B-ENT"
    assert source_plan_rows[0]["source"] == "news"
    assert clarification_rows[0]["needs_clarification"] is True
    assert out_of_scope_rows[0]["domain"] == "general"
    assert typo_rows[0]["alias"] == "贵州茅台"


def test_question_style_reranker_rows_accept_manifest_path(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "classification.jsonl").write_text(
        json.dumps(
            {
                "query": "茅台接下来会怎么走",
                "product_type": "stock",
                "intent_labels": ["market_explanation"],
                "topic_labels": ["price"],
                "question_style": "forecast",
                "sentiment_label": "neutral",
                "split": "train",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (assets_dir / "manifest.json").write_text(
        json.dumps(
            {
                "classification_path": "classification.jsonl",
                "entity_annotations_path": "entity_annotations.jsonl",
                "retrieval_corpus_path": "retrieval_corpus.jsonl",
                "qrels_path": "qrels.jsonl",
                "alias_catalog_path": "alias_catalog.jsonl",
                "source_plan_supervision_path": "source_plan_supervision.jsonl",
                "clarification_supervision_path": "clarification_supervision.jsonl",
                "out_of_scope_supervision_path": "out_of_scope_supervision.jsonl",
                "typo_supervision_path": "typo_supervision.jsonl",
                "sources": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = build_question_style_reranker_rows(assets_dir / "manifest.json")

    assert any(row["question_style"] == "forecast" for row in rows)


def test_load_training_rows_handles_unicode_line_separators_inside_jsonl(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "classification.jsonl").write_text(
        json.dumps(
            {
                "query": "第一句\u2028第二句",
                "product_type": "stock",
                "intent_labels": ["market_explanation"],
                "topic_labels": ["price"],
                "question_style": "why",
                "sentiment_label": "neutral",
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (assets_dir / "manifest.json").write_text(
        json.dumps(
            {
                "classification_path": "classification.jsonl",
                "entity_annotations_path": "entity_annotations.jsonl",
                "retrieval_corpus_path": "retrieval_corpus.jsonl",
                "qrels_path": "qrels.jsonl",
                "alias_catalog_path": "alias_catalog.jsonl",
                "source_plan_supervision_path": "source_plan_supervision.jsonl",
                "clarification_supervision_path": "clarification_supervision.jsonl",
                "out_of_scope_supervision_path": "out_of_scope_supervision.jsonl",
                "typo_supervision_path": "typo_supervision.jsonl",
                "sources": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = load_training_rows(assets_dir / "manifest.json")

    assert rows[0]["query"] == "第一句\u2028第二句"


def test_filter_rows_for_label_honors_available_labels() -> None:
    rows = [
        {"query": "情绪数据", "sentiment_label": "positive", "available_labels": ["sentiment_label"]},
        {"query": "产品数据", "product_type": "stock", "available_labels": ["product_type"]},
        {"query": "显式标签", "product_type": "etf"},
    ]

    filtered = filter_rows_for_label(rows, "product_type")

    assert [row["query"] for row in filtered] == ["产品数据", "显式标签"]


def test_question_style_reranker_rows_skip_unlabeled_manifest_rows(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "classification.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query": "只有情感标签",
                        "sentiment_label": "positive",
                        "available_labels": ["sentiment_label"],
                        "split": "train",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "query": "茅台接下来会怎么走",
                        "product_type": "stock",
                        "intent_labels": ["market_explanation"],
                        "topic_labels": ["price"],
                        "question_style": "forecast",
                        "available_labels": ["product_type", "intent_labels", "topic_labels", "question_style"],
                        "split": "train",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (assets_dir / "manifest.json").write_text(
        json.dumps(
            {
                "classification_path": "classification.jsonl",
                "entity_annotations_path": "entity_annotations.jsonl",
                "retrieval_corpus_path": "retrieval_corpus.jsonl",
                "qrels_path": "qrels.jsonl",
                "alias_catalog_path": "alias_catalog.jsonl",
                "source_plan_supervision_path": "source_plan_supervision.jsonl",
                "clarification_supervision_path": "clarification_supervision.jsonl",
                "out_of_scope_supervision_path": "out_of_scope_supervision.jsonl",
                "typo_supervision_path": "typo_supervision.jsonl",
                "sources": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = build_question_style_reranker_rows(assets_dir / "manifest.json")

    assert not any(row["query"] == "只有情感标签" for row in rows)
    assert any(row["query"] == "茅台接下来会怎么走" for row in rows)
