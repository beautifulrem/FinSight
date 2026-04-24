from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from training.progress import StageProgress

from ..training_manifest import TrainingManifest, TrainingSourceUsage
from ..training_data import (
    build_clarification_supervision_rows_from_records,
    build_out_of_scope_supervision_rows_from_records,
    build_source_plan_supervision_rows_from_records,
    build_typo_supervision_rows_from_records,
)
from .normalize import assign_split_groups, dedupe_rows, load_standardized_rows


def build_training_assets(raw_root: str | Path, output_root: str | Path, enable_translation: bool) -> TrainingManifest:  # noqa: ARG001
    raw_dir = Path(raw_root)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = StageProgress("build_training_assets", 5)

    progress.advance("loading and standardizing raw rows")
    rows = assign_split_groups(dedupe_rows(load_standardized_rows(raw_dir)))
    progress.note(f"standardized rows: {len(rows)}")
    grouped_rows = defaultdict(list)
    source_counts = Counter()
    progress.advance("partitioning sample families")
    for row in rows:
        family = _normalize_family(row.get("sample_family"))
        grouped_rows[family].append(row)
        source_counts[str(row.get("source_id") or "unknown")] += 1

    classification_rows = [*grouped_rows.get("classification", []), *_build_curated_boundary_classification_rows()]
    source_counts["curated_boundary_cases"] += len(classification_rows) - len(grouped_rows.get("classification", []))
    entity_rows = grouped_rows.get("entity", [])
    retrieval_rows = grouped_rows.get("retrieval_corpus", [])
    qrel_rows = grouped_rows.get("qrel", [])
    alias_rows = grouped_rows.get("alias", [])
    progress.advance("building supervision artifacts")
    source_plan_rows = build_source_plan_supervision_rows_from_records(classification_rows)
    clarification_rows = build_clarification_supervision_rows_from_records(classification_rows)
    out_of_scope_rows = build_out_of_scope_supervision_rows_from_records(classification_rows)
    typo_rows = build_typo_supervision_rows_from_records(alias_rows)
    if not retrieval_rows:
        retrieval_rows = _build_retrieval_corpus_rows(qrel_rows)
    progress.note(
        "family counts: "
        f"classification={len(classification_rows)}, "
        f"entity={len(entity_rows)}, "
        f"retrieval_corpus={len(retrieval_rows)}, "
        f"qrel={len(qrel_rows)}, "
        f"alias={len(alias_rows)}"
    )

    progress.advance("writing jsonl artifacts")
    _write_jsonl(output_dir / "classification.jsonl", classification_rows)
    _write_jsonl(output_dir / "entity_annotations.jsonl", entity_rows)
    _write_jsonl(output_dir / "retrieval_corpus.jsonl", retrieval_rows)
    _write_jsonl(output_dir / "qrels.jsonl", qrel_rows)
    _write_jsonl(output_dir / "alias_catalog.jsonl", alias_rows)
    _write_jsonl(output_dir / "source_plan_supervision.jsonl", source_plan_rows)
    _write_jsonl(output_dir / "clarification_supervision.jsonl", clarification_rows)
    _write_jsonl(output_dir / "out_of_scope_supervision.jsonl", out_of_scope_rows)
    _write_jsonl(output_dir / "typo_supervision.jsonl", typo_rows)

    manifest_payload = {
        "classification_path": "classification.jsonl",
        "entity_annotations_path": "entity_annotations.jsonl",
        "retrieval_corpus_path": "retrieval_corpus.jsonl",
        "qrels_path": "qrels.jsonl",
        "alias_catalog_path": "alias_catalog.jsonl",
        "source_plan_supervision_path": "source_plan_supervision.jsonl",
        "clarification_supervision_path": "clarification_supervision.jsonl",
        "out_of_scope_supervision_path": "out_of_scope_supervision.jsonl",
        "typo_supervision_path": "typo_supervision.jsonl",
        "sources": [
            {
                "source_id": source_id,
                "record_count": count,
                "status": "ready",
            }
            for source_id, count in sorted(source_counts.items())
        ],
    }
    progress.advance("writing manifest and training report")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_report.json").write_text(
        json.dumps(
            {
                "sources": manifest_payload["sources"],
                "record_counts": {
                    "classification": len(classification_rows),
                    "entity": len(entity_rows),
                    "retrieval_corpus": len(retrieval_rows),
                    "qrel": len(qrel_rows),
                    "alias": len(alias_rows),
                    "source_plan_supervision": len(source_plan_rows),
                    "clarification_supervision": len(clarification_rows),
                    "out_of_scope_supervision": len(out_of_scope_rows),
                    "typo_supervision": len(typo_rows),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainingManifest(
        root_dir=output_dir,
        classification_path=output_dir / "classification.jsonl",
        entity_annotations_path=output_dir / "entity_annotations.jsonl",
        retrieval_corpus_path=output_dir / "retrieval_corpus.jsonl",
        qrels_path=output_dir / "qrels.jsonl",
        alias_catalog_path=output_dir / "alias_catalog.jsonl",
        source_plan_supervision_path=output_dir / "source_plan_supervision.jsonl",
        clarification_supervision_path=output_dir / "clarification_supervision.jsonl",
        out_of_scope_supervision_path=output_dir / "out_of_scope_supervision.jsonl",
        typo_supervision_path=output_dir / "typo_supervision.jsonl",
        sources=tuple(
            TrainingSourceUsage(
                source_id=item["source_id"],
                record_count=int(item["record_count"]),
                status=item["status"],
            )
            for item in manifest_payload["sources"]
        ),
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def _normalize_family(value: object) -> str:
    family = str(value or "")
    if family == "entity_annotation":
        return "entity"
    if family == "alias_catalog":
        return "alias"
    return family


def _build_retrieval_corpus_rows(qrel_rows: list[dict]) -> list[dict]:
    seen_doc_ids: set[str] = set()
    corpus_rows: list[dict] = []
    for row in qrel_rows:
        doc_id = str(row.get("doc_id") or "").strip()
        doc_text = str(row.get("doc_text") or "").strip()
        if not doc_id or not doc_text or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        corpus_rows.append(
            {
                "sample_family": "retrieval_corpus",
                "doc_id": doc_id,
                "title": "",
                "body": doc_text,
                "text": doc_text,
                "source_id": row.get("source_id"),
                "split_lock_key": row.get("doc_id"),
            }
        )
    return corpus_rows


def _build_curated_boundary_classification_rows() -> list[dict]:
    known_stocks = [
        ("中国平安", "601318.SH"),
        ("贵州茅台", "600519.SH"),
        ("五粮液", "000858.SZ"),
        ("平安银行", "000001.SZ"),
    ]
    unknown_companies = [
        "中华汽车",
        "星辰科技",
        "华新医疗",
        "蓝海能源",
        "银河智能",
        "中瑞生物",
        "东海材料",
        "华南水泥",
        "北辰电力",
        "中原消费",
        "海川装备",
        "明德药业",
    ]
    open_templates = [
        "你觉得{name}怎么样",
        "你觉得{name}现在怎么样",
        "你怎么看{name}",
        "你如何看{name}",
        "怎么看{name}这个公司",
        "{name}现在怎么样",
        "{name}最近怎么样",
        "{name}整体怎么样",
        "{name}值得关注吗",
        "{name}还值得关注吗",
        "{name}值得拿吗",
        "{name}适合继续持有吗",
        "{name}后面怎么看",
        "{name}后续怎么看",
        "帮我评价一下{name}",
        "帮我分析一下{name}",
        "从投资角度看{name}怎么样",
        "从基本面看{name}怎么样",
        "从估值和风险看{name}怎么样",
    ]

    rows: list[dict] = []
    for name, symbol in known_stocks:
        for template in open_templates:
            query = template.format(name=name)
            rows.append(
                _curated_classification_row(
                    query=query,
                    product_type="stock",
                    intent_labels=["hold_judgment", "fundamental_analysis", "market_explanation"],
                    topic_labels=["fundamentals", "valuation", "risk", "price", "news"],
                    question_style="advice",
                    expected_document_sources=["research_note", "news", "announcement"],
                    expected_structured_sources=["market_api", "fundamental_sql"],
                    primary_symbol=symbol,
                    needs_clarification=False,
                )
            )

    for name in unknown_companies:
        for template in open_templates:
            query = template.format(name=name)
            rows.append(
                _curated_classification_row(
                    query=query,
                    product_type="stock",
                    intent_labels=["hold_judgment", "fundamental_analysis", "market_explanation"],
                    topic_labels=["fundamentals", "valuation", "risk", "price", "news"],
                    question_style="advice",
                    expected_document_sources=[],
                    expected_structured_sources=[],
                    primary_symbol="",
                    needs_clarification=True,
                )
            )
    return rows


def _curated_classification_row(
    query: str,
    product_type: str,
    intent_labels: list[str],
    topic_labels: list[str],
    question_style: str,
    expected_document_sources: list[str],
    expected_structured_sources: list[str],
    primary_symbol: str,
    needs_clarification: bool,
) -> dict:
    return {
        "sample_family": "classification",
        "source_id": "curated_boundary_cases",
        "source_version": "v1",
        "split_lock_key": query,
        "query": query,
        "product_type": product_type,
        "intent_labels": intent_labels,
        "topic_labels": topic_labels,
        "question_style": question_style,
        "sentiment_label": "neutral",
        "expected_document_sources": expected_document_sources,
        "expected_structured_sources": expected_structured_sources,
        "primary_symbol": primary_symbol,
        "comparison_symbol": "",
        "needs_clarification": needs_clarification,
        "available_labels": [
            "product_type",
            "intent_labels",
            "topic_labels",
            "expected_document_sources",
            "expected_structured_sources",
            "question_style",
            "sentiment_label",
        ],
    }
