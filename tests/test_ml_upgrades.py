from __future__ import annotations

import json
from pathlib import Path

import pytest

from query_intelligence.nlu.clarification_gate import ClarificationGate
from query_intelligence.nlu.out_of_scope_detector import OutOfScopeDetector
from query_intelligence.nlu.source_plan_reranker import SourcePlanReranker
from query_intelligence.nlu.typo_linker import TypoLinker
from query_intelligence.nlu.classifiers import MultiLabelClassifier
from query_intelligence.service import build_default_service, clear_service_caches
from query_intelligence.training_data import load_training_rows
from training.train_clarification_gate import main as train_clarification_gate_main
from training.train_entity_crf import main as train_entity_crf_main
from training.train_out_of_scope_detector import main as train_out_of_scope_detector_main
from training.prepare_training_run import prepare_training_run
from training.train_question_style_reranker import main as train_question_style_reranker_main
from training.train_all import train_all_models
from training.train_ranker import train_ranker_model
from training.train_source_plan_reranker import main as train_source_plan_reranker_main
from training.sync_and_train import main as sync_and_train_main
from training.train_typo_linker import main as train_typo_linker_main


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def _write_manifest(assets_dir: Path, payload: dict) -> None:
    (assets_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_multilabel_classifier_returns_non_empty_predictions_on_in_domain_query() -> None:
    rows = load_training_rows(ROOT / "data" / "query_labels.csv")
    classifier = MultiLabelClassifier.build_from_records(rows, "intent_labels")

    result = classifier.predict("中国平安还值得持有吗？")

    assert result
    assert any(item["label"] in {"hold_judgment", "risk_analysis", "valuation_analysis"} for item in result)


def test_train_ranker_model_outputs_trained_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "ranker.joblib"

    metadata = train_ranker_model(
        dataset_path=ROOT / "data" / "query_labels.csv",
        output_path=artifact_path,
    )

    assert artifact_path.exists()
    assert metadata["training_rows"] > 0
    assert metadata["positive_rows"] > 0
    assert "feature_names" in metadata


def test_ml_source_planner_adds_research_note_for_etf_buy_sell_query() -> None:
    service = build_default_service()

    result = service.analyze_query("证券ETF最近能买吗？", debug=True)

    assert "research_note" in result["source_plan"]


def test_source_plan_reranker_prioritizes_news_for_news_query() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("最近有哪些茅台新闻？", debug=True)

    assert result["source_plan"][0] == "news"
    assert "announcement" in result["source_plan"]
    assert "market_api" not in result["source_plan"]
    assert "fundamental_sql" not in result["source_plan"]


def test_recent_news_query_drops_price_and_fundamental_sources() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("Any recent news about Moutai?", debug=True)

    assert result["time_scope"] == "recent_1w"
    assert result["source_plan"][0] == "news"
    assert "announcement" in result["source_plan"]
    assert "market_api" not in result["source_plan"]
    assert "fundamental_sql" not in result["source_plan"]


def test_source_plan_reranker_prioritizes_announcement_for_announcement_query() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("贵州茅台最近有什么公告？", debug=True)

    assert result["source_plan"][0] == "announcement"
    assert "market_api" not in result["source_plan"][:2]
    assert "fundamental_sql" not in result["source_plan"]
    assert "industry_sql" not in result["source_plan"]


def test_source_plan_reranker_prioritizes_fundamentals_for_fundamental_query() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("中国平安基本面怎么样？", debug=True)

    assert result["source_plan"][0] == "fundamental_sql"
    assert result["source_plan"].index("fundamental_sql") < result["source_plan"].index("news")


def test_source_plan_reranker_filters_faq_for_stock_risk_query() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("中国平安风险高不高？", debug=True)

    assert "faq" not in result["source_plan"]
    assert "research_note" in result["source_plan"]


def test_source_plan_reranker_filters_generic_product_docs_for_stock_hold_query() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("中国平安从长期看还值得继续拿吗？", debug=True)
    retrieval_result = service.retrieve_evidence(result, top_k=5, debug=True)

    assert "faq" not in result["source_plan"]
    assert "product_doc" not in result["source_plan"]
    assert all(item["source_type"] not in {"faq", "product_doc"} for item in retrieval_result["documents"])


def test_english_recent_news_query_sets_recent_time_scope() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("Any recent news about Moutai?", debug=True)

    assert result["time_scope"] in {"recent_1w", "recent_1m"}


def test_retrieval_documents_include_multiple_relevant_source_types() -> None:
    service = build_default_service()
    nlu_result = service.analyze_query("中国平安还值得持有吗？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    source_types = {item["source_type"] for item in retrieval_result["documents"]}
    assert "research_note" in source_types
    assert len(source_types) >= 2


def test_entity_linker_disambiguates_ping_an_bank_vs_ping_an_insurance() -> None:
    service = build_default_service()

    bank_result = service.analyze_query("平安银行今天为什么跌？", debug=True)
    insurance_result = service.analyze_query("中国平安还值得持有吗？", debug=True)

    assert bank_result["entities"][0]["canonical_name"] == "平安银行"
    assert bank_result["entities"][0]["symbol"] == "000001.SZ"
    assert insurance_result["entities"][0]["canonical_name"] == "中国平安"


def test_ranker_training_uses_qrels_when_available(tmp_path: Path) -> None:
    artifact_path = tmp_path / "ranker.joblib"

    metadata = train_ranker_model(
        dataset_path=ROOT / "data" / "query_labels.csv",
        output_path=artifact_path,
    )

    assert metadata["label_source"] == "qrels"


def test_train_ranker_model_reads_qrels_from_manifest(tmp_path: Path) -> None:
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
    (assets_dir / "retrieval_corpus.jsonl").write_text(
        json.dumps(
            {
                "doc_id": "doc-1",
                "evidence_id": "doc-1",
                "title": "茅台回调",
                "summary": "贵州茅台今天回调",
                "body": "贵州茅台今天回调",
                "source_type": "news",
                "entity_symbols": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (assets_dir / "qrels.jsonl").write_text(
        json.dumps({"query": "贵州茅台今天为什么跌", "doc_id": "doc-1", "evidence_id": "doc-1", "relevance": 2}, ensure_ascii=False),
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
                "sources": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metadata = train_ranker_model(dataset_path=assets_dir / "manifest.json", output_path=tmp_path / "ranker.joblib")

    assert metadata["label_source"] == "qrels"
    assert metadata["training_rows"] > 0


def test_train_ranker_model_can_train_from_manifest_qrels_without_classification_rows(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "classification.jsonl").write_text("", encoding="utf-8")
    (assets_dir / "retrieval_corpus.jsonl").write_text(
        json.dumps(
            {
                "doc_id": "doc-1",
                "evidence_id": "doc-1",
                "title": "政策A",
                "summary": "政策A摘要",
                "body": "政策A内容",
                "source_type": "research_note",
                "entity_symbols": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (assets_dir / "qrels.jsonl").write_text(
        json.dumps({"query": "政策检索问题", "doc_id": "doc-1", "evidence_id": "doc-1", "relevance": 2}, ensure_ascii=False),
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
                "sources": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metadata = train_ranker_model(dataset_path=assets_dir / "manifest.json", output_path=tmp_path / "ranker.joblib")

    assert metadata["label_source"] == "qrels"
    assert metadata["positive_rows"] > 0


def test_train_ranker_model_adds_negative_candidates_for_positive_only_qrels(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "classification.jsonl").write_text("", encoding="utf-8")
    (assets_dir / "retrieval_corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": "doc-1",
                        "evidence_id": "doc-1",
                        "title": "正样本",
                        "body": "茅台研报内容",
                        "source_type": "research_note",
                        "entity_symbols": [],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "doc_id": "doc-2",
                        "evidence_id": "doc-2",
                        "title": "负样本",
                        "body": "完全无关的公告",
                        "source_type": "announcement",
                        "entity_symbols": [],
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    (assets_dir / "qrels.jsonl").write_text(
        json.dumps({"query": "贵州茅台营收是多少", "doc_id": "doc-1", "evidence_id": "doc-1", "relevance": 2}, ensure_ascii=False),
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
                "sources": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metadata = train_ranker_model(dataset_path=assets_dir / "manifest.json", output_path=tmp_path / "ranker.joblib")

    assert metadata["training_rows"] > metadata["positive_rows"]


def test_train_all_models_runs_each_trainer_once(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_run(command: list[str], check: bool) -> None:  # noqa: ARG001
        calls.append((command[2], command[-1]))

    monkeypatch.setattr("training.train_all.subprocess.run", fake_run)

    train_all_models(manifest_path=tmp_path / "manifest.json")

    assert ("training.train_product_type", str(tmp_path / "manifest.json")) in calls
    assert ("training.train_ranker", str(tmp_path / "manifest.json")) in calls
    assert ("training.train_typo_linker", str(tmp_path / "manifest.json")) in calls


def test_sync_and_train_runs_sync_build_and_train(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run(command: list[str], check: bool) -> None:  # noqa: ARG001
        calls.append(command[2])

    monkeypatch.setattr("training.sync_and_train.subprocess.run", fake_run)

    sync_and_train_main()

    assert calls == [
        "scripts.sync_public_datasets",
        "scripts.build_training_assets",
        "training.prepare_training_run",
        "training.train_all",
    ]


def test_prepare_training_run_writes_ready_report(tmp_path: Path, monkeypatch) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    for name in (
        "classification.jsonl",
        "entity_annotations.jsonl",
        "retrieval_corpus.jsonl",
        "qrels.jsonl",
        "alias_catalog.jsonl",
        "source_plan_supervision.jsonl",
        "clarification_supervision.jsonl",
        "out_of_scope_supervision.jsonl",
        "typo_supervision.jsonl",
    ):
        (assets_dir / name).write_text("{\"ok\": true}\n", encoding="utf-8")
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
                "sources": [{"source_id": "demo_source", "record_count": 10, "status": "ready"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (assets_dir / "training_report.json").write_text(
        json.dumps(
            {
                "sources": [{"source_id": "demo_source", "record_count": 10, "status": "ready"}],
                "record_counts": {
                    "classification": 1,
                    "entity": 1,
                    "retrieval_corpus": 1,
                    "qrel": 1,
                    "alias": 1,
                    "source_plan_supervision": 1,
                    "clarification_supervision": 1,
                    "out_of_scope_supervision": 1,
                    "typo_supervision": 1,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    report = prepare_training_run(assets_dir / "manifest.json")

    assert report["status"] == "ready"
    assert (assets_dir / "preflight_report.json").exists()
    assert (tmp_path / "models").exists()
    assert any(item["module"] == "training.train_typo_linker" for item in report["training_modules"])


def test_prepare_training_run_fails_when_required_supervision_is_missing(tmp_path: Path, monkeypatch) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    for name in (
        "classification.jsonl",
        "entity_annotations.jsonl",
        "retrieval_corpus.jsonl",
        "qrels.jsonl",
        "alias_catalog.jsonl",
        "clarification_supervision.jsonl",
        "out_of_scope_supervision.jsonl",
        "typo_supervision.jsonl",
    ):
        (assets_dir / name).write_text("{\"ok\": true}\n", encoding="utf-8")
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
                "sources": [{"source_id": "demo_source", "record_count": 10, "status": "ready"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (assets_dir / "training_report.json").write_text(
        json.dumps(
            {
                "sources": [{"source_id": "demo_source", "record_count": 10, "status": "ready"}],
                "record_counts": {
                    "classification": 1,
                    "entity": 1,
                    "retrieval_corpus": 1,
                    "qrel": 1,
                    "alias": 1,
                    "source_plan_supervision": 1,
                    "clarification_supervision": 1,
                    "out_of_scope_supervision": 1,
                    "typo_supervision": 1,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError):
        prepare_training_run(assets_dir / "manifest.json")

    report = json.loads((assets_dir / "preflight_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert any("source_plan_supervision" in item for item in report["errors"])


def test_question_style_and_sentiment_are_ml_backed_outputs() -> None:
    service = build_default_service()

    why_result = service.analyze_query("中国平安最近为什么涨？", debug=True)
    advice_result = service.analyze_query("证券ETF适合长期拿吗？", debug=True)

    assert why_result["question_style"] == "why"
    assert advice_result["question_style"] == "advice"
    assert advice_result["sentiment_of_user"] in {"neutral", "anxious", "bearish", "bullish"}


def test_question_style_reranker_promotes_forecast_query() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("茅台接下来会怎么走", debug=True)

    assert result["question_style"] == "forecast"


def test_question_style_reranker_promotes_advice_for_suitability_query() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("创业板ETF波动大是不是不适合长期拿", debug=True)

    assert result["question_style"] == "advice"


def test_question_style_reranker_promotes_compare_for_difference_query() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("基金净值和ETF价格到底有啥不同", debug=True)

    assert result["question_style"] == "compare"


def test_entity_crf_training_artifact_is_produced(tmp_path: Path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train_entity_crf.py", str(ROOT / "data" / "query_labels.csv")])

    train_entity_crf_main()

    assert (models_dir / "entity_crf.joblib").exists()


def test_clarification_gate_training_artifact_is_produced(tmp_path: Path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train_clarification_gate.py", str(ROOT / "data" / "query_labels.csv")])

    train_clarification_gate_main()

    assert (models_dir / "clarification_gate.joblib").exists()


def test_typo_linker_training_artifact_is_produced(tmp_path: Path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train_typo_linker.py", str(ROOT / "data" / "alias_table.csv")])

    train_typo_linker_main()

    assert (models_dir / "typo_linker.joblib").exists()


def test_question_style_reranker_training_artifact_is_produced(tmp_path: Path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train_question_style_reranker.py", str(ROOT / "data" / "query_labels.csv")])

    train_question_style_reranker_main()

    assert (models_dir / "question_style_reranker.joblib").exists()


def test_source_plan_reranker_training_artifact_is_produced(tmp_path: Path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train_source_plan_reranker.py", str(ROOT / "data" / "query_labels.csv")])

    train_source_plan_reranker_main()

    assert (models_dir / "source_plan_reranker.joblib").exists()


def test_out_of_scope_detector_training_artifact_is_produced(tmp_path: Path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train_out_of_scope_detector.py", str(ROOT / "data" / "query_labels.csv")])

    train_out_of_scope_detector_main()

    assert (models_dir / "out_of_scope_detector.joblib").exists()


def test_manifest_supervision_artifacts_are_preferred_for_target_trainers(tmp_path: Path, monkeypatch) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    _write_jsonl(
        assets_dir / "classification.jsonl",
        [
            {
                "query": "legacy source plan query",
                "product_type": "stock",
                "intent_labels": ["event_news_query"],
                "topic_labels": ["news"],
                "expected_document_sources": ["news"],
                "expected_structured_sources": [],
                "question_style": "why",
                "sentiment_label": "neutral",
            },
            {
                "query": "legacy clarification query",
                "product_type": "stock",
                "intent_labels": ["buy_sell_timing"],
                "topic_labels": ["price"],
                "expected_document_sources": [],
                "expected_structured_sources": [],
                "primary_symbol": "",
                "comparison_symbol": "",
                "question_style": "advice",
                "sentiment_label": "neutral",
            },
        ],
    )
    _write_jsonl(
        assets_dir / "source_plan_supervision.jsonl",
        [
            {
                "query": "new source plan query",
                "product_type": "stock",
                "intent_labels": ["event_news_query"],
                "topic_labels": ["news"],
                "source": "announcement",
                "label": 1,
            },
            {
                "query": "new source plan query",
                "product_type": "stock",
                "intent_labels": ["event_news_query"],
                "topic_labels": ["news"],
                "source": "market_api",
                "label": 0,
            },
        ],
    )
    _write_jsonl(
        assets_dir / "clarification_supervision.jsonl",
        [
            {
                "query": "new clarification query",
                "product_type": "stock",
                "intent_labels": ["buy_sell_timing"],
                "topic_labels": ["price"],
                "needs_clarification": True,
            },
            {
                "query": "new clarification query",
                "product_type": "stock",
                "intent_labels": ["buy_sell_timing"],
                "topic_labels": ["price"],
                "needs_clarification": False,
            },
        ],
    )
    _write_jsonl(
        assets_dir / "out_of_scope_supervision.jsonl",
        [
            {"query": "new out of scope query", "out_of_scope": 1},
            {"query": "new out of scope query", "out_of_scope": 0},
        ],
    )
    _write_jsonl(
        assets_dir / "typo_supervision.jsonl",
        [
            {"query": "new typo query", "mention": "新别名", "alias": "新公司", "heuristic_score": 1.0, "label": 1},
            {"query": "new typo query", "mention": "新别名", "alias": "旧公司", "heuristic_score": 0.0, "label": 0},
        ],
    )
    _write_jsonl(
        assets_dir / "alias_catalog.jsonl",
        [
            {"normalized_alias": "legacy-alias"},
        ],
    )
    _write_jsonl(assets_dir / "entity_annotations.jsonl", [])
    _write_jsonl(assets_dir / "retrieval_corpus.jsonl", [])
    _write_jsonl(assets_dir / "qrels.jsonl", [])
    _write_manifest(
        assets_dir,
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
    )

    source_rows: list[dict] = []
    clarification_rows: list[dict] = []
    out_of_scope_rows: list[dict] = []
    typo_rows: list[dict] = []

    source_original = SourcePlanReranker.build_from_rows.__func__
    clarification_original = ClarificationGate.build_from_rows.__func__
    out_of_scope_original = OutOfScopeDetector.build_from_rows.__func__
    typo_original = TypoLinker.build_from_rows.__func__

    def capture_source(cls, rows: list[dict]):  # noqa: ANN001
        source_rows.extend(rows)
        return source_original(cls, rows)

    def capture_clarification(cls, rows: list[dict]):  # noqa: ANN001
        clarification_rows.extend(rows)
        return clarification_original(cls, rows)

    def capture_out_of_scope(cls, rows: list[dict]):  # noqa: ANN001
        out_of_scope_rows.extend(rows)
        return out_of_scope_original(cls, rows)

    def capture_typo(cls, rows: list[dict]):  # noqa: ANN001
        typo_rows.extend(rows)
        return typo_original(cls, rows)

    monkeypatch.setattr(SourcePlanReranker, "build_from_rows", classmethod(capture_source))
    monkeypatch.setattr(ClarificationGate, "build_from_rows", classmethod(capture_clarification))
    monkeypatch.setattr(OutOfScopeDetector, "build_from_rows", classmethod(capture_out_of_scope))
    monkeypatch.setattr(TypoLinker, "build_from_rows", classmethod(capture_typo))

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train_source_plan_reranker.py", str(assets_dir / "manifest.json")])
    train_source_plan_reranker_main()
    monkeypatch.setattr("sys.argv", ["train_clarification_gate.py", str(assets_dir / "manifest.json")])
    train_clarification_gate_main()
    monkeypatch.setattr("sys.argv", ["train_out_of_scope_detector.py", str(assets_dir / "manifest.json")])
    train_out_of_scope_detector_main()
    monkeypatch.setattr("sys.argv", ["train_typo_linker.py", str(assets_dir / "manifest.json")])
    train_typo_linker_main()

    assert (models_dir / "source_plan_reranker.joblib").exists()
    assert (models_dir / "clarification_gate.joblib").exists()
    assert (models_dir / "out_of_scope_detector.joblib").exists()
    assert (models_dir / "typo_linker.joblib").exists()
    assert any(row.get("query") == "new source plan query" for row in source_rows)
    assert any(row.get("source") == "announcement" for row in source_rows)
    assert any(row.get("query") == "new clarification query" for row in clarification_rows)
    assert any(row.get("query") == "new out of scope query" for row in out_of_scope_rows)
    assert any(row.get("alias") == "新公司" for row in typo_rows)
    assert all(row.get("query") != "legacy clarification query" for row in clarification_rows)
    assert all(row.get("query") != "legacy source plan query" for row in out_of_scope_rows)
    assert all(row.get("alias") != "legacy-alias" for row in typo_rows)


def test_missing_supervision_artifacts_fall_back_to_existing_logic(tmp_path: Path, monkeypatch) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    _write_jsonl(
        assets_dir / "classification.jsonl",
        [
            {
                "query": "legacy fallback source plan query",
                "product_type": "stock",
                "intent_labels": ["event_news_query"],
                "topic_labels": ["news"],
                "expected_document_sources": ["news"],
                "expected_structured_sources": [],
                "question_style": "why",
                "sentiment_label": "neutral",
            },
            {
                "query": "legacy fallback clarification query",
                "product_type": "stock",
                "intent_labels": ["buy_sell_timing"],
                "topic_labels": ["price"],
                "expected_document_sources": [],
                "expected_structured_sources": [],
                "primary_symbol": "",
                "comparison_symbol": "",
                "question_style": "advice",
                "sentiment_label": "neutral",
            },
            {
                "query": "legacy fallback out of scope query",
                "product_type": "stock",
                "intent_labels": ["buy_sell_timing"],
                "topic_labels": ["price"],
                "expected_document_sources": [],
                "expected_structured_sources": [],
                "question_style": "advice",
                "sentiment_label": "neutral",
            },
        ],
    )
    _write_jsonl(
        assets_dir / "alias_catalog.jsonl",
        [
            {"normalized_alias": "legacy-alias"},
        ],
    )
    _write_jsonl(assets_dir / "entity_annotations.jsonl", [])
    _write_jsonl(assets_dir / "retrieval_corpus.jsonl", [])
    _write_jsonl(assets_dir / "qrels.jsonl", [])
    _write_manifest(
        assets_dir,
        {
            "classification_path": "classification.jsonl",
            "entity_annotations_path": "entity_annotations.jsonl",
            "retrieval_corpus_path": "retrieval_corpus.jsonl",
            "qrels_path": "qrels.jsonl",
            "alias_catalog_path": "alias_catalog.jsonl",
            "sources": [],
        },
    )

    source_rows: list[dict] = []
    clarification_rows: list[dict] = []
    out_of_scope_rows: list[dict] = []
    typo_rows: list[dict] = []

    source_original = SourcePlanReranker.build_from_rows.__func__
    clarification_original = ClarificationGate.build_from_rows.__func__
    out_of_scope_original = OutOfScopeDetector.build_from_rows.__func__
    typo_original = TypoLinker.build_from_rows.__func__

    def capture_source(cls, rows: list[dict]):  # noqa: ANN001
        source_rows.extend(rows)
        return source_original(cls, rows)

    def capture_clarification(cls, rows: list[dict]):  # noqa: ANN001
        clarification_rows.extend(rows)
        return clarification_original(cls, rows)

    def capture_out_of_scope(cls, rows: list[dict]):  # noqa: ANN001
        out_of_scope_rows.extend(rows)
        return out_of_scope_original(cls, rows)

    def capture_typo(cls, rows: list[dict]):  # noqa: ANN001
        typo_rows.extend(rows)
        return typo_original(cls, rows)

    monkeypatch.setattr(SourcePlanReranker, "build_from_rows", classmethod(capture_source))
    monkeypatch.setattr(ClarificationGate, "build_from_rows", classmethod(capture_clarification))
    monkeypatch.setattr(OutOfScopeDetector, "build_from_rows", classmethod(capture_out_of_scope))
    monkeypatch.setattr(TypoLinker, "build_from_rows", classmethod(capture_typo))

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["train_source_plan_reranker.py", str(assets_dir / "manifest.json")])
    train_source_plan_reranker_main()
    monkeypatch.setattr("sys.argv", ["train_clarification_gate.py", str(assets_dir / "manifest.json")])
    train_clarification_gate_main()
    monkeypatch.setattr("sys.argv", ["train_out_of_scope_detector.py", str(assets_dir / "manifest.json")])
    train_out_of_scope_detector_main()
    monkeypatch.setattr("sys.argv", ["train_typo_linker.py", str(assets_dir / "manifest.json")])
    train_typo_linker_main()

    assert (models_dir / "source_plan_reranker.joblib").exists()
    assert (models_dir / "clarification_gate.joblib").exists()
    assert (models_dir / "out_of_scope_detector.joblib").exists()
    assert (models_dir / "typo_linker.joblib").exists()
    assert any(row.get("query") == "legacy fallback source plan query" for row in source_rows)
    assert any(row.get("query") == "legacy fallback clarification query" for row in clarification_rows)
    assert any(row.get("query") == "legacy fallback out of scope query" for row in out_of_scope_rows)
    assert any(row.get("alias") == "legacy-alias" for row in typo_rows)
    assert all(row.get("query") != "new source plan query" for row in source_rows)
    assert all(row.get("query") != "new clarification query" for row in clarification_rows)
    assert all(row.get("query") != "new out of scope query" for row in out_of_scope_rows)
    assert all(row.get("alias") != "新公司" for row in typo_rows)


def test_default_service_builds_entity_boundary_model() -> None:
    service = build_default_service()

    assert service.nlu_pipeline.entity_resolver.boundary_model is not None
    assert service.nlu_pipeline.clarification_gate is not None
    assert service.nlu_pipeline.entity_resolver.typo_linker is not None
    assert service.nlu_pipeline.question_style_reranker is not None
    assert service.nlu_pipeline.source_planner.source_plan_reranker is not None
    assert service.nlu_pipeline.out_of_scope_detector is not None


def test_missing_entity_slot_is_flagged_for_hold_judgment_query() -> None:
    service = build_default_service()

    result = service.analyze_query("后面还值得拿吗？", debug=True)

    assert "missing_entity" in result["missing_slots"]


def test_normalizer_does_not_duplicate_canonical_alias() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("贵州茅台今天为什么跌？", debug=True)

    assert result["normalized_query"] == "贵州茅台今天为什么跌"


def test_generic_etf_lof_query_stays_generic_and_uses_mechanism_sources() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("ETF和LOF有什么区别？", debug=True)

    assert result["entities"] == []
    assert set(result["source_plan"]) == {"research_note", "faq", "product_doc"}
    assert "market_api" not in result["source_plan"]
    assert "news" not in result["source_plan"]


def test_contextless_buy_query_prefers_clarification_over_out_of_scope() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("这个能买吗？", debug=True)

    assert result["product_type"]["label"] != "out_of_scope"
    assert "clarification_required" in result["risk_flags"]
    assert "missing_entity" in result["missing_slots"]
    assert result["source_plan"] == []


def test_english_stop_loss_query_prefers_clarification_over_out_of_scope() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("Should I cut the loss here?", debug=True)

    assert result["product_type"]["label"] != "out_of_scope"
    assert "clarification_required" in result["risk_flags"]
    assert "missing_entity" in result["missing_slots"]
    assert result["source_plan"] == []


def test_chinese_stop_loss_query_prefers_clarification_over_out_of_scope() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("这只该不该止损？", debug=True)

    assert result["product_type"]["label"] != "out_of_scope"
    assert "clarification_required" in result["risk_flags"]
    assert "missing_entity" in result["missing_slots"]
    assert result["source_plan"] == []


def test_missing_entity_query_returns_empty_retrieval_payload() -> None:
    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query("后面还值得拿吗？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert nlu_result["source_plan"] == []
    assert retrieval_result["documents"] == []
    assert retrieval_result["structured_data"] == []
    assert "clarification_required_missing_entity" in retrieval_result["warnings"]


def test_fee_query_filters_zero_and_low_score_documents() -> None:
    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query("公募基金费率包含哪些部分？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert set(nlu_result["source_plan"]) == {"research_note", "faq", "product_doc"}
    assert all(item["retrieval_score"] > 0.05 for item in retrieval_result["documents"])
    assert all(item["source_type"] in {"research_note", "faq", "product_doc"} for item in retrieval_result["documents"])


def test_duplicate_mentions_do_not_duplicate_structured_evidence() -> None:
    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query("平安和中国平安哪个好？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert len(nlu_result["entities"]) == 1
    assert nlu_result["entities"][0]["canonical_name"] == "中国平安"
    evidence_ids = [item["evidence_id"] for item in retrieval_result["structured_data"]]
    assert len(evidence_ids) == len(set(evidence_ids))


def test_generic_product_query_does_not_raise_entity_not_found_risk_flag() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("ETF申赎规则是什么？", debug=True)

    assert "entity_not_found" not in result["risk_flags"]


def test_generic_product_query_filters_out_entity_bound_documents() -> None:
    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query("ETF申赎规则是什么？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert all(not item["entity_hits"] for item in retrieval_result["documents"])


def test_unresolved_stock_compare_query_requires_clarification() -> None:
    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query("中国太保哪个好？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert "missing_entity" in nlu_result["missing_slots"]
    assert "clarification_required" in nlu_result["risk_flags"]
    assert nlu_result["source_plan"] == []
    assert retrieval_result["documents"] == []
    assert retrieval_result["structured_data"] == []


def test_unresolved_stock_query_filters_entity_bound_documents() -> None:
    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query("中国太保哪个好？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert all(not item["entity_hits"] for item in retrieval_result["documents"])


def test_converged_single_entity_drops_ambiguity_flag() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("平安和中国平安哪个好？", debug=True)

    assert len(result["entities"]) == 1
    assert "entity_ambiguous" not in result["risk_flags"]


def test_sector_market_question_is_not_forced_into_clarification() -> None:
    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query("白酒行业今天为什么跌？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert "clarification_required" not in nlu_result["risk_flags"]
    assert nlu_result["source_plan"]
    assert "clarification_required_missing_entity" not in retrieval_result["warnings"]


def test_compare_query_with_only_one_resolved_operand_requires_clarification() -> None:
    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query("上证50和沪深300有什么区别？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert "missing_entity" in nlu_result["missing_slots"]
    assert "clarification_required" in nlu_result["risk_flags"]
    assert nlu_result["source_plan"] == []
    assert retrieval_result["documents"] == []
    assert "clarification_required_missing_entity" in retrieval_result["warnings"]


def test_resolved_entity_query_does_not_keep_missing_entity_slot() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("中国平安还值得持有吗？", debug=True)

    assert result["entities"]
    assert "missing_entity" not in result["missing_slots"]


def test_sector_market_question_does_not_plan_announcement_source() -> None:
    clear_service_caches()
    service = build_default_service()

    result = service.analyze_query("白酒行业今天为什么跌？", debug=True)

    assert "announcement" not in result["source_plan"]
    assert "market_api" not in result["source_plan"]


def test_sector_market_question_retrieves_industry_sql_snapshot() -> None:
    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query("白酒行业今天为什么跌？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert any(item["source_type"] == "industry_sql" for item in retrieval_result["structured_data"])
    assert "industry_sql" in retrieval_result["executed_sources"]
    assert "market_api" not in retrieval_result["executed_sources"]


def test_sector_market_question_keeps_sector_documents() -> None:
    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query("白酒行业今天为什么跌？", debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=5, debug=True)

    assert retrieval_result["documents"]
    assert "risk" not in nlu_result["required_evidence_types"]
