from __future__ import annotations

import json
from pathlib import Path

from query_intelligence.retrieval.doc_retriever import DocumentRetriever


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def test_runtime_documents_jsonl_are_preferred_by_data_loader(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    runtime_dir = data_dir / "runtime"
    runtime_dir.mkdir(parents=True)
    (data_dir / "documents.json").write_text(
        json.dumps([{"evidence_id": "seed_doc", "title": "seed", "summary": "", "body": "", "source_type": "faq"}]),
        encoding="utf-8",
    )
    _write_jsonl(
        runtime_dir / "documents.jsonl",
        [
            {
                "evidence_id": "runtime_doc",
                "source_type": "product_doc",
                "source_name": "runtime",
                "title": "融资融券规则",
                "summary": "融资融券业务说明",
                "body": "投资者开通融资融券需满足适当性要求。",
                "entity_symbols": [],
                "product_type": "stock",
                "credibility_score": 0.8,
            }
        ],
    )
    monkeypatch.setattr("query_intelligence.data_loader.DATA_DIR", data_dir)
    monkeypatch.delenv("QI_DOCUMENTS_PATH", raising=False)

    from query_intelligence.data_loader import clear_data_caches, load_documents

    clear_data_caches()
    try:
        documents = load_documents()
        assert [doc["evidence_id"] for doc in documents] == ["runtime_doc", "seed_doc"]
    finally:
        clear_data_caches()


def test_documents_path_env_override_reads_jsonl(tmp_path: Path, monkeypatch) -> None:
    override_path = tmp_path / "override.jsonl"
    _write_jsonl(
        override_path,
        [
            {
                "evidence_id": "env_doc_1",
                "source_type": "faq",
                "source_name": "env",
                "title": "开户 FAQ",
                "summary": "开户材料",
                "body": "开户需要身份证和银行卡。",
            },
            {
                "evidence_id": "env_doc_2",
                "source_type": "news",
                "source_name": "env",
                "title": "市场新闻",
                "summary": "指数上涨",
                "body": "主要指数今日上涨。",
            },
        ],
    )
    monkeypatch.setenv("QI_DOCUMENTS_PATH", str(override_path))

    from query_intelligence.data_loader import clear_data_caches, load_documents

    clear_data_caches()
    try:
        assert [doc["evidence_id"] for doc in load_documents()] == ["env_doc_1", "env_doc_2"]
    finally:
        clear_data_caches()


def test_materializer_converts_filters_deduplicates_and_docs_are_retrievable(tmp_path: Path) -> None:
    corpus_path = tmp_path / "retrieval_corpus.jsonl"
    output_path = tmp_path / "documents.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "doc_id": "ann-1",
                "title": "平安银行_2023年年度报告",
                "body": "平安银行披露资产质量和营收情况。",
                "source_type": "announcement",
                "source_id": "fir_bench_announcements",
            },
            {
                "doc_id": "ann-1-dup",
                "title": "平安银行_2023年年度报告",
                "body": "平安银行披露资产质量和营收情况。",
                "source_type": "announcement",
                "source_id": "fir_bench_announcements",
            },
            {
                "doc_id": "faq-1",
                "title": "如何开通融资融券？",
                "body": "常见问题：融资融券账户开通需要满足交易经验和风险测评要求。",
                "source_id": "fiqa",
            },
            {
                "doc_id": "skip-1",
                "title": "",
                "body": "plain unrelated text without supported runtime source signals",
                "source_id": "unknown",
            },
        ],
    )

    from query_intelligence.runtime_document_assets import RuntimeDocumentAssetBuilder

    summary = RuntimeDocumentAssetBuilder(corpus_path=corpus_path, max_documents=10).write(output_path)
    documents = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert summary.document_count == 2
    assert summary.skipped_count == 1
    assert summary.duplicate_count == 1
    assert {doc["source_type"] for doc in documents} == {"announcement", "faq"}
    assert all({"evidence_id", "source_name", "provider", "summary", "body", "credibility_score"} <= set(doc) for doc in documents)

    retriever = DocumentRetriever(documents)
    hits = retriever.search(
        {
            "normalized_query": "融资融券如何开通",
            "entity_names": [],
            "keywords": ["融资融券", "开通"],
            "symbols": [],
            "source_plan": ["faq", "product_doc"],
        },
        top_k=3,
    )
    assert hits
    assert hits[0]["source_type"] == "faq"


def test_document_retriever_keeps_source_plan_diversity_when_one_source_dominates() -> None:
    documents = [
        {
            "evidence_id": f"research_{index}",
            "source_type": "research_note",
            "title": f"白酒行业深度报告 {index}",
            "summary": "白酒行业估值、风险、景气度分析。",
            "body": "白酒行业长期研究内容。",
            "entity_symbols": [],
        }
        for index in range(20)
    ]
    documents.append(
        {
            "evidence_id": "news_1",
            "source_type": "news",
            "title": "白酒板块新闻",
            "summary": "白酒板块今日资金关注。",
            "body": "白酒板块新闻跟踪。",
            "entity_symbols": [],
        }
    )

    hits = DocumentRetriever(documents).search(
        {
            "normalized_query": "白酒板块还能买吗",
            "entity_names": ["白酒"],
            "keywords": [],
            "symbols": [],
            "industry_terms": ["白酒"],
            "source_plan": ["news", "research_note"],
        },
        top_k=5,
    )

    assert {hit["source_type"] for hit in hits[:2]} == {"news", "research_note"}
