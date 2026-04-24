from __future__ import annotations

from query_intelligence.retrieval.selector import DocumentSelector


def test_document_selector_prefers_live_url_documents_within_same_source_type() -> None:
    selector = DocumentSelector()

    selected = selector.select(
        [
            {
                "evidence_id": "seed_news",
                "source_type": "news",
                "source_name": "public_news",
                "source_url": None,
                "retrieval_score": 0.8,
                "rank_score": 0.9,
            },
            {
                "evidence_id": "live_news",
                "source_type": "news",
                "source_name": "东方财富",
                "source_url": "https://example.com/news",
                "retrieval_score": 0.5,
                "rank_score": 0.7,
            },
        ],
        source_plan=["news"],
        top_k=1,
    )

    assert selected[0]["evidence_id"] == "live_news"
