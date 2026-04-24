from __future__ import annotations

from scripts.verify_live_sources import _summarize_live_result


def test_live_source_verifier_summarizes_urls_and_non_seed_providers() -> None:
    result = _summarize_live_result(
        {
            "documents": [
                {"source_type": "news", "source_url": "https://example.com/news"},
                {"source_type": "announcement", "source_url": "https://static.cninfo.com.cn/finalpage/a.PDF"},
                {"source_type": "news", "source_url": None},
            ],
            "structured_data": [
                {"source_type": "market_api", "source_name": "tushare"},
                {"source_type": "fundamental_sql", "source_name": "akshare"},
                {"source_type": "market_api", "source_name": "seed"},
            ],
        }
    )

    assert result["has_live_news_url"] is True
    assert result["has_live_announcement_url"] is True
    assert result["has_live_market_provider"] is True
    assert result["has_live_fundamental_provider"] is True
    assert result["live_news_url_count"] == 1
    assert result["live_announcement_url_count"] == 1
    assert result["market_sources"] == ["tushare"]
    assert result["fundamental_sources"] == ["akshare"]
