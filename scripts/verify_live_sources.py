from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.artifacts import ArtifactWriter  # noqa: E402
from query_intelligence.service import build_default_service, clear_service_caches  # noqa: E402


def main() -> None:
    args = _build_parser().parse_args()
    if args.enable_live:
        os.environ["QI_USE_LIVE_MARKET"] = "1"
        os.environ["QI_USE_LIVE_NEWS"] = "1"
        os.environ["QI_USE_LIVE_ANNOUNCEMENT"] = "1"

    clear_service_caches()
    service = build_default_service()
    nlu_result = service.analyze_query(args.query, debug=args.debug)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=args.top_k, debug=args.debug)
    summary = _summarize_live_result(retrieval_result)

    payload = {
        "query": args.query,
        "query_id": nlu_result["query_id"],
        "live_summary": summary,
    }
    if args.write_artifact:
        written = ArtifactWriter(args.output_dir).write(
            query=args.query,
            nlu_result=nlu_result,
            retrieval_result=retrieval_result,
            session_id=args.session_id,
            message_id=args.message_id,
        )
        payload["artifact_dir"] = written["artifact_dir"]
        payload["artifacts"] = written["artifacts"]

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one query with live data providers and report source coverage.")
    parser.add_argument("--query", default="你觉得中国平安怎么样？")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable-live", action="store_true", default=True)
    parser.add_argument("--no-enable-live", dest="enable_live", action="store_false")
    parser.add_argument("--write-artifact", action="store_true", default=True)
    parser.add_argument("--no-write-artifact", dest="write_artifact", action="store_false")
    parser.add_argument("--output-dir", default=os.getenv("QI_API_OUTPUT_DIR", "outputs/query_intelligence"))
    parser.add_argument("--session-id", default="live-source-verification")
    parser.add_argument("--message-id", default=None)
    return parser


def _summarize_live_result(retrieval_result: dict) -> dict:
    documents = retrieval_result.get("documents", [])
    structured = retrieval_result.get("structured_data", [])
    news_urls = [item.get("source_url") for item in documents if item.get("source_type") == "news" and item.get("source_url")]
    announcement_urls = [
        item.get("source_url")
        for item in documents
        if item.get("source_type") == "announcement" and item.get("source_url")
    ]
    market_sources = [
        item.get("source_name")
        for item in structured
        if item.get("source_type") == "market_api" and item.get("source_name") and item.get("source_name") != "seed"
    ]
    fundamental_sources = [
        item.get("source_name")
        for item in structured
        if item.get("source_type") == "fundamental_sql" and item.get("source_name") and item.get("source_name") != "seed"
    ]
    return {
        "live_news_url_count": len(news_urls),
        "live_announcement_url_count": len(announcement_urls),
        "live_market_provider_count": len(market_sources),
        "live_fundamental_provider_count": len(fundamental_sources),
        "news_urls": news_urls[:5],
        "announcement_urls": announcement_urls[:5],
        "market_sources": sorted(set(market_sources)),
        "fundamental_sources": sorted(set(fundamental_sources)),
        "has_live_news_url": bool(news_urls),
        "has_live_announcement_url": bool(announcement_urls),
        "has_live_market_provider": bool(market_sources),
        "has_live_fundamental_provider": bool(fundamental_sources),
    }


if __name__ == "__main__":
    main()
