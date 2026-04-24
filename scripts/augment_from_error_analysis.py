from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_DIR = ROOT / "data"


def build_row(base_query: str, query: str, product_type: str, intents: list[str], topics: list[str], split: str, expected_doc: list[str], expected_struct: list[str]) -> dict:
    return {
        "query": query,
        "product_type": product_type,
        "intent_labels": "|".join(intents),
        "topic_labels": "|".join(topics),
        "question_style": "compare" if "peer_compare" in intents else ("why" if "market_explanation" in intents else ("advice" if {"hold_judgment", "buy_sell_timing"} & set(intents) else "fact")),
        "sentiment_label": "anxious" if any(term in query for term in ["跌", "风险", "波动"]) else ("bullish" if any(term in query for term in ["买", "定投", "持有"]) else "neutral"),
        "expected_document_sources": "|".join(expected_doc),
        "expected_structured_sources": "|".join(expected_struct),
        "primary_symbol": "",
        "comparison_symbol": "",
        "split": split,
        "augmented_from": base_query,
    }


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def augment_from_errors(rows: list[dict], error_report: dict) -> list[dict]:
    existing_queries = {row["query"] for row in rows}
    additions: list[dict] = []

    for error in error_report.get("sample_errors", []):
        query = error["query"]
        expected = error["expected"]
        matched = next((row for row in rows if row["query"] == query), None)
        if matched is None:
            continue

        product_type = matched["product_type"]
        expected_doc = [source for source in matched["expected_document_sources"].split("|") if source]
        expected_struct = [source for source in matched["expected_structured_sources"].split("|") if source]
        split = "train"
        intents = matched["intent_labels"].split("|") if matched.get("intent_labels") else []
        topics = matched["topic_labels"].split("|") if matched.get("topic_labels") else []

        candidates = []
        if error["type"] == "intent" and expected == ["buy_sell_timing"]:
            candidates = [
                query.replace("现在能买吗", "当前适合买入吗"),
                query.replace("适合买入吗", "这个位置可以上车吗"),
            ]
        elif error["type"] == "intent" and expected == ["hold_judgment"]:
            candidates = [
                query.replace("后面还值得拿吗", "后面还适合继续持有吗"),
                query.replace("适合继续持有吗", "从长期看还值得继续拿吗"),
            ]
        elif error["type"] == "intent" and expected == ["product_info"]:
            candidates = [
                query.replace("适合定投吗", "适合作为长期定投工具吗"),
                query.replace("适合长期定投吗", "更适合拿来做长期定投吗"),
            ]
        elif error["type"] == "intent" and expected == ["peer_compare"]:
            candidates = [
                query.replace("哪个好", "谁更适合配置"),
                query.replace("哪个更稳", "谁更有优势"),
            ]
        elif error["type"] == "topic" and expected == ["news", "price"]:
            candidates = [query.replace("为什么跌", "下跌的消息面原因是什么")]
        elif error["type"] == "topic" and expected == ["fundamentals", "risk"]:
            candidates = [query.replace("值得拿吗", "从基本面和风险看还适合持有吗")]
        elif error["type"] == "topic" and expected == ["product_mechanism"]:
            candidates = [query.replace("适合定投吗", "从产品机制看适不适合定投")]
        elif error["type"] == "topic" and expected == ["comparison", "product_mechanism"]:
            candidates = [query.replace("哪个好", "从产品结构看哪个更合适")]

        for candidate in candidates:
            if not candidate or candidate in existing_queries:
                continue
            target_intents = expected if error["type"] == "intent" else intents
            target_topics = expected if error["type"] == "topic" else topics
            additions.append(
                build_row(
                    base_query=query,
                    query=candidate,
                    product_type=product_type,
                    intents=target_intents,
                    topics=target_topics,
                    split=split,
                    expected_doc=expected_doc,
                    expected_struct=expected_struct,
                )
            )
            existing_queries.add(candidate)
    return additions


def build_augmented_qrels(rows: list[dict], documents: list[dict]) -> list[dict]:
    docs_by_id = {doc["evidence_id"]: doc for doc in documents}
    qrels: list[dict] = []
    for row in rows:
        expected_sources = [source for source in row["expected_document_sources"].split("|") if source]
        for evidence_id, doc in docs_by_id.items():
            if doc["source_type"] not in expected_sources:
                continue
            entity_symbols = set(doc.get("entity_symbols", []))
            relevance = 0
            if row.get("primary_symbol") and row["primary_symbol"] in entity_symbols:
                relevance = 2
            elif row.get("comparison_symbol") and row["comparison_symbol"] in entity_symbols:
                relevance = 1
            elif not entity_symbols and doc["source_type"] in {"faq", "product_doc", "research_note"}:
                relevance = 1
            elif doc["source_type"] == "research_note":
                relevance = 1
            if relevance:
                qrels.append({"query": row["query"], "evidence_id": evidence_id, "relevance": relevance})
    return qrels


def main() -> None:
    query_path = DATA_DIR / "query_labels.csv"
    qrels_path = DATA_DIR / "retrieval_qrels.csv"
    report_path = ROOT / "evaluation" / "error_analysis_report.json"
    docs_path = DATA_DIR / "documents.json"

    rows = load_rows(query_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    additions = augment_from_errors(rows, report)
    if additions:
        rows.extend(additions)
        write_rows(query_path, rows)

    documents = json.loads(docs_path.read_text(encoding="utf-8"))
    qrels_rows = build_augmented_qrels(rows, documents)
    write_rows(qrels_path, qrels_rows)
    print(json.dumps({"augmented_rows": len(additions), "total_rows": len(rows), "total_qrels": len(qrels_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
