from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.config import Settings
from query_intelligence.nlu.pipeline import NLUPipeline
from scripts.evaluate_query_intelligence import build_master_eval_set


REPORT_DIR = ROOT / "reports"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    classification_rows = [json.loads(line) for line in (ROOT / "data/training_assets/classification.jsonl").open() if line.strip()]
    qrel_rows = [json.loads(line) for line in (ROOT / "data/training_assets/qrels.jsonl").open() if line.strip()]
    corpus_rows = [json.loads(line) for line in (ROOT / "data/training_assets/retrieval_corpus.jsonl").open() if line.strip()]
    items = build_master_eval_set(classification_rows, qrel_rows, corpus_rows)
    ood_items = [item for item in items if item.domain == "ood"]

    pipeline = NLUPipeline.build_default(Settings(models_dir=str(ROOT / "models")))
    failures: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    product_counts: Counter[str] = Counter()
    for index, item in enumerate(ood_items, start=1):
        out = pipeline.run(item.query, {}, [], False)
        if "out_of_scope_query" not in out.get("risk_flags", []):
            category = classify_failure(item.query, item.bucket, item.language)
            bucket_counts[item.bucket] += 1
            category_counts[category] += 1
            product_counts[str((out.get("product_type") or {}).get("label") or "")] += 1
            failures.append(
                {
                    "query": item.query,
                    "bucket": item.bucket,
                    "language": item.language,
                    "category": category,
                    "product_type": out.get("product_type", {}),
                    "question_style": out.get("question_style"),
                    "intent_labels": [x["label"] for x in out.get("intent_labels", [])],
                    "topic_labels": [x["label"] for x in out.get("topic_labels", [])],
                    "risk_flags": out.get("risk_flags", []),
                    "missing_slots": out.get("missing_slots", []),
                    "source_plan": out.get("source_plan", []),
                    "metadata": item.metadata or {},
                }
            )
        if index % 500 == 0:
            print(json.dumps({"ood_checked": index, "total": len(ood_items), "failures": len(failures)}, ensure_ascii=False), flush=True)

    summary = {
        "ood_total": len(ood_items),
        "ood_failures": len(failures),
        "ood_accuracy": round(1 - len(failures) / max(len(ood_items), 1), 4),
        "bucket_counts": dict(bucket_counts),
        "category_counts": dict(category_counts),
        "predicted_product_type_counts": dict(product_counts),
    }
    raw_path = REPORT_DIR / "2026-04-23-ood-false-negatives.raw.json"
    analysis_path = REPORT_DIR / "2026-04-23-ood-false-negatives.analysis.json"
    markdown_path = REPORT_DIR / "2026-04-23-ood-false-negatives.analysis.md"
    raw_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    analysis_path.write_text(json.dumps({"summary": summary, "sample_failures": failures[:100]}, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary, failures[:80]), encoding="utf-8")
    print(json.dumps({"raw": str(raw_path), "analysis": str(analysis_path), "markdown": str(markdown_path)}, ensure_ascii=False), flush=True)


def classify_failure(query: str, bucket: str, language: str) -> str:
    lowered = query.lower()
    if bucket == "ood_dialogue_public":
        if len(query.strip()) <= 12:
            return "short_dialogue_followup"
        return "multi_turn_dialogue_utterance"
    if re.fullmatch(r"[a-z\\s]{6,}", lowered) and len(set(lowered.split())) <= 2:
        return "gibberish_ascii"
    if any(term in lowered for term in ["computer", "laptop", "python", "sql", "frontend", "program", "戴尔", "电脑", "笔记本"]):
        return "tech_or_coding"
    if any(term in lowered for term in ["hospital", "doctor", "disease", "fever", "cough", "医院", "病情", "孩子", "胸痛"]):
        return "medical_or_hospital"
    if any(term in lowered for term in ["hotel", "restaurant", "museum", "park", "scenic", "景点", "餐厅", "酒店", "苏州", "游乐场", "古镇"]):
        return "travel_food_hotel"
    if any(term in lowered for term in ["car", "suv", "mpv", "大众", "汽车", "三厢车", "蔚揽", "探岳"]):
        return "car_purchase"
    if any(term in lowered for term in ["phone", "call", "price", "how much", "人均消费", "电话", "预算", "多少", "几星级"]):
        return "slot_followup_or_pricing"
    if language == "en":
        return "open_domain_english_qa"
    return "open_domain_chinese_qa"


def render_markdown(summary: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    lines = [
        "# OOD False-Negative Analysis",
        "",
        f"- OOD total: `{summary['ood_total']}`",
        f"- OOD failures: `{summary['ood_failures']}`",
        f"- OOD accuracy: `{summary['ood_accuracy']}`",
        f"- Buckets: `{json.dumps(summary['bucket_counts'], ensure_ascii=False)}`",
        f"- Categories: `{json.dumps(summary['category_counts'], ensure_ascii=False)}`",
        "",
        "## Sample Failures",
        "",
    ]
    for item in failures:
        lines.append(f"- [{item['category']}] `{item['query']}` -> product={item['product_type']}, style={item['question_style']}, intents={item['intent_labels']}, topics={item['topic_labels']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
