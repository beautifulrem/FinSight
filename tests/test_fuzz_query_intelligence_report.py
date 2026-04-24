from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from evaluation.fuzz_query_intelligence_report import build_fuzz_report


ROOT = Path(__file__).resolve().parents[1]


def load_schema(name: str) -> dict:
    path = ROOT / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_fuzz_report_returns_scored_schema_valid_matrix() -> None:
    report = build_fuzz_report()

    assert report["summary"]["total_cases"] >= 25
    assert 0.0 <= report["summary"]["overall_score"] <= 10.0
    assert 0.0 <= report["summary"]["nlu_score"] <= 10.0
    assert 0.0 <= report["summary"]["retrieval_score"] <= 10.0
    assert report["summary"]["passed_cases"] + report["summary"]["failed_cases"] == report["summary"]["total_cases"]
    assert isinstance(report["defect_summary"], list)
    assert report["recommendations"]

    case_ids = {case["id"] for case in report["cases"]}
    assert {
        "colloquial_stock_followup",
        "generic_bucket_index",
        "resolved_stock_comparison",
        "followup_with_dialog_context",
        "lowercase_etf_comparison",
        "stock_fundamental_colloquial",
        "generic_buy_clarification",
        "english_stock_advice",
        "english_announcement_query",
        "english_etf_comparison",
    } <= case_ids

    nlu_schema = load_schema("nlu_result.schema.json")
    retrieval_schema = load_schema("retrieval_result.schema.json")
    for case in report["cases"]:
        jsonschema.validate(case["actual"]["nlu_result"], nlu_schema)
        jsonschema.validate(case["actual"]["retrieval_result"], retrieval_schema)
        assert case["check_results"]
