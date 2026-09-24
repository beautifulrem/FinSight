"""Fuzz matrix for Query Intelligence.

Case definitions (queries, optional dialog context / user profile, and expected checks) live in
``evaluation/fuzz_cases.jsonl`` and are committed. ``build_fuzz_report()`` runs every case through a
Query Intelligence service with live providers disabled, validates both artifacts against the
JSON schemas, evaluates the checks, and returns a scored report. Pass ``report_path`` to load a
previously generated report instead.

    python -m evaluation.fuzz_query_intelligence_report   # writes evaluation/fuzz_query_intelligence_report.{json,md}
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = ROOT / "evaluation" / "fuzz_query_intelligence_report.json"
DEFAULT_CASES_PATH = ROOT / "evaluation" / "fuzz_cases.jsonl"
SCHEMA_DIR = ROOT / "schemas"


def load_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_fuzz_report(
    report_path: str | Path | None = None,
    *,
    service: Any = None,
    cases_path: str | Path = DEFAULT_CASES_PATH,
) -> dict[str, Any]:
    if report_path is not None:
        return deepcopy(json.loads(Path(report_path).read_text(encoding="utf-8")))

    if service is None:
        from query_intelligence.service import build_default_service

        service = build_default_service(
            use_live_market=False,
            use_live_macro=False,
            use_live_news=False,
            use_live_announcement=False,
        )
    nlu_schema = _load_schema("nlu_result.schema.json")
    retrieval_schema = _load_schema("retrieval_result.schema.json")

    cases = []
    for case in load_cases(cases_path):
        result = service.run_pipeline(
            case["query"],
            user_profile=case.get("user_profile") or {},
            dialog_context=case.get("dialog_context") or [],
            top_k=10,
        )
        nlu, retrieval = result["nlu_result"], result["retrieval_result"]
        checks = [
            _schema_check("nlu", nlu, nlu_schema),
            _schema_check("retrieval", retrieval, retrieval_schema),
            *[_evaluate(check, nlu, retrieval) for check in case.get("checks") or []],
        ]
        failed = [check for check in checks if not check["passed"]]
        cases.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                "description": case.get("description"),
                "query": case["query"],
                "status": "passed" if not failed else "failed",
                "passed_checks": len(checks) - len(failed),
                "total_checks": len(checks),
                "check_results": checks,
                "failed_checks": failed,
                "actual": {"nlu_result": nlu, "retrieval_result": retrieval},
            }
        )
    return {
        "summary": _summary(cases),
        "cases": cases,
        "defect_summary": _defects(cases),
        "recommendations": _recommendations(cases),
    }


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _schema_check(scope: str, artifact: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    try:
        jsonschema.validate(artifact, schema)
    except jsonschema.ValidationError as exc:
        return {"scope": scope, "name": "schema", "passed": False, "expected": "schema-valid", "actual": exc.message}
    return {"scope": scope, "name": "schema", "passed": True, "expected": "schema-valid", "actual": "schema-valid"}


def _evaluate(check: dict[str, Any], nlu: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
    scope, name, expected = check["scope"], check["name"], check["expected"]
    actual = _actual(scope, name, nlu, retrieval)
    if name.endswith("_contains"):
        passed = set(expected) <= set(actual or [])
    elif name.endswith("_excludes"):
        passed = not set(expected) & set(actual or [])
        expected = f"exclude {expected}"
    elif name in {"entity_symbols", "comparison_targets", "source_plan"} or name == "warnings":
        passed = sorted(actual or []) == sorted(expected) if name != "source_plan" else list(actual or []) == list(expected)
    else:
        passed = actual == expected
    return {"scope": scope, "name": name, "passed": bool(passed), "expected": expected, "actual": actual}


def _actual(scope: str, name: str, nlu: dict[str, Any], retrieval: dict[str, Any]) -> Any:
    if scope == "nlu":
        base = name.removesuffix("_contains").removesuffix("_excludes")
        if base == "product_type_label":
            return (nlu.get("product_type") or {}).get("label")
        if base == "entity_symbols":
            return [entity.get("symbol") for entity in nlu.get("entities") or [] if entity.get("symbol")]
        return nlu.get(base)
    if name.startswith("coverage."):
        return (retrieval.get("coverage") or {}).get(name.split(".", 1)[1])
    if name == "documents_empty":
        return not retrieval.get("documents")
    if name == "structured_empty":
        return not retrieval.get("structured_data")
    base = name.removesuffix("_contains").removesuffix("_excludes")
    return retrieval.get(base)


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [check for case in cases for check in case["check_results"]]

    def score(selected: list[dict[str, Any]]) -> float:
        return round(10.0 * sum(check["passed"] for check in selected) / len(selected), 2) if selected else 10.0

    passed_cases = sum(case["status"] == "passed" for case in cases)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "offline (all live providers disabled)",
        "total_cases": len(cases),
        "passed_cases": passed_cases,
        "failed_cases": len(cases) - passed_cases,
        "passed_checks": sum(check["passed"] for check in checks),
        "total_checks": len(checks),
        "schema_score": score([check for check in checks if check["name"] == "schema"]),
        "nlu_score": score([check for check in checks if check["scope"] == "nlu" and check["name"] != "schema"]),
        "retrieval_score": score(
            [check for check in checks if check["scope"] == "retrieval" and check["name"] != "schema"]
        ),
        "overall_score": score(checks),
    }


def _defects(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for case in cases:
        for check in case["failed_checks"]:
            grouped.setdefault((check["scope"], check["name"]), []).append(case["id"])
    return [
        {"scope": scope, "check": name, "failed_cases": ids, "count": len(ids)}
        for (scope, name), ids in sorted(grouped.items(), key=lambda item: -len(item[1]))
    ]


def _recommendations(cases: list[dict[str, Any]]) -> list[str]:
    defects = _defects(cases)
    if not defects:
        return ["当前 fuzz 矩阵未暴露明显缺陷，可以继续扩口语、错别字和多轮对话样本。"]
    recommendations = [
        f"{defect['scope']}.{defect['check']} 在 {defect['count']} 个用例失败：{', '.join(defect['failed_cases'][:5])}"
        for defect in defects[:5]
    ]
    if any(defect["scope"] == "retrieval" for defect in defects):
        recommendations.append("离线模式下 live provider 被禁用，部分 retrieval 覆盖/告警检查依赖实时数据，需结合 live 运行复核。")
    return recommendations


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Query Intelligence Fuzz Report",
        "",
        f"- Generated at: {summary['generated_at']} ({summary.get('mode', '')})",
        f"- Cases: {summary['passed_cases']}/{summary['total_cases']} passed; "
        f"checks {summary['passed_checks']}/{summary['total_checks']}",
        f"- Scores: overall {summary['overall_score']}, NLU {summary['nlu_score']}, "
        f"retrieval {summary['retrieval_score']}, schema {summary['schema_score']}",
        "",
        "| Case | Status | Checks |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {case['id']} | {case['status']} | {case['passed_checks']}/{case['total_checks']} |" for case in report["cases"]
    )
    lines += ["", "## Recommendations", "", *[f"- {item}" for item in report["recommendations"]], ""]
    return "\n".join(lines)


def main() -> None:
    report = build_fuzz_report()
    DEFAULT_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    DEFAULT_REPORT_PATH.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
