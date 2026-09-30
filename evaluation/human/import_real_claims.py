"""Turn real market claims the owner collected (研报, 新闻, 微博, 雪球 …) into a held-out claim-check set.

Two steps, so the checker runs once and the labels are written without seeing its verdicts:

1. ``prepare``: reads the filled ``real_claims/claims.csv`` (a copy of ``template.csv``), runs FinSight's claim
   checker **once** on every claim (live sources by default, ``--offline`` for the snapshot), and writes

   * ``claims_real_v1.jsonl``: the claims in the claim benchmark's row format (``expected_verdict`` empty until
     step 2; ``expected_checks`` is not labelled for real claims);
   * ``finsight_run_v1.json``: the checker's verdict and checks for every claim, with commit and time;
   * ``labelling_sheet.csv``: one row per claim with the evidence FinSight retrieved (values, sources, dates)
     but **not** its verdict, and empty ``label`` / ``label_2`` columns for the owner and a second annotator.

2. ``score``: reads the filled labelling sheet, writes the labels into ``claims_real_v1.jsonl``
   (``expected_verdict``) and reports FinSight's agreement with them (accuracy with a Wilson 95% interval,
   Cohen's kappa, confusion matrix, coverage) and, when ``label_2`` is filled, the annotators' agreement.

    python -m evaluation.human.import_real_claims prepare
    python -m evaluation.human.import_real_claims score      # -> evaluation/results/real_claims-v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import (
    HUMAN_DIR,
    RESULTS_DIR,
    agreement,
    display,
    rate,
    read_csv,
    read_jsonl,
    run_config,
    sha256_file,
    write_csv,
    write_jsonl,
    write_result,
)

CLAIMS_DIR = HUMAN_DIR / "real_claims"
TEMPLATE_PATH = CLAIMS_DIR / "template.csv"
INPUT_PATH = CLAIMS_DIR / "claims.csv"
CLAIMS_OUT = CLAIMS_DIR / "claims_real_v1.jsonl"
RUN_OUT = CLAIMS_DIR / "finsight_run_v1.json"
SHEET_PATH = CLAIMS_DIR / "labelling_sheet.csv"
OUT_PATH = RESULTS_DIR / "real_claims-v1.json"
TEMPLATE_COLUMNS = ("id", "claim_text", "source_type", "source_url_or_name", "date_seen", "notes")
SHEET_COLUMNS = ("id", "claim_text", "source_type", "date_seen", "finsight_evidence", "label", "label_2", "notes")
VERDICTS = ("supported", "contradicted", "partially_supported", "unverifiable")
_LABEL_ALIASES = {
    "supported": "supported",
    "支持": "supported",
    "成立": "supported",
    "属实": "supported",
    "contradicted": "contradicted",
    "矛盾": "contradicted",
    "不成立": "contradicted",
    "错误": "contradicted",
    "不属实": "contradicted",
    "partially_supported": "partially_supported",
    "partial": "partially_supported",
    "部分支持": "partially_supported",
    "部分成立": "partially_supported",
    "unverifiable": "unverifiable",
    "无法核实": "unverifiable",
    "无法验证": "unverifiable",
    "不可核实": "unverifiable",
}


def normalise_label(value: str) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text not in _LABEL_ALIASES:
        raise ValueError(f"unknown label {value!r}; use one of {', '.join(VERDICTS)} (或 支持/矛盾/部分支持/无法核实)")
    return _LABEL_ALIASES[text]


# ------------------------------------------------------------------------------------------- prepare
def claim_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Validated input rows: ids unique and non-empty (``R001``… assigned when blank), claim text present."""
    seen, result = set(), []
    for index, row in enumerate(rows, start=1):
        text = row.get("claim_text", "").strip()
        if not text:
            continue
        claim_id = row.get("id", "").strip() or f"R{index:03d}"
        if claim_id in seen:
            raise SystemExit(f"duplicate id {claim_id}")
        seen.add(claim_id)
        result.append({**row, "id": claim_id, "claim_text": text})
    return result


def bench_row(row: dict[str, Any], lang: str) -> dict[str, Any]:
    """A row in the claim benchmark's format (``evaluation/claim_bench/README.md``)."""
    return {
        "id": row["id"],
        "lang": lang,
        "category": "real",
        "claim": row["claim_text"],
        "expected_verdict": None,
        "expected_checks": [],
        "basis": "labelled by the owner from the evidence sheet and public data (import_real_claims score)",
        "source_type": row.get("source_type", ""),
        "source": row.get("source_url_or_name", ""),
        "date_seen": row.get("date_seen", ""),
        "notes": row.get("notes", ""),
    }


_CHECK_KEYS = (
    "target",
    "metric",
    "claimed",
    "claimed_high",
    "claimed_unit",
    "comparator",
    "direction",
    "reference",
    "actual",
    "reference_value",
    "status",
    "reason",
    "evidence_id",
    "source",
    "as_of",
    "note",
)


def run_record(claim_id: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": claim_id,
        "verdict": report["verdict"],
        "checks": [{key: check.get(key) for key in _CHECK_KEYS} for check in report.get("checks") or []],
        "unchecked": [item.get("text") for item in report.get("unchecked") or []],
        "targets": report.get("targets") or [],
        "evidence_sources": report.get("evidence_sources") or [],
    }


_COMPARATOR_SIGN = {"eq": "=", "ne": "≠", "gt": ">", "ge": "≥", "lt": "<", "le": "≤", "approx": "≈", "range": "区间"}


def evidence_text(record: dict[str, Any]) -> str:
    """What FinSight retrieved for a claim, without its verdict or per-number status."""
    lines = []
    for check in record["checks"]:
        subject = " ".join(str(part) for part in (check.get("target"), check.get("metric")) if part) or "（未识别对象）"
        sign = _COMPARATOR_SIGN.get(str(check.get("comparator") or "eq"), str(check.get("comparator")))
        if check.get("claimed") is None:
            claimed = f"声明: 与 {check['reference']} 比较（{sign}）" if check.get("reference") else "声明: 涨跌方向"
        else:
            high = f"~{check['claimed_high']}" if check.get("claimed_high") is not None else ""
            claimed = f"声明 {sign} {check['claimed']}{high}{check.get('claimed_unit') or ''}"
        if check.get("actual") is not None:
            data = f"数据值 {check['actual']}"
            if check.get("reference") and check.get("reference_value") is not None:
                data += f"（对比 {check['reference']}: {check['reference_value']}）"
            where = "，".join(str(part) for part in (check.get("source"), check.get("as_of")) if part)
            lines.append(f"{subject}: {claimed} ↔ {data}" + (f"（{where}）" if where else ""))
        else:
            lines.append(f"{subject}: {claimed} ↔ 未取到可比数据（{check.get('reason') or '无'}）")
    for text in record["unchecked"]:
        lines.append(f"未核对的片段: {text}")
    if not lines:
        lines.append("FinSight 没有从这条声明中读出可核对的数字或比较")
    sources = [
        f"[{item.get('evidence_id')}] {item.get('source_name') or ''} 截至 {item.get('as_of') or '?'}".strip()
        for item in record["evidence_sources"]
    ]
    if sources:
        lines.append("证据: " + "; ".join(dict.fromkeys(sources)))
    return "\n".join(lines)


def prepare(rows: list[dict[str, Any]], *, service: Any, registry: Any) -> tuple[list[dict], list[dict], list[dict]]:
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.chatbot import detect_query_language

    bench, records, sheet = [], [], []
    for row in rows:
        lang = detect_query_language(row["claim_text"])
        report = check_claim(row["claim_text"], service=service, registry=registry, zh=lang == "zh").model_dump()
        record = run_record(row["id"], report)
        bench.append(bench_row(row, lang))
        records.append(record)
        sheet.append(
            {
                "id": row["id"],
                "claim_text": row["claim_text"],
                "source_type": row.get("source_type", ""),
                "date_seen": row.get("date_seen", ""),
                "finsight_evidence": evidence_text(record),
            }
        )
    return bench, records, sheet


# --------------------------------------------------------------------------------------------- score
def score(sheet: list[dict[str, str]], records: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = {record["id"]: record["verdict"] for record in records}
    labels, second, problems = {}, {}, []
    for row in sheet:
        try:
            first = normalise_label(row.get("label", ""))
            other = normalise_label(row.get("label_2", ""))
        except ValueError as exc:
            problems.append(f"{row.get('id')}: {exc}")
            continue
        if first:
            labels[row["id"]] = first
        if other:
            second[row["id"]] = other
    if problems:
        raise SystemExit("invalid labels:\n  " + "\n  ".join(problems))
    ids = [claim_id for claim_id in labels if claim_id in predicted]
    human = [labels[claim_id] for claim_id in ids]
    finsight = [predicted[claim_id] for claim_id in ids]
    result: dict[str, Any] = {
        "claims": len(records),
        "labelled": len(ids),
        "label_distribution": {verdict: human.count(verdict) for verdict in VERDICTS},
        "finsight_verdict_distribution": {verdict: finsight.count(verdict) for verdict in VERDICTS},
        "verdict_accuracy": rate(sum(h == f for h, f in zip(human, finsight, strict=True)), len(ids)),
        "agreement": agreement(human, finsight, VERDICTS, ("human label", "FinSight verdict")),
    }
    checkable = [claim_id for claim_id in ids if labels[claim_id] != "unverifiable"]
    result["coverage"] = {
        "note": "claims the annotator could verify: share FinSight also checked (verdict not unverifiable)",
        **rate(sum(predicted[claim_id] != "unverifiable" for claim_id in checkable), len(checkable)),
    }
    both = [claim_id for claim_id in ids if claim_id in second]
    if both:
        result["inter_annotator"] = agreement(
            [labels[c] for c in both], [second[c] for c in both], VERDICTS, ("label", "label_2")
        )
    result["disagreements"] = [
        {"id": claim_id, "label": labels[claim_id], "finsight": predicted[claim_id]}
        for claim_id in ids
        if labels[claim_id] != predicted[claim_id]
    ]
    return result


def apply_labels(bench: list[dict[str, Any]], sheet: list[dict[str, str]]) -> list[dict[str, Any]]:
    labels = {row["id"]: normalise_label(row.get("label", "")) for row in sheet}
    return [{**row, "expected_verdict": labels.get(row["id"]) or row.get("expected_verdict")} for row in bench]


def _build_service(offline: bool) -> Any:
    from query_intelligence.service import build_default_service

    live = not offline
    return build_default_service(
        use_live_market=live, use_live_macro=live, use_live_news=False, use_live_announcement=False
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Import real claims, run the claim checker once, score labels.")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Run the checker once and write the labelling sheet.")
    prep.add_argument("--input", default=str(INPUT_PATH))
    prep.add_argument("--offline", action="store_true", help="Use the offline snapshot instead of live sources.")
    prep.add_argument("--force", action="store_true", help="Overwrite an existing run (it is meant to run once).")
    scr = sub.add_parser("score", help="Score the filled labelling sheet.")
    scr.add_argument("--sheet", default=str(SHEET_PATH))
    scr.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)

    if args.command == "prepare":
        if RUN_OUT.exists() and not args.force:
            raise SystemExit(f"{display(RUN_OUT)} exists: the checker runs once per claim set (use --force)")
        input_path = Path(args.input)
        if not input_path.exists():
            raise SystemExit(f"{display(input_path)} not found: copy template.csv to claims.csv and fill it")
        rows = claim_rows(read_csv(input_path))
        if not rows:
            raise SystemExit(f"no claims in {display(input_path)}")
        from query_intelligence.agent.tools import build_registry_for_service

        service = _build_service(args.offline)
        bench, records, sheet = prepare(rows, service=service, registry=build_registry_for_service(service))
        write_jsonl(CLAIMS_OUT, bench)
        config = run_config(
            "evaluation.human.import_real_claims",
            argv,
            input=display(input_path),
            input_sha256=sha256_file(input_path),
            data="offline snapshot" if args.offline else "live market and macro sources",
            model=None,
        )
        write_result(RUN_OUT, {"config": config, "claims_sha256": sha256_file(CLAIMS_OUT), "results": records})
        write_csv(SHEET_PATH, SHEET_COLUMNS, sheet)
        print(f"checked {len(records)} claims; fill {display(SHEET_PATH)} (label, optional label_2), then run score")
        return {"config": config, "claims": len(records)}

    run = json.loads(RUN_OUT.read_text(encoding="utf-8"))
    sheet = read_csv(args.sheet)
    body = score(sheet, run["results"])
    write_jsonl(CLAIMS_OUT, apply_labels(read_jsonl(CLAIMS_OUT), sheet))
    report = {
        "config": run_config(
            "evaluation.human.import_real_claims",
            argv,
            checker_run=run["config"],
            sheet=display(args.sheet),
            sheet_sha256=sha256_file(args.sheet),
            claims=display(CLAIMS_OUT),
            claims_sha256=sha256_file(CLAIMS_OUT),
            ci_method="Wilson score 95% for rates; percentile bootstrap (2000, seed 20260930) for kappa",
        ),
        **body,
    }
    out = write_result(args.out, report)
    accuracy = body["verdict_accuracy"]
    print(
        f"labelled {body['labelled']}/{body['claims']}; verdict accuracy {accuracy['rate']} {accuracy['ci95_wilson']}; "
        f"kappa {body['agreement']['cohen_kappa']} -> {display(out)}"
    )
    return report


if __name__ == "__main__":
    main()
