"""Turn real market claims the owner collected (研报, 新闻, 微博, 雪球 …) into a held-out claim-check set.

Two steps, so the checker runs once and the labels are written without seeing its verdict or its reading:

1. ``prepare``: reads the filled ``real_claims/claims.csv`` (a copy of ``template.csv``), runs FinSight's claim
   checker **once** on every claim (live sources by default, ``--offline`` for the offline data), and writes
   two things kept apart:

   * for the **score step only** (annotators do not open these):
     ``finsight_run_v1.json``: FinSight's verdict and its reading of every claim (target, metric, claimed
     number, comparator, per-number status), with commit and time; and ``claims_real_v1.jsonl``: the claims in
     the claim benchmark's row format (``expected_verdict`` empty until step 2);
   * for the **annotators**: ``evidence_v1.jsonl``: the raw records the checker's tools returned for each claim;
     and ``labelling_sheet.csv``: one row per claim with only the claim text, the date it was seen and those
     raw values (every numeric field with unit, source and date; 1688.38亿元, not 168838000000). It is built
     from ``evidence_v1.jsonl`` alone (``annotator_sheet``), so nothing of FinSight's reading can leak into
     it: no metric it picked, no comparator, no claimed-vs-actual pairing, no status or reason code. Empty
     ``label`` / ``label_2`` columns are for the owner and a second annotator.

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
EVIDENCE_OUT = CLAIMS_DIR / "evidence_v1.jsonl"
SHEET_PATH = CLAIMS_DIR / "labelling_sheet.csv"
OUT_PATH = RESULTS_DIR / "real_claims-v1.json"
TEMPLATE_COLUMNS = ("id", "claim_text", "source_type", "source_url_or_name", "date_seen", "notes")
SHEET_COLUMNS = ("id", "claim_text", "date_seen", "evidence", "label", "label_2", "notes")
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


# ------------------------------------------------------------------ what the annotator sees (raw evidence)
# Plain names for the raw payload fields of the retrieved records. These are the data's own fields (every field
# of every record the checker fetched), not FinSight's reading of the claim: which metric it thought the claim
# is about, the comparator, the claimed number and the per-number status stay in ``finsight_run_v1.json``.
_FIELD_NAMES = {
    "close": "收盘价",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "pct_change_1d": "日涨跌幅",
    "pct_change": "涨跌幅",
    "amount": "成交额",
    "volume": "成交量",
    "turnover": "换手率",
    "turnover_rate": "换手率",
    "revenue": "营业收入",
    "net_profit": "净利润",
    "roe": "ROE",
    "roa": "ROA",
    "gross_margin": "毛利率",
    "grossprofit_margin": "毛利率",
    "net_margin": "净利率",
    "netprofit_margin": "净利率",
    "netprofit_yoy": "净利润同比",
    "revenue_yoy": "营收同比",
    "pe_ttm": "市盈率(TTM)",
    "pe": "市盈率",
    "pb": "市净率",
    "ps": "市销率",
    "dividend_yield": "股息率",
    "eps": "每股收益",
    "bps": "每股净资产",
    "metric_value": "数值",
}
_DATE_FIELDS = ("as_of", "trade_date", "report_date", "metric_date", "period")
_SKIP_FIELDS = {
    "symbol",
    "name",
    "product_type",
    "source",
    "source_name",
    "evidence_id",
    "provenance",
    "units_source",
    "units_inferred",
    "metric_units",
    "amount_unit",
    "volume_unit",
    "change_unit",
    "unit",
    "indicator_code",
    "industry_name",
    "price_basis",
    "recent_closes",
    *_DATE_FIELDS,
}


def _plain_number(value: float) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".").replace(",", "")


def _scaled(value: float, unit: str) -> str:
    """``168838000000`` CNY -> ``1688.38亿元``; shares -> ``万股``/``亿股``."""
    for size, prefix in ((1e8, "亿"), (1e4, "万")):
        if abs(value) >= size:
            return f"{_plain_number(round(value / size, 2))}{prefix}{unit}"
    return f"{_plain_number(value)}{unit}"


def _value_text(key: str, value: float, payload: dict[str, Any]) -> str:
    from query_intelligence.agent.tools.units import metric_unit

    units = payload.get("metric_units") if isinstance(payload.get("metric_units"), dict) else {}
    unit = units.get(key) or (payload.get("unit") if key == "metric_value" else None)
    if key == "amount" or unit == "CNY":
        return _scaled(value, "元")
    if key == "volume":
        return _scaled(value, "股" if payload.get("volume_unit") == "share" else "（单位未知）")
    if key in {"close", "open", "high", "low"}:
        return f"{_plain_number(value)}元"
    unit = unit or ("%" if key == "pct_change_1d" else metric_unit(key))
    suffix = {"%": "%", "x": "倍", "CNY/share": "元/股"}.get(str(unit), str(unit or ""))
    return f"{_plain_number(value)}{suffix}"


def evidence_line(item: dict[str, Any]) -> str:
    """One retrieved record as the annotator sees it: what it is, source, date and every numeric value."""
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    title = item.get("title") or payload.get("name") or payload.get("indicator_code") or item.get("evidence_id")
    sources = [
        provenance.get("original_source") or payload.get("source_name") or payload.get("source"),
        item.get("source_name"),
        provenance.get("source_label"),
    ]
    source = " / ".join(dict.fromkeys(str(name) for name in sources if name))
    date = next((str(payload[key]) for key in _DATE_FIELDS if payload.get(key)), None) or item.get("as_of")
    values = []
    for key, value in payload.items():
        if key in _SKIP_FIELDS or isinstance(value, bool) or not isinstance(value, int | float):
            continue
        values.append(f"{_FIELD_NAMES.get(key, key)} {_value_text(key, float(value), payload)}")
    closes = payload.get("recent_closes")
    if isinstance(closes, list) and len(closes) > 1:
        values.append(
            "近期收盘 " + "，".join(f"{row.get('date')} {row.get('close')}" for row in closes if isinstance(row, dict))
        )
    head = f"{title}（来源: {source or '未注明'}；日期: {date or '未注明'}）"
    return f"{head}: {'；'.join(values) if values else '（无数值）'}"


def evidence_text(evidence: list[dict[str, Any]]) -> str:
    """The raw evidence values of one claim, one record per line, with no reference to FinSight's reading."""
    lines = [evidence_line(item) for item in evidence]
    return "\n".join(lines) or "（没有取到任何数据；请只凭公开资料判断）"


class _EvidenceRecorder:
    """Wraps the tool registry during one claim check and keeps every evidence record the tools returned."""

    def __init__(self, registry: Any) -> None:
        self.registry = registry
        self.items: dict[str, dict[str, Any]] = {}

    def run(self, name: str, arguments: Any) -> Any:
        result = self.registry.run(name, arguments)
        for item in getattr(result, "evidence", None) or []:
            dumped = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            self.items.setdefault(str(dumped.get("evidence_id")), dumped)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.registry, name)


def run_checker(
    rows: list[dict[str, Any]], *, service: Any, registry: Any
) -> tuple[list[dict], list[dict], list[dict]]:
    """Step 1a: run the checker once per claim. Returns the bench rows, FinSight's reading and verdict
    (``records``, for the score step only) and the raw evidence each check retrieved (``evidence``)."""
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.chatbot import detect_query_language

    bench, records, evidence = [], [], []
    for row in rows:
        lang = detect_query_language(row["claim_text"])
        recorder = _EvidenceRecorder(registry)
        report = check_claim(row["claim_text"], service=service, registry=recorder, zh=lang == "zh").model_dump()
        bench.append(bench_row(row, lang))
        records.append(run_record(row["id"], report))
        evidence.append({"id": row["id"], "evidence": list(recorder.items.values())})
    return bench, records, evidence


def annotator_sheet(rows: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Step 1b: the labelling sheet from the claim rows and the raw evidence only (never from ``records``)."""
    by_id = {entry["id"]: entry["evidence"] for entry in evidence}
    return [
        {
            "id": row["id"],
            "claim_text": row["claim_text"],
            "date_seen": row.get("date_seen", ""),
            "evidence": evidence_text(by_id.get(row["id"], [])),
        }
        for row in rows
    ]


def prepare(rows: list[dict[str, Any]], *, service: Any, registry: Any) -> tuple[list[dict], list[dict], list[dict]]:
    bench, records, evidence = run_checker(rows, service=service, registry=registry)
    return bench, records, annotator_sheet(rows, evidence)


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
    prep.add_argument("--offline", action="store_true", help="Use the offline data instead of live sources.")
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
        bench, records, evidence = run_checker(rows, service=service, registry=build_registry_for_service(service))
        write_jsonl(CLAIMS_OUT, bench)
        write_jsonl(EVIDENCE_OUT, evidence)
        sheet = annotator_sheet(rows, evidence)
        config = run_config(
            "evaluation.human.import_real_claims",
            argv,
            input=display(input_path),
            input_sha256=sha256_file(input_path),
            data="offline data" if args.offline else "live market and macro sources",
            model=None,
        )
        write_result(
            RUN_OUT,
            {
                "config": config,
                "note": "FinSight's verdicts and readings: used by the score step only; not for annotators",
                "claims_sha256": sha256_file(CLAIMS_OUT),
                "evidence_sha256": sha256_file(EVIDENCE_OUT),
                "results": records,
            },
        )
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
