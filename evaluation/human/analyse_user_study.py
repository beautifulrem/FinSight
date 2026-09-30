"""Analyse the small user study: in-app thumbs up/down and the SUS questionnaire.

Inputs:

* the feedback file the server appends to from ``POST /agent/feedback`` (``QI_FEEDBACK_PATH``; the study guide
  sets it to ``outputs/user_study/feedback.jsonl``, one JSON object per click: ``trace_id``, ``rating``,
  ``comment``, ``query``, ``route``, ``owner``, ``at``). A trace rated more than once counts once, with its
  last rating.
* ``user_study/questionnaire.csv``: a filled copy of ``questionnaire_template.csv`` (SUS items ``q1``–``q10``
  on a 1–5 scale, plus tasks completed and free-text comments).

Output ``evaluation/results/user_study-v1.json``: thumbs-up ratio with a Wilson 95% interval, SUS score
(mean, SD, bootstrap 95% CI, per item), tasks completed, and the most frequent complaints (thumbs-down
comments and "most disliked" answers grouped by keyword, with examples).

    python -m evaluation.human.analyse_user_study
    python -m evaluation.human.analyse_user_study --feedback path/to/feedback.jsonl --since 2026-10-10T00:00
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any

from ..agent_eval.runner import ROOT
from .common import HUMAN_DIR, RESULTS_DIR, display, rate, read_csv, run_config, sha256_file, write_result

STUDY_DIR = HUMAN_DIR / "user_study"
FEEDBACK_PATH = ROOT / "outputs" / "user_study" / "feedback.jsonl"
QUESTIONNAIRE_PATH = STUDY_DIR / "questionnaire.csv"
OUT_PATH = RESULTS_DIR / "user_study-v1.json"
SUS_BENCHMARK = 68.0  # average SUS score across published studies (Sauro & Lewis)
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_RESAMPLES = 2000
COMPLAINT_CATEGORIES: dict[str, str] = {
    "速度慢": r"慢|卡|等(?:待|了)|太久|延迟|转圈|slow",
    "数据旧或不准": r"数据|过时|旧|不准|不对|错误|有误|算错|错了|过期|不是最新",
    "不回答或拒答": r"拒绝|不回答|答不了|无法回答|不支持|没回答|回避|不给",
    "看不懂或太长": r"看不懂|太长|术语|啰嗦|复杂|难懂|太多|冗长|专业",
    "界面操作": r"界面|按钮|找不到|布局|手机|显示|点击|字体|颜色",
    "没有结论或建议": r"没有结论|没用|结论|建议|到底|说了等于没说|模糊",
}


# ------------------------------------------------------------------------------------------- feedback
def load_feedback(path: Path, *, since: str = "") -> list[dict[str, Any]]:
    """One record per trace (its last rating), optionally only clicks at or after ``since`` (ISO, UTC)."""
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if since and str(record.get("at", "")) < since:
            continue
        latest[str(record.get("trace_id"))] = record
    return list(latest.values())


def feedback_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ups = sum(1 for record in records if record.get("rating") == "up")
    downs = sum(1 for record in records if record.get("rating") == "down")
    by_route: dict[str, list[int]] = {}
    for record in records:
        if record.get("rating") in {"up", "down"}:
            by_route.setdefault(str(record.get("route")), []).append(int(record["rating"] == "up"))
    return {
        "rated_answers": ups + downs,
        "browsers": len({record.get("owner") for record in records}),
        "thumbs_up": ups,
        "thumbs_down": downs,
        "thumbs_up_ratio": rate(ups, ups + downs),
        "by_route": {route: rate(sum(values), len(values)) for route, values in sorted(by_route.items())},
        "with_comment": sum(1 for record in records if str(record.get("comment") or "").strip()),
    }


# ------------------------------------------------------------------------------------------------ SUS
def _item_columns(header: list[str]) -> dict[int, str]:
    columns = {}
    for name in header:
        match = re.match(r"\s*q(\d{1,2})(?!\d)", name, re.I)
        if match and 1 <= int(match.group(1)) <= 10:
            columns[int(match.group(1))] = name
    return columns


def sus_score(answers: dict[int, int]) -> float:
    """Standard SUS: odd items contribute (x - 1), even items (5 - x); the sum times 2.5 gives 0–100."""
    total = sum((answers[i] - 1) if i % 2 else (5 - answers[i]) for i in range(1, 11))
    return total * 2.5


def _bootstrap_mean_ci(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_RESAMPLES))
    low = means[max(0, math.floor(0.025 * BOOTSTRAP_RESAMPLES))]
    high = means[min(BOOTSTRAP_RESAMPLES - 1, math.ceil(0.975 * BOOTSTRAP_RESAMPLES) - 1)]
    return [round(low, 2), round(high, 2)]


def questionnaire_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"participants": 0}
    columns = _item_columns(list(rows[0]))
    if len(columns) != 10:
        raise SystemExit(f"questionnaire needs columns q1..q10, found {sorted(columns)}")
    scores, incomplete, invalid, items = [], [], [], {i: [] for i in range(1, 11)}
    tasks = []
    for row in rows:
        raw = {i: row.get(columns[i], "").strip() for i in range(1, 11)}
        if not any(raw.values()):
            continue  # an unused pre-filled participant row
        pid = row.get("participant_id") or "?"
        if not all(raw.values()):
            incomplete.append(pid)
            continue
        try:
            answers = {i: int(float(value)) for i, value in raw.items()}
        except ValueError:
            invalid.append(pid)
            continue
        if any(not 1 <= value <= 5 for value in answers.values()):
            invalid.append(pid)
            continue
        scores.append(sus_score(answers))
        for i, value in answers.items():
            items[i].append(value)
        if row.get("tasks_completed", "").strip():
            tasks.append(float(row["tasks_completed"]))
    mean = statistics.fmean(scores) if scores else None
    return {
        "participants": len(scores),
        "incomplete": incomplete,
        "invalid": invalid,
        "sus_mean": round(mean, 2) if mean is not None else None,
        "sus_sd": round(statistics.stdev(scores), 2) if len(scores) > 1 else None,
        "sus_ci95_bootstrap": _bootstrap_mean_ci(scores),
        "sus_median": statistics.median(scores) if scores else None,
        "sus_scores": scores,
        "above_benchmark_68": None if mean is None else mean > SUS_BENCHMARK,
        "item_means": {f"q{i}": round(statistics.fmean(values), 2) for i, values in items.items() if values},
        "tasks_completed_mean": round(statistics.fmean(tasks), 2) if tasks else None,
    }


# ------------------------------------------------------------------------------------------ complaints
def complaints(feedback: list[dict[str, Any]], rows: list[dict[str, str]], *, examples: int = 3) -> dict[str, Any]:
    texts = [
        {"from": "thumbs_down", "text": str(record.get("comment") or "").strip(), "query": record.get("query")}
        for record in feedback
        if record.get("rating") == "down" and str(record.get("comment") or "").strip()
    ]
    for row in rows:
        for column in ("most_disliked", "comment"):
            if row.get(column, "").strip():
                texts.append({"from": f"questionnaire.{column}", "text": row[column].strip(), "query": None})
    categories: dict[str, list[dict[str, Any]]] = {name: [] for name in COMPLAINT_CATEGORIES}
    other = []
    for item in texts:
        matched = [name for name, pattern in COMPLAINT_CATEGORIES.items() if re.search(pattern, item["text"], re.I)]
        for name in matched:
            categories[name].append(item)
        if not matched:
            other.append(item)
    ranked = sorted(((name, items) for name, items in categories.items() if items), key=lambda pair: -len(pair[1]))
    return {
        "texts": len(texts),
        "top": [{"category": name, "count": len(items), "examples": items[:examples]} for name, items in ranked],
        "uncategorised": other,
        "method": "keyword groups (COMPLAINT_CATEGORIES in analyse_user_study.py); a text can fall in several",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Analyse the user study (feedback file + SUS questionnaire).")
    parser.add_argument("--feedback", default=str(FEEDBACK_PATH))
    parser.add_argument("--questionnaire", default=str(QUESTIONNAIRE_PATH))
    parser.add_argument("--since", default="", help="Only feedback at or after this ISO time (UTC), e.g. 2026-10-10")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)
    feedback_path, questionnaire_path = Path(args.feedback), Path(args.questionnaire)
    feedback = load_feedback(feedback_path, since=args.since)
    rows = read_csv(questionnaire_path) if questionnaire_path.exists() else []
    if not feedback and not rows:
        raise SystemExit(f"nothing to analyse: {display(feedback_path)} and {display(questionnaire_path)} are empty")
    report = {
        "config": run_config(
            "evaluation.human.analyse_user_study",
            argv,
            feedback=display(feedback_path),
            feedback_sha256=sha256_file(feedback_path) if feedback_path.exists() else None,
            questionnaire=display(questionnaire_path),
            questionnaire_sha256=sha256_file(questionnaire_path) if questionnaire_path.exists() else None,
            since=args.since or None,
            ci_method="Wilson score 95% for the thumbs-up ratio; percentile bootstrap (2000, seed 20260930) for SUS",
        ),
        "feedback": feedback_summary(feedback),
        "sus": questionnaire_summary(rows),
        "complaints": complaints(feedback, rows),
    }
    out = write_result(args.out, report)
    ratio = report["feedback"]["thumbs_up_ratio"]
    print(
        f"thumbs up {ratio['rate']} {ratio['ci95_wilson']} over {ratio['n']} rated answers; "
        f"SUS {report['sus'].get('sus_mean')} {report['sus'].get('sus_ci95_bootstrap')} "
        f"(n={report['sus'].get('participants')}) -> {display(out)}"
    )
    return report


if __name__ == "__main__":
    main()
