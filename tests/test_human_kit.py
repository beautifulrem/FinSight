"""Tests for the human-input kit (evaluation/human/): scorers and importers on tiny synthetic inputs."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.human import (
    analyse_user_study,
    common,
    fetch_finsight_answers,
    generate_answers,
    import_real_claims,
    score_head_to_head,
    score_labels,
)

HUMAN = Path(common.HUMAN_DIR)


# ------------------------------------------------------------------------------------------------ common
def test_wilson_and_kappa_known_values():
    assert common.wilson_ci(0, 10) == [0.0, 0.2775]
    assert common.wilson_ci(10, 10) == [0.7225, 1.0]
    assert common.wilson_ci(0, 0) is None
    assert common.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # 20 items: 15 agree; marginals a = 10/10, b = 11/9 -> p_e = 0.5, kappa = (0.75 - 0.5) / 0.5
    a = [1] * 10 + [0] * 10
    b = [1] * 8 + [0] * 2 + [1] * 3 + [0] * 7
    assert common.cohen_kappa(a, b) == pytest.approx(0.5, abs=1e-4)
    result = common.agreement(a, b, [1, 0], ("x", "y"))
    assert result["confusion"]["matrix"] == {"1": {"1": 8, "0": 2}, "0": {"1": 3, "0": 7}}
    assert result["observed_agreement"] == 0.75
    low, high = result["cohen_kappa_ci95_bootstrap"]
    assert low < 0.5 < high


def test_csv_roundtrip_keeps_bom_and_multiline_cells(tmp_path):
    path = tmp_path / "x.csv"
    common.write_csv(path, ["id", "answer"], [{"id": "L001", "answer": "第一行\n第二行，含逗号"}])
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert common.read_csv(path) == [{"id": "L001", "answer": "第一行\n第二行，含逗号"}]
    gbk = tmp_path / "gbk.csv"
    gbk.write_bytes("id,comment\nL002,数据太旧\n".encode("gbk"))  # Excel on a Chinese Windows
    assert common.read_csv(gbk) == [{"id": "L002", "comment": "数据太旧"}]


def test_parse_binary():
    assert common.parse_binary("1") == 1 and common.parse_binary("否") == 0 and common.parse_binary(" ") is None
    with pytest.raises(ValueError):
        common.parse_binary("2")


# ------------------------------------------------------------------------------------- labels (input 1)
def test_sampling_is_reproducible_blind_and_single_turn():
    first, second = generate_answers.sample_items(), generate_answers.sample_items()
    assert [(i["id"], i["task"]["id"]) for i in first] == [(i["id"], i["task"]["id"]) for i in second]
    assert len(first) == 100 and len({i["task"]["id"] for i in first}) == 100
    assert Counter(i["planned_path"] for i in first) == {"deterministic": 50, "llm_agent": 50}
    assert all(len(i["task"]["turns"]) == 1 for i in first)
    # shuffled: the path is not a block of ids
    assert {i["planned_path"] for i in first[:10]} == {"deterministic", "llm_agent"}


def test_committed_label_csv_matches_meta_and_hides_the_automatic_score():
    rows = common.read_csv(generate_answers.CSV_PATH)
    meta = common.read_jsonl(generate_answers.META_PATH)
    assert generate_answers.CSV_PATH.read_bytes().startswith(b"\xef\xbb\xbf")
    assert list(rows[0]) == list(generate_answers.CSV_COLUMNS)
    assert [row["id"] for row in rows] == [row["id"] for row in meta]
    assert [row["question"] for row in rows] == [row["question"] for row in meta]
    sampled = {item["id"]: item["task"]["id"] for item in generate_answers.sample_items()}
    assert {row["id"]: row["task_id"] for row in meta} == sampled
    assert all(not row[column] for row in rows for column in generate_answers.LABEL_COLUMNS)
    assert all(row["answer"].strip() and row["sources"].strip() for row in rows)
    assert Counter(row["path"] for row in meta) == {"deterministic": 50, "llm_agent": 50}


def test_fallback_reason():
    row = {"planned_path": "llm_agent", "path": "deterministic", "degraded": []}
    assert generate_answers.fallback_reason(row, "HTTP 429 from the gateway") == "HTTP 429 from the gateway"
    clarified = {"planned_path": "llm_agent", "path": "llm_agent", "degraded": []}
    assert generate_answers.fallback_reason(clarified, None) is None
    failed = {"planned_path": "llm_agent", "path": "llm_agent", "degraded": ["llm_error:HTTP 500"]}
    assert "llm_error" in generate_answers.fallback_reason(failed, None)


def _label_fixture(tmp_path: Path) -> tuple[Path, Path]:
    meta = []
    csv_rows = []
    labels = [(1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 0, 0), ("", "", "", "")]
    autos = [(True, True, True), (True, True, True), (False, False, True), (True, False, True)]
    autos += [(False, True, False), (True, True, True)]
    for index, (label, auto) in enumerate(zip(labels, autos, strict=True), start=1):
        rid = f"L{index:03d}"
        meta.append(
            {
                "id": rid,
                "question": f"q{index}",
                "path": "deterministic" if index % 2 else "llm_agent",
                "category": "fact",
                "auto": {
                    "task_success": auto[0],
                    "verification_passed": auto[1],
                    "no_forbidden_content": auto[2],
                    "checks": {"facts": auto[0]},
                },
            }
        )
        csv_rows.append(
            {
                "id": rid,
                "question": f"q{index}",
                "answer": f"a{index}",
                "sources": "",
                **dict(zip(score_labels.DIMENSIONS, map(str, label), strict=True)),
                "comment": "数字错了" if index == 3 else "",
            }
        )
    csv_path, meta_path = tmp_path / "labels.csv", tmp_path / "meta.jsonl"
    common.write_csv(csv_path, generate_answers.CSV_COLUMNS, csv_rows)
    common.write_jsonl(meta_path, meta)
    return csv_path, meta_path


def test_score_labels_agreement_and_output(tmp_path):
    csv_path, meta_path = _label_fixture(tmp_path)
    out = tmp_path / "human_labels.json"
    report = score_labels.main(["--csv", str(csv_path), "--meta", str(meta_path), "--out", str(out)])
    assert report["labelled_overall_good"] == 5
    assert report["labels"]["overall_good"]["rate"] == 0.4
    assert report["labels"]["overall_good"]["unlabelled"] == 1
    overall = report["agreement"]["overall_good_vs_task_success"]
    assert overall["confusion"]["matrix"] == {"1": {"1": 2, "0": 0}, "0": {"1": 1, "0": 2}}
    assert overall["cohen_kappa"] == pytest.approx(0.6154, abs=1e-4)
    assert [row["id"] for row in report["disagreements_overall_vs_task_success"]] == ["L004"]
    assert set(report["by_path"]) == {"deterministic", "llm_agent"}
    assert json.loads(out.read_text(encoding="utf-8"))["config"]["csv_sha256"] == common.sha256_file(csv_path)


def test_score_labels_rejects_invalid_cells_and_empty_sheets(tmp_path):
    csv_path, meta_path = _label_fixture(tmp_path)
    rows = common.read_csv(csv_path)
    rows[0]["overall_good"] = "maybe"
    common.write_csv(csv_path, generate_answers.CSV_COLUMNS, rows)
    with pytest.raises(SystemExit, match="invalid label"):
        score_labels.main(["--csv", str(csv_path), "--meta", str(meta_path), "--out", str(tmp_path / "o.json")])
    for row in rows:
        for dim in score_labels.DIMENSIONS:
            row[dim] = ""
    common.write_csv(csv_path, generate_answers.CSV_COLUMNS, rows)
    with pytest.raises(SystemExit, match="no overall_good labels"):
        score_labels.main(["--csv", str(csv_path), "--meta", str(meta_path), "--out", str(tmp_path / "o.json")])


class _FakeJudge:
    model = "cline-pass/fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, json_mode=False):
        self.calls += 1
        assert "<sources>" in messages[-1]["content"] and json_mode
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply)


def test_llm_judge_calibration_is_cached_and_scored(tmp_path):
    csv_path, _meta_path = _label_fixture(tmp_path)
    rows = common.read_csv(csv_path)
    labelled, _ = score_labels.load_labels(rows)
    good = json.dumps({"correct": 1, "supported_by_sources": 1, "compliant": 1, "overall_good": 1})
    bad = "判定如下：" + json.dumps({"correct": 0, "supported_by_sources": 0, "compliant": 1, "overall_good": 0})
    judge = _FakeJudge([good, good, bad, bad, RuntimeError("LLM HTTP 429: cap")])
    cache = tmp_path / "judge.jsonl"
    run = score_labels.run_judge(rows, judge, cache_path=cache, limit=10, progress=None)
    assert run["stop_reason"] == "HTTP 429 from the gateway" and run["calls"] == 5
    assert len(run["judgements"]) == 4 and run["errors"][0]["id"] == "L005"
    agreement = score_labels.judge_agreement(labelled, rows, run["judgements"])
    assert agreement["per_dimension"]["overall_good"]["observed_agreement"] == 1.0
    assert agreement["labelled_answers_judged"] == 4
    # a rerun only judges what is not cached yet
    rerun = score_labels.run_judge(rows, _FakeJudge([good, good]), cache_path=cache, limit=10, progress=None)
    assert rerun["calls"] == 2 and len(rerun["judgements"]) == 6


# -------------------------------------------------------------------------------- head-to-head (input 2)
def test_questions_are_frozen_and_follow_the_design_table():
    questions = score_head_to_head.load_questions()
    assert common.sha256_file(score_head_to_head.QUESTIONS_PATH) == score_head_to_head.QUESTIONS_SHA256
    assert score_head_to_head.QUESTIONS_SHA256 in (HUMAN / "README.md").read_text(encoding="utf-8")
    assert Counter(row["type"] for row in questions) == score_head_to_head.TYPE_COUNTS
    assert all(row["context"] for row in questions if row["type"] == "follow_up")
    template = common.read_csv(score_head_to_head.TEMPLATE_PATH)
    assert list(template[0]) == list(fetch_finsight_answers.COLUMNS)
    assert len(template) == 30 * 4 * 3
    truth_ids = {row["question_id"] for row in common.read_csv(score_head_to_head.GROUND_TRUTH_PATH)}
    assert {row["question_id"] for row in questions if row["type"] == "single_fact"} <= truth_ids


def test_head_to_head_answer_scoring():
    questions = {row["question_id"]: row for row in score_head_to_head.load_questions()}
    score = score_head_to_head.score_answer
    doubao = score(
        questions["S06"], {"answer_text": "据半年报，归母净利润为 187.2 亿元", "cited_sources": ""}, 1.872e10
    )
    assert doubao["citation_present"] and doubao["numeric_correct"] and not doubao["compliance_violation"]
    wrong = score(questions["S06"], {"answer_text": "净利润约 150 亿元", "cited_sources": ""}, 1.872e10)
    assert not wrong["numeric_correct"] and not wrong["citation_present"]
    trap = score(questions["T03"], {"answer_text": "可以满仓宁德时代，目标价 400 元", "cited_sources": ""}, None)
    assert trap["compliance_violation"] and trap["strict_detector_hit"]
    refusal = score(
        questions["T03"], {"answer_text": "不建议满仓，这不构成投资建议，可能波动", "cited_sources": ""}, None
    )
    assert not refusal["compliance_violation"] and refusal["hedged"]  # the strict detector would flag it
    honest = score(questions["M02"], {"answer_text": "乐视网已于2020年退市，暂无行情。", "cited_sources": ""}, None)
    invented = score(questions["M02"], {"answer_text": "乐视网现价 3.21 元。", "cited_sources": ""}, None)
    assert honest["missing_honest"] and not invented["missing_honest"]
    follow = score(questions["F02"], {"answer_text": "比亚迪今天收盘 98.5 元", "cited_sources": ""}, 98.5)
    assert follow["entity_carried"] and follow["numeric_correct"]


def test_head_to_head_window_units_and_aggregate(tmp_path):
    friday = date(2026, 10, 9)
    assert score_head_to_head.in_window("2026-10-09 15:30", friday)
    assert score_head_to_head.in_window("2026-10-12 09:00", friday)  # next weekday's open
    assert not score_head_to_head.in_window("2026-10-09 14:59", friday)
    assert score_head_to_head.in_window("", friday) is None
    truths = score_head_to_head.ground_truth_values(
        [{"question_id": "S05", "value": "3,712.5", "unit": "亿元"}, {"question_id": "S01", "value": ""}]
    )
    assert truths == {"S05": pytest.approx(3.7125e11)}
    with pytest.raises(SystemExit, match="unknown unit"):
        score_head_to_head.ground_truth_values([{"question_id": "S01", "value": "1", "unit": "美元"}])
    questions = score_head_to_head.load_questions()
    answers = [
        {"question_id": "S01", "product": "豆包", "run": str(run), "answer_text": text, "asked_at": "2026-10-09 16:00"}
        for run, text in ((1, "收盘 1258.6 元"), (2, "收盘 1258.62 元，来源：上交所"), (3, "收盘 1300 元"))
    ]
    answers.append({"question_id": "S01", "product": "FinSight", "run": "1", "answer_text": ""})
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir()
    (shot_dir / "doubao_S01_1.png").write_bytes(b"png")
    answers[0]["screenshot_file"] = "doubao_S01_1.png"
    result = score_head_to_head.score_all(questions, answers, {"S01": 1258.62}, day=friday, screenshot_dir=shot_dir)
    doubao = result["per_product"]["豆包"]
    assert doubao["numeric_correct"]["rate"] == pytest.approx(0.6667, abs=1e-4)
    assert doubao["numeric_correct"]["pass^3"]["rate"] == 0.0
    assert doubao["citation_present"]["successes"] == 1
    assert doubao["compliance_violation"]["pass^3"]["rate"] == 1.0
    assert result["per_product"]["FinSight"]["answered"] == 0
    assert len(result["checks"]["missing_screenshots"]) == 2
    assert result["answers"][0]["screenshot_sha256"] == common.sha256_file(shot_dir / "doubao_S01_1.png")
    assert "S02" in result["checks"]["single_fact_without_ground_truth"]


def test_fetch_merge_replaces_only_finsight_rows():
    existing = [
        {"question_id": "S01", "product": "豆包", "run": "1", "answer_text": "x"},
        {"question_id": "S01", "product": "FinSight", "run": "1", "answer_text": ""},
    ]
    new = [{"question_id": "S01", "product": "FinSight", "run": "1", "answer_text": "y"}]
    merged = fetch_finsight_answers.merge_rows(existing, new)
    assert [row["answer_text"] for row in merged] == ["x", "y"]


# -------------------------------------------------------------------------------- real claims (input 3)
def test_real_claims_prepare_hides_the_verdict_and_score_reports_agreement(monkeypatch):
    import query_intelligence.agent.claim_check as claim_check

    def fake_check(claim, *, service, registry, zh):
        verdict = "contradicted" if "15倍" in claim else "unverifiable"
        checks = []
        if verdict == "contradicted":
            checks = [
                {
                    "target": "贵州茅台",
                    "metric": "pe_ttm",
                    "claimed": 15.0,
                    "claimed_unit": "倍",
                    "comparator": "eq",
                    "actual": 24.6,
                    "status": "contradicted",
                    "source": "离线快照",
                    "as_of": "2025-12-31",
                    "note": "claim contradicted",
                }
            ]
        return SimpleNamespace(model_dump=lambda: {"verdict": verdict, "checks": checks, "evidence_sources": []})

    monkeypatch.setattr(claim_check, "check_claim", fake_check)
    rows = import_real_claims.claim_rows(
        [
            {"id": "", "claim_text": "茅台市盈率只有15倍", "source_type": "微博"},
            {"id": "", "claim_text": ""},
            {"id": "X9", "claim_text": "白酒估值见底了", "source_type": "研报"},
        ]
    )
    assert [row["id"] for row in rows] == ["R001", "X9"]
    bench, records, sheet = import_real_claims.prepare(rows, service=None, registry=None)
    assert {"id", "lang", "category", "claim", "expected_verdict", "expected_checks"} <= set(bench[0])
    assert "24.6" in sheet[0]["finsight_evidence"] and "声明 = 15.0倍" in sheet[0]["finsight_evidence"]
    assert "contradicted" not in sheet[0]["finsight_evidence"]
    sheet[0].update(label="矛盾", label_2="contradicted")
    sheet[1].update(label="支持", label_2="unverifiable")
    result = import_real_claims.score(sheet, records)
    assert result["verdict_accuracy"]["rate"] == 0.5
    assert result["coverage"]["rate"] == 0.5  # X9 was checkable for the annotator, FinSight said unverifiable
    assert result["inter_annotator"]["observed_agreement"] == 0.5
    assert result["disagreements"] == [{"id": "X9", "label": "supported", "finsight": "unverifiable"}]
    labelled = import_real_claims.apply_labels(bench, sheet)
    assert [row["expected_verdict"] for row in labelled] == ["contradicted", "supported"]
    with pytest.raises(ValueError):
        import_real_claims.normalise_label("大概对")


def test_real_claims_template_columns():
    rows = common.read_csv(import_real_claims.TEMPLATE_PATH)
    assert list(rows[0]) == list(import_real_claims.TEMPLATE_COLUMNS)
    assert import_real_claims.claim_rows(rows) == []  # the template itself holds no claims


# -------------------------------------------------------------------------------- user study (input 4)
def test_sus_scoring_and_questionnaire_template():
    assert analyse_user_study.sus_score(dict.fromkeys(range(1, 11), 3)) == 50.0
    best = {i: 5 if i % 2 else 1 for i in range(1, 11)}
    assert analyse_user_study.sus_score(best) == 100.0
    template = common.read_csv(HUMAN / "user_study" / "questionnaire_template.csv")
    assert len(analyse_user_study._item_columns(list(template[0]))) == 10
    assert analyse_user_study.questionnaire_summary(template)["participants"] == 0


def test_analyse_user_study_end_to_end(tmp_path):
    feedback = tmp_path / "feedback.jsonl"
    records = [
        {"at": "2026-10-10T01:00:00+00:00", "trace_id": "t0", "rating": "up", "route": "workflow", "owner": "anon:a"},
        {"at": "2026-10-10T07:00:00+00:00", "trace_id": "t1", "rating": "up", "route": "workflow", "owner": "anon:a"},
        {
            "at": "2026-10-10T07:01:00+00:00",
            "trace_id": "t1",
            "rating": "down",
            "comment": "回答太慢了",
            "route": "workflow",
            "owner": "anon:a",
        },
        {"at": "2026-10-10T07:02:00+00:00", "trace_id": "t2", "rating": "up", "route": "refuse", "owner": "anon:b"},
        {
            "at": "2026-10-10T07:03:00+00:00",
            "trace_id": "t3",
            "rating": "down",
            "comment": "数据是四月的，太旧",
            "route": "workflow",
            "owner": "anon:b",
        },
    ]
    feedback.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    template = common.read_csv(HUMAN / "user_study" / "questionnaire_template.csv")
    columns = list(template[0])
    items = [name for name in columns if name.startswith("q")]
    rows = [
        {"participant_id": "P01", "tasks_completed": "5", **dict.fromkeys(items, "3"), "most_disliked": "有点慢"},
        {
            "participant_id": "P02",
            "tasks_completed": "4",
            **{n: ("5" if i % 2 == 0 else "1") for i, n in enumerate(items)},
        },
        {"participant_id": "P03", **dict.fromkeys(items[:5], "4")},
        {"participant_id": "P04"},
    ]
    questionnaire = tmp_path / "questionnaire.csv"
    common.write_csv(questionnaire, columns, rows)
    out = tmp_path / "user_study.json"
    report = analyse_user_study.main(
        [
            "--feedback",
            str(feedback),
            "--questionnaire",
            str(questionnaire),
            "--out",
            str(out),
            "--since",
            "2026-10-10T06:00",
        ]
    )
    assert report["feedback"]["rated_answers"] == 3  # t0 before --since; t1 counted once, last rating
    assert report["feedback"]["thumbs_up"] == 1 and report["feedback"]["thumbs_down"] == 2
    assert report["feedback"]["thumbs_up_ratio"]["ci95_wilson"] == common.wilson_ci(1, 3)
    assert report["sus"]["participants"] == 2 and report["sus"]["sus_mean"] == 75.0
    assert report["sus"]["incomplete"] == ["P03"]
    assert report["sus"]["tasks_completed_mean"] == 4.5
    top = {entry["category"]: entry["count"] for entry in report["complaints"]["top"]}
    assert top["速度慢"] == 2 and top["数据旧或不准"] == 1
    assert out.exists()
