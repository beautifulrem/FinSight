"""``slice_overlap`` scans every task and claim file, not a fixed list (round-10 review J9).

The round-13 version read the dev tasks, the router labels and the dev claims only, so it missed
``tasks/agent_eval_ext_v1.jsonl``, which holds a one-word variant of a round-9 slice task (r9h_ind_en_01).
"""

from __future__ import annotations

import json

from evaluation.agent_eval import slice_overlap


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_a_new_task_file_and_a_new_claim_file_are_scanned_without_a_code_change(tmp_path):
    root = tmp_path
    (root / "tests").mkdir()
    tasks = root / "evaluation" / "agent_eval" / "tasks"
    _write_jsonl(tasks / "agent_eval_v1.jsonl", [{"id": "dev_1", "turns": [{"query": "茅台的市净率是多少倍呢"}]}])
    _write_jsonl(
        tasks / "agent_eval_brand_new_v7.jsonl",
        [{"id": "new_1", "turns": [{"query": "Is CMB's P/B above or below its industry average?"}]}],
    )
    _write_jsonl(
        root / "evaluation" / "claim_bench" / "claims_extra_v2.jsonl",
        [{"id": "c9", "claim": "招商银行的净利润大约是平安银行的三倍半"}],
    )
    slice_dir = root / "heldout_rX"
    _write_jsonl(
        slice_dir / "chat.jsonl",
        [
            {"id": "s_task", "turns": [{"query": "Is CMB's P/B above or below the industry average?"}]},
            {"id": "s_dev", "turns": [{"query": "茅台的市净率是多少倍"}]},
            {"id": "s_new", "turns": [{"query": "两家加起来一共多少亿"}]},
        ],
    )
    _write_jsonl(slice_dir / "claims.jsonl", [{"id": "s_claim", "claim": "招商银行净利润大约是平安银行的三倍半"}])

    slice_overlap.own_texts.cache_clear()
    try:
        report = slice_overlap.overlap_report(slice_dir, root=root)
    finally:
        slice_overlap.own_texts.cache_clear()

    assert report["tasks_with_overlap"] == ["s_claim", "s_dev", "s_task"]
    own = {hit["task"]: hit["own"] for hit in report["hits"]}
    assert own["s_task"] == ["tasks/agent_eval_brand_new_v7.jsonl:new_1"]
    assert own["s_claim"] == ["claim_bench/claims_extra_v2.jsonl:c9"]
    assert own["s_dev"] == ["dev:dev_1"]
    assert len(report["own_files"]) == 3


def test_every_committed_task_and_claim_file_is_an_own_file():
    root = slice_overlap.ROOT
    expected = sorted((root / "evaluation" / "agent_eval" / "tasks").glob("*.jsonl")) + sorted(
        (root / "evaluation" / "claim_bench").glob("*.jsonl")
    )

    assert slice_overlap.own_files() == expected
    names = {path.name for path in expected}
    assert {"agent_eval_ext_v1.jsonl", "agent_eval_shipped_v1.jsonl", "claims_v1_holdout.jsonl"} <= names


def test_the_round9_slice_overlap_includes_the_ext_set_variant_and_the_committed_counts_reproduce():
    report = slice_overlap.overlap_report(slice_overlap.ROOT / "evaluation" / "heldout_r9")
    own = {(hit["task"], location) for hit in report["hits"] for location in hit["own"]}
    assert ("r9h_ind_en_01", "tasks/agent_eval_ext_v1.jsonl:ext_industry_missing_en") in own

    committed = json.loads(
        (slice_overlap.ROOT / "evaluation" / "results" / "slice_overlap-heldout_r9-all-tasks.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(committed["tasks_with_overlap"]) <= set(report["tasks_with_overlap"])
    names = list(committed["chat_runs"])
    assert slice_overlap.without_overlap(committed, names) == committed["chat_runs"]
