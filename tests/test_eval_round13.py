"""Round 13 (round-9 review I12, I13): evaluation honesty checks.

* I13: every writer that can put a file into a committed results folder refuses a dirty working tree before it runs,
  unless ``--allow-dirty`` and a ``--note`` are both given (then the file is labelled).
* I12: the overlap of a held-out slice with the project's own dev tasks, router labels, claims and test literals is
  measured from the slice side and recorded in the run's config.
"""

from __future__ import annotations

import json

import pytest

from evaluation.agent_eval import dirty_guard
from evaluation.agent_eval.results import RESULTS_DIR


def test_dirty_guard_refuses_committed_folders_and_labels_allowed_writes(tmp_path):
    results_file = RESULTS_DIR / "x.json"
    docs_file = dirty_guard.ROOT / "docs" / "results" / "perf" / "x.json"
    scratch = dirty_guard.ROOT / "outputs" / "agent_eval" / "x.json"
    assert dirty_guard.is_committed_results_path(results_file) and dirty_guard.is_committed_results_path(docs_file)
    assert not dirty_guard.is_committed_results_path(scratch) and not dirty_guard.is_committed_results_path(tmp_path)

    assert dirty_guard.check_dirty_write(results_file, "abc1234", allow_dirty=False) is None
    assert dirty_guard.check_dirty_write(scratch, "abc1234-dirty", allow_dirty=False) is None
    with pytest.raises(SystemExit, match="dirty working tree"):
        dirty_guard.check_dirty_write(results_file, "abc1234-dirty", allow_dirty=False)
    with pytest.raises(SystemExit, match="needs a --note"):
        dirty_guard.check_dirty_write(results_file, "abc1234-dirty", allow_dirty=True, notes=[])
    label = dirty_guard.check_dirty_write(results_file, "abc1234-dirty", allow_dirty=True, notes=["prompt edit"])
    assert label and "abc1234-dirty" in label and "--allow-dirty" in label and "prompt edit" in label


def _no_run(*_args, **_kwargs):
    raise AssertionError("a refused dirty write must stop before the run")


def test_gate_update_baseline_refuses_a_dirty_tree_before_running(monkeypatch):
    from evaluation.agent_eval import gate

    monkeypatch.setattr(gate, "_git_commit", lambda: "abc1234-dirty")
    monkeypatch.setattr(gate, "run_eval", _no_run)
    with pytest.raises(SystemExit, match="dirty working tree"):
        gate.main(["--set", "holdout", "--update-baseline"])
    with pytest.raises(SystemExit, match="needs a --note"):
        gate.main(["--set", "holdout", "--update-baseline", "--allow-dirty"])


def test_gate_without_update_does_not_check_the_tree(monkeypatch):
    """A plain gate run writes only to outputs/ (CI on a PR merge commit is clean anyway)."""
    from evaluation.agent_eval import gate

    monkeypatch.setattr(gate, "_git_commit", lambda: "abc1234-dirty")
    called = []
    monkeypatch.setattr(gate, "extras_problems", lambda: called.append(True) or [])
    assert gate.main(["--extras-only"]) == 0 and called


def test_router_eval_refuses_a_dirty_write_into_results(monkeypatch, tmp_path):
    from evaluation.agent_eval import router_eval

    monkeypatch.setattr(router_eval, "_git_commit", lambda: "abc1234-dirty")
    monkeypatch.setattr(router_eval, "evaluate", _no_run)
    with pytest.raises(SystemExit, match="dirty working tree"):
        router_eval.main(["--out", str(RESULTS_DIR / "router_eval-should-not-exist.json")])
    assert not (RESULTS_DIR / "router_eval-should-not-exist.json").exists()

    # scratch output: allowed, and the file records the dirty commit
    monkeypatch.setattr(
        router_eval,
        "evaluate",
        lambda rows: {"queries": 0, "accuracy": None, "per_route": {}, "confusion": {}, "errors": []},
    )
    monkeypatch.setattr(router_eval, "ROUTES", ())
    out = tmp_path / "router.json"
    router_eval.main(["--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["config"]["commit"] == "abc1234-dirty"


def test_claim_bench_and_verifier_stress_refuse_a_dirty_write_into_results(monkeypatch):
    from evaluation.agent_eval import verifier_stress
    from evaluation.claim_bench import run as claim_run

    monkeypatch.setattr(claim_run, "_git_commit", lambda: "abc1234-dirty")
    monkeypatch.setattr(claim_run, "load_claims", _no_run)
    with pytest.raises(SystemExit, match="dirty working tree"):
        claim_run.main(["--set", "dev"])  # default output: evaluation/results/claim_bench-dev.json

    monkeypatch.setattr(verifier_stress, "_git_commit", lambda: "abc1234-dirty")
    monkeypatch.setattr(verifier_stress, "run", _no_run)
    with pytest.raises(SystemExit, match="dirty working tree"):
        verifier_stress.main(["--out", str(RESULTS_DIR / "verifier_stress-should-not-exist.json")])


def test_results_slimmer_needs_a_note_with_allow_dirty(tmp_path, monkeypatch):
    from evaluation.agent_eval import results

    run = {"config": {"commit": "960432d-dirty", "model": None}, "summary": {"task_success": 1.0}, "failures": []}
    source = tmp_path / "run.json"
    source.write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.setattr(results, "ROOT", tmp_path)
    monkeypatch.setattr(results, "RESULTS_DIR", tmp_path / "results")
    with pytest.raises(SystemExit, match="needs a --note"):
        results.main([str(source), "--allow-dirty"])


def test_committed_results_carry_no_unlabelled_dirty_commit():
    """Round-9 review I13: no committed result may name a ``-dirty`` commit without a notes label saying so."""
    from evaluation.agent_eval.results import dirty_provenance

    unlabelled = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or not dirty_provenance(report):
            continue
        config = report.get("config") if isinstance(report.get("config"), dict) else {}
        labels = [*(report.get("notes") or []), config.get("note") or "", *(config.get("notes") or [])]
        notes = " ".join(str(note) for note in labels)
        if "dirty" not in notes.lower():
            unlabelled.append(path.name)
    assert unlabelled == []


# ---- I12: slice overlap ----


def test_slice_overlap_reports_ids_and_locations_never_slice_text(tmp_path):
    from evaluation.agent_eval import slice_overlap

    root = tmp_path
    (root / "tests").mkdir()
    (root / "tests" / "test_x.py").write_text('QUERIES = ["切回深市那只，它的股息率呢", "茅台呢"]\n', encoding="utf-8")
    tasks = root / "evaluation" / "agent_eval" / "tasks"
    tasks.mkdir(parents=True)
    dev = {"id": "dev_1", "turns": [{"query": "which one rose more, and by what margin?"}]}
    (tasks / "agent_eval_v1.jsonl").write_text(json.dumps(dev, ensure_ascii=False) + "\n", encoding="utf-8")
    slice_dir = root / "heldout_rX"
    slice_dir.mkdir()
    rows = [
        {"id": "s_exact", "turns": [{"query": "切回深市那只, 它的股息率呢?"}]},  # punctuation differs only
        {"id": "s_near", "turns": [{"query": "Which one rose more and by what margin"}]},
        {"id": "s_short", "turns": [{"query": "茅台呢"}]},  # 3 normalised characters: below the floor, ignored
        {"id": "s_new", "turns": [{"query": "两家加起来一共多少亿"}]},
    ]
    (slice_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    slice_overlap.own_texts.cache_clear()
    report = slice_overlap.overlap_report(slice_dir, root=root)
    slice_overlap.own_texts.cache_clear()
    assert report["slice_texts"] == 4 and report["tasks_with_overlap"] == ["s_exact", "s_near"]
    by_task = {hit["task"]: hit for hit in report["hits"]}
    assert by_task["s_exact"]["kind"] == "exact" and by_task["s_exact"]["own"] == ["tests/test_x.py:1"]
    assert by_task["s_near"]["own"] == ["dev:dev_1"]
    assert "股息率" not in json.dumps(report, ensure_ascii=False)  # ids and locations only
    assert slice_overlap.is_heldout_path(slice_dir / "chat.jsonl")
    assert not slice_overlap.is_heldout_path(tasks / "agent_eval_v1.jsonl")


def test_slice_overlap_finds_the_three_round8_turns_the_review_named():
    """The round-9 review found three unpublished round-8 slice turns copied or near-copied in round-12 tests and dev
    tasks (r8h_hk_zh_03, r8h_gap_en_04, r8h_hold_zh_03); the slice-side check must report all three."""
    from evaluation.agent_eval import slice_overlap

    report = slice_overlap.overlap_report(slice_overlap.ROOT / "evaluation" / "heldout_r8")
    assert {"r8h_hk_zh_03", "r8h_gap_en_04", "r8h_hold_zh_03"} <= set(report["tasks_with_overlap"])
    own = {(hit["task"], location) for hit in report["hits"] for location in hit["own"]}
    assert ("r8h_hk_zh_03", "tests/test_agent_round12_frame.py:255") in own
    assert ("r8h_gap_en_04", "dev:r12_traded_more_by_how_much_en") in own
    assert ("r8h_hold_zh_03", "tests/test_agent_round12_frame.py:205") in own


def test_sample_tasks_is_seeded_ordered_and_recorded():
    from evaluation.agent_eval.runner import sample_tasks

    tasks = [{"id": f"t{index}"} for index in range(20)]
    subset, info = sample_tasks(tasks, 5, 7)
    again, _ = sample_tasks(tasks, 5, 7)
    assert subset == again and len(subset) == 5
    assert [task["id"] for task in subset] == info["task_ids"]
    assert info["task_ids"] == sorted(info["task_ids"], key=lambda name: int(name[1:]))  # file order kept
    assert info["seed"] == 7 and info["of"] == 20
    full, info_full = sample_tasks(tasks, 0, 7)
    assert full == tasks and info_full["task_ids"] is None
