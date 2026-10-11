"""The README headline numbers are generated from committed result files (round-13 review M14)."""

from __future__ import annotations

import dataclasses
import json
import shutil

import pytest

from evaluation.agent_eval import readme, report


def test_committed_readmes_match_the_result_files():
    assert readme.problems() == []


def test_report_check_includes_the_readme_regions(monkeypatch, capsys):
    monkeypatch.setattr(report.readme_regions, "problems", lambda: ["README.md: a generated region is out of date"])
    assert report.main(["--check"]) == 1
    assert "readme: README.md: a generated region is out of date" in capsys.readouterr().out


def test_every_region_is_marked_once_in_both_readmes():
    for path in readme.READMES.values():
        text = path.read_text(encoding="utf-8")
        for region in readme.REGIONS:
            assert text.count(readme.begin_marker(region)) == 1, (path.name, region)
            assert text.count(readme.end_marker(region)) == 1, (path.name, region)


def test_a_newer_result_makes_the_committed_readme_stale():
    facts = readme.collect()
    newer = dict(facts.verifier, gold_answers=facts.verifier["gold_answers"] + 1)
    stale = readme.render(dataclasses.replace(facts, verifier=newer, verifier_round=facts.verifier_round + 1))
    found = readme.problems(stale)
    assert any("README.md: a generated region is out of date" in problem for problem in found)
    assert any("README_CN.md" in problem for problem in found)


def test_splice_reports_missing_markers_and_keeps_the_rest():
    text = f"intro\n{readme.begin_marker('pitch')}\nold\n{readme.end_marker('pitch')}\noutro\n"
    updated, missing = readme.splice(text, {"pitch": "new", "limits": "x"})
    assert missing == ["limits"]
    assert updated == f"intro\n{readme.begin_marker('pitch')}\nnew\n{readme.end_marker('pitch')}\noutro\n"


def test_latest_slice_and_its_overlap_tiers_come_from_the_files():
    facts = readme.collect()
    latest = facts.slices[-1]
    numbers = readme.slice_rounds()
    assert latest.round == max(numbers)
    # the tiers are recomputed from the overlap scan and both runs' task outcomes
    scan = json.loads((readme.RESULTS / f"{latest.overlap}.json").read_text(encoding="utf-8"))
    excluded = set(scan["tasks_with_overlap"])
    after = json.loads(
        (readme.RESULTS / f"chat_heldout_r{latest.round}-auto-nollm-after-fix.json").read_text(encoding="utf-8")
    )["task_outcomes"]
    kept = set(after) - excluded
    assert latest.tiers[0] == (len(kept), latest.tiers[0][1], sum(after[t] == "1" for t in kept))
    assert latest.tiers[0][0] >= latest.tiers[1][0] >= latest.tiers[2][0]
    rendered = readme.render(facts)
    assert (
        f"(round {latest.round}), {latest.tasks - latest.after} of {latest.tasks} chat tasks"
        in rendered["en"]["limits"]
    )
    assert f"Latest reviewer slice (round {latest.round})" in rendered["en"]["headline-table"]


@pytest.fixture()
def results_copy(tmp_path, monkeypatch):
    """A copy of evaluation/results and of the slice folders, so that files can be added."""
    results = tmp_path / "evaluation" / "results"
    shutil.copytree(readme.RESULTS, results)
    for folder in readme.ROOT.glob("evaluation/heldout_r*"):
        (tmp_path / "evaluation" / folder.name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(readme, "ROOT", tmp_path)
    monkeypatch.setattr(readme, "RESULTS", results)
    readme._json.cache_clear()
    yield tmp_path
    readme._json.cache_clear()


def test_a_slice_counts_as_latest_only_with_its_folder_and_an_after_fix_run(results_copy):
    before = readme.slice_rounds()
    top = max(before) + 1
    results = results_copy / "evaluation" / "results"
    prefix = results / f"chat_heldout_r{max(before)}-auto-nollm-prefix.json"
    shutil.copy(prefix, results / f"chat_heldout_r{top}-auto-nollm-prefix.json")
    (results_copy / "evaluation" / f"heldout_r{top}").mkdir()
    assert readme.slice_rounds() == before  # a first run alone does not make it the latest
    shutil.copy(prefix, results / f"chat_heldout_r{top}-auto-nollm-after-fix.json")
    assert readme.slice_rounds() == [*before, top]


def test_the_highest_verifier_round_wins(results_copy):
    rounds = readme._numbered(r"verifier_stress-round(\d+)")
    results = results_copy / "evaluation" / "results"
    shutil.copy(results / f"{rounds[-1][1]}.json", results / f"verifier_stress-round{rounds[-1][0] + 1}.json")
    assert readme._numbered(r"verifier_stress-round(\d+)")[-1] == (
        rounds[-1][0] + 1,
        f"verifier_stress-round{rounds[-1][0] + 1}",
    )


def test_prepush_and_ci_records_are_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr(readme, "CI_DIR", tmp_path)
    readme._json.cache_clear()
    (tmp_path / "prepush-1111111.md").write_text(
        "Run by `scripts/prepush.sh` (2026-10-01T00:00:00Z).\n| Result | **at least one step failed** |\n"
        "| pytest | `python -m pytest -q tests` | pass | 600 | budget "
        "| 4397 passed, 70 skipped, 5 warnings in 636.17s |\n",
        encoding="utf-8",
    )
    (tmp_path / "prepush-2222222.md").write_text(
        "Run by `scripts/prepush.sh` (2026-10-02T00:00:00Z).\n| Result | **all steps passed** |\n"
        "| pytest | `python -m pytest -q tests` | pass | 600 | budget | 4400 passed, 71 skipped in 600.00s |\n"
        "| pytest-shipped | `x` | pass | 60 | budget | 512 passed, 3 skipped, 4000 deselected in 60.00s |\n",
        encoding="utf-8",
    )
    found = readme.prepush()
    assert found.sha == "2222222" and found.all_passed
    assert found.tests("en") == "4,400 passed, 71 skipped"
    assert found.tests("zh", shipped=True) == "512 个通过、3 个跳过、4,000 个未选中"
    record = {
        "run_id": 1,
        "url": "https://example.invalid/1",
        "head_sha": "abcdef0123",
        "conclusion": "success",
        "created_at": "2026-10-02T00:00:00Z",
        "jobs": [
            {"name": "lint", "conclusion": "success", "minutes": 0.2, "timeout_minutes": 10},
            {"name": "tests", "conclusion": "success", "minutes": 60.0, "timeout_minutes": 120},
        ],
        "recorded_at": "2026-10-02T01:00:00Z",
    }
    (tmp_path / "ci-run-abcdef0.json").write_text(json.dumps(record), encoding="utf-8")
    run = readme.ci_run()
    assert (run.run_id, run.sha, len(run.jobs)) == (1, "abcdef0", 2)


def test_github_anchors_and_chinese_round_numbers():
    assert (
        readme.slug("Verifier stress test (329 gold answers, 5597 corrupted variants, `verifier_stress-round16.json`)")
        == "verifier-stress-test-329-gold-answers-5597-corrupted-variants-verifier_stress-round16json"
    )
    assert readme.slug("CI、推送前检查与人工评测工具包") == "ci推送前检查与人工评测工具包"
    assert [readme.zh_number(n) for n in (6, 10, 12, 20, 23)] == ["六", "十", "十二", "二十", "二十三"]


def test_a_slice_without_its_agent_eval_section_is_reported_not_a_crash(results_copy):
    results = results_copy / "evaluation" / "results"
    top = max(readme.slice_rounds()) + 1
    prefix = results / f"chat_heldout_r{top - 1}-auto-nollm-prefix.json"
    for kind in ("prefix", "after-fix"):
        shutil.copy(prefix, results / f"chat_heldout_r{top}-auto-nollm-{kind}.json")
    (results_copy / "evaluation" / f"heldout_r{top}").mkdir()
    found = readme.problems()
    assert len(found) == 1 and f"Round-{top} held-out chat slice" in found[0]
