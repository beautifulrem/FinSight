"""Round 14 (round-10 review J15): the claims_web_v1 scorer refuses a dirty working tree before it runs when it would
write into a committed results folder, like the other writers (``evaluation/agent_eval/dirty_guard.py``)."""

from __future__ import annotations

import pytest

from evaluation.agent_eval import runner
from evaluation.agent_eval.results import RESULTS_DIR
from evaluation.real_claims_web import score_claims_web_v1


def _no_run(*_args, **_kwargs):
    raise AssertionError("a refused dirty write must stop before the run")


@pytest.fixture
def dirty_tree(monkeypatch):
    monkeypatch.setattr(runner, "_git_commit", lambda: "abc1234-dirty")
    monkeypatch.setattr(runner, "build_offline_service", _no_run)


@pytest.mark.parametrize(
    "argv",
    [
        [],  # the default output, evaluation/results/claims_web_v1-first-run.json
        ["--corrections"],  # evaluation/results/claims_web_v1-with-corrections.json
        [str(RESULTS_DIR / "claims_web_v1-x.json"), "--after-exposure"],
    ],
)
def test_claims_web_scorer_refuses_a_dirty_tree_before_running(dirty_tree, argv):
    with pytest.raises(SystemExit, match="dirty working tree"):
        score_claims_web_v1.main(argv)
    with pytest.raises(SystemExit, match="needs a --note"):
        score_claims_web_v1.main([*argv, "--allow-dirty"])


def test_claims_web_scorer_writes_scratch_output_from_a_dirty_tree(dirty_tree, tmp_path, monkeypatch):
    """Scratch output (outside the committed folders) is not refused; the run reaches the service."""
    reached = []
    monkeypatch.setattr(runner, "build_offline_service", lambda **_kwargs: reached.append(True) or _no_run())
    with pytest.raises(AssertionError, match="stop before the run"):
        score_claims_web_v1.main([str(tmp_path / "x.json")])
    assert reached
