"""Chaos / perf / audit result writers record the commit, never an empty string."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.provenance import ROOT, commit_label, git_state


def test_git_state_in_the_repository_names_a_commit():
    state = git_state()
    assert re.fullmatch(r"[0-9a-f]{7,}", state["commit"]) and isinstance(state["working_tree_clean"], bool)
    assert commit_label(state).startswith(state["commit"])


def test_git_state_outside_a_repository_is_explicitly_unknown(tmp_path):
    state = git_state(tmp_path)
    assert state["commit"] == "unknown" and state["working_tree_clean"] is None and state["commit_error"]
    assert commit_label(state) == "unknown"


def test_every_result_writer_uses_the_shared_provenance_helper():
    writers = ["chaos_drill", "load_test", "measure_startup", "llm_latency_probe", "shared_store_probe"]
    writers.append("audit_data_sources")
    for name in writers:
        source = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert "git_state" in source, name
        assert '"rev-parse", "--short", "HEAD"' not in source, name


def test_committed_chaos_results_carry_a_commit():
    import json

    for path in sorted(Path(ROOT, "docs", "results", "chaos").rglob("chaos-*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report.get("commit"), path
