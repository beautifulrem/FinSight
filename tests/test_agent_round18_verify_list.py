"""Round-18 rule N16 (round-14 review): CI and the pre-push run verify the same held-out slices.

``scripts/prepush.sh`` listed eleven verify scripts while CI ran twelve (``verify_heldout_r13.py`` was missing). Both
now run ``python -m evaluation.agent_eval.verify_slices``, whose list must name every ``verify_*.py`` under
``evaluation/``; neither caller may list a verify script itself.
"""

from __future__ import annotations

import re
from pathlib import Path

from evaluation.agent_eval.verify_slices import VERIFY_SCRIPTS

ROOT = Path(__file__).resolve().parents[1]
_MODULE = "evaluation.agent_eval.verify_slices"


def test_the_list_names_every_verify_script_once():
    listed = [script for script, _env in VERIFY_SCRIPTS]
    scripts = (ROOT / "evaluation").rglob("verify_*.py")
    found = sorted(str(path.relative_to(ROOT)) for path in scripts if path.name != "verify_slices.py")
    assert sorted(listed) == found
    assert len(set(listed)) == len(listed)
    env = dict(VERIFY_SCRIPTS)
    assert env["evaluation/heldout_r9/verify_heldout_r9.py"] == {"QI_EVAL_SNAPSHOT_EXT": "1"}
    for number in range(10, 14):
        assert env[f"evaluation/heldout_r{number}/verify_heldout_r{number}.py"] == {
            "QI_EVAL_SNAPSHOT_EXT": "1",
            "QI_EVAL_SNAPSHOT_FIX": "1",
        }


def test_ci_and_prepush_run_the_shared_list_and_list_no_script_themselves():
    for path in (ROOT / ".github" / "workflows" / "ci.yml", ROOT / "scripts" / "prepush.sh"):
        text = path.read_text(encoding="utf-8")
        assert _MODULE in text, path
        assert not re.search(r"verify_(?:heldout_r\d+|multiturn|test_v3)\.py", text), path
