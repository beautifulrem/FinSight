"""Refuse to write committed evidence from a dirty working tree (round-9 review I13).

Every writer that can put a file under ``evaluation/results/`` (or another committed results folder such as
``docs/results/``) records the commit it ran from; ``runner._git_commit`` suffixes it ``-dirty`` when tracked files
had uncommitted changes. Such a result cannot be reproduced from any commit. The slimmer
(``python -m evaluation.agent_eval.results``) already refused those; this module gives the other writers the same
rule:

* the gate's ``--update-baseline`` (``evaluation/results/gate-<set>.json``),
* ``router_eval --out evaluation/results/...``,
* ``claim_bench.run`` (its default output is ``evaluation/results/claim_bench-<set>.json``),
* ``verifier_stress --out evaluation/results/...``,
* (round 14, J15) ``evaluation.real_claims_web.score_claims_web_v1`` (its default outputs are in
  ``evaluation/results/``).

A dirty run aimed at a committed folder stops before it starts unless ``--allow-dirty`` and a ``--note`` saying
what was uncommitted are both given; the file then carries a ``dirty_tree`` label in its notes. Scratch outputs
(``outputs/``, gitignored) are never refused: their recorded commit already says ``-dirty``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Folders whose files are committed evidence (cited by the READMEs / docs / gate).
COMMITTED_RESULT_DIRS = (ROOT / "evaluation" / "results", ROOT / "docs" / "results")


def is_committed_results_path(path: str | Path) -> bool:
    resolved = Path(path).resolve()
    return any(resolved == folder or folder in resolved.parents for folder in COMMITTED_RESULT_DIRS)


def add_dirty_arguments(parser: argparse.ArgumentParser, *, note: bool = True) -> None:
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Write into a committed results folder from a working tree with uncommitted changes (needs --note); "
        "the file is labelled.",
    )
    if note:
        parser.add_argument("--note", action="append", default=[], help="Provenance note to attach.")


def dirty_label(commit: str | None, notes: list[str] | None = None) -> str:
    detail = "; ".join(note for note in notes or [] if note)
    return (
        f"Recorded from a dirty working tree ({commit}): not reproducible from a commit; written with --allow-dirty"
        + (f" ({detail})." if detail else ".")
    )


def check_dirty_write(
    out: str | Path, commit: str | None, *, allow_dirty: bool, notes: list[str] | None = None
) -> str | None:
    """``None`` when the write is clean or goes to scratch; the label to store when a dirty write is allowed.

    Raises ``SystemExit`` (before any work is done, when called right after argument parsing) when a dirty run
    would write into a committed results folder without ``--allow-dirty`` and a ``--note``.
    """
    if not commit or not commit.endswith("-dirty") or not is_committed_results_path(out):
        return None
    if not allow_dirty:
        raise SystemExit(
            f"refusing to write {out} from a dirty working tree ({commit}): commit the changes and rerun from a "
            "clean tree, write to outputs/ instead, or pass --allow-dirty with a --note saying what was uncommitted"
        )
    if not any(note for note in notes or []):
        raise SystemExit("--allow-dirty needs a --note saying what was uncommitted")
    return dirty_label(commit, notes)
