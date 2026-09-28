"""Commit provenance for chaos / perf / audit result writers.

Every result file under ``docs/results/`` must say which code produced it. ``git_state()`` returns the
short commit and whether tracked files had uncommitted changes, and never an empty string: when git
cannot be run (no repository, a "dubious ownership" refusal on an external volume, no git binary) the
commit is the literal ``"unknown"`` and ``commit_error`` says why. Call it when the run *starts*, so a
commit made while a long run is in progress is not attributed to it.

    from scripts.provenance import git_state
    report.update(git_state())   # {"commit": "4742453", "working_tree_clean": True}
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = "unknown"


def _git(*args: str, cwd: Path) -> tuple[str | None, str | None]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if out.returncode != 0:
        return None, (out.stderr.strip() or f"git {' '.join(args)} exited {out.returncode}")[:200]
    return out.stdout.strip(), None


def git_state(root: Path = ROOT) -> dict[str, Any]:
    """``{"commit": <short sha or "unknown">, "working_tree_clean": bool | None[, "commit_error": str]}``."""
    commit, error = _git("rev-parse", "--short=7", "HEAD", cwd=root)
    status, status_error = _git("status", "--porcelain", "--untracked-files=no", cwd=root)
    state: dict[str, Any] = {
        "commit": commit or UNKNOWN,
        "working_tree_clean": None if status is None else not status,
    }
    if error or status_error:
        state["commit_error"] = error or status_error
    return state


def commit_label(state: dict[str, Any] | None = None) -> str:
    """``4742453`` or ``4742453-dirty`` (or ``unknown``), for files that keep a single commit string."""
    state = state or git_state()
    suffix = "-dirty" if state.get("working_tree_clean") is False else ""
    return f"{state['commit']}{suffix}"
