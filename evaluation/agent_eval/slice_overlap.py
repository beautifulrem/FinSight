"""Overlap of a held-out slice with the project's own dev tasks, router labels, claim rows and test literals.

Round-9 review I12: three turns of the round-8 slice had exact or near copies in round-12 tests and dev tasks (written
before the slice existed, see ``evaluation/heldout_r8/README.md``), and the near-copy tests in
``tests/test_agent_eval.py`` only compared new dev tasks with the slices that were already committed. This module
measures the overlap from the slice side, for any slice file or folder, at the time of a run:

* ``exact``: normalised equality (case, whitespace and punctuation ignored) with an own text;
* ``near``: character 3-gram Jaccard >= 0.8 (the threshold of the dev-task tests) or ``difflib`` ratio >= 0.85
  between normalised texts (the reviewer's measure), for slice texts of at least ``MIN_NEAR_CHARS`` normalised
  characters, so "Moutai?" or "茅台呢" do not count as near copies of everything.

Own texts: every question / claim / document text of EVERY task file under ``evaluation/agent_eval/tasks/`` (the dev
tasks, the router labels, the ext and shipped gate sets, the independent and test sets), of every claim file under
``evaluation/claim_bench/``, and every string literal in ``tests/*.py``. Round-10 review J9: the scan used a fixed
list (dev tasks, router labels, dev claims) and missed ``agent_eval_ext_v1.jsonl``, which holds a one-word variant of
a round-9 slice task; a new task file is now scanned without a code change. The report names the slice task ids and
the own locations (``tests/test_x.py:255``, ``dev:<task id>``, ``tasks/agent_eval_ext_v1.jsonl:<task id>``), never
the slice text, so it can be run on a slice the engineers must not open.

``runner`` attaches the summary to the run's ``config.slice_overlap`` when the task file lives in a ``heldout*``
folder; it can also be run on its own:

    python -m evaluation.agent_eval.slice_overlap evaluation/heldout_r8
    python -m evaluation.agent_eval.slice_overlap /path/to/heldout_r9/chat_r9_heldout.jsonl --out outputs/x.json
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TEXT_KEYS = frozenset({"claim", "query", "text", "title", "body", "question"})
MIN_EXACT_CHARS = 4
MIN_NEAR_CHARS = 10
NEAR_JACCARD = 0.8
NEAR_RATIO = 0.85


def normalise(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def _grams(norm: str) -> frozenset[str]:
    return frozenset(norm[i : i + 3] for i in range(max(1, len(norm) - 2)))


def _collect_texts(row: Any, task: str, found: list[tuple[str, str]], key: str = "") -> None:
    if isinstance(row, dict):
        for name, item in row.items():
            _collect_texts(item, task, found, str(name))
    elif isinstance(row, list):
        for item in row:
            _collect_texts(item, task, found, key)
    elif isinstance(row, str) and key in TEXT_KEYS:
        found.append((task, row))


def jsonl_texts(file: Path) -> list[tuple[str, str]]:
    """``(row id, text)`` for every question / claim / document text of one ``.jsonl`` file."""
    found: list[tuple[str, str]] = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            row = json.loads(line)
            fallback = f"{file.name}:{number}"
            task = str(row.get("id") or fallback) if isinstance(row, dict) else fallback
            _collect_texts(row, task, found)
    return found


def slice_texts(path: str | Path) -> list[tuple[str, str]]:
    """``(task id, text)`` for every question / claim / document text in a slice file or folder (``*.jsonl``)."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    return [item for file in files for item in jsonl_texts(file)]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# Short location prefixes kept from the fixed-list version (reports and tests cite them); every other file is labelled
# with its path under evaluation/.
_PREFIXES = {"agent_eval_v1.jsonl": "dev", "router_labels_v1.jsonl": "router", "claims_v1.jsonl": "claims"}


def own_files(root: Path = ROOT) -> list[Path]:
    """Every task file under ``evaluation/agent_eval/tasks/`` and claim file under ``evaluation/claim_bench/``."""
    evaluation = root / "evaluation"
    return sorted((evaluation / "agent_eval" / "tasks").glob("*.jsonl")) + sorted(
        (evaluation / "claim_bench").glob("*.jsonl")
    )


@lru_cache(maxsize=2)
def own_texts(root: Path = ROOT) -> tuple[tuple[str, str], ...]:
    """``(location, text)`` for the project's own wording: every task and claim file, and the test literals."""
    found: list[tuple[str, str]] = []
    for file in own_files(root):
        prefix = _PREFIXES.get(file.name) or f"{file.parent.name}/{file.name}"
        found.extend((f"{prefix}:{row_id}", text) for row_id, text in jsonl_texts(file))
    for file in sorted((root / "tests").glob("*.py")):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        relative = file.relative_to(root)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and 2 <= len(node.value) <= 300:
                found.append((f"{relative}:{node.lineno}", node.value))
    return tuple(found)


def overlap_report(path: str | Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Exact and near copies of the slice's texts among the project's own texts (ids and locations only)."""
    texts = slice_texts(path)
    own = [(location, normalise(text)) for location, text in own_texts(root)]
    own = [(location, norm) for location, norm in own if norm]
    exact_index: dict[str, list[str]] = {}
    for location, norm in own:
        exact_index.setdefault(norm, []).append(location)
    own_grams = [(location, norm, _grams(norm)) for location, norm in own if len(norm) >= MIN_NEAR_CHARS // 2]
    hits: list[dict[str, Any]] = []
    for task, text in texts:
        norm = normalise(text)
        if len(norm) >= MIN_EXACT_CHARS and norm in exact_index:
            hits.append({"task": task, "kind": "exact", "chars": len(norm), "own": sorted(set(exact_index[norm]))[:5]})
            continue
        if len(norm) < MIN_NEAR_CHARS:
            continue
        mine = _grams(norm)
        best: tuple[float, str] | None = None
        for location, other, grams in own_grams:
            if abs(len(other) - len(norm)) > max(len(other), len(norm)) // 2:
                continue
            jaccard = len(mine & grams) / len(mine | grams)
            matcher = difflib.SequenceMatcher(None, norm, other, autojunk=False)
            ratio = matcher.ratio() if matcher.quick_ratio() >= NEAR_RATIO else 0.0
            score = max(jaccard / NEAR_JACCARD, ratio / NEAR_RATIO)
            if score >= 1.0 and (best is None or score > best[0]):
                best = (score, location)
        if best is not None:
            hits.append({"task": task, "kind": "near", "chars": len(norm), "own": [best[1]]})
    tasks = sorted({hit["task"] for hit in hits})
    return {
        "slice": _display(path, root),
        "slice_texts": len(texts),
        "slice_tasks": len({task for task, _text in texts}),
        "own_texts": len(own),
        "own_files": [_display(file, root) for file in own_files(root)],
        "exact": sum(1 for hit in hits if hit["kind"] == "exact"),
        "near": sum(1 for hit in hits if hit["kind"] == "near"),
        "tasks_with_overlap": tasks,
        "hits": hits,
        "method": (
            f"normalised equality (>= {MIN_EXACT_CHARS} chars); near = 3-gram Jaccard >= {NEAR_JACCARD} or difflib "
            f"ratio >= {NEAR_RATIO} (>= {MIN_NEAR_CHARS} chars); own texts = every evaluation/agent_eval/tasks/*.jsonl "
            "and evaluation/claim_bench/*.jsonl text, string literals in tests/*.py"
        ),
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    """The part stored in a run's config: counts, the slice task ids with an overlap, and the method."""
    keys = ("slice", "slice_texts", "exact", "near", "tasks_with_overlap", "own_files", "method")
    return {key: report[key] for key in keys if key in report}


def is_heldout_path(path: str | Path) -> bool:
    path = Path(path).resolve()
    return any(part.lower().startswith("heldout") for part in (path.parent.name, path.name))


def _display(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else resolved.name


def without_overlap(report: dict[str, Any], run_names: list[str]) -> dict[str, Any]:
    """Pass counts of committed chat runs of the slice (``evaluation/results/<name>.json``) with and without the
    slice tasks the report found, e.g. ``{"chat_heldout_r9-auto-nollm-prefix": {"all": "18/52", ...}}``."""
    from .results import decode_outcomes, load_result

    overlapping = set(report["tasks_with_overlap"])
    counts: dict[str, Any] = {}
    for name in run_names:
        result = load_result(name)
        if not result:
            raise SystemExit(f"evaluation/results/{name}.json not found")
        outcomes = decode_outcomes(result.get("task_outcomes"))
        passed = {task for task, runs in outcomes.items() if runs and all(runs)}
        rest = [task for task in outcomes if task not in overlapping]
        excluded = sorted(task for task in outcomes if task in overlapping)
        counts[name] = {
            "commit": (result.get("config") or {}).get("commit"),
            "all": f"{len(passed)}/{len(outcomes)}",
            "without_overlap": f"{sum(task in passed for task in rest)}/{len(rest)}",
            "excluded": {task: task in passed for task in excluded},
        }
    return counts


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Overlap of a held-out slice with own dev tasks and tests.")
    parser.add_argument("slice", help="A slice .jsonl file or a folder of them.")
    parser.add_argument("--out", default=None, help="Write the full report (ids and locations only) here.")
    parser.add_argument(
        "--chat-runs",
        nargs="*",
        default=[],
        help="Committed chat runs of the slice (evaluation/results stems): pass counts with and without the overlap.",
    )
    args = parser.parse_args(argv)
    report = overlap_report(args.slice)
    if args.chat_runs:
        report["chat_runs"] = without_overlap(report, args.chat_runs)
    if args.out:
        from datetime import UTC, datetime

        from .runner import _git_commit

        given = sys.argv[1:] if argv is None else argv
        command = " ".join(["python -m evaluation.agent_eval.slice_overlap", *given])
        report = {
            "kind": "slice_overlap_scan",
            "config": {
                "slice": report["slice"],
                "commit": _git_commit(),
                "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "command": command,
            },
            **report,
        }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary(report), ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main()
