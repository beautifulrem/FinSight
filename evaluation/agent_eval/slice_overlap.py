"""Overlap of a held-out slice with the project's own dev tasks, router labels, claim rows and test literals.

Round-9 review I12: three turns of the round-8 slice had exact or near copies in round-12 tests and dev tasks (written
before the slice existed, see ``evaluation/heldout_r8/README.md``), and the near-copy tests in
``tests/test_agent_eval.py`` only compared new dev tasks with the slices that were already committed. This module
measures the overlap from the slice side, for any slice file or folder, at the time of a run:

* ``exact``: normalised equality (case, whitespace and punctuation ignored) with an own text;
* ``near``: character 3-gram Jaccard >= 0.8 (the threshold of the dev-task tests) or ``difflib`` ratio >= 0.85
  between normalised texts (the reviewer's measure), for slice texts of at least ``MIN_NEAR_CHARS`` normalised
  characters, so "Moutai?" or "茅台呢" do not count as near copies of everything.

Own texts: every turn of the dev tasks (``agent_eval_v1.jsonl``), the router labels (``router_labels_v1.jsonl``),
the dev claims (``claims_v1.jsonl``) and every string literal in ``tests/*.py``. The report names the slice task ids
and the own locations (``tests/test_x.py:255``, ``dev:<task id>``), never the slice text, so it can be run on a slice
the engineers must not open.

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


def slice_texts(path: str | Path) -> list[tuple[str, str]]:
    """``(task id, text)`` for every question / claim / document text in a slice file or folder (``*.jsonl``)."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    found: list[tuple[str, str]] = []

    def collect(value: Any, key: str, task: str) -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                collect(item, str(name), task)
        elif isinstance(value, list):
            for item in value:
                collect(item, key, task)
        elif isinstance(value, str) and key in TEXT_KEYS:
            found.append((task, value))

    for file in files:
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                row = json.loads(line)
                fallback = f"{file.name}:{number}"
                task = str(row.get("id") or fallback) if isinstance(row, dict) else fallback
                collect(row, "", task)
    return found


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=2)
def own_texts(root: Path = ROOT) -> tuple[tuple[str, str], ...]:
    """``(location, text)`` for the project's own wording: dev task turns, router labels, dev claims, test literals."""
    tasks_dir = root / "evaluation" / "agent_eval" / "tasks"
    found: list[tuple[str, str]] = []
    for task in _jsonl(tasks_dir / "agent_eval_v1.jsonl"):
        found.extend((f"dev:{task['id']}", turn["query"]) for turn in task.get("turns") or [])
    found.extend((f"router:{row['id']}", row["query"]) for row in _jsonl(tasks_dir / "router_labels_v1.jsonl"))
    claims = root / "evaluation" / "claim_bench" / "claims_v1.jsonl"
    found.extend((f"claims:{row['id']}", row["claim"]) for row in _jsonl(claims))
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
        "exact": sum(1 for hit in hits if hit["kind"] == "exact"),
        "near": sum(1 for hit in hits if hit["kind"] == "near"),
        "tasks_with_overlap": tasks,
        "hits": hits,
        "method": (
            f"normalised equality (>= {MIN_EXACT_CHARS} chars); near = 3-gram Jaccard >= {NEAR_JACCARD} or difflib "
            f"ratio >= {NEAR_RATIO} (>= {MIN_NEAR_CHARS} chars); own texts = dev task turns, router labels, dev "
            "claims, string literals in tests/*.py"
        ),
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    """The part stored in a run's config: counts, the slice task ids with an overlap, and the method."""
    return {key: report[key] for key in ("slice", "slice_texts", "exact", "near", "tasks_with_overlap", "method")}


def is_heldout_path(path: str | Path) -> bool:
    path = Path(path).resolve()
    return any(part.lower().startswith("heldout") for part in (path.parent.name, path.name))


def _display(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else resolved.name


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Overlap of a held-out slice with own dev tasks and tests.")
    parser.add_argument("slice", help="A slice .jsonl file or a folder of them.")
    parser.add_argument("--out", default=None, help="Write the full report (ids and locations only) here.")
    args = parser.parse_args(argv)
    report = overlap_report(args.slice)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary(report), ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main()
