"""Decision changes of the deny-by-default lookalike rule over every evaluation query string (round 18, review-14 N5).

Round 17 treated a covered company's short name followed by a *business-line word* (a word list) as another company.
Round 18 denies by default: any name characters after a listed company's name or alias make an unknown company,
unless they are a word that can follow a company name in a question (``nlu/lookalike.py``). This script measures what
that changes on the questions the project already evaluates: every string under a ``query`` / ``question`` /
``claim`` / ``text`` / ``message`` / ``context`` key in the JSON / JSONL files under ``evaluation/`` (task sets,
router labels, held-out slices, claim sets, fuzz cases, red-team cases; not ``evaluation/results`` and not the answer
snapshots in ``fixtures``), deduplicated.

For each string the NLU front end decides which listed securities it names (normalizer, exact matching, then the fuzzy
passes when nothing matched) and which names it sets aside as lookalikes or Hong Kong-only names. The script runs that
with the rule off (``lookalike.DENY_UNKNOWN_EXTENSIONS = False``, the round-17 behaviour) and on, and lists every
string whose decision differs.

    python -m evaluation.agent_eval.lookalike_regression --out evaluation/results/lookalike-regression-r18.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dirty_guard import add_dirty_arguments, check_dirty_write
from .runner import ROOT, _git_commit, build_offline_service

_KEYS = frozenset({"query", "question", "claim", "text", "message", "context"})
_SKIP_DIRS = ("evaluation/results", "evaluation/agent_eval/fixtures")


def _strings(value: Any, key: str = "") -> list[str]:
    if isinstance(value, str):
        return [value] if key in _KEYS else []
    if isinstance(value, dict):
        return [text for name, item in value.items() for text in _strings(item, str(name))]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item, key)]
    return []


def evaluation_strings(root: Path = ROOT) -> list[str]:
    """Every query-like string under ``evaluation/`` (see the module docstring), deduplicated, in file order."""
    found: dict[str, None] = {}
    for path in sorted((root / "evaluation").rglob("*.json*")):
        relative = path.relative_to(root).as_posix()
        if path.suffix not in {".json", ".jsonl"} or relative.startswith(_SKIP_DIRS):
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records = [json.loads(text)]
        for record in records:
            for item in _strings(record):
                item = item.strip()
                if 2 <= len(item) <= 400:
                    found.setdefault(item, None)
    return list(found)


def decision(pipeline: Any, text: str) -> dict[str, list[str]]:
    """The listed symbols the NLU front end resolves in ``text`` and the names it sets aside."""
    normalized, normal_trace = pipeline.normalizer.normalize(text)
    resolver = pipeline.entity_resolver
    entities, _targets, trace = resolver.resolve_exact(normalized)
    if not [entity for entity in entities if entity.get("symbol")]:
        entities, _targets, trace = resolver.resolve(normalized)
    marks = ("lookalike_name:", "hk_only_name:")
    set_aside = sorted({item.split(":", 1)[1] for item in [*normal_trace, *trace] if item.startswith(marks)})
    symbols = sorted({str(entity["symbol"]) for entity in entities if entity.get("symbol")})
    return {"symbols": symbols, "set_aside": set_aside}


def main(argv: list[str] | None = None) -> int:
    from query_intelligence.nlu import lookalike

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "outputs" / "agent_eval" / "lookalike-regression.json"))
    add_dirty_arguments(parser)
    args = parser.parse_args(argv)
    commit = _git_commit()
    label = check_dirty_write(args.out, commit, allow_dirty=args.allow_dirty, notes=args.note)
    strings = evaluation_strings()
    pipeline = build_offline_service(snapshot_ext=True, snapshot_fix=True).nlu_pipeline
    outcomes: dict[str, list[dict[str, list[str]]]] = {}
    for deny in (False, True):
        lookalike.DENY_UNKNOWN_EXTENSIONS = deny
        for text in strings:
            outcomes.setdefault(text, []).append(decision(pipeline, text))
    lookalike.DENY_UNKNOWN_EXTENSIONS = True
    changes = [
        {"text": text, "before": before, "after": after}
        for text, (before, after) in outcomes.items()
        if before != after
    ]
    report = {
        "kind": "lookalike_regression",
        "config": {
            "commit": commit,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": "python -m evaluation.agent_eval.lookalike_regression " + " ".join(argv or []),
            "rule": "query_intelligence/nlu/lookalike.py DENY_UNKNOWN_EXTENSIONS off (round 17) vs on (round 18)",
        },
        "summary": {
            "strings": len(strings),
            "changed": len(changes),
            "lost_a_symbol": sum(bool(set(c["before"]["symbols"]) - set(c["after"]["symbols"])) for c in changes),
            "gained_a_symbol": sum(bool(set(c["after"]["symbols"]) - set(c["before"]["symbols"])) for c in changes),
        },
        "changes": changes,
        "notes": [*args.note, *([label] if label else [])],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    for change in changes:
        print(change["text"], "|", change["before"], "->", change["after"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
