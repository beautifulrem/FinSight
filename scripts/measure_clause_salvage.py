"""Reproduce the "clause salvage: 29% readable" figure (C16) with a committed measurement.

Round 3 replaced the verifier's repair step (B6, ``1bcdde7``): it used to *salvage clauses* (keep the clauses of a
failing sentence whose numbers were supported and glue them back together); it now deletes whole sentences and
falls back to the template answer. The before/after readability was measured once and only quoted in the message
of ``063aeca``. This script repeats that measurement so the number has a file behind it.

Run it from a checkout of ``063aeca~1`` (``2494656``, the commit the "after" numbers in ``063aeca`` were measured
at: the verifier-stress harness with the readability metric, the same gold answers and corrupted variants):

    git worktree add --detach /tmp/fs-2494656 063aeca~1
    cd /tmp/fs-2494656 && /path/to/python /path/to/FinSight/scripts/measure_clause_salvage.py \
        --legacy-commit ca18ae6 --out /path/to/FinSight/evaluation/results/verifier_stress-clause-salvage.json

It runs ``evaluation.agent_eval.verifier_stress`` of that checkout twice with the same seed: once as it is
(whole-sentence repair, "after"), and once with ``repair_answer`` replaced by the clause-salvage implementation of
``--legacy-commit`` (``ca18ae6``, the last commit before B6), loaded from git into a scratch package together with
the evidence module it was written against. Gold answers, variants, the verification that rejects them, and the
readability / fragment / verification checks on the repaired text are the same code in both runs; only the repair
differs. The legacy repair had no template fallback (``fallback`` is ignored), as in the graph at that commit. A
third run feeds the legacy repair the legacy verifier's own report (what to cut decided by the rules of that commit,
as its graph did); readability and verification of the result are still judged by this checkout's code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=Path.cwd(), text=True).strip()


def _load_legacy_verifier(commit: str) -> Any:
    """``query_intelligence/agent/verifier.py`` of ``commit`` (with its ``evidence.py``) as a scratch package."""
    root = Path(tempfile.mkdtemp(prefix="legacy-verifier-"))
    package = root / f"legacy_{commit}"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in ("verifier", "evidence"):
        source = _git("show", f"{commit}:query_intelligence/agent/{name}.py")
        (package / f"{name}.py").write_text(source + "\n", encoding="utf-8")
    sys.path.insert(0, str(root))
    return importlib.import_module(f"legacy_{commit}.verifier")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--legacy-commit", default="ca18ae6", help="Commit whose repair_answer salvages clauses.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    head = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))

    from evaluation.agent_eval import verifier_stress
    from evaluation.agent_eval.runner import DEFAULT_TASKS, load_tasks

    tasks = load_tasks(DEFAULT_TASKS)
    current = verifier_stress.run(tasks, seed=args.seed)

    legacy = _load_legacy_verifier(args.legacy_commit)

    def clause_salvage(draft: dict[str, Any], report: Any, store: Any, *, zh: bool, fallback: Any = None):
        return legacy.repair_answer(draft, report, store, zh=zh)

    def clause_salvage_legacy_report(draft: dict[str, Any], report: Any, store: Any, *, zh: bool, fallback: Any = None):
        # as the graph at the legacy commit ran it: its own verifier's report decides what to cut
        legacy_report = legacy.verify_answer(draft, store, market_precedence=False)
        return legacy.repair_answer(draft, legacy_report, store, zh=zh)

    verifier_stress.repair_answer = clause_salvage
    salvage = verifier_stress.run(tasks, seed=args.seed)
    verifier_stress.repair_answer = clause_salvage_legacy_report
    salvage_legacy_report = verifier_stress.run(tasks, seed=args.seed)

    result = {
        "kind": "verifier_stress_repair_comparison",
        "config": {
            "commit": head + ("-dirty" if dirty else ""),
            "legacy_repair_commit": _git("rev-parse", "--short", args.legacy_commit),
            "tasks": str(DEFAULT_TASKS.relative_to(Path.cwd())),
            "seed": args.seed,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": "python scripts/measure_clause_salvage.py " + " ".join(argv or sys.argv[1:]),
        },
        "gold_answers": current["gold_answers"],
        "variants": current["variants"],
        "same_rejections": current["repair"]["repaired_answers"] == salvage["repair"]["repaired_answers"],
        "whole_sentence_repair": current["repair"],
        # the legacy repair given this commit's verification report (what to cut = today's rules)
        "clause_salvage_repair": salvage["repair"],
        # the legacy repair given the legacy verifier's report (what to cut = the rules of --legacy-commit)
        "clause_salvage_repair_legacy_report": salvage_legacy_report["repair"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    keys = ("repaired_answers", "readable", "with_fragment", "passes_verification", "retained_sentences")
    for label in ("whole_sentence_repair", "clause_salvage_repair", "clause_salvage_repair_legacy_report"):
        print(label, {key: result[label][key] for key in keys})
    return result


if __name__ == "__main__":
    main()
