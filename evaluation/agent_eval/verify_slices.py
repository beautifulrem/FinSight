"""The held-out slices' verify scripts, one list for CI and the pre-push run (round 18, review-14 N16).

Each committed independent slice ships a verify script that re-derives its expected facts from the offline tools.
CI's ``heldout-verify`` job and ``scripts/prepush.sh`` used to list the scripts separately, and the pre-push list
fell behind (it lacked ``verify_heldout_r13.py``). Both now run this module, and ``tests/test_verify_slices.py``
checks that every ``verify_*.py`` under ``evaluation/`` is listed here and that neither caller lists scripts itself.

    python -m evaluation.agent_eval.verify_slices          # run every script, stop at the first failure
    python -m evaluation.agent_eval.verify_slices --list   # print the scripts with their environment
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_EXT = {"QI_EVAL_SNAPSHOT_EXT": "1"}
_EXT_FIX = {"QI_EVAL_SNAPSHOT_EXT": "1", "QI_EVAL_SNAPSHOT_FIX": "1"}
# (script, extra environment): r9 reads the extension snapshot; r10 on also the corrections layer.
VERIFY_SCRIPTS: tuple[tuple[str, dict[str, str]], ...] = (
    ("evaluation/agent_eval/independent/verify_multiturn.py", {}),
    ("evaluation/agent_eval/independent/verify_test_v3.py", {}),
    ("evaluation/heldout_r4/verify_heldout_r4.py", {}),
    ("evaluation/heldout_r5/verify_heldout_r5.py", {}),
    ("evaluation/heldout_r6/verify_heldout_r6.py", {}),
    ("evaluation/heldout_r7/verify_heldout_r7.py", {}),
    ("evaluation/heldout_r8/verify_heldout_r8.py", {}),
    ("evaluation/heldout_r9/verify_heldout_r9.py", _EXT),
    ("evaluation/heldout_r10/verify_heldout_r10.py", _EXT_FIX),
    ("evaluation/heldout_r11/verify_heldout_r11.py", _EXT_FIX),
    ("evaluation/heldout_r12/verify_heldout_r12.py", _EXT_FIX),
    ("evaluation/heldout_r13/verify_heldout_r13.py", _EXT_FIX),
)


def _describe(script: str, extra: dict[str, str]) -> str:
    return " ".join([*(f"{name}={value}" for name, value in extra.items()), "python", script])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the scripts and exit")
    args = parser.parse_args(argv)
    if args.list:
        for script, extra in VERIFY_SCRIPTS:
            print(_describe(script, extra))
        return 0
    for script, extra in VERIFY_SCRIPTS:
        env = {**os.environ, "PYTHONPATH": str(ROOT), **extra}
        print(f"[verify_slices] {_describe(script, extra)}", flush=True)
        if subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=False).returncode:
            print(f"[verify_slices] FAILED: {script}", flush=True)
            return 1
    print(f"{len(VERIFY_SCRIPTS)} verify scripts passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
