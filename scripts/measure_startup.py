"""Measure service start-up: cold build, rebuild in the same process, and a restart with the index on disk.

The document retriever's TF-IDF fit dominates start-up. It is memoised by corpus hash within a process
and, with ``QI_TFIDF_CACHE_DIR``, persisted between processes. Live sources are disabled so only local
work is timed.

    python scripts/measure_startup.py --out docs/results/perf/startup.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = {
    name: "false"
    for name in ("QI_USE_LIVE_MARKET", "QI_USE_LIVE_MACRO", "QI_USE_LIVE_NEWS", "QI_USE_LIVE_ANNOUNCEMENT")
}

_CHILD = """
import json, time
from query_intelligence.service import build_default_service, clear_service_caches
timings = []
for _ in range({builds}):
    clear_service_caches()
    started = time.perf_counter()
    build_default_service()
    timings.append(round(time.perf_counter() - started, 2))
print(json.dumps(timings))
"""


def _run(builds: int, env: dict[str, str]) -> list[float]:
    out = subprocess.run(
        [sys.executable, "-c", _CHILD.format(builds=builds)],
        cwd=ROOT,
        env={**os.environ, **OFFLINE, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    with tempfile.TemporaryDirectory() as cache_dir:
        in_process = _run(2, {"QI_TFIDF_CACHE_DIR": ""})
        first = _run(1, {"QI_TFIDF_CACHE_DIR": cache_dir})
        restart = _run(1, {"QI_TFIDF_CACHE_DIR": cache_dir})
        cache_mb = round(sum(path.stat().st_size for path in Path(cache_dir).iterdir()) / 1e6, 1)
    report = {
        "commit": commit.strip(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "python scripts/measure_startup.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "host": {"platform": platform.platform(), "cpus": os.cpu_count(), "load_avg": list(os.getloadavg())},
        "seconds": {
            "cold_build": in_process[0],
            "rebuild_same_process": in_process[1],
            "cold_build_writing_disk_cache": first[0],
            "restart_with_disk_cache": restart[0],
        },
        "disk_cache_mb": cache_mb,
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
