"""Measure container cold start: ``docker run`` until ``/health`` and ``/ready`` answer 200.

The image should be built from a clean commit, e.g.

    git archive --format=tar <commit> | docker build -f docker/Dockerfile -t finsight:<commit> -
    python scripts/measure_container_startup.py --image finsight:<commit> --runs 3 \
        --out docs/results/perf/startup-container.json

Each run starts a fresh container with the same hardening as the Kubernetes manifest (read-only root
filesystem, non-root, tmpfs for /tmp, /app/state and /app/outputs) and live data off, polls every 0.5 s,
then removes it. ``/ready`` builds the agent and opens the SQLite checkpointer on its first call, so the
time to ready includes that. The commit is read from the image's build label if present, else from the
tag (``finsight:<commit>``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

OFFLINE = ["QI_USE_LIVE_MARKET=0", "QI_USE_LIVE_NEWS=0", "QI_USE_LIVE_ANNOUNCEMENT=0", "QI_USE_LIVE_MACRO=0"]


def _status(url: str) -> int | None:
    request = urllib.request.Request(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return None


def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True).stdout.strip()


def run_once(image: str, port: int, timeout_s: float) -> dict:
    name = f"finsight-startup-{os.getpid()}-{port}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    command = ["run", "-d", "--name", name, "-p", f"127.0.0.1:{port}:8000", "--read-only", "--tmpfs", "/tmp"]
    command += ["--tmpfs", "/app/state:uid=10001", "--tmpfs", "/app/outputs:uid=10001"]
    for item in OFFLINE:
        command += ["-e", item]
    started = time.perf_counter()
    _docker(*command, image)
    health_s = ready_s = None
    ready_body: dict | None = None
    try:
        while time.perf_counter() - started < timeout_s:
            if health_s is None and _status(f"http://127.0.0.1:{port}/health") == 200:
                health_s = round(time.perf_counter() - started, 2)
            if health_s is not None and _status(f"http://127.0.0.1:{port}/ready") == 200:
                ready_s = round(time.perf_counter() - started, 2)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"http://127.0.0.1:{port}/ready", timeout=10) as response:
                    ready_body = json.loads(response.read())
                break
            time.sleep(0.5)
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        service_line = next(
            (line for line in (logs.stdout + logs.stderr).splitlines() if "service loaded in" in line), None
        )
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return {
        "health_200_s": health_s,
        "ready_200_s": ready_s,
        "service_build_log": service_line,
        "ready_checks": (ready_body or {}).get("checks"),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    inspect = json.loads(_docker("image", "inspect", args.image))[0]
    labels = (inspect.get("Config") or {}).get("Labels") or {}
    commit = labels.get("org.opencontainers.image.revision") or args.image.rpartition(":")[2]
    runs = []
    for index in range(args.runs):
        load_before = list(os.getloadavg())
        result = run_once(args.image, args.port, args.timeout)
        result["load_avg_before"] = [round(value, 2) for value in load_before]
        runs.append(result)
        print(f"run {index + 1}: health {result['health_200_s']} s, ready {result['ready_200_s']} s", flush=True)
    ready = [run["ready_200_s"] for run in runs if run["ready_200_s"] is not None]
    health = [run["health_200_s"] for run in runs if run["health_200_s"] is not None]
    docker_info = json.loads(_docker("info", "--format", "{{json .}}"))
    report = {
        "commit": commit,
        "image": args.image,
        "image_id": inspect["Id"],
        # `docker image inspect` .Size is the compressed content size with the containerd image store;
        # `docker images` lists the unpacked size.
        "image_content_size_mb": round(inspect["Size"] / 1e6),
        "image_size_listed": _docker("images", args.image, "--format", "{{.Size}}"),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "python scripts/measure_container_startup.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
        "container": "read-only rootfs, uid 10001, tmpfs /tmp /app/state /app/outputs, live data off, "
        "SQLite checkpointer, no LLM key",
        "host": {
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "docker_vm_cpus": docker_info.get("NCPU"),
            "docker_vm_memory_gb": round(docker_info.get("MemTotal", 0) / 2**30, 1),
        },
        "seconds": {
            "health_200_median": statistics.median(health) if health else None,
            "ready_200_median": statistics.median(ready) if ready else None,
            "ready_200_min": min(ready) if ready else None,
            "ready_200_max": max(ready) if ready else None,
        },
        "runs": runs,
    }
    print(json.dumps(report["seconds"], indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
