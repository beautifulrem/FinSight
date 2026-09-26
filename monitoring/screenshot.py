"""Capture the Grafana dashboard and a Jaeger trace of the monitoring profile as PNGs.

    python monitoring/screenshot.py --grafana http://127.0.0.1:3300 --jaeger http://127.0.0.1:16686 \
        --out docs/assets/ops

Uses Playwright with the locally installed Chrome (``channel="chrome"``) and bypasses any system proxy.
The Jaeger screenshot opens the slowest recent ``finsight.agent.run`` trace of service ``finsight-agent``
(or ``--trace-id``).
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


def _slowest_trace(jaeger: str, operation: str) -> str | None:
    url = f"{jaeger}/api/traces?service=finsight-agent&operation={operation}&limit=50&lookback=6h"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=20) as response:
        data = json.load(response)["data"]
    if not data:
        return None

    def duration(trace: dict) -> int:
        return max(span["duration"] for span in trace["spans"])

    return max(data, key=duration)["traceID"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grafana", default="http://127.0.0.1:3300")
    parser.add_argument("--jaeger", default="http://127.0.0.1:16686")
    parser.add_argument("--trace-id", default=None)
    parser.add_argument("--range", default="now-30m", help="Grafana time range start.")
    parser.add_argument("--out", default="docs/assets/ops")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace_id = args.trace_id or _slowest_trace(args.jaeger, "finsight.agent.run")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--no-proxy-server"])
        page = browser.new_page(viewport={"width": 1600, "height": 2350}, device_scale_factor=1)
        page.goto(f"{args.grafana}/d/finsight-ops/?orgId=1&from={args.range}&to=now&refresh=", wait_until="networkidle")
        page.wait_for_timeout(6000)
        page.screenshot(path=str(out / "grafana-dashboard.png"), full_page=True)
        print("wrote", out / "grafana-dashboard.png")
        if trace_id:
            page.set_viewport_size({"width": 1600, "height": 1100})
            page.goto(f"{args.jaeger}/trace/{trace_id}", wait_until="networkidle")
            page.wait_for_timeout(4000)
            page.screenshot(path=str(out / "jaeger-trace.png"), full_page=False)
            print("wrote", out / "jaeger-trace.png", "trace", trace_id)
        browser.close()


if __name__ == "__main__":
    main()
