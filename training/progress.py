from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


class StageProgress:
    def __init__(self, name: str, total_steps: int) -> None:
        self.name = name
        self.total_steps = max(total_steps, 1)
        self.current_step = 0

    def advance(self, label: str) -> None:
        self.current_step = min(self.current_step + 1, self.total_steps)
        print(self._render(label), flush=True)

    def note(self, message: str) -> None:
        print(f"[{self.name}] {message}", flush=True)

    def heartbeat(self, label: str, elapsed_seconds: float, tick: int) -> None:
        print(self._render_heartbeat(label, elapsed_seconds, tick), flush=True)

    def done(self, payload: dict) -> None:
        print(f"[{self.name}] done {json.dumps(payload, ensure_ascii=False)}", flush=True)

    def _render(self, label: str) -> str:
        width = 24
        filled = round(width * self.current_step / self.total_steps)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{self.name}] [{bar}] {self.current_step}/{self.total_steps} {label}"

    def _render_heartbeat(self, label: str, elapsed_seconds: float, tick: int) -> str:
        width = 24
        head = tick % width
        bar = ["-"] * width
        bar[head] = "#"
        elapsed = _format_duration(elapsed_seconds)
        return f"[{self.name}] [{''.join(bar)}] {self.current_step}/{self.total_steps} {label} elapsed={elapsed}"


def module_label(module_name: str) -> str:
    return module_name.split(".")[-1].replace("train_", "")


class ProgressBar:
    def __init__(self, name: str, total: int, *, width: int = 24, every: int = 1) -> None:
        self.name = name
        self.total = max(int(total), 1)
        self.width = max(int(width), 8)
        self.every = max(int(every), 1)
        self.start = time.monotonic()
        self.last_current = 0

    def update(self, current: int, label: str = "running", *, force: bool = False) -> None:
        current = min(max(int(current), 0), self.total)
        if not force and current != self.total and current % self.every != 0:
            return
        self.last_current = current
        elapsed_seconds = time.monotonic() - self.start
        rate = current / elapsed_seconds if elapsed_seconds > 0 else 0.0
        remaining = (self.total - current) / rate if rate > 0 else 0.0
        filled = round(self.width * current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = 100 * current / self.total
        print(
            f"[{self.name}] [{bar}] {current}/{self.total} {percent:5.1f}% {label} "
            f"elapsed={_format_duration(elapsed_seconds)} eta={_format_duration(remaining)}",
            flush=True,
        )

    def finish(self, label: str = "done") -> None:
        self.update(self.total, label, force=True)


def run_with_heartbeat(
    progress: StageProgress,
    label: str,
    func: Callable[[], T],
    *,
    interval_seconds: float = 5.0,
) -> T:
    stop = threading.Event()
    start = time.monotonic()

    def beat() -> None:
        tick = 0
        while not stop.wait(interval_seconds):
            tick += 1
            progress.heartbeat(label, time.monotonic() - start, tick)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        return func()
    finally:
        stop.set()
        thread.join(timeout=0.2)
        progress.note(f"{label} completed in {_format_duration(time.monotonic() - start)}")


def call_with_optional_fit_progress(build_func: Callable[..., T], rows: list[dict], callback: Callable[[int, int], None]) -> T:
    try:
        return build_func(rows, fit_progress_callback=callback)
    except TypeError as exc:
        if "fit_progress_callback" not in str(exc):
            raise
        return build_func(rows)


def _format_duration(seconds: float) -> str:
    whole = max(int(seconds), 0)
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
