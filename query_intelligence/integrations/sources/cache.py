"""TTL cache for live source results with "last known good" reads.

``get`` returns an entry only while it is fresh. ``get_stale`` also returns expired entries (up to
``max_stale_s`` past expiry) so that a failing chain can serve the last successful live result,
explicitly marked as such, instead of dropping straight to the shipped snapshot.
Values are deep-copied on the way in and out because downstream code enriches payloads in place.
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    stored_at: float
    expires_at: float


class SourceCache:
    def __init__(self, *, max_entries: int = 512, clock: Callable[[], float] = time.monotonic) -> None:
        self._data: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._clock = clock

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None or entry.expires_at < self._clock():
                return None
            return copy.deepcopy(entry.value)

    def get_stale(self, key: str, max_stale_s: float) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None or entry.expires_at + max_stale_s < self._clock():
                return None
            return copy.deepcopy(entry.value)

    def put(self, key: str, value: Any, ttl_s: float) -> None:
        with self._lock:
            if key not in self._data and len(self._data) >= self._max_entries:
                oldest = min(self._data, key=lambda k: self._data[k].stored_at)
                self._data.pop(oldest, None)
            now = self._clock()
            self._data[key] = CacheEntry(value=copy.deepcopy(value), stored_at=now, expires_at=now + max(0.0, ttl_s))

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
