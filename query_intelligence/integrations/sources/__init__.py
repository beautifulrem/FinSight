"""Robust live data acquisition: source catalog, health/circuit breaking, caching, and provenance."""

from .cache import SourceCache
from .catalog import CATALOG, ENDPOINT_SOURCES, SourceInfo, source_for_endpoint, source_label
from .health import CircuitOpenError, SourceHealthRegistry
from .provenance import (
    LAST_KNOWN_GOOD,
    LIVE,
    LIVE_FALLBACK,
    SNAPSHOT,
    build_provenance,
    corpus_provenance,
    freshness,
    snapshot_provenance,
)
from .runtime import (
    AllSourcesFailedError,
    Candidate,
    ChainResult,
    EmptyResultError,
    SourceRuntime,
    SourceTimeoutError,
    get_default_runtime,
    reset_default_runtime,
    runtime_from_settings,
)

__all__ = [
    "CATALOG",
    "ENDPOINT_SOURCES",
    "LAST_KNOWN_GOOD",
    "LIVE",
    "LIVE_FALLBACK",
    "SNAPSHOT",
    "AllSourcesFailedError",
    "Candidate",
    "ChainResult",
    "CircuitOpenError",
    "EmptyResultError",
    "SourceCache",
    "SourceHealthRegistry",
    "SourceInfo",
    "SourceRuntime",
    "SourceTimeoutError",
    "build_provenance",
    "corpus_provenance",
    "freshness",
    "get_default_runtime",
    "reset_default_runtime",
    "runtime_from_settings",
    "snapshot_provenance",
    "source_for_endpoint",
    "source_label",
]
