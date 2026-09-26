from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    tushare_token: str | None = None
    postgres_dsn: str | None = None
    cninfo_announcement_url: str = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    cninfo_static_base: str = "https://static.cninfo.com.cn/"
    request_timeout_seconds: int = 15
    use_live_market: bool = True
    use_live_macro: bool = True
    use_live_news: bool = True
    use_live_announcement: bool = True
    use_postgres_retrieval: bool = False
    training_dataset_path: str | None = None
    models_dir: str = "models"
    training_manifest_path: str | None = None
    enable_external_data: bool = False
    dataset_allowlist: tuple[str, ...] = ()
    enable_translation: bool = False
    force_refresh_data: bool = False
    # Live source robustness (see docs/data-sources.md).
    source_call_timeout_seconds: float = 10.0
    source_failure_threshold: int = 3
    source_cooldown_seconds: float = 60.0
    source_max_cooldown_seconds: float = 600.0
    source_cache_enabled: bool = True
    source_max_stale_seconds: float = 24 * 3600.0
    # Worker pool for guarded upstream calls (bounds threads left behind by hung calls).
    source_max_workers: int = 32
    # Active probing for ``/sources/health?probe=1``: minimum seconds between two probe rounds.
    source_probe_min_interval_seconds: float = 60.0
    # Cross-check live fundamentals between Sina and THS (one extra upstream call per stock).
    source_cross_check_fundamentals: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        live_market_raw = os.getenv("QI_USE_LIVE_MARKET", "true")
        return cls(
            tushare_token=os.getenv("TUSHARE_TOKEN"),
            postgres_dsn=os.getenv("QI_POSTGRES_DSN"),
            cninfo_announcement_url=os.getenv(
                "CNINFO_ANNOUNCEMENT_URL", "https://www.cninfo.com.cn/new/hisAnnouncement/query"
            ),
            cninfo_static_base=os.getenv("CNINFO_STATIC_BASE", "https://static.cninfo.com.cn/"),
            request_timeout_seconds=int(os.getenv("QI_HTTP_TIMEOUT_SECONDS", "15")),
            use_live_market=live_market_raw.lower() in {"1", "true", "yes"},
            use_live_macro=os.getenv("QI_USE_LIVE_MACRO", "true").lower() in {"1", "true", "yes"},
            use_live_news=os.getenv("QI_USE_LIVE_NEWS", live_market_raw).lower() in {"1", "true", "yes"},
            use_live_announcement=os.getenv("QI_USE_LIVE_ANNOUNCEMENT", live_market_raw).lower()
            in {"1", "true", "yes"},
            use_postgres_retrieval=os.getenv("QI_USE_POSTGRES_RETRIEVAL", "").lower() in {"1", "true", "yes"},
            training_dataset_path=os.getenv("QI_TRAINING_DATASET"),
            models_dir=os.getenv("QI_MODELS_DIR", "models"),
            training_manifest_path=os.getenv("QI_TRAINING_MANIFEST"),
            enable_external_data=os.getenv("QI_ENABLE_EXTERNAL_DATA", "").lower() in {"1", "true", "yes"},
            dataset_allowlist=tuple(s.strip() for s in os.getenv("QI_DATASET_ALLOWLIST", "").split(",") if s.strip()),
            enable_translation=os.getenv("QI_ENABLE_TRANSLATION", "").lower() in {"1", "true", "yes"},
            force_refresh_data=os.getenv("QI_FORCE_REFRESH_DATA", "").lower() in {"1", "true", "yes"},
            source_call_timeout_seconds=_env_float("QI_SOURCE_CALL_TIMEOUT_SECONDS", 10.0),
            source_failure_threshold=max(1, int(_env_float("QI_SOURCE_FAILURE_THRESHOLD", 3))),
            source_cooldown_seconds=_env_float("QI_SOURCE_COOLDOWN_SECONDS", 60.0),
            source_max_cooldown_seconds=_env_float("QI_SOURCE_MAX_COOLDOWN_SECONDS", 600.0),
            source_cache_enabled=os.getenv("QI_SOURCE_CACHE", "true").lower() in {"1", "true", "yes"},
            source_max_stale_seconds=_env_float("QI_SOURCE_MAX_STALE_SECONDS", 24 * 3600.0),
            source_max_workers=max(1, int(_env_float("QI_SOURCE_MAX_WORKERS", 32))),
            source_probe_min_interval_seconds=_env_float("QI_SOURCE_PROBE_MIN_INTERVAL_SECONDS", 60.0),
            source_cross_check_fundamentals=os.getenv("QI_SOURCE_CROSS_CHECK", "true").lower() in {"1", "true", "yes"},
        )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default
