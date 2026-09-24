from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Agent traces are written only by tests that configure a sink explicitly.
os.environ.setdefault("QI_AGENT_TRACE_DIR", "off")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def offline_service():
    """Query Intelligence service with every live provider disabled (no network access)."""
    from query_intelligence.service import build_default_service

    return build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
    )
