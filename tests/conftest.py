from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Agent traces are written only by tests that configure a sink explicitly.
os.environ.setdefault("QI_AGENT_TRACE_DIR", "off")
# Likewise the audit log file (the audit tests pass a path explicitly).
os.environ.setdefault("QI_AUDIT_LOG_PATH", "off")
# Services built from the environment stay offline unless QI_TEST_LIVE=1: tests must not depend on (or wait
# for) upstream market sites. Tests of the live defaults delete these variables explicitly.
if os.getenv("QI_TEST_LIVE") != "1":
    for _name in ("QI_USE_LIVE_MARKET", "QI_USE_LIVE_MACRO", "QI_USE_LIVE_NEWS", "QI_USE_LIVE_ANNOUNCEMENT"):
        os.environ.setdefault(_name, "false")

# Graph tests script the classic tool loop turn by turn (tool call, then answer, then revision). Pin the
# classic path; tests/test_agent_perf.py covers the latency switches and their defaults explicitly.
for _name, _value in (
    ("QI_AGENT_PREFETCH", "0"),
    ("QI_AGENT_REVISE_POLICY", "llm"),
    ("QI_AGENT_LLM_STALL_TIMEOUT_S", "0"),
    ("QI_AGENT_VERIFY_DERIVED", "0"),
):
    os.environ.setdefault(_name, _value)

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
