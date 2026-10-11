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

# QI_TEST_SHIPPED_DEFAULT=1 (round-13 review M17): run the suite on what the product serves by default, both offline
# layers on, without the tests listed in tests/seed_data_tests.txt (they assert v1 seed values; see
# pytest_collection_modifyitems below). CI job tests-shipped and the pre-push step pytest-shipped run
# `QI_TEST_SHIPPED_DEFAULT=1 python -m pytest -q tests`. Without it the two lines below pin the v1 seed.
SHIPPED_DEFAULT = os.getenv("QI_TEST_SHIPPED_DEFAULT") == "1"
if SHIPPED_DEFAULT:
    os.environ["QI_OFFLINE_SNAPSHOT_EXT"] = "true"
    os.environ["QI_OFFLINE_SNAPSHOT_FIX"] = "true"
# The offline snapshot extension (data/snapshot/) adds names that earlier tests use as "no offline data" examples
# (平安银行, 宁德时代): tests built from the environment see the v1 snapshot they were written against, like the
# evaluation harness; tests/test_round12_snapshot_ext.py turns the extension on explicitly.
os.environ.setdefault("QI_OFFLINE_SNAPSHOT_EXT", "false")
# Likewise the corrections layer (data/snapshot/structured_data_fix.json) replaces v1 seed values that earlier tests
# assert (五粮液 ROE 29.4, 中国平安 close 53.61); tests/test_round13_snapshot_fix.py turns it on explicitly.
os.environ.setdefault("QI_OFFLINE_SNAPSHOT_FIX", "false")
# tests/test_shipped_default.py removes both switches for its module and runs the API on the shipped default (both
# layers on), so a regression in the shipped data path fails the suite (round-10 review J1).

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

SEED_DATA_LIST = Path(__file__).with_name("seed_data_tests.txt")


def seed_data_tests() -> set[str]:
    """Node ids (``tests/test_x.py::test_y[param]``) of the tests that assert v1 seed values."""
    lines = SEED_DATA_LIST.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")}


def pytest_configure(config):
    config.addinivalue_line("markers", "seed_data: asserts v1 seed values (listed in tests/seed_data_tests.txt)")


def pytest_collection_modifyitems(config, items):
    listed = seed_data_tests()
    seed, rest = [], []
    for item in items:
        path = Path(str(item.path)).resolve()
        relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        node = "::".join([relative, *item.nodeid.split("::")[1:]])
        if node in listed:
            item.add_marker(pytest.mark.seed_data)
            seed.append(item)
        else:
            rest.append(item)
    if SHIPPED_DEFAULT and seed:
        # on the shipped default these tests fail by construction: deselect them (pytest reports the count)
        config.hook.pytest_deselected(items=seed)
        items[:] = rest


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


@pytest.fixture(scope="session")
def shipped_offline_service():
    """(round 17) The offline service on the shipped default (snapshot extension and corrections layer on), built once
    for every module that needs it."""
    from query_intelligence.service import build_default_service

    return build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
        offline_snapshot_fix=True,
    )
