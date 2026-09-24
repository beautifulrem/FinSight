"""Browser end-to-end test of the chat page (headless Chromium via Playwright)."""

from __future__ import annotations

import os
import socket
import threading
import time
from datetime import date
from pathlib import Path

import pytest
import uvicorn
from agent_fakes import StubService, build_fake_registry
from playwright.sync_api import expect, sync_playwright

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app


class PageStub(StubService):
    def analyze_query(self, query, user_profile=None, dialog_context=None, debug=False):
        # Like the real NLU: fall back to an entity mentioned in the dialog context.
        nlu = super().analyze_query(query, user_profile, dialog_context, debug)
        mentioned = any("茅台" in str(item.get("content")) for item in dialog_context or [])
        if not nlu["entities"] and "missing_entity" in nlu["missing_slots"] and mentioned:
            return super().analyze_query(query.replace("这只股票", "贵州茅台"), user_profile, dialog_context, debug)
        return nlu

    def run_pipeline(self, query, user_profile=None, dialog_context=None, top_k=20, debug=False):
        return {
            "nlu_result": self.analyze_query(query),
            "retrieval_result": {"documents": [], "structured_data": [], "warnings": [], "coverage": {}},
        }


class FakeDeepSeek:
    model = "fake"

    def generate(self, record):
        return {
            "answer": f"classic answer to {record['query']}",
            "key_points": ["classic point"],
            "risk_disclaimer": "classic disclaimer",
            "evidence_used": [],
        }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    stub = PageStub()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    app = create_app(
        service=stub,
        app_config={"ui": {"title": "FinSight UI Test"}},
        deepseek_client=FakeDeepSeek(),
        agent_service=AgentService(runtime, trace_sinks=[]),
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def _chromium_path() -> str | None:
    root = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers"))
    candidate = root / "chromium"
    return str(candidate) if candidate.is_file() else None


@pytest.fixture(scope="module")
def page(base_url):
    with sync_playwright() as playwright:
        executable = _chromium_path()
        browser = playwright.chromium.launch(executable_path=executable) if executable else playwright.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        page.goto(base_url)
        yield page
        browser.close()


def _ask(page, text: str) -> None:
    page.fill("#query-input", text)
    page.click("#submit-button")


def test_agent_answer_with_details(page):
    expect(page).to_have_title("FinSight UI Test")
    page.select_option("#mode-select", "auto")

    _ask(page, "贵州茅台的市盈率是多少")

    last = page.locator(".bubble-bot").last
    expect(last).to_contain_text("24.6", timeout=15000)
    expect(last).to_contain_text("Evidence Sources")
    expect(last.locator(".run-details summary")).to_have_text("How this answer was produced")
    last.locator(".run-details summary").click()
    expect(last.locator(".tool-list")).to_contain_text("get_fundamentals")
    expect(last.locator(".next-question").first).to_be_visible()
    expect(page.locator("#status-pill")).to_have_text("Complete")


def test_next_question_chip_fills_input(page):
    chip = page.locator(".bubble-bot").last.locator(".next-question").first
    text = chip.inner_text()
    chip.click()
    expect(page.locator("#query-input")).to_have_value(text)
    page.fill("#query-input", "")


def test_follow_up_uses_session_memory(page):
    # The session already asked about 贵州茅台, so the pronoun is resolved instead of asking back.
    _ask(page, "这只股票能买吗")
    expect(page.locator(".bubble-bot").last).to_contain_text("条件性判断", timeout=15000)


def test_clarification_round_trip(page):
    page.click("#new-session")
    _ask(page, "这只股票能买吗")
    expect(page.locator(".bubble-bot").last).to_contain_text("600519.SH", timeout=15000)
    expect(page.locator("#status-pill")).to_have_text("Waiting for clarification")

    _ask(page, "贵州茅台")

    expect(page.locator(".bubble-bot").last).to_contain_text("Evidence Sources", timeout=15000)


def test_new_session_resets_session_id(page):
    before = page.locator("#session-id").inner_text()
    page.click("#new-session")
    expect(page.locator("#session-id")).not_to_have_text(before)
    expect(page.locator(".bubble-bot").last).to_contain_text("Started a new session")


def test_classic_mode_uses_original_chat_endpoint(page):
    page.select_option("#mode-select", "classic")

    _ask(page, "你好")

    expect(page.locator(".bubble-bot").last).to_contain_text("classic answer to 你好", timeout=15000)
    expect(page.locator(".bubble-bot").last).to_contain_text("classic point")
    page.select_option("#mode-select", "auto")
