"""Browser end-to-end tests of the React chat page (headless Chromium via Playwright).

The page is the committed build in ``query_intelligence/web/dist`` (source: ``frontend/``), served by
the real FastAPI app with a stub NLU service, fake agent tools, and a fake classic LLM client.
"""

from __future__ import annotations

import os
import re
import socket
import threading
import time
from datetime import date
from pathlib import Path

import pytest
import uvicorn
from agent_fakes import StubService, build_fake_registry
from playwright.sync_api import Page, expect, sync_playwright

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app
from query_intelligence.chat.page import DIST_DIR


def test_frontend_build_is_committed():
    # The built app is checked in so Python-only users can run the UI without Node.
    assert (DIST_DIR / "index.html").is_file(), "run `pnpm build` in frontend/"


# Latest-first, like the market providers.
PRICE_HISTORY = [{"trade_date": f"2026-04-{day:02d}", "close": 1380 + day * 1.5} for day in range(22, 2, -1)]


class PageStub(StubService):
    def analyze_query(self, query, user_profile=None, dialog_context=None, debug=False):
        # Like the real NLU: fall back to an entity mentioned in the dialog context.
        nlu = super().analyze_query(query, user_profile, dialog_context, debug)
        mentioned = any("茅台" in str(item.get("content")) for item in dialog_context or [])
        if not nlu["entities"] and "missing_entity" in nlu["missing_slots"] and mentioned:
            return super().analyze_query(query.replace("这只股票", "贵州茅台"), user_profile, dialog_context, debug)
        return nlu

    def run_pipeline(self, query, user_profile=None, dialog_context=None, top_k=20, debug=False):
        structured = []
        if "茅台" in query:
            structured.append(
                {
                    "evidence_id": "price_600519.SH",
                    "source_type": "market_api",
                    "source_name": "tushare",
                    "payload": {
                        "symbol": "600519.SH",
                        "canonical_name": "贵州茅台",
                        "source_name": "tushare",
                        "trade_date": "2026-04-22",
                        "close": 1413.0,
                        "pct_change_1d": 0.42,
                        "history": PRICE_HISTORY,
                    },
                }
            )
        return {
            "nlu_result": self.analyze_query(query),
            "retrieval_result": {"documents": [], "structured_data": structured, "warnings": [], "coverage": {}},
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
def browser():
    with sync_playwright() as playwright:
        executable = _chromium_path()
        options = {"executable_path": executable} if executable else {}
        launched = playwright.chromium.launch(**options)
        yield launched
        launched.close()


def _open(browser, base_url, errors: list[str], **context_options) -> Page:
    context = browser.new_context(**context_options)
    new_page = context.new_page()
    new_page.on("pageerror", lambda exc: errors.append(str(exc)))
    new_page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    new_page.goto(base_url)
    return new_page


@pytest.fixture(scope="module")
def page_errors() -> list[str]:
    return []


@pytest.fixture(scope="module")
def page(browser, base_url, page_errors):
    """One desktop page shared by the conversation tests below (they run in order)."""
    desktop = _open(browser, base_url, page_errors, viewport={"width": 1440, "height": 900})
    yield desktop
    desktop.context.close()


def _ask(page: Page, text: str) -> None:
    page.fill("#query-input", text)
    page.keyboard.press("Enter")


def _last_turn(page: Page):
    return page.locator(".turn").last


def _wait_idle(page: Page) -> None:
    expect(page.locator("#submit-button")).to_be_visible(timeout=15000)


def test_empty_state_and_page_shell(page):
    expect(page).to_have_title("FinSight UI Test")
    expect(page.locator("html")).to_have_attribute("lang", "zh-CN")
    expect(page.get_by_role("heading", level=1)).to_contain_text("A 股问题")
    expect(page.locator(".example-question")).to_have_count(4)
    expect(page.locator("#status-pill")).to_have_text("就绪")
    expect(page.locator(".risk-footer")).to_contain_text("不构成投资建议")
    # The mode switcher is a keyboard-accessible radio group.
    expect(page.get_by_role("radio", name="自动")).to_have_attribute("aria-checked", "true")


def test_agent_answer_with_trace_citations_and_evidence(page):
    page.get_by_role("radio", name="自动").click()
    _ask(page, "贵州茅台的市盈率是多少")

    card = _last_turn(page).locator(".answer-card")
    expect(card).to_contain_text("24.6", timeout=15000)
    _wait_idle(page)
    expect(page.locator("#status-pill")).to_have_text("就绪")
    expect(card.locator(".verification-badge")).to_be_visible()
    expect(card.locator(".disclaimer")).to_be_visible()

    # Collapsible run trace with the tool calls.
    toggle = card.locator(".trace-toggle")
    expect(toggle).to_have_attribute("aria-expanded", "false")
    toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    expect(card.locator(".trace")).to_contain_text("get_fundamentals")
    expect(card.locator(".trace")).to_contain_text("guard_in")

    # Citation chips point into the evidence ledger in the inspector.
    chips = card.locator(".citation-chip")
    expect(chips.first).to_be_visible()
    evidence_id = chips.first.get_attribute("data-evidence-id")
    chips.first.click()
    target = page.locator(f'aside .evidence-item[data-evidence-id="{evidence_id}"]')
    expect(target).to_have_attribute("aria-current", "true")
    expect(page.locator("aside .evidence-item").first).to_contain_text("E1")


def test_inspector_trace_and_run_tabs(page):
    aside = page.locator("aside")
    aside.get_by_role("tab", name=re.compile("过程")).click()
    expect(aside.locator(".waterfall")).to_be_visible()
    expect(aside).to_contain_text("execute_plan")
    aside.get_by_role("tab", name=re.compile("运行")).click()
    expect(aside.locator(".run-details")).to_contain_text("Trace ID")
    expect(aside.locator(".trace-id")).to_have_text(re.compile(r"^[0-9a-f]{32}$"))
    aside.get_by_role("tab", name=re.compile("证据")).click()


def test_next_question_chip_asks_it(page):
    chip = _last_turn(page).locator(".next-question").first
    text = chip.inner_text()
    chip.click()
    expect(_last_turn(page).locator(".user-message")).to_have_text(text)
    expect(_last_turn(page).locator(".answer-card .disclaimer")).to_be_visible(timeout=15000)
    _wait_idle(page)


def test_follow_up_uses_session_memory(page):
    # The session already asked about 贵州茅台, so the pronoun is resolved instead of asking back.
    _ask(page, "这只股票能买吗")
    expect(_last_turn(page).locator(".answer-card")).to_contain_text("条件性判断", timeout=15000)
    _wait_idle(page)


def test_clarification_round_trip(page):
    page.click("#new-session")
    _ask(page, "这只股票能买吗")
    clarification = _last_turn(page).locator(".clarification-card")
    expect(clarification).to_contain_text("600519.SH", timeout=15000)
    expect(page.locator("#status-pill")).to_have_text("等待澄清")
    expect(page.locator(".clarify-banner")).to_be_visible()

    _ask(page, "贵州茅台")

    expect(_last_turn(page)).to_contain_text("正在回答澄清问题")
    expect(_last_turn(page).locator(".answer-card .evidence-count")).to_be_visible(timeout=15000)
    expect(page.locator(".clarify-banner")).to_have_count(0)
    expect(page.locator("#status-pill")).to_have_text("就绪")


def test_settings_show_session_memory_and_new_session_resets_it(page):
    page.click("#settings-toggle")
    dialog = page.get_by_role("dialog")
    expect(dialog.locator(".session-memory li")).to_have_count(1, timeout=10000)
    before = dialog.locator("#session-id").inner_text()
    page.keyboard.press("Escape")
    expect(dialog).to_have_count(0)

    page.click("#new-session")
    expect(page.locator(".notice").last).to_contain_text("已开始新会话")
    page.click("#settings-toggle")
    expect(page.locator("#session-id")).not_to_have_text(before)
    expect(page.get_by_role("dialog")).to_contain_text("本会话还没有已完成的轮次")
    page.keyboard.press("Escape")


def test_classic_mode_uses_original_chat_endpoint_and_charts_prices(page):
    page.get_by_role("radio", name="经典").click()

    _ask(page, "你好")
    card = _last_turn(page).locator(".answer-card")
    expect(card).to_contain_text("classic answer to 你好", timeout=15000)
    expect(card.locator(".key-points")).to_contain_text("classic point")
    _wait_idle(page)

    _ask(page, "贵州茅台最近走势")
    card = _last_turn(page).locator(".answer-card")
    expect(card.locator(".price-chart canvas").first).to_be_visible(timeout=15000)
    expect(card.locator(".price-chart")).to_contain_text("20 个交易日")
    expect(card.locator(".kpi-tile").first).to_contain_text("1,413")
    _wait_idle(page)
    page.get_by_role("radio", name="自动").click()


def test_language_and_theme_toggles(page):
    page.click("#lang-toggle")
    expect(page.locator("html")).to_have_attribute("lang", "en")
    expect(page.locator("#status-pill")).to_have_text("Ready")
    expect(page.get_by_role("radio", name="Auto")).to_be_visible()
    page.click("#lang-toggle")
    expect(page.locator("html")).to_have_attribute("lang", "zh-CN")

    is_dark = "document.documentElement.classList.contains('dark')"
    dark_before = page.evaluate(is_dark)
    page.click("#theme-toggle")
    page.wait_for_function(f"{is_dark} === {str(not dark_before).lower()}")
    expect(page.locator("#theme-toggle")).to_have_attribute("aria-pressed", str(not dark_before).lower())


def test_api_key_setting_is_sent_with_requests(page):
    page.click("#settings-toggle")
    page.fill("#api-key-input", "demo-key-123")
    page.get_by_role("button", name="完成").click()
    with page.expect_request(re.compile(r"/agent/chat/stream$")) as request_info:
        _ask(page, "贵州茅台的市盈率是多少")
    assert request_info.value.headers.get("x-api-key") == "demo-key-123"
    _wait_idle(page)
    page.click("#settings-toggle")
    page.fill("#api-key-input", "")
    page.get_by_role("button", name="完成").click()


def test_stop_detaches_and_the_session_stays_usable(page):
    def slow_stream(route):
        response = route.fetch()
        time.sleep(1.5)
        route.fulfill(response=response)

    page.route("**/agent/chat/stream", slow_stream)
    _ask(page, "贵州茅台的市盈率是多少")
    page.click("#stop-button")
    expect(_last_turn(page)).to_contain_text("已停止")
    expect(page.locator("#status-pill")).to_have_text("就绪")
    page.unroute("**/agent/chat/stream")

    # The stopped run finishes on the server, so the session is not left locked.
    _ask(page, "贵州茅台的市盈率是多少")
    expect(_last_turn(page).locator(".answer-card")).to_contain_text("24.6", timeout=15000)
    _wait_idle(page)


def test_no_browser_errors(page_errors):
    assert page_errors == []


def test_mobile_layout_opens_evidence_sheet(browser, base_url):
    errors: list[str] = []
    mobile = _open(browser, base_url, errors, viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    try:
        expect(mobile.locator("aside")).to_be_hidden()
        mobile.locator(".example-question").first.click()
        chip = mobile.locator(".answer-card .citation-chip").first
        expect(chip).to_be_visible(timeout=15000)
        chip.click()
        sheet = mobile.get_by_role("dialog")
        expect(sheet).to_be_visible()
        expect(sheet.locator('.evidence-item[aria-current="true"]')).to_have_count(1)
        # Nothing overflows horizontally on a phone.
        assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert errors == []
    finally:
        mobile.context.close()
