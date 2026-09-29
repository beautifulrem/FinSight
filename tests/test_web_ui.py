"""Browser end-to-end tests of the React chat page (headless Chromium via Playwright).

The page is the committed build in ``query_intelligence/web/dist`` (source: ``frontend/``), served by
the real FastAPI app with a stub NLU service, fake agent tools, and a fake classic LLM client.
"""

from __future__ import annotations

import json
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
from fastapi.responses import StreamingResponse
from playwright.sync_api import Page, expect, sync_playwright

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.api.app import create_app
from query_intelligence.chat.page import DIST_DIR

REPO = Path(__file__).resolve().parents[1]
# axe-core is a frontend devDependency (pinned in frontend/pnpm-lock.yaml); `pnpm install` provides it.
AXE_JS = REPO / "frontend" / "node_modules" / "axe-core" / "axe.min.js"


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


class StreamingAgentService(AgentService):
    """Emits `answer_delta` events before the final answer, like a server that streams LLM tokens.

    Set ``draft`` to the text to stream (it may differ from the final answer, as when verification
    rewrites it); ``pause`` holds the stream mid-way so a test can see the partial text.
    """

    draft: str | None = None
    pause: float = 0.0

    def stream(self, query, **kwargs):
        for event in super().stream(query, **kwargs):
            if event["event"] == "answer" and self.draft:
                chunks = re.findall(r".{1,6}", self.draft, flags=re.S)
                for i, chunk in enumerate(chunks):
                    yield {"event": "answer_delta", "data": {"text": chunk}}
                    time.sleep(self.pause if i == len(chunks) // 2 else 0.02)
            yield event


SLOW_ANSWER = {
    "status": "ok",
    "session_id": "slow",
    "trace_id": "e" * 32,
    "query": "贵州茅台的市盈率是多少",
    "route": "agent",
    "answer": "贵州茅台市盈率约 24.6 倍 [fundamental_600519.SH]。",
    "evidence_used": ["fundamental_600519.SH"],
    "evidence_sources": [
        {
            "evidence_id": "fundamental_600519.SH",
            "kind": "structured",
            "source_type": "fundamental_sql",
            "title": "贵州茅台 fundamentals",
            "as_of": "2026-09-24",
            "payload": {"provenance": {"mode": "live", "is_live": True, "as_of": "2026-09-24", "freshness": "fresh"}},
        }
    ],
    "tool_calls": [
        {
            "tool": "get_fundamentals",
            "arguments": {"target": "贵州茅台"},
            "ok": True,
            "latency_ms": 90,
            "evidence_ids": ["fundamental_600519.SH"],
            "source": "llm",
            "step": 0,
        }
    ],
    "spans": [
        {"node": name, "started_at": 2000.0 + i, "duration_ms": 5.0}
        for i, name in enumerate(
            ["guard_in", "agent_llm", "agent_tools", "agent_llm", "verify", "compliance", "finalize"]
        )
    ],
    "verification": {"passed": True, "checked_numbers": 1},
    "risk_disclaimer": "以上内容仅基于检索到的证据生成，不构成投资建议。",
}


class SlowCannedStream:
    """A canned agent SSE stream that holds before the tools finish and before the first answer token,
    like an LLM run that spends 10-20 s before it writes. The test releases each gate."""

    def __init__(self) -> None:
        self.tools_done = threading.Event()
        self.first_token = threading.Event()

    def events(self):
        def sse(name: str, data: dict) -> str:
            return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        def node(kind: str, name: str) -> str:
            return sse(kind, {"node": name, "label": name})

        yield sse("session", {"session_id": "slow"})
        yield node("node_start", "guard_in")
        yield node("step", "guard_in")
        yield node("node_start", "agent_llm")
        time.sleep(0.3)
        yield node("step", "agent_llm")
        yield sse("tool_call", {"tool": "get_fundamentals", "arguments": '{"target": "贵州茅台"}'})
        yield node("node_start", "agent_tools")
        self.tools_done.wait(20)
        yield node("step", "agent_tools")
        yield sse(
            "tool_result",
            {"tool": "get_fundamentals", "ok": True, "latency_ms": 90, "evidence_ids": ["fundamental_600519.SH"]},
        )
        yield node("node_start", "agent_llm")
        self.first_token.wait(20)
        for chunk in ["贵州茅台市盈率", "约 24.6 倍", " [fundamental_600519.SH]。"]:
            yield sse("answer_delta", {"text": chunk})
            time.sleep(0.15)
        yield node("step", "agent_llm")
        yield node("node_start", "verify")
        time.sleep(0.2)
        yield sse("answer", SLOW_ANSWER)
        yield sse("done", {"session_id": "slow"})


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def agent_service():
    stub = PageStub()
    runtime = AgentRuntime(stub, build_fake_registry(), None, today=lambda: date(2026, 9, 24))
    return StreamingAgentService(runtime, trace_sinks=[])


_APPS: dict = {}


@pytest.fixture(scope="module")
def base_url(agent_service):
    stub = agent_service.runtime.service
    app = create_app(
        service=stub,
        app_config={"ui": {"title": "FinSight UI Test"}},
        deepseek_client=FakeDeepSeek(),
        agent_service=agent_service,
    )

    @app.post("/test/slow-stream")
    def slow_stream() -> StreamingResponse:
        # Test-only endpoint: the browser's /agent/chat/stream request is routed here (see page.route below).
        return StreamingResponse(app.state.slow_stream.events(), media_type="text/event-stream")

    app.state.slow_stream = SlowCannedStream()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}"
    _APPS[url] = app
    yield url
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
    # The Stop next to the progress works from the keyboard, and focus returns to the composer.
    stop = _last_turn(page).locator(".progress-stop")
    stop.focus()
    page.keyboard.press("Enter")
    expect(_last_turn(page).locator(".stopped-note")).to_contain_text(re.compile(r"已停止（用时 \d+ 秒）"))
    expect(page.locator("#query-input")).to_be_focused()
    expect(page.locator("#status-pill")).to_have_text("就绪")
    # The slow handler is still sleeping; wait for it so it does not race the next request.
    page.unroute_all(behavior="wait")

    # The stopped run finishes on the server, so the session is not left locked.
    _ask(page, "贵州茅台的市盈率是多少")
    expect(_last_turn(page).locator(".answer-card")).to_contain_text("24.6", timeout=15000)
    _wait_idle(page)


def test_progress_is_shown_before_the_first_token_and_folds_away_after(page, base_url):
    stream = SlowCannedStream()
    _APPS[base_url].state.slow_stream = stream
    # A canned slow stream stands in for an LLM run that takes 10-20 s before its first token.
    page.route("**/agent/chat/stream", lambda route: route.continue_(url=f"{base_url}/test/slow-stream"))
    try:
        _ask(page, "贵州茅台的市盈率是多少")
        turn = _last_turn(page)
        progress = turn.locator(".run-progress")
        status = turn.locator(".progress-status")

        # 1. Tools running: plain-language step, the tool and its target, elapsed seconds, a skeleton.
        expect(status).to_have_text("查询数据：正在调用数据工具", timeout=15000)
        expect(status).to_have_attribute("role", "status")
        expect(progress.locator('.progress-steps [aria-current="step"]')).to_contain_text("查询数据")
        expect(progress.locator('.progress-steps li[data-state="done"]')).to_have_count(1)
        tool = progress.locator(".progress-tool").first
        expect(tool).to_have_attribute("data-status", "running")
        expect(tool).to_contain_text("get_fundamentals")
        expect(tool).to_contain_text("· 贵州茅台")
        expect(progress.locator(".answer-skeleton")).to_be_visible()
        expect(progress.locator(".progress-elapsed")).to_have_text(re.compile(r"^已用时 \d+ 秒$"))
        expect(progress.locator(".progress-stop")).to_be_visible()
        expect(turn.locator(".streaming-answer")).to_have_count(0)
        expect(turn.locator(".answer-text")).to_have_count(0)
        expect(page.locator("#status-pill")).to_have_text("分析中")
        # The counter ticks without the panel being re-announced: only the status region is live.
        assert progress.evaluate("el => el.closest('[aria-live]').getAttribute('aria-live')") == "off"
        page.wait_for_timeout(1100)
        expect(progress.locator(".progress-elapsed")).not_to_have_text("已用时 0 秒")
        if AXE_JS.is_file():
            _assert_accessible(page, "progress before the first token (tools running)")

        # 2. Tools done, model reading and drafting: still no answer text.
        stream.tools_done.set()
        expect(status).to_have_text("撰写回答：模型正在阅读数据、组织回答", timeout=15000)
        expect(tool).to_have_attribute("data-status", "ok")
        expect(progress.locator('.progress-steps li[data-state="done"]')).to_have_count(2)
        expect(turn.locator(".streaming-answer")).to_have_count(0)

        # 3. First token: the panel folds into the collapsed run trace above the streaming text.
        stream.first_token.set()
        expect(turn.locator(".streaming-answer, .answer-card[data-streamed]").first).to_be_visible(timeout=15000)
        expect(progress).to_have_count(0)

        # 4. Final answer: no progress state is left, the trace is the normal collapsed trace.
        card = turn.locator(".answer-card")
        expect(card).to_contain_text("24.6", timeout=15000)
        _wait_idle(page)
        expect(turn.locator(".run-progress")).to_have_count(0)
        expect(turn.locator(".progress-status")).to_have_count(0)
        expect(turn.locator(".answer-skeleton")).to_have_count(0)
        expect(card.locator(".trace-toggle")).to_have_attribute("aria-expanded", "false")
        expect(card.locator(".trace-toggle")).to_contain_text("执行过程")

        # Client-measured time to first token and total time in the run details.
        aside = page.locator("aside")
        aside.get_by_role("tab", name=re.compile("运行")).click()
        ttft = aside.locator(".run-ttft")
        expect(ttft).to_have_text(re.compile(r"^\d+(\.\d+)? (ms|s)$"))
        expect(aside.locator(".run-wall")).to_have_text(re.compile(r"^\d+(\.\d+)? s$"))
        first, total = (
            float(text.split()[0]) * (1 if text.endswith(" s") else 0.001)
            for text in (ttft.inner_text(), aside.locator(".run-wall").inner_text())
        )
        assert 1.0 <= first < total, (first, total)
        aside.get_by_role("tab", name=re.compile("证据")).click()
    finally:
        stream.tools_done.set()
        stream.first_token.set()
        page.unroute("**/agent/chat/stream")


def test_streamed_answer_shows_a_caret_then_the_verified_answer(page, agent_service):
    # The draft differs from the final answer, as when verification or compliance rewrites it.
    agent_service.draft = "贵州茅台市盈率约 24.6 倍，这是一段流式草稿 [fundamental_600519.SH]，尚未核验。"
    agent_service.pause = 1.5
    try:
        _ask(page, "贵州茅台的市盈率是多少")
        streaming = _last_turn(page).locator(".streaming-answer")
        expect(streaming).to_contain_text("流式草稿", timeout=15000)
        expect(streaming).not_to_contain_text("fundamental_600519")
        caret = page.evaluate(
            "() => { const el = document.querySelector('.streaming-text > p:last-child, p.streaming-text');"
            " return el ? getComputedStyle(el, '::after').content : null; }"
        )
        assert caret not in (None, "none", "normal")

        card = _last_turn(page).locator(".answer-card[data-streamed]")
        expect(card).to_contain_text("24.6", timeout=15000)
        expect(page.locator(".streaming-answer")).to_have_count(0)
        expect(card).to_have_attribute("data-edited", "true")
        expect(card.locator(".answer-edited")).to_be_visible()
        _wait_idle(page)
    finally:
        agent_service.draft = None
        agent_service.pause = 0.0


def test_feedback_is_posted_with_the_trace_id_and_remembered(page):
    card = _last_turn(page).locator(".answer-card")
    page.route(
        "**/agent/feedback",
        lambda route: route.fulfill(status=200, content_type="application/json", body='{"ok": true}'),
    )
    with page.expect_request(re.compile(r"/agent/feedback$")) as first:
        card.locator(".feedback-down").click()
    body = json.loads(first.value.post_data)
    assert body["rating"] == "down" and body["comment"] is None and body["session_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", body["trace_id"])
    expect(card.locator(".feedback-down")).to_have_attribute("aria-pressed", "true")

    card.locator(".feedback-form textarea").fill("数字不对")
    with page.expect_request(re.compile(r"/agent/feedback$")) as second:
        card.locator(".feedback-form").get_by_role("button", name="提交").click()
    assert json.loads(second.value.post_data)["comment"] == "数字不对"
    expect(card.locator(".feedback-status")).to_have_text("感谢反馈")
    stored = page.evaluate("JSON.parse(localStorage.getItem('finsight.feedback'))")
    assert stored[body["trace_id"]]["comment"] == "数字不对"
    page.unroute("**/agent/feedback")


def test_feedback_is_kept_locally_when_the_endpoint_is_missing(page, page_errors):
    # A server without POST /agent/feedback answers FastAPI's default 404.
    page.route(
        "**/agent/feedback",
        lambda route: route.fulfill(status=404, content_type="application/json", body='{"detail": "Not Found"}'),
    )
    turn_id = page.evaluate(
        "() => document.querySelector(\".answer-card .feedback[data-rating='']\").closest('.turn').dataset.turnId"
    )
    card = page.locator(f'.turn[data-turn-id="{turn_id}"] .answer-card')
    card.locator(".feedback-up").click()
    expect(card.locator(".feedback-status")).to_have_text("已保存在本浏览器（服务端暂不接收反馈）")
    expect(card.locator(".feedback-up")).to_have_attribute("aria-pressed", "true")
    page.unroute("**/agent/feedback")
    # Chrome logs every 4xx response to the console; this one is expected and handled by the UI.
    page_errors[:] = [error for error in page_errors if "status of 404" not in error]


def test_export_answer_as_markdown_with_evidence_and_disclaimer(page):
    card = _last_turn(page).locator(".answer-card")
    card.locator(".export-trigger").click()
    with page.expect_download() as info:
        page.get_by_role("menuitem", name="下载 Markdown（.md）").click()
    download = info.value
    assert re.fullmatch(r"finsight-\d{4}-\d{2}-\d{2}-.+\.md", download.suggested_filename)
    text = Path(download.path()).read_text(encoding="utf-8")
    assert text.startswith("# 贵州茅台的市盈率是多少")
    assert "## 证据" in text and "## 风险提示" in text
    assert re.search(r"\*\*E1\*\* .* · `[\w.]+_600519\.SH`", text)
    assert "[E1]" in text or "[E2]" in text


def _sse(events: list[tuple[str, dict]]) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n" for name, data in events)


def _provenance(mode: str, as_of: str, **extra) -> dict:
    return {"mode": mode, "is_live": mode != "snapshot", "as_of": as_of, "freshness": "fresh", **extra}


RAW_CODES = [
    "out_of_scope_query",
    "verification_failed",
    "budget:",
    "removed_trading_instruction",
    "conditional_prefix",
    "language_mismatch",
    "multi_entity:2",
    "question_style:compare",
    "upstream_error",
    "llm_agent",
]

CANNED_ANSWER = {
    "status": "ok",
    "session_id": "canned",
    "trace_id": "f" * 32,
    "query": "贵州茅台和白酒行业的估值",
    "route": "agent",
    "route_reasons": ["mode:agent", "multi_entity:2", "question_style:compare"],
    "answer": "条件性判断：茅台收于 1413 元 [price_600519.SH]，白酒行业市盈率 27.3 倍 [industry_白酒]。",
    "key_points": ["行业数据来自较早的离线快照 [industry_白酒]"],
    "limitations": ["out_of_scope_query", "白酒行业快照截至 2026-04-21 [industry_白酒]"],
    "risk_disclaimer": "以上内容仅基于检索到的证据生成，不构成投资建议。",
    "evidence_used": ["price_600519.SH", "industry_白酒"],
    "evidence_sources": [
        {
            "evidence_id": "price_600519.SH",
            "kind": "structured",
            "source_type": "market_api",
            "title": "贵州茅台 daily market data",
            "source_name": "akshare_sina",
            "as_of": "2026-09-24",
            "produced_by": "get_price_history",
            "payload": {
                "symbol": "600519.SH",
                "name": "贵州茅台",
                "close": 1413.0,
                "pct_change_1d": -1.14,
                "provenance": _provenance(
                    "live_fallback",
                    "2026-09-24",
                    source_label="新浪财经行情",
                    fallback_reason="eastmoney.quote:error(ProxyError); eastmoney.quote:circuit_open",
                ),
            },
        },
        {
            "evidence_id": "industry_白酒",
            "kind": "structured",
            "source_type": "industry_sql",
            "title": "白酒 industry snapshot",
            "as_of": "2026-04-21",
            "produced_by": "get_fundamentals",
            "payload": {
                "industry_name": "白酒",
                "trade_date": "2026-04-21",
                "pe": 27.3,
                "provenance": {
                    **_provenance("snapshot", "2026-04-21"),
                    "freshness": "stale",
                    "source": "offline_snapshot",
                },
            },
        },
        {
            "evidence_id": "aknews_600519.SH_1",
            "kind": "document",
            "source_type": "news",
            "title": "贵州茅台半年报",
            "source_name": "界面新闻",
            "source_url": "https://example.com/news/1",
            "as_of": "2026-08-15T10:11:51",
            "produced_by": "search_news",
        },
    ],
    "tool_calls": [
        {
            "tool": "get_price_history",
            "arguments": {"target": "600519.SH"},
            "ok": True,
            "latency_ms": 120,
            "evidence_ids": ["price_600519.SH"],
            "source": "llm",
            "step": 0,
        },
        {
            "tool": "search_news",
            "arguments": {"query": "白酒"},
            "ok": False,
            "latency_ms": 3000,
            "evidence_ids": [],
            "error": {"code": "upstream_error", "message": "news source failed"},
            "source": "llm",
            "step": 0,
        },
    ],
    "verification": {"passed": False, "checked_numbers": 2},
    "compliance_notes": ["removed_trading_instruction", "conditional_prefix", "language_mismatch_fallback_to_template"],
    "degraded": ["verification_failed:repaired", "budget:step budget of 6 reached"],
    "answer_source": "llm_agent",
    "llm": {
        "model": "fake-model",
        "calls": 2,
        "steps": 1,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_cache_hit_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 15,
        },
    },
    "nlu_summary": {
        "question_style": "compare",
        "product_type": "stock",
        "entities": [],
        "risk_flags": ["investment_advice_like"],
    },
    "spans": [
        {"node": name, "started_at": 1000.0 + i, "duration_ms": 5.0}
        for i, name in enumerate(
            ["guard_in", "agent_llm", "agent_tools", "agent_llm", "verify", "compliance", "finalize"]
        )
    ],
}


def test_codes_are_humanised_and_stale_data_is_flagged(page):
    body = _sse([("session", {"session_id": "canned"}), ("answer", CANNED_ANSWER), ("done", {"session_id": "canned"})])
    page.route(
        "**/agent/chat/stream",
        lambda route: route.fulfill(status=200, headers={"Content-Type": "text/event-stream"}, body=body),
    )
    try:
        _ask(page, "贵州茅台和白酒行业的估值")
        card = _last_turn(page).locator(".answer-card")
        banner = card.locator(".freshness-banner")
        expect(banner).to_have_attribute("data-level", "warn", timeout=15000)
        expect(banner).to_contain_text("1 条离线快照")
        expect(banner).to_contain_text("日频数据日期相差 156 天")
        expect(card.locator(".kpi-tile .kpi-freshness")).to_contain_text("离线快照")

        card.locator(".trace-toggle").click()
        visible = card.inner_text()
        for code in RAW_CODES:
            assert code not in visible, code
        limitation = card.locator('.limitations [data-code="out_of_scope_query"]')
        expect(limitation).to_have_text("问题不属于金融范畴")
        limitation.hover()
        expect(page.get_by_role("tooltip")).to_contain_text("out_of_scope_query")
        page.mouse.move(0, 0)
        expect(card.locator(".trace")).to_contain_text("达到推理步数上限（6 步）")
        expect(card.locator(".trace")).to_contain_text("已删除交易指令")

        aside = page.locator("aside")
        expect(aside.locator('.freshness-badge[data-mode="snapshot"]')).to_have_count(1)
        expect(aside.locator('.freshness-badge[data-mode="fallback"]')).to_have_count(1)
        expect(aside.locator(".stale-badge")).to_have_count(2)  # the snapshot and the 6-week-old news item
        aside.get_by_role("tab", name=re.compile("运行")).click()
        run = aside.locator(".run-details").inner_text()
        for code in RAW_CODES:
            assert code not in run, code
        expect(aside.locator(".run-details")).to_contain_text("涉及 2 只证券")
        aside.get_by_role("tab", name=re.compile("证据")).click()
        if AXE_JS.is_file():
            _assert_accessible(page, "canned answer: freshness banner, snapshot tile, uncited evidence")
    finally:
        page.unroute("**/agent/chat/stream")


CANNED_CLAIM_REPORT = {
    "claim": "茅台市盈率只有15倍，股价昨天跌了5%",
    "verdict": "partially_supported",
    "checks": [
        {
            "target": "贵州茅台",
            "metric": "pe_ttm",
            "claimed": 15.0,
            "actual": 24.6,
            "status": "contradicted",
            "evidence_id": "fundamental_600519.SH",
            "source": "tushare",
            "as_of": "2025-12-31",
            "note": "",
        },
        {
            "target": "贵州茅台",
            "metric": "close",
            "claimed": 1409.5,
            "actual": 1409.5,
            "status": "supported",
            "evidence_id": "price_600519.SH",
            "source": "tushare",
            "as_of": "2026-04-22",
            "note": "",
        },
        {
            "target": "贵州茅台",
            "metric": None,
            "claimed": 5.0,
            "actual": None,
            "status": "unverifiable",
            "evidence_id": None,
            "source": None,
            "as_of": None,
            "note": "metric not recognised",
        },
    ],
    "targets": [{"name": "贵州茅台", "symbol": "600519.SH"}],
    "evidence_sources": [
        {
            "evidence_id": "fundamental_600519.SH",
            "source_name": "tushare",
            "as_of": "2025-12-31",
            "title": "贵州茅台 fundamentals",
        },
        {
            "evidence_id": "price_600519.SH",
            "source_name": "tushare",
            "as_of": "2026-04-22",
            "title": "贵州茅台 daily market data",
        },
    ],
    "disclaimer": "核查只比对声明中的数字与所列数据源，不评价观点本身，也不构成投资建议。",
}

RAW_CLAIM_CODES = [
    "pe_ttm",
    "pct_change_1d",
    "partially_supported",
    "contradicted",
    "unverifiable",
    "metric not recognised",
]


def test_fact_check_view_shows_verdict_values_and_sources(page):
    requests: list[dict] = []

    def fulfill(route):
        requests.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(status=200, json=CANNED_CLAIM_REPORT)

    page.route("**/agent/claim-check", fulfill)
    try:
        page.get_by_role("tab", name="核查").click()
        expect(page.get_by_role("tab", name="核查")).to_have_attribute("aria-selected", "true")
        expect(page.get_by_role("heading", level=1)).to_have_text("核查一条市场说法")
        expect(page.locator("#query-input")).to_be_hidden()
        expect(page.locator(".claim-example")).to_have_count(3)

        page.locator(".claim-example").first.click()
        report = page.locator(".claim-report")
        expect(report.locator(".claim-verdict")).to_have_text("部分相符", timeout=15000)
        expect(report.locator(".claim-verdict")).to_have_attribute("data-verdict", "partially_supported")
        assert requests == [{"claim": "茅台市盈率只有15倍，股价昨天跌了5%", "language": "zh"}]

        contradicted = report.locator('.claim-check[data-status="contradicted"]')
        expect(contradicted.locator(".claim-metric")).to_have_text("市盈率(TTM)")
        expect(contradicted.locator(".claim-status")).to_have_text("不符")
        expect(contradicted.locator(".claim-claimed")).to_have_text("15 倍")
        expect(contradicted.locator(".claim-actual")).to_have_text("24.6 倍")
        expect(contradicted.locator(".claim-source")).to_contain_text("tushare")
        expect(contradicted.locator(".claim-source time")).to_have_attribute("datetime", "2025-12-31")
        expect(contradicted.locator(".claim-source")).to_contain_text("可能过时")
        expect(report.locator(".claim-check")).to_have_count(3)
        expect(report.locator('.claim-check[data-status="unverifiable"]')).to_contain_text(
            "未能判断这个数字指的是哪个指标"
        )
        expect(report.locator(".claim-disclaimer")).to_contain_text("不构成投资建议")
        visible = report.inner_text()
        for code in RAW_CLAIM_CODES:
            assert code not in visible, code

        # A typed claim goes out on Enter; the chat view keeps its state when switching back.
        page.fill("#claim-input", "贵州茅台市盈率约25倍")
        with page.expect_request(re.compile(r"/agent/claim-check$")) as typed:
            page.keyboard.press("Enter")
        assert json.loads(typed.value.post_data)["claim"] == "贵州茅台市盈率约25倍"
        expect(page.locator(".claim-report .claim-verdict")).to_be_visible()
        if AXE_JS.is_file():
            _assert_accessible(page, "fact-check report (light, zh)")
    finally:
        page.unroute("**/agent/claim-check")
        page.get_by_role("tab", name="问答").click()
    expect(page.locator("#query-input")).to_be_visible()
    assert page.locator(".turn").count() > 0


def test_fact_check_errors_are_explained(page, page_errors):
    page.route(
        "**/agent/claim-check",
        lambda route: route.fulfill(status=500, content_type="application/json", body='{"detail": "boom"}'),
    )
    try:
        page.get_by_role("tab", name="核查").click()
        page.fill("#claim-input", "茅台市盈率只有15倍")
        page.keyboard.press("Enter")
        alert = page.locator(".claim-error")
        expect(alert).to_contain_text("请求失败：boom", timeout=15000)
        page.unroute("**/agent/claim-check")
        page.route("**/agent/claim-check", lambda route: route.fulfill(status=200, json=CANNED_CLAIM_REPORT))
        alert.get_by_role("button", name="重试").click()
        expect(page.locator(".claim-report .claim-verdict")).to_have_text("部分相符")
    finally:
        page.unroute("**/agent/claim-check")
        page.get_by_role("tab", name="问答").click()
    # Chrome logs the 500 response to the console; the UI handled it.
    page_errors[:] = [error for error in page_errors if "status of 500" not in error]


def test_compare_answer_shows_kpi_tiles_for_each_company(page):
    """C18: "贵州茅台和五粮液对比一下" shows the same tiles for both companies, not 8 Moutai tiles."""
    page.get_by_role("radio", name="自动").click()
    _ask(page, "贵州茅台和五粮液对比一下")
    card = _last_turn(page).locator(".answer-card")
    expect(card.locator(".kpi-tile").first).to_be_visible(timeout=15000)
    _wait_idle(page)
    texts = card.locator(".kpi-tile").all_inner_texts()
    moutai = [text for text in texts if text.startswith("贵州茅台")]
    wuliangye = [text for text in texts if text.startswith("五粮液")]
    assert len(moutai) == len(wuliangye) >= 2, texts

    # the same metrics, in the same order
    def labels(tiles: list[str]) -> list[str]:
        return [tile.split("\n")[0].split(" · ")[1] for tile in tiles]

    assert labels(moutai) == labels(wuliangye)
    # C17: this turn's data region has its own name
    number = page.locator(".turn").count()
    expect(card.get_by_role("region", name=f"数据（第 {number} 轮）")).to_be_visible()
    if AXE_JS.is_file():
        _assert_accessible(page, "compare answer with KPI tiles")
    _screenshot(page, card, "compare-kpi-tiles.png")


def test_hearsay_move_claim_is_checked_inline_and_in_the_fact_check_view(page):
    """C2 in the UI: "茅台昨天跌超0.1%" (actual -0.18%) is supported, shown as a fall bigger than 0.1%."""
    _ask(page, "听说茅台昨天跌超0.1%，是真的吗")
    card = _last_turn(page).locator(".answer-card")
    inline = card.locator(".inline-fact-check")
    expect(inline).to_be_visible(timeout=15000)
    _wait_idle(page)
    expect(inline.locator(".claim-verdict")).to_have_attribute("data-verdict", "supported")
    expect(inline.locator(".claim-claimed")).to_contain_text("跌幅 > 0.1%")
    expect(inline.locator(".claim-actual")).to_have_text("-0.18%")
    expect(_last_turn(page).locator(".claim-hint")).to_have_count(0)
    _screenshot(page, card, "move-claim-inline.png")

    inline.locator(".inline-fact-check-open").click()
    try:
        report = page.locator(".claim-check-view .claim-report")
        expect(report.locator(".claim-verdict")).to_have_attribute("data-verdict", "supported", timeout=15000)
        expect(report.locator(".claim-claimed")).to_contain_text("跌幅 > 0.1%")
        page.fill("#claim-input", "茅台昨天跌了超过1%")
        page.keyboard.press("Enter")
        expect(report.locator(".claim-verdict")).to_have_attribute("data-verdict", "contradicted", timeout=15000)
        expect(report.locator(".claim-claimed")).to_contain_text("跌幅 > 1%")
        if AXE_JS.is_file():
            _assert_accessible(page, "move-claim card")
        _screenshot(page, report, "move-claim-card.png")
    finally:
        page.get_by_role("tab", name="问答").click()
    expect(page.locator("#query-input")).to_be_visible()


def _screenshot(page: Page, locator, name: str) -> None:
    """Saved only when QI_UI_SHOTS names a directory (the committed shots come from the real Chrome run)."""
    folder = os.environ.get("QI_UI_SHOTS")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        locator.screenshot(path=str(Path(folder) / name))


def test_refusal_limitations_are_human_readable(page):
    _ask(page, "明天天气怎么样")
    card = _last_turn(page).locator(".answer-card")
    expect(card.locator(".route-badge")).to_be_visible(timeout=15000)
    _wait_idle(page)
    assert "out_of_scope_query" not in card.inner_text()
    expect(card.locator('[data-code="out_of_scope_query"]')).to_have_text("问题不属于金融范畴")


def test_only_the_chat_pane_scrolls(page):
    # Visually hidden (sr-only) text is absolutely positioned; it must stay inside the scrolling pane,
    # or the whole document grows and scrollIntoView() shifts the header off screen.
    assert page.locator(".turn").count() > 3
    assert page.evaluate("document.documentElement.scrollHeight <= window.innerHeight + 1")


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


AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"]


def _serious_violations(page: Page) -> list[dict]:
    if not page.evaluate("typeof window.axe !== 'undefined'"):
        page.add_script_tag(path=str(AXE_JS))
    page.wait_for_timeout(700)  # let entrance animations settle so contrast is measured on final colours
    violations = page.evaluate(
        """async (tags) => {
          const options = { runOnly: { type: "tag", values: tags }, resultTypes: ["violations"] };
          const result = await axe.run(document, options);
          return result.violations.map((v) => ({
            id: v.id, impact: v.impact, help: v.help,
            nodes: v.nodes.slice(0, 4).map((n) => n.target.join(" ") + " :: " + (n.failureSummary || "")),
          }));
        }""",
        AXE_TAGS,
    )
    return [v for v in violations if v["impact"] in ("serious", "critical")]


def _assert_accessible(page: Page, state: str) -> None:
    found = _serious_violations(page)
    assert not found, f"{state}:\n" + json.dumps(found, ensure_ascii=False, indent=1)


def _running_in_ci() -> bool:
    return os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}


@pytest.fixture()
def axe_ready():
    if AXE_JS.is_file():
        return
    message = "axe-core not installed: run `pnpm install --frozen-lockfile` in frontend/"
    if _running_in_ci():
        # CI installs the frontend dependencies for this job; a missing axe must not pass as a skip.
        pytest.fail(message)
    pytest.skip(message)


def test_accessibility_desktop_states(browser, base_url, axe_ready):
    errors: list[str] = []
    desk = _open(browser, base_url, errors, viewport={"width": 1440, "height": 900})
    try:
        _assert_accessible(desk, "empty state (light, zh)")

        _ask(desk, "贵州茅台的市盈率是多少")
        card = _last_turn(desk).locator(".answer-card")
        expect(card).to_contain_text("24.6", timeout=15000)
        _wait_idle(desk)
        card.locator(".trace-toggle").click()
        _assert_accessible(desk, "answer with trace and evidence ledger")

        aside = desk.locator("aside")
        aside.get_by_role("tab", name=re.compile("过程")).click()
        _assert_accessible(desk, "inspector trace tab")
        aside.get_by_role("tab", name=re.compile("运行")).click()
        _assert_accessible(desk, "inspector run tab")

        desk.route(
            "**/agent/feedback",
            lambda route: route.fulfill(status=404, content_type="application/json", body='{"detail": "Not Found"}'),
        )
        card.locator(".feedback-down").click()
        expect(card.locator(".feedback-form textarea")).to_be_visible()
        _assert_accessible(desk, "feedback comment form")
        card.locator(".export-trigger").click()
        expect(desk.get_by_role("menu")).to_be_visible()
        _assert_accessible(desk, "export menu open")
        desk.keyboard.press("Escape")

        desk.click("#settings-toggle")
        expect(desk.get_by_role("dialog")).to_be_visible()
        _assert_accessible(desk, "settings dialog")
        desk.keyboard.press("Escape")

        _ask(desk, "明天天气怎么样")
        expect(_last_turn(desk).locator(".answer-card .route-badge")).to_be_visible(timeout=15000)
        _wait_idle(desk)
        _assert_accessible(desk, "refusal")

        desk.get_by_role("radio", name="经典").click()
        _ask(desk, "贵州茅台最近走势")
        expect(_last_turn(desk).locator(".price-chart canvas").first).to_be_visible(timeout=15000)
        _wait_idle(desk)
        _assert_accessible(desk, "price chart and KPI tiles")
        desk.get_by_role("radio", name="自动").click()

        desk.click("#lang-toggle")
        desk.click("#theme-toggle")
        desk.wait_for_function("document.documentElement.classList.contains('dark')")
        _assert_accessible(desk, "dark theme, English")

        desk.click("#new-session")
        _ask(desk, "这只股票能买吗")
        expect(_last_turn(desk).locator(".clarification-card")).to_be_visible(timeout=15000)
        _assert_accessible(desk, "clarification (dark, English)")
        assert [error for error in errors if "status of 404" not in error] == []
    finally:
        desk.context.close()


def test_accessibility_mobile_dark_english(browser, base_url, axe_ready):
    errors: list[str] = []
    context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    context.add_init_script(
        "localStorage.setItem('finsight.lang', 'en'); localStorage.setItem('finsight.theme', 'dark');"
    )
    mobile = context.new_page()
    mobile.on("pageerror", lambda exc: errors.append(str(exc)))
    mobile.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    try:
        mobile.goto(base_url)
        _assert_accessible(mobile, "mobile empty (dark, en)")

        mobile.locator(".example-question").first.click()
        chip = mobile.locator(".answer-card .citation-chip").first
        expect(chip).to_be_visible(timeout=15000)
        _wait_idle(mobile)
        _assert_accessible(mobile, "mobile answer (dark, en)")
        chip.click()
        expect(mobile.get_by_role("dialog")).to_be_visible()
        _assert_accessible(mobile, "mobile evidence sheet (dark, en)")
        mobile.keyboard.press("Escape")

        # The progress panel before the first token, on a phone.
        stream = SlowCannedStream()
        _APPS[base_url].state.slow_stream = stream
        mobile.route("**/agent/chat/stream", lambda route: route.continue_(url=f"{base_url}/test/slow-stream"))
        try:
            _ask(mobile, "What is Kweichow Moutai's P/E?")
            status = _last_turn(mobile).locator(".progress-status")
            expect(status).to_have_text("Get data: Calling data tools", timeout=15000)
            expect(_last_turn(mobile).locator(".progress-tool")).to_contain_text("· 贵州茅台")
            assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            _assert_accessible(mobile, "mobile progress before the first token (dark, en)")
        finally:
            stream.tools_done.set()
            stream.first_token.set()
        expect(_last_turn(mobile).locator(".answer-card")).to_contain_text("24.6", timeout=15000)
        _wait_idle(mobile)
        mobile.unroute("**/agent/chat/stream")

        mobile.route("**/agent/claim-check", lambda route: route.fulfill(status=200, json=CANNED_CLAIM_REPORT))
        mobile.get_by_role("tab", name="Fact-check").click()
        mobile.locator(".claim-example").first.click()
        expect(mobile.locator(".claim-report .claim-verdict")).to_have_text("Partly supported", timeout=15000)
        expect(mobile.locator('.claim-check[data-status="contradicted"] .claim-actual')).to_have_text("24.6×")
        assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        _assert_accessible(mobile, "mobile fact-check report (dark, en)")
        assert errors == []
    finally:
        context.close()
