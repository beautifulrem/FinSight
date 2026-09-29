"""Intraday (盘中) quotes for 今天/今日/today questions, tested with a frozen Beijing clock and stub providers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from query_intelligence.agent.graph import AgentRuntime
from query_intelligence.agent.service import AgentService
from query_intelligence.agent.tools import ToolContext, build_default_registry
from query_intelligence.integrations.intraday import (
    SHANGHAI_TZ,
    IntradayQuoteError,
    IntradayQuoteProvider,
    asks_about_today,
    market_session,
)

# 2026-09-29 is a Tuesday, 2026-10-03 a Saturday (and inside the National Day closure).
TRADING_MORNING = datetime(2026, 9, 29, 10, 15, tzinfo=SHANGHAI_TZ)
AFTER_CLOSE = datetime(2026, 9, 29, 15, 30, tzinfo=SHANGHAI_TZ)
BEFORE_OPEN = datetime(2026, 9, 29, 9, 0, tzinfo=SHANGHAI_TZ)
WEEKEND = datetime(2026, 10, 3, 10, 15, tzinfo=SHANGHAI_TZ)


class StubQuotes:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def fetch(self, symbol: str, product_type: str = "stock", *, now: datetime | None = None) -> dict:
        self.calls.append((symbol, product_type))
        if self.fail:
            raise IntradayQuoteError("sina.quote: ConnectionError; tencent.qt_quote: ConnectionError")
        return {
            "price": 1452.5,
            "prev_close": 1440.0,
            "pct_change": 0.868,
            "open": 1441.0,
            "high": 1455.0,
            "low": 1438.0,
            "volume": 1234500.0,
            "volume_unit": "share",
            "amount": 1790000000.0,
            "quote_time": "2026-09-29T10:14:57+08:00",
            "endpoint": "https://hq.sinajs.cn/list=sh600519",
            "source": "sina.quote",
            "attempts": ["sina.quote"],
            "fetched_at": "2026-09-29T02:15:00+00:00",
            "fallback_reason": None,
        }


def _agent(offline_service, now: datetime, quotes: StubQuotes | None) -> AgentService:
    context = ToolContext.from_service(offline_service)
    context.intraday_provider = quotes
    context.clock = lambda: now
    runtime = AgentRuntime(offline_service, build_default_registry(context), None, today=lambda: now.date())
    # The offline service has no live market provider; a 今天 question asks for the intraday quote only when
    # live data is on, so switch it on here and let the stub stand in for the real-time source.
    runtime.intraday_quotes = True
    return AgentService(runtime)


def _price_log(response: dict) -> dict:
    calls = [call for call in response["tool_calls"] if call["tool"] == "get_price_history"]
    assert calls, response["tool_calls"]
    return calls[0]


def _price_evidence(response: dict) -> dict:
    return next(item for item in response["evidence_sources"] if item["evidence_id"] == "price_600519.SH")


# ---------------------------------------------------------------- clock and phrasing


@pytest.mark.parametrize(
    ("now", "session"),
    [
        (datetime(2026, 9, 29, 9, 29, tzinfo=SHANGHAI_TZ), "pre_open"),
        (datetime(2026, 9, 29, 9, 30, tzinfo=SHANGHAI_TZ), "morning"),
        (datetime(2026, 9, 29, 11, 45, tzinfo=SHANGHAI_TZ), "lunch_break"),
        (datetime(2026, 9, 29, 13, 0, tzinfo=SHANGHAI_TZ), "afternoon"),
        (datetime(2026, 9, 29, 15, 0, tzinfo=SHANGHAI_TZ), "closed"),
        (WEEKEND, "non_trading_day"),
        (datetime(2026, 10, 1, 10, 0, tzinfo=SHANGHAI_TZ), "non_trading_day"),  # National Day, a Thursday
        (datetime(2026, 9, 29, 2, 15, tzinfo=UTC), "morning"),  # 02:15 UTC is 10:15 in Beijing
    ],
)
def test_market_session_uses_beijing_trading_hours(now, session):
    assert market_session(now) == session


def test_today_phrasing():
    assert asks_about_today("贵州茅台今天股价多少")
    assert asks_about_today("茅台今日涨了吗")
    assert asks_about_today("What is Moutai's price today?")
    assert not asks_about_today("贵州茅台最新股价")
    assert not asks_about_today("贵州茅台的市盈率")


# ---------------------------------------------------------------- the real-time provider (stub HTTP)


class _Response:
    def __init__(self, text: str) -> None:
        self.content = text.encode("gbk")

    def raise_for_status(self) -> None:
        return None


def _sina_line(date: str, clock: str) -> str:
    fields = ["贵州茅台", "1441.00", "1440.00", "1452.50", "1455.00", "1438.00", "1452.40", "1452.50"]
    fields += ["1234500", "1790000000.00"] + ["0"] * 20 + [date, clock, "00"]
    return f'var hq_str_sh600519="{",".join(fields)}";'


def test_provider_parses_sina_with_quote_time():
    urls = []

    def http_get(url, headers, timeout):
        urls.append(url)
        return _Response(_sina_line("2026-09-29", "10:14:57"))

    quote = IntradayQuoteProvider(http_get).fetch("600519.SH", now=TRADING_MORNING)

    assert urls == ["https://hq.sinajs.cn/list=sh600519"]
    assert quote["price"] == 1452.5 and quote["prev_close"] == 1440.0
    assert quote["pct_change"] == pytest.approx(0.8681, abs=1e-4)
    assert quote["quote_time"] == "2026-09-29T10:14:57+08:00"
    assert quote["source"] == "sina.quote" and quote["attempts"] == ["sina.quote"]


def test_provider_falls_back_to_tencent_and_rejects_stale_quotes():
    def http_get(url, headers, timeout):
        if "sinajs" in url:
            return _Response(_sina_line("2026-09-28", "15:00:00"))  # yesterday's quote: rejected
        fields = ["1", "贵州茅台", "600519", "1452.50", "1440.00", "1441.00", "12345"] + ["0"] * 23
        fields += ["20260929101457", "12.50", "0.87", "1455.00", "1438.00", "x", "12345", "179000.00"]
        return _Response(f'v_sh600519="{"~".join(fields)}";')

    quote = IntradayQuoteProvider(http_get).fetch("600519.SH", now=TRADING_MORNING)

    assert quote["source"] == "tencent.qt_quote"
    assert quote["quote_time"] == "2026-09-29T10:14:57+08:00"
    assert quote["amount"] == pytest.approx(1_790_000_000.0)
    assert "not today" in quote["fallback_reason"]

    stale = IntradayQuoteProvider(lambda url, headers, timeout: _Response(_sina_line("2026-09-28", "15:00:00")))
    with pytest.raises(IntradayQuoteError):
        stale.fetch("600519.SH", now=TRADING_MORNING)


# ---------------------------------------------------------------- end to end through the agent (template path)


def test_today_question_in_trading_hours_gets_labelled_intraday_quote(offline_service):
    quotes = StubQuotes()
    response = _agent(offline_service, TRADING_MORNING, quotes).chat("贵州茅台今天股价多少", mode="workflow")

    assert quotes.calls == [("600519.SH", "stock")]
    call = _price_log(response)
    assert call["arguments"].get("intraday") is True
    evidence = _price_evidence(response)
    assert evidence["as_of"] == "2026-09-29T10:14:57+08:00"
    payload = evidence["payload"]
    assert payload["price_basis"] == "intraday"
    assert payload["market_session"] == "morning"
    assert payload["intraday"]["price"] == 1452.5
    provenance = payload["intraday_provenance"]
    assert provenance["source"] == "sina.quote" and provenance["is_live"] is True and provenance["mode"] == "live"
    assert provenance["as_of"] == "2026-09-29T10:14:57+08:00" and provenance["fetched_at"]
    assert "盘中实时价为 1452.5" in response["answer"] and "非收盘价" in response["answer"]
    assert response["verification"]["passed"], response["verification"]
    assert any("盘中实时行情" in item and "不是收盘价" in item for item in response["limitations"])


@pytest.mark.parametrize(
    ("now", "phrase"),
    [(AFTER_CLOSE, "非交易时段"), (BEFORE_OPEN, "非交易时段")],
)
def test_today_question_outside_trading_hours_uses_daily_close_and_says_so(offline_service, now, phrase):
    quotes = StubQuotes()
    response = _agent(offline_service, now, quotes).chat("贵州茅台今天股价多少", mode="workflow")

    assert quotes.calls == []  # no real-time call outside the session
    payload = _price_evidence(response)["payload"]
    assert payload["price_basis"] == "daily_close"
    assert payload["basis_reason"] == "outside_trading_hours"
    assert "最新可用收盘价" in response["answer"]
    note = next(item for item in response["limitations"] if phrase in item)
    assert f"最近交易日收盘价（{payload['as_of'][:10]}）" in note


def test_today_question_on_a_weekend_says_it_is_not_a_trading_day(offline_service):
    quotes = StubQuotes()
    response = _agent(offline_service, WEEKEND, quotes).chat("贵州茅台今天股价多少", mode="workflow")

    assert quotes.calls == []
    payload = _price_evidence(response)["payload"]
    assert payload["price_basis"] == "daily_close" and payload["market_session"] == "non_trading_day"
    assert any("不是 A 股常规交易日" in item for item in response["limitations"])


def test_intraday_source_failure_falls_back_to_daily_close_and_says_so(offline_service):
    quotes = StubQuotes(fail=True)
    response = _agent(offline_service, TRADING_MORNING, quotes).chat("贵州茅台今天股价多少", mode="workflow")

    assert quotes.calls == [("600519.SH", "stock")]
    payload = _price_evidence(response)["payload"]
    assert payload["price_basis"] == "daily_close"
    assert payload["basis_reason"].startswith("intraday_failed")
    assert "最新可用收盘价" in response["answer"]
    assert any("盘中实时行情获取失败" in item for item in response["limitations"])
    assert response["verification"]["passed"]


def test_english_today_question_gets_intraday_quote(offline_service):
    response = _agent(offline_service, TRADING_MORNING, StubQuotes()).chat(
        "What is Kweichow Moutai's share price today?", mode="workflow"
    )

    assert "intraday price is 1452.5" in response["answer"] and "not a close" in response["answer"]
    assert any("intraday quotes" in item for item in response["limitations"])


def test_question_not_about_today_is_unaffected(offline_service):
    quotes = StubQuotes()
    response = _agent(offline_service, TRADING_MORNING, quotes).chat("贵州茅台最近股价多少", mode="workflow")

    assert quotes.calls == []
    call = _price_log(response)
    assert "intraday" not in call["arguments"]
    payload = _price_evidence(response)["payload"]
    assert payload["price_basis"] == "daily_close" and payload["market_session"] is None
    assert payload["basis_reason"] is None
    assert "盘中" not in response["answer"]


def test_offline_runtime_never_requests_intraday(offline_service):
    """Evaluation replays recorded calls: without live data a 今天 question keeps the recorded call key."""
    runtime = AgentRuntime(offline_service, build_default_registry(ToolContext.from_service(offline_service)), None)
    assert runtime.intraday_quotes is False
    assert ToolContext.from_service(offline_service).intraday_provider is None
