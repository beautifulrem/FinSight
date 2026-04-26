from __future__ import annotations

from query_intelligence.retrieval.market_analyzer import MarketAnalyzer


def test_rsi_flat_window_is_neutral() -> None:
    closes = [100.0] * 15

    rsi = MarketAnalyzer._rsi(closes, period=14)

    assert rsi == 50.0


def test_rsi_all_gains_is_100() -> None:
    closes = [float(v) for v in range(1, 16)]

    rsi = MarketAnalyzer._rsi(closes, period=14)

    assert rsi == 100.0
