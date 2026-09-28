"""Tool payload units (round-2 review: amount without unit, ROE as 0.33 vs 29.4) and offline data consistency."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from agent_fakes import build_fake_registry

from query_intelligence.agent.evidence import AgentEvidence
from query_intelligence.agent.tools import ToolOutput
from query_intelligence.agent.tools.units import (
    metric_unit,
    normalise_fundamentals,
    normalise_market,
    normalise_output,
    period_label,
)

ROOT = Path(__file__).resolve().parents[1]
TUSHARE_ROW = {
    "symbol": "600519.SH",
    "close": 1409.5,
    "volume": 26915.93,  # 手
    "amount": 3793827.534,  # 千元
    "pct_change_1d": -0.1778,
    "source": "tushare",
}


def test_tushare_turnover_and_volume_become_cny_and_shares():
    out = normalise_market(TUSHARE_ROW)

    assert out["amount"] == pytest.approx(3_793_827_534.0) and out["amount_unit"] == "CNY"
    assert out["volume"] == pytest.approx(2_691_593.0) and out["volume_unit"] == "share"
    assert out["change_unit"] == "%"
    assert out["units_source"] == {
        "provider": "tushare",
        "amount": "thousand CNY",
        "volume": "lot (100 shares) (from turnover / close)",
    }
    # consistency: turnover ~= close x shares
    assert 0.9 < out["amount"] / (out["close"] * out["volume"]) < 1.1
    assert normalise_market(out) == out  # idempotent (the evaluation replays normalised results)


@pytest.mark.parametrize(
    ("row", "amount", "volume"),
    [
        ({"source": "akshare", "close": 10.0, "amount": 1e8, "volume": 1e5}, 1e8, 1e7),  # 元, 手
        ({"source_name": "sina.kline", "close": 10.0, "amount": 1e8, "volume": 1e7}, 1e8, 1e7),  # 元, 股
        ({"provenance": {"source": "tencent.kline"}, "close": 10.0, "amount": None, "volume": 1e5}, None, 1e7),
        ({"source": "efinance", "volume_unit": "share", "close": 10.0, "amount": 1e8, "volume": 1e7}, 1e8, 1e7),
        # no declared convention (seed ETF row): lots vs shares inferred from turnover ~= close x volume
        ({"close": 4.811, "amount": 4851593064, "volume": 1012592563}, 4851593064, 1012592563),
    ],
)
def test_other_providers(row, amount, volume):
    out = normalise_market(row)

    assert out["amount_unit"] == "CNY" and out["volume_unit"] == "share"
    assert out.get("amount") == (pytest.approx(amount) if amount is not None else None)
    assert out["volume"] == pytest.approx(volume)


def test_fundamentals_get_units_percent_ratios_and_a_period_label():
    data = {
        "report_date": "2025-12-31",
        "metrics": {"roe": 0.33, "pe_ttm": 24.6, "pb": 8.1, "revenue": 168838000000, "netprofit_yoy": -4.53},
        "source": None,
        "industry": {"industry_name": "白酒", "metrics": {"pe": 27.3, "pct_change": -1.05, "turnover": 0.84}},
    }

    out = normalise_fundamentals(data)

    assert out["metrics"]["roe"] == 33.0 and out["units_inferred"] == ["roe"]  # fraction from an unknown source
    assert out["metrics"]["netprofit_yoy"] == -4.53  # growth rates are never rescaled
    assert out["metric_units"] == {"roe": "%", "pe_ttm": "x", "pb": "x", "revenue": "CNY", "netprofit_yoy": "%"}
    assert out["period"] == "FY2025"
    assert out["industry"]["metrics"]["turnover"] == 0.84  # a small rate stays as it is
    assert out["industry"]["metric_units"] == {"pe": "x", "pct_change": "%", "turnover": "%"}
    assert normalise_fundamentals(out) == out


def test_percent_sources_are_never_rescaled():
    out = normalise_fundamentals({"report_date": "2025-09-30", "metrics": {"roe": 0.8}, "source": "tushare"})

    assert out["metrics"]["roe"] == 0.8 and "units_inferred" not in out and out["period"] == "2025Q3"


@pytest.mark.parametrize(
    ("date", "label"),
    [("2024-12-31", "FY2024"), ("20250630", "2025H1"), ("2025-03-31", "2025Q1"), ("2025-09-30", "2025Q3"), ("", None)],
)
def test_period_labels(date, label):
    assert period_label(date) == label


def test_metric_units_cover_the_common_fields():
    assert [metric_unit(name) for name in ("roe", "gross_margin", "revenue_yoy", "pe_ttm", "net_profit", "eps")] == [
        "%",
        "%",
        "%",
        "x",
        "CNY",
        "CNY/share",
    ]


def test_evidence_payloads_are_normalised_with_the_data():
    evidence = AgentEvidence(
        evidence_id="price_600519.SH", kind="structured", source_type="market_api", payload=dict(TUSHARE_ROW)
    )

    output = normalise_output("get_price_history", ToolOutput(data=dict(TUSHARE_ROW), evidence=[evidence]))

    assert output.data["amount_unit"] == "CNY"
    assert output.evidence[0].payload["amount"] == pytest.approx(3_793_827_534.0)
    assert evidence.payload["amount"] == 3793827.534  # the input evidence is not mutated


def test_registry_normalises_every_tool_result():
    registry = build_fake_registry()

    price = registry.run("get_price_history", {"target": "600519.SH"})
    fundamentals = registry.run("get_fundamentals", {"target": "600519.SH"})

    assert price.ok and price.data["amount_unit"] == "CNY" and price.data["change_unit"] == "%"
    assert fundamentals.data["metric_units"]["roe"] == "%" and fundamentals.data["metrics"]["roe"] == 33.0
    assert fundamentals.data["period"] == "FY2025"
    assert fundamentals.evidence[0].payload["roe"] == 33.0


def test_replayed_eval_snapshot_is_normalised():
    from evaluation.agent_eval.replay import ReplayRegistry
    from query_intelligence.agent.tools import ToolSpec
    from query_intelligence.agent.tools.market import PriceHistoryInput

    spec = ToolSpec(name="get_price_history", description="", input_model=PriceHistoryInput, handler=lambda _: None)
    calls = json.loads((ROOT / "evaluation/agent_eval/fixtures/snapshot_v1.json").read_text(encoding="utf-8"))["calls"]
    replay = ReplayRegistry([spec], calls)

    key = next(key for key in calls if key.startswith("get_price_history:") and "600519.SH" in key)
    result = replay.run("get_price_history", json.loads(key.split(":", 1)[1]))

    assert result.ok and result.data["amount"] == pytest.approx(3_793_827_534.0)
    assert result.data["amount_unit"] == "CNY" and result.data["volume_unit"] == "share"
    assert result.evidence[0].payload["amount_unit"] == "CNY"


# ---------------------------------------------------------------- offline data consistency (B25)


def _structured() -> dict:
    return json.loads((ROOT / "data/structured_data.json").read_text(encoding="utf-8"))


def test_fundamental_ratios_are_in_percent_and_in_range():
    for symbol, row in _structured()["fundamental_sql"].items():
        for key, value in row.items():
            if value is None or metric_unit(key) != "%":
                continue
            assert -100 <= value <= 100, (symbol, key, value)
            if key in {"roe", "gross_margin", "grossprofit_margin"}:
                assert abs(value) > 1, f"{symbol} {key}={value} looks like a fraction; the snapshot uses percent"
        assert 0 < row["pe_ttm"] < 500 and 0 < row["pb"] < 100, symbol
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["report_date"]), symbol
        assert row["net_profit"] < row["revenue"], symbol


def test_market_turnover_matches_price_times_volume_after_normalisation():
    for symbol, row in _structured()["market_api"].items():
        out = normalise_market(row)
        if out.get("amount") and out.get("volume") and out.get("close"):
            ratio = out["amount"] / (out["close"] * out["volume"])
            assert 0.8 < ratio < 1.25, (symbol, ratio)


def test_fundamentals_agree_with_the_annual_reports_in_the_document_snapshot():
    """The seed said 1741.2 亿 for FY2025 while the cited 2025 annual-report news says 1688.38 亿."""
    fundamentals = _structured()["fundamental_sql"]
    documents = json.loads((ROOT / "data/documents.json").read_text(encoding="utf-8"))
    pattern = re.compile(
        r"(\d{4})年(?:年度报告|年报)?[^。]{0,20}?营业收入(\d+(?:\.\d+)?)亿元[^。]*?净利润[为]?(\d+(?:\.\d+)?)亿元"
    )
    checked = 0
    for document in documents:
        symbols = {str(item) for item in document.get("symbols") or document.get("entity_symbols") or []}
        text = f"{document.get('summary') or ''}{document.get('body') or ''}"
        for match in pattern.finditer(text):
            year, revenue, profit = match.group(1), float(match.group(2)), float(match.group(3))
            for symbol, row in fundamentals.items():
                if (symbol in symbols or symbol.split(".")[0] in text) and row["report_date"] == f"{year}-12-31":
                    assert row["revenue"] / 1e8 == pytest.approx(revenue, rel=0.01), (symbol, year)
                    assert row["net_profit"] / 1e8 == pytest.approx(profit, rel=0.01), (symbol, year)
                    checked += 1
    assert checked >= 1


def test_eval_snapshots_carry_the_same_fundamentals_as_the_offline_data():
    expected = _structured()["fundamental_sql"]["600519.SH"]
    for name in ("snapshot_v1", "snapshot_holdout_v1", "snapshot_test_v2"):
        text = (ROOT / f"evaluation/agent_eval/fixtures/{name}.json").read_text(encoding="utf-8")
        calls = json.loads(text)["calls"]
        for key, call in calls.items():
            if key.startswith("get_fundamentals:") and "600519" in key and call.get("ok"):
                metrics = call["data"]["metrics"]
                assert (metrics["revenue"], metrics["net_profit"], metrics["roe"]) == (
                    expected["revenue"],
                    expected["net_profit"],
                    expected["roe"],
                ), name


# ---------------------------------------------------------------- sentiment label vs counts (B20)


@pytest.mark.parametrize(
    ("counts", "label"),
    [
        ({"positive": 3}, "positive"),  # the review's case: labelled 中性 from a mean score of 0.55
        ({"neutral": 2, "positive": 1}, "neutral"),
        ({"negative": 2, "neutral": 1}, "negative"),
        ({"positive": 1, "negative": 1}, "neutral"),
        ({}, "neutral"),
    ],
)
def test_sentiment_overall_label_follows_the_counts(counts, label):
    from query_intelligence.agent.tools.sentiment import overall_label

    assert overall_label(counts) == label


def test_recorded_sentiment_with_a_contradicting_label_is_corrected():
    data = {"overall_label": "neutral", "mean_score": 0.5548, "label_counts": {"positive": 3}}
    summary = AgentEvidence(
        evidence_id="sentiment_600519.SH", kind="structured", source_type="sentiment_summary", payload=dict(data)
    )

    output = normalise_output("analyze_sentiment", ToolOutput(data=data, evidence=[summary]))

    assert output.data["overall_label"] == "positive" and output.data["mean_score"] == 0.5548
    assert output.evidence[0].payload["overall_label"] == "positive"
