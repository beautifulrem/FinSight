"""Round 13: the corrections layer (data/snapshot/structured_data_fix.json) and field definitions in answers.

The v1 seed's FY2025 statements of 五粮液 and 中国平安 (and 贵州茅台's ROE, P/E, P/B) contradict the companies' annual
reports, and four of its 2026-04-22 bars (中国平安, 沪深300, 创业板ETF, 证券ETF) disagree with the daily record. The
layer replaces them with values transcribed from the primary documents, states every company's field definitions
(ROE weighted / period-end, 营业收入 / 营业总收入, 归母 / total net profit, basic / diluted EPS), and the answers name
the definition next to the figure. The product applies it by default; the evaluation harness and these tests' other
modules stay pinned to the seed (tests/conftest.py, QI_OFFLINE_SNAPSHOT_FIX=false).
"""

from __future__ import annotations

import json

import pytest

from query_intelligence.data_loader import (
    DATA_DIR,
    SNAPSHOT_FIX_PATH,
    apply_snapshot_corrections,
    load_structured_data,
)


def _base() -> dict:
    return json.loads((DATA_DIR / "structured_data.json").read_text(encoding="utf-8"))


# --- the layer and the loader ----------------------------------------------------------------------------------------
def test_the_layer_replaces_the_contradicted_seed_records_and_leaves_the_rest():
    base = _base()
    corrected = load_structured_data(extended=False, corrected=True)
    statement = corrected["fundamental_sql"]["000858.SZ"]
    # 五粮液 2025年年度报告, 第二节 六、主要会计数据和财务指标 (PDF page 7)
    assert statement["revenue"] == 40528509770.23 and statement["net_profit"] == 8954257202.51
    assert statement["roe"] == 6.89 and statement["eps"] == 2.3068
    assert (
        statement["definitions"]["roe_basis"] == "weighted" and statement["definitions"]["net_profit_basis"] == "归母"
    )
    assert corrected["fundamental_sql"]["601318.SH"]["revenue"] == 1050506e6
    assert corrected["fundamental_sql"]["600519.SH"]["roe"] == 32.53
    assert corrected["market_api"]["601318.SH"]["close"] == 58.03
    assert corrected["market_api"]["000300.SH"]["close"] == 4799.63
    assert corrected["market_api"]["000300.SH"]["pct_change_1d"] == 0.66
    # bars the daily record confirms stay the seed's
    for symbol in ("600519.SH", "000858.SZ", "510300.SH"):
        assert corrected["market_api"][symbol] == base["market_api"][symbol]
    for section in ("industry_sql", "macro_sql", "entity_to_industry"):
        assert corrected[section] == base[section]
    assert load_structured_data(extended=False, corrected=False) == base


def test_extension_records_keep_their_values_and_gain_their_definitions():
    ext_only = load_structured_data(extended=True, corrected=False)
    both = load_structured_data(extended=True, corrected=True)
    for symbol, record in ext_only["fundamental_sql"].items():
        if symbol in _base()["fundamental_sql"]:
            continue
        served = both["fundamental_sql"][symbol]
        assert {key: value for key, value in served.items() if key != "definitions"} == record, symbol
        assert set(served["definitions"]) >= {"roe_basis", "revenue_basis", "net_profit_basis", "eps_basis"}
    midea = both["fundamental_sql"]["000333.SZ"]["definitions"]
    assert midea["roe_basis"] == "period_end" and midea["revenue_basis"] == "营业总收入"
    assert midea["eps_basis"] == "diluted" and midea["netprofit_yoy_basis"] == "total"
    # without the extension the layer adds no record for those names
    assert "000333.SZ" not in load_structured_data(extended=False, corrected=True)["fundamental_sql"]


def test_corrections_win_over_the_seed_and_definitions_never_overwrite_a_record_s_own():
    base = {"market_api": {"A": {"close": 1}}, "fundamental_sql": {"S": {"roe": 1}, "T": {"roe": 2, "definitions": {}}}}
    fix = {
        "market_api": {"A": {"close": 2}},
        "fundamental_sql": {},
        "definitions": {"S": {"roe_basis": "weighted"}, "T": {"roe_basis": "weighted"}, "U": {"roe_basis": "x"}},
    }
    merged = apply_snapshot_corrections(base, fix)
    assert merged["market_api"]["A"] == {"close": 2}
    assert merged["fundamental_sql"]["S"]["definitions"] == {"roe_basis": "weighted"}
    assert merged["fundamental_sql"]["T"]["definitions"] == {}
    assert "U" not in merged["fundamental_sql"]
    assert base["market_api"]["A"] == {"close": 1} and "definitions" not in base["fundamental_sql"]["S"]


def test_the_build_is_reproducible_offline_and_every_figure_is_in_its_quote():
    from scripts.build_snapshot_corrections import cmd_verify

    assert cmd_verify(None) == 0


def test_every_correction_names_its_primary_document():
    fix = json.loads(SNAPSHOT_FIX_PATH.read_text(encoding="utf-8"))
    for symbol, record in fix["fundamental_sql"].items():
        meta = record["_snapshot"]
        assert meta["document_url"].startswith("http://static.cninfo.com.cn/finalpage/"), symbol
        assert "2025" in meta["document"] and meta["source"] == "cninfo.annual_report"
    for symbol, record in fix["market_api"].items():
        assert record["trade_date"] == "2026-04-22" and record["_snapshot"]["cross_check"].endswith("agree"), symbol


def test_the_corrections_setting_follows_the_environment(monkeypatch):
    from query_intelligence.config import Settings
    from query_intelligence.data_loader import snapshot_fix_enabled

    monkeypatch.delenv("QI_OFFLINE_SNAPSHOT_FIX", raising=False)
    assert snapshot_fix_enabled() and Settings.from_env().offline_snapshot_fix
    monkeypatch.setenv("QI_OFFLINE_SNAPSHOT_FIX", "0")
    assert not snapshot_fix_enabled() and not Settings.from_env().offline_snapshot_fix


def test_the_evaluation_harness_is_pinned_to_the_seed_unless_asked(monkeypatch):
    from evaluation.agent_eval.runner import build_offline_service, eval_snapshot_fix
    from query_intelligence.agent.tools import build_registry_for_service

    monkeypatch.delenv("QI_EVAL_SNAPSHOT_FIX", raising=False)
    assert not eval_snapshot_fix()
    pinned = build_registry_for_service(build_offline_service())
    seed = _entry(pinned, "get_fundamentals", "000858.SZ")["data"]
    assert seed["metrics"]["roe"] == 29.4 and "definitions" not in seed  # the seed value the labels were made from
    monkeypatch.setenv("QI_EVAL_SNAPSHOT_FIX", "1")
    assert eval_snapshot_fix()


# --- tools and answers -----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fix_registry():
    from query_intelligence.agent.tools import build_registry_for_service
    from query_intelligence.service import build_default_service

    service = build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
        offline_snapshot_fix=True,
    )
    return build_registry_for_service(service)


def _entry(registry, tool: str, target: str) -> dict:
    result = registry.run(tool, {"target": target})
    data = result.data.model_dump(mode="json") if hasattr(result.data, "model_dump") else result.data
    return {"tool": tool, "ok": result.ok, "data": data, "evidence_ids": [item.evidence_id for item in result.evidence]}


def test_corrected_fundamentals_state_their_definitions_and_their_document(fix_registry):
    data = _entry(fix_registry, "get_fundamentals", "五粮液")["data"]
    assert data["metrics"]["revenue"] == 40528509770.23 and data["valuation_date"] == "2026-04-22"
    assert data["definitions"]["roe_basis"] == "weighted" and data["definitions"]["eps_basis"] == "basic"
    provenance = data["provenance"]
    assert provenance["endpoint"] == "data/snapshot/structured_data_fix.json"
    assert provenance["primary_document"] == "宜宾五粮液股份有限公司2025年年度报告"
    assert "数值依据宜宾五粮液股份有限公司2025年年度报告" in provenance["note"]


def test_a_corrected_quote_is_the_daily_record(fix_registry):
    data = _entry(fix_registry, "get_price_history", "中国平安")["data"]
    assert data["close"] == 58.03 and data["pct_change_1d"] == 0.0 and data["as_of"] == "2026-04-22"
    assert data["provenance"]["endpoint"] == "data/snapshot/structured_data_fix.json"


def test_the_template_names_each_figure_s_definition(fix_registry):
    from query_intelligence.agent.composer import compose_template

    log = [_entry(fix_registry, "get_fundamentals", "五粮液")]
    zh = compose_template(log, zh=True)["answer"]
    assert "ROE（加权） 6.89%" in zh and "归母净利润 89.54 亿元" in zh and "营业收入 405.29 亿元" in zh
    en = compose_template(log, zh=False)["answer"]
    assert "ROE (weighted average) 6.89%" in en and "net profit attributable to shareholders" in en
    midea = compose_template([_entry(fix_registry, "get_fundamentals", "美的集团")], zh=True)["answer"]
    assert "ROE（期末净资产） 19.69%" in midea and "营业总收入 4585.02 亿元" in midea


def test_extra_metrics_name_the_eps_and_growth_definitions(fix_registry):
    from query_intelligence.agent.composer import compose_template

    log = [_entry(fix_registry, "get_fundamentals", "美的集团")]
    answer = compose_template(log, zh=True, query="美的集团的每股收益和净利润增速是多少")["answer"]
    assert "稀释每股收益 5.76 元" in answer
    assert "净利润（含少数股东损益）增速 14.8694%" in answer


def test_a_comparison_of_figures_on_different_definitions_says_so(fix_registry):
    from query_intelligence.agent.composer import compose_template

    log = [_entry(fix_registry, "get_fundamentals", "五粮液"), _entry(fix_registry, "get_fundamentals", "宁德时代")]
    answer = compose_template(log, zh=True, query="五粮液和宁德时代的ROE差多少")["answer"]
    assert "五粮液的ROE为加权平均" in answer and "宁德时代的ROE为按期末净资产" in answer
    assert "口径不同，比较仅供参考" in answer
    same = [_entry(fix_registry, "get_fundamentals", "宁德时代"), _entry(fix_registry, "get_fundamentals", "比亚迪")]
    assert (
        "以上ROE均为按期末净资产口径" in compose_template(same, zh=True, query="宁德时代和比亚迪的ROE差多少")["answer"]
    )


def test_without_the_layer_answers_are_unchanged(offline_service):
    from query_intelligence.agent.composer import compose_template
    from query_intelligence.agent.tools import build_registry_for_service

    registry = build_registry_for_service(offline_service)
    entry = _entry(registry, "get_fundamentals", "五粮液")
    assert "definitions" not in entry["data"]
    answer = compose_template([entry], zh=True)["answer"]
    assert "ROE 29.4%" in answer and "营业收入 1085 亿元" in answer and "（加权）" not in answer
