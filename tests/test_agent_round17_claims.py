"""Round 17 (review 13, M12a and M13): definitions in a claim, and the missing corrections layer in claim-check (M12b-e:
tests/test_agent_round17_claim_parts.py).

* M12a: a clause that defines a metric ("ROE是净利润与净资产之比", "P/E is the share price divided by earnings per
  share") is checked against the glossary's formula; a false definition makes the claim at most partially supported,
  even when the number next to it is true. A threshold reading of P/B ("市净率低于1说明股价低于每股净资产") is a
  definition too, and its "1" is never a value of the company named later.
* M13: with the corrections layer switched on but its file missing, a report that used offline-snapshot figures says
  so (limitation, disclaimer, ``degraded: data_layer_missing:corrections``), as the chat does.

The wordings, companies and pairings are the engineer's own (none is a review sentence or a slice text; the guard in
tests/test_agent_eval.py checks this). Values from the shipped default data (v1 + the 2026-09-30 extension + the
corrections layer), FY2025 report:

=========  =========  =========  =========  ==========  =====  =====  =====  ======  ======
company    rev YoY %  NP 亿      NP YoY %   扣非 亿     PE     PB     ROE    毛利率  净利率
=========  =========  =========  =========  ==========  =====  =====  =====  ======  ======
宁德时代    17.04      722.01     42.18       645.08     15.85  3.55   21.42  26.27   17.04
比亚迪       3.46      326.19    -18.82       294.46     25.81  3.17   13.24  17.74    4.06
美的集团    12.11      439.45     14.87       412.67     13.77  2.89   19.69  26.39    9.58
紫金矿业    14.96      517.77     62.01       507.24     11.71  3.92   27.91  27.73   14.83
中芯国际    16.49       50.41     36.29        41.24    133.05  5.59    3.34  21.62    7.49
招商银行     0.01     1501.81      1.05      1500.07      6.86  0.91   11.80  –       44.49
工商银行     2.00     3685.62      1.04      3681.26      7.89  0.74    8.68  –       43.97
平安银行   -10.40      426.33     -4.21       426.24      5.17  0.48    7.73  –       32.43
中信证券    28.80      300.76     37.26       303.03     10.19  1.37    9.40  –       40.18
长江电力     2.07      345.03      6.13       334.46     19.29  3.23   15.59  61.67   40.01
贵州茅台    -1.21      823.20     -4.53       822.93     21.44  7.22   32.53  91.18   48.76
五粮液     -54.55       89.54    -71.89        88.16     43.63  3.26    6.89  77.54   22.09
中国平安     2.10     1347.78      6.50      1437.73      7.80  1.05   14.00  –       12.83
=========  =========  =========  =========  ==========  =====  =====  =====  ======  ======
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def checker():
    from evaluation.agent_eval.runner import build_offline_service
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    service = build_offline_service(snapshot_ext=True, snapshot_fix=True)
    registry = build_registry_for_service(service)

    def run(claim: str):
        return check_claim(claim, service=service, registry=registry, zh=not claim.isascii())

    return run


def _rows(report):
    return [
        (check.target, check.kind, check.metric, check.comparator, check.claimed, check.claimed_high, check.actual,
         check.reference, check.reference_metric, check.status, check.reason, check.note[:70])
        for check in report.checks
    ] + [("unchecked", item.text) for item in report.unchecked]  # fmt: skip


# ---- M12a: definitions against the glossary ----------------------------------------------------------------
# A false definition next to a true value: never "supported" (the value is checked and supported, the definition is
# contradicted). The comment names what the stated formula really is.
FALSE_DEFINITION_TRUE_VALUE = [
    "净资产收益率就是净利润除以总资产，宁德时代ROE为21.42%",  # that is ROA
    "市盈率等于股价除以每股净资产，招商银行市盈率6.86倍",  # that is P/B
    "毛利率是净利润占营业收入的比例，贵州茅台毛利率91.18%",  # that is the net margin
    "净利率等于毛利润除以营业收入，美的集团净利率约9.6%",  # that is the gross margin
    "每股收益是净利润除以净资产，长江电力每股收益1.41元",  # that is ROE
    "ROE = 净利润 / 营业收入，中芯国际ROE 3.34%",  # that is the net margin
    "市盈率=每股收益/股价，美的集团市盈率13.77倍",  # inverted (earnings yield)
    "市净率指的是总市值与营业收入之比，五粮液市净率3.26倍",  # that is P/S
    "Return on equity is net income over total revenue; Zijin Mining's ROE is about 27.9%",
    "Gross margin equals net profit divided by revenue, and Kweichow Moutai's gross margin is 91.2%",
    "Price-to-book is market value divided by net profit, and Ping An Bank's P/B is 0.48",
    "市净率低于1说明股价高于每股净资产，平安银行市净率0.48倍",  # below book is a P/B under 1
    "A P/B below 1 means the share price is above book value per share; CMB's P/B is 0.91",
]


def test_the_false_definition_set_has_a_dozen_claims():
    assert len(FALSE_DEFINITION_TRUE_VALUE) >= 12
    assert len(set(FALSE_DEFINITION_TRUE_VALUE)) == len(FALSE_DEFINITION_TRUE_VALUE)


@pytest.mark.parametrize("claim", FALSE_DEFINITION_TRUE_VALUE)
def test_a_false_definition_next_to_a_true_value_is_never_supported(checker, claim):
    report = checker(claim)
    assert report.verdict == "partially_supported", _rows(report)
    definitions = [check for check in report.checks if check.kind == "definition"]
    assert [check.status for check in definitions] == ["contradicted"], _rows(report)
    assert definitions[0].evidence_id and definitions[0].evidence_id.startswith("glossary_"), _rows(report)
    values = [check for check in report.checks if check.kind != "definition"]
    assert values and all(check.status == "supported" for check in values), _rows(report)


@pytest.mark.parametrize(
    "claim",
    [
        "市盈率等于股价除以每股收益，贵州茅台市盈率21.44倍",
        "ROE是净利润与净资产之比，招商银行ROE为11.8%",
        "市净率=总市值/净资产，平安银行市净率0.48倍",
        "P/E is the share price divided by earnings per share; CATL's P/E is about 15.9",
        "毛利率是毛利润占营业收入的比重，五粮液毛利率77.54%",
        "净利率即净利润除以营收，长江电力净利率约40%",
        "市净率小于1意味着股价跌破每股净资产，工商银行市净率0.74倍",
    ],
)
def test_a_true_definition_next_to_a_true_value_is_supported(checker, claim):
    report = checker(claim)
    assert report.verdict == "supported", _rows(report)
    kinds = sorted(check.kind for check in report.checks)
    assert kinds == ["definition", "value"], _rows(report)
    assert report.unchecked == [], _rows(report)


@pytest.mark.parametrize(
    ("claim", "verdict"),
    [
        ("市盈率等于总市值除以营业收入", "contradicted"),  # that is P/S
        ("净资产收益率等于归母净利润除以归母净资产", "supported"),
        ("破净是指市净率高于1", "contradicted"),
        ("所谓破净，就是市净率跌破1", "supported"),
        ("EPS is net profit divided by the number of shares", "supported"),
    ],
)
def test_a_definition_alone_is_checked(checker, claim, verdict):
    report = checker(claim)
    assert report.verdict == verdict, _rows(report)
    assert [check.kind for check in report.checks] == ["definition"], _rows(report)


def test_the_threshold_in_a_definition_is_not_the_companys_value(checker):
    # the "1" belongs to the definition; before round 17 it was read as 工商银行's P/B bound
    report = checker("市净率低于1意味着股价低于每股净资产，工商银行市净率0.74倍")
    values = [(check.target, check.metric, check.claimed) for check in report.checks if check.kind == "value"]
    assert values == [("工商银行", "pb", 0.74)], _rows(report)


def test_a_definition_check_names_the_glossary_formula(checker):
    report = checker("ROE等于净利润除以营业总收入")
    (check,) = report.checks
    assert check.metric == "roe" and check.evidence_id == "glossary_ROE", _rows(report)
    assert "净资产" in check.note and "净利率" in check.note, _rows(report)
    assert any(source["evidence_id"] == "glossary_ROE" for source in report.evidence_sources), report.evidence_sources


# ---- M13: the corrections layer switched on but missing --------------------------------------------------------
@pytest.fixture
def without_corrections_file(monkeypatch, tmp_path):
    from query_intelligence import data_loader

    monkeypatch.setattr(data_loader, "SNAPSHOT_FIX_PATH", tmp_path / "structured_data_fix.json")
    data_loader.clear_data_caches()
    yield
    monkeypatch.undo()
    data_loader.clear_data_caches()


def _seed_checker():
    from evaluation.agent_eval.runner import build_offline_service
    from query_intelligence.agent.claim_check import check_claim
    from query_intelligence.agent.tools.defaults import build_registry_for_service

    service = build_offline_service(snapshot_ext=True, snapshot_fix=True)
    registry = build_registry_for_service(service)
    return lambda claim: check_claim(claim, service=service, registry=registry, zh=not claim.isascii())


def test_a_missing_corrections_layer_is_stated_in_the_report(without_corrections_file):
    from query_intelligence.agent.claim_check import CLAIM_LAYER_NOTE_EN, CLAIM_LAYER_NOTE_ZH

    check = _seed_checker()
    report = check("茅台的市盈率差不多24.6倍")  # the v1 seed value; corrected: 21.44
    assert report.degraded == ["data_layer_missing:corrections"], report.model_dump()
    assert report.limitations == [CLAIM_LAYER_NOTE_ZH] and report.disclaimer.startswith(CLAIM_LAYER_NOTE_ZH)
    assert "种子值" in CLAIM_LAYER_NOTE_ZH
    english = check("Moutai's P/E is roughly 24.6")
    assert english.degraded == ["data_layer_missing:corrections"] and english.limitations == [CLAIM_LAYER_NOTE_EN]
    assert "seed" in CLAIM_LAYER_NOTE_EN


def test_a_report_without_snapshot_figures_has_no_layer_note(without_corrections_file):
    report = _seed_checker()("市盈率等于股价除以每股收益")  # a definition: no market or fundamental record used
    assert report.degraded == [] and report.limitations == [], report.model_dump()


def test_no_layer_note_with_the_layer_present(checker):
    report = checker("贵州茅台的市盈率差不多21.4倍")
    assert report.degraded == [] and report.limitations == [], report.model_dump()
    assert not report.disclaimer.startswith("数据更正层")


def test_the_claim_check_endpoint_reports_the_missing_layer(without_corrections_file, monkeypatch):
    from fastapi.testclient import TestClient

    from query_intelligence.agent.graph import AgentRuntime
    from query_intelligence.agent.llm import ScriptedLLM
    from query_intelligence.agent.service import AgentService
    from query_intelligence.agent.tools.defaults import build_registry_for_service
    from query_intelligence.api.app import create_app
    from query_intelligence.service import build_default_service

    monkeypatch.setenv("QI_AGENT_TRACE_DIR", "off")
    service = build_default_service(
        use_live_market=False,
        use_live_macro=False,
        use_live_news=False,
        use_live_announcement=False,
        offline_snapshot_ext=True,
        offline_snapshot_fix=True,
    )
    runtime = AgentRuntime(service, build_registry_for_service(service), ScriptedLLM([]))
    app = create_app(
        service=service, app_config={"deepseek": {"api_key": ""}}, agent_service=AgentService(runtime, trace_sinks=[])
    )
    try:
        response = TestClient(app).post("/agent/claim-check", json={"claim": "茅台市盈率大约24.6倍"})
    finally:
        runtime.close()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["degraded"] == ["data_layer_missing:corrections"] and body["limitations"], body
