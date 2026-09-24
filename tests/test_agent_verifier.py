from __future__ import annotations

from query_intelligence.agent.evidence import AgentEvidence, EvidenceStore
from query_intelligence.agent.verifier import cited_ids, claim_numbers, repair_answer, verify_answer


def _store() -> EvidenceStore:
    store = EvidenceStore()
    store.add(
        AgentEvidence(
            evidence_id="price_600519.SH",
            kind="structured",
            source_type="market_api",
            payload={"close": 1409.5, "pct_change_1d": -0.1778, "as_of": "2026-04-22"},
        )
    )
    store.add(
        AgentEvidence(
            evidence_id="fundamental_600519.SH",
            kind="structured",
            source_type="fundamental_sql",
            payload={"revenue": 174120000000, "roe": 0.33, "pe_ttm": 24.6},
        )
    )
    store.add(
        AgentEvidence(
            evidence_id="news_1",
            kind="document",
            source_type="news",
            text_excerpt="2025年度净利润823.20亿元，同比增长4.5%",
        )
    )
    return store


def test_supported_answer_passes_with_unit_conversions_and_ignored_parameters():
    answer = {
        "answer": (
            "贵州茅台最新收盘价 1409.5 元（2026-04-22），日跌幅约 0.18% [price_600519.SH]。"
            "PE(TTM) 为 24.6 倍，ROE 为 33%，营收约 1741.2 亿元 [fundamental_600519.SH]。"
            "新闻显示净利润 823.2 亿元，同比增长 4.5% [news_1]。RSI(14) 与近5日走势需结合更多数据。"
        ),
        "key_points": ["收盘价 1409.5 元", "2025年报净利润同比增长 4.5%"],
        "evidence_used": ["price_600519.SH"],
    }

    report = verify_answer(answer, _store())

    assert report.passed, report
    assert report.cited_ids == ["price_600519.SH", "fundamental_600519.SH", "news_1"]
    assert report.checked_numbers >= 8


def test_invalid_citation_and_unsupported_number_fail():
    answer = {
        "answer": "茅台市盈率为 35 倍 [fundamental_600519.SH]，目标价 2000 元 [made_up_id]。",
        "key_points": [],
        "evidence_used": [],
    }

    report = verify_answer(answer, _store())

    assert not report.passed
    assert report.invalid_citations == ["made_up_id"]
    assert report.unsupported_numbers == [35.0, 2000.0]
    assert "made_up_id" in report.feedback() and "2000" in report.feedback()


def test_numbers_from_the_question_are_allowed():
    answer = {"answer": "若收益率达到 10%，需要更多证据判断 [price_600519.SH]。", "evidence_used": []}

    assert verify_answer(answer, _store(), query="茅台能涨10%吗").passed


def test_missing_citations_fail_when_evidence_exists_but_not_when_empty():
    answer = {"answer": "目前没有足够信息。", "evidence_used": []}

    assert verify_answer(answer, _store()).missing_citations
    assert verify_answer(answer, EvidenceStore()).passed


def test_repair_removes_unsupported_sentences_and_bad_citations():
    answer = {
        "answer": (
            "收盘价 1409.5 元 [price_600519.SH]。目标价 2000 元 [made_up_id]。ROE 为 33% [fundamental_600519.SH]。"
        ),
        "key_points": ["市盈率 99 倍", "收盘价 1409.5 元"],
        "evidence_used": ["made_up_id"],
    }
    store = _store()
    report = verify_answer(answer, store)

    repaired, notes = repair_answer(answer, report, store, zh=True)

    assert "2000" not in repaired["answer"] and "made_up_id" not in repaired["answer"]
    assert "1409.5" in repaired["answer"] and "33%" in repaired["answer"]
    assert repaired["key_points"] == ["收盘价 1409.5 元"]
    assert repaired["evidence_used"] == ["price_600519.SH", "fundamental_600519.SH"]
    assert any("删除" in note for note in notes)
    assert verify_answer(repaired, store).passed


def test_repair_keeps_a_placeholder_when_everything_is_removed():
    answer = {"answer": "目标价 2000 元。", "key_points": [], "evidence_used": []}
    store = _store()

    repaired, _notes = repair_answer(answer, verify_answer(answer, store), store, zh=False)

    assert repaired["answer"].startswith("The available evidence is not enough")
    assert repaired["evidence_used"] == store.ids()[:5]


def test_claim_number_extraction_skips_dates_codes_and_windows():
    text = "2026年4月22日 600519.SH 在 2026-04-22 的 MA20 与 RSI(14) 以及近20个交易日数据，收盘 1409.5，共 3 篇新闻"

    assert claim_numbers(text) == [1409.5]
    assert claim_numbers("In 2025 revenue grew; FY2024 margin 12.5%") == [12.5]
    assert claim_numbers("目标价 2000 元") == [2000.0]


def test_cited_ids_merges_list_and_inline():
    answer = {"answer": "a [x_1] b [y_2]", "key_points": ["c [x_1]"], "evidence_used": ["z_3"]}

    assert cited_ids(answer) == ["z_3", "x_1", "y_2"]


def test_repair_salvages_supported_clauses():
    answer = {
        "answer": "收盘价 1409.5 [price_600519.SH]，目标价 2600 元，市盈率 55 倍 [made_up]。",
        "evidence_used": [],
    }
    store = _store()

    repaired, _notes = repair_answer(answer, verify_answer(answer, store), store, zh=True)

    assert repaired["answer"] == "收盘价 1409.5 [price_600519.SH]。"
    assert verify_answer(repaired, store).passed
