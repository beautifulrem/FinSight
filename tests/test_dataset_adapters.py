from __future__ import annotations

from pathlib import Path

import pytest

from query_intelligence.external_data.adapters import (
    adapt_baai_finance_instruction_rows,
    adapt_cflue_rows,
    adapt_csprd_rows,
    adapt_dailydialog_rows,
    adapt_financial_news_rows,
    adapt_finfe_rows,
    adapt_fincprg_rows,
    adapt_finnl_rows,
    adapt_fir_bench_announcement_rows,
    adapt_fir_bench_report_rows,
    adapt_msra_rows,
    adapt_mxode_finance_rows,
    adapt_naturalconv_rows,
    adapt_qrecc_rows,
    adapt_risawoz_rows,
    adapt_tnews_rows,
    adapt_t2ranking_rows,
)
from query_intelligence.external_data.label_maps import (
    map_finfe_sentiment,
    map_chnsenticorp_sentiment,
    map_tnews_product_type,
    map_ner_tag_to_ent,
    TNEWS_LABEL_MAP,
    INTENT_KEYWORD_RULES,
    TOPIC_KEYWORD_RULES,
    autolabel_intents,
    autolabel_topics,
    autolabel_question_style,
)
from query_intelligence.external_data.normalize import route_source_to_adapter


def test_map_finfe_sentiment_three_class() -> None:
    assert map_finfe_sentiment(0) == "negative"
    assert map_finfe_sentiment(1) == "neutral"
    assert map_finfe_sentiment(2) == "positive"


def test_map_chnsenticorp_binary() -> None:
    assert map_chnsenticorp_sentiment(0) == "negative"
    assert map_chnsenticorp_sentiment(1) == "positive"


def test_map_tnews_product_type_finance_and_stock() -> None:
    assert map_tnews_product_type(104) == "generic_market"
    assert map_tnews_product_type(114) == "stock"
    assert map_tnews_product_type(103) == "unknown"


def test_tnews_label_map_covers_all_categories() -> None:
    expected_ids = {100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116}
    assert set(TNEWS_LABEL_MAP.keys()) == expected_ids


def test_map_ner_tag_to_ent_keeps_org_drops_per_loc() -> None:
    assert map_ner_tag_to_ent("B-ORG") == "B-ENT"
    assert map_ner_tag_to_ent("I-ORG") == "I-ENT"
    assert map_ner_tag_to_ent("B-PER") == "O"
    assert map_ner_tag_to_ent("B-LOC") == "O"
    assert map_ner_tag_to_ent("O") == "O"


def test_intent_keyword_rules_cover_all_intents() -> None:
    expected_intents = {
        "price_query", "market_explanation", "hold_judgment", "buy_sell_timing",
        "product_info", "risk_analysis", "peer_compare", "fundamental_analysis",
        "valuation_analysis", "macro_policy_impact", "event_news_query", "trading_rule_fee",
    }
    assert set(INTENT_KEYWORD_RULES.keys()) == expected_intents


def test_topic_keyword_rules_cover_all_topics() -> None:
    expected_topics = {
        "price", "news", "industry", "macro", "policy", "fundamentals",
        "valuation", "risk", "comparison", "product_mechanism",
    }
    assert set(TOPIC_KEYWORD_RULES.keys()) == expected_topics


def test_autolabel_intents_matches_keywords() -> None:
    assert "market_explanation" in autolabel_intents("茅台今天为什么跌了")
    assert "peer_compare" in autolabel_intents("茅台和五粮液哪个好")
    assert "trading_rule_fee" in autolabel_intents("这个基金费率多少")


def test_autolabel_topics_matches_keywords() -> None:
    assert "macro" in autolabel_topics("央行降准对市场影响")
    assert "price" in autolabel_topics("股价走势分析")


def test_autolabel_question_style() -> None:
    assert autolabel_question_style("为什么涨了") == "why"
    assert autolabel_question_style("茅台和五粮液比较") == "compare"
    assert autolabel_question_style("建议买入吗") == "advice"
    assert autolabel_question_style("明天会涨吗") == "forecast"
    assert autolabel_question_style("今天收盘价多少") == "fact"
    assert autolabel_question_style("Why did the stock fall?") == "why"
    assert autolabel_question_style("Should I buy this ETF?") == "advice"


def test_route_source_to_adapter_dispatches_known_sources() -> None:
    known_sources = {
        "finfe",
        "chnsenticorp",
        "fin_news_sentiment",
        "msra_ner",
        "peoples_daily_ner",
        "cluener",
        "tnews",
        "thucnews",
        "finnl",
        "t2ranking",
        "csprd",
        "smp2017",
        "mxode_finance",
        "baai_finance_instruction",
        "qrecc",
        "risawoz",
        "naturalconv",
        "dailydialog",
        "fincprg",
        "fir_bench_reports",
        "fir_bench_announcements",
    }
    for source_id in known_sources:
        assert callable(route_source_to_adapter(source_id))


def test_route_source_to_adapter_rejects_unknown_sources() -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        route_source_to_adapter("missing_source")


def test_finfe_adapter_emits_classification_row_for_sentiment() -> None:
    rows = adapt_finfe_rows([{"text": "银行股今天上涨", "label": 2}])

    assert len(rows) == 1
    assert rows[0]["sample_family"] == "classification"
    assert rows[0]["query"] == "银行股今天上涨"
    assert rows[0]["text"] == "银行股今天上涨"
    assert rows[0]["sentiment_label"] == "positive"


def test_msra_adapter_emits_entity_rows_with_ent_tags() -> None:
    rows = adapt_msra_rows(
        [
            {
                "text": "中国平安",
                "tokens": ["中", "国", "平", "安"],
                "tags": ["B-ORG", "I-ORG", "I-ORG", "I-ORG"],
            }
        ]
    )

    assert len(rows) == 1
    assert rows[0]["sample_family"] == "entity"
    assert rows[0]["tags"] == ["B-ENT", "I-ENT", "I-ENT", "I-ENT"]


def test_t2ranking_adapter_collapses_highest_grade_to_two() -> None:
    rows = adapt_t2ranking_rows(
        [
            {
                "query_id": "q-1",
                "query": "贵州茅台",
                "doc_id": "doc-1",
                "doc_text": "贵州茅台公告",
                "relevance": 3,
            }
        ]
    )

    assert len(rows) == 1
    assert rows[0]["sample_family"] == "qrel"
    assert rows[0]["relevance"] == 2


def test_tnews_adapter_keeps_weak_labels_for_non_product_tasks() -> None:
    rows = adapt_tnews_rows([{"sentence": "为什么A股今天下跌", "label": 104}])

    assert len(rows) == 1
    assert rows[0]["sample_family"] == "classification"
    assert rows[0]["product_type"] == "generic_market"
    assert rows[0]["intent_labels"]
    assert rows[0]["topic_labels"]
    assert rows[0]["question_style"] == "why"


def test_tnews_adapter_emits_source_supervision_for_stock_headline() -> None:
    rows = adapt_tnews_rows([{"sentence": "贵州茅台业绩预告，机构上调目标价", "label": 114}])

    assert len(rows) == 1
    row = rows[0]
    assert row["product_type"] == "stock"
    assert "expected_document_sources" in row
    assert "expected_structured_sources" in row
    assert "research_note" in row["expected_document_sources"]
    assert "market_api" in row["expected_structured_sources"]
    assert "expected_document_sources" in row["available_labels"]
    assert "expected_structured_sources" in row["available_labels"]


def test_cflue_adapter_emits_policy_and_faq_source_supervision() -> None:
    rows = adapt_cflue_rows(
        [
            {
                "id": "cflue-1",
                "task": "法规问答",
                "instruction": "请根据最新监管政策解释基金申赎规则",
                "output": "说明申购赎回和监管要求",
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["sample_family"] == "classification"
    assert row["product_type"] == "fund"
    assert "faq" in row["expected_document_sources"]
    assert "product_doc" in row["expected_document_sources"]
    assert "macro_sql" in row["expected_structured_sources"]
    assert "expected_document_sources" in row["available_labels"]


def test_finnl_adapter_emits_source_supervision_for_sequence_payload() -> None:
    rows = adapt_finnl_rows([["贵州茅台发布业绩预告，机构上调目标价", "positive"]])

    assert len(rows) == 1
    row = rows[0]
    assert row["sample_family"] == "classification"
    assert "research_note" in row["expected_document_sources"]
    assert "fundamental_sql" in row["expected_structured_sources"]
    assert "expected_document_sources" in row["available_labels"]
    assert "product_type" in row["available_labels"]


def test_mxode_finance_adapter_emits_full_classification_labels() -> None:
    rows = adapt_mxode_finance_rows(
        [
            {
                "id": "mx-1",
                "prompt": "证券ETF和沪深300指数基金有什么区别？",
                "response": "ETF和指数基金在交易方式上不同。",
                "subset": "Finance-Economics",
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["product_type"] == "etf"
    assert row["question_style"] == "compare"
    assert "product_type" in row["available_labels"]
    assert "expected_document_sources" in row["available_labels"]


def test_baai_finance_instruction_adapter_reads_human_prompt() -> None:
    rows = adapt_baai_finance_instruction_rows(
        [
            {
                "id": "baai-1",
                "conversations": [
                    {"from": "human", "value": "What is the valuation risk of Moutai?"},
                    {"from": "gpt", "value": "Valuation risk is elevated."},
                ],
                "lang": "en",
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["sample_family"] == "classification"
    assert row["question_style"] == "fact"
    assert "valuation_analysis" in row["intent_labels"]


def test_qrecc_adapter_preserves_base_question_style() -> None:
    rows = adapt_qrecc_rows(
        [
            {
                "Conversation_no": 1,
                "Turn_no": 2,
                "Question": "What about its future?",
                "Truth_rewrite": "What is the future outlook for Moutai stock?",
                "Conversation_source": "quac",
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["question_style"] == "forecast"
    assert row["base_question_style"] == "forecast"
    assert "question_style" in row["available_labels"]


def test_risawoz_adapter_emits_question_style_rows_from_dialogue() -> None:
    rows = adapt_risawoz_rows(
        [
            {
                "dialogue_id": "d1",
                "dialogue": [
                    {"turn_id": 0, "user_utterance": "他们家门票是多少？"},
                    {"turn_id": 1, "user_utterance": "谢谢，再见"},
                ],
            }
        ]
    )

    assert len(rows) == 2
    assert all(row["sample_family"] == "classification" for row in rows)
    assert all(row["available_labels"] == ["question_style"] for row in rows)


def test_naturalconv_adapter_emits_ood_only_rows() -> None:
    rows = adapt_naturalconv_rows(
        [
            {
                "dialog_id": "0_1",
                "content": ["你好！", "你是谁？"],
            }
        ]
    )

    assert len(rows) == 2
    assert all(row["available_labels"] == ["out_of_scope_only"] for row in rows)
    assert all(row["product_type"] == "unknown" for row in rows)


def test_dailydialog_adapter_reads_dialogues_txt(tmp_path: Path) -> None:
    raw_dir = tmp_path / "dailydialog"
    train_dir = raw_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "dialogues_train.txt").write_text("Hello there __eou__ Who are you? __eou__\n", encoding="utf-8")

    rows = adapt_dailydialog_rows(raw_dir)

    assert len(rows) == 2
    assert rows[0]["query"] == "Hello there"
    assert rows[1]["available_labels"] == ["out_of_scope_only"]


def test_financial_news_adapter_emits_alias_rows_and_source_supervision() -> None:
    rows = adapt_financial_news_rows(
        [
            {
                "title": "贵州茅台(600519.SH)业绩快报",
                "content": "贵州茅台发布公告",
                "label": 1,
                "company_name": "贵州茅台",
                "symbol": "600519.SH",
                "news_id": "news-1",
            }
        ]
    )

    classification_rows = [row for row in rows if row["sample_family"] == "classification"]
    alias_rows = [row for row in rows if row["sample_family"] == "alias"]

    assert len(classification_rows) == 1
    assert classification_rows[0]["expected_document_sources"]
    assert "news" in classification_rows[0]["expected_document_sources"]
    assert len(alias_rows) >= 2
    assert any(row["alias_text"] == "贵州茅台" for row in alias_rows)
    assert any(row["alias_text"] == "600519.SH" for row in alias_rows)


def test_csprd_adapter_emits_qrels_and_corpus_rows() -> None:
    rows = adapt_csprd_rows(
        [
            {"id": "1", "title": "招股书", "passage": "政策问句", "pos_ctxs": "10,11", "neg_ctxs": "12"},
            {"id": "10", "title": "政策A", "content": "政策A内容"},
        ]
    )

    qrels = [row for row in rows if row["sample_family"] == "qrel"]
    corpus = [row for row in rows if row["sample_family"] == "retrieval_corpus"]

    assert len(qrels) == 3
    assert {row["relevance"] for row in qrels} == {0, 2}
    assert len(corpus) == 1
    assert corpus[0]["doc_id"] == "10"


def test_fincprg_adapter_emits_corpus_and_qrels_from_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "fincprg"
    qrels_dir = raw_dir / "qrels"
    qrels_dir.mkdir(parents=True)
    (raw_dir / "corpus.jsonl").write_text('{"_id":"d1","text":"研报正文"}\n', encoding="utf-8")
    (raw_dir / "queries.jsonl").write_text('{"_id":"q1","text":"道通科技一季度营收是多少"}\n', encoding="utf-8")
    (qrels_dir / "all.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\n", encoding="utf-8")

    rows = adapt_fincprg_rows(raw_dir)

    assert any(row["sample_family"] == "retrieval_corpus" for row in rows)
    assert any(row["sample_family"] == "qrel" and row["relevance"] == 2 for row in rows)


def test_fir_bench_report_and_announcement_adapters_emit_source_typed_corpus() -> None:
    report_rows = adapt_fir_bench_report_rows(
        [{"query": "茅台营收是多少", "title": "研报A", "passage": "营收增长", "label": 1}]
    )
    announcement_rows = adapt_fir_bench_announcement_rows(
        [{"query": "公司公告说了什么", "title": "公告A", "passage": "公告内容", "label": 1}]
    )

    report_corpus = [row for row in report_rows if row["sample_family"] == "retrieval_corpus"]
    announcement_corpus = [row for row in announcement_rows if row["sample_family"] == "retrieval_corpus"]

    assert report_corpus[0]["source_type"] == "research_note"
    assert announcement_corpus[0]["source_type"] == "announcement"


def test_msra_adapter_reads_bio_txt_files(tmp_path: Path) -> None:
    raw_path = tmp_path / "msra_train_bio.txt"
    raw_path.write_text("中\tB-ORG\n国\tI-ORG\n平\tI-ORG\n安\tI-ORG\n\n", encoding="utf-8")

    rows = adapt_msra_rows(raw_path)

    assert len(rows) == 1
    assert rows[0]["text"] == "中国平安"
    assert rows[0]["tags"] == ["B-ENT", "I-ENT", "I-ENT", "I-ENT"]
