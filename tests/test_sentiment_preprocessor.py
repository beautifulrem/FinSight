from __future__ import annotations

import pytest

from sentiment.preprocessor import (
    Preprocessor,
    _extract_target_symbols,
    build_entity_names,
    detect_language,
    extract_and_text_level,
    filter_relevant_sentences,
    normalize_text,
    should_skip_query,
    split_sentences,
)
from sentiment.schemas import FilterMeta, PreprocessedDoc

# ===========================================================================
# should_skip_query
# ===========================================================================


def test_should_skip_query_out_of_scope():
    assert should_skip_query({"product_type": {"label": "out_of_scope"}}) == "product_type=out_of_scope"


def test_should_skip_query_product_info():
    assert should_skip_query({"product_type": {"label": "product_info"}}) == "product_type=product_info"


def test_should_skip_query_trading_rule():
    assert should_skip_query({"product_type": {"label": "trading_rule_fee"}}) == "product_type=trading_rule_fee"


def test_should_skip_query_stock_passes():
    assert should_skip_query({"product_type": {"label": "stock"}}) is None


def test_should_skip_query_missing_product_type():
    assert should_skip_query({}) is None


def test_should_skip_query_product_type_not_dict():
    assert should_skip_query({"product_type": "stock"}) is None


# ===========================================================================
# extract_and_text_level
# ===========================================================================


def test_extract_body_full():
    text, level = extract_and_text_level({
        "body": "公司营收增长显著。",
        "body_available": True,
        "title": "标题",
        "summary": "摘要",
    })
    assert text == "公司营收增长显著。"
    assert level == "full"


def test_extract_body_empty_short():
    text, level = extract_and_text_level({
        "body": "",
        "body_available": True,
        "title": "标题",
        "summary": "摘要",
    })
    assert text == ""
    assert level == "short"


def test_extract_body_available_no_fallback():
    """When body_available=True but body is empty/falsy, do NOT fall back."""
    text, level = extract_and_text_level({
        "body": None,
        "body_available": True,
        "title": "标题",
        "summary": "摘要",
    })
    assert text == ""
    assert level == "short"


def test_extract_title_summary_fallback():
    text, level = extract_and_text_level({
        "title": "贵州茅台营收增长",
        "summary": "营收同比增长15%",
        "body_available": False,
    })
    assert "贵州茅台" in text
    assert "15%" in text
    assert level == "short"


def test_extract_title_only():
    text, level = extract_and_text_level({
        "title": "贵州茅台公告",
        "body_available": False,
    })
    assert text == "贵州茅台公告"
    assert level == "short"


def test_extract_no_text():
    text, level = extract_and_text_level({"body_available": False})
    assert text == ""
    assert level == "short"


# ===========================================================================
# normalize_text
# ===========================================================================


def test_normalize_cjk_spaces():
    assert normalize_text("公 司 报 告") == "公司报告"


def test_normalize_multiple_spaces():
    assert normalize_text("hello   world") == "hello world"


def test_normalize_already_clean():
    assert normalize_text("公司正常经营。") == "公司正常经营。"


def test_normalize_empty():
    assert normalize_text("") == ""


# ===========================================================================
# detect_language
# ===========================================================================


def test_detect_zh_pure():
    assert detect_language("贵州茅台发布了最新财报营收大幅增长") == "zh"


def test_detect_en_pure():
    assert detect_language("Apple Inc. reported record quarterly earnings today.") == "en"


def test_detect_mixed():
    assert detect_language("贵州茅台(Moutai)发布了最新财报 revenue grew 15%") == "mixed"


def test_detect_empty():
    assert detect_language("") == "unknown"


def test_detect_number_heavy():
    """Pure digit/punctuation text should not be misclassified as a language."""
    assert detect_language("12345 67890 3.14 52") == "unknown"


def test_detect_short_cjk_only():
    assert detect_language("茅台") == "zh"


def test_detect_stopword_zh_when_ratio_ambiguous():
    """Stopwords break ties when character ratio is below 10% for both scripts."""
    # "的" + 20 digits → CJK ratio = 1/21 ≈ 4.8%, Latin = 0% → both below 10%
    text = "的" + "1" * 20
    assert detect_language(text) == "zh"


def test_detect_stopword_en_when_ratio_ambiguous():
    """English stopwords break ties when character ratio is too low."""
    # "the" + 28 digits → Latin ratio = 3/31 ≈ 9.7%, CJK = 0% → both below 10%
    text = "the" + "1" * 28
    assert detect_language(text) == "en"


def test_detect_stopword_equal_returns_unknown():
    """When zh and en stopword signals are tied, the result is ambiguous.

    Note: when lingua is available and returns a non-None result, it
    takes priority over the stopword heuristic.  The stopword tiebreaker
    is primarily for the pure-fallback path.
    """
    text = "的 the " + "1" * 26  # both ratios below 10%
    result = detect_language(text)
    # If lingua is available and says "en", that's fine.
    # If lingua is unavailable, the stopword tie should be "unknown".
    assert result in ("unknown", "en")


def test_detect_punctuation_signal_zh():
    """Chinese punctuation helps detect zh when character ratio is low."""
    # "。" (U+3002) is a distinctly Chinese ideographic full stop
    text = "。。。。" + "1" * 15  # CJK: 0, Latin: 0, but 4 Chinese punct
    result = detect_language(text)
    # Without stopwords but with Chinese punct, signal=4 → zh
    assert result == "zh"


def test_detect_stopword_and_punct_combined():
    """Stopword + punctuation together provide strong signal."""
    text = "的。" + "1" * 25  # 1 zh stopword + 。punct + digits
    result = detect_language(text)
    assert result == "zh"


# ===========================================================================
# split_sentences
# ===========================================================================


def test_split_zh_basic():
    result = split_sentences("营收增长。利润下降。前景不明。", "zh")
    assert len(result) == 3
    assert "营收增长" in result[0]


def test_split_zh_semicolon():
    result = split_sentences("营收增长；利润下降；", "zh")
    assert len(result) == 2


def test_split_zh_quotes_balanced():
    result = split_sentences('他说"业绩很好。"她回答"确实如此。"', "zh")
    assert len(result) >= 1


def test_split_zh_single_sentence_no_punct():
    """No sentence-ending punctuation present: return as one sentence."""
    result = split_sentences("公司经营稳健，行业前景广阔", "zh")
    assert len(result) == 1


def test_split_en_basic():
    result = split_sentences("Revenue grew. Profits declined!", "en")
    assert len(result) == 2


def test_split_en_empty():
    assert split_sentences("", "en") == []


def test_split_mixed_fallback():
    result = split_sentences("Revenue grew。利润下降。", "mixed")
    assert len(result) >= 1


# ===========================================================================
# build_entity_names
# ===========================================================================


def test_build_entity_cjk_name():
    entity_map = build_entity_names({
        "entities": [
            {"symbol": "600519.SH", "canonical_name": "贵州茅台", "mention": "茅台"},
        ],
    })
    assert "600519.SH" in entity_map
    assert "贵州茅台" in entity_map
    assert "茅台" in entity_map
    # All map to the same symbol
    assert entity_map["贵州茅台"] == "600519.SH"
    assert entity_map.get("茅台") == "600519.SH"
    # jieba token fragments must NOT be added (avoids false positives)
    assert "贵州" not in entity_map


def test_build_entity_non_cjk_no_jieba():
    """Non-CJK names only contribute exact fields — no token expansion."""
    entity_map = build_entity_names({
        "entities": [
            {"symbol": "AAPL", "canonical_name": "Apple Inc.", "mention": "Apple"},
        ],
    })
    assert entity_map == {"AAPL": "AAPL", "Apple Inc.": "AAPL", "Apple": "AAPL"}


def test_build_entity_no_symbol_skipped():
    entity_map = build_entity_names({
        "entities": [
            {"canonical_name": "NoSymbol", "mention": "NS"},
        ],
    })
    assert len(entity_map) == 0


def test_build_entity_empty():
    assert build_entity_names({}) == {}
    assert build_entity_names({"entities": []}) == {}


# ===========================================================================
# filter_relevant_sentences
# ===========================================================================


def test_filter_exact_match():
    sentences = ["贵州茅台营收增长。", "大盘下跌。", "茅台估值合理。"]
    entity_map = {"贵州茅台": "600519.SH", "茅台": "600519.SH"}
    result, symbols = filter_relevant_sentences(sentences, entity_map, "标题")
    # 3-tier: Strong (贵州茅台/茅台) + Generic (大盘下跌, no entity ref)
    assert len(result) == 3
    assert result[0] == "贵州茅台营收增长。"  # Strong first
    assert result[1] == "茅台估值合理。"       # Strong second
    assert result[2] == "大盘下跌。"           # Generic last
    assert "600519.SH" in symbols


def test_filter_no_match_fallback():
    sentences = ["大盘上涨。", "成交量放大。"]
    entity_map = {"贵州茅台": "600519.SH"}
    result, symbols = filter_relevant_sentences(sentences, entity_map, "茅台观察")
    # 3-tier: both are Generic (no entity reference), title NOT prepended
    assert len(result) == 2
    assert result[0] == "大盘上涨。"
    assert symbols == []


def test_filter_empty_entity_map():
    sentences = ["贵州茅台营收增长。"]
    result, symbols = filter_relevant_sentences(sentences, {}, "标题")
    assert result == sentences
    assert symbols == []


def test_filter_empty_sentences():
    result, symbols = filter_relevant_sentences([], {"茅台": "600519.SH"}, "标题")
    assert result == ["标题"]
    assert symbols == []


# ===========================================================================
# _extract_target_symbols
# ===========================================================================


def test_extract_target_symbols_basic():
    symbols = _extract_target_symbols({
        "entities": [
            {"symbol": "600519.SH", "canonical_name": "贵州茅台"},
            {"symbol": "000858.SZ", "canonical_name": "五粮液"},
        ],
    })
    assert symbols == {"600519.SH", "000858.SZ"}


def test_extract_target_symbols_empty():
    assert _extract_target_symbols({}) == set()
    assert _extract_target_symbols({"entities": []}) == set()


def test_extract_target_symbols_skips_missing_symbol():
    symbols = _extract_target_symbols({
        "entities": [
            {"canonical_name": "NoSymbol"},
            {"symbol": "600519.SH"},
        ],
    })
    assert symbols == {"600519.SH"}


# ===========================================================================
# filter_relevant_sentences — resolver path
# ===========================================================================


def _make_test_resolver():
    """Build a minimal EntityResolver for testing."""
    from sentiment.preprocessor import EntityResolver

    entities = [
        {"entity_id": "1", "canonical_name": "贵州茅台", "symbol": "600519.SH", "entity_type": "stock", "normalized_name": "贵州茅台", "exchange": "SSE"},
        {"entity_id": "2", "canonical_name": "五粮液", "symbol": "000858.SZ", "entity_type": "stock", "normalized_name": "五粮液", "exchange": "SZSE"},
    ]
    aliases = [
        {"entity_id": "1", "alias_text": "茅台", "normalized_alias": "茅台", "priority": "1"},
        {"entity_id": "1", "alias_text": "贵州茅台", "normalized_alias": "贵州茅台", "priority": "2"},
        {"entity_id": "2", "alias_text": "五粮液", "normalized_alias": "五粮液", "priority": "2"},
    ]
    return EntityResolver(entities=entities, aliases=aliases)


def test_filter_with_resolver_target_match():
    """Resolver path: sentences mentioning target entities → strong."""
    resolver = _make_test_resolver()
    sentences = ["贵州茅台营收增长。", "大盘下跌。", "五粮液也有表现。"]
    entity_map = {"贵州茅台": "600519.SH", "茅台": "600519.SH", "五粮液": "000858.SZ"}
    target_symbols = {"600519.SH", "000858.SZ"}

    result, symbols = filter_relevant_sentences(
        sentences, entity_map, "标题",
        entity_resolver=resolver, target_symbols=target_symbols,
    )
    # Strong: two sentences hit target entities; Generic: one sentence
    assert len(result) == 3
    assert result[0] == "贵州茅台营收增长。"
    assert result[1] == "五粮液也有表现。"
    assert result[2] == "大盘下跌。"
    assert "600519.SH" in symbols
    assert "000858.SZ" in symbols


def test_filter_with_resolver_catalog_non_target_excluded():
    """Resolver path: sentences mentioning only catalog non-target entities → excluded.

    五粮液 is in the resolver's entity catalog but NOT a target.
    The resolver identifies it and excludes the sentence.
    """
    resolver = _make_test_resolver()
    sentences = ["五粮液发布财报。", "五粮液股价上涨。", "市场整体走强。"]
    entity_map = {"贵州茅台": "600519.SH", "茅台": "600519.SH"}
    target_symbols = {"600519.SH"}

    result, symbols = filter_relevant_sentences(
        sentences, entity_map, "标题",
        entity_resolver=resolver, target_symbols=target_symbols,
    )
    # 五粮液 is resolved but not a target → excluded (both sentences)
    # 市场整体走强 → no entity → generic
    assert result == ["市场整体走强。"]
    assert symbols == []


def test_filter_with_resolver_non_catalog_heuristic_fallback():
    """Hybrid: resolver finds nothing → jieba POS heuristic checks for non-catalog entities.

    万科 is tagged as 'nz' (proper noun) by jieba POS but is NOT in the test resolver.
    The resolver returns [], then the heuristic catches it.
    """
    resolver = _make_test_resolver()
    sentences = ["万科发布财报。", "市场整体走强。"]
    entity_map = {"贵州茅台": "600519.SH", "茅台": "600519.SH"}
    target_symbols = {"600519.SH"}

    result, symbols = filter_relevant_sentences(
        sentences, entity_map, "标题",
        entity_resolver=resolver, target_symbols=target_symbols,
    )
    # 万科 → jieba tags as nz, not in entity_map → excluded
    # 市场整体走强 → no entity → generic
    assert result == ["市场整体走强。"]
    assert symbols == []


def test_filter_with_resolver_mixed_target_and_non_target():
    """Resolver: sentence with BOTH target and non-target entities → strong (kept)."""
    resolver = _make_test_resolver()
    sentences = ["贵州茅台和微软达成合作。"]
    entity_map = {"贵州茅台": "600519.SH"}
    target_symbols = {"600519.SH"}

    result, symbols = filter_relevant_sentences(
        sentences, entity_map, "标题",
        entity_resolver=resolver, target_symbols=target_symbols,
    )
    # 贵州茅台 is a target → hit_target=True → kept as strong
    assert len(result) == 1
    assert "600519.SH" in symbols


def test_filter_with_resolver_no_entities_generic():
    """Resolver path: sentence with no entities at all → generic (kept)."""
    resolver = _make_test_resolver()
    sentences = ["营收持续增长。", "利润显著提升。", "行业发展前景良好。"]
    entity_map = {"贵州茅台": "600519.SH"}
    target_symbols = {"600519.SH"}

    result, symbols = filter_relevant_sentences(
        sentences, entity_map, "标题",
        entity_resolver=resolver, target_symbols=target_symbols,
    )
    # All generic — no entities resolved at all
    assert len(result) == 3
    assert symbols == []


# ===========================================================================
# Preprocessor — resolver integration
# ===========================================================================


def test_preprocessor_with_default_resolver():
    """process_query works when a resolver is injected."""
    resolver = _make_test_resolver()
    preprocessor = Preprocessor(entity_resolver=resolver)

    nlu = {
        "product_type": {"label": "stock"},
        "entities": [
            {"symbol": "600519.SH", "canonical_name": "贵州茅台", "mention": "茅台"},
        ],
    }
    retrieval = {
        "documents": [{
            "evidence_id": "test_001",
            "source_type": "news",
            "title": "测试标题",
            "body": "贵州茅台今日发布公告。公司营收增长显著。五粮液发布竞品信息。",
            "body_available": True,
        }],
    }
    skip_reason, docs, meta = preprocessor.process_query(nlu, retrieval)
    assert skip_reason is None
    assert meta.analyzed_docs_count == 1
    doc = docs[0]
    # 贵州茅台 sentence → strong (target hit)
    # 公司营收... sentence → generic (no entity)
    # 五粮液... sentence → excluded (non-target entity via resolver)
    assert "贵州茅台" in doc.relevant_excerpt
    assert "公司营收" in doc.relevant_excerpt
    assert "五粮液" not in doc.relevant_excerpt
    assert "600519.SH" in doc.entity_hits


def test_preprocessor_without_resolver_uses_fallback():
    """Without a resolver, substring matching is used (backward compatible)."""
    preprocessor = Preprocessor()  # no resolver

    nlu = {
        "product_type": {"label": "stock"},
        "entities": [
            {"symbol": "600519.SH", "canonical_name": "贵州茅台", "mention": "茅台"},
        ],
    }
    retrieval = {
        "documents": [{
            "evidence_id": "test_001",
            "source_type": "news",
            "title": "测试",
            "body": "贵州茅台营收增长。公司表现良好。",
            "body_available": True,
        }],
    }
    skip_reason, docs, meta = preprocessor.process_query(nlu, retrieval)
    assert skip_reason is None
    assert meta.analyzed_docs_count == 1
    assert "600519.SH" in docs[0].entity_hits


# ===========================================================================
# process_query (integration)
# ===========================================================================

NLU_SAMPLE = {
    "product_type": {"label": "stock"},
    "entities": [
        {"symbol": "600519.SH", "canonical_name": "贵州茅台", "mention": "茅台"},
        {"symbol": "000858.SZ", "canonical_name": "五粮液", "mention": "五粮液"},
    ],
}

DOC_NEWS_BODY = {
    "evidence_id": "news_600519.SH_1",
    "source_type": "news",
    "source_name": "证券时报",
    "publish_time": "2026-04-23T20:17:00",
    "title": "贵州茅台营收增长",
    "summary": "茅台业绩超预期",
    "body": "贵州茅台今日发布公告。公司营收增长显著。五粮液也有不错表现。",
    "body_available": True,
    "rank_score": 0.92,
}

DOC_NEWS_SHORT = {
    "evidence_id": "news_000858.SZ_1",
    "source_type": "news",
    "source_name": "财经网",
    "publish_time": "2026-04-23T21:00:00",
    "title": "五粮液动态",
    "summary": "五粮液发布最新财报",
    "body_available": False,
    "rank_score": 0.88,
}

DOC_FAQ = {
    "evidence_id": "faq_001",
    "source_type": "faq",
    "title": "什么是ETF？",
}

RETRIEVAL_SAMPLE = {
    "documents": [DOC_NEWS_BODY, DOC_NEWS_SHORT, DOC_FAQ],
}


def test_process_query_full_pipeline():
    preprocessor = Preprocessor()
    skip_reason, docs, meta = preprocessor.process_query(NLU_SAMPLE, RETRIEVAL_SAMPLE)

    assert skip_reason is None
    assert meta.analyzed_docs_count == 2
    assert meta.skipped_docs_count == 1
    assert meta.short_text_fallback_count == 1
    assert meta.skipped_by_product_type is False

    # First doc (body available)
    doc1 = docs[0]
    assert doc1.evidence_id == "news_600519.SH_1"
    assert doc1.text_level == "full"
    assert doc1.raw_text == normalize_text("贵州茅台今日发布公告。公司营收增长显著。五粮液也有不错表现。")
    assert "600519.SH" in doc1.entity_hits
    assert doc1.relevant_excerpt is not None
    assert len(doc1.relevant_excerpt) > 0
    assert doc1.source_name == "证券时报"
    assert doc1.publish_time == "2026-04-23T20:17:00"
    assert doc1.rank_score == 0.92
    assert doc1.skipped is False

    # The excerpt should contain the sentence about 贵州茅台
    assert "贵州茅台" in doc1.relevant_excerpt

    # Second doc (short text fallback)
    doc2 = docs[1]
    assert doc2.evidence_id == "news_000858.SZ_1"
    assert doc2.text_level == "short"
    assert not doc2.skipped

    # Third doc (faq skipped)
    doc3 = docs[2]
    assert doc3.evidence_id == "faq_001"
    assert doc3.skipped is True
    assert doc3.skip_reason == "unsupported source_type=faq"


def test_process_query_skip_by_product_type():
    preprocessor = Preprocessor()
    nlu = {"product_type": {"label": "out_of_scope"}, "entities": []}
    skip_reason, docs, meta = preprocessor.process_query(nlu, {"documents": []})

    assert skip_reason == "product_type=out_of_scope"
    assert docs == []
    assert meta.skipped_by_product_type is True


def test_process_query_empty_docs():
    preprocessor = Preprocessor()
    skip_reason, docs, meta = preprocessor.process_query(NLU_SAMPLE, {"documents": []})

    assert skip_reason is None
    assert docs == []
    assert meta.analyzed_docs_count == 0


def test_process_query_per_doc_error_isolation():
    """A bad document should not crash the whole batch."""
    preprocessor = Preprocessor()
    bad_doc = {
        "evidence_id": "bad_001",
        "source_type": "news",
        "title": "Bad Doc",
        "body": object(),  # non-string body will cause errors downstream
        "body_available": True,
    }
    retrieval = {"documents": [bad_doc, DOC_NEWS_BODY]}

    skip_reason, docs, meta = preprocessor.process_query(NLU_SAMPLE, retrieval)
    assert skip_reason is None
    # The bad doc is caught and skipped; the good doc is still processed
    assert meta.analyzed_docs_count == 1
    assert meta.skipped_docs_count == 1
    assert docs[0].evidence_id == "news_600519.SH_1"


def test_preprocessed_doc_skipped_has_empty_entity_hits():
    """Skipped docs should have empty entity_hits, not the full entity list."""
    preprocessor = Preprocessor()
    nlu = NLU_SAMPLE
    retrieval = {"documents": [DOC_FAQ]}
    skip_reason, docs, meta = preprocessor.process_query(nlu, retrieval)
    assert len(docs) == 1
    assert docs[0].skipped is True
    assert docs[0].entity_hits == []
