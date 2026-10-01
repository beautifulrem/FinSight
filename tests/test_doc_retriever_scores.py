"""Round 13 (Cost item 1): the transposed TF-IDF matrix is computed once per index, and the scores stay bit-identical
to ``linear_kernel`` (which transposed the whole corpus matrix on every search)."""

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from query_intelligence.retrieval import doc_retriever
from query_intelligence.retrieval.doc_retriever import DocumentRetriever

DOCS = [
    {"title": "贵州茅台年报", "summary": "营收增长", "body": "茅台 营业收入 净利润", "source_type": "news"},
    {"title": "五粮液公告", "summary": "分红", "body": "五粮液 分红 方案", "source_type": "announcement"},
    {"title": "沪深300ETF", "summary": "费率", "body": "管理费 托管费 跟踪误差", "source_type": "product_doc"},
]


def test_scores_are_identical_to_linear_kernel_and_the_transpose_is_shared():
    first = DocumentRetriever(list(DOCS))
    second = DocumentRetriever(list(DOCS))
    assert second._doc_matrix_t is first._doc_matrix_t  # computed once per fitted matrix
    assert first._doc_matrix_t.format == "csr" and first._doc_matrix_t.shape == first.doc_matrix.T.shape
    for query in ("茅台 净利润", "五粮液分红方案", "ETF 费率", "无关的问题"):
        vector = first.vectorizer.transform([query])
        assert np.array_equal(first._scores(vector), linear_kernel(vector, first.doc_matrix).flatten())


def test_scores_match_on_the_shipped_corpus():
    from query_intelligence.retrieval.pipeline import load_documents

    retriever = DocumentRetriever(load_documents())
    for query in ("中国平安 下跌 原因", "CPI PMI 消费板块 影响", "沪深300ETF的费率和最近走势如何"):
        vector = retriever.vectorizer.transform([query])
        assert np.array_equal(retriever._scores(vector), linear_kernel(vector, retriever.doc_matrix).flatten())


def test_transposed_cache_is_bounded():
    doc_retriever._TRANSPOSED.clear()
    for index in range(6):
        DocumentRetriever([{**DOCS[0], "title": f"标题{index}"}, DOCS[1]])
    assert len(doc_retriever._TRANSPOSED) <= doc_retriever._INDEX_MEMO_SIZE + 2
