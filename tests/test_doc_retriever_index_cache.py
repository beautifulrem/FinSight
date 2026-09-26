from query_intelligence.retrieval import doc_retriever
from query_intelligence.retrieval.doc_retriever import DocumentRetriever

DOCS = [
    {"title": "贵州茅台年报", "summary": "营收增长", "body": "茅台 营业收入 净利润", "source_type": "news"},
    {"title": "五粮液公告", "summary": "分红", "body": "五粮液 分红 方案", "source_type": "announcement"},
]


def test_index_is_reused_for_the_same_corpus_and_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("QI_TFIDF_CACHE_DIR", str(tmp_path))
    doc_retriever._INDEX_MEMO.clear()

    first = DocumentRetriever(list(DOCS))
    second = DocumentRetriever(list(DOCS))

    assert second.vectorizer is first.vectorizer
    assert len(list(tmp_path.glob("tfidf-*.joblib"))) == 1

    # A new process (empty memo) loads the persisted index instead of refitting it.
    doc_retriever._INDEX_MEMO.clear()
    reloaded = DocumentRetriever(list(DOCS))
    assert reloaded.vectorizer is not first.vectorizer
    assert reloaded.vectorizer.vocabulary_ == first.vectorizer.vocabulary_
    assert (reloaded.doc_matrix != first.doc_matrix).nnz == 0


def test_changed_corpus_gets_a_new_index(monkeypatch):
    monkeypatch.delenv("QI_TFIDF_CACHE_DIR", raising=False)
    first = DocumentRetriever(list(DOCS))
    changed = DocumentRetriever([*DOCS, {**DOCS[0], "title": "比亚迪 销量"}])

    assert changed.vectorizer is not first.vectorizer
    assert changed.doc_matrix.shape[0] == 3
    assert len(doc_retriever._INDEX_MEMO) <= doc_retriever._INDEX_MEMO_SIZE


def test_unreadable_cache_file_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setenv("QI_TFIDF_CACHE_DIR", str(tmp_path))
    doc_retriever._INDEX_MEMO.clear()
    DocumentRetriever(list(DOCS))
    for path in tmp_path.glob("tfidf-*.joblib"):
        path.write_bytes(b"not a joblib file")
    doc_retriever._INDEX_MEMO.clear()

    retriever = DocumentRetriever(list(DOCS))

    assert retriever.doc_matrix.shape[0] == 2
