"""Tests for sentiment/classifier.py — label mapping, aggregation, sentence capping."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sentiment.classifier import (
    SentimentClassifier,
    _detect_device,
    _empty_sentiment,
    _LABEL_MAP,
    _LABEL_SCORE,
    _MODEL_IDS,
)
from sentiment.schemas import PreprocessedDoc, SentimentItem


# ===========================================================================
# helpers
# ===========================================================================


def _make_doc(
    evidence_id: str = "doc_001",
    sentences: list[str] | None = None,
    language: str = "zh",
    raw_text: str = "",
    source_type: str = "news",
    **kwargs,
) -> PreprocessedDoc:
    return PreprocessedDoc(
        evidence_id=evidence_id,
        source_type=source_type,
        title=kwargs.get("title", "测试标题"),
        raw_text=raw_text,
        language=language,
        sentences=sentences or [],
        entity_hits=kwargs.get("entity_hits", []),
        text_level=kwargs.get("text_level", "full"),
        relevant_excerpt=kwargs.get("relevant_excerpt"),
        publish_time=kwargs.get("publish_time"),
        source_name=kwargs.get("source_name"),
        rank_score=kwargs.get("rank_score"),
    )


def _with_mock_infer(clf: SentimentClassifier, responses: dict[str, dict[str, float]]):
    """Patch _infer_single to return canned responses keyed by text prefix match."""

    def fake_infer(text: str, language: str) -> dict[str, float]:
        for prefix, scores in responses.items():
            if text.startswith(prefix):
                return scores
        return {"neutral": 1.0}

    patcher = patch.object(clf, "_infer_single", side_effect=fake_infer)
    return patcher


# ===========================================================================
# _detect_device
# ===========================================================================


def test_detect_device_returns_str():
    device = _detect_device()
    assert isinstance(device, str)
    assert device in ("cuda", "mps", "cpu")


# ===========================================================================
# model registry and label maps
# ===========================================================================


def test_model_ids_zh_and_en_registered():
    assert "zh" in _MODEL_IDS
    assert "en" in _MODEL_IDS


def test_label_map_covers_all_classes():
    # Chinese labels map to unified set
    assert set(_LABEL_MAP["zh"].values()) == {"positive", "negative", "neutral"}
    # English labels are identity
    assert set(_LABEL_MAP["en"].values()) == {"positive", "negative", "neutral"}


def test_label_score_ordering():
    assert _LABEL_SCORE["positive"] > _LABEL_SCORE["neutral"] > _LABEL_SCORE["negative"]


# ===========================================================================
# _predict_label_and_score — aggregation logic (model mocked)
# ===========================================================================


class TestPredictLabelAndScore:
    """Test label/score aggregation with mocked _infer_single."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.clf = SentimentClassifier(device="cpu")

    def test_positive_dominant(self):
        with _with_mock_infer(self.clf, {"sent": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}}):
            label, score, conf = self.clf._predict_label_and_score("sent", "zh")
        assert label == "positive"
        assert score > 0.8
        assert conf > 0.9

    def test_negative_dominant(self):
        with _with_mock_infer(self.clf, {"sent": {"positive": 0.05, "neutral": 0.10, "negative": 0.85}}):
            label, score, conf = self.clf._predict_label_and_score("sent", "en")
        assert label == "negative"
        assert score < 0.3
        assert conf > 0.8

    def test_neutral_dominant(self):
        with _with_mock_infer(self.clf, {"sent": {"positive": 0.10, "neutral": 0.80, "negative": 0.10}}):
            label, score, conf = self.clf._predict_label_and_score("sent", "zh")
        assert label == "neutral"
        assert 0.45 < score < 0.55
        assert conf > 0.7

    def test_empty_scores_fallback(self):
        with _with_mock_infer(self.clf, {"sent": {}}):
            label, score, conf = self.clf._predict_label_and_score("sent", "zh")
        assert label == "neutral"
        assert score == 0.5
        assert conf == 0.0


# ===========================================================================
# analyze_document — sentence-by-sentence aggregation
# ===========================================================================


class TestAnalyzeDocument:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.clf = SentimentClassifier(device="cpu")

    def test_majority_label_wins(self):
        """2 positive + 1 neutral → majority positive."""
        doc = _make_doc(
            sentences=["pos1", "pos2", "neu"],
            language="zh",
        )
        with _with_mock_infer(self.clf, {
            "pos1": {"positive": 0.9, "neutral": 0.05, "negative": 0.05},
            "pos2": {"positive": 0.8, "neutral": 0.10, "negative": 0.10},
            "neu":  {"positive": 0.1, "neutral": 0.85, "negative": 0.05},
        }):
            item = self.clf.analyze_document(doc)

        assert isinstance(item, SentimentItem)
        assert item.sentiment_label == "positive"
        assert item.sentiment_score > 0.6
        assert 0.8 < item.confidence < 1.0
        assert item.evidence_id == "doc_001"
        assert item.source_type == "news"

    def test_all_neutral(self):
        doc = _make_doc(sentences=["neu1", "neu2"], language="en")
        with _with_mock_infer(self.clf, {
            "neu1": {"positive": 0.1, "neutral": 0.8, "negative": 0.1},
            "neu2": {"positive": 0.05, "neutral": 0.9, "negative": 0.05},
        }):
            item = self.clf.analyze_document(doc)

        assert item.sentiment_label == "neutral"
        assert 0.45 < item.sentiment_score < 0.55

    def test_falls_back_to_raw_text_when_no_sentences(self):
        doc = _make_doc(sentences=[], raw_text="fallback text", language="zh")
        with _with_mock_infer(self.clf, {"fallback text": {"positive": 0.7, "neutral": 0.2, "negative": 0.1}}):
            item = self.clf.analyze_document(doc)

        assert item.sentiment_label == "positive"

    def test_empty_document_returns_neutral(self):
        doc = _make_doc(sentences=[], raw_text="", language="zh")
        item = self.clf.analyze_document(doc)
        assert item.sentiment_label == "neutral"
        assert item.sentiment_score == 0.5
        assert item.confidence == 0.0

    def test_pass_through_fields_preserved(self):
        doc = _make_doc(
            sentences=["sent"],
            evidence_id="news_abc",
            source_type="announcement",
            source_name="财经网",
            publish_time="2026-04-23T20:00:00",
            entity_hits=["600519.SH"],
            rank_score=0.92,
            text_level="short",
        )
        with _with_mock_infer(self.clf, {"sent": {"positive": 0.6, "neutral": 0.3, "negative": 0.1}}):
            item = self.clf.analyze_document(doc)

        assert item.evidence_id == "news_abc"
        assert item.source_type == "announcement"
        assert item.source_name == "财经网"
        assert item.publish_time == "2026-04-23T20:00:00"
        assert item.entity_symbols == ["600519.SH"]
        assert item.rank_score == 0.92
        assert item.text_level == "short"


# ===========================================================================
# max_sentences capping
# ===========================================================================


def test_sentence_cap_truncates_long_documents():
    clf = SentimentClassifier(device="cpu", max_sentences=3)
    sentences = [f"sent_{i}" for i in range(10)]  # 10 sentences
    doc = _make_doc(sentences=sentences, language="zh")

    responses = {f"sent_{i}": {"positive": 0.7, "neutral": 0.2, "negative": 0.1} for i in range(10)}
    with _with_mock_infer(clf, responses):
        item = clf.analyze_document(doc)

    # Only the first 3 should have been evaluated → relevant_excerpt has 3
    excerpt = item.relevant_excerpt or ""
    assert excerpt.count("sent_") == 3
    assert "sent_0" in excerpt
    assert "sent_2" in excerpt
    assert "sent_3" not in excerpt


# ===========================================================================
# analyze_documents (batch)
# ===========================================================================


def test_skipped_docs_are_excluded():
    clf = SentimentClassifier(device="cpu")
    skipped = _make_doc(sentences=["sent"], evidence_id="skip_me")
    skipped.skipped = True
    skipped.skip_reason = "unsupported source_type=faq"

    normal = _make_doc(sentences=["sent"], evidence_id="normal")
    with _with_mock_infer(clf, {"sent": {"positive": 0.8, "neutral": 0.1, "negative": 0.1}}):
        results = clf.analyze_documents([skipped, normal])

    assert len(results) == 1
    assert results[0].evidence_id == "normal"


def test_doc_error_is_swallowed():
    """A doc that raises during inference should not crash the batch."""
    clf = SentimentClassifier(device="cpu")
    good = _make_doc(sentences=["good"], evidence_id="good")
    bad = _make_doc(sentences=["bad"], evidence_id="bad")

    def failing_infer(text, language):
        if text == "good":
            return {"positive": 0.8, "neutral": 0.1, "negative": 0.1}
        raise RuntimeError("inference failure")

    with patch.object(clf, "_infer_single", side_effect=failing_infer):
        results = clf.analyze_documents([bad, good])

    assert len(results) == 1
    assert results[0].evidence_id == "good"


# ===========================================================================
# _empty_sentiment
# ===========================================================================


def test_empty_sentiment_returns_neutral():
    doc = _make_doc(
        evidence_id="empty",
        source_type="news",
        entity_hits=["600519.SH"],
        rank_score=0.5,
    )
    item = _empty_sentiment(doc)
    assert item.evidence_id == "empty"
    assert item.sentiment_label == "neutral"
    assert item.sentiment_score == 0.5
    assert item.confidence == 0.0
    assert item.entity_symbols == ["600519.SH"]
    assert item.rank_score == 0.5
