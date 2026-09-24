"""Document sentiment tool.

Backends:

* ``classical`` (default): the shipped TF-IDF + linear model (``models/sentiment.joblib``) already
  loaded by the NLU pipeline. Offline, explainable, fast.
* ``finbert``: the downstream FinBERT classifier from ``sentiment/``. Opt in with
  ``QI_AGENT_SENTIMENT_BACKEND=finbert``; requires ``torch``/``transformers`` and model weights
  (Hugging Face download or local cache). Falls back to ``classical`` if it cannot load.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import Counter
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..evidence import AgentEvidence, safe_evidence_id
from .base import ToolFailure, ToolOutput, ToolSpec
from .context import ToolContext
from .documents import _sanitize

logger = logging.getLogger(__name__)

_LABEL_SCORE = {"positive": 0.85, "neutral": 0.5, "negative": 0.15}
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+")
_MAX_SENTENCES = 12


class SentimentBackend(Protocol):
    name: str

    def score(self, text: str) -> tuple[str, float, float]:
        """Return ``(label, bullish_score in [0, 1], confidence)``."""


class ClassicalSentimentBackend:
    name = "classical"

    def __init__(self, classifier: Any) -> None:
        self._classifier = classifier

    def score(self, text: str) -> tuple[str, float, float]:
        from ...nlu.classifiers import _augment_text

        model = self._classifier.model
        probabilities = model.predict_proba([_augment_text(text)])[0]
        by_label = {str(label): float(prob) for label, prob in zip(model.classes_, probabilities, strict=True)}
        label = max(by_label, key=by_label.__getitem__)
        bullish = sum(_LABEL_SCORE.get(name, 0.5) * prob for name, prob in by_label.items())
        return label, round(bullish, 4), round(by_label[label], 4)


class FinBertSentimentBackend:
    name = "finbert"

    def __init__(self) -> None:
        from sentiment.classifier import SentimentClassifier
        from sentiment.preprocessor import detect_language

        self._classifier = SentimentClassifier()
        self._detect_language = detect_language

    def score(self, text: str) -> tuple[str, float, float]:
        label, bullish, confidence = self._classifier._predict_label_and_score(text, self._detect_language(text))
        return label, bullish, confidence


class SentimentInput(BaseModel):
    targets: list[str] = Field(
        min_length=1,
        max_length=3,
        description="Tickers or names of the securities whose news/announcement sentiment should be measured.",
    )
    query: str = Field(default="", max_length=200, description="Optional topic to focus the document search.")
    top_k: int = Field(default=6, ge=1, le=10)


class DocumentSentiment(BaseModel):
    evidence_id: str
    source_type: str
    title: str | None
    publish_time: str | None
    label: str
    score: float = Field(description="Bullishness in [0, 1]; 0.5 is neutral.")
    confidence: float


class SentimentOutput(BaseModel):
    backend: str
    targets: list[str]
    documents: list[DocumentSentiment]
    overall_label: str
    mean_score: float | None
    label_counts: dict[str, int]
    evidence_id: str
    note: str = "Model-estimated tone of retrieved documents; not a forecast or recommendation."


def build_sentiment_tool(context: ToolContext) -> ToolSpec:
    backend_lock = threading.Lock()
    state: dict[str, SentimentBackend | None] = {"backend": None}

    def backend() -> SentimentBackend:
        with backend_lock:
            if state["backend"] is None:
                state["backend"] = _select_backend(context)
            return state["backend"]  # type: ignore[return-value]

    def handler(args: SentimentInput) -> ToolOutput:
        resolved = [context.resolve_target(target) for target in args.targets]
        bundle = context.bundle(
            source_plan=["news", "announcement"],
            query=args.query or " ".join(item.name for item in resolved),
            symbols=[item.symbol for item in resolved],
            names=[item.name for item in resolved],
            product_type=resolved[0].product_type,
        )
        documents = [
            _sanitize(doc)
            for doc in context.retrieve_documents(bundle, args.top_k)
            if doc.get("source_type") in {"news", "announcement"}
        ]
        if not documents:
            raise ToolFailure("not_found", "no news or announcements found to analyze")

        scorer = backend()
        doc_evidence: list[AgentEvidence] = []
        scored: list[DocumentSentiment] = []
        for document in documents:
            item = AgentEvidence.from_document(document, produced_by="analyze_sentiment")
            doc_evidence.append(item)
            text = " ".join(
                part for part in (document.get("title"), document.get("summary") or document.get("body")) if part
            )
            label, bullish, confidence = _score_document(scorer, text)
            scored.append(
                DocumentSentiment(
                    evidence_id=item.evidence_id,
                    source_type=item.source_type,
                    title=item.title,
                    publish_time=item.as_of,
                    label=label,
                    score=bullish,
                    confidence=confidence,
                )
            )

        counts = Counter(item.label for item in scored)
        mean_score = round(sum(item.score for item in scored) / len(scored), 4)
        overall = "positive" if mean_score >= 0.6 else "negative" if mean_score <= 0.4 else "neutral"
        evidence_id = safe_evidence_id("sentiment_" + "_".join(item.symbol for item in resolved))
        output = SentimentOutput(
            backend=scorer.name,
            targets=[item.name for item in resolved],
            documents=scored,
            overall_label=overall,
            mean_score=mean_score,
            label_counts=dict(counts),
            evidence_id=evidence_id,
        )
        summary = AgentEvidence(
            evidence_id=evidence_id,
            kind="structured",
            source_type="sentiment_summary",
            title=f"Document sentiment for {', '.join(output.targets)}",
            source_name=f"{scorer.name} sentiment model",
            payload={
                "overall_label": overall,
                "mean_score": mean_score,
                "label_counts": dict(counts),
                "neutral_score": 0.5,
                "document_ids": [item.evidence_id for item in scored],
            },
            produced_by="analyze_sentiment",
        )
        return ToolOutput(data=output, evidence=[summary, *doc_evidence])

    return ToolSpec(
        name="analyze_sentiment",
        description=(
            "Retrieve recent news and announcements for up to three securities and classify the tone of each "
            "document (positive/neutral/negative) with an aggregate. Describes tone only, never a price forecast."
        ),
        input_model=SentimentInput,
        handler=handler,
        timeout_s=60.0,
        max_retries=0,
        cache_ttl_s=300.0,
    )


def _select_backend(context: ToolContext) -> SentimentBackend:
    requested = os.getenv("QI_AGENT_SENTIMENT_BACKEND", "classical").strip().lower()
    if requested == "finbert":
        try:
            return FinBertSentimentBackend()
        except Exception as exc:  # optional heavy dependency; fall back to the classical model
            logger.warning("FinBERT sentiment backend unavailable, using classical backend: %s", exc)
    classifier = getattr(context.nlu_pipeline, "sentiment_classifier", None)
    if classifier is None:
        from pathlib import Path

        from ...config import Settings
        from ...nlu.classifiers import SingleLabelTextClassifier

        model_path = Path(Settings.from_env().models_dir) / "sentiment.joblib"
        if not model_path.exists():
            raise ToolFailure("unavailable", "no sentiment model is available")
        classifier = SingleLabelTextClassifier.load_model(model_path)
    return ClassicalSentimentBackend(classifier)


def _score_document(scorer: SentimentBackend, text: str) -> tuple[str, float, float]:
    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT.split(text) if sentence and sentence.strip()]
    sentences = sentences[:_MAX_SENTENCES] or [text]
    results = [scorer.score(sentence) for sentence in sentences]
    counts = Counter(label for label, _score, _conf in results)
    label = counts.most_common(1)[0][0]
    bullish = round(sum(score for _label, score, _conf in results) / len(results), 4)
    confidence = round(sum(conf for _label, _score, conf in results) / len(results), 4)
    return label, bullish, confidence
