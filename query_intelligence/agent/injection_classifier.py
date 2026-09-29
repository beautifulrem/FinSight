"""Second-layer, non-lexical injection detector for third-party document text.

A character n-gram TF-IDF + logistic-regression model (``training/injection_classifier/train.py``) scores each
segment (sentence or title) of a document field; segments at or above the model's threshold are treated like
lexical-filter hits (replaced with the redaction marker by ``injection.sanitize_untrusted_text``).

Scope and limits:

* Only document text (tool observations and evidence), never the user's own message: the model was trained on
  document segments, and its false-positive rate on user questions was not measured.
* Text is NFKC-normalised, stripped of invisible characters and confusable-folded before scoring, the same view
  the lexical filter and the answer guard use.
* The model is small and trained on few attacks; its measured recall and false-positive rate on held-out
  attack sets and clean documents are in ``evaluation/results/injection_classifier-r4.json``. It is a layer of
  defense in depth, not a guarantee.
* ``QI_INJECTION_CLASSIFIER=0`` disables it; a missing or unreadable artifact disables it with one warning.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..text_safety import fold

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "injection_classifier.joblib"
MAX_SEGMENT_CHARS = 300
_SEGMENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])|(?<=[.])(?=\s)")


def normalise(text: str) -> str:
    """The view the model scores: NFKC, invisible characters removed, confusables folded, spaces collapsed."""
    return re.sub(r"\s+", " ", fold(text or "")).strip()


def segments(text: str) -> list[str]:
    """Sentence-like segments of ``text`` (original characters), each at most ``MAX_SEGMENT_CHARS`` long."""
    parts: list[str] = []
    for part in _SEGMENT_SPLIT.split(text or ""):
        part = part.strip()
        while len(part) > MAX_SEGMENT_CHARS:
            parts.append(part[:MAX_SEGMENT_CHARS])
            part = part[MAX_SEGMENT_CHARS:]
        if part:
            parts.append(part)
    return parts


@dataclass(frozen=True)
class InjectionClassifier:
    pipeline: Any  # sklearn Pipeline(TfidfVectorizer(char_wb), LogisticRegression)
    threshold: float
    version: str

    def scores(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(p) for p in self.pipeline.predict_proba([normalise(t) for t in texts])[:, 1]]

    def flagged_segments(self, text: str) -> list[str]:
        """The segments of ``text`` scored at or above the threshold (original characters)."""
        parts = [part for part in segments(text) if normalise(part)]
        return [part for part, score in zip(parts, self.scores(parts), strict=True) if score >= self.threshold]


_lock = threading.Lock()
_cached: dict[str, InjectionClassifier | None] = {}


def _artifact_path() -> Path:
    configured = os.getenv("QI_INJECTION_CLASSIFIER_PATH")
    if configured:
        return Path(configured)
    models_dir = Path(os.getenv("QI_MODELS_DIR", "models"))
    if not models_dir.is_absolute() and not (Path.cwd() / models_dir).exists():
        models_dir = Path(__file__).resolve().parents[2] / "models"
    return models_dir / ARTIFACT_NAME


def load_classifier() -> InjectionClassifier | None:
    """The shipped classifier, or ``None`` when disabled or unavailable (cached per artifact path)."""
    if os.getenv("QI_INJECTION_CLASSIFIER", "1").strip().lower() in {"0", "false", "off", "no"}:
        return None
    path = _artifact_path()
    key = str(path)
    if key in _cached:
        return _cached[key]
    with _lock:
        if key not in _cached:
            _cached[key] = _load(path)
    return _cached[key]


def _load(path: Path) -> InjectionClassifier | None:
    try:
        import joblib

        payload = joblib.load(path)
        return InjectionClassifier(
            pipeline=payload["pipeline"], threshold=float(payload["threshold"]), version=str(payload["version"])
        )
    except Exception as exc:  # the lexical filter still runs; the missing layer is logged once
        logger.warning("injection classifier unavailable (%s): %s", path, exc)
        return None


def reset_cache() -> None:
    _cached.clear()
