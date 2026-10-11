"""(round 15, review-11 K4) A small classical classifier for the operation a comparison question asks.

The comparison grammar (``frame.GRAMMAR``) decides the operation from word classes (a lexicon of regular
expressions). A wording outside the lexicon gets no operation, and a clause whose words fit two productions gets the
first one in the table. This module adds a learned reading next to it, within the project's architecture rules
(classical, explainable ML; no transformer, no LLM):

* features: character 1-3-grams of the question (TF-IDF, ``char_wb``) and the grammar's own word-class hits
  (``lex=SUM_WORD``, ``lex=VERSUS``, …), so the lexicon is a feature, not the only signal;
* model: multinomial logistic regression over the operations (``difference``, ``ratio``, ``relative``, ``share``,
  ``share_total``, ``sum``, ``mean``, ``spread``, ``which``, ``total_vs_one``) and ``none`` (not a comparison);
* training: ``python -m training.operation_classifier.train`` — generated wordings (operation phrases × targets ×
  metrics, Chinese and English, seeded) and the dev set's questions labelled by the grammar; the held-out split is
  by *template*, so its accuracy measures wordings the model never saw (``evaluation/results/
  operation_classifier-r15.json``).

It is used in two places only (``frame.frame_operation`` / ``frame.resolve_frame_question``):

* **fallback** — the grammar reads no operation, the question refers to several operands (a plural or a list) and
  asks a quantity, and the classifier gives an operation other than ``none`` with probability ≥ ``FALLBACK_MIN``;
* **tie-breaker** — the clause fits productions of two or more operations; the grammar's first production is kept
  unless the classifier gives another of the matched operations probability ≥ ``TIE_MIN`` and the grammar's choice
  probability ≤ ``TIE_MAX_OTHER``.

Every use is explainable: the frame request records ``classifier`` (operation, probability, top features) and the
route reason gets ``clf:<operation>:<p>:<features>``. ``QI_OPERATION_CLASSIFIER=0`` disables it; a missing artifact
disables it with one warning (the grammar alone then decides, as before round 15).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "operation_classifier.joblib"
OPERATIONS = (
    "difference",
    "ratio",
    "relative",
    "share",
    "share_total",
    "sum",
    "mean",
    "spread",
    "which",
    "total_vs_one",
    "none",
)
# (round 16, review-12 L17) 0.6 -> 0.45 after lexicon dropout: on the held-out templates the grammar reads nothing
# in, the fallback is right on 146/146 at 0.6 and 152/152 at 0.45 (operation_classifier-r16.json, threshold sweep)
FALLBACK_MIN = 0.45
TIE_MIN = 0.9
TIE_MAX_OTHER = 0.05
TOP_FEATURES = 3


# (round 16, review-12 L17) lexicon dropout: a training copy of a question starts with this mark and gets no lexicon or
# structure features, so the character and word n-grams must carry the operation on their own (the wordings the
# grammar misses are exactly those without lexicon hits). Questions at run time never start with it.
DROPOUT_MARK = "⊘"


def strip_mark(text: str) -> str:
    """The text the n-gram vectorizers see (the dropout mark removed). Module-level for pickling."""
    return str(text or "").removeprefix(DROPOUT_MARK)


def lexicon_hits(texts: Any) -> list[dict[str, float]]:
    """The grammar's word classes present in each text (``{"lex=SUM_WORD": 1.0, …}``): the lexicon as features.
    Module-level so a fitted pipeline pickles it by reference."""
    from .frame import _LEXICON, _normalised

    rows = []
    for text in texts:
        if str(text or "").startswith(DROPOUT_MARK):
            rows.append({})
            continue
        clause = _normalised(str(text or ""))
        rows.append({f"lex={name}": 1.0 for name, test in _LEXICON.items() if test(clause)})
    return rows


def structure_hits(texts: Any) -> list[dict[str, float]]:
    """(round 16, review-12 L17) A few structural features the lexicon classes do not carry: whether a total word comes
    before or after the comparison word (``order=sum_then_versus`` / ``order=versus_then_sum``), a list of three or more
    names (``list>=3``) and a yes / no question (``yes_no``). Module-level so a fitted pipeline pickles it by
    reference."""
    from .frame import _SUM, _SUM_WEAK, _VERSUS, _YES_NO, _normalised

    rows = []
    for text in texts:
        if str(text or "").startswith(DROPOUT_MARK):
            rows.append({})
            continue
        clause = _normalised(str(text or ""))
        row: dict[str, float] = {}
        total = _SUM.search(clause) or _SUM_WEAK.search(clause)
        versus = _VERSUS.search(clause)
        if total and versus:
            row["order=sum_then_versus" if total.start() < versus.start() else "order=versus_then_sum"] = 1.0
        if len(re.findall(r"[、，,]|\band\b|和|与|跟|及", clause)) >= 2:
            row["list>=3"] = 1.0
        if _YES_NO.search(clause.strip()):
            row["yes_no"] = 1.0
        rows.append(row)
    return rows


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


@dataclass(frozen=True)
class Reading:
    operation: str
    probability: float
    features: tuple[str, ...]
    use: str  # "fallback" or "tie_break"

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "probability": round(self.probability, 3),
            "top_features": list(self.features),
            "use": self.use,
        }

    def reason(self) -> str:
        return f"clf:{self.operation}:{self.probability:.2f}:{'+'.join(self.features)}"


@dataclass(frozen=True)
class OperationClassifier:
    pipeline: Any  # Pipeline(FeatureUnion(chars, lexicon), LogisticRegression)
    version: str

    @property
    def classes(self) -> list[str]:
        return [str(item) for item in self.pipeline.named_steps["model"].classes_]

    def probabilities(self, text: str) -> dict[str, float]:
        row = self.pipeline.predict_proba([normalise(text)])[0]
        return {label: float(p) for label, p in zip(self.classes, row, strict=True)}

    def top_features(self, text: str, label: str, count: int = TOP_FEATURES) -> tuple[str, ...]:
        """The features that pushed ``text`` towards ``label`` most (value × coefficient), for the trace."""
        features = self.pipeline.named_steps["features"]
        model = self.pipeline.named_steps["model"]
        vector = features.transform([normalise(text)]).tocsr()
        names = feature_names(features)
        row = model.coef_[self.classes.index(label)]
        scored = sorted(
            ((vector.data[i] * row[j], names[j]) for i, j in enumerate(vector.indices)), key=lambda item: -item[0]
        )
        return tuple(name.replace(" ", "_") for weight, name in scored[:count] if weight > 0)

    def read(self, text: str) -> tuple[str, float]:
        probabilities = self.probabilities(text)
        label = max(probabilities, key=probabilities.__getitem__)
        return label, probabilities[label]


def feature_names(features: Any) -> list[str]:
    """Names of the union's features in order: ``chars:<n-gram>``, (round 16) ``words:<n-gram>``, then the dictionary
    features (``lex=<class>``, ``order=…``, ``list>=3``, ``yes_no``)."""
    names: list[str] = []
    for name, transformer in features.transformer_list:
        if name in {"chars", "words"}:
            names += [f"{name}:{item}" for item in transformer.get_feature_names_out()]
        else:
            names += list(transformer.named_steps["dict"].get_feature_names_out())
    return names


_lock = threading.Lock()
_cached: dict[str, OperationClassifier | None] = {}


def _artifact_path() -> Path:
    configured = os.getenv("QI_OPERATION_CLASSIFIER_PATH")
    if configured:
        return Path(configured)
    models_dir = Path(os.getenv("QI_MODELS_DIR", "models"))
    if not models_dir.is_absolute() and not (Path.cwd() / models_dir / ARTIFACT_NAME).exists():
        models_dir = Path(__file__).resolve().parents[2] / "models"
    return models_dir / ARTIFACT_NAME


def load_classifier() -> OperationClassifier | None:
    """The shipped classifier, or ``None`` when disabled or unavailable (cached per artifact path)."""
    if os.getenv("QI_OPERATION_CLASSIFIER", "1").strip().lower() in {"0", "false", "off", "no"}:
        return None
    path = _artifact_path()
    key = str(path)
    if key in _cached:
        return _cached[key]
    with _lock:
        if key not in _cached:
            _cached[key] = _load(path)
    return _cached[key]


def _load(path: Path) -> OperationClassifier | None:
    try:
        import joblib

        payload = joblib.load(path)
        return OperationClassifier(pipeline=payload["pipeline"], version=str(payload["version"]))
    except Exception as exc:  # the grammar alone decides; the missing layer is logged once
        logger.warning("operation classifier unavailable (%s): %s", path, exc)
        return None


def reset_cache() -> None:
    _cached.clear()


# a question that refers to several operands and asks a quantity: the only questions the fallback reads
_SEVERAL = re.compile(
    r"[和与跟及、]|还是|两家|两者|二者|两只|三家|三只|三者|几家|这几|它们|他们|各家|\band\b|\bor\b|\bboth\b|"
    r"\bthem\b|\bthey\b|\bversus\b|\bvs\b|\bthe (?:two|three|four)\b|\ball (?:three|four)\b|,",
    re.IGNORECASE,
)
_ASKS_QUANTITY = re.compile(
    r"多少|几|吗|呢|哪|谁|更|[?？]|\bhow\b|\bwhat\b|\bwhich\b|\bwhose\b|\bis\b|\bdo(?:es)?\b", re.IGNORECASE
)


def fallback_operation(text: str, *, operands: int = 0) -> Reading | None:
    """The classifier's operation for a question the grammar reads none in, or ``None``. ``operands`` is how many
    targets the question names (when the caller knows): two or more count as several without a list word."""
    if not (operands >= 2 or _SEVERAL.search(text or "")) or not _ASKS_QUANTITY.search(text or ""):
        return None
    model = load_classifier()
    if model is None:
        return None
    label, probability = model.read(text)
    if label == "none" or probability < FALLBACK_MIN or not compatible(label, text):
        return None
    return Reading(label, probability, model.top_features(text, label), "fallback")


def compatible(label: str, text: str) -> bool:
    """Structural preconditions the grammar's productions also have: a total compared with another value names a
    total; a spread is not one target's own extremes ("当天最高价和最低价")."""
    from .frame import _OWN_EXTREMES, _SUM, _SUM_WEAK

    if label == "total_vs_one":
        return bool(_SUM.search(text) or _SUM_WEAK.search(text))
    if label == "spread":
        return not _OWN_EXTREMES.search(text)
    return True


def tie_break(text: str, matched: list[str], chosen: str) -> Reading | None:
    """Another of the operations the clause matched, when the classifier is confident of it and not of the grammar's
    choice; ``None`` keeps the grammar's choice."""
    if len(set(matched)) < 2:
        return None
    model = load_classifier()
    if model is None:
        return None
    probabilities = model.probabilities(text)
    label = max(probabilities, key=probabilities.__getitem__)
    if (
        label == chosen
        or label not in matched
        or probabilities[label] < TIE_MIN
        or probabilities.get(chosen, 0.0) > TIE_MAX_OTHER
    ):
        return None
    return Reading(label, probabilities[label], model.top_features(text, label), "tie_break")
