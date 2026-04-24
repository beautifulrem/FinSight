from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from joblib import load
from sklearn.linear_model import LogisticRegression


FEATURE_NAMES = [
    "lexical_score",
    "trigram_similarity",
    "entity_exact_match",
    "alias_match",
    "title_hit",
    "keyword_coverage",
    "intent_compatibility",
    "topic_compatibility",
    "product_type_match",
    "source_credibility",
    "recency_score",
    "is_primary_disclosure",
    "doc_length",
    "time_window_match",
    "ticker_hit",
]


class BaselineRanker:
    def score(self, feature_json: dict) -> float:
        score = (
            0.35 * feature_json["lexical_score"]
            + 0.20 * feature_json["title_hit"]
            + 0.15 * feature_json["entity_exact_match"]
            + 0.10 * feature_json["product_type_match"]
            + 0.10 * feature_json["recency_score"]
            + 0.10 * feature_json["source_credibility"]
        )
        return round(min(score, 1.0), 4)


@dataclass
class MLRanker:
    model: object
    feature_names: list[str]

    @classmethod
    def load_model(cls, path: str | Path) -> "MLRanker":
        payload = load(path)
        return cls(model=payload["model"], feature_names=list(payload["feature_names"]))

    def score(self, feature_json: dict) -> float:
        vector = [[float(feature_json.get(name, 0.0)) for name in self.feature_names]]
        probability = self.model.predict_proba(vector)[0][1]
        return round(float(probability), 4)


class HybridRanker:
    def __init__(self, baseline: BaselineRanker, ml_ranker: MLRanker | None = None, baseline_weight: float = 0.35) -> None:
        self.baseline = baseline
        self.ml_ranker = ml_ranker
        self.baseline_weight = baseline_weight

    @classmethod
    def load_model(cls, path: str | Path) -> "HybridRanker":
        return cls(baseline=BaselineRanker(), ml_ranker=MLRanker.load_model(path))

    def score(self, feature_json: dict) -> float:
        baseline_score = self.baseline.score(feature_json)
        if self.ml_ranker is None:
            return baseline_score
        ml_score = self.ml_ranker.score(feature_json)
        score = self.baseline_weight * baseline_score + (1 - self.baseline_weight) * ml_score
        return round(min(max(score, 0.0), 1.0), 4)
