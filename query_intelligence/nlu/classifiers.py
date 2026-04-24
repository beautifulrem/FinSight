from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import re

from joblib import load
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight


TEXT_CLASSIFIER_BATCH_SIZE = 5_000
TEXT_CLASSIFIER_EPOCHS = 3
FitProgressCallback = Callable[[int, int], None]


PRODUCT_SAMPLES = [
    ("茅台今天为什么跌", "stock"),
    ("五粮液现在能买吗", "stock"),
    ("沪深300ETF最近怎么样", "etf"),
    ("300ETF适合定投吗", "etf"),
    ("这个基金费率高吗", "fund"),
    ("公募基金回撤怎么样", "fund"),
    ("沪深300指数近期表现", "index"),
    ("上证指数今天涨了吗", "index"),
    ("CPI 对市场有什么影响", "macro"),
    ("宏观政策会影响白酒吗", "macro"),
    ("大盘今天为什么跌", "generic_market"),
]

INTENT_SAMPLES = [
    ("茅台今天为什么跌", ["market_explanation"]),
    ("茅台后面还值得拿吗", ["hold_judgment"]),
    ("五粮液现在能买吗", ["buy_sell_timing"]),
    ("这个基金是什么产品", ["product_info"]),
    ("这个基金费率高吗", ["trading_rule_fee"]),
    ("茅台和五粮液哪个好", ["peer_compare"]),
    ("茅台基本面怎么样", ["fundamental_analysis"]),
    ("茅台估值高不高", ["valuation_analysis"]),
    ("宏观政策会影响白酒吗", ["macro_policy_impact"]),
    ("最近有哪些茅台新闻", ["event_news_query"]),
    ("这只基金风险高吗", ["risk_analysis"]),
    ("茅台今天为什么跌后面还值得拿吗", ["market_explanation", "hold_judgment"]),
]

TOPIC_SAMPLES = [
    ("茅台今天为什么跌", ["price", "news"]),
    ("最近有哪些茅台新闻", ["news"]),
    ("白酒行业今天为什么跌", ["industry"]),
    ("宏观政策会影响白酒吗", ["macro", "policy"]),
    ("茅台基本面怎么样", ["fundamentals"]),
    ("茅台估值高不高", ["valuation"]),
    ("这只基金风险高吗", ["risk"]),
    ("茅台和五粮液哪个好", ["comparison"]),
    ("ETF 的申赎规则是什么", ["product_mechanism"]),
]


def _augment_text(text: str) -> str:
    extra_tokens: list[str] = []
    if re.search(r"\b\d{6}\.(?:SH|SZ)\b", text):
        extra_tokens.append("__HAS_TICKER__")
    if "ETF" in text or "etf" in text:
        extra_tokens.append("__HAS_ETF__")
    if any(term in text for term in ["基金", "费率", "申赎", "定投"]):
        extra_tokens.append("__HAS_FUND_TERM__")
    if any(term in text for term in ["为什么", "原因"]):
        extra_tokens.append("__HAS_WHY__")
    if any(term in text for term in ["为啥", "咋", "咋回事"]):
        extra_tokens.append("__HAS_COLLOQUIAL_WHY__")
    if any(term in text for term in ["值得", "持有", "买", "卖", "止盈"]):
        extra_tokens.append("__HAS_ACTION__")
    if any(term in text for term in ["还能拿", "值不值得拿", "还适合拿"]):
        extra_tokens.append("__HAS_HOLD_COLLOQUIAL__")
    if any(term in text for term in ["哪个好", "有什么不同", "有什么区别", "有何区别"]):
        extra_tokens.append("__HAS_COMPARE__")
    if any(term in text for term in ["风险", "波动", "回撤"]):
        extra_tokens.append("__HAS_RISK__")
    if any(term in text for term in ["宏观", "政策", "CPI", "PMI"]):
        extra_tokens.append("__HAS_MACRO__")
    return f"{text} {' '.join(extra_tokens)}".strip()


def estimate_text_classifier_fit_steps(
    row_count: int,
    *,
    batch_size: int = TEXT_CLASSIFIER_BATCH_SIZE,
    epochs: int = TEXT_CLASSIFIER_EPOCHS,
) -> int:
    return max(1, math.ceil(max(row_count, 1) / max(batch_size, 1)) * max(epochs, 1))


def _build_text_feature_union() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word_tfidf",
                HashingVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b\w+\b",
                    n_features=2**18,
                    alternate_sign=False,
                    norm="l2",
                ),
            ),
            (
                "char_tfidf",
                HashingVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    n_features=2**18,
                    alternate_sign=False,
                    norm="l2",
                ),
            ),
        ]
    )


def _iter_batches(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _make_single_label_model(
    texts: list[str],
    labels: list[str],
    *,
    fit_progress_callback: FitProgressCallback | None = None,
    batch_size: int = TEXT_CLASSIFIER_BATCH_SIZE,
    epochs: int = TEXT_CLASSIFIER_EPOCHS,
) -> Pipeline:
    if not texts:
        raise ValueError("Cannot train text classifier with zero rows")
    classes = np.array(sorted(set(labels)))
    if len(classes) < 2:
        raise ValueError("Cannot train text classifier with fewer than two classes")
    batch_size = max(int(batch_size), 1)
    epochs = max(int(epochs), 1)
    total_batches = estimate_text_classifier_fit_steps(len(texts), batch_size=batch_size, epochs=epochs)
    features = _build_text_feature_union()
    classifier = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42, n_jobs=-1)
    y = np.array(labels)
    class_weights = compute_class_weight("balanced", classes=classes, y=y)
    weight_map = dict(zip(classes, class_weights, strict=False))
    rng = np.random.default_rng(42)
    completed = 0
    first_batch = True
    for _ in range(epochs):
        indices = rng.permutation(len(texts))
        for batch_indices in _iter_batches(indices, batch_size):
            batch_texts = [texts[int(index)] for index in batch_indices]
            batch_labels = y[batch_indices]
            batch_weights = np.array([weight_map[label] for label in batch_labels], dtype=float)
            x_batch = features.transform(batch_texts)
            if first_batch:
                classifier.partial_fit(x_batch, batch_labels, classes=classes, sample_weight=batch_weights)
                first_batch = False
            else:
                classifier.partial_fit(x_batch, batch_labels, sample_weight=batch_weights)
            completed += 1
            if fit_progress_callback:
                fit_progress_callback(completed, total_batches)
    return Pipeline([("features", features), ("clf", classifier)])


def _make_multi_label_model(
    texts: list[str],
    y: np.ndarray,
    *,
    fit_progress_callback: FitProgressCallback | None = None,
    batch_size: int = TEXT_CLASSIFIER_BATCH_SIZE,
    epochs: int = TEXT_CLASSIFIER_EPOCHS,
) -> Pipeline:
    if not texts:
        raise ValueError("Cannot train text classifier with zero rows")
    if y.shape[1] == 0:
        raise ValueError("Cannot train multi-label classifier with zero labels")
    batch_size = max(int(batch_size), 1)
    epochs = max(int(epochs), 1)
    total_batches = estimate_text_classifier_fit_steps(len(texts), batch_size=batch_size, epochs=epochs)
    features = _build_text_feature_union()
    classifier = MultiOutputClassifier(
        SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42),
        n_jobs=1,
    )
    binary_classes = [np.array([0, 1]) for _ in range(y.shape[1])]
    rng = np.random.default_rng(42)
    completed = 0
    first_batch = True
    for _ in range(epochs):
        indices = rng.permutation(len(texts))
        for batch_indices in _iter_batches(indices, batch_size):
            batch_texts = [texts[int(index)] for index in batch_indices]
            x_batch = features.transform(batch_texts)
            y_batch = y[batch_indices]
            if first_batch:
                classifier.partial_fit(x_batch, y_batch, classes=binary_classes)
                first_batch = False
            else:
                classifier.partial_fit(x_batch, y_batch)
            completed += 1
            if fit_progress_callback:
                fit_progress_callback(completed, total_batches)
    return Pipeline([("features", features), ("clf", classifier)])


def _predict_multilabel_proba(model: Pipeline, texts: list[str]) -> np.ndarray:
    raw_proba = model.predict_proba(texts)
    if isinstance(raw_proba, list):
        return np.column_stack([proba[:, 1] if proba.shape[1] > 1 else np.zeros(proba.shape[0]) for proba in raw_proba])
    return raw_proba


@dataclass
class ProductTypeClassifier:
    model: Pipeline

    @classmethod
    def build_demo(cls) -> "ProductTypeClassifier":
        return cls.build_from_records([{"query": text, "product_type": label} for text, label in PRODUCT_SAMPLES])

    @classmethod
    def build_from_records(
        cls,
        records: list[dict],
        *,
        fit_progress_callback: FitProgressCallback | None = None,
        batch_size: int = TEXT_CLASSIFIER_BATCH_SIZE,
        epochs: int = TEXT_CLASSIFIER_EPOCHS,
    ) -> "ProductTypeClassifier":
        texts = [_augment_text(record["query"]) for record in records]
        labels = [record["product_type"] for record in records]
        model = _make_single_label_model(
            texts,
            labels,
            fit_progress_callback=fit_progress_callback,
            batch_size=batch_size,
            epochs=epochs,
        )
        return cls(model=model)

    @classmethod
    def load_model(cls, path: str | Path) -> "ProductTypeClassifier":
        return cls(model=load(path))

    def predict(self, text: str, resolved_entities: list[dict]) -> dict:
        if resolved_entities and resolved_entities[0]["entity_type"] in {"stock", "etf", "fund", "index"}:
            return {
                "label": resolved_entities[0]["entity_type"],
                "score": 0.99,
                "trace": ["entity_override"],
            }

        probs = self.model.predict_proba([_augment_text(text)])[0]
        labels = list(self.model.classes_)
        best_idx = max(range(len(probs)), key=probs.__getitem__)
        return {
            "label": labels[best_idx],
            "score": float(round(probs[best_idx], 2)),
            "trace": [f"ml:{labels[best_idx]}"],
        }


@dataclass
class MultiLabelClassifier:
    model: Pipeline
    mlb: MultiLabelBinarizer
    threshold_map: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build_demo(cls, samples: list[tuple[str, list[str]]]) -> "MultiLabelClassifier":
        return cls.build_from_records([{"query": text, "labels": labels} for text, labels in samples], "labels")

    @classmethod
    def build_from_records(
        cls,
        records: list[dict],
        label_field: str,
        *,
        fit_progress_callback: FitProgressCallback | None = None,
        batch_size: int = TEXT_CLASSIFIER_BATCH_SIZE,
        epochs: int = TEXT_CLASSIFIER_EPOCHS,
    ) -> "MultiLabelClassifier":
        train_records = [record for record in records if record.get("split", "train") == "train"]
        valid_records = [record for record in records if record.get("split", "train") == "valid"]
        fit_records = train_records or records
        texts = [_augment_text(record["query"]) for record in fit_records]
        label_sets = [record.get(label_field, []) for record in fit_records]
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(label_sets)
        model = _make_multi_label_model(
            texts,
            y,
            fit_progress_callback=fit_progress_callback,
            batch_size=batch_size,
            epochs=epochs,
        )
        thresholds = {label: 0.2 for label in mlb.classes_}
        if valid_records:
            thresholds = cls._tune_thresholds(model, mlb, valid_records, label_field)
        return cls(model=model, mlb=mlb, threshold_map=thresholds)

    @classmethod
    def load_model(cls, path: str | Path) -> "MultiLabelClassifier":
        payload = load(path)
        mlb = MultiLabelBinarizer()
        mlb.fit([payload["classes"]])
        mlb.classes_ = payload["classes"]
        return cls(model=payload["model"], mlb=mlb, threshold_map=payload.get("threshold_map", {}))

    def predict(self, text: str, threshold: float = 0.2) -> list[dict]:
        probs = _predict_multilabel_proba(self.model, [_augment_text(text)])[0]
        scored = []
        for label, prob in zip(self.mlb.classes_, probs, strict=False):
            scored.append({"label": str(label), "score": float(round(prob, 2))})
        scored.sort(key=lambda item: item["score"], reverse=True)
        selected = [item for item in scored if item["score"] >= self.threshold_map.get(item["label"], threshold)]
        if not selected and scored:
            selected = scored[:1]
        if len(selected) == 1 and len(scored) > 1 and scored[1]["score"] >= max(0.12, selected[0]["score"] - 0.08):
            selected = scored[:2]
        return selected[:4]

    @staticmethod
    def _tune_thresholds(model: Pipeline, mlb: MultiLabelBinarizer, valid_records: list[dict], label_field: str) -> dict[str, float]:
        import logging
        known_labels = set(mlb.classes_)
        for record in valid_records:
            unseen = set(record.get(label_field, [])) - known_labels
            if unseen:
                logging.getLogger(__name__).warning(
                    "Valid labels not in training set (will be ignored): %s — query: %s",
                    unseen,
                    record.get("query", "")[:40],
                )
                break
        texts = [_augment_text(record["query"]) for record in valid_records]
        y_true = mlb.transform([record.get(label_field, []) for record in valid_records])
        y_prob = _predict_multilabel_proba(model, texts)
        threshold_map: dict[str, float] = {}
        candidate_thresholds = [round(value, 2) for value in [0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4]]
        for idx, label in enumerate(mlb.classes_):
            best_threshold = 0.2
            best_score = -1.0
            true_col = y_true[:, idx]
            prob_col = y_prob[:, idx]
            for threshold in candidate_thresholds:
                pred_col = [1 if value >= threshold else 0 for value in prob_col]
                score = f1_score(true_col, pred_col, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            threshold_map[str(label)] = best_threshold
        return threshold_map


@dataclass
class SingleLabelTextClassifier:
    model: Pipeline

    @classmethod
    def build_from_records(
        cls,
        records: list[dict],
        label_field: str,
        *,
        fit_progress_callback: FitProgressCallback | None = None,
        batch_size: int = TEXT_CLASSIFIER_BATCH_SIZE,
        epochs: int = TEXT_CLASSIFIER_EPOCHS,
    ) -> "SingleLabelTextClassifier":
        usable = [record for record in records if record.get(label_field)]
        texts = [_augment_text(record["query"]) for record in usable]
        labels = [record[label_field] for record in usable]
        model = _make_single_label_model(
            texts,
            labels,
            fit_progress_callback=fit_progress_callback,
            batch_size=batch_size,
            epochs=epochs,
        )
        return cls(model=model)

    @classmethod
    def load_model(cls, path: str | Path) -> "SingleLabelTextClassifier":
        return cls(model=load(path))

    def predict(self, text: str) -> dict:
        probs = self.model.predict_proba([_augment_text(text)])[0]
        labels = list(self.model.classes_)
        best_idx = max(range(len(probs)), key=probs.__getitem__)
        return {"label": labels[best_idx], "score": float(round(probs[best_idx], 2))}
