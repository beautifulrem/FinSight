from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from joblib import load
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

from ..training_data import filter_rows_for_label, load_training_rows
from .batch_linear import BATCH_LINEAR_BATCH_SIZE, BATCH_LINEAR_EPOCHS, FitProgressCallback, fit_dict_sgd_classifier


COMPARE_MARKERS = ["比", "相比", "哪个", "谁更", "更稳", "更好", "有什么区别", "有何区别", "有什么不同", "有啥不同", "差别", "一回事吗"]
WHY_MARKERS = ["为什么", "为啥", "咋", "咋回事", "啥原因", "什么原因"]
ADVICE_MARKERS = ["值得", "能买吗", "买吗", "买入", "卖出", "上车", "止盈", "止损", "持有", "拿吗", "还能拿吗", "还能不能拿", "适合长期拿", "适不适合", "不适合", "适合定投", "适合新手"]
FORECAST_MARKERS = ["未来", "接下来", "后市", "会怎么走", "怎么看后市", "后面会怎么走"]

QUESTION_STYLE_VOCAB = ["fact", "why", "compare", "advice", "forecast"]

MANUAL_ROWS = [
    {
        "query": "茅台接下来会怎么走",
        "base_style": "fact",
        "product_type": "stock",
        "intent_labels": ["market_explanation"],
        "topic_labels": ["price"],
        "entity_count": 1,
        "comparison_target_count": 0,
        "question_style": "forecast",
    },
    {
        "query": "创业板ETF波动大是不是不适合长期拿",
        "base_style": "fact",
        "product_type": "etf",
        "intent_labels": ["risk_analysis", "hold_judgment"],
        "topic_labels": ["risk", "product_mechanism"],
        "entity_count": 1,
        "comparison_target_count": 0,
        "question_style": "advice",
    },
    {
        "query": "基金净值和ETF价格到底有啥不同",
        "base_style": "fact",
        "product_type": "etf",
        "intent_labels": ["product_info"],
        "topic_labels": ["product_mechanism", "comparison"],
        "entity_count": 0,
        "comparison_target_count": 2,
        "question_style": "compare",
    },
    {
        "query": "白酒板块今天为什么跌，还能买吗",
        "base_style": "why",
        "product_type": "stock",
        "intent_labels": ["market_explanation", "buy_sell_timing", "hold_judgment"],
        "topic_labels": ["price", "news", "risk"],
        "entity_count": 0,
        "comparison_target_count": 0,
        "question_style": "advice",
    },
    {
        "query": "贵州茅台为什么跌，还值得持有吗",
        "base_style": "why",
        "product_type": "stock",
        "intent_labels": ["market_explanation", "hold_judgment"],
        "topic_labels": ["price"],
        "entity_count": 1,
        "comparison_target_count": 0,
        "question_style": "advice",
    },
    {
        "query": "ETF和LOF是一回事吗",
        "base_style": "fact",
        "product_type": "etf",
        "intent_labels": ["product_info"],
        "topic_labels": ["product_mechanism", "comparison"],
        "entity_count": 0,
        "comparison_target_count": 2,
        "question_style": "compare",
    },
    {
        "query": "基金净值和ETF价格有什么区别",
        "base_style": "fact",
        "product_type": "etf",
        "intent_labels": ["product_info"],
        "topic_labels": ["product_mechanism", "comparison"],
        "entity_count": 0,
        "comparison_target_count": 2,
        "question_style": "compare",
    },
    {
        "query": "创业板ETF波动大是不是不适合长期持有",
        "base_style": "fact",
        "product_type": "etf",
        "intent_labels": ["risk_analysis", "hold_judgment"],
        "topic_labels": ["risk", "product_mechanism"],
        "entity_count": 1,
        "comparison_target_count": 0,
        "question_style": "advice",
    },
    {
        "query": "证券ETF适不适合长期拿",
        "base_style": "fact",
        "product_type": "etf",
        "intent_labels": ["hold_judgment", "product_info"],
        "topic_labels": ["product_mechanism", "risk"],
        "entity_count": 1,
        "comparison_target_count": 0,
        "question_style": "advice",
    },
    {
        "query": "茅台后面会怎么走",
        "base_style": "fact",
        "product_type": "stock",
        "intent_labels": ["market_explanation"],
        "topic_labels": ["price"],
        "entity_count": 1,
        "comparison_target_count": 0,
        "question_style": "forecast",
    },
]


@dataclass
class QuestionStyleReranker:
    vectorizer: DictVectorizer
    model: SGDClassifier

    @classmethod
    def build_from_dataset(cls, dataset_path: str | Path) -> "QuestionStyleReranker":
        rows = build_question_style_reranker_rows(dataset_path)
        return cls.build_from_rows(rows)

    @classmethod
    def build_from_rows(
        cls,
        rows: list[dict],
        *,
        fit_progress_callback: FitProgressCallback | None = None,
        batch_size: int = BATCH_LINEAR_BATCH_SIZE,
        epochs: int = BATCH_LINEAR_EPOCHS,
    ) -> "QuestionStyleReranker":
        features = [cls.make_features(**row) for row in rows]
        labels = [row["question_style"] for row in rows]
        vectorizer, model = fit_dict_sgd_classifier(
            features,
            labels,
            fit_progress_callback=fit_progress_callback,
            batch_size=batch_size,
            epochs=epochs,
        )
        return cls(vectorizer=vectorizer, model=model)

    @classmethod
    def load_model(cls, path: str | Path) -> "QuestionStyleReranker":
        payload = load(path)
        return cls(vectorizer=payload["vectorizer"], model=payload["model"])

    def predict(self, **row: object) -> dict:
        features = self.make_features(**row)
        X = self.vectorizer.transform([features])
        probabilities = self.model.predict_proba(X)[0]
        labels = list(self.model.classes_)
        best_idx = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {"label": labels[best_idx], "score": float(round(probabilities[best_idx], 2))}

    @staticmethod
    def make_features(
        query: str,
        base_style: str,
        product_type: str,
        intent_labels: list[str],
        topic_labels: list[str],
        entity_count: int,
        comparison_target_count: int,
        question_style: str | None = None,  # noqa: ARG004
    ) -> dict[str, float | str]:
        features: dict[str, float | str] = {
            "base_style": base_style,
            "product_type": product_type,
            "entity_count": float(entity_count),
            "comparison_target_count": float(comparison_target_count),
            "has_compare_marker": float(any(term in query for term in COMPARE_MARKERS)),
            "has_why_marker": float(any(term in query for term in WHY_MARKERS)),
            "has_advice_marker": float(any(term in query for term in ADVICE_MARKERS)),
            "has_forecast_marker": float(any(term in query for term in FORECAST_MARKERS)),
        }
        for label in QUESTION_STYLE_VOCAB:
            features[f"base_is_{label}"] = float(base_style == label)
        for label in {"market_explanation", "hold_judgment", "buy_sell_timing", "product_info", "trading_rule_fee", "peer_compare", "fundamental_analysis", "valuation_analysis", "macro_policy_impact", "event_news_query", "risk_analysis"}:
            features[f"intent:{label}"] = float(label in intent_labels)
        for label in {"price", "news", "industry", "macro", "policy", "fundamentals", "valuation", "risk", "comparison", "product_mechanism"}:
            features[f"topic:{label}"] = float(label in topic_labels)
        return features


def build_question_style_reranker_rows(dataset_path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for row in filter_rows_for_label(load_training_rows(dataset_path), "question_style"):
        query = str(row.get("query") or "")
        intent_labels = row.get("intent_labels") or []
        topic_labels = row.get("topic_labels") or []
        base_query = str(row.get("base_query") or query)
        base_style = str(row.get("base_question_style") or heuristic_question_style(base_query, intent_labels))
        entity_count = int(bool(str(row.get("primary_symbol") or "").strip())) + int(bool(str(row.get("comparison_symbol") or "").strip()))
        comparison_target_count = int(row.get("comparison_target_count") or 0)
        if comparison_target_count <= 0:
            comparison_target_count = 2 if ("peer_compare" in intent_labels or any(term in base_query for term in COMPARE_MARKERS)) else 0
        rows.append(
            {
                "query": query,
                "base_style": base_style,
                "product_type": str(row.get("product_type") or "unknown"),
                "intent_labels": intent_labels,
                "topic_labels": topic_labels,
                "entity_count": entity_count,
                "comparison_target_count": comparison_target_count,
                "question_style": str(row.get("question_style") or ""),
            }
        )
    rows.extend(MANUAL_ROWS)
    return rows


def heuristic_question_style(query: str, intent_labels: list[str]) -> str:
    labels = set(intent_labels)
    if "peer_compare" in labels or any(term in query for term in COMPARE_MARKERS):
        return "compare"
    if ("market_explanation" in labels or any(term in query for term in WHY_MARKERS)) and not any(term in query for term in ADVICE_MARKERS):
        return "why"
    if labels.intersection({"hold_judgment", "buy_sell_timing"}):
        return "advice"
    if "market_explanation" in labels or any(term in query for term in WHY_MARKERS):
        return "why"
    if any(term in query for term in FORECAST_MARKERS):
        return "forecast"
    return "fact"


def _split_labels(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.replace(",", "|").split("|") if item.strip()]
