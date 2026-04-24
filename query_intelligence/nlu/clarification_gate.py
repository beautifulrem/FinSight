from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from joblib import load
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

from ..query_terms import contains_bucket_market_term
from ..training_data import load_training_rows
from .batch_linear import BATCH_LINEAR_BATCH_SIZE, BATCH_LINEAR_EPOCHS, FitProgressCallback, fit_dict_sgd_classifier


ENTITY_DEPENDENT_INTENTS = {
    "market_explanation",
    "hold_judgment",
    "buy_sell_timing",
    "peer_compare",
    "fundamental_analysis",
    "valuation_analysis",
    "risk_analysis",
    "event_news_query",
}

GENERIC_PRODUCT_TERMS = {"ETF", "LOF", "基金", "公募基金", "场内基金", "场外基金", "指数基金", "指数"}

COMPARE_MARKERS = ["比", "相比", "哪个", "谁更", "更稳", "更好", "有什么区别", "有何区别", "有什么不同", "差别"]
WHY_MARKERS = ["为什么", "为啥", "咋", "咋回事"]
ADVICE_MARKERS = ["值得", "能买吗", "买吗", "买入", "卖出", "上车", "止盈", "止损", "持有", "拿吗", "还能拿吗", "还能不能拿"]

MANUAL_ROWS = [
    {
        "query": "茅台今天为啥跌, 后面还能拿吗",
        "product_type": "stock",
        "intent_labels": ["market_explanation", "hold_judgment"],
        "topic_labels": ["price", "news", "risk"],
        "time_scope": "today",
        "entity_count": 1,
        "comparison_target_count": 0,
        "needs_clarification": False,
    },
    {
        "query": "贵州茅台最近一周为啥跌",
        "product_type": "stock",
        "intent_labels": ["market_explanation", "event_news_query"],
        "topic_labels": ["price", "news"],
        "time_scope": "recent_1w",
        "entity_count": 1,
        "comparison_target_count": 0,
        "needs_clarification": False,
    },
    {
        "query": "ETF和LOF有什么区别",
        "product_type": "etf",
        "intent_labels": ["product_info", "peer_compare"],
        "topic_labels": ["product_mechanism", "comparison"],
        "time_scope": "unspecified",
        "entity_count": 0,
        "comparison_target_count": 2,
        "needs_clarification": False,
    },
    {
        "query": "300etf和创业板etf哪个好",
        "product_type": "etf",
        "intent_labels": ["peer_compare", "product_info"],
        "topic_labels": ["comparison", "product_mechanism"],
        "time_scope": "unspecified",
        "entity_count": 2,
        "comparison_target_count": 2,
        "needs_clarification": False,
    },
    {
        "query": "这个能买吗",
        "product_type": "stock",
        "intent_labels": ["buy_sell_timing"],
        "topic_labels": ["price", "risk"],
        "time_scope": "unspecified",
        "entity_count": 0,
        "comparison_target_count": 0,
        "needs_clarification": True,
    },
    {
        "query": "后面还值得拿吗",
        "product_type": "stock",
        "intent_labels": ["hold_judgment"],
        "topic_labels": ["risk", "fundamentals"],
        "time_scope": "unspecified",
        "entity_count": 0,
        "comparison_target_count": 0,
        "needs_clarification": True,
    },
    {
        "query": "宽基指数最近能买吗",
        "product_type": "stock",
        "intent_labels": ["buy_sell_timing", "hold_judgment"],
        "topic_labels": ["risk", "price", "fundamentals"],
        "time_scope": "unspecified",
        "entity_count": 0,
        "comparison_target_count": 0,
        "needs_clarification": False,
    },
    {
        "query": "白酒板块最近还能买吗",
        "product_type": "stock",
        "intent_labels": ["buy_sell_timing", "hold_judgment", "market_explanation"],
        "topic_labels": ["industry", "news", "macro"],
        "time_scope": "unspecified",
        "entity_count": 0,
        "comparison_target_count": 0,
        "needs_clarification": False,
    },
]


def _split_labels(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return [item.strip() for item in raw_value.replace(",", "|").split("|") if item.strip()]


@dataclass
class ClarificationGate:
    vectorizer: DictVectorizer
    model: SGDClassifier

    @classmethod
    def build_from_dataset(cls, dataset_path: str | Path) -> "ClarificationGate":
        rows = build_clarification_training_rows(dataset_path)
        return cls.build_from_rows(rows)

    @classmethod
    def build_from_rows(
        cls,
        rows: list[dict],
        *,
        fit_progress_callback: FitProgressCallback | None = None,
        batch_size: int = BATCH_LINEAR_BATCH_SIZE,
        epochs: int = BATCH_LINEAR_EPOCHS,
    ) -> "ClarificationGate":
        feature_rows = [cls.make_features(**row) for row in rows]
        labels = [int(row["needs_clarification"]) for row in rows]
        vectorizer, model = fit_dict_sgd_classifier(
            feature_rows,
            labels,
            fit_progress_callback=fit_progress_callback,
            batch_size=batch_size,
            epochs=epochs,
        )
        return cls(vectorizer=vectorizer, model=model)

    @classmethod
    def load_model(cls, path: str | Path) -> "ClarificationGate":
        try:
            payload = load(path)
            return cls(vectorizer=payload["vectorizer"], model=payload["model"])
        except Exception:
            fallback_dataset = Path(__file__).resolve().parents[2] / "data" / "query_labels.csv"
            return cls.build_from_dataset(fallback_dataset)

    def predict_probability(self, **row: object) -> float:
        features = self.make_features(**row)
        matrix = self.vectorizer.transform([features])
        return float(self.model.predict_proba(matrix)[0][1])

    @staticmethod
    def make_features(
        query: str,
        product_type: str,
        intent_labels: list[str],
        topic_labels: list[str],
        time_scope: str,
        entity_count: int,
        comparison_target_count: int,
        needs_clarification: bool | None = None,  # noqa: ARG004
    ) -> dict[str, float | str]:
        lower_query = query.lower()
        features: dict[str, float | str] = {
            "product_type": product_type,
            "time_scope": time_scope,
            "entity_count": float(entity_count),
            "comparison_target_count": float(comparison_target_count),
            "is_bucket_market_query": float(contains_bucket_market_term(query)),
            "has_compare_marker": float(any(term in query for term in COMPARE_MARKERS)),
            "has_why_marker": float(any(term in query for term in WHY_MARKERS)),
            "has_advice_marker": float(any(term in query for term in ADVICE_MARKERS)),
            "has_generic_product_term": float(any(term.lower() in lower_query for term in {term.lower() for term in GENERIC_PRODUCT_TERMS})),
        }
        for label in ENTITY_DEPENDENT_INTENTS:
            features[f"intent:{label}"] = float(label in intent_labels)
        for label in {"price", "news", "industry", "macro", "policy", "fundamentals", "valuation", "risk", "comparison", "product_mechanism"}:
            features[f"topic:{label}"] = float(label in topic_labels)
        return features


def build_clarification_training_rows(dataset_path: str | Path) -> list[dict]:
    artifact_rows = _load_clarification_supervision_rows(dataset_path)
    if artifact_rows:
        rows = _build_clarification_rows_from_supervision(artifact_rows)
        if rows:
            rows.extend(MANUAL_ROWS)
            if len({int(bool(row["needs_clarification"])) for row in rows}) >= 2:
                return rows
    return build_clarification_training_rows_from_records(load_training_rows(dataset_path))


def build_clarification_training_rows_from_records(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in records:
        query = str(row.get("query") or "")
        product_type = str(row.get("product_type") or "unknown")
        intents = row.get("intent_labels") or []
        topics = row.get("topic_labels") or []
        primary_symbol = str(row.get("primary_symbol") or "")
        comparison_symbol = str(row.get("comparison_symbol") or "")
        entity_count = int(bool(primary_symbol)) + int(bool(comparison_symbol))
        compare_like = bool(comparison_symbol) or any(term in query for term in COMPARE_MARKERS) or "peer_compare" in intents or "comparison" in topics
        generic_product_compare = compare_like and all(term in query for term in ["ETF", "LOF"]) if "ETF" in query or "LOF" in query else False
        mechanism_like = set(topics).issubset({"product_mechanism", "comparison"}) and set(intents).issubset({"product_info", "trading_rule_fee", "peer_compare"})
        needs_clarification = _derive_label(
            query=query,
            product_type=product_type,
            intent_labels=intents,
            topic_labels=topics,
            entity_count=entity_count,
            compare_like=compare_like,
            generic_product_compare=generic_product_compare,
            mechanism_like=mechanism_like,
        )
        rows.append(
            {
                "query": query,
                "product_type": product_type,
                "intent_labels": intents,
                "topic_labels": topics,
                "time_scope": _detect_time_scope(query),
                "entity_count": entity_count,
                "comparison_target_count": 2 if compare_like else 0,
                "needs_clarification": needs_clarification,
            }
        )

    rows.extend(MANUAL_ROWS)
    return rows


def _load_clarification_supervision_rows(dataset_path: str | Path) -> list[dict]:
    artifact_path = _resolve_manifest_artifact_path(dataset_path, "clarification_supervision_path", "clarification_supervision")
    if artifact_path is None or not artifact_path.exists() or artifact_path.stat().st_size == 0:
        return []
    return _load_rows(artifact_path)


def _build_clarification_rows_from_supervision(rows: list[dict]) -> list[dict]:
    built_rows: list[dict] = []
    for row in rows:
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        intents = _split_labels(row.get("intent_labels"))
        topics = _split_labels(row.get("topic_labels"))
        product_type = str(row.get("product_type") or "unknown")
        time_scope = str(row.get("time_scope") or _detect_time_scope(query))
        entity_count = row.get("entity_count")
        if entity_count in (None, ""):
            primary_symbol = str(row.get("primary_symbol") or "").strip()
            comparison_symbol = str(row.get("comparison_symbol") or "").strip()
            entity_count = int(bool(primary_symbol)) + int(bool(comparison_symbol))
        comparison_target_count = row.get("comparison_target_count")
        if comparison_target_count in (None, ""):
            compare_like = bool(row.get("comparison_symbol")) or any(term in query for term in COMPARE_MARKERS) or "peer_compare" in intents or "comparison" in topics
            comparison_target_count = 2 if compare_like else 0
        needs_clarification = _coerce_label(row, "needs_clarification")
        if needs_clarification is None:
            needs_clarification = _coerce_label(row, "label")
        if needs_clarification is None:
            needs_clarification = _derive_label(
                query=query,
                product_type=product_type,
                intent_labels=intents,
                topic_labels=topics,
                entity_count=int(entity_count),
                compare_like=bool(comparison_target_count),
                generic_product_compare=False,
                mechanism_like=False,
            )
        built_rows.append(
            {
                "query": query,
                "product_type": product_type,
                "intent_labels": intents,
                "topic_labels": topics,
                "time_scope": time_scope,
                "entity_count": _coerce_int(entity_count),
                "comparison_target_count": _coerce_int(comparison_target_count),
                "needs_clarification": bool(needs_clarification),
            }
        )
    return built_rows


def _coerce_label(row: dict, *keys: str) -> bool | None:
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            return value
        try:
            return bool(int(float(str(value))))
        except ValueError:
            return str(value).strip().lower() in {"true", "yes", "y", "positive", "1"}
    return None


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _load_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else [payload]
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_manifest_artifact_path(dataset_path: str | Path, *keys: str) -> Path | None:
    path = Path(dataset_path)
    if path.name != "manifest.json" or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        raw_value = payload.get(key)
        if not raw_value:
            continue
        candidate = Path(str(raw_value))
        return candidate if candidate.is_absolute() else path.parent / candidate
    return None


def _derive_label(
    query: str,
    product_type: str,
    intent_labels: list[str],
    topic_labels: list[str],
    entity_count: int,
    compare_like: bool,
    generic_product_compare: bool,
    mechanism_like: bool,
) -> bool:
    if entity_count > 0:
        return False
    if contains_bucket_market_term(query):
        return False
    if product_type == "macro":
        return False
    if product_type in {"fund", "etf"} and (mechanism_like or generic_product_compare):
        return False
    if compare_like:
        return True
    if set(intent_labels).intersection(ENTITY_DEPENDENT_INTENTS) and product_type in {"stock", "index"}:
        return True
    return False


def _detect_time_scope(query: str) -> str:
    if "今天" in query:
        return "today"
    if "最近一周" in query:
        return "recent_1w"
    if "近一个月" in query:
        return "recent_1m"
    if "长期" in query:
        return "long_term"
    return "unspecified"
