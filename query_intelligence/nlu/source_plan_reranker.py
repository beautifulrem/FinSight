from __future__ import annotations

import csv
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

from joblib import load
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

from ..query_terms import contains_bucket_market_term
from ..training_data import load_training_rows, row_supports_label
from .batch_linear import BATCH_LINEAR_BATCH_SIZE, BATCH_LINEAR_EPOCHS, FitProgressCallback, fit_dict_sgd_classifier

SOURCE_PRIORITY = [
    "market_api",
    "news",
    "industry_sql",
    "fundamental_sql",
    "announcement",
    "research_note",
    "faq",
    "product_doc",
    "macro_sql",
]

COMPARE_MARKERS = ["比", "相比", "哪个", "谁更", "更稳", "更好", "有什么区别", "有何区别", "有什么不同", "有啥不同", "差别", "一回事吗"]
WHY_MARKERS = ["为什么", "为啥", "咋", "咋回事", "啥原因", "什么原因"]
ADVICE_MARKERS = ["值得", "能买吗", "买吗", "买入", "卖出", "上车", "止盈", "止损", "持有", "拿吗", "还能拿吗", "还能不能拿", "适合长期拿", "适不适合"]
MAX_SOURCE_PLAN_TRAIN_ROWS = 50000

MANUAL_ROWS = [
    {
        "query": "最近有哪些茅台新闻？",
        "product_type": "stock",
        "intent_labels": ["event_news_query"],
        "topic_labels": ["news"],
        "time_scope": "unspecified",
        "source": "news",
        "label": 1,
    },
    {
        "query": "最近有哪些茅台新闻？",
        "product_type": "stock",
        "intent_labels": ["event_news_query"],
        "topic_labels": ["news"],
        "time_scope": "unspecified",
        "source": "announcement",
        "label": 1,
    },
    {
        "query": "最近有哪些茅台新闻？",
        "product_type": "stock",
        "intent_labels": ["event_news_query"],
        "topic_labels": ["news"],
        "time_scope": "unspecified",
        "source": "market_api",
        "label": 0,
    },
    {
        "query": "贵州茅台最近有什么公告？",
        "product_type": "stock",
        "intent_labels": ["event_news_query"],
        "topic_labels": ["news"],
        "time_scope": "unspecified",
        "source": "announcement",
        "label": 1,
    },
    {
        "query": "贵州茅台最近有什么公告？",
        "product_type": "stock",
        "intent_labels": ["event_news_query"],
        "topic_labels": ["news"],
        "time_scope": "unspecified",
        "source": "market_api",
        "label": 0,
    },
]


@dataclass
class SourcePlanReranker:
    vectorizer: DictVectorizer
    model: SGDClassifier

    @classmethod
    def build_from_dataset(cls, dataset_path: str | Path) -> "SourcePlanReranker":
        rows = build_source_plan_reranker_rows(dataset_path)
        return cls.build_from_rows(rows)

    @classmethod
    def build_from_rows(
        cls,
        rows: list[dict],
        *,
        fit_progress_callback: FitProgressCallback | None = None,
        batch_size: int = BATCH_LINEAR_BATCH_SIZE,
        epochs: int = BATCH_LINEAR_EPOCHS,
    ) -> "SourcePlanReranker":
        features = [cls.make_features(**row) for row in rows]
        labels = [int(row["label"]) for row in rows]
        vectorizer, model = fit_dict_sgd_classifier(
            features,
            labels,
            fit_progress_callback=fit_progress_callback,
            batch_size=batch_size,
            epochs=epochs,
        )
        return cls(vectorizer=vectorizer, model=model)

    @classmethod
    def load_model(cls, path: str | Path) -> "SourcePlanReranker":
        try:
            payload = load(path)
            return cls(vectorizer=payload["vectorizer"], model=payload["model"])
        except Exception as exc:
            fallback_dataset = Path(__file__).resolve().parents[2] / "data" / "query_labels.csv"
            if not fallback_dataset.exists():
                raise RuntimeError(f"source plan reranker model failed to load and fallback dataset is missing: {fallback_dataset}") from exc
            return cls.build_from_dataset(fallback_dataset)

    def score_candidates(
        self,
        query: str,
        product_type: str,
        intent_labels: list[str],
        topic_labels: list[str],
        time_scope: str,
        rule_plan: list[str],
        candidate_sources: list[str],
    ) -> list[dict]:
        candidate_sources = self._filter_marker_conflicts(query, candidate_sources)
        scored: list[dict] = []
        for source in candidate_sources:
            features = self.make_features(
                query=query,
                product_type=product_type,
                intent_labels=intent_labels,
                topic_labels=topic_labels,
                time_scope=time_scope,
                source=source,
                rule_plan=rule_plan,
                label=0,
            )
            X = self.vectorizer.transform([features])
            prob = float(self.model.predict_proba(X)[0][1])
            prob += self._apply_query_marker_boost(query, source)
            prob = max(0.0, min(1.0, prob))
            scored.append({"source": source, "score": round(prob, 4)})
        scored.sort(key=lambda item: (-item["score"], SOURCE_PRIORITY.index(item["source"]) if item["source"] in SOURCE_PRIORITY else 99))
        return scored

    @staticmethod
    def _apply_query_marker_boost(query: str, source: str) -> float:
        if source == "announcement" and "公告" in query:
            return 2.0
        if source == "news" and ("新闻" in query or "news" in query.lower()):
            return 2.0
        return 0.0

    @staticmethod
    def _filter_marker_conflicts(query: str, candidate_sources: list[str]) -> list[str]:
        if "公告" in query:
            return [source for source in candidate_sources if source not in {"news", "market_api", "fundamental_sql", "industry_sql"}]
        return candidate_sources

    @staticmethod
    def make_features(
        query: str,
        product_type: str,
        intent_labels: list[str],
        topic_labels: list[str],
        time_scope: str,
        source: str,
        rule_plan: list[str] | None = None,
        source_id: object | None = None,  # noqa: ARG004
        label: int = 0,  # noqa: ARG004
    ) -> dict[str, float | str]:
        rule_plan = rule_plan or []
        features: dict[str, float | str] = {
            "product_type": product_type,
            "time_scope": time_scope,
            "source": source,
            "rule_contains_source": float(source in rule_plan),
            "rule_rank": float(rule_plan.index(source)) if source in rule_plan else 99.0,
            "is_bucket_market_query": float(contains_bucket_market_term(query)),
            "has_compare_marker": float(any(term in query for term in COMPARE_MARKERS)),
            "has_why_marker": float(any(term in query for term in WHY_MARKERS)),
            "has_advice_marker": float(any(term in query for term in ADVICE_MARKERS)),
        }
        for label_name in {"market_explanation", "hold_judgment", "buy_sell_timing", "product_info", "trading_rule_fee", "peer_compare", "fundamental_analysis", "valuation_analysis", "macro_policy_impact", "event_news_query", "risk_analysis"}:
            features[f"intent:{label_name}"] = float(label_name in intent_labels)
        for label_name in {"price", "news", "industry", "macro", "policy", "fundamentals", "valuation", "risk", "comparison", "product_mechanism"}:
            features[f"topic:{label_name}"] = float(label_name in topic_labels)
        return features


def build_source_plan_reranker_rows(dataset_path: str | Path) -> list[dict]:
    rows: list[dict] = []
    artifact_rows = _load_source_plan_supervision_rows(dataset_path)
    if artifact_rows:
        rows.extend(_build_rows_from_source_plan_supervision(artifact_rows))
    rows.extend(_build_rows_from_legacy_records(load_training_rows(dataset_path)))
    rows.extend(_expand_manual_rows())
    return _cap_rows(rows, MAX_SOURCE_PLAN_TRAIN_ROWS)


def _cap_rows(rows: list[dict], limit: int) -> list[dict]:
    if len(rows) <= limit:
        return rows
    priority_rows = [row for row in rows if str(row.get("source_id") or "") == "curated_boundary_cases"]
    regular_rows = [row for row in rows if str(row.get("source_id") or "") != "curated_boundary_cases"]
    if len(priority_rows) >= limit:
        priority_rows.sort(key=_stable_row_key)
        return priority_rows[:limit]
    regular_rows.sort(key=_stable_row_key)
    return [*priority_rows, *regular_rows[: limit - len(priority_rows)]]


def _stable_row_key(row: dict) -> str:
    payload = "|".join(
        [
            str(row.get("query") or ""),
            str(row.get("source") or ""),
            str(row.get("label") or ""),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_rows_from_legacy_records(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    from .source_planner import RuleSourcePlanner

    rule_planner = RuleSourcePlanner()
    for row in records:
        if not (
            row_supports_label(row, "expected_document_sources")
            or row_supports_label(row, "expected_structured_sources")
        ):
            continue
        query = str(row.get("query") or "")
        product_type = str(row.get("product_type") or "unknown")
        intent_labels = row.get("intent_labels") or []
        topic_labels = row.get("topic_labels") or []
        expected_sources = set(row.get("expected_document_sources") or []) | set(row.get("expected_structured_sources") or [])
        time_scope = _detect_time_scope(query)
        rule_plan = rule_planner.plan(
            product_type,
            [{"label": item, "score": 1.0} for item in intent_labels],
            [{"label": item, "score": 1.0} for item in topic_labels],
            time_scope,
            query=query,
        )["source_plan"]
        for source in SOURCE_PRIORITY:
            rows.append(
                {
                    "query": query,
                    "product_type": product_type,
                    "intent_labels": intent_labels,
                    "topic_labels": topic_labels,
                    "time_scope": time_scope,
                    "source": source,
                    "rule_plan": rule_plan,
                    "label": int(source in expected_sources),
                }
            )
    return rows


def _expand_manual_rows() -> list[dict]:
    from .source_planner import RuleSourcePlanner

    rule_planner = RuleSourcePlanner()
    rows: list[dict] = []
    for row in MANUAL_ROWS:
        rule_plan = rule_planner.plan(
            row["product_type"],
            [{"label": item, "score": 1.0} for item in row["intent_labels"]],
            [{"label": item, "score": 1.0} for item in row["topic_labels"]],
            row["time_scope"],
            query=row["query"],
        )["source_plan"]
        rows.append({**row, "rule_plan": rule_plan})
    return rows


def _load_source_plan_supervision_rows(dataset_path: str | Path) -> list[dict]:
    artifact_path = _resolve_manifest_artifact_path(dataset_path, "source_plan_supervision_path", "source_plan_supervision")
    if artifact_path is None or not artifact_path.exists() or artifact_path.stat().st_size == 0:
        return []
    return _load_rows(artifact_path)


def _build_rows_from_source_plan_supervision(rows: list[dict]) -> list[dict]:
    built_rows: list[dict] = []
    for row in rows:
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        product_type = str(row.get("product_type") or "unknown")
        intent_labels = _split_labels(row.get("intent_labels"))
        topic_labels = _split_labels(row.get("topic_labels"))
        time_scope = str(row.get("time_scope") or _detect_time_scope(query))
        rule_plan = _split_labels(row.get("source_plan") or row.get("rule_plan") or row.get("source_plan_labels"))
        direct_source = str(row.get("source") or row.get("candidate_source") or row.get("target_source") or "").strip()
        if direct_source:
            built_rows.append(
                {
                    "source_id": row.get("source_id"),
                    "query": query,
                    "product_type": product_type,
                    "intent_labels": intent_labels,
                    "topic_labels": topic_labels,
                    "time_scope": time_scope,
                    "source": direct_source,
                    "rule_plan": rule_plan,
                    "label": _coerce_binary_label(row),
                }
            )
            continue
        if rule_plan:
            expected_sources = set(rule_plan)
            for source in SOURCE_PRIORITY:
                built_rows.append(
                    {
                        "source_id": row.get("source_id"),
                        "query": query,
                        "product_type": product_type,
                        "intent_labels": intent_labels,
                        "topic_labels": topic_labels,
                        "time_scope": time_scope,
                        "source": source,
                        "rule_plan": rule_plan,
                        "label": int(source in expected_sources),
                    }
                )
            continue
        expected_sources = set(_split_labels(row.get("expected_document_sources"))) | set(_split_labels(row.get("expected_structured_sources")))
        if expected_sources:
            rule_planner_row = {
                "query": query,
                "product_type": product_type,
                "intent_labels": intent_labels,
                "topic_labels": topic_labels,
                "time_scope": time_scope,
                "source_plan": list(expected_sources),
            }
            built_rows.extend(_expand_source_plan_row(rule_planner_row))
    return built_rows


def _expand_source_plan_row(row: dict) -> list[dict]:
    expected_sources = set(_split_labels(row.get("source_plan")))
    return [
        {
            "query": row["query"],
            "product_type": row["product_type"],
            "intent_labels": row["intent_labels"],
            "topic_labels": row["topic_labels"],
            "time_scope": row["time_scope"],
            "source": source,
            "rule_plan": list(expected_sources),
            "label": int(source in expected_sources),
        }
        for source in SOURCE_PRIORITY
    ]


def _coerce_binary_label(row: dict) -> int:
    for key in ("label", "is_positive", "positive", "target"):
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            return int(value)
        try:
            return int(float(str(value)))
        except ValueError:
            return 1 if str(value).strip().lower() in {"true", "yes", "y", "positive", "1"} else 0
    return 1


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


def _split_labels(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return [item.strip() for item in raw_value.replace(",", "|").split("|") if item.strip()]


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
