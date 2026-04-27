from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from joblib import load
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.class_weight import compute_class_weight

from ..training_data import load_training_rows


NEGATIVE_SAMPLES = [
    "今天天气怎么样",
    "上海今天下雨吗",
    "北京明天多少度",
    "帮我写个Python冒泡排序",
    "解释一下快速排序",
    "你好",
    "早上好",
    "特朗普是谁",
    "今天NBA谁赢了",
    "讲个笑话",
    "帮我翻译这段英文",
    "How is the weather today?",
    "Will it rain in Shanghai today?",
    "Write a Python bubble sort",
    "Who is Donald Trump?",
    "Hello there",
    "Tell me a joke",
    "Translate this sentence into Chinese",
    "Who won the NBA game tonight?",
    "What is the best pizza recipe?",
]

POSITIVE_SAMPLES = [
    "Why did Moutai fall today?",
    "Is Ping An still worth holding?",
    "Which is better, CSI 300 ETF or ChiNext ETF?",
    "What is the difference between ETF and LOF?",
    "Does macro policy affect liquor?",
    "Any recent announcements from Kweichow Moutai?",
]

OUT_OF_SCOPE_BATCH_SIZE = 5_000
OUT_OF_SCOPE_EPOCHS = 3


def _augment_text(text: str) -> str:
    lowered = text.lower()
    extra_tokens: list[str] = []
    if len(text.strip()) <= 24:
        extra_tokens.append("__SHORT_UTTERANCE__")
    if any(term in lowered for term in ["hello", "hi", "thanks", "thank you", "sorry", "bye", "good morning", "good afternoon"]):
        extra_tokens.append("__EN_DIALOGUE_POLITE__")
    if any(term in text for term in ["你好", "谢谢", "抱歉", "不好意思", "再见", "好的", "可以", "行啊"]):
        extra_tokens.append("__ZH_DIALOGUE_POLITE__")
    if _contains_english_token(
        lowered,
        ["hotel", "room", "airport", "flight", "bus", "taxi", "ticket", "restaurant", "tip", "seat", "station", "souvenir", "engine", "car", "store", "office"],
    ):
        extra_tokens.append("__SERVICE_DIALOGUE__")
    if any(term in text for term in ["酒店", "餐厅", "景点", "医院", "公交", "出租车", "机票", "电话", "人均消费", "推荐一款", "推荐一辆"]):
        extra_tokens.append("__SERVICE_DIALOGUE__")
    if any(term in lowered for term in ["dollar", "$", "yuan", "fee", "charge", "price", "how much", "one hundred", "six pm", "7:00"]) or any(
        term in text for term in ["多少", "元", "预算", "价格", "人均消费", "点", "下周二"]
    ):
        extra_tokens.append("__TRANSACTIONAL_DIALOGUE__")
    if any(term in lowered for term in ["translate", "python", "sql", "laptop", "computer", "bubble sort"]) or any(
        term in text for term in ["翻译", "Python", "SQL", "电脑", "笔记本", "台式机", "代码", "冒泡排序"]
    ):
        extra_tokens.append("__UTILITY_TASK__")
    if any(
        term in lowered
        for term in [
            "contract",
            "invoice",
            "bill of lading",
            "insurance premium",
            "current account",
            "guarantee",
            "general manager",
            "board of directors",
            "ceo",
            "stockholders",
            "educational background",
            "tour guide",
            "deposit a check",
            "parcels",
            "film developed",
        ]
    ) or any(term in text for term in ["合同", "发票", "经理", "教育背景", "导游", "保险费", "账户"]):
        extra_tokens.append("__BUSINESS_DIALOGUE__")
    if (
        _contains_english_token(
            lowered,
            ["etf", "lof", "stock", "stocks", "share", "shares", "fund", "funds", "bond", "bonds", "moutai", "macro", "earnings", "valuation"],
        )
        or any(phrase in lowered for phrase in ["ping an", "share price", "stock price", "mutual fund", "index fund", "net profit"])
    ) or any(
        term in text for term in ["基金", "ETF", "LOF", "股票", "财报", "估值", "营收", "净利润", "茅台", "平安", "白酒", "指数", "降息", "降准"]
    ):
        extra_tokens.append("__FINANCE_ANCHOR__")
    return f"{text} {' '.join(extra_tokens)}".strip()


def _contains_english_token(lowered_text: str, tokens: list[str]) -> bool:
    escaped = "|".join(re.escape(token) for token in tokens)
    return bool(re.search(rf"\b(?:{escaped})\b", lowered_text))


def _build_text_feature_union() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word_hash",
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
                "char_hash",
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


@dataclass
class OutOfScopeDetector:
    model: Pipeline

    @classmethod
    def build_from_dataset(cls, dataset_path: str | Path) -> "OutOfScopeDetector":
        rows = build_out_of_scope_training_rows(dataset_path)
        return cls.build_from_rows(rows)

    @classmethod
    def build_from_rows(
        cls,
        rows: list[dict],
        *,
        fit_progress_callback: Callable[[int, int], None] | None = None,
    ) -> "OutOfScopeDetector":
        texts, labels = _prepare_training_examples(rows)
        model = _fit_out_of_scope_pipeline(texts, labels, fit_progress_callback=fit_progress_callback)
        return cls(model=model)

    @classmethod
    def load_model(cls, path: str | Path) -> "OutOfScopeDetector":
        try:
            return cls(model=load(path))
        except Exception as exc:
            fallback_dataset = Path(__file__).resolve().parents[2] / "data" / "query_labels.csv"
            if not fallback_dataset.exists():
                raise RuntimeError(f"out-of-scope detector model failed to load and fallback dataset is missing: {fallback_dataset}") from exc
            return cls.build_from_dataset(fallback_dataset)

    def predict_probability(self, query: str) -> float:
        return float(self.model.predict_proba([_augment_text(query)])[0][1])


def build_out_of_scope_training_rows(dataset_path: str | Path) -> list[dict]:
    artifact_rows = _load_out_of_scope_supervision_rows(dataset_path)
    if artifact_rows and len({int(row["out_of_scope"]) for row in artifact_rows}) >= 2:
        return artifact_rows
    return load_training_rows(dataset_path)


def estimate_out_of_scope_fit_steps(row_count: int) -> int:
    return max(math.ceil(max(row_count, 1) / OUT_OF_SCOPE_BATCH_SIZE) * OUT_OF_SCOPE_EPOCHS, 1)


def _prepare_training_examples(rows: list[dict]) -> tuple[list[str], list[int]]:
    labeled_rows = [row for row in rows if _coerce_label(row, "out_of_scope", "label", "is_out_of_scope") is not None]
    if labeled_rows:
        texts: list[str] = []
        labels: list[int] = []
        for row in labeled_rows:
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            texts.append(_augment_text(query))
            labels.append(int(_coerce_label(row, "out_of_scope", "label", "is_out_of_scope")))
        return texts, labels

    fallback_queries = [str(row.get("query") or "").strip() for row in rows if str(row.get("query") or "").strip()]
    texts = [_augment_text(query) for query in fallback_queries]
    texts.extend(_augment_text(item) for item in POSITIVE_SAMPLES)
    texts.extend(_augment_text(item) for item in NEGATIVE_SAMPLES)
    labels = [0] * len(fallback_queries) + [0] * len(POSITIVE_SAMPLES) + [1] * len(NEGATIVE_SAMPLES)
    return texts, labels


def _fit_out_of_scope_pipeline(
    texts: list[str],
    labels: list[int],
    *,
    fit_progress_callback: Callable[[int, int], None] | None = None,
) -> Pipeline:
    if not texts:
        texts = [_augment_text(item) for item in POSITIVE_SAMPLES + NEGATIVE_SAMPLES]
        labels = [0] * len(POSITIVE_SAMPLES) + [1] * len(NEGATIVE_SAMPLES)
    if len(set(labels)) < 2:
        texts.extend([_augment_text(POSITIVE_SAMPLES[0]), _augment_text(NEGATIVE_SAMPLES[0])])
        labels.extend([0, 1])

    features = _build_text_feature_union()
    features.fit(texts)
    y = np.asarray(labels, dtype=int)
    classes = np.asarray([0, 1], dtype=int)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight_map = {int(label): float(weight) for label, weight in zip(classes, class_weights)}
    classifier = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42, n_jobs=-1)

    total_batches = estimate_out_of_scope_fit_steps(len(y))
    completed = 0
    first_batch = True
    for epoch in range(OUT_OF_SCOPE_EPOCHS):
        order = np.arange(len(y))
        rng = np.random.default_rng(42 + epoch)
        rng.shuffle(order)
        for start in range(0, len(order), OUT_OF_SCOPE_BATCH_SIZE):
            batch_indices = order[start : start + OUT_OF_SCOPE_BATCH_SIZE]
            batch_texts = [texts[int(index)] for index in batch_indices]
            x_batch = features.transform(batch_texts)
            y_batch = y[batch_indices]
            sample_weight = np.asarray([class_weight_map[int(label)] for label in y_batch], dtype=float)
            if first_batch:
                classifier.partial_fit(x_batch, y_batch, classes=classes, sample_weight=sample_weight)
                first_batch = False
            else:
                classifier.partial_fit(x_batch, y_batch, sample_weight=sample_weight)
            completed += 1
            if fit_progress_callback:
                fit_progress_callback(completed, total_batches)

    return Pipeline([("features", features), ("clf", classifier)])


def _load_out_of_scope_supervision_rows(dataset_path: str | Path) -> list[dict]:
    artifact_path = _resolve_manifest_artifact_path(dataset_path, "out_of_scope_supervision_path", "out_of_scope_supervision")
    if artifact_path is None or not artifact_path.exists() or artifact_path.stat().st_size == 0:
        return []
    rows = _load_rows(artifact_path)
    normalized_rows: list[dict] = []
    for row in rows:
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        label = _coerce_label(row, "out_of_scope", "label", "is_out_of_scope")
        if label is None:
            continue
        normalized_rows.append({"query": query, "out_of_scope": int(label)})
    return normalized_rows


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
