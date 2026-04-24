from __future__ import annotations

import math
import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight


ENTITY_LINKER_BATCH_SIZE = 5_000
ENTITY_LINKER_EPOCHS = 3
ENTITY_LINKER_NEGATIVES_PER_ALIAS = 8


def _token_overlap(query: str, text: str) -> float:
    if not text:
        return 0.0
    chars = {char for char in text if char.strip()}
    if not chars:
        return 0.0
    hit = sum(1 for char in chars if char in query)
    return hit / len(chars)


@dataclass
class EntityLinker:
    model: SGDClassifier | None

    @classmethod
    def build_from_catalog(
        cls,
        entities: list[dict[str, str]],
        aliases: list[dict[str, str]],
        *,
        fit_progress_callback: Callable[[int, int], None] | None = None,
        batch_size: int = ENTITY_LINKER_BATCH_SIZE,
        epochs: int = ENTITY_LINKER_EPOCHS,
    ) -> "EntityLinker":
        entity_map = {int(entity["entity_id"]): entity for entity in entities if entity.get("entity_id")}
        if len(entity_map) < 2:
            return cls(model=None)
        X: list[list[float]] = []
        y: list[int] = []
        entity_ids = sorted(entity_map)
        for alias_row in aliases:
            alias_text = alias_row["normalized_alias"]
            target_id = int(alias_row["entity_id"])
            target_entity = entity_map.get(target_id)
            if target_entity is None:
                continue
            candidate_ids = [target_id] + _sample_negative_entity_ids(alias_text, target_id, entity_ids)
            for candidate_id in candidate_ids:
                candidate_entity = entity_map[candidate_id]
                X.append(_build_features(alias_text, alias_text, candidate_entity))
                y.append(1 if candidate_id == target_id else 0)
        model = _fit_entity_linker_model(X, y, fit_progress_callback=fit_progress_callback, batch_size=batch_size, epochs=epochs)
        return cls(model=model)

    def rank(self, query: str, mention: str, candidates: list[dict[str, str]]) -> list[dict[str, object]]:
        if self.model is None:
            scored = [{"candidate": candidate, "score": _heuristic_score(query, mention, candidate)} for candidate in candidates]
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored
        vectors = [_build_features(query, mention, candidate) for candidate in candidates]
        probs = self.model.predict_proba(vectors)[:, 1]
        scored = []
        for candidate, prob in zip(candidates, probs, strict=False):
            scored.append({"candidate": candidate, "score": float(round(prob, 4))})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored


def _build_features(query: str, mention: str, candidate: dict[str, str]) -> list[float]:
    canonical_name = candidate.get("canonical_name", "")
    industry_name = candidate.get("industry_name", "")
    symbol = candidate.get("symbol", "")
    return [
        1.0 if mention == canonical_name else 0.0,
        1.0 if mention and mention in canonical_name else 0.0,
        fuzz.ratio(mention, canonical_name) / 100,
        fuzz.partial_ratio(query, canonical_name) / 100,
        fuzz.partial_ratio(query, industry_name) / 100 if industry_name else 0.0,
        _token_overlap(query, canonical_name),
        _token_overlap(query, industry_name),
        1.0 if symbol and symbol.split(".")[0] in query else 0.0,
    ]


def _sample_negative_entity_ids(alias_text: str, target_id: int, entity_ids: list[int]) -> list[int]:
    digest = hashlib.sha1(f"{alias_text}:{target_id}".encode("utf-8")).digest()
    start = int.from_bytes(digest[:4], "big") % len(entity_ids)
    negatives: list[int] = []
    offset = 0
    while len(negatives) < min(ENTITY_LINKER_NEGATIVES_PER_ALIAS, len(entity_ids) - 1) and offset < len(entity_ids):
        candidate_id = entity_ids[(start + offset) % len(entity_ids)]
        if candidate_id != target_id:
            negatives.append(candidate_id)
        offset += 1
    return negatives


def _heuristic_score(query: str, mention: str, candidate: dict[str, str]) -> float:
    features = _build_features(query, mention, candidate)
    return float(round(max(features[0], features[2], features[3], features[5]), 4))


def _fit_entity_linker_model(
    X: list[list[float]],
    y: list[int],
    *,
    fit_progress_callback: Callable[[int, int], None] | None,
    batch_size: int,
    epochs: int,
) -> SGDClassifier:
    if not X:
        raise ValueError("Cannot train entity linker with zero rows")
    labels = np.asarray(y, dtype=int)
    classes = np.asarray([0, 1], dtype=int)
    if len(set(labels.tolist())) < 2:
        raise ValueError("Cannot train entity linker with fewer than two classes")
    matrix = np.asarray(X, dtype=float)
    batch_size = max(int(batch_size), 1)
    epochs = max(int(epochs), 1)
    total_batches = max(1, math.ceil(len(matrix) / batch_size) * epochs)
    class_weights = compute_class_weight("balanced", classes=classes, y=labels)
    weight_map = dict(zip(classes, class_weights, strict=False))
    model = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42)
    rng = np.random.default_rng(42)
    completed = 0
    first_batch = True
    for _ in range(epochs):
        indices = rng.permutation(len(matrix))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            y_batch = labels[batch_indices]
            sample_weight = np.asarray([weight_map[int(label)] for label in y_batch], dtype=float)
            if first_batch:
                model.partial_fit(matrix[batch_indices], y_batch, classes=classes, sample_weight=sample_weight)
                first_batch = False
            else:
                model.partial_fit(matrix[batch_indices], y_batch, sample_weight=sample_weight)
            completed += 1
            if fit_progress_callback:
                fit_progress_callback(completed, total_batches)
    return model
