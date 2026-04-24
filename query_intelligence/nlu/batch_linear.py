from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight


BATCH_LINEAR_BATCH_SIZE = 5_000
BATCH_LINEAR_EPOCHS = 3
FitProgressCallback = Callable[[int, int], None]


def estimate_batch_fit_steps(
    row_count: int,
    *,
    batch_size: int = BATCH_LINEAR_BATCH_SIZE,
    epochs: int = BATCH_LINEAR_EPOCHS,
) -> int:
    return max(1, math.ceil(max(row_count, 1) / max(batch_size, 1)) * max(epochs, 1))


def fit_dict_sgd_classifier(
    feature_rows: Sequence[dict[str, float | str]],
    labels: Sequence[int | str | bool],
    *,
    fit_progress_callback: FitProgressCallback | None = None,
    batch_size: int = BATCH_LINEAR_BATCH_SIZE,
    epochs: int = BATCH_LINEAR_EPOCHS,
) -> tuple[DictVectorizer, SGDClassifier]:
    if not feature_rows:
        raise ValueError("Cannot train classifier with zero rows")
    y = np.array(list(labels))
    classes = np.array(sorted(set(y.tolist()), key=str))
    if len(classes) < 2:
        raise ValueError("Cannot train classifier with fewer than two classes")

    batch_size = max(int(batch_size), 1)
    epochs = max(int(epochs), 1)
    total_batches = estimate_batch_fit_steps(len(feature_rows), batch_size=batch_size, epochs=epochs)
    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit(feature_rows)
    classifier = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42)
    class_weights = compute_class_weight("balanced", classes=classes, y=y)
    weight_map = dict(zip(classes, class_weights, strict=False))

    rng = np.random.default_rng(42)
    completed = 0
    first_batch = True
    for _ in range(epochs):
        indices = rng.permutation(len(feature_rows))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_features = [feature_rows[int(index)] for index in batch_indices]
            batch_labels = y[batch_indices]
            batch_weights = np.array([weight_map[label] for label in batch_labels], dtype=float)
            x_batch = vectorizer.transform(batch_features)
            if first_batch:
                classifier.partial_fit(x_batch, batch_labels, classes=classes, sample_weight=batch_weights)
                first_batch = False
            else:
                classifier.partial_fit(x_batch, batch_labels, sample_weight=batch_weights)
            completed += 1
            if fit_progress_callback:
                fit_progress_callback(completed, total_batches)
    return vectorizer, classifier
