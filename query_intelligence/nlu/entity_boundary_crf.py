from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import sklearn_crfsuite


def _char_features(text: str, index: int) -> dict[str, object]:
    char = text[index]
    features: dict[str, object] = {
        "bias": 1.0,
        "char": char,
        "is_digit": char.isdigit(),
        "is_alpha": char.isalpha(),
        "is_upper": char.isupper(),
        "is_space": char.isspace(),
    }
    if index > 0:
        prev = text[index - 1]
        features.update(
            {
                "-1:char": prev,
                "-1:is_digit": prev.isdigit(),
                "-1:is_alpha": prev.isalpha(),
            }
        )
    else:
        features["BOS"] = True

    if index > 1:
        prev2 = text[index - 2]
        features.update(
            {
                "-2:char": prev2,
                "-2:is_digit": prev2.isdigit(),
                "-2:is_alpha": prev2.isalpha(),
            }
        )

    if index < len(text) - 1:
        nxt = text[index + 1]
        features.update(
            {
                "+1:char": nxt,
                "+1:is_digit": nxt.isdigit(),
                "+1:is_alpha": nxt.isalpha(),
            }
        )
    else:
        features["EOS"] = True

    if index < len(text) - 2:
        nxt2 = text[index + 2]
        features.update(
            {
                "+2:char": nxt2,
                "+2:is_digit": nxt2.isdigit(),
                "+2:is_alpha": nxt2.isalpha(),
            }
        )
    return features


def _label_query(query: str, aliases: list[str]) -> list[str]:
    labels = ["O"] * len(query)
    ordered_aliases = sorted({alias for alias in aliases if alias}, key=len, reverse=True)
    for alias in ordered_aliases:
        search_start = 0
        while True:
            pos = query.find(alias, search_start)
            if pos == -1:
                break
            span_end = pos + len(alias)
            if any(label != "O" for label in labels[pos:span_end]):
                search_start = pos + 1
                continue
            labels[pos] = "B-ENT"
            for idx in range(pos + 1, span_end):
                labels[idx] = "I-ENT"
            search_start = span_end
    return labels


def _labels_from_spans(query: str, spans: list[dict]) -> list[str]:
    labels = ["O"] * len(query)
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        if start < 0 or end > len(query) or start >= end:
            continue
        labels[start] = "B-ENT"
        for idx in range(start + 1, end):
            labels[idx] = "I-ENT"
    return labels


@dataclass
class EntityBoundaryCRF:
    model: sklearn_crfsuite.CRF

    @classmethod
    def build_from_queries(
        cls,
        queries: list[str],
        alias_values: list[str],
        *,
        feature_progress_callback: Callable[[int, int], None] | None = None,
        verbose: bool = False,
    ) -> "EntityBoundaryCRF":
        X = []
        for index, query in enumerate(queries, start=1):
            X.append([_char_features(query, idx) for idx in range(len(query))])
            if feature_progress_callback:
                feature_progress_callback(index, len(queries))
        y = [_label_query(query, alias_values) for query in queries]
        model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
            verbose=verbose,
        )
        model.fit(X, y)
        return cls(model=model)

    @classmethod
    def build_from_annotations(
        cls,
        annotations: list[dict],
        *,
        feature_progress_callback: Callable[[int, int], None] | None = None,
        verbose: bool = False,
    ) -> "EntityBoundaryCRF":
        X = []
        for index, item in enumerate(annotations, start=1):
            X.append([_char_features(item["query"], idx) for idx in range(len(item["query"]))])
            if feature_progress_callback:
                feature_progress_callback(index, len(annotations))
        y = [_labels_from_spans(item["query"], item["spans"]) for item in annotations]
        model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=150,
            all_possible_transitions=True,
            verbose=verbose,
        )
        model.fit(X, y)
        return cls(model=model)

    def predict_mentions(self, query: str) -> list[str]:
        if not query:
            return []
        labels = self.model.predict_single([_char_features(query, idx) for idx in range(len(query))])
        mentions: list[str] = []
        current = []
        for char, label in zip(query, labels, strict=False):
            if label == "B-ENT":
                if current:
                    mentions.append("".join(current))
                current = [char]
            elif label == "I-ENT" and current:
                current.append(char)
            else:
                if current:
                    mentions.append("".join(current))
                    current = []
        if current:
            mentions.append("".join(current))
        return [mention for mention in mentions if mention.strip()]
