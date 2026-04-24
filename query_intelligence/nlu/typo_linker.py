from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from ..training_manifest import load_training_manifest
from .batch_linear import BATCH_LINEAR_BATCH_SIZE, BATCH_LINEAR_EPOCHS, FitProgressCallback, fit_dict_sgd_classifier


CONFUSION_MAP = {
    "台": ["苔", "茆"],
    "粮": ["梁"],
    "券": ["卷"],
    "行": ["航"],
    "创": ["創"],
    "創": ["创"],
}
MAX_NEGATIVE_ALIASES_PER_MENTION = 8


def _has_term(text: str, term: str) -> float:
    return float(term.lower() in text.lower())


@dataclass
class TypoLinker:
    vectorizer: DictVectorizer
    model: SGDClassifier

    @classmethod
    def build_from_dataset(cls, dataset_path: str | Path) -> "TypoLinker":
        return cls.build_from_rows(build_typo_training_rows_from_dataset(dataset_path))

    @classmethod
    def build_from_aliases(cls, aliases: list[dict[str, str]]) -> "TypoLinker":
        rows = build_typo_training_rows(aliases)
        return cls.build_from_rows(rows)

    @classmethod
    def build_from_rows(
        cls,
        rows: list[dict],
        *,
        fit_progress_callback: FitProgressCallback | None = None,
        batch_size: int = BATCH_LINEAR_BATCH_SIZE,
        epochs: int = BATCH_LINEAR_EPOCHS,
    ) -> "TypoLinker":
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
    def load_model(cls, path: str) -> "TypoLinker":
        from joblib import load

        try:
            payload = load(path)
            return cls(vectorizer=payload["vectorizer"], model=payload["model"])
        except Exception:
            fallback_dataset = Path(__file__).resolve().parents[2] / "data" / "alias_table.csv"
            return cls.build_from_dataset(fallback_dataset)

    def predict_probability(self, query: str, mention: str, alias: str, heuristic_score: float = 0.0) -> float:
        features = self.make_features(query=query, mention=mention, alias=alias, heuristic_score=heuristic_score, label=0)
        X = self.vectorizer.transform([features])
        return float(self.model.predict_proba(X)[0][1])

    @staticmethod
    def make_features(query: str, mention: str, alias: str, heuristic_score: float = 0.0, label: int = 0) -> dict[str, float | str]:  # noqa: ARG004
        mention_lower = mention.lower()
        alias_lower = alias.lower()
        return {
            "mention_len": float(len(mention)),
            "alias_len": float(len(alias)),
            "length_diff": float(abs(len(mention) - len(alias))),
            "lev_similarity": float(Levenshtein.normalized_similarity(mention, alias)),
            "fuzz_ratio": float(fuzz.ratio(mention, alias) / 100),
            "partial_ratio": float(fuzz.partial_ratio(mention, alias) / 100),
            "heuristic_score": float(heuristic_score),
            "exact_casefold": float(mention_lower == alias_lower),
            "mention_in_alias": float(bool(mention and mention in alias)),
            "alias_in_mention": float(bool(alias and alias in mention)),
            "same_prefix_char": float(bool(mention and alias and mention[0] == alias[0])),
            "same_suffix_char": float(bool(mention and alias and mention[-1] == alias[-1])),
            "query_has_etf": _has_term(query, "ETF"),
            "query_has_lof": _has_term(query, "LOF"),
            "alias_has_etf": _has_term(alias, "ETF"),
            "alias_has_lof": _has_term(alias, "LOF"),
            "mention_has_space": float(" " in mention),
            "alias_has_space": float(" " in alias),
            "alias_is_ascii": float(alias.isascii()),
            "mention_is_ascii": float(mention.isascii()),
        }


def build_typo_training_rows(aliases: list[dict[str, str]]) -> list[dict]:
    rows: list[dict] = []
    unique_aliases = []
    seen = set()
    for row in aliases:
        alias = row["normalized_alias"]
        if alias and alias not in seen:
            unique_aliases.append(alias)
            seen.add(alias)

    for alias in unique_aliases:
        positives = _generate_typo_variants(alias)
        for mention in positives:
            rows.append({"query": mention, "mention": mention, "alias": alias, "heuristic_score": 1.0 if mention == alias else 0.8, "label": 1})
            for negative_alias in _select_negative_aliases(alias, unique_aliases):
                if negative_alias == alias:
                    continue
                rows.append(
                    {
                        "query": mention,
                        "mention": mention,
                        "alias": negative_alias,
                        "heuristic_score": 0.0,
                        "label": 0,
                    }
                )
    if rows and len({int(row["label"]) for row in rows}) < 2 and unique_aliases:
        alias = unique_aliases[0]
        synthetic_alias = f"{alias}X" if alias else "NEGATIVE_ALIAS"
        rows.append(
            {
                "query": alias,
                "mention": alias,
                "alias": synthetic_alias,
                "heuristic_score": 0.0,
                "label": 0,
            }
        )
    return rows


def _select_negative_aliases(alias: str, aliases: list[str]) -> list[str]:
    candidates = [
        item
        for item in aliases
        if item != alias
        and (
            abs(len(item) - len(alias)) <= 4
            or any(term in item for term in ["ETF", "LOF"])
        )
    ]
    candidates.sort(key=lambda item: _negative_alias_rank(alias, item))
    return candidates[:MAX_NEGATIVE_ALIASES_PER_MENTION]


def _negative_alias_rank(alias: str, candidate: str) -> tuple[int, int, int, str]:
    same_prefix = int(bool(alias and candidate and alias[0] == candidate[0]))
    same_suffix = int(bool(alias and candidate and alias[-1] == candidate[-1]))
    length_gap = abs(len(alias) - len(candidate))
    digest = hashlib.sha1(f"{alias}|{candidate}".encode("utf-8")).hexdigest()
    return (-same_prefix, -same_suffix, length_gap, digest)


def build_typo_training_rows_from_dataset(dataset_path: str | Path) -> list[dict]:
    artifact_rows = _load_typo_supervision_rows(dataset_path)
    if artifact_rows:
        rows = build_typo_training_rows_from_supervision(artifact_rows)
        if rows and len({int(row["label"]) for row in rows}) >= 2:
            return rows

    aliases = _load_alias_rows_from_dataset(dataset_path)
    return build_typo_training_rows(aliases)


def build_typo_training_rows_from_supervision(rows: list[dict]) -> list[dict]:
    built_rows: list[dict] = []
    for row in rows:
        query = str(row.get("query") or "").strip()
        mention = str(row.get("mention") or row.get("surface_form") or row.get("text") or query).strip()
        alias = str(row.get("alias") or row.get("normalized_alias") or row.get("canonical_name") or "").strip()
        if not mention or not alias:
            continue
        heuristic_score = row.get("heuristic_score", row.get("score", 0.0))
        label = _coerce_label(row)
        if label is None:
            label = 1
        built_rows.append(
            {
                "query": query or mention,
                "mention": mention,
                "alias": alias,
                "heuristic_score": float(heuristic_score or 0.0),
                "label": int(label),
            }
        )
    return built_rows


def _generate_typo_variants(alias: str) -> set[str]:
    variants = {alias}
    if any(char.isalpha() for char in alias):
        variants.add(alias.lower())
        variants.add(alias.upper())
        variants.add(alias.replace("ETF", "ET F").replace("etf", "et f"))
        variants.add(alias.replace("LOF", "LO F").replace("lof", "lo f"))
    if len(alias) >= 3:
        for index in range(len(alias)):
            variants.add(alias[:index] + alias[index + 1 :])
    for index, char in enumerate(alias):
        for replacement in CONFUSION_MAP.get(char, []):
            variants.add(alias[:index] + replacement + alias[index + 1 :])
    return {item for item in variants if item}


def _load_typo_supervision_rows(dataset_path: str | Path) -> list[dict]:
    artifact_path = _resolve_manifest_artifact_path(dataset_path, "typo_supervision_path", "typo_supervision")
    if artifact_path is None or not artifact_path.exists() or artifact_path.stat().st_size == 0:
        return []
    return _load_rows(artifact_path)


def _load_alias_rows_from_dataset(dataset_path: str | Path) -> list[dict[str, str]]:
    path = Path(dataset_path)
    if path.name == "manifest.json" and path.exists():
        manifest_alias_path = _resolve_manifest_artifact_path(dataset_path, "alias_catalog_path")
        if manifest_alias_path is not None and manifest_alias_path.exists() and manifest_alias_path.stat().st_size > 0:
            return _load_alias_rows(manifest_alias_path)
        default_alias_path = Path(__file__).resolve().parents[2] / "data" / "alias_table.csv"
        return _load_alias_rows(default_alias_path) if default_alias_path.exists() and default_alias_path.stat().st_size > 0 else []
    if path.suffix.lower() == ".jsonl":
        return _load_alias_rows(path)
    if path.suffix.lower() == ".json":
        manifest = load_training_manifest(path)
        alias_path = manifest.alias_catalog_path
        if alias_path.exists() and alias_path.stat().st_size > 0:
            return _load_alias_rows(alias_path)
        default_alias_path = Path(__file__).resolve().parents[2] / "data" / "alias_table.csv"
        return _load_alias_rows(default_alias_path) if default_alias_path.exists() and default_alias_path.stat().st_size > 0 else []
    return _load_alias_rows(path)


def _load_alias_rows(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append({str(key): str(value) for key, value in payload.items()})
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [{str(key): str(value) for key, value in item.items()} for item in payload if isinstance(item, dict)]
        return [{str(key): str(value) for key, value in payload.items()}]
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else [payload]
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coerce_label(row: dict) -> int | None:
    for key in ("label", "is_positive", "positive"):
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            return int(value)
        try:
            return int(float(str(value)))
        except ValueError:
            return 1 if str(value).strip().lower() in {"true", "yes", "y", "positive", "1"} else 0
    return None


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
