from __future__ import annotations

import logging
import re
from typing import Any

try:
    import jieba
except ImportError:
    jieba = None  # type: ignore[assignment]

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None  # type: ignore[assignment]

try:
    from lingua import LanguageDetectorBuilder
    from lingua import Language as LinguaLanguage
except ImportError:
    _lingua_available = False
else:
    _lingua_available = True
    _LINGUA_DETECTOR = LanguageDetectorBuilder.from_languages(
        LinguaLanguage.CHINESE, LinguaLanguage.ENGLISH
    ).build()

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:
    _nltk_available = False
else:
    _nltk_available = True
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

from .schemas import (
    FilterMeta,
    Language,
    PreprocessedDoc,
    QueryInput,
    SKIP_PRODUCT_TYPES,
    SUPPORTED_SOURCE_TYPES,
    SentimentItem,
    TextLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

_CJK_SPACE_RE = re.compile(r"(?<=[一-鿿㐀-䶿]) (?=[一-鿿㐀-䶿])")
_CJK_RANGE = re.compile(r"[一-鿿㐀-䶿]")
_LATIN_RANGE = re.compile(r"[a-zA-Z]")

_FUZZY_THRESHOLD = 85


# ---------------------------------------------------------------------------
# 1. Query-level filter
# ---------------------------------------------------------------------------


def should_skip_query(nlu_result: dict[str, Any]) -> str | None:
    """Return a skip reason if the entire query should be skipped, else None."""
    pt = nlu_result.get("product_type", {})
    if isinstance(pt, dict) and pt.get("label") in SKIP_PRODUCT_TYPES:
        return f"product_type={pt['label']}"
    return None


# ---------------------------------------------------------------------------
# 2. Text extraction
# ---------------------------------------------------------------------------


def extract_and_text_level(doc: dict[str, Any]) -> tuple[str, TextLevel]:
    """Extract text from a document, returning (text, text_level).

    text_level is "full" when body was available and non-empty.
    text_level is "short" when falling back to title+summary.
    Returns ("", "short") when no usable text is found.
    """
    body = doc.get("body")
    if doc.get("body_available") and body:
        return normalize_text(body), "full"
    if doc.get("body_available") and not body:
        return "", "short"

    # body unavailable: fall back to title + summary
    title = doc.get("title", "") or ""
    summary = doc.get("summary", "") or ""
    if summary:
        return normalize_text(f"{title}\n{summary}"), "short"
    if title:
        return normalize_text(title), "short"
    return "", "short"


# ---------------------------------------------------------------------------
# 2b. Text normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text by removing inter-CJK spaces and collapsing whitespace."""
    text = _CJK_SPACE_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Language detection
# ---------------------------------------------------------------------------


def detect_language(text: str) -> Language:
    """Detect language using lingua (primary) or character-ratio (fallback)."""
    total = len(text.strip())
    if total == 0:
        return "unknown"

    cjk = len(_CJK_RANGE.findall(text))
    latin = len(_LATIN_RANGE.findall(text))
    cjk_ratio = cjk / total
    latin_ratio = latin / total

    if _lingua_available:
        detected = _LINGUA_DETECTOR.detect_language_of(text)
        if detected is None:
            return "unknown"
        if detected == LinguaLanguage.CHINESE:
            return "mixed" if latin_ratio > 0.10 else "zh"
        return "mixed" if cjk_ratio > 0.10 else "en"

    # Fallback heuristic: 10% threshold (raised from 5%)
    if cjk_ratio > 0.10 and latin_ratio > 0.10:
        return "mixed"
    if cjk_ratio > 0.10:
        return "zh"
    if latin_ratio > 0.10:
        return "en"
    return "unknown"


# ---------------------------------------------------------------------------
# 4. Sentence splitting
# ---------------------------------------------------------------------------


def split_sentences(text: str, language: Language) -> list[str]:
    """Split text into sentences based on language."""
    text = normalize_text(text)
    if not text:
        return []

    if language == "zh":
        return _split_zh(text)
    elif language == "en":
        return _split_en(text)
    else:
        sentences = _split_zh(text)
        if len(sentences) <= 1:
            sentences = _split_en(text)
        if len(sentences) <= 1:
            sentences = re.split(r"(?<=[。！？；.!?])\s*|\n+", text)
            sentences = [s.strip() for s in sentences if s.strip()]
        return sentences


def _split_zh(text: str) -> list[str]:
    """Split Chinese text on 。！？； and newlines, keeping quotes paired."""
    raw = re.split(r"(?<=[。！？；])\s*", text)
    parts: list[str] = []
    for seg in raw:
        for line in seg.split("\n"):
            line = line.strip()
            if line:
                parts.append(line)

    if not parts:
        return []

    # Merge segments to keep Chinese quotes balanced
    _left_dq = "“"  # "
    _right_dq = "”"  # "
    _left_sq = "‘"  # '
    _right_sq = "’"  # '
    merged: list[str] = []
    buf = ""
    for part in parts:
        buf += part
        if (buf.count(_left_dq) == buf.count(_right_dq)
                and buf.count(_left_sq) == buf.count(_right_sq)):
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)

    return merged


def _split_en(text: str) -> list[str]:
    """Split English text using NLTK Punkt tokenizer with regex fallback."""
    if _nltk_available:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    # Fallback: split on sentence-ending punctuation followed by space
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# 5. Entity relevance filtering
# ---------------------------------------------------------------------------


def build_entity_names(nlu_result: dict[str, Any]) -> dict[str, str]:
    """Build a map of entity name text -> entity symbol.

    Each entity contributes its symbol, canonical_name, and mention.
    For CJK names, jieba tokens are also added as keys.
    """
    entity_map: dict[str, str] = {}
    for entity in nlu_result.get("entities", []):
        if not isinstance(entity, dict):
            continue
        symbol = str(entity.get("symbol", "")).strip()
        if not symbol:
            continue
        for field in ("symbol", "canonical_name", "mention"):
            val = entity.get(field)
            if not val:
                continue
            raw = str(val).strip()
            if not raw:
                continue
            entity_map[raw] = symbol
            if jieba is not None and len(raw) >= 2 and _CJK_RANGE.search(raw):
                for token in jieba.cut(raw):
                    token = token.strip()
                    if len(token) >= 2:
                        entity_map[token] = symbol
    return entity_map


def filter_relevant_sentences(
    sentences: list[str],
    entity_map: dict[str, str],
    title: str,
) -> tuple[list[str], list[str]]:
    """Keep sentences mentioning target entities; returns (sentences, matched_symbols).

    Fast path: exact substring match.
    Slow path: rapidfuzz partial_ratio fuzzy match.
    Fallback: title + first 3 sentences, empty symbols.
    """
    if not entity_map:
        return sentences, []

    relevant: list[str] = []
    matched_symbols: list[str] = []

    # Fast path: exact substring matching
    for s in sentences:
        for name, symbol in entity_map.items():
            if name in s:
                relevant.append(s)
                if symbol not in matched_symbols:
                    matched_symbols.append(symbol)
                break

    # Slow path: fuzzy matching
    if not relevant and fuzz is not None:
        for s in sentences:
            for name, symbol in entity_map.items():
                if fuzz.partial_ratio(name, s) >= _FUZZY_THRESHOLD:
                    relevant.append(s)
                    if symbol not in matched_symbols:
                        matched_symbols.append(symbol)
                    break

    if relevant:
        return relevant, matched_symbols

    # Fallback: title + first 3 non-empty sentences
    fallback = [s for s in sentences if s.strip()][:3]
    if title:
        fallback.insert(0, title)
    if not fallback:
        fallback = sentences[:3]
    return fallback, []


# ---------------------------------------------------------------------------
# 6. Full preprocessing pipeline
# ---------------------------------------------------------------------------


class Preprocessor:
    """End-to-end document preprocessing for sentiment analysis."""

    def skip_document(self, doc: dict[str, Any]) -> str | None:
        """Check if an individual document should be skipped. Returns reason or None."""
        source_type = doc.get("source_type", "")
        if source_type not in SUPPORTED_SOURCE_TYPES:
            return f"unsupported source_type={source_type}"
        return None

    def process_query(
        self,
        nlu_result: dict[str, Any],
        retrieval_result: dict[str, Any],
    ) -> tuple[str | None, list[PreprocessedDoc], FilterMeta]:
        """Run the full preprocessing pipeline for one query.

        Returns (query_skip_reason, processed_docs, filter_meta).
        """
        skip_reason = should_skip_query(nlu_result)
        if skip_reason:
            return skip_reason, [], FilterMeta(skipped_by_product_type=True)

        entity_map = build_entity_names(nlu_result)
        docs = retrieval_result.get("documents", [])
        processed: list[PreprocessedDoc] = []
        filter_meta = FilterMeta()

        for doc in docs:
            try:
                doc_skip = self.skip_document(doc)
                if doc_skip:
                    filter_meta.skipped_docs_count += 1
                    processed.append(PreprocessedDoc(
                        evidence_id=doc.get("evidence_id", ""),
                        source_type=doc.get("source_type", ""),
                        title=doc.get("title", "") or "",
                        raw_text="",
                        language="unknown",
                        sentences=[],
                        skipped=True,
                        skip_reason=doc_skip,
                    ))
                    continue

                title = doc.get("title", "") or ""
                raw_text, text_level = extract_and_text_level(doc)

                if not raw_text:
                    filter_meta.skipped_docs_count += 1
                    processed.append(PreprocessedDoc(
                        evidence_id=doc.get("evidence_id", ""),
                        source_type=doc.get("source_type", ""),
                        title=title,
                        raw_text="",
                        language="unknown",
                        sentences=[],
                        skipped=True,
                        skip_reason="empty text after extraction",
                    ))
                    continue

                language = detect_language(raw_text)
                sentences = split_sentences(raw_text, language)

                relevant_sentences, matched_symbols = filter_relevant_sentences(
                    sentences, entity_map, title,
                )
                relevant_excerpt = " ".join(relevant_sentences)

                filter_meta.analyzed_docs_count += 1
                if text_level == "short":
                    filter_meta.short_text_fallback_count += 1

                processed.append(PreprocessedDoc(
                    evidence_id=doc.get("evidence_id", ""),
                    source_type=doc.get("source_type", ""),
                    title=title,
                    publish_time=doc.get("publish_time"),
                    source_name=doc.get("source_name"),
                    rank_score=doc.get("rank_score"),
                    raw_text=raw_text,
                    language=language,
                    sentences=sentences,
                    entity_hits=matched_symbols,
                    text_level=text_level,
                    relevant_excerpt=relevant_excerpt,
                ))

            except Exception:
                logger.exception(
                    "Error processing document %s", doc.get("evidence_id", "unknown")
                )
                filter_meta.skipped_docs_count += 1

        return None, processed, filter_meta
