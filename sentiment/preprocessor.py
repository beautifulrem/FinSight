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
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        _nltk_available = False
    else:
        _nltk_available = True

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

# Optional: reuse NLU EntityResolver for accurate entity matching.
# Falls back gracefully when query_intelligence is not available.
try:
    from query_intelligence.data_loader import load_seed_entities, load_seed_aliases
    from query_intelligence.nlu.entity_resolver import EntityResolver
except ImportError:
    EntityResolver = None  # type: ignore[assignment]
    load_seed_entities = None  # type: ignore[assignment]
    load_seed_aliases = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

_CJK_SPACE_RE = re.compile(r"(?<=[一-鿿㐀-䶿]) (?=[一-鿿㐀-䶿])")
_CJK_RANGE = re.compile(r"[一-鿿㐀-䶿]")
_LATIN_RANGE = re.compile(r"[a-zA-Z]")

_FUZZY_THRESHOLD = 85

# Language-detection heuristics (used when lingua is unavailable)
_ZH_STOPWORDS = frozenset({
    "的", "了", "是", "在", "和", "有", "就", "不", "也", "都",
    "被", "把", "对", "与", "及", "而", "或", "但", "所", "为",
    "从", "到", "要", "会", "能", "个", "以",
})
_EN_STOPWORDS = frozenset({
    "the", "is", "are", "was", "were", "has", "have", "had", "been",
    "this", "that", "these", "those", "with", "from", "after", "before",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "but", "and", "for", "its", "his", "her", "their", "our",
})

_ZH_PUNCT = "。！？；：、，""''《》（）【】…—～"
_EN_PUNCT = ".!?;:\"'\"()-"

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
        if detected == LinguaLanguage.CHINESE:
            return "mixed" if latin_ratio > 0.10 else "zh"
        if detected == LinguaLanguage.ENGLISH:
            return "mixed" if cjk_ratio > 0.10 else "en"
        # lingua returned None — fall through to heuristic below

    # Fallback heuristic: 10% character-ratio threshold (raised from 5%),
    # supplemented by stopword and punctuation signals for tiebreaking.
    if cjk_ratio > 0.10 and latin_ratio > 0.10:
        return "mixed"
    if cjk_ratio > 0.10:
        return "zh"
    if latin_ratio > 0.10:
        return "en"

    # Neither character ratio exceeded 10% — use stopwords + punctuation
    text_lower = text.lower()
    zh_stop = sum(1 for w in _ZH_STOPWORDS if w in text)
    en_stop = sum(1 for w in _EN_STOPWORDS if w in text_lower)
    zh_punct = sum(1 for c in text if c in _ZH_PUNCT)
    en_punct = sum(1 for c in text if c in _EN_PUNCT)

    zh_signal = zh_stop * 2 + zh_punct
    en_signal = en_stop * 2 + en_punct

    if zh_signal > en_signal and zh_signal >= 2:
        return "mixed" if latin_ratio > 0.05 and cjk_ratio > 0.05 else "zh"
    if en_signal > zh_signal and en_signal >= 2:
        return "mixed" if latin_ratio > 0.05 and cjk_ratio > 0.05 else "en"
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
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            pass  # fallback to regex

    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# 5. Entity relevance filtering
# ---------------------------------------------------------------------------


def build_entity_names(nlu_result: dict[str, Any]) -> dict[str, str]:
    """Build a map of entity name text -> entity symbol.

    Each entity contributes its symbol, canonical_name, and mention.
    No jieba token expansion — exact names only, to avoid false positives
    from shared sub-tokens (e.g. "平安" matching 平安银行 when the sentence
    is about 平安证券).
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
    return entity_map


def _extract_target_symbols(nlu_result: dict[str, Any]) -> set[str]:
    """Extract the set of target entity symbols from an NLU result."""
    symbols: set[str] = set()
    for entity in nlu_result.get("entities", []):
        if not isinstance(entity, dict):
            continue
        symbol = str(entity.get("symbol", "")).strip()
        if symbol:
            symbols.add(symbol)
    return symbols


_CORP_PATTERN = re.compile(
    r"[A-Z一-鿿]{2,}(?:公司|股份|集团|银行|证券|保险|实业|控股)"
)


def _has_non_target_entity(sentence: str, entity_map: dict[str, str]) -> bool:
    """Detect if sentence mentions an entity not in the target entity_map.

    Uses jieba POS tagging for Chinese; falls back to pattern matching.
    """
    if jieba is not None:
        try:
            import jieba.posseg as pseg

            for word, flag in pseg.cut(sentence):
                if flag in ("nt", "nz") and len(word) >= 2:
                    if word not in entity_map:
                        return True
        except (ImportError, AttributeError):
            pass

    for m in _CORP_PATTERN.finditer(sentence):
        name = m.group()
        if name not in entity_map:
            return True

    return False


def filter_relevant_sentences(
    sentences: list[str],
    entity_map: dict[str, str],
    title: str,
    *,
    entity_resolver: EntityResolver | None = None,
    target_symbols: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """3-tier entity relevance filtering.

    Strong  — sentence mentions a target entity          → keep (full weight)
    Exclude — sentence mentions a non-target entity       → discard
    Generic — sentence has no entity reference at all      → keep (anaphora candidate)

    Strong sentences come first, followed by Generic.
    If no Strong or Generic survives, falls back to title + first 3 sentences.

    When *entity_resolver* is provided, entity matching uses the NLU
    EntityResolver for accurate alias-aware detection instead of plain
    substring matching.  Non-target detection also switches from jieba POS
    heuristics to the resolver's entity catalog.
    """
    if not entity_map:
        return sentences, []

    strong: list[str] = []
    generic: list[str] = []
    matched_symbols: list[str] = []

    # ---- resolver path ---------------------------------------------------
    if entity_resolver is not None and target_symbols is not None:
        for s in sentences:
            resolved, _comparison_targets, _trace = entity_resolver.resolve_exact(s)
            if resolved:
                hit_target = False
                for candidate in resolved:
                    sym = (candidate.get("symbol") or "").strip()
                    if sym in target_symbols:
                        hit_target = True
                        if sym not in matched_symbols:
                            matched_symbols.append(sym)
                if hit_target:
                    strong.append(s)
                else:
                    # resolved entities exist but none are targets → exclude
                    continue
            else:
                # EntityResolver found nothing, but the sentence could still
                # mention non-catalog entities (e.g. US stocks like 微软, 特斯拉).
                # Fall back to jieba POS heuristic to catch those.
                if _has_non_target_entity(s, entity_map):
                    continue
                generic.append(s)

        # fuzzy fallback when no strong match was found
        if not strong and not generic and fuzz is not None:
            for s in sentences:
                for name, symbol in entity_map.items():
                    if fuzz.partial_ratio(name, s) >= _FUZZY_THRESHOLD:
                        strong.append(s)
                        if symbol not in matched_symbols:
                            matched_symbols.append(symbol)
                        break

        result = strong + generic
        if result:
            return result, matched_symbols

        fallback = [s for s in sentences if s.strip()][:3]
        if title:
            fallback.insert(0, title)
        if not fallback:
            fallback = sentences[:3]
        return fallback, []

    # ---- substring fallback path -----------------------------------------
    for s in sentences:
        # 1. Strong: exact substring match against target entity names
        matched = False
        for name, symbol in entity_map.items():
            if name in s:
                strong.append(s)
                if symbol not in matched_symbols:
                    matched_symbols.append(symbol)
                matched = True
                break

        if matched:
            continue

        # 2. Exclude: non-target entity present
        if _has_non_target_entity(s, entity_map):
            continue

        # 3. Generic: no entity reference found (keep as anaphora candidate)
        generic.append(s)

    # 4. Fuzzy-path for cases where no Strong match was found (fallback)
    if not strong and not generic and fuzz is not None:
        for s in sentences:
            for name, symbol in entity_map.items():
                if fuzz.partial_ratio(name, s) >= _FUZZY_THRESHOLD:
                    strong.append(s)
                    if symbol not in matched_symbols:
                        matched_symbols.append(symbol)
                    break

    result = strong + generic
    if result:
        return result, matched_symbols

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
    """End-to-end document preprocessing for sentiment analysis.

    Parameters
    ----------
    entity_resolver:
        Optional NLU ``EntityResolver`` for accurate entity matching.
        When provided, the resolver is used to detect target and non-target
        entity mentions in each sentence, replacing the simpler substring /
        jieba-POS heuristics.  Build one with :meth:`build_default_resolver`
        or pass your own pre-built instance.
    """

    def __init__(self, entity_resolver: EntityResolver | None = None):
        self._entity_resolver = entity_resolver

    # ------------------------------------------------------------------
    # resolver construction
    # ------------------------------------------------------------------

    @classmethod
    def build_default_resolver(cls) -> EntityResolver | None:
        """Build an ``EntityResolver`` from the seed entity catalog.

        Returns ``None`` when the query_intelligence package or seed data
        is unavailable (the preprocessor will then fall back to substring
        matching).
        """
        if EntityResolver is None or load_seed_entities is None or load_seed_aliases is None:
            return None
        try:
            entities = load_seed_entities()
            aliases = load_seed_aliases()
            if not entities or not aliases:
                return None
            return EntityResolver(entities=entities, aliases=aliases)
        except Exception:
            logger.warning("Failed to build default EntityResolver", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # document-level skip
    # ------------------------------------------------------------------

    def skip_document(self, doc: dict[str, Any]) -> str | None:
        """Check if an individual document should be skipped. Returns reason or None."""
        source_type = doc.get("source_type", "")
        if source_type not in SUPPORTED_SOURCE_TYPES:
            return f"unsupported source_type={source_type}"
        return None

    # ------------------------------------------------------------------
    # main pipeline
    # ------------------------------------------------------------------

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
        target_symbols = _extract_target_symbols(nlu_result)
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
                    sentences,
                    entity_map,
                    title,
                    entity_resolver=self._entity_resolver,
                    target_symbols=target_symbols,
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
                    sentences=relevant_sentences,
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
