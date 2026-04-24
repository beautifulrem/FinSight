from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .entity_boundary_crf import EntityBoundaryCRF
from .entity_linker import EntityLinker
from .typo_linker import TypoLinker

GENERIC_CRYPTO_PRODUCT_MENTIONS = {
    "etf",
    "lof",
    "基金",
    "公募基金",
    "场内基金",
    "场外基金",
    "指数基金",
    "股票",
    "个股",
    "指数",
}


@dataclass
class EntityResolver:
    entities: list[dict[str, str]]
    aliases: list[dict[str, str]]
    linker: EntityLinker | None = None
    boundary_model: EntityBoundaryCRF | None = None
    typo_linker: TypoLinker | None = None

    def __post_init__(self) -> None:
        self._entity_index: dict[int, dict[str, str]] = {
            int(entity["entity_id"]): entity
            for entity in self.entities
            if entity.get("entity_id")
        }
        self._alias_rows_by_normalized: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.aliases:
            alias = row.get("normalized_alias", "")
            if alias:
                self._alias_rows_by_normalized[alias].append(row)
        self._aliases_by_first_char: dict[str, list[str]] = defaultdict(list)
        for alias in self._alias_rows_by_normalized:
            self._aliases_by_first_char[alias[0]].append(alias)

    def resolve(self, query: str) -> tuple[list[dict], list[str], list[str]]:
        resolved_entities: list[dict] = []
        flags: list[str] = []
        trace: list[str] = []

        mention_groups = self._exact_alias_mentions(query)
        if mention_groups:
            for mention_group in mention_groups:
                candidates = []
                for row in mention_group["rows"]:
                    entity = self._entity_by_id(int(row["entity_id"]))
                    if entity is None:
                        continue
                    candidates.append(self._to_candidate(entity, mention_group["text"], "alias_exact", 0.99))
                    trace.append(f"alias_exact: {mention_group['text']}->{entity['canonical_name']}")
                if candidates:
                    resolved_entities.extend(self._disambiguate(query, mention_group["text"], candidates, trace, flags))

        if not resolved_entities:
            fuzzy_alias_groups = self._fuzzy_alias_mentions(query)
            if fuzzy_alias_groups:
                for mention_group in fuzzy_alias_groups:
                    candidates = []
                    for row in mention_group["rows"]:
                        entity = self._entity_by_id(int(row["entity_id"]))
                        if entity is None:
                            continue
                        candidates.append(self._to_candidate(entity, mention_group["text"], "alias_fuzzy", mention_group["score"]))
                        trace.append(f"alias_fuzzy: {mention_group['text']}->{entity['canonical_name']}:{mention_group['score']}")
                    if candidates:
                        resolved_entities.extend(self._disambiguate(query, mention_group["text"], candidates, trace, flags))

        if not resolved_entities:
            fuzzy_candidates: list[dict] = []
            for entity in self.entities:
                if self._should_skip_alias_rows_for_query(query, entity["normalized_name"], [{"entity_id": entity["entity_id"]}]):
                    continue
                score = fuzz.partial_ratio(query, entity["normalized_name"])
                if score >= 90:
                    fuzzy_candidates.append(self._to_candidate(entity, entity["canonical_name"], "fuzzy", round(score / 100, 2)))
                    trace.append(f"fuzzy: {entity['canonical_name']}={score}")
            if fuzzy_candidates:
                resolved_entities.extend(self._disambiguate(query, fuzzy_candidates[0]["mention"], fuzzy_candidates, trace, flags))

        if not resolved_entities and self.boundary_model is not None:
            for mention in self.boundary_model.predict_mentions(query):
                if self._should_skip_crf_mention(mention):
                    trace.append(f"crf_skip:{mention}")
                    continue
                candidate_rows = [
                    row
                    for row in self._candidate_rows_for_crf_mention(mention)
                    if not self._should_skip_alias_rows_for_query(query, row["normalized_alias"], [row])
                ]
                candidates = []
                for row in candidate_rows:
                    entity = self._entity_by_id(int(row["entity_id"]))
                    if entity is None:
                        continue
                    score = 0.92 if row["normalized_alias"] == mention else round(max(fuzz.ratio(mention, row["normalized_alias"]) / 100, 0.78), 2)
                    candidates.append(self._to_candidate(entity, mention, "crf_fuzzy", score))
                if candidates:
                    resolved_entities.extend(self._disambiguate(query, mention, candidates, trace, flags))

        if not resolved_entities:
            flags.append("entity_not_found")
            comparison_targets = self._extract_comparison_targets(query)
            return [], comparison_targets, flags + trace

        resolved_entities = self._dedupe_resolved_entities(query, resolved_entities)
        if len(resolved_entities) == 1:
            flags = [flag for flag in flags if flag != "entity_ambiguous"]
        resolved_entities.sort(key=lambda item: query.find(item["mention"]) if item["mention"] in query else 999)
        comparison_targets = self._extract_comparison_targets(query)
        return resolved_entities, comparison_targets, flags + trace

    def resolve_exact(self, query: str) -> tuple[list[dict], list[str], list[str]]:
        resolved_entities: list[dict] = []
        flags: list[str] = []
        trace: list[str] = []
        for mention_group in self._exact_alias_mentions(query):
            candidates = []
            for row in mention_group["rows"]:
                entity = self._entity_by_id(int(row["entity_id"]))
                if entity is None:
                    continue
                candidates.append(self._to_candidate(entity, mention_group["text"], "alias_exact", 0.99))
                trace.append(f"alias_exact: {mention_group['text']}->{entity['canonical_name']}")
            if candidates:
                resolved_entities.extend(self._disambiguate(query, mention_group["text"], candidates, trace, flags))
        comparison_targets = self._extract_comparison_targets(query)
        if not resolved_entities:
            return [], comparison_targets, ["entity_not_found"]
        return self._dedupe_resolved_entities(query, resolved_entities), comparison_targets, flags + trace

    def _disambiguate(self, query: str, mention: str, candidates: list[dict], trace: list[str], flags: list[str]) -> list[dict]:
        deduped = {}
        for candidate in candidates:
            deduped[candidate["entity_id"]] = max(
                deduped.get(candidate["entity_id"], candidate),
                candidate,
                key=lambda item: item["confidence"],
            )
        deduped_candidates = list(deduped.values())
        if len(deduped_candidates) == 1 or self.linker is None:
            if len(deduped_candidates) > 1:
                flags.append("entity_ambiguous")
            winner = sorted(deduped_candidates, key=lambda item: item["confidence"], reverse=True)[0]
            winner["mention"] = mention
            return [winner]

        ranking = self.linker.rank(
            query,
            mention,
            [self._entity_by_id(candidate["entity_id"]) for candidate in deduped_candidates if self._entity_by_id(candidate["entity_id"])],
        )
        if not ranking:
            flags.append("entity_ambiguous")
            winner = sorted(deduped_candidates, key=lambda item: item["confidence"], reverse=True)[0]
            winner["mention"] = mention
            return [winner]

        best = ranking[0]
        best_entity = best["candidate"]
        if len(deduped_candidates) > 1:
            flags.append("entity_ambiguous")
        trace.append(f"linker:{mention}->{best_entity['canonical_name']}:{best['score']}")
        return [
            {
                "entity_id": int(best_entity["entity_id"]),
                "mention": mention,
                "entity_type": best_entity["entity_type"],
                "canonical_name": best_entity["canonical_name"],
                "symbol": best_entity["symbol"] or None,
                "exchange": best_entity["exchange"] or None,
                "confidence": round(float(best["score"]), 2),
                "match_type": "linked",
            }
        ]

    def _exact_alias_mentions(self, query: str) -> list[dict]:
        raw_matches = []
        candidate_aliases = {
            alias
            for char in set(query)
            for alias in self._aliases_by_first_char.get(char, [])
        }
        for alias in candidate_aliases:
            rows = self._alias_rows_by_normalized[alias]
            if not alias:
                continue
            if self._is_generic_product_mention(alias):
                continue
            for hit in re.finditer(re.escape(alias), query):
                if self._should_skip_exact_alias_hit(query, hit.start(), hit.end(), rows, alias):
                    continue
                raw_matches.append(
                    {
                        "start": hit.start(),
                        "end": hit.end(),
                        "text": rows[0]["alias_text"],
                        "normalized_alias": alias,
                        "priority": min(int(row["priority"]) for row in rows),
                    }
                )

        raw_matches.sort(key=lambda item: (item["start"], -(item["end"] - item["start"]), item["priority"]))
        accepted = []
        occupied: list[tuple[int, int]] = []
        for match in raw_matches:
            if any(not (match["end"] <= start or match["start"] >= end) for start, end in occupied):
                continue
            occupied.append((match["start"], match["end"]))
            accepted.append(match)

        groups = []
        for match in accepted:
            groups.append(
                {
                    "text": match["text"],
                    "rows": self._alias_rows_by_normalized[match["normalized_alias"]],
                }
            )
        return groups

    def _fuzzy_alias_mentions(self, query: str) -> list[dict]:
        raw_matches: list[dict] = []
        query_has_product_term = self._has_product_term(query)
        for alias in self._candidate_fuzzy_aliases(query, query_has_product_term):
            rows = self._alias_rows_by_normalized[alias]
            if not alias or alias in query:
                continue
            if self._is_generic_product_mention(alias):
                continue
            if self._should_skip_alias_rows_for_query(query, alias, rows):
                continue
            if len(alias) < 2 or len(alias) > len(query):
                continue
            if len(alias) <= 2 and not self._is_cjk_string(alias):
                continue
            if query_has_product_term and not self._has_product_term(alias) and len(alias) <= 2:
                continue
            best_match = self._best_fuzzy_substring_match(query, alias)
            if best_match is None:
                continue
            if self._is_generic_product_mention(best_match["text"]):
                continue
            if self._should_skip_generic_suffix_fuzzy_match(best_match["text"], alias):
                continue
            ml_score = self.typo_linker.predict_probability(query=query, mention=best_match["text"], alias=alias, heuristic_score=best_match["score"]) if self.typo_linker else best_match["score"]
            threshold = 0.72 if len(alias) <= 4 else 0.62
            if ml_score < threshold:
                continue
            raw_matches.append(
                {
                    "start": best_match["start"],
                    "end": best_match["end"],
                    "text": best_match["text"],
                    "normalized_alias": alias,
                    "priority": min(int(row["priority"]) for row in rows),
                    "score": round(ml_score, 2),
                }
            )

        raw_matches.sort(key=lambda item: (item["start"], -(item["end"] - item["start"]), item["priority"], -item["score"]))
        accepted = []
        occupied: list[tuple[int, int]] = []
        for match in raw_matches:
            if any(not (match["end"] <= start or match["start"] >= end) for start, end in occupied):
                continue
            occupied.append((match["start"], match["end"]))
            accepted.append(match)

        groups = []
        for match in accepted:
            groups.append(
                {
                    "text": match["text"],
                    "rows": self._alias_rows_by_normalized[match["normalized_alias"]],
                    "score": match["score"],
                }
            )
        return groups

    def _candidate_fuzzy_aliases(self, query: str, query_has_product_term: bool) -> list[str]:
        if len(self._alias_rows_by_normalized) <= 5_000:
            return list(self._alias_rows_by_normalized)
        candidates = {
            alias
            for char in set(query)
            for alias in self._aliases_by_first_char.get(char, [])
        }
        if query_has_product_term:
            lowered_query = query.lower()
            product_terms = ["etf", "lof", "基金"]
            for alias in self._alias_rows_by_normalized:
                lowered_alias = alias.lower()
                if any(term in lowered_query and term in lowered_alias for term in product_terms):
                    candidates.add(alias)
        return list(candidates)

    def _best_fuzzy_substring_match(self, query: str, alias: str) -> dict | None:
        candidate_lengths = {len(alias)}
        if len(alias) >= 3 and not self._is_cjk_string(alias):
            candidate_lengths.update({len(alias) - 1, len(alias) + 1})

        best_match: dict | None = None
        for length in candidate_lengths:
            if length <= 0 or length > len(query):
                continue
            for start in range(len(query) - length + 1):
                text = query[start : start + length]
                if not text.strip():
                    continue
                if alias.isascii() and not text.isascii():
                    continue
                distance = Levenshtein.distance(text, alias)
                if len(alias) <= 4:
                    if distance > 1:
                        continue
                else:
                    similarity = Levenshtein.normalized_similarity(text, alias)
                    if distance > 2 or similarity < 0.78:
                        continue
                score = round(max(Levenshtein.normalized_similarity(text, alias), 0.78), 2)
                candidate = {
                    "start": start,
                    "end": start + length,
                    "text": text,
                    "score": score,
                }
                if best_match is None or (candidate["score"], len(candidate["text"])) > (best_match["score"], len(best_match["text"])):
                    best_match = candidate
        return best_match

    def _is_cjk_string(self, text: str) -> bool:
        return all("\u4e00" <= char <= "\u9fff" for char in text if char.strip())

    def _has_product_term(self, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in ["etf", "lof"]) or any(term in text for term in ["基金", "申赎", "定投"])

    @staticmethod
    def _is_generic_product_mention(value: str) -> bool:
        normalized = value.strip().lower().strip("和与及、/ ")
        return normalized in GENERIC_CRYPTO_PRODUCT_MENTIONS

    def _should_skip_exact_alias_hit(self, query: str, start: int, end: int, rows: list[dict[str, str]], alias_text: str | None = None) -> bool:
        entity_types = {
            entity["entity_type"]
            for row in rows
            if (entity := self._entity_by_id(int(row["entity_id"]))) is not None
        }
        if "sector" not in entity_types:
            return False
        alias_value = alias_text or query[start:end]
        if len(alias_value) > 2:
            return False
        previous_char = query[start - 1] if start > 0 else ""
        next_text = query[end : end + 2]
        if previous_char and self._is_cjk_char(previous_char) and next_text not in {"板块", "行业", "赛道", "方向", "主题"}:
            return True
        return False

    def _should_skip_alias_rows_for_query(self, query: str, alias: str, rows: list[dict[str, str]]) -> bool:
        if not alias:
            return False
        for alias_value in self._embedded_sector_alias_values(alias, rows):
            hits = list(re.finditer(re.escape(alias_value), query))
            if hits and all(self._should_skip_exact_alias_hit(query, hit.start(), hit.end(), rows, alias_value) for hit in hits):
                return True
        return False

    def _embedded_sector_alias_values(self, alias: str, rows: list[dict[str, str]]) -> list[str]:
        entity_types = {
            entity["entity_type"]
            for row in rows
            if (entity := self._entity_by_id(int(row["entity_id"]))) is not None
        }
        values = [alias]
        if "sector" in entity_types:
            for suffix in ("板块", "行业", "赛道", "方向", "主题", "股"):
                if alias.endswith(suffix) and len(alias) > len(suffix):
                    values.append(alias[: -len(suffix)])
        return values

    def _is_cjk_char(self, char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"

    def _should_skip_generic_suffix_fuzzy_match(self, mention: str, alias: str) -> bool:
        if not (self._is_cjk_string(mention) and self._is_cjk_string(alias)):
            return False
        if len(alias) > 6 or len(mention) > 6:
            return False
        generic_suffixes = ["科技", "股份", "集团", "银行", "证券", "汽车", "医药", "生物", "能源", "电子", "控股"]
        for suffix in generic_suffixes:
            if not (alias.endswith(suffix) and mention.endswith(suffix)):
                continue
            alias_prefix = alias[: -len(suffix)]
            mention_prefix = mention[: -len(suffix)]
            if alias_prefix and mention_prefix and alias_prefix != mention_prefix:
                return True
        return False

    def _to_candidate(self, entity: dict[str, str], mention: str, match_type: str, confidence: float) -> dict:
        return {
            "entity_id": int(entity["entity_id"]),
            "mention": mention,
            "entity_type": entity["entity_type"],
            "canonical_name": entity["canonical_name"],
            "symbol": entity["symbol"] or None,
            "exchange": entity["exchange"] or None,
            "confidence": confidence,
            "match_type": match_type,
        }

    def _entity_by_id(self, entity_id: int) -> dict[str, str] | None:
        return self._entity_index.get(entity_id)

    def _extract_comparison_targets(self, query: str) -> list[str]:
        pair_match = re.search(
            r"([^，。? ]+?)(?:和|与|跟)([^，。? ]+?)(?:比|相比|哪个|谁更|更稳|更好|有什么区别|有何区别|有什么不同|差别)",
            query,
        )
        if pair_match:
            return [item for item in pair_match.groups() if item]
        matches = re.findall(r"(?:和|与|跟)([^，。? ]+?)(?:比|相比)", query)
        if matches:
            return matches
        fallback = re.findall(r"(?:和|与|跟)([^，。? ]+?)(?:哪个|谁更|更稳|更好|有什么区别|有何区别|有什么不同|差别)", query)
        return fallback

    def _should_skip_crf_mention(self, mention: str) -> bool:
        normalized = mention.strip().lower()
        if not normalized:
            return True
        if normalized in GENERIC_CRYPTO_PRODUCT_MENTIONS:
            return True
        if len(normalized) <= 1:
            return True
        return False

    def _candidate_rows_for_crf_mention(self, mention: str) -> list[dict[str, str]]:
        exact_rows = self._alias_rows_by_normalized.get(mention, [])
        if exact_rows:
            return exact_rows
        if len(mention) < 3:
            return []
        return [row for alias, rows in self._alias_rows_by_normalized.items() if fuzz.ratio(mention, alias) >= 90 for row in rows]

    def _dedupe_resolved_entities(self, query: str, resolved_entities: list[dict]) -> list[dict]:
        deduped: dict[int, dict] = {}
        for entity in resolved_entities:
            entity_id = int(entity["entity_id"])
            current = deduped.get(entity_id)
            if current is None or self._entity_rank(query, entity) > self._entity_rank(query, current):
                deduped[entity_id] = entity
        return list(deduped.values())

    def _entity_rank(self, query: str, entity: dict) -> tuple[float, int, int]:
        mention = entity.get("mention", "")
        position = query.find(mention) if mention and mention in query else 999
        return (
            float(entity.get("confidence", 0.0)),
            len(mention),
            -position,
        )
