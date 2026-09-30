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

GENERIC_ACTION_PHRASES = {
    "能买",
    "能买吗",
    "值得买",
    "值得拿",
    "还值得",
    "还能拿",
    "止损",
    "止盈",
}

GENERIC_ENTITY_SUFFIX_MENTIONS = {
    "科技",
    "股份",
    "集团",
    "银行",
    "证券",
    "汽车",
    "医药",
    "生物",
    "能源",
    "电子",
    "电力",
    "控股",
    "保险",
}

SECTOR_CONTEXT_SUFFIXES = {"板块", "行业", "赛道", "方向", "主题"}
# Words that place an ambiguous short name in one industry ("平安的保费" is the insurer, "平安的不良率" the bank).
INDUSTRY_CONTEXT = {
    "保险": re.compile(
        r"保险|寿险|财险|产险|保费|险资|承保|偿付|赔付|新业务价值|内含价值|代理人|\binsur(?:ance|er|ers)\b|\bpremiums?\b",
        re.IGNORECASE,
    ),
    "银行": re.compile(
        r"银行|存款|贷款|揽储|不良|息差|拨备|信贷|放贷|\bbank(?:s|ing)?\b|\blend(?:ing|er)?\b|\bloans?\b|\bdeposits?\b|"
        r"non-?performing|net interest margin",
        re.IGNORECASE,
    ),
    "证券": re.compile(r"券商|证券|经纪业务|投行|自营|两融|\bbroker(?:age)?\b|\bsecurities firm\b", re.IGNORECASE),
}
QUESTION_BOUNDARY_CHARS = set("哪谁怎吗么该能好更不")
# Colloquial short names ("美的", "宁王", "工行"; see runtime_entity_assets.COLLOQUIAL_ALIASES) are common word
# fragments ("完美的", "施工行业"): they match only exactly, only as a whole jieba token, and never fuzzily.
COLLOQUIAL_ALIAS_TYPE = "colloquial_alias"
# Degree adverbs ("挺美的", "很美的", "太美的"): a colloquial alias right after one is used as an adjective.
DEGREE_ADVERB_CHARS = set("很挺真太好多更最超蛮怪够极")
# Grammatical particles and question words: a fuzzy window that differs from an alias at one of these is part of the
# sentence, not a misspelt name.
PARTICLE_CHARS = set("是的了吗呢么吧啊呀有在和与及或也都就还没不要会能该多少几谁哪什怎")


class _WordBoundaries:
    """Lazily built jieba tokenizer that knows the colloquial aliases as words (no HMM, deterministic)."""

    def __init__(self, words: set[str]) -> None:
        self._words = sorted(words)
        self._tokenizer = None
        self._last: tuple[str, set[tuple[int, int]]] | None = None  # the last question's spans

    def spans(self, text: str) -> set[tuple[int, int]]:
        if self._tokenizer is None:
            import logging

            import jieba

            jieba.setLogLevel(logging.ERROR)
            tokenizer = jieba.Tokenizer()
            for word in self._words:
                tokenizer.add_word(word)
            self._tokenizer = tokenizer
        last = self._last  # one tuple, replaced whole: safe to read from several threads
        if last is None or last[0] != text:
            last = (text, {(start, end) for _word, start, end in self._tokenizer.tokenize(text, HMM=False)})
            self._last = last
        return last[1]


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
        self._colloquial_aliases = {
            alias
            for alias, rows in self._alias_rows_by_normalized.items()
            if rows and all(row.get("alias_type") == COLLOQUIAL_ALIAS_TYPE for row in rows)
        }
        self._word_boundaries = _WordBoundaries(self._colloquial_aliases)

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
        resolved_entities = self._dedupe_resolved_entities(query, resolved_entities)
        if len(resolved_entities) == 1:
            flags = [flag for flag in flags if flag != "entity_ambiguous"]
        return resolved_entities, comparison_targets, flags + trace

    def resolve_typos_beside(self, query: str, found: list[dict]) -> tuple[list[dict], list[str]]:
        """Typo'd security names next to exact non-security mentions: ``(entities, trace)``.

        ``resolve`` runs fuzzy alias matching only when nothing matched exactly, so "贵州矛台股价多少" resolved to
        贵州茅台 while "贵州矛台的市盈率是多少" (where 市盈率 matches exactly) did not. Here the exact mentions are
        masked out and the rest of the question gets the same fuzzy alias matching; only listed securities are
        kept (a fuzzy metric or sector next to an exact one adds nothing).
        """
        masked = query
        for entity in found:
            # the mention may be a ticker ("510300.SH") while the name is written too ("沪深300ETF (510300.SH)")
            for text in sorted(
                {str(entity.get("mention") or ""), str(entity.get("canonical_name") or "")}, key=len, reverse=True
            ):
                if text:
                    masked = masked.replace(text, " " * len(text))
        if not masked.strip():
            return [], []
        resolved: list[dict] = []
        trace: list[str] = []
        flags: list[str] = []
        for mention_group in self._fuzzy_alias_mentions(masked):
            if len(mention_group["text"].strip()) < 3:
                continue  # a two-character window with one edit ("高的" -> 高铁) is noise, not a typo'd name
            if not re.search(r"[一-鿿]", mention_group["text"]):
                # Latin names differ by one character on purpose: "CSI 3000" is its own index, not a typo of CSI 300.
                continue
            candidates = []
            for row in mention_group["rows"]:
                entity = self._entity_by_id(int(row["entity_id"]))
                if entity is None or entity.get("entity_type") not in {"stock", "etf", "fund", "index"}:
                    continue
                candidates.append(self._to_candidate(entity, mention_group["text"], "alias_fuzzy", mention_group["score"]))
                trace.append(
                    f"alias_fuzzy_beside_exact: {mention_group['text']}->{entity['canonical_name']}:{mention_group['score']}"
                )
            if candidates:
                resolved.extend(self._disambiguate(query, mention_group["text"], candidates, trace, flags))
        return resolved, trace + flags

    def _disambiguate(self, query: str, mention: str, candidates: list[dict], trace: list[str], flags: list[str]) -> list[dict]:
        deduped = {}
        for candidate in candidates:
            deduped[candidate["entity_id"]] = max(
                deduped.get(candidate["entity_id"], candidate),
                candidate,
                key=lambda item: item["confidence"],
            )
        deduped_candidates = list(deduped.values())
        if len(deduped_candidates) > 1:
            chosen = self._industry_or_priority_choice(query, mention, deduped_candidates, trace)
            if chosen is not None:
                flags.append("entity_ambiguous")
                return [chosen]
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

    def alias_candidates(self, mention: str) -> list[dict[str, str]]:
        """Entity rows that share the alias ``mention`` ("平安" -> 中国平安, 平安银行), in alias-priority order."""
        rows = self._alias_rows_by_normalized.get(mention) or self._alias_rows_by_normalized.get(mention.lower()) or []
        seen: dict[int, tuple[int, dict[str, str]]] = {}
        for row in rows:
            entity_id, priority = int(row["entity_id"]), int(row["priority"])
            entity = self._entity_by_id(entity_id)
            if entity is not None and (entity_id not in seen or priority < seen[entity_id][0]):
                seen[entity_id] = (priority, entity)
        return [entity for _priority, entity in sorted(seen.values(), key=lambda item: item[0])]

    def _industry_or_priority_choice(
        self, query: str, mention: str, candidates: list[dict], trace: list[str]
    ) -> dict | None:
        """One policy for a short name shared by companies of different industries ("平安": 中国平安 / 平安银行).

        1. Industry context: words of exactly one candidate's industry elsewhere in the question decide ("平安的保费"
           -> 中国平安, "平安的不良率" -> 平安银行). Other names in the question are masked first, so the 银行 of
           "招商银行" is not context for 平安. Match type ``linked_context``.
        2. Otherwise the alias row with the better priority in the alias table (平安 -> 中国平安), whatever else the
           question says ("平安PE比行业低吗" no longer reads 行 as a bank). Match type ``linked_default``, so the agent
           can resolve it from the conversation or say which company it assumed.
        Candidates of one industry, or of equal priority without context, are left to the linker.
        """
        entities = {candidate["entity_id"]: self._entity_by_id(candidate["entity_id"]) or {} for candidate in candidates}
        industries = {entity_id: str(row.get("industry_name") or "") for entity_id, row in entities.items()}
        if len(set(industries.values())) < 2 or not all(industries.values()):
            return None
        context = query
        for group in self._exact_alias_mentions(query):
            if group["text"] != mention:
                context = context.replace(group["text"], " ")
        context = context.replace(mention, " ")
        cued = [
            candidate
            for candidate in candidates
            if (pattern := INDUSTRY_CONTEXT.get(industries[candidate["entity_id"]])) and pattern.search(context)
        ]
        if len(cued) == 1:
            winner, match_type, reason = cued[0], "linked_context", f"industry:{industries[cued[0]['entity_id']]}"
        else:
            rows = self._alias_rows_by_normalized.get(mention) or self._alias_rows_by_normalized.get(mention.lower()) or []
            priority = {
                candidate["entity_id"]: min(
                    (int(row["priority"]) for row in rows if int(row["entity_id"]) == candidate["entity_id"]),
                    default=99,
                )
                for candidate in candidates
            }
            best = min(priority.values())
            top = [candidate for candidate in candidates if priority[candidate["entity_id"]] == best]
            if len(top) != 1:
                return None
            winner, match_type, reason = top[0], "linked_default", f"alias_priority:{best}"
        trace.append(f"ambiguous_alias:{mention}->{winner['canonical_name']}:{reason}")
        return {**winner, "mention": mention, "match_type": match_type}

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
            colloquial = alias in self._colloquial_aliases
            for hit in re.finditer(re.escape(alias), query):
                if self._should_skip_exact_alias_hit(query, hit.start(), hit.end(), rows, alias):
                    continue
                if colloquial and (hit.start(), hit.end()) not in self._word_boundaries.spans(query):
                    continue  # "完美的" is not 美的集团: a colloquial alias must be a whole word
                if colloquial and hit.start() > 0 and query[hit.start() - 1] in DEGREE_ADVERB_CHARS:
                    continue  # "价格挺美的", "真美的": an adjective after a degree adverb, not the company
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
            if not alias or alias in query or alias in self._colloquial_aliases:
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
            if self._is_generic_action_phrase(best_match["text"]):
                continue
            if self._should_skip_question_word_fuzzy_match(best_match["text"], alias):
                continue
            if self._should_skip_embedded_generic_fuzzy_mention(query, best_match):
                continue
            if self._should_skip_generic_alias_fuzzy_hit(query, best_match, alias, rows):
                continue
            if self._should_skip_generic_suffix_fuzzy_match(best_match["text"], alias):
                continue
            if self._splits_a_word(query, best_match):
                continue
            if self._substitutes_a_particle(best_match["text"], alias):
                continue
            if self._replaces_the_chinese_part(best_match["text"], alias):
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

    def _splits_a_word(self, query: str, match: dict) -> bool:
        """A fuzzy window that starts or ends inside a word of the question is not a typo'd name.

        "价格挺美的" segments as 价格/挺/美的, so the window "格挺美" (one edit from 格林美) starts inside 价格; in
        "有什么影响" the window "有什" (one edit from 有色) ends inside 什么. A typo'd name ("贵州矛台", "五梁液")
        starts and ends at word edges. Only CJK windows of up to three characters are checked: longer windows are
        rarely accidental.
        """
        text = str(match["text"])
        start, end = int(match["start"]), int(match["end"])
        if not re.search(r"[一-鿿]", text):
            # Latin: "CSI 300" inside "CSI 3000" is a different name; the window must end and start at a word edge.
            before = query[start - 1] if start > 0 else " "
            after = query[end] if end < len(query) else " "
            return before.isalnum() or after.isalnum()
        if len(text) > 3 or not self._is_cjk_string(text):
            return False
        spans = self._word_boundaries.spans(query)
        starts = {span_start for span_start, _span_end in spans}
        ends = {span_end for _span_start, span_end in spans}
        return start not in starts or end not in ends

    def _replaces_the_chinese_part(self, text: str, alias: str) -> bool:
        """A window that keeps an alias's Latin part but replaces all of its Chinese characters is another name.

        "比特币ETF能买吗": the window "币ETF" is one edit from 酒ETF (酒ETF鹏华), but the edit is the whole Chinese part
        of that alias. A misspelling keeps some of the Chinese name ("黄今ETF" for 黄金ETF).
        """
        if len(text) != len(alias) or self._is_cjk_string(alias) or not re.search(r"[一-鿿]", alias):
            return False
        chinese = [index for index, char in enumerate(alias) if self._is_cjk_char(char)]
        return all(text[index] != alias[index] for index in chinese)

    @staticmethod
    def _substitutes_a_particle(text: str, alias: str) -> bool:
        """A window that differs from the alias only where it has a grammatical particle is not a typo'd name.

        "最新PMI数据是多少": the window "数据是" is one edit from 数据港, but the edited character is the sentence's
        own "是". A typo replaces a character of the name ("贵州矛台"), not with a particle of the question.
        """
        if len(text) != len(alias):
            return False
        return any(mine != theirs and mine in PARTICLE_CHARS for mine, theirs in zip(text, alias, strict=True))

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

    @staticmethod
    def _is_generic_action_phrase(value: str) -> bool:
        normalized = value.strip().lower().strip("和与及、/ ")
        return normalized in GENERIC_ACTION_PHRASES

    def _should_skip_exact_alias_hit(self, query: str, start: int, end: int, rows: list[dict[str, str]], alias_text: str | None = None) -> bool:
        entity_types = {
            entity["entity_type"]
            for row in rows
            if (entity := self._entity_by_id(int(row["entity_id"]))) is not None
        }
        alias_value = alias_text or query[start:end]
        if self._is_generic_tradable_entity_alias(alias_value, entity_types):
            return True
        if "sector" not in entity_types:
            return False
        if len(alias_value) > 2:
            return False
        previous_char = query[start - 1] if start > 0 else ""
        next_text = query[end : end + 2]
        # "影响保险公司利润", "看好银行股": a sector word followed by a company/stock noun is the sector
        sector_context = next_text in SECTOR_CONTEXT_SUFFIXES or next_text in {"公司", "企业"} or next_text[:1] == "股"
        if previous_char and self._is_cjk_char(previous_char) and not sector_context:
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
            for suffix in (*SECTOR_CONTEXT_SUFFIXES, "股"):
                if alias.endswith(suffix) and len(alias) > len(suffix):
                    values.append(alias[: -len(suffix)])
        return values

    def _is_cjk_char(self, char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"

    def _should_skip_generic_alias_fuzzy_hit(self, query: str, match: dict, alias: str, rows: list[dict[str, str]]) -> bool:
        root = self._generic_entity_suffix_root(alias)
        if root not in GENERIC_ENTITY_SUFFIX_MENTIONS:
            return False
        entity_types = {
            entity["entity_type"]
            for row in rows
            if (entity := self._entity_by_id(int(row["entity_id"]))) is not None
        }
        if self._is_generic_tradable_entity_alias(alias, entity_types):
            return True
        previous_char = query[match["start"] - 1] if match["start"] > 0 else ""
        next_text = query[match["end"] : match["end"] + 2]
        if previous_char and self._is_cjk_char(previous_char) and next_text not in SECTOR_CONTEXT_SUFFIXES:
            return True
        return False

    @staticmethod
    def _generic_entity_suffix_root(value: str) -> str:
        normalized = value.strip().lower().strip("和与及、/ ")
        for suffix in (*SECTOR_CONTEXT_SUFFIXES, "股"):
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                return normalized[: -len(suffix)]
        return normalized

    def _is_generic_tradable_entity_alias(self, alias_value: str, entity_types: set[str]) -> bool:
        root = self._generic_entity_suffix_root(alias_value)
        return root in GENERIC_ENTITY_SUFFIX_MENTIONS and bool(entity_types.intersection({"stock", "fund", "etf", "index"}))

    def _should_skip_question_word_fuzzy_match(self, mention: str, alias: str) -> bool:
        if not (self._is_cjk_string(mention) and self._is_cjk_string(alias)):
            return False
        if len(alias) > 4 or len(mention) > len(alias) + 1:
            return False
        extra_chars = set(mention) - set(alias)
        return bool(extra_chars.intersection(QUESTION_BOUNDARY_CHARS))

    def _should_skip_embedded_generic_fuzzy_mention(self, query: str, match: dict) -> bool:
        if self._generic_entity_suffix_root(match["text"]) not in GENERIC_ENTITY_SUFFIX_MENTIONS:
            return False
        previous_char = query[match["start"] - 1] if match["start"] > 0 else ""
        next_text = query[match["end"] : match["end"] + 2]
        return bool(previous_char and self._is_cjk_char(previous_char) and next_text not in SECTOR_CONTEXT_SUFFIXES)

    def _should_skip_generic_suffix_fuzzy_match(self, mention: str, alias: str) -> bool:
        if not (self._is_cjk_string(mention) and self._is_cjk_string(alias)):
            return False
        if len(alias) > 6 or len(mention) > 6:
            return False
        generic_suffixes = sorted(GENERIC_ENTITY_SUFFIX_MENTIONS, key=len, reverse=True)
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
        return [
            row
            for alias, rows in self._alias_rows_by_normalized.items()
            if alias not in self._colloquial_aliases and fuzz.ratio(mention, alias) >= 90
            for row in rows
        ]

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
