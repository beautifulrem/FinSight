from __future__ import annotations

import math
from datetime import datetime

INTENT_SOURCE_MAP = {
    "market_explanation": {"news", "market_api", "industry_sql"},
    "hold_judgment": {"fundamental_sql", "announcement", "research_note"},
    "buy_sell_timing": {"market_api", "research_note"},
    "product_info": {"faq", "product_doc"},
    "trading_rule_fee": {"faq", "product_doc"},
    "peer_compare": {"market_api", "fundamental_sql"},
    "fundamental_analysis": {"fundamental_sql", "research_note"},
    "valuation_analysis": {"fundamental_sql", "research_note"},
    "macro_policy_impact": {"macro_sql", "news"},
    "event_news_query": {"news", "announcement"},
    "risk_analysis": {"market_api", "faq"},
}

TOPIC_KEYWORDS = {
    "price": ["涨", "跌", "行情", "价格", "收盘", "开盘"],
    "news": ["新闻", "消息", "公告", "报道"],
    "industry": ["行业", "板块", "同行"],
    "macro": ["宏观", "经济", "GDP", "CPI"],
    "policy": ["政策", "监管", "央行"],
    "fundamentals": ["营收", "利润", "财报", "业绩"],
    "valuation": ["估值", "PE", "PB", "市盈率"],
    "risk": ["风险", "波动", "回撤"],
    "comparison": ["比较", "对比", "哪个"],
    "product_mechanism": ["费率", "申赎", "定投", "规则"],
}

TIME_SCOPE_DAYS = {
    "today": 1,
    "recent_3d": 3,
    "recent_1w": 7,
    "recent_1m": 30,
    "recent_1q": 90,
    "long_term": 365,
    "unspecified": 30,
}


class FeatureBuilder:
    def build(self, query_bundle: dict, candidate: dict) -> dict:
        title = candidate.get("title", "")
        body = candidate.get("body", "")
        content = f"{title} {body}"
        keywords = query_bundle.get("keywords", [])
        entity_names = query_bundle.get("entity_names", [])
        symbols = query_bundle.get("symbols", [])
        candidate_symbols = candidate.get("entity_symbols", [])
        normalized_query = query_bundle.get("normalized_query", "")

        lexical_score = float(candidate.get("retrieval_score", 0.0))

        trigram_similarity = self._char_ngram_similarity(normalized_query, title)

        entity_exact_match = (
            1.0 if any(symbol in candidate_symbols for symbol in symbols) else 0.0
        )

        alias_match = (
            1.0 if any(name in content for name in entity_names) else 0.0
        )

        title_hit = (
            1.0 if any(kw in title for kw in keywords + entity_names) else 0.0
        )

        keyword_coverage = (
            round(sum(1 for kw in keywords if kw in content) / max(len(keywords), 1), 4)
            if keywords
            else 0.0
        )

        intent_compatibility = self._intent_source_compatibility(
            query_bundle.get("intent_labels", []),
            candidate.get("source_type", ""),
        )

        topic_compatibility = self._topic_content_compatibility(
            query_bundle.get("topic_labels", []),
            content,
        )

        product_type_match = (
            1.0 if candidate.get("product_type") == query_bundle.get("product_type") else 0.0
        )
        source_credibility = float(candidate.get("credibility_score", 0.5))
        recency_score = self._recency_score(candidate.get("publish_time"))
        is_primary_disclosure = (
            1.0 if candidate.get("source_type") in ("announcement", "product_doc") else 0.0
        )
        doc_length = round(min(len(content) / 5000, 1.0), 4)

        time_window_match = self._time_window_match(
            candidate.get("publish_time"),
            query_bundle.get("time_scope", "unspecified"),
        )

        ticker_hit = (
            1.0 if any(symbol.split(".")[0] in content for symbol in symbols) else 0.0
        )

        return {
            "lexical_score": lexical_score,
            "trigram_similarity": trigram_similarity,
            "entity_exact_match": entity_exact_match,
            "alias_match": alias_match,
            "title_hit": title_hit,
            "keyword_coverage": keyword_coverage,
            "intent_compatibility": intent_compatibility,
            "topic_compatibility": topic_compatibility,
            "product_type_match": product_type_match,
            "source_credibility": source_credibility,
            "recency_score": recency_score,
            "is_primary_disclosure": is_primary_disclosure,
            "doc_length": doc_length,
            "time_window_match": time_window_match,
            "ticker_hit": ticker_hit,
        }

    def _recency_score(self, publish_time: str | None) -> float:
        if not publish_time:
            return 0.3
        try:
            published = datetime.fromisoformat(publish_time.replace("Z", "+00:00"))
            age_days = max((datetime.now(tz=published.tzinfo) - published).days, 0)
        except (ValueError, TypeError):
            return 0.3
        return round(max(0.05, math.exp(-0.693 * age_days / 60)), 4)

    def _char_ngram_similarity(self, text_a: str, text_b: str, n: int = 3) -> float:
        if not text_a or not text_b:
            return 0.0
        ngrams_a = {text_a[i : i + n] for i in range(max(len(text_a) - n + 1, 1))}
        ngrams_b = {text_b[i : i + n] for i in range(max(len(text_b) - n + 1, 1))}
        if not ngrams_a or not ngrams_b:
            return 0.0
        intersection = len(ngrams_a & ngrams_b)
        union = len(ngrams_a | ngrams_b)
        return round(intersection / union, 4) if union else 0.0

    def _intent_source_compatibility(self, intent_labels: list[dict], source_type: str) -> float:
        if not intent_labels or not source_type:
            return 0.5
        score = 0.0
        for intent in intent_labels:
            label = intent["label"] if isinstance(intent, dict) else intent
            compatible_sources = INTENT_SOURCE_MAP.get(label, set())
            if source_type in compatible_sources:
                score = max(score, float(intent["score"]) if isinstance(intent, dict) else 0.8)
        return round(score, 4) if score > 0 else 0.2

    def _topic_content_compatibility(self, topic_labels: list[dict], content: str) -> float:
        if not topic_labels or not content:
            return 0.5
        hits = 0
        total = 0
        for topic in topic_labels:
            label = topic["label"] if isinstance(topic, dict) else topic
            topic_kws = TOPIC_KEYWORDS.get(label, [])
            if topic_kws:
                total += 1
                if any(kw in content for kw in topic_kws):
                    hits += 1
        return round(hits / max(total, 1), 4) if total else 0.5

    def _time_window_match(self, publish_time: str | None, time_scope: str) -> float:
        if not publish_time:
            return 0.3
        try:
            published = datetime.fromisoformat(publish_time.replace("Z", "+00:00"))
            age_days = max((datetime.now(tz=published.tzinfo) - published).days, 0)
        except (ValueError, TypeError):
            return 0.3
        target_days = TIME_SCOPE_DAYS.get(time_scope, 30)
        if age_days <= target_days:
            return 1.0
        overshoot = (age_days - target_days) / max(target_days, 1)
        return round(max(0.1, 1.0 - overshoot * 0.5), 4)
