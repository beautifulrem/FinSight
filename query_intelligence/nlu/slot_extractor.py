from __future__ import annotations

import re


class SlotExtractor:
    def __init__(self, synonyms: dict) -> None:
        self.synonyms = synonyms

    def extract(self, query: str, normalized_query: str, resolved_entities: list[dict] | None = None) -> dict:
        metrics_requested = []
        if "PE" in query or "市盈率" in query:
            metrics_requested.append("pe_ttm")
        if "回撤" in query:
            metrics_requested.append("max_drawdown")
        if "波动率" in query:
            metrics_requested.append("volatility")

        keywords = []
        for candidate in ["下跌", "上涨", "原因", "持有", "风险", "估值", "新闻"]:
            if candidate in query or candidate in normalized_query:
                keywords.append(candidate)
        if "跌" in query and "下跌" not in keywords:
            keywords.append("下跌")

        if any(term in query for term in ["后面", "未来", "接下来"]):
            horizon = "short_term"
        elif any(term in query for term in ["长期", "一年"]):
            horizon = "long_term"
        else:
            horizon = "short_term"

        sentiment = "anxious" if any(term in query for term in ["跌", "慌", "回撤"]) else "neutral"

        missing_slots = []
        if any(term in query for term in ["值得拿", "值得持有"]) and not resolved_entities and "ETF" not in normalized_query:
            missing_slots.append("missing_entity")
        if "和" in query and "比" in query and not re.search(r"(?:和|与|跟)([^，。? ]+?)(?:比|相比)", query):
            missing_slots.append("comparison_target")

        return {
            "keywords": keywords[:8],
            "forecast_horizon": horizon,
            "sentiment_of_user": sentiment,
            "metrics_requested": metrics_requested,
            "missing_slots": missing_slots,
        }

    def detect_question_style(self, query: str, intents: list[dict]) -> str:
        labels = {item["label"] for item in intents}
        advice_terms = ["值得", "能买吗", "买吗", "买入", "卖出", "上车", "止盈", "止损", "持有", "拿吗", "还能拿吗", "值不值得拿"]
        compare_terms = ["比", "相比", "哪个", "谁更", "更稳", "更好", "有什么区别", "有何区别", "有什么不同", "差别"]
        why_terms = ["为什么", "为啥", "咋", "咋回事"]
        if "peer_compare" in labels or any(term in query for term in compare_terms):
            return "compare"
        if ("market_explanation" in labels or any(term in query for term in why_terms)) and not any(term in query for term in advice_terms):
            return "why"
        if "hold_judgment" in labels or "buy_sell_timing" in labels:
            return "advice"
        if "market_explanation" in labels or any(term in query for term in why_terms):
            return "why"
        if any(term in query for term in ["未来", "接下来", "后市"]):
            return "forecast"
        return "fact"
