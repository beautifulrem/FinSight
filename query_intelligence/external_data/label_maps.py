from __future__ import annotations

import re

FINFE_SENTIMENT_MAP: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

CHNSENTICORP_SENTIMENT_MAP: dict[int, str] = {0: "negative", 1: "positive"}

TNEWS_LABEL_MAP: dict[int, str] = {
    100: "unknown",        # 民生
    101: "unknown",        # 文化
    102: "unknown",        # 娱乐
    103: "unknown",        # 体育
    104: "generic_market", # 财经
    106: "macro",          # 房产
    107: "unknown",        # 汽车
    108: "unknown",        # 教育
    109: "unknown",        # 科技
    110: "unknown",        # 军事
    112: "unknown",        # 旅游
    113: "unknown",        # 国际
    114: "stock",          # 股票
    115: "unknown",        # 农业
    116: "unknown",        # 电竞
}

NER_ORG_TAG_MAP: dict[str, str] = {
    "B-ORG": "B-ENT", "I-ORG": "I-ENT",
    "B-NT": "B-ENT", "I-NT": "I-ENT",
}

INTENT_KEYWORD_RULES: dict[str, list[str]] = {
    "price_query": ["价格", "多少钱", "股价", "净值", "报价", "涨了", "跌了", "行情", "走势", "收盘价", "开盘价"],
    "market_explanation": ["为什么涨", "为什么跌", "为啥涨", "为啥跌", "什么原因", "怎么回事", "咋回事", "原因"],
    "hold_judgment": ["值得持有", "继续拿", "还能拿", "值不值", "要不要卖", "该不该持有"],
    "buy_sell_timing": ["能买吗", "可以买", "什么时候买", "什么时候卖", "适合买入", "建仓", "加仓", "减仓", "止盈", "止损"],
    "product_info": ["是什么", "什么产品", "介绍一下", "了解一下", "产品信息", "基金经理", "成立"],
    "risk_analysis": ["风险", "波动", "回撤", "最大亏损", "安全吗", "稳定吗", "波动率"],
    "peer_compare": ["比较", "对比", "哪个好", "区别", "差异", "优劣"],
    "fundamental_analysis": ["基本面", "财报", "营收", "利润", "毛利率", "净利润", "ROE", "业绩"],
    "valuation_analysis": ["估值", "PE", "PB", "市盈率", "市净率", "贵不贵", "便宜吗", "高估", "低估"],
    "macro_policy_impact": ["宏观", "政策", "央行", "利率", "CPI", "PMI", "GDP", "降准", "降息", "货币政策", "财政政策"],
    "event_news_query": ["新闻", "消息", "公告", "最新", "事件", "资讯", "动态"],
    "trading_rule_fee": ["费率", "手续费", "佣金", "交易规则", "申赎", "定投", "分红", "T+", "交易时间"],
}

TOPIC_KEYWORD_RULES: dict[str, list[str]] = {
    "price": ["价格", "股价", "涨", "跌", "行情", "走势", "K线", "成交量", "收盘", "开盘", "净值"],
    "news": ["新闻", "消息", "公告", "动态", "资讯", "快讯", "头条"],
    "industry": ["行业", "板块", "赛道", "产业链", "上下游", "龙头", "细分"],
    "macro": ["宏观", "经济", "GDP", "CPI", "PMI", "央行", "货币", "财政", "就业"],
    "policy": ["政策", "监管", "法规", "规定", "改革", "整顿", "合规", "降准", "降息"],
    "fundamentals": ["基本面", "财报", "营收", "利润", "ROE", "毛利", "净利", "业绩", "增长"],
    "valuation": ["估值", "PE", "PB", "市盈率", "市净率", "溢价", "折价"],
    "risk": ["风险", "回撤", "波动", "亏损", "暴雷", "违约", "清盘"],
    "comparison": ["比较", "对比", "VS", "哪个", "区别", "差异", "优劣"],
    "product_mechanism": ["费率", "规则", "申赎", "定投", "分红", "T+", "ETF联接", "LOF", "场内", "场外"],
}

INTENT_KEYWORD_RULES["price_query"].extend(["price", "quote", "nav", "market price", "close", "open"])
INTENT_KEYWORD_RULES["market_explanation"].extend(["why", "reason", "what happened"])
INTENT_KEYWORD_RULES["hold_judgment"].extend(["hold", "worth holding", "keep holding"])
INTENT_KEYWORD_RULES["buy_sell_timing"].extend(["buy", "sell", "entry", "exit", "should i buy", "should i sell"])
INTENT_KEYWORD_RULES["product_info"].extend(["what is", "introduce", "tell me about", "product info"])
INTENT_KEYWORD_RULES["risk_analysis"].extend(["risk", "volatility", "drawdown", "safe"])
INTENT_KEYWORD_RULES["peer_compare"].extend(["compare", "comparison", "difference", "vs", "better"])
INTENT_KEYWORD_RULES["fundamental_analysis"].extend(["fundamental", "earnings", "revenue", "profit", "roe"])
INTENT_KEYWORD_RULES["valuation_analysis"].extend(["valuation", "pe", "pb", "overvalued", "undervalued"])
INTENT_KEYWORD_RULES["macro_policy_impact"].extend(["macro", "policy", "interest rate", "inflation", "gdp", "cpi"])
INTENT_KEYWORD_RULES["event_news_query"].extend(["news", "announcement", "headline", "update"])
INTENT_KEYWORD_RULES["trading_rule_fee"].extend(["fee", "cost", "commission", "redeem", "subscription", "trading rule"])

TOPIC_KEYWORD_RULES["price"].extend(["price", "quote", "chart", "market price", "nav"])
TOPIC_KEYWORD_RULES["news"].extend(["news", "headline", "announcement", "update"])
TOPIC_KEYWORD_RULES["industry"].extend(["industry", "sector", "supply chain"])
TOPIC_KEYWORD_RULES["macro"].extend(["macro", "economy", "inflation", "gdp", "cpi", "pmi", "central bank"])
TOPIC_KEYWORD_RULES["policy"].extend(["policy", "regulation", "rule", "reform"])
TOPIC_KEYWORD_RULES["fundamentals"].extend(["fundamental", "earnings", "revenue", "profit", "roe"])
TOPIC_KEYWORD_RULES["valuation"].extend(["valuation", "pe", "pb", "premium", "discount"])
TOPIC_KEYWORD_RULES["risk"].extend(["risk", "drawdown", "volatility", "default"])
TOPIC_KEYWORD_RULES["comparison"].extend(["compare", "difference", "vs", "better"])
TOPIC_KEYWORD_RULES["product_mechanism"].extend(["fee", "rule", "redeem", "subscription", "lof", "etf", "fund"])


def map_finfe_sentiment(label: int) -> str:
    return FINFE_SENTIMENT_MAP[label]


def map_chnsenticorp_sentiment(label: int) -> str:
    return CHNSENTICORP_SENTIMENT_MAP[label]


def map_tnews_product_type(label_id: int) -> str:
    return TNEWS_LABEL_MAP.get(label_id, "unknown")


def map_ner_tag_to_ent(tag: str) -> str:
    return NER_ORG_TAG_MAP.get(tag, "O")


def autolabel_intents(text: str) -> list[str]:
    matched: list[str] = []
    for intent, keywords in INTENT_KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text:
                matched.append(intent)
                break
    return matched or ["price_query"]


def autolabel_topics(text: str) -> list[str]:
    matched: list[str] = []
    for topic, keywords in TOPIC_KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text:
                matched.append(topic)
                break
    return matched or ["price"]


def autolabel_question_style(text: str) -> str:
    lowered = text.lower()
    if any(kw in text for kw in ["为什么", "为啥", "原因", "怎么回事", "咋"]) or any(kw in lowered for kw in ["why", "reason"]):
        return "why"
    if any(kw in text for kw in ["比较", "对比", "哪个好", "区别"]) or any(kw in lowered for kw in ["compare", "difference", " vs ", "better"]):
        return "compare"
    if any(kw in text for kw in ["建议", "应该", "该不该", "值不值", "推荐"]) or any(
        kw in lowered for kw in ["should i", "worth", "recommend", "suitable", "good to buy"]
    ):
        return "advice"
    if any(kw in text for kw in ["预测", "会不会", "未来", "走势", "前景", "明天"]) or any(
        kw in lowered for kw in ["forecast", "future", "outlook", "next", "will it"]
    ):
        return "forecast"
    return "fact"
