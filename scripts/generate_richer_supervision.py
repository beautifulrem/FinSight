from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_DIR = ROOT / "data"


STOCK_ENTITIES = [
    {"name": "贵州茅台", "symbol": "600519.SH", "industry": "白酒", "peer": "五粮液", "peer_symbol": "000858.SZ"},
    {"name": "五粮液", "symbol": "000858.SZ", "industry": "白酒", "peer": "贵州茅台", "peer_symbol": "600519.SH"},
    {"name": "中国平安", "symbol": "601318.SH", "industry": "保险", "peer": "平安银行", "peer_symbol": "000001.SZ"},
    {"name": "平安银行", "symbol": "000001.SZ", "industry": "银行", "peer": "中国平安", "peer_symbol": "601318.SH"},
]

ETF_ENTITIES = [
    {"name": "沪深300ETF", "symbol": "510300.SH", "industry": "宽基指数", "peer": "创业板ETF", "peer_symbol": "159915.SZ"},
    {"name": "创业板ETF", "symbol": "159915.SZ", "industry": "成长指数", "peer": "沪深300ETF", "peer_symbol": "510300.SH"},
    {"name": "证券ETF", "symbol": "512880.SH", "industry": "券商", "peer": "沪深300ETF", "peer_symbol": "510300.SH"},
]

INDEX_ENTITIES = [
    {"name": "沪深300", "symbol": "000300.SH", "industry": "宽基指数", "peer": "上证指数", "peer_symbol": "000001.SH"},
    {"name": "上证指数", "symbol": "000001.SH", "industry": "宽基指数", "peer": "沪深300", "peer_symbol": "000300.SH"},
]

MACRO_QUERIES = [
    ("宏观政策会影响白酒吗？", "macro", ["macro_policy_impact"], ["macro", "policy", "industry"], ["news", "research_note"], ["macro_sql", "industry_sql"]),
    ("CPI上行会影响消费股吗？", "macro", ["macro_policy_impact"], ["macro", "industry"], ["news", "research_note"], ["macro_sql", "industry_sql"]),
    ("宏观宽松会利好券商吗？", "macro", ["macro_policy_impact"], ["macro", "policy", "industry"], ["news", "research_note"], ["macro_sql", "industry_sql"]),
    ("利率下行会影响保险股吗？", "macro", ["macro_policy_impact"], ["macro", "industry"], ["news", "research_note"], ["macro_sql", "industry_sql"]),
    ("稳增长政策会影响大盘吗？", "macro", ["macro_policy_impact"], ["macro", "policy", "price"], ["news"], ["macro_sql", "market_api"]),
    ("PMI回升会利好制造业吗？", "macro", ["macro_policy_impact"], ["macro", "industry"], ["news"], ["macro_sql"]),
    ("CPI下降会不会利好消费板块？", "macro", ["macro_policy_impact"], ["macro", "industry"], ["news", "research_note"], ["macro_sql", "industry_sql"]),
    ("经济复苏会影响保险股吗？", "macro", ["macro_policy_impact"], ["macro", "industry"], ["news", "research_note"], ["macro_sql", "industry_sql"]),
    ("降准会利好银行股吗？", "macro", ["macro_policy_impact"], ["macro", "policy", "industry"], ["news", "research_note"], ["macro_sql", "industry_sql"]),
    ("美债收益率变化会影响成长股吗？", "macro", ["macro_policy_impact"], ["macro", "industry"], ["news"], ["macro_sql"]),
]

GENERIC_ROWS = [
    ("场内ETF和场外基金有什么区别？", "etf", ["product_info"], ["product_mechanism", "comparison"], ["faq"], []),
    ("公募基金费率包含哪些部分？", "fund", ["trading_rule_fee"], ["product_mechanism"], ["faq"], []),
    ("这个基金费率高吗？", "fund", ["trading_rule_fee"], ["product_mechanism"], ["faq", "product_doc"], []),
    ("指数基金跟踪误差怎么看？", "index", ["product_info"], ["product_mechanism", "risk"], ["faq"], []),
    ("ETF定投适合什么人？", "etf", ["product_info"], ["product_mechanism"], ["faq", "research_note"], []),
    ("宽基ETF和行业ETF哪个风险更高？", "etf", ["peer_compare", "risk_analysis"], ["comparison", "risk"], ["faq", "research_note"], []),
    ("ETF和LOF有什么区别？", "etf", ["product_info"], ["product_mechanism", "comparison"], ["faq"], []),
    ("基金申赎规则怎么看？", "fund", ["product_info"], ["product_mechanism"], ["faq"], []),
    ("ETF费率一般包括什么？", "etf", ["trading_rule_fee"], ["product_mechanism"], ["faq"], []),
    ("指数基金适合长期配置吗？", "index", ["product_info"], ["product_mechanism"], ["faq", "research_note"], ["market_api"]),
]


def question_style(query: str, intent_labels: list[str]) -> str:
    if "peer_compare" in intent_labels or any(term in query for term in ["哪个", "区别", "更稳", "更好"]):
        return "compare"
    if "market_explanation" in intent_labels or "为什么" in query:
        return "why"
    if "hold_judgment" in intent_labels or "buy_sell_timing" in intent_labels or any(term in query for term in ["能买吗", "值得", "适合"]):
        return "advice"
    if any(term in query for term in ["未来", "接下来", "后面"]):
        return "forecast"
    return "fact"


def sentiment_label(query: str, intent_labels: list[str]) -> str:
    if any(term in query for term in ["跌", "回撤", "波动大", "风险", "为什么跌"]):
        return "anxious"
    if "buy_sell_timing" in intent_labels and any(term in query for term in ["能买吗", "适合买", "定投"]):
        return "bullish"
    if "hold_judgment" in intent_labels and any(term in query for term in ["值得", "适合长期拿"]):
        return "bullish"
    if any(term in query for term in ["卖", "止盈", "止损"]):
        return "bearish"
    return "neutral"


def build_row(
    query: str,
    product_type: str,
    intents: list[str],
    topics: list[str],
    expected_document_sources: list[str],
    expected_structured_sources: list[str],
    primary_symbol: str = "",
    comparison_symbol: str = "",
) -> dict:
    return {
        "query": query,
        "product_type": product_type,
        "intent_labels": "|".join(intents),
        "topic_labels": "|".join(topics),
        "question_style": question_style(query, intents),
        "sentiment_label": sentiment_label(query, intents),
        "expected_document_sources": "|".join(expected_document_sources),
        "expected_structured_sources": "|".join(expected_structured_sources),
        "primary_symbol": primary_symbol,
        "comparison_symbol": comparison_symbol,
    }


def stock_rows(entity: dict) -> list[dict]:
    name = entity["name"]
    symbol = entity["symbol"]
    industry = entity["industry"]
    peer = entity["peer"]
    peer_symbol = entity["peer_symbol"]
    return [
        build_row(f"{name}今天为什么跌？", "stock", ["market_explanation"], ["price", "news"], ["news", "announcement", "research_note"], ["market_api", "industry_sql"], symbol),
        build_row(f"{name}最近为什么涨？", "stock", ["market_explanation"], ["price", "news"], ["news", "research_note"], ["market_api", "industry_sql"], symbol),
        build_row(f"{name}后面还值得拿吗？", "stock", ["hold_judgment"], ["risk", "fundamentals"], ["news", "announcement", "research_note"], ["market_api", "fundamental_sql"], symbol),
        build_row(f"{name}适合继续持有吗？", "stock", ["hold_judgment"], ["risk", "fundamentals"], ["research_note", "announcement"], ["market_api", "fundamental_sql"], symbol),
        build_row(f"{name}现在能买吗？", "stock", ["buy_sell_timing"], ["price", "risk"], ["news", "research_note"], ["market_api", "fundamental_sql"], symbol),
        build_row(f"{name}现在适合买入吗？", "stock", ["buy_sell_timing"], ["price", "risk"], ["news", "research_note"], ["market_api", "fundamental_sql"], symbol),
        build_row(f"{name}风险高不高？", "stock", ["risk_analysis"], ["risk"], ["research_note", "news"], ["market_api", "fundamental_sql"], symbol),
        build_row(f"{name}波动大吗？", "stock", ["risk_analysis"], ["risk", "price"], ["research_note", "news"], ["market_api"], symbol),
        build_row(f"{name}基本面怎么样？", "stock", ["fundamental_analysis"], ["fundamentals"], ["research_note", "announcement"], ["fundamental_sql"], symbol),
        build_row(f"{name}业绩怎么样？", "stock", ["fundamental_analysis"], ["fundamentals"], ["research_note", "announcement"], ["fundamental_sql"], symbol),
        build_row(f"{name}估值高不高？", "stock", ["valuation_analysis"], ["valuation", "risk"], ["research_note", "news"], ["fundamental_sql"], symbol),
        build_row(f"{name}PE高不高？", "stock", ["valuation_analysis"], ["valuation"], ["research_note"], ["fundamental_sql"], symbol),
        build_row(f"{name}现在是不是被低估了？", "stock", ["valuation_analysis"], ["valuation"], ["research_note", "news"], ["fundamental_sql"], symbol),
        build_row(f"{name}现在适合买入还是继续观察？", "stock", ["buy_sell_timing"], ["price", "risk"], ["news", "research_note"], ["market_api", "fundamental_sql"], symbol),
        build_row(f"最近有哪些{name}新闻？", "stock", ["event_news_query"], ["news"], ["news", "announcement"], [], symbol),
        build_row(f"{name}最近有什么公告？", "stock", ["event_news_query"], ["news"], ["announcement"], [], symbol),
        build_row(f"{name}和{peer}哪个更稳？", "stock", ["peer_compare", "risk_analysis"], ["comparison", "risk"], ["research_note", "news"], ["market_api", "fundamental_sql"], symbol, peer_symbol),
        build_row(f"{name}和{peer}哪个好？", "stock", ["peer_compare"], ["comparison"], ["research_note", "news"], ["market_api", "fundamental_sql"], symbol, peer_symbol),
        build_row(f"{name}和{peer}哪个更值得拿？", "stock", ["peer_compare", "hold_judgment"], ["comparison", "risk"], ["research_note", "news"], ["market_api", "fundamental_sql"], symbol, peer_symbol),
        build_row(f"{name}最近涨跌怎么样？", "stock", ["price_query"], ["price"], ["news"], ["market_api"], symbol),
        build_row(f"{name}近一周走势怎么样？", "stock", ["price_query"], ["price"], ["news"], ["market_api"], symbol),
        build_row(f"{name}短期走势强不强？", "stock", ["price_query"], ["price"], ["news"], ["market_api"], symbol),
        build_row(f"{industry}行业今天为什么跌？", "stock", ["market_explanation"], ["industry", "price", "news"], ["news", "research_note"], ["industry_sql"], symbol),
        build_row(f"{industry}行业景气度怎么样？", "stock", ["fundamental_analysis"], ["industry", "fundamentals"], ["research_note"], ["industry_sql"], symbol),
        build_row(f"{industry}行业和大盘比怎么样？", "stock", ["peer_compare"], ["industry", "comparison"], ["research_note"], ["industry_sql", "market_api"], symbol),
        build_row(f"{industry}板块现在风险高吗？", "stock", ["risk_analysis"], ["industry", "risk"], ["research_note", "news"], ["industry_sql"], symbol),
    ]


def etf_rows(entity: dict) -> list[dict]:
    name = entity["name"]
    symbol = entity["symbol"]
    peer = entity["peer"]
    peer_symbol = entity["peer_symbol"]
    return [
        build_row(f"{name}适合定投吗？", "etf", ["product_info"], ["product_mechanism"], ["faq", "product_doc", "research_note"], ["market_api"], symbol),
        build_row(f"{name}适合长期定投吗？", "etf", ["product_info"], ["product_mechanism"], ["faq", "product_doc", "research_note"], ["market_api"], symbol),
        build_row(f"{name}产品有什么特点？", "etf", ["product_info"], ["product_mechanism"], ["product_doc", "faq"], [], symbol),
        build_row(f"{name}跟踪什么指数？", "etf", ["product_info"], ["product_mechanism"], ["product_doc", "faq"], [], symbol),
        build_row(f"{name}最近能买吗？", "etf", ["buy_sell_timing"], ["price", "risk"], ["research_note", "product_doc"], ["market_api", "industry_sql"], symbol),
        build_row(f"{name}现在能买吗？", "etf", ["buy_sell_timing"], ["price", "risk"], ["research_note", "product_doc"], ["market_api", "industry_sql"], symbol),
        build_row(f"{name}适合上车吗？", "etf", ["buy_sell_timing"], ["price", "risk"], ["research_note", "product_doc"], ["market_api", "industry_sql"], symbol),
        build_row(f"{name}波动大吗？", "etf", ["risk_analysis"], ["risk", "price"], ["faq", "product_doc", "research_note"], ["market_api"], symbol),
        build_row(f"{name}风险高不高？", "etf", ["risk_analysis"], ["risk"], ["faq", "product_doc", "research_note"], ["market_api"], symbol),
        build_row(f"{name}适合长期拿吗？", "etf", ["hold_judgment"], ["risk", "product_mechanism"], ["research_note", "product_doc"], ["market_api"], symbol),
        build_row(f"{name}还值得持有吗？", "etf", ["hold_judgment"], ["risk", "price"], ["research_note", "product_doc"], ["market_api"], symbol),
        build_row(f"{name}和{peer}哪个好？", "etf", ["peer_compare"], ["comparison", "product_mechanism"], ["product_doc", "research_note"], ["market_api"], symbol, peer_symbol),
        build_row(f"{name}和{peer}哪个更适合长期定投？", "etf", ["peer_compare", "product_info"], ["comparison", "product_mechanism"], ["product_doc", "research_note"], ["market_api"], symbol, peer_symbol),
        build_row(f"{name}费率高吗？", "etf", ["trading_rule_fee"], ["product_mechanism"], ["faq", "product_doc"], [], symbol),
        build_row(f"{name}申赎规则是什么？", "etf", ["product_info"], ["product_mechanism"], ["faq", "product_doc"], [], symbol),
        build_row(f"{name}最近涨跌怎么样？", "etf", ["price_query"], ["price"], ["product_doc"], ["market_api"], symbol),
        build_row(f"{name}近一周表现怎么样？", "etf", ["price_query"], ["price"], ["product_doc"], ["market_api"], symbol),
        build_row(f"{name}现在风险大吗？", "etf", ["risk_analysis"], ["risk"], ["faq", "research_note"], ["market_api"], symbol),
        build_row(f"{name}和{peer}哪个波动更大？", "etf", ["peer_compare", "risk_analysis"], ["comparison", "risk"], ["research_note", "product_doc"], ["market_api"], symbol, peer_symbol),
        build_row(f"{name}更适合长期配置还是波段？", "etf", ["product_info", "hold_judgment"], ["product_mechanism", "risk"], ["faq", "research_note"], ["market_api"], symbol),
    ]


def index_rows(entity: dict) -> list[dict]:
    name = entity["name"]
    symbol = entity["symbol"]
    peer = entity["peer"]
    peer_symbol = entity["peer_symbol"]
    return [
        build_row(f"{name}近期表现怎么样？", "index", ["price_query"], ["price"], ["product_doc"], ["market_api"], symbol),
        build_row(f"{name}最近涨跌如何？", "index", ["price_query"], ["price"], ["product_doc"], ["market_api"], symbol),
        build_row(f"{name}和{peer}哪个好？", "index", ["peer_compare"], ["comparison"], ["product_doc", "faq"], ["market_api"], symbol, peer_symbol),
        build_row(f"{name}适合长期定投吗？", "index", ["product_info"], ["product_mechanism"], ["product_doc", "faq", "research_note"], ["market_api"], symbol),
        build_row(f"{name}适合长期配置吗？", "index", ["product_info"], ["product_mechanism"], ["product_doc", "faq", "research_note"], ["market_api"], symbol),
        build_row(f"{name}波动大吗？", "index", ["risk_analysis"], ["risk", "price"], ["faq", "product_doc"], ["market_api"], symbol),
        build_row(f"{name}和{peer}哪个更稳？", "index", ["peer_compare", "risk_analysis"], ["comparison", "risk"], ["product_doc", "faq"], ["market_api"], symbol, peer_symbol),
        build_row(f"{name}适合现在开始定投吗？", "index", ["product_info"], ["product_mechanism", "price"], ["product_doc", "research_note"], ["market_api"], symbol),
        build_row(f"{name}和{peer}哪个更适合长期配置？", "index", ["peer_compare", "product_info"], ["comparison", "product_mechanism"], ["product_doc", "faq"], ["market_api"], symbol, peer_symbol),
    ]


def generic_fund_rows() -> list[dict]:
    base = [
        ("场内ETF和场外基金有什么区别？", "etf", ["product_info"], ["product_mechanism", "comparison"], ["faq"], []),
        ("公募基金费率包含哪些部分？", "fund", ["trading_rule_fee"], ["product_mechanism"], ["faq"], []),
        ("这个基金费率高吗？", "fund", ["trading_rule_fee"], ["product_mechanism"], ["faq", "product_doc"], []),
        ("指数基金跟踪误差怎么看？", "index", ["product_info"], ["product_mechanism", "risk"], ["faq"], []),
        ("ETF定投适合什么人？", "etf", ["product_info"], ["product_mechanism"], ["faq", "research_note"], []),
        ("宽基ETF和行业ETF哪个风险更高？", "etf", ["peer_compare", "risk_analysis"], ["comparison", "risk"], ["faq", "research_note"], []),
        ("ETF和LOF有什么区别？", "etf", ["product_info"], ["product_mechanism", "comparison"], ["faq"], []),
        ("基金申赎规则怎么看？", "fund", ["product_info"], ["product_mechanism"], ["faq"], []),
        ("ETF费率一般包括什么？", "etf", ["trading_rule_fee"], ["product_mechanism"], ["faq"], []),
        ("指数基金适合长期配置吗？", "index", ["product_info"], ["product_mechanism"], ["faq", "research_note"], ["market_api"]),
        ("ETF和指数基金是一回事吗？", "etf", ["product_info"], ["product_mechanism", "comparison"], ["faq"], []),
        ("基金净值和ETF价格有什么区别？", "fund", ["product_info"], ["product_mechanism", "comparison"], ["faq"], []),
        ("宽基ETF适合新手吗？", "etf", ["product_info"], ["product_mechanism", "risk"], ["faq", "research_note"], []),
        ("行业ETF适合长期拿吗？", "etf", ["hold_judgment"], ["risk", "product_mechanism"], ["faq", "research_note"], ["market_api"]),
        ("ETF应该看净值还是价格？", "etf", ["product_info"], ["product_mechanism", "price"], ["faq"], []),
        ("基金回撤大说明什么？", "fund", ["risk_analysis"], ["risk"], ["faq"], []),
        ("宽基ETF和指数基金哪个更适合定投？", "etf", ["peer_compare", "product_info"], ["comparison", "product_mechanism"], ["faq", "research_note"], []),
        ("ETF波动大是不是就不适合长期持有？", "etf", ["risk_analysis", "hold_judgment"], ["risk", "product_mechanism"], ["faq", "research_note"], ["market_api"]),
    ]
    return [build_row(*row) for row in base]


def generate_rows() -> list[dict]:
    rows: list[dict] = []
    for entity in STOCK_ENTITIES:
        rows.extend(stock_rows(entity))
    for entity in ETF_ENTITIES:
        rows.extend(etf_rows(entity))
    for entity in INDEX_ENTITIES:
        rows.extend(index_rows(entity))
    for query, product_type, intents, topics, doc_sources, struct_sources in MACRO_QUERIES:
        rows.append(build_row(query, product_type, intents, topics, doc_sources, struct_sources))
    rows.extend(generic_fund_rows())

    deduped = []
    seen_queries = set()
    for row in rows:
        if row["query"] in seen_queries:
            continue
        seen_queries.add(row["query"])
        deduped.append(row)

    for idx, row in enumerate(deduped):
        if idx % 10 in {0, 1}:
            split = "test"
        elif idx % 10 in {2, 3}:
            split = "valid"
        else:
            split = "train"
        row["split"] = split
    return deduped


def generate_qrels(rows: list[dict], documents: list[dict]) -> list[dict]:
    qrels: list[dict] = []
    docs_by_id = {doc["evidence_id"]: doc for doc in documents}
    for row in rows:
        expected_sources = [source for source in row["expected_document_sources"].split("|") if source]
        symbol = row.get("primary_symbol", "")
        comparison_symbol = row.get("comparison_symbol", "")
        for evidence_id, doc in docs_by_id.items():
            if doc["source_type"] not in expected_sources:
                continue
            entity_symbols = set(doc.get("entity_symbols", []))
            relevance = 0
            if symbol and symbol in entity_symbols:
                relevance = 2
            elif comparison_symbol and comparison_symbol in entity_symbols:
                relevance = 1
            elif not entity_symbols and doc["source_type"] in {"faq", "product_doc", "research_note"}:
                relevance = 1
            elif doc["source_type"] == "research_note":
                relevance = 1
            if relevance:
                qrels.append({"query": row["query"], "evidence_id": evidence_id, "relevance": relevance})
    return qrels


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    query_rows = generate_rows()
    documents = json.loads((DATA_DIR / "documents.json").read_text(encoding="utf-8"))
    qrels_rows = generate_qrels(query_rows, documents)
    write_csv(DATA_DIR / "query_labels.csv", query_rows)
    write_csv(DATA_DIR / "retrieval_qrels.csv", qrels_rows)
    print(json.dumps({"query_labels": len(query_rows), "retrieval_qrels": len(qrels_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
