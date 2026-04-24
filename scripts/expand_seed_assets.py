from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_DIR = ROOT / "data"


ENTITY_ADDITIONS = [
    {
        "entity_id": "8",
        "canonical_name": "中国平安",
        "normalized_name": "中国平安",
        "symbol": "601318.SH",
        "market": "CN",
        "exchange": "SSE",
        "entity_type": "stock",
        "industry_name": "保险",
        "status": "active",
    },
    {
        "entity_id": "9",
        "canonical_name": "创业板ETF",
        "normalized_name": "创业板ETF",
        "symbol": "159915.SZ",
        "market": "CN",
        "exchange": "SZSE",
        "entity_type": "etf",
        "industry_name": "成长指数",
        "status": "active",
    },
    {
        "entity_id": "10",
        "canonical_name": "证券ETF",
        "normalized_name": "证券ETF",
        "symbol": "512880.SH",
        "market": "CN",
        "exchange": "SSE",
        "entity_type": "etf",
        "industry_name": "券商",
        "status": "active",
    },
    {
        "entity_id": "11",
        "canonical_name": "平安银行",
        "normalized_name": "平安银行",
        "symbol": "000001.SZ",
        "market": "CN",
        "exchange": "SZSE",
        "entity_type": "stock",
        "industry_name": "银行",
        "status": "active",
    },
]


ALIAS_ADDITIONS = [
    {"alias_id": "11", "entity_id": "8", "alias_text": "平安", "normalized_alias": "平安", "alias_type": "common_alias", "priority": "3", "is_official": "false"},
    {"alias_id": "12", "entity_id": "8", "alias_text": "中国平安", "normalized_alias": "中国平安", "alias_type": "official_name", "priority": "2", "is_official": "true"},
    {"alias_id": "13", "entity_id": "9", "alias_text": "创业板ETF", "normalized_alias": "创业板ETF", "alias_type": "official_name", "priority": "2", "is_official": "true"},
    {"alias_id": "14", "entity_id": "9", "alias_text": "创业板指ETF", "normalized_alias": "创业板指ETF", "alias_type": "common_alias", "priority": "3", "is_official": "false"},
    {"alias_id": "15", "entity_id": "10", "alias_text": "证券ETF", "normalized_alias": "证券ETF", "alias_type": "official_name", "priority": "2", "is_official": "true"},
    {"alias_id": "16", "entity_id": "11", "alias_text": "平安银行", "normalized_alias": "平安银行", "alias_type": "official_name", "priority": "1", "is_official": "true"},
    {"alias_id": "17", "entity_id": "11", "alias_text": "平安", "normalized_alias": "平安", "alias_type": "common_alias", "priority": "3", "is_official": "false"},
]


DOCUMENT_ADDITIONS = [
    {
        "evidence_id": "faq_002",
        "source_type": "faq",
        "source_name": "internal_faq",
        "title": "ETF 定投适合什么投资者",
        "summary": "ETF 定投更适合长期分批投资宽基指数的用户。",
        "body": "ETF 定投通常适合长期投资、分散择时风险、偏好透明指数策略的投资者。",
        "publish_time": "2026-04-01T09:00:00+08:00",
        "product_type": "etf",
        "credibility_score": 0.88,
        "entity_symbols": ["510300.SH", "159915.SZ"],
    },
    {
        "evidence_id": "faq_003",
        "source_type": "faq",
        "source_name": "internal_faq",
        "title": "场内 ETF 和场外基金有什么区别",
        "summary": "交易方式、费率结构和流动性不同。",
        "body": "场内 ETF 像股票一样交易，场外基金按净值申赎，费率结构和流动性也有差异。",
        "publish_time": "2026-04-02T09:00:00+08:00",
        "product_type": "etf",
        "credibility_score": 0.9,
        "entity_symbols": [],
    },
    {
        "evidence_id": "faq_004",
        "source_type": "faq",
        "source_name": "internal_faq",
        "title": "公募基金费率主要包括哪些部分",
        "summary": "基金费率通常包括申购费、赎回费、管理费和托管费。",
        "body": "公募基金的主要费率包括申购费、赎回费、管理费、托管费，不同基金还有销售服务费。",
        "publish_time": "2026-04-03T09:00:00+08:00",
        "product_type": "fund",
        "credibility_score": 0.9,
        "entity_symbols": [],
    },
    {
        "evidence_id": "faq_005",
        "source_type": "faq",
        "source_name": "internal_faq",
        "title": "指数基金的跟踪误差怎么看",
        "summary": "跟踪误差反映基金复制指数的偏离程度。",
        "body": "跟踪误差越小，说明指数基金越贴近标的指数表现。它受费用、抽样复制和现金拖累等因素影响。",
        "publish_time": "2026-04-04T09:00:00+08:00",
        "product_type": "index",
        "credibility_score": 0.88,
        "entity_symbols": ["000300.SH"],
    },
    {
        "evidence_id": "product_doc_002",
        "source_type": "product_doc",
        "source_name": "fund_docs",
        "title": "创业板ETF 产品概要",
        "summary": "创业板 ETF 跟踪创业板指数，波动高于宽基大盘。",
        "body": "创业板 ETF 跟踪成长风格指数，适合希望配置高成长板块的投资者，但波动通常高于宽基ETF。",
        "publish_time": "2026-03-05T10:00:00+08:00",
        "product_type": "etf",
        "credibility_score": 0.91,
        "entity_symbols": ["159915.SZ"],
    },
    {
        "evidence_id": "product_doc_003",
        "source_type": "product_doc",
        "source_name": "fund_docs",
        "title": "证券ETF 产品概要",
        "summary": "证券ETF 聚焦券商板块，弹性高，周期属性更强。",
        "body": "证券ETF 跟踪券商行业指数，受市场成交、资本市场改革和风险偏好影响较大。",
        "publish_time": "2026-03-08T10:00:00+08:00",
        "product_type": "etf",
        "credibility_score": 0.9,
        "entity_symbols": ["512880.SH"],
    },
    {
        "evidence_id": "research_note_001",
        "source_type": "research_note",
        "source_name": "public_research",
        "title": "白酒行业景气度观察：高端酒需求韧性仍在",
        "summary": "高端白酒需求仍有韧性，但短期市场风险偏好影响估值波动。",
        "body": "白酒行业短期受市场风格影响出现回调，但高端酒需求韧性仍在，行业盈利能力维持较高水平。",
        "publish_time": "2026-04-10T09:30:00+08:00",
        "product_type": "stock",
        "credibility_score": 0.8,
        "entity_symbols": ["600519.SH", "000858.SZ"],
    },
    {
        "evidence_id": "research_note_002",
        "source_type": "research_note",
        "source_name": "public_research",
        "title": "保险板块估值复盘：中国平安关注度回升",
        "summary": "保险板块估值处于历史中枢偏下位置。",
        "body": "随着权益市场回暖和负债端改善预期增强，中国平安等保险龙头估值关注度回升。",
        "publish_time": "2026-04-09T09:30:00+08:00",
        "product_type": "stock",
        "credibility_score": 0.79,
        "entity_symbols": ["601318.SH"],
    },
    {
        "evidence_id": "research_note_003",
        "source_type": "research_note",
        "source_name": "public_research",
        "title": "宽基 ETF 定投策略摘要",
        "summary": "宽基 ETF 适合长期定投，波动管理是关键。",
        "body": "宽基 ETF 定投适合长期资金管理，通过定期投入平滑波动并降低择时压力。",
        "publish_time": "2026-04-11T09:30:00+08:00",
        "product_type": "etf",
        "credibility_score": 0.82,
        "entity_symbols": ["510300.SH", "159915.SZ"],
    },
    {
        "evidence_id": "research_note_004",
        "source_type": "research_note",
        "source_name": "public_research",
        "title": "证券ETF交易弹性与择时风险摘要",
        "summary": "证券ETF波动较高，适合风险承受能力较强的投资者。",
        "body": "证券ETF受市场情绪和成交量影响明显，弹性高，但短线波动也更大，买卖时机判断比宽基ETF更重要。",
        "publish_time": "2026-04-14T09:30:00+08:00",
        "product_type": "etf",
        "credibility_score": 0.8,
        "entity_symbols": ["512880.SH"],
    },
    {
        "evidence_id": "news_601318_001",
        "source_type": "news",
        "source_name": "public_news",
        "title": "中国平安估值修复预期升温，保险板块关注度回升",
        "summary": "保险板块估值修复预期增强，中国平安受到资金关注。",
        "body": "随着资本市场情绪修复和负债端改善预期增强，中国平安等龙头保险公司估值修复逻辑受到关注。",
        "publish_time": "2026-04-18T10:00:00+08:00",
        "product_type": "stock",
        "credibility_score": 0.79,
        "entity_symbols": ["601318.SH"],
    },
    {
        "evidence_id": "news_000001SZ_001",
        "source_type": "news",
        "source_name": "public_news",
        "title": "平安银行零售业务修复预期升温",
        "summary": "银行板块估值修复背景下，平安银行受到关注。",
        "body": "随着银行板块估值修复和零售业务改善预期提升，平安银行短期关注度回升。",
        "publish_time": "2026-04-18T09:30:00+08:00",
        "product_type": "stock",
        "credibility_score": 0.78,
        "entity_symbols": ["000001.SZ"],
    },
    {
        "evidence_id": "announcement_000001SZ_001",
        "source_type": "announcement",
        "source_name": "cninfo_seed",
        "title": "平安银行关于季度报告摘要的公告",
        "summary": "披露季度经营与资产质量情况。",
        "body": "平安银行披露季度报告摘要，更新零售、对公和资产质量相关指标。",
        "publish_time": "2026-04-17T18:30:00+08:00",
        "product_type": "stock",
        "credibility_score": 0.97,
        "entity_symbols": ["000001.SZ"],
    },
    {
        "evidence_id": "announcement_601318_001",
        "source_type": "announcement",
        "source_name": "cninfo_seed",
        "title": "中国平安关于年度报告摘要的公告",
        "summary": "披露年度经营与利润情况。",
        "body": "中国平安披露年度报告摘要，更新寿险、财险与综合金融业务经营情况。",
        "publish_time": "2026-04-17T18:00:00+08:00",
        "product_type": "stock",
        "credibility_score": 0.97,
        "entity_symbols": ["601318.SH"],
    },
    {
        "evidence_id": "news_macro_001",
        "source_type": "news",
        "source_name": "macro_public",
        "title": "稳增长政策继续推进，市场关注消费与金融修复",
        "summary": "政策层面继续释放稳增长信号。",
        "body": "宏观政策继续围绕稳增长与促消费展开，市场关注消费和金融板块的后续修复节奏。",
        "publish_time": "2026-04-12T08:30:00+08:00",
        "product_type": "macro",
        "credibility_score": 0.77,
        "entity_symbols": [],
    },
    {
        "evidence_id": "news_etf_001",
        "source_type": "news",
        "source_name": "fund_public",
        "title": "宽基ETF成交活跃，定投资金持续流入",
        "summary": "多只宽基 ETF 成交活跃，资金关注度提升。",
        "body": "近期多只宽基ETF成交活跃，市场对被动投资工具的关注持续提升。",
        "publish_time": "2026-04-13T10:00:00+08:00",
        "product_type": "etf",
        "credibility_score": 0.75,
        "entity_symbols": ["510300.SH", "159915.SZ"],
    }
]


STRUCTURED_MARKET_ADDITIONS = {
    "000300.SH": {
        "symbol": "000300.SH",
        "trade_date": "2026-04-22",
        "open": 3988.4,
        "high": 4012.7,
        "low": 3979.1,
        "close": 4005.2,
        "pct_change_1d": 0.42,
        "volume": 0,
        "amount": 0,
        "history": [
            {"trade_date": "2026-04-22", "open": 3988.4, "high": 4012.7, "low": 3979.1, "close": 4005.2, "pct_change_1d": 0.42, "volume": 0, "amount": 0},
            {"trade_date": "2026-04-21", "open": 3972.0, "high": 3991.2, "low": 3950.3, "close": 3988.5, "pct_change_1d": 0.11, "volume": 0, "amount": 0}
        ],
        "source_name": "seed"
    },
    "159915.SZ": {
        "symbol": "159915.SZ",
        "trade_date": "2026-04-22",
        "open": 2.438,
        "high": 2.471,
        "low": 2.431,
        "close": 2.465,
        "pct_change_1d": 0.86,
        "volume": 684000000,
        "amount": 1670000000,
        "history": [
            {"trade_date": "2026-04-22", "open": 2.438, "high": 2.471, "low": 2.431, "close": 2.465, "pct_change_1d": 0.86, "volume": 684000000, "amount": 1670000000},
            {"trade_date": "2026-04-21", "open": 2.421, "high": 2.446, "low": 2.417, "close": 2.444, "pct_change_1d": 0.31, "volume": 601000000, "amount": 1450000000}
        ],
        "source_name": "seed"
    },
    "512880.SH": {
        "symbol": "512880.SH",
        "trade_date": "2026-04-22",
        "open": 1.012,
        "high": 1.026,
        "low": 1.008,
        "close": 1.021,
        "pct_change_1d": 0.59,
        "volume": 432000000,
        "amount": 441000000,
        "history": [
            {"trade_date": "2026-04-22", "open": 1.012, "high": 1.026, "low": 1.008, "close": 1.021, "pct_change_1d": 0.59, "volume": 432000000, "amount": 441000000},
            {"trade_date": "2026-04-21", "open": 1.006, "high": 1.019, "low": 1.001, "close": 1.015, "pct_change_1d": 0.20, "volume": 398000000, "amount": 404000000}
        ],
        "source_name": "seed"
    },
    "601318.SH": {
        "symbol": "601318.SH",
        "trade_date": "2026-04-22",
        "open": 53.22,
        "high": 53.88,
        "low": 52.95,
        "close": 53.61,
        "pct_change_1d": 0.73,
        "volume": 124000000,
        "amount": 6640000000,
        "history": [
            {"trade_date": "2026-04-22", "open": 53.22, "high": 53.88, "low": 52.95, "close": 53.61, "pct_change_1d": 0.73, "volume": 124000000, "amount": 6640000000},
            {"trade_date": "2026-04-21", "open": 52.88, "high": 53.41, "low": 52.67, "close": 53.22, "pct_change_1d": 0.45, "volume": 117000000, "amount": 6200000000}
        ],
        "source_name": "seed"
    }
}


STRUCTURED_FUNDAMENTAL_ADDITIONS = {
    "000858.SZ": {
        "symbol": "000858.SZ",
        "report_date": "2025-12-31",
        "revenue": 108500000000,
        "net_profit": 37800000000,
        "roe": 29.4,
        "gross_margin": 76.1,
        "pe_ttm": 20.9,
        "pb": 5.4,
        "source_name": "seed"
    },
    "601318.SH": {
        "symbol": "601318.SH",
        "report_date": "2025-12-31",
        "revenue": 1218000000000,
        "net_profit": 121000000000,
        "roe": 15.2,
        "gross_margin": None,
        "pe_ttm": 8.7,
        "pb": 1.1,
        "source_name": "seed"
    }
}


STRUCTURED_INDUSTRY_ADDITIONS = {
    "保险": {
        "industry_name": "保险",
        "trade_date": "2026-04-22",
        "pct_change": 0.68,
        "pe": 11.8,
        "pb": 1.45,
        "turnover": 0.43,
        "source_name": "seed"
    },
    "宽基指数": {
        "industry_name": "宽基指数",
        "trade_date": "2026-04-22",
        "pct_change": 0.42,
        "pe": None,
        "pb": None,
        "turnover": 0.0,
        "source_name": "seed"
    },
    "成长指数": {
        "industry_name": "成长指数",
        "trade_date": "2026-04-22",
        "pct_change": 0.86,
        "pe": None,
        "pb": None,
        "turnover": 0.0,
        "source_name": "seed"
    },
    "券商": {
        "industry_name": "券商",
        "trade_date": "2026-04-22",
        "pct_change": 0.59,
        "pe": 18.2,
        "pb": 1.62,
        "turnover": 0.77,
        "source_name": "seed"
    }
}


STRUCTURED_MACRO_ADDITIONS = {
    "CPI_CN": {"indicator_code": "CPI_CN", "metric_date": "2026-03-31", "metric_value": 0.8, "unit": "%", "source_name": "seed"},
    "PMI_CN": {"indicator_code": "PMI_CN", "metric_date": "2026-03-31", "metric_value": 50.6, "unit": "", "source_name": "seed"},
    "M2_CN": {"indicator_code": "M2_CN", "metric_date": "2026-03-31", "metric_value": 8.1, "unit": "%", "source_name": "seed"},
    "UST10Y_CN_PROXY": {"indicator_code": "CN10Y", "metric_date": "2026-03-31", "metric_value": 2.31, "unit": "%", "source_name": "seed"}
}


ENTITY_TO_INDUSTRY_ADDITIONS = {
    "沪深300ETF": "宽基指数",
    "沪深300": "宽基指数",
    "中国平安": "保险",
    "创业板ETF": "成长指数",
    "证券ETF": "券商"
}


QUERY_LABEL_ROWS = [
    ["茅台今天为什么跌？后面还值得拿吗？", "stock", "market_explanation|hold_judgment", "price|news|risk", "train", "news|announcement|research_note", "market_api|industry_sql|fundamental_sql"],
    ["五粮液现在能买吗？", "stock", "buy_sell_timing", "price|risk", "train", "news|research_note", "market_api|fundamental_sql|industry_sql"],
    ["沪深300ETF适合定投吗？", "etf", "product_info", "product_mechanism", "train", "faq|product_doc|research_note", "market_api"],
    ["这个基金费率高吗？", "fund", "trading_rule_fee", "product_mechanism", "train", "faq|product_doc", ""],
    ["沪深300指数近期表现怎么样？", "index", "price_query", "price", "train", "product_doc", "market_api"],
    ["宏观政策会影响白酒吗？", "macro", "macro_policy_impact", "macro|policy|industry", "train", "news|research_note", "macro_sql|industry_sql"],
    ["中国平安估值高不高？", "stock", "valuation_analysis", "valuation|risk", "train", "research_note|news", "market_api|fundamental_sql|industry_sql"],
    ["中国平安基本面怎么样？", "stock", "fundamental_analysis", "fundamentals|risk", "train", "research_note", "fundamental_sql|industry_sql"],
    ["证券ETF最近能买吗？", "etf", "buy_sell_timing", "price|risk", "train", "product_doc|research_note", "market_api|industry_sql"],
    ["创业板ETF波动大吗？", "etf", "risk_analysis", "risk|price", "train", "product_doc|faq", "market_api"],
    ["场内ETF和场外基金有什么区别？", "etf", "product_info", "product_mechanism|comparison", "train", "faq", ""],
    ["公募基金费率包含哪些部分？", "fund", "trading_rule_fee", "product_mechanism", "train", "faq", ""],
    ["指数基金跟踪误差怎么看？", "index", "product_info", "product_mechanism|risk", "train", "faq", ""],
    ["白酒行业今天为什么跌？", "stock", "market_explanation", "industry|price|news", "train", "news|research_note", "industry_sql|market_api"],
    ["最近有哪些茅台新闻？", "stock", "event_news_query", "news", "train", "news|announcement", ""],
    ["最近有哪些五粮液公告？", "stock", "event_news_query", "news", "train", "announcement", ""],
    ["沪深300ETF和创业板ETF哪个好？", "etf", "peer_compare", "comparison|product_mechanism", "train", "product_doc|faq|research_note", "market_api"],
    ["中国平安和茅台哪个更稳？", "stock", "peer_compare|risk_analysis", "comparison|risk", "train", "research_note|news", "market_api|fundamental_sql"],
    ["大盘今天怎么样？", "generic_market", "price_query", "price", "valid", "news", "market_api"],
    ["白酒行业景气度怎么样？", "stock", "fundamental_analysis", "industry|fundamentals", "valid", "research_note|news", "industry_sql"],
    ["CPI上行会影响消费股吗？", "macro", "macro_policy_impact", "macro|industry", "valid", "news|research_note", "macro_sql|industry_sql"],
    ["沪深300ETF产品有什么特点？", "etf", "product_info", "product_mechanism", "valid", "product_doc|faq", ""],
    ["证券ETF适合长期拿吗？", "etf", "hold_judgment", "risk|price", "test", "product_doc|research_note", "market_api|industry_sql"],
    ["创业板ETF最近涨跌怎么样？", "etf", "price_query", "price", "test", "product_doc", "market_api"],
    ["中国平安最近为什么涨？", "stock", "market_explanation", "price|news", "test", "news|research_note", "market_api|industry_sql"],
    ["中国平安还值得持有吗？", "stock", "hold_judgment", "risk|fundamentals", "test", "research_note|news", "market_api|fundamental_sql"],
    ["宏观宽松会利好券商吗？", "macro", "macro_policy_impact", "macro|industry|policy", "test", "news|research_note", "macro_sql|industry_sql"],
    ["ETF定投适合什么人？", "etf", "product_info", "product_mechanism", "test", "faq|research_note", ""]
]


RETRIEVAL_QRELS_ROWS = [
    ["茅台今天为什么跌？后面还值得拿吗？", "aknews_600519.SH_1", 2],
    ["茅台今天为什么跌？后面还值得拿吗？", "aknews_600519.SH_2", 2],
    ["茅台今天为什么跌？后面还值得拿吗？", "research_note_001", 2],
    ["茅台今天为什么跌？后面还值得拿吗？", "announcement_001", 1],
    ["五粮液现在能买吗？", "aknews_000858.SZ_3", 2],
    ["五粮液现在能买吗？", "research_note_001", 1],
    ["沪深300ETF适合定投吗？", "product_doc_001", 2],
    ["沪深300ETF适合定投吗？", "faq_002", 2],
    ["沪深300ETF适合定投吗？", "research_note_003", 1],
    ["这个基金费率高吗？", "faq_004", 2],
    ["这个基金费率高吗？", "faq_003", 1],
    ["沪深300指数近期表现怎么样？", "product_doc_001", 1],
    ["宏观政策会影响白酒吗？", "news_macro_001", 2],
    ["宏观政策会影响白酒吗？", "research_note_001", 1],
    ["中国平安估值高不高？", "research_note_002", 2],
    ["中国平安估值高不高？", "news_601318_001", 1],
    ["中国平安基本面怎么样？", "announcement_601318_001", 2],
    ["中国平安基本面怎么样？", "research_note_002", 1],
    ["证券ETF最近能买吗？", "research_note_004", 2],
    ["证券ETF最近能买吗？", "product_doc_003", 2],
    ["创业板ETF波动大吗？", "product_doc_002", 2],
    ["创业板ETF波动大吗？", "faq_002", 1],
    ["场内ETF和场外基金有什么区别？", "faq_003", 2],
    ["公募基金费率包含哪些部分？", "faq_004", 2],
    ["指数基金跟踪误差怎么看？", "faq_005", 2],
    ["白酒行业今天为什么跌？", "research_note_001", 2],
    ["白酒行业今天为什么跌？", "aknews_000858.SZ_3", 1],
    ["最近有哪些茅台新闻？", "aknews_600519.SH_1", 2],
    ["最近有哪些茅台新闻？", "aknews_600519.SH_2", 2],
    ["最近有哪些五粮液公告？", "aknews_000858.SZ_1", 1],
    ["沪深300ETF和创业板ETF哪个好？", "product_doc_001", 2],
    ["沪深300ETF和创业板ETF哪个好？", "product_doc_002", 2],
    ["中国平安和茅台哪个更稳？", "research_note_002", 2],
    ["中国平安和茅台哪个更稳？", "research_note_001", 2],
    ["大盘今天怎么样？", "news_macro_001", 1],
    ["白酒行业景气度怎么样？", "research_note_001", 2],
    ["CPI上行会影响消费股吗？", "news_macro_001", 2],
    ["沪深300ETF产品有什么特点？", "product_doc_001", 2],
    ["证券ETF适合长期拿吗？", "research_note_004", 2],
    ["证券ETF适合长期拿吗？", "product_doc_003", 2],
    ["创业板ETF最近涨跌怎么样？", "product_doc_002", 1],
    ["中国平安最近为什么涨？", "news_601318_001", 2],
    ["中国平安最近为什么涨？", "research_note_002", 1],
    ["中国平安还值得持有吗？", "research_note_002", 2],
    ["中国平安还值得持有吗？", "announcement_601318_001", 2],
    ["中国平安还值得持有吗？", "news_601318_001", 1],
    ["宏观宽松会利好券商吗？", "news_macro_001", 2],
    ["宏观宽松会利好券商吗？", "research_note_004", 1],
    ["ETF定投适合什么人？", "faq_002", 2],
    ["ETF定投适合什么人？", "research_note_003", 2]
]


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_docs(existing: list[dict], additions: list[dict]) -> list[dict]:
    docs = {item["evidence_id"]: item for item in existing}
    for item in additions:
        docs[item["evidence_id"]] = item
    return list(docs.values())


def infer_question_style(query: str, intents: str) -> str:
    labels = intents.split("|") if intents else []
    if "peer_compare" in labels or any(term in query for term in ["哪个", "更稳", "更好", "区别"]):
        return "compare"
    if "market_explanation" in labels or "为什么" in query:
        return "why"
    if "hold_judgment" in labels or "buy_sell_timing" in labels or any(term in query for term in ["值得", "能买吗", "适合"]):
        return "advice"
    if any(term in query for term in ["未来", "后面", "接下来"]):
        return "forecast"
    return "fact"


def infer_sentiment(query: str, intents: str) -> str:
    if any(term in query for term in ["跌", "回撤", "波动大", "风险", "为什么跌"]):
        return "anxious"
    if any(term in query for term in ["能买吗", "能买", "值得拿", "值得持有"]):
        return "bullish"
    if any(term in query for term in ["卖", "止盈", "止损"]):
        return "bearish"
    return "neutral"


def main() -> None:
    entity_path = DATA_DIR / "entity_master.csv"
    alias_path = DATA_DIR / "alias_table.csv"
    docs_path = DATA_DIR / "documents.json"
    structured_path = DATA_DIR / "structured_data.json"
    labels_path = DATA_DIR / "query_labels.csv"
    qrels_path = DATA_DIR / "retrieval_qrels.csv"

    with entity_path.open("r", encoding="utf-8") as handle:
        entity_rows = list(csv.DictReader(handle))
    with alias_path.open("r", encoding="utf-8") as handle:
        alias_rows = list(csv.DictReader(handle))
    documents = json.loads(docs_path.read_text(encoding="utf-8"))
    structured = json.loads(structured_path.read_text(encoding="utf-8"))

    entity_by_symbol = {row["symbol"]: row for row in entity_rows}
    for row in ENTITY_ADDITIONS:
        entity_by_symbol[row["symbol"]] = row
    entity_rows = sorted(entity_by_symbol.values(), key=lambda row: int(row["entity_id"]))

    alias_by_id = {row["alias_id"]: row for row in alias_rows}
    for row in ALIAS_ADDITIONS:
        alias_by_id[row["alias_id"]] = row
    alias_rows = sorted(alias_by_id.values(), key=lambda row: int(row["alias_id"]))

    documents = merge_docs(documents, DOCUMENT_ADDITIONS)
    documents.sort(key=lambda item: (item["source_type"], item["evidence_id"]))

    structured.setdefault("market_api", {}).update(STRUCTURED_MARKET_ADDITIONS)
    structured.setdefault("fundamental_sql", {}).update(STRUCTURED_FUNDAMENTAL_ADDITIONS)
    structured.setdefault("industry_sql", {}).update(STRUCTURED_INDUSTRY_ADDITIONS)
    structured.setdefault("macro_sql", {}).update(STRUCTURED_MACRO_ADDITIONS)
    structured.setdefault("entity_to_industry", {}).update(ENTITY_TO_INDUSTRY_ADDITIONS)

    write_csv(entity_path, entity_rows)
    write_csv(alias_path, alias_rows)
    docs_path.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    structured_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")

    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query",
                "product_type",
                "intent_labels",
                "topic_labels",
                "question_style",
                "sentiment_label",
                "split",
                "expected_document_sources",
                "expected_structured_sources",
            ]
        )
        for row in QUERY_LABEL_ROWS:
            query, product_type, intent_labels, topic_labels, split, expected_document_sources, expected_structured_sources = row
            writer.writerow(
                [
                    query,
                    product_type,
                    intent_labels,
                    topic_labels,
                    infer_question_style(query, intent_labels),
                    infer_sentiment(query, intent_labels),
                    split,
                    expected_document_sources,
                    expected_structured_sources,
                ]
            )

    with qrels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "evidence_id", "relevance"])
        writer.writerows(RETRIEVAL_QRELS_ROWS)

    print("expanded data assets written")


if __name__ == "__main__":
    main()
