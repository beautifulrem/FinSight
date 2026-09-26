"""Catalog of upstream data sources.

Every live call is attributed to one *source*: an upstream endpoint family that fails together
(for example all Eastmoney ``push2``/``push2his`` quote hosts are throttled as a unit). Health tracking
and circuit breaking are keyed by these ids, and provenance records carry the human-readable label.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    label: str
    upstream: str
    kinds: tuple[str, ...]


_SOURCES = (
    SourceInfo(
        "eastmoney.quote", "东方财富行情", "push2his.eastmoney.com / push2.eastmoney.com", ("market", "industry")
    ),
    SourceInfo("eastmoney.datacenter", "东方财富数据中心", "datacenter-web.eastmoney.com", ("valuation", "macro")),
    SourceInfo("eastmoney.fund", "天天基金", "fund.eastmoney.com / fundf10.eastmoney.com", ("fund",)),
    SourceInfo("eastmoney.news", "东方财富资讯", "search-api-web.eastmoney.com", ("news",)),
    SourceInfo("eastmoney.announcement", "东方财富公告", "np-anotice-stock.eastmoney.com", ("announcement",)),
    SourceInfo("sina.kline", "新浪财经行情", "finance.sina.com.cn", ("market",)),
    SourceInfo("sina.quote", "新浪实时行情", "hq.sinajs.cn", ("market",)),
    SourceInfo("sina.finance", "新浪财经财务指标", "money.finance.sina.com.cn", ("fundamentals",)),
    SourceInfo("tencent.kline", "腾讯证券行情", "web.ifzq.gtimg.cn", ("market",)),
    SourceInfo("tencent.quote", "腾讯实时行情", "qt.gtimg.cn", ("valuation",)),
    SourceInfo("ths.finance", "同花顺财务摘要", "basic.10jqka.com.cn", ("fundamentals",)),
    SourceInfo("ths.industry", "同花顺行业指数", "d.10jqka.com.cn / q.10jqka.com.cn", ("industry",)),
    SourceInfo("csindex", "中证指数", "csindex.com.cn", ("index_valuation",)),
    SourceInfo("chinabond", "中债收益率曲线", "yield.chinabond.com.cn", ("macro",)),
    SourceInfo("cninfo.announcement", "巨潮资讯公告", "www.cninfo.com.cn", ("announcement",)),
    SourceInfo("cninfo.profile", "巨潮资讯公司概况", "www.cninfo.com.cn", ("industry",)),
    SourceInfo("xueqiu", "雪球", "stock.xueqiu.com", ("fund",)),
    SourceInfo("efinance", "efinance（东方财富K线）", "push2his.eastmoney.com", ("market",)),
    SourceInfo("tushare", "Tushare Pro", "api.tushare.pro", ("market", "fundamentals", "news")),
)

CATALOG: dict[str, SourceInfo] = {info.source_id: info for info in _SOURCES}

# Provider endpoint name (as used in ``provider_endpoint`` traces) -> source id.
ENDPOINT_SOURCES: dict[str, str] = {
    "akshare.stock_zh_a_hist": "eastmoney.quote",
    "akshare.fund_etf_hist_em": "eastmoney.quote",
    "akshare.index_zh_a_hist": "eastmoney.quote",
    "akshare.stock_zh_index_spot_em": "eastmoney.quote",
    "akshare.stock_individual_info_em": "eastmoney.quote",
    "akshare.stock_board_industry_hist_em": "eastmoney.quote",
    "akshare.fund_etf_spot_em": "eastmoney.quote",
    "akshare.stock_value_em": "eastmoney.datacenter",
    "akshare.macro_china_cpi": "eastmoney.datacenter",
    "akshare.macro_china_pmi": "eastmoney.datacenter",
    "akshare.macro_china_money_supply": "eastmoney.datacenter",
    "akshare.macro_china_lpr": "eastmoney.datacenter",
    "akshare.bond_zh_us_rate": "eastmoney.datacenter",
    "akshare.fund_open_fund_info_em": "eastmoney.fund",
    "akshare.fund_overview_em": "eastmoney.fund",
    "akshare.fund_etf_fund_info_em": "eastmoney.fund",
    "akshare.stock_news_em": "eastmoney.news",
    "eastmoney.notice_api": "eastmoney.announcement",
    "akshare.stock_zh_a_daily": "sina.kline",
    "akshare.fund_etf_hist_sina": "sina.kline",
    "akshare.stock_zh_index_daily": "sina.kline",
    "sina.hq_sinajs_cn": "sina.quote",
    "akshare.stock_financial_analysis_indicator": "sina.finance",
    "tencent.fqkline": "tencent.kline",
    "tencent.qt_quote": "tencent.quote",
    "akshare.stock_financial_abstract_ths": "ths.finance",
    "akshare.stock_board_industry_index_ths": "ths.industry",
    "akshare.stock_zh_index_value_csindex": "csindex",
    "akshare.bond_china_yield": "chinabond",
    "cninfo.his_announcement": "cninfo.announcement",
    "akshare.stock_profile_cninfo": "cninfo.profile",
    "akshare.fund_individual_detail_info_xq": "xueqiu",
    "efinance.stock.get_quote_history": "efinance",
    "efinance.fund.get_quote_history": "efinance",
}


def source_for_endpoint(endpoint: str) -> str:
    """Map a provider endpoint to its source id; unknown endpoints get their own id."""
    return ENDPOINT_SOURCES.get(endpoint, endpoint)


def source_label(source_id: str | None) -> str | None:
    if not source_id:
        return None
    info = CATALOG.get(source_id)
    return info.label if info else source_id
