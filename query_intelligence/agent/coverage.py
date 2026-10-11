"""What FinSight covers, and what a question asks for that the evidence does not have.

* ``out_of_coverage``: crypto assets and foreign (US / Hong Kong) equities are finance questions, but
  FinSight's data covers China A-shares, funds/ETFs, indices and China macro only. Such questions get a
  clear "not covered" answer instead of a "which stock?" clarification (which could never succeed).
* ``coverage_gaps``: the deterministic composer restates whatever the tools returned. When the question
  asks for a metric the tools did not return (dividend yield, debt ratio, growth rate) or for a period
  the data does not cover ("2019年营收" with 2025 statements), the gap is stated explicitly instead of
  silently answering with other numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .frame import GENERIC_NAME_TAILS, NET_MARGIN_SHARE_SOURCE, TURNOVER

# --------------------------------------------------------------------------- out of coverage

# Crypto assets by name, ticker or shape. (round 9, E7) Tokens are also named by their ticker next to a fund word
# ("BTC ETF", "ETH现货ETF"), by their project name ("Solana ETF", "币安币"), or as "<X>币" + a fund word ("某某币ETF");
# 人民币 / 港币 / 美元 and money-market funds (货币ETF, 货币基金) are not crypto.
_CRYPTO_TICKERS = r"BTC|ETH|USDT|USDC|DOGE|XRP|SOL|BNB|ADA|DOT|TRX|LTC|SHIB|AVAX|TON|LINK"
_NOT_CRYPTO_BI = "".join(
    f"(?<!{prefix})" for prefix in ("人民", "港", "美", "日", "外", "货", "硬", "纸", "钱", "欧", "英")
)
_CRYPTO = re.compile(
    r"比特币|以太坊|以太币|以太(?=\s*(?:ETF|ETP|现货|期货|基金))|狗狗币|莱特币|瑞波币?|泰达币|币安币?|柴犬币|波场币?|"
    r"艾达币|波卡币|索拉纳|加密货币|加密资产|数字货币|虚拟货币|币圈|山寨币|稳定币|"
    rf"(?<![A-Za-z])(?:{_CRYPTO_TICKERS})(?=\s*(?:ETF|ETP|现货|期货|基金|\bfunds?\b|\btrusts?\b))|"
    r"(?<![A-Za-z])(?:BTC|ETH|USDT|USDC|DOGE|XRP|SOL|BNB|SHIB)(?![A-Za-z])|"
    rf"{_NOT_CRYPTO_BI}币\s*(?=ETF|ETP|现货|期货|基金)|"
    r"\bbitcoin\b|\bethereum\b|\bether\b|\bsolana\b|\bcardano\b|\bpolkadot\b|\bbinance\b|\bripple\b|\btether\b|"
    r"\bcrypto(?:currenc(?:y|ies))?\b|\b(?:doge|lite|stable)coins?\b|\bstablecoins?\b|\btokens?\s+(?:etf|fund)s?\b",
    re.IGNORECASE,
)
# (round 10, F6) Chinese companies listed only in Hong Kong or the US, by the name users write. Several contain or
# resemble an A-share name (平安好医生 / 平安健康 ~ 中国平安, 药明生物 ~ 药明康德, 京东健康 ~ 京东方), so the whole name
# is matched first and the A-share lookalike inside it is not a target (``agent/graph.py`` ``_named_targets``).
# Dual A+H listings (比亚迪, 药明康德, 中芯国际) are A-shares and are not listed here; media and product words
# (网易财经, 百度一下, 腾讯新闻) are not companies asked about.
_HK_US_LISTED = (
    r"平安好医生|平安健康(?:医疗)?|腾讯音乐|腾讯(?!新闻|财经|网|视频|会议|文档|云)|阿里健康|阿里影业|京东健康|京东物流|"
    r"京东集团|京东(?!方)|小米(?:集团|公司)|网易(?!财经|新闻|号|云)|百度(?!一下|搜索|指数|百科|地图|贴吧)|拼多多|快手|"
    r"哔哩哔哩|蔚来(?:汽车)?|理想汽车|小鹏汽车|零跑汽车|携程|贝壳找房|农夫山泉|海底捞|泡泡玛特|蒙牛(?:乳业)?|华润啤酒|"
    r"药明生物|安踏(?:体育)?|李宁公司|中国飞鹤|百胜中国|名创优品|知乎|微博|爱奇艺|金山软件|联想集团|"
    # (round 11, G6) Hong Kong listed subsidiaries and lookalikes of A-share names (比亚迪电子 ~ 比亚迪, 华润置地 ~
    # 华润微, 中国海外发展 ~ 中国海油) and Hong Kong-only blue chips
    r"比亚迪电子|吉利汽车|舜宇光学(?:科技)?|华润置地|华润电力|华润啤酒|华润万象生活|中国海外发展|中信股份|"
    r"石药集团|中国生物制药|碧桂园(?:服务)?|融创中国|长江和记|汇丰控股|友邦保险|香港交易所|港交所|银河娱乐|"
    r"中国旺旺|康师傅|周大福|申洲国际|恒基地产|新鸿基地产|港铁公司"
)
# (round 11, G6) An H share or a Hong Kong ticker: "中国平安H股", "Ping An H shares", "2318.HK", "HK0285". The
# company may also be listed in Shanghai or Shenzhen, but the question asks for the Hong Kong line, which FinSight
# has no data for: the A-share target inside the span is a lookalike, never answered with A-share data.
_H_SHARE = (
    # (round 12, H5) "中芯国际港股": a company name followed by 港股 is its Hong Kong line too
    r"(?:(?![和与跟及或比对同的、，,])[一-鿿A-Za-z]){2,8}?\s*(?:的\s*)?(?:H\s*股(?!指数|ETF|基金|LOF|\(QDII)|港股(?!通))|"
    r"H\s*股(?!东|指数|ETF|基金|LOF|\(QDII)|港股通|"
    r"(?<![0-9A-Za-z.])\d{4,5}\s*\.\s*HK(?![A-Za-z])|(?<![A-Za-z])HK\s?\d{4,5}(?!\d)|"
    r"(?<![0-9A-Za-z.])\d{4,5}\s+HK(?![A-Za-z])|"
    r"\b(?:[a-z][\w&.'-]*\s+){1,3}H[- ]?shares?\b|\bH[- ]shares?\b|\bHKEX\b|"
    # (round 12, H5) "Ping An's Hong Kong listed shares", "the Hong Kong share price of China Merchants Bank"
    r"\b(?:[a-z][\w&.'-]*\s+){0,3}?hong kong[- ]listed\b|"
    r"\b(?:[a-z][\w&.'-]*\s+){0,3}?hong kong (?:share|stock) prices?\b(?: of (?:[a-z][\w&.'-]*\s?){1,4})?|"
    r"\bhong kong (?:shares?|stock|listing|line|counter) of (?:[a-z][\w&.'-]*\s?){1,4}|"
    # (round 13, review-9 I9) "CATL's Hong Kong shares", "Ping An’s Hong Kong stock": a possessive before it
    r"\b[\w&.-]+(?:'s|’s|s')\s+(?:[a-z][\w-]*\s+){0,2}?hong kong (?:shares?|stocks?|line|counter|quote)\b|"
    # (round 12) the Hong Kong line described in words: "它在港交所挂牌的那部分股票", "平安在香港上市的股份",
    # "its shares listed in Hong Kong", "Ping An's Hong Kong listing". As with "…H股", the name before it is part of
    # the span, so the A-share target it names is a lookalike, not answered with A-share data.
    r"(?:(?![和与跟及或比对同的、，,])[一-鿿A-Za-z]){0,8}?(?:在|于)?(?:香港|港交所|联交所|香港交易所)"
    r"(?:上市|挂牌|发行|交易)|"
    r"\b(?:[a-z][\w&.'-]*\s+){0,3}?(?:hong kong (?:listing|line|counter)|"
    r"shares? (?:listed|traded|quoted) (?:in|on(?: the)?) hong kong)\b|"
    r"\b(?:listed|traded|quoted) (?:in|on(?: the)?) hong kong\b"
)
# (round 14, J7) A company's own line outside the A-share market, by any wording: its Hong Kong shares described
# ("中芯国际香港那边", "在港交所交易的那部分", "BYD's HK-listed stock"), its London / Swiss GDRs ("长江电力伦敦GDR",
# "its London GDRs", 全球存托凭证, 沪伦通), its US ADRs ("ICBC ADR", "the NYSE-listed ADR", 美国存托凭证), and the A/H
# premium ("招商银行A股相对H股溢价", "A+H两地", "AH溢价", "A/H premium"), which needs the H-share price. Before round 14
# these were answered with the A-share price, and any "A股" in the question disabled the foreign check
# (``_A_SHARE_ANCHOR``); these spans are checked first, and the name before them is part of the span, so the A-share
# target it names is a lookalike (refused with the note that its A shares are covered). A concept question ("A股和H股
# 有什么区别", "What is an ADR?") names no line and stays in scope.
_ZH_NAME = r"(?:(?![和与跟及或比对同的、，,])[一-鿿A-Za-z]){0,8}?\s*(?:的\s*)?"
_EN_NAME = r"(?:\b[\w&.-]+(?:'s|’s|s')\s+|\b(?:[a-z][\w&.'-]*\s+){1,3}?)?(?:(?:the|its|their)\s+)?"
_HK_LINE = (
    _ZH_NAME + r"(?:在|于)?(?:香港|港股|港交所|联交所|香港交易所|H\s*股)\s*(?:市场)?(?:那边|那头|那里|那块|这边|方面|"
    r"(?:上市|挂牌|交易|发行)?的?(?:那部分|那一部分|部分|那只|那个))|"
    # a named company's H shares ("中国平安的H股价格", "Ping An's H shares"), with "A股" elsewhere in the question too
    r"(?:(?![和与跟及或比对同的、，,])[一-鿿A-Za-z]){2,8}?\s*(?:的\s*)?H\s*股(?!东|指数|ETF|基金|LOF|\(QDII)|"
    r"\b(?:[\w&.-]+(?:'s|’s|s')|(?:[a-z][\w&.'-]*\s+){1,3}?)\s*H[- ]shares?\b|"
    + _EN_NAME
    + r"(?:HK|hong\s+kong)[- ](?:listed|traded|quoted)\b|"
    + _EN_NAME
    + r"(?:listed|traded|quoted)\s+(?:in|on)\s+(?:the\s+)?(?:HK|hong\s+kong)(?:\s+(?:market|exchange|bourse))?\b|"
    + _EN_NAME
    + r"\bHK\s+(?:shares?|stock|line|listing|counter|price|quote|market\s+price)\b"
)
_GDR_LINE = (
    _ZH_NAME + r"(?:(?:伦敦|英国|瑞士|苏黎世|法兰克福|德国)(?:证券)?(?:交易所)?(?:上市|挂牌|交易|发行)?的?\s*"
    r"(?:GDR|全球存托凭证|存托凭证|那边|那部分|股份|股票|部分)|GDR|全球存托凭证|沪伦通(?:GDR|存托凭证)?|中欧通)|"
    + _EN_NAME
    + r"(?:(?:London|Swiss|Zurich|SIX|LSE)[- ](?:listed|traded|quoted)\s+)?(?:GDRs?|global\s+depositary\s+"
    r"(?:receipts?|shares?))\b|"
    + _EN_NAME
    + r"(?:London|Swiss|Zurich|SIX|LSE)[- ](?:listed|traded|quoted)\b|"
    + _EN_NAME
    + r"(?:shares?\s+)?(?:listed|traded|quoted)\s+(?:in|on)\s+(?:the\s+)?(?:London|Zurich|Switzerland|SIX|LSE)\b"
)
_ADR_LINE = (
    _ZH_NAME + r"(?:(?:美国|美股|纽约|纽交所|纳斯达克)(?:证券)?(?:交易所)?(?:上市|挂牌|交易|发行)?的?\s*"
    r"(?:ADR|存托凭证|存托股|那边|那部分|股份|股票|部分)|ADR|美国存托(?:凭证|股票|股份)?)|"
    + _EN_NAME
    + r"(?:(?:NYSE|Nasdaq|US|U\.S\.|New\s+York)[- ](?:listed|traded|quoted)\s+)?(?:ADRs?|ADSs?|American\s+"
    r"depositary\s+(?:receipts?|shares?))\b|"
    + _EN_NAME
    + r"(?:NYSE|Nasdaq|US|U\.S\.)[- ](?:listed|traded|quoted)\s+(?:shares?|stock|line)\b"
)
_AH_PREMIUM = (
    _ZH_NAME + r"(?:A\s*股?\s*(?:[+/／&]|和|与|跟|及|加)\s*H\s*股?(?:两地|股价|价差|溢价|折价|比价)?|"
    r"H\s*股?\s*(?:[+/／&]|和|与|跟|及|加)\s*A\s*股?(?:两地|股价|价差|溢价|折价|比价)?|"
    r"A\s*H\s*股?\s*(?:溢价|折价|比价|价差)|"
    r"A\s*股?(?:相对|相较|对比|比)(?:于)?\s*H\s*股[^，。？?]{0,8}?(?:溢价|折价|贵|便宜|高|低|差|价差)|"
    r"H\s*股(?:相对|相较|对比|比)(?:于)?\s*A\s*股[^，。？?]{0,8}?(?:溢价|折价|贵|便宜|高|低|差|价差))|"
    + _EN_NAME
    + r"(?:A[/-]H|AH)\s+(?:share\s+)?(?:premium|discount|spread|gap|price\s+gap)\b"
    r"(?:\s+(?:on|of|for|in)\s+(?:[\w&.'-]+(?:\s+|(?=[?.!,]|$))){1,3})?|"
    r"\bH[- ]shares?\s+(?:premium|discount)\b|\bpremium\s+(?:of\s+)?(?:its\s+|the\s+)?A[- ]shares?\s+"
    r"(?:over|to|versus|vs\.?)\s+(?:its\s+|the\s+)?H[- ]shares?\b"
)
# (round 15, review-11 K5) The Hong Kong line asked by its stock code, its venue or its currency. Before round 15
# "比亚迪股份(01211)现在多少钱", "中芯国际0981.HK股价" (``\w`` in the ticker's look-behind matched the CJK name, so the
# ticker was missed), "中芯国际在港交所的报价" and "What does China Merchants Bank trade at in Hong Kong dollars?" were
# answered with the A-share price. As with the other line patterns, the name before the code, venue or currency is
# part of the span, so the A-share target it names is a lookalike. An A-share code has six digits and is never one of
# these; a bare four-digit code without a leading zero is a Hong Kong code only right after the company it belongs to
# (``_AH_CODES``, checked in ``_ah_code_spans``), so "茅台1880附近" or "比亚迪2025年报" are not codes.
_HK_CODE_TAIL = (
    r"(?![\d.%％])(?!\s*(?:年|月|日|号|股(?![价票份])|手|元|块|万|亿|个|只|倍|点|%|％|基点|附近|左右|以上|以下|上下|关口|一线|能|可以|"
    r"\b(?:yuan|RMB|CNY|shares|lots|points|times|bp)\b))"
)
_NOT_YEAR_CODE = r"(?!(?:19|20)\d{2}\s*[)）])"
_HK_CODE_LINE = (
    # "中芯国际0981.HK", "比亚迪 (1211.HK)": a ticker right after Chinese text
    _ZH_NAME
    + r"[(（]?\s*(?<![0-9.])\d{4,5}\s*\.\s*HK(?![A-Za-z])\s*[)）]?|"
    # a bracketed code: "比亚迪股份(01211)", "招商银行（3968）", "Ping An (2318)"; "(2025)" is a year
    + _ZH_NAME
    + r"[(（]\s*(?:0\d{3,4}|"
    + _NOT_YEAR_CODE
    + r"[1-9]\d{3})\s*[)）]|"
    + _EN_NAME
    + r"\(\s*(?:0\d{3,4}|"
    + _NOT_YEAR_CODE
    + r"[1-9]\d{3})(?:\s*\.\s*HK)?\s*\)|"
    # a code with its leading zero: "中国平安02318", "0981的股价"
    + _ZH_NAME
    + r"(?<![0-9A-Za-z.])0\d{3,4}(?![A-Za-z])"
    + _HK_CODE_TAIL
)
_HK_VENUE = r"(?:港交所|联交所|香港交易所|香港联交所|香港联合交易所)"
# a name of two to eight characters that is not a conjunction, a comparison or a currency exchange ("换", "兑")
_ZH_NAME2 = r"(?:(?![和与跟及或比对同的、，,换兑])[一-鿿A-Za-z]){2,8}?"
_HK_QUOTE_WORDS = (
    r"(?:报价|价格|股价|收盘价?|行情|走势|市值|市盈率|市净率|估值|成交(?:量|额)?|涨跌幅?|涨幅|跌幅|表现|现价|最新价)"
)
_HKD = r"(?:港元|港币|港幣|HKD|HK\$|HK\s+dollars?|hong\s+kong\s+dollars?)"
_EN_PRICE_VERB = (
    r"(?:trades?|trading|traded|priced?|prices|quoted?|quotes|closes?|closed|closing|worth|valued?|value|stands?)"
)
_HK_VENUE_CURRENCY_LINE = (
    # the venue, then the company: "港交所的平安", "联交所上的招商银行收盘多少"
    _HK_VENUE
    + r"(?:上|里)?(?:上市|挂牌|交易)?的\s*"
    + _ZH_NAME2.replace("比", "")  # the name follows the venue: "联交所的比亚迪"
    + r"(?=$|[\s，,。？?！!]|的|"
    + _HK_QUOTE_WORDS
    + r"|现在|今天|最新|多少|怎么样|如何|涨|跌)|"
    # the company, then the venue and a quote: "中芯国际在港交所的报价", "比亚迪在香港的股价"
    + _ZH_NAME2
    + r"\s*(?:在|于)?(?:"
    + _HK_VENUE
    + r"|香港(?:市场)?)(?:上|里|那边)?的?\s*"
    + _HK_QUOTE_WORDS
    + r"|"
    # the currency: "招商银行港元价格", "中国平安以港币计价的股价", "中国平安现在多少港元"
    + _ZH_NAME2
    + r"\s*(?:的)?\s*(?:以|用|按)?\s*"
    + _HKD
    + r"\s*(?:计价|计)?的?\s*"
    + _HK_QUOTE_WORDS
    + r"|"
    + _ZH_NAME2
    + r"(?:(?![换兑汇])[^，。？?！!；;]){0,8}?多少(?:钱)?\s*"
    + _HKD
    + r"(?!\s*(?:汇率|兑|换))|"
    # "What does China Merchants Bank trade at in Hong Kong dollars?", "Ping An's HKD price"
    r"\b(?:[a-z][\w&.'-]*\s+){1,6}?"
    + _EN_PRICE_VERB
    + r"\b(?:(?!\b(?:convert|conversion|exchange\s+rate)\b)[^.?!]){0,30}?\b(?:in|into)\s+"
    + _HKD
    + r"|"
    r"\b(?:[\w&.-]+(?:'s|’s|s')\s+|(?:[a-z][\w&.'-]*\s+){1,6}?)(?:HKD|HK\$|hong\s+kong\s+dollar)\s+"
    r"(?:share\s+)?(?:price|quote|close|closing\s+price|value|market\s+value)\b"
)
_HK_LINE = _HK_LINE + "|" + _HK_CODE_LINE + "|" + _HK_VENUE_CURRENCY_LINE


# (round 15, K5) Hong Kong codes of A+H companies FinSight covers on the A-share side, with the names a question uses
# for them: a bare four-digit code is the Hong Kong line only right after one of these names ("中国平安2318",
# "招行 3968"). Codes with a leading zero ("0981", "02318") are matched by shape in ``_HK_CODE_LINE``.
# (round 16, review-12 L7) The codes and names come from the dual-listing table (``data/runtime/dual_listings.csv``,
# ``data_loader.load_dual_listings``) instead of a list in the code.
def _dual_listing_rows() -> list[dict[str, str]]:
    from ..data_loader import load_dual_listings

    return [row for row in load_dual_listings() if row.get("a_symbol") and str(row.get("h_code") or "").isdigit()]


def _row_names(row: dict[str, str]) -> tuple[str, ...]:
    """The names the table gives a company: its entity-master name, its short names and its English names."""
    names = [row.get("name_zh") or "", *(row.get("short_zh") or "").split("|"), row.get("name_en") or ""]
    names += (row.get("short_en") or "").split("|")
    return tuple(dict.fromkeys(name.strip() for name in names if name and name.strip()))


_AH_CODES: dict[str, tuple[str, ...]] = {
    row["h_code"][-4:]: _row_names(row) for row in _dual_listing_rows() if row["h_code"][-4] != "0"
}
_BARE_CODE = re.compile(r"(?<![0-9A-Za-z.])([1-9]\d{3})(?![A-Za-z])" + _HK_CODE_TAIL, re.IGNORECASE)


def _ah_code_spans(query: str) -> list[tuple[int, int]]:
    """(round 15, K5) "中国平安2318", "招行 3968 现在多少", "Ping An 2318": a bare Hong Kong code right after the name
    of the company it belongs to (``_AH_CODES``); the span runs from the name to the code."""
    spans = []
    for match in _BARE_CODE.finditer(query or ""):
        names = _AH_CODES.get(match.group(1))
        if not names:
            continue
        before = query[: match.start()].rstrip()
        before = before[:-1].rstrip() if before.endswith("的") else before
        for name in sorted(names, key=len, reverse=True):
            if before.lower().endswith(name.lower()):
                spans.append((len(before) - len(name), match.end()))
                break
    return spans


# (round 16, review-12 L7) The line patterns above take the company as "two to eight characters that are not a
# conjunction or 的" before the line word, so a name that contains 的 (美的集团, "美的港股多少钱一股") escaped
# them and was answered with the A-share price. For every company in the dual-listing table the check is driven by
# data: one of its names (the entity master's name, the table's short names, the alias table's colloquial names,
# its English names), then optionally 的 / 在 / 于, then a line word: 港股, H股, a Hong Kong venue with a listing or
# quote word, its own HKEX code, a Hong Kong-dollar quote, a GDR or an ADR. The name is part of the span, so the
# A-share target it names is a lookalike (refused with the note that its A shares are covered), whatever characters
# the name has.
_ZH_LINE_WORDS = (
    r"(?P<h>港股(?!通)|H\s*股(?!东|指数|ETF|基金|LOF|\(QDII)|"
    # a venue: an exchange by name, or 香港 with a listing or quote word ("在香港有工厂" is not a line)
    r"(?:在|于)?\s*(?:港交所|联交所|香港联合交易所|香港联交所|香港交易所)|"
    r"(?:在|于)?\s*香港(?:市场)?(?:上|里|那边|那头)?(?:上市|挂牌|发行|交易|的?\s*(?:股票|股份|那部分|部分|那只|"
    + _HK_QUOTE_WORDS
    + r"))|"
    # a Hong Kong-dollar quote: "美的港币价格", "美的现在多少港元" (not a currency exchange: 换, 兑, 汇率)
    r"(?:以|用|按)?\s*" + _HKD + r"(?!\s*(?:汇率|兑|换))|"
    r"(?:(?![换兑汇])[^，。？?！!；;]){0,10}?多少(?:钱)?\s*" + _HKD + r"(?!\s*(?:汇率|兑|换))"
    r")|"
    r"(?P<gdr>(?:伦敦|英国|瑞士|苏黎世|法兰克福|德国)(?:证券)?(?:交易所)?(?:上市|挂牌|交易|发行)?的?\s*(?:GDR|全球存托凭证|存托凭证)|"
    r"GDR|全球存托凭证|沪伦通)|"
    r"(?P<adr>(?:美国|美股|纽约|纽交所|纳斯达克)(?:证券)?(?:交易所)?(?:上市|挂牌|交易|发行)?的?\s*(?:ADR|ADS|存托凭证|存托股)|"
    r"ADR|ADS|美国存托(?:凭证|股票|股份)?)"
)
_EN_LINE_WORDS = (
    r"(?P<h>H[- ]?shares?|(?:Hong\s+Kong|HK)[- ](?:shares?|stocks?|listing|line|counter|quotes?|prices?|share\s+price|"
    r"listed|traded|quoted)|(?:HKD|HK\$|Hong\s+Kong\s+dollar)\s+(?:share\s+)?(?:price|quote|close|closing\s+price|"
    r"value|market\s+value)|"
    r"(?:(?!\b(?:convert|conversion|exchange\s+rate)\b)[^.?!]){0,40}?\b(?:in|into)\s+(?:HKD|HK\$|HK\s+dollars?|"
    r"Hong\s+Kong\s+dollars?)|(?:listed|traded|quoted)\s+(?:in|on)\s+(?:the\s+)?(?:HK|Hong\s+Kong))|"
    r"(?P<gdr>(?:(?:London|Swiss|Zurich|SIX|LSE)[- ](?:listed|traded|quoted)\s+)?(?:GDRs?|global\s+depositary\s+"
    r"(?:receipts?|shares?)))|"
    r"(?P<adr>(?:(?:NYSE|Nasdaq|US|U\.S\.)[- ](?:listed|traded|quoted)\s+)?(?:ADRs?|ADSs?|American\s+depositary\s+"
    r"(?:receipts?|shares?)))"
)


@dataclass(frozen=True)
class _DualListingMatcher:
    zh: re.Pattern[str]
    en: re.Pattern[str]
    codes: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...]


def _alias_names(symbols: set[str]) -> dict[str, list[str]]:
    """The alias table's colloquial names of these A-share symbols ("美的", "宁王", "中芯"): Chinese only, two to six
    characters, no spaces or brackets (the table also holds fund names and spaced variants under other types)."""
    from ..data_loader import load_aliases, load_entities

    ids = {
        row.get("entity_id"): row.get("symbol")
        for row in load_entities()
        if row.get("symbol") in symbols and row.get("entity_type") == "stock"
    }
    names: dict[str, list[str]] = {}
    for alias in load_aliases():
        symbol = ids.get(str(alias.get("entity_id") or ""))
        text = str(alias.get("alias_text") or "")
        if symbol and alias.get("alias_type") == "colloquial_alias" and re.fullmatch(r"[一-鿿]{2,6}", text):
            names.setdefault(symbol, []).append(text)
    return names


@lru_cache(maxsize=1)
def _dual_listing_matcher() -> _DualListingMatcher:
    rows = _dual_listing_rows()
    aliases = _alias_names({row["a_symbol"] for row in rows})
    zh_names: set[str] = set()
    en_names: set[str] = set()
    codes = []
    for row in rows:
        names = [*_row_names(row), *aliases.get(row["a_symbol"], [])]
        zh = {name for name in names if re.search(r"[一-鿿]", name)}
        en = {name for name in names if not re.search(r"[一-鿿]", name)}
        zh_names |= zh
        en_names |= en
        code = row["h_code"]
        # its own code after the name: "美的0300", "美的（00300）", "Midea 300.HK"; a bare code without a leading
        # zero also follows ``_HK_CODE_TAIL`` ("比亚迪1211" yes, "比亚迪2025年报" is a year)
        code_part = (
            r"[(（]?\s*(?:HK\s?)?(?<![\d.])0?"
            + re.escape(code[-4:])
            + r"(?:\s*\.\s*HK)?(?![\d.])\s*[)）]?"
            + _HK_CODE_TAIL
        )
        name_part = "|".join(re.escape(name) for name in sorted(zh | en, key=len, reverse=True))
        codes.append(
            (
                re.compile(r"(?:" + name_part + r")\s*(?:的\s*)?" + code_part, re.IGNORECASE),
                tuple(sorted(zh | en)),
            )
        )
    zh_part = "|".join(re.escape(name) for name in sorted(zh_names, key=len, reverse=True))
    en_part = "|".join(re.escape(name) for name in sorted(en_names, key=len, reverse=True))
    return _DualListingMatcher(
        zh=re.compile(r"(?:" + zh_part + r")\s*(?:的\s*)?(?:" + _ZH_LINE_WORDS + r")", re.IGNORECASE),
        en=re.compile(
            r"(?<![\w&.-])(?:" + en_part + r")(?![\w&-])(?:'s|’s)?\s*(?:" + _EN_LINE_WORDS + r")(?![A-Za-z])",
            re.IGNORECASE,
        ),
        codes=tuple(codes),
    )


def dual_listing_spans(query: str) -> list[tuple[int, int, str]]:
    """(round 16, L7) ``(start, end, kind)`` where a company of the dual-listing table is named right before one of
    its foreign lines (``h``, ``gdr``, ``adr``); the span starts at the name."""
    text = query or ""
    if not text.strip():
        return []
    matcher = _dual_listing_matcher()
    found: list[tuple[int, int, str]] = []
    for pattern in (matcher.zh, matcher.en):
        for match in pattern.finditer(text):
            kind = next(kind for kind in ("h", "gdr", "adr") if match.group(kind) is not None)
            found.append((match.start(), match.end(), kind))
    for pattern, _names in matcher.codes:
        found.extend((match.start(), match.end(), "h") for match in pattern.finditer(text))
    return sorted(set(found))


_FOREIGN_LINE = re.compile("|".join((_HK_LINE, _GDR_LINE, _ADR_LINE, _AH_PREMIUM)), re.IGNORECASE)
# the A/H premium itself (not "A股和H股分别多少", which asks for both lines)
_PREMIUM_WORDS = re.compile(r"溢价|折价|比价|价差|\b(?:premium|discount|spread|price\s+gap)\b", re.IGNORECASE)
_FOREIGN_LINE_KINDS = (
    ("ah", re.compile(r"(?=.*(?:" + _PREMIUM_WORDS.pattern + r"))(?:" + _AH_PREMIUM + r")", re.IGNORECASE | re.S)),
    ("gdr", re.compile(_GDR_LINE, re.IGNORECASE)),
    ("adr", re.compile(_ADR_LINE, re.IGNORECASE)),
    ("h", re.compile(_HK_LINE + "|" + _H_SHARE, re.IGNORECASE)),
)
# "A股和H股有什么区别", "What is a GDR?": a concept, not a line of a company
_LINE_CONCEPT = re.compile(
    r"(?:什么是|是什么|什么意思|有什么区别|有何区别|的区别|区别是|含义|定义|怎么理解|指的是|"
    r"\bwhat\s+(?:is|are)\s+(?:an?\s+)?(?:H[- ]shares?|ADRs?|GDRs?|depositary)|\bdifference\s+between\b|"
    r"\bwhat\s+does\s+(?:ADR|GDR|H[- ]share)\s+mean\b)",
    re.IGNORECASE,
)


def foreign_line_spans(query: str) -> list[tuple[int, int]]:
    """(round 14, J7) Where the question names a company's Hong Kong, London/Swiss or US line or its A/H premium."""
    if _LINE_CONCEPT.search(query or ""):
        return []
    spans = [match.span() for match in _FOREIGN_LINE.finditer(query or "") if match.group(0).strip()]
    table = [(start, end) for start, end, _kind in dual_listing_spans(query or "")]
    return sorted({*spans, *_ah_code_spans(query or ""), *table})


def foreign_line_kind(query: str) -> str | None:
    """``ah`` (an A/H premium), ``gdr``, ``adr`` or ``h`` for the line the question asks about, else ``None``."""
    text = query or ""
    if not foreign_line_spans(text) and not re.search(_H_SHARE, text, re.IGNORECASE):
        return None
    if _LINE_CONCEPT.search(text):
        return None
    # (round 16, L7) the line word after a table company's name when no line pattern names the kind
    table = next((kind for _start, _end, kind in dual_listing_spans(text)), "h")
    return next((kind for kind, pattern in _FOREIGN_LINE_KINDS if pattern.search(text)), table)


# US / Hong Kong listed names and markets. Concept-sector phrasing ("苹果概念股", "特斯拉产业链") is an
# A-share theme and stays in scope.
_FOREIGN_EQUITY = re.compile(
    r"(?:苹果公司|苹果股票|苹果股价|苹果的股[价票]|苹果(?=的?(?:市盈率|市净率|市值|财报|营收|净利润|股价|股票|能买|值得买|会涨|会跌))|"
    r"特斯拉|英伟达|微软|谷歌|亚马逊|脸书|奈飞|伯克希尔|台积电|"
    r"腾讯控股|阿里巴巴|美团|小米集团|港股|美股|中概股|纳斯达克|道琼斯|标普500|恒生指数|恒指|" + _HK_US_LISTED + r")"
    r"(?!概念|产业链|供应链|链)|"
    r"\b(?:apple|tesla|nvidia|microsoft|google|alphabet|amazon|netflix|berkshire|tencent|alibaba|meituan|tsmc)\b|"
    r"\b(?:ping an (?:good doctor|healthcare)|tencent music|jd (?:health|logistics)|jd\.com|wuxi biologics|"
    r"alibaba health|xiaomi|baidu|netease|pinduoduo|pdd holdings|kuaishou|bilibili|nio inc|li auto|xpeng)\b|"
    r"(?<![A-Za-z])(?:AAPL|TSLA|NVDA|MSFT|GOOGL?|AMZN|NFLX|META|BABA)(?![A-Za-z])|"
    r"\b(?:us|u\.s\.|american|hong kong) (?:stocks?|shares|equities|market)\b|\bnasdaq\b|\bs&p 500\b|\bdow jones\b|"
    r"\bhang seng\b|\bnyse\b|"
    r"\b(?:byd electronic|geely(?: auto(?:mobile)?)?|sunny optical|china resources land|china overseas land|"
    r"citic limited|cspc|sino biopharm|country garden|hsbc|aia group|ck hutchison)\b|" + _H_SHARE,
    re.IGNORECASE,
)


# A question about how a foreign market affects A-shares ("美股大跌对A股有什么影响") is in scope.
_A_SHARE_ANCHOR = re.compile(
    r"A股|沪深|上证|深证|创业板|科创板|北交所|\bA-?shares?\b|\bChin(?:a|ese) (?:stocks?|equities|market)\b",
    re.IGNORECASE,
)


def foreign_equity_spans(query: str) -> list[tuple[int, int]]:
    """Where the question names a US / Hong Kong listed company or market: an A-share name inside such a span
    ("平安" in 平安好医生) is part of the foreign name, not a target of its own. Since round 14 (J7) also a company's
    Hong Kong / GDR / ADR line or its A/H premium by any wording (``foreign_line_spans``)."""
    spans = [match.span() for match in _FOREIGN_EQUITY.finditer(query or "")]
    return sorted({*spans, *foreign_line_spans(query)})


def out_of_coverage(query: str) -> str | None:
    """``crypto`` or ``foreign_equity`` when the question is about an asset FinSight has no data for. A company's
    foreign line (``foreign_line_spans``) is checked before the A-share anchor: "招商银行A股相对H股溢价多少" asks for
    an H-share price whatever else it says (round 14, J7)."""
    text = query or ""
    if foreign_line_spans(text):
        return "foreign_equity"
    if _A_SHARE_ANCHOR.search(text):
        return None
    if _CRYPTO.search(text):
        return "crypto"
    if _FOREIGN_EQUITY.search(text):
        return "foreign_equity"
    return None


def asks_h_share(query: str) -> bool:
    """Whether the question asks for a Hong Kong (H-share) line or ticker ("中国平安H股", "2318.HK"); since round 14
    (J7) any foreign line of a company (``foreign_line_kind``: Hong Kong, GDR, ADR, the A/H premium)."""
    return bool(re.search(_H_SHARE, query or "", re.IGNORECASE)) or foreign_line_kind(query) is not None


def foreign_line_note(query: str, names: str, *, zh: bool, companies: list[str] | tuple[str, ...] | None = None) -> str:
    """(round 14, J7) What the refused question asked for, and that the company's A shares are covered. (round 17,
    M15) ``companies`` (entity-master names): when none of them has a Hong Kong line in the dual-listing table, an
    H-share question is told that the company has no H shares."""
    kind = foreign_line_kind(query) or "h"
    asked_zh = {
        "h": " H 股（香港上市的股份）",
        "gdr": "在伦敦或瑞士上市的 GDR（全球存托凭证）",
        "adr": "在美国上市的 ADR（美国存托凭证）",
        "ah": " A 股相对 H 股的溢价，需要 H 股（香港上市的股份）的行情",
    }[kind]
    asked_en = {
        "h": "the H shares (the Hong Kong line)",
        "gdr": "the GDRs (global depositary receipts listed in London or Switzerland)",
        "adr": "the ADRs (American depositary receipts listed in the US)",
        "ah": "the A/H premium, which needs the H-share (Hong Kong) price",
    }[kind]
    if kind in {"h", "ah"} and has_hong_kong_line(companies or ()) is False:
        # (round 17, review-13 M15) "茅台港股多少钱": the dual-listing table has no Hong Kong line for the company, so
        # the note says so instead of implying an H share that FinSight merely does not cover
        if zh:
            return (
                f"所问的是{asked_zh}；据港交所证券名单，{names}没有在香港上市的股份（没有 H 股）。{names}在 A 股上市的"
                "股份在覆盖范围内，如需 A 股数据请直接询问。"
            )
        return (
            f" The question asks for {asked_en}; according to the HKEX list of securities, {names} has no shares "
            f"listed in Hong Kong (no H shares). {names}'s A shares are covered, so ask about the A shares directly if "
            "that is what you need."
        )
    if zh:
        return f"所问的是{asked_zh}；{names}在 A 股上市的股份在覆盖范围内，如需 A 股数据请直接询问。"
    return (
        f" The question asks for {asked_en}; {names}'s A shares are covered, so ask about the A shares directly if "
        "that is what you need."
    )


LOOKALIKE_REASON = "lookalike_name"
HK_ONLY_REASON = "hk_only_name"


@dataclass(frozen=True)
class NotCoveredName:
    """(round 17, review-13 M3) A company the question names that the entity master does not have: a Hong Kong-only
    company (``code`` its HKEX code) or a lookalike of a covered company's short name; ``owner`` is the covered
    company it resembles ("" if none)."""

    name: str
    owner: str = ""
    code: str = ""


def not_covered_names(reasons: list[str] | tuple[str, ...]) -> list[NotCoveredName]:
    """The ``hk_only_name:美的置业|03990->美的集团`` and ``lookalike_name:平安证券->中国平安`` entries of an NLU trace
    or of the route reasons, once per name."""
    found: dict[str, NotCoveredName] = {}
    for reason in reasons or []:
        text = str(reason)
        if not text.startswith((f"{LOOKALIKE_REASON}:", f"{HK_ONLY_REASON}:")):
            continue
        body = text.split(":", 1)[1]
        body, _arrow, owner = body.partition("->")
        name, _bar, code = body.partition("|")
        if name.strip() and name.strip() not in found:
            found[name.strip()] = NotCoveredName(name.strip(), owner.strip(), code.strip())
    return list(found.values())


def not_covered_note(names: list[NotCoveredName], *, zh: bool, english: Any = None) -> str:
    """(round 17, M3) What the question named that FinSight has no data on, and that a resembling covered company
    was not used instead ("「美的置业」（港交所代码 03990）只在香港上市…与美的集团名称相近，但不是同一家公司")."""
    parts = []
    for item in names:
        owner_en = ((english(item.owner) if english else None) or item.owner) if item.owner else ""
        if zh:
            head = (
                f"「{item.name}」（港交所代码 {item.code}）在香港上市，不在 FinSight 的数据范围内"
                if item.code
                else f"FinSight 的数据中没有「{item.name}」"
            )
            tail = (
                f"；它与{item.owner}名称相近，但不是同一家公司，没有用{item.owner}的数据回答。" if item.owner else "。"
            )
            parts.append(head + tail)
        else:
            head = (
                f'"{item.name}" (HKEX {item.code}) is listed in Hong Kong and is outside FinSight\'s data'
                if item.code
                else f'FinSight has no data on "{item.name}"'
            )
            tail = (
                f"; its name resembles {owner_en}, but it is a different company, so it was not answered with "
                f"{owner_en}'s figures."
                if item.owner
                else "."
            )
            parts.append(head + tail)
    return ("" if zh else " ").join(parts)


@lru_cache(maxsize=1)
def _a_share_stock_names() -> frozenset[str]:
    from ..data_loader import load_entities

    return frozenset(
        str(row.get("canonical_name"))
        for row in load_entities()
        if row.get("entity_type") == "stock" and row.get("symbol")
    )


def a_share_stocks(names: list[str] | tuple[str, ...]) -> list[str]:
    """(round 17) The names that are A-share stocks of the entity master (not an index such as 恒生指数, which the
    NLU can pick up next to 港股): only these get the "its A shares are covered" note."""
    stocks = _a_share_stock_names()
    return [name for name in names if name in stocks]


def has_hong_kong_line(names: list[str] | tuple[str, ...]) -> bool | None:
    """(round 17, review-13 M15) Whether any of these A-share companies (entity-master names) has a Hong Kong line:
    ``True`` when the dual-listing table or the audit lists one, ``False`` only when every named company is in the
    audit (``data/runtime/hong_kong_line_audit.csv``: the covered companies, each checked against the HKEX list of
    securities) without a Hong Kong code, ``None`` otherwise. The dual-listing table does not list every A+H company,
    so a company outside the audit is never said to have no H shares."""
    from ..data_loader import load_hong_kong_line_audit

    wanted = {str(name) for name in names if name}
    if not wanted:
        return None
    if any(str(row.get("name_zh")) in wanted for row in _dual_listing_rows()):
        return True
    audit = {str(row.get("name_zh")): str(row.get("h_code") or "") for row in load_hong_kong_line_audit()}
    if any(audit.get(name) for name in wanted):
        return True
    if all(name in audit for name in wanted):
        return False
    return None


def out_of_coverage_text(category: str, *, zh: bool) -> str:
    if category == LOOKALIKE_REASON:
        # (round 17, M3) a company absent from the entity master; the note before this names it
        return (
            "FinSight 覆盖 A 股（实体库中的上市公司）、公募基金/ETF、指数和中国宏观数据；如需名称相近的那家公司的数据，"
            "请使用它的全称或代码提问。"
            if zh
            else "FinSight covers China A-shares (the companies in its entity master), funds/ETFs, indices and China "
            "macro data; to ask about the similarly named covered company, use its full name or code."
        )
    if category == "crypto":
        return (
            "FinSight 只覆盖 A 股、公募基金/ETF、指数和中国宏观数据，没有比特币等加密资产的行情或数据，"
            "因此无法回答这个问题。可以换成 A 股或基金相关的问题，例如「贵州茅台最新收盘价是多少？」。"
            if zh
            else "FinSight only covers China A-shares, funds/ETFs, indices and China macro data. It has no data on "
            "crypto assets such as Bitcoin, so it cannot answer this. Try an A-share or fund question instead, "
            'for example "What was Kweichow Moutai\'s latest close?"'
        )
    return (
        "FinSight 只覆盖 A 股、公募基金/ETF、指数和中国宏观数据，没有美股、港股个股或海外市场的数据，"
        "因此无法回答这个问题。可以换成 A 股或基金相关的问题，例如「贵州茅台最新收盘价是多少？」。"
        if zh
        else "FinSight only covers China A-shares, funds/ETFs, indices and China macro data. It has no data on US "
        "or Hong Kong listed stocks or overseas markets, so it cannot answer this. Try an A-share or fund question "
        'instead, for example "What was Kweichow Moutai\'s latest close?"'
    )


# --------------------------------------------------------------------------- requested metrics


@dataclass(frozen=True)
class Metric:
    key: str
    zh: str
    en: str
    pattern: re.Pattern[str]
    fields: tuple[str, ...]  # keys in get_fundamentals ``metrics`` that answer it
    # One field of every group present, the metric can be derived (net margin: revenue and net profit).
    derivable_from: tuple[tuple[str, ...], ...] = ()
    # Why it cannot be given when the inputs are missing ("PEG needs the profit growth rate").
    needs_zh: str = ""
    needs_en: str = ""

    def derivable(self, metrics: dict[str, Any]) -> bool:
        return bool(self.derivable_from) and all(
            any(metrics.get(field) is not None for field in group) for group in self.derivable_from
        )


def _metric(
    key: str,
    zh: str,
    en: str,
    pattern: str,
    fields: tuple[str, ...],
    derivable: tuple[tuple[str, ...], ...] = (),
    needs: tuple[str, str] = ("", ""),
) -> Metric:
    return Metric(key, zh, en, re.compile(pattern, re.IGNORECASE), fields, derivable, *needs)


# Profit growth rates a source may report (percent), in order of preference.
PROFIT_GROWTH_FIELDS = ("netprofit_yoy", "net_profit_yoy", "profit_growth", "dt_netprofit_yoy")


# Order matters: growth rates are matched before the level they grow ("营收增速" is not "营收").
METRICS: tuple[Metric, ...] = (
    _metric(
        "revenue_growth",
        "营收增速",
        "revenue growth",
        r"(?:营收|营业收入|收入)(?:的)?(?:增速|增长率|同比增长|同比|增幅|增长)|revenue growth|sales growth|"
        # (round 12) the first of two coordinated levels: "营收和净利润同比各增长了多少" (zero width after the level,
        # so the second one, "净利润同比", is still matched by its own metric)
        r"(?:营收|营业收入|收入)(?=(?:和|与|及|、)(?:净利润|净利|利润)(?:的)?(?:增速|增长率|同比|增幅|增长))",
        ("revenue_yoy", "revenue_growth", "or_yoy", "tr_yoy"),
    ),
    _metric(
        "profit_growth",
        "净利润增速",
        "net profit growth",
        r"(?:净利润|净利|利润)(?:的)?(?:增速|增长率|同比增长|同比|增幅|增长)|(?:profit|earnings|income) growth|"
        r"(?:净利润|净利|利润)(?=(?:和|与|及|、)(?:营收|营业收入|收入)(?:的)?(?:增速|增长率|同比|增幅|增长))",
        ("netprofit_yoy", "net_profit_yoy", "profit_growth", "dt_netprofit_yoy"),
    ),
    _metric(
        "growth",
        "增速",
        "growth rate",
        r"增速|增长率|同比增长|同比增幅|\bgrowth(?: rate)?\b|\byoy\b|year[- ]over[- ]year",
        (
            "revenue_yoy",
            "revenue_growth",
            "or_yoy",
            "tr_yoy",
            "netprofit_yoy",
            "net_profit_yoy",
            "profit_growth",
            "dt_netprofit_yoy",
        ),
    ),
    # (round 14, review-10 J8) the free-float market cap and the net profit after non-recurring items are answered from
    # their own fields (circ_mv, profit_dedt), never from the total market cap or 归母 net profit; operating, gross and
    # pre-tax profit are named so a question about them is told they are missing, not answered with net profit.
    _metric(
        "float_market_cap",
        "流通市值",
        "free-float market cap",
        r"流通(?:盘)?市值|\bfree[- ]?float(?:ing)? market (?:cap|value)|\bfloat(?:ing)? market (?:cap|value)|"
        r"\bcirculating market (?:cap|value)",
        ("circ_mv",),
    ),
    _metric(
        "market_cap",
        "总市值",
        "market cap",
        r"总市值|市值|market cap(?:itali[sz]ation)?|\bmarket value\b",
        ("total_mv", "market_cap", "total_market_cap"),
    ),
    _metric(
        "profit_dedt",
        "扣非净利润",
        "net profit excluding non-recurring items",
        r"扣除非经常性损益|扣非|\bnon-?recurring\b|"
        r"\b(?:recurring|core|underlying) (?:net )?(?:profit|earnings|income)\b|\bdeducted net profit\b",
        ("profit_dedt",),
    ),
    _metric(
        "operating_profit",
        "营业利润",
        "operating profit",
        r"营业利润(?!率)|经营利润(?!率)|\boperating (?:profit|income)\b(?! margin)|\bEBIT\b",
        ("operate_profit", "operating_profit"),
    ),
    _metric(
        "gross_profit",
        "毛利润",
        "gross profit",
        r"毛利润(?!率)|毛利额|\bgross profits?\b(?! margin)",
        ("gross_profit",),
    ),
    _metric(
        "total_profit",
        "利润总额",
        "total profit",
        # (round 15, review-11 K10) "净利润总额" is the total of net profits, not 利润总额
        r"(?<!净)利润总额|税前利润|\bpre-?tax (?:profit|income)\b|\bprofit before tax\b",
        ("total_profit",),
    ),
    _metric(
        "dividend_yield",
        "股息率",
        "dividend yield",
        # (round 16, review-12 L8) "今年会分红多少": a dividend asked as 分红 / 派息 is not in the data either
        r"股息率|股息|分红|派息|派现|dividend yield|\bdividends?\b",
        ("dividend_yield", "dv_ratio", "dv_ttm"),
    ),
    _metric(
        "debt_ratio",
        "资产负债率",
        "debt ratio",
        r"资产负债率|负债率|负债水平|杠杆率|debt[- ]to[- ](?:asset|equity)|debt ratio|leverage|gearing",
        ("debt_to_assets", "debt_ratio", "liability_ratio", "debt_to_equity"),
    ),
    _metric(
        "gross_margin",
        "毛利率",
        "gross margin",
        r"毛利率|毛利润率|gross margin",
        ("gross_margin", "grossprofit_margin"),
    ),
    # (round 9, E8) net profit as a share of revenue, however it is phrased: "净利润占营收的比重", "(销售)利润率",
    # "每卖100元能落下几块净利润", "profit as a percentage of sales"; not the gross margin ("毛利润率").
    _metric(
        "net_margin",
        "净利率",
        "net margin",
        r"净利率|净利润率|销售净利率|(?<![毛])利润率|"
        r"每(?:赚|卖|收|收入|实现)?[^，。？?,.!！]{0,3}?\d+\s*(?:块|元)(?:钱)?(?:的)?(?:营收|收入|销售额)?"
        r"[^，。？?,.!！]{0,10}?(?:净利润|净利|净赚|利润|落袋)|"
        r"net[- ](?:profit[- ])?margin|profit[- ]margin|(?:net )?(?:profit|income|earnings) as a (?:share|percentage|"
        r"proportion|percent) of (?:revenue|sales)|"
        # (round 11, G5; round 12) "净利润是营收的百分之几", "营收里有多少变成净利润", "What share of that revenue is
        # left as net profit?": one vocabulary with the comparison frame
        f"{NET_MARGIN_SHARE_SOURCE}",
        ("net_margin", "netprofit_margin"),
        (("revenue",), ("net_profit",)),
    ),
    # (round 11, G5) EPS: stated when a source reports it; otherwise named as missing (no share count to derive it),
    # and the template adds the value implied by the latest close and the P/E (TTM), labelled as such.
    _metric(
        "eps",
        "每股收益",
        "EPS",
        # (round 12, H9) "一股赚多少钱", "每股能赚几块": colloquial EPS (not "一股值多少钱", a fair value)
        r"每股收益|每股盈利|每股净利润?|(?<![A-Za-z])EPS(?![A-Za-z])|earnings per share|"
        r"(?:一|每)股(?:能|可以|大概|大约)?赚(?:了)?(?:多少|几)|"
        r"\bhow much (?:does|did) (?:it|each share) earn per share\b",
        ("eps", "basic_eps", "diluted_eps", "eps_ttm"),
    ),
    # P/S needs the market cap, which no configured source has: stated as not computable, never replaced by P/E.
    _metric(
        "ps",
        "市销率",
        "P/S",
        r"市销率|(?<![A-Za-z])P/?S(?![A-Za-z])|price[- ]to[- ]sales",
        ("ps_ttm", "ps"),
        (),
        (
            "市销率等于总市值除以营业收入，当前数据没有总市值或市销率",
            "P/S is the market cap divided by revenue, and the data has neither the market cap nor a P/S figure",
        ),
    ),
    # PEG = P/E ÷ profit growth (percent): derivable only when a source reports the growth rate.
    _metric(
        "peg",
        "PEG",
        "PEG",
        r"(?<![A-Za-z])PEG(?![A-Za-z])|市盈增长比|市盈率相对盈利增长比率",
        ("peg", "peg_ratio"),
        (("pe_ttm", "pe"), PROFIT_GROWTH_FIELDS),
        (
            "PEG 等于市盈率除以净利润增速，当前数据没有净利润增速",
            "PEG is the P/E divided by the net profit growth rate, and the data has no growth rate",
        ),
    ),
    _metric(
        "cash_flow",
        "现金流",
        "cash flow",
        r"现金流|cash ?flow",
        ("operating_cash_flow", "n_cashflow_act", "free_cash_flow"),
    ),
)
_METRIC_BY_KEY = {metric.key: metric for metric in METRICS}
# Labels of extra metrics rendered when present (the template renders PE/PB/ROE/revenue/net profit itself).
# A field shared by several metrics takes the first (most specific) one: revenue_yoy is "营收增速", not "增速".
EXTRA_METRIC_FIELDS: dict[str, Metric] = {}
for _metric_item in METRICS:
    for _field in _metric_item.fields:
        EXTRA_METRIC_FIELDS.setdefault(_field, _metric_item)


# (round 11, G5) "我有1000股五粮液，按最新收盘价值多少钱", "500 shares of Moutai, what are they worth?": the value of a
# stated holding at the last close (shares × close), not a fair-value question. "一股/每股值多少" is a fair value, and
# "10股派…" a dividend per share, so neither is a holding.
_HOLDING_COUNT_ZH = (
    # (round 15, review-11 K7) "1.5万股", "3万股", "2千股": a count with a 万 / 千 multiplier is a holding, not a cap
    # (round 16, review-12 L3) mixed units: "1万2千股", "3千5百股", "1万2股" (12,000)
    r"(?<![每\d.])(?P<n>\d+(?:\.\d+)?(?:\s*[万千百]\s*\d*)+(?=\s*[股份手])|\d[\d,，]*|"
    r"[两二三四五六七八九十百千万][零〇一二两三四五六七八九十百千万]*|"
    r"一[十百千万][零〇一二两三四五六七八九十百千万]*)"
    # (round 12) fund units count like shares ("两万份沪深300ETF"); a lot (手) is 100 shares ("3手茅台")
    r"\s*(?P<unit>股(?![价票东份息市权本派送转配])|份|手(?![续机表头上下动脚指]|工(?!商)))"
)
_HOLDING_VALUE_ZH = re.compile(
    _HOLDING_COUNT_ZH + r"[^。？?！!；;]{0,24}?"
    r"(?:值|市值|价值|总值|总额|合计|一共|总共|算下来|折合)[^。？?！!；;]{0,4}?(?:多少|几)|"
    # (round 15, review-11 K7) a holding listed in clauses ("招商银行 1,000 股；工商银行 3,000 股") and a bare value
    # question at the end ("总市值？", "合计值多少"): the holdings' value, not the companies' market caps
    + _HOLDING_COUNT_ZH.replace("(?P<n>", "(?P<n5>").replace("(?P<unit>", "(?P<unit5>")
    + r"[^？?！!]{0,80}?[。；;，,]\s*(?:那|这些|我的|持仓|全部|一共|总共|合计|加起来|这样)*\s*"
    # (round 16, review-12 L3) "，按最新收盘价算总市值", "，用昨天的收盘价算一下总值"
    r"(?:(?:请|帮我)?(?:按|以|用|拿)(?:最新|最近|昨天|今天|当前|现在|最后)?(?:一个交易日)?(?:的)?"
    r"(?:收盘价?|价格|股价|市价)(?:来|去)?(?:算|计算|折算|估算)(?:一下|下)?(?:的)?\s*)?"
    r"(?:(?:我的|持仓的?|这些|全部|一共|总共|合计)?\s*)"
    r"(?:总市值|市值|总价值|总值|价值|值多少钱?|合计多少)(?:是|为|有)?(?:多少钱?|几)?(?:呢)?\s*[？?]?\s*$"
)
# (round 12) English word orders: "500 shares of X", "200 Moutai shares", "my 200 Moutai shares", the question in the
# next sentence, "the worth of 300 shares"; not "3.88 billion shares"
_EN_COUNT = (
    # (round 16, review-12 L13) "2万 shares", "20k shares", "1.5 thousand shares"
    r"(?<![\d.,])(?P<{name}>\d[\d,]*(?:\.\d+)?\s*(?:万|千|k\b|thousand\b)?|\d[\d,]*)\s+"
    r"(?:(?!billion|million|thousand|bn|mn)[A-Za-z][\w'.-]*\s+){{0,3}}?(?:shares?|units?)\b"
)
_HOLDING_VALUE_EN = re.compile(
    _EN_COUNT.format(name="n")
    + r"[^.?!]{0,60}?\b(?:worth|value|how much)\b|"
    + _EN_COUNT.format(name="n3")
    + r"[^.?!]{0,60}\.\s+(?:so\s+|and\s+)?(?:what|how much|total|combined)\b[^.?!]{0,40}?\b(?:worth|value)\b|"
    r"\bhow much (?:are|is|would) (?:my )?(?P<n2>\d[\d,]*(?:\.\d+)?\s*(?:万|千|k\b|thousand\b)?)\s*"
    r"(?:shares?|units?)\b|"
    r"\bworth of (?:my |the |these |those )?(?P<n4>\d[\d,]*)\s+(?:[A-Za-z][\w'.-]*\s+){0,3}?(?:shares?|units?)\b",
    re.IGNORECASE,
)
# (round 13) The cost of buying a stated number of shares or lots at the last close: "按这个价格，买三手要多少钱",
# "买入500股五粮液大概得花多少钱", "How much would it cost to buy 300 shares at that close?". The same arithmetic as a
# holding's value (count × close), stated as an amount before trading costs; never advice.
_PURCHASE_ZH = re.compile(
    r"(?:买|买入|购买|入手|建仓)\s*" + _HOLDING_COUNT_ZH + r"[^。？?！!；;]{0,12}?"
    r"(?:要|需要|得|需|须|大概|大约|一共|总共|共)*\s*(?:花|用|投入|准备)?\s*(?:多少钱|多少元|多少资金|几万|几千|多少)"
)
_PURCHASE_EN = re.compile(
    r"\bhow much\b[^.?!]{0,40}?\b(?:buy|purchase)\s+(?P<n>\d[\d,]*)\s+(?:[A-Za-z][\w'.-]*\s+){0,3}?"
    r"(?P<unit>shares?|units?|lots?)\b|"
    r"\b(?:buy(?:ing)?|purchas(?:e|ing))\s+(?P<n2>\d[\d,]*)\s+(?:[A-Za-z][\w'.-]*\s+){0,3}?(?P<unit2>shares?|units?|lots?)\b"
    r"[^.?!]{0,40}?\b(?:cost|how much|need)\b|"
    r"\b(?:cost|take|need)\b[^.?!]{0,20}?\b(?:buy|purchase)\s+(?P<n3>\d[\d,]*)\s+(?:[A-Za-z][\w'.-]*\s+){0,3}?"
    r"(?P<unit3>shares?|units?|lots?)\b",
    re.IGNORECASE,
)


def _purchase_match(text: str) -> tuple[int, int, str, tuple[int, int]] | None:
    """``(count in shares, lots or 0, unit word, span)`` of a purchase-cost question, else ``None``."""
    match = _PURCHASE_ZH.search(text)
    if match is not None:
        count = count_value(match.group("n"))
        lots = count if match.group("unit") == "手" else 0
        return (count * 100 if lots else count), lots, match.group("unit"), match.span()
    match = _PURCHASE_EN.search(text)
    if match is not None:
        raw = (match.group("n") or match.group("n2") or match.group("n3") or "").replace(",", "")
        unit = (match.group("unit") or match.group("unit2") or match.group("unit3") or "").lower()
        count = int(raw)
        lots = count if unit.startswith("lot") else 0
        return (count * 100 if lots else count), lots, unit, match.span()
    return None


# (round 14, review-10 J4) gain or loss against a stated cost price: "我300股成本90元，浮亏多少", "我30块买的，现在赚了
# 多少", "I bought 500 shares at 70 — what's my profit?": (close − cost) per share, in percent of the cost, and times
# the shares when a count is stated; never advice
_COST_ZH = re.compile(
    r"(?:成本价?|买入价|买入成本|持仓成本|持仓均价|均价|进价|入手价)(?:是|为|在|大概|大约|约)?\s*(?P<cost>\d+(?:\.\d+)?)\s*(?:元|块)?|"
    r"(?P<cost2>\d+(?:\.\d+)?)\s*(?:(?:元|块)(?:钱)?(?:一股|每股|1股)?\s*(?:的时候|时)?(?:又|再|还|也|另外)?|"
    # (round 17) a bare price right before the buy verb: "我250买了400股"
    r"(?=买|进的|入手|建仓))"
    r"(?:买|买入|买进|入手|进的|进了|建仓|抄底|上车|加仓|补仓)|"
    # (round 17) "买在9元", "成本在40块", "进在30"
    r"(?:买在|买于|进在|入在)\s*(?P<cost6>\d+(?:\.\d+)?)|"
    # (round 17) the count, then the price, after the buy verb: "买了1000股300元", "买入500股，每股40元"
    r"(?:买|买入|买进|入手|建仓)(?:了|入|进)?\s*(?:\d[\d,，]*|[两二三四五六七八九十百千万]+)\s*(?:股|手)\s*[，,]?\s*"
    r"(?:每股|单价|价格|均价)?(?:是|为|在)?\s*(?P<cost3>\d+(?:\.\d+)?)\s*(?:元|块)(?!\s*(?:卖|抛))|"
    # (round 17) the price before the cost word: "按94元的成本", "我是96元的成本"
    r"(?P<cost4>\d+(?:\.\d+)?)\s*(?:元|块)(?:钱)?\s*(?:的)?\s*(?:持仓)?(?:成本价?|买入价|买入成本|均价|进价)"
)
# (round 15, review-11 K2) several purchases at different costs, each a (cost, count) pair: "40元买了500股…43元又买了
# 500股", "先买500股成本40元，又买300股成本45元", "bought 500 shares at 40 and another 500 at 43"
_LOT_ZH = re.compile(
    r"(?P<cost>\d+(?:\.\d+)?)\s*(?:元|块)(?:钱)?(?:一股|每股)?\s*(?:的(?:价格|价位)?)?\s*(?:的时候|时|那会儿?)?"
    r"(?:又|再|还|也|另外|接着|然后)?"
    # (round 16, review-12 L2) 补了 / 加了 / 又入 / 吃进, and the name between the verb and the count ("30元买入紫金矿业
    # 1000股")
    r"(?:买|买入|买进|入手|建仓|加仓|补仓|抄底|补|加|入|吃进|接)(?:了|入|进|回)?\s*(?:[一-鿿A-Za-z]{2,8}?\s*)?"
    + _HOLDING_COUNT_ZH
    + r"|"
    # (round 17) the name between the verb and the count here too: "先买宁德时代300股成本280元"
    + r"(?:买|买入|买进|入手|建仓|加仓|补仓)(?:了|入|进)?\s*(?:[一-鿿A-Za-z]{2,8}?\s*)?"
    + _HOLDING_COUNT_ZH.replace("(?P<n>", "(?P<n2>").replace("(?P<unit>", "(?P<unit2>")
    + r"[^，,。；;]{0,10}?(?:成本价?|买入价|均价|单价|价格|每股)(?:是|为|在|约)?\s*(?P<cost2>\d+(?:\.\d+)?)|"
    # (round 16) "第一次300股75元，第二次200股85元": a count then its cost, after an ordinal purchase word
    + r"(?:第[一二三四五六七八九十\d]+(?:次|笔|批)|头一次|先|后来|然后|再)[^，,。；;：:]{0,4}?"
    + _HOLDING_COUNT_ZH.replace("(?P<n>", "(?P<n3>").replace("(?P<unit>", "(?P<unit3>")
    + r"\s*(?:的)?(?:成本|价格|均价|每股)?\s*(?P<cost3>\d+(?:\.\d+)?)\s*(?:元|块)"
)
_LOT_EN = re.compile(
    r"(?<![\d.,])(?P<n>\d[\d,]*)\s+(?:more\s+)?(?:(?:[A-Za-z][\w'.&-]*\s+){0,3}?(?:shares?|units?)\s+)?"
    # (round 16, review-12 L2) "1,000 Zijin at 32": a name with no "shares"
    r"(?:of\s+[A-Z][\w'&. -]{0,30}?\s+|(?-i:[A-Z])[\w'&.-]*(?:\s+(?-i:[A-Z])[\w'&.-]*){0,2}\s+)?"
    r"(?:at|for)\s*(?:CNY|RMB|¥)?\s*(?P<cost>\d+(?:\.\d+)?)\b(?!\s*(?:shares|%))",
    re.IGNORECASE,
)
_PNL_ASK_ZH = re.compile(
    r"赚了?(?:多少|几)|亏了?(?:多少|几)|浮盈|浮亏|盈亏|盈利(?:了)?多少|亏损(?:了)?多少|收益(?:率)?(?:是)?多少|"
    r"赚(?:还是|了还是)亏|是赚是亏|赚钱了吗|亏钱了吗|回本|"
    # (round 16, L2) "亏了没有", "赚没赚", "浮盈浮亏"
    r"[赚亏]了没有|赚没赚|亏没亏|有没有[赚亏]"
)
_COST_EN = re.compile(
    r"\bpaid\s*(?:CNY|RMB|¥)?\s*(?P<cost3>\d+(?:\.\d+)?)\s*(?:a share|per share|each|apiece|for)\b|"
    r"\b(?:bought|purchased|paid|got in|got into|bought into|went into|picked up|acquired|added|entered)\b[^.?!]{0,40}?"
    r"\b(?:at|for)\s*(?:CNY|RMB|¥)?\s*"
    r"(?P<cost>\d+(?:\.\d+)?)|"
    r"\b(?:cost(?: basis| price)?|entry(?: price)?|average (?:cost|price)|avg\.? cost|(?:purchase|buy) price)\s*"
    r"(?:is|was|of|at|=|:)?\s*(?:CNY|RMB|¥)?\s*(?P<cost2>\d+(?:\.\d+)?)|"
    # (round 17) "my 33 purchase price", "my CNY 96 entry"
    r"\bmy\s+(?:CNY|RMB|¥)?\s*(?P<cost4>\d+(?:\.\d+)?)\s+(?:purchase price|buy price|cost(?: basis)?|entry(?: price)?|"
    r"average cost)\b",
    re.IGNORECASE,
)
_PNL_ASK_EN = re.compile(
    r"\b(?:profit|loss|gain|p&l|pnl|up|down|made|lost|making|losing|underwater|in the (?:red|green))\b|"
    # (round 17) "Where am I now on the remaining stake?", "how do I stand on what's left?"
    r"\bwhere (?:am i|do i stand)\b|\bhow (?:am i doing|do i stand)\b",
    re.IGNORECASE,
)
# (round 17, review-13 M1) a sale reduces the position: "后来32元卖掉了1000股", "又以320元卖了400股", "卖出500股，卖价
# 45元", "减持了一半", "sold 400 at 320", "trimmed 300 shares for 41 each", "sold half". A lot with a sale verb is
# never a purchase; the realised part is the sold shares × (sale price − average cost), the rest is valued at the close.
_SALE_VERB_ZH = (
    r"(?:卖出|卖掉|卖了|抛出|抛掉|抛了|出掉|出了|减持|减仓|减掉|清掉|止盈|止损|割肉|割掉|割了|卖|抛)(?:了|掉|出)?"
)
_FRACTION_ZH = (
    r"(?P<frac>一半|半仓|三分之一|三分之二|四分之一|四分之三|五分之一|"
    r"一成|两成|二成|三成|四成|五成|六成|七成|八成|九成)"
)
_SALE_ZH = re.compile(
    # the price first: "32元卖掉了1000股", "以320元的价格卖了400股"
    r"(?:以|按|在|用)?\s*(?P<price>\d+(?:\.\d+)?)\s*(?:(?:元|块)(?:钱)?(?:一股|每股)?\s*(?:的(?:价格|价位|位置|点位)?)?\s*"
    r"(?:的时候|时|那会儿?)?(?:又|再|还|也|后来|然后|接着|就)?|"
    # (round 17) "88的位置卖了300股", "昨天280卖了一半": a bare price right before the sale verb
    r"(?:的(?:价格|价位|位置|点位)|的时候|时)?(?=(?:卖|抛|出掉|出了|减持|减仓|清掉|止盈|止损|割)))"
    + _SALE_VERB_ZH
    + r"\s*(?:其中)?(?:的)?\s*(?:"
    + _HOLDING_COUNT_ZH.replace("(?P<n>", "(?P<n1>").replace("(?P<unit>", "(?P<unit1>")
    + "|"
    + _FRACTION_ZH.replace("(?P<frac>", "(?P<frac1>")
    + ")|"
    # the count first: "卖出500股，卖价45元", "卖了400股，价格320元"
    + _SALE_VERB_ZH
    + r"\s*(?:其中)?(?:的)?\s*(?:"
    + _HOLDING_COUNT_ZH.replace("(?P<n>", "(?P<n2>").replace("(?P<unit>", "(?P<unit2>")
    + "|"
    + _FRACTION_ZH.replace("(?P<frac>", "(?P<frac2>")
    + r")"
    r"(?:[^，,。；;？?]{0,6}?(?:[，,]\s*)?(?:卖出价|卖价|成交价|价格|均价|每股|单价)?(?:是|为|在)?\s*(?:以|按)?\s*"
    r"(?P<price2>\d+(?:\.\d+)?)\s*(?:元|块))?"
)
_FRACTION_VALUE = {
    "一半": 0.5, "半仓": 0.5, "三分之一": 1 / 3, "三分之二": 2 / 3, "四分之一": 0.25, "四分之三": 0.75, "五分之一": 0.2,
    "一成": 0.1, "两成": 0.2, "二成": 0.2, "三成": 0.3, "四成": 0.4, "五成": 0.5, "六成": 0.6, "七成": 0.7, "八成": 0.8,
    "九成": 0.9, "half": 0.5, "a third": 1 / 3, "one third": 1 / 3, "a quarter": 0.25, "one quarter": 0.25,
}  # fmt: skip
_SALE_EN = re.compile(
    r"\b(?:sold|sell|selling|trimmed|trim|offloaded|unloaded|dumped|cut|reduced|got rid of|took profits? on|cashed out)"
    r"\s+(?:off\s+|out\s+)?(?:(?P<n>\d[\d,]*)|(?P<frac>half|a third|one third|a quarter|one quarter))"
    r"(?:\s+(?:of\s+(?:them|those|it|the(?:se)? shares|my shares|the position|"
    # (round 17) "sold 300 of my 900 BYD shares at 90": the held count inside the sale
    r"my\s+(?P<held>\d[\d,]*)(?:\s+(?!at\b|for\b)[A-Za-z][\w&.'-]*){0,3}?\s+shares?)|shares?|units?|lots?)"
    r"(?:\s+of\s+[A-Z][\w&.'-]*)?)?"
    r"(?:\s+(?:at|for)\s*(?:CNY|RMB|¥)?\s*(?P<price>\d+(?:\.\d+)?)\b(?!\s*%)"
    r"(?:\s*(?:a share|per share|each|apiece))?)?",
    re.IGNORECASE,
)


# (round 17, review-13 M5) the rise needed from the close back to a stated cost: "得涨百分之多少才能回本", "回本还要涨
# 多少", "涨几个点能解套", "How much does it need to rise for me to break even?", "what gain gets me back to my cost"
_RISE_TO_COST = re.compile(
    r"(?:涨|上涨|反弹|回升|拉升)[^，。？?；;]{0,10}?(?:多少|几|百分之)[^，。？?；;]{0,8}?(?:才能|才可以|才会|才|能|可以|就能|就)?"
    r"(?:回本|回得了本|解套|保本|回(?:得)?到成本|涨回成本|回到买入价|持平)|"
    r"(?:回本|解套|保本|回到成本价?|回到买入价|涨回成本价?)[^，。？?；;]{0,8}?(?:需要|要|得|还要|还得|还需|需|至少)?"
    r"(?:再)?(?:涨|上涨|反弹)(?:多少|几|百分之)|"
    r"(?:离|距离?)(?:回本|成本价?|保本)[^，。？?；;]{0,6}?(?:还差|差)(?:多少|几)|"
    r"(?:回本|解套|保本|回到成本价?|涨回成本价?)[^，。？?；;]{0,6}?(?:需要|要|得|还要|还得|还需|需)"
    r"(?:多少|几)个?(?:点|百分点)|"
    r"\b(?:rise|gain|climb|rally|go up|increase|recover)\b[^.?!]{0,40}?\b(?:to )?"
    r"(?:break[- ]?even|get back to (?:my |the )?(?:CNY\s*|RMB\s*|¥\s*)?(?:\d+(?:\.\d+)?\s+)?"
    r"(?:cost|entry|purchase price|buy price)|(?:be|get) back above (?:my )?(?:cost|entry))\b|"
    r"\b(?:what|how much)\b[^.?!]{0,30}?\b(?:rise|gain|increase|move)\b[^.?!]{0,30}?\bbreak[- ]?even\b|"
    r"\bbreak[- ]?even\b[^.?!]{0,30}?\b(?:need|require|take)s?\b[^.?!]{0,20}?\b(?:rise|gain|increase|%|percent)",
    re.IGNORECASE,
)


def _chinese(text: str) -> bool:
    """(round 17) Whether a question is written in Chinese: an English question that ends with a Chinese name added by
    a follow-up rewrite ("… on the rest? (紫金矿业)") is English."""
    cjk = len(re.findall(r"[一-鿿]", text or ""))
    return cjk > 0 and cjk * 3 >= len(re.findall(r"[A-Za-z]", text or ""))


def stated_sales(query: str) -> list[dict[str, Any]]:
    """(round 17, review-13 M1) Every sale the question states, in order: ``{shares, price, fraction, start, end}``
    (``shares`` is ``None`` for a fraction of the position, ``price`` ``None`` when no sale price is given)."""
    text = query or ""
    zh = _chinese(text)
    found: list[dict[str, Any]] = []
    for match in (_SALE_ZH if zh else _SALE_EN).finditer(text):
        groups = match.groupdict()
        raw = groups.get("n") or groups.get("n1") or groups.get("n2")
        frac = groups.get("frac") or groups.get("frac1") or groups.get("frac2")
        price = groups.get("price") or groups.get("price2")
        shares = None
        if raw:
            shares = count_value(raw)
            in_lots = (groups.get("unit1") or groups.get("unit2") or "") == "手"
            if groups.get("n") and re.match(r"\s*lots?\b", text[match.end("n") :], re.IGNORECASE):
                in_lots = True
            shares = shares * 100 if in_lots else shares
        if not shares and not frac:
            continue
        found.append(
            {
                "shares": shares,
                "held": count_value(groups["held"]) if groups.get("held") else None,
                "fraction": _FRACTION_VALUE.get((frac or "").lower()) if frac else None,
                "price": float(price) if price else None,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return found


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    for start, end in spans:
        text = f"{text[:start]}{' ' * (end - start)}{text[end:]}"
    return text


def cost_basis_request(query: str) -> dict[str, Any] | None:
    """``{cost, shares, lots}`` when the question asks the gain or loss against a cost price it states (``shares`` is
    ``None`` when no count is stated), else ``None``."""
    original = query or ""
    zh = _chinese(original)
    sales = stated_sales(original)
    # (round 17, review-13 M1) a sale is never read as a cost, a purchase lot or the held count
    text = _blank(original, [(sale["start"], sale["end"]) for sale in sales])
    found = _COST_ZH.search(text) if zh else _COST_EN.search(text)
    rise = _RISE_TO_COST.search(original)
    asked = (_PNL_ASK_ZH.search(original) if zh else _PNL_ASK_EN.search(original)) or rise
    if found is None and asked is not None and len(detail := cost_lots(text)) >= 2:
        # (round 16) lots written as count then cost ("第一次300股75元，第二次200股85元"): no cost word or buy verb
        request = {"cost": detail[0]["cost"], "shares": sum(lot["shares"] for lot in detail), "lots": 0,
                   "lots_detail": detail}  # fmt: skip
        return _with_sales(request, sales)
    if found is None and asked is not None and zh and _BUY_VERB_ZH.search(text):
        found = _PER_SHARE_PRICE_ZH.search(text)
    if found is None or asked is None:
        return None
    cost = float(next(value for value in found.groupdict().values() if value))
    if cost <= 0:
        return None
    shares, lots = None, 0
    # the count is read outside the cost ("paid 50 a share for 1,000 shares": 1,000, not 50)
    group = next(name for name, value in found.groupdict().items() if value)
    start, end = found.span(group)
    end = end + len(re.match(r"\s*(?:a share|per share|each|apiece)?", text[end:]).group(0))
    rest = f"{text[:start]}{' ' * (end - start)}{text[end:]}"
    counted = re.search(_HOLDING_COUNT_ZH, rest) if zh else _EN_POSITION.search(rest)
    if counted is not None:
        count = count_value(counted.group("n"))
        unit = counted.group("unit").lower()
        lots = count if unit in {"手", "lot", "lots"} else 0
        shares = count * 100 if lots else count
    request: dict[str, Any] = {"cost": cost, "shares": shares if shares and shares > 1 else None, "lots": lots}
    detail = cost_lots(text)
    if len(detail) >= 2:
        # (round 15, review-11 K2) the gain over every lot, never the first lot alone
        request["lots_detail"] = detail
        request["shares"] = sum(lot["shares"] for lot in detail)
        request["lots"] = 0
    elif sales and request["shares"] is None:
        # (round 17) "Bought 1,000 CATL at 300, sold 400 at 320": the one purchase lot written without "shares"
        single = cost_lots(text, minimum=1)
        held = None if zh else _EN_HELD_COUNT.search(text)
        if len(single) == 1 and single[0]["cost"] == cost:
            request["shares"] = single[0]["shares"]
        elif held is not None:
            request["shares"] = count_value(held.group("n"))  # "I hold 600 ICBC bought at 7.2; I sold 200 at 8.5"
    if rise is not None and not sales:
        request["rise_to_cost"] = True  # (round 17, review-13 M5) the rise back to the cost, not the loss in percent
    return _with_sales(request, sales)


_EN_HELD_COUNT = re.compile(
    r"\b(?:hold|own|have|had|holding|owned|got|bought|purchased|picked up)\s+(?P<n>\d[\d,]*)\b"
    r"(?!\.\d|\s*(?:%|percent))",
    re.IGNORECASE,
)
# (round 17) a price per share stated next to a purchase: "我买了3手比亚迪，每股78元"
_PER_SHARE_PRICE_ZH = re.compile(r"(?:每股|单价|成交价|买价)(?:是|为|在)?\s*(?P<cost>\d+(?:\.\d+)?)\s*(?:元|块)")
_BUY_VERB_ZH = re.compile(r"买了|买入|买进|入手|建仓|(?<!要)买(?=\d|[两二三四五六七八九十]|了)")


def _with_sales(request: dict[str, Any], sales: list[dict[str, Any]]) -> dict[str, Any]:
    """(round 17, review-13 M1) The position after the stated sales: ``bought`` (the purchased shares), ``sales``
    (each with its shares; a fraction is taken of the purchased shares), ``shares`` (what remains; ``None`` when the
    purchased count is not stated) and ``average_cost`` (over every purchase lot). A sale of more than was bought
    sets ``oversold``."""
    if not sales:
        return request
    bought = request.get("shares") or next((sale["held"] for sale in sales if sale.get("held")), None)
    lots = request.get("lots_detail") or []
    paid = sum(float(lot["shares"]) * float(lot["cost"]) for lot in lots)
    average = round(paid / bought, 4) if lots and bought else float(request["cost"])
    resolved = []
    for sale in sales:
        shares = sale["shares"]
        if shares is None and sale.get("fraction") and bought:
            shares = round(bought * float(sale["fraction"]))
        if not shares:
            return request  # "卖了一半" without the purchased count: nothing to take half of
        resolved.append({"shares": int(shares), "price": sale.get("price")})
    sold = sum(sale["shares"] for sale in resolved)
    request = {**request, "sales": resolved, "bought": bought, "average_cost": average, "lots": 0}
    if bought is not None:
        request["shares"] = bought - sold
        if bought - sold < 0:
            request["oversold"] = True
    return request


def cost_lots(query: str, *, minimum: int = 2) -> list[dict[str, Any]]:
    """(round 15, review-11 K2) ``[{cost, shares}]`` for every purchase the question states with its own cost
    (two or more, or ``minimum``), in order; empty otherwise. (round 17, review-13 M1) A lot with a sale verb is a
    sale, never a purchase."""
    raw = query or ""
    text = _blank(raw, [(sale["start"], sale["end"]) for sale in stated_sales(raw)])
    found: list[dict[str, Any]] = []
    pattern = _LOT_ZH if re.search(r"[一-鿿]", text) else _LOT_EN
    for match in pattern.finditer(text):
        groups = match.groupdict()
        raw_cost = groups.get("cost") or groups.get("cost2") or groups.get("cost3")
        raw_count = groups.get("n") or groups.get("n2") or groups.get("n3")
        if not raw_cost or not raw_count:
            continue
        count = count_value(raw_count)
        if (groups.get("unit") or groups.get("unit2") or groups.get("unit3") or "") == "手":
            count *= 100
        cost = float(raw_cost)
        if count > 1 and cost > 0:
            found.append({"cost": cost, "shares": count})
    return found if len(found) >= minimum else []


def holding_purchase(query: str) -> bool:
    """Whether a holding-value question asks the cost of buying the shares ("买三手要多少钱") rather than the value of
    shares already held."""
    return _purchase_match(query or "") is not None and not (
        _HOLDING_VALUE_ZH.search(query or "") or _HOLDING_VALUE_EN.search(query or "")
    )


# A fund holding is counted in units (份), a stock holding in shares (股).
_FUND_UNITS = re.compile(r"\d\s*份|[两二三四五六七八九十百千万]\s*份|\bunits?\b", re.IGNORECASE)
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def count_value(raw: str) -> int:
    """A stated count: "2,000", "两千", "1.5万" (15,000), "3万" (30,000), "2千" (2,000)."""
    raw = (raw or "").replace(",", "").replace("，", "").replace(" ", "")
    if raw.isdigit():
        return int(raw)
    thousands = re.fullmatch(r"(\d+(?:\.\d+)?)(?:k|thousand)", raw, re.IGNORECASE)
    if thousands:
        return round(float(thousands.group(1)) * 1000)
    if re.fullmatch(r"\d+(?:\.\d+)?(?:[万千百十]\d*)+", raw):
        # (round 16, review-12 L3) "1万2千" = 12,000, "3千5百" = 3,500, "1.2万" = 12,000, "1万2" = 12,000 (a bare
        # trailing digit counts in the next lower unit)
        lower = {"万": 1000, "千": 100, "百": 10, "十": 1}
        total, last = 0.0, None
        for number, unit in re.findall(r"(\d+(?:\.\d+)?)([万千百十]?)", raw):
            if unit:
                total += float(number) * _CN_UNIT[unit]
                last = unit
            elif last is not None and len(number) == 1:
                total += float(number) * lower[last]
            else:
                total += float(number)
        return round(total)
    return _cn_integer(raw)


def _cn_integer(text: str) -> int:
    total, section, digit = 0, 0, 0
    for char in text:
        if char in _CN_DIGIT:
            digit = _CN_DIGIT[char]
        elif char == "万":
            total += (section + digit) * 10000
            section, digit = 0, 0
        else:
            section += (digit or 1) * _CN_UNIT[char]
            digit = 0
    return total + section + digit


def holding_value_request(query: str) -> tuple[int, tuple[int, int]] | None:
    """``(shares, span)`` when the question asks what a stated number of shares is worth, else ``None``."""
    text = query or ""
    match = _HOLDING_VALUE_ZH.search(text) or _HOLDING_VALUE_EN.search(text)
    if match is None:
        purchase = _purchase_match(text)
        if purchase is None:
            return None
        return (purchase[0], purchase[3]) if purchase[0] > 1 else None
    groups = match.groupdict()
    raw = next((groups[key] for key in ("n", "n2", "n3", "n4", "n5") if groups.get(key)), "")
    shares = count_value(raw)
    if (groups.get("unit") or groups.get("unit5")) == "手":
        shares *= 100  # a board lot is 100 shares
    return (shares, match.span()) if shares > 1 else None


def holding_lots(query: str) -> int | None:
    """The number of lots (手) a holding is stated in ("3手茅台" → 3), else ``None``."""
    match = _HOLDING_VALUE_ZH.search(query or "")
    if match is None:
        purchase = _purchase_match(query or "")
        return purchase[1] if purchase and purchase[1] else None
    groups = match.groupdict()
    if (groups.get("unit") or groups.get("unit5")) != "手":
        return None
    return count_value(groups.get("n") or groups.get("n5") or "")


def in_fund_units(query: str) -> bool:
    """Whether a holding is stated in fund units ("两万份", "4000 units"), not shares."""
    return bool(_FUND_UNITS.search(query or ""))


_HOLDING_COUNT = re.compile(_HOLDING_COUNT_ZH + "|" + _EN_COUNT.format(name="n2"), re.IGNORECASE)


def stated_holding_count(text: str) -> int | None:
    """A number of shares or units stated without a value question ("同样300股", "如果是500股", "800 shares")."""
    match = _HOLDING_COUNT.search(text or "")
    if match is None:
        return None
    count = count_value(match.group("n") or match.group("n2") or "")
    if match.group("unit") == "手":
        count *= 100
    return count if count > 1 else None


_EN_POSITION = re.compile(
    r"(?<![\d.,])(?P<n>\d[\d,]*)\s+(?:(?!billion|million|thousand|bn|mn)[A-Za-z][\w'.-]*\s+){0,3}?"
    r"(?P<unit>shares?|units?|lots?)\b",
    re.IGNORECASE,
)


_EN_BARE_POSITION = re.compile(
    r"(?:,|\band\b|\bplus\b|&)\s*(?:another\s+|also\s+)?(?P<n>\d[\d,]*)\s+(?:of\s+|in\s+)?(?=(?-i:[A-Z]))|"
    # (round 16) a count and a name before the next item of the list ("1,500 CMB and 2,000 ICBC shares")
    r"(?<![\d.,])(?P<n2>\d[\d,]*)\s+(?=(?-i:[A-Z])[\w&.-]*(?:\s+(?-i:[A-Z])[\w&.-]*){0,2}\s*(?:,|\band\b|&))",
    re.IGNORECASE,
)


def holding_positions(query: str) -> list[dict[str, Any]]:
    """(round 13, review-9 I5) Every count in a holding-value question that states two or more positions ("我有200股
    茅台和3手五粮液，一共值多少", "300 Moutai shares and 2 lots of Ping An"): ``{shares, lots, units, start, end}`` in
    order; empty for a single position (``holding_value_request`` covers it)."""
    text = query or ""
    if holding_value_request(text) is None:
        return []
    found: list[dict[str, Any]] = []
    for match in re.finditer(_HOLDING_COUNT_ZH, text):
        count = count_value(match.group("n"))
        lots = count if match.group("unit") == "手" else 0
        found.append(
            {
                "shares": count * 100 if lots else count,
                "lots": lots,
                "units": match.group("unit") == "份",
                "start": match.start(),
                "end": match.end(),
            }
        )
    if not found:
        for match in _EN_POSITION.finditer(text):
            count = int(match.group("n").replace(",", ""))
            unit = match.group("unit").lower()
            lots = count if unit.startswith("lot") else 0
            found.append(
                {
                    "shares": count * 100 if lots else count,
                    "lots": lots,
                    "units": unit.startswith("unit"),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        if found:
            # (round 16, review-12 L3) "800 shares of Midea and 1,200 of CMB": a further count in the same list without
            # its unit takes the unit of the position before it
            taken = [(item["start"], item["end"]) for item in found]
            for match in _EN_BARE_POSITION.finditer(text):
                group = "n" if match.group("n") else "n2"
                if any(start <= match.start(group) < end for start, end in taken):
                    continue
                before = [item for item in found if item["end"] <= match.start()]
                after = [item for item in found if item["start"] >= match.end(group)]
                if not (before or (group == "n2" and after)):
                    continue
                count = int(match.group(group).replace(",", ""))
                unit = before[-1] if before else after[0]
                found.append(
                    {
                        "shares": count * 100 if unit["lots"] else count,
                        "lots": count if unit["lots"] else 0,
                        "units": unit["units"],
                        "start": match.start(group),
                        "end": match.end(group),
                    }
                )
            found.sort(key=lambda item: item["start"])
    found = [item for item in found if item["shares"] > 1]
    return found if len(found) >= 2 else []


def name_forms(name: str, *extra: str) -> list[str]:
    """How a target may be written: its name, the last part of a long Chinese name (贵州茅台 → 茅台), English
    names and aliases, and (round 14, review-10 J5) its Chinese short names from the alias tables (招商银行 → 招行,
    工商银行 → 工行); longest first."""
    from .names import english_aliases, english_name

    forms = {str(item) for item in (name, *extra) if item}
    if len(name) >= 4 and re.fullmatch(r"[一-鿿]+", name) and name[2:] not in GENERIC_NAME_TAILS:
        forms.add(name[2:])
    forms.update(alias for alias in (english_name(name), *english_aliases(name)) if alias)
    forms.update(_chinese_short_names().get(name, ()))
    return sorted(forms, key=len, reverse=True)


@lru_cache(maxsize=1)
def _chinese_short_names() -> dict[str, tuple[str, ...]]:
    """Canonical name → its colloquial / common Chinese aliases (招行, 工行, 茅台), from the runtime and seed alias
    tables and the synonym dictionary."""
    from ..data_loader import load_aliases, load_entities, load_seed_aliases, load_seed_entities, load_synonyms

    found: dict[str, set[str]] = {}
    for entities, aliases in ((load_entities(), load_aliases()), (load_seed_entities(), load_seed_aliases())):
        names = {str(row.get("entity_id")): str(row.get("canonical_name") or "") for row in entities}
        for row in aliases:
            alias = str(row.get("alias_text") or "")
            if row.get("alias_type") not in {"colloquial_alias", "common_alias"} or not re.fullmatch(
                r"[一-鿿]{2,6}", alias
            ):
                continue
            entity_id = str(row.get("entity_id") or "")
            entity_id = str(int(entity_id)) if entity_id.isdigit() else entity_id
            canonical = names.get(entity_id)
            if canonical and alias != canonical:
                found.setdefault(canonical, set()).add(alias)
    for alias, canonical in (load_synonyms().get("alias") or {}).items():
        if re.fullmatch(r"[一-鿿]{2,6}", str(alias)) and alias != canonical:
            found.setdefault(str(canonical), set()).add(str(alias))
    return {name: tuple(sorted(items, key=len, reverse=True)) for name, items in found.items()}


def bind_positions(text: str, positions: list[dict[str, Any]], candidates: dict[str, list[str]]) -> list[str | None]:
    """The candidate key each holding position names: the name right after its unit ("200股茅台", "2 lots of Ping
    An") or inside it ("300 Moutai shares"), else right before it ("茅台200股"), else the first name before the next
    position; ``None`` for a bare count ("我还有5手")."""
    lowered = text.lower()

    def starts(segment: str) -> str | None:
        for key, forms in candidates.items():
            if any(segment.startswith(form.lower()) for form in forms):
                return key
        return None

    def ends(segment: str) -> str | None:
        for key, forms in candidates.items():
            if any(segment.endswith(form.lower()) for form in forms):
                return key
        return None

    bound: list[str | None] = []
    for index, position in enumerate(positions):
        stop = positions[index + 1]["start"] if index + 1 < len(positions) else len(text)
        begin = positions[index - 1]["end"] if index else 0
        inside = lowered[position["start"] : position["end"]]
        after = re.sub(r"^(?:\s|的|of\s)+", "", lowered[position["end"] : stop])
        before = re.sub(r"(?:\s|有|持有|买了|还有|，|,)+$", "", lowered[begin : position["start"]])
        key = next((k for k, forms in candidates.items() if any(form.lower() in inside for form in forms)), None)
        key = key or starts(after) or ends(before)
        if key is None:
            hits = [
                (after.find(form.lower()), k)
                for k, forms in candidates.items()
                for form in forms
                if form.lower() in re.split(r"[，,。；;]", after)[0]
            ]
            key = min(hits)[1] if hits else None
        bound.append(key)
    return bound


# (round 17, review-13 M16) A holding stated with no question ("我持有比亚迪600股", "记一下：我有招行1000股",
# "I hold 500 CATL shares."): before round 17 the NLU read 持有 as hold advice and the answer opened with the advice
# prefix and "问题包含投资建议或预测属性". It is a statement of fact; the answer gives the close without that prefix.
_POSSESSION_ZH = re.compile(
    r"(?:我|本人|俺|咱)?(?:现在|目前|手里|手上|账户里|账户上|仓里)?(?:还)?(?:有|持有|持仓|拿着|拿了|握有|买了|买入了|入了|建仓了|囤了)"
)
_POSSESSION_EN = re.compile(
    r"\b(?:i|we)\s+(?:currently\s+|now\s+|also\s+)?(?:have|hold|own|bought|am\s+holding|'m\s+holding|picked\s+up)\b",
    re.IGNORECASE,
)
_A_QUESTION = re.compile(
    r"[？?]|吗|呢|么|吧|多少|几|怎么|怎样|如何|什么|啥|哪|谁|是否|能不能|要不要|该不该|值不值|行不行|可不可以|会不会|有没有|"
    r"帮我|请|算一下|看看|分析|评价|建议|\b(?:what|how|which|why|when|should|could|would|can|is|are|do|does|"
    r"worth|tell|show|please|help)\b",
    re.IGNORECASE,
)


def holding_statement(query: str) -> bool:
    """(round 17, review-13 M16) The message only states a holding (a possession verb and a share / lot / unit count)
    and asks nothing."""
    text = (query or "").strip()
    if not text or stated_holding_count(text) is None:
        return False
    if not (_POSSESSION_ZH.search(text) or _POSSESSION_EN.search(text)):
        return False
    return not _A_QUESTION.search(text)


def without_holding_value(query: str) -> str:
    """The question with a holding-value request blanked out, for the fair-value and judgment checks."""
    found = holding_value_request(query)
    if found is None:
        return query
    start, end = found[1]
    return f"{query[:start]} {query[end:]}"


def requested_metrics(query: str) -> list[Metric]:
    """Metrics beyond the standard snapshot (PE/PB/ROE/revenue/net profit) that the question asks for."""
    text = query or ""
    found: list[Metric] = []
    consumed: list[tuple[int, int]] = []
    for metric in METRICS:
        for match in metric.pattern.finditer(text):
            if any(start <= match.start() < end for start, end in consumed):
                continue
            consumed.append(match.span())
            if metric not in found:
                found.append(metric)
    return found


_YEAR = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*(?:年|财年|年度)|"
    r"\b(?:in|for|during|of|fy|fiscal|financial year)\s*((?:19|20)\d{2})\b|"
    r"\b((?:19|20)\d{2})\s+(?:revenue|sales|net|profit|earnings|results|annual|full[- ]year|report|financials?|"
    r"dividends?|roe|eps|p/?e|p/?b)",
    re.IGNORECASE,
)
_DATE_LIKE = re.compile(r"(?:19|20)\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?")
_FORWARD = re.compile(r"预计|预测|将会|会不会|目标|展望|明年|forecast|expect|will\b|outlook|target", re.IGNORECASE)


def requested_years(query: str) -> list[int]:
    """Calendar years a question explicitly asks about ("2019年营收", "2023 revenue")."""
    text = _DATE_LIKE.sub(" ", query or "")
    years = []
    for match in _YEAR.finditer(text):
        year = int(next(group for group in match.groups() if group))
        if 1990 <= year <= 2100 and year not in years:
            years.append(year)
    return years


def _year_of(value: Any) -> int | None:
    match = re.match(r"\s*((?:19|20)\d{2})", str(value or ""))
    return int(match.group(1)) if match else None


# (round 17, review-13 M19) A conversion into another currency ("比亚迪的市值换算成美元是多少", "BYD's market cap in US
# dollars"): FinSight's figures are in CNY and its data has no exchange rate. Before round 17 the answer gave the CNY
# figure and said nothing about the conversion; it now says that the rate is missing and that the figures are in CNY.
_FX_ZH = r"(?P<zh>美元|美金|港元|港币|欧元|日元|英镑|澳元|加元|新加坡元|新元|韩元|瑞郎|瑞士法郎|卢布|外币)"
_FX_EN = (
    r"(?P<en>US\s*dollars?|U\.S\.\s*dollars?|USD|US\$|dollars?|HKD|HK\$|Hong\s+Kong\s+dollars?|euros?|EUR|yen|JPY|"
    r"pounds?(?:\s+sterling)?|GBP|sterling|Singapore\s+dollars?|SGD|Australian\s+dollars?|AUD|foreign\s+currency)"
)
_FX_CONVERSION = re.compile(
    r"(?:换算|折算|折合|兑换|转换|转化|换成|折成|算成|合成|计成)(?:成|为|到|一下)?\s*(?:多少)?\s*" + _FX_ZH + r"|"
    r"(?:以|用|按)\s*" + _FX_ZH.replace("?P<zh>", "?P<zh2>") + r"\s*(?:计|计价|计算|来算|算|表示)|"
    r"(?:约合|合|相当于|等于)\s*(?:多少|几)\s*(?:亿|万|千)?\s*" + _FX_ZH.replace("?P<zh>", "?P<zh3>") + r"|"
    r"(?:市值|营收|收入|净利润|利润|股价|价格|收盘价|总资产|分红)[^，。？?！!；;]{0,8}?(?:是|有|为)?(?:多少|几)\s*(?:亿|万|千)?\s*"
    + _FX_ZH.replace("?P<zh>", "?P<zh4>")
    + r"|"
    r"\bconvert(?:ed|ing)?\b[^.?!]{0,60}?\b(?:to|into)\s+" + _FX_EN + r"\b|"
    r"\b(?:in|into)\s+(?:terms\s+of\s+)?" + _FX_EN.replace("?P<en>", "?P<en2>") + r"(?:\s+terms)?\b|"
    r"\b(?:how\s+many|how\s+much\s+in)\s+" + _FX_EN.replace("?P<en>", "?P<en3>") + r"\b|"
    r"\b(?:USD|dollar|euro|yen|sterling)[- ](?:terms|equivalent|value|denominated)\b",
    re.IGNORECASE,
)
_FX_NAME_EN = {"美元": "US dollars", "美金": "US dollars", "港元": "Hong Kong dollars", "港币": "Hong Kong dollars",
               "欧元": "euros", "日元": "yen", "英镑": "pounds sterling"}  # fmt: skip


def _fx_english(name: str) -> str:
    """ "USD", "dollar", "HK$" -> the currency's English name ("" for a currency not listed)."""
    lowered = re.sub(r"\s+", " ", name.lower()).strip()
    if any(word in lowered for word in ("hk", "hong kong")):
        return "Hong Kong dollars"
    if "singapore" in lowered or lowered == "sgd":
        return "Singapore dollars"
    if "australian" in lowered or lowered == "aud":
        return "Australian dollars"
    if "dollar" in lowered or lowered in {"usd", "us$"}:
        return "US dollars"
    if "euro" in lowered or lowered == "eur":
        return "euros"
    if lowered in {"yen", "jpy"}:
        return "yen"
    if "pound" in lowered or "sterling" in lowered or lowered == "gbp":
        return "pounds sterling"
    return ""


def currency_conversion_note(query: str, *, zh: bool) -> str | None:
    """(round 17, review-13 M19) The sentence for a question that asks for a CNY figure in another currency, else
    ``None``. A Hong Kong-dollar price of a company's H shares is a foreign line (``foreign_line_spans``), refused
    before this applies."""
    match = _FX_CONVERSION.search(query or "")
    if match is None:
        return None
    zh_name = next((match.group(name) for name in ("zh", "zh2", "zh3", "zh4") if match.group(name)), None)
    en_name = next((match.group(name) for name in ("en", "en2", "en3") if match.group(name)), None)
    english = _fx_english(en_name or "")
    if zh:
        currency = zh_name or next((key for key, value in _FX_NAME_EN.items() if value == english), "外币")
        return f"当前数据中没有汇率，无法把人民币金额换算成{currency}；以下数值均以人民币计。"
    currency = _FX_NAME_EN.get(zh_name or "", "") or english or "another currency"
    return (
        f"The data has no exchange rate, so the CNY figures cannot be converted to {currency}; the figures below are "
        "in CNY."
    )


def coverage_gaps(
    query: str, tool_log: list[dict[str, Any]], *, zh: bool, names: dict[str, str] | None = None
) -> list[str]:
    """Sentences stating which requested metrics or periods the retrieved data does not contain.

    Only checked against tools that returned data: a failed tool is already reported as missing.
    """
    names = names or {}
    sentences: list[str] = []
    # (round 12, H4) "持有5000份…市值多少" asks for the holding's value, not the company's market cap
    wanted = requested_metrics(without_holding_value(query))
    if _MACRO_GROWTH.search(query or ""):
        # "M2增速" / "M2 growth": the macro indicator is itself the growth rate, not a company metric.
        wanted = [metric for metric in wanted if metric.key != "growth"]
    years = requested_years(query)
    forward = bool(_FORWARD.search(query or ""))
    fundamentals = [entry for entry in tool_log if entry.get("tool") == "get_fundamentals" and entry.get("ok")]
    asked_fundamentals = any(entry.get("tool") == "get_fundamentals" for entry in tool_log)
    for entry in fundamentals:
        data = entry.get("data") or {}
        metrics = data.get("metrics") or {}
        name = str(data.get("name") or names.get(str(data.get("symbol"))) or data.get("symbol") or "")
        missing = [
            metric
            for metric in wanted
            if not any(metrics.get(field) is not None for field in metric.fields) and not metric.derivable(metrics)
        ]
        for metric in [metric for metric in missing if metric.needs_zh]:
            # "茅台的PEG": say what the metric needs and that it is missing, not just "no PEG data"
            missing.remove(metric)
            sentences.append(
                f"无法计算{name}的{metric.zh}：{metric.needs_zh}；以下只列出可得的指标。"
                if zh
                else f"{name}'s {metric.en} cannot be computed: {metric.needs_en}; only the available metrics are "
                "listed below."
            )
        if missing:
            labels = ("、".join(metric.zh for metric in missing)) if zh else ", ".join(metric.en for metric in missing)
            sentences.append(
                f"当前数据源没有{name}的{labels}数据，无法回答这一项；以下只列出可得的指标。"
                if zh
                else f"The current data sources do not include {name}'s {labels}, so that part cannot be answered; "
                "only the available metrics are listed below."
            )
        period = str(data.get("report_date") or "")
        period_year = _year_of(period)
        quarter = requested_quarter(query)
        if quarter and _month_of(period) and _month_of(period) != quarter[1] and not forward:
            # "今年一季度的净利润" with annual statements only: the quarter is not in the data.
            sentences.append(
                f"当前数据中没有所问的{quarter[0]}数据：{name}可得的财务数据报告期为 {period}，"
                "以下数字均属于该报告期，不是所问季度的。"
                if zh
                else f"Data for the requested period ({quarter[0]}) is not available: the latest financial statements "
                f"for {name} are for the period {period}, and the figures below are for that period."
            )
        if years and period_year and period_year not in years and not forward:
            asked = _years_text(years, zh=zh)
            sentences.append(
                f"当前数据中没有所问的{asked}数据：{name}可得的财务数据报告期为 {period}，"
                f"以下数字均属于该报告期，不是{asked}的。"
                if zh
                else f"Data for {asked} is not available: the latest financial statements for {name} are for the "
                f"period {period}, and the figures below are for that period, not {asked}."
            )
    if wanted and not asked_fundamentals:
        labels = ("、".join(metric.zh for metric in wanted)) if zh else ", ".join(metric.en for metric in wanted)
        if any(entry.get("ok") for entry in tool_log):
            sentences.append(
                f"本次检索的证据中没有所问的{labels}数据。"
                if zh
                else f"The retrieved evidence does not include the requested {labels}."
            )
    if years and not fundamentals and not forward:
        for entry in tool_log:
            if entry.get("tool") != "get_price_history" or not entry.get("ok"):
                continue
            data = entry.get("data") or {}
            as_of_year = _year_of(data.get("as_of"))
            # (round 15, review-11 K8) a year the payload's dated closes reach (the previous year's last close, the
            # 52-week window) is not missing
            from .composer import dated_closes

            present = {_year_of(day) for day in dated_closes(data)}
            if as_of_year and as_of_year not in years and min(years) < as_of_year and not set(years) <= present:
                name = str(data.get("name") or data.get("symbol") or "")
                asked = _years_text(years, zh=zh)
                sentences.append(
                    f"当前数据中没有{name}{asked}的行情，只有截至 {data.get('as_of')} 的最新数据。"
                    if zh
                    else f"There is no {asked} market data for {name}; only the latest data (as of "
                    f"{data.get('as_of')}) is available."
                )
    return list(dict.fromkeys(sentences))


_MACRO_GROWTH = re.compile(r"(?<![A-Za-z])m[12](?![A-Za-z0-9])|gdp|cpi|ppi|社融|货币供应|money supply", re.I)
# Macro indicators a question can name, and the indicator codes that answer them.
_MACRO_REQUESTS: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    ("CPI", re.compile(r"cpi|居民消费价格", re.I), ("CPI",)),
    ("PPI", re.compile(r"ppi|工业生产者出厂价格", re.I), ("PPI",)),
    ("PMI", re.compile(r"pmi|采购经理", re.I), ("PMI",)),
    ("M2", re.compile(r"(?<![A-Za-z])m2(?![A-Za-z0-9])|money supply|货币供应", re.I), ("M2",)),
    ("LPR", re.compile(r"lpr|贷款市场报价利率|loan prime rate", re.I), ("LPR",)),
    ("GDP", re.compile(r"gdp|国内生产总值", re.I), ("GDP",)),
    ("社融", re.compile(r"社融|社会融资|total social financing", re.I), ("TSF", "SOCIAL_FINANCING")),
    ("10Y", re.compile(r"国债|\bcgb\b|government bond|treasury", re.I), ("10Y",)),
)


def macro_gaps(query: str, tool_log: list[dict[str, Any]], *, zh: bool) -> list[str]:
    """Macro indicators named in the question that the macro tool did not return ("1-year LPR" with no LPR data)."""
    entries = [entry for entry in tool_log if entry.get("tool") == "get_macro_indicators" and entry.get("ok")]
    if not entries:
        return []
    codes = [
        str(indicator.get("code") or "").upper()
        for entry in entries
        for indicator in (entry.get("data") or {}).get("indicators") or []
        if indicator.get("value") is not None
    ]
    missing = [
        label
        for label, pattern, keys in _MACRO_REQUESTS
        if pattern.search(query or "") and not any(key in code for key in keys for code in codes)
    ]
    if not missing:
        return []
    labels = "、".join(missing) if zh else ", ".join(missing)
    return [
        f"当前数据源没有{labels}的数据，无法回答这一项；以下只列出可得的宏观指标。"
        if zh
        else f"The current data sources do not include {labels}, so that part cannot be answered; only the "
        "available macro indicators are listed below."
    ]


# Foreign central banks and US macro data ("美联储加息对A股有什么影响", "Will a Fed hike hurt A-shares?"): the
# question is in scope (its target is the A-share market), but no source carries the foreign series.
_FOREIGN_MACRO = re.compile(
    r"美联储|联储|FOMC|美国(?:的)?(?:加息|降息|利率|通胀|CPI|国债|经济|就业|非农)|美债|欧洲央行|欧央行|日本央行|日央行|"
    r"\bfed\b|federal reserve|\bu\.?s\.? (?:rates?|inflation|cpi|treasur(?:y|ies)|economy|jobs)|\becb\b|\bboj\b|"
    r"bank of japan|european central bank",
    re.IGNORECASE,
)


def foreign_macro_gaps(query: str, *, zh: bool) -> list[str]:
    """State that foreign central-bank and US macro data are not covered, when the question is about them."""
    if not _FOREIGN_MACRO.search(query or ""):
        return []
    return [
        "当前数据源只有中国的宏观指标（如CPI、PMI、货币供应量和国债收益率），没有美联储等境外央行的政策利率或美国经济数据，"
        "因此无法用数据说明其对A股的具体影响；以下只列出可得的国内证据。"
        if zh
        else "The configured sources carry China macro indicators only (CPI, PMI, money supply, government bond "
        "yields); they have no Federal Reserve or other foreign central-bank rates and no US economic data, so the "
        "effect on A-shares cannot be shown with data. Only the available domestic evidence is listed."
    ]


# Positions and fund flows of an investor group ("国家队最近是不是在加仓中国平安", "Is northbound money flowing into
# Moutai?"): no configured source carries holdings, flows or margin balances. A company's own shareholders are not
# included: their increases and reductions are disclosed in announcements.
_FLOW_SUBJECT = re.compile(
    r"国家队|中央汇金|汇金|证金|社保基金|社保|养老金|险资|公募基金|私募基金|机构投资者|主力资金|主力|游资|北向资金|南向资金|"
    r"北上资金|外资|陆股通|沪股通|深股通|聪明钱|\bnational team\b|\bstate(?:-backed)? funds?\b|"
    r"\b(?:northbound|southbound) (?:money|funds?|capital|investors|flows?)\b|"
    r"\bforeign (?:investors|funds|money|capital)\b|\binstitutional investors?\b|\bsmart money\b",
    re.IGNORECASE,
)
_FLOW_ACTION = re.compile(
    r"加仓|减仓|增持|减持|建仓|清仓|持仓|持股|抄底|出货|买入|卖出|买进|抛售|扫货|买了|卖了|在买|在卖|流入|流出|净买|净卖|"
    r"进场|离场|"
    r"\b(?:buy(?:ing)?|sell(?:ing)?|bought|sold|adding|trimming|accumulating|dumping|inflows?|outflows?|flow(?:ing|ed)?|"
    r"holdings?|positions?|stakes?|net (?:buying|selling|purchases?))\b",
    re.IGNORECASE,
)
_FLOW_DIRECT = re.compile(
    r"资金流向|资金流入|资金流出|两融余额|融资余额|融券余额|持仓数据|\b(?:fund|money|capital) flows?\b|"
    r"\bmargin (?:balance|debt|financing balance)\b|\binstitutional holdings\b",
    re.IGNORECASE,
)


def flow_gaps(query: str, tool_log: list[dict[str, Any]] | None = None, *, zh: bool) -> list[str]:
    """State that holdings and fund-flow data are not covered, when the question asks about them."""
    subject = _FLOW_SUBJECT.search(query or "")
    direct = _FLOW_DIRECT.search(query or "")
    if not direct and not (subject and _FLOW_ACTION.search(query or "")):
        return []
    if any(entry.get("tool") == "explain_concept" and entry.get("ok") for entry in tool_log or []):
        return []  # the glossary answer already says the concept has no data series
    if subject and _FLOW_ACTION.search(query or ""):
        who = subject.group(0)
        if zh:
            return [f"当前数据源没有{who}的持仓或资金流向数据，无法判断其是否在买入或卖出；以下只列出可得的证据。"]
        return [
            f"The configured sources have no holdings or fund-flow data for {who}, so whether they are buying or "
            "selling cannot be shown; only the available evidence is listed."
        ]
    what = direct.group(0)  # type: ignore[union-attr]
    if zh:
        return [f"当前数据源没有{what}数据，无法回答这一项；以下只列出可得的证据。"]
    return [
        f"The configured sources do not include {what} data, so that part cannot be answered; only the available "
        "evidence is listed."
    ]


# A year-to-date move ("今年涨了多少", "年初至今收益", "YTD return", "how has it done this year"), not this year's
# statements ("今年营收").
_YEAR_TO_DATE = re.compile(
    r"今年(?:以来|到现在|至今|迄今)?(?:的|一共|总共|累计|整体|共){0,3}(?:涨|跌|收益|回报|表现|涨幅|跌幅|走势)|"
    r"年初(?:至今|到现在|以来)|年内(?:的|累计|共){0,2}(?:涨|跌|收益|回报|涨幅|跌幅|表现)|"
    r"\bYTD\b|\byear[- ]to[- ]date\b|\bso far this year\b|\bsince the (?:start|beginning) of (?:the|this) year\b|"
    r"\bthis year\b.{0,20}\b(?:return|gain|perform|rise|rose|risen|fall|fell|fallen|up|down|change|move)|"
    r"\b(?:return|gain|performance|change|move|up|down|rise|rose|fall|fell)\b.{0,20}\bthis year\b",
    re.IGNORECASE,
)


def asks_year_to_date(query: str) -> bool:
    return bool(_YEAR_TO_DATE.search(query or ""))


# (round 14) A year-to-date question that names the year's first trading day (or its first close) as the base keeps
# that base instead of the previous year's last close ("从首个交易日的收盘算起", "from the first close of the year").
_YTD_FIRST_CLOSE_BASE = re.compile(
    r"(?:今年|年内|本年)?(?:第一个|首个|头一个)交易日|年初第一(?:个|天)|年内首个收盘|"
    r"\bfirst (?:trading day|close|session) of (?:the|this) year\b|\bthe year's first (?:close|trading day)\b",
    re.IGNORECASE,
)


def ytd_from_first_close(query: str) -> bool:
    return bool(_YTD_FIRST_CLOSE_BASE.search(query or ""))


def year_to_date_gaps(query: str, tool_log: list[dict[str, Any]], *, zh: bool) -> list[str]:
    """A year-to-date question whose price data does not reach back to the first trading day of the year.

    ``get_price_history`` reports ``year_start`` (the first close of the latest close's year) only when its history
    also has a close from the year before, so that close is known to be the year's first. Without it the change
    since the start of the year is stated as unavailable, instead of answering with the latest daily move.
    """
    if not asks_year_to_date(query):
        return []
    sentences = []
    for entry in tool_log:
        if entry.get("tool") != "get_price_history" or not entry.get("ok"):
            continue
        data = entry.get("data") or {}
        if data.get("year_start") or data.get("close") is None:
            continue
        name = str(data.get("name") or data.get("symbol") or "")
        sentences.append(
            f"当前数据中没有{name}今年首个交易日的收盘价，无法计算今年以来的涨跌幅；以下只列出最新一日的行情。"
            if zh
            else f"The data has no close for {name} on the first trading day of this year, so the year-to-date "
            "change cannot be computed; only the latest session is listed below."
        )
    return list(dict.fromkeys(sentences))


# (round 9, E8) A maximum drawdown over a period ("近一年最大回撤", "max drawdown this year") needs the whole period's
# closes; the price tool returns the latest few. Stated as not computable instead of answering with the daily move.
_DRAWDOWN = re.compile(
    r"最大(?:的)?回撤|回撤(?:幅度|了多少|有多(?:大|少|深))|最大跌幅|\bmax(?:imum)?\.? drawdown\b|\bdrawdown\b",
    re.IGNORECASE,
)


def asks_drawdown(query: str) -> bool:
    return bool(_DRAWDOWN.search(query or ""))


# (round 12) The window of a drawdown question: this year ("今年以来最大回撤", "YTD drawdown") or, by default, the
# 52 weeks to the latest close ("近一年", "过去12个月", or no period at all; the answer names the window it used).
_YTD_WINDOW = re.compile(
    r"今年|年初|年内|本年|\bYTD\b|\byear[- ]to[- ]date\b|\bthis year\b|"
    r"\bsince the (?:start|beginning) of (?:the|this) year\b",
    re.IGNORECASE,
)


def drawdown_window(query: str) -> str:
    if not asks_drawdown(query):
        return ""
    return "ytd" if _YTD_WINDOW.search(query or "") else "1y"


# (round 12) The 52-week high and low ("52周最高价", "一年内最高和最低收盘价", "52-week range", "12-month low").
_RANGE_52W = re.compile(
    r"52\s*周|五十二周|(?:近|过去|最近)?(?:一|1)\s*年(?:内|来|以来|里)?(?:的)?(?:最高|最低|高点|低点|新高|新低|区间|价格区间)|"
    r"(?:近|过去|最近)\s*12\s*个?月(?:内|来|以来|里)?(?:的)?(?:最高|最低|高点|低点|区间)|"
    r"\b52[- ]?(?:week|wk)s?\b|\b(?:one|1|12)[- ](?:year|month) (?:high|low|range)s?\b|"
    r"\b(?:high|low)s? (?:of|over|in) the (?:past|last) (?:year|12 months|52 weeks)\b",
    re.IGNORECASE,
)
# (round 14, review-10 J4) "离52周最低点反弹了多少", "how far has it bounced off its 52-week low?";
# "较52周高点跌了多少", "how far below its 52-week high is it?"
_REBOUND_FROM_LOW = re.compile(
    r"(?:低点|最低|低位|新低)[^，。？?；;]{0,12}?(?:反弹|回升|涨回|涨了|涨起来|上涨|拉升)|"
    # (round 16) "现在比一年来的最低收盘价高出多少元"
    r"(?:比|较|离)[^，。？?；;]{0,10}?(?:最低|低点|低位)[^，。？?；;]{0,8}?(?:高出|高了|多出|高(?=多少|几))|"
    r"(?:反弹|回升)[^，。？?；;]{0,6}?(?:多少|几)|"
    r"\b(?:rebound(?:ed)?|bounced?|recovered|rallied|risen|climbed|up)\b[^.?!]{0,30}?\b(?:from|off|since)\b"
    r"[^.?!]{0,20}?\blows?\b|\babove (?:its|the) (?:52[- ]week |one[- ]year )?lows?\b",
    re.IGNORECASE,
)
_BELOW_HIGH = re.compile(
    r"(?:较|离|距|相比|相对|从|(?:现在|目前|如今|当前|股价|收盘价?|最新价)比)[^，。？?；;]{0,10}?(?:最高|高点|高位|新高)[^，。？?；;]{0,12}?(?:跌|回落|回撤|低|下来|差|还差)|"
    r"\b(?:below|off|down from|under)\b (?:its|the) (?:52[- ]week |one[- ]year )?highs?\b",
    re.IGNORECASE,
)
# (round 15, review-11 K4) the rise needed to get back to the 52-week high: (high − close) ÷ close, not the fall from it
_BACK_TO_HIGH = re.compile(
    r"(?:涨|上涨|反弹|回升|拉升)[^，。？?；;]{0,10}?(?:才能|才可以|才会|才|能|可以)?(?:回到|重回|收复|涨回|回升到|回)"
    r"[^，。？?；;]{0,8}?(?:最高|高点|高位|前高)|"
    # (round 17, review-13 M5) the rise asked in the next clause: "要回到52周最高收盘价，还需要上涨百分之几"
    r"(?:回到|重回|收复|涨回|回升到|站回)[^，。？?；;]{0,12}?(?:最高|高点|高位|前高)[^。？?；;]{0,10}?"
    r"(?:要|需要|得|还要|还得|还需|还需要|需)?(?:再)?(?:涨|上涨|反弹)(?:多少|几|百分之(?:多少|几))|"
    r"(?:距离?|离)[^，。？?；;]{0,8}?(?:最高|高点|前高)[^。？?；;]{0,8}?(?:还要|还得|还需|需要|要)(?:再)?(?:涨|上涨)"
    r"(?:多少|几|百分之)|"
    r"(?:距离?|离)[^，。？?；;]{0,8}?(?:最高|高点|前高)[^。？?；;]{0,8}?还差[^。？?；;]{0,4}?(?:百分比|百分之|几个点|%)|"
    r"\b(?:rise|gain|climb|rally|go up|increase)\b[^.?!]{0,40}?\b(?:back to|to (?:get|climb|return) back to|to "
    r"(?:regain|reclaim|retake|revisit|return to|get to|reach|go back to))\b[^.?!]{0,30}?\bhighs?\b|"
    r"\bhow much\b[^.?!]{0,40}?\b(?:back to|regain|reclaim|return to)\b[^.?!]{0,30}?\bhighs?\b",
    re.IGNORECASE,
)
_SHORT_OF_HIGH = re.compile(r"(?:距离?|离)[^，。？?；;]{0,8}?(?:最高|高点|前高)[^。？?；;]{0,8}?还差")
# The session named next to a 52-week question ("今天最高多少，52周最高多少"): then the day's high/low is wanted too.
_SESSION_WORDS = re.compile(r"今天|今日|当天|当日|日内|\btoday\b|\bintraday\b|\bsession\b", re.IGNORECASE)


def asks_range_52w(query: str) -> bool:
    return bool(_RANGE_52W.search(query or ""))


def drawdown_gaps(query: str, tool_log: list[dict[str, Any]], *, zh: bool) -> list[str]:
    """A drawdown question whose price data does not cover the window (the v1 snapshot and the live path keep only
    the latest closes): the drawdown is stated as unavailable. With the window covered (``max_drawdown_1y`` /
    ``max_drawdown_ytd`` in the price data) the template states it instead."""
    window = drawdown_window(query)
    if not window:
        return []
    sentences = []
    for entry in tool_log:
        if entry.get("tool") != "get_price_history" or not entry.get("ok"):
            continue
        data = entry.get("data") or {}
        if data.get("close") is None or data.get(f"max_drawdown_{window}"):
            continue
        name = str(data.get("name") or data.get("symbol") or "")
        if data.get("price_jump_date"):
            sentences.append(_jump_text(name, str(data["price_jump_date"]), "最大回撤", "maximum drawdown", zh=zh))
            continue
        closes = len(data.get("recent_closes") or [])
        held_zh = f"最近 {closes} 个交易日的收盘价" if closes >= 2 else "最新一个交易日的收盘价"
        # no digits: a count of closes is not a figure the verifier can trace to the evidence
        held_en = "a few recent closes" if closes >= 2 else "the latest close"
        sentences.append(
            f"当前数据只有{name}{held_zh}，无法计算所问期间的最大回撤；以下只列出最新行情。"
            if zh
            else f"The data holds only {held_en} for {name}, so the maximum drawdown over the requested period "
            "cannot be computed; only the latest session is listed below."
        )
    return list(dict.fromkeys(sentences))


def _jump_text(name: str, day: str, what_zh: str, what_en: str, *, zh: bool) -> str:
    """A window holding a close-to-close jump beyond every daily limit: likely ex-rights (the closes are unadjusted)."""
    if zh:
        return (
            f"{name}的未复权收盘价在 {day} 有一次超出涨跌幅限制的跳变（可能是送转股除权），无法据此计算{what_zh}；"
            "以下只列出最新行情。"
        )
    return (
        f"{name}'s unadjusted closes jump by more than any daily price limit on {day} (likely an ex-rights "
        f"adjustment), so the {what_en} cannot be computed from them; only the latest session is listed below."
    )


def range_52w_gaps(query: str, tool_log: list[dict[str, Any]], *, zh: bool) -> list[str]:
    """A 52-week high/low question whose price data does not reach back 52 weeks: stated as unavailable, instead
    of answering with the day's high and low."""
    if not asks_range_52w(query):
        return []
    sentences = []
    for entry in tool_log:
        if entry.get("tool") != "get_price_history" or not entry.get("ok"):
            continue
        data = entry.get("data") or {}
        if data.get("close") is None or data.get("range_52w"):
            continue
        name = str(data.get("name") or data.get("symbol") or "")
        if data.get("price_jump_date"):
            sentences.append(
                _jump_text(name, str(data["price_jump_date"]), "52周最高价和最低价", "52-week high and low", zh=zh)
            )
            continue
        sentences.append(
            f"当前数据中{name}的收盘价不足一年，无法给出近一年的最高价和最低价；以下只列出最新行情。"
            if zh
            else f"The data holds less than a year of closes for {name}, so its high and low over the past year cannot "
            "be given; only the latest session is listed below."
        )
    return list(dict.fromkeys(sentences))


def _years_text(years: list[int], *, zh: bool) -> str:
    # "2019年" / "FY2019": forms the verifier reads as dates, not as claimed values.
    return "、".join(f"{year}年" for year in years) if zh else ", ".join(f"FY{year}" for year in years)


# Prefix of the get_fundamentals error for a sector without an industry snapshot (tools/fundamentals.py).
INDUSTRY_NOT_FOUND = "no industry snapshot for"


def failed_target_statements(
    query: str, tool_log: list[dict[str, Any]], *, zh: bool, names: dict[str, str] | None = None
) -> list[str]:
    """Name the targets whose data could not be retrieved (instead of a generic "no evidence" line)."""
    names = names or {}
    per_target: dict[str, list[str]] = {}
    sectors: list[str] = []
    for entry in tool_log:
        if entry.get("ok"):
            continue
        arguments = entry.get("arguments") or {}
        target = arguments.get("target") or ((arguments.get("targets") or [None])[0])
        if not target:
            continue
        if str((entry.get("error") or {}).get("message") or "").startswith(INDUSTRY_NOT_FOUND):
            # "半导体板块估值高吗": the sector is known, the sources have no snapshot for it
            if str(target) not in sectors:
                sectors.append(str(target))
            continue
        kind = {
            "get_fundamentals": ("基本面", "fundamentals"),
            "get_price_history": ("行情", "market data"),
            "compute_indicators": ("技术指标", "technical indicators"),
        }.get(str(entry.get("tool")))
        if kind is None:
            continue
        per_target.setdefault(str(target), [])
        label = kind[0] if zh else kind[1]
        if label not in per_target[str(target)]:
            per_target[str(target)].append(label)
    statements = [
        f"当前数据源没有{sector}行业的估值和行情快照，因此无法判断该行业的估值高低。"
        if zh
        else f"The configured sources have no valuation or market snapshot for the {sector} sector, so its "
        "valuation cannot be assessed."
        for sector in sectors
    ]
    for target, kinds in per_target.items():
        name = names.get(target)
        who = f"{name}（{target}）" if zh and name else (f"{name} ({target})" if name else target)
        statements.append(
            f"当前数据源中没有{who}的{'、'.join(kinds)}数据。"
            if zh
            else f"The current data sources have no {', '.join(kinds)} for {who}."
        )
    return statements


def metric_label(key: str, *, zh: bool) -> str:
    metric = _METRIC_BY_KEY.get(key)
    if metric is None:
        return key
    return metric.zh if zh else metric.en


# --------------------------------------------------------------------------- requested price details

_COUNT_WORDS = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20,
}  # fmt: skip
_COUNT = r"(\d{1,2}|[一两二三四五六七八九十]|one|two|three|four|five|six|seven|eight|nine|ten|twenty)"
_RECENT_CLOSES = re.compile(
    rf"(?:最近|近|过去|前)\s*{_COUNT}\s*(?:个)?(?:交易日|天|日)[^？?。]{{0,8}}?收盘|"
    rf"收盘价?[^？?。]{{0,6}}?(?:最近|近|过去)\s*{_COUNT}\s*(?:个)?(?:交易日|天|日)|"
    rf"(?:last|past|previous|recent)\s+{_COUNT}\s+(?:closes|closing prices|sessions|trading days|days)|"
    rf"{_COUNT}\s+(?:most recent|latest|recent)\s+closes",
    re.IGNORECASE,
)
_CLOSES_WORD = re.compile(r"收盘|\bclos(?:es|ing)\b", re.IGNORECASE)
_PREVIOUS_CLOSE = re.compile(
    r"前一(?:天|日|个交易日)(?:的)?收盘|前收|昨收|上一(?:个)?交易日(?:的)?收盘|\b(?:previous|prior|prev\.?) close\b|"
    r"\bclose (?:before|the day before)\b",
    re.IGNORECASE,
)
# The day's high/low, not "谁的ROE最高" or "is the P/E high or low".
_HIGH = re.compile(
    r"最高价|最高点|日内高点|最高(?=(?:和|与|、|及)最低)|(?:当天|当日|今天|日内)(?:的)?最高|"
    r"\b(?:daily|day'?s|intraday|session|today'?s)\s+highs?\b|\bhighs? and (?:the )?lows?\b|\bhigh/low\b|"
    r"\bhigh price\b",
    re.IGNORECASE,
)
_LOW = re.compile(
    r"最低价|最低点|日内低点|(?<=最高和)最低|(?<=最高与)最低|(?<=最高、)最低|(?:当天|当日|今天|日内)(?:的)?最低|"
    r"\b(?:daily|day'?s|intraday|session|today'?s)\s+lows?\b|\bhighs? and (?:the )?lows?\b|\bhigh/low\b|\blow price\b",
    re.IGNORECASE,
)
_TURNOVER_RATE = re.compile(r"换手率|换手(?:了)?(?:多少|几)|\bturnover (?:rate|ratio)\b", re.IGNORECASE)


def sector_data_question(text: str, entities: list[dict[str, Any]]) -> bool:
    """(round 13) A named sector's own figure that a glossary word names ("白酒行业今天换手率多少"): answered from the
    industry snapshot, not with the concept's definition. Only for a sector the NLU tags (an English question such as
    "the baijiu sector's turnover rate" gets no sector entity and still reads as the glossary concept)."""
    sectors = any(entity.get("entity_type") == "sector" for entity in entities or [])
    return (
        sectors
        and bool(_TURNOVER_RATE.search(text or ""))
        and not re.search(
            r"是什么|什么是|什么意思|含义|定义|怎么算|如何计算|\bwhat is (?:a |the )?turnover rate\b|\bdefin",
            text or "",
            re.I,
        )
    )


_CHANGE_AMOUNT = re.compile(
    r"(?:涨|跌)了?(?:多少|几)(?:钱|元|块)|(?:涨|跌)(?:了)?(?:几|多少)块钱|\bhow many yuan\b|"
    r"\b(?:rose|fell|dropped|gained|moved|changed?)\b[^.?!]{0,20}\bin yuan\b",
    re.IGNORECASE,
)
_OPEN = re.compile(r"开盘价?|\bopen(?:ing)?(?: price)?\b(?! interest)", re.IGNORECASE)
_VOLUME = re.compile(r"成交量|量能|\b(?:trading )?volume\b", re.IGNORECASE)
_AMOUNT = TURNOVER  # (round 12) the comparison frame's turnover vocabulary ("成交了多少钱", "trading value")
_RETURN_DAYS = re.compile(
    rf"(?:近|过去|最近)?\s*{_COUNT}\s*(?:个)?(?:交易日|日|天)(?:的)?(?:收益率?|回报|涨幅|跌幅|涨跌幅?|表现)|"
    rf"\b{_COUNT}[- ](?:day|session)s?\s+(?:return|change|performance|gain|move)\b",
    re.IGNORECASE,
)
_MOVING_AVERAGE = re.compile(
    r"(?<![A-Za-z])MA\s*(\d{1,3})(?!\d)|(\d{1,3})\s*(?:日|天)均线|\b(\d{1,3})[- ]day moving average\b", re.IGNORECASE
)
_ABOVE_MA = re.compile(
    # (round 17, review-13 M8) "跌破净资产" is book value (the P/B), not a moving average
    r"站上|站稳|跌破(?!净资产|每股净资产|净值|账面|发行价)|均线(?:上方|之上|下方|之下)|"
    r"\b(?:above|below) (?:the |its )?(?:\d+[- ]day )?(?:ma\d*|moving average)",
    re.I,
)


def _count(token: str) -> int:
    token = token.lower()
    return int(token) if token.isdigit() else _COUNT_WORDS.get(token, 0)


@dataclass(frozen=True)
class PriceRequest:
    """Price details a question asks for beyond the latest close and daily change."""

    closes: int = 0  # number of recent closes to list
    previous_close: bool = False
    high: bool = False
    low: bool = False
    open: bool = False
    volume: bool = False
    amount: bool = False
    return_days: tuple[int, ...] = ()
    moving_averages: tuple[int, ...] = ()
    above_ma: bool = False
    year_to_date: bool = False
    ytd_first_close: bool = False  # the user named the year's first close as the year-to-date base (round 14)
    range_52w: bool = False  # 52-week (one-year) high and low close
    drawdown: str = ""  # "1y" or "ytd": the window of a maximum drawdown question
    # (round 12, H9) the stock turnover rate (换手率), answered when the source has it, else named missing; the
    # change in yuan against the previous close ("今天跌了多少钱")
    turnover_rate: bool = False
    change_amount: bool = False
    # (round 14, review-10 J4) the move from the 52-week low ("离52周低点反弹了多少") or below the 52-week high
    rebound_52w: bool = False
    off_high_52w: bool = False
    back_to_high_52w: bool = False  # (round 15) the rise needed to get back to the 52-week high

    @property
    def needs_quote(self) -> bool:
        return bool(
            self.closes
            or self.previous_close
            or self.high
            or self.low
            or self.open
            or self.volume
            or self.amount
            or self.turnover_rate
            or self.change_amount
        )

    @property
    def needs_indicators(self) -> bool:
        return bool(self.return_days or self.moving_averages or self.above_ma)


def requested_price_fields(query: str) -> PriceRequest:
    """Parse the price details asked for.

    Examples: "最近五个交易日的收盘价", "daily high and low", "3-day return", "MA5站上了吗".
    """
    text = query or ""
    closes = 0
    match = _RECENT_CLOSES.search(text)
    if match:
        closes = _count(next(group for group in match.groups() if group))
    elif re.search(r"\b(?:last|latest|recent) (?:two|2) closes\b|最近两(?:个|次)收盘", text, re.IGNORECASE):
        closes = 2
    returns = sorted({_count(next(g for g in m.groups() if g)) for m in _RETURN_DAYS.finditer(text)} - {0})
    averages = sorted({int(next(g for g in m.groups() if g)) for m in _MOVING_AVERAGE.finditer(text)})
    above = bool(_ABOVE_MA.search(text))
    if above and not averages:
        averages = [5]
    # "52周最高和最低" asks for the 52-week range, not the day's high and low (unless the day is named too)
    back = bool(_BACK_TO_HIGH.search(text))
    yearly = asks_range_52w(text) or back
    session = not yearly or bool(_SESSION_WORDS.search(text))
    return PriceRequest(
        closes=closes if _CLOSES_WORD.search(text) or match else 0,
        previous_close=bool(_PREVIOUS_CLOSE.search(text)),
        high=bool(_HIGH.search(text)) and session,
        low=bool(_LOW.search(text)) and session,
        open=bool(_OPEN.search(text)),
        volume=bool(_VOLUME.search(text)),
        amount=bool(_AMOUNT.search(text)),
        return_days=tuple(returns),
        moving_averages=tuple(averages),
        above_ma=above,
        year_to_date=asks_year_to_date(text) and not asks_drawdown(text),
        ytd_first_close=ytd_from_first_close(text),
        range_52w=yearly,
        drawdown=drawdown_window(text),
        turnover_rate=bool(_TURNOVER_RATE.search(text)),
        change_amount=bool(_CHANGE_AMOUNT.search(text)),
        rebound_52w=yearly and bool(_REBOUND_FROM_LOW.search(text)),
        # (round 17) "距离52周最高还差多少百分比" reads both ways: the fall from the high and the rise back to it
        off_high_52w=yearly and bool(_BELOW_HIGH.search(text)) and (not back or bool(_SHORT_OF_HIGH.search(text))),
        back_to_high_52w=back,
    )


# --------------------------------------------------------------------------- reporting periods

_QUARTER = re.compile(
    r"(?P<q>[一二三四1-4])季度|第(?P<q2>[一二三四1-4])季度|\bQ(?P<q3>[1-4])\b|"
    r"\b(?P<q4>first|second|third|fourth) quarter\b|(?P<h>上半年|半年报|中报|\bH1\b|\bfirst half\b|\binterim\b)",
    re.IGNORECASE,
)
_QUARTER_MONTH = {"1": 3, "一": 3, "first": 3, "2": 6, "二": 6, "second": 6, "3": 9, "三": 9, "third": 9}
_QUARTER_MONTH.update({"4": 12, "四": 12, "fourth": 12})


def requested_quarter(query: str) -> tuple[str, int] | None:
    """``(label, period-end month)`` for a quarter or half-year request ("今年一季度", "Q3", "上半年")."""
    match = _QUARTER.search(query or "")
    if not match:
        return None
    if match.group("h"):
        return match.group("h"), 6
    token = next(
        group for group in (match.group("q"), match.group("q2"), match.group("q3"), match.group("q4")) if group
    )
    return match.group(0), _QUARTER_MONTH[token.lower()]


def _month_of(value: Any) -> int | None:
    found = re.match(r"\s*(?:19|20)\d{2}-(\d{1,2})", str(value or ""))
    return int(found.group(1)) if found else None


# --------------------------------------------------------------------------- industry and product scope

_INDUSTRY_SCOPE = re.compile(r"行业|板块|同行|同业|\bsector\b|\bindustry\b|\bpeers?\b", re.IGNORECASE)
# Standard fundamentals a question can ask of a company or of its industry snapshot (industry keys in brackets).
_STANDARD: tuple[tuple[str, str, str, re.Pattern[str], tuple[str, ...]], ...] = (
    ("pe", "市盈率", "P/E", re.compile(r"市盈率|(?<![A-Za-z])P/?E(?![A-Za-z])|price[- ]to[- ]earnings", re.I), ("pe",)),
    ("pb", "市净率", "P/B", re.compile(r"市净率|(?<![A-Za-z])P/?B(?![A-Za-z])|price[- ]to[- ]book", re.I), ("pb",)),
    ("roe", "ROE", "ROE", re.compile(r"净资产收益率|(?<![A-Za-z])ROE(?![A-Za-z])|return on equity", re.I), ("roe",)),
    ("revenue", "营业收入", "revenue", re.compile(r"营收|营业收入|\brevenue\b", re.I), ("revenue",)),
    (
        "net_profit",
        "净利润",
        "net profit",
        re.compile(r"净利润|净利(?!率)|\bnet (?:profit|income)\b", re.I),
        ("net_profit",),
    ),
    ("gross_margin", "毛利率", "gross margin", re.compile(r"毛利率|gross margin", re.I), ("gross_margin",)),
)
_FUNDAMENTAL_ASK = re.compile(
    r"市盈率|市净率|净资产收益率|ROE|营收|营业收入|净利润|毛利率|股息率|基本面|财报|(?<![A-Za-z])P/?[EB](?![A-Za-z])|"
    r"\brevenue\b|\bnet (?:profit|income)\b|\bgross margin\b|\bearnings\b|\bfundamentals?\b",
    re.IGNORECASE,
)


def asks_about_industry(query: str) -> bool:
    return bool(_INDUSTRY_SCOPE.search(query or ""))


def industry_gaps(query: str, tool_log: list[dict[str, Any]], *, zh: bool) -> list[str]:
    """Industry metrics the question asks for that the industry snapshot does not contain.

    "ROE跟保险行业平均比呢？": the 保险 snapshot has PE/PB/change but no ROE, so the industry side is stated
    as missing instead of being silently skipped.
    """
    if not asks_about_industry(query):
        return []
    wanted = [item for item in _STANDARD if item[3].search(query or "")]
    sentences: list[str] = []
    seen: set[str] = set()
    for entry in tool_log:
        if entry.get("tool") != "get_fundamentals" or not entry.get("ok"):
            continue
        industry = (entry.get("data") or {}).get("industry") or {}
        name = str(industry.get("industry_name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        metrics = industry.get("metrics") or {}
        missing = [item for item in wanted if not any(metrics.get(key) is not None for key in item[4])]
        if missing:
            labels = "、".join(item[1] for item in missing) if zh else ", ".join(item[2] for item in missing)
            sentences.append(
                f"当前数据源的{name}行业快照没有{labels}，无法给出行业层面的这一项。"
                if zh
                else f"The {name} industry snapshot in the current data has no {labels}, so the industry side of "
                "that cannot be given."
            )
    return sentences


def non_stock_fundamental_gaps(
    query: str,
    tool_log: list[dict[str, Any]],
    *,
    zh: bool,
    names: dict[str, str] | None = None,
    types: dict[str, str] | None = None,
) -> list[str]:
    """P/E, ROE or revenue asked of an ETF, fund or index: company fundamentals do not exist for it."""
    if not types or not _FUNDAMENTAL_ASK.search(query or ""):
        return []
    fetched = {
        str((entry.get("arguments") or {}).get("target"))
        for entry in tool_log
        if entry.get("tool") == "get_fundamentals"
    }
    sentences = []
    for symbol, kind in types.items():
        if kind in {"etf", "fund", "index"} and symbol not in fetched:
            name = (names or {}).get(symbol) or symbol
            label = {"etf": "ETF", "fund": "基金" if zh else "fund", "index": "指数" if zh else "index"}[kind]
            sentences.append(
                f"市盈率、ROE、营收等基本面指标当前只覆盖个股；{name}（{symbol}）是{label}，当前数据中没有它的这类数据。"
                if zh
                else f"P/E, ROE, revenue and similar fundamentals are only covered for individual stocks; {name} "
                f"({symbol}) is an {label}, so the current data has none for it."
            )
    return sentences


_NAMED_INDICATORS = (
    ("RSI(14)", "rsi_14", re.compile(r"(?<![A-Za-z])RSI", re.I)),
    ("MACD", "macd", re.compile(r"MACD", re.I)),
    ("20D vol", "volatility_20d", re.compile(r"波动率|volatility", re.I)),
    ("Bollinger", "bollinger", re.compile(r"布林|bollinger", re.I)),
)


def indicator_gaps(
    query: str, tool_log: list[dict[str, Any]], *, zh: bool, names: dict[str, str] | None = None
) -> list[str]:
    """Requested technical indicators or N-day returns that could not be computed or are null."""
    request = requested_price_fields(query)
    if not request.needs_indicators and not re.search(
        r"RSI|MACD|均线|moving average|volatility|波动率", query or "", re.I
    ):
        return []
    sentences = []
    for entry in tool_log:
        if entry.get("tool") != "compute_indicators":
            continue
        target = str((entry.get("arguments") or {}).get("target") or "")
        name = (names or {}).get(target) or target
        if not entry.get("ok"):
            sentences.append(
                f"当前数据不足以计算{name}的所问技术指标（历史收盘价不够）。"
                if zh
                else f"The current data is not enough to compute the requested indicators for {name} "
                "(too little price history)."
            )
            continue
        data = entry.get("data") or {}
        returns = data.get("pct_change_nd") or {}
        missing = [f"{days}日涨跌幅" if zh else f"{days}-day return" for days in request.return_days
                   if returns.get(f"pct_{days}d") is None]  # fmt: skip
        missing += [f"MA{days}" for days in request.moving_averages if data.get(f"ma{days}") is None]
        for label, key, pattern in _NAMED_INDICATORS:
            if pattern.search(query or "") and data.get(key) is None and label not in missing:
                missing.append(label)
        if missing:
            joined = "、".join(missing) if zh else ", ".join(missing)
            sentences.append(
                f"当前数据中没有{name}的{joined}（历史数据不足）。"
                if zh
                else f"The current data has no {joined} for {name} (not enough history)."
            )
    return sentences
