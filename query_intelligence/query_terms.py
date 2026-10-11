from __future__ import annotations

import re


INDUSTRY_TERMS = [
    "白酒",
    "保险",
    "券商",
    "消费",
    "银行",
    "科技",
    "地产",
    "成长指数",
    "宽基指数",
]

BUCKET_MARKET_TERMS = [
    "行业",
    "板块",
    "大盘",
    *INDUSTRY_TERMS,
]


def contains_bucket_market_term(query: str) -> bool:
    for term in BUCKET_MARKET_TERMS:
        if term not in query:
            continue
        if term in INDUSTRY_TERMS and len(term) <= 2 and not _contains_standalone_short_industry(query, term):
            continue
        return True
    return False


def _contains_standalone_short_industry(query: str, term: str) -> bool:
    sector_markers = {"板块", "行业", "赛道", "方向", "主题"}
    start = 0
    while True:
        index = query.find(term, start)
        if index < 0:
            return False
        previous_char = query[index - 1] if index > 0 else ""
        next_text = query[index + len(term) : index + len(term) + 2]
        if not previous_char or not _is_cjk_char(previous_char) or next_text in sector_markers:
            return True
        start = index + len(term)


def _is_cjk_char(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


# (round 17, review-13 M11) Position-size and allocation requests are advice, like 止损 / 止盈: how many shares or lots
# to buy, how large a position or what share of a portfolio to put in a stock, asked with a verdict ("比较合适",
# "行吗", "should", "is it OK"). Before round 17 "我有10万块，买多少股招行比较合适" (比较 read as a comparison),
# "把仓位的三成放宁德时代行吗" and "Is it OK to put 30% of my portfolio in CATL?" got facts only, with no risk flag and no
# hedge. Factual arithmetic stays a lookup: the value of a stated holding ("我有1000股招行，现在值多少钱"), what a sum
# can buy ("10万块能买多少股", no verdict), a fund's or an index's weights ("持仓比例", "占沪深300的权重").
_VERDICT_ZH = (
    r"(?:比较|较为|更|最)?(?:合适|合理|恰当|稳妥|稳健|为宜|好|划算|妥当)|行吗|行不行|可以吗|可不可以|可行吗|好吗|好不好|"
    r"合适吗|合不合适|合理吗|怎么样|如何|妥吗|靠谱吗|会不会太多|会不会太少|多不多|算多吗|太多了吗|太重了吗|重不重|过重吗"
)
_SHARE_OR_AMOUNT_ZH = (
    r"(?:多少|几)\s*(?:股|手|万|千|块|元|钱|成|份)?|百分之(?:多少|几)|多大(?:的)?(?:比例|比重|仓位)?|多重|多少比例|多少仓位"
)
_PORTION_ZH = (
    r"(?:百分之\s*[一二两三四五六七八九十几百]+|\d+(?:\.\d+)?\s*[%％]|[一二两三四五六七八九十半几]\s*成|"
    r"(?:一|二|两|三|四|五)?(?:半|一半|大半|大部分|小部分|全部|所有))"
)
# a fund's, an institution's or a market's position ("这只基金的仓位多少", "北向资金仓位") is data, not advice
_OTHERS = ("基金", "ETF", "etf", "产品", "机构", "私募", "公募", "北向", "南向", "外资", "社保", "险资", "主力", "游资", "券商", "股东")
_NOT_OTHERS = "".join(f"(?<!{word})(?<!{word}的)(?<!{word}资金)(?<!{word}资金的)" for word in _OTHERS)
_ALLOCATE_VERB_ZH = r"(?:放|买|买入|投|投入|投资|配|配置|压|押|砸|上|建仓|持有|拿)"
POSITION_SIZING = re.compile(
    # "买多少股比较合适", "应该买几手", "该买多少", "建仓多少合适", "投多少钱买招行合适"
    r"(?:买|买入|建仓|配置|配|投|投入|投资|持有)\s*(?:" + _SHARE_OR_AMOUNT_ZH + r")[^，。？?！!；;]{0,14}?"
    r"(?:" + _VERDICT_ZH + r")|"
    r"(?:应该|应当|该|最好|建议|需要|要)(?:要)?\s*(?:买|买入|建仓|配置|投|投入|持有)\s*(?:" + _SHARE_OR_AMOUNT_ZH + r")|"
    # "仓位多少合适", "仓位控制在多少", "几成仓位", "多少仓位比较好", "仓位上限设多少"
    + _NOT_OTHERS
    + r"仓位\s*(?:应该|应当|该|最好|大概|一般)?\s*(?:控制在|放在|保持在|设在|设|定在|定|留|配|配置|占|放|上限|下限)?\s*"
    r"(?:" + _SHARE_OR_AMOUNT_ZH + r")[^，。？?！!；;]{0,8}?(?:" + _VERDICT_ZH + r"|$|[？?])|"
    r"(?:几成|多少|多大|多重)(?:的)?仓位|" + _NOT_OTHERS + r"仓位(?:怎么|如何)(?:分配|配置|安排|控制)|"
    # "把仓位的三成放宁德时代行吗", "三成仓位放宁德时代行吗", "拿一半的钱买比亚迪合适吗", "资金的30%投招行可以吗"
    r"(?:仓位|资金|钱|积蓄|存款|资产|本金|组合|持仓|账户|身家|家当)的?\s*" + _PORTION_ZH + r"(?:的)?\s*(?:仓位|资金|钱|积蓄|资产|"
    r"本金|持仓)?\s*" + _ALLOCATE_VERB_ZH + r"[^，。？?！!；;]{0,14}?(?:" + _VERDICT_ZH + r")|"
    + _PORTION_ZH
    + r"(?:的)?\s*(?:仓位|资金|钱|积蓄|资产|本金|持仓)\s*"
    + _ALLOCATE_VERB_ZH
    + r"[^，。？?！!；;]{0,14}?(?:"
    + _VERDICT_ZH
    + r")|"
    r"(?:重仓|满仓|半仓|全仓|轻仓)[^，。？?！!；;]{0,14}?(?:行吗|行不行|可以吗|合适吗|好吗|合理吗|怎么样)|"
    # "分配多少钱给招行合适", "该拿多少钱买"
    r"(?:分配|分|拿|拿出|留)\s*(?:多少|几)(?:钱|资金|万|成)?[^，。？?！!；;]{0,10}?(?:买|投|配置|放)[^，。？?！!；;]{0,10}?"
    r"(?:" + _VERDICT_ZH + r"|$|[？?])|"
    # English: "how many shares should I buy", "how much should I invest in BYD", "put 30% of my portfolio in CATL",
    # "what percentage of my portfolio should go into ...", "position size", "how big a position"
    r"\bhow\s+many\s+(?:shares|lots|units)\b[^.?!]{0,30}?\b(?:should|shall|would\s+you\s+recommend|to)\b[^.?!]{0,20}?"
    r"\b(?:i|we|one)?\s*(?:buy|purchase|hold|own|get|pick\s+up|add)\b|"
    r"\bhow\s+much\b[^.?!]{0,30}?\b(?:should|shall|do\s+you\s+recommend|would\s+you\s+recommend)\s+(?:i|we|one)\s+"
    r"(?:invest|put|allocate|spend|buy|commit|stake|bet)\b|"
    r"\b(?:put|allocate|invest|place|keep|hold|commit|stake|move|park)\s+(?:about\s+|around\s+|roughly\s+|over\s+|"
    r"up\s+to\s+)?(?:\d+(?:\.\d+)?\s*(?:%|percent|per\s+cent)|half|a\s+third|a\s+quarter|(?:all|most)\s+of)\s+(?:of\s+)?"
    r"(?:my|our|the|a|your)\s+(?:\w+\s+)?(?:portfolio|savings|money|capital|funds|cash|assets|net\s+worth|account)\b|"
    r"\b(?:what|which|how\s+much\s+of\s+a)\s+(?:percentage|percent|share|portion|proportion|fraction|weight(?:ing)?)\s+"
    r"of\s+(?:my|our|a)\s+(?:\w+\s+)?(?:portfolio|savings|money|capital|funds|assets)\b|"
    r"\bposition[- ]siz(?:e|es|ing)\b|\bhow\s+(?:big|large|much)\s+(?:of\s+)?a\s+position\b|"
    r"\b(?:how\s+much|how\s+big)\b[^.?!]{0,20}?\b(?:weight|allocation|exposure)\b[^.?!]{0,30}?\b(?:should|to\s+give)\b",
    re.IGNORECASE,
)



def asks_position_size(query: str) -> bool:
    """(round 17, review-13 M11) The question asks how many shares to buy or how much of a portfolio to put in."""
    return bool(POSITION_SIZING.search(query or ""))
