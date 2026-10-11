"""(round 17, review-13 M12a) Definitional clauses in a claim, checked against the glossary's formulas.

A claim may state what a metric is next to a value of it: "ROE等于净利润除以营业收入，紫金矿业ROE为27.91%". Before
round 17 the definition was an unchecked clause, so a false definition next to a true value made the claim
``supported``. Here a definition is read as a ratio of two quantities (numerator, denominator) and compared with the
formula of the metric's glossary entry (``glossary.py``: ROE is 归母净利润 ÷ 归母净资产, the P/E is price ÷ EPS or
market value ÷ net profit, ...). A threshold reading of the P/B ("市净率低于1说明股价低于每股净资产", "破净是指市净率
低于1") is a definition too: below book value is a P/B under 1.

Every definition found here is one check (``kind="definition"``) that cites the glossary entry
(``glossary_<term>``); the text of the definition is blanked before the numbers are read, so its "1" or its metric
words are never a value of the company named later. Only the forms below are read; a definition in other words stays an
unchecked clause as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .glossary import entry_for_term

# The metrics a claim may define, with their glossary term and the (numerator, denominator) pairs the glossary gives.
_FORMULAS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "roe": ("ROE", (("net_profit", "equity"),)),
    "pe_ttm": ("市盈率", (("price", "eps"), ("market_cap", "net_profit"))),
    "pb": ("市净率", (("price", "bvps"), ("market_cap", "equity"))),
    "gross_margin": ("毛利率", (("gross_profit", "revenue"),)),
    "net_margin": ("净利率", (("net_profit", "revenue"),)),
    "eps": ("每股收益", (("net_profit", "shares"),)),
}
_GLOSSARY_FORMULA = {
    "roe": ("归母净利润 ÷ 归母净资产", "attributable net profit / attributable equity"),
    "pe_ttm": ("股价 ÷ 每股收益（= 总市值 ÷ 归母净利润）", "share price / EPS (= market value / net profit)"),
    "pb": (
        "股价 ÷ 每股净资产（= 总市值 ÷ 归母净资产）",
        "share price / book value per share (= market value / equity)",
    ),
    "gross_margin": ("毛利润 ÷ 营业收入", "gross profit / revenue"),
    "net_margin": ("净利润 ÷ 营业收入", "net profit / revenue"),
    "eps": ("归母净利润 ÷ 普通股股数", "attributable net profit / ordinary shares"),
}
# What a stated pair is, when it is the formula of another ratio (said in the note of a false definition).
_KNOWN_PAIRS = {
    ("net_profit", "assets"): ("总资产收益率（ROA）", "return on assets (ROA)"),
    ("net_profit", "revenue"): ("净利率", "net margin"),
    ("net_profit", "equity"): ("ROE", "ROE"),
    ("net_profit", "shares"): ("每股收益", "EPS"),
    ("gross_profit", "revenue"): ("毛利率", "gross margin"),
    ("price", "eps"): ("市盈率", "P/E"),
    ("market_cap", "net_profit"): ("市盈率", "P/E"),
    ("price", "bvps"): ("市净率", "P/B"),
    ("market_cap", "equity"): ("市净率", "P/B"),
    ("market_cap", "revenue"): ("市销率", "P/S"),
    ("price", "revenue_ps"): ("市销率", "P/S"),
    ("debt", "assets"): ("资产负债率", "debt-to-assets"),
}
_LABEL = {
    "net_profit": ("净利润", "net profit"),
    "equity": ("净资产", "equity"),
    "assets": ("总资产", "total assets"),
    "revenue": ("营业收入", "revenue"),
    "gross_profit": ("毛利润", "gross profit"),
    "price": ("股价", "share price"),
    "market_cap": ("总市值", "market value"),
    "eps": ("每股收益", "EPS"),
    "bvps": ("每股净资产", "book value per share"),
    "shares": ("股数", "shares"),
    "cost": ("营业成本", "cost of revenue"),
    "debt": ("负债", "liabilities"),
    "revenue_ps": ("每股营收", "revenue per share"),
}

# Quantity words, longest forms first within each line (the alternation is tried in order at one position).
_QUANTITIES: tuple[tuple[str, str], ...] = (
    ("gross_profit", r"(?:营业)?收入\s*(?:减去?|−|－|-)\s*(?:营业)?成本|毛利润|毛利(?!率)|gross profit"),
    ("eps", r"每股收益|每股盈利|每股利润|每股净利润?|(?<![A-Za-z])EPS(?![A-Za-z])|earnings per share"),
    (
        "bvps",
        r"每股净资产|每股账面价值|每股账面净资产|(?<![A-Za-z])BVPS(?![A-Za-z])|(?:book value|net assets) per share",
    ),
    ("revenue_ps", r"每股营收|每股营业收入|每股销售额|(?:revenue|sales) per share"),
    (
        "net_profit",
        r"归母净利润|归属于(?:母公司|上市公司)(?:股东|所有者)?的净利润|净利润|净利(?!率)|税后利润|"
        r"net (?:profit|income|earnings)(?: attributable to shareholders)?|earnings(?! per)|profits?(?! margin)",
    ),
    ("assets", r"总资产|资产总额|资产总计|全部资产|total assets|assets"),
    (
        "equity",
        r"归母净资产|归属于(?:母公司|上市公司)(?:股东|所有者)?的?(?:净资产|权益)|净资产(?!收益)|股东权益|所有者权益|"
        r"账面净资产|账面价值|shareholders['’]? equity|(?:total )?equity|book value|net assets",
    ),
    ("revenue", r"营业总收入|营业收入|营收|销售收入|销售额|收入|total revenue|revenues?|sales"),
    ("market_cap", r"总市值|市值|market (?:value|cap(?:itali[sz]ation)?)"),
    ("price", r"每股股价|股票价格|股价|价格|收盘价|(?:share|stock) price|price"),
    ("shares", r"总股本|普通股股数|股本|股数|发行在外的?普通股(?:股数)?|(?:number of )?shares(?: outstanding)?"),
    ("cost", r"营业成本|成本|cost of (?:revenue|sales|goods sold)"),
    ("debt", r"总负债|负债总额|负债|total liabilities|liabilities|debt"),
)
_QUANTITY = "|".join(f"(?:{pattern})" for _name, pattern in _QUANTITIES)
_QUANTITY_LOOKUP = tuple((name, re.compile(rf"(?:{pattern})$", re.I)) for name, pattern in _QUANTITIES)
_FILLER_ZH = r"(?:用|以|拿|将|公司(?:的)?|当期|期末|期初|报告期|加权平均(?:的)?|平均(?:的)?|全部|一家公司(?:的)?)*"
_FILLER_EN = r"(?:(?:the|a|its|their|company's|a company's|firm's|average|period-end|current|annual|weighted)\s+)*"
_Q = rf"{_FILLER_ZH}{_FILLER_EN}(?P<{{name}}>{_QUANTITY})"

_HEADS: tuple[tuple[str, str], ...] = (
    ("roe", r"(?<![A-Za-z])ROE(?![A-Za-z])|净资产收益率|return on equity"),
    (
        "pe_ttm",
        r"市盈率(?:TTM|\s*[（(]TTM[）)])?|(?<![A-Za-z])P/?E(?: ratio)?(?![A-Za-z])|price[- ]to[- ]earnings(?: ratio)?",
    ),
    ("pb", r"市净率|(?<![A-Za-z])P/?B(?: ratio)?(?![A-Za-z])|price[- ]to[- ]book(?: ratio)?"),
    ("gross_margin", r"销售毛利率|毛利率|gross (?:profit )?margin"),
    ("net_margin", r"销售净利率|净利润率|净利率|net (?:profit )?margin"),
    ("eps", r"每股收益|(?<![A-Za-z])EPS(?![A-Za-z])|earnings per share"),
)
_HEAD = "|".join(f"(?:{pattern})" for _name, pattern in _HEADS)
_HEAD_LOOKUP = tuple((name, re.compile(rf"(?:{pattern})$", re.I)) for name, pattern in _HEADS)

_IS_ZH = (
    r"(?:的(?:计算)?(?:公式|方法|算法|定义)\s*(?:是|为)?|的计算方式是)?\s*(?:就|即|也就)?\s*"
    r"(?:等于|是指|指的是|就是|定义为|即为|是|为|指|即|=|＝|：|:)"
)
_DIV_ZH = r"\s*(?:除以|÷|/|／)\s*"
_PAIR_ZH = r"\s*(?:与|和|跟|同|对)\s*"
_RATIO_WORD_ZH = r"\s*(?:之比|的比值|比值|的比率|的比例|相比|的比|相除(?:的结果)?|的商)"
_SHARE_ZH = r"\s*(?:占|在)\s*"
_SHARE_WORD_ZH = r"\s*(?:中|当中)?\s*(?:所占)?\s*(?:的)?\s*(?:比例|比重|百分比|占比)"
_GIVES_ZH = r"\s*(?:得到|算出|所得|得出|计算出)?\s*(?:的)?\s*(?:结果)?\s*(?:就是|即为|便是|即|等于|=|是)\s*"
_IS_EN = r"\s+(?:ratio\s+)?(?:is|equals|=|means|is calculated as|is defined as|is computed as|measures)\s+"
_DIV_EN = r"\s+(?:divided by|over|/|relative to)\s+"
_SHARE_EN = r"\s+as an? (?:share|percentage|proportion|fraction) of\s+"


def _q(name: str) -> str:
    return _Q.format(name=name)


# ROE等于净利润除以净资产 / 市盈率=股价/每股收益 / ROE是净利润与净资产之比 / 毛利率是毛利占营业收入的比重
_FORM_IS_ZH = (
    rf"(?P<head>{_HEAD})\s*{_IS_ZH}\s*{_q('a')}(?:{_DIV_ZH}{_q('b')}|{_PAIR_ZH}{_q('b2')}{_RATIO_WORD_ZH}|"
    rf"{_SHARE_ZH}{_q('b3')}{_SHARE_WORD_ZH})"
)
# 用净利润除以净资产得到的就是ROE
_FORM_GIVES_ZH = rf"{_q('a')}{_DIV_ZH}{_q('b')}{_GIVES_ZH}(?P<head>{_HEAD})"
# P/E is the share price divided by earnings per share; gross margin is gross profit as a share of revenue; ROE is the
# ratio of net income to equity
_FORM_IS_EN = (
    rf"(?P<head>{_HEAD}){_IS_EN}(?:{_q('a')}(?:{_DIV_EN}{_FILLER_EN}{_q('b')}|{_SHARE_EN}{_FILLER_EN}{_q('b3')})|"
    rf"the ratio of\s+{_q('a2')}\s+to\s+{_FILLER_EN}{_q('b2')})"
)
_FORMS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.I) for pattern in (_FORM_IS_ZH, _FORM_GIVES_ZH, _FORM_IS_EN)
)

# Price against book value as a definition of a P/B threshold, in either order, and 破净 defined
_BELOW = r"低于|小于|不到|不足|跌破|<|＜"
_ABOVE = r"高于|大于|超过|>|＞"
_PB = r"(?:市净率|(?<![A-Za-z])P/?B(?![A-Za-z]))"
_PRICE = r"(?:公司的?)?(?:股价|股票价格|每股股价|价格|市值|总市值)"
_BOOK = r"(?:每股净资产|每股账面价值|净资产|账面价值|账面净资产)(?!收益)"
_MEANS = r"(?:时)?\s*[，,]?\s*(?:就|则)?\s*(?:意味着|说明|表示|代表|表明|即|就是|等于说|是指|指的是)\s*"
_SIDE_EN = r"below|under|less than|lower than|above|over|more than|greater than|higher than"
_THRESHOLD_PATTERNS = (
    rf"{_PB}\s*(?P<c1>{_BELOW}|{_ABOVE})\s*(?:1|一)\s*倍?{_MEANS}{_PRICE}\s*(?:已经|已)?\s*"
    rf"(?P<c2>{_BELOW}|{_ABOVE})\s*(?:了)?\s*{_BOOK}",
    rf"{_PRICE}\s*(?P<c2>{_BELOW}|{_ABOVE})\s*(?:了)?\s*{_BOOK}{_MEANS}{_PB}\s*(?P<c1>{_BELOW}|{_ABOVE})"
    r"\s*(?:1|一)(?![\d.])",
    rf"(?:所谓)?破净\s*[，,]?\s*(?:就|也就)?\s*(?:是指|指的是|就是|即|意味着|是说|指|是)\s*(?:{_PB}\s*"
    rf"(?P<c1>{_BELOW}|{_ABOVE})\s*(?:1|一)(?![\d.])|{_PRICE}\s*(?P<c2>{_BELOW}|{_ABOVE})\s*(?:了)?\s*{_BOOK})",
    rf"(?:an?\s+)?(?:P/?B|price[- ]to[- ]book(?: ratio)?)\s+(?:ratio\s+)?(?:of\s+)?(?P<c1>{_SIDE_EN})\s+(?:1|one)\b"
    r"\s*(?:means|implies|indicates|shows|signals|tells you)\s+(?:that\s+)?(?:the\s+)?"
    rf"(?:share price|stock price|price|market value)\s+(?:is\s+)?(?P<c2>{_SIDE_EN})\s+"
    r"(?:the\s+|its\s+)?(?:book value|net assets)(?:\s+per share)?",
)
_THRESHOLD_FORMS: tuple[re.Pattern[str], ...] = tuple(re.compile(pattern, re.I) for pattern in _THRESHOLD_PATTERNS)
_DOWN_WORDS = re.compile(r"低于|小于|不到|不足|跌破|<|＜|below|under|less than|lower than", re.I)
_BOOK_FORMULA = (
    "破净即股价低于每股净资产，也就是市净率低于 1（glossary_市净率）",
    "trading below book value is a share price under book value per share, i.e. a P/B under 1 (glossary_市净率)",
)


@dataclass(frozen=True)
class Definition:
    start: int
    end: int
    metric: str
    term: str
    status: Literal["supported", "contradicted"]
    note: str


def _quantity(word: str) -> str | None:
    word = word.strip()
    for name, pattern in _QUANTITY_LOOKUP:
        if pattern.match(word):
            return name
    return None


def _head(word: str) -> str | None:
    for name, pattern in _HEAD_LOOKUP:
        if pattern.match(word.strip()):
            return name
    return None


def _pair_label(pair: tuple[str, str], *, zh: bool) -> str:
    index = 0 if zh else 1
    return f"{_LABEL[pair[0]][index]} ÷ {_LABEL[pair[1]][index]}"


def _ratio_definition(match: re.Match[str], *, zh: bool) -> Definition | None:
    groups = match.groupdict()
    metric = _head(groups["head"] or "")
    numerator = groups.get("a") or groups.get("a2")
    denominator = groups.get("b") or groups.get("b2") or groups.get("b3")
    if metric is None or not numerator or not denominator:
        return None
    pair = (_quantity(numerator), _quantity(denominator))
    if pair[0] is None or pair[1] is None or pair[0] == pair[1]:
        return None
    a, b = pair
    term, accepted = _FORMULAS[metric]
    if entry_for_term(term) is None:  # the glossary must carry the entry the check cites
        return None
    formula = _GLOSSARY_FORMULA[metric][0 if zh else 1]
    stated = _pair_label((a, b), zh=zh)
    if (a, b) in accepted:
        return Definition(match.start(), match.end(), metric, term, "supported", f"definition: {stated} = {formula}")
    known = _KNOWN_PAIRS.get((a, b))
    if (b, a) in accepted:
        what = "the formula inverted"
    elif known is not None:
        what = f"that is {known[0 if zh else 1]}"
    else:
        what = ""
    note = f"definition: the claim says {stated}{f' ({what})' if what else ''}; the glossary (glossary_{term}) defines "
    note += f"it as {formula}"
    return Definition(match.start(), match.end(), metric, term, "contradicted", note)


def _threshold_definition(match: re.Match[str], *, zh: bool) -> Definition:
    groups = match.groupdict()
    signs = []
    for key in ("c1", "c2"):
        if groups.get(key):
            signs.append(bool(_DOWN_WORDS.fullmatch(groups[key].strip())))
    # 破净 is below book value: every stated side must be "below"; otherwise the two sides must agree
    holds = all(signs) if "破净" in match.group(0) else len(signs) == 2 and signs[0] == signs[1]
    note = f"definition: {_BOOK_FORMULA[0 if zh else 1]}"
    return Definition(match.start(), match.end(), "pb", "市净率", "supported" if holds else "contradicted", note)


def read_definitions(plain: str, *, zh: bool = True) -> list[Definition]:
    """Every definitional clause in the normalised claim, checked against the glossary; non-overlapping, in order."""
    found: list[Definition] = []
    for pattern in _THRESHOLD_FORMS:
        for match in pattern.finditer(plain):
            found.append(_threshold_definition(match, zh=zh))
    for pattern in _FORMS:
        for match in pattern.finditer(plain):
            if any(item.start < match.end() and match.start() < item.end for item in found):
                continue
            definition = _ratio_definition(match, zh=zh)
            if definition is not None:
                found.append(definition)
    return sorted(found, key=lambda item: item.start)


def blank_definitions(text: str, definitions: list[Definition]) -> str:
    """``text`` with every definition's characters blanked (clause punctuation kept), so its words and numbers are not
    read as a value of the company named elsewhere."""
    if not definitions:
        return text
    chars = list(text)
    for item in definitions:
        for index in range(item.start, min(item.end, len(chars))):
            if chars[index] not in "，,。；;！!？?\n":
                chars[index] = " "
    return "".join(chars)
