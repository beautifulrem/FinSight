"""A small curated glossary of A-share market concepts.

Questions such as "北向资金是啥" or "什么是两融" ask about a market mechanism, not about a security. The classical
NLU has no entity for them, so they used to be refused as out of scope (round-3 item C6). The entries below are
curated text, committed with the code, and they are handed to the agent as ordinary evidence (the
``explain_concept`` tool): every answer built from one cites ``glossary_<term>``, so the definition is traceable
like any other evidence item.

The glossary explains concepts only. FinSight has no data series for most of them, and the tool says so, so a
question asking for a number ("北向资金今天净流入多少") gets the definition plus a clear statement that the number
is not in the data, never an invented figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GlossaryEntry:
    """One curated concept: ``term`` is the headword, ``aliases`` other spellings used in questions."""

    term: str
    zh: str
    en: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    has_data_series: bool = False
    #: The word also has an everyday meaning ("国家队队员名单" is about sport): it needs a definition or market cue.
    needs_cue: bool = False

    @property
    def evidence_id(self) -> str:
        return f"glossary_{self.term}"

    @property
    def surface_forms(self) -> tuple[str, ...]:
        return (self.term, *self.aliases)


GLOSSARY: tuple[GlossaryEntry, ...] = (
    GlossaryEntry(
        term="北向资金",
        aliases=("北上资金", "northbound funds", "northbound capital"),
        zh=(
            "北向资金指境外投资者通过沪股通、深股通买卖A股的资金，是沪深港通机制中从香港流入内地市场的方向；"
            "与之相对的南向资金指内地资金通过港股通买卖港股。市场常用其净买入额观察外资态度。"
        ),
        en=(
            "Northbound funds are overseas money buying and selling A-shares through the Shanghai and Shenzhen "
            "Stock Connect, i.e. the Hong Kong-to-mainland direction of Stock Connect; southbound funds are the "
            "opposite direction. Their net buying is often read as a gauge of foreign appetite."
        ),
    ),
    GlossaryEntry(
        term="南向资金",
        aliases=("南下资金", "southbound funds"),
        zh="南向资金指内地投资者通过港股通买卖港股的资金，是沪深港通机制中从内地流向香港市场的方向。",
        en=(
            "Southbound funds are mainland money buying and selling Hong Kong stocks through Stock Connect, the "
            "mainland-to-Hong Kong direction of the scheme."
        ),
    ),
    GlossaryEntry(
        term="融资融券",
        aliases=("两融", "margin trading", "margin financing"),
        zh=(
            "融资融券是券商向投资者出借资金买入证券（融资）或出借证券供其卖出（融券）的信用交易业务，俗称两融。"
            "融资余额反映用借来的钱买股的规模，融券余额反映借券做空的规模；两者合称两融余额，是常用的杠杆指标。"
        ),
        en=(
            "Margin trading and securities lending is credit trading in which a broker lends cash to buy "
            "securities (margin financing) or lends securities to sell (securities lending). The margin balance "
            "shows how much stock is held with borrowed money and is a common gauge of market leverage."
        ),
    ),
    GlossaryEntry(
        term="国家队",
        aliases=("平准基金", "national team"),
        zh=(
            "国家队是市场对具有稳定市场职能的国有背景资金的统称，通常指中央汇金、中国证券金融公司等机构，"
            "以及它们持有或增持的宽基ETF。它不是一个法律概念，也没有公开的统一持仓口径。"
        ),
        en=(
            'The "national team" is a market nickname for state-linked funds with a market-stabilising role, '
            "typically Central Huijin and China Securities Finance, along with the broad-based ETFs they hold. It "
            "is not a legal category and has no single published holdings definition."
        ),
        needs_cue=True,
    ),
    GlossaryEntry(
        term="涨跌停",
        aliases=("涨停", "跌停", "price limit", "limit up", "limit down"),
        zh=(
            "涨跌停是A股的单日价格涨跌幅限制：价格达到上限称为涨停，达到下限称为跌停。限制幅度按板块和股票类型"
            "不同（主板最窄，创业板和科创板更宽，风险警示股更窄），新股上市初期另有规则。"
        ),
        en=(
            'A-shares have a daily price limit: hitting the upper bound is "limit up", the lower bound "limit '
            'down". The width depends on the board and the type of share (narrowest on the main board, wider on '
            "ChiNext and the STAR Market, narrower for risk-flagged shares), with separate rules for new listings."
        ),
    ),
    GlossaryEntry(
        term="换手率",
        aliases=("turnover rate",),
        zh="换手率是一段时间内成交股数占流通股本的比例，用来衡量交易活跃度；同一只股票的换手率高低只能与自身历史或同业比较。",
        en=(
            "The turnover rate is the number of shares traded in a period divided by the free-float share count, a "
            "measure of trading activity that is only meaningful against the same stock's history or its peers."
        ),
    ),
    GlossaryEntry(
        term="ST股",
        aliases=("st股", "风险警示股", "退市风险警示"),
        zh=(
            "ST是交易所对财务或其他状况异常公司加注的风险警示标记，加注后股票简称前带ST；*ST表示存在退市风险。"
            "此类股票的涨跌幅限制更严格，风险显著高于普通股票。"
        ),
        en=(
            "ST marks a company whose financial or other condition the exchange flags as abnormal; *ST marks "
            "delisting risk. Such shares carry tighter daily price limits and materially higher risk."
        ),
    ),
    GlossaryEntry(
        term="打新",
        aliases=("新股申购", "IPO subscription"),
        zh="打新指参与新股发行的申购。A股新股申购按市值配售，中签后才能认购，中签率通常很低。",
        en=(
            '"Playing the new" means subscribing to IPOs. A-share subscriptions are allocated by the market value '
            "of holdings and only successful lots can be bought; hit rates are typically very low."
        ),
    ),
    GlossaryEntry(
        term="限售解禁",
        aliases=("解禁", "lock-up expiry"),
        zh="限售解禁指首发或定增等原因形成的限售股到期可以上市流通。解禁本身只增加可流通股份，并不必然带来抛售。",
        en=(
            "A lock-up expiry is the date restricted shares from an IPO or placement become tradable. It increases "
            "the tradable float; it does not by itself mean the shares will be sold."
        ),
    ),
    GlossaryEntry(
        term="龙虎榜",
        aliases=("龙虎榜数据", "public trading list"),
        zh="龙虎榜是交易所对异常波动或成交活跃的股票公布的买卖前几名营业部席位明细，常被用来观察短线资金行为。",
        en=(
            'The "dragon-tiger list" is the exchange disclosure of the top buying and selling brokerage seats for '
            "stocks with unusual moves or heavy turnover, often read as a window on short-term flows."
        ),
    ),
    GlossaryEntry(
        term="主力资金",
        aliases=("主力", "大单资金", "smart money"),
        zh=(
            "主力资金是市场对大额买卖单背后资金的统称，通常按成交单笔金额划分为大单、中单、小单后统计净额。"
            "它是数据商的口径，不是监管定义，不同数据源的结果可能不同。"
        ),
        en=(
            '"Main-force money" is a market label for the money behind large orders: vendors bucket trades by '
            "order size and net them. It is a data-vendor convention, not a regulatory definition, so different "
            "sources disagree."
        ),
        needs_cue=True,
    ),
    GlossaryEntry(
        term="科创板",
        aliases=("STAR Market",),
        zh="科创板是上海证券交易所面向科技创新企业的板块，实行注册制，日常涨跌幅限制宽于主板。",
        en=(
            "The STAR Market is the Shanghai exchange's board for technology companies, with a registration-based "
            "listing system and a wider daily price limit than the main board."
        ),
    ),
    GlossaryEntry(
        term="北交所",
        aliases=("北京证券交易所", "Beijing Stock Exchange"),
        zh="北交所是服务创新型中小企业的北京证券交易所，由新三板精选层改制而来，交易规则与沪深主板不同。",
        en=(
            "The Beijing Stock Exchange serves innovative small and medium-sized companies; it grew out of the "
            "NEEQ select tier and its trading rules differ from the Shanghai and Shenzhen main boards."
        ),
    ),
    GlossaryEntry(
        term="沪深港通",
        aliases=("沪港通", "深港通", "Stock Connect"),
        zh="沪深港通是内地与香港市场的互联互通机制，包含沪股通、深股通（北向）和港股通（南向）。",
        en=(
            "Stock Connect links the mainland and Hong Kong markets: the Shanghai and Shenzhen Connect legs "
            "(northbound) and the Hong Kong Connect leg (southbound)."
        ),
    ),
    GlossaryEntry(
        term="大宗交易",
        aliases=("block trade",),
        zh="大宗交易是达到规模标准的证券在盘后按约定价格成交的方式，成交价常与收盘价有折价或溢价，不计入当日连续竞价。",
        en=(
            "A block trade is an after-hours negotiated transaction above a size threshold, often at a discount or "
            "premium to the close, and it is not part of the day's continuous auction."
        ),
    ),
    GlossaryEntry(
        term="注册制",
        aliases=("registration-based IPO",),
        zh="注册制是发行上市审核方式：交易所审核、证监会注册，信息披露责任落在发行人和中介机构，与核准制相对。",
        en=(
            "Under the registration-based system the exchange reviews a listing and the CSRC registers it, with "
            "disclosure responsibility on the issuer and its intermediaries, as opposed to the older approval "
            "system."
        ),
    ),
)

_BY_FORM: dict[str, GlossaryEntry] = {}
for _entry in GLOSSARY:
    for _form in _entry.surface_forms:
        _BY_FORM.setdefault(_form.lower(), _entry)
# Longest first: "融资融券" before "两融", "沪深港通" before "沪港通".
_FORMS = sorted(_BY_FORM, key=len, reverse=True)
_LATIN_FORM = re.compile(r"^[a-z0-9 .-]+$")
# A term with an everyday meaning counts only in a question that asks what it means, or that is about the market.
_CUE = re.compile(
    r"是什么|是啥|什么意思|啥意思|什么叫|指的是|是指|含义|定义|概念|怎么理解|如何理解|怎么看|"
    r"股|基金|etf|指数|资金|市场|行情|大盘|估值|仓位|净流入|净买入|买入|卖出|余额|机构|"
    r"\bwhat (?:is|are|does)\b|\bmean\b|\bdefinition\b|\bstocks?\b|\bshares?\b|\bmarket\b|\bfunds?\b|\bequities\b",
    re.IGNORECASE,
)


def lookup_concept(text: str) -> GlossaryEntry | None:
    """The glossary entry a question is about, or ``None``.

    A CJK form must appear in the question; a Latin form must appear as whole words ("national team" but not
    inside a longer word), so ordinary sentences do not match by accident. Terms that also have an everyday
    meaning (``needs_cue``) additionally require a definition question or market wording, so "国家队队员名单"
    (a sports question) matches nothing while "国家队是指什么" does.
    """
    lowered = (text or "").lower()
    if not lowered:
        return None
    for form in _FORMS:
        entry = _BY_FORM[form]
        if _LATIN_FORM.match(form):
            if not re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", lowered):
                continue
        elif form not in lowered:
            continue
        if entry.needs_cue and not _CUE.search(lowered.replace(form, " ")):
            continue
        return entry
    return None


def concept_terms() -> tuple[str, ...]:
    return tuple(entry.term for entry in GLOSSARY)
