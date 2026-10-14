"""(round 17, review-13 M3) A short name of a covered company followed by a business-line word names another company.

"美的置业" (a Hong Kong-listed property developer), "中芯集成" (another STAR-market company), "平安证券" (an unlisted
subsidiary) and "Midea Real Estate" are absent from the entity master, but each starts with a short name of a covered
company (美的 / 中芯 / 平安 / Midea). Before round 17 the short name matched exactly and the answer gave the covered
company's figures. A short name (not the official name) followed by one of the words below is a *lookalike name*:
it is not resolved to the covered company, and the question is answered as naming a company FinSight does not cover.

Exceptions, so a descriptive word keeps its company:
* the extended name is the covered company's own name or starts it (中芯 + 国际 = 中芯国际, 中信 + 证券 = 中信证券);
* the word names the company's own industry ("平安保险" is 中国平安, an insurer; "招行银行" is 招商银行): the word's
  family (保险 / 人寿 / 寿险 / 财险, 银行, 证券 / 券商) appears in the company's name or industry;
* legal-form words (集团, 股份, 有限公司) are not business-line words: "比亚迪股份有限公司" is 比亚迪.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any

# Business-line words that make a short name another company's name (longest first in the pattern).
ZH_SUFFIXES = (
    "置业", "置地", "房地产", "地产", "物业", "集成", "证券", "银行", "保险", "人寿", "财险", "寿险", "信托", "期货", "基金",
    "资本", "资管", "租赁", "控股", "投资", "国际", "电讯", "电信", "基建", "实业", "发展", "健康", "医疗", "电子", "物流",
    "金融", "好医生", "理财",
)  # fmt: skip
# Words that may follow a business-line word inside the same name ("美的置业控股", "中信国际电讯集团").
_ZH_TAIL = ("控股", "集团", "股份有限公司", "有限公司", "股份", "国际", "电讯", "医疗", "科技", "公司")
EN_SUFFIXES = (
    r"real\s+estate", r"propert(?:y|ies)", r"securities", r"bank(?:ing)?", r"insurance", r"life", r"holdings?",
    r"international", r"telecom(?:munications)?", r"electronics?", r"capital", r"land", r"development",
    r"leasing", r"trust", r"investments?", r"asset\s+management", r"futures", r"logistics", r"health(?:care)?",
    r"good\s+doctor", r"financial",
)  # fmt: skip
_ZH_PATTERN = re.compile(
    "(?:" + "|".join(sorted(ZH_SUFFIXES, key=len, reverse=True)) + ")(?:" + "|".join(_ZH_TAIL) + ")*"
)
_EN_PATTERN = re.compile(r"\s+(?:" + "|".join(EN_SUFFIXES) + r")\b(?:\s+(?:holdings?|group|limited|ltd\.?))?", re.I)
# A word that names the company's own industry is a descriptor, not another company.
_FAMILIES = (
    ("保险", "人寿", "寿险", "财险", "insurance", "life"),
    ("银行", "bank", "banking"),
    ("证券", "券商", "securities"),
)


def _family(word: str) -> tuple[str, ...] | None:
    lowered = word.lower()
    return next((family for family in _FAMILIES if any(lowered.startswith(term) for term in family)), None)


def _describes(word: str, owner: str, industry: str) -> bool:
    """The business-line word names the owner's own industry, or extends into the owner's own name."""
    family = _family(word)
    return bool(family) and any(term in owner or term in industry for term in family if not term.isascii())


def lookalike_end(
    text: str,
    end: int,
    *,
    alias: str,
    owner: str,
    industry: str = "",
    names_at: Callable[[str, int], int] | None = None,
) -> int | None:
    """Where a lookalike name that starts with ``alias`` (ending at ``end`` in ``text``) ends, or ``None``.

    ``owner`` is the covered company's canonical (Chinese) name, ``industry`` its industry name; ``names_at(text, i)``
    is the length of the longest name of a listed security starting at ``i`` (0 for none): "茅台五粮液" lists two
    companies. Inside an unknown name only a name of three or more characters ends it ("宁德新能源" is one name,
    although 能源 is a short name of 百川能源)."""
    rest = text[end:]
    if alias.isascii():
        match = _EN_PATTERN.match(rest)
        if match is not None:
            word = match.group(0).strip().split()[0]
            if word[:1].islower() and not text.islower():
                # a name is capitalised ("Midea Real Estate"); "where does BYD land?" and "CATL international sales"
                # use the word as an ordinary word (an all-lower-case question is read as written)
                return None
            return None if _describes(word, owner, industry) else end + match.end()
        return _unknown_extension_end_en(text, end, owner=owner, industry=industry, names_at=names_at)
    match = _ZH_PATTERN.match(rest)
    if match is not None:
        word = next(suffix for suffix in sorted(ZH_SUFFIXES, key=len, reverse=True) if rest.startswith(suffix))
        if owner.startswith(alias + word) or (alias + word) in owner or _describes(word, owner, industry):
            return None
        return end + match.end()
    return _unknown_extension_end_zh(text, end, alias=alias, owner=owner, industry=industry, names_at=names_at)


# (round 18, review-14 N5) Deny by default. The business-line words above were a word list: "紫金黄金" (Zijin Gold, a
# Hong Kong company), "比亚迪半导体", "招银理财", "宁德新能源", "中芯南方", "平安科技" and "Zijin Gold" used a word
# outside it and were answered with the covered company's figures. Now a *name word* right after a listed company's
# name or alias makes a different, unknown company: a noun by jieba's dictionary, or any word that occurs in the name
# of a listed company (A-share entity master and HKEX-only list); in English a capitalised word. Exceptions: a word
# that can follow a company name in a question (below: grammar words, metrics, time words, verbs of the market, legal
# forms), the company's own name going on (紫金 + 矿业), a word naming its own industry (平安 + 保险), another listed
# name ("茅台五粮液"), and a longer alias of the entity table ("平安银行", "宁德时代新能源科技"), matched first. A
# verb, pronoun, time word or numeral after the name ("茅台我按1380买的", "换五粮液看看") is the rest of the question.
DENY_UNKNOWN_EXTENSIONS = True  # switched off only by ``evaluation/agent_eval/lookalike_regression.py`` ("before")
# grammar characters that do not begin a company-name word; those that do (能源, 多元, 可口, 有色, 同方, 比特, 至纯,
# 和讯 …) count only in their question forms, listed with the words below
_ZH_FUNCTION_CHARS = "的地得了吗嘛呢吧啊呀哦哈么和与跟及或比对是在这那哪该每各都也还又就才已曾将想怎咋几啥谁给让把被等们"
_ZH_QUESTION_FORMS = (
    "能买", "能不能", "能否", "能涨", "能跌", "能到", "能赚", "能拿", "能投", "能持", "能长", "能回", "能破", "能翻",
    "多少", "多大", "多高", "多久", "多长", "可以", "可不可以", "可能", "可靠", "要不要", "要买", "要卖", "要涨", "要跌",
    "为什么", "为何", "为啥", "如何", "如果", "何时", "会不会", "会涨", "会跌", "会被", "会有", "会继续", "会再", "会回",
    "会破", "会到", "有没有", "有什么", "有多", "有哪些", "有啥", "有无", "有何", "有望", "有些", "有点", "有关", "同比",
    "同期", "对比", "对于", "对应", "比去年", "比上", "比较", "比同", "比行业", "比其他", "比前", "从去年", "从上",
    "从今年", "到底", "到了", "到期", "到现在", "至今", "此前", "此外", "其中", "其他", "其实",
)  # fmt: skip
_ZH_CONTINUATIONS = (
    # legal forms and the company itself
    "集团", "股份", "公司", "有限", "本身", "自身", "旗下", "所属", "所在", "这家", "这只", "那家", "那只",
    # market data and metrics
    "股价", "股票", "股息", "股东", "股本", "股权", "股性", "股吧", "港股", "A股", "H股", "价格", "价位", "市值", "市盈", "市净",
    "市销", "市场", "收盘", "收入", "收益", "开盘", "最高", "最低", "成交", "换手", "涨", "跌", "走势", "行情", "表现", "估值",
    "营收", "营业", "利润", "净利", "净资", "毛利", "总市", "总资", "总营", "总股", "资产", "负债", "现金", "经营", "主营",
    "业绩", "财报", "财务", "年报", "季报", "中报", "半年", "一季", "二季", "三季", "四季", "季度", "全年", "年度", "分红",
    "派息", "派现", "送转", "配股", "增发", "回购", "减持", "增持", "解禁", "质押", "评级", "研报", "公告", "新闻", "消息",
    "动态", "情况", "基本面", "技术面", "资金", "主力", "北向", "融资", "融券", "龙头", "概念", "板块", "行业", "产量", "产能",
    "产品", "销量", "销售", "订单", "出货", "出口", "海外", "研发", "技术", "董事", "高管", "员工", "管理层", "实控", "ROE",
    "每股", "单季", "累计", "复权", "除权", "除息", "同比", "环比", "相比", "比较", "对比", "较", "占比", "合计", "加起来",
    "差", "排名", "权重", "仓位", "持仓", "持有", "买入", "卖出", "买", "卖", "加仓", "减仓", "清仓", "割肉", "抄底", "定投",
    "投资", "值得", "值不值", "贵", "便宜", "适合", "风险", "前景", "未来", "后市", "接下来", "今后", "将来", "预期", "预测",
    "目标", "上涨", "下跌", "大涨", "大跌", "暴涨", "暴跌", "反弹", "回调", "跳水", "震荡", "横盘", "企稳", "拉升", "冲高",
    "创新高", "创新低", "新高", "新低", "破发", "突破", "连续", "跑赢", "跑输", "好不好", "好吗", "怎么样", "真的", "到底",
    "究竟", "历史", "过去", "以来", "之前", "之后", "之间", "之类", "以上", "以下", "左右", "附近", "应该", "存款",
    "贷款", "规模", "保费", "不良", "拨备", "净息差", "息差", "资本", "偿付", "新业务", "内含价值", "手续费", "佣金",
    "两融", "自营", "经纪", "用户", "客户", "门店", "份额", "交付", "装机", "发电", "电量", "分析", "介绍", "简介",
    "成本", "估个价", "估价", "估一下", "伦敦上市", "美国上市", "香港上市", "瑞士上市", "新加坡上市", "法兰克福上市",
    "美国存托", "全球存托", "存托凭证", "存托股", "伦敦交易所", "数据", "信息", "资料", "代码", "简称", "全称",
    "合理", "现价", "流通", "目标价", "三者", "两者", "四者", "当中", "距离", "距", "影响", "领先", "落后", "贡献",
    "整体", "总体", "基本每股", "核心", "分别", "相对", "作为", "属于", "位于", "位居", "排在", "名列", "位列",
    "第一笔", "第二笔", "第三笔", "第一次", "第二次", "第几",
    # time words
    "今", "去年", "前年", "近", "最", "现在", "目前", "当前", "本周", "本月", "本季", "上周", "上月", "上个", "上半", "上市",
    "下周", "下月", "下个", "下半", "明天", "明年", "昨天", "昨日", "今天", "今日", "早盘", "午盘", "尾盘", "盘中", "盘后",
    "盘前", "年初", "年底", "年末", "月底", "一年", "两年", "三年", "五年", "十年", "一个", "几个", "多少", "多久",
    # spoken
    "呗", "咧", "哇", "嘞", "啦", "喽", "噢", "嗯",
)  # fmt: skip
_ZH_CONTINUATION = re.compile(
    "[" + _ZH_FUNCTION_CHARS + "]|"
    + "|".join(sorted(map(re.escape, (*_ZH_CONTINUATIONS, *_ZH_QUESTION_FORMS)), key=len, reverse=True))
)
# Where an unknown name ends: the same words. Characters that also start name words (能源, 多元, 可口, 有色) do not end
# it on their own; their question forms do (能买, 多少, 可以, 有没有).
_ZH_NAME_STOP = _ZH_CONTINUATION
_FAMILY_WORDS = tuple(term for family in _FAMILIES for term in family if not term.isascii())
_EN_CONTINUATIONS = frozenset(
    """co company corp corporation group ltd limited inc plc shares share stock stocks price prices revenue revenues
    sales profit profits net earnings market dividend dividends close closing open opening valuation value results
    report reports annual quarterly quarter year news performance outlook and or vs versus compared with without in on
    for from to is are was were has have had does do did will would should can could may might the a an of at by since
    today yesterday this last next now currently still also too right better worse bigger higher lower more less
    cheaper plus h""".split()
)
_EN_TITLE_WORD = re.compile(r"\s+([A-Z][a-z][A-Za-z'’-]*)")


def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿"


# A word that can be part of a company name: a noun (jieba's dictionary part of speech: n*, s, j, vn, an, nz), or any
# word of two or more characters that occurs in the name of a listed company (the A-share entity master and the HKEX
# list: 理财, 集成, 发展, 制造 … are verbs in the dictionary but name words in "招银理财", "中芯集成").
_NOUN_TAGS = ("n", "s", "j", "vn", "an")
_POS_WINDOW = 12


@lru_cache(maxsize=1)
def _pos_tokenizer() -> Any:
    import logging

    import jieba
    import jieba.posseg

    jieba.setLogLevel(logging.ERROR)
    return jieba.posseg.POSTokenizer(jieba.Tokenizer())


@lru_cache(maxsize=1)
def _name_words_by_first_char() -> dict[str, tuple[str, ...]]:
    index: dict[str, list[str]] = {}
    for word in _name_words():
        index.setdefault(word[0], []).append(word)
    return {char: tuple(words) for char, words in index.items()}


@lru_cache(maxsize=1)
def _name_words() -> frozenset[str]:
    from ..data_loader import load_entities, load_hk_only_names

    names = [str(row.get("canonical_name") or "") for row in load_entities() if row.get("entity_type") == "stock"]
    names += [str(row.get("name_zh") or "") for row in load_hk_only_names()]
    _pos_tokenizer()  # quiets jieba's log
    import jieba

    tokenizer = jieba.Tokenizer()
    words = {
        word
        for name in names
        for word in tokenizer.cut(name, HMM=False)
        if len(word) >= 2 and all(_is_cjk(char) for char in word)
    }
    return frozenset(word for word in words | set(ZH_SUFFIXES) if not _ZH_CONTINUATION.fullmatch(word))


def _starts_a_name_word(text: str, index: int) -> bool:
    if any(text.startswith(word, index) for word in _name_words_by_first_char().get(text[index], ())):
        return True
    first = next(iter(_pos_tokenizer().cut(text[index : index + _POS_WINDOW], HMM=False)), None)
    return first is not None and first.flag.startswith(_NOUN_TAGS)


def _unknown_extension_end_zh(
    text: str, end: int, *, alias: str, owner: str, industry: str, names_at: Callable[[str, int], int] | None
) -> int | None:
    if not DENY_UNKNOWN_EXTENSIONS or end >= len(text) or not _is_cjk(text[end]):
        return None
    if (alias + text[end]) in owner or (names_at is not None and names_at(text, end) > 0):
        return None  # the company's own name goes on (紫金 + 矿业), or another listed name follows
    if _ZH_CONTINUATION.match(text, end):
        return None
    if text[end] in "同从到至向较" and names_at is not None and names_at(text, end + 1) > 0:
        return None  # "茅台同五粮液比", "招行较平安": a preposition before another listed name
    if any(text.startswith(word, end) and _describes(word, owner, industry) for word in _FAMILY_WORDS):
        return None
    if not _starts_a_name_word(text, end):
        return None  # a verb, pronoun, time word, numeral … : the rest of the question ("茅台我按1380买的", "换五粮液看看")
    stop = end + 1  # the first character is not a continuation (checked above)
    while (
        stop < len(text)
        and _is_cjk(text[stop])
        and not _ZH_NAME_STOP.match(text, stop)
        and not (names_at is not None and names_at(text, stop) >= 3)
    ):
        stop += 1
    return stop


def _unknown_extension_end_en(
    text: str, end: int, *, owner: str, industry: str, names_at: Callable[[str, int], int] | None
) -> int | None:
    if not DENY_UNKNOWN_EXTENSIONS:
        return None
    stop = end
    while (match := _EN_TITLE_WORD.match(text, stop)) is not None:
        word = match.group(1)
        if word.lower() in _EN_CONTINUATIONS or (names_at is not None and names_at(text, match.start(1)) > 0):
            break
        if stop == end and _describes(word, owner, industry):
            return None
        stop = match.end()
    return stop if stop > end else None
