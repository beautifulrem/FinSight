"""Positive text-safety checks for third-party text and for answer text.

Two users:

* ``safe_headline`` decides whether a document title may be *shown* (the evidence list). It is a positive
  shape check, not a blocklist: after NFKC, invisible-character removal and confusable folding (Cyrillic /
  Greek / Armenian look-alikes → Latin), a title is shown only when every character is in an allowed set
  (CJK, ASCII letters and digits, ordinary punctuation), no word mixes scripts (``Ignоre`` with a Cyrillic
  ``о``), and it contains no link, domain, phone number, messaging handle (WeChat / QQ / Telegram / e-mail),
  no instruction or second-person address, and no advice, rating or guarantee wording, and no planted-fact shape
  (corrections, exclusives and rumours, prices and multiples, share-capital actions, AI-addressed text; round 9: an
  insider or unnamed source "revealing" something, a Q&A transcript, a figure "restated"; round 10: a delimited data
  row, a title cut off right after a figure word). The agent's evidence ledger additionally hides a headline that
  states a figure the run's structured data does not contain, in Arabic or Chinese numerals (``agent/graph.py``).
* ``find_prohibited_promotion`` finds what must never appear in *any* answer, whoever wrote it: guaranteed-
  return claims (稳赚不赔, 保本, 保证收益), stock-tip solicitation (荐股, 带单, 喊单, 加微信, 私信, 内幕消息)
  and contact handles offered to the reader, plus (round 7) doubling-and-compensation schemes (资金翻倍，亏损全额
  赔付), principal/interest promises (本金无忧, principal-protected), hype about an imminent move (直接拉升,
  错过再等), tip-sheet hooks (牛股, 建仓名单), ``@handles`` and domains written with spaced dots. The compliance
  guard removes the sentences that contain them.
  Negated or warning uses ("不保证收益", "谨防非法荐股", "no guaranteed return") are left alone.

The template answer never quotes titles at all (``agent/composer.py``); these checks are the layer for
text that is shown or relayed anyway.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad\u180e\u034f\u115f\u1160\u3164]")

# Latin look-alikes from other scripts (a subset of Unicode's confusables.txt covering the letters that render
# identically in common fonts). NFKC already folds full-width, mathematical and circled letters.
_CONFUSABLE_PAIRS = {
    # Cyrillic
    "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "с": "c",
    "т": "t", "у": "y", "х": "x", "ѕ": "s", "і": "i", "ї": "i", "ј": "j", "һ": "h", "ԁ": "d", "ԛ": "q",
    "ԝ": "w", "ӏ": "l", "ɡ": "g", "ь": "b", "ү": "y", "ҳ": "x", "ԍ": "g",
    "А": "A", "В": "B", "Е": "E", "Ё": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ї": "I", "Ј": "J", "Һ": "H", "Ԁ": "D", "Ԛ": "Q",
    "Ԝ": "W", "Ӏ": "I", "Ү": "Y",
    # Greek
    "α": "a", "β": "b", "γ": "y", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
    "υ": "u", "χ": "x", "ω": "w", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Armenian, Latin extensions and IPA
    "օ": "o", "ս": "u", "ց": "g", "հ": "h", "ո": "n", "ı": "i", "ɑ": "a", "ɩ": "i", "ʏ": "y", "ɴ": "n",
    "ʀ": "r", "ᴄ": "c", "ᴏ": "o", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "ᴢ": "z",
}  # fmt: skip
_CONFUSABLES = str.maketrans(_CONFUSABLE_PAIRS)
_FOREIGN_LETTER_SCRIPTS = ("CYRILLIC", "GREEK", "ARMENIAN", "CHEROKEE", "COPTIC", "GEORGIAN")


def fold(text: str) -> str:
    """NFKC, invisible characters removed, confusable letters mapped to Latin (for matching only)."""
    return INVISIBLE.sub("", unicodedata.normalize("NFKC", text or "")).translate(_CONFUSABLES)


def mixed_script_word(text: str) -> str | None:
    """A word that mixes Latin letters with Cyrillic/Greek/... letters (a homoglyph attack), or ``None``."""
    for word in re.findall(r"[^\W\d_]+", unicodedata.normalize("NFKC", INVISIBLE.sub("", text or ""))):
        scripts = set()
        for char in word:
            if "a" <= char.lower() <= "z":
                scripts.add("LATIN")
                continue
            name = unicodedata.name(char, "")
            scripts.update(script for script in _FOREIGN_LETTER_SCRIPTS if name.startswith(script))
        if "LATIN" in scripts and len(scripts) > 1:
            return word
    return None


# ---- detectors (run on folded text) ----

_TLDS = (
    "com|cn|net|org|io|example|top|xyz|vip|cc|me|info|biz|app|site|online|link|club|shop|tv|co|ai|ly|gg|to|ru|"
    "uk|us|de|jp|kr|win|pro|live|fun|wang|ltd|group|tech|store|work|cloud|icu|asia|mobi|name|finance|money|"
    "capital|invest|fund|trade|market|click|bet|bid|loan|tk|ml|ga|cf|gq|pw|ws|la|in|su"
)
_LINK = re.compile(
    r"(?:https?|ftp|javascript|data|vbscript|file|mailto)\s*:"
    r"|\b[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]{1,63})*\.(?:" + _TLDS + r")\b(?:[/?#:]\S*)?"
    r"|www\s*\."
    r"|\bt\.me/\S+",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A handle is 6-20 characters starting with a letter and containing a digit or underscore ("caifu8888").
_HANDLE = r"[A-Za-z](?=[A-Za-z0-9_-]*[\d_])[A-Za-z0-9_-]{5,19}"
_WECHAT_WORD = r"(?:微信|威信|薇信|徽信|v信|(?<![a-z])(?:vx|wx|weixin|wechat)|加v|\+v)"
_GROUP = r"群(?![体众落岛])"
_MESSAGING = re.compile(
    # 加微信 / 加V / 加我QQ / 进群 / 私信: an invitation to move the conversation to a private channel
    r"加\s*(?:我|老师|客服|助理|小编|好友)?\s*(?:微信|威信|薇信|徽信|v信|vx|wx|weixin|wechat|v(?![a-z])|qq|扣扣|企鹅|好友|"
    + _GROUP
    + r")"
    r"|(?:进|入|拉你进|拉你入)\s*(?:vip|粉丝|交流|投资|炒股|内部|付费|收费|会员)?"
    + _GROUP
    + r"|私信|私聊|扫码(?:加|进|入|关注|领取)"
    rf"|{_WECHAT_WORD}\s*(?:号|id)?\s*[:：]?\s*{_HANDLE}"
    # "QQ群736291845", "QQ群号：736291845", "QQ 群（736291845）"
    r"|(?<![a-z])(?:qq|扣扣|企鹅)\s*(?:群号?|号码?)?\s*[:：(（]?\s*\d{5,11}"
    r"|\b(?:telegram|tg)\b\s*(?:群|频道|号|channel|group)?\s*[:：]?\s*@?[A-Za-z][A-Za-z0-9_]{4,31}"
    r"|(?:电报|纸飞机|飞机)\s*(?:群|频道|号)\s*[:：]?\s*@?[A-Za-z0-9_]{4,32}"
    r"|\b(?:whatsapp|discord|signal)\s*[:：]\s*[@+]?[A-Za-z0-9_]{5,}"
    r"|\b(?:dm|message|text|whatsapp|telegram)\s+(?:me|us)\b"
    r"|\bjoin\s+(?:our|my)\s+(?:vip\s+|private\s+|paid\s+|free\s+)?(?:group|channel|chat|telegram|whatsapp|discord|community)",
    re.IGNORECASE,
)
_AT_HANDLE = re.compile(r"(?<![\w.])@[A-Za-z][A-Za-z0-9_]{3,31}\b")
# Phone numbers once spaces and dashes between digits are removed: CN mobile, 400/800 service lines, landlines.
_PHONE_DIGITS = re.compile(r"(?<![\d.])(?:1[3-9]\d{9}|[48]00\d{7}|0\d{2,3}\d{7,8})(?![\d.]?\d)")
_PHONE_FORMATTED = re.compile(r"(?<![\d.])(?:[48]00[-\s]\d{3}[-\s]\d{4}|1[3-9]\d[-\s]\d{4}[-\s]\d{4})(?![\d.]?\d)")
_CONTACT_CUE = re.compile(r"致电|电话|热线|联系|咨询|拨打|手机|来电|call|phone|tel\b|whatsapp|微信|vx", re.IGNORECASE)

# Guaranteed returns, stock-tip solicitation and hype: removed from every answer (unless negated / a warning).
_PROMOTION = re.compile(
    r"稳赚不赔|稳赚|包赚|只赚不赔|稳赢|稳拿|(?<![确担环])保本(?!点)|保收益|保息|保证(?:本金|收益|盈利|回报|赚钱)|收益保证|"
    r"保底收益|承诺收益|零风险|无风险(?:高收益|收益|获利|赚钱)|"
    r"荐股|带单|喊单|跟单(?:群|老师)?|老师带|带你赚|内幕消息|内部消息|内部名单|内部渠道|牛股(?:推荐|名单|池)|必涨股|"
    r"涨停(?:板)?(?:密码|预测|推荐|股池)|收费群|付费群|会员群|vip群|翻倍(?:股|黑马)|黑马股推荐|闭眼(?:买|上车|入)|铁底|"
    r"\bguaranteed?\s+(?:returns?|profits?|gains?|income|yield)|\brisk[- ]free\s+(?:returns?|profits?|gains?)\b|"
    r"\b(?:can(?:no|')?t|cannot)\s+lose\b|\bno[- ]lose\b|\bsure[- ]?(?:fire|thing)\s+(?:win|profit|bet|pick)|"
    r"\bstock\s+tips?\s+(?:group|channel|service)|\binsider\s+(?:tips?|info(?:rmation)?|list)|"
    r"\bback\s+up\s+the\s+truck\b|"
    r"\bonce[- ]in[- ]a[- ](?:decade|lifetime|generation)\s+(?:entry|opportunit|chance|buy)|"
    # doubling-plus-compensation schemes (资金翻倍，亏损全额赔付), principal/interest promises, and hype about an
    # imminent move (直接拉升, 错过再等十年, 必涨停)
    r"(?:资金|收益|本金|账户|利润)翻倍|翻倍[^。；;！!？?\n]{0,12}(?:赔付|包赔|保本|退款|兜底)|"
    r"(?:亏损|亏了|亏本|赔了)[^。；;！!？?\n]{0,6}(?:全额)?(?:赔付|包赔|赔偿|补偿|兜底)|全额赔付|包赔|"
    r"本金无忧|月月付息|保本保息|刚性兑付|"
    r"直接拉升|拉升在即|(?:明早|明天|明日|开盘)[^。；;！!？?\n]{0,6}(?:直接)?拉升|错过(?:就|这次)?再等|必涨停|"
    r"(?:下周|明天|明日|本周)[^。；;！!？?\n]{0,4}必(?:涨|大涨|涨停)|"
    # tip-sheet hooks: 送牛股, 涨停票, 主力建仓名单, 一对一指导, 名额有限
    r"牛股|涨停票|(?:主力|庄家)?建仓名单|一对一(?:指导|带)|名额有限|开户即送|"
    r"\bguaranteed?\s+(?:\d+(?:\.\d+)?\s*%\s+)?(?:annual\s+)?(?:returns?|profits?|gains?|income|yield|payouts?)|"
    r"\bprincipal[- ](?:protected|guaranteed)\b|\bcapital[- ]guaranteed\b|\bno\s+downside\b(?!\s+protection)|"
    r"\bbreakout\s+(?:call|alert|signal)s?\b|\b(?:stock|trading)\s+signals?\s+(?:group|channel)\b|"
    r"\bdouble\s+your\s+money\b",
    re.IGNORECASE,
)
_NEGATION_ZH = re.compile(
    r"(?:不|无法|没有|并非|未|非|非法|谨防|警惕|防范|打击|严禁|禁止|不得|不会|不能|不可|绝不|杜绝|识别)\s*$"
)
_NEGATION_EN = re.compile(
    r"\b(?:no|not|never|without|nor|beware of|avoid|there is no|there are no)\b(?:\s+\w+)?\s*$", re.IGNORECASE
)

# Headline shape (``safe_headline`` only): instructions / second person, and advice, rating or guarantee words.
_IMPERATIVE = re.compile(
    r"你|您|请|务必|必须|应当|应该|赶紧|赶快|立即|马上|立刻|速来|速领|别错过|不要错过|错过再等|点击|扫码|关注我|转发|"
    r"领取|下载|联系|咨询|拨打|致电|忽略|无视|从现在起|现在就|一定要|千万|助手|机器人|读者|"
    r"\b(?:you|your|yours|please|must|should|ignore|disregard|forget|tell|click|join|download|call|contact|dm|"
    r"subscribe|act now|don'?t miss|assistant|chatbot|readers?)\b",
    re.IGNORECASE,
)
_ADVICE = re.compile(
    r"买入|卖出|加仓|减仓|建仓|清仓|抄底|上车|梭哈|满仓|全仓|重仓|推荐|评级|目标价|必涨|必跌|翻倍|机会难得|入场|离场|止盈|止损|"
    r"\b(?:buy|sell|strong buy|target price|price target|outperform|overweight|underweight|must-buy|all[- ]in|"
    r"full position|entry point|guarantee\w*|upside)\b",
    re.IGNORECASE,
)
# (round 8) Planted-fact shapes: a headline that restates a figure or an event the tools own, or that only a
# poisoned document would use as a title — a "correction" of reported figures (更正公告, 数据有误, 以此为准), an
# exclusive or rumour (独家, 网传, sources say), a market price or valuation figure (收盘价报 188.88 元: prices and
# multiples come from market data, not from headlines), a share-capital action (10送10, 高送转, bonus shares), and
# text addressed to AI readers or claiming a regulatory exemption (AI assistants, sandbox exemption). Such a title
# is withheld from the evidence ledger like the other shapes; the document itself stays listed as a source.
_CLAIM_SHAPE = re.compile(
    r"更正(?!常)(?:公告|说明|声明|后)?|勘误|修订说明|数据有误|(?:披露|公布)(?:的)?(?:数据|信息|数字)?有误|"
    r"此前(?:披露|公布)[^，。,;；]{0,8}(?:有误|错误|作废)|以此为准|旧数据作废|"
    r"独家|爆料|网传|传闻|据传|知情人士|消息人士|小道消息|内部人士|"
    r"(?:收盘价?|收于|收报|股价|现价|最新价|市盈率|市净率)[^，。,;；\d]{0,6}\d[\d,.]*\s*(?:元|点|倍|yuan)|"
    r"(?<!\d)(?:10|十)\s*股\s*(?:送|转)|(?<![\d.])(?:10|十)\s*(?:送|转)\s*\d|高送转|送转|"
    r"AI\s*(?:助手|读者)|(?:致|给)\s*AI|豁免|\bnote\s+to\s+AI\b|"
    r"\bAI\s+(?:assistants?|readers?|chatbots?)\b|\b(?:assistants|chatbots?)\b|"
    r"\bsandbox\b|\bexemption\b|\bexclusive\b|\brumou?rs?\b|\bsources\s+(?:say|said|claim)\b|"
    r"\bcorrect(?:ion|ed)\b|\brestate(?:d|ment)\b|\berrat(?:um|a)\b|\bbonus\s+(?:shares?|issue)\b|"
    r"\bstock\s+split\b|\b(?:close[ds]?|closing\s+price|share\s+price|last\s+price)\s+(?:at\s+|of\s+|up\s+|down\s+)?"
    r"(?:CNY|RMB|¥)?\s*\d|"
    # (round 9, E4) unconfirmed-source shapes: an insider or unnamed source "revealing" something (透露, 据悉, 知情人士,
    # 市场传言, insiders, people familiar, leaked), a question-and-answer transcript used as a headline (问：…答：), and
    # a restatement of a figure ("实为", "实际应为") — none is how a filing or a news desk titles a report
    r"透露|据悉|据了解|知情|(?:人士|高管|管理层)(?:称|表示|说|指出)|坊间|传言|风传|传出|"
    r"问\s*[:：][^。]{0,60}?答\s*[:：]|(?<![\w])Q\s*[:：].{0,80}?(?<![\w])A\s*[:：]|实为|实际应?为|"
    r"\binsiders?\b|\bpeople\s+familiar\b|\bleak(?:ed|s)?\b|\bwhispers?\b|\bunconfirmed\b|\breportedly\b",
    re.IGNORECASE,
)
_MARKUP = re.compile(r"[<>{}\[\]`|\\^~]|!\[|\]\(|&#|\\u[0-9a-f]{4}|\*\*|__", re.IGNORECASE)
# (round 10, F3) Two more figure shapes a headline never has. A *data row*: fields joined by delimiters, two or more of
# them bare numbers ("code,name,price,pe 000858.SZ,五粮液,99.9,8.8"), i.e. a table or an export whose numbers are
# figures without units; thousands separators and years are not fields. A *cut figure*: the headline ends in a number
# right after a figure word ("…拟每10股派现金红利3", "营收同比增长12"), the shape of a title cut off mid-figure (the
# split variant of a planted document), whose value cannot be checked at all.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_ROW_DELIMITER = re.compile(r"[,;]")
_NUMERIC_FIELD = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
_YEAR_FIELD = re.compile(r"(?:19|20)\d{2}")
_CUT_FIGURE = re.compile(
    r"(?:派发?|派息|派现|红利|股息|分红|营收|收入|利润|净利|毛利率?|净利率|市盈率|市净率|(?<![A-Za-z])(?:PE|PB|ROE|EPS)|"
    r"股价|价格|收于|收报|报|涨幅?|跌幅?|增长|增加|减少|下滑|下降|上升|提升|同比|环比|为|至|达到?|约|超过?|逾|近)"
    r"\s*[:：]?\s*[-+]?(?!(?:19|20)\d{2}$)\d+(?:\.\d+)?$",
    re.IGNORECASE,
)


# (round 11, G8) A dramatic financial claim with no figure ("净利润腰斩", "暴雷", "崩盘", "退市风险", "plunges",
# "halved") is the figure-free form of a planted headline: the figure rules above never see it, and nothing in the run
# can confirm it. It is a claim shape unless the title names an official source (a filing title "关于…的公告", an
# annual/interim report, an exchange or a regulator). Measured on the shipped corpus (data/runtime/documents.jsonl +
# data/documents.json, 38,446 titles, 5,874 shown before) and the replay snapshots (77 titles): 0 newly hidden.
_DRAMATIC_CLAIM = re.compile(
    r"腰斩|暴雷|爆雷|崩盘|退市风险|闪崩|暴跌|巨亏|断崖式|血亏|爆仓|雪崩|跳水|崩塌|坍塌|暴增|暴涨|狂飙|飙升|造假|违约|"
    r"爆表|翻车|塌方|(?<!最)大跌|(?<!最)大涨|"
    r"\bhalved\b|\bcollapse[sd]?\b|\bcrash(?:es|ed)?\b|\bplunge[sd]?\b|\bplummet(?:s|ed)?\b|\bmeltdown\b|"
    r"\bblow-?up\b|\bdelisting\s+risk\b|\bwiped\s+out\b|\bfraud\b|\bdefault(?:s|ed)\b|\bsoar(?:s|ed)\b|"
    r"\bskyrocket(?:s|ed)?\b",
    re.IGNORECASE,
)
_OFFICIAL_SOURCE = re.compile(
    r"关于[^，。]{1,40}的(?:提示性)?公告|年度报告|半年度报告|季度报告|业绩预告|业绩快报|证监会|上交所|深交所|北交所|"
    r"交易所|国家统计局|人民银行|央行|财政部|国资委|金融监管总局|"
    r"\bannual\s+report\b|\binterim\s+report\b|\bCSRC\b|\bstock\s+exchange\b|\bfiling\b",
    re.IGNORECASE,
)


def _data_row(folded: str) -> str | None:
    for chunk in _THOUSANDS.sub("", folded).split():
        fields = [field for field in _ROW_DELIMITER.split(chunk) if field]
        numeric = [f for f in fields if _NUMERIC_FIELD.fullmatch(f) and not _YEAR_FIELD.fullmatch(f)]
        if len(fields) >= 3 and len(numeric) >= 2:
            return chunk
    return None


# Allowed characters of a shown headline (folded text): CJK ideographs, ASCII letters/digits/space and
# ordinary punctuation.
_HEADLINE_CHARS = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbfA-Za-z0-9 ,.:;?!'\"()%/&+\-，。、：；？！“”‘’《》〈〉（）—…·％]+"
)
MAX_HEADLINE_CHARS = 80


@dataclass(frozen=True)
class Finding:
    kind: str  # link | email | messaging | phone | promotion | mixed_script | markup | charset | imperative | advice
    # | claim (a planted-fact shape: correction, exclusive/rumour, price figure, share-capital action, AI-addressed)
    text: str


def contact_findings(text: str, *, strict: bool) -> list[Finding]:
    """Links, e-mail, messaging handles and phone numbers in ``text``.

    ``strict`` (headlines) flags any phone-shaped number and any ``@handle``; otherwise (answers, which quote
    prices, volumes and amounts) a phone number needs a contact cue nearby or a phone-style grouping.
    """
    folded = fold(text)
    found = [Finding("link", m.group(0)) for m in _LINK.finditer(folded)]
    found += [Finding("email", m.group(0)) for m in _EMAIL.finditer(folded)]
    found += [Finding("messaging", m.group(0)) for m in _MESSAGING.finditer(folded)]
    if strict:
        found += [Finding("messaging", m.group(0)) for m in _AT_HANDLE.finditer(folded)]
    squeezed = re.sub(r"(?<=\d)[\s\-–—.]+(?=\d{3})", "", folded) if strict else folded
    for match in _PHONE_FORMATTED.finditer(folded):
        found.append(Finding("phone", match.group(0)))
    for match in _PHONE_DIGITS.finditer(squeezed):
        window = squeezed[max(0, match.start() - 12) : match.end() + 6]
        if strict or _CONTACT_CUE.search(window):
            found.append(Finding("phone", match.group(0)))
    return found


def find_prohibited_promotion(text: str) -> list[Finding]:
    """Guarantee / solicitation / hype wording and contact handles that no answer may contain."""
    folded = fold(text)
    # "加 微 信", "稳-赚-不-赔": separators between CJK characters are dropped before matching.
    compact = re.sub(r"(?<=[\u4e00-\u9fff])[\s\-_.·•*|/\\~～]+(?=[\u4e00-\u9fff])", "", folded)
    found: list[Finding] = []
    for candidate in dict.fromkeys((folded, compact)):
        for match in _PROMOTION.finditer(candidate):
            before = candidate[max(0, match.start() - 24) : match.start()]
            if _NEGATION_ZH.search(before[-6:]) or _NEGATION_EN.search(before):
                continue
            found.append(Finding("promotion", match.group(0)))
        if found:
            break
    contacts = contact_findings(text, strict=False)
    found += [item for item in contacts if item.kind != "link" or not _allowed_domain(item.text)]
    # An answer never needs to name a social-media handle ("@AshareAlphaSignals") or a domain written with
    # spaces around the dots to dodge link detection ("ping-an-insider . example . com").
    found += [Finding("messaging", m.group(0)) for m in _AT_HANDLE.finditer(folded)]
    found += [Finding("link", m.group(0)) for m in _SPACED_DOMAIN.finditer(folded) if re.search(r"\s", m.group(0))]
    return found


# lower-case labels only, the first one with a letter: "2025. Com…" at a sentence end is not a domain
_SPACED_DOMAIN = re.compile(
    r"\b(?=[a-z0-9-]*[a-z])[a-z0-9][a-z0-9-]{1,62}(?:\s?\.\s?[a-z0-9-]{1,63})*\s?\.\s?"
    r"(?:com|cn|net|org|example|io|top|xyz|vip|cc|info|biz)\b"
)


# Domains an answer may name: exchanges, regulators and statistics offices.
_ALLOWED_DOMAINS = re.compile(
    r"(?:^|\.)(?:cninfo\.com\.cn|sse\.com\.cn|szse\.cn|bse\.cn|csrc\.gov\.cn|pbc\.gov\.cn|stats\.gov\.cn|"
    r"gov\.cn|hkex\.com\.hk|sec\.gov)$",
    re.IGNORECASE,
)


def _allowed_domain(link: str) -> bool:
    host = re.sub(r"^(?:https?://)?", "", link.strip(), flags=re.IGNORECASE).split("/")[0].split("?")[0].split(":")[0]
    return bool(host) and "." in host and bool(_ALLOWED_DOMAINS.search(host.lower()))


def headline_findings(title: str) -> list[Finding]:
    """Everything that stops ``title`` from being shown as a headline (empty = safe)."""
    text = re.sub(r"\s+", " ", INVISIBLE.sub("", title or "")).strip()
    folded = re.sub(r"\s+", " ", fold(text)).strip()
    found: list[Finding] = []
    if not folded:
        return [Finding("charset", "")]
    word = mixed_script_word(text)
    if word:
        found.append(Finding("mixed_script", word))
    if len(folded) > MAX_HEADLINE_CHARS:
        found.append(Finding("charset", f"longer than {MAX_HEADLINE_CHARS} characters"))
    outside = _HEADLINE_CHARS.sub("", folded)
    if outside:
        found.append(Finding("charset", outside[:20]))
    found += [Finding("markup", m.group(0)) for m in _MARKUP.finditer(folded)]
    found += contact_findings(text, strict=True)
    found += [Finding("promotion", m.group(0)) for m in _PROMOTION.finditer(folded)]
    found += [Finding("imperative", m.group(0)) for m in _IMPERATIVE.finditer(folded)]
    found += [Finding("advice", m.group(0)) for m in _ADVICE.finditer(folded)]
    found += [Finding("claim", m.group(0)) for m in _CLAIM_SHAPE.finditer(folded)]
    row = _data_row(folded)
    if row:
        found.append(Finding("claim", row))
    cut = _CUT_FIGURE.search(folded)
    if cut:
        found.append(Finding("claim", cut.group(0)))
    dramatic = _DRAMATIC_CLAIM.search(folded)
    if dramatic and not _OFFICIAL_SOURCE.search(folded):
        found.append(Finding("claim", dramatic.group(0)))
    return found


def safe_headline(title: str | None) -> str | None:
    """The title (invisible characters removed, whitespace collapsed) when it passes the shape check, else ``None``."""
    if not title or headline_findings(title):
        return None
    return re.sub(r"\s+", " ", INVISIBLE.sub("", title)).strip()
