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
  row, a title cut off right after a figure word; round 12: a title cut mid-clause, characters spaced out one by one,
  a key-value record, an advertisement, figure-free audit-opinion, trading-halt and restructuring events; round 14:
  romanised Chinese, i.e. pinyin buy-solicitation or several multi-syllable pinyin words). The agent's evidence ledger
  additionally hides a headline that states a figure the run's structured data does not contain, in Arabic or Chinese
  numerals, and (round 14) a headline asserting regulatory action, fraud or delisting, or claiming to be a regulator's
  notice, unless the document comes from an official source (``regulatory_title_withheld``; ``agent/graph.py``).
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


# (round 18, review-14 N7) Chinese uppercase financial numerals (大写, as on cheques and contracts) and their
# traditional forms, read as the lower-case numerals every figure rule knows: "叁点柒倍" is "三点七倍", "玖拾亿元整" is
# "九十亿元整". Only inside a run of numerals that holds a 大写 character and is a figure: two or more numeral
# characters ("玖拾", "叁点柒", "陆万"), or one followed by a unit ("叁倍", "伍成", "伍圆"). "陆家嘴", "队伍",
# "拾取" and "登陆" keep their characters. A 圆 right after such a figure is the currency unit 元.
_CAPITAL_DIGITS = {
    "壹": "一", "贰": "二", "貳": "二", "叁": "三", "參": "三", "叄": "三", "肆": "四", "伍": "五", "陆": "六",
    "陸": "六", "柒": "七", "捌": "八", "玖": "九", "拾": "十", "佰": "百", "仟": "千", "萬": "万", "億": "亿",
}  # fmt: skip
_CAPITAL_TABLE = str.maketrans(_CAPITAL_DIGITS)
_CAPITAL_RUN = re.compile("[〇零一二两三四五六七八九十百千万亿点點" + "".join(_CAPITAL_DIGITS) + "]+(?P<yuan>[圆圓])?")
_CAPITAL_UNIT_AFTER = re.compile(r"[倍成亿億万萬元圆圓角%％]|个百分点")


def capital_numerals(text: str) -> str:
    """``text`` with figures in Chinese uppercase financial numerals written in lower-case numerals (same length)."""
    if not any(char in _CAPITAL_DIGITS for char in text):
        return text

    def lower(match: re.Match[str]) -> str:
        run = match.group(0)
        digits = run[: len(run) - len(match.group("yuan") or "")]
        if not any(char in _CAPITAL_DIGITS for char in digits):
            return run
        if len(digits) < 2 and not (match.group("yuan") or _CAPITAL_UNIT_AFTER.match(text, match.end())):
            return run
        return digits.translate(_CAPITAL_TABLE) + ("元" if match.group("yuan") else "")

    return _CAPITAL_RUN.sub(lower, text)


def fold(text: str) -> str:
    """NFKC, invisible characters removed, confusable letters mapped to Latin and (round 18) Chinese uppercase
    financial numerals written as lower-case numerals (for matching only)."""
    return capital_numerals(INVISIBLE.sub("", unicodedata.normalize("NFKC", text or "")).translate(_CONFUSABLES))


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
    # (round 15, K15) "逢低加码" (traditional 加碼 reads the same after ``_simplified``)
    r"逢低(?:加码|布局|吸纳|买)|加码买|"
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
    r"股价|价格|收于|收报|报|涨幅?|跌幅?|增长|增加|减少|下滑|下降|上升|提升|同比|环比|为|至|达到?|约|超过?|逾|近|"
    r"收益率?|回报率?|年化)"
    r"\s*[:：]?\s*[-+]?(?!(?:19|20)\d{2}$)\d+(?:\.\d*)?$",
    re.IGNORECASE,
)


# (round 15, review-11 K15) Planted financial-fact headlines the round-11 reviewer showed in the evidence list in their
# split (title cut in half) and title-only variants, written in shapes the rules above did not read:
# * *traditional characters and Taiwanese terms*: "貴州茅臺本益比僅 5.2" (本益比 = 市盈率, 僅 = 仅). Headline checks
#   read the title with common traditional characters mapped to simplified ones (``_TRADITIONAL``) and 本益比 /
#   股價淨值比 as 市盈率 / 市净率, so every keyword rule applies to them ("法人建議逢低加碼" is advice);
# * *a limiter cut before or at its figure*: "…仅", "…只有四", "…仅 5.2", "高达 12": a valuation or result claim stopped
#   right where its value starts, Arabic or Chinese numerals;
# * *a metric cut at its change word*: "归母净利润同", "营收下", "market value has reache": a figure claim cut before
#   its value;
# * *Latin look-alike words in another script*: "ΡΕ(ΤΤΜ)" in Greek capitals reads as "PE (TTM)" but no Latin keyword
#   rule sees it; a word of two or more letters that are all Greek / Cyrillic confusables of Latin letters is a
#   homoglyph word like a mixed-script one.
# Measured on the shipped corpus (data/runtime/documents.jsonl + data/documents.json, 38,446 titles, 5,874 shown
# before) and the replay snapshots (747 titles, 445 shown before): see ``SECURITY.md`` for the count newly hidden.
_TRADITIONAL_PAIRS = (
    "臺台灣湾僅仅淨净營营價价漲涨億亿萬万報报導导盤盘為为達达約约與与業业績绩稅税東东長长開开當当現现創创歷历預预測测"
    "幣币國国際际銀银資资產产總总額额減减權权衝冲擊击點点來来後后會会將将對对進进這这還还經经濟济據据傳传聞闻獨独調调"
    "罰罚議议碼码買买賣卖倉仓擇择機机優优勢势險险虧亏損损補补鎖锁時时間间問问題题說说話话請请讀读寫写書书訊讯號号錢钱"
    "費费貨货證证監监處处決决違违規规範范檢检査查詐诈騙骗偽伪實实圓圆匯汇兌兑換换帳账戶户單单線线綫线動动態态勝胜敗败"
    "獎奖紅红戰战爭争縮缩擴扩張张頂顶鋒锋鋰锂電电車车輛辆軟软體体網网絡络華华興兴發发財财計计劃划條条層层級级類类標标"
    "準准確确認认識识雙双齊齐豐丰嚴严廣广礦矿藥药醫医療疗壽寿貸贷債债協协務务環环節节氣气溫温驗验訂订購购銷销運运輸输"
    "鐵铁鋼钢煉炼熱热頭头緊紧寬宽兩两個个們们嗎吗啟启從从無无雖虽讓让給给該该應应須须樣样簡简檔档漢汉擁拥護护師师員员"
    "顧顾團团隊队廠厂聯联"
)
_TRADITIONAL = str.maketrans(dict(zip(_TRADITIONAL_PAIRS[::2], _TRADITIONAL_PAIRS[1::2], strict=True)))
_TAIWAN_TERMS = (("本益比", "市盈率"), ("股价净值比", "市净率"), ("净值比", "市净率"))


def _simplified(folded: str) -> str:
    """``folded`` with common traditional characters and Taiwanese valuation terms read as simplified ones."""
    text = folded.translate(_TRADITIONAL)
    for term, simplified in _TAIWAN_TERMS:
        text = text.replace(term, simplified)
    return text


_CN_DIGITS = r"[零〇一二两三四五六七八九十百千万亿点]+"
_LIMITER = r"(?:(?<!不)仅为?|只有|只剩|(?<![人天英奇口刚方])才|不到|不足|高达|低至|最低|最高)"
_CUT_LIMIT = re.compile(
    # a limiter at the very end, or followed by a bare figure (Arabic or Chinese numerals) and nothing else
    r"(?:(?<!不)仅为?|只有|只剩|(?<![人天英奇口刚方])才|高达|低至)\s*$|"
    + _LIMITER
    + r"\s*[:：]?\s*(?:[-+]?\d+(?:\.\d*)?|"
    + _CN_DIGITS
    + r")\s*$"
)
_CUT_CHANGE = re.compile(
    # a metric followed by the start of a change word and nothing else: "归母净利润同", "营收下", "市值较上" (a whole
    # change word, "毛利率提升", is a complete headline)
    r"(?:净利润|净利|营业收入|营收|收入|利润|市值|股价|市盈率|市净率|每股收益|毛利率|净利率|(?<![A-Za-z])(?:ROE|EPS))"
    r"(?:同比?|环比?|较上?年?|下|上)\s*$|"
    # "Kweichow Moutai's market value has reache", "net profit fell", "P/E ratio stands": a metric and a verb, no value
    r"\b(?:market\s+(?:value|cap\w*)|valuation|net\s+(?:profit|income)|revenues?|sales|earnings|profits?|"
    r"(?:share|stock)\s+price|P/?E(?:\s+ratio)?|P/?B(?:\s+ratio)?|dividend(?:\s+yield)?|EPS|ROE)\s+"
    r"(?:(?:has|have|had|is|was|were|will|now)\s+)*(?:\w+ly\s+)?(?:reach\w*|hit\w*|ris\w*|rose|fell|fall\w*|drop\w*|"
    r"climb\w*|surg\w*|soar\w*|plung\w*|jump\w*|slid\w*|slump\w*|stand\w*|stood|grew|grow\w*|declin\w*|increas\w*|"
    r"decreas\w*|top\w*|tumbl\w*|doubl\w*|halv\w*|exceed\w*|surpass\w*|came\s+in|come\w*)\s*$",
    re.IGNORECASE,
)


# (round 16, review-12 L18) Figures written so that the digit rules above do not read them, from the round-12 reviewer's
# planted headlines that the evidence list showed:
# * *English numbers in words*: "P/E is now just four point two times" (``spelled_numbers`` reads it as "4.2 times",
#   so the ledger's figure check and every cut / claim rule see a figure);
# * *digits spaced out or split by soft hyphens*: "市盈率 6 . 7 倍", "1 2 3 亿元", "仅­5­.­9" (the soft hyphens are
#   invisible characters, removed by ``fold``; the spaced digits are joined: ``joined_digits``);
# * *a figure cut after its decimal point*: "…市盈率仅5." (the bare figure of ``_CUT_FIGURE`` / ``_CUT_LIMIT`` may end
#   on a dot);
# * *an English limiter before a bare figure at the end*: "…P/E is now just 4." (``_CUT_LIMIT_EN``) and an English
#   valuation multiple with its value ("P/E of 4.2 times", ``_CLAIM_MULTIPLE_EN``), like 市盈率…倍 in ``_CLAIM_SHAPE``;
# * a year or a report and a metric with no value ("贵州茅台2025年净利", in ``_CUT_PHRASE``) and "fell by half"
#   ("halved" in words, in ``_DRAMATIC_CLAIM``).
_EN_UNIT_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]  # fmt: skip
_EN_TENS_WORDS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_EN_VALUES = {
    **{word: value for value, word in enumerate(_EN_UNIT_WORDS)},
    **{word: 20 + 10 * index for index, word in enumerate(_EN_TENS_WORDS)},
}
_EN_WORD = r"(?:" + "|".join(sorted([*_EN_VALUES, "hundred", "thousand"], key=len, reverse=True)) + r")"
_EN_DIGIT_WORD = r"(?:" + "|".join(_EN_UNIT_WORDS[:10]) + r")"
_EN_NUMBER = re.compile(
    r"\b(?P<int>" + _EN_WORD + r"(?:(?:\s*-\s*|\s+(?:and\s+)?)" + _EN_WORD + r")*)\b"
    r"(?P<point>\s+point(?P<decimals>(?:\s+" + _EN_DIGIT_WORD + r")*)\b)?",
    re.IGNORECASE,
)


def _english_integer(words: list[str]) -> int:
    total = current = 0
    for word in words:
        if word == "hundred":
            current = max(current, 1) * 100
        elif word == "thousand":
            total += max(current, 1) * 1000
            current = 0
        else:
            current += _EN_VALUES[word]
    return total + current


def spelled_numbers(text: str) -> str:
    """(round 16, L18) ``text`` with English numbers written in words as digits: "four point two times" → "4.2
    times", "forty-two percent" → "42 percent", "twelve billion" → "12 billion", "four point" (cut) → "4.". For
    matching only."""

    def digits(match: re.Match[str]) -> str:
        words = [word for word in re.split(r"[\s-]+", match.group("int").lower()) if word and word != "and"]
        number = str(_english_integer(words))
        if match.group("point") is None:
            return number
        decimals = "".join(str(_EN_VALUES[word]) for word in (match.group("decimals") or "").lower().split())
        return f"{number}.{decimals}"

    return _EN_NUMBER.sub(digits, text or "")


def joined_digits(text: str) -> str:
    """(round 16, L18) Digits spaced out or around a spaced decimal point joined: "6 . 7 倍" → "6.7 倍", "1 2 3 亿元" →
    "123 亿元". Single digits only, so "2025 10 11" or "Top 10 2025" keep their spaces."""
    joined = re.sub(r"(?<=\d)\s*[.．]\s+(?=\d)|(?<=\d)\s+[.．]\s*(?=\d)", ".", text or "")
    return re.sub(r"(?<![\d.])\d(?:\s\d)+(?![\d.])", lambda match: re.sub(r"\s", "", match.group(0)), joined)


_CUT_LIMIT_EN = re.compile(
    r"\b(?:just|only|merely|barely|a\s+mere|as\s+(?:low|high)\s+as|under|below|nearly|almost)\s+[-+]?\d+(?:\.\d*)?\s*$",
    re.IGNORECASE,
)
_CLAIM_MULTIPLE_EN = re.compile(
    r"(?<![A-Za-z])(?:P/?E|P/?B|price[- ]to[- ](?:earnings|book))(?:\s+ratio)?\b[^.;\d]{0,24}?[-+]?\d[\d,.]*\s*"
    r"(?:x|times)\b",
    re.IGNORECASE,
)


def lookalike_word(text: str) -> str | None:
    """(round 15, K15) A word of two or more Greek / Cyrillic / Armenian letters that are all confusables of Latin
    letters ("ΡΕ", "ТТМ"): it reads as a Latin abbreviation that the Latin keyword rules never see."""
    for word in re.findall(r"[^\W\d_]+", unicodedata.normalize("NFKC", INVISIBLE.sub("", text or ""))):
        if len(word) < 2 or any("a" <= char.lower() <= "z" or "一" <= char <= "鿿" for char in word):
            continue
        if all(char in _CONFUSABLE_PAIRS for char in word):
            return word
    return None


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
    r"\bskyrocket(?:s|ed)?\b|"
    # (round 16, review-12 L18) "profit fell by half", "cut in half": "halved" in words
    r"\b(?:fell|falls?|fallen|dropped|drops?|declined?|plunged?|slid|shrank|shrunk|sank|tumbled?|cut|slashed)\s+"
    r"(?:by\s+)?(?:half|in\s+half)\b|"
    # (round 12, H10) figure-free company events a regulator or the company itself would announce: a modified audit
    # opinion ("无法表示意见", "否定意见"; "标准无保留意见" is the clean opinion), a trading halt or a restructuring.
    # Shown only when the title names an official source, like the rest of this list.
    r"无法表示意见|否定意见|(?<!无)保留意见|非标准?(?:审计)?意见|停牌|(?:重大)?资产重组|筹划重组|借壳|"
    r"\b(?:disclaimer\s+of\s+opinion|adverse\s+opinion|qualified\s+opinion|trading\s+halt(?:ed)?|restructuring)\b",
    re.IGNORECASE,
)
# (round 12, H10) An advertisement is not a headline: an ad label ("（广告）", "【推广】", "Sponsored") or a
# wealth-product pitch ("专属VIP理财"); a market headline about yields ("银行理财年化收益率跌破3%") is not an ad.
_ADVERTISEMENT = re.compile(
    r"[（(【\[]\s*(?:广告|推广|赞助|软文)\s*[）)】\]]|^\s*(?:广告|推广|赞助)\s*[:：]|"
    r"(?:VIP|专属|尊享|高端)\s*理财|理财\s*(?:VIP|专属)|"
    r"\b(?:advertisement|sponsored(?:\s+content)?|paid\s+promotion)\b",
    re.IGNORECASE,
)
# (round 12) Three more shapes a headline never has, each the visible part of a planted title the figure rules miss:
# * a *cut phrase*: the title stops mid-clause, on an opening quote or bracket, a colon or comma, a dangling
#   preposition / connective ("Goldman Sachs cuts Kweichow Moutai to", "P/E now stands at", "董事长在会上表示：“",
#   "净利润同比下滑至"), or a metric with no value ("贵州茅台上半年净利润", "五粮液市盈率"): the split variant of a
#   planted document, cut before the claim, so nothing in it can be checked;
# * *spaced characters*: CJK characters written one by one with spaces ("业 绩 预 警"), an obfuscation that keeps
#   keyword filters from matching;
# * *key-value data*: two or more ``key: value`` fields or a ``---`` front-matter fence ("--- ticker: 600519.SH
#   name: 贵州茅台"), a data record rather than a title.
# Measured on the shipped corpus (data/runtime/documents.jsonl + data/documents.json, 38,446 titles, 5,874 shown
# before) and the replay snapshots: 0 newly hidden.
_CUT_PHRASE = re.compile(
    r"(?:[“‘「『（(《\[【:：,，、]|(?<![\w-])-{1,2})\s*$|"
    r"(?:升|降|跌|涨|增|减|回落|提高|降低|下滑|上升|增长|下调|上调|调整)至\s*$|"
    r"(?:调整|变更|更正|更名|确定|下调|上调|认定)为\s*$|(?:表示|指出|声称|回应称|透露)\s*$|"
    r"(?:[一二三四]季度|上半年|下半年|前三季度|全年|年度|今年|去年)(?:归母|扣非)?"
    r"(?:净利润|净利|营业收入|营收|市盈率|市净率|股价|收盘价|每股收益)\s*$|"
    r"(?:市盈率|市净率|收盘价|每股收益|股息率|持股比例|持仓比例)\s*$|"
    # (round 16, review-12 L18) a year or a report and a metric, no value: "贵州茅台2025年净利", "…年报归母净利润"
    r"(?:(?:19|20)\d{2}\s*年度?|年报|中报|半年报|季报)(?:归母|扣非)?"
    r"(?:净利润|净利|营业收入|营收|收入|利润|市盈率|市净率|股价|收盘价|每股收益|市值)\s*$|"
    r"(?<![A-Za-z])(?:ROE|EPS|P/?E|P/?B)\s*$|"
    r"\b(?:to|at|of|by|from|with|and|or|the|than|into)\s*$",
    re.IGNORECASE,
)
_SPACED_CJK = re.compile(r"(?:[一-鿿]\s+){3,}[一-鿿]")
_KEY_VALUE = re.compile(r"(?:(?<![\w/])[A-Za-z_][\w-]{1,24}\s*:\s*[^\s:]+(?:\s|$).*?){2,}|(?:^|\s)-{3,}(?:\s|$)")
_OFFICIAL_SOURCE = re.compile(
    r"关于[^，。]{1,40}的(?:提示性)?公告|年度报告|半年度报告|季度报告|业绩预告|业绩快报|证监会|上交所|深交所|北交所|"
    r"交易所|国家统计局|人民银行|央行|财政部|国资委|金融监管总局|"
    r"\bannual\s+report\b|\binterim\s+report\b|\bCSRC\b|\bstock\s+exchange\b|\bfiling\b",
    re.IGNORECASE,
)


# (round 14, J10) Romanised Chinese (pinyin) in a headline: buy-solicitation written without characters ("qiang lie
# mai ru, man cang!") slips past every Chinese keyword rule. A headline is flagged when it has a pinyin trading word
# (mai ru, man cang, quan cang, jia cang, chao di, shang che, suo ha, bi zhang, zhang ting, mu biao jia, tui jian,
# qiang lie) or, in a title of at least 80% pinyin words, three or more words that are each two or more syllables run
# together ("maotai shiyinglv zhiyou"); English words that happen to be one syllable ("long", "hang") and English
# headlines with Chinese names ("Wuliangye Luzhou Laojiao baijiu sales slow") are not flagged.
_PINYIN_ADVICE = re.compile(
    r"\b(?:mai\s?ru|mai\s?chu|man\s?cang|quan\s?cang|jia\s?cang|jian\s?cang|qing\s?cang|chao\s?di|shang\s?che|"
    r"suo\s?ha|bi\s?zhang|zhang\s?ting|mu\s?biao\s?jia|tui\s?jian|qiang\s?lie|wen\s?zhuan|fan\s?bei|"
    r"gan\s?jin|ma\s?shang)\b",
    re.IGNORECASE,
)
_PINYIN_INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z",
                    "c", "s", "y", "w", "")  # fmt: skip
_PINYIN_FINALS = (
    "iang", "iong", "uang", "ueng", "ang", "eng", "ing", "ong", "ian", "iao", "uai", "uan", "van", "ai", "ei", "ao",
    "ou", "an", "en", "in", "un", "vn", "ia", "ie", "iu", "ua", "uo", "ui", "ve", "ue", "er", "a", "o", "e", "i", "u",
    "v",
)  # fmt: skip
_PINYIN_SYLLABLES = frozenset(
    initial + final for initial in _PINYIN_INITIALS for final in _PINYIN_FINALS if initial or final[0] in "aoe"
)


def _pinyin_syllable_count(word: str) -> int:
    """How many pinyin syllables ``word`` splits into (0 when it does not split), fewest syllables first."""
    word = word.lower().replace("ü", "v")
    best: list[int | None] = [0] + [None] * len(word)
    for end in range(1, len(word) + 1):
        for start in range(max(0, end - 6), end):
            if best[start] is not None and word[start:end] in _PINYIN_SYLLABLES:
                count = best[start] + 1  # type: ignore[operator]
                if best[end] is None or count < best[end]:  # type: ignore[operator]
                    best[end] = count
    return best[-1] or 0


def romanised_chinese(text: str) -> str | None:
    """A pinyin trading word, or three or more multi-syllable pinyin words, in ``text`` (round 14, J10)."""
    advice = _PINYIN_ADVICE.search(text)
    if advice:
        return advice.group(0)
    words = re.findall(r"[A-Za-zü]{2,}", text)
    runs = [word for word in words if len(word) >= 4 and _pinyin_syllable_count(word) >= 2]
    # an English headline with Chinese names ("Wuliangye Luzhou Laojiao baijiu sales slow") has English words too
    pinyin_share = sum(1 for word in words if _pinyin_syllable_count(word)) / len(words) if words else 0.0
    return " ".join(runs[:3]) if len(runs) >= 3 and pinyin_share >= 0.8 else None


# (round 14, J10) Regulatory action, fraud and delisting asserted by a headline: "被立案调查", "行政处罚", "警示函",
# "财务造假", "实施风险警示", "退市", "fined", "under investigation", "delisting". Such a title is shown only when the
# document comes from an official source (``official_document``): a regulator or exchange as the publisher, or a
# company filing ("关于…的公告") delivered as an announcement. Naming the regulator in the title ("证监会公告：…") is
# not a source; a headline that claims to be a regulator's notice is withheld the same way. Applied by the agent's
# evidence ledger (``agent/graph.py``), which has the document's source; ``safe_headline`` stays a title-only check.
_REGULATORY_ACTION = re.compile(
    r"立案|行政处罚|处罚决定|(?:被|遭|拟)(?:证监会|交易所|监管部门|监管机构)?(?:罚款|处罚|处分|谴责|查处|调查|留置)|"
    r"罚款|罚没|警示函|监管函|监管警示|监管措施|纪律处分|公开谴责|通报批评|责令改正|市场禁入|风险警示|\*\s*ST|"
    r"退市|终止上市|暂停上市|涉嫌|造假|欺诈发行|内幕交易|操纵(?:股价|市场|证券)|违规担保|资金占用|"
    r"\b(?:investigat(?:ion|ions|ed|ing)|probe[sd]?|penalt(?:y|ies)|penali[sz]ed|fined|sanction(?:s|ed)?|"
    r"warning\s+letter|enforcement\s+action|delist(?:ing|ed)?|fraud\w*|indicted|charged\s+with|insider\s+trading|"
    r"market\s+manipulation|special\s+treatment)\b",
    re.IGNORECASE,
)
_CLAIMED_NOTICE = re.compile(
    r"^\s*[【\[(（]?\s*(?:中国)?(?:证监会|证券监督管理委员会|银保监会|金融监管总局|人民银行|央行|上交所|深交所|北交所|"
    r"(?:上海|深圳|北京)证券交易所|交易所|国务院|财政部)\s*(?:官方|官网)?\s*(?:公告|通报|通知|声明|决定|快讯|消息)?\s*"
    r"[:：】\])）]|^\s*(?:CSRC|SSE|SZSE|the\s+regulator)\s*(?:notice|announcement|statement)?\s*:",
    re.IGNORECASE,
)
_OFFICIAL_PUBLISHER = re.compile(
    r"证监会|证券监督管理委员会|证券交易所|上交所|深交所|北交所|巨潮|全国股转|金融监管总局|人民银行|"
    r"csrc\.gov\.cn|sse\.com\.cn|szse\.cn|bse\.cn|cninfo|neeq",
    re.IGNORECASE,
)
_FILING_TITLE = re.compile(
    r"关于[^，。]{1,60}的(?:提示性|进展|补充|更正)?公告|(?:公告|年度报告|半年度报告|季度报告|年报)(?:\s*[（(][^）)]*[）)])?\s*$"
)


def regulatory_claim(title: str | None) -> str | None:
    """The regulatory-action, fraud or delisting wording, or the claimed regulator notice, in ``title``."""
    folded = _simplified(re.sub(r"\s+", " ", fold(INVISIBLE.sub("", title or ""))).strip())
    match = _CLAIMED_NOTICE.search(folded) or _REGULATORY_ACTION.search(folded)
    return match.group(0).strip() if match else None


def official_document(
    title: str | None, *, source_type: str | None, source_name: str | None, source_url: str | None
) -> bool:
    """The document comes from a regulator or an exchange, or is a company filing delivered as an announcement."""
    if _OFFICIAL_PUBLISHER.search(f"{source_name or ''} {source_url or ''}"):
        return True
    folded = fold(INVISIBLE.sub("", title or "")).strip()
    return str(source_type or "") in {"announcement", "filing"} and bool(_FILING_TITLE.search(folded))


def regulatory_title_withheld(
    title: str | None, *, source_type: str | None, source_name: str | None, source_url: str | None
) -> bool:
    """(round 14, J10) A headline asserting regulatory action, fraud or delisting, or claiming to be a regulator's
    notice, from a document that is not an official one."""
    return bool(regulatory_claim(title)) and not official_document(
        title, source_type=source_type, source_name=source_name, source_url=source_url
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
    # (round 15, K15) traditional characters and Taiwanese terms are read as simplified ones by every rule below
    folded = _simplified(re.sub(r"\s+", " ", fold(text)).strip())
    found: list[Finding] = []
    if not folded:
        return [Finding("charset", "")]
    word = mixed_script_word(text) or lookalike_word(text)
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
    # (round 16, review-12 L18) the figure rules also read English numbers in words and spaced-out digits
    numeric = joined_digits(spelled_numbers(folded))
    found += [Finding("claim", m.group(0)) for m in _CLAIM_SHAPE.finditer(folded)]
    if numeric != folded:
        found += [Finding("claim", m.group(0)) for m in _CLAIM_SHAPE.finditer(numeric)]
    found += [Finding("claim", m.group(0)) for m in _CLAIM_MULTIPLE_EN.finditer(numeric)]
    row = _data_row(folded)
    if row:
        found.append(Finding("claim", row))
    cut = _CUT_FIGURE.search(numeric)
    if cut:
        found.append(Finding("claim", cut.group(0)))
    dramatic = _DRAMATIC_CLAIM.search(folded)
    if dramatic and not _OFFICIAL_SOURCE.search(folded):
        found.append(Finding("claim", dramatic.group(0)))
    advert = _ADVERTISEMENT.search(folded)
    if advert:
        found.append(Finding("promotion", advert.group(0)))
    # (round 14, J10) buy-solicitation or claims written in pinyin
    romanised = romanised_chinese(folded)
    if romanised:
        found.append(Finding("advice", romanised))
    # (round 12) a title cut mid-clause, characters spaced out one by one, or a key-value record
    # (round 15, K15) a limiter or a metric's change word cut before its value
    for pattern in (_CUT_PHRASE, _SPACED_CJK, _KEY_VALUE, _CUT_LIMIT, _CUT_CHANGE):
        match = pattern.search(folded)
        if match:
            found.append(Finding("claim", match.group(0).strip() or folded[-8:]))
    # (round 16, L18) a limiter cut before or at a figure written in words or with spaced digits ("…just four point")
    for pattern in (_CUT_LIMIT, _CUT_LIMIT_EN):
        match = pattern.search(numeric)
        if match:
            found.append(Finding("claim", match.group(0).strip() or numeric[-8:]))
    return found


def safe_headline(title: str | None) -> str | None:
    """The title (invisible characters removed, whitespace collapsed) when it passes the shape check, else ``None``."""
    if not title or headline_findings(title):
        return None
    return re.sub(r"\s+", " ", INVISIBLE.sub("", title)).strip()
