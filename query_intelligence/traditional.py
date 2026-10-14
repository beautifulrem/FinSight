"""Traditional → simplified Chinese, for the NLU front end, the agent and the claim checker.

Round 14: published claims from Hong Kong and Taiwan outlets ("工商銀行年度實現淨利潤3,707.66億元") are read with the
same rules as simplified ones. Round 17 (review-13 M9): questions too ("長江電力的市淨率是多少" was refused as out of
scope, "招商銀行的淨利潤是多少" answered with the close only): the NLU pipeline and the agent convert the question
before any rule or model reads it.

No converter library is a dependency. Two character tables are merged:

* the hand-written table below (the vocabulary of market claims: company names, metrics, units, moves, common
  function words), which wins where both have a row, and
* the one-to-one rows of OpenCC's ``TSCharacters.txt`` (Apache-2.0, ``data/runtime/opencc_ts_characters.tsv``, built by
  ``scripts/build_opencc_table.py``): a traditional character that is never itself a simplified one, so 乾 / 於 / 徵
  are not in it and a simplified name is never rewritten.

Then a few Taiwan / Hong Kong financial terms (每股盈餘 = 每股收益, 年增 = 同比增长). A text with no traditional
character is returned unchanged; characters outside both tables are kept as written.

Round 18 (review-14 N11 / N12): ``normalise_text`` is what the NLU pipeline, the agent and the claim checker call. It
reads full-width Latin letters and digits as ASCII (``fold_width``: "ＢＹＤ的ＰＥ" -> "BYD的PE"; the full-width
Chinese punctuation the parsers split clauses on is kept), converts traditional characters (``to_simplified``) and
rewrites the Hong Kong / Taiwan financial vocabulary of ``REGIONAL_TERMS`` to the mainland term (純利 -> 净利润,
股東應佔溢利 -> 归母净利润, 市賬率 -> 市净率, 按年增長 -> 同比增长, 本益比 -> 市盈率 ...), in traditional or simplified
characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PAIRS = (
    "營营業业淨净潤润歸归屬属於于東东億亿萬万價价漲涨盤盘報报總总額额資资產产權权稅税純纯損损減减長长較较與与為为這这"
    "個个們们來来時时間间國国際际銀银鐵铁電电車车礦矿證证險险會会計计財财務务經经濟济發发開开關关實实現现幣币對对從从"
    "後后當当種种點点單单雙双數数據据員员費费貨货買买賣卖場场達达創创歷历紀纪錄录預预測测調调異异倉仓儲储備备戶户圖图"
    "書书訊讯號号樣样機机構构規规劃划環环節节頭头線线體体層层標标準准臺台灣湾寧宁貴贵華华興兴廣广豐丰陽阳雲云滬沪蘇苏"
    "鄭郑鳳凤龍龙島岛衛卫藥药醫医療疗農农糧粮鋼钢鋁铝銅铜鋰锂製制導导習习學学師师動动態态靜静釋释攤摊兩两歲岁週周記记"
    "載载齊齐擁拥處处獎奖勵励圓圆幾几儘尽盡尽碼码應应該该選选擇择續续繼继綜综維维護护縮缩擴扩張张顯显認认識识議议論论"
    "談谈說说話话讀读請请讓让譯译詳详細细紹绍終终結结約约給给統统績绩紅红緊紧練练組组織织網网絡络級级紙纸裡里裏里麼么"
    "沒没還还邊边過过進进運运遠远遞递門门問问閉闭區区協协賺赚虧亏貶贬負负債债擔担餘余歐欧頁页類类項项順顺領领題题願愿"
    "顧顾風风飛飞馬马驗验鬥斗魚鱼鳥鸟麥麦黃黄齡龄係系徵征樂乐極极橫横檔档殘残氣气決决況况滿满漢汉潛潜無无爭争獲获畫画"
    "盜盗監监眾众確确礎础穩稳積积稱称簡简築筑編编聯联聲声職职腦脑舊旧藝艺蘭兰術术衝冲裝装複复見见視视覺觉觀观訂订設设"
    "許许評评試试詢询語语誤误諾诺變变質质購购賽赛趨趋跡迹軟软輸输轉转辦办連连遷迁郵邮鄉乡針针錢钱錯错鍵键閱阅陸陆隊队"
    "階阶隨随雖虽雜杂難难靈灵響响須须養养館馆驅驱髮发鬆松億亿蓋盖產产績绩淨净漲涨跌跌幣币圍围園园團团夥伙優优價价儀仪"
    "傳传僅仅償偿創创劃划勢势勝胜匯汇廠厂厲厉參参雙双嚴严囑嘱壓压壞坏奪夺奮奋婦妇屆届屬属歲岁幣币廢废彈弹徑径復复憂忧"
    "懷怀戰战撥拨據据攜携擊击敗败斷断昇升晉晋暫暂歡欢氫氢減减湧涌溫温滙汇漁渔濕湿瀏浏燈灯爐炉狀状獨独環环瑪玛瓊琼畢毕"
    "盞盏碩硕禮礼穀谷窮穷筆笔範范糾纠紋纹細细終终絕绝絲丝綠绿緩缓縣县績绩繁繁罰罚聞闻腳脚臨临舉举艦舰莊庄華华萬万蔣蒋"
    "虛虚蟲虫補补襯衬覽览觸触訴诉詐诈誠诚誰谁課课誼谊謀谋謂谓證证譽誉豬猪貝贝負负貢贡貧贫貸贷貿贸資资賈贾賓宾賜赐賠赔"
    "賦赋賴赖贈赠贊赞跡迹蹤踪軍军輕轻輩辈輪轮輯辑辭辞農农這这進进遊游遺遗邁迈鄰邻醜丑釀酿鈔钞鈴铃銷销鋪铺鍋锅鎖锁鎮镇"
    "鏈链鐘钟鑑鉴閃闪閒闲闆板闖闯陳陈陣阵陰阴陽阳隱隐雞鸡霧雾頂顶頓顿預预頒颁頗颇頻频顆颗額额顏颜颱台飄飘飯饭飲饮飾饰"
    "餅饼騰腾驚惊髒脏魯鲁鮮鲜鯨鲸鹽盐麵面黨党齒齿亞亚鋒锋峯峰凱凯啟启嶼屿憑凭擬拟燁烨紡纺綱纲縱纵聖圣蕭萧蘋苹譚谭"
    "豔艳貓猫趙赵輝辉迴回鄒邹釣钓銳锐錦锦鍾钟闊阔陝陕隴陇頌颂騏骐鵬鹏麗丽龐庞嶺岭贏赢藍蓝灣湾紹绍鑒鉴瑤瑶臺台"
)
_HAND = {_PAIRS[index]: _PAIRS[index + 1] for index in range(0, len(_PAIRS), 2)}
_OPENCC_TABLE = Path(__file__).resolve().parents[1] / "data" / "runtime" / "opencc_ts_characters.tsv"


@lru_cache(maxsize=1)
def _opencc_rows() -> dict[str, str]:
    """The one-to-one rows of OpenCC's table; empty when the file is missing (the hand table still applies)."""
    try:
        lines = _OPENCC_TABLE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    rows: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#") or line.startswith("traditional\t"):
            continue
        traditional, simplified = line.split("\t", 1)
        rows[traditional] = simplified
    return rows


@lru_cache(maxsize=1)
def _table() -> tuple[dict[int, str], frozenset[str]]:
    merged = {**_opencc_rows(), **_HAND}
    return str.maketrans(merged), frozenset(char for char, simple in merged.items() if char != simple)

# Taiwan / Hong Kong financial terms, after the character table: (pattern, simplified term). Applied to traditional
# text only ("2025年增5%" in simplified text is a year, then 增); the terms that read the same in either script are in
# ``REGIONAL_TERMS`` (round 18, which took over 每股盈余, 股东权益报酬率 and 纯益 from this table).
_TERMS = (
    (re.compile(r"年增(?=\s*[+\d])"), "同比增长"),
    (re.compile(r"年减(?=\s*\d)"), "同比下降"),
    (re.compile(r"人币"), "人民币"),
)


def has_traditional(text: str) -> bool:
    traditional = _table()[1]
    return any(char in traditional for char in text or "")


def to_simplified(text: str) -> str:
    """The text with the table's traditional characters and terms in simplified Chinese; unchanged otherwise."""
    if not text or not has_traditional(text):
        return text
    simplified = text.translate(_table()[0])
    for pattern, term in _TERMS:
        simplified = pattern.sub(term, simplified)
    return simplified


# ---------------------------------------------------------------------------------------------------------
# (round 18, review-14 N12) width
# ---------------------------------------------------------------------------------------------------------
# Full-width digits and Latin letters (U+FF10-FF19, U+FF21-FF3A, U+FF41-FF5A) and the full-width percent sign are the
# ASCII characters NFKC maps them to. Nothing else in the full-width block is folded: the parsers split clauses on
# ，。；：？！ and read （） as brackets, so those stay as written; ． / ／ / ＆ / ＇ are folded only between ASCII
# letters or digits ("３．２", "Ｐ／Ｅ", "Ｓ＆Ｐ", "ＢＹＤ＇ｓ"). Circled or mathematical digits (①, 𝟏) are not
# folded either: a list marker is not a number.
_WIDTH = str.maketrans(
    {
        **{code: code - 0xFEE0 for code in range(0xFF10, 0xFF1A)},
        **{code: code - 0xFEE0 for code in range(0xFF21, 0xFF3B)},
        **{code: code - 0xFEE0 for code in range(0xFF41, 0xFF5B)},
        0xFF05: "%",
    }
)
_WIDTH_JOINERS = re.compile(r"(?<=[0-9A-Za-z])[．／＆＇](?=[0-9A-Za-z])")
_JOINER_ASCII = {"．": ".", "／": "/", "＆": "&", "＇": "'"}
_HAS_WIDE = re.compile(r"[％０-９Ａ-Ｚａ-ｚ]")


def fold_width(text: str) -> str:
    """Full-width letters, digits and % as ASCII; full-width Chinese punctuation kept (see ``_WIDTH``)."""
    if not text or not _HAS_WIDE.search(text):
        return text
    folded = text.translate(_WIDTH)
    return _WIDTH_JOINERS.sub(lambda match: _JOINER_ASCII[match.group(0)], folded)


# ---------------------------------------------------------------------------------------------------------
# (round 18, review-14 N11) Hong Kong / Taiwan financial vocabulary
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RegionalTerm:
    """A Hong Kong or Taiwan financial term and the mainland term it is read as. Patterns run on simplified characters
    (after ``to_simplified``), so one row covers 純利 and 纯利."""

    pattern: re.Pattern[str]
    term: str
    region: str  # "HK", "TW" or "HK/TW"
    note: str


# A change word after 按年 / 按季 ("按年增长", "按年下跌"): 按年 alone ("按年计算") is not a year-on-year comparison.
_CHANGE_AFTER = r"(?=\s*(?:录得)?(?:增|升|上升|上涨|涨|跌|下跌|降|下降|减|减少|下滑|回落|[+\-−]?\d))"

# Sources: the wording of Hong Kong exchange filings and press (HKEX-listed issuers' results announcements in Chinese:
# "本公司擁有人應佔溢利", "股本回報率", "每股盈利", "按年"; HK financial press: "純利", "市賬率", "收市價", "營業額")
# and of Taiwan's market (TWSE's daily table "個股日本益比、殖利率及股價淨值比"; Taiwanese results reports: "每股盈餘",
# "稅後純益", "股東權益報酬率", "年增率"). The mainland terms are the ones the NLU and the claim checker already read.
# Longer forms come first. Terms whose Hong Kong meaning differs from the mainland word of the same spelling (收益 is
# revenue in Hong Kong filings, income in the mainland; 除稅後溢利 includes minority interests) are left out on purpose.
REGIONAL_TERMS: tuple[RegionalTerm, ...] = (
    RegionalTerm(
        re.compile(r"(?:本公司|母公司|公司)?(?:普通股)?(?:股东|拥有人|权益持有人)应占(?:年度|年内|期内|本年度)?(?:溢利|利润|纯利|盈利)"),
        "归母净利润",
        "HK",
        "profit attributable to owners / shareholders of the company: the mainland 归属于母公司股东的净利润",
    ),
    RegionalTerm(re.compile(r"纯利率|纯益率"), "净利率", "HK/TW", "net profit margin"),
    RegionalTerm(
        re.compile(r"税后纯益|纯利润|纯利|纯益"),
        "净利润",
        "HK/TW",
        "HK press 纯利 / TW 稅後純益: the headline net profit (mainland colloquial 纯利润 too)",
    ),
    RegionalTerm(
        re.compile(r"股东权益报酬率|股东权益回报率|股东资金回报率|股本回报率|权益回报率"),
        "净资产收益率",
        "HK/TW",
        "return on equity (ROE); TW 股東權益報酬率, HK 股本回報率 / 股東權益回報率",
    ),
    RegionalTerm(re.compile(r"市账率|股价净值比|股价账面值比"), "市净率", "HK/TW", "price-to-book (P/B): HK 市賬率, TW 股價淨值比"),
    RegionalTerm(re.compile(r"本益比"), "市盈率", "TW", "price-to-earnings (P/E): TW 本益比"),
    RegionalTerm(re.compile(r"每股盈利|每股盈余"), "每股收益", "HK/TW", "earnings per share (EPS): HK 每股盈利, TW 每股盈餘"),
    RegionalTerm(
        re.compile(r"每股资产净值|每股账面值|每股净值"), "每股净资产", "HK/TW", "book value per share: HK 每股資產淨值, TW 每股淨值"
    ),
    RegionalTerm(re.compile(r"营业额(?!税)"), "营业收入", "HK", "turnover / revenue: HK 營業額"),
    RegionalTerm(re.compile(r"收市价"), "收盘价", "HK", "closing price: HK 收市價"),
    RegionalTerm(re.compile(r"年增率"), "同比增速", "TW", "year-on-year growth rate: TW 年增率"),
    RegionalTerm(re.compile(r"按年" + _CHANGE_AFTER), "同比", "HK", "year on year: HK 按年 (only before a change word)"),
    RegionalTerm(re.compile(r"按季" + _CHANGE_AFTER), "环比", "HK", "quarter on quarter: HK 按季 (only before a change word)"),
)


def regional_terms(text: str) -> str:
    """``text`` with the Hong Kong / Taiwan financial terms of ``REGIONAL_TERMS`` in their mainland form."""
    for row in REGIONAL_TERMS:
        text = row.pattern.sub(row.term, text)
    return text


def normalise_text(text: str) -> str:
    """What every reader of a question or a claim sees: width folded, traditional characters converted, regional
    financial terms in their mainland form (``normalise_text_steps`` also names the steps that changed the text)."""
    return normalise_text_steps(text)[0]


def normalise_text_steps(text: str) -> tuple[str, list[str]]:
    """``(text, steps)``: the steps (``fullwidth_to_ascii``, ``traditional_to_simplified``, ``regional_terms``) that
    changed the text, for the NLU's explainability trace."""
    steps: list[str] = []
    if not text:
        return text, steps
    folded = fold_width(text)
    if folded != text:
        steps.append("fullwidth_to_ascii")
    simplified = to_simplified(folded)
    if simplified != folded:
        steps.append("traditional_to_simplified")
    mainland = regional_terms(simplified)
    if mainland != simplified:
        steps.append("regional_terms")
    return mainland, steps
