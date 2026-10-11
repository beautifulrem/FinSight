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
"""

from __future__ import annotations

import re
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

# Taiwan / Hong Kong financial terms, after the character table: (pattern, simplified term)
_TERMS = (
    (re.compile(r"每股盈余"), "每股收益"),
    (re.compile(r"股东权益报酬率"), "净资产收益率"),
    (re.compile(r"税后纯益|纯益"), "净利润"),
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
