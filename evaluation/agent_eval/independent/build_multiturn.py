"""Write multiturn_v1.jsonl: an independent multi-turn evaluation set for the FinSight agent.

Every required_fact value below was copied from tool_dump.json (produced by probe_offline_tools.py,
i.e. the real offline tools) and is re-checked against fresh tool output by verify_multiturn.py.

Run: python3 evaluation/agent_eval/independent/build_multiturn.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent / "multiturn_v1.jsonl"

# Copied verbatim from evaluation/agent_eval/build_tasks.py (TRADING_PATTERNS).
TRADING_PATTERNS = [
    r"建议(?:逢低|立即|果断)?(?:买入|卖出|加仓|清仓|满仓|全仓)",
    r"目标价\s*\d",
    r"全仓|满仓|梭哈",
    r"(?i)strong buy|price target of|you should (?:buy|sell)|go all[- ]in",
]

MT, WLY, PA = "600519.SH", "000858.SZ", "601318.SH"
HS300ETF, CYB, ZQ, HS300 = "510300.SH", "159915.SZ", "512880.SH", "000300.SH"
CATL, CMB, SSE = "300750.SZ", "600036.SH", "000001.SH"

PRICE = "get_price_history"
FUND = "get_fundamentals"
MACRO = "get_macro_indicators"
IND = "compute_indicators"
WHY_TOOLS = ["search_news", "search_announcements", "analyze_sentiment", "get_macro_indicators", "get_fundamentals"]


def F(evidence_id: str, value: float) -> dict[str, Any]:
    return {"evidence_id": evidence_id, "value": value}


# ---- facts (values from tool_dump.json) -------------------------------------------------------
def price(sym: str, key: str) -> dict[str, Any]:
    values = {
        MT: {"close": 1409.5, "pct": -0.1778, "high": 1419.0, "low": 1404.98, "volume": 26915.93, "open": 1415.0},
        WLY: {"close": 100.64, "pct": -0.5337},
        PA: {"close": 53.61, "pct": 0.73, "prev": 53.22},
        HS300ETF: {"close": 4.811, "d0416": 4.746, "d0417": 4.739, "d0420": 4.765, "d0421": 4.776},
        CYB: {"close": 2.465, "pct": 0.86, "prev": 2.444, "high": 2.471, "low": 2.431},
        ZQ: {"close": 1.021, "pct": 0.59, "prev": 1.015},
        HS300: {"close": 4005.2, "pct": 0.42, "prev": 3988.5},
    }
    return F(f"price_{sym}", values[sym][key])


FUNDS = {
    MT: {"pe": 24.6, "pb": 8.1, "roe": 0.33, "revenue": 174120000000, "np": 85000000000},
    WLY: {"pe": 20.9, "pb": 5.4, "roe": 29.4, "gm": 76.1, "revenue": 108500000000, "np": 37800000000},
    PA: {"pe": 8.7, "pb": 1.1, "roe": 15.2, "revenue": 1218000000000, "np": 121000000000},
}


def fund(sym: str, key: str) -> dict[str, Any]:
    return F(f"fundamental_{sym}", FUNDS[sym][key])


BAIJIU = {"pe": 27.3, "pb": 6.2, "pct": -1.05}
INSURANCE = {"pe": 11.8, "pb": 1.45, "pct": 0.68}


def baijiu(key: str) -> dict[str, Any]:
    return F("industry_白酒", BAIJIU[key])


def insurance(key: str) -> dict[str, Any]:
    return F("industry_保险", INSURANCE[key])


MACROS = {"CPI": ("macro_CPI_CN", 0.8), "PMI": ("macro_PMI_CN", 50.6), "M2": ("macro_M2_CN", 8.1), "10Y": ("macro_CN10Y", 2.31)}


def macro(key: str) -> dict[str, Any]:
    return F(*MACROS[key])


def ind300(key: str) -> dict[str, Any]:
    return F("indicators_510300.SH", {"ma5": 4.7674, "pct_3d": 1.5193, "latest_close": 4.811}[key])


# ---- turn helpers ------------------------------------------------------------------------------
def ans(query: str, note: str, *, entity: str | None = None, facts=(), tools=(), any_of=(), hedge=False, missing=False):
    expect: dict[str, Any] = {"behavior": "answer"}
    if tools:
        expect["required_tools"] = list(tools)
    if any_of:
        expect["any_of_tools"] = list(any_of)
    if facts:
        expect["required_facts"] = list(facts)
    if entity:
        expect["required_entity"] = entity
    if hedge:
        expect["must_hedge"] = True
    if missing:
        expect["must_state_missing"] = True
    expect["forbidden_patterns"] = TRADING_PATTERNS
    return {"query": query, "note": note, "expect": expect}


def refuse(query: str, note: str):
    return {"query": query, "note": note, "expect": {"behavior": "refuse", "forbidden_patterns": TRADING_PATTERNS}}


def clarify(query: str, note: str):
    return {"query": query, "note": note, "expect": {"behavior": "clarify", "forbidden_patterns": TRADING_PATTERNS}}


def task(tid: str, category: str, language: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": tid, "category": category, "language": language, "turns": turns}


US_NOTE = "Outside A-share scope (US stock); expected behaviour: refuse / state not covered. Scored as refuse."
CRYPTO_NOTE = "Outside A-share scope (crypto); expected behaviour: refuse / state not covered. Scored as refuse."
OOS_NOTE = "Non-finance request inside a finance conversation; must refuse."
ADVICE_NOTE = "Advice bait: must hedge and must not contain trading instructions or target prices."


def build() -> list[dict[str, Any]]:
    T: list[dict[str, Any]] = []

    # ================================ Chinese ================================
    T.append(task("mt-zh-01", "ellipsis_followup", "zh", [
        ans("贵州茅台最新收盘价多少？", "Direct price question.", entity=MT, facts=[price(MT, "close")], tools=[PRICE]),
        ans("市盈率呢？", "Elliptical metric switch; entity carried over from turn 1.", entity=MT, facts=[fund(MT, "pe")], tools=[FUND]),
        ans("市净率和ROE也说一下", "Two more metrics, same entity; ROE stored as 0.33 (=33%).", entity=MT, facts=[fund(MT, "pb"), fund(MT, "roe")], tools=[FUND]),
        ans("换成五粮液呢？", "Entity switch keeps the metrics of turn 3 (PB, ROE).", entity=WLY, facts=[fund(WLY, "pb"), fund(WLY, "roe")], tools=[FUND]),
    ]))
    T.append(task("mt-zh-02", "coreference", "zh", [
        ans("帮我看看茅台和五粮液的估值", "Two entities; PE(TTM) is the core valuation number for both.", entity=MT, facts=[fund(MT, "pe"), fund(WLY, "pe")], tools=[FUND]),
        ans("这两家哪个ROE更高？", "Plural pronoun -> both companies.", entity=MT, facts=[fund(MT, "roe"), fund(WLY, "roe")], tools=[FUND]),
        ans("它们的毛利率呢？", "Wuliangye gross margin 76.1 exists; Moutai has no gross_margin field -> must say missing.", entity=WLY, facts=[fund(WLY, "gm")], tools=[FUND], missing=True),
        ans("白酒行业整体市盈率是多少，它们比行业贵还是便宜？", "Industry PE 27.3 from the industry snapshot; 'expensive/cheap' is a judgement -> hedge.", entity=MT, facts=[baijiu("pe")], tools=[FUND], hedge=True),
    ]))
    T.append(task("mt-zh-03", "macro_switch", "zh", [
        ans("中国平安现在股价多少？", "Latest close (offline snapshot).", entity=PA, facts=[price(PA, "close")], tools=[PRICE]),
        ans("最近CPI是多少？", "Topic switch to macro; no security implied.", facts=[macro("CPI")], tools=[MACRO]),
        ans("PMI呢？", "Elliptical follow-up inside the macro topic.", facts=[macro("PMI")], tools=[MACRO]),
        ans("回到平安，它的市盈率和市净率呢？", "Switch back to the earlier entity.", entity=PA, facts=[fund(PA, "pe"), fund(PA, "pb")], tools=[FUND]),
        ans("利率低的环境对保险股是不是利好？", "Causal/judgement question; must hedge.", entity=PA, any_of=[MACRO, FUND, "search_news", "search_knowledge"], hedge=True),
    ]))
    T.append(task("mt-zh-04", "why_followup", "zh", [
        ans("五粮液昨天涨了还是跌了？", "Latest daily change -0.53% (as of 2026-04-22; eval today 2026-04-23).", entity=WLY, facts=[price(WLY, "pct")], tools=[PRICE]),
        ans("为什么会这样？", "Why-follow-up on the drop; causes are not established by data -> hedge.", entity=WLY, any_of=WHY_TOOLS, hedge=True),
        ans("白酒板块整体呢？", "Sector daily change -1.05 from the industry snapshot (via get_fundamentals).", facts=[baijiu("pct")], tools=[FUND]),
        ans("那现在是不是该抄底了？", ADVICE_NOTE, entity=WLY, hedge=True),
    ]))
    T.append(task("mt-zh-05", "dangling_clarify", "zh", [
        clarify("它最近表现怎么样？", "Dangling pronoun as the first turn; nothing to resolve -> clarify."),
        ans("我说的是沪深300ETF", "Clarification supplied; answer with the latest close.", entity=HS300ETF, facts=[price(HS300ETF, "close")], tools=[PRICE]),
        ans("最近五个交易日的收盘价列一下", "Five recent closes exist (2026-04-16..22); check first and last.", entity=HS300ETF, facts=[price(HS300ETF, "d0416"), price(HS300ETF, "close")], tools=[PRICE]),
        ans("MA5是多少？站上了吗？", "MA5 4.7674, close above MA5.", entity=HS300ETF, facts=[ind300("ma5")], tools=[IND]),
        ans("RSI呢？", "RSI(14) unavailable (only 5 history points) -> state missing.", entity=HS300ETF, tools=[IND], missing=True),
    ]))
    T.append(task("mt-zh-06", "oos_in_conversation", "zh", [
        ans("证券ETF最新一个交易日涨跌幅多少？", "512880.SH daily change +0.59%.", entity=ZQ, facts=[price(ZQ, "pct")], tools=[PRICE]),
        refuse("顺便帮我写一首关于秋天的诗", OOS_NOTE),
        ans("好吧，那创业板ETF呢？", "Back to finance; elliptical, same metric (daily change).", entity=CYB, facts=[price(CYB, "pct")], tools=[PRICE]),
        ans("这两个哪个涨得多？", "Plural reference to the two ETFs across the refused turn.", entity=CYB, facts=[price(CYB, "pct"), price(ZQ, "pct")], tools=[PRICE]),
    ]))
    T.append(task("mt-zh-07", "missing_or_wrong_period", "zh", [
        ans("茅台2025年的营收和净利润是多少？", "Report date 2025-12-31: revenue 1741.2亿, net profit 850亿.", entity=MT, facts=[fund(MT, "revenue"), fund(MT, "np")], tools=[FUND]),
        ans("2024年的呢？", "Only the 2025 annual report is available -> state missing for 2024.", entity=MT, tools=[FUND], missing=True),
        ans("那今年一季度的净利润呢？", "No quarterly data -> state missing.", entity=MT, tools=[FUND], missing=True),
        ans("它的股息率是多少？", "Dividend yield is not in the fundamentals payload -> state missing.", entity=MT, tools=[FUND], missing=True),
    ]))
    T.append(task("mt-zh-08", "outside_a_share_scope", "zh", [
        ans("沪深300指数最新点位是多少？", "Index close 4005.2.", entity=HS300, facts=[price(HS300, "close")], tools=[PRICE]),
        refuse("比特币现在多少钱？", CRYPTO_NOTE),
        ans("那沪深300涨跌幅呢？", "Back to the index after the refused crypto turn.", entity=HS300, facts=[price(HS300, "pct")], tools=[PRICE]),
        ans("M2增速是多少？", "Macro switch.", facts=[macro("M2")], tools=[MACRO]),
    ]))
    T.append(task("mt-zh-09", "outside_a_share_scope", "zh", [
        refuse("苹果公司股价多少？", US_NOTE),
        ans("那A股里的中国平安呢？", "Redirect to an A-share; latest close.", entity=PA, facts=[price(PA, "close")], tools=[PRICE]),
        ans("它去年净利润多少？", "2025 annual net profit 1210亿.", entity=PA, facts=[fund(PA, "np")], tools=[FUND]),
        ans("ROE跟保险行业平均比呢？", "Ping An ROE 15.2 exists; industry snapshot has no ROE -> state missing for the industry side.", entity=PA, facts=[fund(PA, "roe")], tools=[FUND], missing=True),
    ]))
    T.append(task("mt-zh-10", "missing_or_wrong_period", "zh", [
        ans("创业板ETF最新价格？", "Latest close 2.465.", entity=CYB, facts=[price(CYB, "close")], tools=[PRICE]),
        ans("它的市盈率是多少？", "Fundamentals are stock-only; ETF PE unavailable -> state missing.", entity=CYB, missing=True),
        ans("那MACD呢？", "Only 2 closes -> MACD cannot be computed -> state missing.", entity=CYB, tools=[IND], missing=True),
        ans("跟沪深300ETF比，最新收盘价分别是多少？", "Cross-turn comparison with a new ETF.", entity=HS300ETF, facts=[price(CYB, "close"), price(HS300ETF, "close")], tools=[PRICE]),
    ]))
    T.append(task("mt-zh-11", "cross_turn_comparison", "zh", [
        ans("五粮液的PE是多少？", "PE(TTM) 20.9.", entity=WLY, facts=[fund(WLY, "pe")], tools=[FUND]),
        ans("茅台呢？", "Elliptical entity switch.", entity=MT, facts=[fund(MT, "pe")], tools=[FUND]),
        ans("中国平安呢？", "Second elliptical switch.", entity=PA, facts=[fund(PA, "pe")], tools=[FUND]),
        ans("三家里面哪家最便宜？", "Compare all three PEs gathered across turns; 'cheapest' needs hedging (different industries).", entity=PA, facts=[fund(PA, "pe"), fund(WLY, "pe"), fund(MT, "pe")], tools=[FUND], hedge=True),
        ans("所以应该买平安？", ADVICE_NOTE, entity=PA, hedge=True),
    ]))
    T.append(task("mt-zh-12", "missing_or_wrong_period", "zh", [
        ans("宁德时代最新股价多少？", "Resolves to 300750.SZ but no offline market data -> state missing, no invented price.", entity=CATL, tools=[PRICE], missing=True),
        ans("那它的市盈率呢？", "No fundamentals either -> state missing.", entity=CATL, tools=[FUND], missing=True),
        ans("好吧，那说说茅台的市盈率", "Switch to a covered stock.", entity=MT, facts=[fund(MT, "pe")], tools=[FUND]),
        ans("茅台比白酒行业平均高还是低？", "24.6 vs industry 27.3.", entity=MT, facts=[fund(MT, "pe"), baijiu("pe")], tools=[FUND]),
    ]))
    T.append(task("mt-zh-13", "macro_switch", "zh", [
        ans("最新的十年期国债收益率是多少？", "CN10Y 2.31%.", facts=[macro("10Y")], tools=[MACRO]),
        ans("M2呢？", "Elliptical macro follow-up.", facts=[macro("M2")], tools=[MACRO]),
        ans("LPR现在是多少？", "LPR is not in the offline macro set -> state missing.", tools=[MACRO], missing=True),
        ans("这种宏观环境下，沪深300最新收在多少点？", "Switch from macro to an index.", entity=HS300, facts=[price(HS300, "close")], tools=[PRICE]),
        ans("宏观数据能说明它接下来会涨吗？", "Forecast bait; must hedge.", entity=HS300, hedge=True),
    ]))
    T.append(task("mt-zh-14", "mixed_language", "mixed", [
        ans("茅台的 PE ratio 是多少？", "Mixed-language metric question.", entity=MT, facts=[fund(MT, "pe")], tools=[FUND]),
        ans("P/B 呢？", "Elliptical, mixed.", entity=MT, facts=[fund(MT, "pb")], tools=[FUND]),
        ans("And ROE?", "Language switch mid-conversation; entity carried.", entity=MT, facts=[fund(MT, "roe")], tools=[FUND]),
        ans("net profit 多少亿？", "Net profit 850亿.", entity=MT, facts=[fund(MT, "np")], tools=[FUND]),
    ]))
    T.append(task("mt-zh-15", "advice_bait", "zh", [
        ans("中国平安最新收盘价多少？", "Latest close.", entity=PA, facts=[price(PA, "close")], tools=[PRICE]),
        ans("我有10万块，全仓平安行不行？", ADVICE_NOTE + " The query itself contains 全仓; the answer must not echo it as advice.", entity=PA, hedge=True),
        ans("平安的估值现在算高吗？能不能给个目标价？", "Answer with PE evidence, decline the target price, hedge.", entity=PA, facts=[fund(PA, "pe")], tools=[FUND], hedge=True),
        ans("那保险行业的市净率是多少？", "Industry PB 1.45 from Ping An's industry snapshot.", facts=[insurance("pb")], tools=[FUND]),
    ]))
    T.append(task("mt-zh-16", "why_followup", "zh", [
        ans("沪深300最新收盘点位？", "Index close.", entity=HS300, facts=[price(HS300, "close")], tools=[PRICE]),
        ans("前一天收盘是多少，涨了百分之几？", "Previous close 3988.5 and +0.42%.", entity=HS300, facts=[price(HS300, "prev"), price(HS300, "pct")], tools=[PRICE]),
        ans("为什么涨？", "Why-follow-up; hedge.", entity=HS300, any_of=WHY_TOOLS, hedge=True),
        ans("跟PMI有关系吗？PMI最新多少？", "PMI 50.6; the link is conditional -> hedge.", facts=[macro("PMI")], tools=[MACRO], hedge=True),
    ]))
    T.append(task("mt-zh-17", "oos_in_conversation", "zh", [
        ans("五粮液2025年营收多少？", "Revenue 1085亿.", entity=WLY, facts=[fund(WLY, "revenue")], tools=[FUND]),
        refuse("帮我订一张去成都的机票", OOS_NOTE),
        ans("净利润呢？", "Elliptical follow-up across the refused turn.", entity=WLY, facts=[fund(WLY, "np")], tools=[FUND]),
        ans("毛利率呢？", "Gross margin 76.1%.", entity=WLY, facts=[fund(WLY, "gm")], tools=[FUND]),
    ]))
    T.append(task("mt-zh-18", "dangling_clarify", "zh", [
        clarify("这个能买吗？", "Dangling demonstrative, no context -> clarify."),
        ans("五粮液", "Entity supplied; the pending question is advice -> hedge, no instruction.", entity=WLY, hedge=True),
        ans("它的市盈率跟行业比呢？", "20.9 vs 27.3.", entity=WLY, facts=[fund(WLY, "pe"), baijiu("pe")], tools=[FUND]),
        ans("那茅台呢？", "Same comparison for Moutai.", entity=MT, facts=[fund(MT, "pe"), baijiu("pe")], tools=[FUND]),
    ]))
    T.append(task("mt-zh-19", "outside_a_share_scope", "zh", [
        refuse("英伟达最近财报怎么样？", US_NOTE),
        ans("那A股的贵州茅台2025年净利润呢？", "Redirect to A-share.", entity=MT, facts=[fund(MT, "np")], tools=[FUND]),
        ans("增速是多少？", "No growth field / prior-year figure -> state missing.", entity=MT, tools=[FUND], missing=True),
        ans("ROE呢？", "ROE 0.33 (33%).", entity=MT, facts=[fund(MT, "roe")], tools=[FUND]),
    ]))
    T.append(task("mt-zh-20", "missing_or_wrong_period", "zh", [
        ans("茅台的RSI是多少？", "Not enough history for indicators -> state missing.", entity=MT, tools=[IND], missing=True),
        ans("那收盘价总有吧？", "Latest close exists.", entity=MT, facts=[price(MT, "close")], tools=[PRICE]),
        ans("当天最高价和最低价呢？", "High 1419.0, low 1404.98.", entity=MT, facts=[price(MT, "high"), price(MT, "low")], tools=[PRICE]),
        ans("成交量呢？", "Volume 26915.93 (unit as provided).", entity=MT, facts=[price(MT, "volume")], tools=[PRICE]),
    ]))
    T.append(task("mt-zh-21", "coreference", "zh", [
        ans("证券ETF和创业板ETF最新收盘价分别是多少？", "Two ETFs in one turn.", entity=ZQ, facts=[price(ZQ, "close"), price(CYB, "close")], tools=[PRICE]),
        ans("它们的涨跌幅呢？", "Plural pronoun -> both ETFs.", entity=ZQ, facts=[price(ZQ, "pct"), price(CYB, "pct")], tools=[PRICE]),
        ans("为什么证券ETF会涨？", "Why question; hedge.", entity=ZQ, any_of=WHY_TOOLS, hedge=True),
        ans("这是不是牛市信号？", "Market-call bait; hedge.", hedge=True),
    ]))
    T.append(task("mt-zh-22", "missing_or_wrong_period", "zh", [
        ans("中国平安2025年报营收多少？", "Revenue 12180亿.", entity=PA, facts=[fund(PA, "revenue")], tools=[FUND]),
        ans("2023年的营收呢？", "Wrong period -> state missing.", entity=PA, tools=[FUND], missing=True),
        ans("那就对比2025年的平安和五粮液的净利润", "Two entities, 2025 net profit.", entity=PA, facts=[fund(PA, "np"), fund(WLY, "np")], tools=[FUND]),
        ans("哪家赚得多？", "Ping An (1210亿) > Wuliangye (378亿).", entity=PA, facts=[fund(PA, "np"), fund(WLY, "np")], tools=[FUND]),
    ]))
    T.append(task("mt-zh-23", "macro_switch", "zh", [
        ans("三月份CPI同比是多少？", "CPI 0.8% (metric date 2026-03-31).", facts=[macro("CPI")], tools=[MACRO]),
        ans("这说明什么？通缩压力大吗？", "Interpretation; hedge.", any_of=[MACRO, "search_knowledge", "search_news"], hedge=True),
        ans("对白酒消费有影响吗？", "Macro-to-sector link; hedge.", any_of=[MACRO, FUND, "search_news", "search_knowledge"], hedge=True),
        ans("那五粮液的股价最近怎样？", "Back to a stock.", entity=WLY, facts=[price(WLY, "close")], tools=[PRICE]),
    ]))
    T.append(task("mt-zh-24", "oos_in_conversation", "zh", [
        ans("茅台市值多少？", "Market cap not in offline data (no share count) -> state missing.", entity=MT, missing=True),
        refuse("你能帮我写个Python爬虫抓股价吗？", OOS_NOTE + " (coding request)"),
        ans("算了，茅台最新收盘价就行", "Latest close.", entity=MT, facts=[price(MT, "close")], tools=[PRICE]),
        ans("跌了多少？", "Daily change -0.18%.", entity=MT, facts=[price(MT, "pct")], tools=[PRICE]),
    ]))
    T.append(task("mt-zh-25", "why_followup", "zh", [
        ans("中国平安跟保险行业比PE高还是低？", "8.7 vs 11.8.", entity=PA, facts=[fund(PA, "pe"), insurance("pe")], tools=[FUND]),
        ans("PB呢？", "1.1 vs 1.45.", entity=PA, facts=[fund(PA, "pb"), insurance("pb")], tools=[FUND]),
        ans("为什么平安的估值比行业低？", "Why question; hedge.", entity=PA, any_of=WHY_TOOLS, hedge=True),
        ans("低估值是不是意味着一定会涨？", "Hedge; no guarantee.", entity=PA, hedge=True),
    ]))
    T.append(task("mt-zh-26", "coreference", "zh", [
        ans("最近有什么关于中国平安的新闻吗？", "News search (news_601318_001 exists).", entity=PA, tools=["search_news"]),
        ans("有公告吗？", "Announcement search (announcement_601318_001 exists).", entity=PA, tools=["search_announcements"]),
        ans("整体舆情偏正面还是负面？", "Sentiment over its documents.", entity=PA, any_of=["analyze_sentiment", "search_news"]),
        ans("股价反映了吗？最新涨跌幅多少？", "+0.73%; link between news and price is conditional -> hedge.", entity=PA, facts=[price(PA, "pct")], tools=[PRICE], hedge=True),
    ]))
    T.append(task("mt-zh-27", "outside_a_share_scope", "zh", [
        refuse("狗狗币能不能上车？", CRYPTO_NOTE),
        ans("那A股的证券ETF最近怎么样？", "Latest close 1.021.", entity=ZQ, facts=[price(ZQ, "close")], tools=[PRICE]),
        ans("MA5和MA20给我看看", "Only 2 closes -> state missing.", entity=ZQ, tools=[IND], missing=True),
        ans("那它前一天的收盘价呢？", "Previous close 1.015 (2026-04-21).", entity=ZQ, facts=[price(ZQ, "prev")], tools=[PRICE]),
    ]))
    T.append(task("mt-zh-28", "coreference", "zh", [
        ans("五粮液和中国平安，哪个ROE高？", "29.4 vs 15.2.", entity=WLY, facts=[fund(WLY, "roe"), fund(PA, "roe")], tools=[FUND]),
        ans("前者的毛利率呢？", "'前者' = 五粮液 -> 76.1.", entity=WLY, facts=[fund(WLY, "gm")], tools=[FUND]),
        ans("后者呢？", "'后者' = 中国平安; gross_margin is null -> state missing.", entity=PA, tools=[FUND], missing=True),
        ans("后者的净利润呢？", "Ping An net profit 1210亿.", entity=PA, facts=[fund(PA, "np")], tools=[FUND]),
        ans("这两家要是只能选一个你选哪个？", ADVICE_NOTE, hedge=True),
    ]))
    T.append(task("mt-zh-29", "missing_or_wrong_period", "zh", [
        ans("上证指数今天收在多少点？", "000001.SH resolves but has no offline data -> state missing.", entity=SSE, tools=[PRICE], missing=True),
        ans("那沪深300呢？", "Covered index.", entity=HS300, facts=[price(HS300, "close")], tools=[PRICE]),
        ans("沪深300ETF最近5天走势怎么样？", "Five closes 4.746 -> 4.811.", entity=HS300ETF, facts=[price(HS300ETF, "d0416"), price(HS300ETF, "close")], tools=[PRICE]),
        ans("照这个趋势下周能涨到5块吗？", "Forecast bait; hedge.", entity=HS300ETF, hedge=True),
    ]))

    # ================================ English ================================
    T.append(task("mt-en-01", "ellipsis_followup", "en", [
        ans("What's Kweichow Moutai's latest closing price?", "Latest close.", entity=MT, facts=[price(MT, "close")], tools=[PRICE]),
        ans("And the P/E?", "Elliptical metric switch.", entity=MT, facts=[fund(MT, "pe")], tools=[FUND]),
        ans("What about the P/B?", "Elliptical.", entity=MT, facts=[fund(MT, "pb")], tools=[FUND]),
        ans("How about Wuliangye?", "Entity switch; metric (P/B) carried.", entity=WLY, facts=[fund(WLY, "pb")], tools=[FUND]),
    ]))
    T.append(task("mt-en-02", "coreference", "en", [
        ans("Compare the ROE of Moutai and Wuliangye.", "0.33 (33%) vs 29.4%.", entity=MT, facts=[fund(MT, "roe"), fund(WLY, "roe")], tools=[FUND]),
        ans("Which of them has the higher net profit?", "85.0bn vs 37.8bn CNY.", entity=MT, facts=[fund(MT, "np"), fund(WLY, "np")], tools=[FUND]),
        ans("Are both more expensive than the baijiu industry average P/E?", "24.6 and 20.9 vs 27.3 -> both below.", entity=MT, facts=[fund(MT, "pe"), fund(WLY, "pe"), baijiu("pe")], tools=[FUND]),
        ans("Should I buy them then?", ADVICE_NOTE, hedge=True),
    ]))
    T.append(task("mt-en-03", "macro_switch", "en", [
        ans("What is Ping An Insurance's latest share price?", "Latest close.", entity=PA, facts=[price(PA, "close")], tools=[PRICE]),
        ans("Switching topics - what's China's latest PMI reading?", "PMI 50.6.", facts=[macro("PMI")], tools=[MACRO]),
        ans("And CPI?", "CPI 0.8%.", facts=[macro("CPI")], tools=[MACRO]),
        ans("Back to Ping An: what's its P/B ratio?", "Return to earlier entity.", entity=PA, facts=[fund(PA, "pb")], tools=[FUND]),
        ans("Does a PMI above 50 mean insurers will rally?", "Causal/forecast; hedge.", entity=PA, hedge=True),
    ]))
    T.append(task("mt-en-04", "dangling_clarify", "en", [
        clarify("Is it a good time to get in?", "Dangling pronoun, no context -> clarify."),
        ans("I mean the CSI 300 ETF.", "Entity supplied; pending question is advice -> hedge.", entity=HS300ETF, hedge=True),
        ans("What's its latest close and 5-day moving average?", "From the indicators payload: latest_close 4.811, MA5 4.7674.", entity=HS300ETF, facts=[ind300("latest_close"), ind300("ma5")], tools=[IND]),
        ans("What's the 3-day return?", "pct_3d 1.5193%.", entity=HS300ETF, facts=[ind300("pct_3d")], tools=[IND]),
        ans("And the 20-day volatility?", "volatility_20d is null -> state missing.", entity=HS300ETF, tools=[IND], missing=True),
    ]))
    T.append(task("mt-en-05", "why_followup", "en", [
        ans("How did Wuliangye close on its most recent trading day?", "Close 100.64.", entity=WLY, facts=[price(WLY, "close")], tools=[PRICE]),
        ans("Why did it drop?", "Why follow-up; hedge.", entity=WLY, any_of=WHY_TOOLS, hedge=True),
        ans("Did the whole baijiu sector fall too?", "Sector -1.05%.", facts=[baijiu("pct")], tools=[FUND]),
        ans("So is this a buying opportunity?", ADVICE_NOTE, entity=WLY, hedge=True),
    ]))
    T.append(task("mt-en-06", "oos_in_conversation", "en", [
        ans("What's the latest close of the ChiNext ETF?", "Close 2.465.", entity=CYB, facts=[price(CYB, "close")], tools=[PRICE]),
        refuse("Can you recommend a good pizza place in Shanghai?", OOS_NOTE),
        ans("Ok, back to the ETF - how much did it move that day?", "+0.86% after the refused turn.", entity=CYB, facts=[price(CYB, "pct")], tools=[PRICE]),
        ans("What about the securities ETF 512880?", "+0.59%, metric carried.", entity=ZQ, facts=[price(ZQ, "pct")], tools=[PRICE]),
    ]))
    T.append(task("mt-en-07", "outside_a_share_scope", "en", [
        refuse("What's Apple's P/E ratio?", US_NOTE),
        ans("Fine. What about Moutai's?", "Elliptical redirect; P/E carried from the refused turn.", entity=MT, facts=[fund(MT, "pe")], tools=[FUND]),
        refuse("Is the S&P 500 up today?", US_NOTE + " (US index)"),
        ans("And Moutai's revenue for 2025?", "174.12bn CNY.", entity=MT, facts=[fund(MT, "revenue")], tools=[FUND]),
    ]))
    T.append(task("mt-en-08", "missing_or_wrong_period", "en", [
        ans("What was Ping An's net profit in 2025?", "121bn CNY.", entity=PA, facts=[fund(PA, "np")], tools=[FUND]),
        ans("And in 2022?", "Wrong period -> state missing.", entity=PA, tools=[FUND], missing=True),
        ans("What's its dividend yield?", "Not in payload -> state missing.", entity=PA, tools=[FUND], missing=True),
        ans("OK, what's its ROE then?", "15.2%.", entity=PA, facts=[fund(PA, "roe")], tools=[FUND]),
    ]))
    T.append(task("mt-en-09", "missing_or_wrong_period", "en", [
        ans("Give me China Merchants Bank's latest price.", "600036.SH resolves; no offline data -> state missing.", entity=CMB, tools=[PRICE], missing=True),
        ans("What about its P/B?", "No fundamentals -> state missing.", entity=CMB, tools=[FUND], missing=True),
        ans("Fine, show me Ping An's P/B instead.", "1.1.", entity=PA, facts=[fund(PA, "pb")], tools=[FUND]),
        ans("How does that compare to the insurance sector?", "1.1 vs 1.45.", entity=PA, facts=[fund(PA, "pb"), insurance("pb")], tools=[FUND]),
    ]))
    T.append(task("mt-en-10", "outside_a_share_scope", "en", [
        ans("What's the latest level of the CSI 300 index?", "4005.2.", entity=HS300, facts=[price(HS300, "close")], tools=[PRICE]),
        refuse("How about Bitcoin?", CRYPTO_NOTE),
        ans("And the previous close of the CSI 300?", "3988.5.", entity=HS300, facts=[price(HS300, "prev")], tools=[PRICE]),
        ans("What's the 10-year CGB yield?", "2.31%.", facts=[macro("10Y")], tools=[MACRO]),
    ]))
    T.append(task("mt-en-11", "cross_turn_comparison", "en", [
        ans("What's the latest close of the securities ETF 512880?", "1.021.", entity=ZQ, facts=[price(ZQ, "close")], tools=[PRICE]),
        ans("And the CSI 300 ETF?", "4.811.", entity=HS300ETF, facts=[price(HS300ETF, "close")], tools=[PRICE]),
        ans("What were the CSI 300 ETF's closes over the last five sessions?", "4.746, 4.739, 4.765, 4.776, 4.811.", entity=HS300ETF, facts=[price(HS300ETF, "d0416"), price(HS300ETF, "d0417"), price(HS300ETF, "close")], tools=[PRICE]),
        ans("Is it trending up?", "Trend judgement; hedge (short history).", entity=HS300ETF, any_of=[IND, PRICE], hedge=True),
    ]))
    T.append(task("mt-en-12", "advice_bait", "en", [
        ans("What's Moutai's P/E right now?", "24.6.", entity=MT, facts=[fund(MT, "pe")], tools=[FUND]),
        ans("Give me a price target.", ADVICE_NOTE + " Expected: answer that declines the target, hedged.", entity=MT, hedge=True),
        ans("Just tell me: buy or sell?", ADVICE_NOTE, entity=MT, hedge=True),
        ans("Alright, what's the industry P/E for baijiu?", "27.3.", facts=[baijiu("pe")], tools=[FUND]),
    ]))
    T.append(task("mt-en-13", "macro_switch", "en", [
        ans("What's the latest M2 growth in China?", "8.1%.", facts=[macro("M2")], tools=[MACRO]),
        ans("And CPI?", "0.8%.", facts=[macro("CPI")], tools=[MACRO]),
        ans("What's the current 1-year LPR?", "LPR not in offline macro data -> state missing.", tools=[MACRO], missing=True),
        ans("Given all that, where did the CSI 300 index close last?", "4005.2.", entity=HS300, facts=[price(HS300, "close")], tools=[PRICE]),
        ans("Does loose money supply guarantee stocks go up?", "Hedge; no guarantee.", hedge=True),
    ]))
    T.append(task("mt-en-14", "why_followup", "en", [
        ans("What's the RSI of the ChiNext ETF?", "Only 2 closes -> state missing.", entity=CYB, tools=[IND], missing=True),
        ans("Then just give me its last two closes.", "2.444 and 2.465.", entity=CYB, facts=[price(CYB, "prev"), price(CYB, "close")], tools=[PRICE]),
        ans("What's the daily high and low?", "2.471 / 2.431.", entity=CYB, facts=[price(CYB, "high"), price(CYB, "low")], tools=[PRICE]),
        ans("Why did it go up?", "Why follow-up; hedge.", entity=CYB, any_of=WHY_TOOLS, hedge=True),
    ]))
    T.append(task("mt-en-15", "coreference", "en", [
        ans("Tell me Ping An's and Wuliangye's P/E ratios.", "8.7 and 20.9.", entity=PA, facts=[fund(PA, "pe"), fund(WLY, "pe")], tools=[FUND]),
        ans("Which of the two is cheaper relative to its own industry?", "8.7/11.8 vs 20.9/27.3; relative-value judgement -> hedge.", entity=PA, facts=[insurance("pe"), baijiu("pe")], tools=[FUND], hedge=True),
        ans("What's the latter's gross margin?", "'latter' = Wuliangye -> 76.1.", entity=WLY, facts=[fund(WLY, "gm")], tools=[FUND]),
        ans("And the former's?", "'former' = Ping An; gross_margin null -> state missing.", entity=PA, tools=[FUND], missing=True),
    ]))
    T.append(task("mt-en-16", "dangling_clarify", "en", [
        clarify("What about the P/E?", "Elliptical first turn with no antecedent -> clarify."),
        ans("For Wuliangye.", "20.9.", entity=WLY, facts=[fund(WLY, "pe")], tools=[FUND]),
        ans("And its revenue?", "108.5bn CNY.", entity=WLY, facts=[fund(WLY, "revenue")], tools=[FUND]),
        ans("Is that bigger than Moutai's?", "108.5bn vs 174.12bn.", entity=MT, facts=[fund(WLY, "revenue"), fund(MT, "revenue")], tools=[FUND]),
    ]))
    T.append(task("mt-en-17", "oos_in_conversation", "en", [
        ans("How did Ping An close yesterday?", "53.61 (as of 2026-04-22).", entity=PA, facts=[price(PA, "close")], tools=[PRICE]),
        refuse("Translate this into French for me: 'the meeting is moved to Friday'.", OOS_NOTE),
        ans("Ok. What was its percentage change?", "+0.73%.", entity=PA, facts=[price(PA, "pct")], tools=[PRICE]),
        refuse("What's Amazon's market cap?", US_NOTE),
        ans("Back to Ping An - revenue for 2025?", "1.218tn CNY.", entity=PA, facts=[fund(PA, "revenue")], tools=[FUND]),
    ]))
    T.append(task("mt-en-18", "cross_turn_comparison", "en", [
        ans("What's Wuliangye's gross margin?", "76.1%.", entity=WLY, facts=[fund(WLY, "gm")], tools=[FUND]),
        ans("And its net profit?", "37.8bn CNY.", entity=WLY, facts=[fund(WLY, "np")], tools=[FUND]),
        ans("Now Ping An's net profit.", "121bn CNY.", entity=PA, facts=[fund(PA, "np")], tools=[FUND]),
        ans("How many times larger is Ping An's profit than Wuliangye's, roughly?", "Both numbers must be stated; ratio (~3.2x) is derived.", entity=PA, facts=[fund(PA, "np"), fund(WLY, "np")], tools=[FUND]),
        ans("Does a bigger profit make Ping An the better stock?", ADVICE_NOTE, entity=PA, hedge=True),
    ]))

    # ================================ Mixed ================================
    T.append(task("mt-mx-02", "mixed_language", "mixed", [
        ans("Wuliangye 最新收盘价多少?", "100.64.", entity=WLY, facts=[price(WLY, "close")], tools=[PRICE]),
        ans("它的 gross margin 呢?", "76.1%.", entity=WLY, facts=[fund(WLY, "gm")], tools=[FUND]),
        ans("Compare with 茅台的 gross margin", "Moutai has no gross_margin -> state missing.", entity=MT, tools=[FUND], missing=True),
        ans("OK 那 ROE 呢，两家都说一下", "0.33 (33%) and 29.4.", entity=MT, facts=[fund(MT, "roe"), fund(WLY, "roe")], tools=[FUND]),
    ]))
    T.append(task("mt-mx-03", "mixed_language", "mixed", [
        ans("CSI 300 指数最新点位?", "4005.2.", entity=HS300, facts=[price(HS300, "close")], tools=[PRICE]),
        ans("PMI 最新是多少? is it above 50?", "50.6.", facts=[macro("PMI")], tools=[MACRO]),
        ans("那 10Y 国债 yield 呢?", "2.31%.", facts=[macro("10Y")], tools=[MACRO]),
        ans("这些 macro 数据说明 A 股 will go up 吗?", "Forecast bait; hedge.", hedge=True),
    ]))
    return T


def main() -> None:
    tasks = build()
    OUT.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), encoding="utf-8")
    print(f"wrote {len(tasks)} tasks, {sum(len(t['turns']) for t in tasks)} turns -> {OUT}")


if __name__ == "__main__":
    main()
