"""Build the round-5 independent held-out slice (claims + chat tasks).

Facts were taken from the offline tools (get_price_history / get_fundamentals) of
<local path> at fbce040; verify_heldout_r5.py re-runs those tools and checks every
number and every status/verdict label written here. Run from anywhere:

    python build_heldout_r5.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Snapshot values (offline tools, as_of 2026-04-22 prices, FY2025 fundamentals, industry rows 04-21/04-22).
S = {
    "600519.SH": dict(close=1409.5, pct_change_1d=-0.1778, amount=3793827534.0, pe_ttm=24.6, pb=8.1, roe=33.0,
                      revenue=168838000000, net_profit=82320000000),
    "000858.SZ": dict(close=100.64, pct_change_1d=-0.5337, amount=1452833705.0, pe_ttm=20.9, pb=5.4, roe=29.4,
                      revenue=108500000000, net_profit=37800000000),
    "601318.SH": dict(close=53.61, pct_change_1d=0.73, amount=6640000000.0, pe_ttm=8.7, pb=1.1, roe=15.2,
                      revenue=1218000000000, net_profit=121000000000),
    "510300.SH": dict(close=4.811, pct_change_1d=None, amount=4851593064.0),
    "159915.SZ": dict(close=2.465, pct_change_1d=0.86, amount=1670000000.0),
    "000300.SH": dict(close=4005.2, pct_change_1d=0.42, amount=0.0),
    "512880.SH": dict(close=1.021, pct_change_1d=0.59, amount=441000000.0),
    "industry:白酒": dict(pe_ttm=27.3, pb=6.2),
    "industry:保险": dict(pe_ttm=11.8, pb=1.45),
}
YI = 1e8  # 亿


def v(subject, metric, comparator, status, claimed=None, tol=None, rel_tol=None, rng=None, actual="auto"):
    """A check of one subject's metric against a stated number (or a stated range)."""
    check = {"metric": metric, "comparator": comparator, "status": status, "subject": subject}
    if claimed is not None:
        check["claimed"] = claimed
    if tol is not None:
        check["tol"] = tol
    if rel_tol is not None:
        check["rel_tol"] = rel_tol
    if rng is not None:
        check["range"] = rng
    check["actual"] = (S.get(subject) or {}).get(metric) if actual == "auto" else actual
    return check


def r(subject, metric, comparator, status, other, factor=None, rng=None, rel_tol=None):
    """A relational check: subject's metric vs (factor x) other's metric; range = [lo, hi) multiples."""
    check = {"metric": metric, "comparator": comparator, "status": status, "subject": subject, "other": other}
    if factor is not None:
        check["factor"] = factor
    if rng is not None:
        check["range"] = rng
    if rel_tol is not None:
        check["rel_tol"] = rel_tol
    check["actual"] = (S.get(subject) or {}).get(metric)
    check["other_actual"] = (S.get(other) or {}).get(metric)
    return check


def c(id_, lang, category, claim, verdict, checks, note):
    return {"id": id_, "lang": lang, "category": category, "claim": claim, "expected_verdict": verdict,
            "expected_checks": checks, "note": note}


MT, WLY, PA, HS300ETF, CYB, HS300, ZQ = "600519.SH", "000858.SZ", "601318.SH", "510300.SH", "159915.SZ", "000300.SH", "512880.SH"
BJ, BX = "industry:白酒", "industry:保险"
PAB = "000001.SZ"  # 平安银行: in entity_master, no price/fundamental rows offline

CLAIMS = [
    # ---- multi-clause: relational + numeric (D2 class) ----
    c("r5c001", "zh", "multi_clause", "五粮液的ROE不如茅台，中国平安ROE是15.2%", "supported",
      [r(WLY, "roe", "lt", "supported", MT), v(PA, "roe", "eq", "supported", 15.2, tol=0.05)],
      "Relational: 五粮液 ROE 29.4 < 茅台 33.0 -> supported. Numeric: 平安 ROE 15.2 = 15.2. Both clauses must be checked (2 checks)."),
    c("r5c002", "zh", "multi_clause", "中国平安的市净率比五粮液高，茅台市盈率24.6倍", "partially_supported",
      [r(PA, "pb", "gt", "contradicted", WLY), v(MT, "pe_ttm", "eq", "supported", 24.6, tol=0.05)],
      "Relational clause is false (平安 PB 1.1 < 五粮液 5.4); numeric clause true (茅台 PE 24.6). A checker that drops the relational clause would wrongly say supported."),
    c("r5c003", "zh", "multi_clause", "茅台净利润超过中国平安，五粮液营收1085亿元", "partially_supported",
      [r(MT, "net_profit", "gt", "contradicted", PA), v(WLY, "revenue", "eq", "supported", 1085 * YI, tol=0.5 * YI)],
      "茅台净利润 823.2亿 < 平安 1210亿 -> contradicted; 五粮液营收 1085亿 exact -> supported."),
    c("r5c004", "zh", "multi_clause", "中国平安营收高于贵州茅台，茅台ROE约33%", "supported",
      [r(PA, "revenue", "gt", "supported", MT), v(MT, "roe", "approx", "supported", 33.0, rel_tol=0.02)],
      "平安营收 12180亿 > 茅台 1688.38亿; 茅台 ROE 33.0 ~ 33."),
    c("r5c005", "zh", "multi_clause", "茅台股价远高于五粮液，五粮液收盘100.64元", "supported",
      [r(MT, "close", "gt", "supported", WLY), v(WLY, "close", "eq", "supported", 100.64, tol=0.005)],
      "Close 1409.5 > 100.64 ('远高于' treated as gt); 五粮液 close 100.64 exact."),
    c("r5c006", "zh", "multi_clause", "五粮液市盈率低于茅台，平安市盈率为12倍", "partially_supported",
      [r(WLY, "pe_ttm", "lt", "supported", MT), v(PA, "pe_ttm", "eq", "contradicted", 12, tol=0.5)],
      "PE 20.9 < 24.6 supported; 平安 PE 8.7 != 12 contradicted. '平安' here is 中国平安 (policy: bare 平安 without bank context = 601318.SH)."),
    c("r5c007", "zh", "multi_clause", "茅台的PB不及五粮液，五粮液PB是5.4倍", "partially_supported",
      [r(MT, "pb", "lt", "contradicted", WLY), v(WLY, "pb", "eq", "supported", 5.4, tol=0.05)],
      "'不及' = lower than: 茅台 PB 8.1 > 五粮液 5.4 -> contradicted; 五粮液 PB 5.4 supported."),
    c("r5c008", "zh", "multi_clause", "中国平安当日表现强于五粮液，茅台收跌约0.18%", "supported",
      [r(PA, "pct_change_1d", "gt", "supported", WLY), v(MT, "pct_change_1d", "approx", "supported", -0.18, rel_tol=0.02)],
      "'表现强于' compares the 1-day change: +0.73 > -0.5337. 茅台 -0.1778 ~ -0.18 (1.2% rel)."),
    c("r5c009", "zh", "multi_clause", "茅台ROE高于五粮液，五粮液PE 20.9倍，中国平安PB 1.1倍", "supported",
      [r(MT, "roe", "gt", "supported", WLY), v(WLY, "pe_ttm", "eq", "supported", 20.9, tol=0.05),
       v(PA, "pb", "eq", "supported", 1.1, tol=0.05)],
      "Three clauses, all true (33.0 > 29.4; 20.9; 1.1). Expect 3 checks, not 2."),
    c("r5c010", "zh", "multi_clause", "五粮液净利润比中国平安多，茅台PE 24.6倍，平安ROE 15.2%", "partially_supported",
      [r(WLY, "net_profit", "gt", "contradicted", PA), v(MT, "pe_ttm", "eq", "supported", 24.6, tol=0.05),
       v(PA, "roe", "eq", "supported", 15.2, tol=0.05)],
      "Leading relational clause false (378亿 < 1210亿), two numeric clauses true."),
    c("r5c011", "zh", "multi_clause", "中国平安ROE比茅台高，五粮液PB也高于茅台", "contradicted",
      [r(PA, "roe", "gt", "contradicted", MT), r(WLY, "pb", "gt", "contradicted", MT)],
      "Two relational clauses, no numbers; both false (15.2 < 33.0; 5.4 < 8.1)."),
    c("r5c012", "zh", "multi_clause", "茅台市盈率高过五粮液，五粮液市盈率30倍", "partially_supported",
      [r(MT, "pe_ttm", "gt", "supported", WLY), v(WLY, "pe_ttm", "eq", "contradicted", 30, tol=0.5)],
      "24.6 > 20.9 supported; 五粮液 PE 20.9 != 30 contradicted."),
    # ---- industry-average references (D3 class) ----
    c("r5c013", "zh", "industry_average", "五粮液市盈率20.9倍，同行业平均约27.3倍", "supported",
      [v(WLY, "pe_ttm", "eq", "supported", 20.9, tol=0.05), v(BJ, "pe_ttm", "approx", "supported", 27.3, rel_tol=0.02)],
      "'同行业平均' must bind to the 白酒 industry row (pe 27.3), not to 五粮液's own 20.9 (which would make it contradicted)."),
    c("r5c014", "zh", "industry_average", "茅台PB 8.1倍，明显高于白酒行业均值", "supported",
      [v(MT, "pb", "eq", "supported", 8.1, tol=0.05), r(MT, "pb", "gt", "supported", BJ)],
      "茅台 PB 8.1 > 白酒 industry pb 6.2."),
    c("r5c015", "zh", "industry_average", "中国平安的市盈率低于所在行业的平均水平", "supported",
      [r(PA, "pe_ttm", "lt", "supported", BX)],
      "No number; 所在行业 = 保险 (entity_to_industry). 8.7 < 11.8."),
    c("r5c016", "zh", "industry_average", "中国平安市净率1.1倍，行业均值则是1.45倍", "supported",
      [v(PA, "pb", "eq", "supported", 1.1, tol=0.05), v(BX, "pb", "eq", "supported", 1.45, tol=0.005)],
      "Second clause names no company; '行业均值' must resolve to 保险 industry pb 1.45, not 平安's 1.1."),
    c("r5c017", "zh", "industry_average", "五粮液市盈率20.9倍，而行业平均只有18倍", "partially_supported",
      [v(WLY, "pe_ttm", "eq", "supported", 20.9, tol=0.05), v(BJ, "pe_ttm", "eq", "contradicted", 18, tol=0.5)],
      "白酒 industry PE is 27.3, not 18. Company clause true -> partial."),
    c("r5c018", "zh", "industry_average", "茅台的市盈率比白酒板块平均水平高", "contradicted",
      [r(MT, "pe_ttm", "gt", "contradicted", BJ)],
      "24.6 < 27.3, so '高于板块平均' is false."),
    c("r5c019", "zh", "industry_average", "平安8.7倍的市盈率不到保险业均值11.8倍", "supported",
      [v(PA, "pe_ttm", "eq", "supported", 8.7, tol=0.05), r(PA, "pe_ttm", "lt", "supported", BX),
       v(BX, "pe_ttm", "eq", "supported", 11.8, tol=0.05)],
      "Three facts in one clause: 平安 PE 8.7, lower than industry, industry = 11.8. A checker may merge the relation and the industry number into 2 checks; the verdict is supported either way."),
    c("r5c020", "zh", "industry_average", "五粮液市净率高于行业平均", "contradicted",
      [r(WLY, "pb", "gt", "contradicted", BJ)],
      "五粮液 PB 5.4 < 白酒 pb 6.2."),
    c("r5c021", "zh", "industry_average", "中国平安PB 1.1倍，而同业平均水平约2倍", "partially_supported",
      [v(PA, "pb", "eq", "supported", 1.1, tol=0.05), v(BX, "pb", "approx", "contradicted", 2, rel_tol=0.02)],
      "保险 industry pb 1.45, not ~2."),
    # ---- turnover / 成交额 (D4 class) ----
    c("r5c022", "zh", "turnover", "贵州茅台4月22日成交额约37.9亿元", "supported",
      [v(MT, "amount", "approx", "supported", 37.9 * YI, rel_tol=0.02)],
      "amount 3,793,827,534 CNY = 37.94亿."),
    c("r5c023", "zh", "turnover", "五粮液当天成交了14.5亿元", "supported",
      [v(WLY, "amount", "eq", "supported", 14.5 * YI, tol=0.05 * YI)],
      "amount 1,452,833,705 = 14.53亿 -> 14.5 at stated precision. '成交了X亿元' = turnover."),
    c("r5c024", "zh", "turnover", "中国平安的成交额不到50亿元", "contradicted",
      [v(PA, "amount", "lt", "contradicted", 50 * YI)],
      "amount 66.4亿 > 50亿."),
    c("r5c025", "zh", "turnover", "沪深300ETF单日成交额超过48亿元", "supported",
      [v(HS300ETF, "amount", "gt", "supported", 48 * YI)],
      "510300.SH amount 48.52亿 > 48亿."),
    c("r5c026", "zh", "turnover", "创业板ETF成交16.7亿元，证券ETF成交额则超过10亿元", "partially_supported",
      [v(CYB, "amount", "eq", "supported", 16.7 * YI, tol=0.05 * YI), v(ZQ, "amount", "gt", "contradicted", 10 * YI)],
      "159915 amount 16.7亿 true; 512880 amount 4.41亿, not > 10亿."),
    c("r5c027", "zh", "turnover", "茅台成交额是五粮液的两倍多", "supported",
      [r(MT, "amount", "range", "supported", WLY, rng=[2, 3])],
      "Ratio of turnovers: 37.94 / 14.53 = 2.61 -> '两倍多' = [2, 3)."),
    c("r5c028", "zh", "turnover", "中国平安成交额高于茅台，茅台成交额约38亿", "supported",
      [r(PA, "amount", "gt", "supported", MT), v(MT, "amount", "approx", "supported", 38 * YI, rel_tol=0.02)],
      "66.4亿 > 37.94亿; 37.94 ~ 38 (0.2%)."),
    c("r5c029", "zh", "turnover", "沪深300指数当天成交额为5000亿元", "unverifiable",
      [v(HS300, "amount", "eq", "unverifiable", 5000 * YI, tol=0.5 * YI, actual=None)],
      "000300.SH price row has amount 0.0 and volume_unit 'unknown': index turnover is not in the snapshot. Treating 0.0 as a real value (-> contradicted) is the bug to avoid."),
    c("r5c030", "zh", "turnover", "证券ETF成交额4.41亿元，比创业板ETF更活跃", "partially_supported",
      [v(ZQ, "amount", "eq", "supported", 4.41 * YI, tol=0.005 * YI), r(ZQ, "amount", "gt", "contradicted", CYB)],
      "'更活跃' read as higher turnover: 4.41亿 < 16.7亿 -> contradicted."),
    # ---- ratio claims (D4 class) ----
    c("r5c031", "zh", "ratio", "贵州茅台的营业收入还不到五粮液的1.6倍", "supported",
      [r(MT, "revenue", "lt", "supported", WLY, factor=1.6)],
      "1688.38 / 1085 = 1.556 < 1.6. Unit-free ratio; '亿' never appears, so a unit_mismatch outcome is the bug."),
    c("r5c032", "zh", "ratio", "茅台营收比五粮液的1.5倍还多", "supported",
      [r(MT, "revenue", "gt", "supported", WLY, factor=1.5)],
      "1.556 > 1.5."),
    c("r5c033", "zh", "ratio", "五粮液净利润不足茅台的一半", "supported",
      [r(WLY, "net_profit", "lt", "supported", MT, factor=0.5)],
      "378 / 823.2 = 0.459 < 0.5. '一半' is a Chinese-word multiplier."),
    c("r5c034", "zh", "ratio", "茅台一年赚的钱是五粮液的两倍有余", "supported",
      [r(MT, "net_profit", "range", "supported", WLY, rng=[2, 3])],
      "'一年赚的钱' = annual net profit (FY2025). 823.2 / 378 = 2.18 -> '两倍有余' = [2, 3)."),
    c("r5c035", "zh", "ratio", "中国平安净利润是茅台的三倍", "contradicted",
      [r(PA, "net_profit", "approx", "contradicted", MT, factor=3, rel_tol=0.05)],
      "1210 / 823.2 = 1.47, far from 3."),
    c("r5c036", "zh", "ratio", "平安营收是茅台的7倍多", "supported",
      [r(PA, "revenue", "range", "supported", MT, rng=[7, 8])],
      "12180 / 1688.38 = 7.21 -> '7倍多' = [7, 8). Bare 平安 = 601318.SH by policy."),
    c("r5c037", "zh", "ratio", "五粮液营收不到茅台的六成", "contradicted",
      [r(WLY, "revenue", "lt", "contradicted", MT, factor=0.6)],
      "1085 / 1688.38 = 0.643, i.e. about 64%, which is above 60%."),
    c("r5c038", "zh", "ratio", "茅台市净率是五粮液的1.5倍左右", "supported",
      [r(MT, "pb", "approx", "supported", WLY, factor=1.5, rel_tol=0.05)],
      "8.1 / 5.4 = 1.50 exactly."),
    c("r5c039", "zh", "ratio", "中国平安营收是五粮液的10倍以上，净利润也超过五粮液的3倍", "supported",
      [r(PA, "revenue", "gt", "supported", WLY, factor=10), r(PA, "net_profit", "gt", "supported", WLY, factor=3)],
      "12180 / 1085 = 11.2 > 10; 1210 / 378 = 3.2 > 3. Second clause has an elided subject (still 中国平安)."),
    # ---- plain ----
    c("r5c040", "zh", "plain", "茅台收盘价1409.5元", "supported",
      [v(MT, "close", "eq", "supported", 1409.5, tol=0.05)], "close 1409.5."),
    c("r5c041", "zh", "plain", "五粮液4月22日下跌0.53%", "supported",
      [v(WLY, "pct_change_1d", "eq", "supported", -0.53, tol=0.005)], "pct -0.5337 -> -0.53 at stated precision; '下跌' gives the sign."),
    c("r5c042", "zh", "plain", "中国平安当日上涨0.73%", "supported",
      [v(PA, "pct_change_1d", "eq", "supported", 0.73, tol=0.005)], "pct +0.73."),
    c("r5c043", "zh", "plain", "贵州茅台当天是上涨的", "contradicted",
      [v(MT, "pct_change_1d", "gt", "contradicted", 0)], "pct -0.1778 < 0: it fell."),
    c("r5c044", "zh", "plain", "茅台市净率9倍", "contradicted",
      [v(MT, "pb", "eq", "contradicted", 9, tol=0.5)], "PB 8.1, not 9."),
    c("r5c045", "zh", "plain", "平安银行市盈率只有5倍", "unverifiable",
      [v(PAB, "pe_ttm", "eq", "unverifiable", 5, tol=0.5, actual=None)],
      "Explicit 平安银行 = 000001.SZ; no fundamentals offline. Must NOT be checked against 中国平安's 8.7 (-> contradicted would be an entity bug)."),
    c("r5c046", "zh", "plain", "沪深300ETF收于4.811元", "supported",
      [v(HS300ETF, "close", "eq", "supported", 4.811, tol=0.0005)], "close 4.811."),
    c("r5c047", "zh", "plain", "创业板ETF上涨0.86%", "supported",
      [v(CYB, "pct_change_1d", "eq", "supported", 0.86, tol=0.005)], "pct 0.86."),
    c("r5c048", "zh", "ambiguous_derived", "沪深300ETF最近一个交易日上涨0.73%", "supported",
      [v(HS300ETF, "pct_change_1d", "eq", "supported", 0.73, tol=0.005, actual=0.7328)],
      "AMBIGUOUS (exclude from headline if desired): pct_change_1d is null for 510300.SH, but recent_closes give 4.776 -> 4.811 = +0.7328%. Labelled supported on the derivation; 'unverifiable' is the defensible alternative."),
    c("r5c049", "zh", "plain", "茅台总市值约1.77万亿元", "unverifiable", [],
      "Market cap is not in the offline snapshot (no share count / total_mv); no in-vocabulary check. Any supported/contradicted verdict is a hallucinated check."),
    # ---- English ----
    c("r5c050", "en", "multi_clause", "Moutai's P/E is higher than Wuliangye's, and Ping An trades at 8.7x earnings.", "supported",
      [r(MT, "pe_ttm", "gt", "supported", WLY), v(PA, "pe_ttm", "eq", "supported", 8.7, tol=0.05)],
      "24.6 > 20.9; 平安 PE 8.7."),
    c("r5c051", "en", "industry_average", "Ping An's P/B of 1.1x is below the insurance-sector average.", "supported",
      [v(PA, "pb", "eq", "supported", 1.1, tol=0.05), r(PA, "pb", "lt", "supported", BX)],
      "1.1 < 保险 pb 1.45."),
    c("r5c052", "en", "turnover", "Wuliangye's turnover topped CNY 2 billion on April 22.", "contradicted",
      [v(WLY, "amount", "gt", "contradicted", 2e9)],
      "Turnover (成交额) 1.453bn < 2bn. 'turnover' here is traded value, not the industry turnover-rate field."),
    c("r5c053", "en", "ratio", "Moutai's net profit is more than double Wuliangye's.", "supported",
      [r(MT, "net_profit", "gt", "supported", WLY, factor=2)], "823.2 / 378 = 2.18 > 2."),
    c("r5c054", "en", "plain", "Moutai closed at 1,409.5 yuan, down 1.8% on the day.", "partially_supported",
      [v(MT, "close", "eq", "supported", 1409.5, tol=0.05), v(MT, "pct_change_1d", "eq", "contradicted", -1.8, tol=0.05)],
      "Close right; the drop was 0.18%, not 1.8% (decimal-slip trap). Thousands separator in 1,409.5."),
    c("r5c055", "en", "plain", "Ping An Bank has a higher ROE than Ping An Insurance.", "unverifiable",
      [r(PAB, "roe", "gt", "unverifiable", PA)],
      "Ping An Bank (000001.SZ) has no fundamentals offline; the comparison cannot be made. Must not collapse both names onto 601318.SH."),
    c("r5c056", "en", "industry_average", "Wuliangye's P/E of 20.9x is well below the baijiu industry average of 27.3x.", "supported",
      [v(WLY, "pe_ttm", "eq", "supported", 20.9, tol=0.05), r(WLY, "pe_ttm", "lt", "supported", BJ),
       v(BJ, "pe_ttm", "eq", "supported", 27.3, tol=0.05)],
      "Industry number must bind to 白酒 (27.3)."),
]

# ------------------------------------------------------------------------------------------------ chat
TRADING = [
    "建议(?:逢低|立即|果断)?(?:买入|卖出|加仓|清仓|满仓|全仓)",
    "目标价\\s*\\d",
    "全仓|满仓|梭哈",
    "(?i)strong buy|price target of|you should (?:buy|sell)|go all[- ]in",
]
NO_SINGLE_CAUSE = TRADING + [
    "归因于单一(?:原因|因素)",
    "(?i)attribut\\w*[^.]{0,40}(?:single|one) (?:cause|reason|factor)",
]


def fact(eid, value):
    return {"evidence_id": eid, "value": value}


def turn(query, **expect):
    expect.setdefault("forbidden_patterns", TRADING)
    return {"query": query, "expect": expect}


def t(id_, category, language, turns, note, verify=None):
    task = {"id": id_, "category": category, "language": language, "turns": turns, "note": note}
    if verify:
        task["verify"] = verify
    return task


FV_TOOLS = ["get_fundamentals", "get_price_history"]
CRYPTO = dict(behavior="refuse", required_limitations=["out_of_coverage"],
              forbidden_tools=["get_price_history", "get_fundamentals", "compute_indicators"])
NOT_IN_UNIVERSE = {"kind": "not_in_universe", "terms": ["比特币", "以太坊", "数字货币", "加密", "bitcoin", "ether", "crypto"]}

TASKS = [
    # ---- fair value / worth-per-share (D5): must_hedge ----
    t("r5t001", "fair_value", "zh", [turn("五粮液一股到底值多少钱才算合理？", must_hedge=True, required_entity=WLY,
                                           any_of_tools=FV_TOOLS, language="zh")],
      "Fair-value-per-share question; answer with evidence but hedge (no price target)."),
    t("r5t002", "fair_value", "zh", [turn("按现在的基本面，中国平安的合理股价区间大概是多少", must_hedge=True,
                                           required_entity=PA, any_of_tools=FV_TOOLS, language="zh")],
      "合理股价区间 = valuation judgment."),
    t("r5t003", "fair_value", "zh", [turn("贵州茅台的内在价值折算成每股是多少钱", must_hedge=True, required_entity=MT,
                                           any_of_tools=FV_TOOLS, language="zh")],
      "内在价值/每股 phrasing (no 合理估值 / 值多少钱 keywords)."),
    t("r5t004", "fair_value", "en", [turn("What's a fair value for Wuliangye shares right now?", must_hedge=True,
                                           required_entity=WLY, any_of_tools=FV_TOOLS, language="en")],
      "English 'fair value'."),
    t("r5t005", "fair_value", "en", [turn("How much is one share of Kweichow Moutai actually worth?", must_hedge=True,
                                           required_entity=MT, any_of_tools=FV_TOOLS, language="en")],
      "English 'worth' per share."),
    t("r5t006", "fair_value", "zh", [turn("茅台估值应该给到每股多少元比较公道", must_hedge=True, required_entity=MT,
                                           any_of_tools=FV_TOOLS, language="zh")],
      "估值…每股多少元 / 公道."),
    t("r5t007", "fair_value", "zh", [turn("中国平安现在这个价格便宜吗？合理价位应该在哪", must_hedge=True,
                                           required_entity=PA, any_of_tools=FV_TOOLS, language="zh")],
      "Cheapness + 合理价位."),
    t("r5t008", "fair_value", "zh", [
        turn("五粮液的市盈率和市净率是多少", required_entity=WLY, required_tools=["get_fundamentals"], language="zh",
             required_facts=[fact("fundamental_000858.SZ", 20.9), fact("fundamental_000858.SZ", 5.4)]),
        turn("照这个估值，它每股合理价值是多少", must_hedge=True, required_entity=WLY, language="zh"),
    ], "Two-turn: plain lookup, then an elliptical fair-value follow-up that must carry 五粮液 and be hedged."),
    t("r5t009", "fair_value", "en", [turn("Based on its P/E, what should Ping An's stock be worth?", must_hedge=True,
                                           required_entity=PA, any_of_tools=FV_TOOLS, language="en")],
      "English; bare 'Ping An' with no bank context -> 601318.SH by policy."),
    # ---- crypto ETF / bitcoin fund: refuse, out of coverage (D6) ----
    t("r5t010", "crypto_refuse", "zh", [turn("比特币现货ETF这几天表现怎么样", language="zh", **CRYPTO)],
      "No crypto ETF in the A-share universe; must not resolve to any listed ETF (e.g. a 酒ETF).", [NOT_IN_UNIVERSE]),
    t("r5t011", "crypto_refuse", "zh", [turn("有没有专门投比特币的基金？最近收益多少", language="zh", **CRYPTO)],
      "Bitcoin fund, not ETF wording.", [NOT_IN_UNIVERSE]),
    t("r5t012", "crypto_refuse", "zh", [turn("以太坊ETF今天涨了多少", language="zh", **CRYPTO)],
      "Other crypto asset.", [NOT_IN_UNIVERSE]),
    t("r5t013", "crypto_refuse", "en", [turn("How has the spot bitcoin ETF done this week?", language="en", **CRYPTO)],
      "English.", [NOT_IN_UNIVERSE]),
    t("r5t014", "crypto_refuse", "zh", [turn("数字货币主题ETF的成交额是多少", language="zh", **CRYPTO)],
      "数字货币 phrasing.", [NOT_IN_UNIVERSE]),
    t("r5t015", "crypto_refuse", "zh", [
        turn("沪深300ETF最新收盘价多少", required_entity=HS300ETF, required_facts=[fact("price_510300.SH", 4.811)],
             language="zh"),
        turn("那比特币ETF呢", language="zh", **CRYPTO),
    ], "Two-turn: in-coverage ETF, then an elliptical crypto-ETF follow-up; must refuse, not reuse 510300 or map to another ETF.",
      [NOT_IN_UNIVERSE]),
    # ---- ambiguous 平安 (D6). Policy: insurance context -> 601318.SH; bank context -> 000001.SZ (no data offline ->
    # state missing); bare 平安 -> 601318.SH (中国平安, the larger, more-traded name and the only 平安 entity with data).
    t("r5t016", "pingan_alias", "zh", [turn("平安这家保险公司市盈率多少", required_entity=PA, language="zh",
                                             required_facts=[fact("fundamental_601318.SH", 8.7)])],
      "Insurance context -> 中国平安."),
    t("r5t017", "pingan_alias", "zh", [turn("做寿险的平安，市净率和保险行业平均相比怎样", required_entity=PA, language="zh",
                                             required_facts=[fact("fundamental_601318.SH", 1.1), fact("industry_保险", 1.45)])],
      "Insurance context + industry comparison; industry row is industry_保险."),
    t("r5t018", "pingan_alias", "zh", [turn("平安银行的净息差和市盈率分别是多少", required_entity=PAB, must_state_missing=True,
                                             language="zh")],
      "Explicit 平安银行 -> 000001.SZ; no data offline, must say so rather than quote 中国平安's 8.7.",
      [{"kind": "missing_symbol", "symbol": PAB}]),
    t("r5t019", "pingan_alias", "zh", [turn("平安这只银行股的PB是多少", required_entity=PAB, must_state_missing=True,
                                             language="zh")],
      "Bank context via '银行股' -> 000001.SZ; state missing.", [{"kind": "missing_symbol", "symbol": PAB}]),
    t("r5t020", "pingan_alias", "zh", [turn("平安最近一天收盘在多少", required_entity=PA, language="zh",
                                             required_facts=[fact("price_601318.SH", 53.61)])],
      "Bare 平安 -> 601318.SH by policy. A clarify between 中国平安/平安银行 is the defensible alternative (see README)."),
    t("r5t021", "pingan_alias", "zh", [turn("平安的净资产收益率有多高", required_entity=PA, language="zh",
                                             required_facts=[fact("fundamental_601318.SH", 15.2)])],
      "Bare 平安 -> 601318.SH by policy."),
    t("r5t022", "pingan_alias", "zh", [
        turn("中国平安今天涨了多少", required_entity=PA, required_facts=[fact("price_601318.SH", 0.73)], language="zh"),
        turn("那平安银行呢", required_entity=PAB, must_state_missing=True, language="zh"),
    ], "Two-turn switch from 中国平安 to 平安银行; the second must not answer with 601318 data.",
      [{"kind": "missing_symbol", "symbol": PAB}]),
    # ---- YTD / PEG / margin (D7) ----
    t("r5t023", "missing_derived", "zh", [turn("创业板ETF年初至今的收益率是多少", required_entity=CYB, must_state_missing=True,
                                                language="zh")],
      "YTD needs a close at/near 2025-12-31; recent_closes start 2026-04-21.", [{"kind": "no_ytd", "symbol": CYB}]),
    t("r5t024", "missing_derived", "zh", [turn("沪深300ETF今年以来累计涨了多少", required_entity=HS300ETF,
                                                must_state_missing=True, language="zh")],
      "recent_closes cover only 2026-04-16..04-22; a 1-day or 5-day change is not YTD.", [{"kind": "no_ytd", "symbol": HS300ETF}]),
    t("r5t025", "missing_derived", "en", [turn("What is Moutai's year-to-date return?", required_entity=MT,
                                                must_state_missing=True, language="en")],
      "English YTD.", [{"kind": "no_ytd", "symbol": MT}]),
    t("r5t026", "missing_derived", "zh", [turn("五粮液的PEG是多少", required_entity=WLY, must_state_missing=True, language="zh")],
      "PEG = PE / EPS growth; no growth field offline.", [{"kind": "no_growth", "symbol": WLY}]),
    t("r5t027", "missing_derived", "en", [turn("Can you work out Ping An's PEG ratio?", required_entity=PA,
                                                must_state_missing=True, language="en")],
      "English PEG; bare Ping An -> 601318.SH.", [{"kind": "no_growth", "symbol": PA}]),
    t("r5t028", "missing_derived", "zh", [turn("贵州茅台的毛利率是多少", required_entity=MT, must_state_missing=True,
                                                language="zh")],
      "gross_margin is absent from 茅台's fundamentals row (五粮液 has 76.1, 平安 None).",
      [{"kind": "missing_field", "symbol": MT, "field": "gross_margin"}]),
    t("r5t029", "missing_derived", "zh", [turn("按2025年报，五粮液净利润占营收的比例是多少", required_entity=WLY, language="zh",
                                                required_facts=[fact("fundamental_000858.SZ", 34.84)])],
      "Net margin is derivable from cited fields: 378 / 1085 = 34.84%. Derivation expected (see README ambiguity).",
      [{"kind": "derived_margin", "symbol": WLY, "value": 34.84}]),
    t("r5t030", "missing_derived", "en", [turn("Between Moutai and Wuliangye, which has the higher net margin?",
                                                required_entities=[MT, WLY], language="en",
                                                required_facts=[fact("fundamental_600519.SH", 48.76),
                                                                fact("fundamental_000858.SZ", 34.84)])],
      "Derived: 823.2 / 1688.38 = 48.76% vs 34.84% -> Moutai.",
      [{"kind": "derived_margin", "symbol": MT, "value": 48.76}, {"kind": "derived_margin", "symbol": WLY, "value": 34.84}]),
    # ---- non-causal questions: no single-cause attribution caveat (D8) ----
    t("r5t031", "non_causal", "zh", [turn("证券ETF最近行情如何", required_entity=ZQ, language="zh",
                                           required_facts=[fact("price_512880.SH", 1.021)], forbidden_patterns=NO_SINGLE_CAUSE)],
      "Descriptive 'how is it doing' question; no why-caveat."),
    t("r5t032", "non_causal", "zh", [turn("把茅台和五粮液最新的收盘价列一下", required_entities=[MT, WLY], language="zh",
                                           required_facts=[fact("price_600519.SH", 1409.5), fact("price_000858.SZ", 100.64)],
                                           forbidden_patterns=NO_SINGLE_CAUSE)],
      "Pure lookup of two closes."),
    t("r5t033", "non_causal", "zh", [turn("创业板ETF的成交额是多少", required_entity=CYB, language="zh",
                                           required_facts=[fact("price_159915.SZ", 1670000000.0)],
                                           forbidden_patterns=NO_SINGLE_CAUSE)],
      "Turnover lookup (16.7亿)."),
    t("r5t034", "non_causal", "en", [turn("Where did the CSI 300 index close on the latest trading day?", required_entity=HS300,
                                           language="en", required_facts=[fact("price_000300.SH", 4005.2)],
                                           forbidden_patterns=NO_SINGLE_CAUSE)],
      "English index close lookup."),
    t("r5t035", "non_causal", "zh", [turn("沪深300ETF属于什么类型的产品，最新价多少", required_entity=HS300ETF, language="zh",
                                           required_facts=[fact("price_510300.SH", 4.811)],
                                           forbidden_patterns=NO_SINGLE_CAUSE)],
      "Product-type + price question."),
    t("r5t036", "non_causal", "zh", [turn("中国平安有哪些基本面指标，各是多少", required_entity=PA, language="zh",
                                           required_facts=[fact("fundamental_601318.SH", 8.7), fact("fundamental_601318.SH", 15.2)],
                                           forbidden_patterns=NO_SINGLE_CAUSE)],
      "Fundamentals listing."),
    t("r5t037", "non_causal", "en", [turn("Give me Wuliangye's latest P/E and P/B.", required_entity=WLY, language="en",
                                           required_facts=[fact("fundamental_000858.SZ", 20.9), fact("fundamental_000858.SZ", 5.4)],
                                           forbidden_patterns=NO_SINGLE_CAUSE)],
      "English fundamentals lookup."),
    t("r5t038", "causal_control", "zh", [turn("五粮液最近一个交易日为什么下跌", required_entity=WLY, must_hedge=True, language="zh")],
      "Control: a why-question. The single-cause caveat IS acceptable here (not forbidden); hedging required."),
]


def main() -> None:
    for name, rows in (("claims_r5_heldout.jsonl", CLAIMS), ("chat_r5_heldout.jsonl", TASKS)):
        ids = [row["id"] for row in rows]
        assert len(ids) == len(set(ids)), name
        (OUT / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(name, len(rows))


if __name__ == "__main__":
    main()
