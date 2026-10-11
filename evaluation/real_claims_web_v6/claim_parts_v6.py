"""Hand-written parts of the sixth set's claims (what each sentence asserts). No value here comes from the data files:
`claimed` is the number as written in the sentence; label_claims_v6.py computes every compared value.

kinds: value (one metric vs the served value), direction (a band word, README tables), relation (two served values),
definition (a stated definition), null (not checkable, with the reason).
"""

from __future__ import annotations

FY = None  # FY2025 annual figures carry no date
D930, D422 = "2026-09-30", "2026-04-22"


def V(entity, metric, claimed, unit="", qual="exact", basis=None, date=FY, note=None, alt=None):
    p = {"kind": "value", "entity": entity, "metric": metric, "claimed": claimed, "unit": unit, "qual": qual}
    if basis:
        p["basis_named"] = basis
    if date:
        p["date"] = date
    if note:
        p["note"] = note
    if alt:
        p["alt_primary_basis"] = alt
    return p


def DIR(entity, metric, band, words, basis=None, date=FY, note=None):
    p = {"kind": "direction", "entity": entity, "metric": metric, "band": band, "words": words}
    if basis:
        p["basis_named"] = basis
    if date:
        p["date"] = date
    if note:
        p["note"] = note
    return p


def REF(entity, metric, date=FY, basis=None):
    r = {"entity": entity, "metric": metric}
    if date:
        r["date"] = date
    if basis:
        r["basis_named"] = basis
    return r


def REL(op, a, b, claimed=None, qual="exact", words=None, note=None):
    p = {"kind": "relation", "op": op, "a": a, "b": b, "words": words}
    if claimed is not None:
        p["claimed"] = claimed
        p["qual"] = qual
    if note:
        p["note"] = note
    return p


def DEF(entity, metric, stated, standard=True, basis=None, note=None):
    p = {"kind": "definition", "entity": entity, "metric": metric, "stated": stated, "standard": standard}
    if basis:
        p["basis_named"] = basis
    if note:
        p["note"] = note
    return p


def NUL(what, reason):
    return {"kind": "null", "what": what, "reason": reason}


KF_YOY = "扣非 YoY is not served"
PP = "change in percentage points (prior-year ratio not served)"

PARTS: dict[str, list[dict]] = {
    "w6_01": [
        V("中国平安", "revenue", "10505.06", "亿", note="主营收入 read as generic revenue"),
        V("中国平安", "revenue_yoy", "2.1", "%"),
        V("中国平安", "net_profit", "1347.78", "亿", basis="归母"),
        V("中国平安", "netprofit_yoy", "6.45", "%", basis="归母"),
        V("中国平安", "profit_dedt", "1437.73", "亿"),
        NUL("扣非净利润同比上升22.47%", KF_YOY),
    ],
    "w6_02": [
        DEF("长江电力", "profit_dedt", "扣非归母净利润 = 剔除偶然(非经常性)收益后的归母净利润"),
        NUL("扣非归母净利润2025年同比增长2.89%", KF_YOY),
        NUL("扣非增速低于归母净利润增速", "comparison needs the 扣非 YoY, which is not served"),
    ],
    "w6_03": [
        DIR("上证指数", "pct", "up", "收涨", date=D422),
        NUL("深证成指收涨", "深证成指 not in the data"),
        DIR("创业板指", "pct", "up", "收涨", date=D422),
        REL("gt", REF("创业板指", "pct", D422), REF("上证指数", "pct", D422), words="深强沪弱 (创业板指 vs 沪指)"),
        NUL("深强沪弱 (深证成指 vs 沪指)", "深证成指 not in the data"),
    ],
    "w6_04": [
        V("贵州茅台", "net_margin", "50.53", "%", alt="total",
          note="generic 'net profit margin' compared with served 归母净利润/营业收入"),
        NUL("net profit margin declined 1.75 percentage points", PP),
    ],
    "w6_05": [
        V("宁德时代", "revenue_yoy", "17", "%"),
        DIR("宁德时代", "revenue_yoy", "up", "稳增", note="稳增 = 稳步增长 (README: g > 0)"),
        V("宁德时代", "netprofit_yoy", "42", "%", note="generic 利润"),
        DIR("宁德时代", "netprofit_yoy", "strong_up", "狂飙", note="狂飙 read in the sense of 飙升 / surged (g ≥ 20)"),
    ],
    "w6_06": [
        V("创业板指", "close", "3135.28", "点", date=D930),
    ],
    "w6_07": [
        DIR("上证指数", "pct", "slight_up", "微红收官", date=D930, note="微红 read in the sense of 微涨 (0 < c ≤ 1)"),
    ],
    "w6_08": [
        V("比亚迪", "net_margin", "4.06", "%", alt="total"),
        NUL("净利率同比下降1.12个百分点", PP),
    ],
    "w6_09": [
        V("上证指数", "pct", "0.31", "%", date=D930),
        V("上证指数", "close", "3842.19", "点", date=D930),
        NUL("深证成指微跌0.11%", "深证成指 not in the data"),
        V("创业板指", "pct", "-0.23", "%", date=D930),
    ],
    "w6_10": [
        V("贵州茅台", "net_margin", "52", "%", alt="total"),
        NUL("净利率和2024年持平", PP),
    ],
    "w6_11": [
        V("中信证券", "profit_dedt", "303.03", "亿"),
        NUL("扣非同比增长40.51%", KF_YOY),
        NUL("扣非增速高于归母净利润增速", "comparison needs the 扣非 YoY, which is not served"),
    ],
    "w6_12": [
        V("紫金矿业", "net_profit", "517.77", "亿", basis="归母"),
        V("紫金矿业", "netprofit_yoy", "61.55", "%", basis="归母"),
        V("紫金矿业", "prior_net_profit", "320.51", "亿", basis="归母", note="上年同期 归母净利润"),
    ],
    "w6_13": [
        V("平安银行", "net_profit", "426", "亿"),
        V("平安银行", "prior_net_profit", "445", "亿", note="2024年的445亿"),
        DIR("平安银行", "netprofit_yoy", "slight_down", "略低于", note="略低于 read in the sense of 略降 (−5 ≤ g < 0)"),
    ],
    "w6_14": [
        V("贵州茅台", "profit_dedt", "82293107655.25", "元"),
        NUL("2024年扣非净利润86,240,905,977.42元", "prior-year 扣非 not served"),
        NUL("扣非同比下滑4.58%", KF_YOY),
    ],
    "w6_15": [
        V("五粮液", "revenue", "40.529", "billion"),
        V("五粮液", "revenue_yoy", "-54.55", "%"),
    ],
    "w6_16": [
        V("上证指数", "close", "3842", "点", date=D930),
        V("上证指数", "pct", "0.3", "%", date=D930),
    ],
    "w6_17": [
        V("紫金矿业", "profit_dedt", "50.724", "billion"),
        NUL("扣非 year-on-year increase of 60.05%", KF_YOY),
    ],
    "w6_18": [
        V("宁德时代", "net_margin", "17", "%", qual="approx", basis="归母",
          note="net profit margin attributable to the parent = 归母净利润/营业收入"),
    ],
    "w6_19": [
        NUL("毛利润同比增长349.26亿元", "毛利润 amount not served"),
        V("紫金矿业", "gross_margin", "27.73", "%"),
        NUL("综合毛利率同比增7.36pct", PP),
    ],
    "w6_20": [
        V("美的集团", "revenue", "458.5", "billion", basis="营业总收入", note="'total revenue' = 营业总收入"),
        V("美的集团", "net_profit", "43.95", "billion", basis="归母"),
        V("美的集团", "revenue_yoy", "10", "%", qual="double_digit"),
        V("美的集团", "netprofit_yoy", "10", "%", qual="double_digit", note="generic 'profit'"),
    ],
    "w6_21": [
        V("招商银行", "revenue_yoy", "0.01", "%"),
        V("招商银行", "netprofit_yoy", "1.2", "%", note="generic 淨利潤", alt="归母"),
        REL("gt", REF("招商银行", "netprofit_yoy"), REF("招商银行", "revenue_yoy"),
            words="淨利潤增幅亦僅為略高的 (higher than the revenue growth)"),
    ],
    "w6_22": [
        DEF("招商银行", "roe", "ROE = yearly earnings per unit of shareholder equity (134 per 1,000)"),
        V("招商银行", "roe", "13.4", "%", note="134 yuan per 1,000 yuan of equity = 13.4%"),
        NUL("China Construction Bank earns 100 yuan per 1,000", "建设银行 not in the data"),
    ],
    "w6_23": [
        V("美的集团", "profit_dedt", "412.7", "亿"),
        NUL("扣非同比增長15.5%", KF_YOY),
    ],
    "w6_24": [
        V("宁德时代", "revenue", "4237.02", "亿"),
        V("宁德时代", "revenue_yoy", "17.04", "%"),
        V("宁德时代", "net_profit", "722.01", "亿", basis="归母"),
        V("宁德时代", "netprofit_yoy", "42.28", "%", basis="归母"),
        V("宁德时代", "profit_dedt", "645.08", "亿"),
        NUL("扣非同比增长43.37%", KF_YOY),
    ],
    "w6_25": [
        NUL("中国移动2025年归母净利润1370.95亿元", "中国移动 not in the data"),
        V("贵州茅台", "net_profit", "823.2", "亿", basis="归母"),
        V("宁德时代", "net_profit", "722.01", "亿", basis="归母"),
    ],
    "w6_26": [
        V("上证指数", "pct", "0.3", "%", date=D930),
        NUL("CSI 300 rose 0.3 percent on 2026-09-30", "沪深300 bars are served for 2026-04-21/22 only"),
    ],
    "w6_27": [
        V("中信证券", "revenue", "748.54", "亿", basis="营业收入"),
        V("中信证券", "revenue_yoy", "17.35", "%", basis="营业收入",
          note="17.35% is growth against the originally reported 2024 营业收入; the served 28.8 is against the restated "
               "2024 figure (sources.json note). The sentence names 营业收入, not a prior-year basis, so it is compared."),
        V("中信证券", "net_profit", "300.76", "亿", basis="归母"),
        V("中信证券", "netprofit_yoy", "38.58", "%", basis="归母"),
        REL("gt", REF("中信证券", "netprofit_yoy"), REF("中信证券", "revenue_yoy"), words="净利润增速显著高于营收增速"),
    ],
    "w6_28": [
        V("贵州茅台", "revenue", "1688.38", "亿", basis="营业收入"),
        V("贵州茅台", "revenue_yoy", "-1.21", "%"),
        DIR("贵州茅台", "revenue_yoy", "slight_down", "微降"),
    ],
    "w6_29": [
        V("平安银行", "revenue", "1314", "亿"),
        V("平安银行", "revenue_yoy", "-10", "%",
          note="'营收同比2024年下滑了10个百分点' read as revenue YoY −10% (an amount has no percentage-point change)"),
    ],
    "w6_30": [
        REL("ratio", REF("五粮液", "revenue", basis="营业收入"), REF("五粮液", "prior_revenue"), claimed="0.5", qual="lt",
            words="营业收入仅为上年的不足一半"),
        REL("ratio", REF("五粮液", "net_profit"), REF("五粮液", "prior_net_profit"), claimed="0.3", qual="lt",
            words="净利润缩水至不足三成"),
        NUL("扣非净利润跌幅超过七成", KF_YOY),
    ],
    "w6_31": [
        V("贵州茅台", "revenue", "1688.38", "亿", basis="营业收入"),
        V("贵州茅台", "revenue_yoy", "-1.21", "%"),
        V("贵州茅台", "net_profit", "82.32", "亿", basis="归母"),
        V("贵州茅台", "netprofit_yoy", "-4.53", "%", basis="归母"),
        V("贵州茅台", "profit_dedt", "82.29", "亿"),
        NUL("扣非同比下降4.58%", KF_YOY),
    ],
    "w6_32": [
        DIR("比亚迪", "revenue_yoy", "slight_up", "小幅增长"),
    ],
    "w6_33": [
        V("中国平安", "profit_dedt", "1437.73", "亿"),
        NUL("約6,700億新台幣", "New Taiwan dollar conversion (foreign currency)"),
        NUL("扣非年增22.5%", KF_YOY),
    ],
    "w6_34": [
        V("比亚迪", "net_profit", "32.6", "billion", basis="归母"),
        NUL("($4.7 billion)", "US-dollar amount"),
        V("比亚迪", "prior_net_profit", "40.3", "billion", basis="归母", note="down from 40.3 billion yuan in 2024"),
    ],
    "w6_35": [
        REL("opposite_sign", REF("比亚迪", "revenue_yoy", basis="营业收入"), REF("比亚迪", "netprofit_yoy"),
            words="营业收入与净利润变动背离 (changes in opposite directions)"),
    ],
    "w6_36": [
        REL("lt_magnitude_both_negative", REF("平安银行", "netprofit_yoy"), REF("平安银行", "revenue_yoy"),
            words="[净利润]降幅小于营收降幅"),
    ],
    "w6_37": [
        V("中芯国际", "profit_dedt", "4.124", "billion"),
        NUL("扣非 increase of 55.9%", KF_YOY),
    ],
    "w6_38": [
        V("紫金矿业", "gross_margin", "27.73", "%"),
        NUL("毛利率同比上升7.35个百分点", PP),
        V("紫金矿业", "net_margin", "18.28", "%", alt="total"),
        NUL("净利率较上年同期上升5.31个百分点", PP),
    ],
    "w6_39": [
        V("五粮液", "revenue", "405.29", "亿", basis="营业收入"),
        V("五粮液", "revenue_yoy", "-54.55", "%"),
        V("五粮液", "net_profit", "89.54", "亿", basis="归母"),
        V("五粮液", "netprofit_yoy", "-71.89", "%", basis="归母"),
    ],
    "w6_40": [
        V("比亚迪", "revenue", "8039.65", "亿"),
        V("比亚迪", "revenue_yoy", "3.46", "%"),
        V("比亚迪", "net_profit", "326.19", "亿", basis="归母"),
        V("比亚迪", "netprofit_yoy", "-18.97", "%", basis="归母"),
    ],
    "w6_41": [
        V("平安银行", "revenue", "1314.42", "亿", basis="营业收入"),
        V("平安银行", "revenue_yoy", "-10.4", "%"),
        NUL("降幅较2024年收窄0.53pct", "prior-year growth rate not served"),
        V("平安银行", "net_profit", "426.33", "亿", basis="归母"),
        V("平安银行", "netprofit_yoy", "-4.2", "%", basis="归母"),
        V("平安银行", "roe", "9.15", "%", note="generic ROE; the report's weighted ROE is 9.15", alt="weighted"),
        NUL("ROE同比-0.93pct", PP),
    ],
    "w6_42": [
        V("中芯国际", "revenue", "673.23", "亿"),
        V("中芯国际", "revenue_yoy", "16.5", "%"),
        V("中芯国际", "net_profit", "50.41", "亿", basis="归母"),
        V("中芯国际", "netprofit_yoy", "36.3", "%", basis="归母"),
        V("中芯国际", "profit_dedt", "41.24", "亿"),
        NUL("扣非同比55.9%", KF_YOY),
        NUL("扣非同比暴增", "direction of the 扣非 YoY, which is not served"),
    ],
    "w6_43": [
        V("中芯国际", "net_profit", "50.41", "亿", qual="approx", basis="归母"),
        V("中芯国际", "netprofit_yoy", "36.3", "%", basis="归母"),
        DIR("中芯国际", "netprofit_yoy", "strong_up", "大幅增长", basis="归母"),
    ],
    "w6_44": [
        V("招商银行", "revenue", "337.5", "billion", note="bank 'operating income' = 营业收入"),
        DIR("招商银行", "revenue_yoy", "flat", "was flat"),
    ],
    "w6_45": [
        V("招商银行", "roe", "13.44", "%", note="generic 净资产收益率", alt="weighted"),
        REL("gt", REF("招商银行", "roe"), REF("工商银行", "roe"), words="超过六大行ROE水平 (工商银行)"),
        NUL("超过六大行ROE水平 (农业银行, 中国银行, 建设银行, 交通银行, 邮储银行)", "the other five big banks are not in the data"),
    ],
    "w6_46": [
        V("创业板指", "close", "3752.76", "点", date=D422),
        V("创业板指", "pct", "1.73", "%", date=D422),
    ],
    "w6_47": [
        V("五粮液", "pe_ttm", "13.74", "倍", date=D422,
          note="served PE on 2026-04-22 uses FY2025 归母净利润 (annual report published 2026-04-30)"),
    ],
    "w6_48": [
        V("中信证券", "revenue", "74.8", "billion"),
        NUL("($10.4 billion)", "US-dollar amount"),
        V("中信证券", "net_profit", "30.1", "billion"),
    ],
    "w6_49": [
        V("工商银行", "net_profit", "3707.66", "亿", basis="total",
          note="净利润 contrasted with 其中归属于母公司股东净利润 in the same sentence: the group (total) net profit"),
        V("工商银行", "netprofit_yoy", "1.0", "%", basis="total"),
        V("工商银行", "net_profit", "3685.62", "亿", basis="归母"),
    ],
    "w6_50": [
        V("招商银行", "revenue_yoy", "0.01", "%"),
        V("招商银行", "netprofit_yoy", "1.2", "%", basis="归母"),
    ],
    "w6_51": [
        V("美的集团", "net_profit", "439", "亿", basis="归母"),
        V("美的集团", "netprofit_yoy", "14", "%", basis="归母"),
    ],
    "w6_52": [
        V("宁德时代", "revenue", "4237", "亿"),
        REL("ratio", REF("宁德时代", "revenue"), REF("比亚迪", "revenue"), claimed="0.5", qual="lt", words="不到比亚迪一半"),
        V("宁德时代", "net_profit", "722", "亿"),
        REL("ratio_minus_one", REF("宁德时代", "net_profit"), REF("比亚迪", "net_profit"), claimed="1", qual="n_more",
            words="赚的钱比比亚迪多一倍多 (1 < A/B − 1 < 2)"),
        V("宁德时代", "net_margin", "18", "%", note="比例 = net margin; '高到18%'", alt="total"),
    ],
    "w6_53": [
        V("紫金矿业", "net_profit", "517.77", "亿", basis="归母"),
        V("紫金矿业", "netprofit_yoy", "61.55", "%", basis="归母"),
        REL("ratio", REF("紫金矿业", "net_profit"), REF("紫金矿业", "prior_net_profit"), claimed="2", qual="near",
            words="接近翻倍式增长 (近 N with N = 2: 1.8 ≤ ratio ≤ 2)"),
    ],
    "w6_54": [
        V("五粮液", "close", "100.64", "元", date=D422),
        V("五粮液", "pct", "-0.53", "%", date=D422),
    ],
    "w6_55": [
        V("长江电力", "gross_margin", "61.67", "%"),
        NUL("较去年同期的59.13%有所上升", "prior-year gross margin not served"),
        NUL("高于行业平均44.02%", "industry average not in the data"),
    ],
    "w6_56": [
        V("中信证券", "roe", "10.59", "%", basis="weighted"),
        NUL("同比+2.5pct", PP),
    ],
    "w6_57": [
        V("平安银行", "net_profit", "426.33", "亿", basis="归母"),
        V("平安银行", "netprofit_yoy", "-4.2", "%", basis="归母"),
        NUL("较2024年11.6%的降幅大幅收窄", "prior-year growth rate not served"),
    ],
    "w6_58": [
        V("长江电力", "revenue", "862.40", "亿"),
        V("长江电力", "revenue_yoy", "2.07", "%"),
        DIR("长江电力", "revenue_yoy", "slight_up", "小幅增长"),
        V("长江电力", "net_profit", "345", "亿", qual="gte", basis="归母", note="站上345亿元关口"),
        V("长江电力", "netprofit_yoy", "6.17", "%", basis="归母"),
        REL("gt", REF("长江电力", "netprofit_yoy"), REF("长江电力", "revenue_yoy"), words="利润增速大幅甩开营收增速"),
    ],
    "w6_59": [
        V("宁德时代", "close", "291.11", "元", date=D930),
        V("宁德时代", "pct", "1.5", "%", date=D930),
        NUL("换手率0.7%", "turnover rate"),
        V("宁德时代", "volume", "29.7", "万手", date=D930),
        V("宁德时代", "amount", "86.14", "亿", date=D930),
    ],
    "w6_60": [
        V("创业板指", "pct", "1.73", "%", date=D422),
        V("创业板指", "close", "3752.76", "点", date=D422),
    ],
}

CONTEXT: dict[str, list[str]] = {
    "w6_02": ["含金量极高 (opinion)", "真实反映主业经营实力的核心指标 (opinion)"],
    "w6_05": ["盈利质量全面上行 (opinion)"],
    "w6_06": ["its lowest close of 2026 (superlative)"],
    "w6_07": ["在银行、医药、白酒等权重板块支撑下 (sector attribution)"],
    "w6_25": ["位居前三位 (ranking; excluding financials and oil/petrochemicals)"],
    "w6_32": ["出海贡献销量增量 (sales volume, not served)"],
    "w6_36": ["靠的是大幅压降信用减值损失和业务管理费 (cause; impairment and expenses not served)"],
    "w6_45": ["整体业绩保持平稳 (vague)", "盈利效率位居股份制银行最优 (ranking)"],
    "w6_47": ["这数儿太少见了 (opinion)"],
    "w6_53": ["核心盈利水平創歷史新高 (superlative)"],
    "w6_60": ["创近11年新高 (superlative)"],
}
