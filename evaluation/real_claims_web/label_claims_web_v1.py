"""Label ``collected_v1.jsonl`` from the offline data and write ``claims_web_v1.jsonl``.

Every expected verdict is computed here from the offline tools' output (``get_fundamentals`` /
``get_price_history`` on the offline service with the 2026-09-30 snapshot extension on, the product default);
no fact is typed in. Each claim is split into parts, one per stated number; a part is a predicate over tool values
under the rounding rules below and is True, False or None (the fact is not in the offline data). The verdict
follows the checker's documented aggregation (``ClaimReport.verdict``): all parts true -> supported; no part true
but one false -> contradicted; some true and some false or not in the data -> partially_supported; no part decided
-> unverifiable. Qualitative context without a number of its own (创历史新高, 位列第一, 排名第二, 股价再创新高) is not a
part; it is listed in ``context_not_checked``.

Rounding rules (README, "Labelling rules"):

* an exact figure ("4237.02亿元", "1.50%", "5.8元") holds when |claimed - actual| <= max(half a unit of the claim's
  last written digit, 1% of |actual|); a growth rate or a change also needs the right sign;
* 约 / 左右 / about: within 5% of the claimed number;
* 突破 / 超 / 超过 / 站上 / 首破 (a bound): actual > N (突破 / 站上 / 破: >= N); 涨超1%: the rise is above 1%;
* 近 / 将近 / 接近 N: 0.9 N <= actual <= N;
* a value derived from two offline fields (net margin = net profit / revenue; prior-year profit = profit / (1 + YoY);
  yuan / point change = close - close / (1 + pct); a fund's daily change from its last two closes) is labelled from
  that derivation; a derived value whose own rounding exceeds the claim's precision is None (not in the data).

Run from the repo root::

    python -m evaluation.real_claims_web.label_claims_web_v1
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COLLECTED = HERE / "collected_v1.jsonl"
OUT = HERE / "claims_web_v1.jsonl"

sys.path.insert(0, str(ROOT))

from evaluation.agent_eval.runner import build_offline_service  # noqa: E402
from query_intelligence.agent.tools import build_registry_for_service  # noqa: E402

_service = build_offline_service(snapshot_ext=True)
_registry = build_registry_for_service(_service)
_cache: dict[str, dict[str, Any]] = {}

Value = float | None
Part = tuple[str, Callable[[], Value | bool | None]]


def payload(evidence_id: str) -> dict[str, Any]:
    if evidence_id not in _cache:
        kind, _, target = evidence_id.partition("_")
        tool = "get_price_history" if kind == "price" else "get_fundamentals"
        for item in _registry.run(tool, {"target": target}).evidence:
            _cache[item.evidence_id] = item.payload
        if evidence_id not in _cache:
            raise SystemExit(f"{tool}({target}) did not return {evidence_id}")
    return _cache[evidence_id]


def fund(symbol: str, *keys: str) -> Value:
    """A fundamentals field (first key present); None when the offline data does not carry it."""
    data = payload(f"fundamental_{symbol}")
    for key in keys:
        value = data.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def price(symbol: str, key: str) -> Value:
    value = payload(f"price_{symbol}").get(key)
    if not isinstance(value, int | float) or (key == "amount" and value <= 0):
        return None
    return float(value)


def yi(value: Value) -> Value:
    return None if value is None else value / 1e8


def net_margin(symbol: str) -> Value:
    revenue, profit = fund(symbol, "revenue"), fund(symbol, "net_profit")
    return None if not revenue or profit is None else profit / revenue * 100


def gross_margin(symbol: str) -> Value:
    return fund(symbol, "grossprofit_margin", "gross_margin")


def prior_year(symbol: str, key: str, growth_key: str) -> Value:
    value, growth = fund(symbol, key), fund(symbol, growth_key)
    return None if value is None or growth is None else value / (1 + growth / 100)


def fall(symbol: str) -> Value:
    """Prior-year net profit minus this year's."""
    before, now = prior_year(symbol, "net_profit", "netprofit_yoy"), fund(symbol, "net_profit")
    return None if before is None or now is None else before - now


def change_from_pct(symbol: str) -> Value:
    """close - previous close, previous close = close / (1 + pct / 100)."""
    close, pct = price(symbol, "close"), daily_change(symbol)
    return None if close is None or pct is None else close - close / (1 + pct / 100)


def daily_change(symbol: str) -> Value:
    """pct_change_1d, or (for a fund whose source leaves it empty) the change of its last two closes."""
    pct = price(symbol, "pct_change_1d")
    if pct is not None:
        return pct
    data = payload(f"price_{symbol}")
    closes = [row for row in data.get("recent_closes") or [] if isinstance(row.get("close"), int | float)]
    if len(closes) >= 2 and closes[-1].get("date") == data.get("as_of"):
        return (closes[-1]["close"] / closes[-2]["close"] - 1) * 100
    return None


# ------------------------------------------------------------------ predicates (None: not in the offline data)
def eq(actual: Value, claimed: float, step: float) -> bool | None:
    if actual is None:
        return None
    if claimed and actual and (claimed > 0) != (actual > 0):
        return False
    return abs(actual - claimed) <= max(step / 2, abs(actual) * 0.01)


def approx(actual: Value, claimed: float) -> bool | None:
    return None if actual is None else abs(actual - claimed) <= abs(claimed) * 0.05


def above(actual: Value, bound: float, *, inclusive: bool = False) -> bool | None:
    if actual is None:
        return None
    return actual >= bound if inclusive else actual > bound


def nearly(actual: Value, target: float) -> bool | None:
    return None if actual is None else 0.9 * target <= actual <= target


def absent() -> None:
    return None


CATL, BYD, MIDEA, ZIJIN, SMIC = "300750.SZ", "002594.SZ", "000333.SZ", "601899.SH", "688981.SH"
CMB, ICBC, PAB, CITIC, CYPC = "600036.SH", "601398.SH", "000001.SZ", "600030.SH", "600900.SH"
MOUTAI, WLY, PINGAN = "600519.SH", "000858.SZ", "601318.SH"
SSE, CHINEXT, GOLD_ETF, CSI300, CSI300_ETF = "000001.SH", "399006.SZ", "518880.SH", "000300.SH", "510300.SH"

# id -> (category, parts, context not checked, note comparing the offline value with the primary source)
LABELS: dict[str, tuple[str, list[Part], list[str], str]] = {
    "w01": ("annual_figures", [
        ("营收 4237.02亿", lambda: eq(yi(fund(CATL, "revenue")), 4237.02, 0.01)),
        ("净利润 722.01亿", lambda: eq(yi(fund(CATL, "net_profit")), 722.01, 0.01)),
    ], [], "offline = annual report"),
    "w02": ("loose_or_relational", [
        ("净利润增速 / 营收增速 近2.5倍 [2.25, 2.5]",
         lambda: nearly(None if fund(CATL, "revenue_yoy") is None else
                        (fund(CATL, "netprofit_yoy") or 0) / (fund(CATL, "revenue_yoy") or 1), 2.5)),
    ], [], "offline YoY 42.18% / 17.04% (report 42.28% / 17.04%)"),
    "w03": ("annual_ratios", [
        ("毛利率 26.3%", lambda: eq(gross_margin(CATL), 26.3, 0.1)),
        ("净利率 18.1% (offline: 归母净利润 / 营收)", lambda: eq(net_margin(CATL), 18.1, 0.1)),
    ], ["创近五年新高"], "18.1% is total net profit / revenue; the offline net margin is 归母 / revenue (17.04%)"),
    "w04": ("daily_price", [
        ("收盘 291.11 (2026-09-30)", lambda: eq(price(CATL, "close"), 291.11, 0.01)),
        ("上涨 4.31元 (close - close/(1+pct))", lambda: eq(change_from_pct(CATL), 4.31, 0.01)),
        ("涨幅 1.50%", lambda: eq(daily_change(CATL), 1.50, 0.01)),
    ], ["高开高走"], "offline = source"),
    "w05": ("market_value", [
        ("成交额 86.14亿", lambda: eq(yi(price(CATL, "amount")), 86.14, 0.01)),
        ("换手率 0.70% (not in the offline data)", absent),
        ("总市值 1.35万亿", lambda: eq(None if fund(CATL, "total_mv") is None else fund(CATL, "total_mv") / 1e12,
                                    1.35, 0.01)),
    ], [], "offline = source"),
    "w06": ("annual_ratios", [
        ("归母净利润 326.19亿", lambda: eq(yi(fund(BYD, "net_profit")), 326.19, 0.01)),
        ("同比 -18.97%", lambda: eq(fund(BYD, "netprofit_yoy"), -18.97, 0.01)),
        ("净利率 4.20%", lambda: eq(net_margin(BYD), 4.20, 0.01)),
    ], [], "4.20% is total net profit / revenue; offline 归母 basis 4.06%; offline YoY -18.82% (report -18.97%)"),
    "w07": ("loose_or_relational", [
        ("净利润 326亿", lambda: eq(yi(fund(BYD, "net_profit")), 326, 1)),
        ("2024 净利润 403亿 (profit / (1 + YoY))", lambda: eq(yi(prior_year(BYD, "net_profit", "netprofit_yoy")), 403,
                                                             1)),
        ("少了近80亿 [72, 80]", lambda: nearly(yi(fall(BYD)), 80)),
    ], [], "offline net profit and YoY give 2024 = 401.8亿"),
    "w08": ("english_or_traditional", [
        ("fell 19%", lambda: eq(fund(BYD, "netprofit_yoy"), -19, 1)),
        ("32.62 billion yuan", lambda: eq(yi(fund(BYD, "net_profit")), 326.2, 0.1)),
        ("$4.72 billion (USD, not in the offline data)", absent),
    ], [], "offline = report"),
    "w09": ("english_or_traditional", [
        ("毛利率 17.74%", lambda: eq(gross_margin(BYD), 17.74, 0.01)),
        ("毛利率 同比 -1.7pct (prior year not in the offline data)", absent),
        ("淨利率 4.2%", lambda: eq(net_margin(BYD), 4.2, 0.1)),
        ("淨利率 同比 -1.15pct (prior year not in the offline data)", absent),
    ], [], "4.2% is total net profit / revenue; offline 归母 basis 4.06%"),
    "w10": ("annual_figures", [
        ("营业总收入 4585亿", lambda: eq(yi(fund(MIDEA, "revenue")), 4585, 1)),
        ("归母净利润 439.5亿", lambda: eq(yi(fund(MIDEA, "net_profit")), 439.5, 0.1)),
    ], ["业绩历史新高"], "offline revenue is 营业总收入 (4585.02亿; 营业收入 4564.52亿)"),
    "w11": ("annual_figures", [
        ("营业总收入 4585亿", lambda: eq(yi(fund(MIDEA, "revenue")), 4585, 1)),
        ("同比 +12.1%", lambda: eq(fund(MIDEA, "revenue_yoy"), 12.1, 0.1)),
    ], [], "offline = report (营业总收入)"),
    "w12": ("annual_ratios", [
        ("每股收益 5.8元", lambda: eq(fund(MIDEA, "eps"), 5.8, 0.1)),
        ("加权ROE 19.70%", lambda: eq(fund(MIDEA, "roe"), 19.70, 0.01)),
    ], [], "offline EPS is diluted 5.76 (basic 5.80); offline ROE 19.69 ~ weighted 19.70"),
    "w13": ("annual_figures", [
        ("营收 3491亿", lambda: eq(yi(fund(ZIJIN, "revenue")), 3491, 1)),
        ("同比 +15%", lambda: eq(fund(ZIJIN, "revenue_yoy"), 15, 1)),
    ], [], "offline = report"),
    "w14": ("annual_ratios", [
        ("归母净利润 517.77亿", lambda: eq(yi(fund(ZIJIN, "net_profit")), 517.77, 0.01)),
        ("同比 +61.55%", lambda: eq(fund(ZIJIN, "netprofit_yoy"), 61.55, 0.01)),
        ("ROE 33.04%", lambda: eq(fund(ZIJIN, "roe"), 33.04, 0.01)),
    ], [], "offline ROE 27.91 is not the weighted ROE (report 33.04%); offline YoY 62.01% (report 61.55%)"),
    "w15": ("loose_or_relational", [
        ("归母净利润 突破517亿 (>= 517)", lambda: above(yi(fund(ZIJIN, "net_profit")), 517, inclusive=True)),
        ("同比增长超60%", lambda: above(fund(ZIJIN, "netprofit_yoy"), 60)),
    ], [], "offline = report"),
    "w16": ("annual_ratios", [
        ("基本每股收益 1.95元", lambda: eq(fund(ZIJIN, "eps"), 1.95, 0.01)),
        ("每股收益 同比 +61% (EPS growth not in the offline data)", absent),
    ], [], "offline EPS is diluted 1.91 (basic 1.95)"),
    "w17": ("daily_price", [
        ("收涨 1.12% (2026-09-30)", lambda: eq(daily_change(ZIJIN), 1.12, 0.01)),
        ("收盘 29.79", lambda: eq(price(ZIJIN, "close"), 29.79, 0.01)),
    ], [], "offline = source"),
    "w18": ("annual_ratios", [
        ("毛利率 22%", lambda: eq(gross_margin(SMIC), 22, 1)),
        ("同比 +3个百分点 (prior year not in the offline data)", absent),
    ], [], "offline gross margin 21.62% = report 21.6%"),
    "w19": ("annual_ratios", [
        ("毛利率 21%", lambda: eq(gross_margin(SMIC), 21, 1)),
        ("同比 +3个百分点 (prior year not in the offline data)", absent),
    ], [], "offline 21.62%: the post rounds it to 21 (the company says 22)"),
    "w20": ("english_or_traditional", [
        ("營收 93.27億美元 (USD, not in the offline data)", absent),
        ("年增 16.2%", lambda: eq(fund(SMIC, "revenue_yoy"), 16.2, 0.1)),
    ], [], "16.2% is the USD (IFRS) growth; offline RMB growth 16.49%"),
    "w21": ("annual_figures", [
        ("营收 同比 +16%", lambda: eq(fund(SMIC, "revenue_yoy"), 16, 1)),
        ("净利 +39%", lambda: eq(fund(SMIC, "netprofit_yoy"), 39, 1)),
        ("月产能 突破百万片 (not in the offline data)", absent),
    ], ["创历史新高"], "39% is the IFRS / USD figure; offline (CAS) 36.29%"),
    "w22": ("annual_figures", [
        ("净利润 1501.81亿", lambda: eq(yi(fund(CMB, "net_profit")), 1501.81, 0.01)),
        ("同比 +1.21%", lambda: eq(fund(CMB, "netprofit_yoy"), 1.21, 0.01)),
    ], [], "offline YoY 1.05% is total net profit growth; 归母 growth in the report is 1.21%"),
    "w23": ("annual_figures", [
        ("营业收入 3375.32亿", lambda: eq(yi(fund(CMB, "revenue")), 3375.32, 0.01)),
        ("同比增加 0.44亿 (offline YoY 0.01% is too coarse to derive it)", absent),
        ("增幅 0.01%", lambda: eq(fund(CMB, "revenue_yoy"), 0.01, 0.01)),
    ], [], "offline = report"),
    "w24": ("loose_or_relational", [
        ("招行 ROE 13.44%", lambda: eq(fund(CMB, "roe"), 13.44, 0.01)),
        ("工商银行 ROE 9.45%", lambda: eq(fund(ICBC, "roe"), 9.45, 0.01)),
        ("中国银行 ROE 8.94% (not in the offline data)", absent),
        ("招行 ROE 远高于 工商银行", lambda: None if fund(CMB, "roe") is None or fund(ICBC, "roe") is None
         else (fund(CMB, "roe") or 0) > (fund(ICBC, "roe") or 0)),
    ], [], "offline ROE 11.8 / 8.68 are not the weighted ROE / ROAE (report 13.44% / 9.45%)"),
    "w25": ("loose_or_relational", [
        ("平安银行 净利润 同比 -4.2%", lambda: eq(fund(PAB, "netprofit_yoy"), -4.2, 0.1)),
        ("招行 归母净利润 同比 +1.21%", lambda: eq(fund(CMB, "netprofit_yoy"), 1.21, 0.01)),
    ], [], "offline CMB YoY 1.05% is total net profit growth (归母 1.21%)"),
    "w26": ("annual_ratios", [
        ("招行 加权ROE > 13%", lambda: above(fund(CMB, "roe"), 13)),
        ("连续3年回落 (history not in the offline data)", absent),
    ], [], "offline ROE 11.8 is not the weighted ROE (report 13.44%)"),
    "w27": ("daily_price", [
        ("收盘 41.26 (2026-09-30)", lambda: eq(price(CMB, "close"), 41.26, 0.01)),
        ("上涨 1.85%", lambda: eq(daily_change(CMB), 1.85, 0.01)),
    ], [], "offline = source"),
    "w28": ("annual_figures", [
        ("营业收入 8013.95亿", lambda: eq(yi(fund(ICBC, "revenue")), 8013.95, 0.01)),
        ("同比 +1.9%", lambda: eq(fund(ICBC, "revenue_yoy"), 1.9, 0.1)),
    ], [], "8013.95亿 / 1.9% is the IFRS figure; offline (CAS) 8382.70亿 / 2.0%"),
    "w29": ("annual_figures", [
        ("营业收入 8382.70亿", lambda: eq(yi(fund(ICBC, "revenue")), 8382.70, 0.01)),
        ("同比 +2.0%", lambda: eq(fund(ICBC, "revenue_yoy"), 2.0, 0.1)),
    ], ["位列第一"], "offline = report"),
    "w30": ("annual_figures", [
        ("营收 同比 -1.53%", lambda: eq(fund(ICBC, "revenue_yoy"), -1.53, 0.01)),
        ("连续五年收缩 (history not in the offline data)", absent),
    ], [], "offline = report (+2.0%); the post is wrong"),
    "w31": ("daily_price", [
        ("工商银行 涨超1% (2026-09-30)", lambda: above(daily_change(ICBC), 1)),
        ("中国银行 涨超1% (not in the offline data)", absent),
    ], ["股价再创新高"], "offline = source (+1.47%)"),
    "w32": ("market_value", [
        ("工商银行 总市值 2.81万亿", lambda: eq(None if fund(ICBC, "total_mv") is None else
                                          fund(ICBC, "total_mv") / 1e12, 2.81, 0.01)),
    ], ["排名第二"], "offline total market value 2.95万亿 (2026-09-30); the source's basis is not stated"),
    "w33": ("annual_figures", [
        ("营业收入 1314.42亿", lambda: eq(yi(fund(PAB, "revenue")), 1314.42, 0.01)),
        ("同比 -10.4%", lambda: eq(fund(PAB, "revenue_yoy"), -10.4, 0.1)),
    ], [], "offline = report"),
    "w34": ("annual_figures", [
        ("归母净利润 426.33亿", lambda: eq(yi(fund(PAB, "net_profit")), 426.33, 0.01)),
        ("同比 -4.21%", lambda: eq(fund(PAB, "netprofit_yoy"), -4.21, 0.01)),
    ], [], "offline = report"),
    "w35": ("annual_figures", [
        ("营收 748.54亿", lambda: eq(yi(fund(CITIC, "revenue")), 748.54, 0.01)),
        ("营收 同比 +17.35%", lambda: eq(fund(CITIC, "revenue_yoy"), 17.35, 0.01)),
        ("归母净利润 300.76亿", lambda: eq(yi(fund(CITIC, "net_profit")), 300.76, 0.01)),
        ("净利润 同比 +38.58%", lambda: eq(fund(CITIC, "netprofit_yoy"), 38.58, 0.01)),
    ], [], "17.35% is against pre-restatement 2024 (restated: 28.79%, offline 28.8%); offline profit YoY 37.26% "
           "(report 38.58%)"),
    "w36": ("annual_ratios", [
        ("净利率 41.42%", lambda: eq(net_margin(CITIC), 41.42, 0.01)),
        ("较上年 +6.01个百分点 (prior year not in the offline data)", absent),
    ], [], "41.42% is total net profit / revenue; offline 归母 basis 40.18%"),
    "w37": ("annual_figures", [
        ("营业收入 748.3亿", lambda: eq(yi(fund(CITIC, "revenue")), 748.3, 0.1)),
        ("同比 +29%", lambda: eq(fund(CITIC, "revenue_yoy"), 29, 1)),
    ], [], "flash-report figure; offline = final report 748.54亿 / +28.8%"),
    "w38": ("loose_or_relational", [
        ("净利 首破300亿 (> 300)", lambda: above(yi(fund(CITIC, "net_profit")), 300)),
        ("飙升 39%", lambda: eq(fund(CITIC, "netprofit_yoy"), 39, 1)),
    ], [], "offline profit YoY 37.26% (report 38.58%)"),
    "w39": ("daily_price", [
        ("涨 0.15% (2026-09-30)", lambda: eq(daily_change(CITIC), 0.15, 0.01)),
        ("成交额 14.26亿", lambda: eq(yi(price(CITIC, "amount")), 14.26, 0.01)),
    ], [], "offline = source"),
    "w40": ("annual_figures", [
        ("营收 862.42亿", lambda: eq(yi(fund(CYPC, "revenue")), 862.42, 0.01)),
        ("归母净利 345.03亿", lambda: eq(yi(fund(CYPC, "net_profit")), 345.03, 0.01)),
        ("营收 同比 +2.07%", lambda: eq(fund(CYPC, "revenue_yoy"), 2.07, 0.01)),
        ("净利 同比 +6.17%", lambda: eq(fund(CYPC, "netprofit_yoy"), 6.17, 0.01)),
    ], [], "offline = report (profit YoY 6.13% vs 6.17%)"),
    "w41": ("annual_figures", [
        ("归母净利 同增 7.4%", lambda: eq(fund(CYPC, "netprofit_yoy"), 7.4, 0.1)),
        ("现金分红比例 71% (not in the offline data)", absent),
    ], [], "offline = report (6.17%); 7.4% is the growth of 利润总额"),
    "w42": ("annual_figures", [
        ("净利润 341.67亿", lambda: eq(yi(fund(CYPC, "net_profit")), 341.67, 0.01)),
        ("同比 +5.14%", lambda: eq(fund(CYPC, "netprofit_yoy"), 5.14, 0.01)),
    ], [], "pre-report figures; offline = final report 345.03亿 / +6.13%"),
    "w43": ("annual_ratios", [
        ("每股收益 1.4101元", lambda: eq(fund(CYPC, "eps"), 1.4101, 0.0001)),
        ("同比增加 0.0820元 (prior EPS not in the offline data)", absent),
        ("增长 6.17% (EPS growth not in the offline data)", absent),
    ], [], "offline = report"),
    "w44": ("annual_figures", [
        ("营业总收入 1720.54亿 (offline revenue)", lambda: eq(yi(fund(MOUTAI, "revenue")), 1720.54, 0.01)),
        ("归母净利润 823.20亿", lambda: eq(yi(fund(MOUTAI, "net_profit")), 823.20, 0.01)),
    ], [], "offline revenue is 营业收入 1688.38亿 (营业总收入 1720.54亿 includes 利息收入)"),
    "w45": ("loose_or_relational", [
        ("总营收 约1721亿 (within 5%)", lambda: approx(yi(fund(MOUTAI, "revenue")), 1721)),
    ], [], "offline revenue is 营业收入 1688.38亿, inside 约 (5%)"),
    "w46": ("annual_figures", [
        ("营收 1688.38亿", lambda: eq(yi(fund(MOUTAI, "revenue")), 1688.38, 0.01)),
        ("同比 -1.21% (v1 snapshot has no YoY for 茅台)", lambda: eq(fund(MOUTAI, "revenue_yoy"), -1.21, 0.01)),
        ("归母净利润 823.2亿", lambda: eq(yi(fund(MOUTAI, "net_profit")), 823.2, 0.1)),
    ], [], "offline = report"),
    "w47": ("annual_ratios", [
        ("ROE 32.53%", lambda: eq(fund(MOUTAI, "roe"), 32.53, 0.01)),
    ], ["领跑行业"], "offline ROE 33.0 (v1 seed) vs weighted 32.53% in the report"),
    "w48": ("daily_price", [
        ("跌 0.18% (2026-04-22)", lambda: eq(daily_change(MOUTAI), -0.18, 0.01)),
        ("成交额 37.94亿", lambda: eq(yi(price(MOUTAI, "amount")), 37.94, 0.01)),
    ], [], "offline (v1) = source"),
    "w49": ("annual_figures", [
        ("营业收入 405.29亿", lambda: eq(yi(fund(WLY, "revenue")), 405.29, 0.01)),
        ("同比 -54.55% (v1 snapshot has no YoY for 五粮液)", lambda: eq(fund(WLY, "revenue_yoy"), -54.55, 0.01)),
    ], [], "v1 seed (1085亿 revenue, 378亿 profit) does NOT match the FY2025 report (405.29亿, 89.54亿)"),
    "w50": ("annual_ratios", [
        ("基本每股收益 2.31元 (v1 snapshot has no EPS for 五粮液)", lambda: eq(fund(WLY, "eps"), 2.31, 0.01)),
        ("加权ROE 6.89%", lambda: eq(fund(WLY, "roe"), 6.89, 0.01)),
    ], [], "v1 seed ROE 29.4 does NOT match the FY2025 report (6.89%)"),
    "w51": ("daily_price", [
        ("跌 0.53% (2026-04-22)", lambda: eq(daily_change(WLY), -0.53, 0.01)),
        ("成交额 14.53亿", lambda: eq(yi(price(WLY, "amount")), 14.53, 0.01)),
    ], [], "offline (v1) = source"),
    "w52": ("annual_figures", [
        ("归母净利润 1347.78亿", lambda: eq(yi(fund(PINGAN, "net_profit")), 1347.78, 0.01)),
        ("同比 +6.45% (v1 snapshot has no YoY for 中国平安)", lambda: eq(fund(PINGAN, "netprofit_yoy"), 6.45, 0.01)),
    ], [], "v1 seed (1210亿 profit, 12180亿 revenue) does NOT match the FY2025 report (1347.78亿, 10505.06亿)"),
    "w53": ("daily_price", [
        ("收盘 58.03 (2026-04-22)", lambda: eq(price(PINGAN, "close"), 58.03, 0.01)),
        ("上涨 0.0%", lambda: eq(daily_change(PINGAN), 0.0, 0.1)),
        ("成交额 33.65亿", lambda: eq(yi(price(PINGAN, "amount")), 33.65, 0.01)),
    ], [], "v1 snapshot (53.61, +0.73%, 66.4亿) and the source disagree; not resolved against an exchange record"),
    "w54": ("daily_price", [
        ("沪指 涨 0.31% (2026-09-30)", lambda: eq(daily_change(SSE), 0.31, 0.01)),
        ("深成指 跌 0.11% (not in the offline data)", absent),
        ("创业板指 跌 0.23%", lambda: eq(daily_change(CHINEXT), -0.23, 0.01)),
    ], [], "offline = source"),
    "w55": ("daily_price", [
        ("创业板(指) 跌 0.23% (2026-09-30)", lambda: eq(daily_change(CHINEXT), -0.23, 0.01)),
        ("科创(50 指数) 跌 2.51% (the index is not in the offline data; 科创50ETF is another instrument)", absent),
        ("大盘(沪指) 微涨 0.31%", lambda: eq(daily_change(SSE), 0.31, 0.01)),
    ], [], "offline = source"),
    "w56": ("daily_price", [
        ("上涨 11.74点 (close - close/(1+pct))", lambda: eq(change_from_pct(SSE), 11.74, 0.01)),
        ("涨幅 0.31%", lambda: eq(daily_change(SSE), 0.31, 0.01)),
        ("报 3842.19点", lambda: eq(price(SSE, "close"), 3842.19, 0.01)),
    ], [], "offline = source"),
    "w57": ("daily_price", [
        ("沪市成交额 6793.99亿 (the index row carries no turnover)",
         lambda: eq(yi(price(SSE, "amount")), 6793.99, 0.01)),
        ("深市成交额 7586.19亿 (not in the offline data)", absent),
    ], [], "market turnover is not in the offline data"),
    "w58": ("daily_price", [
        ("报收 8.633 (2026-09-30)", lambda: eq(price(GOLD_ETF, "close"), 8.633, 0.001)),
        ("收涨 1.34%", lambda: eq(daily_change(GOLD_ETF), 1.34, 0.01)),
        ("成交金额 26.95亿", lambda: eq(yi(price(GOLD_ETF, "amount")), 26.95, 0.01)),
    ], [], "offline = source"),
    "w59": ("daily_price", [
        ("沪深300指数 涨 0.66% (2026-04-22)", lambda: eq(daily_change(CSI300), 0.66, 0.01)),
        ("北证50指数 涨 1.59% (not in the offline data)", absent),
        ("科创50指数 涨 1.71% (not in the offline data)", absent),
    ], [], "v1 snapshot +0.42% (4005.2 / 3988.5) vs +0.66% in two media reports; not resolved"),
    "w60": ("daily_price", [
        ("沪深300ETF 涨 0.73% (2026-04-22; from the last two closes)",
         lambda: eq(daily_change(CSI300_ETF), 0.73, 0.01)),
        ("成交额 48.52亿", lambda: eq(yi(price(CSI300_ETF, "amount")), 48.52, 0.01)),
    ], [], "offline (v1) = source"),
}


# Claims the company's own report (or the IFRS / USD / total-profit basis the claim uses) supports, but the offline
# data contradicts because its field has another definition or a wrong seed value; and claims where the offline price
# and the published price disagree without a primary record to settle it. Counted in the README; labels unchanged.
OFFLINE_DIFFERS = {
    "w03", "w06", "w09", "w14", "w16", "w20", "w21", "w22", "w24", "w25", "w26", "w28", "w35", "w36", "w44", "w47",
    "w49", "w50", "w52",
}
UNRESOLVED = {"w53", "w59"}


def verdict(truths: list[bool | None]) -> str:
    decided = [value for value in truths if value is not None]
    if not decided:
        return "unverifiable"
    if all(truths):
        return "supported"
    if not any(decided):
        return "contradicted"
    return "partially_supported"


def main() -> int:
    rows = [json.loads(line) for line in COLLECTED.read_text(encoding="utf-8").splitlines() if line.strip()]
    if {row["id"] for row in rows} != set(LABELS):
        raise SystemExit("collected ids and LABELS differ")
    out = []
    for row in rows:
        category, parts, context, note = LABELS[row["id"]]
        evaluated = []
        for description, predicate in parts:
            value = predicate()
            evaluated.append({"part": description, "true": None if value is None else bool(value)})
        text = row["published_text"].replace("[", "").replace("]", "")
        out.append({
            "id": row["id"],
            "lang": "en" if text.isascii() else "zh",
            "category": category,
            "claim": text,
            **{key: value for key, value in row.items() if key != "id"},
            "subject_inserted": "[" in row["published_text"],
            "expected_verdict": verdict([item["true"] for item in evaluated]),
            "also_acceptable": [],
            "parts": evaluated,
            "context_not_checked": context,
            "offline_vs_primary": note,
            "label_follows_offline_difference": (
                "unresolved" if row["id"] in UNRESOLVED else row["id"] in OFFLINE_DIFFERS
            ),
        })
    with OUT.open("w", encoding="utf-8") as handle:
        for item in out:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for item in out:
        counts[item["expected_verdict"]] = counts.get(item["expected_verdict"], 0) + 1
    print(f"wrote {len(out)} labelled claims to {OUT}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
